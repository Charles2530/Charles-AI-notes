# Roofline 建模与性能案例

做算子优化时，最常见的失败不是“技术不够强”，而是没有先判断瓶颈类型，benchmark 设计不可靠，看到局部提速就误以为端到端受益，或者没有把案例还原成通用方法。

因此这一页专门把 roofline、benchmark 设计和几个典型案例放在一起，帮助你把“性能感觉”变成“性能判断”。

这页适合和 [GEMM 与 Attention Kernel](gemm-attention-and-fused-kernels.md)、[Triton 编程模型与自动调优](triton-programming-model-and-autotuning.md) 一起读。前者提供热点算子的具体结构，后者解释 tile、program id 和 autotune 如何落到实现；Roofline 则负责先判断该往算力、带宽、layout 还是 launch 开销上用力。

## Roofline 的真正用途

Roofline 不是为了精确预测每个 kernel 的最终性能，而是为了先问一个更关键的问题：

**这个 kernel 更可能受算力限制，还是受带宽限制？**

**其直觉形式为**：

$$
\text{Performance}
\le
\min(\text{Peak FLOPs}, \text{Bandwidth} \times \text{Arithmetic Intensity}).
$$

这行式子在说：实际性能不会超过两个上限中更小的那个。`Peak FLOPs` 是硬件计算峰值；`Bandwidth × Arithmetic Intensity` 是按数据搬运能力推出来的带宽上限；Arithmetic Intensity 表示每搬 1 byte 数据能做多少 FLOPs。

### 1.1 算术强度为什么重要

如果一个算子做了很多计算却只搬很少数据，它更可能接近算力上限；  
如果一个算子大量搬数据但每字节只做很少计算，它更可能受带宽限制。

这能帮你决定该优先想 tile 和 Tensor Core，还是优先想 fusion、layout、向量化与减少写回。

## 完整案例：decode attention 为什么常被 KV 带宽卡住

先看一个单层 decode attention。假设：

| 参数 | 值 |
| --- | ---: |
| batch size \(B\) | 8 |
| attention heads \(H\) | 32 |
| head dim \(D\) | 128 |
| context length \(T\) | 8192 |
| KV dtype | BF16，2 bytes |

每个请求只 decode 1 个新 token。对每层 attention 来说，主要计算有两段：\(QK^\top\) 和 \(PV\)。粗略 FLOPs 为：

\[
\text{FLOPs}_{QK}=2\times B\times H\times T\times D
=2\times8\times32\times8192\times128
\approx0.54GFLOPs
\]

这行式子在估算 \(QK^\top\) 的乘加主项：\(B\) 是 batch size，\(H\) 是 head 数，\(T\) 是历史 KV 长度，\(D\) 是 head dim；前面的 2 是常见 FLOPs 口径里一次乘加约等于 2 个 FLOPs。

\[
\text{FLOPs}_{PV}\approx0.54GFLOPs
\]

所以总计算量约 \(1.07GFLOPs\)。而 KV 读取量约为：

\[
\text{Bytes}_{KV}=2\times B\times H\times T\times D\times2
\approx1.0GiB
\]

其中前面的 2 表示 K 和 V，最后的 2 表示 BF16 每元素 2 bytes。Q、输出和少量 scale / metadata 相比 KV 读取很小，可以先忽略。

于是算术强度约为：

\[
AI=\frac{1.07GFLOPs}{1.0GiB}\approx1FLOP/byte
\]

这里 \(AI\) 是 arithmetic intensity，分子是总计算量，分母是主要 HBM 读写量；它越低，越说明每搬一次数据只做很少计算。

如果目标 GPU 的有效 HBM 带宽按 2.5TB/s 估算，那么 roofline 给出的带宽上限大约是：

\[
2.5TB/s\times1FLOP/byte\approx2.5TFLOP/s
\]

这远低于现代 GPU 的峰值 Tensor Core 算力，说明这个 decode attention 更像 bandwidth-bound，而不是 compute-bound。优化方向应优先减少 KV 读取、改善 KV layout、分页连续性、向量化加载、cache 命中和 dequant 融合，而不是只盯着 matmul 峰值算力。

### 2.1 加上实测：怎么判断偏差

假设 microbenchmark 测到该层 kernel 平均耗时 0.55ms，则：

\[
\text{Measured bandwidth}=\frac{1.0GiB}{0.55ms}\approx1.95TB/s
\]

\[
\text{Achieved FLOPs}=\frac{1.07GFLOPs}{0.55ms}\approx1.95TFLOP/s
\]

这两行式子分别把“搬了多少数据”和“做了多少计算”除以实测 latency。前者看是否接近 HBM 带宽，后者看是否接近计算峰值；两个指标一起看，才能判断瓶颈是不是和 roofline 预测一致。

这两个数字都接近 bandwidth-bound roofline 的判断：不是 FLOPs 太少，而是每一步都要扫一大段 KV。此时如果换成 FP8 / INT8 KV，把 KV bytes 理论上减半到约 0.5GiB，理想 latency 是 0.275ms。但真实可能只降到 0.33ms，因为还要付 scale 读取、dequant、layout 对齐、page table 和 softmax 等开销。

| 版本 | KV bytes | 单层 attention latency | kernel speedup | 解释 |
| --- | ---: | ---: | ---: | --- |
| BF16 KV | 1.0GiB | 0.55ms | 1.0x | HBM 读取主导 |
| FP8 / INT8 KV | 0.5GiB + scale | 0.33ms | 1.67x | bytes 降了，但 dequant 和调度还在 |

### 2.2 为什么端到端收益小于 kernel 收益

假设一个 32 层模型的 decode step 端到端是 58ms，其中：

| 部分 | baseline |
| --- | ---: |
| attention KV 读取相关 kernel | \(32\times0.55=17.6ms\) |
| MLP / GEMM / norm / residual | 24.0ms |
| runtime 调度、采样、Python/CUDA graph 边界 | 6.0ms |
| 通信、同步、batch 形状不齐 | 10.4ms |
| 合计 | 58.0ms |

把 attention KV kernel 从 0.55ms 优化到 0.33ms 后，32 层节省：

\[
32\times(0.55-0.33)=7.04ms
\]

这行式子只计算 attention KV kernel 在 32 层里节省的时间：每层少 \(0.22ms\)，乘以 32 层得到总节省。

端到端变为约 51ms，整体 speedup 是：

\[
\frac{58}{51}\approx1.14x
\]

所以“单 kernel 1.67x”最后只变成“端到端 1.14x”并不矛盾。Amdahl 定律、其他算子、新的 dequant 开销、runtime 调度和 shape 分布都会吃掉部分收益。好的性能报告必须同时写 kernel 级收益和端到端收益，并解释差额去了哪里。

### 2.3 这个案例怎么迁移到世界模型

视频世界模型 rollout 也常遇到类似问题：长 horizon 下，模型要反复读取历史 latent、KV cache 或 memory token。如果算术强度低，优化重点不是“让 matmul 更炫”，而是压缩 cache、减少重复读取、把当前窗口和远期 memory 分层、融合 dequant 和 attention，并用真实 rollout shape 做 benchmark。

### 2.4 世界模型案例：视频 latent rollout 什么时候不是带宽瓶颈

再看一个和 decode attention 相反的例子。假设一个 action-conditioned 视频世界模型要同时评估 16 条候选动作序列，每条使用 8 帧历史 latent，每帧 latent grid 为 \(16\times16=256\) token，因此每条序列长度：

\[
L=8\times256=2048
\]

这里 \(L\) 是每条候选序列的 token 长度：8 帧历史，每帧 256 个 latent token，所以一条序列共有 2048 个 token。

模型 hidden size 为 1024，16 个 attention head，head dim 64，BF16。只看一个 Transformer block 的全注意力和投影，粗略计算：

| 项 | 估算 |
| --- | ---: |
| candidates \(B\) | 16 |
| sequence length \(L\) | 2048 |
| hidden size \(d\) | 1024 |
| attention matmul FLOPs \(QK^\top + PV\) | \(\approx 275GFLOPs\) |
| QKV + output projection FLOPs | \(\approx 275GFLOPs\) |
| block total FLOPs | \(\approx 550GFLOPs\) |
| input latent bytes | \(16\times2048\times1024\times2 \approx 64MiB\) |
| naive score matrix bytes | \(16\times16\times2048^2\times2 \approx 2.0GiB\) |

如果用 FlashAttention 避免 score matrix 落 HBM，主要字节不是 2.0GiB 的 score，而是 Q/K/V、输出和若干中间状态。即使按 350MiB 有效读写估算，算术强度也大约是：

\[
AI\approx\frac{550GFLOPs}{350MiB}\approx1500FLOPs/byte
\]

这行式子和前面的 decode attention 对照：分子是一个 block 的粗略计算量，分母是 FlashAttention 路径下的有效读写量。\(AI\) 高到约 1500 FLOPs/byte，说明瓶颈更可能转向计算、occupancy 或调度，而不是单纯 HBM 读取。

这和上一节 decode attention 的 \(1FLOP/byte\) 完全不同。decode attention 扫 KV，常是 bandwidth-bound；这个视频 latent full-attention block 更接近 compute-bound 或调度/occupancy-bound。此时如果只做 FP8 latent cache，收益可能不如减少 token 数和 attention 结构本身。

假设把视觉 tokenizer 和 memory selection 改掉，只保留 3 帧高分辨率接触窗口、3 帧中分辨率历史和低频布局 memory，把有效 token 从 2048 压到 768。attention 项按 \(L^2\) 缩放，projection 项按 \(L\) 缩放：

| 版本 | \(L\) | attention FLOPs | projection FLOPs | block total | 假设实测 latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原始 latent | 2048 | 275G | 275G | 550G | 4.1ms |
| token 压缩后 | 768 | 39G | 103G | 142G | 1.35ms |

kernel 级 speedup 约为：

\[
\frac{4.1}{1.35}\approx3.0x
\]

这个比值是 kernel 级 speedup：只比较 video latent transformer block 自身的 latency，不包含 tokenizer、候选排序、memory 写回和其他头部开销。

但端到端 rollout 未必有 3 倍。假设一次候选动作评估的总耗时是 38ms：

| 部分 | baseline | token 压缩后 |
| --- | ---: | ---: |
| video latent transformer blocks | 18ms | 7ms |
| VAE / tokenizer / connector | 5ms | 5ms |
| action head / risk head | 4ms | 4ms |
| runtime 调度和候选排序 | 6ms | 6ms |
| memory 读写和 replay 记录 | 5ms | 7ms |
| 合计 | 38ms | 29ms |

端到端 speedup 只有 \(38/29\approx1.31x\)。原因是 token 压缩后 transformer 变快了，但 connector、risk/action head、候选排序和 memory 写回占比上升；如果压缩还引入额外 gather/scatter 或多尺度 memory 读取，系统收益会继续被吃掉。

这个例子给世界模型算子优化一个很具体的判断：先算 \(L^2\) attention、projection 和 latent bytes，再决定是做低比特 cache、token pruning、局部 attention，还是改 episode tokenization。对视频 rollout，减少无关 token 常常比单纯压 dtype 更有效；但接触帧、风险帧和动作分叉附近 token 不能乱剪，否则 kernel 变快了，闭环成本可能更高。

### 2.5 硬证据模块：Roofline 案例要写到任务指标

Roofline 报告的最低格式是 shape、FLOPs、bytes、arithmetic intensity、实测带宽和端到端收益。世界模型还要追加任务证据。

| 证据项 | 最小记录 | 世界模型追加项 |
| --- | --- | --- |
| 本页解决哪项成本 | bandwidth-bound、compute-bound、launch-bound 中哪一个 | 是压 rollout KV、视频 latent attention，还是 tokenization |
| 可复算数字 | shape、dtype、FLOPs、bytes、AI、latency | 候选动作数、horizon、相机/latent token 数 |
| 失败案例 | kernel speedup 大但端到端小 | 端到端小之外，还要看动作排序是否变坏 |
| 证据等级 | microbenchmark 或 production trace | 是否有 hard replay / closed-loop task bucket |
| 验收指标 | achieved bandwidth / FLOPs、Amdahl breakdown | action ranking、risk recall、latent drift、cost per success |

因此，一个完整结论应像这样：`该 video latent block 的 AI 约 1500 FLOPs/byte，更偏 compute/occupancy；token 压缩让 block latency 4.1ms -> 1.35ms，但端到端 rollout 38ms -> 29ms。hard replay 中 contact event F1 不降、top-1 candidate ranking 不翻转，才允许上线。`

## Benchmark 设计的三层目标

一个有价值的 benchmark 应至少服务三个目标：看单 kernel 极限，看真实 shape 分布下的稳定收益，看端到端系统是否真的受益。

如果只满足第一个，很容易得到“实验室很快，线上不明显”的结论。

## 微基准的基本规范

较稳妥的 microbenchmark 通常要充分 warmup，固定 shape、dtype 和 layout，区分单 kernel latency 与 end-to-end latency，多轮重复，避免把 JIT 或首次编译成本算进去，并报告环境。

## 一个 GEMM 案例怎么看

面对 GEMM 案例时，建议先看 \(M/N/K\) 多大、shape 是否规则、是否是高频服务热点、是否已经走 Tensor Core，以及 epilogue 是否可融合。

如果是规则大矩阵，roofline 往往提示它更偏 compute-heavy；  
如果是小而碎的 GEMM，真正的问题可能反而是 launch 和 shape。

## 一个 Softmax / Norm 案例怎么看

这类算子的关键问题通常不是 FLOPs，而是数据读写次数、reduction 路径、向量化和是否可融合。

roofline 常提醒你：这类算子更可能是 bandwidth-bound，因此优化方向应偏向减少中间写回、提高向量化、改善访存局部性和融合周边操作。

## 一个 Attention 案例怎么看

Attention 常常是混合型案例：数学公式看起来计算很多，但真正性能又常被 I/O 主导；长上下文下 score matrix materialization 很痛，decode 场景下 KV cache 读取更痛。

因此对 attention 来说，roofline 的价值在于提醒你先拆阶段：prefill、decode、paged KV 和 prefix reuse 的访问模式并不相同。

## 端到端收益为什么常常小于单 kernel 收益

原因通常是该 kernel 在整体中占比有限，优化后出现了新瓶颈，额外引入了数据准备成本，或者调度和通信没有同步优化。

这也是为什么好的案例拆解，不会只给“kernel 提速 2 倍”，还会解释端到端提升多少、为什么不是同等倍数，以及剩余瓶颈在哪里。

## 如何把案例拆解成可复用方法

面对任何优化案例，都可以问四件事：它解决的是带宽问题、算力问题，还是 launch / layout 问题；它复用了 tiling、fusion、online reduction、persistent、specialization 中的哪些通用模式；它依赖什么 shape、dtype、拓扑和请求分布边界；它能迁移到哪些别的算子上。

这样案例才会变成“方法库”，而不是孤立技巧。

## 直觉例子

Roofline 像是在看一台机器究竟是马力不够，还是输送带太慢；benchmark 像是在不同工况下测这台机器；案例拆解则是在弄清楚某次改造到底是换了更强的引擎，还是把物流路径改顺了。只有把这三件事放在一起，性能优化才会从“感觉更快”变成“知道为什么更快”。

## 阅读结论

Roofline 建模与性能案例的核心意义，在于把性能优化从“经验活”变成“结构化分析”。只要你能先判断瓶颈类型，再设计可信 benchmark，再把案例抽象成可复用模式，就会比单纯堆技巧更接近真正成熟的算子工程。

## 工程收束

Roofline 不是展示图，而是把 FLOPs、带宽、算术强度、steady state 和真实 workload 对齐的排查工具。最容易出错的是字节数估错、只跑理想 shape、忽略服务态请求；上线前应同时做理论估算和实测，用 case study 解释偏差，并保留基准脚本。
