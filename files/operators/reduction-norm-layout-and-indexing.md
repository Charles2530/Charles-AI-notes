# Reduction、Norm、Layout 与 Indexing：小算子为什么能拖慢大模型

大模型系统里最容易被低估的，不一定是最大的 GEMM，而是那些每层、每步、每个 token 都会经过的小算子：reduction、softmax、LayerNorm、RMSNorm、RoPE、gather、scatter、layout transform、pack/unpack。它们单次 FLOPs 不多，但会反复读写整行 hidden state、KV cache 或 routing buffer；在 decode、小 batch、长上下文和 MoE 场景里，存在感会被放大。

这页先不从“有哪些 kernel”开始，而是从一条数据流开始：一个 token 的 hidden state 进入 Transformer block 后，除了 GEMM 和 attention，还要不断经历“读一整行、算统计量、改 layout、按索引重排、再写回”。如果这些步骤被拆成许多小 kernel，GPU trace 里就会出现密密麻麻的短条，每条都不大，却让端到端 latency 下不来。

## 先跟着一行 hidden state 走一遍

假设有一行 hidden state：

\[
x \in \mathbb{R}^{D}
\]

这里的 \(x\) 表示一个 token 在某一层里的 hidden state，\(D\) 是 hidden size。它可能先做 residual add，再做 RMSNorm，然后进 QKV projection；projection 后 Q/K 要做 RoPE；attention 前要算 softmax；MoE 层要按 expert routing gather/scatter token；低比特路径还要 dequant、requant 或读取 scale。这里很多操作都不是大矩阵乘，而是围绕一行、一块、一组 token 做统计和搬运。

| 操作 | 看起来在算什么 | 真正考验什么 |
| --- | --- | --- |
| Reduction | sum、max、mean、variance、argmax | warp/block 协作、同步、累加精度 |
| Softmax | 指数归一化 | max/sum reduction、数值稳定、分块读写 |
| LayerNorm / RMSNorm | 每行归一化 | 行内统计量、向量化 load/store、融合 residual/scale |
| RoPE | 旋转 Q/K 维度 | 紧贴 projection，避免额外写回 |
| Gather / Scatter | 按索引读取或写回 | 不规则访存、coalescing、cache 命中 |
| Layout transform | transpose、permute、contiguous、pack | stride、对齐、中间张量、后续 kernel 能否吃满 |

小算子的共同点是：它们通常更像带宽问题，而不是算力问题。优化的第一步不是问“还能不能少算几个 FLOPs”，而是问“这行数据被从 HBM 读写了几次，中间值有没有机会留在 register 或 shared memory 里”。

## Reduction：很多数压成少数统计量

Reduction 的任务是把一组值压成更少的统计量，例如 sum、max、mean、variance、argmax，或 top-k 之前的局部最大值。它天然需要线程协作：每个 thread 只能先处理自己负责的一段，再把局部结果汇总到 warp、block，必要时再做跨 block 的第二阶段归约。

| 层级 | 适合什么 | 风险 |
| --- | --- | --- |
| warp 内 reduction | 一行或一小块数据能被少量 warp 覆盖 | 分支和 mask 要正确处理尾部元素 |
| block 内 reduction | hidden size 较大、需要共享内存暂存 | shared memory bank conflict、同步开销 |
| 多 block reduction | 行很长或全局统计 | 原子操作、第二阶段 kernel、非确定性 |

Reduction 还会改变数值行为。浮点加法不满足严格结合律，归约顺序一变，低位误差也会变。训练里做 loss、grad norm、variance、softmax sum 时，如果累加精度太低或极值处理不稳，单个 kernel 的微小误差会在很多层、很多 step 后放大。

所以 reduction kernel 的正确性不只看结果“大致接近”。上线前要覆盖极值输入、空 mask、非 2 的幂长度、不同 dtype、不同 stride、不同 batch/head layout，以及是否接受非确定性。

## Softmax：不是一行公式，而是两次归约加稳定化

Softmax 公式很短：

\[
\mathrm{softmax}(x)_i =
\frac{\exp(x_i)}{\sum_j \exp(x_j)}
\]

这个式子表示：第 \(i\) 个位置的输出，是它自己的指数分数除以整行所有指数分数之和。高性能实现不能直接照公式先 `exp` 再求和。为了避免指数溢出，通常先做 max reduction：

\[
m=\max_j x_j
\]

再计算：

\[
p_i=\frac{\exp(x_i-m)}{\sum_j \exp(x_j-m)}
\]

这意味着 softmax 至少要处理最大值统计、指数、sum 统计和归一化写回。长序列 attention 里，如果把整行 score 矩阵写到 HBM，再读回来做 softmax，再写回概率矩阵，I/O 会非常重。FlashAttention 一类算法的直觉就是把 softmax 统计量在线维护在 tile 流程里，让 score 不必完整 materialize。

读 softmax kernel 时，先看三件事：它是否用稳定 max；是否分块维护 running max 和 running sum；是否把 mask、scale、dropout 或 causal boundary 放进同一条数据流。只看 `exp` 的成本，会错过真正的瓶颈。

## LayerNorm / RMSNorm：公式简单，数据流不简单

LayerNorm 对一行 hidden state 做均值和方差统计：

\[
\mu=\frac{1}{D}\sum_i x_i,\qquad
\sigma^2=\frac{1}{D}\sum_i (x_i-\mu)^2
\]

这里的 \(\mu\) 表示这一行 hidden state 的均值，\(\sigma^2\) 表示这一行内部的方差，二者都要先通过行内 reduction 得到。然后输出：

\[
y_i=\gamma_i\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta_i
\]

RMSNorm 去掉均值中心化，只用均方根尺度：

\[
y_i=\gamma_i\frac{x_i}{\sqrt{\frac{1}{D}\sum_i x_i^2+\epsilon}}
\]

这也是为什么 RMSNorm 在许多 LLaMA-style 模型里更常见：它少算一个均值相关统计量，路径更短，和 residual/scale 融合更自然。但 RMSNorm 不是“免费”。它仍要读整行、做平方和 reduction、开方、缩放，再写回；低精度下还要决定统计量是否用 FP32 累加。

Norm kernel 的工程画像通常是：

| 关注点 | 为什么重要 |
| --- | --- |
| 行是否连续 | 决定 load/store 能否向量化 |
| hidden size 是否对齐 | 决定向量宽度、尾部 mask 和特化路径 |
| 统计量累加 dtype | 决定低精度稳定性 |
| 是否融合 residual / scale / cast | 决定是否少一次 HBM 往返 |
| 是否被图编译器捕获 | 决定端到端是否真的减少 launch |

如果 trace 里 norm 本体很快，但整层仍然慢，常见原因是 norm 前后多了 cast、contiguous、dequant、residual add 或 framework fallback。优化 norm 时要看整条邻近数据流，而不是只替换一个 kernel。

## Layout：`view` 很便宜，`contiguous` 可能很贵

PyTorch 里很多张量操作看起来只是改 shape，但真正成本取决于 stride。`view` 通常要求底层存储能按新 shape 解释；`permute` 可以只改 stride；但如果后续 kernel 需要连续布局，调用 `contiguous()` 就可能触发真实拷贝。

这件事在 GPU kernel 里更尖锐。Layout 决定 thread 访问地址是否连续、是否能 coalesced access、是否能向量化、Tensor Core 是否拿到合适矩阵布局、shared memory 是否有 bank conflict。一个主算子 benchmark 很快，不代表端到端快；如果前后不断做 transpose、pack、unpack 或 swizzle，收益会被搬运吃掉。

| 症状 | 可能原因 | 排查方式 |
| --- | --- | --- |
| GEMM 很快，整层慢 | 前后有 layout transform | 在 Nsight Systems 看 `transpose/contiguous/permute` 短 kernel |
| 自定义 kernel 对某些输入慢 | stride 非预期或对齐不足 | 打印 shape、stride、storage offset、dtype |
| `torch.compile` 后收益不稳 | dynamic shape 或 layout 变化导致 fallback | 看 graph break、kernel cache 和 fallback count |
| Tensor Core 利用率低 | dtype 对了但 layout/tile 不匹配 | 用 Nsight Compute 看 memory pattern 和 tensor pipe |

Layout 不是“排版细节”，它是 kernel 接口的一部分。写算子文档时，必须说明支持哪些 stride、哪些 memory format、哪些 alignment，哪些输入会被复制或退回慢路径。

## Indexing、Gather 与 Scatter：不规则访存会改变问题形状

Gather/scatter 类操作的难点不在算术，而在地址不可预测。Embedding lookup、MoE token dispatch、top-k、采样、KV page table、稀疏 attention、packing/unpacking 都会引入索引。

对 GPU 来说，规则连续访问最好；不规则访问会让相邻 thread 读不同位置，降低 coalescing，cache 命中也更难预测。scatter 还可能引入写冲突，需要 atomic 或额外排序。MoE 里 token 先按 expert 分组，做 grouped GEMM，再 combine 回原顺序，真正的成本往往在 dispatch/combine/layout，而不是 expert GEMM 本身。

遇到这类 kernel，不要用 FLOPs 判断。更应该问：

1. 索引是否能排序、分桶或预处理；
2. 是否能把相邻 token 放到相邻 expert 或相邻 page；
3. gather 后的 layout 是否适合后续 GEMM；
4. scatter 是否存在写冲突或 atomic 热点；
5. 是否能把 gather、compute、scatter 的一部分融合。

如果这些问题答不上来，单独优化一个 gather kernel 很可能只是在局部变快，端到端仍被重排和同步拖住。

## Fusion 与向量化：省的是 HBM 往返和 launch

小算子很适合融合，原因不是“融合听起来高级”，而是它们经常在同一行数据上连续工作。把 residual add、norm、scale、cast 合在一起，能让中间值停在 register 或 shared memory 中；把 bias、activation、quant/dequant 放进 GEMM epilogue，能减少一次中间张量写回；把 RoPE 放进 Q/K 后处理路径，能少一次额外 kernel launch。

| 可融合链路 | 主要收益 | 需要小心 |
| --- | --- | --- |
| residual + RMSNorm + cast | 少读写 hidden state | 统计量精度、对齐和尾部 mask |
| bias + activation | 减少中间 activation 写回 | epilogue 寄存器压力 |
| RoPE + Q/K post-process | 少一次 projection 后写回 | head layout、position offset |
| dequant + matmul / norm | 减少低比特额外搬运 | scale layout、fallback、误差 |
| top-k 前局部 reduction | 减少全量 materialize | 排序精度和边界条件 |

但融合也会失败。一个 fused kernel 如果寄存器压力太高、只支持少数 shape、和 graph capture 不兼容，或者把原本可复用的中间结果藏起来，就可能让尾部 workload 变慢。好的做法是按 shape bucket 做 dispatch：高频、对齐、稳定的 shape 走 fused path，长尾 shape 保留通用实现。

## 读 trace 时的排查顺序

当你怀疑 reduction、norm、layout 或 indexing 是瓶颈，可以按下面顺序排查。

| 现象 | 先看什么 | 常见修复 |
| --- | --- | --- |
| GPU trace 里短 kernel 很密 | launch 数、graph break、是否 eager | 合并小算子、`torch.compile`、CUDA Graph |
| DRAM throughput 高但算力低 | arithmetic intensity、读写次数 | fusion、减少写回、向量化 |
| 某些 batch 特别慢 | shape bucket、尾部 mask、fallback | bucket 化、特化高频 shape |
| norm 数值偶发漂移 | 累加 dtype、epsilon、极值输入 | FP32 accumulation、边界测试 |
| softmax 出 NaN | max trick、mask、低精度 exp | 稳定 softmax、mask 审计 |
| MoE dispatch 慢 | token 分布、expert 分组、combine | 排序/分桶、grouped GEMM、减少 scatter |
| layout transform 频繁 | stride、contiguous、memory format | 上游固定布局、让后续 kernel 接受 stride |

这类优化最怕只看单 kernel microbenchmark。上线前至少要同时报告：单 kernel latency、端到端层耗时、P95/P99、支持 shape 覆盖率、fallback 率、数值误差和质量回归。小算子优化不是把一个函数写快，而是把“数据从哪里来、在哪里停、写到哪里去”重新组织好。

## 实战判断

Reduction、norm、layout transform 和 indexing kernel 的共同主题是：它们常常更受内存系统支配，调用频率又极高，因此要从数据流而不是算子名来读。一个“很小”的 kernel 可能拖慢服务，是因为它让同一行 hidden state 或 KV cache 多走了几趟 HBM，或者让后续主算子拿不到合适 layout。

如果只记住一个工作流，就是：先用 trace 找短 kernel 和 layout 搬运，再用 roofline 判断带宽或算力，再看 shape/stride/dtype 是否命中高效路径，最后才决定写 fused kernel、改 layout、做 shape specialization，还是把问题交给图编译和 runtime dispatch。

## 外部精读

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)：核对 thread hierarchy、memory hierarchy、coalescing、shared memory 和 warp-level primitives。
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)：从内存访问、并行组织和 profiling 角度理解小算子优化。
- [PyTorch Tensor Views](https://docs.pytorch.org/docs/stable/tensor_view.html)：理解 `view`、`reshape`、`permute`、`contiguous` 与 stride/copy 行为。
- [PyTorch LayerNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html) 与 [PyTorch RMSNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html)：核对 API 语义和公式边界。
- [Triton Programming Guide](https://triton-lang.org/main/programming-guide/chapter-1/introduction.html)：学习用 block、offset、mask 和 stride 表达 reduction、softmax、norm 这类 kernel。
- [FlashAttention](https://arxiv.org/abs/2205.14135)：理解在线 softmax 和 I/O-aware attention 如何把 reduction 统计量放进分块数据流。
- [Layer Normalization](https://arxiv.org/abs/1607.06450) 与 [RMSNorm](https://arxiv.org/abs/1910.07467)：理解归一化公式、训练稳定性与 RMSNorm 的设计动机。
- [智源/OneFlow：全栈 Transformer 推理优化](https://hub.baai.ac.cn/view/33449)：学习中文工程长文如何从硬件、算子、显存、KV cache 和调度分层讲解；具体事实仍回论文和官方文档核对。
