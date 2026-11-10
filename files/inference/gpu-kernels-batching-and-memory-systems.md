# 同一层在prefill和decode里为何有不同成本

一个4096×4096的BF16线性层，权重主体32MiB。Prefill可一次处理许多输入token，低batch decode可能每次只处理一行。权重没变，每次读入后能复用的次数却变了。

## 从矩阵shape算出量级

对 $Y=XW$，X为M×4096，主计算约 $2M\times4096^2$ FLOPs。M=1时约3355万FLOPs，至少读取32MiB权重，理想强度约1FLOP/byte；M=128时计算增加128倍，而权重仍有机会只读一次，复用潜力大得多。

真实执行还涉及输入输出、缓存、tile与指令选择。这个算账解释瓶颈可能不同，不保证某个GPU达到理想下界。

把同一层放进一个简化 Roofline 里，可以看到为什么 batch 会改变瓶颈。若权重读取为 32 MiB、计算量为 33.6 MFLOPs，算术强度约为 1 FLOP/byte；即使 GPU 的峰值算力很高，内存读取下界仍可能占主要部分。把 M 从 1 增到 128，计算量和输入 token 同时增加，权重如果能在 cache 或片上存储中复用，强度会提高；如果权重每轮都从 HBM 重新读取，理论收益就会被打折。

这个估算只用来选择测量方向。真实 kernel 还会读写输出、处理 bias 和 activation、做边界 mask，权重也可能因为 tile 排布被重复加载。报告时应给出“逻辑字节数”和 profile 中的实际读写指标，不能把公式下界当作硬件实测带宽。

## Batch把权重读取分给更多请求

持续批处理可把多条正在decode的请求合成一轮，让同一权重用于更多行。吞吐可能提高，但新请求还要等待调度；长短请求混合又会改变KV读取与尾延迟。

静态batch等最长请求全部完成再一起结束，会让短请求占着槽位。Continuous batching可以在迭代间移除完成请求、补入新请求，但每轮工作量与状态也更动态。

## KV给并发设另一条上限

32层、8个KV头、头维128、BF16时，每token KV主体128KiB。32k历史的单请求约4GiB，四条约16GiB。即使权重能放下，并发也可能被KV挤满。

按页分配减少连续大块预留需求。页大小为4、长度分别5和7的两请求，各需2页，总分配16个槽位、有效12个，仍有4个尾部空槽。分页改善分配，不会消除所有碎片。

如果页大小改成 16，同样两条请求各只占一页，但一条长度 5、一条长度 7 的尾部浪费变成 20 个槽位；页更小可以减少尾部浪费，却增加 block table 条目和地址计算。页大小的选择要和请求长度分布、KV 访问 kernel 以及 allocator 元数据一起测，不能只看单请求的理论占用。

![vLLM logical and physical block table 原论文图](../assets/images/paper-figures/inference/vllm-logical-physical-block-table.png){ width="760" }

<small>图源：[Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)，Figure 5。原论文图意：vLLM 用 block table 把 logical KV blocks 翻译到 physical KV blocks，使请求的 KV cache 不必连续存放。</small>

## 量化和融合分别改哪段数据流

权重量化减少常驻和读取字节，但增加解包、scale或转换。KV量化作用于历史状态，精度与页面布局另有约定。融合减少中间写回与launch，但可能增加寄存器占用。

先用trace确认时间花在权重GEMM、attention扫描KV、转换还是调度。只看峰值算力或显存剩余，无法决定该优化哪一项。

![Low-bit cache hierarchy data transmission 原图](../assets/images/paper-deep-dives/inference/low-bit-llm-survey/figure-4-cache-hierarchy.png){ width="690" }

<small>图源：`A Survey of Low-bit Large Language Models`，Figure 4。原论文图意：以 NVIDIA A100 为例展示 weight 和 activation 在 host、global memory、L2、shared memory、registers 与 MatMul kernel 之间的数据传输。</small>

![KV cache quantization 原图](../assets/images/paper-deep-dives/inference/low-bit-llm-survey/figure-6-kv-cache.png){ width="620" }

<small>图源：`A Survey of Low-bit Large Language Models`，Figure 6。原论文图意：展示 quantized KV cache 如何减少缓存存储和数据传输，并在 attention forward 前 dequantize。</small>

## 同一个实验同时保留两种边界

Microbenchmark固定shape、dtype与layout，验证kernel本身；完整请求测试加入队列、continuous batch和真实长度分布，验证系统收益。若单kernel更快但端到端无改善，检查其占比以及新引入的转换、padding或等待。

可以用一个两阶段的时间账本避免混淆。假设 prefill 需要 30 ms，decode 需要 70 ms，量化只把 decode kernel 从 70 ms 降到 50 ms，那么整条请求从 100 ms 降到 80 ms，端到端收益是 20%。如果队列等待平均 120 ms，用户看到的时间从 220 ms 降到 200 ms，收益只剩约 9%。同一个 kernel 优化在不同到达率下会呈现不同的用户收益。

因此压测报告要同时列出服务前的计算时间和用户端的请求时间。P50 变快但 P99 变慢，常见原因是新的 kernel 或更大的 batch 增加了资源争用；只报平均 tokens/s 会把这类尾部变化隐藏掉。


## 继续读

要把 GPU 时间线接到服务指标，可以继续看[注意力与 KV kernel](../operators/serving-attention-and-kv-kernels.md)、[通信 kernel 与重叠](../operators/communication-kernels-and-overlap.md)和[推理成本模型](cost-modeling-and-slo-design.md)。

## 对照源码与延伸阅读

- [GEMM的形状、tile与复用](https://github.com/triton-lang/triton/blob/cd2e01c6686d6503802e6cdfac56d5a5dfdb27a1/python/tutorials/03-matrix-multiplication.py)
- [动态请求与KV预算共同调度](https://github.com/vllm-project/vllm/blob/fa3b9d1377219247060d3c8b49eae199c9e7e154/vllm/v1/core/sched/scheduler.py)

## 用 Roofline 先判断应该优化哪里

对一个 kernel，算术强度可以粗略写成

\[
I=\frac{\text{FLOPs}}{\text{bytes moved}}.
\]

若 (I) 很低，增加理论 FLOPs 峰值通常帮助不大，应该先看权重/KV 读取、布局和缓存；若 (I) 较高但实际利用率低，再检查 tile、占用率、寄存器和 kernel launch。Roofline 只是筛选假设，真实 kernel 还会受边界、同步、稀疏和多 stream 影响。

## Prefill 与 decode 的 batch 形状不同

Prefill 通常有较大的 (M)，适合把权重读取分摊给许多 token；decode 每个请求每轮新增一个或少量 token，(M) 可能接近活动请求数，且每次都要读取历史 KV。Continuous batching 改变的是每轮参与的请求集合，不会自动让 decode 变成 prefill 的矩阵形状。

测试时至少按以下维度分桶：输入长度、输出长度、活动请求数、KV 命中、quant dtype 和 kernel 路径。平均一个 tokens/s 数字会把 prefill 与 decode 的收益混在一起。

## kernel 优化的验收顺序

1. **数值对照**：固定输入与 dtype，比较 max error、相对误差和 NaN/Inf。
2. **shape 对照**：覆盖常见与边界 M/N/K、非整 tile、不同 stride。
3. **微基准**：预热后测稳定区间、p50/p95 和 launch 次数。
4. **算子链**：检查融合后是否减少中间写回和同步。
5. **端到端**：放回真实 batch、调度、KV 和请求 trace。

单个 kernel 加速 2 倍，若只占总时间 5%，端到端理论上也只有有限收益；若融合改变了寄存器占用，可能在另一个 shape 上退化。报告要列出覆盖的 shape 和设备。

## 来源与改编说明

- [Triton 矩阵乘教程](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)用于对照 tile、指针和 autotune 的最小实现。
- [Triton fused softmax 教程](https://github.com/triton-lang/triton/blob/cd2e01c6686d6503802e6cdfac56d5a5dfdb27a1/python/tutorials/02-fused-softmax.py)用于理解减少中间写回的融合路径。
- [PagedAttention](https://arxiv.org/abs/2309.06180)用于连接 KV 分页、并发和内存系统。

## 自测

1. 为什么 decode 的低 batch 矩阵乘更容易受内存和 launch 影响？
2. 如何用算术强度判断先测 GEMM 还是先测 KV 读取？
3. 一个 kernel 微基准变快但端到端不变，下一步要补哪些证据？
