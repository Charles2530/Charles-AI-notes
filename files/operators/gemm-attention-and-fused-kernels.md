# 从矩阵乘的一个输出块理解融合

矩阵乘的公式只有一行 $C_{ij}=\sum_k A_{ik}B_{kj}$，kernel 却要同时决定输出 tile、数据复用和累加器寿命。把这三个决定写清楚以后，GEMM、attention 和 epilogue fusion 的相同点与差别就能落到地址和依赖上。

## 四乘四矩阵里的两乘两输出块

设A、B都是4×4，当前program负责C的前两行、前两列。沿K每次走2个位置，先读A左上2×2和B左上2×2，得到部分和；再读A的前两行后两列与B的后两行前两列，加入同一累加器。

这块C有4个输出，各需要4次乘加，共16次乘加，按一次乘法和一次加法各计1FLOP，共32FLOPs。A的一个元素在块内参与两列输出，B的一个元素参与两行输出。Tile让这份复用在片上发生。

官方Triton矩阵乘教程按这一结构执行，输出块循环外创建FP32 accumulator，K循环中 `tl.dot` 累加，循环后才写回。它还通过调整program遍历顺序改善L2复用，但这不会改变数学上的元素对应关系。

## 先把 shape 和 stride 分开写

逻辑公式与内存布局是两件事。以行主序为例，

\[
A\in\mathbb R^{M\times K},\quad
B\in\mathbb R^{K\times N},\quad
C\in\mathbb R^{M\times N},
\]

元素地址可以写成

\[
\operatorname{addr}(A_{i,k})=A_0+i s^A_M+k s^A_K,
\]

其中 stride 是以元素为单位的步长。转置 view 可能让 \(s^A_M\) 与 \(s^A_K\) 交换，逻辑 shape 仍然可以显示成 `[M,K]`。kernel 若只按连续布局计算，结果会在某些 shape 上悄悄错误。

一个小测试可以同时覆盖连续和转置输入：让 `A[i,k] = 10*i + k`、`B[k,j] = 100*k + j`，用高精度 Python 参考逐元素比较。不要只用全 1 矩阵，那个输入会掩盖轴交换。

tile 形状 `[BM, BN, BK]` 也要和边界 mask 一起记录。若 M 不是 BM 的整数倍，最后一个 tile 会包含无效行；mask 只保护加载和存储，不会自动修正错误的 pointer 计算。先让一个小矩阵只落在边界 tile，再恢复较大 shape，通常比直接跑 benchmark 更快发现问题。

## 用M的变化看计算强度

设BF16权重为4096×4096，主体32MiB。M=1时，主要计算约3355万FLOPs，若至少读一遍权重，算术强度约1FLOP/byte。M=128时，同一权重服务128行输入，计算约42.95亿FLOPs，忽略较小输入输出时强度约128FLOPs/byte。

实际字节还包括输入、输出、重复读取和缓存行为，这里只是理想复用估算。但它已经说明同一层在decode与prefill中可能有不同瓶颈，不能只看N和K相同就复用性能结论。

## 累加精度是接口的一部分

输入可以是 FP8 或 BF16，累加器却常用 FP32。若每一项乘积量级约为 1，K=4096 时直接用低精度累加会让舍入误差经历数千次叠加；FP32 累加通常能把这部分误差压低，再按输出 dtype 写回。具体误差仍取决于输入分布、缩放和硬件指令。

因此数值对照至少保存三份结果：高精度参考、低精度输入但高精度累加、最终 kernel 输出。第一份与第二份的差异属于量化和输入转换，第二份与第三份的差异才更接近 kernel 实现或累加路径。只比较最终输出，无法知道误差来自哪里。

对于 `GEMM+bias+GELU`，激活函数还可能改变误差的放大方式。测试应覆盖接近 0、较大正值、较大负值和边界 tile，而不是只在随机输入上报一个最大误差。

## Epilogue 可以省掉哪次写回

若需要 $Y=\operatorname{GELU}(AB+b)$，分开实现可能先把AB写到显存，再读出加bias和激活。若把bias和激活接在GEMM的最终累加器之后，便可直接写Y，减少中间结果的一次写入和一次读取。

对于 $M\times N$ 的BF16中间结果，理论上可省 $4MN$ 字节。收益还要扣除新增寄存器压力、特殊函数计算与对原GEMM调度的影响。若融合迫使高效库GEMM退回较差实现，节省的读写可能不足以补偿计算损失。

判断是否融合可以先列一个成本表：

| 项目 | 分开实现 | 融合实现可能的变化 |
| --- | --- | --- |
| 中间结果写回 | 写一次，后续再读一次 | 可能省掉 |
| 累加器存活时间 | GEMM 结束即释放 | 需要继续保存到 epilogue |
| 寄存器压力 | 较低 | bias、激活和临时量增加 |
| 调度选择 | 可直接使用成熟 GEMM | 可能限制 tile 或 occupancy |
| 数值路径 | 每层边界明确 | 激活和舍入顺序改变 |

这张表的作用是提醒实验不要把“少一次显存访问”直接写成“必然更快”。最终要用相同输入、相同 dtype 和相同输出误差做端到端测量。

## Attention 多了归一化依赖

Attention先算 $S=QK^\top/\sqrt d$，再按行softmax，最后算PV。它不能把任意一个小score块独立softmax后直接相加，因为分母取决于整行所有key。

FlashAttention用在线最大值、分母和输出累积修正这一依赖，既保持全行归一化，又不把完整S与P写回。具体逐块计算见[FlashAttention](long-context-and-flashattention-evolution.md)。它仍执行dense attention的配对计算，节省主要来自中间矩阵I/O和执行组织。

在线 softmax 可以用三个量说明。处理到当前 tile 时，保存行最大值 \(m\)、未归一化分母 \(l\) 和未归一化 value 累加 \(u\)。新 tile 的局部统计量为 \(m_b,l_b,u_b\)，合并时令

\[
m' = \max(m,m_b),
\]

\[
l'=e^{m-m'}l+e^{m_b-m'}l_b,
\qquad
u'=e^{m-m'}u+e^{m_b-m'}u_b.
\]

最后输出为 \(u'/l'\)。旧统计量和新统计量都按新的最大值重新缩放，所以不需要把所有 score 留在显存中。若把每个 tile 的 softmax 直接相加，就丢失了跨 tile 的分母，结果不会等于完整行 softmax。

这个推导还给出一个很小的数值测试。把一行 score 分成两块，分别用在线公式和完整矩阵 softmax 计算，比较输出和最大值。测试应覆盖一个 tile 的分数远大于另一个 tile、分数全为负，以及 mask 使某些位置无效的情况。

## 比较实现时保留完整表达式

如果目标是 `GEMM+bias+GELU`，基准就应比较相同表达式、相同dtype和相同输入layout。单测裸GEMM后再把其速度宣称为完整层速度，会漏掉转换和后处理。先检查数值误差，再分别测kernel和完整层，才能判断融合究竟改变了哪项成本。

## 性能测试要把启动和同步写进协议

GPU kernel 的第一次调用可能包含 JIT、模块加载和缓存建立。计时前先预热固定次数，并在每次区间前后使用设备同步；否则 CPU 计时可能只测到提交命令的时间。报告至少包括 warmup 次数、重复次数、p50、p95、输入 shape、dtype、设备和是否包含转换。

一个可回放的 benchmark 卡片可以写成

```text
expression: matmul + bias + gelu
shape: M=128, K=4096, N=4096
layout: A row-major, B transposed-view
dtype: bf16 input, fp32 accumulate, bf16 output
timed_region: end_to_end
warmup: 50
repeats: 200
metrics: p50, p95, max_abs_error
```

对 decode 还要记录 batch 和有效 token 数，对 prefill 还要记录序列长度。只改变 M 而保留其他条件时，才能观察算术强度变化；否则缓存、padding 和调度变化会混在一起。

融合 kernel 的收益也要与显存峰值一起报告。寄存器或共享内存增加可能降低 occupancy，单次延迟下降却让并发请求的吞吐变差。服务系统中还需测批处理后的 p95，而不是只测单个 kernel 的最短时间。

![FlashAttention IO-aware tiling 原论文图](../assets/images/paper-figures/operators/flashattention-io-aware-tiling.png){ width="760" }

<small>图源：[FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)，Figure 1。原论文图意：左侧展示 FlashAttention 如何按块把 Q/K/V 放入 SRAM，避免 materialize 完整 attention matrix；右侧展示在 GPT-2 attention 上相对 PyTorch 实现的加速效果。</small>

## 对照源码与延伸阅读

- [Triton GEMM 的块循环与融合入口](https://github.com/triton-lang/triton/blob/cd2e01c6686d6503802e6cdfac56d5a5dfdb27a1/python/tutorials/03-matrix-multiplication.py)
- [Attention 的在线归约](https://github.com/triton-lang/triton/blob/cd2e01c6686d6503802e6cdfac56d5a5dfdb27a1/python/tutorials/06-fused-attention.py)

如果想继续往实现边界走，可以看[CuTe 的 layout 与 local_tile](cutlass-cute-and-compiler-stack.md)理解线程片段，再看[长上下文与 FlashAttention 演进](long-context-and-flashattention-evolution.md)核对在线 softmax 的跨 tile 统计量。融合的收益要回到完整层和服务 p95，不能只看一个 kernel 的最短时间。

## 从一个输出块继续做实验

固定 4×4 小矩阵，分别实现裸 GEMM、GEMM+bias、GEMM+bias+GELU。先比较每个版本的输出，再测中间结果写回次数。随后改变 batch，把同一权重用于更多行，观察算术强度为何变化。这个过程把“融合减少 I/O”与“融合增加寄存器压力”同时保留下来。

Attention 不能把每个 score tile 独立归一化。任何在线 softmax 实现都必须维护跨 tile 的最大值和归一化统计量，并证明它与完整行 softmax 等价到允许的数值误差内。

## 来源与改编说明

- **GEMM 实现**：[Triton matrix multiplication](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)。
- **Attention 机制**：[FlashAttention](https://arxiv.org/abs/2205.14135)；论文图表和 benchmark 只在原硬件与设置下成立。
- **自测**：列出融合后可能减少的一次写回和一次读取，再列出可能增加的寄存器、特殊函数或调度成本。
