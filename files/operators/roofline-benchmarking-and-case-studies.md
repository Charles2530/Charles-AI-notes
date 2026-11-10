# Roofline怎样解释同一GPU上的两种attention

一个 attention kernel 跑得慢，先要知道它是在等算力还是等数据。Roofline 把计算吞吐和内存带宽放到同一张图上：计算运算量 F、搬运字节 M、算术强度 I=F/M，再比较 $\min(P,B I)$。它给的是上限，实际程序仍可能被并行度、访存模式和调度限制。

如果设备峰值算力为 (P)，显存带宽为 (B)，两条上限的交点为

\[
I^*=P/B.
\]

当算术强度低于 (I^*) 时，带宽线更可能先成为上限；高于 (I^*) 后，计算线更可能先成为上限。这里的“更可能”很重要，因为峰值只在对应指令、shape、占用率和数据路径都能达到时才有意义。一个 kernel 落在带宽侧，不代表所有剩余时间都能靠增加带宽解决。

假设某设备的有效峰值是 1000 TFLOP/s、带宽是 2 TB/s，则 (I^*=500\) FLOPs/byte。前面的 decode 示例 (I=1) 远低于交点，首先应查 K/V 流量、缓存复用和访存事务；prefill 的强度若接近或超过交点，才更值得查矩阵乘吞吐、Tensor Core 利用率和寄存器压力。这个数值是量级示例，不是任意 GPU 的规格。

## 单query decode的低强度从哪来

假设batch为8、query头数32、KV头数也为32、历史长度8192、head dimension128，BF16 K/V。忽略softmax等小项，QK与PV运算量为

$$
F=4\times8\times32\times8192\times128
=1{,}073{,}741{,}824\ \text{FLOPs}.
$$

每个K/V至少读取一次时，主体字节为

$$
M=2\times8\times32\times8192\times128\times2
=1{,}073{,}741{,}824\ \text{bytes}=1\ \text{GiB}.
$$

于是理想强度为1FLOP/byte。若有效带宽假设为2TB/s，读取下界约0.537毫秒，不能把这个下界写成已测延迟。

若GQA改为8个KV头、仍有32个query头，KV主体减到四分之一。理想复用下强度升为4FLOPs/byte，但实际kernel若重复读取共享KV，收益会小于该估算。

GQA 改变的是 K/V 的存储头数，不会自动把 QK 的逻辑配对数除以四。仍有 32 个 query 头时，QK 和 PV 的有效 FLOPs 要按 query 头计，K/V 字节才按 8 个头计。报告时把 `Hq`、`Hkv` 和每个 query 头对应的 KV 组写出来，避免用“头数变少”同时改掉计算和内存两个分子。

一个小的 shape sweep 可以把这个口径写清楚：

| (H_q) | (H_{kv}) | (T) | 需要观察 |
| ---: | ---: | ---: | --- |
| 32 | 32 | 2048 | MQA/GQA 前的基线 |
| 32 | 8 | 2048 | K/V 流量与实际复用 |
| 32 | 1 | 2048 | 极端共享是否引入广播开销 |
| 32 | 8 | 8192 | 长上下文下的带宽和缓存行为 |

每一行都应保持 batch、dtype、mask 和计时边界一致。若只拿最有利的组合报告 GQA 加速，读者无法判断收益来自头映射还是其他条件变化。

## Prefill的query多了，复用也变了

考虑单头长度T=4096、d=128的非因果dense attention。QK与PV主运算约 $4T^2d=8.59$ GFLOPs。Q、K、V、O四个BF16张量主体合计约4MiB。如果能有效分块复用、避免完整score写回，理想算术强度远高于单query decode。

然而完整T×T分数矩阵单份BF16就有32MiB，显式写S/P会显著增加流量。FlashAttention减少这部分I/O，但受片上容量影响，K/V块仍可能被多个query块读取，不能把4MiB直接当作真实HBM总流量。

因果mask还会减少部分有效配对，计算计数需对应实现与报告口径。报告TFLOPS时先写明使用dense还是因果有效FLOPs。

## 为什么没有达到roofline仍不能直接说kernel差

计算与带宽峰值需要合适shape、指令、数据复用和足够并行。小矩阵可能无法填满设备；不规则索引可能产生更多物理事务；寄存器spill和同步也会改变实际成本。Roofline中的M若只是逻辑张量大小，而非实际搬运，预测误差就可能来自模型本身。

因此先在相同计时边界下测kernel，再看实际带宽、指令和资源。不要从一个低util数字直接推断原因。

## 逻辑字节不等于实际 HBM 流量

公式中的 (M) 通常按张量元素数乘 dtype 字节数估算。现实 kernel 还会受到 tile 重读、cache 命中、对齐、向量化宽度和写分配的影响。一个 K/V tile 可能由多个 query tile 共享，也可能因为驻留不足被重新从 HBM 读取。

因此最好同时记录三种量：

| 量 | 来源 | 用途 |
| --- | --- | --- |
| 逻辑 bytes | shape、dtype、mask 的公式 | 定义问题和理想下界 |
| 估计 bytes | tile、缓存和重复读取模型 | 形成可检验假设 |
| 实际 bytes | profiler 或硬件计数器 | 验证流量是否真的发生 |

如果实际 bytes 远高于逻辑下界，先检查 stride、tile 顺序和 cache 复用；如果 bytes 接近下界但速度仍低，才把注意力转向指令吞吐、同步或并行度。没有计数器时可以保留“未知”，不要把逻辑字节冒充成测量值。

## 算子加速如何折算到整步

教学设定中，attention占完整请求时间30%，其余70%不变。如果attention提速2倍，整体加速为

$$
S=\frac1{0.7+0.3/2}\approx1.176.
$$

若加入布局转换、队列变化或并发资源竞争，还可能更小。反之，节省显存让batch增大时，也可能出现系统级额外收益。这个公式只描述固定串行占比的情形。

## 一个可比较的结果应包含什么

记录shape、head映射、mask、dtype、layout、设备和软件版本；分别报告预热后的分布与冷启动成本；给出数值误差、逻辑FLOPs/bytes定义和完整层耗时。本文没有附GPU实测，因此所有时间下界都只用于检验计算与单位。

基准还要固定同步方法和重复策略。GPU 异步提交会让普通 CPU 计时低估耗时，第一次调用又可能包含 JIT 或缓存建立。至少分开保存 `warmup_ms`、稳态 p50/p95、冷启动时间和 batch 内有效 token 数。

若优化前后的数值误差、逻辑 FLOPs 口径或 mask 不同，TFLOPS 不能直接比较。因果 attention 可以少算一部分配对，也可以只在实现中跳过写回；两者的有效 FLOPs 定义要写在结果卡片里。

```text
kernel: fused_attention
shape: B=8, Hq=32, Hkv=8, T=8192, d=128
mask: causal
dtype: bf16
logical_flops: ...
logical_bytes: ...
measured_hbm_bytes: ...
warmup: 50
steady_repeats: 200
latency_p50_p95: ... / ...
accuracy_check: max_abs_error=...
```

这张卡片让一次 roofline 判断可以被复算，也能指出模型、kernel 和系统层之间的边界。

![Triton roofline comparison](../assets/images/paper-figures/operators/triton-roofline-comparison.png){ width="820" }

<small>图源：[Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf)。原图表达不同实现相对 roofline 的位置；读图重点是把 microbenchmark 结果先放到“算力上限 vs 带宽上限”的坐标里，再判断优化方向。它不能证明某个 kernel 在端到端服务里必然更快。</small>

## 对照源码与延伸阅读

- [分块attention的实际计算路径](https://github.com/triton-lang/triton/blob/cd2e01c6686d6503802e6cdfac56d5a5dfdb27a1/python/tutorials/06-fused-attention.py)
- [不同shape上的矩阵乘基准组织](https://github.com/triton-lang/triton/blob/cd2e01c6686d6503802e6cdfac56d5a5dfdb27a1/python/tutorials/03-matrix-multiplication.py)

如果还不熟悉计时边界，先看[profiling、调试与数值稳定性](profiling-debugging-and-numerical-stability.md)；如果要把 roofline 判断接到真实请求，继续看[工作负载建模与 shape bucket](workload-modeling-and-shape-buckets.md)。公式负责缩小假设范围，trace 和硬件计数器负责确认假设。

## 先用量级筛选假设

Roofline 适合回答“在当前逻辑工作量下，算力还是带宽更可能先到上限”。它不能直接回答 kernel 是否写得好，也不能替代实际 trace。建议先用公式算出 F、M 和 I，再用设备峰值给出上界，最后用事件计时与硬件计数器验证偏差来自哪里。

报告时区分逻辑 FLOPs、有效 FLOPs、实际字节和估计字节。因果 mask、GQA、tile 复用和重复读取都会改变口径；如果公式和 benchmark 使用不同计数，TFLOPS 百分比会失去可比性。

## 来源与改编说明

- **实现参考**：[Triton fused attention tutorial](https://github.com/triton-lang/triton/blob/cd2e01c6686d6503802e6cdfac56d5a5dfdb27a1/python/tutorials/06-fused-attention.py)。
- **系统背景**：[Triton 论文](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf)展示 tile 级表达与 roofline 对照。
- **自测**：解释为什么 attention 占总时间 30% 时，即使它单独加速 2 倍，整步加速仍只有约 1.176 倍。
