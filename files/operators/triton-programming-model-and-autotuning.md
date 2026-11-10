# 用 Triton 把一段向量分给多个 program

一个向量加 kernel 只差几行代码，为什么仍可能读错非连续 view、在尾块越界，或者 benchmark 看起来比 PyTorch 快却在真实调用里变慢？Triton 的 program、地址、mask、stride 和 launch 参数需要一起看，长度 1000 的向量加正好能把这些边界暴露出来。

## 一个 program 负责256个位置

令BLOCK=256。program 0处理0到255，program 1处理256到511，program 3处理768到1023。总共只需4个program，但最后24个位置超出输入范围。

`tl.arange(0,BLOCK)`产生块内索引集合，`tl.program_id(0)*BLOCK`给出当前块起点。这个program不等于一个CUDA线程；块内这些元素由编译器分配到线程和寄存器，具体分配受dtype、操作和配置影响。

下面是完整教学实现，需要支持的GPU、PyTorch与Triton环境。这里没有在GPU上测量它的速度。

```python
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(X, Y, Z, N, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < N
    x = tl.load(X + offsets, mask=valid, other=0.0)
    y = tl.load(Y + offsets, mask=valid, other=0.0)
    tl.store(Z + offsets, x + y, mask=valid)

def add(x, y):
    assert x.is_cuda and y.is_cuda and x.device == y.device
    assert x.ndim == y.ndim == 1 and x.shape == y.shape
    assert x.is_contiguous() and y.is_contiguous() and x.dtype == y.dtype
    z = torch.empty_like(x)
    if x.numel():
        add_kernel[(triton.cdiv(x.numel(), 256),)](x, y, z, x.numel(), BLOCK=256)
    return z

x = torch.arange(1000, device='cuda', dtype=torch.float32)
y = torch.ones_like(x)
torch.testing.assert_close(add(x, y), x + y)
```

Load和store都带mask。无效位置填0不会影响有效位置的逐元素加法，但归约类算子要选择不同填充值，例如softmax的无效logit应填负无穷。

## 为什么 wrapper 要检查连续性

本例使用 `X+offsets`，假定相邻逻辑元素在存储中也相邻。若传入 `big_tensor[::2]`，逻辑shape虽然正确，stride却为2，当前kernel会读取错误位置。可选择在wrapper拒绝这种输入，也可把stride作为参数传进kernel并改地址计算。

自动调用 `contiguous()` 是第三种选择，但它增加一次拷贝。若只测kernel，不把这次转换算进完整算子延迟，就会高估收益。

## 矩阵乘只是换成二维输出块

对 $C_{M\times N}=A_{M\times K}B_{K\times N}$，一个program负责一块 $B_M\times B_N$ 的C，沿K反复读取A、B小块并累加。官方矩阵乘教程在循环中调用 `tl.dot`，最后把累加结果转换并写回。

$B_M,B_N$增大会增加复用，也增加输出累加器；$B_K$改变每次读入量和流水节奏。选择tile时必须同时看矩阵shape、寄存器、shared memory和尾块浪费。没有一个BLOCK设置能覆盖所有问题。

## Autotune 比较的是有限候选

`triton.autotune`可以在给定shape key下测试一组配置，例如不同tile、warp数和流水stage。它不会自动发明正确的索引、修复越界或理解业务流量。

对有副作用的kernel，候选试跑可能多次修改输出，必须按接口正确重置状态。调优key还要包含真正影响性能或语义的维度。遗漏stride、dtype等条件，可能复用错误选择；把每个无关值都纳入key，又会增加编译和缓存数量。

首次JIT与调优耗时应和预热后的执行时间分别报告。线上请求若会触发新shape，这些冷启动成本就需要计入对应延迟。

## stride 是 kernel 接口的一部分

如果矩阵逻辑尺寸相同，下面两个输入仍然不是同一种布局：

```python
x = torch.randn((8, 16), device="cuda")
x_view = x[:, ::2]       # shape [8, 8]，最后一维 stride 为 2
x_contig = x_view.contiguous()
```

只接收 `X + offsets` 的 kernel 只能处理 `x_contig`。要支持 `x_view`，wrapper 必须把 `stride_m`、`stride_n` 传入，并在 kernel 中按 `row * stride_m + col * stride_n` 计算地址。这个版本覆盖更多输入，却可能减少合并访问；把所有 stride 情况塞进同一个 autotune key，也会扩大编译缓存。接口选择应由真实调用分布决定。

对矩阵乘，最小的形状 contract 至少包括 `M,N,K`、A/B/C 的 stride、dtype、是否转置和累加精度。只把 `M,N,K` 放进 key，可能让连续布局的最优配置被非连续输入复用；只把每个 stride 原值放进去，又可能为大量等价布局重复编译。可以先把 stride 归一化为 `contiguous`、`row_major`、`col_major` 等类别，再按类别测量。

## autotune 的候选要有理由

一个矩阵乘的最小骨架可以写成：

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8, num_stages=4),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn,
                  stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    ...
```

这里的候选只是示意，不能直接当成所有 GPU 和 shape 的最佳配置。`BLOCK_M`、`BLOCK_N` 影响输出复用和寄存器数量，`BLOCK_K` 影响 K 维循环和内存访问，`num_warps` 与 `num_stages` 又会改变占用率和流水。候选集合应来自一个小型 shape sweep，保留每项配置的正确性结果、寄存器数和稳态时间。

如果 `M,N,K` 的范围很大，可以把 key 拆成 `"M_BUCKET"`、`"N_BUCKET"`、`"K_BUCKET"`，例如按是否小于 64、是否为 128 的倍数以及 K 的数量级分桶。分桶过粗会复用错误配置，分桶过细会增加 JIT 与调优冷启动。上线前应把实际请求 shape 的频率乘进候选选择，而不是只看单个最大矩阵。

## 正确性、冷启动和稳态计时要分开

一个可复查的 kernel benchmark 至少分三段：

1. 用随机输入、边界值、非整除 shape 和非连续 view 与 PyTorch reference 比较，记录最大绝对误差、相对误差和是否发生 NaN。
2. 单独测第一次调用，把 JIT、autotune、CUDA context 和可能的内存分配列为 cold-start 成本。
3. 预热后同步计时，固定迭代次数，报告中位数、p95、输入 shape、dtype、GPU、driver 和 Triton commit。若 wrapper 发生 `contiguous()`，完整算子计时必须把拷贝包含进去。

`torch.cuda.synchronize()` 应放在计时边界附近，否则 CPU 记录到的可能只是 kernel launch 时间。对短 kernel，可以用 CUDA events 减少主机计时误差；无论采用哪种方式，都要说明同步协议。没有运行 GPU benchmark 时，页面只能给出预期瓶颈和实验方法，不能填写一个看起来精确的加速倍数。

![Triton roofline comparison](../assets/images/paper-figures/operators/triton-roofline-comparison.png){ width="760" }

<small>图源：[Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf)，Figure 1。原图比较 Triton、cuBLAS、AutoTVM、Tensor Comprehensions 和 PlaidML 在矩阵乘 roofline 图上的位置；这里的重点是 Triton 的价值不是替代所有库，而是在高层框架和手写 CUDA 之间提供 tile 级表达能力。</small>

## 对照源码与延伸阅读

- [官方向量加教程](https://github.com/triton-lang/triton/blob/cd2e01c6686d6503802e6cdfac56d5a5dfdb27a1/python/tutorials/01-vector-add.py)
- [官方矩阵乘的地址、tile和autotune](https://github.com/triton-lang/triton/blob/cd2e01c6686d6503802e6cdfac56d5a5dfdb27a1/python/tutorials/03-matrix-multiplication.py)

## 本页的练习阶梯

先把向量长度改成 1、255、256、257，检查尾块 mask；再传入非连续 view，观察 wrapper 为什么拒绝它；最后给矩阵乘加入一个 bias 和激活，比较分开写回与 epilogue fusion 的数值和计时。每次只改一个条件，并分别记录 JIT、autotune 和稳态执行时间。

Triton 的高层表达减少了手写线程索引的负担，但并不自动保证地址、边界、stride、数值精度和性能正确。对每个 kernel，先写 reference 实现和 `torch.testing.assert_close`，再运行 benchmark。

## 来源与改编说明

- **主教程**：[Triton 官方矩阵乘教程](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)。
- **编程模型**：[Triton 官方向量加教程](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)，用于核对 program ID、mask、load/store 和 wrapper 的边界。
- **调优边界**：[Triton `autotune` 文档](https://triton-lang.org/main/python-api/generated/triton.autotune.html)，用于核对候选配置、key 和调优缓存行为。
- **代码版本**：仓库链接固定到 commit，提交新版本时应重新检查 API 和候选配置。
- **自测**：说明为什么把 `contiguous()` 放到 wrapper 里会改变完整算子成本，以及 autotune key 漏掉 stride 可能造成什么错误。

继续阅读：[算子测试、回归与维护](testing-regression-and-maintenance.md)处理 reference、误差阈值和版本回归；[Roofline 与 benchmark 案例](roofline-benchmarking-and-case-studies.md)把 kernel 时间放回端到端算术强度和实际 HBM 流量。
