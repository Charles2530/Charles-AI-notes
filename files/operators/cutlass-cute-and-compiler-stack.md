# CuTe 怎样把矩阵坐标映射到线程和存储

一个 CuTe kernel 能顺利编译，并不说明它把 A、B、C 的坐标接对了。矩阵乘同时涉及数学坐标、内存位置和线程负责的片段，CuTe 用 layout 表达这些映射，CUTLASS 再把复制、矩阵运算和流水组件组合起来。读源码时先追一块数据怎样移动，比先记模板类型名称更容易。

## 从二维坐标算出线性地址

一个2×3的行优先矩阵，layout可写成shape(2,3)、stride(3,1)。坐标(i,j)映射到 $3i+j$，因此(1,2)位于偏移5。若stride为(1,2)，则同一坐标位于偏移5，但其他坐标顺序不同，(0,1)从偏移1变成2。

Shape描述有效坐标范围，stride描述地址增量。逻辑shape相同，不意味着连续维相同，也不意味着最合适的线程分工相同。

## local_tile先选出一个block负责的区域

CuTe的 `sgemm_1.cu`把全局A、B、C构造成tensor视图，用 `local_tile`按CTA坐标选出当前输出块及相关K块。随后为copy和计算分别建立线程分区。

负责搬A的线程布局与负责累加C的线程布局可以不同。一种布局优化全局读取，另一种优化计算与寄存器分配。二者在shared memory处对接，因此必须保证大家对数据坐标的解释一致。

<!-- systems-workflow-visual -->
```mermaid
flowchart LR
  M["全局矩阵shape与stride"] --> T["local_tile · 当前CTA区域"] --> P["线程copy分区"] --> S["shared memory"] --> C["计算分区与register累加"] --> E["epilogue写回"]
```

搬运布局与计算布局可以不同，shared memory是它们交换数据的接口。

## 主循环沿K逐块推进

该入门实现先copy当前A/B块到shared memory，等待异步拷贝完成并做block同步，再调用 `gemm`把乘积加进寄存器片段。计算后再次同步，避免下一轮写shared memory时还有线程在读旧块。

最后的epilogue将累加器按 $\alpha AB+\beta C$ 写回。这里alpha/beta与深度学习里的量化scale没有相同含义，不能仅凭变量名关联。

这个教程展示清楚的数据流，不能把它的每个同步点当成最高性能流水设计。更高级实现会重叠搬运与计算，并使用具体架构的矩阵指令和存储能力。

### 用一个小矩阵追一条坐标

设全局 A 为 `2×4`、B 为 `4×3`，输出 C 为 `2×3`，CTA 负责完整的 C。若当前输出坐标是 `(1,2)`，它需要累加

\[
C_{1,2}=A_{1,0}B_{0,2}+A_{1,1}B_{1,2}+A_{1,2}B_{2,2}+A_{1,3}B_{3,2}.
\]

当 K tile 宽度为 2 时，前两项来自第一个 tile，后两项来自第二个 tile。线程分区可以让一个线程负责 C 的一个片段，也可以让一个 warp 协同计算多个元素；无论采用哪种策略，shared memory 中的 A/B 坐标都必须能回到上面的四项。先用这个小矩阵打印 copy 后的 tile，再解释 `make_tensor`、`local_tile` 与 `partition`，比直接展开模板更容易发现转置或 stride 错误。

如果 C 的初始值不为零，epilogue 还要计算 `alpha * (A @ B) + beta * C`。把 beta 路径遗漏后，零初始化测试仍可能通过，所以测试至少要包含一个非零 C 和两个不同的 alpha/beta。

## 图编译器与kernel库位于不同层

图编译器能看到算子之间的数据依赖，可能合并逐元素操作、选择layout或调用库GEMM。Kernel库专注一个或一组计算的设备执行。若图层在库调用前插入转置复制，单测GEMM速度就不足以解释完整层。

Triton也允许直接描述块、地址与归约；CuTe/CUTLASS暴露的布局与架构组件有不同组织方式。选择哪种工具，应看是否需要现成库覆盖、定制epilogue、特殊layout或硬件指令，而非给语言排固定性能顺序。

## 阅读一个新kernel的可执行顺序

先固定M/N/K和输入stride，手算一个输出坐标对应哪些输入。再找到tile坐标、线程分区和copy，确认数据在shared与register的存活区间。最后检查同步与epilogue，并把编译出的资源使用与profile对照。

如果某一步无法确定坐标含义，先缩成小矩阵打印或构造参考实现。模板能编译只说明类型组合合法，不证明它计算了预期矩阵方向。

![Triton roofline comparison 原论文图](../assets/images/paper-figures/operators/triton-roofline-comparison.png){ width="760" }

<small>图源：[Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations](https://doi.org/10.1145/3315508.3329973)，Figure 1。原论文图意：在矩阵乘 \(C=AB^T\) 上比较 cuBLAS、Triton、Auto-TVM、Tensor Comprehensions 和 PlaidML 相对 roofline model 的性能位置。</small>

## 对照源码与延伸阅读

- [CuTe sgemm_1 的layout、local_tile、copy与gemm](https://github.com/NVIDIA/cutlass/blob/e13f5cd4142c43efc37c2f6339923eb4a57afb99/examples/cute/tutorial/sgemm_1.cu)

## 阅读 CuTe/CUTLASS 的检查点

先给一个小矩阵写出 shape、stride 和坐标到地址的映射，再追 `local_tile` 如何选择 CTA 区域，最后看 copy、shared memory、register fragment 和 epilogue。模板类型很长时，优先追数据坐标；只有坐标关系稳定后，才解释线程分区与架构指令。

如果最终结果只在转置输入或非连续 stride 上出错，先回到 layout 和 `local_tile`，不要马上怀疑 Tensor Core。一个小矩阵、一个手算地址和一次真实 kernel 输出，通常比完整模板展开更快地把问题缩小。

最小回归可以覆盖四组输入。连续行优先、连续列优先、带 padding 的 stride，以及 M/N/K 不是 tile 整数倍的尾部。每组都用 CPU 参考矩阵乘比较最大误差，并保存编译出的 kernel 资源信息。某个 layout 只在边界 shape 失败，往往说明 mask 或 copy 分区没有覆盖尾部，不一定是矩阵指令本身的问题。

**来源与改编说明**：[CuTe sgemm_1](https://github.com/NVIDIA/cutlass/blob/e13f5cd4142c43efc37c2f6339923eb4a57afb99/examples/cute/tutorial/sgemm_1.cu) 是本页的源码入口。它适合解释数据流，不应被描述成所有 GPU 上的最高性能实现。自测时把行优先 stride 改成列优先，重新计算一个输出坐标的地址和线程片段。

读完坐标映射后，可以接[矩阵乘、attention 与融合 kernel](gemm-attention-and-fused-kernels.md)看 tile 怎样影响复用和写回，再看[PTX、SASS 与编译器检查](ptx-sass-and-compiler-inspection.md)确认高层 layout 最终落成了什么指令。两处都要以目标 shape 和实际 trace 为准，不能凭模板类型名字判断性能。
