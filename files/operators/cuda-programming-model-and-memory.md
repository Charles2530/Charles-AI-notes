# CUDA 编程模型与内存层次

CUDA 之所以长期是 AI kernel 的主语言，不只是因为它能在 GPU 上写程序，而是因为它把并行执行模型、内存层次和同步机制暴露得足够直接。理解 CUDA 的关键不是背 API，而是看清三件事：线程如何被调度，数据如何在 register、shared memory、L2 和 HBM 之间移动，为什么很多优化本质上都是重新安排访存和并发。

这页是算子专题的基础页，后续 [GEMM、Attention 与融合 Kernel](gemm-attention-and-fused-kernels.md)、[Triton 编程模型与 Autotuning](triton-programming-model-and-autotuning.md)、[CUTLASS、CuTe 与编译栈](cutlass-cute-and-compiler-stack.md) 都依赖这里的心智模型。

## 最小心智模型

一个 CUDA 程序可以先理解成：

1. CPU 发起 kernel launch；
2. GPU 启动大量 thread；
3. thread 按 block 组织，block 再组成 grid；
4. 硬件把 thread 分组成 warp 执行；
5. 数据从 HBM 进入更靠近计算单元的层级，完成计算后写回。

CPU 更强调低延迟单线程、复杂控制流和大 cache；GPU 更强调大量线程并发、高带宽吞吐、用 warp 切换隐藏访存延迟。AI 负载有大量规则、重复、数据并行的数值运算，因此天然适合 GPU。

这个模型的落点是：写 CUDA kernel 时，真正要设计的是**每个 block 合作处理哪块数据、每个 warp 如何访问内存、热点中间值放在哪一层、同步边界在哪里**。

## 执行层次

CUDA 的并行层级可以从小到大看：

| 层级 | 含义 | 优化关注点 |
| --- | --- | --- |
| thread | 最小执行实体 | register 使用、局部计算 |
| warp | 通常 32 个 thread 的调度组 | 分支发散、coalescing、warp primitive |
| block | 可共享 shared memory 的线程组 | tile 大小、同步、shared memory 占用 |
| grid | 一个 kernel 的 block 集合 | 全局并行度、shape 覆盖、边界处理 |
| SM | 执行 block/warp 的硬件单元 | occupancy、调度、资源驻留 |

虽然编程时写的是 thread，但硬件调度的核心粒度通常是 warp。warp 内线程走不同分支时会发生 branch divergence，硬件需要串行化不同路径，效率会下降。

Block 的重要性在于它定义了 shared memory 的共享范围和 `__syncthreads()` 的同步边界。很多高性能 kernel 的第一步，就是决定一个 block 内要合作完成多大的 tile。

## 内存层次

GPU 内存不是一块统一空间，而是明显分层：

| 层级 | 典型特点 | 常见用途 |
| --- | --- | --- |
| register | thread 私有，最快，容量小 | 累加器、热点标量、局部中间值 |
| shared memory | block 内共享，低延迟，高带宽 | tile 缓冲、局部归约、producer-consumer pipeline |
| L1 / texture / constant | 面向特定访问模式 | 只读数据、局部缓存、常量广播 |
| L2 | 多 SM 共享 | 跨 block 缓存、KV/page 访问缓冲 |
| global memory / HBM | 容量最大，延迟高 | 参数、激活、KV cache、输出张量 |

AI kernel 的性能常常不是被算术指令卡住，而是被数据搬运卡住。优化的中心思想通常是：少读写 HBM，把可重用数据留在 shared memory 或 register，让访存 coalesced，并尽量把多个小操作融合到一次读写路径里。

Register 最快，但 register pressure 会反噬。寄存器用得太多会降低 SM 上可驻留 warp 数，甚至 spill 到 local memory，而 local memory 本质上还是走 global memory 路径，代价很高。

## 访存与数据流

Global memory 访问最好让一个 warp 内线程读取连续、对齐地址，这样硬件可以把多次小访问合并成更少的大事务。Embedding、稀疏路由、KV cache 页面访问和不规则 gather/scatter 都容易破坏 coalescing。

Shared memory 的典型价值是做 tile 缓冲。以 GEMM 为例，一块 \(A\) 和一块 \(B\) 被搬入 shared memory 后，可以服务多个 MMA/FMA，而不是每个 thread 都从 HBM 重复读取。这就是 tiling 的核心。

Shared memory 也有 bank conflict 问题。如果多个线程同时访问同一 bank 且模式不佳，访问会被串行化。高性能 kernel 常通过 padding、swizzle、layout 变换和特殊 tile 排布减少冲突。

较新的 GPU 架构支持异步拷贝和多级流水，典型数据流是：

1. 从 HBM 预取下一块 tile；
2. 放入 shared memory；
3. 从 shared memory 装入 register；
4. Tensor Core 计算当前 tile；
5. 同时准备下一轮数据。

优秀 kernel 的复杂度往往来自这里：不是公式复杂，而是为了让搬运和计算重叠。

## Occupancy 与并发

Occupancy 通常指 SM 上实际驻留 warp 数占理论最大 warp 数的比例。它重要，但不是唯一目标。

Occupancy 太低时，GPU 难以用其他 warp 隐藏访存延迟；但 occupancy 高也不一定快，因为 kernel 可能仍然受限于 HBM 带宽、bank conflict、同步开销或 Tensor Core 利用率。

影响 occupancy 的主要因素包括：

1. 每个 thread 的 register 使用量；
2. 每个 block 的 shared memory 使用量；
3. block size 和 warp 数；
4. 编译器生成的局部变量与 spill；
5. 形状边界导致的有效线程比例。

实践中，很多优秀 GEMM/Attention kernel 会主动牺牲一部分 occupancy，换取更大的 tile、更高的数据重用或更好的 Tensor Core pipeline。目标不是“满 occupancy”，而是整体吞吐更高。

## Tensor Core 与 Warp Primitive

Tensor Core 是现代 AI GEMM、Attention 和低精度计算的性能关键。要真正吃到收益，通常要满足：

1. dtype 匹配，如 FP16/BF16/FP8/INT8；
2. tile 大小和对齐满足 MMA 指令要求；
3. 数据 layout 适合加载到 Tensor Core；
4. epilogue 不打断主计算流水；
5. M/N/K 维度足够规则。

不是所有矩阵乘都天然跑满 Tensor Core。decode 阶段的小 batch、细碎 shape、边界 mask、额外 transpose 和复杂 epilogue 都可能把理论吞吐吃掉。

Warp-level primitive 是另一类高频工具，包括 `__shfl_*`、`__ballot_sync`、`__syncwarp` 等。它们常用于 warp 内 reduction、softmax 局部 max/sum、layernorm 小向量归约、稀疏路由中的条件组合。很多小算子优化的核心，是尽量避免回到 shared memory 和 block-wide sync。

## AI Kernel 常见形态

| Kernel 类型 | 性能主矛盾 | 常见优化 |
| --- | --- | --- |
| GEMM | 数据重用与 Tensor Core 利用 | tiling、shared memory、register blocking、epilogue fusion |
| Attention | score 矩阵 I/O 与 KV 访问 | FlashAttention、online softmax、paged attention |
| Softmax / Reduce | 归约和数值稳定 | warp reduce、分块归约、避免中间写回 |
| LayerNorm / RMSNorm | 带宽和归约 | 向量化 load/store、融合 residual/scale |
| Quant / Dequant | 类型转换和 scale 访问 | per-token/per-channel scale、融合 GEMM epilogue |
| MoE 路由 | 不规则访存和负载不均 | token grouping、permute fusion、expert batching |

这些 kernel 表面不同，底层问题高度相似：shape 是否规则，数据是否连续，重用是否足够，中间张量是否被 materialize，launch 是否过多，精度和数值稳定是否满足要求。

## 调试与优化清单

排查 CUDA kernel 时建议按下面顺序看，而不是直接改代码：

1. **先定瓶颈**：是 HBM、Tensor Core、L2、shared memory、同步还是 launch；
2. **看访存**：global load/store 是否 coalesced，是否有不必要中间写回；
3. **看资源**：register、shared memory、occupancy 是否处在合理区间；
4. **看 shape**：边界、mask、小 batch、长尾 shape 是否拖慢主路径；
5. **看融合**：相邻小算子是否可以合并，epilogue 是否可以承接后处理；
6. **看数值**：低精度、归约顺序、softmax 稳定性是否改变结果；
7. **看回归**：固定输入、固定 seed、固定 shape bucket 做性能与正确性测试。

CUDA 优化的核心不是把代码写得更“底层”，而是让数据更少地绕远路。只要能减少 HBM 往返、提升片上重用、降低同步和 launch 开销，kernel 才有机会接近硬件上限。
