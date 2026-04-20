# CUDA 编程模型与内存层次

CUDA 之所以长期是 AI kernel 的主语言，不只是因为它“能在 GPU 上写程序”，而是因为它把 GPU 的并行执行模型、内存层次和同步机制暴露得足够直接，使得工程师可以针对 AI 负载做非常激进的优化。理解 CUDA 的关键，不在于记住 API，而在于真正理解三件事：

1. GPU 线程是怎样被调度和执行的；
2. 数据在 register、shared memory、L2 和 HBM 之间如何流动；
3. 为什么很多所谓优化，本质上是在重新安排访存和并发。

## 1. CUDA 的最小心智模型

**可以先把 CUDA 程序理解成**：

1. CPU 侧发起 kernel launch；
2. GPU 侧启动大量 thread；
3. thread 按 block 组织，block 再组成 grid；
4. GPU 硬件把这些 thread 分组成 warp 执行；
5. 数据从 global memory 进入更靠近计算单元的内存层次，再被计算并写回。

这套模型看似简单，但几乎所有性能差异都埋在其中。

### 1.1 为什么 GPU 不像 CPU 那样写

CPU 更强调低延迟单线程执行、复杂控制流和大 cache 层级；GPU 更强调：

1. 大量线程并发；
2. 高带宽吞吐；
3. 通过线程切换隐藏内存访问延迟；
4. 用 SIMT 模型把大量相似计算铺开。

AI 任务恰好有大量规则、重复、数据并行的数值运算，这让 GPU 天然合适。

## 2. Thread、Warp、Block、Grid

CUDA 的并行层级从小到大可理解为：

1. `thread`：最小执行实体；
2. `warp`：通常 32 个 thread 的执行组；
3. `block`：一组 thread，可共享 shared memory 和同步；
4. `grid`：一整个 kernel 的 block 集合。

### 2.1 为什么 warp 是关键单位

虽然我们编程时写的是 thread，但硬件调度的核心粒度通常是 warp。warp 内线程在很多情况下执行相同指令流，只是处理不同数据。这就是 SIMT 的核心直觉：看起来像多线程，实质上带有很强的数据并行色彩。

如果 warp 内线程走不同分支，就会出现 branch divergence，导致硬件需要串行化执行不同路径，效率下降。

### 2.2 Block 为什么重要

Block 的作用不只是“组织线程”，更重要的是：

1. 它定义了 shared memory 的共享范围；
2. 它定义了 `__syncthreads()` 的同步边界；
3. 它影响单个 SM 上能同时驻留多少 block；
4. 它决定了 tile 的常见编程粒度。

很多 kernel 的第一步设计，就是先决定 block 内要合作完成多大的 tile。

## 3. SIMT、SPMD 与实际执行

AI System Docs 中关于 SIMD、SIMT 与 CUDA 的讨论很有帮助：从编程视角看，CUDA 更像 SPMD，即“写一份程序，很多线程各自运行”；但从硬件执行视角看，它又保留了强烈的 SIMT 特征，也就是 warp 内多个线程往往共同执行同一指令流。

这个视角很重要，因为它解释了很多现象：

1. 为什么分支发散会慢；
2. 为什么 contiguous access 如此重要；
3. 为什么 warp-level primitive 能显著加速归约；
4. 为什么很多优化都围绕让 warp 内线程协同搬运和计算。

## 4. SM、Occupancy 与隐藏延迟

一个 GPU 上有多个 SM（Streaming Multiprocessor）。每个 SM 可以同时驻留多个 warp，通过 warp 切换隐藏内存延迟。如果一个 warp 在等待 global memory，硬件可以切换到另一个可运行 warp。

### 4.1 Occupancy 是什么

Occupancy 通常指某个 SM 上实际驻留 warp 数占理论最大 warp 数的比例。很多初学者把它当成唯一性能指标，这是不对的，但它仍是一个重要信号，因为：

1. occupancy 太低时，延迟隐藏能力不足；
2. occupancy 受 register 使用量、shared memory 使用量、block 大小共同制约；
3. 某些 kernel 为了更多寄存器或更大 tile，可能主动牺牲 occupancy。

### 4.2 为什么 occupancy 高不一定就快

因为性能不只由并发数决定，还取决于：

1. 是否受限于带宽；
2. 是否有 bank conflict；
3. 是否有过多同步；
4. 是否真的提升了 Tensor Core 利用率。

很多优秀 kernel 并不会追求“满 occupancy”，而是追求更合理的算子流水与数据重用。

## 5. CUDA 内存层次

一个最基础但必须掌握的图景是，GPU 内存不是一块统一空间，而是有明显层级：

1. `register`：每个 thread 私有，最快但容量小；
2. `shared memory`：block 内共享，延迟低、带宽高；
3. `L1 / texture / constant`：面向特定访问模式和缓存场景；
4. `L2`：多个 SM 共享；
5. `global memory / HBM`：容量最大，但访问代价高。

### 5.1 为什么内存层次比指令更重要

对大量 AI 算子来说，真正慢的往往不是算术本身，而是数据搬运。  
如果数据每次都从 HBM 取，用完又写回 HBM，那么即使数学操作很简单，性能也会被带宽完全卡住。

这也是为什么 kernel 优化的中心思想通常是：

1. 尽量少读写 HBM；
2. 尽量把重用数据留在 shared memory 或 register；
3. 尽量让访存 coalesced；
4. 尽量把多个操作 fuse 到一次读写中。

## 6. Global Memory 与 Coalescing

global memory 是容量最大的存储层，但访问延迟高、带宽宝贵。  
理想情况下，一个 warp 的线程应该尽量访问连续、对齐的地址，这样硬件可以把多次小访问合并成更少、更大的事务。

### 6.1 Coalesced access 为什么关键

如果 warp 中 32 个线程访问的是连续地址，那么访存事务可以被较好合并；  
如果访问分散、步长大、存在不规则 gather/scatter，就会出现：

1. 更多访存事务；
2. 带宽利用率下降；
3. cache 命中变差；
4. 整体 kernel 变得 memory-bound。

Embedding、indexing、稀疏路由、KV cache 页面访问都很容易出现这种问题。

## 7. Shared Memory：许多高性能 kernel 的中枢

shared memory 可被 block 内 thread 共同访问，延迟比 global memory 小得多，带宽也更高。  
**常见作用包括**：

1. 作为 tile 缓冲区；
2. 存放中间归约结果；
3. 减少重复访问 global memory；
4. 配合异步拷贝形成 producer-consumer pipeline。

### 7.1 为什么 GEMM 几乎离不开 shared memory

**矩阵乘最大的机会来自数据重用**：  
一块 \(A\) 子矩阵和一块 \(B\) 子矩阵一旦被搬入 shared memory，就可以服务多个乘加操作，而不必每个 thread 都去 HBM 重复读取。

**这就是所谓的 tiling**：先搬小块，再在小块内部反复算。

### 7.2 Bank conflict 是什么

shared memory 通常按 bank 组织。如果同一时刻多个线程访问同一个 bank 且模式不佳，就会发生 bank conflict，导致访问串行化。

这也是为什么很多高性能实现会引入：

1. padding；
2. swizzle；
3. layout 变换；
4. 特殊的 tile 排布。

目的不是“写得复杂”，而是减少 bank conflict。

## 8. Register：最快，但最容易被忽略的约束

register 是 thread 私有、速度最快的存储层。很多 kernel 优化的真正收益，来自：

1. 尽量把热点中间值放在 register；
2. 避免不必要写回 shared memory 或 global memory；
3. 让累计值在 register 中完成。

### 8.1 Register pressure 为什么会反噬

寄存器不是无限的。如果一个 kernel 用了太多 register，会导致：

1. 一个 SM 上可同时驻留的 warp 变少；
2. occupancy 下降；
3. 甚至出现 register spill 到 local memory。

local memory 本质上还是走 global memory 路线，性能代价很大。所以“更多寄存器”也不是总比“更少寄存器”好。

## 9. Warp-level Primitive

现代 CUDA 优化大量使用 warp-level primitive，例如：

1. `__shfl_*` 做 warp 内数据交换；
2. `__ballot_sync` 收集条件位；
3. `__syncwarp` 做更细粒度同步。

它们之所以重要，是因为很多小规模归约、prefix 操作、投票和重排，完全没必要回到 shared memory 再同步整个 block。warp-level 原语可以让这些操作更轻量。

### 9.1 典型用途

1. warp 内 reduction；
2. softmax 中局部 max / sum；
3. layernorm 的小向量归约；
4. 稀疏路由和索引映射中的条件组合。

许多 “warp softmax”“warp reduce” 的高性能实现，本质上就是在尽量减少 shared memory 和 block-wide sync。

## 10. Tensor Core 与 MMA

对于矩阵乘和 attention 这类核心 AI 算子，Tensor Core 是性能关键。它提供了更高吞吐的矩阵乘累加路径，但前提是：

1. 数据类型匹配；
2. tile 大小和布局匹配；
3. 数据装载方式适合 Tensor Core 指令；
4. 编译器或模板库能生成正确的 MMA 序列。

### 10.1 为什么不是所有 GEMM 都天然吃满 Tensor Core

因为真正决定能否高效利用 Tensor Core 的，不只是“调用了 matrix multiply”，还包括：

1. K 维是否够大；
2. tile 切分是否合适；
3. 访存是否连续；
4. epilogue 是否打断流水；
5. 数据布局是否导致额外 transpose。

这也是为什么 CUTLASS、CuTe、手写 MMA pipeline 仍然很有价值。

## 11. 异步拷贝与多级流水

较新的 GPU 架构支持更强的异步拷贝与内存流水机制。直觉上，就是把：

1. 从 global memory 取 tile；
2. 放入 shared memory；
3. 再从 shared memory 装入寄存器；
4. 执行 Tensor Core 计算；

这几步尽量重叠起来，形成 pipeline，而不是等前一步完全结束再做下一步。

### 11.1 为什么这对 GEMM 和 Attention 特别重要

因为这两类算子几乎都具有“分块搬运 + 分块计算”的天然结构。  
只要 pipeline 做得好，就能在一边算当前 tile 的同时，一边预取下一个 tile，减少内存等待。

## 12. 启动配置：threads per block 不是拍脑袋

选择 `blockDim` 时要综合考虑：

1. 每个 block 的工作量；
2. warp 数是否足够；
3. shared memory 使用量；
4. register 使用量；
5. 是否利于 coalesced access；
6. 是否利于后续归约。

初学者最常见错误之一，就是把线程数当作“越多越好”。现实里，很多 kernel 的最佳 block size 都来自功能需求、寄存器预算和 occupancy 的折中，而不是固定套 `256` 或 `512`。

## 13. CUDA 中最常见的几类 AI kernel

### 13.1 GEMM

核心问题是 tile、数据重用、Tensor Core 和 epilogue fusion。

### 13.2 Softmax / Reduce

**核心问题是**：

1. 局部 max / sum 的归约；
2. 数值稳定性；
3. warp 与 block 粒度切换；
4. 是否需要分块在线归一化。

### 13.3 LayerNorm / RMSNorm

**核心问题是**：

1. 每行统计量计算；
2. 访存连续性；
3. 向量化 load/store；
4. 是否与 scale/bias/残差融合。

### 13.4 Attention

**核心问题是**：

1. 是否 materialize 大矩阵；
2. 在线 softmax；
3. Q/K/V tile 复用；
4. causal mask、padding mask 与分块调度。

### 13.5 Indexing / Gather / Scatter

核心问题更多在不规则访存和原子操作，而不是纯数学计算。

## 14. 一个 GEMM 直觉例子

假设要计算 \(C = AB\)。  
最朴素的实现是：每个 thread 负责一个输出元素，直接从 global memory 反复读取 \(A\) 和 \(B\)。  
这种实现通常非常慢，因为：

1. 同一行或同一列数据被重复读取很多次；
2. global memory 带宽压力大；
3. 没有利用 Tensor Core；
4. 中间累加没有很好地留在 register。

**优化后的思路通常是**：

1. 把 \(A\) 和 \(B\) 切成 tile；
2. 搬到 shared memory；
3. 再搬到 register；
4. 用 warp 或 warp-group 协作做 MMA；
5. 最后把结果写回 global memory。

这类思路几乎贯穿了现代所有高性能 GEMM。

## 15. 一个 Softmax 直觉例子

看似简单的 softmax，真正高效实现时要处理两件事：

1. **数值稳定**：需要减去最大值；
2. **性能稳定**：不能因为多次 global memory 往返而慢。

因此很多高性能 softmax 会采用：

1. 先做局部 max reduction；
2. 再做局部 exp 和 sum reduction；
3. 尽可能在寄存器或 shared memory 中完成；
4. 若序列太长，则分块并使用在线归约。

这正是很多 FlashAttention 实现的基础。

## 16. 一个 LayerNorm / RMSNorm 直觉例子

Norm 类算子常被误以为很轻，但它们在 LLM 推理和训练中调用极其频繁，而且往往是内存带宽受限。性能关键通常不是更多 FLOPs，而是：

1. 向量化加载；
2. warp / block 内快速归约；
3. 融合 scale、bias 或 residual；
4. 减少中间结果写回。

所以小算子也很值得优化，尤其在 decode 场景中。

## 17. 训练与推理场景下的不同关注点

CUDA kernel 在训练里更偏向：

1. 大批量 GEMM 吞吐；
2. attention 反向；
3. optimizer step；
4. 混合精度与梯度缩放。

**在推理里更偏向**：

1. decode 下小而频繁的 kernel；
2. paged attention；
3. quantized GEMM；
4. KV cache 访存；
5. 多请求动态 batch 下的 irregularity。

同一种算法，训练与推理的最优实现可以完全不同。

## 18. 常见性能问题排查清单

如果某个 CUDA kernel 表现不佳，通常先检查：

1. 是否 coalesced load/store；
2. 是否存在明显 branch divergence；
3. shared memory 是否有 bank conflict；
4. register 是否过高导致 occupancy 下降；
5. 是否有大量不必要同步；
6. 是否因为 layout 不合理导致额外 transpose；
7. 是否明明是 bandwidth-bound，却在盲目追求更多计算优化。

## 19. 常见正确性问题排查清单

如果 kernel 结果不对，常见原因包括：

1. 越界访问；
2. mask 逻辑错误；
3. block 边界条件处理不对；
4. 原子操作竞争；
5. 数值精度或累加精度不够；
6. 异步拷贝与同步点顺序错误；
7. 只在某些 shape 下才触发的 layout bug。

## 20. 一个形象比喻

可以把 CUDA 程序想成在一个多层仓库里组织工人搬运和装配零件。HBM 像远处的大仓库，shared memory 像车间内的临时料架，register 像工人手边的工具盘。  
如果每次装配都要跑回大仓库拿一个零件，效率一定很差；真正高效的做法是先按块把常用材料搬到车间，再按工位分配到手边，最后连续完成多步装配。

绝大多数高性能 CUDA kernel，本质上都在做这件事：把数据搬运组织得像一条高效生产线，而不是一群线程各干各的。

## 21. 小结

理解 CUDA 编程模型，关键不在于记住多少 API，而在于掌握 GPU 的执行与内存本质：warp 如何执行、SM 如何隐藏延迟、数据在各层内存中如何流动、以及为什么 tile、coalescing、shared memory、register 和 Tensor Core 会成为一切高性能 kernel 的核心词汇。

只要这套心智模型建立起来，后面再去看 Triton、CUTLASS、FlashAttention、PagedAttention、量化 GEMM 或自定义 fused kernel，就不会把它们看成一堆零散技巧，而会看到它们共享的底层逻辑。
