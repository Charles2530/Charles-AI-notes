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

## 22. Memory Transaction 视角下该怎么看“访存是否健康”

很多 CUDA 初学者只会说“要 coalesced”，但在真正调 kernel 时，更实用的视角是：**一次 warp 访存到底被拆成了多少个 transaction，搬了多少无用字节，命中了哪一级 cache，以及这些 transaction 是否能与计算重叠。**

### 22.1 为什么 transaction 粒度比源代码更真实

你在代码里看到的是：

1. `x[idx]`
2. `x[idx + stride]`
3. `float4` 或 `half2` 向量化加载

但硬件实际执行时关心的是：

1. **warp 内地址是否落在少数连续段上**；
2. **访问大小是否与 cache line / segment 对齐**；
3. **是否因为对齐差而多搬了一整块数据**；
4. **写回时是否触发 read-modify-write 或部分写放大**。

### 22.2 常见的“代码看着连续，硬件其实不连续”

1. **leading dimension 太怪**，导致相邻线程跨行跳跃；
2. **结构体布局不紧凑**，不同字段交错存放；
3. **量化权重和 scale 分开放置**，dequant 时访问两套不同行步长数组；
4. **page/block 索引层级太多**，最终地址映射分散。

### 22.3 一条很实用的判断线

如果某个 kernel 在 profile 里呈现出：

1. **HBM 带宽不高**；
2. **SM busy 也不高**；
3. **warp stall reason 偏向 memory dependency**；

那常见原因不是“GPU 太慢”，而是 transaction 组织很差，导致**既没有算满，也没有搬满**。

## 23. Warp Scheduler 真正在隐藏什么

理解 warp scheduler 的价值，关键是明白 GPU 不是让一个 warp 一口气做完所有事，而是在**大量 warp 间切换，用并发去填平等待**。

### 23.1 哪些等待最常见

1. **global memory latency**；
2. **shared memory 依赖**；
3. **长流水线指令等待**，如某些 Tensor Core 或特殊函数；
4. **barrier / sync 等待**；
5. **scoreboard dependency**，即指令结果尚未准备好。

### 23.2 为什么 occupancy 不是唯一答案

很多人以为只要 occupancy 高，warp scheduler 就能把等待全部藏掉。实际上还要看：

1. **每个 warp 自身有没有足够的 ILP**；
2. **可运行 warp 是否都卡在同一类依赖上**；
3. **是否因为 shared memory / register 占用导致 block 太少**；
4. **访存模式是否让所有 warp 同时被同一热点拖住**。

也就是说，**occupancy 提供的是“可以切换的人多不多”，而不是“切换后一定有活可干”**。

## 24. `cp.async` 与多阶段流水为什么重要

Hopper 之前后，很多高性能 GEMM 和 attention kernel 的质变都来自一件事：**把全局访存搬运和块内计算做成显式流水线**。

### 24.1 传统同步式 tile 流程

1. 从 global memory 搬一块到 shared memory；
2. `__syncthreads()`；
3. 计算这一块；
4. 再搬下一块；
5. 再同步。

这种流程简单，但会产生明显的“搬运空洞”。

### 24.2 `cp.async` 带来的变化

通过异步复制，kernel 可以：

1. **在计算当前 tile 时，后台预取下一 tile**；
2. **减少整块阻塞式等待**；
3. **构造 double buffering / multi-stage pipeline**；
4. **更细粒度地安排 producer-consumer 节奏**。

### 24.3 为什么不是所有 kernel 都该上多阶段

多阶段流水也有成本：

1. **shared memory 占用增加**；
2. **同步和状态管理更复杂**；
3. **寄存器压力上升**；
4. **小尺寸或低复用场景下收益有限**。

所以它更适合那些**tile 复用高、搬运成本显著、计算与搬运都足够重**的 kernel。

## 25. TMA、Cluster Launch 与 Hopper 之后的新心智

随着 Hopper 架构引入 **Tensor Memory Accelerator**、线程块簇和更强的异步能力，高性能 kernel 的设计边界又往前推了一步。

### 25.1 TMA 主要带来的不是“更快拷贝”这么简单

它实际在改变三件事：

1. **从更高维张量视角描述搬运**，而不是只靠 thread 手工逐元素拷；
2. **减少拷贝时对通用寄存器和线程参与的依赖**；
3. **让更复杂的 tile 和 layout 转换更自然地进入流水线**。

### 25.2 Thread Block Cluster 适合什么

当单个 block 的 shared memory 已不足以承载期望工作集时，cluster 允许更大范围的协作，适合：

1. **更大的 tile 协同**；
2. **跨 block 的 producer-consumer**；
3. **某些需要局部更大共享域的稀疏或分块算法**。

但 cluster 并不是白送的：**可移植性、调度约束和调优复杂度都更高**。

## 26. Launch Bounds、Register Pressure 与可持续优化

很多 kernel 在第一次调优时会遇到一个经典矛盾：**寄存器多了，单线程更强；寄存器太多，SM 同时驻留 warp 变少。**

### 26.1 `launch_bounds` 在帮你做什么

它本质上是在给编译器一个约束：你预期 block 大小和最少驻留 block 数大概是什么。编译器据此会调整 register 分配与生成策略。

### 26.2 常见误区

1. **盲目压 register**。结果虽然 occupancy 上去了，但 spill 到 local memory，反而更慢；
2. **盲目追大 tile**。看似 Tensor Core 利用更高，实则把 shared memory 和 register 全吃光；
3. **只测一个 shape**。某配置在大 shape 好，但在服务态小 shape 下严重退化。

### 26.3 一个更稳妥的调法

建议把调优分成三步：

1. **先确认瓶颈是带宽还是计算**；
2. **再分别扫 tile、寄存器和 pipeline stages**；
3. **最后看 shape family，而不是只看单点最佳值**。

## 27. Cache 行为不是黑盒

很多 kernel 不会显式控制 L1/L2，但不意味着 cache 行为不可理解。一个实用角度是区分：

1. **流式读一次就丢的数据**；
2. **同 SM 内短时间会复用的数据**；
3. **跨 block 也可能复用的数据**；
4. **写回后很快又会被后续 kernel 读的数据**。

### 27.1 为什么这对 AI kernel 特别重要

AI kernel 里经常出现下面几类模式：

1. **Q/K/V 块内重用**；
2. **dequant 时权重块和 scale 块重复读取**；
3. **MoE 路由索引先读再散射再 gather**；
4. **epilogue 刚写回的结果下一 kernel 立刻再读**。

如果系统能通过 fusion、layout 或 launch 顺序改善这些复用，收益往往不小。

### 27.2 一个常见错误

把所有数据都当成“希望 cache 住”的对象。其实很多 streaming 数据根本不该强求缓存；更应把缓存预算留给真正会复用的块。

## 28. Occupancy、ILP 与 MLP 的三角关系

讨论 kernel 性能时，单提 occupancy 常常过于粗糙。更完整的视角是：

1. **occupancy**：同时驻留多少 warp；
2. **ILP**：单个 warp 内独立指令是否足够多；
3. **MLP**：是否能并发发起多个内存请求。

### 28.1 为什么这三者会互相牵制

1. **更大 tile** 可能提高 ILP，但吃掉 register，压低 occupancy；
2. **更激进预取** 可能提高 MLP，但增加状态管理复杂度；
3. **更高 occupancy** 可能要求更小 block 或更少寄存器，从而降低单 warp 效率。

真正的最优解，通常不是某个指标极致，而是三者在特定 shape 和硬件上的平衡点。

## 29. Persistent Kernel 在推理里为什么常见

服务态 decode 场景里，很多请求很碎，kernel launch 开销、调度抖动和形状不规则都会被放大。**persistent kernel** 的价值就在于：让一批常驻 block 持续从工作队列里取任务，而不是为每个小任务都重新 launch。

### 29.1 它解决的问题

1. **减少 launch overhead**；
2. **提高小任务吞吐稳定性**；
3. **让 runtime 更容易做细粒度调度**；
4. **更适合 paged KV、grouped GEMM 这类不规则工作负载**。

### 29.2 它带来的新问题

1. **负载均衡更难**；
2. **调试更复杂**；
3. **容易出现某些 block 长期处理重任务导致尾部拖慢**；
4. **需要更明确的工作队列和同步协议**。

## 30. Kernel 设计常见模式

从 BBuf 的 CUDA 优化笔记、CUTLASS/CuTe 和很多现代推理 kernel 里，可以总结出几类高频模式：

1. **tile + pipeline**：GEMM、FlashAttention、量化 GEMM；
2. **warp-specialized**：不同 warp 承担加载、计算、归约等不同角色；
3. **persistent work queue**：服务态小任务或动态 shape；
4. **split-K / grouped dispatch**：让不规则矩阵也能吃满硬件；
5. **epilogue fusion**：把 bias、scale、activation、residual、dequant 合入写回前；
6. **layout-aware kernel**：通过 swizzle、interleave、reorder 减少 bank conflict 和 transaction 浪费。

### 30.1 为什么模式比单个技巧重要

因为真正工程里，硬件架构会变，框架会变，某条指令的最优用法也会变；但**数据搬运、协同粒度、工作分解和复用策略**这四件事不会变。

## 31. 一个面向 AI 算子的 CUDA 设计清单

准备写一个 CUDA kernel 时，建议先强制回答以下问题：

1. **目标 shape family 是什么**；
2. **它更像 compute-bound 还是 memory-bound**；
3. **是否存在天然 tile 复用**；
4. **是否值得用 shared memory 或直接走寄存器/向量化加载**；
5. **边界块和 mask 会不会特别重**；
6. **是否应该 persistent 或 grouped**；
7. **是否需要 fusion 才能体现价值**；
8. **正确性最脆弱的边界 shape 是什么**。

### 31.1 为什么这份清单重要

因为很多“优化失败”的根因不是实现差，而是一开始就把问题建模错了：把服务态小矩阵当训练态大 GEMM 来做，把 bandwidth-bound 的事情当 compute-bound 来卷，把 runtime 问题误判成 kernel 问题。

## 32. 小结

CUDA 编程模型真正难的地方，不在语法，而在于你是否能从**transaction、warp 调度、流水线、内存层次和形状分解**这几个维度同时看一个 kernel。只要能把这些层真正串起来，看待 GPU 上的 GEMM、Attention、KV Cache、量化算子和融合 kernel 就会从“很多零碎技巧”变成“少数几类可复用设计模式”的组合。
