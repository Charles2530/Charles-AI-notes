# CUDA 中，一个元素怎样找到负责它的线程

长度 1000 的向量加法看起来没有研究价值，却正好能暴露 CUDA kernel 最基本的错误。一个线程负责哪个输出，地址怎样由 stride 算出来，最后一个 block 会不会越界，答案都可以在几行代码和一个 CPU 参考实现里核对。矩阵乘、attention 和 fused optimizer 只是把同一组问题扩大了。

## 从输出位置反推 launch

若每个block有256个线程，需要4个block。线程的全局索引为

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < n) out[i] = x[i] + y[i];
```

block 0负责0到255，block 3理论上覆盖768到1023，最后24个线程必须跳过访问。这个条件需要保护读取和写入；只给store加判断，越界load仍然可能出错。

CUDA的block被调度到SM，线程按warp执行。在NVIDIA CUDA设备上warp包含32个线程，因此256线程block包含8个warp。Block之间没有天然的全局执行顺序，不能让后一个block假定前一个block已经写完数据。

![GH100 GPU block diagram](../assets/images/paper-deep-dives/foundations/gpu-utilization-misleading/figure-1-gh100-sms.png){ width="920" }

<small>图源：[NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)。原图展示 GH100 的 GPC、TPC、SM、L2、HBM 控制器和互连模块。这里用它说明：kernel 不是跑在一个抽象“GPU”上，而是被分配到许多 SM 和共享内存系统里；SM 利用率高，不代表 HBM、L2、launch、同步和通信都不是瓶颈。</small>

## 连续索引为什么有帮助

若x是连续FP32数组，相邻32个线程访问相邻32个元素，逻辑上覆盖128字节连续数据。硬件能将请求组织成较少的内存事务，实际事务数量还受对齐和架构影响。

如果每个线程改为读取 `x[i*32]`，同一warp访问就跨得很远。数学上仍是32次load，物理上可能需要更多事务。这个差别称为访存合并问题，不能仅由元素数量判断。

二维张量也一样。行优先矩阵的元素地址为 `base + row*stride_row + col*stride_col`。维度相同而stride不同的输入，会让同一索引代码访问完全不同的位置。

## 数据在哪一层被重复使用

向量加中每个输入通常只使用一次，刻意先搬到shared memory不一定有益。矩阵乘不同，一块A会与多列B相乘，一块B会被多行A使用。把这些块放进片上可共享存储，可以避免重复从更远层级取数据。

寄存器主要保存线程的局部值与累加器，shared memory供同一block协作，L2是更大范围共享的缓存，HBM保存主体张量。编译器因寄存器不足而使用的local memory，名字里虽有local，却通常位于设备内存地址空间，经缓存访问，并不等于片上寄存器。

CUTLASS的CuTe入门GEMM依次执行global到shared的copy、等待与block同步、GEMM累加，再同步以防下一轮覆盖尚在读取的数据。数据复用带来收益，也引入生命周期与同步要求。

## 同步必须对应真实依赖

一个线程刚写完shared memory，不代表同一block其他线程已经看到了各自需要的全部值。协作计算需要适当同步。反过来，没有共享依赖时添加同步只会增加等待。

全block屏障不能随意放在部分线程进入的分支里，否则其他线程可能永远不会到达。边界处理常通过mask让所有线程沿相同协作结构执行，再忽略无效元素。

![Nsight Systems GPU starvation example](../assets/images/tools/nsight-systems/gpu-starvation.png){ width="760" }

<small>图源：[Nsight Systems Exposes New GPU Optimization Opportunities](https://developer.nvidia.com/blog/nsight-systems-exposes-gpu-optimization/)。原图展示 GPU 时间线中的空洞。这里用它说明：如果 GPU 没拿到活，瓶颈可能在 CPU 调度、同步、数据准备或 launch 组织，而不是某个 kernel 算得慢。</small>

## 更多驻留线程为何未必更快

SM可同时驻留多少block，受到线程、寄存器和shared memory资源限制。更大tile提高复用，也可能增加累加器与缓存需求，减少并发block；溢出到local memory后甚至新增访存。

Occupancy描述驻留程度，不直接等于性能目标。若一个kernel已经充分利用矩阵指令和数据复用，强行提高occupancy可能破坏更重要的条件。先用[成本模型](kernel-cost-models-and-selection-heuristics.md)估算需要搬多少数据，再用[profiling](profiling-debugging-and-numerical-stability.md)确认实际瓶颈。


## 对照源码与延伸阅读

- [CuTe sgemm_1 的 copy、同步与累加循环](https://github.com/NVIDIA/cutlass/blob/e13f5cd4142c43efc37c2f6339923eb4a57afb99/examples/cute/tutorial/sgemm_1.cu)
- [相同向量加任务的 Triton 映射](https://github.com/triton-lang/triton/blob/cd2e01c6686d6503802e6cdfac56d5a5dfdb27a1/python/tutorials/01-vector-add.py)

## 一个最小实验怎样逐层加复杂度

先实现长度 1000 的向量加，比较连续输入和 stride=2 的切片；再把每个线程处理一个元素改成每个线程处理多个元素；最后才尝试 shared memory 或向量化加载。每一步保留参考输出、kernel 时间和完整 wrapper 时间。这样可以观察边界 mask、访存模式和 launch 开销分别带来什么变化。

不要先追求 occupancy。若向量加的工作集很小，kernel 时间可能主要由 launch 和同步决定；若 stride 很大，瓶颈可能是非合并访问；若线程每次处理多个元素，寄存器和尾部处理又会改变条件。先根据工作集和访问模式提出假设，再用 Nsight 或计时验证。

## 读一个地址公式的四个问题

面对任何 kernel，先写出一个输出元素对应的输入地址，再检查：

1. 线程是否可能计算同一个输出两次？
2. 所有 load 和 store 是否都有同一套边界保护？
3. stride 是否来自真实 tensor view，而不是默认连续布局？
4. 不同线程是否在共享写入前后有明确同步？

这四问能在编译前发现许多越界、重复写入和转置误读。性能优化应在数学映射正确后进行。

## 还要检查 warp 和 shared memory 的细节

一个 warp 中的线程若走了不同分支，硬件会分段执行这些路径。边界判断本身通常不可避免，真正需要留意的是让大量有效线程长期分成互斥分支。可以先统计每个分支覆盖的 lane，再看它是否出现在内层循环里。

shared memory 也有 bank conflict。不同线程访问同一 bank 的不同地址时，访问可能被拆成多个事务；访问同一个地址则可能通过广播处理。具体冲突模式受数据类型、bank 宽度和架构影响，不能只凭“用了 shared memory”判断快慢。

这些现象都应放回完整时间线。一个 kernel 的访存事务、warp stall、寄存器使用和 occupancy 彼此相关，单独优化其中一项可能把压力转移到另一层。NVIDIA 的 [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) 解释线程层级、同步和内存模型，[CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) 给出合并访问、共享内存和测量方法；[Nsight Systems 文档](https://docs.nvidia.com/nsight-systems/) 用于查看 CPU、GPU 和通信的时间关系。

## 实现依据

向量加和二维地址公式来自 CUDA 官方编程模型；CuTe 的 [sgemm_1](https://github.com/NVIDIA/cutlass/blob/e13f5cd4142c43efc37c2f6339923eb4a57afb99/examples/cute/tutorial/sgemm_1.cu) 用来对照 global、shared、同步和累加器之间的数据流，Triton 的[向量加教程](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)用来核对相同问题在另一种编程接口下怎样表达。文中的长度 1000、stride 例子和实验顺序是本站的教学设计，不是某张 GPU 的性能报告。

最小回放可以只测三件事：连续输入、`x[::2]` 形成的非连续 view、以及最后一个 block 的尾部。每次同时保存 CPU 参考输出、完整 wrapper 时间、kernel 时间和错误最大值。只有输出一致以后，才有资格讨论 occupancy 或向量化。
