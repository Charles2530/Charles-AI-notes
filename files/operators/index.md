# 算子与编译器专题：把性能问题落到内存和形状

算子优化不是把某个 kernel 换成更快的实现就结束了。一次矩阵乘、attention 或 reduction 能否提速，取决于输入 shape、数据布局、内存层级、并行切分、数值格式和运行时是否真的选中了这条路径。PyTorch、Triton、CUTLASS、cuBLAS、FlashAttention 与编译器工具链只是把这些约束暴露在不同层。

读这个专题时，先追踪一块 tensor 的去向：它从哪里读入，在哪里重排，经过几次 HBM 往返，在哪个 kernel 中计算，最后是否因为 layout 或 dispatch 又付出一次转换成本。很多“算力不够”的问题，最后都落在搬运、空转和形状不匹配上。

## 先建立一条性能判断链

一个可复用的排查顺序是：先用 workload 和 shape 描述问题，再用 profiler 找到热点，接着判断热点受算力、带宽、同步还是 launch 限制，最后才决定融合、重排、量化或改并行策略。只看单个 kernel 的理论 TFLOPS，无法解释动态 batch、长序列和多卡通信下的端到端结果。

| 先问什么 | 你要找的证据 | 建议入口 |
| --- | --- | --- |
| 工作负载是什么形状 | M/N/K、序列长度、batch、稀疏度、读写字节数 | [Workload Modeling 与 Shape Bucket](workload-modeling-and-shape-buckets.md) |
| 时间花在哪里 | kernel 时间、launch 间隙、同步和通信 trace | [Profiling、调试与数值稳定](profiling-debugging-and-numerical-stability.md) |
| 瓶颈属于哪一层 | 算术强度、带宽利用率、occupancy、L2 命中 | [Roofline 建模与性能案例](roofline-benchmarking-and-case-studies.md) |
| 改动是否真的有效 | 相同输入桶下的端到端延迟、吞吐、误差和回归 | [测试、回归与维护](testing-regression-and-maintenance.md) |

## 建立硬件和内存直觉

| 你想弄清的问题 | 建议先读 |
| --- | --- |
| CUDA block、warp、shared memory、global memory 如何协作 | [CUDA 编程模型与内存层次](cuda-programming-model-and-memory.md) |
| GPU interconnect、NUMA、拓扑为什么影响分布式性能 | [GPU 互联与拓扑映射](gpu-interconnects-and-topology-mapping.md) |
| Roofline 怎样判断瓶颈在算力还是带宽 | [Roofline 建模与性能案例](roofline-benchmarking-and-case-studies.md) |
| profiling、NVTX、数值稳定如何一起排查 | [Profiling、调试与数值稳定](profiling-debugging-and-numerical-stability.md) |

## 进入常见高价值 kernel

| 主题 | 建议先读 |
| --- | --- |
| GEMM、attention、fused kernel 的基本模式 | [GEMM、Attention 与融合 Kernel](gemm-attention-and-fused-kernels.md) |
| FlashAttention 和长上下文为什么改变 IO 模型 | [FlashAttention 与长上下文演进](long-context-and-flashattention-evolution.md) |
| reduction、norm、layout、indexing 为什么经常拖慢训练 | [Reduction、Norm、Layout 与 Indexing](reduction-norm-layout-and-indexing.md) |
| low precision、FP8、量化 kernel 如何服务训练和推理 | [低精度与量化 Kernel](low-precision-and-quantized-kernels.md)、[FP8 训练与优化器 Kernel](fp8-training-and-optimizer-kernels.md) |

## 进入工具栈和维护

| 目标 | 建议先读 |
| --- | --- |
| 用 Triton 写可调优 kernel | [Triton 编程模型与自动调优](triton-programming-model-and-autotuning.md) |
| 理解 CUTLASS、CuTe 和编译器栈 | [CUTLASS、CuTe 与编译器栈](cutlass-cute-and-compiler-stack.md) |
| 选择 runtime dispatch 和 shape bucket | [Runtime Dispatch 与 Kernel 选择](runtime-dispatch-and-kernel-selection.md)、[Workload Modeling 与 Shape Bucket](workload-modeling-and-shape-buckets.md) |
| 做自定义 op 的测试、回归和维护 | [自定义 Op 与框架集成](custom-op-development-and-framework-integration.md)、[测试、回归与维护](testing-regression-and-maintenance.md) |

算子篇的判断标准很朴素：改动后有没有更少的数据搬运、更好的并行度、更稳定的数值、更明确的 shape 适配，以及更可维护的回归测试。

## 三个容易误读的地方

**理论 FLOPS 高，不代表请求更快。** decode、reduction 和小矩阵 GEMM 往往受内存访问或 kernel launch 限制，峰值算力没有被吃满是结果，不一定是根因。

**融合越多，不代表一定越好。** 融合可以减少中间张量和 launch，但也可能扩大寄存器压力、降低 occupancy，或让动态 shape 失去可复用的实现。要把收益放回真实 shape 桶和完整请求链路里测。

**编译器 fallback 也是产品行为。** Inductor、TensorRT 或自定义 dispatch 在某些 dtype、mask、stride、序列长度下可能回退到通用实现。上线前应记录实际命中的 kernel、覆盖的 shape，以及 fallback 对 P99 的影响。

## 按问题进入

| 现在遇到的问题 | 推荐路径 |
| --- | --- |
| GPU 利用率低，但 FLOPs 看起来不高 | [CUDA 编程模型与内存层次](cuda-programming-model-and-memory.md) → [Roofline](roofline-benchmarking-and-case-studies.md) → [Profiling](profiling-debugging-and-numerical-stability.md) |
| attention 变快了，端到端却没有改善 | [FlashAttention 与长上下文演进](long-context-and-flashattention-evolution.md) → [Runtime Dispatch 与 Kernel 选择](runtime-dispatch-and-kernel-selection.md) |
| 自定义 kernel 只在少数输入上有效 | [Workload Modeling 与 Shape Bucket](workload-modeling-and-shape-buckets.md) → [Triton 编程模型与自动调优](triton-programming-model-and-autotuning.md) |
| 多卡扩展后通信吞掉计算收益 | [GPU 互联与拓扑映射](gpu-interconnects-and-topology-mapping.md) → [通信 Kernel 与计算通信重叠](communication-kernels-and-overlap.md) |
