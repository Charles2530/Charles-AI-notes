# 算子与编译器专题：把性能问题落到内存和形状

现代 AI 系统里，PyTorch、Triton、CUTLASS、cuBLAS、FlashAttention、TensorRT、XLA/Inductor 共同决定一个模型能否跑满硬件。定位性能问题时，先把瓶颈落到三个对象：tensor shape、memory movement、kernel dispatch。

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
