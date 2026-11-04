# 算子与编译器

一个张量操作先有数学定义，再由程序把输出位置映射到输入地址，划分成GPU工作块，最后由runtime选择并执行kernel。性能问题可以出现在任一层。这个专题沿数据与状态变化讲机制，再提供实现和诊断入口。

## 从能手算的kernel开始

[CUDA](cuda-programming-model-and-memory.md)把1000个元素分给线程，解释warp、内存与同步；[Triton](triton-programming-model-and-autotuning.md)提供完整向量加代码，再连接二维tile。随后用[GEMM与融合](gemm-attention-and-fused-kernels.md)看同一块输入如何复用，用[归约、Norm与索引](reduction-norm-layout-and-indexing.md)理解多个元素如何共同决定输出。

## 模型里的具体计算路径

| 想弄清的问题 | 正文 |
| --- | --- |
| 不存完整分数矩阵怎样做softmax | [FlashAttention](long-context-and-flashattention-evolution.md) |
| 一次decode如何按页找到历史KV | [服务attention](serving-attention-and-kv-kernels.md) |
| 4-bit码值怎样参与乘加 | [低精度kernel](low-precision-and-quantized-kernels.md) |
| token怎样按专家重排再合并 | [MoE算子](moe-routing-and-sparse-kernels.md) |
| 低精度前向以后怎样更新参数 | [FP8训练与优化器](fp8-training-and-optimizer-kernels.md) |
| 四卡梯度怎样同步与重叠 | [通信算子](communication-kernels-and-overlap.md)、[物理拓扑](gpu-interconnects-and-topology-mapping.md) |

## 从解释走到实现

[CuTe/CUTLASS](cutlass-cute-and-compiler-stack.md)解释坐标、存储与线程布局；[高级调度](advanced-kernel-patterns-and-shape-specialization.md)沿tile讲双缓冲和persistent；[DeepGEMM机制](deepgemm-fp8-gemm-and-mega-moe.md)与[源码入口](deepgemm-source-and-integration.md)提供固定版本的阅读路线。

把kernel接入模型时，继续看[自定义算子注册](custom-op-development-and-framework-integration.md)、[runtime分流](runtime-dispatch-and-kernel-selection.md)和[测试](testing-regression-and-maintenance.md)。

## 性能判断与查验工具

先用[workload与shape分桶](workload-modeling-and-shape-buckets.md)确定实际工作，再用[成本模型](kernel-cost-models-and-selection-heuristics.md)和[Roofline](roofline-benchmarking-and-case-studies.md)算量级。测量方法见[Profiling](profiling-debugging-and-numerical-stability.md)，需要下钻时读[PTX/SASS](ptx-sass-and-compiler-inspection.md)。

[硬件排查](hardware-aware-debug-checklist.md)用于缩小异常范围，[性能误判](performance-anti-patterns-and-failure-modes.md)用可复算反例说明局部加速为何未必改善完整请求。各页明确区分源码事实、数学推导和需要目标GPU验证的性能结论。

## 对照源码与延伸阅读

- [从向量加开始的官方教程入口](https://github.com/triton-lang/triton/blob/cd2e01c6686d6503802e6cdfac56d5a5dfdb27a1/python/tutorials/01-vector-add.py)
