# 算子与编译器

同一个矩阵乘，数学公式没有变，换一种 stride、batch 或 GPU 就可能走完全不同的路径。程序要把输出位置映射回输入地址，kernel 要安排线程和片上存储，runtime 还要在很多候选里选一个能运行的版本。性能问题往往出在这些连接处，而不在公式本身。

这一组页面从一个能手算的小算子开始，逐渐走到 attention、MoE、FP8、通信和自定义算子。每到一层，都把数学结果、设备执行和完整请求分开看。

## 先从一个能手算的 kernel 开始

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

## 再把它接进实现

[CuTe/CUTLASS](cutlass-cute-and-compiler-stack.md)解释坐标、存储与线程布局；[高级调度](advanced-kernel-patterns-and-shape-specialization.md)沿tile讲双缓冲和persistent；[DeepGEMM机制](deepgemm-fp8-gemm-and-mega-moe.md)与[源码入口](deepgemm-source-and-integration.md)提供固定版本的阅读路线。

把kernel接入模型时，继续看[自定义算子注册](custom-op-development-and-framework-integration.md)、[runtime分流](runtime-dispatch-and-kernel-selection.md)和[测试](testing-regression-and-maintenance.md)。

## 最后才谈性能

先用[workload与shape分桶](workload-modeling-and-shape-buckets.md)确定实际工作，再用[成本模型](kernel-cost-models-and-selection-heuristics.md)和[Roofline](roofline-benchmarking-and-case-studies.md)算量级。测量方法见[Profiling](profiling-debugging-and-numerical-stability.md)，需要下钻时读[PTX/SASS](ptx-sass-and-compiler-inspection.md)。

[硬件排查](hardware-aware-debug-checklist.md)用于缩小异常范围，[性能误判](performance-anti-patterns-and-failure-modes.md)用可复算反例说明局部加速为何未必改善完整请求。各页明确区分源码事实、数学推导和需要目标GPU验证的性能结论。

## 对照源码与延伸阅读

- [从向量加开始的官方教程入口](https://github.com/triton-lang/triton/blob/cd2e01c6686d6503802e6cdfac56d5a5dfdb27a1/python/tutorials/01-vector-add.py)

## 从教程到 kernel 的推荐顺序

先读 [Triton 向量加](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)，把 program、offset、mask 和 launch 一一对应；再读 [Triton 矩阵乘](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)，观察二维 tile、K 维循环和 autotune；最后进入 GEMM、FlashAttention、通信和 DeepGEMM。每一步都先验证数值，再讨论性能。

性能结论按三层记录：公式给出工作量，microbenchmark 给出某个 kernel 的时间，端到端 trace 判断它是否位于关键路径。三层证据不能相互替代。

## 读完一层就做一次验收

不要等读完所有算子再开始写 kernel。每一层都可以留下一个很小的证据包：

| 阶段 | 最小任务 | 需要保存的结果 |
| --- | --- | --- |
| 向量与 tile | 对长度 1000 的输入处理尾块 | 参考输出、mask、边界 shape |
| GEMM | 用两个非连续 stride 的矩阵计算一个 tile | stride、输出 checksum、dtype |
| attention / KV | 跟踪一次 query 读取哪些历史块 | block 映射、有效 token 数 |
| 特化与调度 | 用真实 shape 桶比较通用路径和快路径 | p50/p95、fallback 次数、错误样本 |

如果某个阶段只有“代码能编译”而没有参考输出，先不要进入下一层。kernel 领域很容易把能启动、数值正确、局部变快和完整请求变快混成一个结论；这张表强制把它们拆开。

## 来源与改编说明

- **官方实现**以 Triton 文档和固定 commit 的教程为准，版本更新后重新核对 API。
- **系统直觉**可结合 [BRRRR](https://horace.io/brrr_intro.html) 和 Roofline 页面，把内存、算力和调度成本分开。
- **自测**：给一个 shape、dtype 和 stride，说明需要记录哪些信息才能判断某个 kernel 的慢来自带宽、并行度、转换还是调度。
- **自测**：给一个 `M=130, N=257, K=96`、非连续输入的矩阵乘，写出尾块、stride、mask、参考实现和计时边界；再说明哪一项证据缺失时不能下性能结论。
