# Triton 编程模型与自动调优来源台账

这份台账服务于 [Triton 编程模型与自动调优](../operators/triton-programming-model-and-autotuning.md)。正文不把 Triton 写成“Python 写 CUDA”，而是围绕 program instance、block 地址计算、stride、matmul tile、autotune key、JIT/cache/fallback 和端到端验证来组织。

## 使用方式

第一类来源是 Triton 官方教程和 API。Vector Addition 教程支撑正文里的 `program_id`、`tl.arange`、offset、mask、`tl.load`、`tl.store` 最小心智模型。Matrix Multiplication 教程支撑 \(M,N,K\)、`BLOCK_M/N/K`、`tl.dot`、group ordering、L2 locality 和 autotune 配置。`triton.autotune` API 支撑 `configs`、`key`、`reset_to_zero`、`restore_value` 和候选配置搜索边界。

第二类来源是 Triton 论文和 OpenAI 技术博客。Triton 论文用于说明它的定位是 tiled neural network computations 的中间语言和编译器，而不是普通 Python kernel；正文使用 roofline comparison 图，是为了说明 Triton 处在高层框架与手写 CUDA 之间的生产力/性能折中。OpenAI 技术博客用于补充 program-level abstraction 和神经网络 kernel 的工程动机。

第三类来源是 PyTorch compiler 文档。Triton 在实际项目里经常不是单独出现：`torch.compile` / Inductor 可能生成 Triton kernel，高层库也可能内部封装 Triton 路径。正文关于 graph capture、JIT cache、fallback、动态 shape 和线上冷启动的说法，要回到 PyTorch compiler 文档和真实 trace 验证。

第四类来源是 GPU 优化背景。CUDA Best Practices、Nsight 和 CUTLASS/CuTe 资料用于建立共通判断：coalescing、occupancy、memory throughput、tile 复用、epilogue fusion、Tensor Core 路径和 shape bucket。正文只把这些作为 Triton 读法背景，不展开成完整 CUDA 教程。

第五类来源是中文教程。Datawhale Triton 教程和中文社区资料适合学习教学顺序：先用 vector add 建立 offsets/mask，再进 matmul tile 和 autotune。事实、API 语义、性能 claim 和图片来源仍以官方文档、论文和稳定技术博客为准。

## 写作边界

正文不把 Triton 写成现成库替代品。常规 GEMM、conv、attention 若已有 cuBLAS/cuDNN/FlashAttention/CUTLASS 高质量实现，优先用库做 baseline；Triton 的价值主要在库不覆盖、需要快速融合或 shape 相对固定的自定义路径。

正文不把 autotune 写成自动发明算法。它只在给定 `triton.Config` 候选里搜索；候选的数据流、地址计算、tile 设计和端到端指标仍由工程师负责。

正文不只看热态 microbenchmark。Triton 的 JIT 编译、autotune、cache miss、fallback、dynamic shape 和 graph capture 不稳定都可能进入真实训练或服务路径。上线判断必须包含 cold start、warmup、TTFT/TPOT/P99、训练 step time 和回归测试。

正文不把 stride 和 mask 当作语法细节。Triton kernel 的正确性经常毁在非 contiguous layout、最后一个 tile、batch/head stride、dtype cast 和边界 mask 上。讲法要让读者优先看地址矩阵，而不是只看算子公式。

## 来源

| 来源 | 类型 | 本站使用方式 | 关联主题 |
| --- | --- | --- | --- |
| [Triton Vector Addition tutorial](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html) | 官方教程 | 支撑 `program_id`、offset、mask、load/store 的最小示例。 | 编程模型 |
| [Triton Matrix Multiplication tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html) | 官方教程 | 支撑 matmul tile、`tl.dot`、group ordering、autotune 和 L2 locality。 | matmul |
| [Triton `autotune` API](https://triton-lang.org/main/python-api/generated/triton.autotune.html) | 官方 API | 支撑 `configs`、`key`、reset/restore 和候选配置搜索语义。 | autotune |
| [Triton paper](https://doi.org/10.1145/3315508.3329973) | 论文 / 图源 | 使用 roofline comparison 图说明 tiled neural network DSL 的定位。 | 图源 |
| [Introducing Triton](https://openai.com/index/triton/) | 技术博客 | 支撑 Triton 作为自定义 neural network kernel 中间层的工程动机。 | 写作讲法 |
| [PyTorch `torch.compile` docs](https://docs.pytorch.org/docs/stable/torch.compiler.html) | 官方文档 | 支撑 Inductor、graph capture、编译缓存和生成 kernel 的上层路径。 | compiler |
| [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) | 官方文档 | 支撑 coalescing、occupancy、memory throughput、profiling 背景。 | GPU 基础 |
| [Nsight Compute Roofline](https://docs.nvidia.com/nsight-compute/2023.3/NsightCompute/index.html) | 官方文档 | 支撑不要只看 FLOPs，要用 bandwidth/roofline 判断 kernel 瓶颈。 | 性能分析 |
| [CUTLASS documentation](https://github.com/NVIDIA/cutlass) | 官方仓库/文档 | 作为 Triton 之外的模板化 GEMM/attention kernel family 对照。 | kernel stack |
| [Datawhale awesome-triton](https://datawhalechina.github.io/awesome-triton/) | 中文教程 | 借鉴中文教程式讲法，事实和 API 细节回官方文档核对。 | 中文讲法 |
