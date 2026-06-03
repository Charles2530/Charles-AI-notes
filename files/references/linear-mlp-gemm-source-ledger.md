# 线性层、MLP 与 GEMM 来源台账

这份台账服务于 [线性层、MLP 与 GEMM](../foundations/linear-layers-mlp-and-gemm.md)。正文不把 Linear、MatMul、GEMM 写成同义词，而是按三层讲：模型语义、数学/API 操作和硬件执行。

## 使用方式

第一类来源是框架 API。PyTorch `nn.Linear` 文档用于确认 `in_features`、`out_features`、weight shape、bias 和前向定义，避免把论文里的 \(xW+b\) 和实际存储 layout 混淆。

第二类来源是 GPU 线性代数与 kernel 文档。cuBLAS、cuBLASLt、CUTLASS GEMM API、CUTLASS 3.x blog 和 Triton paper 帮助本站解释 GEMM 的 \(M/N/K\)、tile、MMA、epilogue fusion、layout、dtype 和 roofline 视角。

第三类来源是大模型并行论文。Megatron-LM Figure 3a 用来说明张量并行不是抽象切模型，而是切 MLP 的 Linear/GEMM 维度，并把通信安排在必要位置。

第四类来源是本站内部系统页。量化、低精度 kernel、CUDA 内存层次、GEMM/Attention 融合 kernel、DeepGEMM 和推理 runtime 页面都依赖这篇基础页提供的概念边界。

中文材料主要用于学习讲法：先把“Linear 是模型层，GEMM 是执行层”讲清楚，再引入 \(M,N,K\)、prefill/decode 和 epilogue。事实和图片来源仍以论文、官方文档和稳定技术博客为主。

这页还服务一个全站统一口径：以后遇到 Linear、MLP、MatMul、GEMM、BMM、grouped GEMM、fused dequant GEMM 时，优先说明它位于模型、API、kernel 还是 runtime 哪一层。这样量化、MoE、推理服务和算子页不会把同一个词混用成不同含义。

## 写作边界

正文不把所有矩阵乘都叫 GEMM。MatMul 可以是框架 API、batched tensor op 或高维 broadcasting；GEMM 是 BLAS / kernel 语境中的 dense matrix-matrix multiplication family。

正文不把 FLOPs 当成 latency。GEMM 的真实速度还取决于 tile、layout、dtype、epilogue、memory movement、launch、workspace、shape bucket 和 runtime dispatch。

正文不把量化收益只归因于 bit 数。低比特权重只有在 unpack、scale、dequant、matmul、requant 或 residual 能走高效融合路径时，才可能转成端到端收益。

正文不把张量并行写成笼统“切模型”。Megatron 的价值在于具体切 Linear 权重矩阵，并控制 GeLU 和 all-reduce 出现的位置。

## 来源

| 来源 | 类型 | 本站使用方式 | 关联主题 |
| --- | --- | --- | --- |
| [PyTorch `nn.Linear`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html) | 官方文档 | 支撑 weight shape、bias、前向公式和 API 约定。 | Linear API |
| [NVIDIA cuBLAS documentation](https://docs.nvidia.com/cuda/cublas/index.html) | 官方文档 | 支撑 BLAS/GEMM 库接口和 NVIDIA GPU 矩阵乘背景。 | GEMM |
| [NVIDIA CUTLASS GEMM API](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api.html) | 官方文档 | 支撑 threadblock、warp、instruction、epilogue 等 GEMM 层级解释。 | GEMM kernel |
| [CUTLASS 3.x blog](https://developer.nvidia.com/blog/cutlass-3-x-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design/) | 官方技术博客 | 支撑可组合 GEMM kernel 和 epilogue fusion 的讲法。 | GEMM design |
| [Triton paper](https://doi.org/10.1145/3315508.3329973) | 论文 / 图源 | 使用 roofline comparison 图解释 arithmetic intensity、tiling 和自定义 kernel 价值。 | Triton / roofline |
| [Megatron-LM](https://arxiv.org/abs/1909.08053) | 论文 / 图源 | 使用 Figure 3a 说明 MLP 的列并行、行并行和通信位置。 | Tensor parallel |
| [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) | 工程仓库 | 作为后续低精度和 MoE grouped GEMM 的系统延伸，不作为本页主要图源。 | Low-bit GEMM |
| [NVIDIA Tensor Cores](https://www.nvidia.com/en-us/data-center/tensor-cores/) | 官方技术页 | 背景性解释 Tensor Core 为什么是现代 GEMM 热点路径。 | Hardware |
