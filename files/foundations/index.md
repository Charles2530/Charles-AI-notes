# 基础概念专题：把术语变成可推理的结构

基础篇不是背名词表，而是给后面的训练、推理、量化、VLM、VLA 和世界模型准备共同语言。读这里时要抓三件事：张量在什么维度上流动，目标函数在逼近什么分布，工程实现为什么会受内存、数值和数据分布限制。

本站会参考高质量博客和微信公众号的讲法来安排节奏：先给直觉，再拆机制，最后说边界。但事实、公式和图示仍以论文、官方文档和可复现工程资料为准，来源集中记录在[外部阅读总账](../references/external-reading-ledger.md)及各专题台账里。

## 先建立计算视角

| 你卡住的问题 | 建议先读 |
| --- | --- |
| 不知道 shape、broadcast、反向图为什么总出错 | [张量、Shape 与计算图](tensors-shapes-and-computation-graphs.md) |
| 想把矩阵乘、MLP、GEMM 和参数量联系起来 | [线性层、MLP 与 GEMM](linear-layers-mlp-and-gemm.md) |
| 想理解卷积、特征图和视觉 token 的关系 | [卷积与视觉特征提取](convolution-and-feature-extraction.md) |
| 想知道 autograd、checkpoint 和显存重算怎么接上 | [Autograd、激活检查点与显存](autograd-activation-checkpointing-and-memory.md) |

## 再进入大模型机制

| 你想解释的现象 | 建议先读 |
| --- | --- |
| token 如何互相读信息，mask 又在挡什么 | [Transformer 输入与注意力](transformer-attention-and-tokenization.md) |
| 位置编码为什么影响长上下文外推 | [位置编码、Mask 与上下文](positional-encoding-masks-and-context.md) |
| normalization 和 residual 为什么决定训练稳定性 | [Normalization、Residual 与 Activation](normalization-residual-and-activation.md) |
| MoE、Mamba、SSM 这些架构和 Transformer 如何互补 | [MoE 与大模型架构基础](moe-and-large-model-architecture-basics.md)、[Mamba 与混合 SSM Transformer](mamba-and-hybrid-ssm-transformers.md) |

## 最后补上学习目标

| 后续方向 | 基础篇里要先会什么 |
| --- | --- |
| 训练 | [优化与训练基础](optimization-and-training-basics.md)、[数据与数据集基础](data-and-dataset-basics.md) |
| 推理 | [生成与解码基础](generation-and-decoding-basics.md)、[数值、内存与运行时基础](numerics-memory-and-runtime-basics.md) |
| 扩散与世界模型 | [概率、潜变量与生成模型](probability-latent-variables-and-generative-models.md)、[预训练目标与表征学习](pretraining-objectives-and-representation-learning.md) |
| 多模态与 Agent | [多模态 CoT 与推理基础](multimodal-cot-and-reasoning-basics.md)、[Prompt、CoT 与 RAG 基础](prompt-engineering-cot-and-rag-basics.md) |

## 如果从下游专题倒回来读

| 下游方向 | 先补的基础 | 再接的专题入口 |
| --- | --- | --- |
| 分布式训练 / Megatron / DeepSpeed | [张量、Shape 与计算图](tensors-shapes-and-computation-graphs.md)、[线性层、MLP 与 GEMM](linear-layers-mlp-and-gemm.md)、[Autograd、激活检查点与显存](autograd-activation-checkpointing-and-memory.md) | [分布式训练与 Checkpoint](../training/distributed-training-and-checkpointing.md) |
| 低比特训练与量化 | [数值、内存与运行时基础](numerics-memory-and-runtime-basics.md)、[Normalization、Residual 与 Activation](normalization-residual-and-activation.md)、[模型压缩、剪枝与 NAS](model-compression-pruning-and-nas-basics.md) | [低比特训练与数值格式](../training/low-bit-training-and-numerics.md)、[FP8 与混合精度推理](../quantization/fp8-and-mixed-precision-serving.md) |
| 推理服务 / KV cache / 投机解码 | [Transformer 输入与注意力](transformer-attention-and-tokenization.md)、[位置编码、Mask 与上下文](positional-encoding-masks-and-context.md)、[生成与解码基础](generation-and-decoding-basics.md) | [推理运行时](../inference/serving-runtimes-vllm-sglang-and-tensorrt-llm.md)、[KV、缓存与投机解码](../inference/caching-routing-and-speculative.md) |
| 扩散、一致性模型与蒸馏 | [概率、潜变量与生成模型](probability-latent-variables-and-generative-models.md)、[模型蒸馏入门](model-distillation-basics.md)、[数值、内存与运行时基础](numerics-memory-and-runtime-basics.md) | [一步生成、蒸馏与整流](../diffusion/distillation.md) |
| 世界模型 / JEPA / Dreamer | [预训练目标与表征学习](pretraining-objectives-and-representation-learning.md)、[概率、潜变量与生成模型](probability-latent-variables-and-generative-models.md)、[泛化、正则化与分布偏移](generalization-regularization-and-distribution-shift.md) | [世界模型专题](../world-models/index.md) |
| VLM / Omni / 多模态报告 | [卷积与视觉特征提取](convolution-and-feature-extraction.md)、[多模态 CoT 与推理基础](multimodal-cot-and-reasoning-basics.md)、[Prompt、CoT 与 RAG 基础](prompt-engineering-cot-and-rag-basics.md) | [视觉 Tokenizer 与连接器](../vlm/visual-tokenizers-and-connectors.md) |
| VLA / 具身智能 | [张量、Shape 与计算图](tensors-shapes-and-computation-graphs.md)、[卷积与视觉特征提取](convolution-and-feature-extraction.md)、[数据与数据集基础](data-and-dataset-basics.md)、[数据划分与评测指标](data-splits-metrics-and-evaluation-basics.md) | [VLA 动作表示与控制接口](../vla/action-representation-and-interfaces.md)、[具身智能从零路线](../embodied-ai/beginner-learning-path.md) |
| RAG / Agent / Judge | [Prompt、CoT 与 RAG 基础](prompt-engineering-cot-and-rag-basics.md)、[生成与解码基础](generation-and-decoding-basics.md)、[数据划分与评测指标](data-splits-metrics-and-evaluation-basics.md) | [RAG、Agent 与评估](../knowledge-qa/rag-agent-and-evaluation.md) |

读基础篇最有用的方式，是每读完一节就问：这个概念在后面的系统里控制了什么代价、什么能力、什么失败模式。能回答这个问题，名词就不再只是名词。
