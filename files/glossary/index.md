# 术语表

这一页把知识库中反复出现的术语做统一整理。目标不是百科定义，而是在本站语境里说明它主要指什么、最容易和什么概念混淆。

!!! note "初学者先抓住"

    术语表不是要背诵，而是用来防止“同一个词在不同层意思不同”。例如 checkpoint 在训练里是恢复资产，KV cache 在推理里是显存主角，sampler 在扩散和数据管线里又各有语境。

| 使用场景 | 建议做法 |
| --- | --- |
| 读论文时遇到缩写 | 先确认基本含义，再回到论文导读判断它改的是训练、推理、系统还是评测哪一层 |
| 排查系统问题 | 先区分术语属于训练、推理、量化、runtime 还是部署工具，避免把不同层的问题混在一起 |
| 扩写新页面 | 新术语优先补到这里，再从正文链接回来，减少各页重复解释 |

## 训练与优化

| 术语 | 含义 |
| --- | --- |
| `Activation` | 神经网络中间层输出。量化时 activation 往往比权重更难压低比特，因为动态范围更不稳定。 |
| `AdamW` | Transformer 训练中常用优化器，把 weight decay 与梯度更新解耦。 |
| `DPO` | Direct Preference Optimization，常用偏好对齐方法，直接利用偏好对优化策略分布。 |
| `ELBO` | Evidence Lower Bound，变分推断目标，在扩散模型和世界模型中常出现。 |
| `LoRA` | Low-Rank Adaptation，用低秩增量替代全量参数更新的参数高效微调方法。 |
| `QAT` | Quantization-Aware Training，在训练中模拟量化误差，让模型适应低比特约束。 |
| `PTQ` | Post-Training Quantization，训练完成后再进行量化。 |
| `SFT` | Supervised Fine-Tuning，用高质量示范数据教模型遵循任务格式和行为。 |

## 推理与服务系统

| 术语 | 含义 |
| --- | --- |
| `Continuous Batching` | 连续批处理，允许请求在 decode 过程中动态加入和退出，提高 GPU 利用率。 |
| `KV Cache` | Transformer 推理时缓存历史 token 的 key/value 张量，长上下文服务的核心显存瓶颈之一。 |
| `Prefix Cache` | 缓存重复 prompt 前缀，避免反复 prefill。模板化 agent 和企业问答中价值很高。 |
| `Speculative Decoding` | 草稿模型或草稿路径先生成候选 token，再由目标模型并行验证的推理加速方法。 |
| `TTFT` | Time To First Token，首 token 延迟。 |
| `TPOT` | Time Per Output Token，生成阶段每 token 延迟。 |
| `vLLM` | LLM serving runtime，常与 paged attention、continuous batching、KV 管理一起讨论。 |
| `SGLang` | 面向结构化生成和 agent workload 的 serving/runtime 生态。 |

## 量化与数值

| 术语 | 含义 |
| --- | --- |
| `AWQ` | Activation-aware Weight Quantization，量化时考虑 activation 对权重误差的放大。 |
| `BF16` | 16 位浮点格式，指数范围接近 FP32，训练稳定性通常好于 FP16。 |
| `FP8` | 8 位浮点格式，常见 E4M3、E5M2 等变体，依赖 scale 策略和硬件支持。 |
| `GPTQ` | 典型后训练权重量化方法，通过二阶近似减少量化误差。 |
| `MXFP8` | Microscaling FP8，每块配缩放因子，常用于更细粒度低精度训练或推理。 |
| `Outlier` | 激活或权重中的极端值，会显著影响量化和低精度训练稳定性。 |
| `Scale` | 量化中的缩放因子，用于把原始数值映射到低比特表示范围。 |

## 扩散与生成模型

| 术语 | 含义 |
| --- | --- |
| `CFG` | Classifier-Free Guidance，用条件分支和无条件分支组合增强条件对齐。 |
| `DDPM` | Denoising Diffusion Probabilistic Models，现代扩散模型主流路线起点。 |
| `DDIM` | Denoising Diffusion Implicit Models，不重训模型也能减少采样步数的重要方法。 |
| `DMD` | Distribution Matching Distillation，一步扩散蒸馏代表路线之一。 |
| `DMD2` | DMD 的改进版本，重点改善一步蒸馏的模糊和训练不稳。 |
| `DPM-Solver` | 面向 diffusion ODE 的高阶专用求解器，用于少步高质量采样。 |
| `Heun` | 二阶 predictor-corrector 数值求解器，常用于扩散采样。 |
| `LCM` | Latent Consistency Models，少步扩散采样路线之一。 |
| `Phased DMD` | 将 DMD 分布匹配阶段化的方法，目标是更稳地保住结构与细节。 |

## 表征、检索与多模态

| 术语 | 含义 |
| --- | --- |
| `CLIP` | Contrastive Language-Image Pre-training，图文对齐和多模态检索代表方法。 |
| `Embedding` | 向量表示。检索和对比学习中，embedding 空间几何很重要。 |
| `Hard Negative` | 困难负样本，看起来很像但关键语义不同的负样本。 |
| `InfoNCE` | 对比学习经典目标，要求模型把正确匹配项排到候选最前。 |
| `MoCo` | Momentum Contrast，通过动量编码器和队列维护大量负样本。 |
| `BYOL` | Bootstrap Your Own Latent，不显式依赖负样本的自监督方法。 |
| `RAG` | Retrieval-Augmented Generation，用检索结果增强生成模型。 |

## 世界模型、具身与控制

| 术语 | 含义 |
| --- | --- |
| `World Model` | 面向决策的环境预测模型，学习世界如何随动作、时间和目标变化。 |
| `RSSM` | Recurrent State-Space Model，Dreamer 等 latent world model 的核心结构之一。 |
| `MPC` | Model Predictive Control，滚动优化动作序列的经典规划方法。 |
| `WAM` | World-Action Model，强调动作和世界未来的联合建模。 |
| `VAM` | Video-Action Model，强调视频表示与动作建模的统一。 |
| `Near-miss` | 接近失败但尚未造成事故的样本，常用于风险训练和数据回流。 |
| `Counterfactual` | 反事实条件变化，用于比较“如果当时换个动作会怎样”。 |

## 框架与部署工具

| 术语 | 含义 |
| --- | --- |
| `CUDA` | NVIDIA GPU 编程模型和运行时。 |
| `Triton` | 面向张量块 kernel 的 DSL 和 codegen 工具。 |
| `CUTLASS` | CUDA 高性能线性代数和 GEMM 类 kernel 模板库。 |
| `CuTe` | CUTLASS 中强调 layout algebra 和 tile 组合表达的抽象层。 |
| `ONNX Runtime / ORT` | 跨平台推理 runtime，可通过不同 Execution Provider 对接多类硬件。 |
| `TensorRT-LLM` | NVIDIA 面向 LLM 推理优化的 runtime / kernel 生态。 |
| `NCCL` | NVIDIA Collective Communications Library，分布式训练和推理通信基元。 |
| `LightLLM` | 偏轻量、高性能和易扩展的 LLM serving framework。 |
