# 基础概念专题：把数学工具组装成模型机制

基础概念用于把前置数学接到模型结构、训练目标、数据、生成、多模态、检索和工程成本。按下面分组定位需要补的机制；读专题或论文卡住时，用“从下游专题倒回来读”反查。

## 计算、梯度与运行时

这一组先建立模型执行视角：公式怎样落到 tensor，tensor 怎样落到计算图和 GEMM，backward 怎样得到梯度，最后怎样变成显存和运行时成本。

| 你需要解释的现象 | 建议阅读 |
| --- | --- |
| 公式里的下标、期望、loss 和梯度怎样读成接口 | [读懂公式的最小数学](symbols-and-minimal-math.md) |
| shape、axis、broadcast、stride 和计算图为什么会导致错位或 OOM | [张量、Shape 与计算图](tensors-shapes-and-computation-graphs.md) |
| Linear、MLP、参数量、FLOPs、M/N/K 和 GEMM 怎样对应 | [线性层、MLP 与 GEMM](linear-layers-mlp-and-gemm.md) |
| backward 为什么需要 activation，checkpointing 为什么用重算换显存 | [自动微分与激活显存](autograd-activation-checkpointing-and-memory.md) |
| 采样、argmax、round、teacher 和旧策略怎样获得训练信号 | [梯度估计、离散变量与训练信号](gradient-estimators-and-training-signals.md) |
| BF16、FP8、INT4、带宽、kernel、TTFT 和 TPOT 怎样形成运行时账本 | [数值、显存与运行时](numerics-memory-and-runtime-basics.md) |

## 模型结构

这一组回答信息在网络里怎样流动：token 读谁、位置怎样进入 attention、深层网络怎样稳定、视觉怎样形成多尺度特征、稀疏专家和状态空间怎样改变计算路径。

| 你需要解释的现象 | 建议阅读 |
| --- | --- |
| 文本、图像、视频和动作怎样变成 token，再经过 Q/K/V 和 FFN | [Transformer、Tokenization 与注意力](transformer-attention-and-tokenization.md) |
| RoPE、ALiBi、causal mask、packing 和 context extrapolation 在控制什么 | [位置编码与 Mask](positional-encoding-masks-and-context.md) |
| Norm、residual 和 activation 为什么共同决定深层训练稳定性 | [归一化、残差与激活](normalization-residual-and-activation.md) |
| 卷积、channel、感受野、U-Net 和视觉 token 怎样连接 | [卷积与视觉特征](convolution-and-feature-extraction.md) |
| Total Params、Activated Params、router、expert 和 all-to-all 怎样读 | [MoE 与大模型架构](moe-and-large-model-architecture-basics.md) |
| SSM、selective scan、Mamba 和混合 Transformer 怎样维护序列状态 | [Mamba 与混合 SSM 架构](mamba-and-hybrid-ssm-transformers.md) |

## 训练目标与数据

这一组回答模型被要求学什么、梯度是否可信、数据和评测能否支撑部署结论。

| 你需要解释的现象 | 建议阅读 |
| --- | --- |
| loss、batch、学习率、optimizer、clipping 和训练曲线怎样组成一次可靠更新 | [优化与训练入门](optimization-and-training-basics.md) |
| next-token、masked、contrastive、denoising、BC 和 anti-collapse 各自保留什么 | [预训练目标与表征学习](pretraining-objectives-and-representation-learning.md) |
| likelihood、VAE、ELBO、EBM、diffusion 和 RSSM 的概率接口有什么不同 | [概率与潜变量模型](probability-latent-variables-and-generative-models.md) |
| soft target、hidden state、rationale 和少步生成怎样从 teacher 迁移 | [模型蒸馏入门](model-distillation-basics.md) |
| 样本、标签、processor、metadata、清洗和版本怎样组成训练契约 | [数据与数据集基础](data-and-dataset-basics.md) |
| split、污染、指标、置信区间、分桶和协议怎样形成评测证据 | [数据划分与评测指标](data-splits-metrics-and-evaluation-basics.md) |
| 训练 loss 为什么不等于部署风险，正则化和 OOD 怎样一起判断 | [泛化、正则化与分布偏移](generalization-regularization-and-distribution-shift.md) |

## 生成、多模态与检索

这一组把模型输出接到真实使用流程：怎样选择 token，怎样融合图像和视频证据，怎样把外部知识和工具变成可验证输入。

| 你需要解释的现象 | 建议阅读 |
| --- | --- |
| greedy、top-k/top-p、beam、结构化解码、投机解码和 KV cache 怎样配合 | [生成与解码基础](generation-and-decoding-basics.md) |
| 模态对齐、视觉证据、CoT、工具和 verifier 怎样组成多模态推理链 | [多模态推理基础](multimodal-cot-and-reasoning-basics.md) |
| Prompt、CoT、RAG、ReAct 和工具调用怎样组成可验证信息流 | [Prompt、CoT 与 RAG](prompt-engineering-cot-and-rag-basics.md) |

## 压缩与架构搜索

| 你需要解释的现象 | 建议阅读 |
| --- | --- |
| 参数少为什么不一定快，量化、蒸馏、剪枝、NAS 和 kernel 应怎样匹配瓶颈 | [模型压缩、剪枝与 NAS](model-compression-pruning-and-nas-basics.md) |

## 从下游专题倒回来读

下面的表把四层知识放在同一行。读论文卡住时，从当前列向左补；准备继续深入时，向右走。

| 数学先修 | 基础机制 | 专题入口 | 论文例子 |
| --- | --- | --- | --- |
| [线性代数](../prerequisite-math/linear-algebra-and-geometry.md)、[数值与复杂度](../prerequisite-math/numerics-complexity-and-estimation.md) | [张量与计算图](tensors-shapes-and-computation-graphs.md)、[Linear/GEMM](linear-layers-mlp-and-gemm.md)、[自动微分](autograd-activation-checkpointing-and-memory.md) | [分布式训练与 Checkpoint](../training/distributed-training-and-checkpointing.md) | [ZeRO](../paper-deep-dives/foundations/zero.md)、[Megatron-LM](../paper-deep-dives/foundations/megatron-lm.md) |
| [谱、Hessian 与 Fisher](../prerequisite-math/spectral-second-order-and-low-rank.md)、[Jacobian](../prerequisite-math/differential-jacobian-and-gradient-flow.md)、[数值](../prerequisite-math/numerics-complexity-and-estimation.md) | [运行时数值](numerics-memory-and-runtime-basics.md)、[梯度估计/STE](gradient-estimators-and-training-signals.md)、[模型压缩](model-compression-pruning-and-nas-basics.md) | [PTQ / GPTQ / AWQ / SmoothQuant](../quantization/ptq-gptq-awq-smoothquant.md) | [Attn-QAT](../paper-deep-dives/foundations/attn-qat.md)、[Low-bit LLM Survey](../paper-deep-dives/inference/low-bit-llm-survey.md) |
| [序列与掩码](../prerequisite-math/sequence-position-and-masking.md)、[离散选择](../prerequisite-math/discrete-selection-topk-and-routing.md)、[数值](../prerequisite-math/numerics-complexity-and-estimation.md) | [Transformer](transformer-attention-and-tokenization.md)、[位置编码](positional-encoding-masks-and-context.md)、[生成与解码](generation-and-decoding-basics.md) | [KV、缓存与投机解码](../inference/caching-routing-and-speculative.md) | [vLLM / PagedAttention](../paper-deep-dives/inference/vllm-pagedattention.md)、[EAGLE-2](../paper-deep-dives/inference/eagle-2.md) |
| [微积分与 ODE/SDE](../prerequisite-math/calculus-and-newton-leibniz.md)、[概率](../prerequisite-math/probability-distributions-and-expectation.md)、[生成目标](../prerequisite-math/latent-likelihood-and-generative-objectives.md) | [概率与潜变量](probability-latent-variables-and-generative-models.md)、[梯度估计](gradient-estimators-and-training-signals.md)、[模型蒸馏](model-distillation-basics.md) | [Score Matching 到 SDE](../diffusion/score-matching-sde-and-probability-flow.md)、[一步生成与蒸馏](../diffusion/distillation.md) | [DPM-Solver++](../paper-deep-dives/diffusion/dpm-solver-plus-plus.md)、[DMD2](../paper-deep-dives/diffusion/dmd2.md) |
| [概率](../prerequisite-math/probability-distributions-and-expectation.md)、[随机过程](../prerequisite-math/stochastic-processes-and-markov.md)、[价值与策略](../prerequisite-math/value-policy-and-rl-objectives.md)、[控制](../prerequisite-math/dynamic-systems-and-control.md) | [预训练目标](pretraining-objectives-and-representation-learning.md)、[概率与潜变量](probability-latent-variables-and-generative-models.md)、[梯度估计](gradient-estimators-and-training-signals.md) | [RSSM、Dreamer 与规划](../world-models/rssm-dreamer-and-planning.md) | [DreamerV2](../paper-deep-dives/world-models/dreamerv2.md)、[MuZero](../paper-deep-dives/world-models/muzero.md) |
| [线性代数](../prerequisite-math/linear-algebra-and-geometry.md)、[信息论](../prerequisite-math/information-theory-and-divergence.md)、[卷积频域](../prerequisite-math/convolution-frequency-and-multiscale.md) | [卷积与视觉特征](convolution-and-feature-extraction.md)、[表示学习](pretraining-objectives-and-representation-learning.md)、[多模态推理](multimodal-cot-and-reasoning-basics.md) | [VLM 架构与训练](../vlm/architecture-and-training.md) | [V-JEPA](../paper-deep-dives/world-models/v-jepa.md)、[BAGEL](../paper-deep-dives/world-models/bagel.md) |
| [三维几何](../prerequisite-math/geometry-vision-and-3d.md)、[控制](../prerequisite-math/dynamic-systems-and-control.md)、[分布偏移](../prerequisite-math/causality-counterfactual-and-distribution-shift.md) | [数据与数据集](data-and-dataset-basics.md)、[评测指标](data-splits-metrics-and-evaluation-basics.md)、[多模态推理](multimodal-cot-and-reasoning-basics.md) | [VLA 动作表示](../vla/action-representation-and-interfaces.md)、[具身智能路线](../embodied-ai/beginner-learning-path.md) | [RT-2](../paper-deep-dives/embodied-ai/rt2.md)、[SpatialVLA](../paper-deep-dives/embodied-ai/spatialvla.md)、[VGGT](../paper-deep-dives/embodied-ai/vggt.md) |
| [序列](../prerequisite-math/sequence-position-and-masking.md)、[概率](../prerequisite-math/probability-distributions-and-expectation.md)、[采样评估](../prerequisite-math/sampling-estimation-and-calibration.md) | [Prompt / CoT / RAG](prompt-engineering-cot-and-rag-basics.md)、[生成与解码](generation-and-decoding-basics.md)、[评测指标](data-splits-metrics-and-evaluation-basics.md) | [RAG、Agent 与长上下文系统](../inference/rag-agents-and-long-context-systems.md) | [Gemini 2.5](../paper-deep-dives/technical-reports/gemini-2-5.md) |
