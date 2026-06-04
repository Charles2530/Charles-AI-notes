# 知识 QA 来源台账

这页记录 [知识 QA](../knowledge-qa/foundations-and-transformer.md) 板块的来源口径。它不是普通链接收藏，而是说明：哪些外部材料只用于判断“高频问法和覆盖面”，哪些材料才能支撑事实、机制、边界和继续阅读。

## 维护规则

| 规则 | 说明 |
| --- | --- |
| 面经只定问题，不定答案 | 公开面经和题库用于抽取常见问法、主题覆盖和追问方向，不复制原文答案。 |
| 事实回到原始来源 | 技术机制、性能 claim、模型细节和数值边界优先回到论文、官方文档、官方仓库或站内来源台账。 |
| 每个答案要能面试复述 | QA 的 `面试回答` 字段要能支撑 1-2 分钟口头回答，必要时补最小公式和变量解释。 |
| 每个答案要有边界 | QA 的 `边界` 字段必须说明不能外推到哪里，尤其区分论文结果、官方 demo、系统吞吐和真实闭环。 |
| 继续读优先站内 | QA 的 `继续读` 优先链接到站内长文；站内没有完整解释时才补外部原始来源。 |

## 覆盖面来源

这些材料用于判断 AI 技术面试和知识速查里哪些问题反复出现。它们帮助组织问题，但不直接作为强事实证据。

| 来源 | 类型 | 对 QA 的作用 | 使用边界 |
| --- | --- | --- | --- |
| [Deep Learning Interviews](https://arxiv.org/abs/2201.00650) | 书 / arXiv | 覆盖深度学习、数学、优化和模型基础问题，帮助保证基础 QA 不只围绕 LLM。 | 用作主题覆盖参考；具体机制仍回本站基础页和原始论文。 |
| [wdndev/llm_interview_note](https://github.com/wdndev/llm_interview_note) | 中文开源 LLM 面试笔记 | 覆盖 tokenizer、Transformer、训练、微调、RL、RAG 和 Agent 等中文高频问法。 | 只学习问题组织和覆盖范围，不复制答案。 |
| [AI 应用开发工程师面试宝典](https://github.com/guocong-bincai/ai-interview-guide) | 中文 AI 应用面试题库 | 提醒 QA 覆盖 RAG、Agent、工程架构、部署、评估和安全。 | 应用开发题库偏产品落地，事实仍需官方文档或站内系统页核验。 |
| [AgentGuide](https://github.com/adongwanai/AgentGuide) | Agent / RAG / 面试资料 | 补充手撕 attention、RAG、Agent、模型评估和大厂面经覆盖。 | 用作追问方向，不作为 benchmark 或技术成熟度证据。 |
| [AI 智能体与大模型应用开发面试题库](https://github.com/didilili/ai-agents-from-zero/blob/main/AI%E6%99%BA%E8%83%BD%E4%BD%93%E4%B8%8E%E5%A4%A7%E6%A8%A1%E5%9E%8B%E5%BA%94%E7%94%A8%E5%BC%80%E5%8F%91%E9%9D%A2%E8%AF%95%E9%A2%98%E5%BA%93.md) | 应用开发题库 | 提醒 QA 标注 RAG、Agent、工程治理、安全和项目实战问题。 | 不把题库表述写成本站结论。 |
| 牛客公开大模型面经和题库 | 面经平台 | 抽取国内算法岗常问的训练、推理、量化、RLHF、RAG、Agent 和部署问题。 | 内容质量参差；只做高频信号，不做事实来源。 |

## 事实核验来源：基础与 Transformer

| 问题域 | 主要站内入口 | 原始或官方来源 |
| --- | --- | --- |
| Transformer、attention、mask、tokenizer | [Transformer 输入与注意力](../foundations/transformer-attention-and-tokenization.md)、[位置编码与 Mask](../foundations/positional-encoding-masks-and-context.md) | [Attention Is All You Need](https://arxiv.org/abs/1706.03762)、[RoFormer / RoPE](https://arxiv.org/abs/2104.09864)、[Hugging Face BPE 课程](https://huggingface.co/docs/course/chapter6/5) |
| 张量、shape、MLP、GEMM | [张量、Shape 与计算图](../foundations/tensors-shapes-and-computation-graphs.md)、[线性层、MLP 与 GEMM](../foundations/linear-layers-mlp-and-gemm.md) | [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)、[Triton paper](https://doi.org/10.1145/3315508.3329973) |
| 生成与解码 | [生成与解码基础](../foundations/generation-and-decoding-basics.md) | [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751)、[Hugging Face 生成策略](https://huggingface.co/docs/transformers/generation_strategies) |

## 事实核验来源：训练与对齐

| 问题域 | 主要站内入口 | 原始或官方来源 |
| --- | --- | --- |
| 预训练、数据、优化器 | [模型训练从零入门](../training/beginner-model-training-primer.md)、[目标函数、优化器与 LR 日程](../training/objectives-optimizers-and-schedules.md) | [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/)、[Smol Training Playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook) |
| 分布式训练 | [分布式训练与 Checkpoint](../training/distributed-training-and-checkpointing.md)、[Megatron / DeepSpeed](../training/megatron-lm-deepspeed-and-open-training-stacks.md) | [Megatron-LM](https://arxiv.org/abs/1909.08053)、[ZeRO](https://arxiv.org/abs/1910.02054)、[DeepSpeed ZeRO 文档](https://deepspeed.readthedocs.io/en/latest/zero3.html) |
| 对齐、RLHF、DPO、GRPO | [预训练、微调与对齐](../training/pretraining-finetuning-alignment.md)、[Policy Gradient、Actor-Critic、PPO 与 GRPO](../reinforcement-learning/policy-gradient-actor-critic-ppo-grpo.md) | [DeepSeek-R1](../paper-deep-dives/technical-reports/deepseek-r1.md)、[偏好数据与对齐陷阱](../training/preference-data-and-alignment-pitfalls.md) |

## 事实核验来源：推理、量化与算子

| 问题域 | 主要站内入口 | 原始或官方来源 |
| --- | --- | --- |
| KV cache、batching、运行时 | [推理运行时](../inference/serving-runtimes-vllm-sglang-and-tensorrt-llm.md)、[KV、缓存与投机解码](../inference/caching-routing-and-speculative.md) | [vLLM PagedAttention blog](https://vllm.ai/blog/2023-06-20-vllm)、[SGLang paper](https://arxiv.org/abs/2312.07104)、[TensorRT-LLM KV Cache](https://nvidia.github.io/TensorRT-LLM/features/kvcache.html) |
| 投机解码、长上下文 | [上下文压缩、KV 驱逐与记忆层级](../inference/context-compression-kv-eviction-and-memory-hierarchies.md)、[EAGLE 专题](../paper-deep-dives/inference/eagle.md) | [SpecInfer](https://arxiv.org/abs/2305.09781)、[EAGLE-2](https://arxiv.org/abs/2406.16858) |
| 量化 | [PTQ / GPTQ / AWQ / SmoothQuant](../quantization/ptq-gptq-awq-smoothquant.md)、[FP8 与混合精度推理](../quantization/fp8-and-mixed-precision-serving.md) | [Hugging Face Quantization concept guide](https://huggingface.co/docs/transformers/quantization/concept_guide)、[GPTQ](https://arxiv.org/abs/2210.17323)、[AWQ](https://arxiv.org/abs/2306.00978)、[SmoothQuant](https://arxiv.org/abs/2211.10438) |
| CUDA、Triton、FlashAttention | [CUDA 编程模型与内存层次](../operators/cuda-programming-model-and-memory.md)、[FlashAttention 与长上下文](../operators/long-context-and-flashattention-evolution.md)、[Triton 编程模型与自动调优](../operators/triton-programming-model-and-autotuning.md) | [Triton paper](https://doi.org/10.1145/3315508.3329973)、[CUTLASS GEMM docs](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api.html)、[Nsight Compute Roofline](https://docs.nvidia.com/nsight-compute/2023.3/NsightCompute/index.html#rooflines) |

## 事实核验来源：RAG、Agent、评估与多模态

| 问题域 | 主要站内入口 | 原始或官方来源 |
| --- | --- | --- |
| RAG、Agent、长上下文 | [RAG、Agent 与长上下文系统](../inference/rag-agents-and-long-context-systems.md)、[观测与在线评估](../inference/observability-and-online-evaluation.md) | [OpenAI Structured Outputs](https://openai.com/index/introducing-structured-outputs-in-the-api/)、[SGLang documentation](https://docs.sglang.io/) |
| VLM 与视觉连接器 | [VLM 架构](../vlm/architecture-and-training.md)、[视觉 Tokenizer 与连接器](../vlm/visual-tokenizers-and-connectors.md) | [CLIP](https://arxiv.org/abs/2103.00020)、[BLIP-2](https://arxiv.org/abs/2301.12597)、[LLaVA](https://llava-vl.github.io/) |
| 扩散、视频生成 | [扩散训练与表示](../diffusion/training.md)、[视频与多模态扩散](../diffusion/video-and-multimodal-diffusion.md) | [DDPM](https://arxiv.org/abs/2006.11239)、[Score SDE](https://arxiv.org/abs/2011.13456)、[DMD](https://arxiv.org/abs/2311.18828) |
| 世界模型、VLA、具身智能 | [从零理解世界模型](../world-models/beginner-world-models-primer.md)、[VLA 动作表示](../vla/action-representation-and-interfaces.md)、[具身智能从零路线](../embodied-ai/beginner-learning-path.md) | [Dreamer](../paper-deep-dives/world-models/dreamer.md)、[V-JEPA](../paper-deep-dives/world-models/v-jepa.md)、[RT-2](https://arxiv.org/abs/2307.15818)、[Open X-Embodiment](https://robotics-transformer-x.github.io/) |

## QA 页面覆盖登记

| QA 页面 | 计划覆盖 | 主要核验入口 |
| --- | --- | --- |
| [基础与 Transformer](../knowledge-qa/foundations-and-transformer.md) | 机器学习基础、Transformer、tokenizer、位置编码、解码 | foundations、training、inference source ledgers |
| [训练与对齐](../knowledge-qa/training-and-alignment.md) | 预训练、SFT、RLHF/DPO/GRPO、分布式训练、实验治理 | training、reinforcement-learning、paper-deep-dives |
| [推理服务与量化](../knowledge-qa/inference-serving-and-quantization.md) | KV cache、运行时、投机解码、量化和上线验收 | inference、quantization、paper-deep-dives |
| [算子与性能](../knowledge-qa/operators-and-performance.md) | GEMM、FlashAttention、Triton/CUDA、profiling、通信 | operators、training、inference |
| [RAG、Agent 与评估](../knowledge-qa/rag-agent-and-evaluation.md) | RAG、Agent、tool calling、LLM-as-judge、安全与在线评估 | inference、foundations、evaluation source ledger |
| [多模态与生成模型](../knowledge-qa/multimodal-generative-and-embodied.md) | VLM、扩散、视频生成、多模态评估 | vlm、diffusion、technical-reports |
| [世界模型](../knowledge-qa/world-models.md) | RSSM、Dreamer、JEPA、动作条件建模、WAM/VAM、闭环评测 | world-models、paper-deep-dives/world-models、claim ledger |
| [具身智能与 VLA](../knowledge-qa/embodied-ai-and-vla.md) | VLA 动作接口、数据闭环、Sim2Real、机器人视觉、评测与部署 | embodied-ai、vla、paper-deep-dives/embodied-ai |

## 后续维护检查

新增或修改 QA 时，按下面顺序检查。

1. 问题是否来自真实高频困惑，而不是为了凑目录。
2. `面试回答` 是否能在 1-2 分钟内说清对象、机制、公式或工程取舍。
3. `追问展开` 是否覆盖常见追问，而不是重复定义。
4. `边界` 是否明确不能外推的范围。
5. `继续读` 是否至少给出一个站内长文或原始来源。
6. 如果引用前沿论文、官方 demo 或系统吞吐，是否遵守 [全站证据与复现状态标准](evidence-and-reproducibility-standard.md) 和 [证据判断原则](site-judgement-principles.md)。
