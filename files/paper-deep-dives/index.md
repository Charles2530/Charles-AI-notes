# 论文精读专题：读机制、证据和不能外推的部分

论文精读页不追求把摘要翻译一遍，而是回答四个问题：这篇工作真正改了哪一个机制；作者用什么实验支撑这个 claim；图表里哪些信息可以迁移到工程判断；还有哪些结论不能外推到别的模型、数据或部署场景。

下面按领域列出已整理的 deep dive。读的时候不必从头到尾扫清单，可以先从自己正在研究的系统问题进入：训练显存、推理延迟、扩散少步、世界模型、VLA 或技术报告。写法会参考高质量博客和公众号的“问题驱动式精读”，但结论仍回到论文原文、代码、官方报告和本站来源台账。

## 怎么读一篇精读

| 阅读动作 | 目的 |
| --- | --- |
| 先读图和核心实验 | 看作者真正想证明什么，而不是只看方法名。 |
| 再读方法拆解 | 判断改动发生在目标函数、架构、数据、系统还是评测。 |
| 最后读局限和失败模式 | 决定这篇工作能不能迁移到你的任务。 |

## 基础与训练

- [ZeRO：为什么数据并行会浪费这么多显存](foundations/zero.md)
- [Megatron-LM：Tensor Model Parallel](foundations/megatron-lm.md)
- [MagiAttention：超长上下文分布式 Attention](foundations/magiattention.md)
- [Attn-QAT：4-bit Attention 量化感知训练](foundations/attn-qat.md)
- [Muon：LLM 预训练优化器](foundations/muon-is-scalable.md)
- [SLA / SLA2：DiT 稀疏线性 Attention](foundations/sla-sparse-linear-attention.md)
- [GPU Utilization：为什么 100% 不等于训练跑满](foundations/gpu-utilization-misleading.md)

## 推理系统

- [vLLM / PagedAttention：KV cache 分页管理](inference/vllm-pagedattention.md)
- [EAGLE：为什么 draft 不一定要是小模型](inference/eagle.md)
- [EAGLE-2：动态 Draft Tree](inference/eagle-2.md)
- [EAGLE-3：训练时见过自己犯错](inference/eagle-3.md)
- [KVSlimmer：非对称 KV 合并](inference/kvslimmer.md)
- [Low-bit LLM Survey：低比特大模型系统](inference/low-bit-llm-survey.md)
- [Fast-FoundationStereo：实时 Zero-Shot 双目匹配](inference/fast-foundation-stereo.md)

## 扩散生成

- [Score SDE：连续时间生成过程](diffusion/score-sde.md)
- [DPM-Solver++：Guidance 场景 ODE 求解器](diffusion/dpm-solver-plus-plus.md)
- [CPS：Flow-RL 的系数保持采样](diffusion/cps.md)
- [DMD：一步扩散蒸馏](diffusion/dmd.md)
- [DMD2：更稳的少步分布匹配蒸馏](diffusion/dmd2.md)
- [Phased DMD：分阶段少步蒸馏](diffusion/phased-dmd.md)
- [AnyFlow：任意步视频扩散蒸馏](diffusion/anyflow.md)
- [CausVid：流式自回归视频扩散](diffusion/causvid.md)
- [Diffusion Forcing：next-token 与全序列扩散](diffusion/diffusion-forcing.md)
- [Wan2.1：开源视频生成系统路线](diffusion/wan.md)

## 世界模型

- [PlaNet：像素规划的潜变量动力学](world-models/planet.md)
- [MuZero：只预测对规划有用的世界模型](world-models/muzero.md)
- [Dreamer：想象轨迹中训练控制策略](world-models/dreamer.md)
- [DreamerV2：离散世界模型玩 Atari](world-models/dreamerv2.md)
- [DreamerV3：latent 里训练策略](world-models/dreamerv3.md)
- [JEPA：预测式表征学习路线](world-models/jepa.md)
- [H-JEPA：层级 JEPA 与潜变量 EBM](world-models/h-jepa.md)
- [V-JEPA：视频潜变量预测表征](world-models/v-jepa.md)
- [Genie：无动作标签学可交互环境](world-models/genie.md)
- [LWM：百万 token 视频语言世界模型](world-models/lwm.md)
- [RingAttention：近无限上下文训练](world-models/ringattention.md)
- [Self Forcing：对齐视频训练与推理](world-models/self-forcing.md)
- [Towards Video World Models：视频世界模型门槛](world-models/towards-video-world-models.md)
- [World Models Survey：具身世界模型综述](world-models/world-models-embodied-ai-survey.md)
- [World Model for Robot Learning：机器人学习综述](world-models/world-model-for-robot-learning-survey.md)
- [LingBot-World：视频基础模型到世界模拟器](world-models/lingbot-world.md)
- [Lance：统一多模态世界建模启发](world-models/lance.md)
- [BAGEL：交错多模态预训练](world-models/bagel.md)
- [Emu3.5：原生多模态世界学习器](world-models/emu3-5.md)
- [Beyond Language Modeling：多模态预训练与世界模型](world-models/beyond-language-modeling.md)

## 具身智能

- [RT-2：视觉语言模型接到机器人动作](embodied-ai/rt2.md)
- [GR-2：Web 视频知识迁移到机器人](embodied-ai/gr2.md)
- [GR-3：少样本长时程 VLA](embodied-ai/gr3.md)
- [π0.5：开放世界 VLA](embodied-ai/pi05.md)
- [kai0：资源受限下的高可靠机器人操作](embodied-ai/kai0.md)
- [Qwen-RobotManip：先对齐再扩规模的机器人操作基础模型](embodied-ai/qwen-robotmanip.md)
- [DM0.5：开放世界具身基础模型](embodied-ai/dm05.md)
- [GenCeption：视频生成模型如何成为通用视觉学习器](embodied-ai/genception.md)
- [SpatialVLA：3D 空间表征接入 VLA](embodied-ai/spatialvla.md)
- [Video Prediction Policy：预测视觉表征训练策略](embodied-ai/video-prediction-policy.md)
- [Fast-WAM：训练时学世界，推理时不生成未来](embodied-ai/fast-wam.md)
- [Motus：统一 latent action 世界模型](embodied-ai/motus.md)
- [DreamZero：WAM 零样本策略](embodied-ai/dreamzero.md)
- [VO-DP：语义-几何自适应扩散策略](embodied-ai/vodp.md)
- [VGGT：前向恢复相机、深度、点云与轨迹](embodied-ai/vggt.md)
- [VGGT-Ω：几何重建怎样变成具身空间表征](embodied-ai/vggt-omega.md)
- [D4RT：动态场景的 4D 重建与跟踪](embodied-ai/d4rt.md)
- [Depth Anything：无标注数据释放单目深度](embodied-ai/depth-anything.md)
- [Depth Anything V2：单目深度数据配方](embodied-ai/depth-anything-v2.md)
- [Depth Anything 3：任意视角 3D 几何底座](embodied-ai/depth-anything-3.md)
- [MapAnything：统一前向 Metric 3D 重建骨干](embodied-ai/mapanything.md)

## 技术报告

- [DeepSeek-V3：671B MoE、MLA 与 FP8](technical-reports/deepseek-v3.md)
- [DeepSeek-R1：RL 激发推理能力](technical-reports/deepseek-r1.md)
- [DeepSeek-V4：百万上下文与 FP4 后训练](technical-reports/deepseek-v4-pro.md)
- [Qwen3：Thinking 模式、预训练与蒸馏](technical-reports/qwen3.md)
- [Qwen3.5-Omni：Thinker-Talker 与全模态 Agent](technical-reports/qwen35-omni.md)
- [Cosmos 3：面向 Physical AI 的全模态世界模型平台](technical-reports/cosmos3.md)
- [Kimi K2：MuonClip、万亿 MoE 与 Agent 数据](technical-reports/kimi-k2.md)
- [LongCat-Video：长视频生成、GRPO 与高效推理](technical-reports/longcat-video.md)
- [Gemini 2.5：动态 Thinking 与长上下文 Agent](technical-reports/gemini-2-5.md)
- [GPT-4o System Card：Omni 安全与多模态评测](technical-reports/gpt-4o-system-card.md)
- [InCoder-32B：工业代码模型与执行验证训练](technical-reports/incoder-32b.md)
- [Nemotron 3 Super：Mamba-MoE 与异步 RL](technical-reports/nemotron-3-super.md)
