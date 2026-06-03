# 预训练、微调与对齐来源台账

这份台账服务于 [预训练、微调与对齐](../training/pretraining-finetuning-alignment.md)。正文的写作目标是把三类训练信号拆开：预训练塑造能力边界，SFT 塑造行为格式，偏好/RL 塑造选择倾向；现代后训练则把可验证奖励、蒸馏、工具、安全和线上回流组织成流水线。

## 使用方式

第一类来源是预训练和能力边界。GPT-3、FLAN、LIMA 等来源帮助本站解释：next-token prediction 或 instruction tuning 的目标看似简单，但真正影响能力的是数据分布、规模、示范质量、任务覆盖和评测接口。正文不要把预训练 loss 直接等价为产品能力。

第二类来源是 RLHF 和偏好优化。InstructGPT、PPO、DPO、HH-RLHF、Constitutional AI、IPO、ORPO 等来源帮助本站解释 demonstration、reward model、pairwise preference、reference policy、KL/clip 约束和偏好数据风险。正文重点不在罗列算法，而在说明偏好信号到底改变了模型哪种选择倾向。

第三类来源是现代后训练流水线。DeepSeek-R1、Qwen3 和 SmolLM3 这类报告说明后训练已经不是单个 SFT 或 RL 算法，而是 cold start、reasoning RL/RLVR、thinking mode、general RL、distillation、工具和安全评测的组合。正文使用这些来源时要避免把特定团队 recipe 写成普适规律。

第四类来源是中文讲法与长文组织。高质量博客、技术长文和微信公众号适合学习中文解释节奏：先给读者一个问题归因框架，再解释公式和训练流程，最后给评测与排查清单。事实、公式、版本语义和性能 claim 仍以论文、官方报告、官方博客或代码仓库为准；微信公众号链接稳定性弱，不作为本站长期链接校验对象。

## 写作边界

不要把预训练、SFT 和对齐写成“模型越来越聪明”的三段神话。三者改变的是监督信号和分布压力：预训练学习数据分布，SFT 学示范行为，偏好/RL 学选择倾向。

不要把对齐写成万能修补层。偏好训练更擅长塑形已有能力、修风格和边界；如果底座缺少数学、代码、长上下文或领域知识，最后一层对齐很难凭空补齐。

不要只看 reward 或胜率平均值。对齐证据至少要按任务、语言、长度、风险等级、标注者、线上场景和长尾失败桶拆开看，并同时监控 KL、回答长度、事实性、安全拒答、能力保持和用户真实成功率。

不要把 DPO、IPO、ORPO 等写成免维护替代品。它们可以简化优化流程，但偏好数据质量、candidate 分布、reference model、标注指南和回归评测仍然决定结果上限。

## 图片使用

本页没有使用 image2，也没有生成新图片；正文只复用了仓库里已经存在的论文图和技术报告图，并在图注中保留图源和读图方法。

| 正文图片 | 本地路径 | 原始来源 | 本站用途 |
| --- | --- | --- | --- |
| InstructGPT RLHF pipeline | `../assets/images/paper-figures/training/instructgpt-rlhf-pipeline.png` | [InstructGPT](https://arxiv.org/abs/2203.02155) Figure 2 | 说明 RLHF 是 SFT、reward modeling、PPO 三步流程。 |
| Qwen3 post-training pipeline | `../assets/images/paper-deep-dives/technical-reports/qwen3/figure-1-posttrain-pipeline.png` | [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) Figure 1 | 说明现代后训练是一条包含 cold start、RL、fusion、distillation 的生产线。 |

## 来源

| 来源 | 类型 | 本站使用方式 | 关联问题 |
| --- | --- | --- | --- |
| [GPT-3](https://arxiv.org/abs/2005.14165) | 论文 | 支撑大规模 next-token 预训练如何诱导广泛语言和少样本能力。 | 预训练能力边界 |
| [InstructGPT](https://arxiv.org/abs/2203.02155) | 论文 / 图源 | 支撑 SFT、reward model、PPO 三步 RLHF 流程和人类偏好评测。 | RLHF |
| [PPO](https://arxiv.org/abs/1707.06347) | 论文 / 图源 | 支撑 clipped surrogate、old policy 和小步更新解释。 | RL 优化 |
| [HH-RLHF](https://arxiv.org/abs/2204.05862) | 论文 | 支撑 helpful/harmless 数据和偏好建模边界。 | 偏好数据 |
| [FLAN](https://arxiv.org/abs/2210.11416) | 论文 | 支撑 instruction tuning 的任务覆盖和泛化讨论。 | SFT / instruction tuning |
| [Constitutional AI](https://arxiv.org/abs/2212.08073) | 论文 | 支撑 AI feedback、规则化偏好和安全对齐讨论。 | RLAIF / safety |
| [Self-Instruct](https://arxiv.org/abs/2212.10560) | 论文 | 支撑用强模型生成指令数据时要检查质量和分布。 | 数据合成 |
| [LIMA](https://arxiv.org/abs/2305.11206) | 论文 | 支撑少量高质量 SFT 数据也可能显著塑造行为格式。 | SFT 数据质量 |
| [DPO](https://arxiv.org/abs/2305.18290) | 论文 | 支撑 closed-form preference objective 和 reference model 解释。 | 直接偏好优化 |
| [Llama 2](https://arxiv.org/abs/2307.09288) | 技术报告 | 支撑 SFT、RLHF、安全数据、red teaming 与能力保持共同看。 | 后训练与安全 |
| [IPO](https://arxiv.org/abs/2402.03300) | 论文 | 支撑偏好目标设计会影响过优化与校准。 | 偏好优化 |
| [ORPO](https://arxiv.org/abs/2403.07691) | 论文 | 支撑不显式 reference 的偏好优化路线可作为横向对照。 | 偏好优化 |
| [DeepSeek-R1](https://arxiv.org/abs/2501.12948) | 技术报告 | 支撑 RLVR、GRPO、可验证奖励和 reasoning RL 讨论。 | 现代后训练 |
| [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) | 技术报告 / 图源 | 使用 post-training pipeline 图解释 cold start、reasoning RL、fusion、general RL 和蒸馏。 | 现代后训练 |
| [Hugging Face RLHF blog](https://huggingface.co/blog/rlhf) | 官方技术博客 | 学习把 RLHF 词表、流程和直觉讲清楚。 | 讲法参考 |
| [SmolLM3 blog](https://huggingface.co/blog/smollm3) | 官方训练复盘 | 学习多阶段 pretraining、reasoning、SFT/alignment 如何组织成训练 recipe。 | 现代训练复盘 |
| [Datawhale LLM Universe](https://datawhalechina.github.io/llm-universe/) | 中文教程 | 学习中文入门文章如何把模型训练、提示、RAG 和评测串成学习路径。 | 中文讲法 |

## 后续写作提醒

写预训练、SFT 和对齐时，段落最好围绕“监督信号从哪里来”展开。预训练段落问数据分布和目标，SFT 段落问示范质量和任务覆盖，偏好段落问 chosen/rejected、reward、reference 和评测分桶，现代后训练段落问每个阶段的验收信号和回滚条件。

如果新增模型训练报告，先判断它是在讲底座预训练、instruction tuning、preference optimization、RLVR、distillation、工具训练还是安全评测。不要把所有后训练都写成“做了 RLHF”，也不要把一个团队的 pipeline 外推为所有模型都必须采用的固定流程。
