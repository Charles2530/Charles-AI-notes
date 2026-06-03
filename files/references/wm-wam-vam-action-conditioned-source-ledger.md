# WM / WAM / VAM 动作条件建模来源台账

这份台账服务于 [WM / WAM / VAM：动作到底怎样进入世界模型](../world-models/wm-wam-vam-and-action-conditioned-modeling.md)。正文不把 WM、WAM、VAM 写成缩写词表，而是围绕一个接口问题讲清楚：动作是条件、输出、联合变量，还是视频表征的下游消费对象；这个位置决定模型能否做反事实、规划、闭环刷新和失败恢复。

## 使用方式

第一类来源是 classical world model 和 model-based RL。World Models、PlaNet、Dreamer、DreamerV2、DreamerV3 和 Google/DeepMind 官方博客用于解释 latent dynamics、imagined rollout、reward/value/continuation、actor/critic 和 planner。正文使用这类来源时，重点放在“动作作为条件，未来作为评估对象”，不要把 Dreamer 写成普通视频预测器。

第二类来源是 action-conditioned foundation world model。Genie、Genie 2、V-JEPA 2-AC 等材料用于解释从 action-free 视频表征到 action-conditioned planning 的过渡。正文要强调：大规模视频预训练可以提供动态先验，但控制仍需要动作接口、goal/cost、真实观测刷新和闭环证据。

第三类来源是 world action model 和 video-action joint modeling。DreamZero / World Action Models、Video Prediction Policy、GR-2/GR-3 等论文和项目页用于解释动作与未来视觉共同生成的路线。正文要把 open-loop video、offline action prediction 和 real robot closed-loop 分开，不把漂亮未来视频当成动作成功证据。

第四类来源是 VLA/action interface。RT-1、RT-2、Open X-Embodiment、ACT、Diffusion Policy、OpenVLA 和动作接口来源台账用于提醒：policy 输出动作不等于理解动作后果。VLA 可以直接输出动作，WM/WAM/VAM 则必须额外说明动作后果、风险、延迟和恢复。

第五类来源是中文讲法与技术长文。Datawhale、机器之心、智源社区、量子位、PaperWeekly 和微信公众号长文适合学习如何把世界模型、VLA、机器人策略和视频生成写成“接口 -> 机制 -> 证据 -> 边界”的结构。事实、图源、性能 claim 和论文版本仍以论文、官方博客、项目页和代码仓库为准。

## 写作边界

不要把“能生成未来视频”写成“能规划”。open-loop video prediction 只能证明可视未来质量或表征能力；规划还需要动作进入条件、候选动作能改变未来、cost/reward/risk 能比较候选、闭环刷新能纠偏。

不要把“能输出动作”写成“有世界模型”。一个 VLA policy 可能直接从观测和语言输出动作 token，但如果它不预测或评估动作后果，本站不把它当成可规划 world model。

不要把 WAM 写成视频模型旁边加动作头。WAM 的核心 claim 是动作和未来世界联合建模，并且动作可以在真实观测刷新后继续有效。正文应要求同时看视频、动作、真实执行、失败回放和跨场景泛化。

不要用 FVD、CLIPScore、视频清晰度替代机器人指标。VAM 的最终证据应包含 success rate、closed-loop rollout、跨场景泛化、失败恢复、动作平滑度和推理延迟；视频质量只是中间诊断。

不要把 action-free foundation model 直接称为 robot planner。V-JEPA、Genie 这类模型可以提供强视频/动态先验，但规划真实动作必须补上 action-conditioned dynamics、goal cost、controller、safety filter 和环境反馈。

## 图片使用

本页没有使用 image2，也没有生成新图片；正文只复用了仓库里已经存在的论文图，并在图注中保留图源和读图方法。Dreamer Figure 3 与 V-JEPA 2 Figure 7 原图自带长 caption，主题页正文没有复用，避免打断阅读。

| 正文图片 | 本地路径 | 原始来源 | 本站用途 |
| --- | --- | --- | --- |
| DreamZero joint video-action architecture | `../assets/images/paper-deep-dives/world-models/dreamzero/figure-4-architecture.png` | [World Action Models are Zero-shot Policies](https://arxiv.org/abs/2602.15922) Figure 4 | 说明训练侧联合去噪未来视频和动作，推理侧用真实观测刷新闭环。 |
| Video Prediction Policy overview | `../assets/images/paper-deep-dives/embodied-ai/vpp/figure-1-overview.png` | [Video Prediction Policy](https://arxiv.org/abs/2412.14803) Figure 1 | 说明视频预测中间表征如何作为机器人 policy 的动态状态。 |

## 来源

| 来源 | 类型 | 本站使用方式 | 关联问题 |
| --- | --- | --- | --- |
| [World Models](https://worldmodels.github.io/) | 项目页 / 博客式论文 | 建立 latent model、dream environment 和 controller 的早期直觉。 | WM 直觉 |
| [PlaNet](https://planetrl.github.io/) | 项目页 / 论文 | 支撑 planning from pixels、RSSM 和 CEM planning 的接口讲法。 | action-conditioned planning |
| [Dream to Control](https://arxiv.org/abs/1912.01603) | 论文 | 支撑 world model、actor、critic 和 imagined rollout 的关系。 | Dreamer |
| [DreamerV3](https://arxiv.org/abs/2301.04104) | 论文 | 支撑 fixed hyperparameters、symlog/two-hot/free bits 和跨域 model-based RL。 | WM 基线 |
| [Google Research: Introducing Dreamer](https://research.google/blog/introducing-dreamer-scalable-reinforcement-learning-using-world-models/) | 官方博客 | 借鉴“learning the world model / learning behaviors from imagined futures”的解释顺序。 | 博客讲法 |
| [Genie](https://sites.google.com/view/genie-2024/) | 项目页 / 论文 | 支撑从视频生成可交互环境、latent action model 和 action-controllable generation 的讨论。 | foundation world model |
| [Genie 2 DeepMind blog](https://deepmind.google/blog/genie-2-a-large-scale-foundation-world-model/) | 官方博客 | 支撑 action-controllable playable world、foundation world model 和交互边界。 | action-controllable WM |
| [Genie 3](https://deepmind.google/models/genie/) | 官方模型页 | 记录 2026 年 Google DeepMind 对可交互世界模型的最新入口；只作能力形态和官方定位参考。 | 交互世界模型 |
| [V-JEPA 2 research page](https://ai.meta.com/research/publications/v-jepa-2-self-supervised-video-models-enable-understanding-prediction-and-planning/) | 官方研究页 / 图源 | 支撑 action-free 预训练到 V-JEPA 2-AC latent planning 的两阶段讲法。 | latent planning |
| [V-JEPA 2 paper](https://arxiv.org/abs/2506.09985) | 论文 | 支撑 candidate action sequence、goal representation 和 model predictive control 的细节。 | V-JEPA 2-AC |
| [DreamZero / World Action Models](https://dreamzero0.github.io/) | 项目页 / 图源 | 支撑未来视频和未来动作 joint generation、closed-loop control、7Hz real-time claim 和 real robot evidence。 | WAM |
| [World Action Models are Zero-shot Policies](https://arxiv.org/abs/2602.15922) | 论文 | 支撑 WAM 作为 zero-shot policy 和 video-action joint modeling 的核心 claim。 | WAM |
| [Video Prediction Policy](https://video-prediction-policy.github.io/) | 项目页 / 图源 | 支撑 predictive visual representation 如何接机器人动作。 | VAM |
| [Video Prediction Policy paper](https://arxiv.org/abs/2412.14803) | 论文 | 支撑 text-conditioned video prediction、representation aggregation 和 robot policy 证据。 | VAM |
| [RT-2 DeepMind blog](https://deepmind.google/blog/rt-2-new-model-translates-vision-and-language-into-action/) | 官方博客 | 对比 VLA 直接输出动作与世界模型预测动作后果的差异。 | VLA 对照 |
| [Open X-Embodiment / RT-X](https://robotics-transformer-x.github.io/) | 项目页 / 论文 | 支撑跨机器人动作口径、动作空间和数据混合的接口问题。 | action data |
| [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) | 项目页 / 论文 | 补充连续动作分布、receding horizon 和多峰动作建模。 | policy 对照 |
| [全站外部阅读台账](external-reading-ledger.md) | 站内台账 | 记录高质量博客、中文公众号和技术媒体的讲法学习边界。 | 写作方法 |
| [机器之心](https://www.jiqizhixin.com/) | 中文技术媒体 / 公众号生态 | 学习“背景 -> 方法 -> 实验 -> 局限”的中文论文讲解节奏，不作为强事实唯一来源。 | 中文讲法 |
| [量子位](https://www.qbitai.com/) | 中文技术媒体 / 公众号生态 | 学习从产业问题和路线对比切入的写法，事实回原论文或官方材料核对。 | 中文讲法 |
| [智源社区](https://hub.baai.ac.cn/) | 中文社区 / 编译转载 | 学习技术长文和编译材料如何组织中文背景，转载内容追到原文。 | 中文讲法 |
| [PaperWeekly](https://zhuanlan.zhihu.com/paperweekly) | 中文论文解读 / 公众号生态 | 学习论文速读如何把贡献、方法图和实验表拆成读者能跟上的顺序。 | 中文讲法 |

## 后续写作提醒

写 WM/WAM/VAM 时，正文最好围绕“谁消费未来”组织证据。Planner 消费未来，就看 candidate ranking 和 closed-loop MPC；policy 消费未来，就看 action chunk 和真实观测刷新；risk checker 消费未来，就看 near-miss recall 和 calibration；data engine 消费未来，就看 failure replay 和 hard-negative yield。不要让缩写替代证据。

如果引用中文技术长文，优先吸收结构：先给接口问题，再讲机制，再给实验边界。不要直接搬用二手性能数字，也不要让公众号式标题替代可核对证据。
