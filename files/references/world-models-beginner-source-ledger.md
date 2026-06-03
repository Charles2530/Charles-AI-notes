# 世界模型入门来源台账

这份台账服务于 [从零理解世界模型](../world-models/beginner-world-models-primer.md)。正文的目标不是把世界模型写成视频生成、强化学习或 JEPA 的模型名清单，而是先建立一个可复用接口：belief state + action -> future useful for decision。

## 使用方式

第一类来源是经典 latent control world model。World Models、PlaNet、Dreamer 帮助本站解释 observation、belief state、latent dynamics、imagined rollout、reward/value 和 policy/planner 如何接起来。正文使用这类来源时，重点放在“未来如何被决策模块消费”，不要把它们写成普通视频预测器。

第二类来源是生成式交互环境。Towards Video World Models 和 Genie 2 帮助本站解释视频世界模型、state-space world model、action-controllable playable world、counterfactual trajectory 和模拟器化方向。正文使用这类来源时要强调证据边界：demo 和 open-loop consistency 不等于真实闭环控制收益。

第三类来源是预测式表征和 JEPA。V-JEPA、V-JEPA 2 和相关专题用于说明 latent prediction 可以学习时空状态，但要进入规划仍需要 action-conditioned dynamics、goal interface、planner 和闭环验证。

第四类来源是中文讲法和高质量技术长文。中文博客、公众号和社区长文适合学习组织方式：先区分视频生成与决策世界模型，再按 state、action、rollout、evidence、closed-loop 的顺序讲。事实、图源、性能 claim 和论文版本仍以论文、官方博客、项目页和代码仓库为准。

## 写作边界

不要把“能生成视频”写成“能规划”。视频 demo 只能说明可视未来质量或交互原型，不能自动证明动作因果、reward/risk 接口和闭环收益。

不要把“强视频表征”写成“完整世界模型”。V-JEPA-style latent prediction 能学习时空状态，但如果没有动作条件、planner、goal/cost 和真实观测刷新，它更像世界模型的状态层候选。

不要把 Dreamer-style control world model 和 Genie-style playable world 混成一个层级。前者主要服务 policy learning 或 planning，后者主要服务可交互模拟、数据引擎和环境生成。二者都可能成为世界模型，但证据和风险不同。

不要只看 open-loop prediction loss。世界模型证据要从 reconstruction、prediction、action counterfactual、reward/risk、planning ablation 一路追到 closed-loop success。

## 图片使用

本页没有使用 image2，也没有生成新图片；正文只复用了仓库里已经存在的论文图或技术长文图，并在图注中保留图源和读图方法。

| 正文图片 | 本地路径 | 原始来源 | 本站用途 |
| --- | --- | --- | --- |
| State-space video world model | `../assets/images/paper-deep-dives/world-models/towards-video-world-models/figure-5-state-space-video-world-model.png` | [Towards Video World Models](https://www.xunhuang.me/blogs/world_model.html) Figure 5 | 说明 observation、state、action、rollout 和 planner/policy 的接口。 |

## 来源

| 来源 | 类型 | 本站使用方式 | 关联问题 |
| --- | --- | --- | --- |
| [World Models](https://worldmodels.github.io/) | 项目页 / 论文入口 | 支撑 VAE、MDN-RNN、controller 和“在模型里练习”的早期直觉。 | 经典入口 |
| [PlaNet](https://arxiv.org/abs/1811.04551) | 论文 | 支撑 latent dynamics、planning from pixels、CEM planning 和 belief state。 | latent planning |
| [PlaNet project page](https://planetrl.github.io/) | 项目页 | 支撑 planning from pixels 的可视化入口。 | 经典入口 |
| [Dreamer](https://arxiv.org/abs/1912.01603) | 论文 / 图源 | 支撑 latent imagination、actor/value、reward model 和 imagined rollout。 | latent control |
| [Dreamer project page](https://danijar.com/project/dreamer/) | 项目页 | 支撑 Dreamer 相比 PlaNet 的接口变化。 | 经典入口 |
| [Genie 2 DeepMind blog](https://deepmind.google/blog/genie-2-a-large-scale-foundation-world-model/) | 官方博客 | 支撑 action-controllable playable world、counterfactual trajectories 和 world simulator 方向。 | 生成式环境 |
| [Towards Video World Models](https://www.xunhuang.me/blogs/world_model.html) | 技术长文 / 图源 | 使用 state-space world model 图解释视频世界模型的接口。 | 视频 WM |
| [V-JEPA](https://arxiv.org/abs/2404.08471) | 论文 | 支撑 latent video prediction 和不重建像素的表征学习路线。 | JEPA |
| [V-JEPA 2 research page](https://ai.meta.com/research/publications/v-jepa-2-self-supervised-video-models-enable-understanding-prediction-and-planning/) | 官方研究页 / 图源 | 支撑 action-free 预训练到 action-conditioned planning 的两阶段讲法。 | JEPA planning |
| [V-JEPA 2 paper](https://arxiv.org/abs/2506.09985) | 论文 | 支撑 V-JEPA 2-AC、robot planning 和 closed-loop 证据边界。 | planning |

## 后续写作提醒

写世界模型入门页时，每段最好能回答四件事：状态从哪里来，动作如何进入，rollout 被谁消费，证据停在哪一层。只要这四件事没讲清楚，再多模型名和 demo 都容易让读者把视频生成、表征学习和决策世界模型混在一起。
