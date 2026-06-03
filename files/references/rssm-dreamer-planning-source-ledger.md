# RSSM、Dreamer 与规划来源台账

这份台账服务于 [RSSM、Dreamer 与规划](../world-models/rssm-dreamer-and-planning.md)。正文不把 Dreamer 系列写成版本号列表，而是围绕一条机制链讲：状态为什么需要历史，posterior / prior 为什么分开，world model loss 怎样塑造可控制 latent，PlaNet 和 Dreamer 怎样消费 imagined future，以及真实机器人和视频世界模型为什么不能直接混为一谈。

## 使用方式

第一类来源是早期概念和像素规划。World Models 项目页用于给读者建立“内部模型里练习”的直觉；PlaNet 论文和项目页用于解释 RSSM、latent dynamics、CEM planning 和 planning from pixels。正文使用 PlaNet Figure 2 说明 RNN、SSM、RSSM 的结构差异。

第二类来源是 Dreamer 系列主线。Dreamer 论文、项目页和 Google Research 官方博客用于解释 latent imagination、actor/value、\(\lambda\)-return、value bootstrap 和 PlaNet 到 Dreamer 的接口变化。正文使用 Dreamer Figure 3 和 Figure 4 说明：Dreamer 不是视频预测器，而是把 world model 变成 actor-critic 的训练环境。

第三类来源是 DreamerV2 / DreamerV3 的稳定化设计。DreamerV2 支撑 discrete latent、KL balancing 和 Atari 证据；DreamerV3 支撑 symlog、symexp two-hot、free bits、fixed hyperparameters、Minecraft diamond 和 scaling。正文使用 DreamerV3 Figure 3(a) 说明跨域稳健化技巧仍然服务同一个目标：让 latent state 能被 policy learning 消费。

第四类来源是真实机器人和边界。DayDreamer 用于说明 Dreamer 能进入真实硬件在线学习，但仍受任务、动作接口、安全和硬件约束影响；MuZero 用于对比不重建图像、只学 reward/value/policy 相关 latent dynamics 的另一条路线。

中文和博客材料主要学习讲法：先把“普通视频预测”和“用于控制的 world model”区分清楚，再沿着 state、dynamics、loss、planner / actor、evidence 逐层解释。事实判断以论文、项目页、官方博客和可复查材料为准；链接不稳定或事实链不够一手的中文资料不进入来源表。

## 写作边界

正文不把 RSSM 说成一个普通视觉 encoder。RSSM state 必须包含历史、动作和不确定性，并且能被 prior rollout。

正文不把图像重建说成最终目标。图像重建是塑造 latent 的训练信号，真正的控制证据来自 reward、continuation、value、real environment return 和 closed-loop success。

正文不把 PlaNet 和 Dreamer 混成同一种决策接口。PlaNet 每步 CEM 在线搜索，Dreamer 在 imagined rollout 中训练 actor，再用 actor 直接行动。

正文不把 DreamerV3 的 benchmark 成绩外推成通用机器人能力。DreamerV3 证明固定配置跨多个 RL domain 的稳定性；真实机器人还需要动作接口、安全、恢复、数据回流和硬件约束。

正文不把视频世界模型和 Dreamer-style control world model 混用。Open-loop 视频预测能说明可视未来质量，不能直接证明 policy learning；control return 能证明任务回报，也不能直接证明可视模拟器质量。

## 图片使用

本页没有使用 image2，也没有生成新图片；正文只复用了仓库里已经存在的论文图，并在图注中保留图源和读图方法。

| 正文图片 | 本地路径 | 原始来源 | 本站用途 |
| --- | --- | --- | --- |
| PlaNet latent dynamics | `../assets/images/paper-deep-dives/world-models/planet/figure-2-latent-dynamics.png` | [PlaNet](https://arxiv.org/abs/1811.04551) Figure 2 | 说明 RNN、SSM、RSSM 的状态结构差异。 |
| Dreamer horizon ablation | `../assets/images/paper-deep-dives/world-models/dreamer/figure-4-horizon.png` | [Dreamer](https://arxiv.org/abs/1912.01603) Figure 4 | 说明 imagined horizon 是收益和模型误差的折中。 |
| DreamerV3 world model learning | `../assets/images/paper-deep-dives/world-models/dreamerv3/x2.png` | [DreamerV3](https://arxiv.org/abs/2301.04104) Figure 3(a) | 说明 DreamerV3 的稳定化技巧仍服务可决策 latent state。 |

## 来源

| 来源 | 类型 | 本站使用方式 | 关联主题 |
| --- | --- | --- | --- |
| [World Models](https://worldmodels.github.io/) | 项目页 / 早期博客式论文 | 建立 latent dream、VAE、MDN-RNN、controller 的直觉背景。 | 概念入口 |
| [PlaNet project page](https://planetrl.github.io/) | 项目页 | 支撑 planning from pixels、latent planning 和 CEM 的高层解释。 | PlaNet |
| [Learning Latent Dynamics for Planning from Pixels](https://arxiv.org/abs/1811.04551) | 论文 / 图源 | 使用 Figure 2 解释 RNN、SSM、RSSM；支撑 RSSM 和 latent overshooting。 | RSSM |
| [Dreamer project page](https://danijar.com/project/dreamer/) | 项目页 | 支撑 latent imagination、actor/value 和 Dreamer 相比 PlaNet 的变化。 | Dreamer |
| [Dream to Control](https://arxiv.org/abs/1912.01603) | 论文 / 图源 | 使用 Figure 3 和 Figure 4 解释组件图和 horizon tradeoff。 | Dreamer |
| [Google Research: Introducing Dreamer](https://research.google/blog/introducing-dreamer-scalable-reinforcement-learning-using-world-models/) | 官方博客 | 借鉴它对 world model learning、behavior learning、PlaNet vs Dreamer 的讲法。 | 官方解释 |
| [DreamerV2](https://arxiv.org/abs/2010.02193) | 论文 | 支撑 discrete latent、KL balancing、Atari 证据和 model-based vs model-free 对比。 | DreamerV2 |
| [DreamerV3 project page](https://danijar.com/project/dreamerv3/) | 项目页 | 支撑 fixed hyperparameters、Minecraft diamond、scaling 和跨域结果的解释。 | DreamerV3 |
| [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104) | 论文 / 图源 | 使用 Figure 3(a) 解释 world model learning；支撑 symlog、two-hot、free bits。 | DreamerV3 |
| [DayDreamer](https://danijar.com/project/daydreamer/) | 项目页 | 支撑真实机器人在线学习的证据和边界。 | 机器人 |
| [DayDreamer paper](https://arxiv.org/abs/2206.14176) | 论文 | 支撑 4 类真实机器人、在线学习、无仿真器的实验描述。 | 机器人 |
| [MuZero](https://arxiv.org/abs/1911.08265) | 论文 | 对比只学习 reward/value/policy 相关 dynamics、不重建图像的路线。 | 边界对比 |
| [VITALab DreamerV3 note](https://vitalab.github.io/article/2023/01/19/DreamerV3.html) | 技术博客 | 借鉴 RSSM、symlog、free bits、two-hot 的二次讲解方式。 | 博客讲法 |
