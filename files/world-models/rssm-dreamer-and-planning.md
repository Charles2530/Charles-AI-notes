# RSSM、Dreamer 与规划

Dreamer 的核心问题很朴素：真实环境交互太贵，能不能先学一个世界模型，再在模型想象出的轨迹里训练策略？RSSM 是这条线里的关键表示结构，它把历史记忆和随机潜变量结合起来，让 agent 在部分可观测环境里保持 belief state。

## 核心问题

单帧图像不是完整状态。机器人只看一张图，不知道速度和接触历史；Atari 只看当前帧，也不一定知道物体运动方向。RSSM 的目标是把历史动作和观测压进一个 latent state，让模型既记住过去，又表达不确定性。

最小结构可以写成：

\[
h_t=f_\theta(h_{t-1}, z_{t-1}, a_{t-1})
\]

\[
z_t \sim q_\phi(z_t \mid h_t, x_t)
\]

\[
z_{t+1} \sim p_\theta(z_{t+1} \mid h_{t+1})
\]

\(h_t\) 是 deterministic hidden state，负责记历史；\(z_t\) 是 stochastic state，负责表达当前不确定性；posterior \(q_\phi\) 用真实观测校正 belief；prior \(p_\theta\) 用于没有未来观测时 rollout。

## RSSM 为什么比普通 RNN 更适合世界模型

普通 RNN 把所有信息塞进一个确定性 hidden state。RSSM 多了随机 latent，让模型可以表达“我不确定世界真实状态是哪一种”。这对部分可观测场景很重要：同一张观测图可能对应不同速度、遮挡和接触状态。

![PlaNet latent dynamics model designs](../assets/images/paper-deep-dives/world-models/planet/figure-2-latent-dynamics.png){ width="860" }

<small>图源：[Learning Latent Dynamics for Planning from Pixels](https://arxiv.org/abs/1811.04551)，Figure 2。原图比较 RNN、SSM 和 RSSM 三种 latent dynamics 结构。本站用它说明：RSSM 同时保留 deterministic history 和 stochastic state，适合把历史记忆、不确定性和 rollout 放在一起。</small>

读这张图时，重点看 RSSM 的两条状态线：一条是确定性记忆，一条是随机潜状态。真实观测进入 posterior 校正当前 state；规划或 imagined rollout 时，模型只能依赖 prior 往前预测。

## Dreamer 的三件事

Dreamer 把模型学习、价值学习和策略学习接成闭环：

```text
真实环境收集 episode
  -> 学 representation / dynamics / reward / continue
  -> 在 latent space imagined rollout
  -> 用 imagined trajectory 训练 actor 和 critic
  -> 回真实环境执行并继续收集数据
```

![Dreamer components 原论文图](../assets/images/paper-deep-dives/world-models/dreamer/figure-3-components.png){ width="860" }

<small>图源：[Dreamer](https://arxiv.org/abs/1912.01603)，Figure 3。原图表达 representation model、transition model、observation model、reward model、actor 和 value model 的关系。本站用它说明：Dreamer 不是一个单模型，而是一组围绕 latent rollout 训练控制策略的模块。</small>

第一件事是学习世界模型。模型要从真实 transition 中学会编码观测、预测下一 latent、重建观测、预测 reward 和 continue。它学到的不是完美真实世界，而是足够服务策略学习的压缩动态。

第二件事是在 latent space 里 rollout。因为 latent 远比像素便宜，模型可以想象多条未来轨迹，用它们训练 actor 和 critic。

第三件事是把策略带回真实环境。Dreamer 的收益必须通过真实环境回报或任务成功率证明，不能只看 reconstruction loss。

## Planning 和 policy learning 的区别

PlaNet 更偏 planning：给定当前 belief，用模型评估候选 action sequence，选回报最高的动作。Dreamer 更偏 policy learning：训练一个 actor，让它直接从 latent state 输出动作。

| 路线 | 每步决策时做什么 | 优势 | 代价 |
| --- | --- | --- | --- |
| PlaNet / MPC | 在线搜索候选动作序列 | 更直接利用模型预测 | 在线计算贵。 |
| Dreamer | 训练 actor，执行时直接出动作 | 执行快，适合长期学习 | actor 可能继承模型偏差。 |

两者都依赖世界模型，但消费方式不同。读论文时要先定位：模型输出是给 planner 在线搜索，还是给 actor/critic 离线或在线训练。

## Dreamer 证据怎么看

![Dreamer horizon ablation](../assets/images/paper-deep-dives/world-models/dreamer/figure-4-horizon.png){ width="760" }

<small>图源：[Dreamer](https://arxiv.org/abs/1912.01603)，Figure 4。原图表达 imagined horizon 对控制性能的影响。本站用它说明：rollout horizon 不是越长越好，太短看不到长期后果，太长会放大模型误差。</small>

这张图是理解世界模型评测的好例子。Horizon 太短，策略看不到长期收益；horizon 太长，latent dynamics 的误差会积累。世界模型的工程问题往往就在这个折中里：多想几步可能更聪明，也可能更自信地错。

![Dreamer video prediction](../assets/images/paper-deep-dives/world-models/dreamer/figure-5-video.png){ width="860" }

<small>图源：[Dreamer](https://arxiv.org/abs/1912.01603)，Figure 5。原图展示模型从 latent rollout 解码出的预测帧。本站用它说明：可视化有助于诊断模型是否捕捉到运动和结构，但控制能力仍要看 reward 和闭环表现。</small>

## 常见误解

| 误解 | 更准确的说法 |
| --- | --- |
| RSSM 是为了生成高清视频 | 它主要服务 belief state 和 latent rollout。 |
| imagined rollout 越长越好 | 长 horizon 会积累模型误差。 |
| reconstruction 好就能控制好 | 控制需要 reward、continue、value 和 action sensitivity。 |
| Dreamer 省掉真实环境 | 它减少真实试错，但仍需要真实交互校正模型和策略。 |

## 外部精读

- [PlaNet](https://arxiv.org/abs/1811.04551)：RSSM 和 latent planning 的前身。
- [Dreamer](https://arxiv.org/abs/1912.01603)：imagined actor-critic 的核心论文。
- [DreamerV2](https://arxiv.org/abs/2010.02193)：离散 latent 和 Atari 结果。
- [DreamerV3](https://arxiv.org/abs/2301.04104)：更通用的世界模型 agent。
- [World Models](https://worldmodels.github.io/)：更早的直觉入口。
