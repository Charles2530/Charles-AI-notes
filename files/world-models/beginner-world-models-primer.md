# 从零理解世界模型

世界模型不是“能生成未来视频”的模型。更准确地说，它是智能体内部的一套预测器：给定当前状态和候选动作，预测未来状态、观测、奖励、风险或终止信号，帮助策略或规划器选择下一步。

## 核心问题

智能体做动作之前，最好能先问一句：如果我这样做，会发生什么？世界模型回答的就是这个问题。

```text
历史观测 + 当前状态 + 候选动作 -> 未来状态 / 观测 / 奖励 / 风险
```

如果输出只是一段好看的视频，但动作换了未来不变，或者未来不能被 planner、policy、risk checker 使用，那它还不是一个可靠的决策世界模型。

## 观测、状态和 latent

观测 observation 是传感器看到的东西：图像、深度、文本、触觉、机器人关节状态。真实状态 state 是足以预测未来的局面：物体位置、速度、接触、遮挡、任务阶段。latent 是模型学到的压缩表示，用来近似真实状态。

现实中我们通常只有部分观测。单帧图像看不到速度，看不到杯子背后的遮挡物，也不一定知道上一秒夹爪是不是已经碰到物体。因此世界模型需要记忆历史，把多帧和动作整合成 belief state。

## 动作条件是分水岭

普通视频预测可以问：“接下来可能发生什么？”动作条件世界模型要问：“如果我执行动作 \(a_t\)，接下来会发生什么？”

一个最小形式是：

\[
p_\theta(z_{t+1}, r_t, d_t \mid z_t, a_t)
\]

这里 \(z_t\) 是当前 latent state，\(a_t\) 是动作，\(z_{t+1}\) 是下一状态，\(r_t\) 是奖励或进展信号，\(d_t\) 是终止或失败信号。白话读法：模型用当前状态和动作，预测下一步世界怎么变，以及这步动作好不好、是否结束。

## Rollout：在模型里向未来展开

Rollout 是从当前状态出发，连续预测多步未来：

```text
z_t --a_t--> z_{t+1} --a_{t+1}--> z_{t+2} --...--> z_{t+H}
```

如果 rollout 在 latent space 里发生，它更便宜，适合规划和策略训练；如果 rollout 解码成视频，它更可视化，适合人类检查、数据标注和生成式模拟。二者不是谁替代谁，而是服务位置不同。

![State-space video world model](../assets/images/paper-deep-dives/world-models/towards-video-world-models/figure-5-state-space-video-world-model.png){ width="900" }

<small>图源：[Towards Video World Models](https://www.xunhuang.me/blogs/world_model.html)，Figure 5。原图表达 observation、latent state、action、rollout 和 policy/planner 的关系。本站用它说明：世界模型的关键接口是 state + action -> future，而不是单纯生成一段视频。</small>

看这张图时，先找三件事：state 从哪里来，action 在哪里进入，rollout 的输出被谁消费。如果没有消费方，模型最多是演示器；如果 action 不影响未来，模型就无法做反事实规划。

## Dreamer 这条线：在想象里训练策略

Dreamer 系列代表了 classic model-based RL 的一条清晰路线：先学一个 latent dynamics model，再在 latent imagined rollout 里训练 actor 和 critic。它不追求生成最高清的视频，而是追求 latent 足够服务控制。

![Dreamer overview 原论文图](../assets/images/paper-deep-dives/world-models/dreamer/figure-1-overview.png){ width="860" }

<small>图源：[Dreamer: Reinforcement Learning with Latent World Models](https://arxiv.org/abs/1912.01603)，Figure 1。原图表达从真实经验学习世界模型，再用 imagined trajectories 训练 actor 和 critic。本站用它说明：世界模型的价值在于减少真实环境试错，并把未来预测转成策略学习信号。</small>

## 生成式世界模型这条线：让未来可看、可交互

Genie、Genie 2、视频世界模型和机器人动作条件视频模型更强调外部可见的未来：生成环境、视频 rollout、交互式场景或动作后果。这类模型的优势是可视化强，容易做人类检查；难点是长时一致性、动作可控性、物理约束和低延迟。

世界模型不要被二分成“latent 派”和“视频派”。真实系统常常需要二者：latent 用于快速决策，视频用于检查、标注和模拟，几何或物理模块用于约束风险。

## 如何判断一篇世界模型论文

| 问题 | 为什么重要 |
| --- | --- |
| 预测对象是什么 | latent、视频、reward、risk、done 的用途不同。 |
| 动作是否真的进入模型 | 没有 action sensitivity 就无法做反事实决策。 |
| rollout 能滚多远 | 长 horizon 容易漂移，不能只看短 demo。 |
| 输出被谁使用 | planner、policy、risk checker、人类审查需要不同接口。 |
| 证据是什么 | open-loop loss、视频 demo、ablation、closed-loop success 不能混为一谈。 |

## 常见误解

| 误解 | 更准确的说法 |
| --- | --- |
| 能生成视频就是世界模型 | 还要看动作、状态、消费方和闭环收益。 |
| 世界模型必须高清 | 控制用 latent 可能比高清像素更重要。 |
| open-loop 预测好就能规划好 | 规划需要动作敏感、风险校准和 rollout 稳定。 |
| 世界模型可以替代真实数据 | imagined data 需要门禁，否则会把模型偏差回灌进策略。 |

## 外部精读

- [World Models](https://worldmodels.github.io/)：最经典的世界模型入门之一。
- [Dreamer](https://arxiv.org/abs/1912.01603)：理解 latent world model 和 imagined actor-critic。
- [PlaNet](https://arxiv.org/abs/1811.04551)：理解 pixel-to-latent planning。
- [DeepMind Genie 2](https://deepmind.google/blog/genie-2-a-large-scale-foundation-world-model/)：看生成式交互环境路线。
- [Meta V-JEPA 2](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/)：看预测表征和机器人/视频理解方向。
