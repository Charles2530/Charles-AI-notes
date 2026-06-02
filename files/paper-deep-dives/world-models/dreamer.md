# Dreamer：为什么可以在想象轨迹里训练控制策略

Checked date: 2026-06-02  
Repro status: 论文级理解；本站未复现实验，只引用原论文图和公开结果。

Dreamer 的问题很清楚：真实环境交互贵，能不能先学一个 latent world model，再在这个模型想象出来的轨迹里训练 actor 和 value？它的贡献不是“生成未来图像很好看”，而是把 representation learning、latent dynamics、reward prediction 和 actor-critic 接成一条可训练控制链。

## 论文解决的问题

Model-free RL 每提升一点能力都要和真实环境大量交互。Model-based RL 试图先学习环境模型，再用模型做规划或训练策略。Dreamer 选择的是后一种：在 latent space 中 rollout，并用 imagined trajectories 训练 actor 和 critic。

核心链路是：

```text
真实 episode -> 学 latent world model -> imagined rollout -> actor/value learning -> 回真实环境执行
```

## 方法结构

![Dreamer overview 原论文图](../../assets/images/paper-deep-dives/world-models/dreamer/figure-1-overview.png){ width="860" }

<small>图源：[Dreamer: Reinforcement Learning with Latent World Models](https://arxiv.org/abs/1912.01603)，Figure 1。原图表达从真实环境数据学习世界模型，再用 imagined trajectories 训练 actor 和 critic。本站用它说明：Dreamer 的关键不是像素重建，而是让 latent rollout 变成策略学习信号。</small>

Dreamer 的 world model 包含五个部分：representation model 把观测变成 latent state；transition model 预测下一 latent；observation model 用于重建；reward model 预测奖励；continue model 预测 episode 是否继续。Actor 和 value model 不直接在像素上训练，而是在 latent imagination 中学习。

![Dreamer components 原论文图](../../assets/images/paper-deep-dives/world-models/dreamer/figure-3-components.png){ width="860" }

<small>图源：[Dreamer](https://arxiv.org/abs/1912.01603)，Figure 3。原图表达 representation、transition、observation、reward、actor、value 的连接方式。本站用它定位每个模块的消费关系：world model 产生 imagined latent sequence，actor/value 消费这些 sequence。</small>

## RSSM 在这里起什么作用

Dreamer 使用 RSSM。它把 deterministic hidden state 和 stochastic latent state 结合起来：hidden state 记历史，stochastic state 表达不确定性。真实观测到来时用 posterior 校正 state；想象未来时用 prior rollout。

这正好匹配部分可观测控制任务。单帧图像不包含完整速度、接触和遮挡信息；RSSM 用历史动作和观测补足 belief state。

## Actor-critic 为什么能在 latent 里学

Dreamer 在模型中生成 latent trajectory，然后根据 reward model 和 value model 估计 return。Actor 通过这些 imagined returns 更新参数。直观上，它不是等真实环境告诉它每条尝试的结果，而是在模型里快速试很多短未来。

这条路径省真实交互，但引入模型偏差。world model 错得越自信，actor 越可能学到利用模型漏洞的行为。所以 Dreamer 的证据必须看真实环境 performance，而不能只看 reconstruction 或 video prediction。

## 关键实验怎么读

![Dreamer horizon ablation](../../assets/images/paper-deep-dives/world-models/dreamer/figure-4-horizon.png){ width="760" }

<small>图源：[Dreamer](https://arxiv.org/abs/1912.01603)，Figure 4。原图展示 imagined horizon 对控制性能的影响。本站用它解释：world model rollout horizon 是一个偏差-收益折中，太短看不到长期后果，太长会放大模型误差。</small>

Horizon 图是 Dreamer 里最值得带走的证据之一。它提醒我们，世界模型不是滚得越远越好；真实系统里也一样，长 horizon planning 会同时增加信息量和模型偏差。

![Dreamer video prediction](../../assets/images/paper-deep-dives/world-models/dreamer/figure-5-video.png){ width="860" }

<small>图源：[Dreamer](https://arxiv.org/abs/1912.01603)，Figure 5。原图展示 latent rollout 解码后的预测视频。本站用它作为诊断图：可视化能帮助发现模型是否捕捉运动和结构，但不能替代闭环控制指标。</small>

视频预测图适合检查模型有没有学到粗略动态，但它不是最终指标。Dreamer 的主张要靠真实控制回报支撑。

## 可复用经验

| 经验 | 迁移到工程时怎么用 |
| --- | --- |
| 先学紧凑 latent dynamics | 不必所有系统都追求高清未来视频。 |
| rollout horizon 要调 | 长期规划和模型误差需要折中。 |
| world model 输出要被 actor/value 消费 | 只给人看 demo 不足以证明决策价值。 |
| 真实环境指标是最终证据 | reconstruction loss 不能替代任务成功率。 |

## 边界

Dreamer 的实验环境和真实机器人、自动驾驶、多模态 agent 相比仍然更规整。把 Dreamer 思路迁移到真实世界时，还要处理多相机、动作延迟、不可逆失败、安全约束、稀疏奖励和数据分布漂移。

## 外部精读

- [Dreamer paper](https://arxiv.org/abs/1912.01603)
- [PlaNet paper](https://arxiv.org/abs/1811.04551)
- [DreamerV2](https://arxiv.org/abs/2010.02193)
- [DreamerV3](https://arxiv.org/abs/2301.04104)
- [World Models](https://worldmodels.github.io/)
