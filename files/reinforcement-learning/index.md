# 强化学习阅读入口：从回报信号到可控训练

强化学习在本站里有两条线：一条是传统 RL 的状态、动作、价值和策略梯度；另一条是大模型、世界模型和 VLA 里的偏好优化、在线反馈、环境交互和数据闭环。读这一章时，先不要急着记算法名，先问清楚奖励来自哪里，策略能观察到什么，采样成本由谁承担。

高质量中文长文和公众号通常会先用具体任务讲清“为什么监督学习不够”，再引入 MDP、value、policy 和 advantage。这里也采用这种讲法，但每个结论会尽量落回论文或开源训练流程。

## 建立 RL 语言

| 你想弄清的问题 | 建议先读 |
| --- | --- |
| state、action、reward、return 为什么要被放进 MDP | [MDP、价值函数与 Bellman](mdp-value-bellman.md) |
| value、Q、advantage 分别在估什么 | [MDP、价值函数与 Bellman](mdp-value-bellman.md) |
| policy gradient、actor-critic、PPO、GRPO 怎样从采样里更新策略 | [Policy Gradient、Actor-Critic、PPO 与 GRPO](policy-gradient-actor-critic-ppo-grpo.md) |

## 进入大模型和世界模型

| 场景 | 建议先读 |
| --- | --- |
| 在 latent imagination 里训练策略 | [世界模型中的强化学习](rl-for-world-models.md) |
| 用 verl 理解 RLHF/GRPO 的训练数据流 | [verl 训练流程](verl-code-flow.md) |
| 把 RL 和训练章节接起来 | [偏好数据与对齐陷阱](../training/preference-data-and-alignment-pitfalls.md) |
| 把 RL 和世界模型接起来 | [RSSM、Dreamer 与规划](../world-models/rssm-dreamer-and-planning.md) |

## 阅读时的判断题

每个 RL 方法都可以用四个问题检查：奖励是不是稳定且可被策略利用；探索成本是不是可承受；value 或 reward model 的误差会不会被策略放大；离线数据、在线 rollout 和评测环境之间有没有分布偏移。能回答这些问题，比记住一个算法缩写更可靠。
