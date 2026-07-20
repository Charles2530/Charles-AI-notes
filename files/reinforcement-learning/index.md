# 强化学习：从交互闭环到策略优化

强化学习（Reinforcement Learning, RL）是一类序列决策方法：**agent 通过与环境交互学习 policy，使期望累计回报最大化**。动作会改变未来的数据分布，训练信号来自一段时间后的结果，而不是固定输入上的逐步标签。

## 什么是强化学习

一个强化学习任务至少要说明：agent 能观察什么、能采取什么 action、环境怎样响应、reward 如何衡量结果，以及一段交互何时结束。策略 \(\pi\) 把 observation 或 state 映射为动作分布；训练目标不能逐步比对标准答案，需要最大化轨迹的期望累计回报：

\[
J(\pi)=\mathbb{E}_{\tau\sim\pi}\left[\sum_{t=0}^{T-1}\gamma^t r_t\right]
\]

这里 \(\tau\) 是按策略采集的 trajectory，\(r_t\) 是单步 reward，\(\gamma\) 决定多远的未来仍影响当前选择。期望同时覆盖策略采样和环境随机性，所以一次成功不等于策略已经学会，必须看重复运行后的回报分布与真实任务指标。

## Agent 与环境怎样形成闭环

以仓储机器人把货箱送到出库口为例，一个时间步可以这样展开：

1. 环境给出 `observation_t`：前置相机图像、激光雷达、当前速度和局部目标。
2. policy 根据观测输出 `action_t`：左右轮速度。
3. 环境执行动作，机器人向前移动或转弯，并返回 `reward_t`、`observation_{t+1}` 以及是否结束。
4. agent 用新观测继续决策，直到送达、碰撞，或达到时间上限。

这不是静态数据管线。policy 决定机器人会走到哪些通道、看到哪些遮挡、遇到哪些风险；policy 更新后，下一批 rollout 的数据分布也会改变。完整的数据闭环与术语定义见 [Agent、环境与训练闭环](rl-agent-environment-loop.md)。

## 强化学习和监督学习有什么不同

四种学习范式最关键的差别，是训练信号从哪里来、策略能否影响后续数据：

| 范式 | 数据来源 | 学习信号 | 策略是否在训练中改变数据分布 | 典型限制 |
| --- | --- | --- | --- | --- |
| 监督学习 | 固定的输入与正确标签 | 单样本预测误差 | 否 | 依赖可靠标签，通常不表达动作的长期后果 |
| 模仿学习 | 专家 demonstration | 复现专家动作 | 部署时会，纯行为克隆训练时不会 | 偏离专家数据后容易累积误差 |
| Offline RL | 固定历史 trajectory | reward、return 与价值估计 | 训练采样不变 | 新策略动作可能超出数据支持范围 |
| Online RL | 当前 policy 与环境持续交互 | rollout 得到的 reward / return | 是 | 交互昂贵、不安全或回报稀疏时成本很高 |

模仿学习回答“专家在这里会怎么做”，RL 回答“什么动作能带来更高长期回报”。Offline RL 虽不在训练期间与环境交互，仍要根据 reward 比较长期决策；online RL 则把采样和优化放在持续变化的闭环里。

## 强化学习真正难在哪里

- **信用分配**：货箱最终送达才得分，早期哪次减速或绕行贡献了成功并不直接可见。
- **探索与安全冲突**：agent 需要尝试未知动作，但真实机器人不能靠碰撞来遍历失败方式。
- **部分可观测**：单帧相机看不到货架后的叉车，当前 observation 不等于完整 state。
- **分布持续变化**：policy 更新会改变它访问的状态和采集的数据，训练目标随闭环数据一起变化。
- **奖励错设**：若只奖励速度，机器人可能超速或走危险捷径；代理指标提高不代表真实目标改善。
- **估计误差会被放大**：value、world model 或 reward model 的小偏差，可能被策略主动利用。

因此，RL 实验不能只报告 reward 曲线。还要检查成功率、约束违反、数据覆盖、策略熵、价值估计误差和分布外行为。

## 什么任务适合强化学习

RL 更适合满足这些条件的任务：动作会影响未来；没有便宜的逐步正确标签；结果可以用回报或约束评价；能够承受足够的仿真、离线数据或在线交互；并且需要在探索、延迟后果或长期规划之间权衡。例如游戏控制、可重置仿真中的机器人策略、带可验证结果的推理，以及世界模型中的规划与策略学习。

当**正确动作标签很便宜**、**交互不安全**，或 **reward 无法代表真实目标**时，RL 通常不是第一选择。此时应先考虑监督学习、模仿学习、规则规划、搜索或离线评估；即使最后采用 RL，也应先解决安全边界、奖励有效性和数据支持问题。

## 完整学习路线 {#完整学习路线}

| 阶段 | 章节 |
| --- | --- |
| 建立语言 | 1. [强化学习专题首页](index.md) |
| 建立语言 | 2. [Agent、环境与训练闭环](rl-agent-environment-loop.md) |
| 长期价值 | 3. [多臂老虎机与探索](multi-armed-bandits-and-exploration.md) |
| 长期价值 | 4. [MDP、回报、价值函数与 Bellman](mdp-value-bellman.md) |
| 长期价值 | 5. [动态规划、Monte Carlo 与 TD](dynamic-programming-monte-carlo-td.md) |
| 经典深度 RL | 6. [Q-learning 与 DQN](q-learning-and-dqn.md) |
| 经典深度 RL | 7. [Policy Gradient 与 REINFORCE](policy-gradient-and-reinforce.md) |
| 经典深度 RL | 8. [Actor-Critic、GAE 与 PPO](policy-gradient-actor-critic-ppo-grpo.md) |
| 经典深度 RL | 9. [连续控制与 SAC](continuous-control-and-sac.md) |
| 低成本与模型驱动 | 10. [Offline Reinforcement Learning](offline-reinforcement-learning.md) |
| 低成本与模型驱动 | 11. [Model-based RL 与世界模型](rl-for-world-models.md) |
| 大模型与具身系统 | 12. [LLM 强化学习：RLHF、RLVR 与 GRPO](llm-rlhf-rlvr-and-grpo.md) |
| 大模型与具身系统 | 13. [VLA、机器人与闭环强化学习](vla-robotics-and-closed-loop-rl.md) |
| 大模型与具身系统 | 14. [RL 系统工程与 verl](verl-code-flow.md) |
| 综合 | 15. [综合实践与算法选择](practice-and-algorithm-selection.md) |

## 按方向进入课程

| 你的目标 | 建议入口与顺序 |
| --- | --- |
| 从零建立经典 RL 基础 | 当前页 -> [交互闭环](rl-agent-environment-loop.md) -> [多臂老虎机与探索](multi-armed-bandits-and-exploration.md) -> [MDP 与 Bellman](mdp-value-bellman.md) -> [动态规划、Monte Carlo 与 TD](dynamic-programming-monte-carlo-td.md) -> [Q-learning 与 DQN](q-learning-and-dqn.md) -> [Policy Gradient 与 REINFORCE](policy-gradient-and-reinforce.md) -> [Actor-Critic、GAE 与 PPO](policy-gradient-actor-critic-ppo-grpo.md) -> [连续控制与 SAC](continuous-control-and-sac.md) |
| 现代 RL 应用完整路线 | [连续控制与 SAC](continuous-control-and-sac.md) -> [Offline RL](offline-reinforcement-learning.md) -> [Model-based RL 与世界模型](rl-for-world-models.md) -> [LLM 强化学习](llm-rlhf-rlvr-and-grpo.md) -> [VLA 与机器人闭环](vla-robotics-and-closed-loop-rl.md) -> [verl](verl-code-flow.md) -> [综合实践与算法选择](practice-and-algorithm-selection.md) |
| 理解 PPO 与后续大模型强化学习 | [MDP 与 Bellman](mdp-value-bellman.md) -> [Policy Gradient 与 REINFORCE](policy-gradient-and-reinforce.md) -> [Actor-Critic、GAE 与 PPO](policy-gradient-actor-critic-ppo-grpo.md) -> [LLM 强化学习](llm-rlhf-rlvr-and-grpo.md) -> [verl 训练流程](verl-code-flow.md) -> [综合实践与算法选择](practice-and-algorithm-selection.md) |
| 研究世界模型与规划 | [交互闭环](rl-agent-environment-loop.md) -> [MDP 与 Bellman](mdp-value-bellman.md) -> [Offline RL](offline-reinforcement-learning.md) -> [Model-based RL 与世界模型](rl-for-world-models.md) |
| 进入 VLA 与机器人闭环 | [交互闭环](rl-agent-environment-loop.md) -> [连续控制与 SAC](continuous-control-and-sac.md) -> [Offline RL](offline-reinforcement-learning.md) -> [Model-based RL 与世界模型](rl-for-world-models.md) -> [VLA 与机器人闭环](vla-robotics-and-closed-loop-rl.md) |

方向入口用于缩短到目标主题的路径。读现代 RL 论文前，至少应能区分 observation 与 state、reward 与 return、on-policy 与 offline 数据。

## 学完后如何判断自己真的理解了

你应能独立完成以下检查，而不只是复述算法缩写：

1. 把一个具体任务写成 observation、state、action、reward、termination / truncation 与 trajectory。
2. 画出从 `environment step` 到 rollout batch、target / advantage、loss 和 policy update 的数据流。
3. 解释 reward、return、value、Q 与 advantage 各自估计什么，以及 Bellman 或 policy gradient 在哪里使用它们。
4. 根据动作空间、数据来源、交互成本和奖励质量，在监督学习、模仿学习、offline RL 与 online RL 之间作出可辩护的选择。
5. 运行最小实验并诊断：表现变化来自策略真的改善，还是 reward hacking、价值偏差、数据不足或评测泄漏。
6. 面对 PPO、SAC、GRPO、Dreamer 或机器人 RL 系统，能指出 rollout 来自哪个 policy、优化目标依赖哪些估计、部署时最大的分布与安全风险是什么。
