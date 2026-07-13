# Agent、环境与训练闭环

## 本章回答什么

强化学习训练到底在循环什么？本章用一个仓储机器人任务回答：环境如何产生数据，policy 如何选择动作，一批 trajectory 又如何变成 return、advantage 或 target，最后更新 policy。

读完后，你应能从一次 `environment step` 追到一次参数更新，并说明为什么更新后的策略会采到不同的数据。

## 先修知识

先读 [强化学习导读](index.md)，确认 RL 的目标与适用边界。数学上需要理解随机变量、条件分布和期望；不熟悉时可先看 [概率分布、随机变量与期望](../prerequisite-math/probability-distributions-and-expectation.md) 和 [随机过程与马尔可夫性](../prerequisite-math/stochastic-processes-and-markov.md)。

本章只建立术语和数据流，不推导 Bellman 方程或 policy gradient。

## 最小任务

任务是让移动机器人从仓库入口把货箱送到指定出库口：

| 要素 | 最小定义 |
| --- | --- |
| observation | 前置相机图像、局部激光雷达、轮速、上一动作和目标方向 |
| state | 机器人与所有货架、行人、叉车的真实位姿、速度等动力学量，以及任务进度 |
| action | 左右轮速度，连续取值 |
| reward | 成功送达 `+100`，碰撞 `-100`，每步 `-0.01` |
| termination | 成功送达或发生不可恢复碰撞 |
| truncation | 运行达到 1,000 步时间上限 |

一个时间步里，policy 收到 `observation_t`，输出 `action_t`；环境执行动作后返回 `reward_t`、`observation_{t+1}` 和 `terminated/truncated`。若任一结束标志为真，当前 episode 停止；两者的学习含义不能混为一谈。

## 核心机制

### 先把术语钉牢

| 术语 | 定义 | 仓库任务中的例子 |
| --- | --- | --- |
| observation | agent 在当前时刻实际收到的信息 | 相机、雷达、轮速与目标方向 |
| state | 给定 action 后，足以确定下一状态分布的当前信息，即对相关历史的充分统计量 | 包括被货架遮挡的叉车真实位置、速度与运动状态 |
| action | agent 交给环境、会影响下一步的决策量 | 左右轮速度 |
| reward | 环境对一次转移给出的标量反馈 | 碰撞 `-100`，普通移动 `-0.01` |
| termination | 任务在语义上到达终止状态 | 送达成功或不可恢复碰撞 |
| truncation | 因外部限制截断交互，但任务状态本身未必终止 | 达到 1,000 步上限 |
| episode | 从 reset 开始，到 termination 或 truncation 为止的一次完整交互 | 一次送货尝试 |
| trajectory | 按时间排列的 observation、action、reward 与结束标志序列 | 机器人本次送货留下的全量记录 |
| policy | 根据 observation 或 state 产生动作或动作分布的规则 \(\pi(a\mid o)\) | 神经网络输出轮速分布 |
| rollout | 用某个 policy 在一个或多个环境中实际采集 trajectory 的过程或结果 | 64 个并行仓库仿真采一批送货轨迹 |
| environment step | 环境接收一个 action、推进一次动力学，并返回 reward、下一 observation 和结束标志 | 执行轮速 50 ms 后返回新传感器读数 |

### Observation 不等于 state

货架会遮住迎面驶来的叉车。环境的完整 state 包含叉车位置，但单帧相机 observation 看不到它。这种设定称为**部分可观测马尔可夫决策过程（POMDP）**：环境内部仍有潜在的 Markov state，agent 每步只收到这个 state 的部分 observation，因此需要用历史帧、记忆模块或 belief state 推断当前状态与风险。

### Reward 不等于完整目标

每步都有 `-0.01` 的时间惩罚，但只有送达时才给 `+100`；因此，**稀疏的是成功信号**，而不是所有 reward 都为零。入口处选择绕开拥堵通道，可能要几百步后才体现为成功，形成**延迟奖励**。算法需要把终局结果归因到此前动作；额外加入进度奖励虽能变密，却可能诱导机器人追求局部距离而忽视碰撞风险。

### Policy 既是模型，也是数据采集规则

初始 policy 常撞在入口附近，因此 rollout 大多是短失败轨迹；更新后它进入仓库深处，采到新的路口、遮挡与叉车交互。也就是说，**改变 policy 就改变了下一批训练数据的分布**。这正是 online RL 与固定监督数据训练的根本差异之一。

## 训练数据流

一次完整训练迭代可以压缩成三行：

```text
observation_t -> policy -> action_t -> environment
environment -> reward_t + observation_{t+1} + terminated/truncated
trajectory batch -> return / advantage / target -> loss -> policy update
```

逐段看这条链路：

1. **采样**：固定当前版本 policy，在并行环境中反复执行 step，记录 observation、action、reward、下一 observation、结束标志，以及算法需要的 log-prob 或 value。
2. **组批**：把多个 episode 或固定长度片段整理成 trajectory batch。truncation 处是否 bootstrap value，取决于算法和环境时间上限语义；不能直接当作真实终止。
3. **构造学习信号**：Monte Carlo 方法算 return，actor-critic 算 advantage 与 value target，Q-learning 类方法构造 TD target。
4. **优化**：由学习信号构造 policy loss、value loss 或 Q loss，反向传播并更新参数。
5. **重新采样**：新 policy 再进入环境。对 on-policy 算法，旧 batch 很快失效；对 replay 或 offline 方法，也必须处理行为策略与当前策略的分布差异。

仓库任务中，若 batch 几乎全是入口碰撞，问题可能不是 loss 写错，而是数据没有覆盖成功路径。若更新后成功率上升但安全违规也上升，则 reward 或约束没有表达完整目标。

## 常见失败

| 失败 | 表现 | 应先检查什么 |
| --- | --- | --- |
| 把 termination 和 truncation 合并 | 时间上限处 value target 被错误置零 | 环境 API、bootstrap mask 与时间上限语义 |
| observation 缺少历史 | 相同画面下动作摇摆，遇遮挡就失败 | frame stack、记忆状态与传感器覆盖 |
| 奖励过于稀疏 | 长期没有成功 trajectory，梯度近似噪声 | 探索、示范数据、课程设计或安全的 reward shaping |
| 奖励塑形偏离目标 | reward 上升但真实送达率或安全性下降 | 单项 reward、约束指标与独立任务评测 |
| 混用过期 rollout | policy ratio 或 advantage 与采样策略不匹配 | rollout policy 版本、old log-prob 与权重同步 |
| reset 或记录错位 | 下一 episode 的 observation 接到上一条 trajectory | done mask、buffer 边界和时间下标 |

## 自检与练习

1. 为什么货架后的叉车位置属于 state，却不一定属于当前 observation？给出一种让 policy 利用历史的方法。
2. 手工跟踪三步记录：写出每一步的 `observation_t`、`action_t`、`reward_t`、`observation_{t+1}` 和结束标志，标明 trajectory 在哪里结束。
3. 若机器人在第 1,000 步被时间上限截断，但仍处于正常通道，解释为什么这与碰撞 termination 的 value target 不应自动相同。
4. 把成功奖励从 `+100` 改成每向目标靠近一米 `+1`，列出至少一种 reward hacking 风险和一个独立评测指标。
5. 比较训练早期与训练后期各 100 条 rollout 的状态覆盖；说明观察到的变化为何可能来自 policy，而不是环境本身。

## 下一步

下一章进入 [MDP、价值函数与 Bellman](mdp-value-bellman.md)，把本章的 state、transition、reward 和 trajectory 写成可递推的长期价值问题。若 return、value、advantage 或 policy objective 仍不熟悉，可并行阅读 [价值函数、策略梯度与强化学习目标](../prerequisite-math/value-policy-and-rl-objectives.md)。
