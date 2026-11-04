# 算法选择与实验结果怎样判断

算法名称只给出更新规则的一部分。实际选择首先取决于能否交互、动作是否可枚举、观察能否支持状态判断，以及奖励是否表达任务目标。一个满足这些条件的简单基线，比没有数据支持的复杂目标更容易解释。

## Loss 在降，机器人或游戏得分却没有变

先不要把这个现象归因于算法不够先进。可以冻结一批转移，沿一条样本重算奖励、后继状态、终止 mask、目标与当前预测。若 DQN 把 reset 后的新起点当终点后的状态，loss 仍可以下降，因为网络正在拟合一个一致但错误的目标。

接下来确认 actor 或 Q 参数实际变化，再冻结策略独立评测。训练时的 epsilon 探索、奖励 shaping 和尚未结束的部分回报，都会让训练日志与真正执行目标不同。若样本计算正确、参数更新正常，才进一步研究探索覆盖、估计偏差和超参数。

本页的命令输出分别用来区分这些层次。读者可以按当前故障跳到对应部分，不必把所有算法各跑一遍再作判断。

## 从任务约束缩小候选

若动作不改变后续状态，先考虑 Bandit；若只有监督标签或高质量示范，先建立监督或模仿基线。存在延迟后果后，再根据数据来源与动作空间选择候选。

| 条件 | 可考虑的方法 | 首先核对的技术限制 |
| --- | --- | --- |
| 已知小型转移模型、动作可枚举 | DP、value iteration | 模型与奖励是否准确 |
| 可在线交互、有限离散动作 | DQN | replay 覆盖、Q target 与终止 mask |
| 可持续获取新鲜轨迹 | PPO | old anchor、优势、更新幅度与采样延迟 |
| 连续动作、希望重用历史交互 | SAC | 动作密度坐标、critic 偏差与温度 |
| 只能使用固定日志 | Offline RL 与 behavior cloning 基线 | 数据 support、行为偏移与离线评估假设 |
| 有可校准的动作条件模型 | MPC、Dyna、模型内 actor-critic | 候选动作上的模型偏差与规划成本 |
| 同一 prompt 可采多回答且奖励可比较 | GRPO、RLOO 或 PPO 等 | group id、奖励可信度、长度与概率约定 |

连续动作并不排除 PPO，SAC 也不天然适用于任意固定日志。观察缺少关键历史时，先修正状态表示或记忆接口；奖励只测格式时，换优化器不会自动得到正确内容。各方法的推导分别见 [DQN](q-learning-and-dqn.md)、[PPO](policy-gradient-actor-critic-ppo-grpo.md)、[SAC](continuous-control-and-sac.md)、[Offline RL](offline-reinforcement-learning.md)与[世界模型](rl-for-world-models.md)。

## GridWorld 输出中的三个问题

从仓库根目录运行 NumPy 示例

```bash
python files/assets/examples/reinforcement-learning/gridworld.py --seed 7
```

固定示例输出为

```text
policy_value_mean=-4.464
optimal_value_start=-1.900, greedy_policy=[1, 1, 2, 1, 1, -1]
q_mean=-1.166, q_max=0.000
```

第一行评估均匀随机策略，并对六个状态价值取平均。第二行是 value iteration 的最优起点价值，最短路径奖励 $-1,-1,0$ 在折扣 $0.9$ 下得到 $-1.9$；策略数组按状态给出上、右、下、左的动作编号，终点用 $-1$ 表示无动作。第三行汇总有限交互学到的 Q 表，包含不同状态动作项。

因此 `policy_value_mean`、`optimal_value_start` 和 `q_mean` 不能直接比较谁更大就说谁训练更好。它们的策略、统计对象和聚合方式都不同。若终点值异常，先检查进入终点的 reward 和后续 bootstrap；若短训练的 Q 与精确解不同，先看探索覆盖和更新次数，不能要求一次有限采样等于 DP 解。机制解释见 [Bellman](mdp-value-bellman.md)与 [DP/MC/TD](dynamic-programming-monte-carlo-td.md)。

## Bandit 输出怎样区分运气与决策

```bash
python files/assets/examples/reinforcement-learning/bandit.py --seed 7 --steps 2000
```

这组固定任务中，epsilon-greedy 的平均遗憾为 `0.191`、最佳臂选择比例为 `0.895`，UCB 为 `0.050` 与 `0.955`。这些值只描述 seed 7 所生成的均值与噪声。两种方法共享潜在奖励噪声，降低了单次比较的环境差异，却没有消除随机任务与算法参数的影响。

如果比较多个 seed，应先在每个相同 seed 内计算策略差值，再统计差值的均值与波动。把同一训练 run 中高度相关的数千步当成数千个独立重复，会夸大证据量。最好同时报告任务数、每个任务交互预算和 reward 分布，详见 [Bandit 正文](multi-armed-bandits-and-exploration.md)。

## 连续控制中的有限更新能证明什么

依赖安装方式见[示例 README](../assets/examples/reinforcement-learning/README.md)。具备 PyTorch 与 Gymnasium 后可运行

```bash
python files/assets/examples/reinforcement-learning/deep_rl_smoke.py sac --seed 7 --steps 256
```

SAC 路径使用 Pendulum-v1 的三维观察与 `[-2,2]` 动作。Replay 达到 64 条后开始更新，256 步共 193 次更新。`actor_parameter_delta` 和 `critic_parameter_delta` 为正，说明参数确实变化；有限 `actor_loss`、`critic_loss` 和 `alpha` 说明本次短流程没有立即出现数值异常。它们都不能代替回报收敛证据。

还需要区分 `completed_episode_return_mean` 与 `current_partial_episode_return`。未结束轨迹只有部分回报，不能作为完整 episode 均值，也不能因没有完整 episode 就填零。PPO 也可用于连续控制，但本仓库 PPO smoke 使用 CartPole，不能把这两个默认运行结果当成 Pendulum 上的 PPO/SAC 公平对比。

同样 100 万环境步的两个实验，如果一个每步更新一次、另一个更新十次，交互预算相同而训练计算不同。比较时应一起报告真实 environment steps、optimizer updates、wall time、模型规模与评测协议。模型想象步数和机器人真实步数也应分开。

## Verifier 组内目标的数值证据

```bash
python files/assets/examples/reinforcement-learning/policy_objectives.py --seed 7
```

脚本中 format 组奖励全为 $1$，优势精确为零；arithmetic 组奖励为 $[0,1,2]$，优势约为 $[-1.224745,0,1.224745]$；最终 `clipped_policy_loss=0.000795`。逐点 log-ratio 出现负值是允许的，脚本没有计算完整分布的精确 KL，也没有训练真实语言模型。

这组输出回答“目标函数按哪个方向计算”，不能回答“语言模型是否变聪明”。真实 RLVR 还需要隐藏任务正确率、零方差组比例、回答长度和奖励分量。若公开 reward 快速上涨但隐藏正确率下降，应先检查 verifier 漏洞、数据泄漏和格式捷径。公式见 [LLM 强化学习](llm-rlhf-rlvr-and-grpo.md)，分布式字段对应关系见 [verl](verl-code-flow.md)。

## 让每种结论对应一份证据

| 想表达的结论 | 所需证据 | 仅靠什么还不够 |
| --- | --- | --- |
| 更新公式实现正确 | 小数值例子、边界和梯度方向检查 | loss 是有限值 |
| 训练程序能运行 | 数据流、参数变化、资源与恢复检查 | 只成功初始化模型 |
| 某任务学到了策略 | 足够训练与独立闭环评测 | 短 smoke 或训练 loss 下降 |
| 方法在该设置更好 | 相同预算、多个独立 run 与误差范围 | 单次最好成绩 |
| 能跨场景泛化 | 明确未见场景与硬件上的评测 | 同一环境重复调参 |

这些证据可以逐层增加，不能相互替代。最有用的失败记录通常是一条可复算轨迹，保留观察、动作、奖励、终止原因、目标值和更新前后概率，再配上它所代表的任务条件。这样才能区分数据协议错误、目标设计问题与算法假设不成立。
