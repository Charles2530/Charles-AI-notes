# Agent、环境与训练闭环

把机器人放进仓库，任务是从入口移动到货架。策略看到相机、激光雷达、轮速和目标位置，输出左右轮速度。环境执行动作后返回新传感器数据、奖励和结束标志。强化学习的基本单位是这次转移；网络结构只是决定动作的一种实现。

## 一条轨迹记录了什么

本文用 $r_t$ 表示执行 $a_t$ 后收到的奖励，轨迹写作

\[
\tau=(o_0,a_0,r_0,o_1,a_1,r_1,\ldots,o_T).
\]

有些教材将同一个奖励写作 $r_{t+1}$，二者只差索引约定。实际存储必须保证每行的动作、奖励和后继观察对应同一次环境调用。

| 对象 | 仓库机器人的具体含义 | 数据中应如何理解 |
| --- | --- | --- |
| observation $o_t$ | 此刻收到的图像、雷达和轮速 | 策略实际可访问的输入 |
| state $s_t$ | 机器人与障碍物位置、速度等完整环境信息 | 满足 Markov 假设所需的状态，未必可直接看到 |
| action $a_t$ | 本周期左右轮速度命令 | 必须同时说明单位、范围和持续时间 |
| reward $r_t$ | 到达目标、碰撞、耗时产生的数值反馈 | 属于执行该动作后的转移 |
| policy $\pi$ | 由观察或历史产生动作的规则 | 可以随机采样，也可以确定输出 |
| episode | 一次重置之后到结束之前的交互 | 定义任务统计与重置边界 |
| rollout | 为训练或评测收集的一段轨迹 | 可能只覆盖 episode 的一部分 |

教学奖励可以定义为到达目标 $+100$、不可恢复碰撞 $-100$、其他每步 $-0.01$。奖励规则决定偏好。例如同样成功的两条路径，折扣与步成本通常偏好较早完成；如果额外给“接近目标”奖励，却没有检查任务完成，机器人可能重复靠近再远离刷分。每个奖励分量都需要对应一个实际行为目的。

一次 environment step 的顺序如下，优化器通常在收集若干步后才运行。

```text
observation_t -> policy -> action_t -> environment.step
environment.step -> reward_t + observation_{t+1} + terminated + truncated
trajectory batch -> return / advantage / target -> loss -> policy update
```

## 跟踪一次仓库转弯

设机器人在同一路口两次看见叉车，当前距离都为 2 米。第一段历史中，0.2 秒前叉车还在 2.4 米，表示正在接近；第二段历史中，之前只有 1.6 米，表示正在远离。如果两次都只保存当前距离，策略收到相同输入，却要面对不同碰撞概率。

一次真实控制可以按下表记录。数值是教学示意，时间顺序比字段名称更重要。

| 时刻 | 发生了什么 | 记录的对象 |
| --- | --- | --- |
| 10.00 s | 采集图像与距离，叉车正在接近 | `observation_capture_time` 与历史 |
| 10.03 s | 策略提出左轮 0.1、右轮 0.2 m/s | `proposed_action` |
| 10.04 s | 控制器限制右轮为 0.15 m/s | `executed_action` |
| 10.14 s | 本周期结束，读取后继观察 | 与实际动作配对的 `next_observation` |
| 10.14 s | 未碰撞、未到目标 | 步奖励 −0.01，`terminated=False` |

如果采样器此时恰好收满一个 batch，只是结束了这段数据，机器人任务仍未结束。若随后 reset，必须把 10.14 秒的最后观察保留下来；新回合入口画面不能替代它。动作持续时间和实际执行值还会进入动力学监督，这与仅保存策略输出不同。

这些时间戳是建议的日志字段，并非某个环境 API 自动保证。Gymnasium 的 `step` 返回通常组织为 observation、reward、terminated、truncated、info，包装器怎样保存最后观察需要按实际版本检查。

## 为什么一张图像可能不够

前述两段叉车历史产生相似的当前图像，却需要不同的避障动作。若策略只看当前图像，同一个输入下的后果就像额外噪声。

完全状态的 Markov 条件是，给定 $s_t,a_t$ 后，下一状态与奖励不再依赖更早历史。机器人观察往往不满足该条件，因此可使用历史 $h_t=(o_{\le t},a_{<t})$，或用 belief 表示对隐藏状态的条件分布

\[
b_t(s)=p(s_t=s\mid h_t).
\]

在已知转移 $P$ 和观测模型 $O$ 时，belief 的更新先根据动作预测状态，再用新观察修正

\[
b_{t+1}(s')\propto O(o_{t+1}\mid s')\sum_sP(s'\mid s,a_t)b_t(s).
\]

循环网络、历史帧或记忆模块是在学习一个可供决策的历史摘要，并不自动等于精确 belief。是否足够，要看遮挡、延迟、速度变化等需要记忆的情形中，策略能否区分后果。[随机过程与马尔可夫性](../prerequisite-math/stochastic-processes-and-markov.md)给出条件独立的数学解释。

## 结束任务与停止采样是两回事

假设目标已到达，后续任务价值为零，`terminated=True`。如果只是采样器达到 1000 步上限，机器人仍可能继续完成任务，则是外部 `truncated=True`。两者都需要停止当前采样段或重置环境，价值目标却不同。

令奖励为 $1$，折扣 $0.9$，结束前真实后继状态的价值估计为 $6$。真实终止时目标为 $1$；外部截断时目标仍是 $1+0.9\times6=6.4$。这就是 bootstrap mask 通常使用 `1 - terminated` 的原因。

```python
reward, gamma, next_value = 1.0, 0.9, 6.0
terminal_target = reward
truncated_target = reward + gamma * next_value
assert terminal_target == 1.0
assert abs(truncated_target - 6.4) < 1e-12
```

这里的 `next_value` 必须来自截断前的最后观察，不能来自自动重置后新 episode 的初始观察。向量环境和自动重置包装器尤其需要核对 final observation 的接口。若任务本身规定“必须在 1000 步以内完成”，截止时刻属于真实任务终止；剩余时间也应进入状态，否则同一位置在不同剩余时间下的价值无法区分。[Gymnasium 的时间限制说明](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)详细区分了这些情况。

训练还需要第二种边界。GAE 的误差递推不能越过重置进入下一条轨迹，即使上一段因外部截断而保留了 value bootstrap。因此“是否继续估计未来价值”和“是否继续串接下一个采样误差”应分别记录，见 [GAE 与 PPO](policy-gradient-actor-critic-ppo-grpo.md)。

## 一份能被重放和解释的数据

每次交互至少保留观察、实际执行动作、奖励、后继观察、终止原因、episode id 和时间顺序。On-policy 更新还需要采样策略的动作 log probability；动作经安全过滤器改变时，还要分开记录提出动作与执行动作。采样策略版本、奖励版本和环境版本会影响数据含义。

训练闭环随后分成两件工作。评估当前策略时保持参数不变，估计它在固定任务分布上的表现；改进策略时使用新的价值或梯度更新参数。混用训练探索动作和评测动作、在评测中继续更新、只报告未结束 episode 的部分回报，都会改变结论。[MDP 与 Bellman](mdp-value-bellman.md)把这种交互写成递推方程，[实践与算法选择](practice-and-algorithm-selection.md)解释如何读实验输出。
