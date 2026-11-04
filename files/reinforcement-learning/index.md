# 强化学习

强化学习研究怎样根据行动的后果改进决策。系统收到观察，选择动作，环境给出奖励和新观察；动作既改变当前得分，也改变以后能看到的数据。机器人为了抓住杯子先绕开障碍，语言模型为了答对问题先进行推理，都涉及眼前成本与后续收益的关系。是否适合用强化学习，取决于能否定义决策过程、获得有意义的反馈，并承担探索或离线外推的代价。

## 导航任务怎样产生学习信号

设一个教学任务连续三次行动得到奖励 $1,0,2$，折扣因子 $\gamma=0.9$。从起点看的回报为

\[
G_0=1+0.9\times0+0.9^2\times2=2.62.
\]

第二步虽然没有即时奖励，却可能是获得最后两分的必要动作。只拟合“哪一步的奖励最大”，会丢掉这条因果链。强化学习里的价值函数估计从某个状态或动作继续执行策略能得到多少回报；策略则决定如何行动。价值预测和策略改进因此相互依赖。

这里有三种容易混在一起的量。reward 是环境给某次转移的反馈，return 是一段未来反馈的累计，value 是在策略与环境随机性下对 return 的条件期望。单条轨迹回报高，不保证策略平均表现好；value 高，也不表示眼前奖励高。它可能来自预计稍后获得的机会，也可能来自错误外推。

## 先辨认数据怎样产生

| 设置 | 训练中有什么 | 优化的直接对象 | 容易漏掉的限制 |
| --- | --- | --- | --- |
| 监督学习 | 输入与目标标签 | 标签预测误差 | 标签未必表达长期后果 |
| 模仿学习 | 示范状态与动作 | 模仿行为的概率或距离 | 自己行动后可能进入示范未覆盖的状态 |
| Offline RL | 固定的状态、动作、奖励和后继状态日志 | 数据约束下的长期回报 | 无法现场验证没见过的动作 |
| Online RL | 能持续与环境交互 | 交互产生的回报 | 探索消耗时间、资源并可能产生失败 |

Off-policy 和 offline 也不同。DQN、SAC 可以从旧策略的 replay 学习，同时继续收集新数据，因此属于 online off-policy 方法。Offline RL 的训练数据固定，错误动作价值缺少新反馈纠正。[离线强化学习](offline-reinforcement-learning.md)专门处理这一困难。

## 按问题查阅 {#完整学习路线}

这些文章按技术依赖连接，可以围绕眼前的问题阅读。数学符号不熟时，可随时返回[概率与期望](../prerequisite-math/probability-distributions-and-expectation.md)。

| 当前问题 | 对应正文 |
| --- | --- |
| observation、state、action 与一次环境交互分别是什么 | [Agent、环境与训练闭环](rl-agent-environment-loop.md) |
| 未知选项该怎样探索，探索成本如何计算 | [多臂老虎机与探索](multi-armed-bandits-and-exploration.md) |
| 延迟奖励怎样写成可递推的价值 | [MDP、价值函数与 Bellman](mdp-value-bellman.md) |
| 有模型和只有轨迹时怎样估计同一个价值 | [动态规划、Monte Carlo 与 TD](dynamic-programming-monte-carlo-td.md) |
| 怎样通过动作价值学习离散决策 | [Q-learning 与 DQN](q-learning-and-dqn.md) |
| 奖励为什么能对动作概率求梯度 | [Policy Gradient 与 REINFORCE](policy-gradient-and-reinforce.md) |
| 怎样用 critic 降低方差，并约束策略更新 | [Actor-Critic、GAE 与 PPO](policy-gradient-actor-critic-ppo-grpo.md) |
| 连续动作的概率、熵和 replay 怎样协作 | [连续控制与 SAC](continuous-control-and-sac.md) |
| 固定日志为什么会诱发分布外动作高估 | [Offline Reinforcement Learning](offline-reinforcement-learning.md) |
| 怎样利用环境预测进行规划或策略学习 | [Model-based RL 与世界模型](rl-for-world-models.md) |
| 人类偏好、可验证奖励怎样训练语言模型 | [LLM 强化学习](llm-rlhf-rlvr-and-grpo.md) |
| 分布式系统怎样保存采样与更新的概率关系 | [verl 训练流程](verl-code-flow.md) |
| 动作块怎样接上真实反馈、恢复和人工接管 | [VLA、机器人与闭环强化学习](vla-robotics-and-closed-loop-rl.md) |
| 算法假设和实验结果怎样一起判断 | [实践与算法选择](practice-and-algorithm-selection.md) |

## 从一次更新核对到实际行为

价值方法先检查 reward、bootstrap 和 Q 的尺度；策略梯度先检查动作 log probability、advantage 的符号和优化方向；模型方法先检查预测是否会随动作正确变化。它们最后都要回到闭环行为。训练 loss 降低、奖励上涨和真实任务成功，是三种不同证据。

仓库的[可运行示例](../assets/examples/reinforcement-learning/README.md)提供 Bandit、GridWorld、策略目标数值计算，以及 DQN、REINFORCE、PPO、SAC 的短更新流程。前几种适合逐项核对公式；深度 RL 的短运行只检查数据流、有限 loss 和参数变化，不能据此宣称收敛。每篇文章会说明具体检查了什么，以及哪些结论仍需要多个随机种子和独立环境评测。

本站 [REINFORCE](policy-gradient-and-reinforce.md) 与 [PPO](policy-gradient-actor-critic-ppo-grpo.md) 使用同一张六格地图，正文给出可独立运行的完整 NumPy 学习循环，并用环境的精确价值评估训练前后变化。它们让轨迹、目标和参数更新对应起来；两者训练预算不同，文中的结果不用于算法性能排名。
