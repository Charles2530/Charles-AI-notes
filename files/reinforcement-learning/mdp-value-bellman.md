# MDP、价值函数与 Bellman

强化学习的基础不是某个算法，而是一套描述闭环决策的语言。只要把 `state`、`action`、`reward`、`policy`、`value` 和 `Bellman equation` 理顺，后面的 PPO、GRPO、Dreamer 和世界模型都会清楚很多。

!!! note "初学者先抓住"
    强化学习要解决的是“现在做什么会让未来更好”。所以它不只关心当前动作是否看起来合理，还关心这个动作会把系统带到什么状态，以及后面还能拿到多少奖励。

## 1. MDP 是最小决策模型

MDP 通常写成：

\[
\mathcal{M}=(\mathcal{S}, \mathcal{A}, P, R, \gamma)
\]

| 符号 | 简洁解释 |
| --- | --- |
| \(\mathcal{S}\) | state space：环境可能处在的状态集合 |
| \(\mathcal{A}\) | action space：agent 可以选择的动作集合 |
| \(P(s_{t+1}\mid s_t,a_t)\) | transition：执行动作后状态如何变化 |
| \(R(s_t,a_t)\) | reward：当前动作或结果带来的反馈 |
| \(\gamma\) | discount factor：未来奖励打多少折扣 |

如果状态不能完全观测，就变成 POMDP。机器人和 VLA 常常是 POMDP，因为摄像头只能看到一部分世界，抽屉里、遮挡后、过去发生过的接触都可能影响下一步。

!!! note "难点解释：state 不等于 observation"
    `observation` 是传感器看到的东西，例如当前图像；`state` 是足以预测未来的内部描述，例如物体位置、速度、接触关系和任务进度。VLM 往往处理 observation，世界模型要努力学习更接近 state 的 latent。

## 2. Policy 是行为规则

策略写成：

\[
\pi_\theta(a_t\mid s_t)
\]

意思是：给定当前状态 \(s_t\)，模型以什么概率选择动作 \(a_t\)。在语言模型 RLHF 中，状态可以理解为 prompt 加已经生成的前缀，动作就是下一个 token。

| 术语 | 简洁解释 |
| --- | --- |
| deterministic policy | 同一状态总输出同一动作，适合稳定执行 |
| stochastic policy | 同一状态按概率采样动作，适合探索和生成 |
| behavior policy | 负责采样数据的策略，可能是旧模型或示范策略 |
| target policy | 正在被优化或评估的策略 |

!!! example "有趣例子：走迷宫"
    如果每次到路口都固定向左走，这是 deterministic policy。如果有 70% 概率向左、30% 概率向右，这是 stochastic policy。训练早期需要一定随机性，否则 agent 可能永远发现不了右边有出口。

## 3. Reward、Return 和 Value

单步 reward 是当前反馈：

\[
r_t = R(s_t,a_t)
\]

return 是从当前时刻开始累计的未来奖励：

\[
G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots
\]

value 是在某个策略下，从状态 \(s\) 出发能拿到的期望 return：

\[
V^\pi(s)=\mathbb{E}_\pi[G_t\mid s_t=s]
\]

Q-value 则多固定了当前动作：

\[
Q^\pi(s,a)=\mathbb{E}_\pi[G_t\mid s_t=s,a_t=a]
\]

Advantage 衡量“这个动作比平均策略好多少”：

\[
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)
\]

!!! note "难点解释：reward 和 value 不要混"
    reward 像这一步考试得了几分，value 像从现在开始预计整门课最终能拿多少分。一个动作当前 reward 可能低，但如果它把系统带到更好的未来状态，value 仍然可能很高。很多长时任务难，就难在短期反馈和长期收益不一致。

## 4. Bellman 方程

Bellman 方程把“当前价值”拆成“当前奖励 + 下一状态价值”：

\[
V^\pi(s)=\mathbb{E}_{a\sim\pi,\,s'\sim P}
\left[
R(s,a)+\gamma V^\pi(s')
\right]
\]

Q-value 也可以写成：

\[
Q^\pi(s,a)=\mathbb{E}_{s'\sim P}
\left[
R(s,a)+\gamma \mathbb{E}_{a'\sim\pi} Q^\pi(s',a')
\right]
\]

这就是动态规划、TD learning、actor-critic 和世界模型规划的共同基础。

!!! note "图像化理解：Bellman 是可递归的未来账本"
    Bellman 不是神秘公式，它只是说：要知道现在值多少钱，就看现在立刻得到多少，加上下一步状态值多少钱。世界模型的价值在这里很直接：如果能预测 \(s'\) 和 \(r\)，就能在执行前估算未来账本。

## 5. 从 Bellman 到 Q-learning

Q-learning 直接学习 \(Q(s,a)\)：在状态 \(s\) 做动作 \(a\) 后，长期看大概能拿多少回报。它用 Bellman 思路构造 TD target：

\[
y_t = r_t + \gamma \max_{a'} Q_{\text{target}}(s_{t+1}, a')
\]

再让当前网络的 \(Q(s_t,a_t)\) 靠近这个 target：

\[
L=\left(y_t-Q_\theta(s_t,a_t)\right)^2
\]

| 术语 | 简洁解释 |
| --- | --- |
| Q-learning | 学状态-动作价值 \(Q(s,a)\)，再选价值最高的动作 |
| TD target | 用“当前奖励 + 下一状态估计价值”拼出的训练目标 |
| bootstrapping | 用模型自己的下一步估计来训练当前估计 |
| replay buffer | 保存过去交互样本，训练时随机抽取，减少样本相关性 |
| target network | 慢更新或延迟更新的目标网络，让 TD target 不至于每步剧烈漂移 |

!!! example "有趣例子：餐厅排队"
    你在商场里选择排哪家餐厅。reward 是这次吃得是否满意，value 是“排这家之后，等待、用餐、赶电影整体值不值”。Q-learning 不只记“这家好吃”，还要记“现在排队 50 分钟时还值不值”。

DQN 把 Q-learning 接到深度网络上，用图像作为输入，用神经网络预测每个动作的 Q-value。它的两个关键工程技巧是 experience replay 和 target network。

![Deep Q-learning with Experience Replay: Breakout reward](../assets/images/paper-figures/reinforcement-learning/dqn-breakout-average-reward.png){ width="420" }
![Deep Q-learning with Experience Replay: Seaquest reward](../assets/images/paper-figures/reinforcement-learning/dqn-seaquest-average-reward.png){ width="420" }
![Deep Q-learning with Experience Replay: Breakout value](../assets/images/paper-figures/reinforcement-learning/dqn-breakout-average-v.png){ width="420" }
![Deep Q-learning with Experience Replay: Seaquest value](../assets/images/paper-figures/reinforcement-learning/dqn-seaquest-average-v.png){ width="420" }

<small>图源：[Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)，Figure 2。原论文图意：左侧两图是 Breakout 和 Seaquest 训练中的平均回合奖励；右侧两图是固定状态集合上的平均最大预测 action-value。</small>

!!! note "图解：reward 曲线为什么抖，value 曲线为什么更平滑"
    平均回合 reward 很容易抖，因为策略一点小变化就会让整局游戏进入不同状态分布。右侧的平均最大 Q-value 用固定状态集合观察模型估计，像一个验证集上的长期价值读数，所以更平滑。初学者读训练曲线时不要只盯 reward：value、KL、entropy、success rate 和真实闭环评测经常要一起看。

## 6. Value 真的学到了什么

DQN 的 Seaquest 图很适合解释“value 是未来机会”，而不只是当前分数。

<div class="atlas-value-figure">
  <figure class="atlas-value-curve">
    <img src="../../assets/images/paper-figures/reinforcement-learning/dqn-seaquest-value-curve.png" alt="DQN Seaquest value curve" />
  </figure>
  <div class="atlas-value-frames">
    <figure>
      <img src="../../assets/images/paper-figures/reinforcement-learning/dqn-seaquest-frame-a.png" alt="DQN Seaquest frame A" />
      <figcaption>A</figcaption>
    </figure>
    <figure>
      <img src="../../assets/images/paper-figures/reinforcement-learning/dqn-seaquest-frame-b.png" alt="DQN Seaquest frame B" />
      <figcaption>B</figcaption>
    </figure>
    <figure>
      <img src="../../assets/images/paper-figures/reinforcement-learning/dqn-seaquest-frame-c.png" alt="DQN Seaquest frame C" />
      <figcaption>C</figcaption>
    </figure>
  </div>
</div>

<small>图源：[Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)，Figure 3。原论文图意：左图是 Seaquest 中 30 帧片段的预测 value；A、B、C 是曲线上三个时间点对应的游戏画面。</small>

!!! note "图解：A、B、C 三个点在讲什么"
    A 点敌人出现，模型预测“现在有得分机会”，value 上升。B 点鱼雷快击中敌人，未来得分几乎要兑现，value 达到峰值。C 点敌人消失，机会被消耗，value 回落。这个例子能帮你分清 reward 和 value：reward 是打中那一下的即时分数，value 是打中之前模型已经看到的未来机会。

## 7. Model-free 与 Model-based

| 路线 | 学什么 | 优点 | 风险 |
| --- | --- | --- | --- |
| Model-free RL | 直接学习 policy 或 value | 不需要显式 dynamics，接口简单 | 样本效率低，长时信用分配难 |
| Model-based RL | 学 transition / world model，再规划或训练策略 | 可用 imagined rollout，提高样本效率 | 模型误差会被 planner 或 actor 利用 |
| Imitation / BC | 模仿示范动作 | 稳定、省交互、工程常用 | 出分布后容易连环偏离 |
| Offline RL | 只用离线数据学习策略 | 适合昂贵或危险环境 | 分布外动作的价值估计容易虚高 |

VLA 常从 BC 起步，因为机器人实机探索代价高。世界模型和 RL 的作用，是在 BC 之外补上“动作后果”和“长期奖励”的闭环。

## 8. On-policy、Off-policy 与 Offline RL

这三个词很容易混：

| 类型 | 数据从哪里来 | 更新时最怕什么 |
| --- | --- | --- |
| on-policy RL | 当前 policy 刚采样的数据 | 样本贵、方差大、吞吐低 |
| off-policy RL | replay buffer 里的历史数据，仍会继续收新数据 | 旧数据和当前策略差太远 |
| offline RL | 固定数据集，不再和环境交互 | 分布外动作的价值被高估 |

![Online, off-policy and offline RL](../assets/images/paper-figures/reinforcement-learning/offline-rl-online-offpolicy-offline.png){ width="920" }

<small>图源：[Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems](https://arxiv.org/abs/2005.01643)，Figure 1。原论文图意：对比 classic online RL、classic off-policy RL 和 offline RL 的数据收集与训练关系。</small>

!!! note "图解：三种 RL 的关键差别"
    左图 online RL 每轮用当前 policy 收新数据；中图 off-policy RL 把多轮 policy 的数据放进 buffer 反复使用；右图 offline RL 的数据集一开始就固定，训练过程不能再试错。机器人、医疗、自动驾驶和大模型后训练经常关心 offline 或接近 offline 的设置，因为真实试错成本高。

Offline RL 不是“把监督学习数据拿来做 RL”这么简单。它真正难的是：模型可能会想选数据里几乎没出现过的动作，而这些动作的 Q-value 只是神经网络外推出来的，可能虚高。

![Offline RL support versus distribution constraints](../assets/images/paper-figures/reinforcement-learning/offline-rl-support-vs-distribution.png){ width="920" }

<small>图源：[Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems](https://arxiv.org/abs/2005.01643)，Figure 4。原论文图意：用一维 lineworld 说明 distribution constraint 和 support constraint 的差别。</small>

!!! note "图解：support constraint 为什么比“完全模仿分布”更灵活"
    图里 behavior policy 很偏向向左走，但只要数据支持向右这个动作，support constraint 仍允许学习出向右到达目标的策略。distribution constraint 更像“按照数据里的概率来”，容易把低概率但正确的动作压住。对 VLA 来说，这对应一个常见矛盾：既要避免分布外危险动作，又不能把策略锁死在低质量示范的频率上。

Offline RL 也可能从不完整轨迹里拼出更好的策略。

![Offline RL Maze2D trajectory stitching](../assets/images/paper-figures/reinforcement-learning/offline-rl-maze2d-stitching.png){ width="620" }

<small>图源：[Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems](https://arxiv.org/abs/2005.01643)，Figure 6。原论文图意：展示 Maze2D 中利用轨迹组合结构找到更短路径的例子。</small>

!!! note "图解：trajectory stitching 是什么"
    数据里可能没有一条完整最优轨迹，但有“从 1 到 2”的片段和“从 2 到 3”的片段。好的 offline RL 应该能把这些片段组合起来，推出从 1 到 3 的更优行为。这比行为克隆更进一步：BC 只模仿已有路径，offline RL 尝试用 Bellman/value 把片段的长期价值接起来。

## 9. 信用分配和探索

强化学习最难的两个问题：

| 问题 | 简洁解释 |
| --- | --- |
| credit assignment | 一段轨迹最后成功了，究竟是哪几个早期动作贡献最大 |
| exploration | 当前策略没试过的动作，怎么知道有没有更高回报 |

在 LLM/VLM 后训练里，credit assignment 也很明显：完整回答最后得 1 分，但哪几个 token 真的让答案正确？在机器人里更明显：任务成功可能来自前 30 秒里某个微小对齐动作。

!!! warning "常见误区：奖励越密越好"
    密集奖励能降低学习难度，但也可能把模型带偏。比如只奖励“靠近杯子”，机器人可能学会把夹爪贴近杯子却不稳定抓取。好的 reward 要贴近最终任务，同时避免被策略钻空子。

## 10. 和世界模型的接口

世界模型本质上补上了 MDP 里的 transition：

\[
\hat P_\theta(s_{t+1}\mid s_t,a_t)
\]

如果还能预测 reward、done 和 risk：

\[
\hat r_t,\hat d_t,\hat \rho_t = f_\theta(s_t,a_t)
\]

就可以在模型内部做 planning 或 imagined rollout。后面的 PlaNet、Dreamer、RLVR-World 都在围绕这件事展开：用可学习模型近似环境，再把强化学习的长期优化接上去。
