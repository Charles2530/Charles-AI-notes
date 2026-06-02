# MDP、价值函数与 Bellman

强化学习最核心的问题不是“用哪个算法”，而是：**一个 agent 现在做出的动作，会怎样改变未来，以及怎样把未来的好坏反推回当前动作。** MDP 给出决策问题的语言，value function 给出未来收益的度量，Bellman 方程给出递归学习未来的方法。

如果这三件事没有讲清楚，后面的 PPO、GRPO、Dreamer、Offline RL 和世界模型都会变成一串名词。反过来，只要你能读懂 Bellman 在做什么，就能看出大多数 RL 方法到底是在学 policy、学 value、学 transition，还是在用世界模型替代真实环境试错。

## MDP 在描述什么

MDP，全称 Markov Decision Process，通常写成：

\[
\mathcal{M}=(\mathcal{S}, \mathcal{A}, P, R, \gamma)
\]

这行公式不是为了显得数学化，它是在说一个决策问题需要五个部件：

| 部件 | 问的问题 | 例子 |
| --- | --- | --- |
| \(\mathcal{S}\) | 世界可能处在什么状态 | 游戏画面、机器人位姿、prompt 加生成前缀 |
| \(\mathcal{A}\) | agent 能做什么动作 | 移动摇杆、机械臂控制量、下一个 token |
| \(P\) | 动作会怎样改变状态 | 向右走后角色位置变化，夹爪闭合后物体可能被抓起 |
| \(R\) | 当前结果好不好 | 得分、成功信号、规则奖励、人类偏好分数 |
| \(\gamma\) | 多远的未来还算重要 | 近视控制取小一些，长期任务取大一些 |

把“找钥匙”写成 MDP：状态是你已经检查过哪些地方、当前看到什么、还剩多少时间；动作是看桌面、开抽屉、翻外套；奖励是找到钥匙加分，浪费时间扣分。MDP 的意义是让“现在做什么”可以被分解成“这个动作会把我带到什么状态，以及后面还有多少机会”。

### State 不是 observation

很多读者第一次学 RL 会把 `state` 和 `observation` 混在一起。`observation` 是传感器当前看到的东西，`state` 是足以预测未来的内部描述。摄像头只看到桌面，这是 observation；钥匙可能在外套里、刚才抽屉已经查过、桌面反光导致识别不可靠，这些才更接近 state。

机器人和 VLA 多数时候面对的是 POMDP，也就是部分可观测决策问题。模型只能看到图像、语言、关节角和历史动作的一部分，必须用历史窗口、RNN/RSSM hidden state、Transformer memory 或 latent belief 把缺失信息压进内部状态。世界模型的一个基本任务，就是把 observation 变成更接近 state 的 latent。

## Markov 假设在省什么

MDP 的关键假设是 Markov property：

\[
P(s_{t+1}\mid s_1,a_1,\ldots,s_t,a_t)=P(s_{t+1}\mid s_t,a_t)
\]

这行公式的意思是：如果当前状态 \(s_t\) 已经足够完整，那么预测下一状态 \(s_{t+1}\) 时，不需要再把更早的完整历史都拿出来。左边把从第一步到当前的历史全列出来，右边只保留当前状态和当前动作。Markov 假设不是说真实世界没有历史，而是说“有用历史已经被状态编码进去了”。

这也是世界模型为什么重视 latent state。一个只看当前 RGB 图像的模型，很难判断杯子是否刚被推过、门是否卡住、物体背后是否有遮挡；一个好的 latent dynamics 应该把这些会影响未来的因素保留下来。RSSM、belief state、KV memory 和长上下文，本质上都在帮模型构造更接近 Markov 的状态。

## Policy 是行为规则

策略写成：

\[
\pi_\theta(a_t\mid s_t)
\]

公式读法很直接：在状态 \(s_t\) 下，参数为 \(\theta\) 的策略选择动作 \(a_t\) 的概率是多少。确定性策略可以理解成“同一状态总输出同一动作”；随机策略输出一个分布，允许探索和多样性。

LLM 后训练也能放进这个语言里：状态是 prompt 加已经生成的前缀，动作是下一个 token，trajectory 是完整回答，reward 是规则检查器、reward model 或人类偏好给出的分数。机器人 VLA 里，状态通常包含图像、语言、关节状态和历史动作，动作可能是连续控制向量，也可能是 action token 或 action chunk。

## Reward、return、value 的差别

单步 reward 是当前反馈：

\[
r_t = R(s_t,a_t)
\]

它只回答“这一步好不好”。但 RL 关心的是长期后果，所以需要 return：

\[
G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2}+\cdots
\]

这里 \(G_t\) 是从第 \(t\) 步开始往后看的总收益。当前奖励权重是 1，下一步奖励乘 \(\gamma\)，再下一步乘 \(\gamma^2\)。\(\gamma\) 越接近 1，越远的未来越能影响当前决策。

Value function 把这种未来收益变成“状态的价格”：

\[
V^\pi(s)=\mathbb{E}_\pi[G_t\mid s_t=s]
\]

这行公式在说：如果现在处在状态 \(s\)，之后都按照策略 \(\pi\) 行动，那么平均能拿到多少 return。注意这里有期望 \(\mathbb{E}\)，因为策略可能采样不同动作，环境 transition 也可能随机。

Q-value 则多固定了当前动作：

\[
Q^\pi(s,a)=\mathbb{E}_\pi[G_t\mid s_t=s,a_t=a]
\]

它问的是：如果当前状态是 \(s\)，第一步强制做动作 \(a\)，之后再按策略 \(\pi\) 行动，平均能拿多少。很多控制问题真正需要的是 Q-value，因为 agent 要在多个动作之间比较。

| 概念 | 关心什么 | 常见用途 |
| --- | --- | --- |
| reward \(r_t\) | 当前一步反馈 | 训练 reward model、规则奖励、任务打分 |
| return \(G_t\) | 从当前开始的累计未来反馈 | Monte Carlo policy gradient |
| value \(V(s)\) | 当前状态整体值多少钱 | baseline、critic、bootstrap |
| Q-value \(Q(s,a)\) | 当前动作之后值多少钱 | Q-learning、DQN、action selection |
| advantage \(A(s,a)\) | 当前动作比平均动作好多少 | PPO、Actor-Critic、GRPO 的相对更新 |

## Bellman 方程真正递归了什么

Bellman 方程把“当前价值”拆成“当前奖励 + 下一状态价值”：

\[
V^\pi(s)=\mathbb{E}_{a\sim\pi,\,s'\sim P}
\left[
R(s,a)+\gamma V^\pi(s')
\right]
\]

这个公式可以按三步读：

1. 在状态 \(s\) 下，策略 \(\pi\) 采样动作 \(a\)。
2. 环境 transition \(P\) 根据 \(s,a\) 产生下一状态 \(s'\)。
3. 当前状态的价值等于当前奖励 \(R(s,a)\)，加上下一状态价值 \(V^\pi(s')\) 的折扣。

Bellman 的厉害之处在于它把一个长远问题变成局部递归。你不用等到整局结束才知道当前状态有没有价值，可以用“下一步奖励 + 下一状态价值估计”来更新当前价值。这就是 bootstrap。

Q-value 的 Bellman 形式是：

\[
Q^\pi(s,a)=\mathbb{E}_{s'\sim P}
\left[
R(s,a)+\gamma \mathbb{E}_{a'\sim\pi}Q^\pi(s',a')
\right]
\]

这里当前动作 \(a\) 已经固定，所以外层只对下一状态 \(s'\) 取期望；到了下一状态，再按策略 \(\pi\) 采样后续动作 \(a'\)。这就是“先固定当前动作，再评估未来行为”的表达。

## Discount 是时间预算，不是玄学参数

\(\gamma\) 决定未来奖励对当前决策还有多大影响。一个粗略的 effective horizon 是：

\[
H_{\text{eff}}\approx\frac{1}{1-\gamma}
\]

如果机器人控制频率是 `20Hz`，可以粗略估算：

| \(\gamma\) | 等效步数 | 等效真实时间 |
| ---: | ---: | ---: |
| `0.95` | `20` steps | `1s` |
| `0.99` | `100` steps | `5s` |
| `0.995` | `200` steps | `10s` |

这不是说 \(\gamma\) 越大越好。更长 horizon 可以看到长期后果，但也会放大 transition error、reward misspecification 和 value overestimation。假设世界模型每一步有 `2%` 的关键接触误判概率，那么 100 步内至少出现一次误判的概率是：

\[
1-0.98^{100}\approx86.7\%
\]

所以在世界模型和机器人控制里，\(\gamma\)、rollout horizon、控制频率和模型误差必须一起定。短 horizon 容易贪心，长 horizon 容易被模型漏洞欺骗。

## 从 Bellman 到 Q-learning

Q-learning 用 Bellman 思路构造 TD target：

\[
y_t = r_t+\gamma\max_{a'}Q_{\text{target}}(s_{t+1},a')
\]

这行公式在说：当前训练目标 \(y_t\) 等于这一步真实拿到的奖励 \(r_t\)，加上下一状态里“当前估计最好的动作价值”的折扣值。然后让当前网络预测靠近这个目标：

\[
L=\left(y_t-Q_\theta(s_t,a_t)\right)^2
\]

第一行是“未来账本”，第二行是“把当前估计往账本上靠”。这里最危险的地方也很明显：\(y_t\) 里面有模型自己的估计 \(\max Q\)。如果下一状态价值被高估，当前价值也会被带偏。DQN 用 target network 和 replay buffer 稳定这个过程。

![Deep Q-learning with Experience Replay: Breakout reward](../assets/images/paper-figures/reinforcement-learning/dqn-breakout-average-reward.png){ width="420" }

![Deep Q-learning with Experience Replay: Seaquest reward](../assets/images/paper-figures/reinforcement-learning/dqn-seaquest-average-reward.png){ width="420" }

![Deep Q-learning with Experience Replay: Breakout value](../assets/images/paper-figures/reinforcement-learning/dqn-breakout-average-v.png){ width="420" }

![Deep Q-learning with Experience Replay: Seaquest value](../assets/images/paper-figures/reinforcement-learning/dqn-seaquest-average-v.png){ width="420" }

<small>图源：[Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)，Figure 2。原图表达：Breakout 和 Seaquest 训练中的平均奖励曲线与固定状态集合上的平均最大 Q-value。本站用它说明：reward 曲线反映策略实际玩出的结果，value 曲线反映模型对固定状态未来机会的估计，两者都要看。</small>

## Value 学到的是未来机会

DQN 的 Seaquest 例子很适合解释 value：value 不是当前分数，而是“接下来可能得分的机会”。

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

<small>图源：[Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)，Figure 3。原图表达：Seaquest 中 30 帧片段的预测 value，以及 A/B/C 三个时刻的游戏画面。本站读法：A 点敌人出现，未来得分机会变大；B 点鱼雷即将命中，value 达到高点；C 点机会消耗后 value 回落。</small>

这个图能帮你分清 reward 和 value：reward 是“打中敌人那一下”的即时分数，value 是打中之前模型已经看到的未来机会。世界模型中的 value head、reward head、risk head 也是类似逻辑：不是为了复述画面，而是为了判断未来后果。

## On-policy、off-policy 和 offline 的差别

Bellman 方程让 RL 可以从历史数据里学习，但数据来源会决定风险。

| 类型 | 数据从哪里来 | 主要风险 |
| --- | --- | --- |
| on-policy RL | 当前策略刚采样的数据 | 样本贵，rollout 吞吐低，方差大 |
| off-policy RL | replay buffer 里的历史数据，还会继续收新数据 | 旧数据和当前策略差太远 |
| offline RL | 固定数据集，不再和环境交互 | 分布外动作的 value 容易被高估 |

![Online, off-policy and offline RL](../assets/images/paper-figures/reinforcement-learning/offline-rl-online-offpolicy-offline.png){ width="920" }

<small>图源：[Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems](https://arxiv.org/abs/2005.01643)，Figure 1。原图表达：classic online RL、classic off-policy RL 和 offline RL 的数据收集与训练关系。本站读法：机器人、医疗、自动驾驶和大模型后训练常接近 offline 设置，因为真实试错成本高。</small>

Offline RL 最难的是分布外动作。模型可能会选择数据里几乎没出现过的动作，而这个动作的 Q-value 只是网络外推出来的，可能虚高。保守价值估计、behavior constraint、support constraint 和真实回放校准，都是在解决这个问题。

![Offline RL support versus distribution constraints](../assets/images/paper-figures/reinforcement-learning/offline-rl-support-vs-distribution.png){ width="920" }

<small>图源：[Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems](https://arxiv.org/abs/2005.01643)，Figure 4。原图表达：distribution constraint 和 support constraint 的差别。本站读法：VLA 既要避免完全没见过的危险动作，又不能死板模仿示范数据里频率最高但并非最优的动作。</small>

## 和世界模型的接口

世界模型补的是 MDP 里的 transition：

\[
\hat P_\theta(s_{t+1}\mid s_t,a_t)
\]

这里的帽子 \(\hat{}\) 很重要：它表示“模型预测”，不是真实环境。如果世界模型还预测 reward、done 和 risk：

\[
(\hat s_{t+1},\hat r_t,\hat d_t,\hat \rho_t)=f_\theta(s_t,a_t)
\]

那么 planner 或 actor 就可以在执行前比较候选动作后果。Bellman 语言会变成：用 \(\hat s_{t+1}\) 和 \(\hat r_t\) 估计未来价值，再选择让 return 更高、risk 更低的动作。

这也是世界模型最大的机会和最大风险。机会是把真实试错换成内部 rollout，减少机器人小时数和昂贵评测；风险是策略会利用模型漏洞，在 latent 里拿高分，真实世界失败。因此读任何世界模型论文都要问四个问题：

| 问题 | 为什么关键 |
| --- | --- |
| transition 是否 action-sensitive | 换动作时未来预测是否真的改变 |
| reward/done/risk 是否校准 | 未来账本是否反映真实任务结果 |
| rollout horizon 是否合理 | 长时错误是否被 value 或 planner 放大 |
| 是否有闭环验证 | open-loop prediction 好看不等于策略更好 |

## 外部精读

- [Sutton & Barto: Reinforcement Learning, 2nd Edition](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)：MDP、value、Bellman、dynamic programming 的经典教材。
- [OpenAI Spinning Up: RL Intro](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)：用工程视角解释 policy、trajectory、return 和 RL 算法族。
- [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)：DQN、experience replay、target network 和 value 曲线的原始来源。
- [Offline RL Tutorial, Review, and Perspectives](https://arxiv.org/abs/2005.01643)：理解 offline RL、分布外动作和保守约束的好入口。

## 本页结论

MDP 定义决策问题，policy 定义怎么行动，reward 评价当前反馈，value 把未来机会压成数字，Bellman 方程把长期未来拆成可递归学习的局部目标。世界模型并没有绕开这些概念，它只是把真实 transition 换成可学习的 \(\hat P_\theta\)，让 agent 能在执行前估算动作后果。
