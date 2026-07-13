# 价值函数、策略梯度与强化学习目标

这篇回答一个问题：return、value、Q 函数、advantage、policy gradient、PPO、GRPO 和 KL penalty 为什么会反复出现在世界模型、VLA 后训练和大模型推理报告里。

强化学习的数学核心很朴素：模型不是只预测一个标签，而是在一串状态和动作里最大化长期收益。难点也在这里：奖励可能延迟，轨迹有随机性，策略一改，后面的数据分布也会跟着改。

## Return：一条轨迹最后值多少钱

从时间 \(t\) 开始的折扣回报可以写成：

\[
G_t
=
\sum_{k=0}^{\infty}\gamma^k r_{t+k}
\]

这里 \(r_{t+k}\) 是未来第 \(k\) 步奖励，\(\gamma\in[0,1]\) 是折扣因子。读这行式子时，重点不是求和技巧，而是时间偏好：\(\gamma\) 越接近 1，越重视长期后果；\(\gamma\) 越小，越重视眼前奖励。

机器人任务里，如果只在成功时给一个终局 reward，\(G_t\) 会很稀疏，训练信号很弱。世界模型和 VLA 论文常引入 progress、value、risk 或 termination head，就是为了把长期结果变成更密的学习信号。

## Value 和 Q：把未来压成一个数

状态价值函数：

\[
V^\pi(s)
=
\mathbb{E}_{\pi}[G_t\mid s_t=s]
\]

这里条件 \(s_t=s\) 固定当前状态，期望 \(\mathbb E_\pi\) 则对策略后续动作和环境转移产生的所有可能轨迹取平均。

动作价值函数：

\[
Q^\pi(s,a)
=
\mathbb{E}_{\pi}[G_t\mid s_t=s,a_t=a]
\]

\(V^\pi(s)\) 问的是“在状态 \(s\) 继续按策略 \(\pi\) 做，平均能得到多少回报”。\(Q^\pi(s,a)\) 多问一步：“如果第一步指定做动作 \(a\)，后面再按策略走，平均能得到多少回报”。

这两个函数是把长轨迹折叠成局部判断的工具。MuZero 做 planning 时要估计 value，Dreamer 在 imagined trajectory 里训练 actor-critic，机器人世界模型用 value/risk head 排候选动作，都是这个想法的变体。

## Bellman：价值可以递推

Bellman 期望方程写成：

\[
V^\pi(s)
=
\mathbb{E}_{a\sim\pi,\,s'\sim P}
\left[
r(s,a)+\gamma V^\pi(s')
\right]
\]

它说：当前价值等于当前奖励加上下一个状态价值的折扣期望。这个公式之所以重要，是因为它把“长期回报”改写成“一步奖励 + 下一步价值”，从而可以递推学习。

如果把它用于样本更新，就会出现 TD error：

\[
\delta_t
=
r_t+\gamma V(s_{t+1})-V(s_t)
\]

\(\delta_t\) 可以理解为预测误差：当前 value 预测得太低还是太高。actor-critic、Dreamer、PPO 和很多机器人后训练方法都在某种形式上使用这个误差信号。

## 多步回报和 λ-return：在一阶 TD 与整条轨迹之间折中

一步 TD 只看 \(r_t+\gamma V(s_{t+1})\)，Monte Carlo return 则尽量看完整条未来轨迹。很多世界模型和 actor-critic 方法会用 \(n\)-step return 折中：

\[
G_t^{(n)}
=
\sum_{k=0}^{n-1}\gamma^k r_{t+k}
+
\gamma^n V(s_{t+n})
\]

这行公式表示先累积前 \(n\) 步真实或想象奖励，再用第 \(n\) 步的 value 估计剩余未来。

\(\lambda\)-return 把不同长度的 \(n\)-step return 加权混合：

\[
G_t^\lambda
=
(1-\lambda)
\sum_{n=1}^{\infty}
\lambda^{n-1}G_t^{(n)}
\]

\(\lambda\) 越小，越接近短视但低方差的一步 TD；\(\lambda\) 越接近 1，越接近长视但高方差的 Monte Carlo。Dreamer、DreamerV2/V3 和很多 imagined rollout 训练都喜欢这类目标，因为 world model 里的长 rollout 会积累模型误差，完全只看一步又太短视。

## Advantage：这个动作比平均水平好多少

优势函数写成：

\[
A^\pi(s,a)
=
Q^\pi(s,a)-V^\pi(s)
\]

它问的不是“这个动作绝对好不好”，而是“在这个状态下，它比策略通常会做的动作好多少”。这比直接用 return 稳，因为不同状态本来难度不同，直接比较原始回报会混掉状态差异。

kai0 的 Stage Advantage、PPO 的 GAE、GRPO 的 group advantage，都在解决同一个问题：给策略更新一个更稳定的相对进展信号。

## Policy gradient：让更好动作更可能发生

策略目标可以写成：

\[
J(\theta)
=
\mathbb{E}_{\tau\sim\pi_\theta}[G(\tau)]
\]

经典 policy gradient 形式是：

\[
\nabla_\theta J(\theta)
=
\mathbb{E}
\left[
\nabla_\theta\log\pi_\theta(a_t\mid s_t) A_t
\right]
\]

这行式子的直觉很直接：如果 \(A_t>0\)，说明动作比平均好，就提高它的 log probability；如果 \(A_t<0\)，说明动作比平均差，就降低它的 log probability。

但策略不能改得太猛。大模型 RLHF、RLVR 和 VLA 后训练中，策略一旦偏离 reference model 太多，可能语言能力退化、格式崩坏，或在 learned simulator 里钻漏洞。

## PPO 和 GRPO：控制更新步子

PPO 常用概率比值：

\[
r_t(\theta)
=
\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\theta_{\text{old}}}(a_t\mid s_t)}
\]

如果 \(r_t\) 远大于 1，说明新策略把这个动作概率提高太多；远小于 1，说明压低太多。PPO 用 clipping 限制这个比值，避免一次更新把策略推远。

GRPO 的直觉是：对同一个问题采样一组回答，用组内 reward 标准化得到相对优势。一个常见抽象是：

\[
A_i
=
\frac{r_i-\operatorname{mean}(r_{1:G})}
{\operatorname{std}(r_{1:G})+\epsilon}
\]

这让训练不必为每个 token 都训练一个独立 value model，而是用同组候选之间的相对好坏来提供 advantage。DeepSeek-R1、Kimi-K2、LongCat-Video 这类报告里出现 GRPO、reward curve、KL 和 group computation 时，可以先按这个框架读。

![PPO and GRPO](../assets/images/paper-deep-dives/technical-reports/deepseek-r1/figure-grpo.png){ width="760" }

<small>图源：`DeepSeek-R1` 技术报告相关图，本站已本地化于论文专题。原图对比 PPO 和 GRPO 的训练流程。本站在这里用它说明：PPO 依赖 value/GAE 估计优势，GRPO 用同组候选的 reward 做相对优势。</small>

## KL penalty：别离参考策略太远

很多大模型 RL 目标会加 KL 惩罚：

\[
\mathcal{L}
=
\mathcal{L}_{\text{reward}}
-
\beta D_{\mathrm{KL}}
\left(
\pi_\theta \,\|\, \pi_{\text{ref}}
\right)
\]

这里 \(\pi_{\text{ref}}\) 是参考模型，\(\beta\) 控制约束强度。直觉是：reward 鼓励模型朝高分回答走，KL 约束防止它为了分数偏离原有语言分布太远。

KL 的定义和不对称性在 [信息论、熵与分布距离](information-theory-and-divergence.md) 里已经讲过；这里更关心它在策略优化里的角色：作为行为边界，而不是单独的评测指标。

这也是为什么读 RL 论文不能只看 reward 曲线。reward 上升但 KL 暴涨，可能意味着模型在利用 reward 漏洞；KL 很稳但 reward 不动，可能说明约束太强。

## 常见误读

| 误读 | 更稳的理解 |
| --- | --- |
| reward 高就代表策略好 | reward 可能有漏洞，必须看真实任务和分布外行为 |
| value 是最终答案 | value 是对未来回报的估计，误差会影响 planning 和策略更新 |
| \(\lambda\)-return 越长越好 | 长回报偏差小但方差和模型误差更大，需要折中 |
| advantage 只是 reward 的别名 | advantage 是相对当前状态基线的好坏 |
| PPO/GRPO 只是训练技巧 | 它们决定策略每次能被推多远，也决定稳定性边界 |

## 读完以后怎么判断

看到 return、value、advantage、PPO、GRPO 或 KL penalty，先问：奖励来自哪里，长期回报如何估计，优势是相对什么基线，策略更新有没有受约束，评测是否能发现 reward hacking 或 simulator exploitation。这个问题链会支撑 Dreamer、MuZero、DeepSeek-R1、LongCat-Video 和具身世界模型后训练论文。

## 相关阅读与下一步

- 站内下一步：[MDP、价值函数与 Bellman](../reinforcement-learning/mdp-value-bellman.md)。
- 站内下一步：[Policy Gradient、Actor-Critic、PPO 与 GRPO](../reinforcement-learning/policy-gradient-actor-critic-ppo-grpo.md)。
- 论文精读：[Dreamer](../paper-deep-dives/world-models/dreamer.md)。
- 论文精读：[DeepSeek-R1](../paper-deep-dives/technical-reports/deepseek-r1.md)。
