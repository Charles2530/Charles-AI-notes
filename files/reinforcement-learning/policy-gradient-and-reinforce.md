# Policy Gradient 与 REINFORCE

仍然使用前面六个格子的地图。智能体从左上角出发，目标是右下角；普通移动奖励为 $-1$，进入终点奖励为 $0$。我们已经能用价值表判断哪个动作好，现在换一个问题：只让策略与环境交互，能否直接把“上、右、下、左”的概率调好，而不先求出每个动作的 Q 值？

```text
0  1  2
3  4  5*
```

这正是策略梯度的起点。本文一直使用这张地图，把动作概率、轨迹概率、回报权重和一次实际训练连起来。环境实现是仓库已有的 [GridWorld](../assets/examples/reinforcement-learning/gridworld.py)，奖励与动作编号沿用 [Bellman 正文](mdp-value-bellman.md)。

## 先决定策略究竟有什么参数

每个非终止格子保存四个 logits，记作 $\theta_{s,a}$，经过 softmax 得到动作概率

\[
\pi_\theta(a\mid s)=\frac{\exp\theta_{s,a}}{\sum_j\exp\theta_{s,j}}.
\]

参数表形状是 $[6,4]$，其中终点行不会被用于行动。初始 logits 全为零，所以每个方向概率都是 $0.25$。策略按照这个分布采样动作，环境按固定规则移动。撞到边界会原地停留，但仍损失一步奖励。

在状态 $0$ 采到“向右”时，我们需要知道怎样调整这一行 logits。对 log-softmax 求导可得

\[
\frac{\partial\log\pi_\theta(a\mid s)}{\partial\theta_{s,j}}
=\mathbb 1[j=a]-\pi_\theta(j\mid s).
\]

因此初始策略中，“向右”这个已采动作的 score 向量为 $[-0.25,0.75,-0.25,-0.25]$。沿这个方向增加参数，会提高向右概率，同时降低其他动作概率。剩下的问题是，这个方向应该乘多大权重，以及何时应该反过来走。

## 环境不可微，梯度从哪里来

设一条轨迹为 $\tau=(s_0,a_0,r_0,\ldots,s_T)$，奖励 $r_t$ 在执行 $a_t$ 后收到。目标是最大化从起点计算的折扣回报

\[
J(\theta)=\mathbb E_{\tau\sim p_\theta}\left[R(\tau)\right],\qquad
R(\tau)=\sum_{t=0}^{T-1}\gamma^t r_t.
\]

地图中的状态跳转是离散操作，不能直接对“走到右边格子”求导。但我们可以对走出这条路径的概率求导。轨迹概率分解为

\[
p_\theta(\tau)=p(s_0)\prod_{t=0}^{T-1}\pi_\theta(a_t\mid s_t)P(s_{t+1}\mid s_t,a_t).
\]

初始状态和环境规则不含策略参数，所以对 log probability 求导时，只有动作概率留下

\[
\nabla_\theta\log p_\theta(\tau)=\sum_t\nabla_\theta\log\pi_\theta(a_t\mid s_t).
\]

再使用 $\nabla p=p\nabla\log p$，把目标梯度变成可以采样估计的期望

\[
\begin{aligned}
\nabla_\theta J
&=\sum_\tau\nabla_\theta p_\theta(\tau)R(\tau)\\
&=\mathbb E_{\tau\sim p_\theta}\left[R(\tau)\sum_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)\right].
\end{aligned}
\]

这里用离散轨迹求和表达，连续变量时对应积分，并需要允许交换求导与期望的条件。实际操作就是先按当前策略采若干完整轨迹，计算每条轨迹回报，再把各动作的 score 乘回报后累加。无需对环境反向传播，也无需知道转移概率表。

## 为什么一个动作只承担它之后的奖励

考虑一条绕路轨迹 $0\to1\to0\to1\to2\to5$。在第二次到达状态 $1$ 时，之前的两次绕路成本已经发生。它们不能由接下来的动作改变，却会让整条回报更差。如果把全程回报乘到每个动作上，就把这段无关的噪声也交给后续动作承担。

数学上可以去掉这些过去奖励，因为在给定当前状态时，$\mathbb E_a[\nabla\log\pi(a\mid s)]=\nabla\sum_a\pi(a\mid s)=0$。得到 reward-to-go 形式

\[
G_t=\sum_{k=t}^{T-1}\gamma^{k-t}r_k,\qquad
\nabla_\theta J=\mathbb E\left[\sum_t\gamma^tG_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)\right].
\]

外层 $\gamma^t$ 把当前位置的回报换算到起点时刻，内层 $G_t$ 从当前位置开始折扣。本文代码保留两者。若使用 $\gamma=1$，或把时间权重吸收到折扣状态分布，公式外观会不同；直接对所有采样时间步均匀平均则是另一种常见实现约定，不能在推导中悄悄删掉权重。

最短路径 $0\to1\to2\to5$ 的奖励为 $[-1,-1,0]$，$\gamma=0.9$ 时三个 return 为 $[-1.9,-1,0]$。它们可以从后向前算，最后一步为零，上一项等于当前奖励加 $0.9$ 倍后一项。这个倒序扫描正是后面代码里的 `carry`。

## 回报全为负，策略还能学会走最短路吗

可以，但单条样本的方向未必符合“好路径就增加概率”的直觉。对最短路径的第一个向右动作，原始回报 $-1.9$ 乘 score，得到 $[0.475,-1.425,0.475,0.475]$，单独做这次更新会降低向右概率。更差、更长的路径也会受到惩罚，策略改进来自整个采样期望，而不是每条轨迹都能独立识别好坏。

加入 baseline 后，相对表现更容易看清。初始均匀策略在状态 $0$ 的真实价值约为 $-6.7169$；最短路径的优势为 $-1.9-(-6.7169)=4.8169$，现在它会提高向右概率。若一条很长路径得到 $-9$，同一 baseline 下优势为负，方向相反。

Baseline 不必精确才保持期望梯度，只要在给定状态时不依赖本次动作，就有

\[
\mathbb E_a[b(s)\nabla\log\pi(a\mid s)]=b(s)\nabla1=0.
\]

代码用以前批次的回报训练一个状态 baseline，并在本批 actor 更新期间固定。这样它估计“平常从这里出发会得到多少”，当前轨迹只需提供偏离预期的部分。对正回报同样如此，回报 $2$ 若不减 baseline，仍是增加已采动作概率的信号；只有相对 baseline 为负时才得到负权重。

## 把这些量接成一个能训练的程序

下面代码从仓库根目录运行，只需 NumPy。它直接求 softmax 的解析梯度，以便看清每个更新量；把参数表替换成神经网络时，自动微分会承担同样的链式求导。

每批先固定策略采 32 个完整回合，随后统一更新一次 actor。Baseline 在 actor 更新之后才吸收本批回报，避免本次动作反过来改变自己的 baseline。达到保护性上限仍没结束会报错，不会把未观察的未来当成零。

```python
import sys
import numpy as np
sys.path.insert(0, 'files/assets/examples/reinforcement-learning')
from gridworld import GridWorld, policy_evaluation

rng = np.random.default_rng(7)
gamma = 0.9
theta = np.zeros((6, 4))
baseline = np.zeros(6)

def probabilities(logits):
    weights = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return weights / weights.sum(axis=-1, keepdims=True)

def collect_episode():
    env = GridWorld()
    state = env.reset()
    trajectory = []
    for _ in range(1000):
        p = probabilities(theta[state])
        action = int(rng.choice(4, p=p))
        next_state, reward, terminated = env.step(action)
        trajectory.append((state, action, reward, p))
        state = next_state
        if terminated:
            return trajectory
    raise RuntimeError('episode incomplete; do not treat missing future as zero')

start = policy_evaluation(GridWorld(), probabilities(theta))[0]
for iteration in range(60):
    batch = [collect_episode() for _ in range(32)]
    gradient = np.zeros_like(theta)
    return_sums, visits = np.zeros(6), np.zeros(6)
    for trajectory in batch:
        returns = np.zeros(len(trajectory))
        carry = 0.0
        for t in reversed(range(len(trajectory))):
            carry = trajectory[t][2] + gamma * carry
            returns[t] = carry
        for t, (state, action, _, p) in enumerate(trajectory):
            score = -p.copy()
            score[action] += 1.0
            advantage = returns[t] - baseline[state]
            gradient[state] += gamma**t * advantage * score
            return_sums[state] += returns[t]
            visits[state] += 1
    theta += 0.3 * gradient / len(batch)
    seen = visits > 0
    baseline[seen] += 0.2 * (return_sums[seen]/visits[seen] - baseline[seen])
end = policy_evaluation(GridWorld(), probabilities(theta))[0]
print(f'exact_start_value: {start:.6f} -> {end:.6f}')
print('start_action_probabilities:', probabilities(theta[0]).round(3))
assert end > start
```

固定 seed 下输出为

```text
exact_start_value: -6.716893 -> -2.084056
start_action_probabilities: [0.027 0.387 0.563 0.023]
```

这是本站 GridWorld 上真实执行的表格策略训练，不是论文成绩。向右和向下都能走到最短路径，因此两者都获得较大概率是合理的；向上和向左在起点撞边界，概率降了下来。最优起点价值是 $-1.9$，当前仍为 $-2.084056$，说明随机策略仍有绕路概率。

`policy_evaluation` 只在训练前后调用，用小地图已知转移计算精确评测值。Actor 和 baseline 的训练只读取采样轨迹，没有调用这份环境模型。这个区分使我们既能演示 model-free 学习，也能用精确解检查它是否改善。

## 换成网络后，什么还一样，什么容易误读

对神经策略，常写最小化损失 $L=-\sum_t\gamma^t\operatorname{stopgrad}(G_t-b_t)\log\pi_\theta(a_t\mid s_t)$，负号把梯度下降转成回报梯度上升。停止梯度的是回报权重，策略 logprob 必须保留计算图。连续动作的各维 log density 要按联合分布规则求和，才能与每条样本的标量 advantage 对齐。

这个 loss 是获得正确局部梯度的工具，不是期望回报的数值估计。采样分布随策略改变，对同一批旧动作反复降低 loss，可能让策略过度偏爱偶然成功的样本；loss 下降不能替代重新采样后的 return。REINFORCE 的基本循环因此是采新数据、做更新、再采新数据。

同批回报均值包含自身样本，不自动满足 action-independent baseline 条件。在独立同状态样本的简单情形中，它会给期望梯度乘上 $1-1/N$；再除以样本标准差还会引入随机缩放。常见归一化可以有工程用途，但需要与理论无偏 baseline 分开解释。

下一篇 [Actor-Critic、GAE 与 PPO](policy-gradient-actor-critic-ppo-grpo.md)继续使用这张地图，解决“完整回报太慢、估计太噪、旧数据想多用几次”三个具体困难。已有 [CartPole smoke](../assets/examples/reinforcement-learning/deep_rl_smoke.py)可按[运行说明](../assets/examples/reinforcement-learning/README.md)执行；其 REINFORCE 使用归一化相对回报，和本文保留外层折扣的解析示例不是逐项相同实现。

```bash
python files/assets/examples/reinforcement-learning/deep_rl_smoke.py reinforce --seed 7 --steps 256
```

推导与实现的组织参考了 [Spinning Up 策略优化正文](https://github.com/openai/spinningup/blob/038665d62d569055401d91856abb287263096178/docs/spinningup/rl_intro3.rst)及同 commit 的 [VPG 实现](https://github.com/openai/spinningup/blob/038665d62d569055401d91856abb287263096178/spinup/algos/pytorch/vpg/vpg.py)。原始 REINFORCE 论文见 [Williams, 1992](https://doi.org/10.1007/BF00992696)。
