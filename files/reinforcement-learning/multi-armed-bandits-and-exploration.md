# 多臂老虎机，怎样发现早期运气造成的误判

设有三个按钮，真实平均奖励分别为 1.0、1.2、0.8，每次按下还有随机噪声。我们不知道这些均值。第一次各试一次，恰好得到 1.6、0.7、0.9，贪心策略会偏爱第一个按钮，真正最好的第二个却排最后。探索的困难就从这里开始，眼前估计高，既可能是质量好，也可能只是运气好。

这种动作不改变下一轮奖励分布的任务称为多臂老虎机。先去掉状态转移，可以单独研究“多拿收益”与“补充信息”怎样取舍。

## 不断累积的样本怎样修正判断

记按钮 $a$ 的计数为 $N(a)$、样本均值为 $Q(a)$。新奖励 $R$ 到来后，增量更新为

\[
N(a)\leftarrow N(a)+1,\qquad
Q(a)\leftarrow Q(a)+\frac{R-Q(a)}{N(a)}.
\]

它来自把旧样本总和写成 $(N-1)Q$。假设第二个按钮已经试两次，均值仍为 0.7，第三次得到 1.9，更新后为 $0.7+(1.9-0.7)/3=1.1$。新证据能纠正早期低估，但策略必须愿意再试它，估计才会改变。

若永远只按第一个按钮，第二个的 Q 会永远停在早期值。增加贪心动作的样本只能提高对第一个的认识，不能自动揭示第二个更好。这就是探索与普通均值估计之间的联系。

当环境非平稳时，所有历史等权的均值还会越来越迟钝。常数步长 $Q\leftarrow Q+\alpha(R-Q)$ 让旧信息逐渐衰减，代价是持续保留更多噪声。平稳估计和跟踪变化是不同目标。

## Epsilon-greedy 用固定概率购买信息

$\epsilon$-greedy 在大部分时刻选最大 Q，以概率 $\epsilon$ 均匀随机选一个按钮。三个按钮、$\epsilon=0.1$、最大 Q 唯一时，贪心按钮被选的概率是 $0.9+0.1/3\approx0.9333$，其他各约 0.0333。随机分支仍可能选到原本最好的估计。

这个办法让低估按钮获得新样本，却不区分“明显差”和“还没了解”。运行很久以后，它仍把随机预算平均花到所有按钮上。减小 epsilon 会减少这项持续成本，也可能让误判纠正得更慢。

并列时也要有明确规则。`np.argmax` 取第一个最大值，初始全零会偏向索引 0。随机打破并列是另一种行为策略，需要独立随机数来源，不能悄悄改变环境噪声序列。

## UCB 让计数决定探索强度

UCB 给少见动作更高探索 bonus，常见形式为

\[
U_t(a)=Q(a)+\sqrt{\frac{2\log t}{N(a)}}.
\]

假设第 33 次选择前，均值为 `[1.0,0.7,0.8]`，计数为 `[20,2,10]`。三个分数约为 `[1.591,2.570,1.636]`，于是第二个按钮虽然当前均值最低，却因只试过两次得到选择。

随着它被反复采样，bonus 缩小，实际均值成为更重要的依据。未访问按钮的分母为零，需要先各试一次，或把其优先级设为最高。这里 $t=\sum_aN(a)+1$，不要把轮次和计数混用。

经典 UCB1 的遗憾界有奖励范围和独立性等假设。[原始分析](https://doi.org/10.1023/A:1013689704352)讨论有界奖励。下面用高斯噪声演示同样形式的策略，不直接继承有界奖励版本的理论常数，也不把 bonus 当成已校准的真实置信区间。

## 在同一个随机世界里比较两种决策

为避免一种策略多调用随机数后改变环境，我们先生成每个时刻、每个按钮的潜在奖励表。两种策略看到的是同一个世界，但每次只能读取自己选择的按钮。策略随机数另用一个生成器。

```python
import numpy as np

means = np.array([1.0, 1.2, 0.8])
steps = 2000
noise = np.random.default_rng(7).normal(size=(steps, 3))
potential_rewards = means[None, :] + noise

def run(method):
    rng = np.random.default_rng(17)
    q, counts = np.zeros(3), np.zeros(3, dtype=int)
    total_reward = regret = 0.0
    for t in range(1, steps + 1):
        if method == 'ucb' and np.any(counts == 0):
            action = int(np.flatnonzero(counts == 0)[0])
        elif method == 'ucb':
            action = int(np.argmax(q + np.sqrt(2.0 * np.log(t) / counts)))
        elif rng.random() < 0.1:
            action = int(rng.integers(3))
        else:
            action = int(np.argmax(q))
        reward = potential_rewards[t - 1, action]
        counts[action] += 1
        q[action] += (reward - q[action]) / counts[action]
        total_reward += reward
        regret += means.max() - means[action]
    assert counts.sum() == steps and np.isfinite(q).all()
    return q, counts, total_reward, regret

for method in ['epsilon', 'ucb']:
    q, counts, reward, regret = run(method)
    print(method, 'counts', counts, 'means', q.round(3),
          'reward', round(reward, 3), 'regret', round(regret, 3))
```

Q 的更新只使用已选奖励，真实 `means` 只用于构造环境和评测 regret，不能被决策器偷看。使用相同奖励表降低了对比噪声，并不保证两种算法实际观察相同样本，也不能用一个 seed 证明普遍优劣。

## 收到高奖励，不代表这个动作没有探索代价

第二个按钮的真实均值 1.2 最好。按第一个按钮本轮即使幸运得到 2，期望机会损失仍为 $1.2-1.0=0.2$。常用伪遗憾为

\[
\operatorname{Regret}_T=\sum_{t=1}^T(\mu_* - \mu_{a_t}).
\]

这按真实均值比较动作，不能用最佳按钮本轮的一个随机奖励减实际奖励逐项替代。真实业务通常不知道所有均值，因此直接可算遗憾是合成环境的便利。线上系统需要另行设计随机对照或其他评测。

仓库已有[十臂示例](../assets/examples/reinforcement-learning/bandit.py)使用同样的配对噪声原则，可从根目录运行

```bash
python files/assets/examples/reinforcement-learning/bandit.py --seed 7 --steps 2000
```

它输出累计奖励、平均遗憾和最佳臂选择比例。比较多个随机任务时，先算同 seed 内两种方法的差，再汇总差值的均值与波动。时间步彼此依赖，不能把一次运行的两千步当成两千次独立实验。

如果按钮会改变下一轮可选状态，当前均值已不够用。地图里向左可能眼前吃亏却进入更有价值的区域，需要 [MDP 与 Bellman](mdp-value-bellman.md)表示延迟后果。这时探索还会改变未来能收集到什么数据，复杂性超过本文的无状态问题。
