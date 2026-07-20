# 多臂老虎机与探索

## 本章回答什么

当每次只能从多个选项里试一个，而每个选项的平均收益未知时，怎样在“利用当前最优选项”和“探索尚未了解的选项”之间分配机会？本章用同一个固定 seed 的 10 臂任务比较 epsilon-greedy 与 UCB，并把估计、动作选择和指标串成最小实验。

Bandit 没有随动作变化的 state transition，也没有跨多步传播的 delayed return，因此不是完整的序列决策强化学习。它隔离了探索问题，适合作为从 [Agent、环境与训练闭环](rl-agent-environment-loop.md) 走向 MDP 的桥梁。

## 先修知识

需要理解随机变量、均值和采样误差；可回看 [概率分布、随机变量与期望](../prerequisite-math/probability-distributions-and-expectation.md)。代码只依赖 Python 与 NumPy，不依赖 Gym。

先固定两个词：**利用（exploitation）**是选择当前估计价值最大的 arm，**探索（exploration）**是尝试信息不足或暂时看起来不优的 arm。探索可能牺牲当前 reward，却可能改善后续选择所依据的估计。

## 最小任务

环境有 10 个 arm。第 (a) 个 arm 的真实均值为 \(\mu_a\)，每次拉动得到带噪声的 reward。真实最优 arm 为 \(a^*=\arg\max_a\mu_a\)。实验固定一次真实均值，每一步只生成一行包含全部 arm 的潜在噪声，让两种算法在该步面对同一个任务与反事实 reward，处理完后丢弃该行。

每一步的 regret 定义为真实均值差：

\[
\operatorname{regret}_t=\mu_{a^*}-\mu_{a_t}
\]

它衡量这次选择相对最佳 arm 少了多少**期望 reward**，不是本次带噪声 reward 的差。运行比较：

```bash
python files/assets/examples/reinforcement-learning/bandit.py --seed 7 --steps 2000
```

脚本输出累计 reward、平均 regret 和最佳 arm 选择比例。固定 seed 用于复现实验轨迹，不代表单个 seed 足以支持算法优劣结论。

## 核心机制

### 样本均值：只更新被选择的 arm

对 arm (a) 的第 (N(a)) 次观测，样本均值可以增量更新：

\[
Q(a)\leftarrow Q(a)+\frac{R-Q(a)}{N(a)}
\]

对应实现位于 [`bandit.py`](../assets/examples/reinforcement-learning/bandit.py)：

```python
def update_sample_average(values, counts, arm, reward):
    new_count = int(counts[arm]) + 1
    new_value = values[arm] + (reward - values[arm]) / new_count
    values[arm] = new_value
    counts[arm] = new_count
```

实际函数会先验证浮点 values、非负整数 counts、arm 和有限 reward，再计算两个新值并写回，因此非法输入不会只改 count 而留下半次更新。这个估计默认 reward 分布近似 stationary，即同一个 arm 的真实均值不会随时间系统性漂移。若均值持续变化，所有历史样本等权的样本均值会反应迟缓，此时常数步长更新可能更合适，但需要另行分析步长带来的偏差与方差。

### Epsilon-greedy：显式保留随机探索

epsilon-greedy 以概率 \(\epsilon\) 均匀随机选 arm，否则利用当前最大估计。示例约定并列最大时选择最小索引，使 `epsilon=0` 的行为确定；随机探索使用 `numpy.random.default_rng(seed)`，相同 seed 可复现。

固定 epsilon 很容易解释，但它不会因为某个 arm 已被充分估计而自动减少对它的随机探索。epsilon 的合适取值依赖 horizon、噪声和非平稳程度，不能由单条 reward 曲线直接决定。

### UCB：把估计值与不确定性奖励相加

示例使用 UCB1 形式：

\[
a_t=\arg\max_a\left[Q(a)+\sqrt{\frac{2\log t}{N(a)}}\right]
\]

其中 `t` 是从 1 开始的交互步，调用时必须满足 `t == counts.sum() + 1`。实现先按索引尝试每个 `count == 0` 的 arm，再计算分数，因此没有 `log(0)` 或除零；这也是一种乐观处理：未尝试 arm 不会因为初始估计低而永久被忽略。

UCB 的 bonus 会随选择次数增加而缩小，在这个 stationary、有限臂、特定噪声的教学设定中可能比固定 epsilon 更有效率。它不是在所有环境中都优于 epsilon-greedy；非平稳 reward、重尾噪声或不同超参数都可能改变比较结果。

## 训练数据流

Bandit 的数据流没有下一 state：

```text
当前 arm 价值估计 -> 探索规则 -> arm
环境的固定均值与当前噪声 -> reward
arm + reward -> count / sample-average update -> 下一步选择
```

为了让比较尽量公平，[`bandit.py`](../assets/examples/reinforcement-learning/bandit.py) 先固定真实均值，再逐步生成共享的 arm 噪声行；算法只决定该步读取哪个 arm 对应的潜在样本。task RNG 与 epsilon-greedy 的 policy RNG 来自同一 seed 派生的独立子流，因此策略随机调用不会改变环境 reward。两种算法仍会选择不同 arm，所以观察到的 reward 序列不同，这是策略行为的一部分。

进入 [`gridworld.py`](../assets/examples/reinforcement-learning/gridworld.py) 后，数据流多出 `state -> action -> next_state`。其中终止转移的 TD target 不 bootstrap：

```python
def td_target(reward, gamma, next_value, terminated):
    return float(reward) if terminated else float(reward + gamma * next_value)
```

这一步把“哪个 arm 更好”的即时选择，推进到“当前动作怎样改变未来状态价值”的序列问题。

## 常见失败

| 失败 | 表现 | 检查方式 |
| --- | --- | --- |
| 只看累计 reward | 不同噪声实现下曲线排名反复变化 | 同时报告 regret、最佳 arm 比例，并跨多个 seed 汇总 |
| 两算法使用不同任务 | 差异可能来自真实均值或噪声，而非选择规则 | 固定真实均值，并共享预生成潜在 reward 流 |
| 未处理未尝试 arm | UCB 出现除零 warning，或 arm 永远不被选 | 算分前返回第一个 `count == 0` 的 arm |
| 把样本均值当真实值 | 少量幸运样本造成过早利用 | 查看每个 arm 的 count 与估计不确定性 |
| 忽略 stationary 假设 | 环境均值漂移后仍长期相信旧样本 | 分时段检查估计误差，考虑有限窗口或常数步长 |
| 把 Bandit 当完整 RL | 无法解释 transition、return 与终止 bootstrap | 转到 GridWorld，显式记录 state transition |

reward 曲线本身尤其嘈杂：它同时受真实 arm 均值、动作选择和单次采样噪声影响。单次运行中较高的累计 reward 可能来自幸运噪声，不能直接证明探索策略更好。

## 自检与练习

1. 手算某个 arm 依次得到 `2`、`4` 后的 count 与样本均值，解释为什么其他 arm 不变。
2. 令 `epsilon=0` 且两个 arm 的估计并列最大，确认示例选择哪个索引；再说明这种 tie rule 是否改变长期结论。
3. 写出 UCB 在所有 count 都大于零后才计算 bonus 的原因，并解释 bonus 怎样随 count 变化。
4. 用 seed `7`、`8`、`9` 分别运行实验；比较三项指标，而不是只比较累计 reward。
5. 运行下面的 GridWorld 与测试，找出终止转移 target 中被屏蔽的项：

```bash
python files/assets/examples/reinforcement-learning/gridworld.py --seed 7
python -m pytest tests/test_reinforcement_learning_examples.py -q
```
