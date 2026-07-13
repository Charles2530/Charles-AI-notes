# 动态规划、Monte Carlo 与 TD

## 本章回答什么

已经有了 Bellman 方程，怎样真正算出 value？本章用同一个 2×3 GridWorld 比较三条路径：Dynamic Programming（DP）枚举已知模型，Monte Carlo（MC）等待完整 episode，Temporal-Difference（TD）从单步样本 bootstrap。

三类方法解决的是同一个预测问题，却依赖不同数据、产生不同偏差与方差。读完后，你应能说明 policy evaluation、policy improvement、policy iteration、value iteration、first-visit Monte Carlo、TD(0) 和 n-step return 之间的依赖关系，并正确处理 terminal mask。

## 先修知识

先读 [MDP、价值函数与 Bellman](mdp-value-bellman.md)。本章直接使用 \(V^\pi\)、\(Q^\pi\)、Bellman expectation、Bellman optimality、return 和 discount，不重新定义它们。

代码实现与运行前提见 [强化学习示例 README](../assets/examples/reinforcement-learning/README.md)。示例只依赖 Python 与 NumPy；完整源码是 [`gridworld.py`](../assets/examples/reinforcement-learning/gridworld.py)。

## 最小任务

`GridWorld` 是确定性的 2×3 地图，状态按行编号：

```text
0  1  2
3  4  5*
```

状态 `0` 是起点，`5` 是 terminal。动作 `0/1/2/3` 是上、右、下、左；撞边界时留在原状态。每个非终止转移 reward 为 `-1`，进入或从终点转移的 reward 为 `0` 且 `terminated=True`。因此 `0 -> 1 -> 2 -> 5` 的 reward 序列是 `-1, -1, 0`，而不是每步都 `-1`。

从仓库根目录运行：

```bash
python files/assets/examples/reinforcement-learning/gridworld.py --seed 7
```

当前示例输出：

```text
policy_value_mean=-4.464
optimal_value_start=-1.900, greedy_policy=[1, 1, 2, 1, 1, -1]
q_mean=-1.166, q_max=0.000
```

前两行分别来自均匀随机 policy evaluation 与 value iteration。`greedy_policy` 中动作 `1` 是右、`2` 是下；terminal 的 `-1` 是“无动作”的 sentinel。第三行是脚本附带的 Q-learning smoke test，不是本章用来证明收敛的实验。

## 核心机制

### 1. Policy evaluation：已知模型下求 \(V^\pi\)

若能查询每个 \((s,a)\) 的 transition 与 reward，可反复应用 Bellman expectation backup：

\[
V_{k+1}(s)=\sum_a\pi(a\mid s)\sum_{s'}P(s'\mid s,a)
\left[R(s,a,s')+\gamma V_k(s')\right]
\]

这行式子在第 \(k\) 轮旧数组 \(V_k\) 上计算整张新数组 \(V_{k+1}\)。[`policy_evaluation`](../assets/examples/reinforcement-learning/gridworld.py) 正是这种同步更新：先创建 `updated = np.zeros_like(values)`，所有 state 都读旧 `values`，一轮结束后才替换。若改成 in-place 更新，仍可能收敛到同一不动点，但每轮数值与收敛速度会不同，不能把两种实验逐轮硬对齐。

对均匀随机 policy，\(\pi(a\mid s)=1/4\)。边界动作会原地不动，也必须计入四个动作的平均；漏掉它会改变被评估的 policy。

### 2. Policy improvement 与 policy iteration

有了 \(V^\pi\)，可用一次前瞻计算每个动作的分数：

\[
q_\pi(s,a)=\sum_{s'}P(s'\mid s,a)
\left[R(s,a,s')+\gamma V^\pi(s')\right]
\]

其中 \(q_\pi(s,a)\) 表示当前先做 \(a\)、之后遵循 \(\pi\) 的价值。policy improvement 令新策略在每个状态选择 \(\arg\max_a q_\pi(s,a)\)。交替执行“policy evaluation 评估当前策略”和“policy improvement 变成 greedy”，直到策略稳定，就是 policy iteration。

这里先完整评估再改进，概念边界最清楚。实际实现可以提前停止评估或交错更新，但必须明确自己使用的是哪种近似，不能把尚未收敛的 \(V\) 当成精确 \(V^\pi\)。

### 3. Value iteration：直接应用 Bellman optimality

Value iteration 把 evaluation 与 improvement 压进一次 backup：

\[
V_{k+1}(s)=\max_a\sum_{s'}P(s'\mid s,a)
\left[R(s,a,s')+\gamma V_k(s')\right]
\]

这行式子使用 Bellman optimality 的 \(\max_a\)，所以目标是 \(V^*\)，不是某个固定 \(V^\pi\)。示例的 [`value_iteration`](../assets/examples/reinforcement-learning/gridworld.py) 也做同步更新，并在最大绝对变化小于 `tolerance` 时停止；若达到 `max_iterations` 仍不满足容差，则抛出 `RuntimeError`，不会返回伪装成收敛的数组。

在本任务中，从 `0` 到 `5` 最少三次转移，reward 是 `-1,-1,0`。取 \(\gamma=0.9\)，最优起点 value 为 \(-1-0.9=-1.9\)，与 CLI 的 `optimal_value_start=-1.900` 一致。

### 4. First-visit Monte Carlo：等 episode 结束

若不知道完整 \(P,R\)，可以按 policy 采样 episode，再从末尾反算每个时刻的 return。first-visit Monte Carlo 只用一个 episode 中某 state 第一次出现处的 \(G_t\) 更新该 state；下一个 episode 再计一次。对上述最短 episode，\(\gamma=0.9\) 时：

```text
states:   0    1    2    5
rewards:    -1   -1    0
returns: -1.9 -1.0  0.0
```

示例的增量样本均值更新为：

```python
def monte_carlo_update(values, counts, state, return_):
    new_count = int(counts[state]) + 1
    new_value = float(values[state] + (return_ - values[state]) / new_count)
    values[state] = new_value
    counts[state] = new_count
```

真实函数还会验证数组、索引、有限 return 与计数溢出。MC target 不含当前 value 估计，因此没有 bootstrap bias；但必须等到 episode 结束，长轨迹中的随机 reward 会带来较高方差。

### 5. TD(0)：一步样本加 bootstrap

TD(0) 每次 transition 都更新 predecessor state \(s_t\)。非终止 target 为 \(y_t=r_t+\gamma V(s_{t+1})\)，terminal target 为 \(y_t=r_t\)；两类转移都执行：

\[
V(s_t)\leftarrow V(s_t)+\alpha
\left[y_t-V(s_t)\right]
\]

其中 \(\alpha\) 是步长，方括号内是 TD error。非终止 target 用当前估计 \(V(s_{t+1})\) 作为学习目标的一部分，这就是 bootstrap；terminal transition 仍更新 \(V(s_t)\)，只是没有终点之后的 bootstrap。目标因此可能有估计偏差，但通常比完整 return 的方差低，也能在线更新。

### 6. n-step return 与 bias-variance

MC 和 TD(0) 之间可以用 n-step return 连起来：

\[
G_t^{(n)}=\sum_{k=0}^{n-1}\gamma^k r_{t+k}+\gamma^n V(s_{t+n})
\]

这行式子先使用 \(n\) 个真实 reward，再从 \(s_{t+n}\) bootstrap。\(n=1\) 就是 TD(0) target；若 \(n\) 步内遇到真正 terminal，则求和在该转移结束且不添加 bootstrap 项。当 \(n\) 延伸到 episode 末尾且不再 bootstrap 时，就得到 Monte Carlo return。较小 \(n\) 通常方差低但更依赖当前 value，bias 可能较大；较大 \(n\) 减少 bootstrap 依赖，却累积更多采样噪声。这就是 n-step target 的 bias-variance 权衡，不是“\(n\) 越大越准确”的单调关系。

### 7. Terminal mask 与 truncation 边界

把结束边界写进 TD error：

\[
\delta_t=r_t+\gamma(1-d_t)V(s_{t+1})-V(s_t)
\]

其中 \(d_t=1\) 只表示**真正 terminal**：任务语义上已结束，后面没有可计入当前 return 的价值。此时 bootstrap 乘零。源码中的实现完全对应这个边界：

```python
def td_target(reward, gamma, next_value, terminated):
    return float(reward) if terminated else float(reward + gamma * next_value)
```

`truncated=True` 常表示时间上限、采样片段长度或外部中断，而非 MDP 到达 terminal。时间上限处是否 bootstrap，要看是否仍有有效 successor state 及其 value：若环境返回的是可继续任务的有效下一状态，通常保留 bootstrap；若截断同时销毁了状态语义或任务本身确实以时限定义失败，则需按该任务定义处理。不要把 `terminated or truncated` 无条件塞给 \(d_t\)。

## 训练数据流

三类估计的输入边界可以压缩为：

```text
已知模型 DP: P,R + policy/value -> 枚举全部转移 -> Bellman backup -> 新 value
first-visit MC: sampled episode -> 从末尾算 return -> 每个 state 首次出现处更新
TD / n-step: sampled transition 或短片段 -> reward + bootstrap value -> 在线更新
```

DP 的“训练数据”是可枚举模型，不需要 rollout；MC 与 TD 都需要 policy 产生样本，因此它们估计的是采样 policy 所覆盖的数据分布。仅用 on-policy trajectory 做 prediction，不会因为把 target 写成 TD 就自动成为 optimal control。

一次样本记录至少保留 `state, action, reward, next_state, terminated, truncated`。先依据结束语义构造 bootstrap mask，再计算 MC/TD target，最后更新 value；若先丢掉两个结束标志的区别，后面无法可靠恢复。

## 常见失败

| 失败 | 表现 | 修正 |
| --- | --- | --- |
| 混淆 in-place 与 synchronous DP | 每轮数组和讲义不同就判错 | 先确认读取旧数组还是立即写回；比较最终不动点与停止条件 |
| MC 未等 episode 结束 | 用不完整后缀冒充完整 return | 等真正结束，或明确改用 n-step / bootstrapped target |
| terminal target leakage | 终点 value 被加到前一状态 target | 对真正 terminal 令 \(d_t=1\) |
| tolerance / limit 错误 | 容差非正、上限耗尽仍返回结果 | 用正有限容差；未收敛达到上限时显式失败 |
| prediction / control 混淆 | 用 on-policy 样本评估 \(\pi\)，却声称得到最优策略 | prediction 固定 policy；control 还需要 improvement 或最优性更新 |
| 所有 truncation 都置零 | 时间片段末端系统性低估 value | 判断是否有有效 successor state/value，再决定 bootstrap |

## 自检与练习

1. **手算 Bellman update**：令初始 \(V_0=0\)、\(\gamma=0.9\)。对状态 `2` 的四个动作分别算一次 target，再写出 value iteration 的 \(V_1(2)\)。注意向下进入 `5` 的 reward 是 `0`。
2. **比较目标**：对 transition `(0, right, -1, 1, False)`，设当前 \(V(1)=-2\)，算 TD(0) target；再解释为什么只看这条 transition 不能得到 MC return。
3. **代码实验**：运行 `gridworld.py`，把 `gamma` 从 `0.9` 改为 `0.5` 传给 `value_iteration`，预测并验证 `optimal_value_start`；不要修改环境 reward 语义。
4. **边界题**：分别为“到达目标”和“采样器每 32 步切片但保留下一状态”写出 \(d_t\)，解释两者为何不同。

## 下一步

权威延伸阅读可用 [Sutton & Barto, Reinforcement Learning: An Introduction, 2nd edition](http://incompleteideas.net/book/the-book-2nd.html)：Chapter 4 是 Dynamic Programming，Chapter 5 是 Monte Carlo Methods，Chapter 6 是 Temporal-Difference Learning，与本章顺序对应。

课程下一站是 Q-learning 与 DQN，但独立页面尚未创建。请从 [完整学习路线](index.md#完整学习路线) 查看当前可读章节与后续状态；若要继续理解 advantage 如何进入策略更新，可先读现有的 [Policy Gradient、Actor-Critic、PPO 与 GRPO](policy-gradient-actor-critic-ppo-grpo.md)。
