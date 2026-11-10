# 动态规划、Monte Carlo 与 TD，怎样估计同一个价值

六格地图中的策略在状态 0 选择向右，经过 $0\to1\to2\to5$ 到达终点，奖励为 $[-1,-1,0]$。只凭这条经历，应该怎样修改各状态的价值？如果还能查询环境所有动作后果，计算又有什么不同？动态规划、Monte Carlo 与 TD 共享价值定义，真正的差别在于它们拿到未来信息的方式。

环境、终止奖励和动作编号沿用 [Bellman](mdp-value-bellman.md)。本文把策略固定下来讨论预测，再说明怎样进入控制，避免把评估一个策略和寻找最优策略混成同一项计算。

```mermaid
flowchart LR
    M[能查询的模型\nP 和 R] --> DP[DP\n对所有后果求期望]
    E[一条完整轨迹\n直到终点] --> MC[MC\n先算 G_t]
    T[一次转移\nr_t, s_{t+1}] --> TD[TD\n立即 bootstrap]
    DP --> V[更新同一个 V(s)]
    MC --> V
    TD --> V
```

三种方法最后都在改 $V(s)$，区别只在目标从哪里来。DP 读取环境模型，MC 等一整条轨迹结束，TD 在当前转移后就拿后继状态的估计补上未来。把这三个输入放在一起，后面的数字才不会看起来像三套互不相干的公式。

## 有模型时，当前没走过的分支也能算

假设策略在每个非终止状态均匀选四个方向。动态规划可以查询所有动作与后继状态，以概率加权构造目标

\[
V_{k+1}(s)=\sum_a\pi(a\mid s)\sum_{s'}P(s'\mid s,a)
[R(s,a,s')+\gamma V_k(s')].
\]

在状态 2，只有向下进入终点得到 0，其他三个动作得到 −1。初始 $V_0=0$ 时，一次备份为 −0.75。虽然实际采到的轨迹从 2 向下、回报为 0，DP 仍计入策略可能向其他方向走的后果。它做的是完整期望，而非拟合这一条好运路径。

固定策略反复评估得到 $V^\pi$。Policy iteration 在评估之后把每个状态的动作改成当前最好的，再重新评估；value iteration 则直接在备份中取动作最大值。仓库 [GridWorld](../assets/examples/reinforcement-learning/gridworld.py)的 `policy_evaluation` 和 `value_iteration` 分别实现这两种算子，不要只因为都返回数组就认为它们在估计相同对象。

## 没有模型时，先等这条轨迹走完

Monte Carlo 用真实完整回报作目标。折扣为 0.9 时，从末尾倒算

\[
G_2=0,\quad G_1=-1+0.9\times0=-1,\quad
G_0=-1+0.9\times(-1)=-1.9.
\]

假设当前价值在状态 0、1、2、5 上分别为 $[-5,-3,-1,0]$，步长 0.1。MC 更新 $V(s_t)\leftarrow V(s_t)+0.1(G_t-V(s_t))$ 后，它们分别变为 −4.69、−2.8、−0.9、0。目标不含旧 value，因此旧 critic 的错误没有被直接复制到目标里。

这条路径恰好最短，不能把一次 MC 更新当作均匀策略的精确价值。均匀策略还会绕路，需要多条按该策略产生的轨迹平均。MC 的随机性来自整段未来，优点是直接使用观察到的结果，代价是等待完整回报以及较大的采样波动。

如果同一状态在一条轨迹里出现多次，first-visit MC 只用时间上的第一次访问，every-visit 使用每次访问。先倒序算所有 return，再正序挑首次状态，能避免把“倒序遇到的第一次”错当成时间上第一次。

## TD 用当前估计补齐尚未发生的未来

TD(0) 在每次转移后就形成目标

\[
y_t=r_t+\gamma b_tV(s_{t+1}),\qquad
\delta_t=y_t-V(s_t),\qquad
V(s_t)\leftarrow V(s_t)+\alpha\delta_t.
\]

$b_t$ 在真实终止时为零。沿同一条路径，使用更新前的价值 $[-5,-3,-1,0]$，三个目标为 $[-3.7,-1.9,0]$，residual 为 $[1.3,1.1,1]$。步长 0.1 后，值变为 $[-4.87,-2.89,-0.9]$。

这与 MC 的差别来自对未来的信任位置。第一步 TD 还不知道状态 1 之后这次走得很好，先相信旧估计 −3；MC 看到了后续实际 −1。TD 可以立即更新，但把后继估计误差带入当前目标。若后继 V 高估 3，这一项给目标带来 $0.9\times3=2.7$ 的高估。

把两种更新并排运行，更容易看清旧值、目标与结果。

```python
import numpy as np

states = np.array([0, 1, 2])
next_states = np.array([1, 2, 5])
rewards = np.array([-1., -1., 0.])
terminated = np.array([False, False, True])
old = np.array([-5., -3., -1., -2., -1., 0.])
gamma, alpha = 0.9, 0.1
returns = np.zeros(3)
carry = 0.0
for t in reversed(range(3)):
    carry = rewards[t] + gamma * carry
    returns[t] = carry

td_targets = rewards + gamma * (~terminated) * old[next_states]
mc, td = old.copy(), old.copy()
mc[states] += alpha * (returns - old[states])
td[states] += alpha * (td_targets - old[states])
print('MC target', returns, 'updated', mc[states])
print('TD target', td_targets, 'updated', td[states])
assert np.allclose(mc[states], [-4.69, -2.8, -0.9])
assert np.allclose(td[states], [-4.87, -2.89, -0.9])
```

代码特意冻结旧数组，使目标来源一目了然。若在线更新过程中再次访问已经更新的状态，后面的 bootstrap 会使用新值，逐步数值可能不同，必须说明更新顺序。

## 多走一步再 bootstrap，会得到什么

$n$-step target 使用前 $n$ 步真实奖励，再接一个 value

\[
G_t^{(n)}=\sum_{k=0}^{n-1}\gamma^k r_{t+k}+\gamma^nV(s_{t+n}).
\]

从状态 0 开始，取 $n=2$，目标为 $-1-0.9+0.9^2(-1)=-2.71$。它比一步 TD 的 −3.7 更依赖真实轨迹，但仍使用了状态 2 的旧估计。使用完整三步并在终点去掉尾值，就得到 MC 的 −1.9。

增大 $n$ 减少了对早期 value 的依赖，同时引入更多随机奖励，不能保证在所有任务都更准确。GAE 将多个时间尺度的 residual 组合起来，详见 [Actor-Critic 与 PPO](policy-gradient-actor-critic-ppo-grpo.md)，其中继续使用这条轨迹。

真实终止与采样边界仍需区分。若只收集到状态 2，任务尚未结束，应保留它的 bootstrap；若已进入终点 5，尾部价值为零。把外部截断后的未知未来当零，会改变估计目标。完整交互定义见[Agent 与环境](rl-agent-environment-loop.md)。

### `terminated` 和 `truncated` 不能合并成一个 done

在 Gymnasium 风格的环境里，`terminated=True` 表示任务按照环境规则结束，例如到达终点或撞毁；`truncated=True` 表示采样窗口或时间上限到了，未来状态仍可能继续。对 TD 目标，通常只在真实终止时去掉 bootstrap：

```python
bootstrap = 0.0 if terminated else 1.0
target = reward + gamma * bootstrap * next_value
```

假设时间上限在状态 2 截断，`reward=-1`、`gamma=0.9`、`next_value=-1`。正确目标是 `-1.9`；如果把 `truncated` 当成终止，目标会变成 `-1`。这一个差值会沿多步回报传回更早状态，尤其在固定长度 rollout 中影响明显。

数据集若只保存一个 `done` 字段，事后通常无法可靠恢复这两个原因。采集时至少保留 `terminated`、`truncated`、`final_observation` 和时间步；跨环境转换时再写一条明确的映射规则。否则同一套 TD 代码在不同环境上会学习到不同的终止语义。

## 从价值预测走向动作改进

固定策略的 MC/TD 只告诉我们这个策略怎样。要改进动作，需要学习 Q 或利用环境模型比较动作。SARSA 把下一次实际选择的动作价值接进目标，Q-learning 接下一状态最大的动作价值，区别见 [Q-learning 与 DQN](q-learning-and-dqn.md)。

从仓库根目录运行

```bash
python files/assets/examples/reinforcement-learning/gridworld.py --seed 7
```

`policy_value_mean` 是均匀策略的 DP 评估摘要，`optimal_value_start` 来自 value iteration，`q_mean` 是有限采样 Q 表的均值。它们跨不同策略和统计对象，不能直接把最大数字当最好方法。系统教材可对照 [Sutton 与 Barto](http://incompleteideas.net/book/the-book-2nd.html) 第 4、5、6 章，分别讨论动态规划、MC 与 TD。

## 算法和环境依据

六格地图和 `GridWorld` 是本站的可执行教学环境，公式和算法边界对照 [Sutton 与 Barto](http://incompleteideas.net/book/the-book-2nd.html) 第 4、5、6 章。`V_0`、策略、终止奖励和更新顺序都固定以后，DP、MC 与 TD 的差别才有可比性；页面里的 −0.75、−2.71 和 −1.9 是机制算例，不是学习曲线。

`terminated` 与 `truncated` 的接口语义对照 [Gymnasium step API](https://gymnasium.farama.org/api/env/#gymnasium.Env.step)。页面中的 −1.9 与 −1 是用来检查 bootstrap mask 的算例，不代表任何环境的实测回报。

运行示例后，分别记录策略评估、value iteration 和有限采样 Q 表的对象、输入数据和更新顺序。不要把三个数组的最大值放在同一列就叫作算法排名。
