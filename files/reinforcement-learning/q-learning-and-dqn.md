# Q-learning 与 DQN

六格地图中的智能体在状态 1 偶尔为了探索向左绕路。它仍能用这次经历学习以后怎样贪心走向终点，前提是分清实际执行的动作与构造目标时假设的动作。知道每个动作的长期价值，就能选择价值最高的动作。Q-learning 直接学习这个动作价值，不必先得到完整环境模型；DQN 用神经网络代替价值表，把方法扩展到图像等大输入。难点在于，网络拟合的目标本身也来自价值估计。

## 同一次转移，可以形成不同目标

记一次交互为 $(s_t,a_t,r_t,s_{t+1})$。SARSA 使用行为策略实际选择的下一动作 $a_{t+1}$，Q-learning 则使用下一状态价值最大的动作

\[
y_t^{\mathrm{SARSA}}=r_t+\gamma b_tQ(s_{t+1},a_{t+1}),
\qquad y_t^Q=r_t+\gamma b_t\max_a Q(s_{t+1},a).
\]

其中 $b_t$ 在真实终止时为零。两者都以 $Q(s_t,a_t)\leftarrow Q(s_t,a_t)+\alpha[y_t-Q(s_t,a_t)]$ 更新当前表项。

假设在状态 0 向右走到 1，奖励为 −1。当前对状态 1 的估计中，向右为 −1、向左为 −3，行为策略为了探索选择向左，$\gamma=0.9$。SARSA 目标是 $-1+0.9(-3)=-3.7$，Q-learning 目标是 $-1+0.9(-1)=-1.9$。前者把下一步确实会探索绕路的后果算进来；后者朝下一步贪心选择的目标更新。

Q-learning 的行为策略可以是 $\epsilon$-greedy，目标策略却是贪心策略，这已经构成 off-policy。Replay 是数据重用机制，并不是 Q-learning 成为 off-policy 的原因。

在有限表格、充分访问状态动作对、合适步长与有界奖励等条件下，可以讨论 Q-learning 的收敛。神经网络版本不自动继承这些保证。

## 先让表格版本在地图上走出来

下面从仓库根目录运行。训练只调用 `step` 获取真实转移，评测再冻结 Q 按贪心策略走一条完整路径。30 步采样边界没有被伪装成环境终止，最后一条转移仍保留 bootstrap。

```python
import sys
import numpy as np
sys.path.insert(0, 'files/assets/examples/reinforcement-learning')
from gridworld import GridWorld, q_learning_update

env = GridWorld()
rng = np.random.default_rng(7)
q = np.zeros((6, 4))
for episode in range(200):
    state = env.reset()
    for step in range(30):
        action = int(rng.integers(4)) if rng.random() < 0.2 else int(q[state].argmax())
        nxt, reward, terminal = env.step(action)
        q_learning_update(q, state, action, reward, nxt, 0.2, 0.9, terminal)
        state = nxt
        if terminal:
            break
state = env.reset()
path, total = [state], 0.0
for t in range(30):
    state, reward, terminal = env.step(int(q[state].argmax()))
    path.append(state)
    total += 0.9**t * reward
    if terminal:
        break
assert terminal, 'evaluation did not reach the goal'
assert abs(total + 1.9) < 1e-8
print('path', path, 'discounted return', total)
```

这里探索动作影响收集了哪些经验，贪心目标影响每条经验怎样更新 Q。两者共同让起点绕路动作逐渐变差，短路径被保留。一次 seed 的成功并不证明任意任务收敛，但可以与地图的精确最优回报 −1.9 对照，确认更新确实形成了可执行策略。

## DQN 拟合什么张量

把六行 Q 表换成网络后，可以把地图状态编码成六维 one-hot，网络对每个输入输出四个 Q。一般地，网络输入一批状态 $[B,d]$，输出每个动作的 Q 值 $[B,|\mathcal A|]$。训练样本只执行过一个动作，因此用动作索引 `gather` 得到 $[B]$ 的 $Q_\theta(s_t,a_t)$，不把同一奖励当成所有动作的标签。

DQN 维护在线网络 $Q_\theta$ 与目标网络 $Q_{\bar\theta}$，目标为

\[
y_t=r_t+\gamma b_t\max_a Q_{\bar\theta}(s_{t+1},a),\qquad
L(\theta)=\mathbb E\left[\ell(Q_\theta(s_t,a_t)-\operatorname{stopgrad}(y_t))\right].
\]

目标分支停止梯度，优化器只让当前预测靠近当次固定的目标。若同时对两侧求导，网络可能通过移动目标降低误差，已经不再是这里的半梯度 TD 更新。$\ell$ 可用平方误差，也可用 Huber loss 限制大残差对梯度的影响；Huber 不会修复错误奖励或错误终止标志。

Replay buffer 把历史转移混合后采样，减少连续轨迹的相关性并重用交互；目标网络在若干步内冻结或缓慢跟随在线网络，减缓目标漂移。函数逼近、自举和 off-policy 的组合仍可能不稳定，两个机制都不构成无条件收敛保证。

历史上应区分 [2013 年 Atari DQN 预印本](https://arxiv.org/abs/1312.5602)与 [2015 年 Nature DQN](https://doi.org/10.1038/nature14236)，后者的算法明确使用独立 target network。不能把两个版本的实现细节合并成同一算法快照。

## 最大化为什么容易过估计

假设两个动作真实价值都为 $3$，估计误差有时为正、有时为负。取最大值会更常挑到误差为正的那个，即使每个动作单独的估计都没有系统偏差。Double DQN 将选动作和评估动作分开

\[
a^*=\arg\max_aQ_\theta(s_{t+1},a),\qquad
 y_t^{\mathrm{Double}}=r_t+\gamma b_t Q_{\bar\theta}(s_{t+1},a^*).
\]

例如在线网络给出 $[5,4]$，目标网络给出 $[2,3]$，奖励为 $1$、折扣为 $0.9$。普通 DQN 用目标网络的最大值 $3$，得到 $3.7$；Double DQN 用在线网络选第一个动作，再读取目标网络的 $2$，得到 $2.8$。它减轻同一个噪声估计既选择又评价造成的偏差，不能保证所有 Q 都是严格下界。

```python
online_next, target_next = [5.0, 4.0], [2.0, 3.0]
best = max(range(len(online_next)), key=online_next.__getitem__)
dqn = 1.0 + 0.9 * max(target_next)
double_dqn = 1.0 + 0.9 * target_next[best]
assert abs(dqn - 3.7) < 1e-12
assert abs(double_dqn - 2.8) < 1e-12
```

原始方法与 Atari 评测见 [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)。这组数值是机制算例，不是论文实验结果。

## 一次短运行能检查哪些事情

仓库[`deep_rl_smoke.py`](../assets/examples/reinforcement-learning/deep_rl_smoke.py)提供 CartPole 的 DQN 更新流程，依赖安装与命令矩阵见[示例说明](../assets/examples/reinforcement-learning/README.md)。

```bash
python files/assets/examples/reinforcement-learning/deep_rl_smoke.py dqn --seed 7 --steps 256
```

代码使用 `epsilon = max(0.1, 1 - step / steps)`。在 `steps=256`、从零编号的步数中，首次碰到下界是 `step=231`。若忘记随机探索、只看网络初始最大输出，会把初始化偏好误当成学到的策略。

Replay 中真实终止会去掉 bootstrap，外部截断保留最后有效后继状态的价值。`current_partial_episode_return` 是尚未结束那一段的回报，不能作为完整 episode 均值。有限 loss、非零参数变化和正确交互数量只能表明更新链路可执行；长时间价值发散、策略退化和多 seed 的性能仍需另外评测。

检查异常时先重算一条样本的奖励、mask、下一状态 Q、目标与当前 Q。若 Q 尺度远高于可获得的折扣回报，检查奖励尺度、真实终止和最大化偏差；若训练误差很低但执行很差，检查 replay 覆盖、观察是否近似 Markov，以及评测环境。固定日志不能主动补充遗漏动作的反馈，这会进入[离线强化学习](offline-reinforcement-learning.md)的问题范围。动作连续到无法枚举时，则可用显式 actor，见 [SAC](continuous-control-and-sac.md)。
