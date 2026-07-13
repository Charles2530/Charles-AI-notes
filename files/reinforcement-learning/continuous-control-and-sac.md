# 连续控制与 SAC

## 本章回答什么

DQN 能对有限 action 取 \(\max_a Q(s,a)\)，但电机力矩是连续值，无法逐个枚举。Soft Actor-Critic（SAC）用 reparameterized squashed Gaussian actor 产生连续动作，以 twin Q、replay buffer、target network 和 maximum entropy objective 进行 off-policy 学习。

本章解释动作坐标、tanh log-prob 修正、温度与更新顺序。短 Pendulum smoke 只证明 critic、actor、temperature 和 Polyak target update 都实际执行，不证明控制策略收敛。

## 先修知识

先读 [Actor-Critic、GAE 与 PPO](policy-gradient-actor-critic-ppo-grpo.md)，理解 actor、critic、advantage 与 terminal bootstrap。DQN 的离散动作限制见 [Q-learning 与 DQN](q-learning-and-dqn.md)。依赖隔离与命令见 [示例 README](../assets/examples/reinforcement-learning/README.md)。

## 最小任务

Pendulum-v1 的 observation 有 3 维，action 是区间 `[-2, 2]` 内的单个连续力矩。SAC actor 先输出 normalized action \((-1,1)\)，再映射到环境区间。`--steps 256` 表示总 interaction 数，不是先收集 256 条数据再开始训练。

replay warm-up 的 batch size 是 64 条 transition。每次 interaction 先写入 replay；仅当 buffer 长度 `<64` 时跳过优化，所以第 64 条 transition 写入后立即执行首个更新，此后每条新 transition 都执行一次。当前 inclusive condition 下，更新数是 \(256-64+1=193\)，也就是 193 次 optimizer update。

运行 [`deep_rl_smoke.py`](../assets/examples/reinforcement-learning/deep_rl_smoke.py)：

```bash
python files/assets/examples/reinforcement-learning/deep_rl_smoke.py sac --seed 7 --steps 256
```

输出列出共享 interaction/episode 摘要、critic loss、actor loss、temperature \(\alpha\) 和独立的 actor/critic 参数变化。Pendulum 在 200 steps 才 time-limit truncation，因此合法的 `--steps 128` 会报告 `completed_episode_return_mean=unavailable`，并把已采奖励单列为 `current_partial_episode_return`，不会把 partial 冒充完整 episode 均值。只要这些量有限且参数确实改变，smoke 契约就满足；负回报或单个 seed 的差异不能用来判断收敛与算法优劣。

## 核心机制

### 为什么 DQN 不能直接枚举连续动作

DQN target 依赖 \(\max_{a'}Q(s',a')\)。当 action 是连续向量时，枚举不存在可行的有限集合；每个 target 内再运行数值优化又昂贵且不稳定。SAC 单独学习 actor \(a\sim\pi_\theta(\cdot\mid s)\)，用它给出高价值且有探索性的动作。

### Reparameterization 与 squashed Gaussian

actor 输出 \(\mu_\theta(s)\) 与 \(\sigma_\theta(s)\)，先采标准噪声，再构造动作：

\[
\epsilon\sim\mathcal N(0,I),\quad
u=\mu_\theta(s)+\sigma_\theta(s)\odot\epsilon,\quad
\tilde a=\tanh(u)
\]

这种 reparameterization 让梯度穿过 \(u\) 与 \(\tilde a\) 回到 actor 参数。它是 squashed Gaussian，而不是先从 Gaussian 采样后简单裁剪；裁剪会在边界产生不可逆的概率质量。

tanh 改变概率密度，因此 logp 必须包含 tanh Jacobian correction：

\[
\log\pi(\tilde a\mid s)=
\log\mathcal N(u;\mu,\sigma)-
\sum_i\log\left(1-\tanh^2(u_i)+\varepsilon\right)
\]

这里第二项补偿变量变换，\(\varepsilon\) 只用于数值稳定。漏掉它会让 entropy 与 temperature 更新基于错误密度。

### Normalized action 与环境 action scaling

若环境上下界是 \(l,h\)，action scaling 为：

\[
a_{\mathrm{env}}=\frac{h-l}{2}\tilde a+\frac{h+l}{2}
\]

critic 和 replay 都使用真实环境动作坐标。smoke 使用 scale-corrected environment-action logprob：除 tanh correction 外，还减去缩放 Jacobian 的常数 \(\sum_i\log((h_i-l_i)/2)\)。把 normalized actor 输出直接传给 Pendulum 会把可用力矩从 `[-2,2]` 错缩成 `[-1,1]`；若 logp 又留在 normalized 坐标，entropy temperature 的数值口径也会和环境动作密度不一致。

### Maximum entropy、twin Q 与 target

SAC 同时偏好高回报和高 entropy。critic 的 entropy-regularized target 是：

\[
y_t=r_t+\gamma(1-d_t)
\left[\min_{i=1,2}Q_{\bar\phi_i}(s_{t+1},a_{t+1})
-\alpha\log\pi_\theta(a_{t+1}\mid s_{t+1})\right]
\]

twin Q 取较小值以减轻过估计；target network 用 Polyak averaging 缓慢跟随 online critics；`terminated` 屏蔽整个 bootstrap，truncation 有有效 next observation 时保留。actor 最小化 \(\alpha\log\pi-\min(Q_1,Q_2)\)。smoke 把一维动作的目标写成命名常量 `target_entropy=-1.0`，temperature \(\alpha\) 据此更新。由于 differential entropy 会随坐标缩放而改变，这个数值必须和上一节的 environment-action logprob 坐标一起解释，而不能脱离密度定义比较。

这些 transition 来自 replay buffer，因而 critic 可以重复使用旧策略采集的数据；这也是 SAC 相对 PPO 的 off-policy 数据边界。

完整 SAC 算法与 maximum entropy 目标的主要来源见 [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290)。

## 训练数据流

```text
actor/environment action -> Pendulum transition -> replay buffer
next actor + target twin Q -> entropy target -> update online twin critics
current actor samples -> min online Q + logp -> update actor
detached logp -> target entropy objective -> update temperature
online critics -> Polyak averaging -> target critics
```

更新顺序要明确：critic target 不回传到 target Q 或 next action；actor update 需要 Q 对 action 的梯度，但不应顺便更新 critic 参数；temperature 使用 detached logp；最后再做 Polyak update。每个 optimizer 的参数集合应互不混淆。

## 常见失败

| 失败 | 结果 | 修正 |
| --- | --- | --- |
| Gaussian action 直接 clip | 边界概率与梯度不一致 | reparameterization 后 tanh squash |
| 漏 tanh Jacobian correction | logp、entropy、temperature 全部偏 | 用变量变换后的密度 |
| 漏 action scaling | 环境只收到部分动作范围 | 从 `[-1,1]` 仿射映射到 action space |
| 用单 Q target | 过估计更易被 actor 利用 | twin Q 并取 target min |
| terminal 仍 bootstrap | 终点之后凭空增加价值 | 只对 true `terminated` 置零 |
| actor backward 更新 critic | 参数职责混乱、额外梯度 | actor 步临时冻结 critic 参数 |
| 256 steps 后比较回报 | 把随机初始化当算法结论 | 长训练、多 seed、独立评估与置信区间 |

## 自检与练习

1. 对 `reward=1, next_min_q=10, next_logp=2, gamma=.9, alpha=.5`，算 terminal 与非 terminal target。
2. Pendulum action 上界为 2，若 normalized action 为 0.5，环境 action 是多少？
3. 说明 tanh Jacobian correction 与 environment action scaling correction 分别来自哪次变量变换。
4. 检查 Polyak 系数两端：\(\tau=1\) 与 \(\tau=0\) 各会发生什么。

## 下一步

SAC 已把经典模型无关的连续控制链路接通。固定日志和分布外动作风险见 [Offline Reinforcement Learning](offline-reinforcement-learning.md)；learned dynamics 与规划见 [Model-based RL 与世界模型](rl-for-world-models.md)；连续 action chunk、安全约束与真实部署见 [VLA、机器人与闭环强化学习](vla-robotics-and-closed-loop-rl.md)。课程地图中仅“综合实践与算法选择”仍处于规划状态，见 [完整学习路线](index.md#完整学习路线)。
