# Policy Gradient 与 REINFORCE

## 本章回答什么

动作来自随机采样，尤其是离散 action 时，梯度从哪里来？Policy Gradient 不对采出的 action id 求导，而是用 score-function estimator 提高高回报动作的 log probability。REINFORCE 是这条思路最直接的 Monte Carlo 实现。

本章负责 trajectory objective、return-to-go、baseline 与探索；critic、GAE 和 PPO 的复用约束留到下一章。

## 先修知识

先读 [Q-learning 与 DQN](q-learning-and-dqn.md)，对照 value-based control 与直接策略优化。概率分布、期望与 log probability 的数学准备见 [概率、分布与期望](../prerequisite-math/probability-distributions-and-expectation.md)。安装和输出契约见 [示例 README](../assets/examples/reinforcement-learning/README.md)。

## 最小任务

CartPole-v1 只有两个离散 action。策略网络输出 logits，`Categorical(logits=...)` 采样 action，并保留所采动作的 `log_prob`。完成若干 trajectory 后，从末尾计算 return-to-go，并执行一次 policy optimizer update。

运行 [`deep_rl_smoke.py`](../assets/examples/reinforcement-learning/deep_rl_smoke.py)：

```bash
python files/assets/examples/reinforcement-learning/deep_rl_smoke.py reinforce --seed 7 --steps 256
```

脚本会完成当前 episode，即实际采样数可能略大于请求值。它验证 sampled action 的 logp 与 return 确实产生有限参数变化；它不声称 256 steps 足以收敛或解决任务。

## 核心机制

### Trajectory objective 与 log-prob trick

策略目标是轨迹回报的期望：

\[
J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}[R(\tau)]
\]

这里 \(\tau\) 是完整 trajectory，\(R(\tau)\) 是其折扣回报。对采样分布使用 log-prob trick，可得到 score-function estimator：

\[
\nabla_\theta J(\theta)=
\mathbb{E}\left[\sum_t G_t\nabla_\theta
\log\pi_\theta(a_t\mid s_t)\right]
\]

其中 \(G_t\) 是从时刻 \(t\) 开始的 return-to-go。高 \(G_t\) 会沿着提高该 sampled action 概率的方向更新；低回报则相反。梯度源是 `Categorical` 对所采 action 的 logp，而不是离散 action 数值本身。

### Return-to-go 与 episode 边界

REINFORCE 从 trajectory 尾部递推：

\[
G_t=r_t+\gamma(1-d_t)G_{t+1}
\]

这里 \(d_t=1\) 表示真正 terminal，防止一个 episode 的 reward 泄漏进前一个。若环境因时间上限 truncation，当前无 critic 的 smoke run 会在采样边界截断 tail return；这是有限 horizon Monte Carlo 估计，可能相对未截断任务有偏，不能谎称已经 bootstrap 未观测未来。

### Baseline 降低方差

可将 \(G_t\) 换成 \(G_t-b(s_t)\)。只要是 action-independent baseline，即 baseline 不依赖当前所采 action，它在期望上不改变 policy gradient，却能降低不同 trajectory 带来的方差。批内 return normalization 也是常见尺度处理，但它依赖 batch 统计，不等同于学得的 value function。

### 探索与 entropy

随机策略本身就是探索来源。策略过早变尖会减少新 action 的机会，因此可监控或奖励 entropy；但 entropy bonus 只调节动作分布，不会修复错误 reward。训练用随机采样，评估可同时报告 deterministic argmax 与 stochastic policy，且必须注明口径。

REINFORCE 的原始来源是 Williams 的论文 [Simple statistical gradient-following algorithms for connectionist reinforcement learning](https://doi.org/10.1007/BF00992696)。这项引用支持 score-function 更新的历史来源，不代表短 smoke 命令复现了论文实验。

## 训练数据流

| 阶段 | 保存的数据 | 作用 |
| --- | --- | --- |
| trajectory | observation、sampled action、reward、terminated/truncated | 确定完整采样边界 |
| logp | \(\log\pi_\theta(a_t\mid s_t)\) | 提供策略参数的梯度路径 |
| return | 每一步 return-to-go \(G_t\) | 给 sampled action 加权 |
| loss | \(-\mathrm{mean}(G_t\,\log p_t)\) | 转成梯度下降目标并更新 policy |

一条 trajectory 必须保留自己的边界；将多个 episode 拼接后直接反向累计 return 会造成跨 terminal leakage。实现中 logp 在采样时保留计算图，return 只是不可导权重。

## 常见失败

| 失败 | 结果 | 修正 |
| --- | --- | --- |
| 对 sampled action id 求导 | 梯度不存在或含义错误 | 对该 action 的 log probability 求导 |
| 使用整局 return 给所有步且不说明 | 信用分配更粗、方差更高 | 使用 return-to-go，并明确估计器 |
| 拼接 episode 后不 reset return | 后一局 reward 泄漏到前一局 | 在 true terminal 处重置递推 |
| baseline 依赖当前 action | 可能改变期望梯度 | 使用 action-independent baseline 或正确 advantage estimator |
| return 尺度很大 | 单批更新剧烈波动 | 检查 reward、discount、normalize 与学习率 |
| 短回报变好即声称收敛 | 高方差估计被当成结论 | 多 seed、独立评估、置信区间 |

## 自检与练习

1. 给定 rewards `[1, 2]` 和 \(\gamma=0.5\)，手算两个 return-to-go。
2. 解释为什么 \(b(s_t,a_t)\) 不能不加修正地充当 baseline，而 \(b(s_t)\) 可以。
3. 比较 `Categorical.sample()` 与 `argmax` 采集的数据；哪一个仍代表当前 stochastic policy？
4. 将 return normalization 移除，比较 loss 尺度，但不要用一次运行判断策略质量。
