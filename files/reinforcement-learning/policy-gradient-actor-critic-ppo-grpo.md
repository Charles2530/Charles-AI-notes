# Actor-Critic、GAE 与 PPO

## 本章回答什么

REINFORCE 的 Monte Carlo return 方差很高，怎样用 critic bootstrap，又怎样在同一批 rollout 上做几次更新而不让新策略离采样策略太远？Actor-Critic 分开决策与估值，GAE 在 bias-variance 间插值，PPO 用 old logprob、probability ratio 与 clipped surrogate 限制更新。

本页保留稳定 URL，但职责只覆盖经典 Actor-Critic、GAE 与 PPO。GRPO、LLM 后训练和系统成本以后续课程桥接，不在这里展开。

## 先修知识

先读 [Policy Gradient 与 REINFORCE](policy-gradient-and-reinforce.md)，确保理解 trajectory、logp、return-to-go 与 action-independent baseline。运行环境与所有命令见 [示例 README](../assets/examples/reinforcement-learning/README.md)。

## 最小任务

PPO smoke 仍使用 CartPole-v1。actor 输出离散 action 分布，critic 输出 \(V(s)\)。rollout 保存 old logp、old value、reward 与 `terminated`，按 episode 或采样片段计算 GAE，再对一个 epoch 的 minibatch 执行联合更新。

运行 [`deep_rl_smoke.py`](../assets/examples/reinforcement-learning/deep_rl_smoke.py)：

```bash
python files/assets/examples/reinforcement-learning/deep_rl_smoke.py reinforce --seed 7 --steps 256
python files/assets/examples/reinforcement-learning/deep_rl_smoke.py ppo --seed 7 --steps 256
```

前者提供无 critic 的对照。PPO 输出共享 interaction/episode 摘要、loss、`sampled_approximate_kl`，以及独立的 `actor_parameter_delta` 和 `critic_parameter_delta`；两个非零 delta 分别证明 actor 与 critic 被更新。两个命令都只验证有限计算链和真实 optimizer update，不是学习曲线或收敛证明。

## 核心机制

### Actor、critic、target 与 advantage

actor \(\pi_\theta(a\mid s)\) 负责采样动作；critic \(V_\phi(s)\) 估计当前状态按该策略继续的期望回报。一步 TD residual 是：

\[
\delta_t=r_t+\gamma(1-d_t)V_\phi(s_{t+1})-V_\phi(s_t)
\]

其中 \(d_t=1\) 只对应 `terminated`。critic 拟合 return 或 value target；actor 使用 advantage 判断 sampled action 比当前状态平均预期好多少。critic 的估计误差会进入 actor，因此 value loss 下降不自动意味着策略变好。

### GAE 与结束语义

Generalized Advantage Estimation（GAE）反向递推：

\[
\hat A_t=\delta_t+\gamma\lambda(1-d_t)\hat A_{t+1}
\]

\(\lambda=0\) 更接近一步 TD，bootstrap 依赖强、方差较低；\(\lambda\) 接近 1 时使用更长 reward 后缀，方差通常更高。true terminal 同时屏蔽 delta 的 next value 和 advantage recurrence。

truncation 不是 terminal：如果时间上限仍返回有效 final observation，应从该状态 bootstrap；但环境控制仍会 reset，GAE recurrence 不能穿过 reset 泄漏到下一 episode。smoke 实现把每个 episode 或 rollout cutoff 分段计算，并为非 terminal 片段保存独立 bootstrap value。

### Old policy、ratio 与 clipped surrogate

rollout 时保存并 detach old logprob。更新时计算 current logp 与 probability ratio：

\[
r_t(\theta)=\exp\left(\log\pi_\theta(a_t\mid s_t)-
\log\pi_{\mathrm{old}}(a_t\mid s_t)\right)
\]

PPO 的 clipped surrogate objective 为：

\[
L^{\mathrm{CLIP}}=\mathbb{E}_t\left[
\min\left(r_t\hat A_t,
\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)\hat A_t\right)
\right]
\]

其中 \(\epsilon\) 表示允许 probability ratio 偏离 1 的 clip 区间宽度。positive advantage 的 ratio 超过 \(1+\epsilon\) 后不再增加 surrogate；negative advantage 在 ratio 低于 \(1-\epsilon\) 时也采用更保守项。代码以负号把最大化 surrogate 转成最小化 loss。

![PPO clipped surrogate objective 原论文图](../assets/images/paper-figures/training/ppo-clipped-surrogate.png){ width="760" }

<small>图源：[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)，Figure 1。原图展示 clipped surrogate 随 probability ratio 的保守下界；它解释 clip 的局部形状，不保证任意多轮更新仍留在可信区域。</small>

### Entropy、KL 与 minibatch 复用

entropy bonus 防止离散策略过早变成确定性。PPO 对一批数据做有限 epoch/minibatch 复用，所以是 approximately on-policy：epoch 太多、学习率太高或 KL 激增后，old-policy importance ratio 不再可信。

smoke 在所有 minibatch optimizer step 完成后，用 final actor 对完整 rollout 的已采 action 重新计算 logp，再报告 `old_logp - final_logp` 的样本均值。这个 sampled approximate KL 是 rollout 上的 Monte Carlo 诊断量，可能为负；它不是对完整 action distribution 求和得到的、保证非负的精确 KL divergence。若将每个 minibatch 的临时值覆盖到同一个变量，最后只会报告末个 minibatch，不能代表最终策略对完整 rollout 的偏移。

old policy 是这批 rollout 的行为策略快照，用来构造 ratio。reference policy 则常用于限制模型长期偏移，例如 LLM RL 的 KL penalty；它可以跨很多轮固定。两者用途不同，不能把 reference logprob 填到 PPO old logprob 的分母。

## 训练数据流

```text
actor + environment -> rollout(state, action, reward, old logprob, old value)
segment boundary + bootstrap value + terminated mask -> GAE -> advantage/return
minibatch -> current logprob/value -> clipped actor loss + value loss + entropy
all optimizer updates -> final actor on complete rollout -> sampled approximate KL
then discard or refresh rollout
```

rollout 与优化必须分阶段。old logprob 在采样后固定，advantage 也按该批数据固定；若每个 minibatch 重新用已更新 critic 生成 advantage，训练目标会在一个 epoch 内漂移。

## 常见失败

| 失败 | 表现 | 修正 |
| --- | --- | --- |
| old logprob 未 detach | 旧策略分支也收到梯度 | rollout 时保存数值并 detach |
| terminal 与 truncation 共用 mask | 时间上限 value 偏低 | terminal 屏蔽；有效 truncation final state 保留 bootstrap |
| GAE 穿过 environment reset | 下一 episode reward 污染上一段 | 分段递推，reset 同时结束 recurrence |
| minibatch epoch 过多 | ratio、KL 快速偏离 1/0 | 限制复用，监控 clip fraction 与 KL |
| 只调 clip epsilon | 仍出现大更新 | 同时检查 advantage scale、学习率、梯度范数和 epoch |
| critic loss 很低就宣布学会 | actor 可能没有改善 | 独立评估 policy return 与多 seed |

## 自检与练习

1. 设 advantage 为 `2`、ratio 为 `1.4`、\(\epsilon=0.2\)，算负 clipped loss；再算 advantage 为 `-2`、ratio 为 `0.6`。
2. 给定一次 true terminal 和一次 time-limit truncation，分别写出 value bootstrap 与 GAE recurrence 的处理。
3. 解释 old policy 与 reference policy 为何可能参数相同但语义仍不同。
4. 增加 PPO minibatch epoch，观察完整 rollout 上的 sampled approximate KL，并解释它为什么可能为负、为什么不等于精确 KL；这只是更新幅度诊断，不是收益改善证明。

## 下一步

离散 CartPole 可以枚举 action logits；连续控制需要可重参数化策略与 off-policy critic，见 [连续控制与 SAC](continuous-control-and-sac.md)。GRPO、RLHF、RLVR 与 DPO 的边界见 [LLM 强化学习：RLHF、RLVR 与 GRPO](llm-rlhf-rlvr-and-grpo.md)；世界模型路线见 [Model-based RL 与世界模型](rl-for-world-models.md)，工程路线见 [verl 训练流程](verl-code-flow.md)。
