# 目标函数、优化器与学习率日程

模型架构决定系统允许学什么，数据决定系统能看到什么，目标函数、优化器和学习率日程决定系统实际沿着怎样的路径学到什么。同一个模型、同一批数据，仅仅更换 loss 配方和 schedule，就可能得到截然不同的训练轨迹。

这页给训练目标和优化配置一个统一入口。更系统的数据配方见 [Scaling、课程学习与数据混合](scaling-curriculum-and-data-mixture.md)，数值异常排查见 [训练稳定性、数值异常与故障排查](stability-numerics-and-failure-triage.md)。

## 目标函数在定义学习压力

大多数训练问题可以写成：

\[
\min_\theta \mathcal{J}(\theta)
=
\mathbb{E}_{(x,y)\sim p_{\text{data}}}
[\ell(f_\theta(x),y)]
+ \lambda \Omega(\theta)
\]

现代基础模型复杂在于：\(y\) 可能是下一个 token、噪声、偏好排序、奖励信号、图像 patch 或动作；\(\ell\) 可能由多个子目标组成；样本分布是多源、分阶段、随时间变化的。

不同目标本质上在回答不同问题：

| 目标 | 问题 | 常见用途 |
| --- | --- | --- |
| 交叉熵 | 怎样拟合数据分布 | LLM、代码、多模态 token |
| 扩散损失 | 怎样恢复被噪声破坏的信号 | 图像、视频、音频生成 |
| 对比损失 | 怎样塑造表示空间几何 | 检索、对齐、表征学习 |
| 偏好目标 | 怎样偏向更受欢迎的输出 | RLHF、DPO、对齐 |
| 强化学习目标 | 怎样最大化长期回报 | agent、机器人、工具使用 |
| 多任务目标 | 怎样同时保留多个能力 | VLM、VLA、世界模型 |

所以“哪种 loss 更好”没有统一答案，必须看它给模型施加了什么学习压力。

## 常见目标族

自回归语言模型最经典的目标是 next-token prediction：

\[
\mathcal{L}_{\text{CE}} =
-\sum_{t=1}^{T}\log p_\theta(x_t\mid x_{<t})
\]

它简单、稳定、可扩展，但不直接优化有用性、偏好和业务风险。

扩散模型常用噪声、\(x_0\) 或 velocity 预测目标，例如：

\[
\mathcal{L}_{\text{diff}}
=
\mathbb{E}_{x_0,\epsilon,t}
\left[\|\epsilon-\epsilon_\theta(x_t,t,c)\|_2^2\right]
\]

不同参数化会影响梯度尺度、采样器兼容性和训练稳定性。

对比学习常用 InfoNCE：

\[
\mathcal{L}_{\text{InfoNCE}}
=
-\log
\frac{\exp(\mathrm{sim}(z,z^+)/\tau)}
{\exp(\mathrm{sim}(z,z^+)/\tau)+\sum_i\exp(\mathrm{sim}(z,z_i^-)/\tau)}
\]

它优化的是表示几何，而不是直接分类。

偏好学习如 DPO 会比较优选回答和劣选回答，推动策略相对参考模型向偏好方向移动。这类目标对参考模型、偏好数据、KL 约束和学习率非常敏感。

## 多目标训练

实际系统很少只优化一个损失。多模态、具身和世界模型常写成：

\[
\mathcal{L}
=
\lambda_1\mathcal{L}_{\text{task}}
+\lambda_2\mathcal{L}_{\text{aux}}
+\lambda_3\mathcal{L}_{\text{align}}
+\lambda_4\mathcal{L}_{\text{reg}}
\]

真正难点不是写公式，而是：

1. 权重如何设；
2. 不同 loss scale 是否可比；
3. 哪些目标早期强、后期弱；
4. 不同数据源上的目标是否要分开归一化；
5. 一个目标提升时，另一个目标是否退化。

多目标训练常常要和 curriculum、分阶段训练、冻结/解冻策略一起设计。否则辅助目标很容易压制主目标，或者后训练目标把预训练能力冲掉。

## 优化器的角色

一阶优化器可以抽象为：

\[
\theta_{t+1}=\theta_t-\eta_t u_t
\]

其中 \(u_t\) 是基于梯度和历史统计构成的更新方向，\(\eta_t\) 是学习率。

| 优化器 | 特点 | 常见注意点 |
| --- | --- | --- |
| SGD | 简单、经典 | 大模型中通常不如自适应方法方便 |
| Momentum SGD | 平滑更新方向 | 视觉和表征任务仍常见 |
| Adam | 一阶/二阶矩自适应 | 状态内存高，超参经验成熟 |
| AdamW | 解耦 weight decay | Transformer 默认选择 |
| Adafactor | 节省优化器状态 | 行为和调参经验不同于 AdamW |
| Lion 等变体 | 试图减少状态或改变更新几何 | 需要充分大规模证据 |

工业大模型训练里，AdamW 仍常作为默认起点，不是因为它理论上永远最优，而是因为稳定、可预期、经验最丰富。

## 学习率日程

学习率日程几乎和优化器同样重要。常见形式是 warmup + decay：

\[
\eta_t =
\begin{cases}
\eta_{\max}\frac{t}{T_w}, & t\le T_w\\
\eta_{\max}f(t), & t>T_w
\end{cases}
\]

Warmup 的作用不只是平滑优化，它还能让混合精度、梯度统计、dataloader 和分布式通信进入稳定区间。Decay 可以是 cosine、linear、constant with decay 或多阶段 schedule。

学习率设计要和以下因素一起看：

1. effective batch；
2. 训练阶段和数据切换；
3. loss 组成和权重；
4. 梯度裁剪；
5. 混合精度和 loss scaling；
6. checkpoint 恢复后的 global step。

很多“模型不稳定”其实是 schedule 和阶段切换不匹配。

## 正则、裁剪与稳定项

常见稳定手段包括：

1. weight decay；
2. gradient clipping；
3. dropout / stochastic depth；
4. label smoothing；
5. KL / reference constraint；
6. auxiliary loss；
7. norm、logit、activation 相关约束。

这些手段都不是越多越好。它们会改变优化几何和能力分布。比如过强 KL 会限制偏好学习收益，过强 clipping 会掩盖真正的数值异常，过强 auxiliary loss 会让模型偏向容易优化的中间任务。

使用这些项时要记录触发频率、实际 loss scale、对主指标的影响，以及是否在特定数据桶上造成退化。

## 配置验收清单

目标函数、优化器和日程上线前至少要回答：

1. 主 loss 和辅助 loss 的权重如何确定；
2. loss scale 是否随数据源或阶段变化；
3. 学习率、warmup、decay 和 effective batch 是否匹配；
4. optimizer state、scheduler state 和 scaler 是否能完整 checkpoint；
5. gradient clipping、weight decay、KL 等稳定项是否有监控；
6. 数据切换点、loss 切换点和 LR 切换点是否对齐；
7. 是否按能力桶观察收益和退化；
8. 恢复训练后曲线是否连续。

训练目标不是静态公式，优化器也不是默认选项。它们共同定义训练轨迹。成熟训练系统会把 loss 配方、数据阶段、学习率、batch、数值稳定和 checkpoint 状态作为同一个配置资产管理。
