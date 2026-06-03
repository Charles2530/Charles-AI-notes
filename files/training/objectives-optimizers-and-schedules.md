# 目标函数、优化器与学习率日程：训练轨迹由什么决定

同一个模型、同一批数据，只要换一个 loss 配方、优化器或学习率日程，训练轨迹就可能完全不同。目标函数定义“模型被什么惩罚”；优化器决定“梯度怎样变成参数更新”；学习率日程决定“每一步走多大、什么时候收力”。

这篇只回答一个问题：**为什么训练不是把 loss 写出来然后交给 AdamW 就结束？**

## 目标函数定义学习压力

大多数训练问题都可以写成：

\[
\min_\theta \mathcal{J}(\theta)
=
\mathbb{E}_{(x,y)\sim p_{\text{data}}}
[\ell(f_\theta(x),y)]
+ \lambda \Omega(\theta)
\]

这里 \(\theta\) 是模型参数，\((x,y)\) 是从训练数据分布采样的输入和目标，\(f_\theta(x)\) 是模型预测，\(\ell\) 是主损失，\(\Omega(\theta)\) 是正则或约束项，\(\lambda\) 控制约束强度。读这行公式时，重点不是每个符号叫什么，而是它在问：模型在哪些样本上犯错、错在哪里、哪些犯错方式更不能接受。

目标函数不是客观真理，而是你给模型施加的训练压力。交叉熵让模型拟合下一个 token；扩散损失让模型学会去噪；偏好损失让模型相对参考模型偏向 chosen；RL 目标让模型先采样，再根据 reward 调整概率。不同目标对应不同学习信号，不能只比较 loss 数值。

## 监督学习：答案在数据里

自回归语言模型的 next-token objective 是：

\[
\mathcal{L}_{\text{CE}}
=
-
\sum_{t=1}^{T}
\log p_\theta(x_t\mid x_{<t})
\]

这里 \(x_t\) 是第 \(t\) 个 token，\(x_{<t}\) 是前缀。公式读法很直接：正确 token 概率越高，\(-\log p_\theta\) 越小；正确 token 概率越低，惩罚越大。

SFT 仍然是交叉熵，只是输入变成指令 \(x\)，目标变成示范回答 \(y\)：

\[
\mathcal{L}_{\text{SFT}}
=
-
\sum_{t=1}^{T}
\log p_\theta(y_t\mid x,y_{<t})
\]

SFT 教模型“在这个输入下模仿这份示范回答”。它适合学习格式、风格、工具调用模板和领域表达，但它不直接告诉模型哪个回答更受人类偏好，也不保证真实任务风险最低。

## 偏好学习：答案变成相对比较

InstructGPT 的标注界面很适合提醒我们：很多对齐信号不是 token 级标准答案，而是同一个 prompt 下多个回答的相对排序。

![InstructGPT labeler ranking interface 原论文图](../assets/images/paper-figures/training/instructgpt-labeler-ranking.png){ width="860" }

<small>图源：[Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)，Appendix Figure 19(b)。原图表达：标注者在同一个 prompt 下比较多个模型输出，并从最好到最差排序。本站读法：偏好数据描述的是“哪个回答更好”，不是“每个 token 应该是什么”。</small>

DPO 把 chosen / rejected pair 直接变成一个 closed-form preference objective。常见写法是：

\[
\mathcal{L}_{\text{DPO}}
=
-
\log \sigma
\left(
\beta
\log
\frac{\pi_\theta(y^+\mid x)}
{\pi_{\text{ref}}(y^+\mid x)}
-
\beta
\log
\frac{\pi_\theta(y^-\mid x)}
{\pi_{\text{ref}}(y^-\mid x)}
\right)
\]

这里 \(y^+\) 是更受偏好的回答，\(y^-\) 是较差回答，\(\pi_{\text{ref}}\) 是参考模型，\(\beta\) 控制偏好强度。读这行公式时抓住相对性：相对参考模型，提高 chosen 的概率，压低 rejected 的概率。DPO 不需要显式 reward model 和 PPO rollout，但仍然强依赖偏好数据质量、参考模型选择、长度偏置和学习率。

## RL 目标：先采样，再评价

RLHF/RLVR 的目标更间接。模型先按当前策略采样回答，再由 reward model、规则、测试、verifier 或环境给分。一个简化目标是：

\[
\max_\theta\;
\mathbb{E}_{y\sim \pi_\theta(\cdot\mid x)}
\left[
r(x,y)
-
\beta\,\mathrm{KL}(\pi_\theta\|\pi_{\text{ref}})
\right]
\]

这里 \(r(x,y)\) 是完整回答或轨迹的奖励，KL 项限制当前策略不要离参考模型太远。读这行公式时，重点是 reward 不是每个 token 的标准答案。PPO/GRPO 必须把“这个回答整体更好”转成 token 概率更新，所以才需要 old log prob、advantage、clip、KL 或 group baseline 这些稳定装置。

GRPO 可以用同一 prompt 的多条采样回答构造组内优势：

\[
A_i=
\frac{r_i-\mathrm{mean}(\{r_1,\ldots,r_G\})}
{\mathrm{std}(\{r_1,\ldots,r_G\})+\epsilon}
\]

这里 \(r_i\) 是同题第 \(i\) 个回答的奖励，\(G\) 是同题采样数量。公式读法是：不把难题和易题直接比较，而是在同一道题内部看哪个回答更好。这样可以省掉 value model，但会增加 rollout、verifier 和组内方差管理的成本。

![GRPO 原图](../assets/images/paper-deep-dives/technical-reports/deepseek-r1/figure-grpo.png){ width="920" }

<small>图源：[DeepSeek-R1](https://arxiv.org/abs/2501.12948)，Supplementary Fig. 1。原图表达：PPO 使用 value model 估计 advantage，GRPO 直接从同一问题的一组 sampled outputs 的 reward scores 中估计相对 advantage。本站读法：GRPO 省掉 critic，但把压力转移到多采样、reward/verifier 吞吐和长度偏置控制。</small>

## 多目标训练：loss 权重不等于梯度贡献

多模态、世界模型和 VLA 很少只有一个 loss。常见形式是：

\[
\mathcal{L}
=
\lambda_1\mathcal{L}_{\text{task}}
+\lambda_2\mathcal{L}_{\text{aux}}
+\lambda_3\mathcal{L}_{\text{align}}
+\lambda_4\mathcal{L}_{\text{reg}}
\]

这里 \(\lambda_i\) 是各目标权重，但权重不是全部。一个 token 密集、尺度稳定的 video loss，即使权重只有 1.0，也可能贡献大部分梯度；一个稀疏但关键的 risk loss，即使很重要，也可能因为样本少、梯度弱而被淹没。

例如 VLA/world model 同时训练视频预测、动作重建和风险预测：

```text
L = 1.0 * video_loss + 0.2 * action_loss + 0.05 * risk_loss
```

如果 video loss 降了，真实抓取 success 和 risk recall 反而下降，根因可能不是模型“不会动作”，而是训练压力被视频预测占掉了。要看 per-loss grad norm、分桶指标、动作敏感性、风险召回和 closed-loop success。多目标训练的难点不是把公式写长，而是让每个目标的梯度贡献和任务价值匹配。

## 优化器决定怎样走

目标函数定义地形，优化器决定怎么在地形上移动。最抽象的一阶更新可以写成：

\[
\theta_{t+1}=\theta_t-\eta_t u_t
\]

这里 \(\eta_t\) 是第 \(t\) 步学习率，\(u_t\) 是由当前梯度和历史统计构成的更新方向。SGD 的 \(u_t\) 接近当前梯度，Momentum 会平滑历史方向，Adam/AdamW 会用一阶和二阶矩自适应缩放不同参数的步长。

![Loss landscape visualization 原论文图](../assets/images/paper-figures/foundations/loss-landscape-figure-resnet56.png){ width="820" }

<small>图源：[Visualizing the Loss Landscape of Neural Nets](https://arxiv.org/abs/1712.09913)，loss landscape visualization。原图表达：通过二维切片可视化网络附近的 loss surface。本站读法：训练不是一步跳到最低点，而是在复杂地形里按梯度、历史统计和学习率慢慢走。</small>

常见优化器可以按“更新方向怎么被修正”来读：

| 优化器 | 更新方向的特点 | 什么时候常用 |
| --- | --- | --- |
| SGD | 直接沿梯度走 | 小模型、视觉、可控实验 |
| Momentum SGD | 用历史梯度平滑方向 | 视觉和表征训练仍常见 |
| Adam | 用一阶/二阶矩自适应缩放 | Transformer 训练的基础选择 |
| AdamW | 把 weight decay 从 Adam 梯度更新中解耦 | 大模型训练默认起点 |
| Adafactor | 压缩二阶统计，省 optimizer state | 超大模型或显存受限 |
| Muon | 对矩阵动量做正交化 | 新近大规模预训练路线，需要场景验证 |

AdamW 流行不是因为它理论上永远最优，而是因为它稳定、经验丰富、和 Transformer 训练配方匹配。它的关键来自 [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)：在 Adam 这类自适应方法里，把 L2 正则直接加到 loss 里，并不等价于经典 weight decay；AdamW 把 decay 从梯度更新中解耦，使正则行为更清晰。

## 学习率日程决定什么时候快走、什么时候收力

学习率不是一个常数旋钮。常见日程是 warmup + decay：

\[
\eta_t =
\begin{cases}
\eta_{\max}\frac{t}{T_w}, & t\le T_w\\
\eta_{\max}f(t), & t>T_w
\end{cases}
\]

这里 \(T_w\) 是 warmup 步数，\(\eta_{\max}\) 是峰值学习率，\(f(t)\) 是 decay 函数。读这行公式时，先看两个阶段：warmup 让训练从小步开始，避免早期梯度统计、混合精度、通信和 optimizer state 还没稳定时直接大步走；decay 让后期更新变小，减少在好解附近来回震荡。

LR schedule 必须和 effective batch、数据阶段、loss 切换、gradient clipping、混合精度和 checkpoint resume 一起看。很多“模型突然不稳”不是 optimizer 坏了，而是数据切换点、loss 权重切换点和 LR 位置没对齐。

## 稳定项不是越多越好

常见稳定项包括 weight decay、gradient clipping、dropout、label smoothing、KL/reference constraint、auxiliary loss、activation/logit 约束。它们都在改变训练压力。

| 稳定项 | 保护什么 | 过强时会怎样 |
| --- | --- | --- |
| Weight decay | 减少权重无约束增长 | 欠拟合或能力下降 |
| Gradient clipping | 防止异常梯度毁掉一步更新 | 掩盖真实数值异常，更新变钝 |
| KL constraint | 防止后训练偏离底座太远 | reward 学不动，模型过度保守 |
| Dropout / smoothing | 减少过拟合和过度自信 | 学习速度下降，细节能力弱 |
| Auxiliary loss | 给中间能力提供信号 | 辅助目标压制主目标 |

使用这些项时，不能只记录是否开启。要记录实际触发频率、loss scale、gradient norm、对主指标的影响，以及是否只在特定数据桶上造成退化。

## 一次配置变更应该怎样验收

改目标函数、优化器或 LR 日程时，最低限度要回答：

| 问题 | 为什么重要 |
| --- | --- |
| 主 loss 和辅助 loss 的梯度贡献是多少 | loss 权重不等于训练压力 |
| 学习率、warmup、decay 和 effective batch 是否匹配 | 决定早期稳定和后期收敛 |
| optimizer/scheduler/scaler state 能否完整 checkpoint | resume 后曲线不应断裂 |
| clipping、KL、weight decay 是否有实际监控 | 避免稳定项默默接管训练 |
| 数据阶段和 loss 阶段是否对齐 | 防止切换点造成能力冲刷 |
| 分桶评测是否同时看收益和退化 | 平均分会掩盖长尾能力损失 |

成熟训练系统会把 loss 配方、数据阶段、optimizer、scheduler、混合精度、checkpoint state 和评测分桶作为同一个配置资产管理，而不是把它们分散在不同脚本里。

## 外部精读

- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)：理解 SFT、偏好排序、reward model 和 PPO 的数据链。
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)：理解 DPO 如何把偏好 pair 变成 closed-form objective。
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948)：理解 GRPO/RLVR 在长推理任务里的动机。
- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)：理解 AdamW 为什么要解耦 weight decay。
- [Visualizing the Loss Landscape of Neural Nets](https://arxiv.org/abs/1712.09913)：建立优化地形的直觉。

## 阅读结论

训练配置不是一堆默认项。目标函数决定模型被什么信号推动，优化器决定梯度怎样变成更新方向，学习率日程决定这条路径什么时候大步探索、什么时候小步收敛。真正的训练调参不是问“哪个 loss/optimizer 最好”，而是问每个训练压力是否对准任务价值、是否被正确放大、是否被稳定项压住，以及是否在关键能力桶上带来真实收益。
