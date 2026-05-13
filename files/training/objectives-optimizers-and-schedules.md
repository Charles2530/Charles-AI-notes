# 目标函数、优化器与 LR 日程

模型架构决定系统允许学什么，数据决定系统能看到什么，目标函数、优化器和学习率日程决定系统实际沿着怎样的路径学到什么。同一个模型、同一批数据，仅仅更换 loss 配方和 schedule，就可能得到截然不同的训练轨迹。

这页给训练目标和优化配置一个统一入口。更系统的数据配方见 [Scaling、课程学习与数据配比](scaling-curriculum-and-data-mixture.md)，数值异常排查见 [训练稳定性与故障排查](stability-numerics-and-failure-triage.md)。

!!! note "初学者先抓住"

    训练配方可以先按“目标、步长、刹车”理解：loss 告诉模型什么叫错，optimizer 决定怎样改参数，learning rate schedule 决定每一步迈多大。三者必须一起看，单独换一个组件，往往会改变整条学习轨迹。

!!! example "有趣例子：厨师调味"

    模型像厨师，loss 像食客反馈，optimizer 像改菜谱的方法，learning rate 像每次加盐的量。反馈再准确，如果一次加半罐盐，菜也会报废；如果每次只加一粒盐，改进又会慢到不可接受。

!!! tip "主线定位：降低训练成本和错误优化成本"

    目标函数、优化器与 LR 日程在世界模型高效训练里负责回答：训练信号到底在奖励什么，哪些能力被大桶数据主导，哪些阶段切换会破坏稳定性。它降低训练成本和错误优化成本；如果 loss 与闭环目标错位，算力会被稳定地花在错误方向上。

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

图文对齐和样本匹配常用 InfoNCE：

\[
\mathcal{L}_{\text{InfoNCE}}
=
-\log
\frac{\exp(\mathrm{sim}(z,z^+)/\tau)}
{\exp(\mathrm{sim}(z,z^+)/\tau)+\sum_i\exp(\mathrm{sim}(z,z_i^-)/\tau)}
\]

它优化的是表示几何，而不是直接分类。

偏好学习如 DPO 会比较优选回答和劣选回答，推动策略相对参考模型向偏好方向移动。这类目标对参考模型、偏好数据、KL 约束和学习率非常敏感。

强化学习目标比普通监督学习多一层间接性。监督学习里，目标答案 \(y\) 通常已经给定；强化学习里，模型先按当前策略采样动作或回答，再根据奖励判断这次采样是否值得加强。语言模型 RLHF 可以简化成：

\[
\max_\theta\;
\mathbb{E}_{y\sim \pi_\theta(\cdot\mid x)}
\left[
r_\phi(x,y)-\beta\,\mathrm{KL}(\pi_\theta\|\pi_{\text{ref}})
\right].
\]

这里 \(\pi_\theta\) 是当前模型，\(r_\phi\) 是 reward model，\(\pi_{\text{ref}}\) 常是 SFT/reference 模型。这个目标和交叉熵最大的区别是：模型不是模仿固定答案，而是在一组可能回答里尝试提高高奖励回答的概率，同时用 KL 防止自己离参考模型太远。

!!! note "初学者解释：RLHF 不是让 reward 直接替代 loss"
    Reward model 给的是完整回答的偏好分数，不是每个 token 的标准答案。PPO 这类算法要把“这个回答整体更好”转成 token 级概率更新，因此需要 value、advantage、旧策略概率、clip 和 KL 这些稳定装置。缺少这些装置，模型很容易为了追高 reward 走偏，例如变长、套模板、过度拒答或牺牲事实性。

## GRPO / RLVR：同一道题多采样，再做组内比较

DeepSeek-R1、Qwen3 等技术报告里反复出现 `GRPO` 和 `RLVR`。这部分如果只理解成“又一种 PPO 变体”，会漏掉重点。它真正回答的是：当奖励来自数学答案、代码测试、选择题、格式检查这类 verifier 时，怎样把“最终是否成功”转成稳定的策略更新。

![GRPO 原图](../assets/images/paper-deep-dives/technical-reports/deepseek-r1/figure-grpo.png){ width="920" }

<small>图源：[DeepSeek-R1](https://arxiv.org/abs/2501.12948)，Supplementary Fig. 1。原论文图意：PPO 使用 value model 估计 advantage；GRPO 省去 value model，直接从同一问题的一组 sampled outputs 的 reward scores 中估计相对 advantage。</small>

!!! note "这张图帮你补哪块基础"
    PPO 里通常有 policy、reference、reward model 和 value model。value model 要猜“当前前缀未来大概能拿多少奖励”，这在长 CoT 里很难，因为中间推理可能被后面推翻。GRPO 换了一个更朴素的比较方式：同一道题一次采样多条回答，谁通过 verifier、谁格式正确、谁结果更好，就在这一组里相对更高。

    这张图支撑的工程判断是：GRPO 省掉 value model，降低 critic 训练和显存成本；但它把成本转移到组内多采样、reward/verifier 吞吐和 rollout freshness 上。因此不能只问“要不要 critic”，还要算每个 prompt 的采样组大小、平均输出长度和 verifier 延迟。

GRPO 的 advantage 通常来自同题组内标准化：

\[
A_i=
\frac{r_i-\mathrm{mean}(\{r_1,\ldots,r_G\})}
{\mathrm{std}(\{r_1,\ldots,r_G\})}.
\]

这里 \(G\) 是同一道题采样的回答数，\(r_i\) 是第 \(i\) 条回答的奖励。直觉很像同一场考试里比较多份草稿：不是问“这份草稿绝对值多少分”，而是问“它比同题其他解法更好吗”。这样可以省掉 critic / value model，也减少一个大模型训练和显存负担。

`RLVR` 指的是 reinforcement learning with verifiable rewards。它最适合这类任务：

| 任务 | Verifier 例子 | 为什么适合 RL |
| --- | --- | --- |
| 数学 | 最终答案匹配、符号化检查 | 中间步骤难标，但结果可验证 |
| 代码 | 单元测试、编译器、执行结果 | 真实成功信号比偏好打分更可靠 |
| 选择题 / STEM | 标准答案、格式检查 | reward 稳定，采样成本可控 |
| 工具调用 | schema、返回码、任务完成信号 | 能评价完整轨迹是否闭环 |

但它也有边界。开放写作、主观审美、复杂安全判断没有天然 verifier，这时仍要依赖 reward model、judge、人审或规则组合。即使有 verifier，也可能出现 reward hacking：模型学会迎合格式、利用测试漏洞、输出不可读但能过检查的答案。因此 GRPO / RLVR 应和 rejection sampling、格式约束、人工抽检、长度监控、通过率分桶和真实任务评测一起看。

!!! example "有趣例子：一题多解的补习班"
    老师不给每一步打分，只看最后答案是否正确。学生同一道题写 16 种解法，正确且清楚的解法下次更可能被采用，明显错的解法概率下降。GRPO 就是在把这种“同题多解、组内比较”的信号变成模型参数更新。

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

!!! note "常见误区：loss 降了就一定更好"

    Loss 是训练压力的代理，不是最终能力本身。某个辅助 loss 下降，可能只是模型学会了更容易的中间任务；偏好 loss 改善，也可能伴随事实性、长上下文或格式稳定性退化。更稳的读法是：每次改 loss、权重或 schedule，都要同步看分桶评测、失败样本和关键能力回归。

### 多目标训练的排查链：辅助 loss 把主目标压住了

假设一个 VLA/world model 同时训练视频预测、动作重建和风险预测：

```text
L = 1.0 * video_loss + 0.2 * action_loss + 0.05 * risk_loss
```

训练后 `video_loss` 从 `0.092` 降到 `0.078`，但真实抓取 success 从 `74%` 掉到 `66%`，risk recall 从 `69%` 掉到 `55%`。这时不能只看 loss 权重数字，因为不同 loss 的尺度和梯度范数不一定可比：

| Signal | Loss Weight | Loss Change | Grad Norm Share | 任务后果 |
| --- | ---: | ---: | ---: | --- |
| video prediction | `1.0` | `-15%` | `82%` | 画面更像 |
| action reconstruction | `0.2` | `-3%` | `13%` | 动作差异弱 |
| risk prediction | `0.05` | `+9%` | `5%` | 风险漏报 |

这里的机制是：视频预测 token 多、loss 稳、梯度占比大，模型自然把容量花在让未来“看起来像”；动作和风险信号稀疏，虽然更接近闭环成败，却被主损失淹没。

修复链应该按目标函数而不是按模块拍脑袋：

```text
症状：视频预测更好，动作执行和风险指标下降
指标：per-loss grad norm、bucket loss、action sensitivity、risk recall、closed-loop success
机制：多目标 loss 的数值权重不等于梯度贡献，也不等于任务价值
修复：梯度归一化、分阶段训练、提高高价值桶采样、risk/action 辅助头延后冻结或单独 warmup
反例：过度放大稀有风险 loss 会让模型保守，主分布动作变慢或拒绝执行
边界：如果模型只做离线视频生成，video loss 可以主导；一旦接 planner，动作和风险目标必须有硬门槛
```

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
| Muon | 对矩阵权重的动量做正交化 | 主要用于大规模预训练，通常与 AdamW/Adam 混合处理非矩阵参数 |
| Adafactor | 节省优化器状态 | 行为和调参经验不同于 AdamW |
| Lion 等变体 | 试图减少状态或改变更新几何 | 需要充分大规模证据 |

工业大模型训练里，AdamW 仍常作为默认起点，不是因为它理论上永远最优，而是因为稳定、可预期、经验最丰富。

Muon 代表另一类思路：把 Transformer 中的矩阵参数当作线性算子来更新，通过 Newton-Schulz 近似正交化 momentum，减少少数奇异方向主导训练的问题。它在 [Muon 专题讲解](../paper-deep-dives/foundations/muon-is-scalable.md) 中被系统验证到 5.7T token MoE 预训练，但在 SFT、后训练和已有 AdamW checkpoint 迁移上仍要单独验证。

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
