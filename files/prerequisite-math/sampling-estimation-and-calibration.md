# 采样、估计与校准

这篇回答一个问题：论文里说的 Monte Carlo、估计器、方差、置信区间、acceptance rate、calibration 和 ECE 到底在衡量什么。

很多模型目标写成期望，但真实系统只能采样；很多论文报告平均指标，但真实部署关心长尾；很多方法依赖 confidence，但 confidence 未必等于真实成功率。采样、估计和校准，就是把这些不确定性讲清楚。

## Monte Carlo 估计为什么会抖

如果目标是：

\[
\mu=\mathbb{E}_{X\sim p}[f(X)]
\]

但我们无法直接算整个分布，就从 \(p\) 独立采样 \(N\) 次，并定义：

\[
X_i\overset{\mathrm{i.i.d.}}{\sim}p,
\qquad
\hat\mu_N=\frac{1}{N}\sum_{i=1}^N f(X_i).
\]

这里 \(\hat\mu_N\) 是 \(\mu\) 的 Monte Carlo 估计。这个从样本得到数字的规则叫估计器；它为什么会抖，要在区分偏差和方差后，再看样本均值的方差与标准误。

后续例子：世界模型评测里，如果用 learned world model 做 Monte Carlo rollout 来估计某个 policy 的 value，就必须问两件事：采了多少条轨迹，模型误差会不会随 rollout 长度累积。

## 估计器：你用什么规则从样本得到数字

估计器是一个从样本到数字的规则。上面的 \(\hat\mu_N\) 是均值估计器；另一个例子是对 \(f(X)\) 的样本方差：

\[
\hat{\sigma}^2
=
\frac{1}{N-1}
\sum_{i=1}^{N}(f(X_i)-\hat\mu_N)^2
\]

这里分母用 \(N-1\)，是为了修正用样本均值估计真实均值带来的偏差。读论文结果时，平均值只是估计，方差和样本数决定这个估计有多可靠。

## 偏差和方差：估计错在哪里

估计误差常被拆成两类。偏差问的是“平均来说是否系统性偏离真值”，方差问的是“换一批样本会不会抖得很厉害”。

一个估计器 \(\hat{\theta}\) 的偏差可以写成：

\[
\operatorname{Bias}(\hat{\theta})
=
\mathbb{E}[\hat{\theta}]-\theta
\]

如果一个 judge model 总是高估某类回答，它有偏差；如果机器人评测每次成功率波动很大，它方差高。增加样本数主要降低方差，不会自动消除数据采样偏、标注偏或 simulator 偏差。

## 方差和标准误：平均数有多可信

若被估计量的单样本方差有限，即

\[
\sigma^2=\operatorname{Var}_p[f(X)]<\infty,
\qquad
\operatorname{Var}(\hat\mu_N)=\frac{\sigma^2}{N}
\]

第二项说明换一批样本时 \(\hat\mu_N\) 会怎样波动。增加 \(N\) 会降低方差，但不会让有限样本的结果完全不抖；单个 \(f(X)\) 的方差 \(\sigma^2\) 越大，同样预算下的均值估计越不稳定。

样本均值的标准误，也就是这个估计器的标准差，为：

\[
\mathrm{SE}
=
\frac{\sigma}{\sqrt{N}}
\]

这行公式说明：样本数变成 4 倍，平均值的不确定性只减半。对于机器人成功率、长上下文 acceptance、视频偏好评分这类指标，多采样很贵，所以报告方差、置信区间或任务桶分布非常重要。

在强化学习里，\(X_i\) 可以是一条轨迹，\(f(X_i)\) 是它的 return。同一策略也可能因动作采样、环境转移、初始状态和稀有事件得到很不一样的 return，因此 Monte Carlo policy gradient 容易有高方差。如果样本彼此相关，上面的 \(\sigma^2/N\) 不能原样套用，有效信息通常少于同样数量的独立样本。

## Baseline 为什么能降方差

Policy gradient 常把 return \(G_t\) 换成 \(G_t-b(s_t)\)。只要给定状态后 baseline 不依赖当前采样动作，常见做法是只依赖状态，它贡献的期望梯度就是 0：

\[
\mathbb E_{A\sim\pi_\theta(\cdot\mid s)}
\left[b(s)\nabla_\theta\log\pi_\theta(A\mid s)\right]
=b(s)\nabla_\theta\sum_a\pi_\theta(a\mid s)
=0.
\]

所以减去这样的 baseline 不改变期望 policy gradient，却能先扣掉“这个状态本来通常就能得到的回报”。当 \(b(s)\) 能较好预测该状态下的平均 return 时，剩下的量更接近动作相对平均水平的好坏，样本间波动往往更小。baseline 选得差不保证降方差，但它与当前采样动作独立这一条件保证了上述无偏性质；critic 估计的 state value 是常见选择。算法里的具体用法见 [Policy Gradient、Actor-Critic、PPO 与 GRPO](../reinforcement-learning/policy-gradient-actor-critic-ppo-grpo.md)。

## 重要性采样：数据来自另一个分布时怎样改权重

目标分布是 \(p\)，样本却来自 \(q\) 时，可以用概率比改写期望：

\[
\mathbb E_p[f(X)]
=
\mathbb E_q\left[\frac{p(X)}{q(X)}f(X)\right].
\]

这里 \(p(X)/q(X)\) 是重要性权重：\(q\) 很少采到、但 \(p\) 认为重要的样本会获得更大权重。这个等式要求在所有 \(p(x)f(x)\) 会贡献期望的地方都有 \(q(x)>0\)；如果行为分布从不覆盖目标分布的重要区域，事后加权无法补回根本没见过的样本。

即使支持集条件满足，权重也可能非常不均匀。少数巨大权重会支配总和，大量近零权重样本几乎不提供信息，于是名义上有很多样本，实际有效样本信息却集中在极少数样本上，估计方差可能很高。

PPO 使用的每步 action probability ratio 不是完整的 trajectory importance weight，不能完全修正更新策略与 rollout 策略造成的 state-distribution mismatch。对 ratio 做 clipping 会引入偏差，换取更稳定的有限样本更新；因此 rollout policy 仍需与 update policy 保持接近，不能把 clipping 当成任意复用旧轨迹的许可。概率比和更新约束的具体形式见 [Policy Gradient、Actor-Critic、PPO 与 GRPO](../reinforcement-learning/policy-gradient-actor-critic-ppo-grpo.md)。

## 置信区间：别只看一个平均数

一个常见近似是：

\[
\hat\mu_N\pm 1.96\cdot \mathrm{SE}
\]

这可以粗略理解成 95% 置信区间。它不是说真实均值有 95% 概率在这个区间里，而是说如果重复很多次同样实验，这种构造区间的方法大约 95% 能覆盖真实均值。

论文里如果两个方法平均分差很小，但置信区间高度重叠，就不能轻易说一个方法“显著更好”。

如果指标是二元成功率，例如机器人任务成功 / 失败，可以用更贴近二项分布的粗略近似：

\[
\hat p
\pm
1.96
\sqrt{
\frac{\hat p(1-\hat p)}{n}
}
\]

这里 \(\hat p\) 是样本成功率，\(n\) 是试验次数。这个近似在样本数太少或成功率非常接近 0 / 1 时会不稳，但足够提醒读者：`成功 18/20` 和 `成功 90/100` 看起来都是 90%，可信度并不一样。

## 分层估计：平均数要覆盖任务桶

很多评测不是一个均匀分布。长上下文请求、数学题、工具调用、失败恢复、不同机器人任务和视频偏好，都可能来自不同任务桶。整体平均可以写成加权和：

\[
\mu
=
\sum_g w_g\mu_g
\]

这里 \(g\) 是任务桶，\(w_g\) 是该桶权重，\(\mu_g\) 是桶内平均表现。如果线上真实权重 \(w_g\) 和论文 benchmark 权重不同，整体结论会变。

所以通用读法不是“平均分更高就结束”，而是看每个关键桶：简单样本、困难样本、长尾、分布外、失败恢复、长输出和高并发场景是否都被覆盖。

## Acceptance rate：候选被接受的比例

投机解码里常看 acceptance rate：

\[
\mathrm{acceptance}
=
\frac{\text{accepted draft tokens}}
{\text{proposed draft tokens}}
\]

它衡量 draft model 提出的 token 有多少被 target model 接受。速度提升不只取决于 draft 快不快，也取决于接受率是否足够高。

![Draft confidence calibration](../assets/images/paper-deep-dives/inference/eagle-2/figure-6-confidence-calibration.png){ width="690" }

<small>图源：[EAGLE-2](https://arxiv.org/abs/2406.16858)，Figure 6。原图比较 draft confidence 与真实 acceptance rate。本站在这里用它说明：confidence 只有经过校准，才能作为昂贵验证前的便宜代理信号。</small>

## 校准：模型说 80% 时，真的有 80% 吗

校准关心预测概率和真实频率是否一致。比如把所有 confidence 在 0.8 附近的样本拿出来，如果真实成功率也约 0.8，就说明这个桶校准较好。

ECE（Expected Calibration Error）常写成：

\[
\mathrm{ECE}
=
\sum_{m=1}^{M}
\frac{|B_m|}{n}
\left|
\mathrm{acc}(B_m)
-
\mathrm{conf}(B_m)
\right|
\]

逐项读：

| 符号 | 含义 |
| --- | --- |
| \(B_m\) | 第 \(m\) 个 confidence 桶 |
| \(|B_m|/n\) | 这个桶占总样本比例 |
| \(\mathrm{acc}(B_m)\) | 这个桶里的真实准确率 |
| \(\mathrm{conf}(B_m)\) | 这个桶里的平均置信度 |

如果模型总是说 0.95，但真实成功率只有 0.7，ECE 会变大。世界模型的 risk head、reward model、draft confidence、judge model 都需要关注校准，而不是只看平均分。

## 排序相关：只关心谁更好时看 rank

有些任务不需要精确预测数值，只需要把候选排对。例如世界模型评估多个 action candidate 时，top-1 是否一致可能比 value 数字本身更重要。

Spearman rank correlation 衡量两个排序是否一致。它常用于比较想象环境中的 policy ranking 和真实环境中的 ranking 是否同向。

如果 learned world model 给出的 value 数值偏高，但仍能把好策略排在坏策略前，它对候选筛选可能仍有价值；反过来，平均误差小但排序错，会直接伤规划。

## 常见误读

| 误读 | 更稳的理解 |
| --- | --- |
| 样本平均就是事实 | 它只是有限样本估计，要看方差和样本数 |
| confidence 高就可靠 | 要看校准，尤其是分布外和长尾任务 |
| acceptance rate 高就一定快 | draft 成本、验证并行、tree 构造和调度也会吃掉收益 |
| 增加样本数能解决所有不确定性 | 它主要降低方差，不能自动修正系统性偏差 |
| 总平均能代表所有用户 | 任务桶权重变化会改变整体结论 |
| 相关性高就代表因果 | calibration / correlation 只说明统计关系，不自动说明机制 |

## 读完以后怎么判断

看到采样指标、平均分、confidence 或 calibration 图时，先问样本来自哪里、样本数多少、方差多大、confidence 是否校准、平均收益是否覆盖 P95/P99 和任务桶。这样读 EAGLE、GRPO、world model evaluation 和 judge model 论文会清楚很多。

## 相关阅读与下一步

- 站内下一步：[概率论、分布与期望](probability-distributions-and-expectation.md)。
- 站内下一步：[随机过程与马尔可夫性](stochastic-processes-and-markov.md)。
- 论文精读：[EAGLE-2：动态 Draft Tree](../paper-deep-dives/inference/eagle-2.md)。
- 论文精读：[DeepSeek-R1：RL 激发推理能力](../paper-deep-dives/technical-reports/deepseek-r1.md)。
