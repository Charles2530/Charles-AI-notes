# 采样、估计与校准

这篇回答一个问题：论文里说的 Monte Carlo、估计器、方差、置信区间、acceptance rate、calibration 和 ECE 到底在衡量什么。

很多模型目标写成期望，但真实系统只能采样；很多论文报告平均指标，但真实部署关心长尾；很多方法依赖 confidence，但 confidence 未必等于真实成功率。采样、估计和校准，就是把这些不确定性讲清楚。

## Monte Carlo：用样本平均近似期望

如果目标是：

\[
\mu=\mathbb{E}_{X\sim p}[f(X)]
\]

但我们无法直接算整个分布，就可以采样 \(N\) 个样本：

\[
\hat{\mu}
=
\frac{1}{N}\sum_{i=1}^{N} f(x_i),
\quad x_i\sim p
\]

这里 \(\hat{\mu}\) 是 Monte Carlo 估计。样本越多，估计通常越稳，但成本也越高。

后续例子：世界模型评测里，如果用 learned world model 做 Monte Carlo rollout 来估计某个 policy 的 value，就必须问两件事：采了多少条轨迹，模型误差会不会随 rollout 长度累积。

## 估计器：你用什么规则从样本得到数字

估计器是一个从样本到数字的规则。比如样本均值：

\[
\hat{\mu}
=
\frac{1}{N}\sum_{i=1}^N x_i
\]

样本方差：

\[
\hat{\sigma}^2
=
\frac{1}{N-1}
\sum_{i=1}^{N}(x_i-\hat{\mu})^2
\]

这里分母用 \(N-1\)，是为了修正用样本均值估计真实均值带来的偏差。读论文结果时，平均值只是估计，方差和样本数决定这个估计有多可靠。

## 方差和标准误：平均数有多可信

如果单个样本的标准差是 \(\sigma\)，样本均值的标准误大约是：

\[
\mathrm{SE}
=
\frac{\sigma}{\sqrt{N}}
\]

这行公式说明：样本数变成 4 倍，平均值的不确定性大约只减半。对于机器人成功率、长上下文 acceptance、视频偏好评分这类指标，多采样很贵，所以报告方差、置信区间或任务桶分布非常重要。

## 置信区间：别只看一个平均数

一个常见近似是：

\[
\hat{\mu}\pm 1.96\cdot \mathrm{SE}
\]

这可以粗略理解成 95% 置信区间。它不是说真实均值有 95% 概率在这个区间里，而是说如果重复很多次同样实验，这种构造区间的方法大约 95% 能覆盖真实均值。

论文里如果两个方法平均分差很小，但置信区间高度重叠，就不能轻易说一个方法“显著更好”。

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
| 相关性高就代表因果 | calibration / correlation 只说明统计关系，不自动说明机制 |

## 读完以后怎么判断

看到采样指标、平均分、confidence 或 calibration 图时，先问样本来自哪里、样本数多少、方差多大、confidence 是否校准、平均收益是否覆盖 P95/P99 和任务桶。这样读 EAGLE、GRPO、world model evaluation 和 judge model 论文会清楚很多。

## 相关阅读与下一步

- 站内下一步：[概率论、分布与期望](probability-distributions-and-expectation.md)。
- 站内下一步：[随机过程与马尔可夫性](stochastic-processes-and-markov.md)。
- 论文精读：[EAGLE-2：动态 Draft Tree](../paper-deep-dives/inference/eagle-2.md)。
- 论文精读：[DeepSeek-R1：RL 激发推理能力](../paper-deep-dives/technical-reports/deepseek-r1.md)。
