# 缩放律、幂律与实验曲线

这篇回答一个问题：论文精读里那些 scaling law、log-log 曲线、Pareto frontier、ablation 和 loss curve 应该怎么看。

很多论文不是只报告一个最终分数，而是报告“数据更多会怎样”“模型更大会怎样”“推理步数更多会怎样”“速度和质量如何权衡”。这些图背后的数学并不复杂，但很容易被误读。

## 幂律：规模变大时按固定指数变化

常见缩放律可以写成：

\[
L(N)=aN^{-\alpha}+b
\]

这里 \(N\) 可以是参数量、数据量、计算量或测试时 token 数；\(L(N)\) 是 loss 或 error；\(a\) 是尺度系数，\(\alpha\) 是缩放指数，\(b\) 是不可继续下降的底部误差。

如果 \(\alpha\) 大，说明规模增加带来的收益下降较快；如果 \(\alpha\) 小，说明继续加规模仍有较慢但稳定的收益。注意 \(b\) 很关键：曲线不会无限下降，数据噪声、任务不可约误差和评测上限都会形成地板。

## log-log 图：把幂律看成直线

对幂律取对数：

\[
\log(L-b)
\approx
\log a-\alpha \log N
\]

在 log-log 图上，幂律关系会接近直线，斜率大约是 \(-\alpha\)。这就是为什么很多 scaling law 图喜欢用对数坐标：它能把“按指数缩放”的趋势变成更容易读的直线。

读 log-log 图时要避免一个视觉陷阱：同样的水平距离代表倍数变化，不是加法变化。从 \(10^6\) 到 \(10^7\) 和从 \(10^9\) 到 \(10^{10}\) 在图上可能一样宽，但实际资源规模完全不同。

![EAGLE-3 scaling law](../assets/images/paper-deep-dives/inference/eagle-3/figure-1-scaling-law-eagle3.png){ width="760" }

<small>图源：[EAGLE-3](https://arxiv.org/abs/2503.01840)，Figure 1 相关 scaling law 图。本站在这里用它说明：读缩放曲线时要看横轴代表什么资源、纵轴代表什么收益、斜率是否稳定，以及外推是否仍在数据支持范围内。</small>

## 边际收益：继续加资源还值不值

如果一个方法从 1B 数据到 10B 数据收益很大，但从 100B 到 1T 收益很小，就说明边际收益在下降。论文里的 scaling curve 不只是证明“更大更好”，更重要的是帮助判断下一笔预算该投在哪里。

读曲线时先问：

| 问题 | 为什么重要 |
| --- | --- |
| 横轴是什么 | 参数、数据、FLOPs、采样步数、测试时 token，意义完全不同 |
| 纵轴是什么 | loss、accuracy、speedup、acceptance length、human preference 不能混读 |
| 是否对数坐标 | 线性视觉直觉在 log 坐标下容易误判 |
| 点的范围多大 | 只在小范围拟合的曲线不能随便外推 |
| 是否有误差条 | 没有方差就难判断差异是否可靠 |

## 插值和外推：曲线支持到哪里

如果实验点覆盖 \(10^8\) 到 \(10^{10}\) tokens，在这个区间内读趋势更像插值；把同一斜率推到 \(10^{13}\) tokens，就是外推。外推会遇到几类风险：

| 风险 | 例子 |
| --- | --- |
| 数据分布变了 | 低质量数据比例上升，scaling 指数变差 |
| 瓶颈转移 | 算法 speedup 到某点后被内存、调度或网络卡住 |
| 指标饱和 | benchmark 接近上限后，loss 下降不再转成任务收益 |
| 训练 recipe 变了 | 学习率、batch、token mixture 或后训练不再可比 |

所以论文里的 fitted line 不是许愿线。更稳的写法是：在观测范围内支持某个趋势；超过范围时只能作为假设，需要新实验验证。

## Pareto frontier：不能只看一个指标

很多系统论文同时优化质量、延迟、显存、吞吐和成本。Pareto frontier 表示：在当前候选里，没有另一个方案能在所有指标上都更好。

如果一个模型质量更高但延迟也更高，它不一定支配另一个模型；如果一个方案质量相同但更快，它才明显更优。Gemini、Kimi、Nemotron、推理 runtime、量化和视频生成报告里，经常要按 Pareto 思路读图。

## Ablation：一次只拆一个因素

消融实验的目标是回答“这个模块有没有贡献”。理想消融只改变一个因素：

\[
\Delta
=
\mathrm{score}_{\text{full}}
-
\mathrm{score}_{\text{without module}}
\]

但真实论文里常见问题是：去掉模块后训练预算、超参、数据流或系统开销也变了。读 ablation 时要问：两个设置除了这个模块之外是否真的可比。

## 曲线斜率和拐点：机制可能在变

训练 loss 曲线突然变平，可能是学习率、数据质量、模型容量或优化瓶颈；speedup 曲线突然饱和，可能是新瓶颈从 compute 转到 I/O 或调度；test-time scaling 曲线拐头，可能说明更多 token 开始带来噪声或重复推理。

所以曲线不是装饰。它常常比单点表格更能揭示机制：方法在哪个区间有效，在哪个区间开始失效。

## 读曲线时同时看误差和成本

曲线上的点如果来自单次 run，可能只是随机波动。训练曲线最好有多 seed 或至少有平滑窗口；评测曲线最好有置信区间；系统曲线最好报告硬件、batch、prompt/output 长度和并发分布。

还要把收益除以成本看。一个方法让 score 从 70 到 72，但计算量增加 4 倍；另一个方法从 70 到 71，但几乎免费，二者服务的产品区间可能完全不同。Pareto frontier 的意义就在这里：它不是选“最高分”，而是选“给定预算下最合适的点”。

## 常见误读

| 误读 | 更稳的理解 |
| --- | --- |
| scaling law 证明无限扩展有效 | 它只支持观测区间附近的趋势，远距离外推要保守 |
| 曲线更低就一定更好 | 要看纵轴是 loss、error、latency 还是 cost |
| 拟合线很直就能放心外推 | 训练分布、系统瓶颈和指标上限都可能改变斜率 |
| ablation 少一点就是模块没用 | 可能是指标不敏感、训练未调好或作用只在长尾任务 |
| Pareto 图上最高点最好 | 还要看成本、延迟和部署约束 |

## 读完以后怎么判断

看到 scaling law、loss curve、speedup curve 或 ablation 表，先问横轴资源、纵轴指标、坐标尺度、误差范围、是否可外推、单点收益是否转移到端到端任务。这样读 EAGLE、Genie、Muon、DeepSeek、Kimi、视频生成和推理系统论文会更稳。

## 相关阅读与下一步

- 站内下一步：[采样、估计与校准](sampling-estimation-and-calibration.md)。
- 站内下一步：[数值计算、复杂度与估算](numerics-complexity-and-estimation.md)。
- 论文精读：[EAGLE-3：训练时见过自己犯错](../paper-deep-dives/inference/eagle-3.md)。
- 论文精读：[GPU Utilization：为什么 100% 不等于训练跑满](../paper-deep-dives/foundations/gpu-utilization-misleading.md)。
