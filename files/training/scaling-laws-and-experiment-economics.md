# 固定训练预算下，参数和 token 怎样交换

你手里只有一笔训练预算，怎样判断应该扩大模型，还是多喂一些 token？Scaling law 试图回答这个问题。它不是一条放之四海而皆准的比例，而是一组实验在特定数据、架构和优化设置下留下的曲线。

曲线能帮你选下一组实验，不能代替下游任务评测。一个模型的 held-out loss 变好了，工具调用、长上下文和机器人控制仍可能没有变化。

## 先把拟合假设写在纸上

一种常见近似为

$$
L(N,D)=A N^{-\alpha}+B D^{-\beta}+L_\infty,
$$

N为参数量，D为训练token曝光，A/B和指数由实验拟合。$L_\infty$表示该模型下的渐近项。这个式子将有限模型与有限数据的影响分开，但真实数据混合、训练不足和架构变化可能破坏假设。

[Chinchilla](https://arxiv.org/abs/2203.15556)的IsoFLOP实验在固定计算量下比较多个模型/token组合，观察损失谷底。曲线提供的是原设置下的证据，不能把某个参数/token比例永久当成所有模型的规则。

![Chinchilla IsoFLOP curves 原论文图](../assets/images/paper-figures/training/chinchilla-isoflop-curves.png){ width="920" }

<small>图源：[Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)，Figure 4。原论文图意：固定 FLOP 预算时，不同模型大小会对应不同最终 loss；曲线谷底给出该预算下更合适的参数量，并外推出参数和 token 的 scaling 关系。</small>

## 用约束推一次最优点

对稠密自回归模型，忽略额外attention、重计算等成本，使用 $C\approx6ND$。令 $D=C/(6N)$，代入损失后对N求导，最优点满足

$$
\alpha A N^{-\alpha}=\beta B D^{-\beta}.
$$

由此得到量级关系

$$
N_*\propto C^{\beta/(\alpha+\beta)},\qquad
D_*\propto C^{\alpha/(\alpha+\beta)}.
$$

若两个指数相等，参数与token随预算按相同幂次增长。这个结论依赖拟合式和计算约束，没有假定所有实际模型指数相同。

![Chinchilla optimal tokens and parameters 原论文图](../assets/images/paper-figures/training/chinchilla-tokens-vs-params.png){ width="620" }

<small>图源：[Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)，Figure 15。原论文图意：在固定训练 FLOP 预算下，三种估计方法给出相近的最优 token 数和参数量关系。</small>

## 计算token预算时不要偷偷扣掉重复

D统计实际进入训练计算的token曝光。重复文本即使信息增量有限，也会消耗计算并贡献梯度。不能用 $D(1-重复率)(1-噪声率)$ 伪造一个通用“有效token数”，再直接代入原拟合。

数据质量变化可以单独建模或重新拟合，并用真实实验验证。独立样本规模、曝光次数和任务覆盖应分别记录。

## 拟合之前先看实验是否可比较

固定tokenizer、损失口径、数据分布、训练日程和评测集，覆盖多个N/D组合。只沿“模型越大、训练越久”一条对角线收集点，难以区分两项贡献。

保留未参与拟合的规模或预算做预测检验，观察残差与外推误差。若小模型未充分优化、某组学习率不合适，曲线可能反映训练配置缺陷而非架构容量。

## 训练最优与总成本最优可以不同

更小模型若训练更久，训练成本可能增加，但部署时每次请求更便宜。假设额外训练成本为K，每请求节省c，且质量与服务目标不变，则至少约K/c次请求才能抵回投入。

这个式子还没包含研发时间、失败实验和系统迁移。预测请求量不确定时，应展示不同场景，不把单点预算写成保证回本。

使用scaling结论时，明确它预测的是heldout loss、当前任务质量还是总成本。前者改善以后，仍需独立验证工具、长上下文或控制等具体能力。

## Kaplan 与 Chinchilla 不能只比较一个比例

[Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)和 [Chinchilla](https://arxiv.org/abs/2203.15556)都拟合了损失与规模、数据和计算的关系，但实验预算、模型配方和优化条件不同。前者常被引用来说明损失随模型和数据呈幂律变化，后者在固定 FLOP 的比较中强调许多早期模型训练 token 偏少。把后者的参数与 token 比例直接套到另一种架构、数据去重策略或训练阶段，超出了论文证据。

读到一个 scaling 结论时，至少记录四项信息。它拟合的目标是验证损失还是下游分数，计算量怎样定义，模型是否充分训练，预测点是否落在实验范围内。四项缺一项，曲线仍可作为选点线索，却不能当成定律。

## 一张实验表应当能看出哪些组合可比

| run | 参数量 N | 训练 token D | 估算 FLOP C | 完成状态 | 验证 loss | 备注 |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| A | 125M | 1B | $7.5\times10^{17}$ | 收敛 | 2.85 | 基线 |
| B | 350M | 350M | $7.35\times10^{17}$ | 未充分训练 | 3.02 | 不能与 A 直接比较 |
| C | 350M | 1B | $2.1\times10^{18}$ | 收敛 | 2.61 | 预算不同 |

B 的 loss 较高，可能来自 token 太少，也可能来自训练未完成。只有先把完成状态和数据配方写清楚，拟合才有解释空间。表格中的 FLOP 仍是估算值，实际还会包含 attention、重计算、通信和 padding；端到端成本需要另记墙钟时间与 GPU 小时。

## 从曲线到下一次实验

可以把拟合工作拆成四步。先锁定 tokenizer、数据版本、优化器、序列长度和验证集；然后在参数量和 token 数上各取至少两个水平，避免所有点落在同一条对角线上；接着留出一个没有参与拟合的组合；最后用预测误差决定下一次实验，而不是看到幂律很直就继续扩大预算。

一个实用的结果卡片可以写成

```text
question_id: scaling-2026-01
changed: model_size, training_tokens
fixed: tokenizer, data_mix, optimizer, eval_set
budget: 8 GPU-hours per point
fit_target: heldout_loss
heldout_point: N=700M, D=2B
decision_rule: if prediction error > 0.05, add a cross point before scaling up
downstream_check: long_context_exact_match, tool_success
```

这个卡片把“拟合得很好”与“值得继续投入”分开。若验证 loss 的变化无法改变模型大小、数据比例或停止条件，继续添加同分布点的价值很低，应把预算移到能区分下游能力的评测。

## 用一个小预算检查拟合有没有越界

假设同一 tokenizer 和数据混合下，预算 (C) 固定，准备了三个模型规模和三种 token 预算。不要只保留每条对角线上的最低 loss，还要保存每个组合的实际 FLOPs、训练是否充分和验证集结果。若某个小模型在预算结束时仍明显没有收敛，它在曲线上的位置不能和充分训练的模型直接比较。

拟合后留出一个没有参与选择的组合。例如用两种预算和两种规模拟合，第三种规模只用来检查外推误差。若预测在留出点上系统偏高或偏低，先检查数据分布、优化器和 token 计数，再决定是否扩大模型或延长训练。一次漂亮的幂律拟合，不能替代这个留出检查。

博士阶段做实验时，建议把“下一次实验会改变什么决策”写在表格旁边。若无论 loss 怎样变化都不会改变模型规模、数据配比或停止条件，继续收集更多拟合点的价值就很低，应转向能区分下游能力的评测。


## 对照源码与延伸阅读

- [模型规模、token曝光与实际训练循环的连接](https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/train.py)
- [Kaplan scaling laws](https://arxiv.org/abs/2001.08361)与 [Chinchilla compute-optimal training](https://arxiv.org/abs/2203.15556)用于对照拟合对象和预算假设

## 相关阅读

训练规模的结论要接到实际训练记录，建议继续看[训练数据与配比](scaling-curriculum-and-data-mixture.md)、[优化器与学习率](objectives-optimizers-and-schedules.md)和[训练评测与消融](evaluation-and-ablation-methodology.md)。这些页面分别处理数据分布、更新计数和比较协议，不能用一个 scaling 指数替代。

**来源与改编说明**：拟合形式和计算最优讨论参考 [Chinchilla](https://arxiv.org/abs/2203.15556)，训练循环字段参考 [nanoGPT](https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/train.py)。幂律关系只在相近数据、架构和优化条件下解释，页面中的成本回收式是教学估算。
