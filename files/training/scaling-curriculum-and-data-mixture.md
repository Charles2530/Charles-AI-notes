# 数据配比怎样改变每次更新的方向

从两种来源各抽一半样本，不一定让它们贡献一半训练信号。样本长度、loss归一化与训练顺序都会改变实际目标。先算曝光，再看梯度，才能判断一个数据配方是否被正确执行。

## 样本比例与token比例不同

来源A每条平均100个目标token，B每条1000个。各抽一半样本、按所有目标token平均loss时，B约占90.9%的监督位置。若希望来源损失等权，则需要分别归一化再组合，或者调整采样与长度策略。

设两个来源期望损失为 $L_A,L_B$，混合目标 $L=\alpha L_A+(1-\alpha)L_B$。梯度也是相同加权和，但只有采样器与分母实现了这一约定，配置里的alpha才真有这个含义。

## 配比可以直接转动梯度

教学设定中，A梯度为 `[2,0]`，B为 `[-1,3]`。各半混合得到 `[0.5,1.5]`；A权重提高到0.8，得到 `[1.4,0.6]`。更新方向已经明显改变。

这不意味着某种来源必然损害另一种能力。真实梯度随参数和输入变化，需要在独立子集上检查结果。标签“代码”“数学”“对话”只描述来源，不能代替当前实验的因果证据。

## 小数据源可能被反复曝光

假设总预算100B目标token，小来源有2B独立token，却按20%采样。它预计贡献20B曝光，相当于平均重复10遍；大来源80B若贡献80B，平均约1遍。

重复并不自动无效，但会改变过拟合与记忆风险。报告配方时同时给独立数据规模和实际曝光量，避免把20B曝光写成20B新数据。

## 相同数据换顺序也会走不同轨迹

每个batch的梯度在当前参数上计算。先看A再看B时，B遇到的参数已被A修改；调换顺序后不再是同一组数。学习率随时间变化时，后到的数据还可能获得不同更新幅度。

Curriculum可以先安排某类样本，再逐步改变难度或来源。它是一项需要验证的训练假设，不应预设“从易到难”必定更好。对照实验需保持总token、模型与计算预算一致，再比较顺序。

![Wan data provisioning](../assets/images/paper-deep-dives/diffusion/wan/figure-data-stage.png){ width="920" }

<small>图源：[Wan: Open and Advanced Large-Scale Video Generative Models](https://arxiv.org/abs/2503.20314)，Figure 3。原论文图意：不同训练阶段的数据供给策略会动态调整 motion、quality 和 category 的比例。</small>

## 用实际计数检验配方

采样器按来源记录样本数、非padding token、目标token和重复曝光。Packing、过滤和截断都可能让实际占比偏离请求的比例。分布式loader还需避免多worker重复读同一片段。

配方调整后，用固定评测子集比较收益与退化，并保留一组来源不重叠的heldout输入。若同时改模型大小、token预算与采样比例，结论就无法单独归因到配方。

## 把“比例”写成采样器能够执行的协议

训练配置中的 `weight_A=0.7` 仍然不够。至少要说明它作用在样本、序列、有效 token 还是 loss 上。设第 (i) 个来源在一个 update 中抽到 (n_i) 个样本，第 (i) 个样本的有效目标 token 数为 (m_{ij})，则按所有 token 平均的损失是

\[
L_{\text{token}}=\frac{\sum_i\sum_{j=1}^{n_i}\sum_{k=1}^{m_{ij}}\ell_{ijk}}
{\sum_i\sum_{j=1}^{n_i}m_{ij}}.
\]

这会天然偏向长样本。若目标是来源等权，可以先计算每个来源自己的 token 平均损失，再用来源权重组合：

\[
L_{\text{source}}=\sum_i w_i\left(\frac{1}{\sum_jm_{ij}}\sum_{j,k}\ell_{ijk}\right),\qquad \sum_iw_i=1.
\]

两种定义都合理，但优化方向不同。实现时应在日志里同时记录 `source_examples`、`source_target_tokens`、`source_loss_sum` 和 `source_loss_mean`，否则只看到一个总 loss，无法知道配方是按哪种分母执行的。

一个容易漏掉的情况是 gradient accumulation。若每个 micro-batch 单独平均后再平均，短 batch 与长 batch 的权重会被改变。更稳妥的做法是累加 loss numerator 和有效 token denominator，等累积到 optimizer update 时再除；如果框架只能传入已经平均的 loss，就要记录每个 micro-batch 的有效 token 并做加权聚合。

## Curriculum 是一条时间表，不是“从易到难”的口号

可以把第 (i) 个来源在训练进度 (t) 的概率写成 (p_i(t))。最简单的两阶段协议是

\[
p_A(t)=
\begin{cases}
0.8,&t<t_0\\
0.4,&t\ge t_0,
\end{cases}
\qquad p_B(t)=1-p_A(t).
\]

实际系统还要给 (t) 定义清楚：是 optimizer update、消费 token，还是已经完成的 epoch。按 wall-clock 切换会受到排队和吞吐变化影响，按 token 切换更接近训练目标。课程安排还可能与学习率同时变化，因此对照实验至少需要：

| 实验 | 数据顺序 | 总有效 token | 学习率和模型 | 用途 |
| --- | --- | --- | --- | --- |
| baseline | 固定混合比例 | 相同 | 相同 | 估计普通配方 |
| curriculum | 分阶段比例 | 相同 | 相同 | 测试顺序本身 |
| shuffled curriculum | 同一比例但打乱阶段边界 | 相同 | 相同 | 检查阶段边界是否是偶然因素 |

如果课程只在早期降低训练 loss，却让后期的长尾评测退化，不能只报告前期曲线。固定一组不参与采样器选择的评测集，分别报告总体、按来源和按难度的指标。

## 分布式数据管线要审计哪些数

在多 worker、多 rank 和 packing 同时存在时，配置比例与真实比例之间可能有多层偏差。一次审计应沿下面的链路打印计数：

```text
原始记录数
  -> 过滤后记录数
  -> 每来源抽样数
  -> 每来源有效 token
  -> packing 后目标 token
  -> 每个 rank 消费的 token
  -> optimizer update 的累计 token
```

每个 rank 只需记录局部计数，再由 rank 0 汇总；不要让所有 rank 同时向同一个 W&B Run 写同名 step。检查以下不变量：

- `sum(source_target_tokens) == total_target_tokens`，允许的差异只能来自明确的 padding/mask 约定。
- 重新开始 checkpoint 后，数据游标、随机种子和 sampler epoch 不应悄悄回到旧位置。
- `drop_last`、动态 batch 和长度分桶不会让某个来源系统性丢弃更多长样本。
- 多次遍历小数据源时，日志中能区分独立 token 数与重复曝光数。

## 用一个小型矩阵判断是否真的改善

假设有代码、自然语言和数学三个来源。一次配方实验至少保留下表，而不是只写“总验证 loss 下降”：

| 评测桶 | baseline | 新配方 | 变化 | 解释问题 |
| --- | ---: | ---: | ---: | --- |
| 代码 heldout | 1.20 | 1.05 | -0.15 | 是否是代码来源重复曝光带来的 |
| 数学 heldout | 1.80 | 1.83 | +0.03 | 是否牺牲了边界能力 |
| 长上下文 | 2.10 | 1.96 | -0.14 | packing 与长度分布是否变化 |
| 新来源 heldout | 2.30 | 2.29 | -0.01 | 是否只是记住了训练来源 |

把“整体变好”拆成这些桶后，下一步才可能决定是改变比例、增加数据、修复过滤，还是接受某种 trade-off。最小可执行实验不需要先训练大模型；用同一初始化的小模型和固定 token 预算，就可以先验证采样器与 loss 分母是否按预期工作。

## 继续读

继续看[数据质量、去重与治理](data-quality-dedup-and-governance.md)和[评测与消融方法](evaluation-and-ablation-methodology.md)。

## 来源与改编说明

- [Dive into Deep Learning 的数据预处理与训练章节](https://zh.d2l.ai/)提供教材化的 data loader、batch 和训练循环入口，本文沿用其“先明确数据对象，再谈优化”的顺序。
- [Chinchilla: Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)提供参数、训练 token 与计算预算需要一起讨论的原始依据；本文不把其 scaling 关系外推到所有模型或数据配方。
- [Wan 训练报告](https://arxiv.org/abs/2503.20314)的阶段化数据供给用于说明 curriculum 的记录方式；图中比例属于该系统设置，不能直接当作通用配方。

## 自测

1. 两个来源各抽 100 条，但平均长度分别为 100 和 1000 token；按 token 平均 loss 时，两个来源各贡献多少比例的监督位置？
2. 一个 micro-batch 有 1000 个有效 token，另一个有 200 个；若分别平均 loss 再平均，和按有效 token 聚合有什么区别？
3. 课程在第 10B token 切换一次，为什么用 optimizer step 记录切换位置可能在不同吞吐实验中产生不同实际数据量？
4. 新配方总 loss 下降但数学 heldout 上升，下一步应先检查配比、过滤、评测泄漏还是学习率？请写出一项能区分它们的对照实验。


## 对照源码与延伸阅读

- [固定token窗口采样与更新进度的实际接口](https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/train.py)
