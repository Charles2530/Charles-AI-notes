# 模型训练从零入门

训练不是“把数据喂给模型”。训练是一条生产线：数据被采样成 batch，模型给出预测，loss 衡量预测和目标的差距，反向传播计算梯度，优化器更新参数，评测和 checkpoint 决定这次实验能不能相信。

## 核心问题

如果你只看一条 loss 曲线，很容易误以为训练就是让数字下降。真正的问题是：**模型是否在正确数据上，用正确目标，以可恢复、可比较、可扩展的方式学习了你想要的能力**。

这个问题可以拆成六步：

```text
数据 -> batch -> forward -> loss -> backward -> optimizer step -> eval/checkpoint
```

其中任何一环出错，loss 都可能继续下降，但模型变得不可用。

## 一次训练迭代到底发生了什么

最小监督学习形式是：

\[
\min_\theta \; \mathbb{E}_{(x,y)\sim \mathcal{D}}[\ell(f_\theta(x), y)]
\]

这里 \(x\) 是输入，\(y\) 是目标，\(f_\theta\) 是参数为 \(\theta\) 的模型，\(\ell\) 是 loss。白话说：从数据分布 \(\mathcal{D}\) 里抽样，让模型预测，再用 loss 惩罚错误，最后更新参数。

一次训练 step 可以写成：

```text
batch = sample(dataset)
pred = model(batch.inputs)
loss = criterion(pred, batch.targets)
grad = backward(loss)
optimizer.update(parameters, grad)
```

LLM 的预训练只是这个框架的特殊形式：输入是一段 token，目标通常是下一个 token；扩散模型的目标可能是噪声、score 或 velocity；偏好训练的目标可能是让 chosen response 比 rejected response 更高分。

## Batch 不是越大越好

Batch 是一次参数更新看到的数据集合。batch 大，梯度噪声更小，吞吐可能更好；batch 过大，优化动态会变，泛化和稳定性可能受影响。工程里还要区分 micro-batch、global batch、gradient accumulation：

| 名词 | 含义 | 常见坑 |
| --- | --- | --- |
| micro-batch | 单张卡一次 forward/backward 的样本数 | 太大会 OOM，太小会设备吃不满。 |
| gradient accumulation | 多个 micro-batch 累积后再 step | loss scaling、日志和学习率步数容易对错。 |
| global batch | 所有卡、所有累积合起来的有效 batch | 改卡数时如果不调 LR，训练曲线会变。 |

训练日志里的 `step` 不一定代表同样数量的 token。比较两个 run 时，优先看 consumed tokens、effective batch、learning rate schedule、数据版本和 eval 指标。

## Loss 是训练信号，不是最终答案

交叉熵预训练里，模型输出每个词的概率分布，目标是让真实下一个 token 的概率更高。一个常见形式是：

\[
\mathcal{L}_{\text{CE}} = -\sum_t \log p_\theta(x_t \mid x_{<t})
\]

这行公式表示：每个位置都问“模型给真实 token 的概率是多少”，概率越低，惩罚越大。它能训练语言建模能力，但不能直接保证模型会遵守指令、会安全拒答、会使用工具，或者会在机器人上稳定执行动作。

所以大模型训练通常分阶段：

| 阶段 | 主要学习什么 | 典型风险 |
| --- | --- | --- |
| 预训练 | 语言、知识、模式、代码和世界统计 | 数据污染、能力偏差、token 不够。 |
| SFT | 按指令回答、格式和任务示范 | 过拟合模板，覆盖底座能力。 |
| 偏好/RL | 偏好排序、安全边界、工具行为 | reward hacking，judge 偏差。 |
| 蒸馏 | 把强模型行为转给小模型 | 复制错误，丢掉探索能力。 |

## 训练系统先算三张账

第一张是显存账。训练时显存不只存参数，还要存梯度、优化器状态、激活、中间 buffer 和通信缓存。AdamW 里，优化器状态通常比参数还大；反向传播还需要保留或重算激活。

![ZeRO memory optimization stages 原论文图](../assets/images/paper-figures/training/zero-memory-stages.png){ width="860" }

<small>图源：[ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)，Figure 1。原图表达 ZeRO 把普通数据并行中重复保存的 optimizer states、gradients、parameters 分阶段切到不同 data-parallel ranks。本站用它解释训练显存账：省显存不是一个开关，而是决定哪些状态常驻、哪些分片、哪些按需 gather。</small>

第二张是时间账。训练慢可能是 GPU 算不满，也可能是通信、输入管线、checkpoint I/O、pipeline bubble 或 kernel launch 造成的。

![GPipe pipeline parallelism 原论文图](../assets/images/paper-figures/training/gpipe-pipeline-parallelism.png){ width="860" }

<small>图源：[GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965)，Figure 2(c)。原图表达把 mini-batch 切成 micro-batch 后，不同 accelerator 可以同时处理不同模型分段。本站用它说明 pipeline parallelism 主要在处理设备空转和时间排布，而不是直接改变目标函数。</small>

第三张是质量账。吞吐提升、显存降低、MFU 变高，都不能自动证明模型更好。训练系统改动必须回到 validation、downstream eval、人工评估、失败桶和可复现实验资产。

## Scaling law 的直觉

训练预算不是只堆参数。Chinchilla 的结论提醒我们：固定计算量下，模型大小和训练 token 数需要配平。

![Chinchilla IsoFLOP curves 原论文图](../assets/images/paper-figures/training/chinchilla-isoflop-curves.png){ width="920" }

<small>图源：[Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)，Figure 4。原图表达在固定 FLOP 预算下，不同模型大小有不同最终 loss，曲线谷底对应 compute-optimal 的参数/token 配比。本站用它解释：模型太大但 token 不够会欠训练，模型太小则容量不足。</small>

这张图不需要你记住某个具体数字。它真正告诉你：训练设计是一笔预算分配，参数、token、数据质量、上下文长度和训练步数必须一起看。

## Checkpoint 是训练资产，不只是备份

一个可信 checkpoint 应该能回答：它来自哪份代码、哪份数据、哪个 config、哪个 seed、哪个 optimizer state、哪个 tokenizer、哪个训练进度。只保存模型权重，往往不足以恢复训练或解释结果。

工程上至少要保存：

| 资产 | 为什么重要 |
| --- | --- |
| model weights | 推理和继续训练需要。 |
| optimizer / scheduler state | 恢复训练曲线需要。 |
| tokenizer / processor | token id 变了，模型输入就变了。 |
| data manifest | 证明训练看过什么、没看过什么。 |
| config / environment | 复现实验和排查回归需要。 |
| eval artifacts | 证明模型质量变化不是错觉。 |

## 常见误解

| 误解 | 更准确的说法 |
| --- | --- |
| loss 下降就代表模型更好 | loss 只是训练目标上的信号，还要看 eval 和任务失败。 |
| GPU 利用率 100% 就代表训练高效 | 可能只是设备忙，不代表 Tensor Core、通信和数据管线高效。 |
| batch 改大只影响速度 | batch 会改变优化噪声、LR 语义和收敛行为。 |
| checkpoint 只是防断电 | checkpoint 是实验可恢复和可审计的证据链。 |

## 外部精读

- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)：补模型结构直觉。
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)：理解参数和 token 的预算配平。
- [ZeRO](https://arxiv.org/abs/1910.02054)：理解大模型训练显存账。
- [GPipe](https://arxiv.org/abs/1811.06965)：理解 pipeline parallelism 的时间账。
- [DeepSpeed ZeRO tutorial](https://www.deepspeed.ai/tutorials/zero/)：把论文概念接到工程配置。
