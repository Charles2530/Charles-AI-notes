# 训练图解阅读顺序：先看链路，再看论文图

训练相关论文图很容易看花：数据配方、优化器、并行策略、checkpoint、吞吐曲线、评测表全部挤在一起。这个页面保留“按图找文章”的入口，但阅读顺序不再是图越多越好，而是先确认自己要解释哪一段训练链路。

如果你只想快速补直觉，先读表格里的主题页；如果你正在复盘一个实验，再回到原论文图和来源台账核对细节。本站没有为这个入口生成新图片，也不会用 image2 补图。

## 按训练链路阅读

| 想解决的问题 | 直接阅读 |
| --- | --- |
| 完全不知道模型训练在做什么 | [模型训练从零入门](beginner-model-training-primer.md) |
| 想理解 loss、optimizer 和 LR schedule | [目标函数、优化器与 LR 日程](objectives-optimizers-and-schedules.md) |
| 想分清预训练、SFT、偏好和 RLHF | [预训练、微调与对齐](pretraining-finetuning-alignment.md) |
| 想理解显存、并行和 checkpoint | [分布式训练与 Checkpoint](distributed-training-and-checkpointing.md) |
| 想理解 Megatron / DeepSpeed | [Megatron / DeepSpeed](megatron-lm-deepspeed-and-open-training-stacks.md) |
| 想理解 FP8、低比特训练和数值稳定 | [低比特训练与数值格式](low-bit-training-and-numerics.md) |
| 想理解实验经济学和 scaling law | [Scaling Law 与训练经济学](scaling-laws-and-experiment-economics.md) |
| 想排查训练失败 | [训练稳定性与故障排查](stability-numerics-and-failure-triage.md) |

## 图解材料怎么用

| 看图时先问 | 为什么 |
| --- | --- |
| 图里展示的是数据流、模型结构、并行切分还是评测结果 | 不同图不能支撑同一种结论。 |
| 横纵轴和单位是什么 | 吞吐、FLOP、token、step、wall time 很容易被混用。 |
| 这张图能不能迁移到你的模型规模和硬件 | 论文图常只覆盖特定 batch、context、cluster 和数据配方。 |

## 三种常见读图误区

第一种误区是把训练曲线当成模型能力本身。loss 下降只能说明训练目标在当前数据分布上更容易被拟合，不自动说明下游任务、长上下文、工具调用或多轮对话变好了。看到 loss 曲线时，要同时找验证集、任务评测、数据变化和学习率 schedule。

第二种误区是把系统吞吐图当成算法优势。吞吐提升可能来自 packing、更大的 batch、更短的序列、更好的 checkpoint 策略，也可能只是硬件、并行配置或 dataloader 改了。看到 tokens/s、MFU、GPU utilization 时，要检查它是不是端到端指标，是否包含数据读取、通信、checkpoint 和评测。

第三种误区是把流程图当成完整实现。很多论文图为了讲清主线，会省略数据过滤、重试、异常样本、失败恢复和人工审核。真正落地时，流程图最有价值的部分不是箭头本身，而是它提醒你每个箭头都需要日志、版本、指标和回滚点。

## 从图回到实验证据

读训练图时，可以按四步走。

| 步骤 | 该确认什么 |
| --- | --- |
| 找对象 | 图在解释数据、目标函数、模型结构、并行系统还是评测。 |
| 找变量 | 哪个变量被作者改变，哪些变量保持不变。 |
| 找代价 | 改动把成本放在数据、训练时长、显存、通信还是后处理。 |
| 找外推边界 | 结论是否只在特定模型规模、数据规模、硬件或任务上成立。 |

如果一张图只能说明“作者的系统在作者的设置里更快”，它仍然有价值，但不能直接推出“我的训练也会更快”。这正是图解页要补的判断能力：既看懂图，也知道它没有证明什么。

## 和训练主线的对应关系

训练图大致可以映射到四条主线。数据图通常连接到去重、过滤、混合和 packing；优化图连接到 loss、optimizer、LR schedule 和梯度稳定性；系统图连接到并行、通信、checkpoint 和低比特数值；评测图连接到 ablation、数据泄漏、任务覆盖和回归分析。读图时把它放回这四条主线，信息会少很多噪声。

## 读完后的检查

读完一张训练图，最好能用自己的话复述三句话：作者改了哪一个变量；这个变量为什么会影响 loss、吞吐或评测；如果换成你的数据、模型规模和集群配置，最可能失效的假设是什么。能说清这三句，图才真正进入了你的工程判断；再能指出需要补哪一个 ablation 或日志字段，就可以把它用于实验复盘，也能避免把漂亮曲线误读成通用规律。

## 外部精读起点

- [Chinchilla](https://arxiv.org/abs/2203.15556)：理解参数量、token 数和训练 FLOP 的预算关系。
- [InstructGPT](https://arxiv.org/abs/2203.02155)：理解 SFT、reward model 和 RLHF 数据链路。
- [ZeRO](https://arxiv.org/abs/1910.02054)：理解数据并行为什么会重复保存模型状态。
- [GPipe](https://arxiv.org/abs/1811.06965)：理解 pipeline parallelism 和 micro-batch 如何减少设备空转。
