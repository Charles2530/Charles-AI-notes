# 训练图解旧入口

这个页面保留给旧链接和搜索结果使用；它不再作为训练专题的阅读入口。训练章节现在优先让读者直接进入一篇能独立讲清楚问题的文章，而不是先看一页论文图索引。

## 推荐阅读顺序

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

## 外部精读

- [Chinchilla](https://arxiv.org/abs/2203.15556)：理解参数量、token 数和训练 FLOP 的预算关系。
- [InstructGPT](https://arxiv.org/abs/2203.02155)：理解 SFT、reward model 和 RLHF 数据链路。
- [ZeRO](https://arxiv.org/abs/1910.02054)：理解数据并行为什么会重复保存模型状态。
- [GPipe](https://arxiv.org/abs/1811.06965)：理解 pipeline parallelism 和 micro-batch 如何减少设备空转。
