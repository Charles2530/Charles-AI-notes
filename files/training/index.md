# 训练系统阅读入口：从 loss 到数据、并行和复盘

训练章节关注的是一个模型怎样从数据、目标函数和算力预算里被“塑形”。如果只盯 loss 曲线，很容易漏掉更关键的变量：数据分布是否干净，token 是否被浪费，优化器和 LR schedule 是否匹配，显存和通信是否拖慢集群，评测和 ablation 是否真的能解释改动。

这里会参考优秀中文技术博客和公众号的组织方式，把训练讲成一条可复盘的工程链路：先讲直觉，再讲数据、目标、系统和故障。具体结论以论文、开源训练栈和官方文档为准。

## 第一次读训练

| 你想弄清的问题 | 建议先读 |
| --- | --- |
| 完全不知道训练过程在优化什么 | [模型训练从零入门](beginner-model-training-primer.md) |
| loss、optimizer、LR schedule 如何共同决定更新方向 | [目标函数、优化器与 LR 日程](objectives-optimizers-and-schedules.md) |
| 预训练、SFT、偏好学习、RLHF 的数据链路如何不同 | [预训练、微调与对齐](pretraining-finetuning-alignment.md) |
| 训练为什么会 NaN、发散或越训越差 | [稳定性、数值与故障排查](stability-numerics-and-failure-triage.md) |

## 做一次可复盘实验

| 环节 | 建议先读 |
| --- | --- |
| 数据从采集、去重、过滤到混合 | [数据质量、去重与治理](data-quality-dedup-and-governance.md)、[Scaling、Curriculum 与数据混合](scaling-curriculum-and-data-mixture.md) |
| tokenizer、packing、输入流水线如何影响吞吐 | [输入流水线、Packing 与吞吐治理](input-pipelines-packing-and-throughput-governance.md) |
| 分布式并行和 checkpoint 如何决定可训练规模 | [分布式训练与 Checkpoint](distributed-training-and-checkpointing.md) |
| Megatron、DeepSpeed 等训练栈各自负责什么 | [Megatron / DeepSpeed 与开放训练栈](megatron-lm-deepspeed-and-open-training-stacks.md) |
| 实验记录、指标和治理怎么保留证据 | [W&B 实验追踪与治理](wandb-experiment-tracking-and-governance.md) |

## 扩展到后训练和经济学

| 目标 | 建议先读 |
| --- | --- |
| 估算 token、参数量、FLOP 和预算的关系 | [Scaling Law 与实验经济学](scaling-laws-and-experiment-economics.md) |
| 构建偏好数据、judge model 和后训练数据引擎 | [偏好数据与对齐陷阱](preference-data-and-alignment-pitfalls.md)、[后训练数据引擎与 Judge Model](post-training-data-engines-and-judge-models.md) |
| 低比特、FP8 和数值格式如何进入训练 | [低比特训练与数值格式](low-bit-training-and-numerics.md) |
| 集群运维、checkpoint、失败恢复如何落地 | [集群运维与实验管理](cluster-operations-and-experiment-management.md) |

读训练最要紧的是保留因果证据。一次改动如果没有数据版本、训练配置、吞吐、评测和失败样本，事后就很难判断它到底改善了模型，还是只是换了一个随机种子。
