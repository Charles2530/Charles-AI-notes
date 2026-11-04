# 训练

训练让数据通过损失和梯度改变参数。下一词预测把一段文本变成错开一位的X/Y，模型输出词表logits，交叉熵衡量目标概率，optimizer执行更新。数据、数值精度和多卡通信都围绕这条计算链增加约束。

## 先看一次真正可复算的更新

[一段文本怎样变成参数更新](beginner-model-training-primer.md)用4×4的bigram参数表手算loss与梯度，提供标准库和PyTorch对应代码，再对照nanoGPT。想理解optimizer为何给出不同更新，接着读[目标、优化器与日程](objectives-optimizers-and-schedules.md)。

<!-- systems-workflow-visual -->
```mermaid
flowchart LR
  X["数据与X/Y"] --> F["前向logits"] --> L["loss与分母"] --> G["backward梯度"] --> U["optimizer更新"]
  U --> X
  U --> C["checkpoint与独立评测"]
```

每个专题都能落到这条链的一个位置。数据处理先改变输入，数值与分布式策略再改变同一更新的执行方式。

## 决定模型实际看到什么

[输入流水线](input-pipelines-packing-and-throughput-governance.md)解释X/Y、目标mask、padding和packing边界。[数据吞吐](data-systems-and-optimization.md)连接生产速度与GPU等待；[去重和过滤](data-quality-dedup-and-governance.md)说明处理如何改变分布；[数据配比](scaling-curriculum-and-data-mixture.md)计算来源权重、重复曝光和顺序。

预训练、SFT、偏好与奖励的区别，见[同一问答的不同监督](pretraining-finetuning-alignment.md)。进一步看[偏好数据](preference-data-and-alignment-pitfalls.md)、[失败轨迹与judge](post-training-data-engines-and-judge-models.md)以及[MTP与投机解码](mtp-and-speculative-decoding.md)。

## 把同一更新扩展到真实系统

| 当前问题 | 正文 |
| --- | --- |
| 参数更新在低精度里消失 | [低比特数值路径](low-bit-training-and-numerics.md) |
| Loss突然变NaN | [首个坏值定位](stability-numerics-and-failure-triage.md) |
| 模型状态放不下、多个rank目标不一致 | [分布式更新与checkpoint](distributed-training-and-checkpointing.md) |
| 分不清TP与状态分片的框架分工 | [从MLP理解训练栈](megatron-lm-deepspeed-and-open-training-stacks.md) |
| 慢rank、故障与保存间隔 | [集群作业时间线](cluster-operations-and-experiment-management.md) |

## 用实验决定下一步

[Scaling与训练预算](scaling-laws-and-experiment-economics.md)推导参数和token的交换关系；[评测与消融](evaluation-and-ablation-methodology.md)区分观察、统计证据与因果归因；[实验记录](wandb-experiment-tracking-and-governance.md)从一次真实数值更新说明曲线怎么记。图片出处集中在[来源索引](image-sources.md)。

## 对照源码与延伸阅读

- [nanoGPT训练循环](https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/train.py)
- [PyTorch基础优化教程](https://github.com/pytorch/tutorials/blob/5672cd98a7671eda4d9122b10f8eec580f74d03e/beginner_source/basics/optimization_tutorial.py)
