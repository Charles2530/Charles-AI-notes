# 训练

训练脚本看起来常常很复杂，真正发生的事情却可以从一轮更新开始追。数据先变成输入和目标，模型产生 logits，loss 把两者放在一起比较，梯度和 optimizer 改动参数。packing、混合精度、多卡和 checkpoint 都是在这条链上增加的新状态。

## 先把一次更新跑通

[一段文本怎样变成参数更新](beginner-model-training-primer.md)用4×4的bigram参数表手算loss与梯度，提供标准库和PyTorch对应代码，再对照nanoGPT。想理解optimizer为何给出不同更新，接着读[目标、优化器与日程](objectives-optimizers-and-schedules.md)。

<!-- systems-workflow-visual -->
```mermaid
flowchart LR
  X["数据与X/Y"] --> F["前向logits"] --> L["loss与分母"] --> G["backward梯度"] --> U["optimizer更新"]
  U --> X
  U --> C["checkpoint与独立评测"]
```

后面的专题都可以沿这条链回看。数据处理改变模型实际看到的样本，数值和分布式策略改变同一轮更新怎样完成，评测则检查更新后的模型有没有在目标任务上变好。

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

如果要把一个教程假设真正变成博士阶段可复查的研究记录，沿[从假设到证据的研究闭环](research-loop-from-hypothesis-to-evidence-1123-2351.md)执行：本站提出机制和反例，`server_monitor` 运行并观察任务，PaperPilot 保存论文证据与笔记版本，最后把条件和失败样本写回正文。

## 对照源码与延伸阅读

- [nanoGPT训练循环](https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/train.py)
- [PyTorch基础优化教程](https://github.com/pytorch/tutorials/blob/5672cd98a7671eda4d9122b10f8eec580f74d03e/beginner_source/basics/optimization_tutorial.py)
## 外部材料怎样接进来

第一次自己写训练循环，可以先看 [Karpathy 的 recipe](https://karpathy.github.io/2019/04/25/recipe/)，照着检查数据，让小模型过拟合，再建立一个不花哨的 baseline。[Hugging Face LLM Course](https://huggingface.co/learn/llm-course/chapter1/1)适合补 tokenizer、Transformer、数据和微调。等这些对象都能在代码里找到，再去读 [Stanford CS336](https://stanford-cs336.github.io/spring2025/) 和 [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)，系统、通信和数据配方才不会变成一串缩写。

遇到任何训练专题，都回到同一条线索。X/Y 从哪里来，logits 如何变成 loss，backward 累了哪些梯度，optimizer 改了谁，checkpoint 保存了什么，独立评测又看到了什么。能把改动放回这条链上，通常比记住框架名称更有用。

## 从能跑到能复查，中间有三道门

一个训练实验至少要过三道门。第一道是小数据过拟合，确认数据、mask、loss 和梯度方向没有错；第二道是中断恢复，比较连续运行与从 checkpoint 恢复后的参数、优化器状态和数据进度；第三道是独立评测，确认训练集 loss 下降没有被写成泛化能力。

| 门 | 最小问题 | 通过时留下什么 |
| --- | --- | --- |
| 数据与梯度 | 一个 batch 能否过拟合，padding 是否改变分母 | batch 快照、loss 分解、梯度范数 |
| 状态恢复 | 第 N 步中断后是否沿同一数据顺序继续 | checkpoint manifest、恢复差值 |
| 结果解释 | 新旧版本哪些样本改变，是否超过 seed 波动 | 配对结果、失败 ID、评测版本 |

这三道门不要求一开始就使用大模型。小模型和短数据足以发现大量接口错误；等它们通过，再增加 token、GPU、并行度和复杂数据。每次扩展只改变一个主要因素，后面才知道性能或质量变化从哪里来。

可以把 `server_monitor` 当作运行证据入口，把 PaperPilot 当作论文条件与解释入口。两边都保存同一个 run ID，正文只引用能回放的字段，不把笔记里的猜测写成实验结果。
