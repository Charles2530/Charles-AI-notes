# 训练与对齐中的问题定位

训练loss下降、模型回答却没有改善时，先确定这个loss实际奖励什么。数据目标错位、归约权重不对和评价协议变化，都可能形成“优化成功、任务没改善”的结果。

## Loss很快接近零，应该高兴还是检查数据

先让输入和标签可读。自回归任务若标签没有右移，模型可能只学复制当前位置；SFT若mask只留下模板或固定结束符，loss也可能很低，却没有训练真正答案。

抽取几条经过全部processor后的样本，逐位置打印输入、目标与loss mask；再检查训练和验证是否共享原文、用户或相邻片段。只有目标与划分确认后，曲线才有可解释性。见[数据怎样成为样本](../foundations/data-and-dataset-basics.md)。

## 梯度累积为什么不总等于大batch

若两个micro-batch分别有2和8个有效token，对各自平均loss再等权平均，会给两组各50%权重；真正的全token平均需要20%和80%。累积次数相同不代表目标相同。

此外BatchNorm、随机增强、dropout状态和优化器更新次数也可能不同。先用固定小输入比较累积后的梯度，再加入随机与分布式因素。loss缩放应与DDP平均约定配套，不能在各层随意除以batch或卡数。见[分布式训练](../training/distributed-training-and-checkpointing.md)。

## 混合精度出了NaN，梯度裁剪能解决吗

裁剪限制有限梯度的范数，不会把已出现的NaN修成正确值。先定位第一个非法张量，检查输入、除零、log参数、softmax全mask和溢出。若使用loss scaling，应先unscale，再按真实梯度阈值裁剪。

BF16指数范围较大也不保证稳定，较低尾数精度仍可能丢小更新；FP32累加、主参数和优化器状态配置也影响行为。把学习率调小可以作为对照，但不能代替定位。见[训练稳定性](../training/stability-numerics-and-failure-triage.md)。

## LoRA参数很少，为什么训练显存仍高

LoRA减少需要优化的增量参数及相关梯度状态，模型仍需执行基础网络的前向与反向传播路径，并保存部分激活。长序列、大batch和注意力中间状态可能继续占主要显存。

QLoRA进一步量化冻结基础权重，但反向仍要经过解量化后的计算来更新适配器，不能解释成“所有训练均在4-bit完成”。先按权重、梯度、优化器与激活分别量占用。见[QLoRA与量化训练](../quantization/qlora-and-quantized-training.md)。

## SFT、DPO和RL究竟在使用什么监督

SFT学习示范目标的条件概率；DPO在特定推导与约定下用偏好对比较策略和参考模型的log概率差；RL根据策略采样结果和奖励更新行为。它们都可改变回答习惯，但数据来源与目标不同。

例如同题“简短正确”和“冗长错误”被偏好标成后者，优化再稳定也会学到错误偏好。先确认胜负标签的依据、拒答与长度混杂、参考模型和token归约，再讨论算法。见[偏好数据与对齐陷阱](../training/preference-data-and-alignment-pitfalls.md)。

## Reward上升，为什么任务成功率反而下降

奖励可能只是任务的代理。奖励模型偏爱长答案时，策略可以通过变长获得更高分，而不提高正确率；规则奖励只检查最终格式时，也可能遗漏真实行为。

固定一组独立任务，用真实验证器或人工复核比较奖励与任务结果，并按长度、难度和来源分组。GRPO组内全相同奖励还可能没有可用相对信号，需要查看采样分布和奖励离散程度，不能仅靠提高batch解决。见[强化学习与后训练](../reinforcement-learning/index.md)。

## 恢复checkpoint后为什么不完全复现

参数和优化器状态只覆盖一部分训练状态。数据顺序、采样器位置、随机数状态、混合精度scaler、scheduler和分布式实现都可能影响下一步。某些教学checkpoint只支持大致继续训练，没有承诺逐步完全一致。

先比较恢复前后同一个固定batch的预测、梯度和一次更新，再检查数据进度与随机状态。即使记录完整，跨硬件或不同kernel仍可能带来数值差异。见[checkpoint与恢复](../training/distributed-training-and-checkpointing.md)。

## 快速自测

从一次“loss 下降但任务没改善”的实验中列出四项记录：样本经过的处理、loss 分母、独立评测、恢复状态。逐项说明哪一项缺失会让结论失去解释。完整实验方法见[评测与消融](../training/evaluation-and-ablation-methodology.md)。

**来源与改编说明**：自回归目标和 checkpoint 约定对照 [PyTorch distributed checkpoint 文档](https://pytorch.org/docs/stable/distributed.checkpoint.html)，偏好优化可查 [DPO 论文](https://arxiv.org/abs/2305.18290) 与 [TRL 文档](https://huggingface.co/docs/trl/)。本页示例用于定位接口，不是算法效果报告。
