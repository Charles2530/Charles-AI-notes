# 训练出现NaN时，怎样找到第一个坏值

Loss变成NaN通常已是传播结果。输入、某层输出、loss、梯度或优化器状态中，任何一项先出现非有限值，都可能继续污染后续计算。最有用的证据是触发batch和更新前状态。

## 先用同一个batch重放

保存可复现的输入、mask、随机状态和更新前checkpoint，检查是否在相同位置复现。若每次从新随机batch开始，故障可能暂时消失，却无法判断修复是否有效。

先检查token ID是否越界、有效目标是否为零、输入和参数是否有限。全mask样本若导致以0为分母，问题发生在目标定义，不应靠降低学习率掩盖。

## 一个softmax溢出的完整路径

对logits `[1000,1001]`直接计算指数，会超出常用浮点范围。稳定softmax先减最大值，得到 `[-1,0]`，指数约 `[0.3679,1]`，概率约 `[0.2689,0.7311]`。

```python
import math
z = [1000.0, 1001.0]
e = [math.exp(x - max(z)) for x in z]
print([x / sum(e) for x in e])
```

如果这一层已经产生NaN，后面加loss scaling不能恢复。应先修前向计算或检查输入为什么超出预期范围。

## 沿执行顺序缩小范围

比较各层输入输出，找首个由有限变为非有限的操作；若前向正常，再看loss与backward；若梯度正常但step后权重异常，检查优化器状态、学习率与更新实现。

分布式运行中，一张卡的坏梯度可能通过all-reduce传给全部rank。应记录同步前的局部状态，否则所有卡同时报错会掩盖源头。同步等待也可能只是某rank提前失败，不能自动归因到网络。

<!-- systems-workflow-visual -->
```mermaid
flowchart LR
  I["固定输入与更新前状态"] --> F["逐层forward有限值"] --> L["loss与有效分母"] --> B["backward梯度"] --> O["optimizer状态与写回"]
  F --> E["首个由有限变非有限的位置"]
  L --> E
  B --> E
  O --> E
```

从左到右找最早异常，避免把最后传播到loss的NaN当成发生位置。

## 梯度裁剪可以限制更新信号，但不证明根因消失

全局范数裁剪将g乘以 $\min(1,c/\|g\|)$。例如g=[3,4]、阈值c=2，范数5，裁剪后为[1.2,1.6]。它保留方向并减小幅度。

若g已含NaN，范数计算也可能无效。若异常来自错误数据或损失分母，长期裁剪只是改变其后果。混合精度时先unscale，再按实际梯度裁剪。

![Exploding gradients in recurrent training](../assets/images/paper-figures/foundations/pascanu13-figure-6-exploding-gradients.png)

<small>图源：[Pascanu et al., On the difficulty of training recurrent neural networks](https://proceedings.mlr.press/v28/pascanu13.html)。原图表达参数空间里的 cliff 会让梯度突然变大；读图重点是把 loss spike、`nan/inf` 和 grad norm 异常先当作可定位的数值现象，而不是直接归因于“模型太大”。它不能覆盖所有 Transformer 训练事故，只提供理解梯度爆炸和 clipping 的直觉入口。</small>

## 恢复后跳变先检查状态完整性

相同权重配上不同Adam矩、学习率计数、RNG或数据位置，会走出不同轨迹。用一个短实验对比连续运行与中断恢复后的几步，在相同容差下检查loss、更新与输入顺序。

如果运行本来含非确定性算子，逐bit相同未必是合理要求，但差异范围应可解释。修复完成的证据是触发输入与相关短程回放通过，而非长作业暂时没再报错。


## 对照源码与延伸阅读

- [AMP、clip、step与checkpoint的真实顺序](https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/train.py)
- [稳定softmax减最大值的实现](https://github.com/triton-lang/triton/blob/cd2e01c6686d6503802e6cdfac56d5a5dfdb27a1/python/tutorials/02-fused-softmax.py)
