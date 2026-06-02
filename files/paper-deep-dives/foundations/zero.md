# ZeRO：为什么数据并行会浪费这么多显存

Checked date: 2026-06-02  
Repro status: 论文级理解；本站未复现实验，只引用原论文图和公开 DeepSpeed 文档。

ZeRO 解决的问题很直接：普通 data parallel 会在每张 GPU 上重复保存完整模型状态。模型一大，显存先被 optimizer states、gradients、parameters 吃掉，batch 和模型规模都上不去。ZeRO 的思路是把这些状态按 data-parallel ranks 分片，减少冗余。

## 论文解决的问题

普通数据并行里，每张卡都有完整参数、完整梯度和完整优化器状态。对于 AdamW，优化器通常还要保存一阶矩和二阶矩，显存占用非常可观。

如果每张卡都保存同一份状态，扩展 GPU 数量不会降低单卡模型状态显存。ZeRO 的核心问题就是：哪些状态真的需要每张卡常驻？哪些可以切分，需要时再通信拿回来？

## ZeRO 三个阶段

![ZeRO memory optimization stages 原论文图](../../assets/images/paper-figures/training/zero-memory-stages.png){ width="860" }

<small>图源：[ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)，Figure 1。原图表达 ZeRO-DP 三个阶段分别分片 optimizer states、gradients、parameters，并比较单设备模型状态显存。本站用它说明：ZeRO 是逐步减少 data parallel 冗余状态，而不是一个单一开关。</small>

三阶段可以这样读：

| 阶段 | 分片对象 | 直觉 |
| --- | --- | --- |
| ZeRO-1 | optimizer states | Adam 的动量和方差不必每张卡都完整保存。 |
| ZeRO-2 | optimizer states + gradients | 梯度也可以按 DP rank 分片和 reduce。 |
| ZeRO-3 | optimizer states + gradients + parameters | 参数本身也分片，需要计算时 gather。 |

越往后省显存越多，但通信和实现复杂度也越高。ZeRO-3 不是免费午餐：forward/backward 时参数 gather、通信 overlap、checkpoint 保存和恢复都会更复杂。

## 显存账怎么拆

训练显存常被误说成“模型多大”。更准确的拆法是：

```text
parameters + gradients + optimizer states + activations + temporary buffers + communication buffers
```

ZeRO 主要处理前三类模型状态。Activation checkpointing 处理激活；mixed precision 处理 dtype；pipeline/tensor parallel 改变模型切分；offload 把部分状态搬到 CPU/NVMe，但会引入带宽和延迟问题。

## ZeRO 和 Megatron 的关系

Megatron-LM 更偏模型并行：把 Transformer 的矩阵、层或序列切到不同 GPU 上。ZeRO 更偏数据并行里的状态分片：把每个 DP rank 上重复保存的状态拆开。两者常常一起用，因为一个解决模型怎么拆，一个解决状态怎么放。

工程上不要把 ZeRO、FSDP、TP、PP 混成“分布式训练技巧”。它们动的是不同账本：参数计算、状态存储、激活生命周期、通信路径和 checkpoint 语义。

## 论文结果怎么读

ZeRO 的实验重点不是某个下游任务分数，而是证明在给定硬件上能训练更大的模型，并保持可接受吞吐。读这类系统论文时要看：最大模型规模、单卡显存、吞吐、batch size、通信开销和扩展效率。

如果一个系统只展示“显存省了”，但吞吐掉得很厉害，或者 checkpoint 不可恢复，工程价值就要打折。ZeRO 真正可复用的是状态生命周期思想，而不是某个固定配置。

## 可复用经验

| 经验 | 工程含义 |
| --- | --- |
| 先拆显存账 | 不知道谁占显存，就不知道该用 ZeRO、checkpoint 还是低精度。 |
| 省显存会换通信 | ZeRO 阶段越深，越要看通信 overlap 和网络拓扑。 |
| checkpoint 语义会变 | 分片状态保存、恢复和转换都要进入实验资产管理。 |
| 和模型并行组合 | 大模型常需要 ZeRO + TP/PP/CP，而不是单独一种策略。 |

## 边界

ZeRO 不会自动改善模型质量，也不解决数据、目标函数、优化器稳定性和评测问题。它让更大模型或更大 batch 能放进硬件，但训练是否有效仍取决于完整实验设计。

## 外部精读

- [ZeRO paper](https://arxiv.org/abs/1910.02054)
- [DeepSpeed ZeRO tutorial](https://www.deepspeed.ai/tutorials/zero/)
- [PyTorch FSDP documentation](https://pytorch.org/docs/stable/fsdp.html)
- [Megatron-LM paper](https://arxiv.org/abs/1909.08053)
- [DeepSpeed documentation](https://www.deepspeed.ai/)
