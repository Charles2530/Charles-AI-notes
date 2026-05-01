# 自动微分、激活显存与 Checkpointing

训练大模型时，显存压力往往不是权重单独造成的，而是权重、梯度、optimizer state 和中间激活一起造成的。自动微分让训练变简单，也带来了保存计算图和激活的显存成本。

![Sublinear memory computation graph 原论文图](../assets/images/paper-figures/foundations/sublinear-memory-figure-1-computation-graph.png){ width="760" }

<small>图源：[Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174)，Figure 1。原论文图意：把神经网络前向计算表示成有向计算图，反向传播时需要中间节点的值；若不保存全部中间值，就需要在 backward 前重新计算部分节点。</small>

!!! note "图解：checkpointing 的核心是少存、多算"
    这张图把训练看成一张计算图：前向时每个节点产生中间激活，反向时梯度需要沿图反传并读取这些激活。全保存最省计算但占显存；activation checkpointing 只保存少数关键节点，反向时从最近 checkpoint 重新跑一段前向，换回缺失激活。训练显存不够时，不一定先换模型，可以先看 checkpointing、batch size、sequence length、ZeRO/FSDP、低精度和重算策略。

!!! note "初学者先抓住"
    推理只要算出答案，训练还要记住“答案是怎么来的”，这样 backward 才能算 gradient。显存压力很多时候不是权重本身，而是中间激活、梯度和 optimizer state 一起叠加。

!!! example "有趣例子：解题草稿纸"
    Activation 像解题过程中的草稿。全保存最省时间但占纸；checkpointing 像只保存关键步骤，回头需要时再重算中间过程。它省显存，但会多花计算时间。

!!! tip "学完本页你应该能"
    看到训练 OOM 时，能估计权重、梯度、optimizer state 和 activation 谁是主因；看到 activation checkpointing、ZeRO/FSDP、offload 或长上下文训练时，能说明它们分别在省哪一类状态、代价是什么。

## 1. 自动微分在做什么

现代框架会记录前向计算图，然后自动根据链式法则计算梯度。

```text
y = model(x)
loss = loss_fn(y, target)
loss.backward()
```

这背后会记录：

1. 哪些张量参与了计算；
2. 每个算子的 backward 规则；
3. backward 时需要哪些中间值；
4. 梯度应该累积到哪些参数上。

## 2. 为什么训练比推理更占显存

推理只需要前向输出，训练还需要保存反向传播所需状态。

| 项目 | 推理 | 训练 |
| --- | --- | --- |
| 权重 | 需要 | 需要 |
| 中间激活 | 通常可释放 | 需要保存或重算 |
| 梯度 | 不需要 | 需要 |
| optimizer state | 不需要 | AdamW 通常需要额外状态 |
| checkpoint | 可选 | 必须治理 |

AdamW 训练中，单个参数可能对应权重、梯度、一阶动量、二阶动量等多份状态，因此训练显存通常远高于推理。

## 3. Activation Checkpointing 的核心折中

Activation checkpointing 的思想是：前向时不保存所有中间激活，只保存少数 checkpoint；反向时重新计算丢掉的激活。

它用更多计算换更少显存。

```text
Forward:
  save only selected activations
  discard intermediate activations

Backward:
  recompute missing activations
  compute gradients
```

这在大模型训练中非常常见，尤其是长上下文、多模态和视频模型。

## 4. 一个简单例子

假设有 6 层网络：

```text
x -> L1 -> L2 -> L3 -> L4 -> L5 -> L6 -> loss
```

不做 checkpoint 时，可能保存每一层激活。  
做 checkpoint 时，只保存 \(L2\)、\(L4\)、\(L6\) 的输出。反向传播到 \(L3\) 时，再从 \(L2\) 重新计算 \(L3\)、\(L4\)。

这样显存下降，但训练时间会上升。

## 5. 显存排查顺序

遇到 OOM 时，可以按下面顺序检查：

1. batch size 是否过大；
2. sequence length 或图像分辨率是否过高；
3. activation checkpointing 是否开启；
4. 是否使用 BF16/FP16/FP8 等混合精度；
5. optimizer state 是否可分片；
6. 是否需要 ZeRO、FSDP、TP/PP/CP；
7. 数据加载或缓存是否意外占显存。

## 6. 和后续专题的关系

- [训练稳定性](../training/stability-numerics-and-failure-triage.md)：理解 OOM、NaN、梯度异常。
- [分布式训练与 Checkpoint](../training/distributed-training-and-checkpointing.md)：理解恢复语义和状态分片。
- [Megatron-LM、DeepSpeed 与训练系统栈](../training/megatron-lm-deepspeed-and-open-training-stacks.md)：理解 ZeRO、TP、PP、CP。
- [数值、显存与运行时基础](numerics-memory-and-runtime-basics.md)：理解显存、dtype 和 runtime 的关系。

## 小结

自动微分让训练更容易，但训练系统必须为计算图和中间激活付出显存代价。Activation checkpointing 是最常见的折中：少存一点，多算一点。
