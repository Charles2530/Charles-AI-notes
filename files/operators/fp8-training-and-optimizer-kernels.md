# FP8 训练与优化器 Kernel

低精度话题里，`FP8` 和优化器 kernel 值得单独拿出来讲，因为它们主要影响的是训练主路径，而不是单纯的推理压缩。权重量化更多关注部署态显存和带宽；而 `FP8` 训练与 fused optimizer kernel 关心的是：在大规模训练中，如何把前向、反向、梯度、主权重、优化器状态和数值缩放组织成一套既快又稳的执行路径。

这页建议和 [低比特训练与数值格式](../training/low-bit-training-and-numerics.md)、[分布式训练与 Checkpoint](../training/distributed-training-and-checkpointing.md)、[低精度与量化 Kernel](low-precision-and-quantized-kernels.md) 一起读。训练页解释数值格式和稳定性边界，分布式页解释状态切分与恢复，本页则关注这些约束落到 kernel、scale metadata 和 optimizer step 时会怎样影响吞吐。

!!! note "初学者先抓住"

    FP8 训练不是把 dtype 改小，而是重写训练主路径里的数值契约。前向、反向、scale、梯度、optimizer state、主权重和 checkpoint 都要配合，否则省下的带宽可能换来不稳定的更新。

!!! note "难点解释：为什么优化器 kernel 也重要"

    大模型训练里 AdamW 等优化器会频繁读写参数、一阶动量、二阶动量和梯度。若这些更新分散成很多小 kernel，会浪费带宽和 launch；fused optimizer kernel 能把多次读写合并，但也要小心数值精度和状态恢复。

## 1. 为什么 FP8 训练和推理量化不是一回事

推理量化更多是在“压缩已有模型”；  
FP8 训练则是在“用更低精度直接完成学习过程”。  
这意味着它要同时面对前向数值范围、反向梯度范围、累加精度、scale 更新和 optimizer state 的保存格式。

也就是说，FP8 训练不是只换一个 dtype，而是整条训练 pipeline 的重构。

## 2. FP8 训练真正依赖什么

要让 FP8 训练可用，通常至少需要合适的 per-tensor 或更细粒度 scaling，某些路径保留更高精度累加，稳定的 cast / dequant / requant kernel，对异常值和 outlier 有合理处理，并让训练框架、kernel、optimizer 路径配套。

因此 FP8 更像系统级能力，而不是某个局部技巧。

## 3. Scale 元数据为什么是训练的一部分

在 FP8 路径里，真正参与计算的并不只是张量本身，还包括 scale、history 或统计量、cast 策略、溢出检测或 guard。这些额外元数据会影响 kernel 输入输出格式、张量布局、fused path 设计和 checkpoint 状态。

所以 FP8 训练 kernel 的核心不只是低精度乘法，而是“低精度张量 + 缩放元数据”的联合管理。

## 4. 为什么优化器 kernel 也是主战场

在大模型训练里，优化器 step 的成本不容忽视，因为参数很多、optimizer state 很大、每步都要更新，还常与 ZeRO / FSDP 分片状态耦合。

如果优化器实现仍然是多次读写参数、动量和方差，并拆成分散的 elementwise kernel，训练吞吐就会被明显拖慢。

## 5. Fused Optimizer 的意义

Fused optimizer kernel 的核心，是把原本分散的多步 elementwise 操作合并起来，例如读梯度、更新一阶矩和二阶矩、计算校正、应用权重衰减、更新参数。

### 5.1 为什么这比看起来更重要

因为这些步骤本质上都是内存带宽敏感的 elementwise 操作。  
如果拆成多个 kernel，同一组参数和状态会被反复读写，launch 次数也会增加，大模型下总时间会很可观。

## 6. FP8 与优化器之间的精度分层

**一个很现实的问题是**：  
即使前向和反向更多地走 FP8，优化器状态也未必适合全面降到同样精度。

常见原因是 moment 累积对精度敏感，长时间训练误差会积累，不同参数组的动态范围差异也很大。

因此实际系统常采用更复杂的精度分层：前向低精度，累加更高精度，optimizer state 保留更稳定格式，最终参数更新路径再做折中。

## 7. 训练态低精度 kernel 的常见热点

常见热点包括 FP8 GEMM、cast / scale 相关 kernel、gradient unscale / clip、optimizer step、master weight 更新，以及分布式梯度聚合前后的数据格式转换。这些路径如果彼此不协调，就会让理论上的 FP8 收益被大量格式转换吃掉。

## 8. 一个形象比喻

如果把训练看成一条长期运转的工厂流水线，那么 FP8 训练不只是把原材料改成更轻的包装，而是要求整个仓储、计量、称重和装配流程都能适应更轻、更敏感的零件。优化器 kernel 则像最后的总装和校准工位，如果这一步仍然非常笨重，前面节省下来的吞吐也会被吃掉。

## 9. 小结

FP8 与优化器 kernel 的主题，核心不是“更低精度”，而是“训练主路径是否能以更低精度仍然稳定运行”。这要求 kernel、scale、优化器状态、mixed precision 策略和分布式训练框架一起工作。  
因此它是训练系统能力的一部分，而不只是单个算子技巧。

## 工程收束

FP8 训练与优化器 kernel 要围绕 scale 管理、累加精度、统计同步、优化器状态和回退路径来验收。只看显存下降会漏掉 scale 漂移、训练/推理 FP8 路径不一致等问题；上线前应按层验证 FP8 可承受度，保留 BF16 回退，监控 scale 分布，并做长跑数值回归。
