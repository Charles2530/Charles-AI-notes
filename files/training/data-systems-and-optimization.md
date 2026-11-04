# 每秒多少训练token真正到达GPU

GPU完成一个训练step后，需要下一批已经解码、分词并组装好的张量。数据系统太慢会让设备等待；数据提前准备好，但大部分位置是padding，也会浪费预算。两种情况在“GPU很忙”这一指标里不容易区分。

## 先定义每步处理的量

设一批4条序列，长度分别为100、200、300、400，统一pad到400，总槽位1600，真实token1000，padding占37.5%。若只有回答部分参与loss，有效监督token还会更少。

因此至少区分物理处理槽位、非padding token和参与loss的目标token。比较两个packing方案时，不能一个报告槽位tokens/s，另一个报告监督tokens/s。

![Chinchilla tokens vs parameters](../assets/images/paper-figures/training/chinchilla-tokens-vs-params.png){ width="760" }

<small>图源：[Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)。原图表达在固定计算预算下参数量和训练 token 数需要共同选择；读图重点是把数据系统视为训练优化的一部分：有效 token、采样、packing 和吞吐都会改变实际学习预算。它不能证明某个数据配方最优。</small>

## 生产速度怎样变成等待

教学设定中，GPU每批计算100毫秒，loader每批准备150毫秒。若完全串行，一批250毫秒；有足够预取后，稳态仍受较慢的150毫秒限制。把GPU算得更快到50毫秒，对稳态吞吐没有帮助，等待反而更明显。

若loader准备80毫秒、GPU计算100毫秒，预取可把大部分准备时间隐藏在计算里。Buffer能平滑抖动，不能长期弥补生产速度低于消费速度。

## 一条样本经过哪些变化

原始记录先解码、规范化、去重与切分，再分词和采样，最后形成X/Y与mask。每一步都可能改变训练分布。按文档随机采样和按token位置随机采样，长文档得到的权重不同；按来源样本数配比，也不等于按有效token配比。

nanoGPT的 `get_batch` 从预分词memmap中随机取固定窗口，X与Y错开一位。它的输入路径简单，适合看清模型接口，但没有覆盖一般多模态解码、流式分片与复杂packing。完整边界见[输入流水线](input-pipelines-packing-and-throughput-governance.md)。

## 吞吐优化要保持同一个目标

把两条序列拼在一起可以减少padding，但要同时处理attention可见范围、位置编号与目标移位。若原本独立的问答拼接后互相看见，模型目标已经改变，不能只当作无损提速。

改变batch或梯度累积时，也要核对有效token总数与loss分母。只保持张量shape相同，不能保证每步优化同一个经验目标。

## 中断恢复需要已消费进度

预取器可能已经读到下一批，模型却还没完成当前更新。保存最远读取位置会跳过尚未训练的数据；只保存随机种子也无法重建所有buffer顺序。

将已提交optimizer step与已消费样本位置对应起来，再说明如何恢复shuffle、worker和预取状态。具体实现可以选择精确重放或允许有限重复，但需要明确语义，并用短程中断实验验证。

最终把每步耗时拆成数据等待、计算、通信和保存，辅以监督token计数。这样才能判断下一次优化应改存储、预处理、packing还是模型执行。


## 对照源码与延伸阅读

- [memmap读取、get_batch与预取位置](https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/train.py)
