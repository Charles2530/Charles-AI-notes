# 基础计算与 Transformer 问答

这页适合在代码能运行、结果却不符合预期时查阅。每个问题从一个可观察的现象出发，说明应该比较哪些数值，再链接到完整推导。

## Shape都对，为什么算出的结果仍然错

Shape只记录轴长度，不记录轴含义。给2×2矩阵加`[10,20]`，默认沿列广播；如果原意是每行加一个数，需要`[[10],[20]]`。两种输出都是2×2，单靠shape断言发现不了错误。

先用轴长度互不相等的小例子，例如batch2、length3、hidden5，再赋予容易追踪的值。注意力应沿key轴softmax，每个query行的有效权重和为1；padding和禁止位置应按约定为零。输入轴交换错误时，具体元素比较通常比最终loss更敏感。完整例子见[张量与计算图](../foundations/tensors-shapes-and-computation-graphs.md)。

## 为什么反向梯度要相加，不能覆盖

设 \(L=x^2+x\)，x=2时梯度是5。平方分支贡献4，直接相加分支贡献1。如果局部反向写`grad = contribution`，其中一条会覆盖另一条。共享embedding、残差、多项损失都有同样需求。

另一种累积发生在多次backward之间。框架常把参数梯度保留到显式清空，所以梯度累积可以实现更大的有效batch；忘记清空也可能无意中混入上一步梯度。它与单次图内分支累积不是同一个层次。见[自动微分](../foundations/autograd-activation-checkpointing-and-memory.md)。

## `eval()`以后为什么还会记录梯度

`eval()`切换模块行为，例如关闭dropout随机屏蔽、让BatchNorm使用运行统计；它不改变参数的`requires_grad`，也不关闭图记录。普通推理通常还会使用`no_grad()`或适当的inference模式。

反过来，`no_grad()`也不会自动把模型切到eval状态。若训练模式的dropout仍在，输出依然可能随机。比较训练与推理差异时，分别检查模块状态、梯度模式和随机数。重计算、共享状态和编译优化又各有约定，不能把一个开关概括成“进入推理”。

## 因果训练一次算完整序列，为什么不算偷看答案

输入`[BOS,A,B]`，标签`[A,B,EOS]`。B位置只看B及其之前的输入，用来预测EOS；A位置无法看B，因此不能直接读取自己的目标。训练完整序列已知，所以可把各位置并行计算，mask限制它们的依赖。

把输入与标签不小心设成完全相同，或者在softmax后才草率处理未来位置，都可能让loss异常低。先逐位置打印输入、目标和可见key，再看[Transformer完整计算](../foundations/transformer-attention-and-tokenization.md)。

## Attention权重很高，能证明模型靠这个词作答吗

权重控制某个头、某一层如何混合value。最终输出还经过其他头、输出投影、残差和后续层。高权重位置的value可能很小，其他路径也可能补偿。因此权重可用于观察计算，却不单独构成因果解释。

要测试某个信息是否影响答案，可以设计保持其他条件尽量不变的替换或遮挡对照，再观察结果，但输入修改也可能引入分布变化。解释时应报告实际干预，而非把可视化亮色直接等同于“重要性”。

## 用KV cache后输出为什么可能不同

在同一标准因果模型、相同输入、位置和数值语义下，缓存应复用本来就不随未来token变化的历史K/V。若差异很大，先核对position id、mask、缓存长度、batch重排和前缀条件；如果只是小误差，可能来自不同矩阵形状和浮点归约顺序。

对照时固定解码方式，并在每一步比较logits，而非只看最终文本。接近并列的候选会把微小数值差异放大为后续完全不同的生成。缓存是否正确与最终文字是否相同，需要分层检查。见[位置、mask与上下文](../foundations/positional-encoding-masks-and-context.md)。

## Greedy、采样与“最好答案”是什么关系

Greedy每步选最高概率，不保证整段概率最高；整段概率最高也不保证事实正确。Top-p等采样则先修改候选分布，再随机选择，日志中的原始概率与实际行为概率可能不同。

如果任务有确定性验证器，例如代码测试，可以生成多个候选再检查；这增加覆盖和计算成本，并不改变原模型保证。验证器检查了哪些行为、有没有漏掉边界，必须另行说明。见[生成与解码](../foundations/generation-and-decoding-basics.md)。

## 快速自测

挑一个“结果看起来对但可能有错”的张量或生成问题，分别写出轴含义、mask、随机状态、验证器和最终任务指标。若只能给出最终 loss 或一张 attention 图，说明还缺哪条证据。完整计算路径见[Transformer 与 attention](../foundations/transformer-attention-and-tokenization.md)。

## 五分钟最小排查

下面的代码不运行模型，只把最容易被 shape 掩盖的两个错误显式打印出来。它使用互不相等的轴长度，避免 `[2, 2]` 这类偶然合法的例子把问题藏起来。

```python
import numpy as np

# b=2, l=3, d=4，每个轴都填入可追踪的整数
x = np.arange(2 * 3 * 4).reshape(2, 3, 4)
feature_bias = np.array([100, 200, 300, 400])
position_bias = np.array([10, 20, 30]).reshape(1, 3, 1)

by_feature = x + feature_bias
by_position = x + position_bias
assert by_feature.shape == x.shape
assert by_position.shape == x.shape
assert by_feature[1, 2, 3] == x[1, 2, 3] + 400
assert by_position[1, 2, 3] == x[1, 2, 3] + 30

# 只允许当前位置读取自己和过去的 key
scores = np.arange(9, dtype=float).reshape(3, 3)
future = np.triu(np.ones((3, 3), dtype=bool), k=1)
masked = np.where(future, -np.inf, scores)
weights = np.exp(masked - np.max(masked, axis=-1, keepdims=True))
weights /= weights.sum(axis=-1, keepdims=True)
assert np.allclose(weights[np.triu(future)], 0.0)
assert np.allclose(weights.sum(axis=-1), 1.0)
```

这段检查只证明广播轴和因果 mask 的数值约定，不能证明模型学到了正确任务。若真实代码的 `logits` 与这里的结果不一致，先固定一个 batch、一个位置和一个 head 打印中间张量，再检查数据布局、mask 语义和归约轴；不要先调学习率或增大模型。

**来源与改编说明**：张量和自动微分的 API 约定对照 [PyTorch Tensor 文档](https://pytorch.org/docs/stable/tensors.html) 与 [Autograd 文档](https://pytorch.org/docs/stable/autograd.html)，因果注意力对照 [Transformer 原论文](https://arxiv.org/abs/1706.03762)，生成参数对照 [Transformers generation 文档](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)。示例数字和矩阵是教学构造，不能替代具体实现的单元测试。
