# 张量、Shape 与计算图来源台账

## 使用方式

这份台账服务于 [张量、Shape 与计算图](../foundations/tensors-shapes-and-computation-graphs.md)。正文讲法来自三类资料。

第一类是框架官方文档。PyTorch Tensor Views、Storage、`Tensor.view` 和 Autograd Mechanics 用来确定 storage、stride、view/copy、动态计算图和 saved tensors 的边界；NumPy broadcasting 用来确定广播规则；einops 用来学习如何把轴语义写进 reshape 表达式。

第二类是系统论文。TensorFlow 系统论文用来解释 dataflow graph 的基本图像：节点是算子，边是张量，系统可以基于图做设备放置和执行优化。本站只借它讲计算图直觉，不把 TensorFlow 图模式等同于 PyTorch eager autograd。

第三类是高质量中文讲法。《动手学深度学习》的数据操作、线性代数和注意力章节适合作为中文读者的基础补充：它善于把 shape、axis、broadcast 和代码例子放在一起讲，而不是只给术语表。

## 写作边界

正文不把张量写成抽象数学名词，而是写成 storage + metadata + module contract。公式必须解释它在工程里回答什么问题：某个索引怎么映射到底层 storage、某个 Linear 为什么只要求最后一维、某个多模态 token 长度为什么会乘法放大。

正文不把 `reshape` 当成无成本操作。只要涉及 `view`、`reshape`、`transpose`、`permute`、`contiguous`，都要同时交代它可能改变 axis semantics、stride/layout 或 copy 行为。

正文不把计算图只讲成“forward/backward 箭头”。重点是：forward 产生的中间 tensor 为什么要被 backward 保存，保存它们为什么会变成训练显存压力。

## 来源

| 来源 | 类型 | 本站使用方式 | 关联主题 |
| --- | --- | --- | --- |
| [PyTorch Tensor Views](https://docs.pytorch.org/docs/main/tensor_view.html) | 官方文档 | 解释 view 共享底层数据、transpose 可能产生 non-contiguous view、reshape/flatten 可能返回 view 或 copy。 | view / stride |
| [torch.Tensor.view](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.view.html) | 官方文档 | 引入 view 对 stride/contiguity 的要求，以及不确定时使用 reshape 的实践边界。 | view 条件 |
| [PyTorch Storage](https://docs.pytorch.org/docs/stable/storage.html) | 官方文档 | 用 storage、shape、stride、offset、dtype 解释 tensor 的底层结构。 | tensor metadata |
| [torch.reshape](https://docs.pytorch.org/docs/stable/generated/torch.reshape) | 官方文档 | 支撑“reshape 可能零拷贝，也可能复制，不能依赖具体行为”的说法。 | reshape |
| [NumPy Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) | 官方文档 | 采用右对齐、相等或为 1 的广播规则，并提醒广播可能导致低效内存使用。 | broadcasting |
| [numpy.broadcast_to](https://numpy.org/doc/stable/reference/generated/numpy.broadcast_to.html) | 官方文档 | 补充 broadcast view 可能 readonly、non-contiguous、多个元素引用同一内存位置。 | broadcast view |
| [Einops basics](https://einops.rocks/1-einops-basics/) | 官方教程 | 借鉴用 `b c h w -> b h w c` 这类表达方式显式标注轴语义。 | axis semantics |
| [einops.rearrange API](https://einops.rocks/api/rearrange/) | 官方文档 | 支撑 rearrange 作为 reader-friendly smart element reordering 的写法。 | reshape 可读性 |
| [PyTorch Autograd Mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html) | 官方文档 | 解释动态图、链式法则、`.grad_fn` 和 saved tensors。 | computation graph |
| [torch.autograd](https://docs.pytorch.org/docs/autograd) | 官方文档 | 补充 backward、grad、forward-mode AD 和 saved tensor hooks 的边界。 | autograd |
| [TensorFlow: Large-Scale Machine Learning on Heterogeneous Distributed Systems](https://arxiv.org/abs/1603.04467) | 论文 | 使用 Figure 2 说明 dataflow graph 直觉。 | computation graph |
| [TensorFlow: A system for large-scale machine learning](https://arxiv.org/abs/1605.08695) | 论文 | 补充 TensorFlow graph 如何面向分布式和异构设备执行。 | graph execution |
| [动手学深度学习：数据操作](https://zh.d2l.ai/chapter_preliminaries/ndarray.html) | 中文教材 | 学习面向中文读者解释张量、广播、基本数组操作的讲法。 | 中文讲法 |
| [动手学深度学习：线性代数](https://zh.d2l.ai/chapter_preliminaries/linear-algebra.html) | 中文教材 | 学习 axis、降维、矩阵乘和张量符号的基础解释。 | 轴与矩阵 |
| [动手学深度学习：多头注意力](https://zh.d2l.ai/chapter_attention-mechanisms/multihead-attention.html) | 中文教材 | 参考如何在注意力代码中逐步标注 queries/keys/values 的 shape 变化。 | attention shape |
