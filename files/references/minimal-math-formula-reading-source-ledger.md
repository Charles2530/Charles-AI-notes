# 读懂公式的最小数学来源台账

这份台账服务于 [读懂公式的最小数学](../foundations/symbols-and-minimal-math.md)。正文不再把数学写成符号查表，而是围绕一条读法展开：公式声明接口、概率、loss、梯度、约束、可见性和成本。

## 使用方式

第一类来源是基础教材。D2L 的线性代数、概率和自动微分章节帮助本站把 shape、矩阵乘法、期望、梯度和自动求导讲成可读的入门路径。

第二类来源是计算图和反向传播讲解。CS231n、colah 和 PyTorch autograd 文档帮助本站解释：backward 不是孤立技巧，而是沿计算图用链式法则传递梯度；forward 中保存的 tensor 会影响训练显存。

第三类来源是矩阵微积分材料。The Matrix Calculus You Need For Deep Learning 和 The Matrix Cookbook 这类资料用于校验矩阵乘法、梯度形状和符号约定；正文只保留初学者真正需要的部分。

第四类来源是框架和数组文档。NumPy broadcasting、PyTorch tensor/autograd 文档帮助本站说明 shape、broadcasting、saved tensors 和隐式对齐为什么是工程问题。

图源使用 Attention Is All You Need 的 Transformer 架构图和 Sublinear Memory 的计算图图示。前者说明公式背后是模块接口，后者说明梯度和显存来自计算图依赖。

这页还承担一个全站写作约束：数学解释不能从正文里抽离。符号表可以辅助检索，但不能替代“这行公式在解决什么问题”的解释。本站后续改稿时，遇到公式应当就地说明变量来源、shape、可见条件、优化目标和边界，而不是让读者跳到一个孤立附录里自己拼。

中文材料主要用于降低入门阻力。D2L 的中文版本适合给读者补线性代数、概率和自动微分；英文材料如 CS231n、colah、explained.ai 和 PyTorch 文档用于校验计算图、矩阵微积分和框架实现。两类材料的角色不同：中文材料帮读者进门，英文/官方材料帮本站保持机制准确。

## 写作边界

正文不做大而全数学教材。它只覆盖读 AI 文章最常见的公式：shape、线性层、softmax、cross entropy、期望、梯度、KL、时间下标和复杂度。

正文不鼓励背孤立符号。同一个 \(H\) 可能是 height、horizon、head count；同一个 \(D\) 可能是 hidden dimension、dataset 或 depth。判断必须回到上下文、shape 和模块接口。

正文不把公式解释集中塞到统一附录。每篇技术文章仍然应该在公式旁边解释变量、输入输出和边界；本页只是读法底座。

正文不把概率当成真实置信度。Softmax 输出是模型分布，可能校准不好；KL 是有方向的分布差异，不是普通对称距离。

正文不把复杂度写成真实 latency。\(O(L^2)\)、\(B\cdot L\cdot D\cdot bytes\) 这类公式只说明增长趋势或数量级；真实运行还要看 kernel、layout、cache、batch、dtype、device 和编译器。

正文不把公式读法变成 checklist 风格提示框。文章可以给读者一个稳定读法，但每个小节仍要讲清具体知识：矩阵乘法怎样重组特征，softmax 怎样归一化分数，交叉熵为什么惩罚正确答案低概率，梯度为什么沿计算图传播。

## 来源

| 来源 | 类型 | 本站使用方式 | 关联主题 |
| --- | --- | --- | --- |
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | 论文 / 图源 | 使用 Figure 1 说明公式背后是 Transformer 模块接口和数据流。 | 模块接口 |
| [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174) | 论文 / 图源 | 使用 Figure 1 说明 forward、backward、计算图和中间张量保存。 | 计算图 |
| [D2L：线性代数](https://zh.d2l.ai/chapter_preliminaries/linear-algebra.html) | 中文教材 | 支撑张量、矩阵乘法、范数和轴语义的入门解释。 | 线性代数 |
| [D2L：自动微分](https://zh.d2l.ai/chapter_preliminaries/autograd.html) | 中文教材 | 支撑计算图和自动求导讲法。 | 自动微分 |
| [D2L：概率](https://zh.d2l.ai/chapter_preliminaries/probability.html) | 中文教材 | 支撑概率、期望和采样分布的解释。 | 概率 |
| [CS231n: Optimization and Backpropagation](https://cs231n.github.io/optimization-2/) | 教程 | 借鉴计算图和链式法则讲法。 | 反向传播 |
| [colah: Calculus on Computational Graphs](https://colah.github.io/posts/2015-08-Backprop/) | 技术博客 | 借鉴用图解释反向传播的教学结构。 | 反向传播讲法 |
| [The Matrix Calculus You Need For Deep Learning](https://explained.ai/matrix-calculus/) | 技术教程 | 校验矩阵微积分和梯度 shape 解释。 | 矩阵微积分 |
| [PyTorch Autograd Mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html) | 官方文档 | 支撑 saved tensors、动态计算图和 backward 细节。 | 框架实现 |
| [NumPy Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) | 官方文档 | 支撑 broadcasting 右对齐和 singleton axes 解释。 | Shape |
