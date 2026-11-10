# 基础知识

很多论文卡在一行很小的地方。一个张量的第二维到底是序列还是特征，loss 的分母有没有算进 padding，梯度更新的又是哪一组参数。这里先把这些地方算明白，Transformer、扩散和强化学习才不会只剩名词。

每页都从一个能手算的小例子开始，再接到公式、代码和更大的模型。熟悉的部分可以跳过，遇到符号对不上时再回来查，也不必先把整套数学读完。

## 先看一轮更新究竟做了什么

设一个模型根据两个特征预测数值，输入为 \(x=[1,2]\)，权重为 \(w=[1,1]\)，偏置为 \(b=0\)。模型给出

\[
\hat y=xw+b=1\times1+2\times1=3.
\]

若真实值是 \(y=5\)，取损失 \(L=\frac12(\hat y-y)^2=2\)。损失对预测值的导数是 \(\hat y-y=-2\)，因此对两个权重的梯度分别为 \(-2\) 和 \(-4\)，对偏置的梯度为 \(-2\)。学习率取 \(0.1\)，一次梯度下降得到 \(w=[1.2,1.4]\)、\(b=0.2\)。用同一输入重算，预测值变为 \(4.2\)，损失降为 \(0.32\)。

这个小例子已经包含了训练的基本结构。输入和参数有形状，模型按规则计算输出，损失衡量输出与目标的偏差，梯度决定参数如何更新。训练集上的误差变小还不能证明模型能处理新数据，后面还需要数据划分和评测。

## 计算如何写成程序

| 遇到的对象 | 具体解释 | 深入阅读 |
| --- | --- | --- |
| 张量与形状 | 一个 batch 有多少样本，每个数属于哪个样本或特征，矩阵如何相乘 | [张量、Shape 与计算图](tensors-shapes-and-computation-graphs.md) |
| 线性层与非线性 | 如何把输入特征混合成新特征，以及为什么要叠加激活函数 | [线性层、MLP 与 GEMM](linear-layers-mlp-and-gemm.md) |
| 损失与更新 | 误差如何变成梯度，学习率和 batch 如何影响一次更新 | [优化与训练](optimization-and-training-basics.md) |
| 自动微分 | 多层计算如何沿链式法则传播梯度，为什么反向传播要保存中间值 | [自动微分与激活显存](autograd-activation-checkpointing-and-memory.md) |

上面的多输入例子可以手算。[可运行的机制示例](../assets/examples/foundations/mechanisms.py)还包含线性层、标量线性回归的一次更新、带因果掩码的 attention 和解码概率比较，方便把公式与程序输出对照。

## 模型学到什么，由数据和目标共同决定

同一网络可以用于分类、预测下一个 token，也可以从带噪图像中预测噪声。网络结构决定可用的计算方式，训练目标决定哪些输出会受到奖励或惩罚，数据决定模型实际接触到哪些情况。

| 问题 | 阅读入口 |
| --- | --- |
| 样本、标签、tokenizer 和数据清洗是什么关系 | [数据与数据集](data-and-dataset-basics.md) |
| 为什么要区分训练集、验证集和测试集 | [数据划分与评测指标](data-splits-metrics-and-evaluation-basics.md) |
| 预测、对比、去噪这些目标分别要求模型保留什么 | [预训练目标与表征学习](pretraining-objectives-and-representation-learning.md) |
| 训练误差下降后，新样本为什么仍会失败 | [泛化、正则化与分布偏移](generalization-regularization-and-distribution-shift.md) |
| 模型如何从另一个模型的输出中学习 | [模型蒸馏](model-distillation-basics.md) |
| 潜变量、似然与生成目标怎样连接 | [概率与潜变量模型](probability-latent-variables-and-generative-models.md) |

## 从模型计算到文字、图像与应用

语言模型把上下文转成下一个 token 的概率分布。[Transformer 与注意力](transformer-attention-and-tokenization.md)解释上下文如何进入计算，[生成与解码](generation-and-decoding-basics.md)解释概率如何变成输出文本。模型结构的其余分支见[模型结构总览](../model-architectures/index.md)。

图像和视频通过编码器进入模型，[多模态推理](multimodal-cot-and-reasoning-basics.md)讨论视觉信息如何被使用。[Prompt、CoT 与 RAG](prompt-engineering-cot-and-rag-basics.md)则解释任务描述、中间推理和外部资料如何改变模型的输入条件。

## 从一条计算链判断还缺哪块知识

上面的回归例子只需可微函数。把输出改成离散采样动作，普通链式法则就不能直接穿过选择，需要[梯度估计](gradient-estimators-and-training-signals.md)。把输出改成可能有多个解释的观测，就要定义条件分布、潜变量与似然；把同一计算放进更深的网络，归一化、残差和激活会影响数值与梯度传播。

这些文章的分工取决于问题改变了什么。读完一个公式，可以先尝试说出每个输入在训练与推理时是否可得，再用正文数字复算一次，最后对照实现。公式写得熟练，却不知道采样来自哪里或loss平均了哪些位置，仍容易在实现中改变目标。

## 数学与工程参考

[符号与最小数学](symbols-and-minimal-math.md)解释下标、求和、期望和导数的读法。[数学工具](../prerequisite-math/index.md)按问题连接线性代数、概率、微积分及更深入的推导，不需要先通读整套数学才能开始读模型。

遇到采样或离散选择不可直接求导时，查[梯度估计与训练信号](gradient-estimators-and-training-signals.md)；遇到显存、精度和速度问题时，查[数值、显存与运行时](numerics-memory-and-runtime-basics.md)。这些问题在具体计算中出现后，再展开对应细节会更容易理解。
## 来源驱动的入门顺序

如果第一次系统学习，先读[D2L 中文教材](https://zh.d2l.ai/)中与线性模型、MLP、注意力和训练对应的章节，再用[Nielsen 的手写数字叙事](http://neuralnetworksanddeeplearning.com/chap1.html)复习“问题—损失—梯度—更新”的连续过程。想把一轮更新算到代码里，再读[CS231n 两层网络案例](https://cs231n.github.io/neural-networks-case-study/)。这些来源承担不同任务，视觉直觉、教材推导和可运行案例不要互相替代。

已经能独立读 PyTorch 代码后，再把[Stanford CS336 Spring 2025](https://cs336.stanford.edu/spring2025/)当作进阶路线。它把语言模型拆成数据、tokenizer、Transformer、训练、评测和系统问题，适合用来检查本站基础页之间是否缺了一条连接。课程作业和讲义的版本会变化，本站只借鉴主题顺序和问题意识，具体 API、实验数字和代码仍回到对应官方仓库或固定版本。

本站的主干检查点是：能写出输入与参数的 shape，能解释一个 loss 的分母，能复算一次梯度更新，并能指出训练误差下降还缺哪一种独立证据。

## 基础阶段的四个检查点

读完基础主线后，不必用背诵章节名来判断是否学会。尝试完成四个小任务：

1. 给一个 `[B,L,D]` 张量标出样本、序列和特征轴，并说明一次线性层改变了哪一维。
2. 对一个带 padding 的 batch 写出 loss 分子和分母，解释为什么无效位置不能悄悄进入平均。
3. 手算一次参数更新，再说明训练集 loss 下降还缺什么测试才能支持泛化结论。
4. 给出一个离散动作采样例子，指出普通链式法则在哪一步断开，以及需要哪类梯度估计或替代目标。

如果其中一题只能复述术语，回到对应正文和小程序重算。基础知识的目标是让后续页面中的 shape、mask、目标和证据分母都能被独立核对。
