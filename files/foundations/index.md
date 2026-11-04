# 基础知识

模型接收数字，经过一系列计算给出预测；训练根据预测误差修改参数。文本、图像和动作会带来不同的数据形式，但张量、函数、损失和梯度始终是理解这些计算的共同语言。

这里按计算对象组织内容。每篇会先固定一个问题，再把输入、公式、中间结果和实现接起来。已经熟悉的部分可以跳过；遇到某个符号或操作再补相应数学，不需要按固定阶段完成整套内容。

## 从两个输入到一次参数更新

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
