# 对比学习总览

对比学习的核心，是让“正样本更近、负样本更远”。

!!! tip "基础知识入口"
    对比学习依赖 embedding space、batch 内矩阵计算和 loss 优化。如果这些概念还不稳，可以先看 [张量与计算图](../foundations/tensors-shapes-and-computation-graphs.md)、[Transformer 与 Attention](../foundations/transformer-attention-and-tokenization.md) 和 [优化与训练基础](../foundations/optimization-and-training-basics.md)。

下面这张 SimCLR 原论文图先把视觉对比学习压缩成一个训练闭环：同一张图经过两种增强形成正对，encoder 和 projection head 得到表示，再用对比损失把正对拉近、把 batch 里的其他样本推远。

![SimCLR 原论文框架图](../assets/images/paper-figures/contrastive-learning/simclr-figure-1.png){ width="820" }

<small>图源：[A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)，Figure 1。原论文图意：SimCLR 从同一图像采样两种 augmentation，经过 encoder \(f(\cdot)\) 和 projection head \(g(\cdot)\)，用 contrastive loss 最大化同源视图表示的一致性。</small>

!!! note "图解：SimCLR 图里的训练闭环"
    先看图的左侧：同一张原图会被做两次随机增强，得到两条不同视图，它们在训练里被当作正样本对。中间的 encoder \(f(\cdot)\) 负责提取可迁移表示，projection head \(g(\cdot)\) 则把表示投到更适合做对比损失的空间。最后的 contrastive loss 不只是拉近这两个正样本，还会把 batch 中其他图像视为负样本推远。真正难的是三件事：增强不能破坏语义，batch 里的负样本不能含太多假负样本，projection head 学到的空间要能服务下游任务。

**最经典的目标是 InfoNCE**：

\[
\mathcal{L}_{\text{InfoNCE}} = - \log \frac{\exp(\text{sim}(z_i, z_i^+)/\tau)}{\exp(\text{sim}(z_i, z_i^+)/\tau) + \sum_j \exp(\text{sim}(z_i, z_j^-)/\tau)}
\]

这个式子看起来像一条普通损失函数，但背后真正塑造的是表示空间的几何结构：什么应该聚在一起，什么应该分开，哪些差异要被保留，哪些差异可以被忽略。

## 1. 为什么它重要

它不仅影响自监督视觉，也深刻影响：

- CLIP 这类图文模型：用图像和文本的相似度学习跨模态语义空间。
- 检索系统：把查询和候选都变成 embedding，再按距离召回。
- 表征学习：让模型先学可迁移表示，而不是只服务单一标签任务。
- 多模态对齐：把图像、文本、音频等不同输入放到可比较空间。
- 向量数据库召回：用近邻搜索快速找到语义相近内容。

可以说，对比学习是现代“向量空间理解”最核心的训练范式之一。

## 2. 一个例子：金毛犬的两张照片

把一只金毛犬的两种图像增强视为正对，把其他图像视为负对。训练后，模型会倾向于把“同一只狗的不同视角”聚到一起，把不同对象分开。

这个例子看似简单，但已经包含对比学习的本质：

- 正样本应该近：同一对象、同一语义或同一图文对应该在 embedding 空间里靠近。
- 负样本应该远：无关对象或语义不同的样本应该被拉开。
- 相似性定义：距离由任务目标和数据构造决定，不是天然固定的。

### 2.1 为什么这个例子还不够真实

真实世界的问题通常更复杂。比如：

1. 两张狗图是不是同一种狗，还是同一只狗；
2. 不同拍摄环境该不该被忽略；
3. 颜色差异、背景差异、动作差异中哪些算语义，哪些算噪声；
4. 在文本检索里，同义句算正样本还是假负样本。

也就是说，对比学习真正难的地方从来不是“有没有损失函数”，而是“正负定义到底是什么”。

## 3. 对比学习真正解决的问题

在没有标签时，模型并不知道什么是“同类”。对比学习的关键就是自己构造监督信号，让模型学会一个有结构的嵌入空间：

\[
x \xrightarrow{f_\theta} z
\]

**这个空间随后可以被用于**：

- 分类：用 embedding 上的线性分类器或近邻判断类别。
- 检索：按向量距离找到语义相近的图像、文本或商品。
- 聚类：把相似样本自动分组，发现数据结构。
- 多模态对齐：让图像、文本等不同模态可以互相比较和召回。

### 3.1 表示空间为什么比单一任务更重要

因为一旦嵌入空间足够有结构，很多下游任务只是“读取这个空间里的信息”，而不必每次从头学习。一个好的表示空间常常意味着：

1. 线性分类更容易；
2. 检索近邻更可信；
3. 多模态对齐更稳定；
4. 长尾迁移更自然。

## 4. 为什么它对多模态特别重要

图像和文本原本在完全不同的输入空间里。对比学习让它们可以共享一个可比较的嵌入空间，于是：

- 文本搜图：把文字查询映射到图像 embedding 空间里找近邻。
- 图像搜文：用图片找到标题、说明、标签或相关文档。
- 相似内容聚合：把语义相近但形式不同的内容归到同一区域。

这也是 CLIP、商品搜索、图文检索能大规模落地的重要原因。

![CLIP pre-training and zero-shot transfer 原论文图](../assets/images/paper-figures/contrastive-learning/clip-figure-1.png){ width="920" }

<small>图源：[Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)，Figure 1。原论文图意：CLIP 用图像编码器和文本编码器做图文对比学习；zero-shot 分类时，把类别文本 prompt 编码成分类器权重，与图像表示做相似度匹配。</small>

!!! note "图解：CLIP 把分类问题改写成检索问题"
    这张图左侧是训练：正确图文对要靠近，错误配对要远离。右侧是 zero-shot 分类：类别名先被写成文本 prompt，再和图片 embedding 做相似度。初学者要注意，CLIP 的强项是全局语义对齐和检索，不等于它天然能读小字、做精确定位或完成复杂视觉推理。

### 4.1 多模态里的特殊难点

在多模态场景中，负样本问题会更棘手。因为很多图文对之间存在弱相关、部分重叠或语义近似。如果把这些都粗暴地当负样本，就会在嵌入空间里强行拉开本该靠近的点。

这也是为什么多模态系统常常需要：

1. 更大的 batch；
2. 更谨慎的 hard negative 设计；
3. 更强的数据清洗；
4. 更细的评测和长尾分析。

## 5. 两条最重要的发展线

### 自监督视觉线

如：

- SimCLR：用强数据增广和大 batch 对比学习视觉表示。
- MoCo：用动量编码器和队列扩展负样本规模。
- BYOL：不显式使用负样本，通过在线网络和目标网络避免 collapse。

**它们主要在回答**：没有标签时，怎样学出有用表征。

### 多模态对齐线

如：

- CLIP 风格图文对齐：用图文对比学习把图片和文本放进共享空间。
- 向量检索：把 embedding 作为检索索引的基础表示。
- 跨模态召回：用一种模态查询另一种模态，例如文搜图或图搜文。

**它们主要在回答**：不同模态之间，怎样建立统一语义空间。

### 5.1 还有一条常被忽略的系统线

随着对比学习进入生产环境，它又衍生出一条系统主线：

1. 向量索引和检索；
2. hard negative 挖掘；
3. embedding 评测与回流；
4. 在线召回质量监控。

这说明对比学习并不只是训练范式，也是一类系统基础设施的起点。

## 6. 为什么“自监督”不等于“无监督万能”

对比学习虽然不依赖显式标签，但依然非常依赖：

1. 数据增广设计；
2. 正负样本构造；
3. batch 与队列策略；
4. 假负样本控制；
5. 长尾评测。

如果这些环节没做好，模型依然可能学到捷径，而不是稳定语义结构。

## 7. 一个更现实的理解方式

可以把对比学习想成在训练一名“向量世界里的地图绘制者”。它的任务不是直接回答问题，而是把世界里的样本排布成一张地图：

1. 相似的样本应该住得近；
2. 不同的样本应该拉开；
3. 同一语义在不同模态中也要对齐；
4. 长尾和细粒度差异不能全被抹平。

地图画得好，后面的检索、分类、多模态对齐和推荐系统都会受益；地图画歪了，后面所有基于 embedding 的系统都会出问题。

## 8. 阅读建议

如果你是第一次系统学习这个方向，建议按下面顺序：

1. [基础方法](foundations.md)
2. [多模态与检索](multimodal-and-retrieval.md)
3. [难负样本与增广设计](hard-negatives-and-augmentation-design.md)
4. [评测与失效模式](evaluation-and-failure-modes.md)
5. [自蒸馏与非对比方法](self-distillation-and-non-contrastive-methods.md)

先理解 SimCLR/MoCo/BYOL 这类基础构件，再看它们如何自然延伸到 CLIP、检索系统与非对比表征学习。

## 9. 一个总判断

对比学习的价值，不在于它是一种“无标签也能学”的技巧，而在于它提供了一种组织表示空间的方式。现代多模态检索、向量数据库和很多自监督视觉能力，实际上都建立在这张表示空间地图是否画得足够合理之上。理解对比学习，也是在理解今天大量基础模型系统的底层几何。 

## 快速代码示例

```python
import torch
import torch.nn.functional as F

def info_nce(z1, z2, temperature=0.07):
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    logits = z1 @ z2.T / temperature
    labels = torch.arange(z1.size(0), device=z1.device)
    return F.cross_entropy(logits, labels)
```

这段代码演示了一个最小可用的 **InfoNCE** 步骤：先把两路表示做归一化，再用相似度矩阵除温度得到 logits，最后用对角正样本标签计算交叉熵。实际训练时通常还会加入双向损失、大 batch 或队列来提升负样本质量。


## 学习路径与阶段检查

对比学习建议按“目标函数 -> 样本制度 -> 下游消费 -> 失效分析”读。它不是单个 loss，而是一套表示空间设计方法。

| 阶段 | 先读 | 读完要能回答 |
| --- | --- | --- |
| 1. 基础目标 | [基础方法](foundations.md) | InfoNCE、temperature、batch negatives 和 projection head 分别控制什么 |
| 2. 样本制度 | [难负样本与增广设计](hard-negatives-and-augmentation-design.md) | 正样本、负样本、hard negative 和 false negative 的边界如何定义 |
| 3. 多模态消费 | [多模态与检索](multimodal-and-retrieval.md) | 表示空间如何被 CLIP、向量检索、召回和重排系统消费 |
| 4. 替代路线和验收 | [自蒸馏与非对比方法](self-distillation-and-non-contrastive-methods.md)、[评测与失效模式](evaluation-and-failure-modes.md) | 不用显式负样本时如何防 collapse，表示质量如何按任务桶和长尾样本验收 |

读完后建议接 [VLM 总览](../vlm/index.md)：很多 VLM 的图文对齐、检索增强和数据清洗问题，本质上都在消费对比学习留下的 embedding 空间。
