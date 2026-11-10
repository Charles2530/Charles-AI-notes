# 模型结构

结构决定信息在哪里交换、哪些计算共享参数、历史以什么形式保留。训练目标再决定这些参数被优化成什么。理解新架构时，先画出输入到输出的计算，通常比先记模型名称更容易判断它改变了什么。

## 同一组特征怎样经过不同模块

输入是`[B,L,D]`。线性层把每个位置的D维变为H维，通常不交换位置；卷积读取邻近窗口并共享权重；attention让每个query按内容读取其他位置；SSM将历史压进状态并递推更新。它们可以组合，也可以替换某个特定环节。

| 要理解的运算 | 对应文章 | 可手算的主线 |
| --- | --- | --- |
| 逐位置变换特征 | [Linear与MLP](../foundations/linear-layers-mlp-and-gemm.md) | 两个输入怎样得到两个输出及权重梯度 |
| 在局部窗口复用参数 | [卷积](../foundations/convolution-and-feature-extraction.md) | 3×3图像上的四个窗口与多尺度 |
| 按内容读取其他位置 | [Attention](../foundations/transformer-attention-and-tokenization.md) | 两个位置的QK分数、mask和value加权 |
| 改变可见性和位置关系 | [位置与Mask](../foundations/positional-encoding-masks-and-context.md) | 两篇文档打包后如何隔离 |
| 连接深层计算 | [Norm、残差和激活](../foundations/normalization-residual-and-activation.md) | (1,3)的归一化与残差导数 |
| 为不同token选择参数 | [MoE](../foundations/moe-and-large-model-architecture-basics.md) | 四个token的dispatch、专家与combine |
| 将历史变成递推状态 | [Mamba与SSM](../foundations/mamba-and-hybrid-ssm-transformers.md) | 标量状态更新与仿射scan |

## 缓存历史还是压缩历史

标准因果attention保存各历史位置的K/V，新query仍能有针对性地读取它们，状态成本通常随长度增长。固定维SSM每步更新压缩状态，状态大小不随历史长度线性增长，却不能无损保留任意历史细节。

混合架构可以在部分层保留attention、部分层使用SSM。MoE又作用于前馈参数选择，与这条历史表示轴不同。将“SSM+attention+MoE”逐层分开计算，才能说明存储、算术和通信分别来自哪里。

## 改结构也会改变可部署的计算

[剪枝与结构搜索](../foundations/model-compression-pruning-and-nas-basics.md)用一个两层网络说明删除通道后哪些矩阵必须一起改变。[模型蒸馏](../foundations/model-distillation-basics.md)则改变学生获得的训练信号，不一定改变结构。

参数量、激活参数、FLOPs和延迟各有含义。一个更窄的层可能降低矩阵计算量，却使GPU利用率变差；一个MoE可能增加总参数，同时减少每token使用的专家参数，却增加通信。继续看[算子实现](../operators/index.md)和[运行时](../inference/index.md)，才能把结构优势与具体执行条件接起来。

## 读结构图时如何追到正文

先找到输入和输出张量，再给每条箭头写shape，确认在哪一步交换位置、在哪一步改变通道、在哪一步引入随机或离散选择。随后读[表示学习目标](../foundations/pretraining-objectives-and-representation-learning.md)，检查训练到底奖励哪些变化。这样就能区分结构本身的性质与某套训练配方得到的能力。
## 来源驱动的入门顺序

先用[Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)建立输入、注意力和输出的整体图，再用[Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)把公式对到实现；卷积感受野可用[Distill 交互教程](https://distill.pub/2019/computing-receptive-fields/)补足。Mamba、MoE 和 Mamba-2 应放在 attention、卷积、状态和参数共享的对照建立后阅读。

每篇结构文章先问三件事：信息在哪里交换，历史以什么形式保存，代价落在算力、显存还是通信。模型名称只作为索引，不能替代这三项计算描述。

## 读一张新架构图的工作表

面对论文或项目页中的结构图，可以按下面顺序填表：

| 问题 | 记录内容 |
| --- | --- |
| 输入是什么 | batch、序列/空间尺寸、通道、dtype、是否含历史状态 |
| 第一次改变 shape 的位置 | 投影、patch、下采样、路由或 tokenizer |
| 信息在哪里交换 | 局部窗口、全局 attention、卷积、跨模态连接或状态递推 |
| 哪些变量跨时间保存 | KV、SSM state、memory token、检索结果或外部数据库 |
| 哪些路径是条件的 | top-k 专家、mask、稀疏激活、控制分支或安全过滤 |
| 输出由谁消费 | loss、采样器、planner、控制器、缓存或评测器 |

填完后再估算每个阶段的工作量和状态生命周期。若论文只展示“更高质量”而没有说明新增状态、训练目标或评测条件，先把结论限定在作者设置内。

## 用四个最小实验比较结构

1. **形状实验**：给定 `[B,L,D]`，逐层打印 shape、stride 和参数量。
2. **信息实验**：遮掉一个位置、一个模态或一段历史，观察输出是否按结构预期改变。
3. **成本实验**：改变 L、batch、专家数或状态长度，分别记录算量、峰值显存和通信。
4. **任务实验**：固定训练配方，只替换结构，比较独立评测、长上下文和失败桶。

例如把 attention 换成固定状态的 SSM，不能只比较参数量；要同时看长序列下的信息保留、状态更新、prefill/decode 代价和任务结果。把 dense FFN 换成 MoE，也要报告路由负载、通信和专家容量，而不是只报告激活参数量。

## 自测：结构改变还是训练改变

如果一个新模型在同一 benchmark 上更好，先问它是否同时使用了更多数据、不同 tokenizer、更长训练或额外检索。只有在这些条件固定，或者被单独列成实验变量时，才能把差异归因于结构。结构入口页只负责提出这个问题，具体证据要回到对应训练和评测页面。
