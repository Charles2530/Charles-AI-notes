# AI 基础知识地图

这一组页面是本站后续专题的公共地基。它不要求你先学完整本数学教材，而是帮你把论文和工程文档里最常见的概念按顺序串起来：

<!-- [补充] -->
| 读者 | 建议读法 | 工程落点 |
| --- | --- | --- |
| 初学者 | 先抓“AI 基础知识地图”要解决的问题、关键术语和最小例子；公式只看变量方向。 | 能说清输入、输出和常见失败信号。 |
| 工程师 | 直接对照结构、指标、成本和边界条件，必要时回到前置页补概念。 | 落到数据、算力、延迟、显存、评测和回滚检查。 |


!!! note "读法定位"
    这页先回答“AI 基础知识地图”在「基础知识」里的位置：它解决什么局部问题，依赖哪些前置，最后会影响哪类工程或研究判断。
    **前置**：先看本页要补哪一个最小概念；公式或术语卡住时回到术语表，不需要一次吃完整个数学体系。 必要时先看本页路线，或查 [术语表](../glossary/index.md)。
    **主线关系**：把符号、张量、优化、评测和运行时这些前置打稳，后面的扩散、VLM/VLA、训练与系统页才不会断层。

```text
数据 -> 张量 / Token -> 模型模块 -> 目标 / Loss -> 泛化 -> 解码 / Runtime -> 评测证据
```

如果你刚开始读 AI 论文，先别急着背模型名字。先弄清数据怎么进模型、模型每一步在算什么、训练信号从哪里来、最后用什么证据证明它真的变好。

!!! note "先读哪几页"
    推荐顺序是：本页 -> [数据与数据集基础](data-and-dataset-basics.md) -> [符号与最小数学地图](symbols-and-minimal-math.md) -> [张量、Shape 与计算图](tensors-shapes-and-computation-graphs.md) -> [Transformer 输入与注意力](transformer-attention-and-tokenization.md) -> [预训练目标与表示学习](pretraining-objectives-and-representation-learning.md) -> [优化与训练入门](optimization-and-training-basics.md) -> [泛化、正则化与分布偏移](generalization-regularization-and-distribution-shift.md)。读完这些，再按兴趣补生成解码、模型蒸馏、多模态、MoE、Mamba、显存和评测。

<!-- [补充] -->
> **一句话理解**：AI 基础知识地图 可以先当成一张入门地图：它把零散术语连成“输入、变化、输出”三件事。先不用背公式，先看清哪一步在改变信息、哪一步在改变成本。

## 一张图先建立全局感

![Transformer architecture 原论文图](../assets/images/paper-figures/transformer/attention-is-all-you-need-figure-1.png){ width="500" .atlas-figure-tall }

<small>图源：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)，Figure 1。原论文图意：展示 Transformer encoder-decoder 的整体结构，包括 token embedding、positional encoding、multi-head attention、feed-forward network、residual connection、normalization 和最终 softmax。</small>

!!! note "论文图解卡：这张 Transformer 图怎么读"
    **先看哪里**：从底部的 input/output embedding 看起，再沿着 attention、feed-forward、residual、normalization 往上走。  
    **图中模块**：embedding 把 token 变成向量；attention 让 token 互相读取信息；feed-forward/MLP 逐 token 做非线性变换；residual 和 norm 让深层训练更稳；softmax 把输出变成概率。  
    **对应基础页**：token 和 attention 看 Transformer 页，MLP 看线性层页，residual/norm 看稳定性页，softmax 和概率看符号页与概率页。  
    **容易误读**：这张图是结构图，不是显存图。真实系统还要考虑 KV cache、dtype、kernel、batch 和 runtime。

## 基础主线：从数据到证据

### 2.1 先看数据本身，再变成张量和 token

文本、图像、视频、音频、动作轨迹，进入模型前先是样本、标签、schema、来源、split 和版本。然后它们才会经过 tokenizer / processor 变成张量。张量的 shape 决定模块能不能接上，也决定显存和计算大概有多贵。

```text
数据来源 / 样本 / 标签 / schema / split / version
  -> tokenizer / processor
文本 -> token id -> embedding: [B, L, D]
图像 -> patch token -> visual embedding: [B, N, D]
视频 -> frame / patch / time token -> long sequence
动作 -> action value / action token -> policy input or target
```

先读：

| 页面 | 解决的问题 |
| --- | --- |
| [数据与数据集基础](data-and-dataset-basics.md) | 数据来源、样本/标签、processor、清洗、去重、泄漏和版本怎么读 |
| [符号与最小数学地图](symbols-and-minimal-math.md) | 公式和 shape 字母到底怎么读 |
| [张量、Shape 与计算图](tensors-shapes-and-computation-graphs.md) | 数据如何组织成张量，shape 为什么是接口 |
| [卷积与视觉特征](convolution-and-feature-extraction.md) | 图像局部特征、多尺度和 UNet 怎么理解 |

### 2.2 模型模块把表示一步步变换

大多数模型结构都可以先拆成几类模块：线性层负责投影，MLP 负责逐 token 非线性变换，attention 负责跨 token 读信息，norm/residual 负责稳定深层网络。

| 模块 | 初学者理解 | 先读 |
| --- | --- | --- |
| Linear / MLP | 把向量重新组合，再加入非线性 | [线性层、MLP 与 GEMM](linear-layers-mlp-and-gemm.md) |
| Attention | 每个 token 按相关性读取其它 token | [Transformer 输入与注意力](transformer-attention-and-tokenization.md) |
| Position / Mask | 告诉模型顺序和可见性 | [位置编码与 Mask](positional-encoding-masks-and-context.md) |
| Norm / Residual | 让深层训练更稳定 | [Norm、残差与激活函数](normalization-residual-and-activation.md) |
| MoE / Mamba | 在容量、长上下文和推理成本上做结构取舍 | [MoE](moe-and-large-model-architecture-basics.md)、[Mamba](mamba-and-hybrid-ssm-transformers.md) |

### 2.3 目标、Loss 和梯度让模型学习

训练不是“模型突然懂了”，而是反复做四件事：

```text
算预测 -> 算 loss -> 反向传播 gradient -> optimizer 更新参数
```

最小公式是：

\[
\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}
\]

<!-- [补充] -->
| 符号 | 含义 | 取值范围 |
| --- | --- | --- |
| \(\theta\) | 模型或策略参数 | 实值张量，形状由模型结构决定 |
| \(\eta\) | 公式中的变量或中间量 | 按上下文定义 |


这里 \(\theta\) 是模型参数，\(\mathcal{L}\) 是 loss，\(\nabla_\theta \mathcal{L}\) 是梯度，\(\eta\) 是学习率。第一次看到这行公式时，只要把它读成“沿着让错误变小的方向更新参数”就够了。

先读：

| 页面 | 解决的问题 |
| --- | --- |
| [概率与潜变量模型](probability-latent-variables-and-generative-models.md) | 为什么模型学的是分布，\(p_\theta\)、latent、KL、ELBO 怎么读 |
| [预训练目标与表示学习](pretraining-objectives-and-representation-learning.md) | next-token、masked modeling、contrastive、denoising 和行为克隆分别在学什么 |
| [优化与训练入门](optimization-and-training-basics.md) | loss、gradient、optimizer、训练曲线怎么连起来 |
| [泛化、正则化与分布偏移](generalization-regularization-and-distribution-shift.md) | loss 下降后，为什么还要看 train-val gap、正则化和 OOD |
| [模型蒸馏入门](model-distillation-basics.md) | teacher/student、soft target、rationale 和少步生成蒸馏怎么读 |
| [自动微分与激活显存](autograd-activation-checkpointing-and-memory.md) | 为什么训练要保存计算图和激活，checkpointing 为什么省显存 |

### 2.4 解码和 Runtime 决定能不能跑得动

算法公式正确，不代表工程上能跑。长上下文、多图视频、MoE、低比特和高并发都会把问题变成系统成本。

```text
参数量 -> 权重显存
序列长度 -> attention / KV cache
dtype -> bytes / kernel 支持
batch 和 shape -> 吞吐 / 延迟
runtime -> 调度、缓存、融合和 fallback
```

先读 [生成与解码基础](generation-and-decoding-basics.md) 和 [数值、显存与运行时](numerics-memory-and-runtime-basics.md)。读完后至少能分清：temperature/top-p 改的是生成选择，KV cache 和 runtime 改的是执行成本；模型文件变小、FLOPs 下降、延迟下降、显存下降也不是同一件事。

### 2.5 评测证据决定结论是否可信

一个方法说“更好”，要追问它到底在哪个证据上更好：

| 证据 | 能说明什么 | 不能说明什么 |
| --- | --- | --- |
| train loss | 训练目标是否被优化 | 真实任务一定成功 |
| validation / test | 固定数据上的泛化趋势 | 部署长尾和安全边界 |
| ablation | 哪个设计可能有用 | 全部场景都适用 |
| latency / throughput | 系统成本 | 质量一定不掉 |
| closed-loop / online eval | 真实交互表现 | 所有风险都已覆盖 |

先读 [数据划分与评测指标](data-splits-metrics-and-evaluation-basics.md)。如果只看一个总分，很容易把平均收益误读成关键场景可靠。

### 2.6 决策和 RL 的最小入口

如果你要读世界模型、VLA、机器人或 RLHF，基础区至少要认得这五个词：

| 词 | 最小读法 | 后面会用在哪里 |
| --- | --- | --- |
| state | 当前局势或历史压缩状态 | 世界模型 latent、VLA 观测 |
| action | 模型或 agent 选择的下一步 | token、工具调用、机器人动作 |
| reward | 行为好坏的反馈信号 | RLHF、RLVR、控制任务 |
| policy | 从 state 到 action 的规则或分布 | LLM 采样策略、VLA 策略 |
| value | 从当前状态预计未来能拿多少回报 | PPO、Dreamer、规划 |

这部分不在基础区展开算法推导。要系统学习，直接进 [MDP、价值函数与 Bellman](../reinforcement-learning/mdp-value-bellman.md)。

## 两条常见学习路线

### 3.1 从零读 AI 模型

| 顺序 | 页面 |
| --- | --- |
| 1 | [数据与数据集基础](data-and-dataset-basics.md) |
| 2 | [符号与最小数学地图](symbols-and-minimal-math.md) |
| 3 | [张量、Shape 与计算图](tensors-shapes-and-computation-graphs.md) |
| 4 | [线性层、MLP 与 GEMM](linear-layers-mlp-and-gemm.md) |
| 5 | [Transformer 输入与注意力](transformer-attention-and-tokenization.md) |
| 6 | [预训练目标与表示学习](pretraining-objectives-and-representation-learning.md) |
| 7 | [概率与潜变量模型](probability-latent-variables-and-generative-models.md) |
| 8 | [优化与训练入门](optimization-and-training-basics.md) |
| 9 | [泛化、正则化与分布偏移](generalization-regularization-and-distribution-shift.md) |
| 10 | [数据划分与评测指标](data-splits-metrics-and-evaluation-basics.md) |

### 3.2 为后续专题补课

| 你想读 | 先补什么 |
| --- | --- |
| 扩散模型 | 卷积/UNet、概率分布、score、优化 |
| VLM / VLA | 数据 schema、token、视觉特征、attention、位置/mask、动作符号、BC |
| 量化 | dtype、bytes、GEMM、runtime、评测 |
| 推理系统 | 解码、attention、KV cache、batch、latency、runtime |
| 训练系统 | 数据版本、预训练目标、loss、梯度、自动微分、activation、checkpoint、蒸馏信号 |
| 世界模型 | latent、动作条件分布、state/action/reward、rollout、评测桶 |
| 技术报告 | MoE、Mamba、低精度、activated params、证据边界 |

## 读论文时的四步法

读任何新论文，先按这四步拆：

1. **变量**：输入、输出、token、latent、action、reward 是什么 shape 和语义？
2. **路径**：数据经过哪些模块，哪些参数被训练，推理时哪些状态被缓存？
3. **目标**：loss 或概率目标到底在优化什么？
4. **证据**：图表证明的是 benchmark、消融、系统吞吐、真实闭环，还是 demo？

如果这四步还不熟，先不要急着进入论文专题页。先按下面的症状表回看基础页，等能稳定回答变量、路径、目标和证据，再去对照不同论文方向。

## 进阶查阅：遇到症状回看哪页

| 现场症状 | 先回看 |
| --- | --- |
| 数据来源、标签或版本说不清 | [数据与数据集基础](data-and-dataset-basics.md) |
| shape 对不上、长视频一接入就 OOM | [张量、Shape 与计算图](tensors-shapes-and-computation-graphs.md) |
| attention 一长就慢，KV cache 压不住 | [Transformer 输入与注意力](transformer-attention-and-tokenization.md)、[数值、显存与运行时](numerics-memory-and-runtime-basics.md) |
| 不知道某个预训练 loss 在学什么 | [预训练目标与表示学习](pretraining-objectives-and-representation-learning.md) |
| loss 下降但关键任务不涨 | [优化与训练入门](optimization-and-training-basics.md)、[泛化、正则化与分布偏移](generalization-regularization-and-distribution-shift.md)、[数据划分与评测指标](data-splits-metrics-and-evaluation-basics.md) |
| 采样参数一改，输出质量和成本都变 | [生成与解码基础](generation-and-decoding-basics.md) |
| 小模型很像 teacher，但上线关键桶退化 | [模型蒸馏入门](model-distillation-basics.md)、[数据划分与评测指标](data-splits-metrics-and-evaluation-basics.md) |
| latent 很省但 planner 用不了 | [概率与潜变量模型](probability-latent-variables-and-generative-models.md) |
| 低比特模型变小但不快 | [线性层、MLP 与 GEMM](linear-layers-mlp-and-gemm.md)、[数值、显存与运行时](numerics-memory-and-runtime-basics.md) |
| 看世界模型或 VLA 时分不清 state/action/reward | [MDP、价值函数与 Bellman](../reinforcement-learning/mdp-value-bellman.md) |
| RAG 或工具调用结果不可复现 | [Prompt、CoT 与 RAG 入门](prompt-engineering-cot-and-rag-basics.md)、[数据划分与评测指标](data-splits-metrics-and-evaluation-basics.md) |

## 本页结论

基础知识不是“学完才能开始”的门槛，而是一张回查地图。第一次读只要打通主线：数据有来源和版本，数据变成张量/token，模块处理表示，目标和 loss 产生梯度，泛化决定能否离开训练集，解码和 runtime 决定成本，评测决定结论。后面读扩散、VLM、量化、训练、推理、强化学习和世界模型时，遇到卡点再回到对应基础页补细节。

!!! tip "下一站"
    - 按导航顺序继续：[数据与数据集基础](data-and-dataset-basics.md)。
    - 概念或符号卡住时，先查 [术语表](../glossary/index.md)，再回到当前页。
