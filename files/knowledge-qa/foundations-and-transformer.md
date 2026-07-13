# 基础与 Transformer QA

这一页用面试高频问法快速复盘机器学习基础、Transformer、tokenizer、attention、位置编码和解码策略。回答按技术面试的正常体量写：先给主回答，再补公式、机制拆解、追问和边界。

## 机器学习与深度学习基础

### Q：机器学习和深度学习的核心差别是什么？

**面试回答。** 机器学习的核心是从数据中学习函数 \(f_\theta(x)\)，让输入 \(x\) 映射到预测 \(y\)。传统机器学习通常需要人工设计特征，例如文本里的 TF-IDF、图像里的 HOG；深度学习把特征抽取也变成可训练模块，用多层神经网络从原始数据中学习表示。可以这样概括：传统 ML 更强调“特征工程 + 相对浅的模型”，深度学习更强调“端到端表示学习 + 大容量模型 + 大规模数据和计算”。

**追问展开。** 面试里如果继续问“深度学习为什么能强”，不要只说层数多。更完整的回答是：深度网络通过组合低层特征形成高层抽象，优化目标把表示学习和任务学习绑在一起；Transformer、CNN、VLM 都是在不同结构先验下学习表示。  
**易错点 / 边界。** 深度学习不自动等于泛化好；数据偏差、训练目标错配、评测污染和分布外样本仍会让大模型失效。  
**继续读。** [数据与数据集基础](../foundations/data-and-dataset-basics.md) / [泛化、正则化与分布偏移](../foundations/generalization-regularization-and-distribution-shift.md)

### Q：监督学习、无监督学习和自监督学习怎么区分？

**面试回答。** 监督学习使用人工标签，例如 \((x, y)\) 中的 \(y\) 是类别、分数或标注结果；无监督学习没有显式标签，常见目标是聚类、密度估计或表示学习；自监督学习从数据本身构造监督信号。LLM 的 next-token prediction 就是典型自监督：给定前缀 \(x_{<t}\)，预测 \(x_t\)。

\[
\mathcal{L}_{\text{LM}}=-\sum_t \log p_\theta(x_t \mid x_{<t})
\]

**追问展开。** 自监督不是“没有目标”，而是标签来自数据结构本身。比如 masked language modeling 预测被遮住的 token，contrastive learning 判断图文是否匹配，扩散模型预测噪声或 score。  
**易错点 / 边界。** 自监督学到的是训练分布里的可预测结构，不保证事实正确、价值对齐或任务最优。  
**继续读。** [预训练目标与表示学习](../foundations/pretraining-objectives-and-representation-learning.md) / [预训练、微调与对齐](../training/pretraining-finetuning-alignment.md)

### Q：训练集、验证集和测试集为什么必须分开？

**面试回答。** 训练集用于更新参数，验证集用于调超参、早停和选择模型，测试集用于最后估计泛化误差。如果把测试集参与模型选择，就相当于把测试分布的信息泄漏给训练流程，最终指标会偏乐观。一个简洁说法是：训练集回答“怎么学”，验证集回答“选哪个”，测试集回答“真实表现估计是多少”。

**追问展开。** 对时间序列、用户行为、机器人轨迹和生产日志，随机切分往往不够。更合理的切分可能按时间、用户、场景、设备、任务或数据来源分组，避免同源样本同时出现在训练和测试里。  
**易错点 / 边界。** 测试集指标也只是对某个分布的估计；上线后仍要监控数据漂移、长尾失败和安全事件。  
**继续读。** [数据划分、指标与评估基础](../foundations/data-splits-metrics-and-evaluation-basics.md)

### Q：过拟合、欠拟合和泛化该怎么解释？

**面试回答。** 欠拟合是模型或训练过程连训练分布都学不好，表现为训练误差和验证误差都高；过拟合是模型把训练集里的噪声或偶然模式也学进去，表现为训练误差低但验证或测试误差高。泛化能力指模型在未见样本上的表现，实际关心的是训练分布和部署分布之间是否足够一致。

\[
\text{generalization gap} = \mathcal{E}_{\text{test}} - \mathcal{E}_{\text{train}}
\]

**追问展开。** 常见缓解手段包括更多更干净的数据、正则化、数据增强、早停、模型容量控制和更稳的评测切分。对大模型，还要关注数据污染和 benchmark 过拟合。  
**易错点 / 边界。** 训练 loss 和验证 loss 的趋势是信号，不是完整诊断；类别不平衡、分布偏移和指标设计错误也会伪装成过拟合。  
**继续读。** [泛化、正则化与分布偏移](../foundations/generalization-regularization-and-distribution-shift.md) / [评测与消融方法](../training/evaluation-and-ablation-methodology.md)

### Q：梯度消失和梯度爆炸为什么会发生？

**面试回答。** 反向传播本质上是链式法则，深层网络中梯度会经过许多雅可比矩阵连乘。如果这些局部导数的谱范数长期小于 1，梯度会指数衰减；长期大于 1，梯度会放大。RNN 时代这个问题特别典型，Transformer 里残差连接、LayerNorm、初始化和梯度裁剪都在缓解这个问题。

**追问展开。** 面试中可以把解决方案按层次说：结构上用残差和归一化，优化上用 warmup、合理学习率和梯度裁剪，数值上用 BF16/FP32 累加和 loss scale 检查。  
**易错点 / 边界。** loss NaN 不一定就是梯度爆炸，也可能是坏数据、低精度溢出、除零、分布式通信错误或自定义 kernel bug。  
**继续读。** [归一化、残差与激活](../foundations/normalization-residual-and-activation.md) / [训练稳定性与失败排查](../training/stability-numerics-and-failure-triage.md)

### Q：交叉熵为什么常用于分类和语言模型？

**面试回答。** 交叉熵衡量真实分布 \(q\) 和模型预测分布 \(p_\theta\) 的差异。分类中，如果真实标签是 one-hot，交叉熵就是最大化正确类别概率；语言模型中，就是最大化训练文本里每个 next token 的条件概率。

\[
H(q,p_\theta)=-\sum_i q_i \log p_\theta(i), \quad
\mathcal{L}_{\text{CE}}=-\log p_\theta(y\mid x)
\]

**追问展开。** 它和 maximum likelihood 是一回事的两种说法：最小化负对数似然等价于最大化数据在模型下出现的概率。实际训练时还会关心 label smoothing、class imbalance、ignore index 和 token packing。  
**易错点 / 边界。** 交叉熵低不等于回答有用或事实正确；开放生成还要看评估集、偏好、工具结果和用户任务成功。  
**继续读。** [目标函数、优化器与 LR 日程](../training/objectives-optimizers-and-schedules.md) / [生成与解码基础](../foundations/generation-and-decoding-basics.md)

## Transformer 与 Attention

### Q：Transformer 相比 RNN 最大的结构变化是什么？

**面试回答。** Transformer 用 self-attention 让序列中任意两个 token 可以直接交互，不再依赖 RNN 那种逐步传递 hidden state 的递归结构。训练时，所有位置可以并行计算 attention 和 loss，所以长序列训练吞吐更好；建模上，任意距离的 token 只需要一层 attention 就能建立联系。

**追问展开。** 但生成阶段仍是自回归的：第 \(t\) 个 token 必须等前 \(t-1\) 个 token 生成出来。推理优化因此围绕 KV cache、batching、投机解码和服务调度展开。  
**易错点 / 边界。** Transformer 的 \(O(n^2)\) attention 会让长上下文成本变高；它解决了 RNN 的并行训练问题，但引入了新的显存和推理缓存问题。  
**继续读。** [Transformer 输入与注意力](../foundations/transformer-attention-and-tokenization.md) / [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

### Q：Self-attention 里的 Q、K、V 分别是什么？

**面试回答。** Q、K、V 是同一个 token 表示经过三个线性层得到的投影。Query 表示“当前位置想找什么信息”，Key 表示“每个位置能被怎样匹配”，Value 表示“真正被读出的内容”。attention score 来自 Q 和 K 的点积，输出是对 V 的加权求和。

\[
\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
\]

**追问展开。** 除以 \(\sqrt{d_k}\) 是为了控制点积方差，避免维度变大后 softmax 过于尖锐。面试时要强调 Q/K 决定“读哪里”，V 决定“读什么”。  
**易错点 / 边界。** Q/K/V 是计算接口，不代表模型显式存了人类可解释的数据库；单个 head 的语义也不能随意贴标签。  
**继续读。** [Transformer 输入与注意力](../foundations/transformer-attention-and-tokenization.md) / [张量、Shape 与计算图](../foundations/tensors-shapes-and-computation-graphs.md)

### Q：Attention 为什么是 \(O(n^2)\)？

**面试回答。** 对长度为 \(n\) 的序列，每个 query token 都要和所有 key token 算相似度，所以 score matrix 是 \(n \times n\)。如果 hidden size 固定，attention score 的元素数量随序列长度平方增长；训练时还要保存中间结果用于反向传播。

\[
QK^\top \in \mathbb{R}^{n \times n}, \quad \text{cost} \approx O(n^2 d)
\]

**追问展开。** 这个复杂度主要影响训练和 prefill；decode 阶段每步只新增一个 query，但要读取历史所有 K/V，因此更像 \(O(n d)\) 的逐步读缓存问题。  
**易错点 / 边界。** 长上下文性能不能只看理论复杂度，实际还受 FlashAttention、KV cache layout、batch 调度和硬件带宽影响。  
**继续读。** [FlashAttention 与长上下文](../operators/long-context-and-flashattention-evolution.md) / [KV、缓存与投机解码](../inference/caching-routing-and-speculative.md)

### Q：Multi-head attention 为什么要分多个 head？

**面试回答。** 多头注意力把模型维度切成多个子空间，每个 head 单独做 attention，再把结果拼接回去。这样模型可以在不同表示子空间里学习不同关系，例如局部邻近、长距离依赖、语法关系、视觉区域或特殊 token 对齐。

**追问展开。** 如果模型维度是 \(d_{\text{model}}\)，head 数是 \(h\)，常见设置是每个 head 维度 \(d_h=d_{\text{model}}/h\)。head 数增加会改变单 head 维度、并行粒度和 KV cache 组织，不是越多越好。  
**易错点 / 边界。** 多头提供表达能力，但具体 head 是否有清晰语义要靠分析；不能把某个 head 的可视化个例当成稳定机制。  
**继续读。** [Transformer 输入与注意力](../foundations/transformer-attention-and-tokenization.md) / [VLM 架构：视觉表征、连接器与记忆](../vlm/architecture-and-training.md)

### Q：MHA、MQA 和 GQA 的差别是什么？

**面试回答。** MHA 中每个 query head 都有独立 K/V；MQA 让所有 query heads 共享一组 K/V；GQA 介于两者之间，让一组 query heads 共享 K/V。推理 decode 时，K/V 要常驻并反复读取，所以减少 K/V heads 可以显著降低 KV cache 和内存带宽压力。

**追问展开。** 面试里可以这样说取舍：MHA 表达能力强但 KV 成本高，MQA 最省 KV 但可能损失质量，GQA 在质量和成本之间折中。现代 LLM 常用 GQA，是因为服务端 decode 更看重 KV 带宽。  
**易错点 / 边界。** GQA/MQA 的质量影响和训练配方有关；不能把已有模型改个配置就认为无损。  
**继续读。** [推理运行时](../inference/serving-runtimes-vllm-sglang-and-tensorrt-llm.md) / [上下文压缩、KV 驱逐与记忆层级](../inference/context-compression-kv-eviction-and-memory-hierarchies.md)

### Q：为什么现代 LLM 多用 decoder-only 架构？

**面试回答。** decoder-only 架构用 causal mask 做 next-token prediction，预训练目标统一、数据容易构造、生成接口自然。问答、代码、摘要、工具调用都可以转成“给定上下文继续生成”的形式，这让一个模型可以覆盖大量任务。相比 encoder-decoder，它的训练和推理路径更统一，适合大规模自回归语言建模。

**追问展开。** encoder-only 仍适合理解类任务，如分类、检索 embedding、序列标注；encoder-decoder 仍适合翻译、摘要等输入输出边界清晰的任务。decoder-only 胜出的是通用生成和统一接口，不是所有任务的理论最优。  
**易错点 / 边界。** 架构只决定信息流和目标接口，能力仍取决于数据、规模、训练稳定性和对齐。  
**继续读。** [预训练目标与表示学习](../foundations/pretraining-objectives-and-representation-learning.md) / [生成与解码基础](../foundations/generation-and-decoding-basics.md)

### Q：LayerNorm、残差连接和 MLP 在 Transformer 里分别做什么？

**面试回答。** 残差连接让每层学习增量变换，帮助信息和梯度跨层流动；LayerNorm 稳定每个 token 的特征分布，让训练更可控；MLP 对每个 token 独立做非线性变换，增强表示能力。一个常见 Transformer block 可以粗略写成：

\[
x' = x + \operatorname{Attention}(\operatorname{LN}(x)), \quad
y = x' + \operatorname{MLP}(\operatorname{LN}(x'))
\]

**追问展开。** Attention 负责 token 间交互，MLP 负责 token 内特征加工；二者交替堆叠，逐层形成上下文化表示。Pre-LN 和 Post-LN 的差异会影响深层训练稳定性。  
**易错点 / 边界。** 这些模块看似基础，但激活函数、归一化位置、宽度比例和残差缩放都会影响训练稳定和吞吐。  
**继续读。** [归一化、残差与激活](../foundations/normalization-residual-and-activation.md) / [线性层、MLP 与 GEMM](../foundations/linear-layers-mlp-and-gemm.md)

### Q：MoE 为什么能扩参数但不同比例增加计算？

**面试回答。** MoE 的核心是稀疏激活：模型有很多专家参数，但每个 token 只路由到 top-k 个专家。总参数量可以很大，单 token 实际计算只经过少量专家，所以计算量不按总参数线性增长。

\[
y = \sum_{i \in \operatorname{TopK}(g(x))} g_i(x) E_i(x)
\]

其中 \(g_i(x)\) 是 router 给专家 \(E_i\) 的权重。  
**追问展开。** 面试时要补系统成本：MoE 需要专家并行、负载均衡、all-to-all 通信和 router 稳定训练。它不是“免费变大”，而是把 dense compute 换成 routing 和通信问题。  
**易错点 / 边界。** MoE 的收益依赖硬件拓扑、batch、token 分布和 serving 实现；小规模或低并发场景未必划算。  
**继续读。** [MoE 与大模型架构](../foundations/moe-and-large-model-architecture-basics.md) / [MoE 路由与多模型服务](../inference/moe-routing-and-multi-model-serving.md)

## Tokenizer、位置编码与 Mask

### Q：Tokenizer 为什么会影响模型能力和成本？

**面试回答。** Tokenizer 决定文本如何被拆成模型输入 token，因此直接影响上下文长度、训练样本密度、稀有词处理、多语言效率和推理成本。比如同一句中文、英文、代码在不同 tokenizer 下 token 数可能差异很大，token 数越多，attention、KV cache 和计费都越高。

**追问展开。** 一个好的回答要提到：tokenizer 是模型输入接口，不是普通预处理。它会影响模型如何看到词边界、数字、标点、Unicode、代码缩进和特殊符号。BPE、WordPiece、SentencePiece 都是在词表大小、覆盖率和压缩率之间取舍。  
**易错点 / 边界。** Tokenizer 不能单独解决理解问题；它只是把离散文本映射成 token 序列，语义仍要靠模型和训练目标学习。  
**继续读。** [Transformer 输入与注意力](../foundations/transformer-attention-and-tokenization.md) / [Hugging Face BPE 课程](https://huggingface.co/docs/course/chapter6/5)

### Q：BPE 的直觉和训练过程是什么？

**面试回答。** BPE 从字符或字节级 token 开始，统计语料中最常出现的相邻 token pair，并不断把最高频 pair 合并成新 token。这样常见词或片段会被压成少量 token，罕见词仍能拆成更小单位表示。

**追问展开。** 简单例子：如果语料里 `t` 和 `h` 经常相邻，就可能合并成 `th`；后续 `th` 和 `e` 又可能合并成 `the`。训练过程本质是用有限词表做语料压缩。  
**易错点 / 边界。** BPE 合并来自训练语料统计，因此会继承语料语言分布；代码、中文、多语言、数字和 emoji 都可能出现效率或切分问题。  
**继续读。** [Transformer 输入与注意力](../foundations/transformer-attention-and-tokenization.md) / [Karpathy minBPE](https://github.com/karpathy/minbpe)

### Q：位置编码为什么必要？

**面试回答。** Self-attention 只看 token 之间的相似度，如果没有位置信息，它本身对序列顺序不敏感。位置编码告诉模型 token 在序列中的位置或相对距离，否则“猫追狗”和“狗追猫”会更难区分。

**追问展开。** 位置可以通过绝对 embedding 加到输入，也可以像 RoPE 一样进入 Q/K 的几何关系，或像 ALiBi 一样进入 attention score。关键不是“给 token 编号”，而是让模型在注意力计算中感知顺序和距离。  
**易错点 / 边界。** 长上下文能力不仅取决于位置编码，还取决于训练长度、数据、attention 实现、KV cache 和评测任务。  
**继续读。** [位置编码与 Mask](../foundations/positional-encoding-masks-and-context.md) / [RoFormer / RoPE](https://arxiv.org/abs/2104.09864)

### Q：RoPE 的公式直觉是什么？

**面试回答。** RoPE 把每个位置 \(m\) 映射成对 Q/K 向量的旋转变换。二维情况下可以写成：

\[
R_m =
\begin{bmatrix}
\cos m\theta & -\sin m\theta \\
\sin m\theta & \cos m\theta
\end{bmatrix}
\]

对 \(q\) 和 \(k\) 分别旋转后，点积会自然包含相对位置 \(m-n\) 的信息。  
**追问展开。** 面试时可以强调：RoPE 不是把位置向量加到 token 上，而是让位置进入 Q/K 的内积几何。它适合 decoder-only LLM，是因为 causal attention 里相对距离非常重要。  
**易错点 / 边界。** RoPE 外推到更长上下文还需要缩放、训练和评测配合；不能只靠公式保证百万上下文稳定。  
**继续读。** [位置编码与 Mask](../foundations/positional-encoding-masks-and-context.md) / [FlashAttention 与长上下文](../operators/long-context-and-flashattention-evolution.md)

### Q：Causal mask 到底屏蔽了什么？

**面试回答。** Causal mask 阻止当前位置看到未来 token，使第 \(t\) 个位置只能基于 \(x_{<t}\) 预测 \(x_t\)。训练时虽然所有位置并行计算，但每个位置的 attention score 会把未来位置置为 \(-\infty\)，softmax 后概率为 0。

\[
M_{ij} =
\begin{cases}
0, & j \le i \\
-\infty, & j > i
\end{cases}
\]

**追问展开。** 这就是 decoder-only 能并行训练自回归目标的关键：并行算所有位置，但信息可见性仍符合逐 token 生成。  
**易错点 / 边界。** mask 正确不代表数据管线正确；label shift、packing、跨样本 attention 和评测泄漏都可能出错。  
**继续读。** [位置编码与 Mask](../foundations/positional-encoding-masks-and-context.md) / [输入流水线、Packing 与吞吐治理](../training/input-pipelines-packing-and-throughput-governance.md)

## 生成与表示

### Q：Greedy、beam search、top-k 和 top-p 怎么选？

**面试回答。** Greedy 每步选概率最高 token，稳定但容易单调；beam search 保留多条高概率路径，适合翻译、摘要这类目标相对确定的任务；top-k 从最高的 k 个 token 采样；top-p 从累计概率达到 \(p\) 的候选集合采样，更适合开放生成。temperature 控制分布尖锐程度：

\[
p_i = \operatorname{softmax}(z_i / T)
\]

\(T\) 越高越随机，越低越保守。  
**追问展开。** 面试里可以按任务回答：代码补全和结构化输出偏低温，创意写作偏采样，机器翻译可以考虑 beam。  
**易错点 / 边界。** 解码策略只改变输出分布，不会增加模型知识；事实性和工具可靠性要靠检索、约束、评估和验证。  
**继续读。** [生成与解码基础](../foundations/generation-and-decoding-basics.md) / [Hugging Face 生成策略](https://huggingface.co/docs/transformers/generation_strategies)

### Q：为什么 beam search 在开放文本里可能变差？

**面试回答。** Beam search 最大化序列累计概率，开放生成时高概率路径往往更安全、更重复、更模板化。语言模型的局部概率最高不等于全局文本最有信息量，所以 beam 可能导致退化、重复或缺少多样性。

**追问展开。** Nucleus sampling 的动机是动态选择概率质量最高的一小团候选，而不是固定 k。这样既避免低概率乱采样，也避免只沿最高概率路径走到底。  
**易错点 / 边界。** 多样性和事实性是不同维度；更随机可能更有趣，也可能更容易出错。  
**继续读。** [生成与解码基础](../foundations/generation-and-decoding-basics.md) / [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751)

### Q：Embedding 是不是就等于语义？

**面试回答。** Embedding 是模型把离散对象映射到连续向量空间后的表示，常常编码语义、语法、位置、风格或任务相关信息。它不是“语义本身”，而是训练目标下有用的压缩表示。相似度通常用点积或 cosine：

\[
\cos(a,b)=\frac{a^\top b}{\|a\|\|b\|}
\]

**追问展开。** 在 RAG 中，embedding 相似代表“向量空间接近”，不保证事实等价。数字、否定、时间、实体名和精确约束常常需要 BM25、metadata filter 或 rerank 补足。  
**易错点 / 边界。** 向量相似度不能替代任务评估；检索、聚类、推荐都要看真实召回和最终效果。  
**继续读。** [预训练目标与表示学习](../foundations/pretraining-objectives-and-representation-learning.md) / [RAG、Agent 与长上下文系统](../inference/rag-agents-and-long-context-systems.md)

### Q：Prompt、CoT 和 RAG 分别改变了什么？

**面试回答。** Prompt 改变模型看到的上下文和任务说明，CoT 让模型显式或隐式组织中间推理步骤，RAG 把外部知识检索进上下文。三者都不改模型参数，而是改变推理时的信息条件和生成约束。

**追问展开。** 可以用条件概率解释：模型生成的是 \(p_\theta(y \mid x, c)\)，prompt、检索文档、工具结果都在改变条件 \(c\)。如果条件错了，模型可能生成很流畅但错误的答案。  
**易错点 / 边界。** CoT 不保证推理正确，RAG 不保证检索正确，prompt 不保证安全；生产系统要有评估和验证。  
**继续读。** [prompt、CoT 与 RAG 基础](../foundations/prompt-engineering-cot-and-rag-basics.md) / [RAG、Agent 与长上下文系统](../inference/rag-agents-and-long-context-systems.md)

### Q：为什么“上下文更长”不等于“记忆更好”？

**面试回答。** 长上下文只是允许模型看到更多 token，不保证模型能可靠找到、理解并使用远处信息。随着上下文变长，attention 成本、KV cache、指令冲突、信息干扰和位置外推问题都会增加。

**追问展开。** 面试中可以把长上下文和 RAG 对比：长上下文适合一次性读入相关材料，RAG 适合从大知识库中选择相关材料。真实系统常组合摘要、检索、缓存和记忆分层，而不是只扩大窗口。  
**易错点 / 边界。** Needle-in-a-haystack 只测某类定位能力，不代表复杂推理、长程一致性或多轮 agent 可靠。  
**继续读。** [RAG、Agent 与长上下文系统](../inference/rag-agents-and-long-context-systems.md) / [上下文压缩、KV 驱逐与记忆层级](../inference/context-compression-kv-eviction-and-memory-hierarchies.md)

### Q：为什么大模型会出现幻觉？

**面试回答。** 语言模型生成的是条件概率最高的一段文本，而不是从事实数据库里直接取值。当训练数据缺失、上下文证据不足、问题有歧义、检索错误或模型被要求回答超出能力的问题时，它仍可能生成流畅但不真实的内容。

**追问展开。** 公式上可以说模型在做 \(p_\theta(y\mid x)\) 的采样或解码；这个概率来自训练分布，不自带“事实校验器”。降低幻觉要结合 RAG、工具调用、引用、校验器、拒答策略和在线反馈。  
**易错点 / 边界。** 幻觉不是单一问题；事实错误、推理错误、引用错误、工具错误和过时知识要分开诊断。  
**继续读。** [RAG、Agent 与长上下文系统](../inference/rag-agents-and-long-context-systems.md) / [观测与在线评估](../inference/observability-and-online-evaluation.md)
