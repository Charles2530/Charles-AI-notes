# AI 基础知识地图

这一组页面是后续专题的公共底座。它不追求把数学、深度学习和系统工程全部讲完，而是把阅读扩散模型、VLM、VLA、量化、训练、推理和算子时最常反复出现的概念提前讲清。

!!! note "在主线里的位置"
    基础知识不是世界模型高效训练的终点，而是全站共用语言。读这里时优先抓住张量、token、attention、概率、优化、显存和 runtime 如何支撑后续的状态表示、动作条件、长序列训练与低比特系统；不必把它当成从头刷完的教材。

![Transformer architecture 原论文图](../assets/images/paper-figures/transformer/attention-is-all-you-need-figure-1.png){ width="500" .atlas-figure-tall }

<small>图源：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)，Figure 1。原论文图意：展示 Transformer encoder-decoder 的整体结构，包括 token embedding、positional encoding、multi-head attention、feed-forward network、residual connection、normalization 和最终 softmax。</small>

!!! note "图解：基础知识最好按层读"
    这张 Transformer 原图几乎串起了后续所有基础概念：输入先变成 token embedding，再加入位置编码；attention 负责跨 token 信息交互；feed-forward/MLP 负责逐 token 非线性变换；残差和归一化保证深层训练稳定；最后通过 softmax 形成概率输出。遇到新名词时，先判断它属于输入表示、模型结构、训练目标还是运行系统，概念就不容易混在一起。

!!! tip "这一组怎么读"
    基础知识不是考试大纲，先按“够用即止”读。每页只需要抓住三件事：这个概念解决什么问题，它在后续哪个专题会反复出现，遇到错误时它能帮你排查哪一类原因。公式和实现细节可以第二遍再补。

<div class="atlas-meta-grid">
  <div>
    <strong>先建立词汇表</strong>
    <p>知道 tensor、token、embedding、loss、gradient、runtime 分别属于哪一层。</p>
  </div>
  <div>
    <strong>再理解数据流</strong>
    <p>看清输入如何变成表示，表示如何经过模块，loss 如何反向影响参数。</p>
  </div>
  <div>
    <strong>最后接工程问题</strong>
    <p>把显存、带宽、dtype、kernel、checkpoint 和评测指标接到真实系统里。</p>
  </div>
</div>

## 阅读分层与小地图

| 分层 | 推荐内容 | 读到什么程度 |
| --- | --- | --- |
| 主线必读 | 张量/Shape、Transformer/Attention、概率潜变量、优化、数值显存运行时 | 能读懂后续页面里的状态、动作、loss、KV 和低比特成本 |
| 工程查阅 | 自动微分、位置编码与 mask、GEMM、运行时基础 | 碰到显存、shape、mask 或 kernel 问题时查 |
| 论文扩展 | Transformer、DDPM、Chinchilla、FlashAttention 等原图 | 用原图建立概念，不把二手解释当作证据 |
| 基础补课 | 线性代数、概率、优化、计算图 | 哪个概念卡住补哪个 |
| 暂时可跳过 | 与当前专题无关的细节推导 | 先服务主线阅读，不追求一次学完 |

本专题是**公共语言层**。它不直接证明世界模型能训练得更高效，但能让读者看懂后面每个成本账为什么成立。

## 为什么要单独有基础知识模块

很多专题页面都会反复出现同一批基础概念：

- 扩散模型会讲 `UNet`、`DiT`、`score`、`SDE/ODE`、`ELBO`。
- VLM、VLA 和 agent 会讲 `token`、`embedding`、`cross-attention`、视觉编码器、动作 token、多模态上下文和 CoT。
- 量化会讲 `FP8`、`INT4`、显存、带宽、kernel 和 runtime。
- 训练会讲 loss、反向传播、optimizer、checkpoint、mixed precision。
- 算子与编译器会讲 GEMM、attention kernel、通信、访存和数值误差。

如果每个专题都从零解释，页面会重复且分散；如果完全不解释，初学者会很难把概念串起来。因此这里把通用部分抽出来，后面专题只需要链接回来。

## 建议按折叠分组阅读

基础知识现在按“先看数据如何表示，再看模型如何处理，再看训练如何成立，最后看系统如何跑起来”的顺序组织。侧边栏里的二级分组可以折叠；读者不必一次展开全部，遇到后续专题卡壳时回到对应分组即可。

| 分组 | 先解决的问题 | 页面 |
| --- | --- | --- |
| 表示与计算 | 数据如何变成张量，基础模块如何把张量变成特征 | [张量、Shape 与计算图](tensors-shapes-and-computation-graphs.md)、[线性层、MLP 与 GEMM](linear-layers-mlp-and-gemm.md)、[卷积与视觉特征](convolution-and-feature-extraction.md) |
| 结构与上下文 | token 之间如何交互，长序列和深层网络为什么能工作 | [Transformer 输入与注意力](transformer-attention-and-tokenization.md)、[位置编码与上下文 Mask](positional-encoding-masks-and-context.md)、[Mamba 与混合 SSM 架构](mamba-and-hybrid-ssm-transformers.md)、[Norm、残差与激活函数](normalization-residual-and-activation.md) |
| 交互与多模态 | prompt、RAG、CoT、视觉和多模态上下文如何接到模型 | [Prompt、CoT 与 RAG 入门](prompt-engineering-cot-and-rag-basics.md)、[多模态推理入门](multimodal-cot-and-reasoning-basics.md) |
| 训练目标与证据 | 模型在优化什么，能力是否真的变好，评测证据是否可靠 | [概率与潜变量模型](probability-latent-variables-and-generative-models.md)、[优化与训练入门](optimization-and-training-basics.md)、[数据划分与评测指标](data-splits-metrics-and-evaluation-basics.md) |
| 系统执行与成本 | 训练为什么吃显存，低精度、runtime 和硬件瓶颈如何限制效率 | [自动微分与激活显存](autograd-activation-checkpointing-and-memory.md)、[数值、显存与运行时](numerics-memory-and-runtime-basics.md) |

## 一条最短主线

如果只想快速建立直觉，可以抓住这条链：

\[
\text{Tensor} \rightarrow \text{Module} \rightarrow \text{Loss} \rightarrow \text{Gradient} \rightarrow \text{Runtime}
\]

也就是：

1. 数据先被组织成张量。
2. 张量通过 Conv、Attention、MLP 等模块流动。
3. 模型输出和目标之间形成 loss。
4. 反向传播把 loss 转成参数梯度。
5. runtime、kernel 和硬件决定这套计算能否高效执行。

这条链一旦打通，后面读扩散、量化和训练系统时会容易很多。

## 和后续专题的接口

| 基础页 | 后续最常用在哪里 | 典型问题 |
| --- | --- | --- |
| 张量与计算图 | 训练、算子、VLM、VLA | 为什么 shape 对不上，为什么显存爆了 |
| 线性层与 GEMM | Transformer、量化、算子 | 为什么大模型性能经常卡在矩阵乘 |
| 卷积与特征提取 | 扩散 UNet、视觉编码器、VLM | 为什么局部纹理和层级特征重要 |
| Transformer 与 Attention | LLM、VLM、DiT、推理 KV cache | 为什么 attention 既强又贵 |
| Mamba 与状态空间模型 | 长上下文、流式推理、混合 Mamba-Transformer 架构 | 为什么 recurrent state 能降低 KV cache 压力，为什么混合架构会流行 |
| 提示工程、CoT 与 RAG | 推理服务、Agent、企业知识库、工具系统 | prompt 如何定义任务边界，RAG 如何组织证据 |
| 多模态、CoT 与推理 | VLM、Omni、agent、世界模型 | 不同模态如何进模型，thinking token 如何服务复杂任务 |
| 位置编码与 Mask | 长上下文、VLM、VLA、推理 | 为什么 token 顺序、可见性和上下文成本必须一起看 |
| 归一化与残差 | 深层网络、低精度训练 | 为什么模型能训深，为什么数值会崩 |
| 概率与生成模型 | 扩散、世界模型、VAE | 为什么要采样，为什么要学分布 |
| 优化与训练 | 预训练、后训练、QAT | 为什么 loss 下降不等于能力可靠 |
| 自动微分与显存 | 大模型训练、多模态训练 | 为什么训练显存远高于推理，checkpointing 如何省显存 |
| 数据与评测 | 训练、量化、推理、VLM | 为什么平均分不够，为什么需要分桶和回放 |
| 数值与运行时 | 量化、推理、算子 | 为什么格式能保存不等于能跑快 |

## 反向症状索引：训练里出事时先回看哪页

基础知识最有用的读法，是从真实故障倒推概念。世界模型训练或 VLA/WAM 系统里遇到下面这些症状时，可以先回到对应基础页，而不是只在模型名之间横跳。

| 现场症状 | 先回看 | 能做出的判断 |
| --- | --- | --- |
| 多相机视频一接入就 OOM，或者 `32k` context 很快被吃满 | [张量、Shape 与计算图](tensors-shapes-and-computation-graphs.md)、[Transformer 输入与注意力](transformer-attention-and-tokenization.md) | 是 patch/token 数、activation、KV cache 还是中间 materialize 在爆 |
| 长上下文 rollout token/s 掉得很快，KV 显存压不住 | [位置编码与上下文 Mask](positional-encoding-masks-and-context.md)、[数值、显存与运行时](numerics-memory-and-runtime-basics.md) | 是 \(L^2\) attention、KV 生命周期、mask 结构还是 dtype/kernel 支持问题 |
| rollout 越滚越漂，risk head 很自信但错 | [概率与潜变量模型](probability-latent-variables-and-generative-models.md)、[数据划分与评测指标](data-splits-metrics-and-evaluation-basics.md) | 是不确定性没校准、latent 丢了关键状态，还是评测桶没覆盖长尾 |
| loss 下降但闭环成功率不涨 | [优化与训练入门](optimization-and-training-basics.md)、[数据划分与评测指标](data-splits-metrics-and-evaluation-basics.md) | 是目标函数不对、数据分布偏了，还是平均指标掩盖关键任务退化 |
| FP8/低比特短跑稳定，长跑后 loss spike 或 NaN | [Norm、残差与激活函数](normalization-residual-and-activation.md)、[数值、显存与运行时](numerics-memory-and-runtime-basics.md) | 是 activation outlier、scale 策略、累加精度还是敏感层保护不足 |
| 训练吞吐低但 GPU 利用率看似不差 | [线性层、MLP 与 GEMM](linear-layers-mlp-and-gemm.md)、[自动微分与激活显存](autograd-activation-checkpointing-and-memory.md) | 是 GEMM shape 不合适、launch 太碎、checkpoint 重算过多还是数据搬运问题 |
| 视频状态很会描述画面，但控制时丢小物体或接触细节 | [卷积与视觉特征](convolution-and-feature-extraction.md)、[多模态推理入门](multimodal-cot-and-reasoning-basics.md) | 是视觉 encoder 没保局部细节、token 压缩太狠，还是多模态对齐丢了任务变量 |
| Agent/RAG 工具链回答像样但不可回放、不可评测 | [Prompt、CoT 与 RAG 入门](prompt-engineering-cot-and-rag-basics.md)、[数据划分与评测指标](data-splits-metrics-and-evaluation-basics.md) | 是输入契约、证据组织、输出 schema 还是评测闭环没建立 |

## 从论文专题回填的基础概念

论文专题里已经出现了一批会反复回到基础层的概念。读这些内容时，不要只记模型名字，先把它们归到下面几类：

| 专题来源 | 回填到基础层的知识点 | 建议回看 |
| --- | --- | --- |
| [Muon](../paper-deep-dives/foundations/muon-is-scalable.md)、[Kimi K2](../paper-deep-dives/technical-reports/kimi-k2.md)、[DeepSeek-V4](../paper-deep-dives/technical-reports/deepseek-v4-pro.md) | 优化器不只是 AdamW 参数表；Muon / MuonClip 强调更新矩阵几何、谱范数约束和大规模 MoE 训练稳定性 | [优化与训练入门](optimization-and-training-basics.md) |
| [Nemotron 3 Super](../paper-deep-dives/technical-reports/nemotron-3-super.md) | Mamba-heavy backbone、GQA、LatentMoE 和 shared-weight MTP 说明“注意力、状态空间、MoE、预测头”是不同结构层 | [Mamba 与混合 SSM 架构](mamba-and-hybrid-ssm-transformers.md)、[Transformer 输入与注意力](transformer-attention-and-tokenization.md) |
| [GPT-4o System Card](../paper-deep-dives/technical-reports/gpt-4o-system-card.md)、[Qwen3.5-Omni](../paper-deep-dives/technical-reports/qwen35-omni.md) | 多模态不是“把图片转文字再给 LLM”；音频、视觉、文本、动作 token 都会改变输入表示、评测和安全边界 | [多模态推理入门](multimodal-cot-and-reasoning-basics.md)、[数据划分与评测指标](data-splits-metrics-and-evaluation-basics.md) |
| [Qwen3](../paper-deep-dives/technical-reports/qwen3.md)、推理与 Agent 系统 | CoT、thinking budget、RAG 和工具调用都属于“上下文如何被组织和使用”的问题；prompt 只是其中一个接口 | [Prompt、CoT 与 RAG 入门](prompt-engineering-cot-and-rag-basics.md)、[多模态推理入门](multimodal-cot-and-reasoning-basics.md) |
| [SLA / SLA2](../paper-deep-dives/foundations/sla-sparse-linear-attention.md)、[RingAttention](../paper-deep-dives/world-models/ringattention.md) | attention 的数学定义和实际执行路径不同；稀疏、线性、blockwise 和 sequence parallel 都是在改变可见性、访存和通信方式 | [位置编码与上下文 Mask](positional-encoding-masks-and-context.md)、[数值、显存与运行时](numerics-memory-and-runtime-basics.md) |

## 进入专题前的检查

基础模块不需要一次性背完，但读后续专题前最好能形成四个判断：

| 检查点 | 你应该能说清什么 | 如果卡住，回看 |
| --- | --- | --- |
| 表示 | 数据如何变成 tensor、token、embedding、recurrent state 或 latent state | [张量、Shape 与计算图](tensors-shapes-and-computation-graphs.md)、[Transformer 输入与注意力](transformer-attention-and-tokenization.md)、[Mamba 与混合 SSM 架构](mamba-and-hybrid-ssm-transformers.md)、[多模态推理入门](multimodal-cot-and-reasoning-basics.md) |
| 目标 | loss、概率目标或生成目标到底在约束什么 | [概率与潜变量模型](probability-latent-variables-and-generative-models.md)、[优化与训练入门](optimization-and-training-basics.md) |
| 执行 | 训练和推理为什么会受显存、带宽、dtype 和 kernel 影响 | [自动微分与激活显存](autograd-activation-checkpointing-and-memory.md)、[数值、显存与运行时](numerics-memory-and-runtime-basics.md) |
| 证据 | 一个方法变好，是平均分变好，还是关键任务桶真的变好 | [数据划分与评测指标](data-splits-metrics-and-evaluation-basics.md) |

读扩散、VLM、训练系统或推理服务时，如果发现自己只能记住术语，不能判断它属于哪一层，优先回到这张表补接口。

## 一个总判断

基础知识不是“学完才能开始”的门槛，而是一个随时回查的地图。读后面的高级专题时，如果发现概念开始混乱，优先回到这里判断它属于哪一层，再继续追具体方法。
## 深度标杆页

基础知识专题的标杆页要负责给后续世界模型、训练、推理和算子章节打地基。其它基础页应向这些页面补直觉、公式、最小例子和跨专题链接。

| 标杆页 | 对齐标准 |
| --- | --- |
| [张量、Shape 与计算图](tensors-shapes-and-computation-graphs.md) | 把 shape、计算图、数据流和框架执行联系起来 |
| [Transformer 输入与注意力](transformer-attention-and-tokenization.md) | 讲清 token、attention、KV 和上下文成本 |
| [概率与潜变量模型](probability-latent-variables-and-generative-models.md) | 支撑扩散、VAE、RSSM、世界模型和不确定性 |
| [优化与训练入门](optimization-and-training-basics.md) | 支撑 loss、optimizer、稳定性和训练诊断 |
| [数值、显存与运行时](numerics-memory-and-runtime-basics.md) | 把 dtype、显存、带宽、runtime 和系统成本接起来 |
