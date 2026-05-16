---
paper_date: 2025-07
paper_title: "Gemini 2.5: Pushing the Frontier with Advanced Reasoning, Multimodality, Long Context, and Next Generation Agentic Capabilities"
paper_topic: 技术报告
---

# Gemini 2.5：动态 Thinking 与长上下文 Agent

!!! info "报告信息"
    - 技术报告：`Gemini 2.5: Pushing the Frontier with Advanced Reasoning, Multimodality, Long Context, and Next Generation Agentic Capabilities`
    - 模型：`Gemini 2.5 Pro`、`Gemini 2.5 Flash`、`Gemini 2.5 Flash-Lite`
    - 链接：[arXiv:2507.06261](https://arxiv.org/abs/2507.06261)、[PDF](https://arxiv.org/pdf/2507.06261)、[DeepMind PDF mirror](https://storage.googleapis.com/deepmind-media/gemini/gemini_v2_5_report.pdf)
    - 版本：arXiv 2025-07-07 提交，2025-12-19 修订到 v6
    - 关键词：native multimodal sparse MoE、teacher distillation、TPUv5p、Pathways、SFT、reward model、RL、dynamic thinking、1M context、agentic tool use、multimodal post-training

Gemini 2.5 这份报告的核心不是单个 benchmark 第一，而是展示 Google 如何把 **原生多模态 MoE 架构、跨数据中心 TPU 训练基础设施、teacher distillation、长上下文、工具使用和 RL 驱动的 dynamic thinking** 组合成一条完整的前沿闭源模型路线。

| Dimension | Gemini 2.5 |
| --- | --- |
| Model Family | Gemini 2.5 Pro, Gemini 2.5 Flash, Gemini 2.5 Flash-Lite |
| Architecture | native multimodal sparse MoE Transformer |
| Context Length | 1M text tokens |
| Output Length | 64K tokens for Gemini 2.5 Pro / Flash |
| Input Modalities | Text, Image, Video, Audio |
| Main Capability Upgrade | dynamic thinking, coding, math/science reasoning, long context, video/audio understanding, agentic tool use |
| Knowledge Cutoff | January 2025 |
| Training Stack | TPUv5p, Pathways, JAX/PAX, multi-pod synchronous training |

!!! warning "阅读边界"
    这是闭源模型技术报告。报告披露了系统设计、训练阶段、基础设施和大量评测，但没有公开参数规模、精确数据配比、完整优化器超参和权重。因此它适合学习大模型系统路线和训练组织方式，不适合当作可完全复现的 recipe。

## Evidence Snapshot

| Claim | Direct Source | Figure/Table/Setting | Evidence Type | Repro Status | Can Support | Cannot Prove |
| --- | --- | --- | --- | --- | --- | --- |
| Gemini 2.5 把 native multimodal、long context 和 dynamic thinking 组织成闭源前沿模型路线 | [arXiv:2507.06261](https://arxiv.org/abs/2507.06261) | model family, evaluation and infrastructure sections | Paper Result | Paper Only | 技术报告可支持系统设计和能力分层分析 | 不证明训练配方、数据或权重可复现 |
| TPU/Pathways/JAX/PAX 基础设施支撑大规模多模态训练 | 同上 | training infrastructure description | System Throughput | Paper Only | 前沿模型训练依赖跨系统协同 | 不证明其他硬件栈可获得同等收益 |
| 本站将其作为闭源技术报告阅读样例 | 本页与 [证据标准](../../references/evidence-and-reproducibility-standard.md) | 工程归纳 | Site Inference | Site Inference | 可训练读者区分报告自述与复现实验 | 不应作为独立复现证据 |

## 论文位置

Gemini 2.5 可以放在 Gemini 1.5 / 2.0 之后看。Gemini 1.5 的代表特征是长上下文和多模态输入，Gemini 2.0 强调更强的 agentic / tool-use 能力，Gemini 2.5 则把“模型会不会思考、思考多久、何时使用工具、如何处理长视频和长上下文”推到更中心的位置。

这条路线可以压成：

```text
Gemini 2.0 native multimodal sparse MoE
  + larger teacher models and dataset distillation
  + high-quality multimodal / multilingual / code / math pre-training data
  + TPUv5p + Pathways multi-pod training infrastructure
  + SFT / reward models / RL with larger post-training compute
  + dynamic thinking and tool-use training
  -> Gemini 2.5 Pro / Flash / Flash-Lite
```

和 DeepSeek-R1 / Qwen3 一类报告相比，Gemini 2.5 的公开细节更少，但系统跨度更大。DeepSeek-R1 更聚焦可验证推理 RL，Qwen3 更聚焦开源模型家族和 thinking / non-thinking 统一，Gemini 2.5 则把 reasoning、工具、长上下文、视频、音频、搜索和多产品部署放进同一套模型报告里。

![Gemini 2.5 cost-quality Pareto original](https://ar5iv.labs.arxiv.org/html/2507.06261/assets/x1.png){ width="920" }

<small>图源：`Gemini 2.5` technical report，Figure 1，ar5iv 从 arXiv source 渲染的原图。原图展示 Gemini 2.X 不同模型在 cost / price 与 internal quality aggregate 之间的 Pareto 关系，强调 2.5 Pro、2.5 Flash 和 2.5 Flash-Lite 覆盖不同成本和质量点。</small>

!!! note "这张 Pareto 图怎么读"
    这张图的重点不是某个公开 benchmark，而是模型家族设计。Pro 是高质量端，Flash 是质量、速度和成本折中，Flash-Lite 是低成本高吞吐端。技术报告把 2.5 做成系列，而不是只发布一个最强模型，说明训练和后训练目标已经被产品形态反向约束：同一代能力要同时服务深度推理、低延迟对话、工具调用和大规模在线请求。

## 模型家族与接口

报告 Table 1 把 Gemini 1.5、2.0 和 2.5 的接口差异放在一起。下面保留英文列名，便于和原文对照。

| Feature | Gemini 1.5 Flash | Gemini 1.5 Pro | Gemini 2.0 Flash-Lite | Gemini 2.0 Flash | Gemini 2.5 Flash | Gemini 2.5 Pro |
| --- | --- | --- | --- | --- | --- | --- |
| Input modalities | Text, Image, Video, Audio | Text, Image, Video, Audio | Text, Image, Video, Audio | Text, Image, Video, Audio | Text, Image, Video, Audio | Text, Image, Video, Audio |
| Output modalities | Text | Text | Text | Text, Image* | Text, Audio* | Text, Audio* |
| Context length | 1M | 2M | 1M | 1M | 1M | 1M |
| Output length | 8K | 8K | 8K | 8K | 64K | 64K |
| Gemini App Availability | Yes | Yes | No | Yes | Yes | Yes |
| Supports tool use? | No | No | No | Yes | Yes | Yes |
| Thinking | No | No | No | Yes* | Dynamic | Dynamic |
| Knowledge cutoff | Nov 2023 | Nov 2023 | Jun 2024 | Jun 2024 | Jan 2025 | Jan 2025 |

<small>表源：`Gemini 2.5` technical report，Table 1。星号项在原报告中表示实验模型或通过 Live API 提供的能力。</small>

这张表里最值得抓住的是两行：`Supports tool use?` 和 `Thinking`。Gemini 2.0 开始把工具使用变成主路径，Gemini 2.5 则把 thinking 从实验功能变成 Pro / Flash 的动态能力。也就是说，Gemini 2.5 的推理增强不只是“多吐一些 CoT token”，而是和工具、上下文、速度、成本一起进入模型接口设计。

模型 ID 在报告中也单独列出：

| Model | Stable API ID | Preview API ID |
| --- | --- | --- |
| Gemini 2.5 Pro | `gemini-2.5-pro` | `gemini-2.5-pro-preview-06-05` |
| Gemini 2.5 Flash | `gemini-2.5-flash` | `gemini-2.5-flash-preview-05-20` |
| Gemini 2.5 Flash-Lite | `gemini-2.5-flash-lite` | `gemini-2.5-flash-lite-preview-06-17` |

<small>表源：`Gemini 2.5` technical report，Table 2。原表用于说明 2.5 系列的稳定和 preview API 标识。</small>

## 架构与预训练

报告明确写到 Gemini 2.5 延续并扩展 Gemini 2.0 架构：底座是 **native multimodal sparse MoE Transformer**。这里有三个关键词。

第一是 `native multimodal`。文本、图像、视频、音频不是通过独立外部模型临时拼接，而是在同一模型族内作为原生输入模态处理。这会影响 tokenizer、数据混合、attention 上下文、post-training 和安全评测。

第二是 `sparse MoE`。报告没有公开总参数量、激活参数量和 expert 数，但选择 MoE 的目标很清楚：在较高模型容量下控制每 token 激活计算，并为 Pro / Flash / Flash-Lite 这样的多成本点模型家族提供扩展空间。

第三是 `distillation`。报告提到 Gemini 2.5 Pro 和 Flash 都从更大的 teacher models 中蒸馏而来，并使用改进的 post-training 和 dataset distillation 技术。这意味着 2.5 的能力不只来自一次大规模预训练，也来自 teacher 产生的数据、筛选规则、偏好信号和后训练迁移。

| Training Ingredient | Reported Detail | Engineering Meaning |
| --- | --- | --- |
| Architecture | native multimodal sparse MoE Transformer | 用统一架构承载文本、图像、视频、音频和工具场景，同时控制激活计算 |
| Teacher distillation | Pro and Flash are distilled from larger teacher models | 把更大模型能力压到可部署模型，降低服务成本 |
| Pre-training data | web documents, code, math, video, audio, image-text interleaved data | 能力来源覆盖通用知识、代码、STEM、多模态理解和多语 |
| Language coverage | over 100 languages | 多语不是后训练补丁，而是进入预训练数据工程 |
| Data processing | filtering, deduplication, quality and safety filters | 训练收益依赖数据治理，而不只是 token 数 |
| Knowledge cutoff | January 2025 | 对事实问答、搜索工具和 benchmark 解读都很重要 |

!!! note "训练细节里最关键的不是参数量"
    Gemini 2.5 没有披露参数规模，但报告反复强调数据质量、蒸馏、后训练和基础设施。这说明闭源前沿模型的技术壁垒已经不只是“模型有多大”，而是 teacher 模型、数据筛选、后训练环境、自动评测、RL 稳定性和跨数据中心训练系统一起构成的能力栈。

## 训练基础设施：TPUv5p、Pathways 与跨数据中心训练

Gemini 2.5 的基础设施部分值得仔细看，因为它解释了大模型训练为什么是系统工程，而不是单机算法问题。

报告中写到，Gemini 2.5 在 TPUv5p 上训练，使用多个 8960-chip pod，且这些 pod 分布在多个数据中心。训练软件栈使用 Pathways SDK、JAX 和 PAX，提供单控制器编程模型和 many-way SPMD 分区。

| Infrastructure Component | Gemini 2.5 Detail | Why It Matters |
| --- | --- | --- |
| Accelerator | TPUv5p | 大规模矩阵计算和高带宽互联是训练吞吐基础 |
| Scale | multiple 8960-chip pods | 单个训练 run 跨越多个完整 TPU pod |
| Geography | multiple data centers | 资源调度和容灾能力进入训练系统本身 |
| Programming stack | Pathways SDK, JAX, PAX | 把跨 pod / 跨数据中心并行隐藏在统一编程模型下 |
| Parallelism | many-way SPMD, model/data parallelism | 支撑 MoE、多模态和长上下文训练的计算/显存划分 |
| Synchronization | synchronous data-parallelism across TPU pods | 让超大 batch 和多 pod 训练保持一致更新 |
| Data loading | side-car loaders streaming compressed data from object storage | 输入管线不被远端数据读取和预处理拖垮 |

跨数据中心同步训练的难点在于通信、故障和静默错误。报告提到它们在训练中做了三类系统设计。

第一是全局同步和调度。训练需要跨 inter-DCN 网络做 global all-reduce，并通过自定义调度让不同 slice 的工作负载尽量均匀。否则某个 slice 慢一点，就会拖住整个同步 step。

第二是弹性恢复。系统可以在 slice 粒度上恢复失败切片，而未受影响的 slice 继续保留状态。报告给出的结果是，恢复过程中训练吞吐可保持在 97% 以上，故障恢复时间在数十秒量级。

第三是 SDC 检测。SDC 指 silent data corruption，也就是硬件或系统没有显式报错，但计算结果已经错了。报告采用 split-phase SDC detection：疑似异常 step 会 replay，确认错误后再回滚或修复。报告披露约 0.25% 的 steps 进入 replay，其中约 6% 被确认为真实 corruption；早期实验里 TPU 计算利用率约 93.4%，额外 replay / rollback 计算约 4.5%，downtime 约 0.6%。

!!! note "为什么 SDC 细节很重要"
    在小模型训练里，偶发数值错误常常被当成随机波动；在跨数万芯片的大模型训练里，静默错误会变成确定会遇到的系统问题。Gemini 2.5 把 replay、rollback 和 slice-level recovery 写进报告，说明前沿模型训练已经默认要为硬件错误、网络抖动和数据输入故障设计控制面。

## 后训练：SFT、Reward Model 与 RL

报告的 post-training 部分没有给出完整数据规模，但把训练阶段讲得比较清楚：Gemini 2.5 继续使用 SFT、reward models 和 RL，并在 Gemini 1.5 / 2.0 的基础上加大了后训练 compute、数据多样性和稳定性工程。

| Post-training Component | Gemini 2.5 Detail | Training Role |
| --- | --- | --- |
| SFT | higher-quality supervised data with model-assisted quality-control loops | 建立指令跟随、格式、领域任务和多模态行为基础 |
| Reward Models | improved reward models for reasoning, tool use and multimodal tasks | 给开放生成任务、复杂偏好和非规则任务提供学习信号 |
| RL | increased compute and longer, more stable RL runs | 把推理、工具、代码和多模态任务的策略行为继续拉高 |
| Verifiable Rewards | code execution and other automatically checkable signals | 在代码、数学、工具任务上提供低噪声奖励 |
| Model-based Rewards | model judges for generative or subjective tasks | 覆盖无法用单一规则判分的回答质量和偏好 |
| Multimodal Data | diverse image, video and audio post-training data | 防止后训练只优化文本对话，保住原生多模态能力 |

这里最重要的工程点是：Gemini 2.5 的 RL 不只服务数学题。报告明确把 reasoning、tool use、multimodal capability、coding 和 factuality 放进同一个后训练框架。它的奖励来源也分两类：可验证任务用执行器或规则给强信号，开放任务用 reward model / judge 给软信号。

### 代码后训练

Gemini 2.5 的代码能力提升来自三条线。

1. 预训练数据里包含大量高质量 code 和软件工程相关数据；
2. 后训练阶段使用 web、repositories 和 synthetic code data，并强调复杂真实软件工程问题；
3. Google 内部软件工程师贡献了大量 curated coding tasks，用来提升真实开发任务的覆盖度。

报告把 SWE-Bench Verified、Aider Polyglot 等软件工程 benchmark 作为重要证据。读这些结果时要注意，它们不只是“会写函数”的指标，而是在测试模型能否理解仓库、修改多文件、遵循测试和处理工具反馈。

### 事实性和搜索

Gemini 2.5 还把事实性训练和工具使用结合起来。报告说模型会被训练去判断什么时候需要 Search，如何整合检索结果，如何处理冲突证据，以及如何引用来源。

这和普通 RAG 的区别在于：RAG 可以由外部系统决定检索，模型只负责读 retrieved chunks；Gemini 2.5 的目标更接近 agentic search：模型自己决定是否搜索、搜什么、读哪些结果、如何综合和引用。

## Dynamic Thinking

Gemini 2.5 的关键变化是 `dynamic thinking`。它不是给所有请求固定一段很长的思考 token，而是让模型根据任务难度动态决定是否思考、思考多少、什么时候停止。

报告把 thinking 放在 RL 之后讲，这很重要。Thinking 不是单纯的 prompt trick，而是后训练行为：模型通过 RL 和推理轨迹相关训练学会在复杂任务中分解问题、检查中间结果、使用工具和调整解题路径。

| Mechanism | Gemini 2.5 Meaning | Training / Inference Impact |
| --- | --- | --- |
| Dynamic thinking | model decides how much reasoning to allocate | 同时优化质量、延迟和成本 |
| Thinking budget | Flash / Flash-Lite can expose controls over thinking cost | 产品侧可以用预算控制 test-time compute |
| RL for reasoning | post-training rewards encourage better multi-step reasoning | 让思考行为和任务成功率绑定，而不是只变长 |
| Tool-aware reasoning | thinking can interact with search, code execution and other tools | 复杂任务中把内部推理和外部反馈组合起来 |
| Long output | 64K output length for 2.5 Pro / Flash | 支撑长推理、长代码修改、长报告和多步 agent 输出 |

<small>原图位置：`Gemini 2.5` technical report 中 thinking budget 相关图，见 [DeepMind PDF mirror](https://storage.googleapis.com/deepmind-media/gemini/gemini_v2_5_report.pdf#page=6)。该图展示 Gemini 2.5 Flash 在不同 thinking budget 下的 benchmark 表现变化，用来说明 test-time compute 与任务质量之间的可控权衡。</small>

!!! note "不要把 dynamic thinking 等同于可见 CoT"
    报告讨论的是模型内部推理和训练方式，不等于产品必须暴露完整 chain-of-thought。对工程系统来说，更有用的抽象是 test-time compute allocation：简单请求少想或不想，复杂数学、代码、搜索和长上下文任务多分配推理预算。

Dynamic thinking 的工程意义在于，它把“推理质量”和“服务成本”拉到同一个控制面。固定长思考会让简单任务变慢变贵；完全不思考又会损害复杂任务。动态预算让模型族可以在 Pro / Flash / Flash-Lite 之间覆盖不同 SLA。

## Agentic 能力与工具训练

Gemini 2.5 报告把 agentic capability 放在核心位置。这里的 agent 不是简单多轮聊天，而是模型在长上下文和多步环境中调用工具、读取反馈、修正计划并完成任务。

| Agentic Area | Reported Training Signal | Capability Target |
| --- | --- | --- |
| Tool use | tool-use demonstrations, API calls, execution feedback | 学会选择工具、填参数、读结果 |
| Search / browsing | search decision, evidence integration, source citation | 在知识不确定或过期时主动检索 |
| Code execution | executable rewards, tests, repository tasks | 用运行反馈修复代码和验证答案 |
| Multimodal tools | image/video/audio inputs with tool interaction | 在多模态任务里结合外部工具 |
| Long-horizon tasks | extended context and multi-step trajectories | 跨多轮、多文件、多证据保持任务状态 |

报告中的 Deep Research agent 是一个很好的例子：它不是一次性问答，而是把 reasoning 和 search 迭代组合起来，逐步形成报告。这里真正训练的是“决策流程”：什么时候搜索，什么时候继续读，什么时候整合，什么时候停止。

这类能力和 [RAG、Agent 与长上下文系统](../../inference/rag-agents-and-long-context-systems.md) 里的工程问题直接相关。模型越会自己使用工具，外部 agent 框架就越需要把权限、安全、成本、缓存、日志和回滚做好；否则模型能力越强，系统风险也越大。

## 多模态与长上下文训练

Gemini 2.5 的多模态不是只做图文问答。报告把视觉、视频、音频和多语都放进模型能力范围，尤其强调长视频和实时音频。

| Modality | Gemini 2.5 Training / System Detail | Why It Matters |
| --- | --- | --- |
| Text | large-scale multilingual and high-quality web/code/math data | 通用知识、推理和代码能力基础 |
| Image | interleaved image-text data and multimodal post-training | 支撑视觉问答、图表、UI 和文档理解 |
| Video | more efficient video tokenizer, large-scale video pre/post-training | 更长视频上下文和更强时序理解 |
| Audio | non-semantic and causal representations, real-time-oriented training | 支撑语音、环境音和实时交互 |
| Multilingual | over 100 languages, tokenizer and data-mixture improvements | 减少多语能力只靠翻译迁移 |

视频部分的一个具体细节很有价值：报告说 Gemini 2.5 的视频 tokenizer 比前代更高效，每帧视觉 token 数从上一代的 258 降到 66。以 1M context 计算，这让可处理视频时长从约 1 小时提高到约 3 小时。这个提升不是“模型更聪明”这么简单，而是 tokenization、上下文预算和视频采样策略一起带来的。

!!! note "长视频能力首先是 token budget 问题"
    1M context 听起来很大，但视频会迅速吃掉 token。每帧 258 个 token 和 66 个 token 的差别，会直接决定模型能看几十分钟还是几小时。对视频/世界模型系统来说，视觉 tokenizer、帧率、压缩特征和上下文调度往往和 Transformer 本体同样重要。

音频部分也体现了类似思路。报告提到使用 non-semantic representations 处理不需要转写成文本的任务，并使用 causal representations 支撑实时原生音频理解。这说明 Gemini 2.5 不只是把语音先转 ASR 文本再喂给 LLM，而是在模型侧保留更多音频特征和实时约束。

## Benchmark 结果怎么看

Gemini 2.5 报告覆盖大量 benchmark，包括通用文本、数学、科学、代码、软件工程、长上下文、多模态、视频、音频和 agentic browsing。读这些结果时，最好按能力族看，而不是记单个分数。

| Benchmark Area | Evidence in Report | Interpretation |
| --- | --- | --- |
| Reasoning | math/science and difficult QA benchmarks | dynamic thinking 和 RL 后训练提高复杂推理 |
| Coding | LiveCodeBench, SWE-Bench Verified, Aider Polyglot | 不只考函数补全，也考真实仓库修改和工具反馈 |
| Long Context | MRCR and long-context retrieval / reasoning tasks | 1M context 不是只做 needle retrieval，还要长程综合 |
| Multimodal | MMMU, MathVista, video and audio evaluations | 原生多模态训练在图表、视觉推理和视频理解上体现收益 |
| Agentic Browsing | Deep Research and browsing-oriented evaluations | 搜索、证据整合和引用来源成为模型后训练目标 |
| Cost / Latency | Pro / Flash / Flash-Lite Pareto and speed comparisons | 模型家族用不同计算预算覆盖不同产品场景 |

这类评测的解读要谨慎。Gemini 2.5 是闭源模型，很多对比来自作者报告的统一评测设置；不同厂商的工具权限、thinking 配置、搜索环境、temperature、上下文长度和系统 prompt 都可能影响结果。更稳的结论是：报告证据支持 Gemini 2.5 在 reasoning、coding、多模态、长上下文和 agentic 任务上整体升级，但不应该把所有数字直接外推到自己的线上系统。

## 训练路线的可复用经验

如果把 Gemini 2.5 当作工程参考，它最值得复用的不是某个隐藏参数，而是几条系统原则。

| Reusable Principle | Gemini 2.5 Example | Project Implication |
| --- | --- | --- |
| 模型家族先于单点模型 | Pro / Flash / Flash-Lite 覆盖质量、速度、成本 | 训练目标要提前考虑部署层级和 SLA |
| 蒸馏是主线能力工程 | Pro / Flash 从更大 teacher 蒸馏 | 大模型团队需要把 teacher、数据生成和筛选做成长期系统 |
| 后训练不只做偏好对齐 | SFT/RM/RL 覆盖 reasoning、tool、code、多模态 | reward 和评测要按任务族拆分 |
| Thinking 要和预算绑定 | dynamic thinking / thinking budget | 推理质量必须和延迟、token 成本共同优化 |
| 多模态能力依赖 tokenizer | video token 从 258/frame 降到 66/frame | 模态压缩和上下文管理直接决定可用能力 |
| 训练系统要假设故障常态化 | slice recovery, SDC replay, cross-DC synchronization | 大规模训练需要控制面、容错和数据管线工程 |
| Agent 训练需要环境反馈 | search, browsing, code execution, tool APIs | 静态 SFT 不足以训练可靠工具使用 |

对自己的项目来说，最直接的启发是：不要把“推理模型”只理解成长 CoT 数据集。真正可落地的推理模型需要四类东西一起工作：可验证或可评估的任务环境、足够稳定的 RL / reward pipeline、能控制 test-time compute 的推理接口，以及可观测的工具执行系统。

## 局限与不可外推

Gemini 2.5 报告很强，但也有明显边界。

1. **不可复现性**：没有公开权重、参数规模、精确数据配比、optimizer 配方和训练超参。
2. **评测依赖设置**：thinking budget、工具权限、搜索环境和系统 prompt 会影响 benchmark。
3. **内部指标不可比较**：Figure 1 的 internal quality aggregate 不能和其他厂商公开指标直接比较。
4. **Agent benchmark 仍不稳定**：多步工具任务对环境、权限、运行失败和缓存很敏感。
5. **多模态压缩细节有限**：报告给了 video token 数等关键线索，但没有完整公开 tokenizer 和训练数据构造。
6. **Thinking 不等于可解释性**：动态思考提高任务表现，不代表用户能看到或审计完整内部推理。

这些边界不削弱报告价值，反而说明应该如何读闭源技术报告：它适合提供系统路线和工程判断，不适合当作逐项复现清单。

## 项目启发

如果要把这篇报告转成自己的研发任务，可以按下面拆。

| Workstream | What To Build | Checkpoint |
| --- | --- | --- |
| Data | 多语、代码、数学、多模态和工具轨迹的数据治理管线 | 是否能按任务族追踪数据质量和失败模式 |
| Post-training | SFT、reward model、RL、verifier、judge 和 rollout 系统 | 是否有可复用的 reward / evaluation harness |
| Thinking | 可配置的 reasoning budget 和任务难度路由 | 是否能在质量、延迟、成本之间做在线权衡 |
| Tools | 搜索、代码执行、API 调用、权限和日志 | 是否能复盘每次工具调用和失败原因 |
| Long Context | tokenizer、packing、检索、压缩和 KV 成本管理 | 1M context 是否真的转化为任务成功率 |
| Infra | 分布式训练、checkpoint、replay、故障恢复和数据输入 | 长 run 是否能在硬件故障下继续稳定推进 |

Gemini 2.5 最值得学习的一句话是：前沿模型能力不是单一算法堆出来的，而是训练数据、teacher、post-training、推理预算、工具环境和大规模基础设施一起优化出来的。

## 延伸阅读

- [大模型训练路线图](../../training/index.md)：预训练、后训练、数据治理和分布式系统主线。
- [后训练数据引擎与 Judge 模型](../../training/post-training-data-engines-and-judge-models.md)：理解 SFT、judge、reward 和数据回流。
- [Policy Gradient、PPO 与 GRPO](../../reinforcement-learning/policy-gradient-actor-critic-ppo-grpo.md)：连接 reasoning RL 和 reward-driven post-training。
- [RAG、Agent 与长上下文系统](../../inference/rag-agents-and-long-context-systems.md)：理解搜索、工具调用和长上下文推理服务系统。
- [VLM/VLA 总览](../../vlm/index.md)：连接多模态表示、视频、音频和动作系统。
