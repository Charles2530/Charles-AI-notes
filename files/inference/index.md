# 推理总览

<div class="atlas-hero">
  <div class="atlas-kicker">Inference Systems</div>
  <p class="atlas-lead">这一专题围绕在线推理系统展开，重点讨论服务系统、运行时、KV cache、长上下文、MoE、多模型路由、在线评测与容量治理。</p>
  <div class="atlas-chip-row">
    <span class="atlas-chip">服务系统</span>
    <span class="atlas-chip">运行时</span>
    <span class="atlas-chip">KV Cache</span>
    <span class="atlas-chip">容量运维</span>
  </div>
</div>

## 专题定位

<div class="atlas-meta-grid">
  <div>
    <strong>核心问题</strong>
    <p>如何把训练好的模型变成一条可运营的在线系统，在质量、时延、吞吐、成本和稳定性之间做持续平衡。</p>
  </div>
  <div>
    <strong>适合读者</strong>
    <p>适合做在线服务、RAG、Agent、模型路由、多租户平台和推理基础设施的人。</p>
  </div>
  <div>
    <strong>阅读方式</strong>
    <p>建议先看服务系统和运行时，再进入缓存、长上下文、MoE、多模型和在线评测。</p>
  </div>
</div>

## 推荐入口

<div class="atlas-card-grid">
  <a class="atlas-card" href="serving-systems/">
    <strong>服务系统</strong>
    <p>先建立 prefill、decode、排队、批处理和尾延迟的总体系统图。</p>
  </a>
  <a class="atlas-card" href="serving-runtimes-vllm-sglang-and-tensorrt-llm/">
    <strong>vLLM、SGLang 与 TensorRT-LLM</strong>
    <p>理解运行时选型如何影响 cache、量化、结构化生成和多模型集成。</p>
  </a>
  <a class="atlas-card" href="context-compression-kv-eviction-and-memory-hierarchies/">
    <strong>上下文压缩、KV 淘汰与内存分层</strong>
    <p>快速切入长上下文系统最现实的内存与缓存管理问题。</p>
  </a>
</div>

推理系统的目标，不只是“模型能跑”，而是“在成本、延迟、吞吐和稳定性之间做平衡”。

对自回归模型，单请求时延可粗略写成：

\[
T \approx T_{\text{queue}} + T_{\text{prefill}} + T_{\text{decode}} + T_{\text{post}}
\]

若输出长度为 \(n\)，则常可进一步近似为：

\[
T_{\text{decode}} \approx n \cdot T_{\text{step}}
\]

!!! tip "基础知识入口"
    推理系统里的 `KV Cache`、latency、throughput、runtime、kernel 和低精度路径，都依赖基础执行模型。建议先看 [Transformer 与 Attention](../foundations/transformer-attention-and-tokenization.md)、[位置编码、Mask 与上下文](../foundations/positional-encoding-masks-and-context.md) 和 [数值、显存与运行时基础](../foundations/numerics-memory-and-runtime-basics.md)。

## 1. 推理系统为什么和训练系统完全不同

**训练关心的是**：

- loss：优化目标是否下降，训练信号是否合理。
- 收敛：参数更新是否稳定走向可用解，而不是震荡或发散。
- 泛化：模型在未见数据和长尾任务上是否仍然可靠。

**推理关心的是**：

- 首 token 延迟：用户看到第一段输出前等多久。
- 每 token 生成速度：流式输出过程中每个 token 的间隔。
- QPS：单位时间能处理多少请求。
- tail latency：P95/P99 是否稳定，决定最差体验。
- 单位请求成本：每次调用消耗多少 GPU、显存和工程资源。

因此同一个模型，训练做得很好，不代表在线服务就自然高效。

### 1.1 为什么很多团队会误判推理问题

因为训练视角很容易把问题看成“模型是否强”。但推理视角里，很多关键问题其实是：

1. 请求是否被正确分流；
2. 上下文是否被合理装配；
3. KV cache 是否被有效管理；
4. prefill 和 decode 是否被分开优化；
5. 尾延迟是否被控制。

所以推理更多是系统设计问题，而不是单模型问题。

## 2. 一个直观例子：短问题 + 长 PDF

用户问一句很短的话，但带了 20 页 PDF。真正贵的往往不是解码，而是前面的 prefill，因为要先把整段上下文编码进 KV cache。

**这说明推理至少要区分两类负载**：

- prefill：一次性处理输入上下文并写入 KV cache，长输入时成本高。
- decode：逐 token 生成输出，常受 KV cache 读写和调度影响。

它们的瓶颈和优化手段并不一样。

### 2.1 再看一个反例：短轮次 agent

如果系统是屏幕 agent 或工具 agent，请求上下文可能不长，但每次都需要快速多轮调用。此时瓶颈可能不在 prefill，而在：

1. 小 batch 下的 kernel 利用率；
2. 高频工具链切换；
3. 路由和缓存污染；
4. 多轮状态膨胀。

这说明“推理慢”并不是一个统一问题。

## 3. 推理要解决的四个核心问题

### 延迟

用户多久能看到第一段输出。

更细一点说，延迟至少要拆成**排队、网络、prefill、decode 和后处理**几个阶段。不同业务对延迟的敏感点并不一样。聊天产品往往首先在意首 token 时间，agent 系统更在意整条多步链路的总完成时间，而批量离线生成则可能根本不在乎 TTFT，却非常在意整体吞吐和资源利用率。

### 吞吐

同样资源下一秒能服务多少 token 或多少请求。

吞吐不只是“卡打满没有”。它还涉及请求混部后的有效吞吐、短长请求混合时的吞吐退化、连续批处理对不同阶段的影响，以及高峰流量下系统是否还能稳定维持接近设计值的服务能力。很多线上系统平均吞吐看上去不错，但一旦混入长上下文、工具调用或图像输入，请求画像一变，吞吐就会迅速塌掉。

### 成本

每次请求到底花多少钱。

成本既包括直接的 GPU 成本，也包括显存占用、缓存驻留、工程维护复杂度、失败重试和容量冗余。一个表面上更快的方案，若需要更高的显存水位、更复杂的旁路服务或更脆弱的缓存策略，最终未必更便宜。推理优化里，真正需要关注的是**单位可接受质量对应的单位成本**，而不是抽象意义上的“越省越好”。

### 稳定性

P95/P99 会不会突然拉高。

稳定性回答的是系统能否长期经营，而不是能否偶尔跑出漂亮 benchmark。线上常见的问题包括长请求挤占短请求、某一类 prompt 触发显存峰值、prefix cache 失效导致瞬时流量放大，以及多模型路由切换后某条慢路径被放大。只看平均值，很容易把这些风险藏起来。

### 3.1 还有一个经常被漏掉的问题：可观测性

一个推理系统若无法回答“为什么今天比昨天慢”“为什么某类请求集中失败”“为什么某些用户突然都掉到了慢模型上”，那它即使暂时跑得起来，也很难长期维护。

可观测性的价值在于，它把推理系统从“黑箱服务”变成“可诊断系统”。至少要能分阶段看排队时间、prefill 时间、decode 时间、cache 命中率、路由分布、显存水位和异常请求桶。否则团队即使知道系统出了问题，也很难知道该先改调度、改路由、改缓存还是改模型。

## 4. 一个统一理解

可以把在线推理看成一个队列系统：

\[
\text{requests} \rightarrow \text{scheduling} \rightarrow \text{GPU compute} \rightarrow \text{responses}
\]

因此很多问题其实不是“模型慢”，而是：

- 调度不合理：长短请求、在线和后台任务没有被正确排队。
- cache 管理差：KV、prefix 或检索缓存没有被有效复用和回收。
- 请求类型混在一起：长文档、短聊天、agent 和多模态请求互相拖慢。
- 路由策略没分层：简单任务和高价值复杂任务走了同一条昂贵路径。

### 4.1 推理为什么天然是多目标优化

很多优化只能同时改善一部分目标：

1. 更 aggressive batching 提升吞吐，但可能拉高 TTFT；
2. 更强检索提升质量，但拉长关键路径；
3. 量化降低成本，但可能损伤长尾能力；
4. 小模型更快，但复杂任务可能需要升级回大模型。

所以推理决策本质上是多目标权衡，而不是“尽量更快”这么单一。

## 5. 推理优化的主要抓手

### 服务系统设计

如 continuous batching、并行策略、资源池化。

服务系统设计决定请求怎样进入 GPU、怎样被合批、怎样和其他请求共享资源。它不仅影响平均吞吐，还影响尾延迟、租户隔离和故障放大方式。一个好的服务系统，会明确区分热路径与慢路径、长请求与短请求、在线用户流量与后台批量任务，而不是把所有东西都塞进同一个资源池里。

### 缓存

如 KV cache、prefix cache、检索缓存。

缓存是推理系统最像“放大器”的组件。命中时能显著降低重复计算和首 token 延迟，失效时却可能造成显存膨胀、冷热不均和回收抖动。更重要的是，缓存从来不是只靠技术实现就能成功，它还依赖 prompt 结构是否稳定、会话边界是否清楚、上下文拼装是否一致以及淘汰策略是否贴近真实流量。

### 路由

如不同请求走不同模型或链路。

路由负责把“不同难度、不同价值、不同形态”的请求送往不同路径。它既可以是显式规则，例如高价值用户走更强模型，也可以是隐式分类器，例如根据上下文长度、工具需求和历史失败率选择不同链路。一个成熟的推理系统，几乎不可能只靠单模型单路径覆盖所有请求。

### 加速推理方法

如 speculative decoding、量化、kernel 优化。

这类方法解决的是单位时间内如何多做一点有效计算，但它们的价值必须回到系统里判断。比如 speculative decoding 要看 draft 模型命中率和额外调度成本，量化要看长尾质量与 kernel 支持，kernel 优化要看是否真的命中热路径。脱离真实流量分布去谈加速，往往只能得到局部最优。

### 上下文管理

如压缩、裁剪、淘汰和外部记忆分层。

上下文管理本质上是在回答两个问题：**哪些内容必须保留，哪些内容应该外置**。随着 RAG、agent 和记忆系统接入，推理的瓶颈越来越不只是生成，而是上下文如何组织。一个设计合理的系统，不会默认把所有历史都塞进 prompt，而会根据任务目标、召回质量、缓存状态和时延预算来动态决定保留什么。

### 5.1 为什么这些抓手必须一起看

**因为单点优化常常会转移瓶颈**：

1. 量化后显存省了，但 cache 成了新瓶颈；
2. 检索做强了，但上下文拼接太长；
3. speculative 解码快了，但排队和路由变差；
4. continuous batching 提升了吞吐，却把尾延迟拉高。

**推理系统的难点就在这里**：你很少能只动一个旋钮而不影响其他部分。

## 6. 推理文档在本知识库里的位置

这里的推理章节不是只讲某个 kernel 或某个 serving 框架，而是把完整的在线系统拆成若干关键层：

1. 服务系统与调度；
2. `vLLM`、`SGLang`、`TensorRT-LLM` 等运行时如何承接 cache、batching 与 kernel；
3. 解耦式 prefill、KV 服务与容量运维如何把上下文和增量生成分层处理；
4. 缓存、路由与投机执行；
5. RAG、agent 与长上下文；
6. 上下文压缩与内存分层；
7. GPU kernel、batching 与显存系统；
8. 多模型与 MoE 路由；
9. 成本建模、SLO 和线上评测。

这样阅读时会更容易建立系统图，而不是只记一些零散优化技巧。

## 7. 推荐阅读顺序

**建议先读**：

1. [服务系统](serving-systems.md)
2. [vLLM、SGLang 与 TensorRT-LLM](serving-runtimes-vllm-sglang-and-tensorrt-llm.md)
3. [解耦式 Prefill、KV 服务与容量运维](disaggregated-prefill-kv-services-and-capacity-ops.md)
4. [缓存、路由与投机执行](caching-routing-and-speculative.md)
5. [RAG、Agent 与长上下文系统](rag-agents-and-long-context-systems.md)
6. [GPU Kernel、Batching 与内存系统](gpu-kernels-batching-and-memory-systems.md)
7. [可观测性与在线评测](observability-and-online-evaluation.md)

如果你更关心成本与容量规划，再继续看：

1. [MoE 路由与多模型服务](moe-routing-and-multi-model-serving.md)
2. [成本建模与 SLO 设计](cost-modeling-and-slo-design.md)
3. [上下文压缩、KV 淘汰与内存分层](context-compression-kv-eviction-and-memory-hierarchies.md)

## 阶段检查与下一站

推理专题不要按“优化技巧清单”来读，而要按线上请求的生命周期来读。

| 阶段 | 关键问题 | 相关页面 | 下一站 |
| --- | --- | --- | --- |
| 1. 请求进入系统 | 如何排队、合批、分流，TTFT 和 TPOT 谁是主瓶颈 | [服务系统](serving-systems.md)、[成本建模与 SLO 设计](cost-modeling-and-slo-design.md) | [训练总览](../training/index.md) 中的评测与成本 |
| 2. 模型在 GPU 上跑 | runtime、kernel、batching 和 KV cache 如何决定吞吐 | [vLLM、SGLang 与 TensorRT-LLM](serving-runtimes-vllm-sglang-and-tensorrt-llm.md)、[GPU Kernel、Batching 与内存系统](gpu-kernels-batching-and-memory-systems.md) | [算子与编译器](../operators/index.md) |
| 3. 上下文被管理 | 长上下文、RAG、agent 和记忆系统如何控制成本与错误传播 | [RAG、Agent 与长上下文系统](rag-agents-and-long-context-systems.md)、[上下文压缩、KV 淘汰与内存分层](context-compression-kv-eviction-and-memory-hierarchies.md) | [VLM](../vlm/index.md) 或 [世界模型](../world-models/index.md) |
| 4. 系统可运营 | 线上分桶、异常回放、容量规划和多模型路由如何闭环 | [可观测性与在线评测](observability-and-online-evaluation.md)、[MoE 路由与多模型服务](moe-routing-and-multi-model-serving.md) | [量化](../quantization/index.md) |

读完推理模块后，应该能把“慢、贵、不稳、掉点”拆成请求画像、调度、缓存、runtime、kernel、模型质量和评测口径中的具体责任层。

## 快速代码示例

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    tensor_parallel_size=2,
    max_model_len=8192,
)
params = SamplingParams(temperature=0.2, max_tokens=256)
outputs = llm.generate(
    ["请总结下面错误日志并给出修复建议：..."],
    params,
)
print(outputs[0].outputs[0].text)
```

这段代码给出一个最小 `vLLM` 推理入口：初始化模型时设置并行与上下文长度，采样参数独立管理，再统一走 `generate`。在生产中可继续扩展成批处理、流式输出和多路路由。

## 8. 一个总判断

推理系统不是“把训练好的模型部署一下”，而是把调度、缓存、路由、上下文管理、加速策略和业务目标放到同一个运行框架里。真正成熟的推理团队，不只是知道某个模型多快，而是知道什么请求该走什么路径、什么成本是值得的、哪些尾部风险正在积累，以及一旦流量和任务形态改变，系统会先从哪里出问题。 

换句话说，推理的难点不是“做一次优化”，而是**持续地和流量分布一起演化**。业务会变，模型会变，请求长度会变，缓存命中模式会变，硬件和 runtime 也会变。只有把系统拆成可观察、可调度、可回退的几层结构，推理优化才不会沦为一次次救火。
