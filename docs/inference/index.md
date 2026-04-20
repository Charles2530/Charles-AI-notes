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
  <a class="atlas-card" href="serving-systems.md">
    <strong>服务系统</strong>
    <p>先建立 prefill、decode、排队、批处理和尾延迟的总体系统图。</p>
  </a>
  <a class="atlas-card" href="serving-runtimes-vllm-sglang-and-tensorrt-llm.md">
    <strong>vLLM、SGLang 与 TensorRT-LLM</strong>
    <p>理解运行时选型如何影响 cache、量化、结构化生成和多模型集成。</p>
  </a>
  <a class="atlas-card" href="context-compression-kv-eviction-and-memory-hierarchies.md">
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

## 1. 推理系统为什么和训练系统完全不同

**训练关心的是**：

- loss
- 收敛
- 泛化

**推理关心的是**：

- 首 token 延迟
- 每 token 生成速度
- QPS
- tail latency
- 单位请求成本

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

- prefill
- decode

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

### 吞吐

同样资源下一秒能服务多少 token 或多少请求。

### 成本

每次请求到底花多少钱。

### 稳定性

P95/P99 会不会突然拉高。

### 3.1 还有一个经常被漏掉的问题：可观测性

一个推理系统若无法回答“为什么今天比昨天慢”“为什么某类请求集中失败”“为什么某些用户突然都掉到了慢模型上”，那它即使暂时跑得起来，也很难长期维护。

## 4. 一个统一理解

可以把在线推理看成一个队列系统：

\[
\text{requests} \rightarrow \text{scheduling} \rightarrow \text{GPU compute} \rightarrow \text{responses}
\]

因此很多问题其实不是“模型慢”，而是：

- 调度不合理
- cache 管理差
- 请求类型混在一起
- 路由策略没分层

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

### 缓存

如 KV cache、prefix cache、检索缓存。

### 路由

如不同请求走不同模型或链路。

### 加速推理方法

如 speculative decoding、量化、kernel 优化。

### 上下文管理

如压缩、裁剪、淘汰和外部记忆分层。

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

## 8. 一个总判断

推理系统不是“把训练好的模型部署一下”，而是把调度、缓存、路由、上下文管理、加速策略和业务目标放到同一个运行框架里。真正成熟的推理团队，不只是知道某个模型多快，而是知道什么请求该走什么路径、什么成本是值得的、哪些尾部风险正在积累，以及一旦流量和任务形态改变，系统会先从哪里出问题。 
