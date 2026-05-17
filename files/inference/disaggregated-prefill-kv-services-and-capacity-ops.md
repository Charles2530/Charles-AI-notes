# 解耦 Prefill 与 KV 服务

随着上下文越来越长、请求越来越异质、模型越来越多，把所有工作都塞进一套统一推理实例，不再总是合理。长文档问答、RAG、Agent 和多模型路由场景中，prefill 与 decode 的资源特征明显不同，KV cache 也越来越像独立资源层。解耦式 prefill、KV 服务化和更细粒度容量运维，正是在这个背景下出现。

!!! note "读法定位"
    这页先回答“解耦 Prefill 与 KV 服务”在「推理」里的位置：它解决什么局部问题，依赖哪些前置，最后会影响哪类工程或研究判断。
    **前置**：先知道 transformer 解码、GPU kernel、显存和吞吐/延迟的区别。 必要时先回 [推理入口](index.md)、[基础知识](../foundations/index.md) 或 [术语表](../glossary/index.md)。
    **主线关系**：把请求生命周期、KV Cache、batching、runtime、SLO 和成本连接起来，看模型上线后每一毫秒花在哪里。

!!! note "初学者先抓住"
    Prefill 像一次性处理长输入，decode 像逐 token 生成答案。两者瓶颈不同，长上下文系统里把它们拆开，常常是为了避免长请求拖垮短请求。

!!! example "有趣例子：图书馆复印和现场问答"
    复印一本厚书需要大机器和长时间，回答一个短问题需要快速响应。如果两者排同一条队，短问题会被厚书复印堵住。解耦式 prefill/decode 就是在拆队列。

## 为什么拆分 Prefill 和 Decode

Prefill 和 decode 的资源画像不同：

1. prefill 更偏大计算块和高并行；
2. decode 更偏小步 memory-bound；
3. prefill 常被长上下文拉爆；
4. decode 更容易被高并发和 KV 管理拖慢。

当长文档请求和短聊天请求共用一套实例时，长 prefill 会占住 GPU，短对话 decode 被牵连，TTFT 和 TPOT 一起恶化。解耦的核心目标不是“更快一点”，而是避免长短请求互相拖垮，并让 prefill、decode 和 KV 分别按自身瓶颈扩容。

## 基本架构

一个常见抽象是：

| 层级 | 职责 |
| --- | --- |
| Prefill tier | 高吞吐编码长上下文，生成初始 KV |
| Decode tier | 低延迟增量生成，持续消费 KV |
| KV layer | 承载、转移、复用和回收 KV 状态 |
| Router | 根据请求长度、SLO、租户和缓存命中决定路径 |
| Capacity controller | 管理队列、配额、扩缩容和降级 |

这类架构适合长文档、RAG、Agent、多租户企业助手和 prefix 复用明显的工作负载。不适合低并发、短输入短输出、单模型且无明显缓存复用的简单场景。

## KV 为什么像独立服务层

传统理解里，KV 是模型实例里的内部状态。但一旦系统开始做 prefix cache、prefill/decode 解耦、多阶段生成复用、多模型级联复用，KV 就从实现细节变成系统资源。

系统必须显式回答：

1. KV 放在哪；
2. 谁拥有它；
3. 如何命名和索引；
4. 何时转移、何时失效；
5. 如何统计命中率和成本；
6. 如何做多租户隔离和权限控制。

KV 管理开始接近存储系统问题：热冷分层、分配回收、页面管理、生命周期、一致性和版本。

## 网络与拓扑约束

Prefill 产生的状态要交给 decode 层，系统就不只是 GPU 算力问题，还会变成网络问题：

1. 状态转移带宽是否足够；
2. 延迟是否抵消解耦收益；
3. 是否跨节点或跨机柜传输；
4. 传输失败如何重试或回退；
5. 中间态如何加密和隔离。

调度器需要同时感知 GPU 拓扑、网络拓扑、KV 热度、队列拥塞和租户优先级。否则 prefill 池和 decode 池的分离，可能只是把瓶颈从计算转移到网络。

## 容量规划

解耦架构下，容量要分层估算：

| 资源 | 关键指标 |
| --- | --- |
| Prefill | input token/s、prefix cache hit、长上下文占比 |
| Decode | output token/s、TPOT、batch 动态性 |
| KV | 显存占用、页碎片、命中率、转移带宽 |
| Router | 各路径流量、升级/降级率、队列长度 |
| 网络 | KV transfer latency、跨域流量、失败重试 |

如果 prefix cache 命中率高，prefill 容量需求可能比直觉小很多；如果 prompt 版本频繁变化导致 cache 不命中，预期收益会大幅缩水。

## 风险与失效模式

解耦式架构常见风险包括：

1. KV 转移延迟吃掉收益；
2. 状态一致性和版本管理复杂；
3. debug 维度变多，问题跨 prefill、KV、decode 三层；
4. 网络拓扑不合理导致隐藏瓶颈；
5. 多租户隔离不清晰，引入安全风险；
6. 短请求被复杂架构拖慢；
7. cache 命中假设与真实流量不符。

因此它应当由 workload 驱动，而不是作为默认复杂化方案。

## 验收清单

上线前建议回答：

1. prefill 是否确实是主要瓶颈；
2. 长上下文或 prefix 复用是否足够多；
3. KV 转移成本是否小于解耦收益；
4. decode tier 的 p95/p99 是否改善；
5. cache miss、传输失败和状态过期时如何回退；
6. 多租户权限、审计和删除策略是否覆盖 KV 层；
7. 扩缩容是否能分别作用于 prefill、decode 和 KV。

运维上还要准备回放和压测集。回放集用于比较同一批请求在解耦前后的 TTFT、TPOT、KV 转移耗时和 cache 命中；压测集用于验证长文档洪峰、短请求洪峰、cache miss 洪峰和单租户大流量是否会把某一层打穿。没有这两类验证，解耦架构很容易在平均指标上好看，但在真实峰值下暴露新的尾延迟。

另外，解耦后的容量报表要分层展示，而不是只给一个总 GPU 利用率。prefill tier 忙、decode tier 空，和 decode tier 忙、prefill tier 空，对扩容决策完全不同。KV 层也需要单独显示热 key、过期 key、碎片率和跨节点转移量。

落地时还要避免把“KV 服务化”做成过早抽象。更务实的路径通常是先在单 runtime 内做好 prefix/KV 生命周期，再扩展到同节点共享，最后才考虑跨节点 KV 服务。每一步都应有明确收益指标，否则复杂度会先于收益到来。

如果没有明确收益，保持单体 runtime 并强化调度和缓存，通常是更稳的选择。

解耦式 prefill 的价值不在架构本身，而在于它是否让异质请求的容量、延迟和显存治理变得更可控。

## 真实排查案例：解耦后 TTFT 好了，TPOT 变差

**输入症状**：长文档问答接入解耦式 prefill 后，TTFT 从 3.4s 降到 2.1s，但生成阶段 TPOT 从 45ms/token 增到 68ms/token，用户看到首 token 更快，后续流式输出变卡。

**关键指标**：prefill tier 利用率下降，decode tier queue depth 上升；KV 转移耗时 P95 达到 180ms；decode tier 上 KV page miss 增多；网络出入流量与长上下文请求峰值高度相关。

**Nsight / trace 观察**：trace 中 prefill 结束后不是立刻 decode，而是出现 KV transfer gap；decode kernel 本身没有退化，但 paged attention 前的 KV metadata 同步和 page lookup 时间增加。

**判断**：解耦收益被 KV 转移与 page locality 侵蚀。系统只优化了 prefill 计算，没有保证 KV 落在 decode 最合适的位置。

**修复**：把长上下文请求按 session 粘到固定 decode pool；限制跨节点 KV 转移，优先同节点共享；KV metadata 做批量同步；按请求桶记录 `prefill_saved_ms - kv_transfer_ms - decode_extra_ms` 的净收益。

**反例**：如果 prefill 占比极高、prefix 复用稳定、KV 转移在同节点内完成，解耦通常会明显改善 TTFT。反过来，短请求和低复用请求不该默认进入复杂解耦路径。

!!! tip "下一站"
    - 回到本专题入口：[推理](index.md)，确认这页在整条路线中的位置。
    - 按导航顺序继续：[RAG、Agent 与长上下文系统](rag-agents-and-long-context-systems.md)。
    - 概念或符号卡住时，先查 [术语表](../glossary/index.md)，再回到当前页。
