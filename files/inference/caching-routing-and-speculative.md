# 缓存、路由与投机执行

大模型在线推理进入生产环境后，真正拖垮系统的往往不是“模型不够快”，而是 KV cache 爆显存、不同类型请求互相阻塞、大模型被低价值请求占满、decode 太慢导致长答案体验差。缓存、路由与投机执行，是把推理资源用到刀刃上的三类机制。

!!! note "初学者先抓住"
    缓存减少重复计算，路由决定请求走哪条模型链路，投机执行用便宜路径先猜、昂贵路径再验证。三者都是在问：有限 GPU 资源应该花在哪里最划算。

!!! example "有趣例子：机场安检"
    常旅客通道像 prefix cache，提前准备好的证件信息能复用；不同旅客分到不同通道像路由；先用快速扫描发现疑似问题再人工复核，像 speculative decoding 的“先猜再验”。

## KV Cache 是第二套显存大户

Transformer decode 会为历史 token 保存每层 key 和 value。粗略估算：

\[
\text{KV bytes}
\approx
2 \times B \times L \times H \times d_h \times T \times b
\]

其中 \(B\) 是并发请求数，\(L\) 是层数，\(H\) 是头数，\(d_h\) 是每头维度，\(T\) 是上下文长度，\(b\) 是每元素字节数。

这解释了为什么权重量化后模型已经能装下，但长上下文并发请求仍会把显存撑爆。长上下文系统里，KV cache 常常比参数更快成为瓶颈。

KV 管理要关注：

1. 请求生命周期；
2. page/block 分配；
3. 长短请求混跑造成的碎片；
4. prefix 复用；
5. eviction 和抢占策略；
6. 与调度器的配合。

## 缓存不只有 KV

完整服务中常见缓存包括：

| 缓存 | 作用 | 适合场景 |
| --- | --- | --- |
| Prefix cache | 复用系统 prompt、工具 schema、固定模板 | 企业问答、agent、客服 |
| KV cache | 保存已处理 token 的 K/V | 所有自回归 decode |
| 检索缓存 | 复用 RAG 召回结果 | 重复知识查询 |
| 工具结果缓存 | 复用 API 或数据库查询 | 工具调用 agent |
| 会话状态缓存 | 保存多轮目标、约束、工具状态 | 长会话和流程型任务 |

缓存命中率可以写成：

\[
\text{hit rate}=\frac{\text{cache hits}}{\text{total lookups}}
\]

但高命中率不一定等于高收益。要看命中的是 prefill 大头、decode KV、外部工具延迟，还是低成本的小片段。

## 路由是资源分配

不是所有请求都配得上最大模型。给定请求特征 \(\phi(x)\) 和模型池 \(\mathcal{M}\)，路由可以抽象为：

\[
m^\star =
\arg\min_{m\in\mathcal{M}}
\left(
\lambda_1\text{cost}(m)
+\lambda_2\text{latency}(m)
-\lambda_3\text{quality}(m,x)
\right)
\]

实际路由常按以下维度：

1. 请求长度：短问题走低延迟链路，长文档走大上下文链路；
2. 任务类型：闲聊、代码、文档、表格、工具调用分流；
3. 置信度：小模型先答，低置信或触发规则再升级；
4. 用户等级或 SLA：不同客户走不同质量与成本策略；
5. 风险等级：高风险任务走更强模型、工具校验或人工审核。

路由失败有三类典型表现：复杂问题被送去小模型导致错误；规则越来越多导致路由器难维护；没有在线反馈闭环，离线策略逐渐脱离真实流量。

## 投机执行

`Speculative decoding` 的基本流程是：

1. 草稿路径先生成多个候选 token；
2. 目标模型一次性并行验证；
3. 接受通过验证的前缀；
4. 错误位置由目标模型修正并继续。

平均每轮前进 token 数与 acceptance 直接相关。一个够用的速度模型是：

\[
\text{Speedup}
\approx
\frac{\mathbb{E}[A]\cdot C_{\text{decode}}}
{C_{\text{draft}}+C_{\text{verify}}+C_{\text{overhead}}}
\]

其中 \(\mathbb{E}[A]\) 是平均接受 token 数，\(C_{\text{draft}}\) 是草稿开销，\(C_{\text{verify}}\) 是验证开销，\(C_{\text{overhead}}\) 包括调度、KV 管理和 fallback。

投机执行有用的条件是：draft 足够便宜、接受率足够高、verify 能高效并行、系统开销没有把收益吃掉。

## 三者如何耦合

缓存、路由和投机执行不是三套独立机制。

| 机制 | 会影响什么 |
| --- | --- |
| Prefix cache | 降低 prefill 成本，改变长短请求调度优先级 |
| KV cache | 决定并发容量、decode 稳定性和 eviction 策略 |
| 路由 | 决定请求进入哪个模型池和缓存池 |
| 投机执行 | 改变 decode step 形态和 KV 生命周期 |
| Dynamic batching | 同时影响缓存命中、路由延迟和 speculative acceptance |

例如，长输出请求最适合 speculative，但它也占用更久 KV；复杂文档请求可能 prefix cache 收益高，但 prefill 峰值大；小模型路由节省成本，但如果错误导致重试，端到端成本反而上升。

## 观测指标

建议至少按请求桶记录：

1. KV 占用、碎片率、eviction 次数；
2. prefix cache hit rate 和节省的 prefill token；
3. 路由命中模型、升级率、降级率、错误率；
4. speculative acceptance、fallback 率、p50/p95/p99 latency；
5. TTFT、TPOT、tokens/s、GPU 利用率；
6. 不同长度、任务、SLA、温度配置下的收益差异。

只看全局平均延迟会掩盖问题。缓存和投机通常只在特定桶里有明显收益，路由也可能在少数高价值任务上失败。

## 实践清单

上线前建议回答：

1. 当前瓶颈是 prefill、decode、KV 显存、排队还是外部工具；
2. 哪些前缀、检索结果和工具结果值得缓存；
3. KV eviction 是否会破坏质量或会话一致性；
4. 路由策略是否有在线反馈和安全兜底；
5. speculative 是否按长度、任务和温度分桶验证；
6. 当 acceptance 下降、p99 变差或 KV 压力升高时，是否能自动降级；
7. baseline 路径是否始终可回退。

缓存、路由和投机执行的共同目标，是在不牺牲质量和可靠性的前提下，把有限 GPU 显存和 decode 时间分配给最值得的请求。
