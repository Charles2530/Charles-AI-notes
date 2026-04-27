# 推理优化清单与容量规划

推理系统上线后，团队很快会遇到两个问题：现在还能撑多久，下一步该先优化哪里。这两个问题表面像“算资源”，本质是在做系统建模。大模型推理容量不是固定常数，而是请求分布、输入输出长度、模型结构、batch 策略、缓存命中、工具调用和 SLO 共同作用的结果。

这页把推理优化从零散技巧整理成容量规划和瓶颈归因框架。

## 请求画像先于资源估算

设请求到达率为 \(\lambda\)，平均服务时间为 \(S\)，负载强度可粗略写成：

\[
\rho=\lambda S
\]

当 \(\rho\) 接近系统有效处理能力时，排队时间和尾延迟会快速恶化。LLM/VLM 系统中 \(S\) 不是常数，通常更接近：

\[
S\approx
T_{\text{prefill}}(L_{\text{in}})
+T_{\text{decode}}(L_{\text{out}})
+T_{\text{tool}}
+T_{\text{network}}
\]

因此容量规划必须先统计真实请求画像：

1. 输入长度分布；
2. 输出长度分布；
3. 长上下文和多模态占比；
4. 工具调用轮次和外部依赖耗时；
5. 模型路由分布；
6. prefix / KV cache 命中率；
7. 峰值流量和多租户叠加。

不要只看平均 token。少量超长 PDF、代码仓库或多图请求，可能占掉大部分算力和显存。

## Prefill、Decode 与 Queue

推理总延迟至少应拆成：

\[
T_{\text{total}}
=
T_{\text{queue}}
+T_{\text{prefill}}
+T_{\text{decode}}
+T_{\text{overhead}}
\]

Prefill 一次性处理全部输入 token，长上下文时成本高；decode 每次只生成少量 token，但要重复很多轮，并持续访问 KV cache。Queue 则反映容量、调度和优先级是否健康。

很多“延迟高”不能笼统归因于模型太大，而要问：

1. 是队列太长；
2. 是 prefill 太重；
3. 是 decode 太慢；
4. 是 KV cache 压力太高；
5. 是工具或网络拖慢；
6. 是调度和 batching 没组织好。

不同瓶颈对应完全不同的优化动作。

## 分层优化清单

| 层级 | 关注点 | 常见动作 |
| --- | --- | --- |
| 请求层 | 输入/输出长度、RAG 拼接、多模态大小 | prompt 收缩、检索裁剪、输出限长、任务分流 |
| 调度层 | queue、continuous batching、priority、routing | 动态 batching、优先级、SLA 隔离、路由 |
| 模型层 | 参数量、量化、draft、early exit | 量化、蒸馏、speculative、模型分层 |
| 缓存层 | prefix、KV、工具结果 | prefix cache、KV paging、eviction、结果缓存 |
| Kernel 层 | GEMM、attention、norm、decode kernel | FlashAttention、paged attention、fused kernels |
| 系统层 | 多实例、网络、存储、外部工具 | 横向扩容、隔离、降级、异步化 |

优化顺序应由瓶颈决定，而不是由技术新不新决定。很多团队一上来做 speculative 或手写 kernel，但真正瓶颈可能只是 prompt 太长、RAG 拼接过量、路由不合理或 cache 命中低。

## 容量模型

容量规划至少要估计三类资源：

1. **计算容量**：prefill FLOPs、decode step、Tensor Core 利用率；
2. **显存容量**：权重、KV cache、batch 中间状态、碎片；
3. **服务容量**：queue、batching、工具、网络、实例副本。

一个实用方法是按请求桶建模，而不是建一个全局平均模型：

| 请求桶 | 主要成本 | 典型优化 |
| --- | --- | --- |
| 短问答 | queue 和 launch 开销 | batching、轻量模型、prefix cache |
| 长文档 | prefill 和 KV | 检索裁剪、上下文压缩、prefill 分离 |
| 长输出 | decode | speculative、draft model、输出策略 |
| 工具 agent | 外部调用和多轮 | 工具缓存、状态压缩、异步执行 |
| 多模态 | 编码器和跨模态 token | 图像裁剪、视觉缓存、模型路由 |

如果 10% 的长请求占据 60% 的 GPU 时间，优化策略就不能平均铺开。

## SLO 驱动优化

容量规划必须和 SLO 绑定。常见指标包括：

1. TTFT：首 token 延迟；
2. TPOT：每输出 token 延迟；
3. p50/p95/p99 请求延迟；
4. tokens/s 和 requests/s；
5. 错误率、超时率、降级率；
6. 单请求成本和单 token 成本；
7. 任务成功率和质量指标。

不同业务的 SLO 不同。企业问答可以接受更高延迟但要求证据可靠；屏幕 agent 更看重每步交互延迟；批处理文档分析更看重吞吐和成本。没有 SLO，优化会变成无限追求更快，却不清楚哪些改动值得做。

## 降级与扩容策略

当资源紧张时，系统不应只有“一起变慢”。应提前设计降级：

1. 低优先级请求切到小模型；
2. 长上下文请求限制最大输入；
3. 关闭多样本采样、二次重排或高成本工具；
4. 批处理任务延后执行；
5. 高风险任务保留强模型和人工审核；
6. p99 恶化时关闭 speculative 或降低 batch。

扩容也要按瓶颈扩。若瓶颈是 KV 显存，单纯增加计算实例未必有效；若瓶颈是外部工具，增加 GPU 也不会改善端到端延迟。

## 验收清单

做推理优化前后，至少回答：

1. 瓶颈属于 queue、prefill、decode、KV、tool、network 还是 kernel；
2. 优化收益在哪些请求桶成立；
3. p95/p99 是否改善，是否牺牲 TTFT；
4. 质量、错误率和降级率是否变化；
5. cache 命中、KV 占用和 batch 形态是否改变；
6. 峰值流量下收益是否还成立；
7. 回退路径是否稳定；
8. 单请求成本是否真的下降。

推理优化的核心不是收集技巧，而是建立请求画像、瓶颈分解、容量模型和 SLO 之间的闭环。
