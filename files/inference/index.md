# 推理系统专题：把模型变成稳定服务

推理不是“把训练好的模型跑起来”这么简单。线上系统真正要解决的是延迟、吞吐、成本、显存、排队、缓存、路由、观测和质量回退之间的冲突。读这一章时，可以先把每个问题放进一条请求的生命周期：prefill 读上下文，decode 逐 token 生成，KV cache 占住显存，scheduler 决定谁先跑，SLO 决定系统能不能交付。

高质量博客常把 vLLM、SGLang、TensorRT-LLM 的系统直觉讲得很清楚；本站会借鉴这种“从一次请求出发”的讲法，再回到论文、官方文档和运行时实现。

## 先看一条请求怎么跑

| 你想弄清的问题 | 建议先读 |
| --- | --- |
| vLLM、SGLang、TensorRT-LLM 分别解决什么 | [推理运行时](serving-runtimes-vllm-sglang-and-tensorrt-llm.md) |
| 一个服务从 API 到 GPU scheduler 要经历哪些环节 | [推理服务系统](serving-systems.md) |
| batching、attention kernel、显存带宽如何限制吞吐 | [GPU Kernel、Batching 与内存系统](gpu-kernels-batching-and-memory-systems.md) |
| prefill 和 decode 为什么越来越适合拆开 | [Prefill/KV 服务与容量运维](disaggregated-prefill-kv-services-and-capacity-ops.md) |

## 再看成本怎么被压住

| 目标 | 建议先读 |
| --- | --- |
| 用 prefix cache、routing、speculative decoding 减少重复计算 | [缓存、路由与投机解码](caching-routing-and-speculative.md) |
| 控制长上下文的 KV 显存和延迟 | [上下文压缩、KV 驱逐与记忆层级](context-compression-kv-eviction-and-memory-hierarchies.md) |
| 同时服务 MoE、多模型和多租户请求 | [MoE 路由与多模型服务](moe-routing-and-multi-model-serving.md) |
| 给业务定价、定 SLO、算容量 | [成本建模与 SLO 设计](cost-modeling-and-slo-design.md) |

## 最后补齐线上治理

| 线上问题 | 建议先读 |
| --- | --- |
| 怎么知道服务正在变慢、变贵或变差 | [观测与在线评测](observability-and-online-evaluation.md) |
| RAG、Agent、长上下文系统怎样避免“能跑但不可控” | [RAG、Agent 与长上下文系统](rag-agents-and-long-context-systems.md) |
| 上线前如何做容量规划和回退策略 | [优化清单与容量规划](optimization-checklist-and-capacity-planning.md) |

推理系统的阅读重点不是记住某个框架，而是判断瓶颈在计算、显存、调度、网络还是质量控制。能定位瓶颈，工具选择就会自然很多。
