# KV、缓存与投机解码来源台账

这份台账服务于 `files/inference/caching-routing-and-speculative.md`。正文定位是推理请求生命周期页，不做 vLLM、EAGLE 或路由系统的完整综述；目标是解释 prefix/KV cache、routing 和 speculative decoding 如何共同分配 GPU 时间、KV 显存和强模型调用次数。

## 资料取舍

| 主题 | 主要来源 | 在正文中的用法 |
| --- | --- | --- |
| PagedAttention / KV cache | [vLLM / PagedAttention blog](https://vllm.ai/blog/vllm)、[PagedAttention paper](https://arxiv.org/abs/2309.06180) | 支撑 KV cache block/page、block table、continuous batching 和碎片管理。 |
| Speculative decoding | [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) | 支撑 draft 提议、target 验证，并保持 target distribution 的边界。 |
| Token tree verification | [SpecInfer](https://arxiv.org/abs/2305.09781) | 支撑 token tree、parallel verification 和 accepted prefix 的读法。 |
| 工程评测 | [vLLM speculative decoding blog](https://vllm.ai/blog/spec-decode) | 支撑按 draft 方法、acceptance、QPS 和 workload 分桶报告收益。 |
| EAGLE 系列 | [EAGLE](https://arxiv.org/abs/2401.15077)、[EAGLE-2](https://arxiv.org/abs/2406.16858)、[EAGLE-3](https://arxiv.org/abs/2503.01840) | 支撑 feature-level draft、dynamic draft tree 和 train-inference mismatch。 |
| 模型路由 | [RouteLLM](https://arxiv.org/abs/2406.18665) | 支撑把请求路由写成成本、延迟、质量和风险之间的选择。 |
| 程序化生成与 prefix 复用 | [SGLang](https://arxiv.org/abs/2312.07104) | 作为 prefix/radix cache、复杂生成程序和 runtime 分工的背景，不在正文中展开。 |

## 写作边界

正文不把 PagedAttention 写成 attention kernel 优化。它解决的是 KV cache 分配、碎片和共享；FlashAttention 解决的是 attention score/probability 中间矩阵的 I/O；二者可以组合，但不是同一层。

正文不把 speculative decoding 写成质量提升或小模型替代大模型。经典投机解码的目标是在保持 target model 分布的前提下降低 target decode 次数；质量筛选、reranking 或 verifier sampling 是另一类问题。

正文不把路由写成固定规则表。RouteLLM 一类工作说明，请求级模型路由本质上是成本和质量之间的选择；真实服务还要加入 SLO、风险、租户、缓存命中、工具链和回退成本。

## 图片使用口径

正文保留三张已有图片，没有生成新图。vLLM logical/physical block table 图用于说明 KV cache 不必物理连续；SpecInfer workflow 图用于说明 draft tree 与 target verification；EAGLE-2 dynamic draft tree 图用于说明投机预算要按 confidence 分配。

这些图只承担机制解释。正文不把 vLLM 图写成所有 serving 框架的默认实现，不把 SpecInfer 图写成所有投机解码都必须使用 tree verification，也不把 EAGLE-2 图写成所有请求都适合动态树。

## 核对口径

正文中的 KV 显存公式是成本近似，用来说明层数、并发、上下文长度、KV head 数、head 维度和 dtype bytes 的乘法关系。真实系统还要加上 page metadata、fragmentation、prefix sharing、eviction、draft 分支、temporary buffer 和 runtime 调度。

正文中的 speculative speedup 公式也是判断近似，用来提醒读者 acceptance 是核心变量。真实 speedup 还取决于 draft 模型实现、verify kernel、batching、tree mask、KV 分支、sampling 策略、QPS 和 P95/P99。

正文把缓存、路由和投机解码放在一页，是因为它们都在改变请求生命周期中的资源分配；但它们的验证指标必须分开记录。cache 看 hit 和 saved tokens，routing 看 route distribution 和升级/回退，speculative 看 accepted length 和 reject/rollback。

## 与其他页面的分工

这页和 `inference-serving-systems-source-ledger.md` 的分工是：服务系统页讲完整 serving runtime 生命周期；本页专门讲 cache、routing 和 speculative 三类资源分配机制。

这页和 `flashattention-long-context-source-ledger.md` 的分工是：FlashAttention 页讲 attention kernel、KV cache 和 sequence parallel 三张长上下文账；本页讲在线请求中 KV cache、prefix cache、routing 和 draft verification 如何组合。

这页和 `generation-decoding-source-ledger.md` 的分工是：生成解码页讲采样、beam、MTP 和 speculative 的生成算法边界；本页讲 speculative 在 serving 系统里的 acceptance、P99、KV 生命周期和 workload 分桶。

## 未采用的写法

- 没有把 prefix cache、PagedAttention、routing 和 speculative decoding 写成全局开关。
- 没有只报告平均 tokens/s 或平均 latency。
- 没有把 draft 速度当成 speculative decoding 的唯一指标。
- 没有生成新图，正文只复用仓库已有论文图。
