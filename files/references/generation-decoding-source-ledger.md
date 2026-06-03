# 生成与解码来源台账

## 使用方式

这份台账服务于 [生成与解码](../foundations/generation-and-decoding-basics.md)。正文不把 generation parameters 写成参数表，而是把解码解释成状态机：logits、mask、penalty、temperature、truncation、sampling/search、verification、KV cache 和 runtime。

第一类来源是解码论文和教程。Holtzman 的 Neural Text Degeneration、Hugging Face generation docs/blog 和 OpenAI structured outputs 帮助本站解释 greedy、beam、top-k、top-p、temperature、constraints 和 schema/grammar 的机制边界。

第二类来源是 test-time search。Self-Consistency、verifier sampling、Program-of-Thoughts 和代码执行验证帮助本站区分“多候选提高任务质量”和“投机解码降低延迟”。

第三类来源是推理系统。Speculative Decoding、SpecInfer、vLLM speculative decoding、PagedAttention 和 TensorRT-LLM KV cache 文档帮助本站把生成策略放回 prefill、decode、KV cache、batch、acceptance rate 和 P99 里看。

## 写作边界

正文不把 temperature 当成事实性开关。它只改变 logits 分布的形状，不能补模型知识、检索证据或业务验证。

正文不把结构化输出等同于正确输出。Grammar/schema 可以保证格式合法，事实、引用和工具参数仍需要外部校验。

正文不把 speculative decoding 写成质量提升。标准投机解码的目标是保持 target model 分布并减少 target forward 次数；质量筛选属于 verifier sampling 或 reranking。

## 来源

| 来源 | 类型 | 本站使用方式 | 关联主题 |
| --- | --- | --- | --- |
| [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) | 论文 | 使用 Figure 5 说明 next-token distribution 形状随上下文变化，支持 top-p 的自适应候选集合直觉。 | nucleus sampling |
| [Closing the Curious Case of Neural Text Degeneration](https://arxiv.org/abs/2310.01693) | 论文 | 补充 truncation sampling 为什么有效的后续理论讨论。 | truncation sampling |
| [Hugging Face Generation Strategies](https://huggingface.co/docs/transformers/generation_strategies) | 官方文档 | 核对 greedy、beam、sampling、constraints、cache 等生成配置入口。 | 解码策略 |
| [Hugging Face: How to generate text](https://huggingface.co/blog/how-to-generate) | 技术博客 | 用可运行例子解释 greedy、beam、top-k、top-p 和 repetition penalty。 | 教程 |
| [OpenAI Structured Outputs](https://openai.com/index/introducing-structured-outputs-in-the-api/) | 官方博客 | 解释 JSON Schema 转 CFG、动态 constrained decoding 与普通 prompt JSON 的差别。 | 结构化解码 |
| [Self-Consistency](https://arxiv.org/abs/2203.11171) | 论文 | 支撑多条推理路径采样后聚合/筛选的 test-time search 视角。 | verifier sampling |
| [Program of Thoughts](https://arxiv.org/abs/2211.12588) | 论文 | 说明把计算交给程序执行可以作为外部验证器。 | 工具验证 |
| [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) | 论文 | 解释 draft/verify 如何在保持 target distribution 的前提下降低延迟。 | speculative decoding |
| [SpecInfer](https://arxiv.org/abs/2305.09781) | 论文 | 参考 token tree verification 与 serving 场景下的投机推理。 | speculative serving |
| [vLLM speculative decoding blog](https://vllm.ai/blog/spec-decode) | 官方工程博客 | 参考 acceptance rate、QPS、draft overhead 和 workload 分桶的报告方式。 | serving |
| [PagedAttention paper](https://arxiv.org/abs/2309.06180) | 论文 | 使用 KV block / block table 图说明 decode 阶段动态 KV 显存管理。 | KV cache |
| [vLLM PagedAttention blog](https://vllm.ai/blog/2023-06-20-vllm) | 官方工程博客 | 参考 continuous batching、KV memory waste 和 block sharing 的工程讲法。 | serving runtime |
| [TensorRT-LLM KV Cache System](https://nvidia.github.io/TensorRT-LLM/advanced/kv-cache-reuse.html) | 官方文档 | 补充 KV cache reuse、blocks、eviction 和 attention window 等部署机制。 | KV cache |
