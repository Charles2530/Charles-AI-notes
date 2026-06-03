# 推理服务系统来源台账

这份台账服务于 [推理服务系统](../inference/serving-systems.md)。正文不把 serving 写成框架列表，而是围绕一次请求的生命周期讲：queue、prefill、KV cache、decode、batching、disaggregation、routing、speculative decoding 和 observability 分别解决什么问题。

## 使用方式

第一类来源是 KV cache 和 continuous batching。vLLM PagedAttention 博客和论文图支撑正文里的 block table、逻辑 token 到物理 block 映射、paged KV cache 和 continuous batching。正文使用 vLLM PagedAttention 图说明 serving runtime 的核心是请求长度变化时的内存管理。

第二类来源是调度论文。Orca 支撑 iteration-level scheduling 的概念：生成式模型的 batch 不是一次固定矩阵计算，而是在每个 decode step 都变化的流。DistServe 支撑 prefill / decode 分离和 goodput-oriented serving 的讨论。

第三类来源是生产框架文档。Hugging Face TGI、SGLang 和 TensorRT-LLM 文档用于说明成熟 serving 栈的真实接口：prefix cache、RadixAttention、in-flight batching、KV cache system、quantization、multi-GPU 和 engine 构建。

第四类来源是工程优化博客。NVIDIA inference optimization blog 和智源转载的 OneFlow / Transformer 推理优化中文长文用于校准讲法：推理优化要按模型结构、算子、显存、KV、batching、调度和 SLO 分层，而不是只写“换框架更快”。

## 写作边界

正文不把平均 tokens/s 当作用户体验。TTFT、TPOT、P95/P99、端到端任务时间和每成功任务成本必须分开。

正文不把 vLLM、SGLang、TensorRT-LLM 写成互斥排行榜。它们面向的 workload 和系统接口不同：paged KV、程序化生成、硬件编译和低比特部署分别解决不同瓶颈。

正文不把 PagedAttention 说成万能优化。它主要改善 KV 分配、碎片和共享；长 prompt prefill、decode bandwidth、sampling、工具调用和网络仍可能是瓶颈。

正文不把 prefill / decode disaggregation 当作默认最优。它适合规模、SLO 和输入输出分布足够清晰的服务；小规模部署可能被状态传输和调度复杂度抵消。

## 来源

| 来源 | 类型 | 本站使用方式 | 关联主题 |
| --- | --- | --- | --- |
| [vLLM PagedAttention blog](https://vllm.ai/blog/2023-06-20-vllm) | 官方博客 / 图源 | 使用 PagedAttention 图解释 KV block、block table、continuous batching 和内存碎片。 | KV cache |
| [vLLM paper](https://arxiv.org/abs/2309.06180) | 论文 | 支撑 PagedAttention、serving throughput 和 memory management 的论文证据。 | vLLM |
| [Orca](https://www.usenix.org/conference/osdi22/presentation/yu) | OSDI 论文 | 支撑 iteration-level scheduling、selective batching 和生成式模型服务调度。 | 调度 |
| [DistServe](https://arxiv.org/abs/2401.09670) | 论文 | 支撑 prefill / decode disaggregation、goodput 和 SLO-oriented serving。 | 分离式服务 |
| [Hugging Face TGI](https://huggingface.co/docs/text-generation-inference/index) | 官方文档 | 支撑生产 LLM serving 栈的功能入口和部署边界。 | Serving stack |
| [SGLang docs](https://docs.sglang.ai/) | 官方文档 | 支撑程序化生成、prefix reuse、RadixAttention 和复杂 agent workload。 | 生成程序 |
| [TensorRT-LLM docs](https://nvidia.github.io/TensorRT-LLM/) | 官方文档 | 支撑 in-flight batching、KV cache system、quantization、engine build 和 NVIDIA 部署路径。 | 硬件部署 |
| [NVIDIA inference optimization blog](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/) | 官方工程博客 | 支撑 batching、quantization、KV cache、speculative decoding 等优化手段的工程视角。 | 工程优化 |
| [SpecInfer](https://arxiv.org/abs/2305.09781) | 论文 / 图源 | 支撑 speculative decoding 需要同时看 draft cost、acceptance 和 target verification。 | 投机解码 |
| [EAGLE](https://arxiv.org/abs/2401.15077) | 论文 | 支撑 feature-level draft 和端到端 speedup 要按 workload 解释。 | 投机解码 |
| [智源转载：全栈 Transformer 推理优化](https://hub.baai.ac.cn/view/33449) | 中文长文 | 借鉴按模型、算子、显存、调度、工程分层讲解的中文写法。 | 中文讲法 |
