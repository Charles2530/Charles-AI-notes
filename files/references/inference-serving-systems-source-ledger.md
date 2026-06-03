# 推理服务系统来源台账

这份台账服务于 [推理服务系统](../inference/serving-systems.md)。正文的写法参考高质量技术博客和中文公众号常用的讲解方式：先从“为什么快模型上线后仍然慢”这个用户体感问题进入，再把 queue、prefill、KV cache、decode、batching、routing、speculative decoding 和 observability 拆成一张请求账单。事实、术语和系统边界以论文、官方博客和官方文档为准。

## 使用方式

第一类来源是 KV cache 和 continuous batching。vLLM PagedAttention 博客与论文支撑 block table、逻辑 token 到物理 block 映射、paged KV cache、prefix sharing 和 continuous batching。正文复用 vLLM 系统图说明 serving runtime 的核心不是模型类，而是 scheduler、KV manager、allocator 和 worker 的协作。

第二类来源是调度论文。Orca 支撑 iteration-level scheduling：生成式模型的 batch 不是一次固定矩阵乘，而是在每个 decode step 都变化的流。DistServe 支撑 prefill / decode disaggregation、goodput 和 SLO-oriented serving 的讨论。

第三类来源是生产框架和官方文档。Hugging Face TGI、SGLang、TensorRT-LLM KV Cache System 和 NVIDIA Dynamo 用于核对成熟 serving 栈的真实接口：prefix cache、RadixAttention、in-flight batching、KV block pool、reuse、offload、eviction、routing 和 disaggregated serving。

第四类来源是 speculative decoding 和工程优化。Hugging Face assisted generation、vLLM speculative decoding、SpecInfer、EAGLE 和 NVIDIA inference optimization blog 用于解释 draft cost、acceptance rate、target verification、batching、quantization 和 KV cache 等优化为什么要按 workload 解读。

第五类来源是讲法参考。智源转载的全栈 Transformer 推理优化长文、中国信通院 2026 推理优化报告，以及其他中文技术博客、知乎专栏和微信公众号，只用于学习讲解节奏：从用户体感慢切入，按请求生命周期拆瓶颈，再落到 SLO、成本和观测闭环。稳定性不足、版权不清或没有原始出处的图和结论不进入正文。

## 写作边界

正文不把平均 tokens/s 当作用户体验。TTFT、TPOT、P95/P99、端到端任务时间和每成功任务成本必须分开。

正文不把 vLLM、SGLang、TensorRT-LLM、TGI 或 Dynamo 写成互斥排行榜。它们面向的 workload 和系统接口不同：paged KV、程序化生成、硬件编译、低比特部署、routing 和 disaggregated serving 分别解决不同瓶颈。

正文不把 PagedAttention 说成万能优化。它主要改善 KV 分配、碎片和共享；长 prompt prefill、decode bandwidth、sampling、工具调用和网络仍可能是瓶颈。

正文不把 prefill / decode disaggregation 当作默认最优。它适合规模、SLO 和输入输出分布足够清晰的服务；小规模部署可能被状态传输和调度复杂度抵消收益。

正文不把 speculative decoding 当作通用加速按钮。它只在 draft 足够便宜、接受率较高、target decode 是主瓶颈时更容易带来端到端收益。

正文没有使用 image2 或其他生成式工具作图。唯一图片来自仓库已有资产，并在正文标出原始来源。

## 图片使用

| 图片 | 原始来源 | 本站使用方式 |
| --- | --- | --- |
| `vllm-system-overview-padded.png` | [vLLM PagedAttention blog](https://vllm.ai/blog/2023-06-20-vllm) | 说明 scheduler、KV cache manager、CPU/GPU block allocator 和 worker 的系统关系。 |

## 来源

| 来源 | 类型 | 本站使用方式 | 关联主题 |
| --- | --- | --- | --- |
| [vLLM PagedAttention blog](https://vllm.ai/blog/2023-06-20-vllm) | 官方博客 / 图源 | 使用 PagedAttention 图解释 KV block、block table、continuous batching 和内存碎片。 | KV cache |
| [vLLM paper](https://arxiv.org/abs/2309.06180) | 论文 | 支撑 PagedAttention、serving throughput 和 memory management 的论文证据。 | vLLM |
| [vLLM speculative decoding blog](https://vllm.ai/blog/spec-decode) | 官方博客 | 支撑 speculative decoding 需要看 workload、acceptance rate、QPS 和端到端效果。 | 投机解码 |
| [Orca](https://www.usenix.org/conference/osdi22/presentation/yu) | OSDI 论文 | 支撑 iteration-level scheduling、selective batching 和生成式模型服务调度。 | 调度 |
| [DistServe](https://arxiv.org/abs/2401.09670) | 论文 | 支撑 prefill / decode disaggregation、goodput 和 SLO-oriented serving。 | 分离式服务 |
| [Hugging Face: Assisted Generation](https://huggingface.co/blog/assisted-generation) | 官方技术博客 | 支撑 draft / verify / acceptance 的 speculative decoding 直觉。 | 投机解码 |
| [Hugging Face TGI](https://huggingface.co/docs/text-generation-inference/index) | 官方文档 | 支撑生产 LLM serving 栈的功能入口和部署边界。 | Serving stack |
| [SGLang docs](https://docs.sglang.ai/) | 官方文档 | 支撑程序化生成、prefix reuse、RadixAttention 和复杂 agent workload。 | 生成程序 |
| [TensorRT-LLM KV Cache System](https://nvidia.github.io/TensorRT-LLM/features/kvcache.html) | 官方文档 | 支撑 KV block pool、reuse、offload、eviction、GQA/MQA 和 attention window。 | 硬件部署 |
| [NVIDIA Dynamo docs](https://docs.nvidia.com/dynamo/latest/index.html) | 官方文档 | 支撑 disaggregated serving、routing、KV-aware 调度和多节点推理服务方向。 | 生产 serving |
| [NVIDIA inference optimization blog](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/) | 官方工程博客 | 支撑 batching、quantization、KV cache、speculative decoding 等优化手段的工程视角。 | 工程优化 |
| [SpecInfer](https://arxiv.org/abs/2305.09781) | 论文 | 支撑 speculative decoding 需要同时看 draft cost、acceptance 和 target verification。 | 投机解码 |
| [EAGLE](https://arxiv.org/abs/2401.15077) | 论文 | 支撑 feature-level draft 和端到端 speedup 要按 workload 解释。 | 投机解码 |
| [智源转载：全栈 Transformer 推理优化](https://hub.baai.ac.cn/view/33449) | 中文长文 | 借鉴按模型、算子、显存、调度、工程分层讲解的中文写法。 | 中文讲法 |
| [中国信通院：大模型推理优化关键技术与应用实践研究报告（2026）](https://www.caict.ac.cn/kxyj/qwfb/ztbg/202604/P020260415615308750699.pdf) | 中文产业报告 | 参考产业侧如何组织模型压缩、PD 分离、KV 多级存储和服务调度主题。 | 产业视角 |
