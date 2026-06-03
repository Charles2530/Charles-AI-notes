# 推理运行时来源台账

这份台账服务于 [推理运行时：按请求生命周期选择 vLLM、SGLang 与 TensorRT-LLM](../inference/serving-runtimes-vllm-sglang-and-tensorrt-llm.md)。正文不把 runtime 写成框架排行榜，而是按请求生命周期拆解：入队、prefill、decode、streaming、KV cache 复用、cache 释放、尾延迟和质量回归。台账的作用是记录这套写法背后的主要来源，以及哪些说法不能越界。

第一类来源是 vLLM 与 PagedAttention。vLLM 博客和 PagedAttention 论文支撑正文中关于 KV cache 动态增长、内存碎片、block table、非连续物理 block、parallel sampling cache sharing 与 continuous batching 的说法。正文把 vLLM 写成通用 serving 的强基线，但不把它写成所有 workload 下的绝对最快方案；benchmark 仍必须按 prompt/output 长度、并发和 SLO 条件复验。

第二类来源是 SGLang。SGLang 论文支撑“复杂 LLM 应用是 structured language model programs”的写法：frontend language 负责表达 generation、parallelism 与控制流，runtime 通过 RadixAttention 复用 KV cache，并通过压缩状态机加速结构化输出解码。SGLang 文档用于核对当前生产 serving、OpenAI-compatible API、prefix caching、多硬件支持和部署入口。正文不把 SGLang 写成只适合 agent，也不把 RadixAttention 写成所有 prefix cache 的唯一实现；它只是说明程序结构可以成为 runtime 优化信号。

第三类来源是 TensorRT-LLM 官方文档。TensorRT-LLM 文档支撑正文中关于 engine、online serving、paged attention、in-flight batching、chunked prefill、parallelism、量化、guided decoding、speculative decoding 和 benchmark 工具的描述。KV Cache System 页面支撑 block pool、cross-request reuse、radix search tree、eviction、offload、retention policy、GQA/MQA、attention window、cache salting 和 KV cache dtype 的边界。正文把 TensorRT-LLM 写成硬件路径明确时的高控制面选项，而不是研究试验的默认最轻路径。

第四类来源是更广的 serving 系统背景。Orca 用 iteration-level scheduling 解释为什么自回归 generation 不能只按整请求 batch；Hugging Face Text Generation Inference 用来对照成熟 serving 栈、API 和维护状态。正文引用这些来源是为了提醒读者：vLLM、SGLang 和 TensorRT-LLM 不是孤立发明，它们都在解决大模型在线 serving 的调度、cache 和尾延迟问题。

## 主要来源

| 来源 | 类型 | 本页使用方式 | 对应段落 |
| --- | --- | --- | --- |
| [vLLM PagedAttention blog](https://vllm.ai/blog/vllm) | 官方工程博客 | 支撑 KV cache 大、动态、碎片化，PagedAttention 用 block 管理 KV，continuous batching 与 cache sharing 是 serving 关键。 | vLLM |
| [PagedAttention paper](https://arxiv.org/abs/2309.06180) | 论文 | 核对 PagedAttention、vLLM serving 系统、KV memory management 与 benchmark 条件。 | vLLM、Benchmark |
| [vLLM documentation](https://docs.vllm.ai/en/latest/) | 官方文档 | 核对当前 serving、OpenAI-compatible API、prefix caching、structured outputs、speculative decoding、量化和部署入口。 | vLLM、选择表 |
| [SGLang paper](https://arxiv.org/abs/2312.07104) | 论文 | 支撑 structured language model programs、frontend/runtime 分工、RadixAttention 和结构化输出解码。 | SGLang |
| [SGLang documentation](https://docs.sglang.io/) | 官方文档 | 核对生产 serving、prefix caching、OpenAI-compatible API、多硬件和部署入口。 | SGLang |
| [TensorRT-LLM documentation](https://nvidia.github.io/TensorRT-LLM/) | 官方文档 | 支撑 engine、online serving、IFB、paged attention、chunked prefill、parallelism、quantization、benchmark 和 feature matrix。 | TensorRT-LLM |
| [TensorRT-LLM KV Cache System](https://nvidia.github.io/TensorRT-LLM/features/kvcache.html) | 官方文档 | 支撑 block pool、reuse、radix tree、eviction、offload、retention policy、GQA/MQA、cache salting 和 dtype 控制。 | TensorRT-LLM、Benchmark |
| [Hugging Face Text Generation Inference](https://huggingface.co/docs/text-generation-inference/index) | 官方文档 | 对照另一个成熟 serving 栈，并记录官方对 TGI 维护状态和下游 runtime 的说明。 | 外部精读 |
| [Orca](https://www.usenix.org/conference/osdi22/presentation/yu) | 系统论文 | 作为 iteration-level scheduling 和自回归 serving 调度背景。 | 生命周期、Benchmark |

## 写作边界

正文中的 KV cache 公式是容量近似，用来解释并发数、上下文长度、层数、KV head、head 维度和 dtype bytes 的乘法关系。真实 serving 还要加入 page/block 元数据、allocator overhead、prefix sharing、cache salting、eviction/offload、draft 分支、temporary buffer、tensor parallel 切分和 scheduler policy。因此公式不能替代实际 profile。

正文不把 prefill 与 decode 写成绝对分离的工程阶段。现代 runtime 可能有 chunked prefill、prefill/decode disaggregation、overlap scheduler 或 mixed batch，但教学上先拆开两者有助于理解：prefill 常接近 dense compute，decode 常被 KV cache 与 memory bandwidth 影响。

正文不使用单个 tokens/s 判定 runtime 好坏。推理 benchmark 必须至少写清模型、硬件、prompt/output 长度分布、并发、batch policy、streaming、prefix cache、量化、多 GPU、speculative decoding 和 SLO。没有这些条件，tokens/s、TTFT、inter-token latency、P95/P99 都容易被误读。

正文不把 vLLM、SGLang、TensorRT-LLM 写成互斥替代品。它们可以同时出现在一个生产系统里：vLLM 做通用 serving，SGLang 服务复杂程序化生成，TensorRT-LLM 服务固定硬件上的高性能路径。选择取决于 workload、团队能力和观测指标。

正文不新增图片，也没有使用 image2 或任何图像生成。页面只复用了仓库已有的 `files/assets/images/paper-figures/inference/vllm-system-overview-padded.png`，用于说明 vLLM 中请求调度、KV block 管理和模型执行之间的关系。图片承担机制说明，不承担性能排行榜或全 runtime 通用架构图的作用。

## 与站内其他台账的分工

[推理服务系统来源台账](inference-serving-systems-source-ledger.md)覆盖更完整的 serving 系统页，包括 API、scheduler、cache、monitoring 和服务化边界。本台账专门服务 vLLM、SGLang、TensorRT-LLM 三个运行时的比较页，重点是如何按 workload 选 runtime。

[KV、缓存与投机解码来源台账](caching-routing-speculative-source-ledger.md)专门解释 prefix/KV cache、routing 和 speculative decoding 如何分配 GPU 时间、KV 显存和强模型调用次数。本台账只在 runtime 选择层面引用这些机制，不展开 speculative decoding 的算法细节。

[FlashAttention 与长上下文来源台账](flashattention-long-context-source-ledger.md)负责 attention kernel、decode KV 读取和长上下文并行机制。本台账只引用其中的 prefill/decode 成本口径，不替代 FlashAttention、Flash-Decoding 或 long-context parallelism 的专题说明。

[FP8 与混合精度推理来源台账](fp8-mixed-precision-serving-source-ledger.md)负责 FP8、KV cache 量化、scale 和质量回归。本台账只把量化作为 runtime 选择条件之一，具体 dtype、校准和 fallback 仍回到低精度专题。

## 后续维护

推理 runtime 文档变化很快。维护这页时优先复核官方文档，而不是只看早期论文或社区 benchmark。尤其需要复查的点包括：vLLM 文档的 latest 与 stable 差异、SGLang 文档域名与功能入口、TensorRT-LLM 的 KV cache 页面路径、TGI 的维护状态、以及各 runtime 对新模型架构、FP8/FP4、speculative decoding、guided decoding 和 disaggregated serving 的支持边界。
