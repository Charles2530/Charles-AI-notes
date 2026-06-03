# FlashAttention 与长上下文来源台账

这份台账服务于 [FlashAttention 与长上下文](../operators/long-context-and-flashattention-evolution.md)。正文不把 FlashAttention 写成“又一个 attention 名词”，而是围绕一条主线组织：标准 attention 的公式很短，但长上下文的真实瓶颈来自 score/probability matrix 的 HBM 往返、KV cache 的生命周期、decode 阶段的长历史读取，以及跨设备 sequence parallel。

## 使用方式

第一类来源是 FlashAttention 系列论文和作者资料。FlashAttention 原论文支撑正文里的 IO-aware tiling、HBM/SRAM 区分、online softmax、exact attention 和 backward 重算。FlashAttention-2 支撑“减少 HBM 往返后，GPU 内部工作分配和非 matmul FLOPs 会成为下一层瓶颈”的写法。FlashAttention-3 支撑“attention kernel 会随 Hopper、WGMMA、TMA、FP8 和异步流水继续演化”的边界。

第二类来源是 serving 侧资料。PyTorch Flash-Decoding 和 vLLM PagedAttention 用来把 prefill 与 decode 拆开：prefill 是大块 dense attention，decode 是短 query 反复扫长 KV cache。正文里关于 KV cache 公式、page/block 管理、continuous batching、prefix reuse 和 tail latency 的说法，优先回到 vLLM、PyTorch 和推理系统资料核对。

第三类来源是长上下文并行论文。RingAttention、DeepSpeed Ulysses 和相关 context parallel 方法用来说明：当上下文继续变长，问题不再只是单 kernel 的片上复用，而是把 sequence dimension 拆到多卡，并在通信、显存和计算重叠之间做工程权衡。正文引用 RingAttention 图，是为了让读者看到“序列块沿设备环传递 K/V”的机制，而不是把它当作所有长上下文训练的默认方案。

第四类来源是 GPU 底层和中文工程长文。CUDA Best Practices、NVIDIA 文档和智源转载的 Transformer 推理优化长文，用来校准讲法：不要只说 FLOPs，要把 HBM、SRAM/shared memory、register、kernel launch、shape、dtype、KV cache、scheduler 和服务指标放在一条成本链上。中文长文的价值主要在叙事结构，不作为强性能 claim 的唯一证据。

## 写作边界

正文不把 FlashAttention 说成近似 attention。只要讨论 FlashAttention 原始设定，就应明确它是 exact dense attention 的 IO-aware 实现；如果后续讨论 sparse、sliding window、block sparse 或任意 mask kernel，必须单独说明它们是否改变可见结构。

正文不把“省显存”写成单一结论。FlashAttention 省的是 attention 中间矩阵和 HBM I/O；PagedAttention 省的是动态 KV 分配浪费和碎片；MQA/GQA 减少的是 KV head 数；KV quantization 减少的是每个元素字节数；activation checkpointing 省的是训练中间激活常驻。它们可以组合，但不能互相替代。

正文不把 benchmark 迁移到所有 workload。FlashAttention 的加速与序列长度、head dimension、mask、dropout、dtype、GPU 架构、框架版本和 kernel path 相关。Serving 的收益还要看 TTFT、TPOT、P95/P99、batch 动态性、prefix 命中率、KV watermark 和请求长度分布。

正文不把长上下文长度当成最终质量。最大 context length 只是容量声明；真正的长上下文能力要看 retrieval、multi-hop、代码仓库理解、长视频/多图输入、工具调用、对抗干扰和成本延迟。上下文越长，如果有效利用率越低，就不应写成“能力自然提升”。

## 来源

| 来源 | 类型 | 本站使用方式 | 关联主题 |
| --- | --- | --- | --- |
| [FlashAttention](https://arxiv.org/abs/2205.14135) | 论文 / 图源 | 支撑 IO-aware exact attention、HBM/SRAM、online softmax、backward 重算和 Figure 1 图。 | attention kernel |
| [FlashAttention-2](https://arxiv.org/abs/2307.08691) | 论文 | 支撑 better parallelism、work partitioning、减少非 matmul FLOPs、跨 block 并行等讲法。 | GPU 并行 |
| [FlashAttention-2 blog](https://crfm.stanford.edu/2023/07/17/flash2.html) | 作者/机构博客 | 学习如何把第一代 IO-aware 思想推进到 GPU 内部并行分工。 | 写作讲法 |
| [FlashAttention-3](https://arxiv.org/abs/2407.08608) | 论文 | 支撑 Hopper、异步流水、低精度和新硬件路径对 attention kernel 的影响。 | Hopper kernel |
| [Flash-Decoding for long-context inference](https://pytorch.org/blog/flash-decoding/) | 官方工程博客 | 支撑 decode 阶段短 query、长 KV、跨 KV block 并行和长上下文推理的区别。 | decode |
| [vLLM / PagedAttention blog](https://vllm.ai/blog/vllm) | 官方工程博客 | 支撑 KV cache page/block、碎片、sharing、continuous batching 和 serving throughput。 | KV cache |
| [vLLM paper](https://arxiv.org/abs/2309.06180) | 论文 | 用作 PagedAttention 和高吞吐 serving 的论文证据。 | serving runtime |
| [Ring Attention](https://arxiv.org/abs/2310.01889) | 论文 / 图源 | 支撑 sequence blocks、ring pass K/V、blockwise attention 和近无限上下文训练叙事。 | sequence parallel |
| [DeepSpeed Ulysses](https://arxiv.org/abs/2309.14509) | 论文 | 支撑把 sequence dimension 放进分布式并行、用 all-to-all 交换 QKV 的长序列训练路径。 | 长序列训练 |
| [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) | 官方文档 | 支撑 HBM/shared memory/register、memory coalescing、profiling 和 GPU 优化背景。 | GPU 基础 |
| [NVIDIA Nsight Compute Roofline](https://docs.nvidia.com/nsight-compute/2023.3/NsightCompute/index.html) | 官方文档 | 支撑不要只看 FLOPs，要用 bandwidth、arithmetic intensity 和 roofline 判断瓶颈。 | 性能分析 |
| [智源转载：全栈 Transformer 推理优化](https://hub.baai.ac.cn/view/33449) | 中文长文 | 借鉴按硬件、算子、显存、KV、调度和服务指标分层讲解的中文写法。 | 中文讲法 |

## 图片使用口径

正文保留三张已有图片，没有生成新图。FlashAttention IO-aware tiling 图用于说明 HBM 与片上 SRAM 的数据流差异；FlashAttention speedup 图用于提醒读者 benchmark 必须绑定 GPU、shape、mask、dtype 和实现路径；RingAttention 图用于说明长上下文会从单 kernel 扩展到跨设备 sequence parallel。

这些图只承担机制解释，不作为装饰。正文不把 FlashAttention 的速度图外推成所有 attention 的固定加速倍数，也不把 RingAttention 图写成所有长上下文系统的默认方案。

## 核对口径

正文把长上下文拆成三张账：attention kernel 的中间矩阵 I/O，decode 阶段的 KV cache 生命周期，以及更长上下文下的跨设备 sequence parallel。这是本站自己的教学组织方式；每张账背后的事实边界分别回到 FlashAttention 系列、vLLM/PagedAttention/Flash-Decoding、RingAttention/Ulysses 等来源核对。

正文中的 KV cache 容量公式是成本账近似，目的是让读者看到 \(B\)、\(T\)、层数、KV head 数、head 维度和 dtype bytes 的乘法关系。真实服务还要考虑 page size、碎片、prefix sharing、eviction、offload、calibration 和 runtime 调度，因此公式不能替代服务 profile。
