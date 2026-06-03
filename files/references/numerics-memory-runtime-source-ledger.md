# 数值、显存与运行时来源台账

这页记录数值格式、混合精度、显存账、KV cache、roofline、kernel 和 serving runtime 相关来源。它服务的写作原则是：不要把 FP8、INT4、KV cache、FLOPs、kernel 和 runtime 写成概念清单，而要说明每项技术到底省哪张账、代价转移到哪里、如何用 trace 和质量回归证明收益。

## 使用方式

写数值和运行时文章时，先把来源分成五类。

第一类是数值格式与混合精度。NVIDIA mixed precision、Transformer Engine、FP8 Formats、Microscaling 和低比特训练报告帮助本站解释 FP16、BF16、FP8、INT8/INT4 的表示范围、scale、amax、累加精度和敏感路径保护。正文要强调 FP8 不是 dtype 开关，而是 mixed precision + scale + kernel + fallback。

第二类是显存账。ZeRO、QLoRA、activation checkpointing、KV cache 文档和训练系统资料帮助本站把 parameters、gradients、optimizer states、activations、KV cache、workspace 分开。写任何“省显存”技术时，都要明确它省的是常驻状态、峰值状态、通信 buffer 还是请求生命周期。

第三类是带宽和 roofline。Nsight Compute roofline、FlashAttention、CUTLASS/CuTe 和 kernel 优化资料帮助本站解释 FLOPs 不等于速度。正文应让读者理解 arithmetic intensity、HBM/SRAM 读写、fusion、layout 和 shape specialization，而不是只看 MACs。

第四类是 serving runtime。vLLM PagedAttention、TensorRT-LLM KV cache、SGLang、ONNX Runtime 和 TensorRT 文档帮助本站解释：高并发推理是请求调度、continuous batching、paged KV、prefix reuse、cache eviction、fallback 和 tail latency 的组合问题。

第五类是工程讲法和中文入口。General Compute、OneFlow、智源社区和产业报告适合学习如何把硬件、kernel、runtime 和质量回归串成文章；强 claim、版本行为和性能数据仍回官方文档、论文或仓库核对。

## 写作边界

本页不展开每个量化方法的完整算法，也不替代推理系统、算子或训练稳定性专题。它只建立共同成本模型：dtype 决定表示和 bytes，resident tensors 决定容量，bandwidth/roofline 决定瓶颈，kernel/runtime 决定收益能否兑现，quality buckets 和 P95/P99 决定是否真的可上线。

如果一个技术只减少模型文件大小，但没有减少热路径读写、没有命中低精度 kernel、没有改善真实请求分布的 TTFT/TPOT/P99，本站不把它写成“加速已完成”。同理，系统吞吐变好但数学、代码、长上下文、OCR、工具调用或闭环任务退化，也不能算训练/部署成功。

## 来源

| 来源 | 类型 | 本站学什么 | 落到页面 |
| --- | --- | --- | --- |
| [NVIDIA Mixed Precision Training Guide](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html) | 官方文档 | FP16/BF16、loss scaling、Tensor Core 和混合精度训练路径。 | 混合精度 |
| [Transformer Engine FP8 Delayed Scaling](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/fp8_delayed_scaling/fp8_delayed_scaling.html) | 官方文档 | delayed scaling、amax history 和 FP8 recipe。 | FP8 scale |
| [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433) | 论文 | E4M3/E5M2、scale 设计和训练曲线证据。 | FP8 |
| [Microscaling Data Formats](https://arxiv.org/abs/2310.10537) | 论文 | per-block scaling 与 MX 格式如何服务更低 bit 训练/推理。 | MXFP |
| [Nsight Compute Roofline](https://docs.nvidia.com/nsight-compute/2023.3/NsightCompute/index.html) | 官方文档 | 用 roofline 判断 compute-bound 和 memory-bound。 | roofline |
| [FlashAttention](https://arxiv.org/abs/2205.14135) | 论文 | IO-aware exact attention 如何减少 HBM/SRAM 读写。 | attention kernel |
| [vLLM PagedAttention blog](https://vllm.ai/blog/2023-06-20-vllm) | 官方工程博客 | KV cache block/page 管理和 continuous batching 为什么影响 serving。 | KV cache |
| [PagedAttention paper](https://arxiv.org/abs/2309.06180) | 论文 | near-zero KV memory waste 和跨请求共享的系统证据。 | serving memory |
| [TensorRT-LLM KV Cache System](https://nvidia.github.io/TensorRT-LLM/features/kvcache.html) | 官方文档 | KV reuse、offload、eviction 和 cache hierarchy。 | 生产 runtime |
| [General Compute: Quantization for Inference](https://www.generalcompute.com/blog/quantization-for-inference-gptq-awq-smoothquant-fp8) | 工程博客 | GPTQ、AWQ、SmoothQuant、FP8 按瓶颈分工，而不是按 bit 数排序。 | 量化选型 |
| [Hugging Face Quantization concept guide](https://huggingface.co/docs/transformers/quantization/concept_guide) | 官方文档 | 权重、激活、KV、校准和 runtime support 的概念入口。 | 量化基础 |
| [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) | 技术报告 | MLA、MoE、FP8 和系统优化如何共同塑造成本账。 | 技术报告读法 |
| [OneFlow 官方博客](https://blog.csdn.net/OneFlow_Official) | 中文工程博客 | 学习用 kernel、显存、带宽和分布式视角写工程文章。 | 中文讲法 |
