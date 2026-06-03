# FP8 与混合精度推理来源台账

这份台账服务于 `files/quantization/fp8-and-mixed-precision-serving.md`。正文定位是推理服务判断页，不做低比特训练综述；重点是解释 FP8 在服务里省哪张账、scale 如何成为控制面、哪些路径应保高精度、上线时如何证明收益。

## 资料取舍

| 主题 | 主要来源 | 在正文中的用法 |
| --- | --- | --- |
| FP8 格式 | [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433) | 支撑 E4M3/E5M2、scale 和任务曲线证据。 |
| FP8 scale recipe | [NVIDIA Transformer Engine FP8 delayed scaling](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/fp8_delayed_scaling/fp8_delayed_scaling.html) | 支撑 delayed scaling、amax history 和 recipe 不是 dtype 开关的写法。 |
| FP8 硬件背景 | [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/) | 支撑 FP8 Tensor Core 和硬件路径必须与 kernel 命中一起看的判断。 |
| 大模型 FP8 mixed precision | [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) | 支撑 mixed precision framework、scale、累加和通信路径共设计。 |
| KV cache 量化 | [vLLM Quantized KV Cache](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/) | 支撑 KV cache 量化要单独考虑 scale 校准、per-head 支持和 backend 边界。 |
| 量化生态 | [Hugging Face Quantization concept guide](https://huggingface.co/docs/transformers/quantization/concept_guide) | 支撑把 FP8 放到权重、激活、KV cache 和 runtime support 的整体生态里理解。 |

## 写作边界

正文避免把 FP8 写成“比 BF16 快、比 INT4 稳”的固定结论。FP8 是否值，取决于服务瓶颈、硬件、kernel、shape、scale、fallback 和质量桶。若 decode 卡在 KV cache 或调度上，权重 GEMM 改成 FP8 未必能显著改善端到端 TPOT。

正文也避免把 FP8 与 INT8/INT4 按 bit 数线性排序。INT4 可能更省常驻显存，但需要更强的 group scale、fused dequant、kernel 和校准；FP8 压缩不如 INT4 激进，但在支持 FP8 Tensor Core 的硬件上，可能更容易获得稳定低精度 GEMM 路径。

## 图片使用口径

正文保留三张已有图片，没有生成新图。FP8 Formats 的训练曲线用于说明格式定义必须回到任务证据；DeepSeek-V3 的 FP8 framework 图用于说明生产形态是 mixed precision 系统；DeepSeek-V3 的 accumulation 图用于说明低精度乘法、scale 和累加精度要一起设计。

这些图片不是装饰，也不用于暗示训练结论可以直接外推到所有推理服务。正文只抽取机制：格式、scale、累加、高精度保护和系统共设计。

## 核对口径

正文中的“scale 是控制面”来自 FP8 格式论文、Transformer Engine 文档和 vLLM KV cache 文档的共同边界：FP8 值不是直接存成 8 bit 就结束，而是要有 scale 统计、更新、校准、元数据和 fallback。正文中的“混合精度才是生产形态”来自 DeepSeek-V3 报告的系统图和大模型低精度实践。

正文把服务指标拆成 TTFT、TPOT、throughput、P95/P99、显存、kernel trace、fallback 和质量桶，是本站自己的工程组织方式；它不声称任何单一来源给出了完整上线门禁，而是把论文/文档事实落到服务验证。

## 服务验收口径

正文把 prefill 和 decode 分开，是为了避免把 GEMM benchmark 等同于在线收益。prefill 阶段更容易受大矩阵吞吐影响；decode 阶段可能被 KV cache、batch 调度、小 shape、采样和网络输出限制。FP8 是否有效，要看真实请求分布下的端到端指标，而不是只看单个 kernel 或单层矩阵。

质量桶也按任务拆开。文本 LLM 看数学、代码、长上下文、工具调用和结构化输出；VLM 看 OCR、表格、坐标、细粒度视觉问答和 projector；VLA 或世界模型看动作抖动、risk head、near-miss 和 closed-loop success。这些分桶是本站的上线判断框架，来源文档提供机制事实，具体门禁应由项目自己的离线集和线上监控决定。

## 与其他页面的分工

这份台账不替代 `low-bit-training-numerics-source-ledger.md`。低比特训练页关心反向传播、optimizer state、checkpoint 和长跑稳定性；本页关心推理服务里的热路径、KV cache、fallback、tail latency 和质量回归。两者都会谈 scale，但训练页强调训练状态链，本页强调服务控制面。

本页也不替代 PTQ/GPTQ/AWQ/SmoothQuant 页面。那些页面讨论整数权重量化、激活离群值和校准方法；FP8 页只把它们作为选型对照，提醒读者不要按 bit 数排序，而要按瓶颈、硬件路径和质量风险选路线。

## 未采用的写法

- 没有生成新图。
- 没有把 FP8 写成全模型 dtype 开关。
- 没有用单个 benchmark 证明 FP8 上线成功。
- 没有把 DeepSeek-V3 的训练结论直接说成所有推理服务的结论。
- 没有把 KV cache 量化和权重量化混在一起，因为它们的瓶颈、scale 和质量风险不同。
