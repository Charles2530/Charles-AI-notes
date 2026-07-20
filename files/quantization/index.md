# 量化专题：低比特、质量与运行时取舍

量化需要在模型质量、显存、带宽、kernel 支持、校准数据和业务风险之间做取舍。一个量化方案能不能上线，取决于误差从哪里来、有没有被校准吸收、运行时是否真能吃到低比特，以及评测是否覆盖了你关心的任务。

## 先分清量化路线

| 你想弄清的问题 | 建议先读 |
| --- | --- |
| PTQ、GPTQ、AWQ、SmoothQuant 到底差在哪里 | [PTQ / GPTQ / AWQ / SmoothQuant](ptq-gptq-awq-smoothquant.md) |
| activation outlier 为什么会让 INT8/INT4 变难 | [Activation Outliers 与校准策略](activation-outliers-and-calibration-strategies.md) |
| FP8、混合精度和低比特整数各适合什么 | [FP8 与混合精度推理](fp8-and-mixed-precision-serving.md) |
| QLoRA 为什么能省显存，但不是纯推理量化 | [QLoRA 与量化训练](qlora-and-quantized-training.md) |

## 再看能不能真正落地

| 落地问题 | 建议先读 |
| --- | --- |
| QAT、attention kernel 和 KV cache 如何互相牵制 | [QAT、Kernel 与 KV Cache](qat-kernels-and-kv-cache.md) |
| 多模态、VLM、VLA 量化为什么更容易伤语义和动作 | [多模态与 VLA 量化](quantization-for-multimodal-and-vla-models.md) |
| 量化运行时和框架支持如何决定实际收益 | [量化运行时与框架](quantization-runtimes-and-frameworks.md) |
| 服务硬件、吞吐和成本如何一起评估 | [Serving 栈与硬件取舍](serving-stacks-and-hardware-tradeoffs.md) |

## 做上线判断

| 判断维度 | 建议先读 |
| --- | --- |
| 需要一页比较不同方法 | [量化方法对比表](comparison-table.md) |
| 需要公式和符号最小集 | [量化符号与最小数学](symbols-and-minimal-math.md) |
| 需要上线前检查清单 | [量化评测与部署清单](evaluation-and-deployment-checklist.md) |

量化最常见的误判是只看模型文件变小。真正的验收要同时看质量回归、端到端延迟、吞吐、显存峰值、kernel 命中率和长上下文场景的 KV 行为。
