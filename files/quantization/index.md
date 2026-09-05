# 量化专题：低比特、质量与运行时取舍

量化需要在模型质量、显存、带宽、kernel 支持、校准数据和业务风险之间做取舍。一个量化方案能不能上线，取决于误差从哪里来、有没有被校准吸收、运行时是否真能吃到低比特，以及评测是否覆盖了你关心的任务。

量化可以看成一条误差预算。权重、激活、KV cache 和累加器分别承担不同的误差；校准数据决定哪些通道会被保护；硬件和 kernel 决定低比特是否真的转化成吞吐或成本收益。只看模型文件大小，会漏掉这条链上最容易失败的环节。

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

## 一套可复用的量化决策顺序

先确定要解决的瓶颈，再选择量化对象。显存不足时优先盘点权重、KV 和 activation 的占比；带宽受限时关注权重读取和 cache 访问；算力不足时确认硬件是否有对应的低比特矩阵指令。随后用代表性校准集覆盖长短上下文、不同语言、工具调用和多模态输入，最后在真实 runtime 中比较质量与服务指标。

| 目标 | 先量化什么 | 主要风险 |
| --- | --- | --- |
| 缩小常驻权重 | 权重，常见为 INT8/INT4 或 FP8 | 离群通道、格式转换和 kernel 不支持 |
| 降低长上下文显存 | KV cache，必要时配合权重量化 | 累积误差、长程检索和生成稳定性 |
| 降低训练显存 | LoRA/QLoRA、低比特 optimizer 或 activation | 梯度噪声、累加精度和训练收敛 |
| 保护多模态能力 | 对视觉 encoder、connector、action head 分层处理 | OCR、细粒度定位和动作精度退化 |

## 读结果时看四个对照

同一模型要同时看 FP16/BF16 基线、量化后的离线任务、量化后的真实 runtime，以及长上下文和高并发场景。离线准确率没有下降，可能只是测试集太短；模型文件变小，可能只是把成本转移到了反量化、workspace 或 KV。最终要问的是单位有效结果的成本有没有下降，质量退化是否发生在可接受的任务桶内。
