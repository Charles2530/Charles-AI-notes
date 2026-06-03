# 量化图解阅读顺序：看清误差从哪里来

量化图常会同时画权重分布、激活离群值、分组尺度、kernel 路径和部署收益。读图前先确认它在解释哪件事：数值误差、校准策略、训练适配、运行时支持，还是端到端服务成本。这个页面保留“按图找文章”的入口，但不再把图当成目录装饰。

本站没有为这个入口生成新图片，也不会用 image2 补图。涉及论文图时，优先使用原论文、官方文档或已本地化的可核验图片。

## 按量化问题阅读

| 想解决的问题 | 直接阅读 |
| --- | --- |
| 想一次分清 PTQ / GPTQ / AWQ / SmoothQuant | [PTQ / GPTQ / AWQ / SmoothQuant](ptq-gptq-awq-smoothquant.md) |
| 想理解 activation outlier 和校准 | [Activation Outliers 与校准策略](activation-outliers-and-calibration-strategies.md) |
| 想理解 FP8 和混合精度推理 | [FP8 与混合精度推理](fp8-and-mixed-precision-serving.md) |
| 想理解 QLoRA 和量化训练 | [QLoRA 与量化训练](qlora-and-quantized-training.md) |
| 想理解 QAT、kernel 和 KV cache | [QAT、Kernel 与 KV Cache](qat-kernels-and-kv-cache.md) |
| 想比较不同量化路线 | [量化方法对比表](comparison-table.md) |
| 想看部署时用哪些框架 | [量化运行时与框架](quantization-runtimes-and-frameworks.md) |
| 想做上线验收 | [评测与部署检查](evaluation-and-deployment-checklist.md) |

## 图解材料怎么用

| 看图时先问 | 为什么 |
| --- | --- |
| 图里量化的是权重、激活、KV cache 还是 optimizer state | 不同对象的误差和收益完全不同。 |
| scale 是 per-tensor、per-channel、per-group 还是 block-wise | granularity 决定精度、metadata 和 kernel 复杂度。 |
| benchmark 是模型质量、kernel 吞吐还是端到端延迟 | 文件变小不等于服务变快。 |

## 三种常见读图误区

第一种误区是只看 bit 数。INT4、NF4、FP8、W4A16、W8A8 这些标签只告诉你表示格式的一部分，不告诉你分组尺度、校准数据、异常通道、zero point、metadata、kernel 路径和反量化开销。看到低比特图时，先问它量化了哪些张量，哪些张量仍保留高精度。

第二种误区是把离线 perplexity 当成上线质量。量化误差可能在普通文本上不明显，却在数学、代码、长上下文、多模态定位、工具调用或动作输出里放大。评测图如果只给平均分，就要继续找失败样本、分任务指标和长尾 case。

第三种误区是把 kernel benchmark 当成服务收益。一个低比特 GEMM 在 microbenchmark 里更快，不代表端到端服务更快；真实链路还包含 KV cache、prefill/decode 比例、batching、调度、通信、反量化和内存碎片。读部署图时，要区分算子级速度、模型级吞吐和业务级延迟。

## 从图回到上线判断

读量化图时，可以按四步走。

| 步骤 | 该确认什么 |
| --- | --- |
| 找对象 | 图在量化权重、激活、KV cache、optimizer state 还是整条推理链路。 |
| 找误差 | 误差来自 rounding、clipping、outlier、group scale 还是校准分布偏移。 |
| 找补偿 | 方法用二阶近似、activation-aware scaling、QAT、LoRA 还是混合精度来补偿。 |
| 找收益 | 收益体现在显存、带宽、吞吐、延迟、成本还是可部署模型规模。 |

如果一张图只展示模型文件变小，它还没有回答“能不能上线”。上线判断至少要同时看质量回归、延迟曲线、吞吐上限、显存峰值、长上下文 KV 行为和运行时支持。

## 和量化主线的对应关系

量化图大致对应四条主线。误差图解释为什么低比特会伤模型；校准图解释用哪些样本估计 scale 和 outlier；kernel 图解释低比特是否真能减少内存搬运和计算；部署图解释收益能不能穿过运行时、调度和评测。把图放回这四条线，就能判断它是在讲算法、系统，还是只是在展示一个局部现象。

## 读完后的检查

读完一张量化图，最好能复述三句话：它量化了什么对象；它用什么机制降低或接受误差；它的收益是否已经穿过运行时和业务评测。说不清这三句，就先不要把图里的 speedup 或压缩率当成上线结论。

## 外部精读起点

- [Hugging Face Quantization concepts](https://huggingface.co/docs/transformers/quantization/concept_guide)：先建立低比特格式、权重/激活/KV 和运行时的共同语言。
- [GPTQ](https://arxiv.org/abs/2210.17323)：理解二阶误差补偿式 PTQ。
- [AWQ](https://arxiv.org/abs/2306.00978)：理解为什么重要权重通道要由激活决定。
- [SmoothQuant](https://arxiv.org/abs/2211.10438)：理解 activation outlier 如何迁移到权重侧。
