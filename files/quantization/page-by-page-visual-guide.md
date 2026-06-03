# 量化图解旧入口

这个页面保留给旧链接和搜索结果使用；它不再作为量化专题的阅读入口。量化章节现在优先讲清每一种方法解决什么误差、依赖什么校准和运行时、为什么未必直接变快。

## 推荐阅读顺序

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

## 外部精读

- [Hugging Face Quantization concepts](https://huggingface.co/docs/transformers/quantization/concept_guide)：先建立低比特格式、权重/激活/KV 和运行时的共同语言。
- [GPTQ](https://arxiv.org/abs/2210.17323)：理解二阶误差补偿式 PTQ。
- [AWQ](https://arxiv.org/abs/2306.00978)：理解为什么重要权重通道要由激活决定。
- [SmoothQuant](https://arxiv.org/abs/2211.10438)：理解 activation outlier 如何迁移到权重侧。
