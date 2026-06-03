# PTQ、GPTQ、AWQ 与 SmoothQuant 来源台账

这份台账服务于 [PTQ、GPTQ、AWQ 与 SmoothQuant](../quantization/ptq-gptq-awq-smoothquant.md)。正文的写作目标不是按 bit 数或论文时间线罗列方法，而是解释量化如何在数值误差、系统瓶颈和部署运维之间分配预算。

## 使用方式

第一类来源是基础量化和工具链文档。Hugging Face Transformers quantization 文档、PaddleNLP 量化教程和 General Compute 工程博客帮助本站解释：同样叫 INT4/INT8，weight-only、W8A8、KV cache、group size、Q/DQ 融合和 runtime kernel 可能完全不同。正文引用这类来源时，用它们建立部署读法，不把博客里的具体性能 claim 当作普适事实。

第二类来源是经典 PTQ 方法。GPTQ、AWQ、SmoothQuant 和 LLM.int8() 分别支撑二阶重构、activation-aware 通道保护、activation outlier migration 和 mixed-precision decomposition。正文每个方法只承担一个解释职责：它保护哪类误差、依赖哪类校准、适合哪类系统瓶颈。

第三类来源是中文讲法。中文官方文档、技术博客和高质量微信公众号适合学习文章组织方式：先用显存和服务瓶颈建立问题，再讲量化公式，最后给校准、kernel 和评测清单。事实、公式、版本语义和性能 claim 仍以论文、官方文档、官方项目页或代码仓库为准。

## 写作边界

不要把 `INT4` 写成自动加速。低 bit 只说明存储表示更小；端到端速度取决于 packing layout、dequant、低比特 GEMM、batch shape、runtime 调度和 fallback kernel。

不要把 PTQ 校准写成形式步骤。校准集决定 scale、敏感通道、Hessian 近似和跳层策略；如果它不覆盖真实线上长尾，量化误差会在代码、数学、长上下文、多模态或动作任务里暴露。

不要把 weight-only 量化写成解决全部显存问题。长上下文服务里 KV cache 可能成为更大瓶颈；多模态服务里视觉 token、connector、workspace 和 prefill 计算也可能主导成本。

不要把 perplexity 不掉写成上线完成。必须同时检查任务分桶、工具调用、长文档引用、OCR/表格、闭环控制、安全边界、P95/P99、kernel trace 和量化配置可追溯性。

## 图片使用

本页没有使用 image2，也没有生成新图片；正文只复用了仓库里已经存在的论文图，并在图注中保留图源和读图方法。

| 正文图片 | 本地路径 | 原始来源 | 本站用途 |
| --- | --- | --- | --- |
| GPTQ procedure | `../assets/images/paper-figures/quantization/gptq-figure-2-procedure.png` | [GPTQ](https://arxiv.org/abs/2210.17323) Figure 2 | 说明 GPTQ 用近似 Hessian 和逐列补偿保护层输出。 |
| AWQ activation-aware | `../assets/images/paper-figures/quantization/awq-figure-1-activation-aware.png` | [AWQ](https://arxiv.org/abs/2306.00978) Figure 1 | 说明权重通道重要性由 activation 使用方式决定。 |
| SmoothQuant intuition | `../assets/images/paper-figures/quantization/smoothquant-figure-2-intuition.png` | [SmoothQuant](https://arxiv.org/abs/2211.10438) Figure 2 | 说明 activation outlier 如何拉大量化范围，以及平滑迁移的动机。 |

## 来源

| 来源 | 类型 | 本站使用方式 | 关联问题 |
| --- | --- | --- | --- |
| [Hugging Face Quantization concept guide](https://huggingface.co/docs/transformers/quantization/concept_guide) | 官方文档 | 核对量化对象、低比特概念和 Transformers 生态入口。 | 概念入口 |
| [Hugging Face: selecting a quantization method](https://huggingface.co/docs/transformers/quantization/selecting) | 官方文档 | 支撑按硬件、内存、延迟和任务目标选择量化路线。 | 选型 |
| [GPTQ](https://arxiv.org/abs/2210.17323) | 论文 / 图源 | 支撑 Hessian-aware weight-only PTQ、逐列量化和误差补偿。 | GPTQ |
| [AWQ](https://arxiv.org/abs/2306.00978) | 论文 / 图源 | 支撑 activation-aware 权重通道保护和低成本校准。 | AWQ |
| [SmoothQuant](https://arxiv.org/abs/2211.10438) | 论文 / 图源 | 支撑 W8A8、activation outlier migration 和等价缩放公式。 | SmoothQuant |
| [LLM.int8()](https://arxiv.org/abs/2208.07339) | 论文 | 支撑大模型 activation outlier 与 mixed-precision decomposition 的早期解释。 | activation outlier |
| [ZeroQuant](https://arxiv.org/abs/2206.01861) | 论文 | 支撑 PTQ、分层量化和系统部署视角的背景。 | PTQ 系统 |
| [PaddleNLP 大模型量化教程](https://paddlenlp.readthedocs.io/en/stable/llm/quantization.html) | 中文官方文档 | 学习 GPTQ、AWQ 等参数和工具链如何落地。 | 中文实践 |
| [General Compute: Quantization for Inference](https://www.generalcompute.com/blog/quantization-for-inference-gptq-awq-smoothquant-fp8) | 工程博客 | 学习用 prefill/decode 瓶颈解释 weight-only、W8A8 和 FP8 的工程分工。 | 讲法参考 |

## 后续写作提醒

量化页面要围绕“瓶颈和误差预算”展开。先问线上瓶颈是权重显存、decode 权重带宽、prefill GEMM、KV cache、通信还是多模态前端；再问量化对象是权重、activation、KV 还是训练状态；最后才选择 GPTQ、AWQ、SmoothQuant、FP8、KV cache quant 或 QAT。

新增量化方法时，不要只写压缩率和 benchmark 分数。必须补上校准数据、scale 粒度、group size、zero point、跳层策略、kernel 后端、硬件型号、Q/DQ 是否融合、评测桶和失败样本。
