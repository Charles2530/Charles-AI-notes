# PTQ、GPTQ、AWQ 与 SmoothQuant

量化不是“把 FP16 改成 INT4”。量化是在问：哪些张量可以低比特表示，误差怎么控制，校准数据是否代表真实输入，kernel 和 runtime 是否支持，质量回归能不能被测出来。

## 核心问题

LLM 量化最常见的目标是省显存、提吞吐、降低部署成本。但不同方法处理的是不同误差来源：GPTQ 更像权重重构，AWQ 更强调激活感知的权重保护，SmoothQuant 则把激活离群值的一部分压力迁移到权重上。

先把量化对象分清：

| 对象 | 为什么量化 | 主要风险 |
| --- | --- | --- |
| 权重 | 减少模型常驻显存和 GEMM 带宽 | 低比特误差伤害推理质量。 |
| 激活 | 减少运行时中间张量和计算成本 | 离群值会让 scale 很难选。 |
| KV cache | 长上下文服务显存大头 | 影响注意力读取历史的精度。 |
| optimizer state | 训练阶段省显存 | 稳定性和收敛风险更高。 |

## PTQ：不用重新训练，但要校准

Post-Training Quantization 在模型训练完成后做量化。它的吸引力是成本低，不需要完整再训练；它的风险是校准集、scale、granularity 和 kernel 支持一旦不合适，质量会掉得很隐蔽。

最基础的 uniform quantization 可以理解成：用 scale 把浮点数映射到整数格点，再在推理时近似恢复。

\[
q = \operatorname{round}(x / s), \quad \hat{x}=s q
\]

\(s\) 是 scale。scale 太大，小数差异被抹掉；scale 太小，离群值会溢出或被截断。group-wise、channel-wise、token-wise 等 granularity，本质上都在决定“多少数共享一个 scale”。

## GPTQ：按重构误差处理权重

GPTQ 的问题意识是：如果直接逐个权重量化，会把误差传到层输出上。更好的做法是利用校准数据估计权重误差对输出的影响，再逐步量化和补偿。

![GPTQ procedure 原论文图](../assets/images/paper-figures/quantization/gptq-figure-2-procedure.png){ width="760" }

<small>图源：[GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)，Figure 2。原图表达 GPTQ 按列量化权重，并用近似 Hessian 信息更新尚未量化的列。本站用它说明：GPTQ 不是简单 round，而是尽量让量化后的层输出接近原模型。</small>

读 GPTQ 时抓住两个点：它主要是 weight-only PTQ；它依赖校准样本估计哪些权重误差更影响输出。工程上常见 GPTQ 模型适合降低权重显存，但是否提速取决于对应 INT4 kernel 和 serving runtime。

## AWQ：保护重要权重通道

AWQ 观察到，不是所有权重同样重要。某些通道对应的激活更大、更关键，直接低比特量化会明显伤害输出。AWQ 的思路是根据激活分布找到重要通道，通过 scaling 保护它们。

![AWQ activation-aware 原论文图](../assets/images/paper-figures/quantization/awq-figure-1-activation-aware.png){ width="760" }

<small>图源：[AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978)，Figure 1。原图表达少量显著激活通道对应的权重更需要保护。本站用它解释：AWQ 的关键词是 activation-aware，即用激活统计决定哪些权重量化误差更危险。</small>

AWQ 常被用于 4-bit weight-only 部署。它的优势是校准和部署成本相对友好；限制是它仍然主要围绕权重量化，激活、KV cache 和具体 kernel 路径要另看。

## SmoothQuant：把激活离群值迁移给权重

激活量化难点在 outliers。某些通道激活值特别大，导致 scale 被迫变大，普通值的分辨率就变差。SmoothQuant 的核心是利用线性层的缩放等价性：把激活里的大幅度缩小，同时把权重对应通道放大，使整体矩阵乘结果尽量不变。

![SmoothQuant intuition 原论文图](../assets/images/paper-figures/quantization/smoothquant-figure-2-intuition.png){ width="720" }

<small>图源：[SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438)，Figure 2。原图表达将 activation outliers 的量化难度平滑迁移到 weight。本站用它说明：SmoothQuant 的目标不是单纯压权重，而是让 activation 也更容易进入 INT8 计算路径。</small>

![SmoothQuant main idea](../assets/images/paper-figures/quantization/smoothquant-figure-5-main-idea.png){ width="760" }

<small>图源：[SmoothQuant](https://arxiv.org/abs/2211.10438)，Figure 5。原图表达通过 per-channel smoothing factor 调整 activation 和 weight 的尺度。本站用它解释：量化前的尺度重分配是方法核心，scale 选择直接决定精度和稳定性。</small>

SmoothQuant 更适合理解 W8A8 这类权重和激活都量化的路径。它常常要和硬件支持、INT8 GEMM、runtime kernel 一起看。

## 三者怎么选

| 方法 | 主要处理 | 适合场景 | 先检查 |
| --- | --- | --- | --- |
| GPTQ | 权重量化后的重构误差 | 想用 4-bit weight-only 降显存 | 校准集、group size、kernel 支持。 |
| AWQ | 激活重要通道对应的权重保护 | 想低成本部署 4-bit LLM | 是否有对应 serving kernel，长尾任务是否掉分。 |
| SmoothQuant | 激活离群值导致的 W8A8 难题 | 想走 INT8 activation + weight 计算 | activation 统计是否稳定，硬件是否真支持。 |

不要只按榜单选方法。更稳的流程是：先确定目标硬件和 runtime，再确定量化对象，然后选校准集和评测集，最后用真实 prompt 长度、batch、KV cache 策略跑端到端 benchmark。

## 常见误解

| 误解 | 更准确的说法 |
| --- | --- |
| INT4 一定比 FP16 快 | 如果 kernel 或内存路径不支持，可能只省显存不提速。 |
| 校准集随便取一点就行 | 校准集决定 scale 和重要通道判断，分布不对会掉长尾能力。 |
| weight-only 量化解决所有显存问题 | 长上下文服务里 KV cache 可能更大。 |
| perplexity 不掉就能上线 | 还要看指令、代码、工具、长上下文、多模态和安全失败桶。 |

## 外部精读

- [Hugging Face Quantization concept guide](https://huggingface.co/docs/transformers/quantization/concept_guide)：量化概念和 Transformers 生态入口。
- [GPTQ](https://arxiv.org/abs/2210.17323)：理解 Hessian-aware weight-only PTQ。
- [AWQ](https://arxiv.org/abs/2306.00978)：理解 activation-aware weight protection。
- [SmoothQuant](https://arxiv.org/abs/2211.10438)：理解 activation outlier migration。
- [llm-awq project](https://github.com/mit-han-lab/llm-awq)：看 AWQ 的实际工具链。
