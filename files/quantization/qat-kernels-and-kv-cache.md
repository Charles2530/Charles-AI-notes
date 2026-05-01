# QAT、Kernel 与 KV Cache

很多量化讨论只停留在“精度掉不掉”，但真实部署里，量化是否成功还取决于 `QAT`、kernel 和 `KV cache` 三件事是否匹配。只压权重而不看激活、kernel 和缓存，常会出现“模型文件小了，系统却没有明显变快”的结果。

!!! note "初学者先抓住"

    量化落地要同时过三关：模型能不能适应低精度，kernel 能不能真的跑快，KV cache 能不能在长上下文里省显存。只看离线 perplexity 或文件大小，很容易误判线上收益。

!!! example "有趣例子：行李压缩袋"

    把衣服压小只是第一步；如果箱子拉链、安检流程和取用方式都不配合，旅行并不会更顺。量化也是一样：权重压小了，但 runtime、kernel、KV layout 不配合，服务不一定更快。

## 量化的统一写法

均匀量化可写成：

\[
q=\mathrm{clip}\left(\left\lfloor \frac{x}{s}\right\rceil+z,q_{\min},q_{\max}\right),
\qquad
\hat{x}=s(q-z)
\]

其中 \(s\) 是 scale，\(z\) 是 zero-point，\(q\) 是量化值，\(\hat{x}\) 是反量化近似。

部署时要同时决定：

1. 量化对象：weight、activation、KV cache、optimizer state；
2. 粒度：per-tensor、per-channel、per-group、per-token、per-head；
3. 格式：INT8/INT4、FP8、NF4、MXFP8 等；
4. 累加精度和反量化位置；
5. kernel 是否支持对应 layout 和 packing。

## QAT 在优化什么

PTQ 便宜，但在极低比特、激活量化或高精度业务中不一定稳。QAT 会在训练时显式模拟量化：

\[
\hat{x}=\mathrm{Dequant}(\mathrm{Quant}(x))
\]

让模型在训练期间适应量化噪声。

QAT 不是简单“重新训一遍”，而是在优化：

1. 权重分布与量化区间；
2. 激活离群值；
3. 误差在层间的传播；
4. attention、MLP、输出头在低比特环境下的稳定性；
5. 哪些层保留高精度，哪些层进入低精度路径。

训练中常用 fake quant 和 STE。取整不可导，STE 近似让未被 clip 区间的梯度直接传回。它不完美，但工程上实用。

## 什么时候值得做 QAT

更值得考虑 QAT 的情况：

1. 目标是 INT4、FP8 activation 或更激进低比特；
2. PTQ 后结构化抽取、长推理、多轮工具调用明显掉点；
3. 业务对小质量退化很敏感；
4. 想同时处理权重、激活和 KV cache；
5. 目标量化格式和 kernel 栈已经基本确定。

不一定值得的情况：

1. PTQ 已经满足质量和成本；
2. kernel 对目标格式支持差；
3. 训练预算不足以稳定调参；
4. 服务瓶颈其实在 KV、路由、工具或 prefill；
5. 量化格式仍在频繁变化。

QAT 更像 PTQ 不够稳时的下一层保险，而不是默认第一步。

## Kernel 决定收益是否兑现

量化方案理论上好，不代表部署一定快。真正落点取决于 kernel：

1. 是否支持该 bit 宽和 packing；
2. 是否支持 group size、scale 粒度和 layout；
3. 是否能 fuse dequant、GEMM、bias、activation、requant；
4. 是否避免频繁 dequant-requant；
5. 是否能高效利用 Tensor Core、寄存器和带宽；
6. 是否与 serving runtime 的 batch 和 KV 管理兼容。

某种 4bit 方法论文质量最好，但如果现有 kernel 对它支持差，线上吞吐可能不如一个略逊但实现成熟的方案。量化工程选择常常是“当前硬件和 kernel 上最值得跑的格式”，而不是“论文表格里最优的格式”。

## KV Cache 是系统瓶颈

权重量化减少模型参数显存，但长上下文和高并发下，KV cache 仍可能成为大头：

\[
M_{\text{KV}}\sim
2\cdot B\cdot L\cdot T\cdot H\cdot d_h\cdot b
\]

KV 量化的收益直接作用于并发容量和长上下文成本，但风险也更大：

1. decode 每步都依赖历史 KV，误差会持续影响输出；
2. attention 对 K/V 误差敏感度和层、head、位置有关；
3. 不同请求长度和动态 batch 会改变 cache 访问模式；
4. KV scale、page layout、eviction 策略会和 runtime 深度耦合。

KV cache 量化必须在真实长上下文、长输出和多轮任务上验证，不能只看短文本困惑度。

## 端到端链路

成熟量化设计要同时拉通：

| 环节 | 要确认什么 |
| --- | --- |
| 训练/QAT | fake quant、scale、保留高精层、loss 稳定 |
| 导出 | 权重 packing、metadata、scale 格式 |
| Kernel | GEMM、attention、dequant、requant 是否高效 |
| Runtime | dynamic batching、KV paging、prefix cache 是否兼容 |
| 评测 | 质量、延迟、吞吐、显存、长上下文稳定性 |

只要其中一环没跟上，整体收益就可能被吃掉。训练时假设 per-channel INT4，线上 kernel 只高效支持另一种 packing，最终部署格式就会和训练假设不一致。

## 评测清单

量化上线前建议同时看：

1. 模型质量：通用、代码、数学、结构化抽取、长上下文、多轮工具；
2. 数值稳定：极值输入、长序列、低温/高温、重复生成；
3. 性能：TTFT、TPOT、tokens/s、batch 下吞吐；
4. 显存：权重、activation、KV cache、metadata；
5. Kernel：是否走目标低比特路径，有无频繁 dequant；
6. Runtime：dynamic batching、prefix cache、KV eviction 是否正常；
7. 回退：异常请求是否能切回高精路径。

QAT、kernel 和 KV cache 必须作为一条链设计。量化成功的标志不是模型文件变小，而是在真实服务负载下，以可接受质量换来可验证的显存、吞吐和成本收益。
