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

部署时要同时决定量化对象是 weight、activation、KV cache 还是 optimizer state，粒度是 per-tensor、per-channel、per-group、per-token 还是 per-head，格式是 INT8/INT4、FP8、NF4、MXFP8 还是别的低精度格式，并明确累加精度、反量化位置，以及 kernel 是否支持对应 layout 和 packing。

## QAT 在优化什么

PTQ 便宜，但在极低比特、激活量化或高精度业务中不一定稳。QAT 会在训练时显式模拟量化：

\[
\hat{x}=\mathrm{Dequant}(\mathrm{Quant}(x))
\]

让模型在训练期间适应量化噪声。

QAT 不是简单“重新训一遍”，而是在优化权重分布与量化区间、激活离群值、误差在层间的传播、attention / MLP / 输出头在低比特环境下的稳定性，以及哪些层保留高精度、哪些层进入低精度路径。

训练中常用 fake quant 和 STE。取整不可导，STE 近似让未被 clip 区间的梯度直接传回。它不完美，但工程上实用。

## 什么时候值得做 QAT

更值得考虑 QAT 的情况，是目标已经进入 INT4、FP8 activation 或更激进低比特，PTQ 后在结构化抽取、长推理、多轮工具调用上明显掉点，业务对小质量退化很敏感，团队想同时处理权重、激活和 KV cache，并且目标量化格式和 kernel 栈已经基本确定。

不一定值得做 QAT 的情况也很常见：PTQ 已经满足质量和成本，kernel 对目标格式支持差，训练预算不足以稳定调参，服务瓶颈其实在 KV、路由、工具或 prefill，或者量化格式仍在频繁变化。

QAT 更像 PTQ 不够稳时的下一层保险，而不是默认第一步。

### 世界模型里什么时候值得做 QAT

如果对象是 VLM/VLA 或动作条件世界模型，QAT 的触发条件要比文本模型更具体：

| 现象 | 说明 | QAT 是否值得 |
| --- | --- | --- |
| 文本/VQA 几乎不掉，但动作坐标抖 | 误差集中在 connector 或 action head | 先混合精度保护，必要时对 action path 做 QAT |
| 短视频预测正常，长 rollout 漂移 | KV / memory / latent cache 误差累积 | 值得做 KV 或 latent cache fake quant 训练 |
| 风险概率整体偏低 | risk head 校准被破坏 | 不宜激进低精，优先重校准和高精保留 |
| kernel 支持 FP8 主干，但 connector 回退 | runtime 路径不一致 | 先修 kernel / layout，再考虑 QAT |
| 低精 rollout 能省大量并发成本 | 系统收益明确 | 值得用小规模 QAT 换稳定上线 |

世界模型 QAT 的目标不是“所有层都适应低精度”，而是让误差出现在不影响决策的地方。对接触、风险、动作输出这些小误差会放大的路径，保高精通常比强行 QAT 更划算。

## Kernel 决定收益是否兑现

量化方案理论上好，不代表部署一定快。真正落点取决于 kernel 是否支持该 bit 宽和 packing，是否支持 group size、scale 粒度和 layout，是否能 fuse dequant、GEMM、bias、activation、requant，是否避免频繁 dequant-requant，是否能高效利用 Tensor Core、寄存器和带宽，以及是否与 serving runtime 的 batch 和 KV 管理兼容。

某种 4bit 方法论文质量最好，但如果现有 kernel 对它支持差，线上吞吐可能不如一个略逊但实现成熟的方案。量化工程选择常常是“当前硬件和 kernel 上最值得跑的格式”，而不是“论文表格里最优的格式”。

## KV Cache 是系统瓶颈

权重量化减少模型参数显存，但长上下文和高并发下，KV cache 仍可能成为大头：

\[
M_{\text{KV}}\sim
2\cdot B\cdot L\cdot T\cdot H\cdot d_h\cdot b
\]

KV 量化的收益直接作用于并发容量和长上下文成本，但风险也更大。decode 每步都依赖历史 KV，误差会持续影响输出；attention 对 K/V 误差的敏感度和层、head、位置有关；不同请求长度和动态 batch 会改变 cache 访问模式；KV scale、page layout、eviction 策略也会和 runtime 深度耦合。

KV cache 量化必须在真实长上下文、长输出和多轮任务上验证，不能只看短文本困惑度。

### rollout KV 量化：一笔显存账

以 32 层、hidden size 4096、batch 16、上下文 8192、BF16 KV 为例，KV cache 近似为：

\[
2\times B\times L\times T\times d_{\text{model}}\times b
=2\times16\times32\times8192\times4096\times2
\approx64GiB
\]

如果 KV 改成 INT8 / FP8，理论上可降到约 32GiB，再加 scale metadata。对世界模型 rollout，这个差异很实际：同样显存下可以把候选动作数从 8 提到 16，或者把 horizon 从 4 秒拉到 8 秒。

但 rollout KV 和普通聊天 KV 不一样。聊天任务里 KV 误差主要伤下一 token；世界模型里 KV 误差会伤未来 latent、动作分叉和风险估计。低精 KV 如果让“向左抓”和“向右推”的未来变得更像，规划器会以为动作无关紧要。

### rollout KV 量化验收清单

不要用单个 perplexity 或短视频 loss 验收 rollout KV。建议同时过四道门：

| 门禁 | 指标 | 合格信号 |
| --- | --- | --- |
| 开环预测 | latent MSE / LPIPS / temporal consistency | 相对 BF16 只小幅退化，关键对象不漂移 |
| 动作敏感性 | 不同动作未来差异、counterfactual ranking | 低精 KV 下 action ranking 不翻转 |
| 风险校准 | collision / slip / lost-object ECE、near-miss recall | 风险不系统性偏低 |
| 闭环收益 | success rate、recovery rate、collision rate、cost per success | 显存或吞吐收益没有被失败率吃掉 |

一个实用灰度标准可以写成：在 hard replay 桶里，低精 KV 的动作候选排序与 BF16 一致率不低于 98%，near-miss recall 下降不超过 1 个百分点，闭环成功率下降不超过预设业务阈值；否则先只量化远期 memory KV，不量化当前窗口和风险相关层。

### 硬证据模块：KV/QAT 必须用 world-model 指标验收

KV cache 和 QAT 页最容易滑回通用 LLM 指标，例如 perplexity、MMLU 或 tokens/s。对世界模型来说，这些只能算基础门禁，真正要看低精度是否改变了未来分叉和风险排序。

| 证据项 | 最小可复算例子 | 失败案例 | 验收指标 |
| --- | --- | --- | --- |
| 本页解决哪项成本 | BF16 KV vs INT8/FP8 KV 的 GiB、带宽、latency 对比 | 文件小了，但 dequant/layout 让端到端不快 | KV GiB、kernel hit rate、P95 rollout latency |
| Action sensitivity | 同一历史下 8 条候选动作的低精/高精未来排序 | 低精 KV 让 left/right/grasp 未来更像 | candidate ranking agreement、latent delta |
| Risk calibration | hard-negative replay 上 risk head 对比 | 低精后滑落/碰撞风险系统性偏低 | near-miss recall、ECE、false negative |
| Long-horizon consistency | 2s/8s/30s rollout 对比 | 远期 memory KV 压坏，物体身份漂移 | object permanence、latent drift |
| 证据等级 | microbench、end-to-end latency、闭环小桶要分开报告 | 只给 KV kernel speedup，没有任务结果 | throughput + task bucket + failure replay |

一个可接受的报告格式应写成：`INT8 rollout KV 让显存从 64GiB 降到 32GiB，单 kernel 1.6x，但端到端 rollout 只有 1.18x；hard replay 中 top-1 action ranking agreement 为 99.1%，near-miss recall 下降 0.4pp，因此只量化远期 KV 和非风险路径。` 如果缺少后半句，就还不是世界模型量化证据。

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

量化上线前建议同时看模型质量、数值稳定、性能、显存、kernel、runtime 和回退路径。质量评测要覆盖通用、代码、数学、结构化抽取、长上下文和多轮工具；稳定性要看极值输入、长序列、低温/高温和重复生成；性能要看 TTFT、TPOT、tokens/s 和 batch 下吞吐；显存要拆开权重、activation、KV cache 和 metadata；kernel 要确认是否走目标低比特路径、有无频繁 dequant；runtime 要验证 dynamic batching、prefix cache、KV eviction 是否正常；异常请求还要能切回高精路径。

QAT、kernel 和 KV cache 必须作为一条链设计。量化成功的标志不是模型文件变小，而是在真实服务负载下，以可接受质量换来可验证的显存、吞吐和成本收益。
