# 低比特训练与数值格式：误差会被写回参数

低比特训练最容易被讲错的地方，是把它讲成“训练版量化”。推理量化的误差大多停在一次前向里；训练里的低精度会进入 loss、反向传播、梯度规约和 optimizer update，最后写回参数。下一步训练看到的，已经是被上一轮舍入、截断、scale 失配改过的模型。

不要急着比较 FP8、MXFP8、FP4、NVFP4 谁更省 bit。更重要的问题是：**哪些张量路径可以低精度，哪些路径必须被高精度保护，scale 和累加状态怎样随训练一起管理？**

这也是低比特训练和低比特推理的根本差别。推理系统关心的是一次请求的质量、延迟和 fallback；训练系统还要关心误差会不会沿着参数更新闭环放大。一个 FP8 kernel 能跑，只说明算子可用；一套低比特训练 recipe 能长跑，才说明格式、scale、累加、checkpoint 和监控真正合在了一起。

## 先把训练看成一张状态链

一轮训练不是只有 `W @ X`。如果把低比特放进训练，至少会碰到四类状态。

| 状态链路 | 低精度可能作用在哪里 | 省下的东西 | 最容易出问题的地方 |
| --- | --- | --- | --- |
| 前向 | 权重、激活、attention/MLP GEMM | HBM 读写、Tensor Core 吞吐、activation cache | logits、loss、残差流、outlier |
| 反向 | 保存的 activation、反向 GEMM、梯度 | 峰值显存、反向吞吐 | 梯度方向、loss spike、NaN/Inf |
| 优化器 | 梯度 bucket、一阶/二阶动量、master weight | 常驻显存、ZeRO/FSDP shard 体积 | Adam 更新尺度、长历史误差 |
| 系统状态 | scale、amax history、loss scaler、checkpoint manifest | 恢复一致性、端到端吞吐 | resume 后轨迹漂移、metadata 开销 |

AdamW 的显存账能帮助定位这件事。只看参数本身时，BF16 参数约是 \(2\Psi\) bytes；但混合精度训练还可能保存 BF16 梯度、FP32 master weight、FP32 一阶动量和 FP32 二阶动量，模型状态常被粗略估成 \(16\Psi\) bytes 量级。这里 \(\Psi\) 是参数个数，不包括 activation、通信 buffer 和 kernel workspace。

这笔账说明了低比特训练的一个现实：只把 GEMM 输入换成 FP8，收益会被 optimizer state、activation cache 和通信状态吃掉一部分。低比特训练真正要省的不是某一个 tensor，而是整条训练状态链。

## FP8 为什么先成为训练入口

整数低比特常写成：

\[
q=\mathrm{round}(x/s), \qquad \hat{x}=s q
\]

这里 \(x\) 是原始浮点张量，\(s\) 是 scale，\(q\) 是整数或低比特表示，\(\hat{x}\) 是反量化后参与计算的近似值。INT8/INT4 的优点是规则简单、kernel 成熟；难点是动态范围主要靠 \(s\) 扛。一个张量里如果大多数值在 \([-2,2]\)，但偶尔有一个 1000，scale 为了容纳 1000 会让普通值挤在很少的桶里；如果截断 1000，又可能伤到关键 token 或关键层。

FP8 仍然是浮点数。一个浮点值可粗略写成：

\[
x=(-1)^{s_{\mathrm{sign}}}\times 2^e\times m
\]

其中符号位决定正负，指数 \(e\) 决定动态范围，尾数 \(m\) 决定相邻可表示值之间有多细。FP8 只有 8 bit，所以必须在指数和尾数之间取舍。

| 格式 | 直觉 | 训练里常见用途 |
| --- | --- | --- |
| E4M3 | 指数少一点，尾数多一点，普通范围更细 | 权重、激活、前向 GEMM |
| E5M2 | 指数多一点，尾数少一点，范围更宽 | 梯度、反向里范围更大的张量 |

这不是硬规则。DeepSeek-V3 的 FP8 框架大量使用 E4M3，同时靠细粒度 scale 和更安全的累加路径保护稳定性。FP8 比 INT4 更适合作为训练入口，不是因为“8 bit 天然安全”，而是它保留指数位，更容易适配训练张量不断漂移的动态范围。

![FP8 training loss curves 原论文图](../assets/images/paper-figures/foundations/fp8-formats-figure-1-training-loss.png){ width="700" }

<small>图源：[FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)，Figure 1。原图比较 BF16 与 FP8 路线在 GPT/BERT 类模型上的 loss 或 perplexity 曲线；读图时重点不是“FP8 也能训”，而是格式选择、scale 策略、累加精度和敏感算子保护共同成立时，曲线才可能贴近 BF16。</small>

这张图给的是底线，不是通行证。迁移到新模型、新激活函数、更长上下文或更长 token budget 时，还要重新监控 loss spike、梯度范数、activation percentile、scale 饱和率和 NaN/Inf。

## scale 是控制面，不是注脚

一次 FP8 cast 可以理解成：

\[
x_{\mathrm{fp8}}=\mathrm{FP8Cast}(x/s), \qquad \hat{x}=s\cdot x_{\mathrm{fp8}}
\]

\(s\) 把原始张量 \(x\) 映射到 FP8 可表示范围内，\(x_{\mathrm{fp8}}\) 是存进去的低精度值，\(\hat{x}\) 是计算时恢复出来的近似值。scale 太小会 overflow 或截断大值；scale 太大又会让普通值落到过粗的表示点上。训练过程中张量分布一直变，scale 也必须跟着变。

NVIDIA Transformer Engine 的 delayed scaling recipe 就是在做这件事：运行时记录历史 amax，再用历史最大值估计下一段时间的 scale。这里 amax 是一段张量里观测到的最大绝对值。它比固定 scale 更能适应训练漂移，也比每次精确重算更便宜；但它不是免费午餐。如果 outlier 突然出现，历史 scale 可能来不及反应；如果历史窗口被少数极端值支配，普通值会长期被压粗。

scale 粒度同样重要。

| scale 粒度 | 好处 | 代价 |
| --- | --- | --- |
| per-tensor | 元数据少，kernel 简单 | 容易被一个 outlier 拖住 |
| per-token / per-channel | 更贴合局部分布 | scale 更多，layout 和 kernel 更复杂 |
| per-block / MX | 在局部适配和硬件实现之间折中 | block size、scale 类型、累加方式都要协同设计 |

低比特训练的第一性问题不是“选 FP8 还是 BF16”，而是“每条张量路径用什么 scale 粒度，scale 如何更新，scale 状态如何保存，异常时如何回退”。

## 训练误差会闭环放大

推理量化常常可以用 calibration 估计一次性误差，因为模型权重不再变化。训练不同。低精度误差会改变梯度，梯度会改变参数，参数又会改变下一步激活分布。

```text
参数分布 -> 激活分布 -> scale/rounding -> 梯度 -> optimizer update -> 新参数分布
```

这个闭环让“短跑稳定”不等于“长跑稳定”。[Scaling FP8 Training to Trillion-Token LLMs](https://arxiv.org/abs/2409.12517) 把 FP8 训练推到更长 token 规模后，指出 SwiGLU outlier amplification 会带来稳定性问题。这类结果比前几千 step 的 loss 重合更值得重视，因为真实预训练的风险经常出现在中后段。

activation 是低比特训练的主战场之一。权重、梯度和 optimizer state 可以被 ZeRO/FSDP 分片，但 activation cache 依赖 micro-batch、sequence length、层数和是否重计算，常常直接压在单卡峰值显存上。反向传播还要读取前向保存的 activation；如果保存时压得太粗，梯度方向会被污染。

这也是 COAT 这类工作把 optimizer states 和 activation 作为目标的原因。只把 linear GEMM 改成 FP8，不能覆盖训练显存的大头，也不能解释最常见的反向不稳定。

## mixed precision 是路径图，不是全局 dtype

![DeepSeek-V3 FP8 framework 原论文图](../assets/images/paper-deep-dives/technical-reports/deepseek-v3/figure-6-fp8-framework.png){ width="920" }

<small>图源：[DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)，Figure 6。原图展示 DeepSeek-V3 的 FP8 mixed precision training framework：GEMM 相关路径进入 FP8，部分状态、累加、embedding、MoE dispatch/combination 和 optimizer 相关路径保留更高精度或特殊处理。</small>

这张图最该看的不是颜色，而是边界。一个训练系统会把 `Fprop`、`Dgrad`、`Wgrad`、activation cache、通信、optimizer update 拆成不同路径。某些矩阵乘可以用 FP8 输入，某些中间量必须高精度累加，某些模块要保留 BF16/FP32，某些通信张量还要考虑量化、反量化和 overlap 开销。

| 边界 | 为什么不能粗暴 FP8 |
| --- | --- |
| loss / normalization / softmax | 小概率、归一化统计和 logits 差值对舍入误差敏感 |
| optimizer update | 一阶/二阶动量积累很多 step，误差不是一次性的 |
| distributed communication | 低精度通信能省带宽，但 scale metadata、dequant 和 overlap 会影响端到端速度 |
| checkpoint / resume | scale、amax history、RNG、并行拓扑不一致，会让恢复后的训练轨迹漂移 |

DeepSeek-V3 的系统意义在这里：它把 FP8 放进大规模 MoE 预训练，而不是单独展示一个量化算子。读这类报告时，要同时看格式、scale、累加、并行、通信和 checkpoint，而不是摘一句“使用 FP8 训练”。

## 累加精度决定长维求和能不能扛住

![DeepSeek-V3 FP8 accumulation 原论文图](../assets/images/paper-deep-dives/technical-reports/deepseek-v3/figure-7-fp8-accumulation.png){ width="880" }

<small>图源：[DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)，Figure 7。原图展示细粒度量化、tile/group scale 与提高累加精度的设计；读图时重点看 FP8 元素只负责输入表示，GEMM 的部分和最终累加仍要由更安全的精度路径保护。</small>

矩阵乘不是两个低精度数相乘就结束。一个输出元素来自很多乘积的求和：

\[
y_{ij}=\sum_k a_{ik}w_{kj}
\]

\(a_{ik}\) 是第 \(i\) 个输入向量的第 \(k\) 个元素，\(w_{kj}\) 是权重矩阵第 \(k,j\) 个元素，\(y_{ij}\) 是输出。即使 \(a\) 和 \(w\) 用 FP8 存储，求和过程也不能随便用 FP8 累加，因为很多小误差会在 \(k\) 维上聚起来。

所以论文和系统报告常同时写 “FP8 GEMM” 和 “increased precision accumulation”。前者省带宽、提吞吐；后者防止长维度求和把误差滚大。对于长上下文、宽 MLP、MoE expert 或大 batch，累加维度越长，累加精度越不能被当作实现细节。

## activation、梯度、optimizer state 是三张账

低比特训练最容易写糊的一点，是把所有张量都叫“量化对象”。它们的统计性质完全不同。

**activation** 随输入、层、token 位置和训练阶段变化。attention、SwiGLU、MoE gate、residual stream 都可能制造 outlier。activation cache 还要在反向中复用，所以压缩 activation 不是只看前向误差，而是要看反向梯度是否还对。

**梯度** 的范围和稀疏结构跟 loss、batch、clip、parallel reduction 相关。梯度低精度化如果引入系统性 bias，optimizer 会沿错误方向更新。梯度通信低精度还要看 all-reduce 前后的 scale 是否一致，不能只看单卡局部误差。

**optimizer state** 是长历史变量。Adam 的 \(m_t\) 和 \(v_t\) 分别累积一阶与二阶统计：

\[
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\qquad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
\]

\(g_t\) 是当前 step 的梯度，\(m_t\) 是动量，\(v_t\) 是平方梯度的滑动平均，\(\beta_1,\beta_2\) 控制历史保留比例。optimizer state 的低精度误差会跨很多 step 积累，所以它和 activation 的压缩策略不能混为一谈。

FP8-LM 把梯度和 optimizer state 纳入低精度训练框架，COAT 则明确把 optimizer states 和 activation 作为显存主目标。它们共同说明：低比特训练的收益来自整条训练状态链，而不是某一个 GEMM kernel。

## FP4/MXFP 更像系统共设计

当 bit 数继续降到 FP4，元素本身能表达的点太少，scale 就从辅助变量变成核心机制。Microscaling 的思路是让一小块元素共享 scale：

\[
x_i \approx s_{\mathrm{block}}\cdot q_i
\]

\(q_i\) 是 block 内第 \(i\) 个低比特元素，\(s_{\mathrm{block}}\) 是这一小块共享的缩放因子。相比 per-tensor scale，per-block scale 更能贴合局部分布；相比 per-element scale，它又不至于产生过多元数据。OCP MX 规范把这类 block-scaled format 标准化，论文里的 MXFP8、MXFP4、MXINT8 都沿着这个方向展开。

FP4 的困难也在这里。元素只有 4 bit，最大值附近、零附近和中间区间的可表示点都非常稀疏。单纯把 BF16 张量 cast 成 FP4 几乎没有训练稳定性的希望，必须配合 block scale、Hadamard rotation 或其他 outlier 平滑、高精度保护层、随机舍入、无偏梯度估计和原生 kernel。

因此 FP4 训练不是“FP8 再小一半”。它更像数制、scale、optimizer、kernel 和硬件共同设计。没有硬件低精度 Tensor Core、没有 fused dequant/accumulation、没有 scale metadata 的 layout 支持，理论上的 4 bit 存储很容易被 cast、metadata 和额外 kernel 开销抵消。

## 一次低比特训练实验怎样验收

低比特训练报告如果只给 loss 曲线和吞吐，证据是不够的。至少要把四类问题说清楚。

| 要证明的事 | 应该看什么 | 为什么 |
| --- | --- | --- |
| 训练动力学没坏 | loss spike、梯度范数、NaN/Inf、seed/LR 鲁棒性 | 低精度误差会写回参数，不能只看平均 loss |
| 数值控制面有效 | amax、scale 饱和率、activation percentile、overflow/underflow | scale 失配通常先出现在统计量里，后出现在 loss 里 |
| 系统收益兑现 | peak memory、step time、kernel profile、通信/计算 overlap | 低 bit 可能省某项显存，也可能增加 cast、metadata 和同步开销 |
| 恢复一致 | scale/amax history、loss scaler、RNG、数据游标、并行拓扑、分片 manifest | checkpoint load 成功不等于训练轨迹可复现 |

对于世界模型、VLA 或机器人策略，指标还要往下游走一步。低精度 checkpoint 和 BF16 checkpoint 应比较 action ranking、event divergence、near-miss recall、closed-loop return，而不是只比较语言 perplexity。低比特训练最终省的是训练成本，不能买来动作条件、风险估计或长时 latent 的退化。

## 最后判断

低比特训练可以按四句话记：

1. FP8 能进入训练，是因为它保留指数位，并配合 scale、累加和 mixed precision 保护。
2. scale 是控制面，不是注脚；scale 粒度、更新方式和 checkpoint 状态决定训练能不能复现。
3. activation、梯度和 optimizer state 的风险不同，不能用一个“量化误差”统一带过。
4. FP4/MXFP4 的收益更依赖硬件和 kernel，共同设计比单个量化公式更重要。

## 外部材料

- [NVIDIA Transformer Engine: FP8 Delayed Scaling](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/fp8_delayed_scaling/fp8_delayed_scaling.html)：理解 amax history、recipe 和 FP8 runtime 状态。
- [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)：理解 E4M3/E5M2 为什么服务训练和推理。
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)：看大规模模型如何把 FP8、MoE、通信和累加精度放到同一个系统里。
- [COAT: Compressing Optimizer states and Activations for Memory-Efficient FP8 Training](https://arxiv.org/abs/2410.19313)：理解 activation 与 optimizer state 为什么是训练显存主战场。
- [Scaling FP8 Training to Trillion-Token LLMs](https://arxiv.org/abs/2409.12517)：看短跑稳定和长跑稳定为什么不是一回事。
- [Microscaling Data Formats for Deep Learning](https://arxiv.org/abs/2310.10537)：理解 MXFP/MXINT 这类 block scale 格式的动机。
- [NVIDIA 中文技术博客：FP8 在大模型训练中的应用、挑战及实践](https://developer.nvidia.cn/blog/fp8-llm-app-challenges/)：适合作为中文工程讲法入口，再回到论文和官方文档核对细节。
