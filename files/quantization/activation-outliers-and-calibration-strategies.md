# 激活离群值与校准策略

量化最容易被简化成“把 FP16 换成 INT8/INT4”。但真正决定成败的，常常不是权重，而是 activation。权重是训练后固定的；activation 会随着每个 prompt、每张图、每段上下文和每个 batch 动态变化。

!!! note "初学者先抓住"
    activation outlier 指的是某层激活里少数特别大的值。它们会迫使 scale 变粗，让大多数普通值挤在很少的量化格子里。于是平均误差可能看起来还行，但关键细节已经丢了。

## 1. 先复习量化误差从哪里来

均匀量化：

\[
q=\mathrm{clip}\left(\mathrm{round}\left(\frac{x}{s}\right)+z,\ q_{\min},q_{\max}\right)
\]

反量化：

\[
\hat{x}=s(q-z)
\]

误差：

\[
e=x-\hat{x}
\]

| 符号 | 含义 |
| --- | --- |
| \(x\) | 原始浮点激活或权重 |
| \(q\) | 低比特整数 |
| \(\hat{x}\) | 反量化后的近似值 |
| \(s\) | scale，一个量化格子的宽度 |
| \(z\) | zero-point |
| \(e\) | 量化误差 |

**读作什么**：量化误差来自三处：rounding 会把值吸到最近格点；clip 会把超出范围的值截断；后续层会把前面的误差继续放大或改变方向。

## 2. outlier 怎样把 scale 拉粗

对称量化里，一个常见 scale 选择是：

\[
s=\frac{\max(|x|)}{q_{\max}}
\]

| 符号 | 含义 |
| --- | --- |
| \(\max(|x|)\) | 当前量化范围里的最大绝对值 |
| \(q_{\max}\) | 正向最大可表示整数 |
| \(s\) | 每个整数格子覆盖的真实数值宽度 |

**小账**：假设大多数 activation 在 \([-2,2]\)，但有一个 outlier 是 \(30\)。INT4 的 \(q_{\max}=7\)，则：

\[
s=\frac{30}{7}\approx4.29
\]

这意味着 \(-2\) 到 \(2\) 的主体区域总共不到一个整数格子的宽度。很多不同的小值反量化后会变得几乎一样。

!!! warning "常见误解"
    “不裁掉任何 outlier”听起来安全，但它会让 scale 变大，牺牲主体区域分辨率。低比特量化经常是在 clipping error 和 rounding resolution 之间折中。

## 3. clipping 和 resolution 的权衡

如果设置截断阈值 \(\alpha\)，可以把量化范围限制在 \([-\alpha,\alpha]\)：

\[
x_{\mathrm{clip}}=\mathrm{clip}(x,-\alpha,\alpha)
\]

\[
s=\frac{\alpha}{q_{\max}}
\]

| 符号 | 含义 |
| --- | --- |
| \(\alpha\) | 截断阈值 |
| \(x_{\mathrm{clip}}\) | 截断后的值 |
| \(s\) | 截断后重新计算的 scale |

**读作什么**：\(\alpha\) 小，主体值分辨率更细，但 outlier 被裁得更狠；\(\alpha\) 大，outlier 保留更多，但普通值分辨率更粗。

| 阈值选择 | 好处 | 风险 |
| --- | --- | --- |
| 很大 | 少 clipping | 主体值全挤在少数格子里 |
| 很小 | 主体值更细 | 关键 outlier 被截断 |
| 任务校准 | 按真实数据折中 | 依赖校准集代表性 |

## 4. 为什么 LLM 里 activation outlier 常见

| 来源 | 直觉 | 后果 |
| --- | --- | --- |
| 残差叠加 | 多层信息不断累积 | 某些通道幅度越来越大 |
| LayerNorm/RMSNorm 后通道不均衡 | 归一化不保证每个通道同等温和 | 少数 hidden channel 变成尖峰 |
| 稀有 token | 罕见符号、代码、数字、格式触发异常路径 | 校准集没覆盖时容易线上掉点 |
| 长上下文 | 历史信息和位置影响累积 | 后半段 token 分布不同于短校准 |
| 多模态输入 | 图像、表格、UI、视频特征分布异质 | 局部细节通道特别敏感 |

**一句话**：权重分布是静态的，activation 分布是输入驱动的。校准集越不像真实流量，activation scale 越可能选错。

## 5. per-tensor、per-channel、group-wise 怎么选

设激活：

\[
X\in\mathbb{R}^{B\times T\times d}
\]

| 符号 | 含义 |
| --- | --- |
| \(B\) | batch size |
| \(T\) | token 数或时间步 |
| \(d\) | hidden dimension |
| \(X\) | 某层 activation |

| 粒度 | scale 覆盖范围 | 适合什么 | 风险 |
| --- | --- | --- | --- |
| per-tensor | 整个 \(X\) | 简单 baseline | 一个 outlier 拖全局 |
| per-token | 每个 token | 输入动态变化大 | 在线统计开销 |
| per-channel | 每个 hidden 通道 | 通道差异明显 | scale 读取增加 |
| group-wise | 每组通道 | 精度和系统折中 | group size 影响质量和 kernel |

**读作什么**：更细的粒度能把 outlier 影响限制在更小范围，但会增加 scale 元数据和 kernel 复杂度。

## 6. 校准本质上在选什么

校准不是随便喂几条样本。它是在用一小批代表性输入，估计线上会出现的张量范围、scale、zero-point、截断阈值和敏感通道。

常见目标包括：

| 目标 | 公式或直觉 | 适合场景 |
| --- | --- | --- |
| MSE | \(\min \|x-\hat{x}\|_2^2\) | 直接控制数值误差 |
| KL divergence | 让量化后分布接近原分布 | 分类或概率分布校准 |
| 输出误差 | \(\min \|XW-\hat{X}\hat{W}\|^2\) | 关注层输出 |
| 任务代理损失 | 用困惑度、准确率或任务分数选配置 | 更贴近实际使用 |

| 符号 | 含义 |
| --- | --- |
| \(x\) | 原始张量 |
| \(\hat{x}\) | 量化再反量化张量 |
| \(XW\) | 原始线性层输出 |
| \(\hat{X}\hat{W}\) | 量化后线性层输出 |

**读作什么**：最小 MSE 不一定等于最高任务质量。任务越复杂，越要用真实任务桶参与校准和选择。

## 7. 动态量化和静态量化

| 类型 | scale 什么时候确定 | 优点 | 代价 |
| --- | --- | --- | --- |
| 动态量化 | 推理时按当前输入计算 | 适应输入变化 | 每次请求多统计和 Q/DQ 开销 |
| 静态量化 | 校准阶段提前固定 | 推理路径固定，更易优化 | 校准集不准会掉点 |
| weight-only | 权重离线量化，activation 高精 | 稳、部署成熟 | 计算核心未必低比特 |

**读作什么**：动态更灵活，静态更容易快。生产选择不是绝对优劣，而是看输入分布变化、目标延迟和 kernel 支持。

## 8. SmoothQuant 为什么能帮 activation

线性层：

\[
Y=XW
\]

SmoothQuant 使用：

\[
Y=(XD^{-1})(DW)
\]

| 符号 | 含义 |
| --- | --- |
| \(D\) | 对角缩放矩阵 |
| \(XD^{-1}\) | outlier 被平滑后的 activation |
| \(DW\) | 接收难度迁移后的权重 |

**读作什么**：输出保持不变，但 activation 的量化难度降低。因为权重是静态的，放大后的权重可以离线处理；activation 是动态的，线上更难临时补救。

![SmoothQuant intuition](../assets/images/paper-figures/quantization/smoothquant-figure-2-intuition.png){ width="740" }

<small>图源：[SmoothQuant](https://arxiv.org/abs/2211.10438)，Figure 2。原论文图意：activation outlier 会减少有效量化 bit；SmoothQuant 把一部分难度从 activation 迁移到 weight。</small>

!!! note "图解"
    图里的 activation 尖峰就是本页说的 outlier。它们不一定多，但会决定 scale。SmoothQuant 的价值在于先改变数值分布形态，再做量化。

## 9. outlier channel 是否应该保高精

有些系统会对严重 outlier 通道做特殊处理：

| 策略 | 做法 | 适合什么 |
| --- | --- | --- |
| 高精保留 | 少数通道继续 FP16/BF16 | 极敏感层或安全模块 |
| channel splitting | 把 outlier 通道拆分分摊 | 降低单通道峰值 |
| mixed precision | 不同层不同 bit | 生产服务质量保护 |
| rotation/shift | 变换分布后再量化 | outlier 集中明显 |

**核心逻辑**：不是所有维度都值得同等低比特对待。保护少数关键通道，常常比全层提高 bit 更划算。

## 10. 校准集怎么选

一个好的校准集不一定大，但必须像真实流量。

| 场景 | 校准集必须覆盖 |
| --- | --- |
| 聊天/助手 | 短问答、长指令、多轮、工具返回、拒答/安全样本 |
| RAG | 长文档、表格、引用、噪声 chunk、跨段检索 |
| 代码 | 长文件、特殊符号、缩进、罕见库名 |
| 数学 | 长推导、公式、边界条件 |
| VLM | 小字、表格、截图、低清图、图表 |
| VLA | 轨迹片段、失败恢复、细动作、遮挡、长 horizon |

!!! warning "校准集太干净的后果"
    如果校准集只有短、干净、常见的样本，scale 会过于乐观。上线后遇到长上下文、代码块、复杂表格或机器人异常状态时，activation 分布可能完全不同。

## 11. KV cache 里的离群问题

KV cache 也是 activation 的一种动态状态。长上下文中，K/V 张量会随层数、位置、输入内容变化。

\[
\mathrm{KV\ Memory}\approx2\times L\times B\times T\times H_{\text{kv}}\times d_h\times \mathrm{bytes}
\]

| 符号 | 含义 |
| --- | --- |
| \(L\) | 层数 |
| \(B\) | batch 或并发序列 |
| \(T\) | 上下文长度 |
| \(H_{\text{kv}}\) | KV head 数 |
| \(d_h\) | 每个 head 维度 |
| \(\mathrm{bytes}\) | 每个元素字节数 |

**读作什么**：KV cache 低比特能显著省长上下文显存，但如果 K/V 的量化误差改变注意力分布，模型可能丢长程细节、引用错位或推理不稳。

## 12. 评测时看什么

| 检查项 | 为什么重要 |
| --- | --- |
| per-layer activation range | 找出 outlier 严重层 |
| scale/amax 日志 | 观察校准和线上是否分布漂移 |
| 任务桶回归 | 平均分掩盖长尾失败 |
| 长上下文专项 | KV 和后段 token 最容易出问题 |
| kernel trace | 确认动态统计和 Q/DQ 没吃掉收益 |
| 高风险样本回放 | 找安全头、动作头、OCR、代码等局部能力退化 |

## 13. 一个财务文档 VLM 例子

常规问答样本里，量化模型几乎不掉分；但上线后，复杂发票的税率、小数点、跨页金额关联频繁出错。排查发现校准集里缺少高分辨率表格和低清扫描件，activation scale 主要适配了普通段落问答，导致表格结构和小字通道的分辨率不足。

解决思路不是盲目提高全模型 bit，而是：

1. 补充表格、小字、低清图、跨页金额样本做校准。
2. 对视觉 connector、OCR 相关层和结构头保守混合精度。
3. 按字段级准确率、金额误差、跨页一致性单独回归。

## 14. 小结

activation outlier 是低比特量化中最容易让初学者低估的问题。它让 scale 变粗，让普通值失去分辨率，也让校准集质量变成部署成败的关键。理解 outlier 后，再看 AWQ、SmoothQuant、KV cache quant 和混合精度，就会发现它们都在处理同一个核心问题：把有限的数值表示能力分给最需要的地方。
