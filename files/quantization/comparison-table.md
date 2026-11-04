# 量化方法与数值格式对照

选择前先写清对象。压权重、压激活、压KV和减少微调状态，解决的是不同预算。下面的表按职责比较，不把算法名称当作质量排行榜。

| 名称 | 它规定或改变什么 | 需要的数据或训练 | 输出后还要确认什么 |
| --- | --- | --- | --- |
| 直接舍入PTQ | 训练后选择范围、scale和码值 | 权重本身；激活量化通常需范围统计 | 截断误差、输出误差、可用kernel |
| GPTQ | 利用校准输入相关性补偿权重量化误差 | 代表性校准输入 | 分组、阻尼、排列与packing兼容 |
| AWQ | 利用激活搜索通道缩放与截断 | 校准输入及块输出比较 | 等价变换、scale布局、WQLinear执行 |
| SmoothQuant | 在激活与权重之间迁移范围 | 激活通道统计 | 所有分支等价、W/A精度和部署布局 |
| QAT | 前向模拟量化并继续训练可训练参数 | 训练数据与近似反向规则 | 模拟量化是否对齐真实kernel |
| QLoRA | 冻结量化底座，训练低秩adapter | 微调数据 | 底座身份、adapter路径和合并后误差 |

QAT是训练制度，GPTQ/AWQ是PTQ方法，QLoRA描述微调结构；三者不处在同一条“谁更先进”的轴上。可组合性由具体实现决定。

## W4A16 仍然缺哪些信息

W和A分别表示权重与激活的数值路径。W4A16只给出位数概况，仍需补充码本、group size、zero point、scale dtype、packing、累加方式与未量化模块。

| 表示 | 需要特别分清的地方 | 主要可能收益 |
| --- | --- | --- |
| 均匀INT4/INT8 | 码值范围、对称性、分组、zero point | 存储和带宽；原生计算收益取决于kernel |
| FP8 | 具体浮点格式、scale约定、块布局与累加 | 支持路径上的容量、带宽或矩阵计算 |
| NF4 | 非均匀码本索引与块尺度 | 冻结权重的紧凑表示，常见于QLoRA |
| KV低精度 | K/V各自粒度、页布局、写入和读取尺度 | 长上下文驻留容量与读取带宽 |

## 用一个需求挑选候选

若BF16权重放不下而输入激活预算较小，可先比较weight-only候选。若大batch矩阵乘占主要时间，且目标硬件有成熟W/A低精度路径，再评估激活量化。若权重已能放下、并发仍被历史状态挤满，则先算KV预算。若目的是在有限显存下微调，转到QLoRA的训练状态分析。

这些是筛选候选的方法。最终结论需要同一模型、同一输入、相同质量约束及明确版本的实验，不能从表中推导固定速度倍数或统一质量顺序。

详细计算见[量化数学](symbols-and-minimal-math.md)、[三种PTQ方法](ptq-gptq-awq-smoothquant.md)、[QLoRA](qlora-and-quantized-training.md)与[运行时](quantization-runtimes-and-frameworks.md)。

![SmoothQuant main idea](../assets/images/paper-figures/quantization/smoothquant-figure-5-main-idea.png){ width="820" }

<small>图源：[SmoothQuant](https://arxiv.org/abs/2211.10438)，Figure 5。原图表达把激活中的 outlier 难度迁移到权重侧，使 W8A8 更容易部署；读图重点是把量化方法对照表里的 PTQ/GPTQ/AWQ/SmoothQuant/FP8 分到不同瓶颈。它不能证明 SmoothQuant 比所有方法更好，只解释一类激活 outlier 处理思路。</small>

## 对照源码与延伸阅读

- [GPTQ 实际补偿流程](https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/gptq.py)
- [QLoRA 实际加载与训练设置](https://github.com/artidoro/qlora/blob/7f4e95a68dc076bea9b3a413d2b512eca6d004e5/qlora.py)
