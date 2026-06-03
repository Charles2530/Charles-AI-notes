# 归一化、残差与激活来源台账

这页记录归一化、残差连接、Transformer Norm placement、激活函数和 activation outlier 相关来源。它服务的写作原则是：不要把 BatchNorm、LayerNorm、RMSNorm、Pre-Norm、GELU、SwiGLU 写成术语清单，而要说明它们分别在“尺度、梯度路径、非线性选择、低精度数值”里解决什么问题。

## 使用方式

写基础组件文章时，先把材料分成四类。

第一类是优化稳定性来源。ResNet、BatchNorm、LayerNorm、RMSNorm 和 Pre-LN/Post-LN 论文帮助本站解释深层网络为什么会坏，以及组件如何改变训练动力学。正文应优先讲机制：残差把 \(H(x)\) 改成 \(x+F(x)\)，LayerNorm 改变统计轴，RMSNorm 保留尺度归一化，Pre-Norm 给 residual stream 留出更直接的 identity path。

第二类是激活与门控来源。ReLU、GELU、Swish、GLU variants 和 LLaMA 这类来源帮助本站解释 MLP 不是“线性层后随便接一个函数”，而是在 token 内部做通道筛选。写 SwiGLU 时必须把 gate/value 两条投影分支讲清楚，不能把它简化成 Swish 曲线。

第三类是大模型工程来源。Megatron-LM、LLaMA、DeepNet、Sebastian Raschka FAQ 和 Naoki Shibuya 博客帮助本站把论文公式落到现代 block 写法、训练稳定性和效率取舍。博客主要学习讲法：先讲一个可见问题，再把公式嵌进去解释，而不是把公式放成孤立符号表。

第四类是低比特与 outlier 来源。LLM.int8()、SmoothQuant、AWQ、FP8 训练报告和低比特综述帮助本站说明：Norm 控制整体尺度并不等于消除所有 outlier；activation distribution、scale granularity、accumulation 和 residual add 都会影响实际质量。

## 写作边界

归一化不能被写成“防止梯度消失”的万能句。BatchNorm、LayerNorm、RMSNorm 的统计轴、训练/推理行为和适用架构都不同。残差连接也不能只写“解决梯度消失”，它更关键的是把层改成学习相对输入的更新量，并给反向传播一个 identity 项。激活函数不能只列公式，必须说明硬门、软门、self-gating 和 GLU-style gating 分别改变哪些通道。

中文材料和微信公众号风格适合学习入口设计、例子节奏和术语表达；强事实仍回论文、官方文档或可复查的公开博客。不能直接稳定访问或版权状态不清的图，不放进正文，只把讲法迁移成自己的解释。

## 来源

| 来源 | 类型 | 本站学什么 | 落到页面 |
| --- | --- | --- | --- |
| [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) | 论文 | degradation problem、residual function、identity shortcut 如何让深层网络更易优化。 | 残差连接、ResNet 图解 |
| [Batch Normalization](https://arxiv.org/abs/1502.03167) | 论文 | batch/channel 统计、训练/推理 running stats、早期深层 CNN 稳定化路径。 | BatchNorm、CNN 归一化 |
| [How Does Batch Normalization Help Optimization?](https://arxiv.org/abs/1805.11604) | 论文 | BatchNorm 不只减少 internal covariate shift，更让优化地形更平滑。 | 归一化机制边界 |
| [Layer Normalization](https://arxiv.org/abs/1607.06450) | 论文 | 单样本 hidden 维统计、训练推理一致性、RNN/Transformer 适配原因。 | LayerNorm 公式 |
| [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) | 论文 | RMSNorm 去掉 mean-centering、保留 re-scaling invariance 和效率收益。 | RMSNorm 公式 |
| [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) | 论文 | Post-LN 初始化梯度、warmup、Pre-LN 稳定性的理论解释。 | Pre-Norm / Post-Norm |
| [Megatron-LM](https://arxiv.org/abs/1909.08053) | 论文 | BERT-like 模型中 LayerNorm/residual 顺序影响训练 loss 曲线。 | Transformer 稳定性图 |
| [DeepNet](https://arxiv.org/abs/2203.00555) | 论文 | 极深 Transformer 仍需要 residual scaling 和初始化配方。 | Pre-Norm 边界 |
| [Rectified Linear Units Improve Restricted Boltzmann Machines](https://www.cs.toronto.edu/~hinton/absps/reluICML.pdf) | 论文 | ReLU 硬门、稀疏激活和深层网络训练突破的早期来源。 | 激活函数基础 |
| [Gaussian Error Linear Units](https://arxiv.org/abs/1606.08415) | 论文 | GELU 是 \(x\Phi(x)\)，用输入值软权重而非按正负硬切。 | GELU 公式 |
| [Searching for Activation Functions](https://arxiv.org/abs/1710.05941) | 论文 | Swish/SiLU 的 self-gating 形状和 \(\beta\) 参数。 | SiLU / Swish |
| [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) | 论文 | ReGLU、GEGLU、SwiGLU 是 FFN 门控结构，而非单变量曲线。 | SwiGLU |
| [LLaMA](https://arxiv.org/abs/2302.13971) | 论文 | RMSNorm、SwiGLU、RoPE 等现代 decoder block 组合。 | 现代 LLM block |
| [Sebastian Raschka RMSNorm FAQ](https://sebastianraschka.com/faq/docs/rmsnorm-vs-layernorm.html) | 技术博客 | 用简洁工程语言解释 RMSNorm 为什么在 LLaMA-style 模型里流行。 | RMSNorm 讲法 |
| [Naoki Shibuya: SwiGLU](https://naokishibuya.github.io/blog/2023-04-30-swiglu-2020/index.html) | 技术博客 | 从 ReLU/GELU/Swish 到 GLU variants 的逐步展开和图示方式。 | 激活曲线、SwiGLU |
| [Hugging Face 中文 ResNet 课程](https://hugging-face.cn/learn/computer-vision-course/unit2/cnns/resnet) | 中文课程 | 用读者熟悉的 ResNet 入口解释残差网络，不把论文问题讲成术语堆叠。 | 残差入口 |
| [Datawhale RMSNorm Tutorial](https://datawhalechina.github.io/llm-algo-leetcode/02_PyTorch_Algorithms/01_RMSNorm_Tutorial.html) | 中文工程教程 | 用公式、PyTorch 实现和混合精度溢出解释 RMSNorm，不把它写成孤立术语。 | RMSNorm 中文讲法 |
| [Datawhale SwiGLU Activation](https://datawhalechina.github.io/llm-algo-leetcode/02_PyTorch_Algorithms/02_SwiGLU_Activation.html) | 中文工程教程 | 从 GLU 双分支、参数量对齐和矩阵融合解释 SwiGLU。 | SwiGLU 中文讲法 |
| [LLM.int8()](https://arxiv.org/abs/2208.07339) | 论文 | 大模型少数 activation outlier 如何破坏朴素 INT8 矩阵乘。 | activation outlier |
| [SmoothQuant](https://arxiv.org/abs/2211.10438) | 论文 | 通过等价缩放把 activation outlier 难度迁移到权重侧。 | W8A8、低比特 |
| [AWQ](https://arxiv.org/abs/2306.00978) | 论文 | activation-aware 的权重通道保护视角。 | 量化敏感通道 |
