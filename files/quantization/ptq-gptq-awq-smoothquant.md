# GPTQ、AWQ 与 SmoothQuant 怎样减少量化误差

最直接的训练后量化，是给训练好的权重选定格子，再逐个舍入。它能马上减小表示范围，但相同的权重误差会被输入放大、抵消或沿后续层传播。本页从一个线性层出发，解释为什么 GPTQ 要补偿其他权重，AWQ 要看激活，而 SmoothQuant 会同时改动归一化和线性层。

下文统一用 $Y=XW$，X 的每行是一条输入，W 的每列是一个输出通道。源码里的 `nn.Linear.weight` 通常是 W 的转置。

## 先看逐个舍入漏掉了什么

假设输入的两个通道在校准样本里总相等，$x=[u,u]$，权重为 $w=[0.6,0.6]^\top$。原输出为 $1.2u$。如果整数格子间隔为 1，独立舍入得到 $[1,1]$，输出变成 $2u$。

现在先把第一个权重量化成 1，再允许第二个尚未量化的权重补偿。要暂时保持总和 1.2，第二个权重应改成 0.2；最终量化成 0，输出为 u。误差从 $0.8u$ 降到 $-0.2u$。这个二维教学例说明，按输入相关性补偿，比把各个权重孤立地看待更有希望保住输出。

它也暴露了校准边界。若真实输入变为 $[u,-u]$，这组补偿就未必合适。量化算法优化的是它看到的输入分布。

<!-- systems-workflow-visual -->
```mermaid
flowchart LR
  W["w=0.6,0.6 · 输入u,u"] --> N["独立round得到1,1"] --> E["输出2u · 误差0.8u"]
  W --> A["先将首权重量化成1"] --> B["补偿第二权重到0.2"] --> Q["再round成0"] --> O["输出u · 误差-0.2u"]
```

补偿利用了当前输入通道的相关性。换一种输入关系，较小误差的结论也可能改变。

## GPTQ 先量化一列，再修正剩余列

GPTQ 针对层输出重建目标

$$
\min_{\hat W}\|XW-X\hat W\|_F^2.
$$

若只看一个输出通道，其二阶曲率与 $X^\top X$ 有关。相关输入允许一个方向的量化偏差由另一个方向部分补偿。原始 `gptq.py` 的 `add_batch` 累积输入统计；`fasterquant` 加入对角阻尼，计算逆矩阵相关分解，再按列量化并更新未处理的权重。

代码里的 `Hinv` 名称尤其容易误读。完成两次 Cholesky 操作之后，这个变量存的是逆矩阵的上三角 Cholesky 因子，不能把循环中的每个除法直接解释成“除以 Hessian 对角”。分块处理用于组织补偿计算，group size 则决定多大一组权重共用量化参数，两者也要分开。

![GPTQ procedure 原论文图](../assets/images/paper-figures/quantization/gptq-figure-2-procedure.png){ width="760" }

<small>图源：[GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)，Figure 2。原图表达 GPTQ 按列量化权重，并用近似 Hessian 信息更新尚未量化的列；这里的重点是 GPTQ 的重点是层输出重构，而不是单个权重逐个 round。</small>

校准输入有完全相关的通道时，$X^\top X$ 可能奇异。阻尼能改善数值问题，也会改变所求近似。上面的二维例只说明补偿直觉，没有模拟源码的阻尼、分块与所有量化细节。

## AWQ 用激活判断哪些权重误差更昂贵

对某个输入通道 i，输出误差包含 $x_i E_{i,:}$。一个权重行本身不大，但若对应激活经常很大，舍入这一行仍会产生显著误差。AWQ 利用校准激活搜索通道缩放，并在支持的位置把等价变换吸收到相邻算子。

设正对角矩阵 D 为通道尺度，先写成 $XW=(XD^{-1})(DW)$，再量化 DW。放大某个权重通道，有机会让它相对量化格距获得更细的有效表示；相应输入必须反向缩小。由于同组其他权重及 scale 也会改变，放大并不自动改善结果，需要实际计算层或块的输出误差。

AWQ 的 `pre_quant.py` 会注册 hook 收集线性层输入，再调用缩放与截断搜索，并把结果应用到网络。随后 `quantizer.py` 明确区分 `pseudo_quantize_model_weight` 和 `real_quantize_model_weight`。前者用浮点张量模拟舍入误差，后者构造可用于真实低比特执行的 `WQLinear`。

AWQ 常见实现仍将相关权重打包到低比特格式。“保护重要通道”不应被理解为必然留下若干 FP16 权重。具体是否混用精度，取决于实现和配置。

![AWQ activation-aware 原论文图](../assets/images/paper-figures/quantization/awq-figure-1-activation-aware.png){ width="900" }

<small>图源：[AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978)，Figure 1。原图表达少量显著 activation 通道对应的权重更需要保护；这里的重点是 AWQ 的关键词是 activation-aware，重要性来自运行时输入如何使用这些通道，而不只来自权重绝对值。</small>

## SmoothQuant 把动态激活的范围转移给静态权重

如果还希望激活也进入低比特矩阵乘，动态激活离群值会使每次请求的 scale 难以选择。SmoothQuant 使用同样的等价缩放，但重点是让激活更容易量化。

在本页方向下，把输入通道除以 $s_i$、对应权重行乘以 $s_i$。源码按激活通道最大幅度 a 与权重通道最大幅度 w 构造近似为

$$
s_i=\frac{a_i^\alpha}{w_i^{1-\alpha}},
$$

并带有防零截断。增大 $\alpha$ 通常向激活侧分配更多平滑力度，也会让权重侧承担更大范围。`smooth_ln_fcs` 通过缩放 LayerNorm 的 weight/bias 和下游 Linear 权重完成折叠；Llama 类 RMSNorm 路径则没有同样的 bias。

若归一化输出同时送给 Q、K、V 三个投影，所有消费同一输入的分支都要保持等价。只改一条分支，整个网络就变了。实际支持哪些模型结构，需要看该版本的结构分支。

![SmoothQuant intuition 原论文图](../assets/images/paper-figures/quantization/smoothquant-figure-2-intuition.png){ width="760" }

<small>图源：[SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438)，Figure 2。原图表达 activation outlier 会拉大量化范围，使主体值失去分辨率；SmoothQuant 通过等价缩放把部分难度迁移到 weight。</small>

## 怎样从这三种方法走到可用模型

先固定浮点参考模型、tokenizer、输入和输出比较方式。再用独立校准集估计统计量、搜索缩放或补偿，得到模拟量化结果。只有这一步的误差可接受，才值得打包并送进目标 runtime，比较模拟结果和实际 kernel 是否对齐。

GPTQ、AWQ、SmoothQuant 描述的是处理误差的办法，W4A16、W8A8 描述的是数值路径，两组名字不能互相替代。同样的算法可以有不同 group size、scale 格式和打包布局。后续的 [运行时](quantization-runtimes-and-frameworks.md)会继续追踪一个 packed byte 怎样参与乘法。


## 对照源码与延伸阅读

- [GPTQ add_batch 与 fasterquant](https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/gptq.py)
- [AWQ 激活收集、缩放与截断](https://github.com/mit-han-lab/llm-awq/blob/d6e797a42b9ef7778de8ee2352116e0f48a78d61/awq/quantize/pre_quant.py)
- [SmoothQuant 等价变换](https://github.com/mit-han-lab/smoothquant/blob/c61476d728e42ae0d8a35e7e78494edcac3237b5/smoothquant/smooth.py)
