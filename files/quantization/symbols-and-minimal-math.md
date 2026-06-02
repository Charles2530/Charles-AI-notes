# 量化符号与最小数学地图

这页不是数学考试，而是读量化教程和论文时随手查的地图。量化公式大多在回答三件事：


```text
原来的数是什么 -> 用低比特怎么近似 -> 这个近似会不会伤模型输出
```

!!! note "初学者先抓住"
    看到公式时先问：这个符号是权重、激活还是 KV cache；scale 是一整层共用还是每组共用；误差是在单个数上看，还是在矩阵乘输出上看。

## 最常见的量化公式

<!-- [融合：算法基础] 来源：llm-algo-leetcode《01. Data Types and Precision | 大模型的数据格式与混合精度 (Data Types & Precision)》-->
> **算法视角**：量化可以看成在权重或激活旁边维护 scale / zero-point，把高精度矩阵乘改写成“低比特存储 + 反量化或融合计算”的数据布局问题。
>
> 延伸阅读：[01. Data Types and Precision | 大模型的数据格式与混合精度 (Data Types & Precision)](https://datawhalechina.github.io/llm-algo-leetcode/01_Hardware_Math_and_Systems/01_Data_Types_and_Precision.html)

<!-- [融合：复杂度分析] 来源：llm-algo-leetcode《01. Data Types and Precision | 大模型的数据格式与混合精度 (Data Types & Precision)》-->
| 操作 | 时间复杂度 | 空间复杂度 | 说明 |
|------|-----------|-----------|------|
| 逐元素量化 / 反量化 | $O(N)$ | $O(N)$ 或按 block 存 scale | 关键收益通常来自带宽和存储减少，而不是渐进复杂度下降；参考：[对应章节](https://datawhalechina.github.io/llm-algo-leetcode/01_Hardware_Math_and_Systems/01_Data_Types_and_Precision.html) |
| 量化 GEMM | $O(MNK)$ | 权重侧约按 bit-width 缩放 | 反量化若能融合进 matmul epilogue，才更容易转成端到端收益。 |

\[
q = \mathrm{clip}\left(\mathrm{round}\left(\frac{x}{s}\right)+z,\ q_{\min},\ q_{\max}\right)
\]


\[
\hat{x}=s(q-z)
\]


**读作什么**：把 \(x\) 除以 scale，变成“第几个格子”；加上 zero-point 后得到整数；如果整数超过低比特范围，就裁到边界；计算时再把整数还原成近似浮点值 \(\hat{x}\)。

**小例子**：若 \(s=0.5,z=0\)，则 \(x=1.2\) 先变成 \(1.2/0.5=2.4\)，round 后 \(q=2\)，反量化 \(\hat{x}=0.5\times2=1.0\)。误差是 \(0.2\)。

!!! warning "常见误解"
    \(q\) 不是模型真正想要的输出，它只是低比特存储或低比特计算中的中间表示。模型质量最终看的是 \(\hat{x}\) 进入矩阵乘、attention 和后续层以后，输出是否仍然接近原模型。

## bit 数和整数范围

如果使用 \(b\) bit，有符号整数通常能表示：

\[
q_{\min}=-2^{b-1},\qquad q_{\max}=2^{b-1}-1
\]


**读作什么**：bit 越少，可用格子越少。INT8 大约有 256 个格子，INT4 只有 16 个格子。格子少不一定不能用，但 scale 和 outlier 会变得更关键。

| 格式 | 有符号整数范围 | 直觉 |
| --- | --- | --- |
| INT8 | \([-128,127]\) | 格子较多，激活量化更常见 |
| INT4 | \([-8,7]\) | 压缩强，常用于 weight-only |
| INT2 | \([-2,1]\) | 极激进，通常需要特殊训练或强保护 |

## scale 怎么选

最简单的对称量化会用最大绝对值决定 scale：

\[
s=\frac{\max(\lvert x\rvert)}{q_{\max}}
\]


**读作什么**：最大的数要刚好放进最大整数格子里，因此 scale 等于“最大真实值 / 最大整数值”。

**离群值例子**：一组激活大多在 \([-2,2]\)，但有一个值是 \(30\)。INT4 的 \(q_{\max}=7\)，scale 会变成 \(30/7\approx4.29\)。这时 \(-2\) 到 \(2\) 的主体区域只有一两个格子可用，细节几乎被抹掉。

!!! note "为什么 activation outlier 麻烦"
    权重是固定的，可以离线分析；激活随每个输入变化。一个少见 token、长上下文位置或图像局部细节，都可能让某层 activation 突然出现大值，迫使 scale 变粗。

## 对称和非对称量化

对称量化常写成：

\[
q=\mathrm{round}\left(\frac{x}{s}\right),\qquad \hat{x}=sq
\]


非对称量化常写成：

\[
q=\mathrm{round}\left(\frac{x}{s}\right)+z,\qquad \hat{x}=s(q-z)
\]


**读作什么**：如果数据天然围绕 0，对称量化够用；如果数据整体偏移，zero-point 可以让整数范围利用得更充分。

## 量化粒度：一个 scale 管多大

设一个激活矩阵：

\[
X\in\mathbb{R}^{B\times T\times d}
\]


不同量化粒度的区别，是 scale 覆盖的范围不同：

| 粒度 | scale 形状直觉 | 优点 | 风险 |
| --- | --- | --- | --- |
| per-tensor | 整个 \(X\) 一个 scale | 元数据最少，简单 | 一个 outlier 影响所有值 |
| per-token | 每个 token 一个 scale | 适合动态激活 | 在线统计和 kernel 更复杂 |
| per-channel | 每个 hidden channel 一个 scale | 能保护通道差异 | scale 读取更多 |
| group-wise | 每组通道或 token 一个 scale | 精度和成本折中 | group size 要和 kernel 匹配 |
| per-element | 每个元素一个 scale | 误差最小 | 元数据通常太贵 |

!!! tip "实用判断"
    scale 越细，误差通常越小；但 scale 本身也要存、要读、要参与 dequant。量化收益不是只看 \(q\) 变小，也要看 scale 和 dequant 是否拖慢热路径。

## 线性层里误差看什么

原始线性层：

\[
Y=XW
\]


量化后近似为：

\[
\hat{Y}=\hat{X}\hat{W}
\]


误差可以粗略写成：

\[
\Delta Y = Y-\hat{Y}
\]


**读作什么**：真正重要的不是每个权重是否近似，而是量化后的矩阵乘输出是否仍然接近原输出。GPTQ、AWQ 和 SmoothQuant 都是在用不同方式控制这个输出误差。

## GPTQ 里的二阶直觉

GPTQ 常用下面的目标来描述一层权重量化：

\[
\min_{\hat{W}}\ \|WX-\hat{W}X\|_2^2
\]


**读作什么**：不要只让 \(\hat{W}\) 像 \(W\)，而是让它们乘上真实激活 \(X\) 以后输出尽量像。Hessian 或近似二阶信息用来判断哪些方向更敏感。

!!! note "初学者够用理解"
    二阶信息不是神秘魔法。它大致是在说：同样大小的权重误差，落在不同方向上，对输出和 loss 的伤害不同。GPTQ 尝试在量化一部分权重后，补偿剩余权重，减少输出变化。

## AWQ 里的激活敏感通道

AWQ 的核心直觉可以写成：

\[
Y=\sum_i X_i W_i
\]


**读作什么**：有些权重本身看起来不大，但它所在通道的激活经常很大，最终对输出很重要。AWQ 用校准激活找这些通道，并在量化前给它们更好的保护。

## SmoothQuant 的重参数化

线性层输出可以写成：

\[
Y=XW
\]


SmoothQuant 使用一个可合并的对角缩放矩阵 \(D\)：

\[
Y=(XD^{-1})(DW)
\]


**读作什么**：输出没有变，但难量化的 activation outlier 被压平了一部分；代价转移到更容易离线处理的权重上。

**直觉例子**：搬家时不是把所有箱子都变轻，而是把最难搬、最容易挡路的箱子重新分配到更好处理的位置。

## LoRA 和 QLoRA 的符号

LoRA 把权重更新写成：

\[
W' = W+\Delta W,\qquad \Delta W=BA
\]


QLoRA 的关键是：

\[
W_{\text{base}}\approx \hat{W}_{\text{4bit}},\qquad W'= \hat{W}_{\text{4bit}} + BA
\]


**读作什么**：底座权重用 4bit 存储并冻结，训练时只更新小的 LoRA adapter。它主要省微调显存，不等于所有计算都在 4bit 里完成。

## KV cache 显存公式

自回归生成中，每层都要缓存历史 token 的 Key 和 Value。一个常见估算：

\[
\mathrm{KV\ Memory}\approx 2 \times L \times B \times T \times H_{\text{kv}} \times d_h \times \mathrm{bytes}
\]


**读作什么**：KV cache 随层数、并发、上下文长度、KV head 数和 dtype 线性增长。把 BF16 KV 改成 INT8，理论上 bytes 减半，但还要看 scale、dequant 和长程质量。

**小账**：若 \(L=32,B=8,T=16384,H_{\text{kv}}d_h=4096,\mathrm{bytes}=2\)，则 KV 本体约：

\[
2\times32\times8\times16384\times4096\times2\approx64\mathrm{GiB}
\]


这解释了为什么长上下文服务里，权重已经 4bit 后，显存仍可能被 KV cache 吃满。

## 读量化公式的四句检查

1. 这个公式量化的是 \(W\)、\(X\)、\(K/V\)，还是训练状态？
2. scale 是 per-tensor、per-channel、per-token，还是 group-wise？
3. 误差是在单个数上看，还是在 \(XW\)、attention 或最终任务上看？
4. 低比特路径是否真的由目标 runtime 和 kernel 执行？

如果这四句答不清，先不要比较哪个方法更强。量化里最容易迷路的地方，就是把数值格式、误差控制算法和系统执行路径混成一件事。
