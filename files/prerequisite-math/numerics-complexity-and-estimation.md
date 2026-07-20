# 数值计算、复杂度与估算

论文里的短公式进入训练和推理系统后，会变成 FLOPs、bytes、精度、稳定性和复杂度问题。数值计算和复杂度把“公式能不能算、算得准不准、算得贵不贵”说清楚。

## Big-O：看规模增长趋势

复杂度符号 \(O(\cdot)\) 描述输入规模变大时成本怎样增长，不给出精确时间。

![Big-O complexity](../assets/images/prerequisite-math/big-o-complexity.svg){ width="700" }

<small>图源：[Wikimedia Commons: Comparison computational complexity](https://commons.wikimedia.org/wiki/File:Comparison_computational_complexity.svg)。原图比较不同复杂度曲线的增长速度。这里的重点是：线性、平方、指数复杂度在小规模时差别不明显，规模一大就会决定系统是否可用。</small>

attention score 的显存常粗略写成：

\[
O(BHL^2)
\]

这里 \(B\) 是 batch，\(H\) 是 head 数，\(L\) 是序列长度。平方项来自所有 query 和 key 两两比较。长上下文、多图、多帧视频一旦把 \(L\) 拉大，成本会非常快地上升。

如果 \(L\) 从 4k 增到 8k，\(L^2\) 会变成 4 倍，不是翻倍。这个数量级判断解释了为什么长上下文系统要做 FlashAttention、block attention、KV 压缩和上下文分层：瓶颈在于平方增长太快。

## FLOPs：算了多少乘加

FLOPs 衡量浮点运算次数。一个矩阵乘：

\[
C=AB,\quad
A\in\mathbb{R}^{M\times K},
B\in\mathbb{R}^{K\times N}
\]

大约需要：

\[
2MKN
\]

次浮点操作。训练和推理系统里，GEMM、attention、MLP、卷积的成本常常先从 FLOPs 估算开始。

为什么是 \(2MKN\)：输出矩阵有 \(M\times N\) 个元素，每个元素要做长度为 \(K\) 的点积，约 \(K\) 次乘法和 \(K\) 次加法，所以是 \(2K\)。合起来就是 \(2MKN\)。后面看 MLP 或 GEMM benchmark 时，\(M,N,K\) 对应 batch/token 数、输入通道和输出通道。

但 FLOPs 不等于速度。一个算子可能 FLOPs 不高，却因为读写太多数据而慢。

## Bytes：数据搬运也要付钱

显存和带宽常用 bytes 估算。一个 hidden tensor：

\[
\text{bytes}
=
B\cdot L\cdot D\cdot \text{bytes per element}
\]

BF16/FP16 通常是 2 bytes，FP32 是 4 bytes，FP8 是 1 byte，INT4 是半个 byte。量化能省存储和带宽，但会引入 scale、dequant、误差和 kernel 支持问题。

举个数：`B=1`、`L=8192`、`D=4096`、BF16 时，一个 hidden tensor 大约是

\[
1\times8192\times4096\times2
\approx 64\text{MiB}
\]

这还只是一个张量。训练时同一层可能还要保存 Q/K/V、MLP 中间激活、norm 统计和 backward 需要的缓存。

推理服务里，KV cache 的 bytes 往往比权重更容易成为长上下文瓶颈：

\[
\text{KV bytes}
\approx
2\cdot \text{layers}\cdot B\cdot L\cdot D_{\text{kv}}\cdot \text{bytes}
\]

前面的 2 表示 K 和 V。

如果层数 32、\(B=1\)、\(L=8192\)、\(D_{\text{kv}}=1024\)、BF16 为 2 bytes，那么 KV cache 大约是：

\[
2\times32\times1\times8192\times1024\times2
\approx 1\text{GiB}
\]

这解释了为什么“权重量化后模型变小”并不自动解决长上下文服务：请求越长，KV cache 越像主角。

## 浮点误差：机器数不是实数

浮点数用有限 bit 表示实数，因此会有舍入误差、上溢、下溢和精度损失。softmax 如果直接算：

\[
\frac{e^{s_i}}{\sum_j e^{s_j}}
\]

当 \(s_i\) 很大时，指数可能溢出。稳定写法通常会先减最大值：

\[
\mathrm{softmax}(s_i)
=
\frac{e^{s_i-\max(s)}}{\sum_j e^{s_j-\max(s)}}
\]

这不改变结果，却能避免数值爆炸。

原因是所有 logit 同时减去同一个常数，softmax 比值不变：

\[
\frac{e^{s_i-c}}{\sum_j e^{s_j-c}}
=
\frac{e^{s_i}/e^c}{\sum_j e^{s_j}/e^c}
=
\frac{e^{s_i}}{\sum_j e^{s_j}}
\]

选择 \(c=\max(s)\) 可以让最大指数项变成 \(e^0=1\)，避免 \(e^{1000}\) 这类溢出。

## 条件数：输入小误差会不会被放大

有些计算对输入误差很敏感。条件数可以理解为“输入误差被输出放大的倍数”。矩阵求逆、线性方程求解、某些归一化和低精度累加都会遇到类似问题。

在大模型里，数值稳定性会体现在：

| 场景 | 常见问题 |
| --- | --- |
| FP16/BF16 训练 | 梯度下溢、loss scaling |
| FP8 训练 | scale 选择、amax 历史、累加精度 |
| softmax / attention | 大 logits 溢出、mask 处理错误 |
| normalization | 方差太小、epsilon 选择 |
| 量化推理 | outlier channel、校准集不代表真实请求 |

## 估算：先算数量级，再跑 benchmark

一个有用的工程习惯是先做数量级估算。例如多相机视频输入：

\[
L=Cams\times Frames\times Patches
\]

如果 4 个相机、16 帧、每帧 256 个 patch：

\[
L=4\times16\times256=16384
\]

这时 full attention 的 \(L^2\) 成本已经非常高。你不用等模型 OOM 才知道风险，先算一遍就能决定是否需要 resampler、局部 attention、token pruning 或 memory hierarchy。

## 常见误读

| 误读 | 更稳的理解 |
| --- | --- |
| Big-O 大就一定慢 | 常数、硬件、实现和真实 shape 也重要 |
| FLOPs 少就一定快 | 可能被内存带宽、launch、通信或 layout 卡住 |
| 低精度只是换 dtype | scale、累加、校准和 kernel 都会影响质量和速度 |
| benchmark 平均值够了 | P95/P99、长尾 shape 和端到端指标同样关键 |

## 读完以后怎么判断

看到一个系统优化 claim，先问它省的是 FLOPs、bytes、通信还是 latency；再问质量指标有没有回归；最后问 benchmark 是否覆盖真实 shape 分布。这个判断会贯穿推理服务、量化、算子、长上下文和世界模型高效训练。
