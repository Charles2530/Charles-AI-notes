# 数值计算、复杂度与估算

给定一次 `[1024,4096]×[4096,4096]` 的矩阵乘法，先估计乘加次数与数据字节数，再判断硬件可能受计算还是带宽限制。这样，Big-O、FLOPs、算术强度和运行时间就都对应同一个工作负载。

这些计算是理想化估算。实现还会重复加载、使用缓存、分配工作区并调度线程，最后需要测量；但先写清单位与规模，能排除明显不可能的性能解释。

## Big-O：看规模增长趋势

复杂度符号 \(O(\cdot)\) 描述输入规模变大时成本怎样增长，不给出精确时间。

![Big-O complexity](../assets/images/prerequisite-math/big-o-complexity.svg){ width="700" }

<small>图源：[Wikimedia Commons: Comparison computational complexity](https://commons.wikimedia.org/wiki/File:Comparison_computational_complexity.svg)。原图比较不同复杂度曲线的增长速度。这里的重点是：线性、平方、指数复杂度在小规模时差别不明显，规模一大就会决定系统是否可用。</small>

attention score 的显存常粗略写成：

\[
O(BHL^2)
\]

这里 \(B\) 是 batch，\(H\) 是 head 数，\(L\) 是序列长度。平方项来自所有 query 和 key 两两比较。长上下文、多图、多帧视频一旦把 \(L\) 拉大，成本会非常快地上升。

如果 \(L\) 从 4k 增到 8k，\(L^2\) 会变成 4 倍，不是翻倍。这个数量级判断需要与实现分开。标准稠密attention的算术量仍随长度平方增长；FlashAttention通过分块避免把完整分数矩阵写入高带宽显存，降低中间存储和IO，不会自动把稠密attention的全部算术变成线性。局部或稀疏attention则通过减少连接改变计算集合，通常也改变模型可见的信息。

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

举一个教学shape，\(M=1024,K=N=4096\)，矩阵乘约需34.36 GFLOPs。若A、B、C都用2字节存储，且理想情况下各读写一次，数据量为

\[
2(MK+KN+MN)=50{,}331{,}648\ \mathrm{bytes}=48\ \mathrm{MiB}.
\]

算术强度 \(I=\mathrm{FLOPs}/\mathrm{bytes}\) 约为682.7 FLOP/byte。换成单行 \(M=1\)，运算量约33.55 MFLOPs，而权重仍需读约32 MiB，算术强度接近1。批量越大，权重越有机会被多个输入复用，这解释了单token decode与大批prefill为什么可能受不同瓶颈限制。

假设一台虚构设备可持续达到100 TFLOP/s和1 TB/s，理想时间下界为

\[
t\ge\max\left(\frac{\mathrm{FLOPs}}{100\times10^{12}},
\frac{\mathrm{bytes}}{10^{12}}\right).
\]

大矩阵的两项约为0.344 ms和0.0503 ms，算力项占主导；单行的两项约为0.000336 ms和0.0336 ms，带宽项占主导。这是roofline数量级估算，不是实测延迟。缓存、转置、额外读写、kernel启动、通信和利用率都可能增加时间，实际权重缓存命中还会改变外存流量。

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

对有限logits，这不改变实数运算下的结果，并避免指数的正向上溢。它不保证整个模型数值稳定。

原因是所有 logit 同时减去同一个常数，softmax 比值不变：

\[
\frac{e^{s_i-c}}{\sum_j e^{s_j-c}}
=
\frac{e^{s_i}/e^c}{\sum_j e^{s_j}/e^c}
=
\frac{e^{s_i}}{\sum_j e^{s_j}}
\]

选择 \(c=\max(s)\) 可以让最大指数项变成1。例如 \((1000,999)\) 先减1000，变成 \((0,-1)\)，softmax约为 \((0.731059,0.268941)\)。如果一整行都被mask成负无穷，减最大值会产生未定义的“负无穷减负无穷”；必须按算子约定处理全mask行，不能指望减最大值解决。

相近数字相减也可能丢失有效位。计算很小 \(x\) 的 \(e^x-1\)，应考虑专门的 `expm1(x)`；计算 \(\log(1+x)\) 可用 `log1p(x)`。这类函数通过适当实现减少消去误差，不能靠最后转成高精度补回已经丢掉的位。

## 条件数：输入小误差会不会被放大

线性系统 \(Ax=b\) 的二范数条件数为 \(\kappa_2(A)=\|A\|_2\|A^{-1}\|_2\)。在可逆及小扰动条件下，它给出右端项相对误差可能被放大的上界；它描述问题本身的敏感性，算法稳定性则描述实现额外引入多少误差。

取

\[
A=\begin{bmatrix}1&1\\1&1.0001\end{bmatrix},\quad
b=\begin{bmatrix}2\\2.0001\end{bmatrix}.
\]

两行相减可得 \(0.0001x_2=0.0001\)，解为 \((1,1)\)。只把 \(b_2\) 从2.0001改成2.0002，解就变成 \((0,2)\)。输入的绝对变化只有0.0001，解却明显变化；两条方程几乎重复，因而很难稳定区分两个未知量。即使求解器实现完全正确，这种敏感性仍然存在。

在大模型里，数值稳定性会体现在：

| 场景 | 常见问题 |
| --- | --- |
| FP16训练 | 指数范围较小，可能溢出或梯度下溢，动态loss scaling常用于改善后者 |
| BF16训练 | 指数范围接近FP32、有效精度更低，仍需关注舍入和累加误差 |
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

## 为什么批次变大会改变同一权重的利用率

固定K、N，M从1增加时，乘加量与M成正比，权重矩阵的原始大小却不变。理想情况下，读入一块权重可用于更多输入行，因此每搬一个字节完成的计算增加。M很小时，这种复用不足，峰值FLOP/s可能完全用不上。

这个推理只说明趋势。tile划分、缓存容量、实际读取次数和输出写回决定真实强度。把矩阵数学上只出现一次理解为硬件只读一次，会低估流量；把每次乘法都按从HBM重读，又会高估流量。

## 从单位检查一个估算

```python
M, K, N = 1024, 4096, 4096
flops = 2*M*K*N
traffic = 2*(M*K + K*N + M*N)
print(flops/1e9, traffic/2**20, flops/traffic)
print(max(flops/100e12, traffic/1e12)*1e3)
kv = 2*32*1*8192*1024*2
print(kv/2**30)
```

输出依次给出约34.36 GFLOPs、48 MiB、682.7 FLOP/byte、理想下界0.344 ms和1 GiB KV。GB使用 \(10^9\) 字节，GiB使用 \(2^{30}\) 字节；把两种单位混用会制造约7%的容量差异。上述数值没有执行GPU计算，也没有包含分配器、padding、量化scale和临时workspace。


- [GPU Puzzles 的索引与访存示例](https://github.com/srush/GPU-Puzzles/blob/b3c4b237d7f0dc6d82055b753c8ea6e0cbb845eb/GPU_puzzlers.py)，逐项追踪哪些线程读写哪些元素，避免只数算术运算。
