# 线性层到 GEMM

如果把现代大模型拆到最底层，会发现大量计算都在做矩阵乘。但这里要先分清三层概念：`Linear` 是模型层，`MatMul` 是数学/框架里的矩阵乘操作，`GEMM` 是底层线性代数库和 GPU kernel 常用的 dense matrix-matrix multiplication 形式。它们相关，但不是同一个词。


![Triton roofline comparison 原论文图](../assets/images/paper-figures/operators/triton-roofline-comparison.png){ width="490" }

<small>图源：[Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations](https://doi.org/10.1145/3315508.3329973)，Figure 1。原论文图意：在 \(C=AB^T\) 矩阵乘上比较 cuBLAS、Triton、Auto-TVM、Tensor Comprehensions 和 PlaidML 相对 roofline model 的性能位置。</small>

## 先看线性层公式

线性层最常见的公式是：

\[
y=xW+b
\]


如果：

\[
x\in\mathbb{R}^{B\times L\times D},\quad W\in\mathbb{R}^{D\times D_{\text{out}}}
\]


这行式子在说：输入 \(x\) 有 batch 维 \(B\)、序列维 \(L\)、hidden 维 \(D\)；权重 \(W\) 把 \(D\) 维输入映射到 \(D_{\text{out}}\) 维输出。

那么：

\[
y\in\mathbb{R}^{B\times L\times D_{\text{out}}}
\]


读作：batch 和 token 数不变，每个 token 的向量从 \(D\) 维被投影到 \(D_{\text{out}}\) 维。

## 再分清 MatMul 和 GEMM

<!-- [融合：算法基础] 来源：llm-algo-leetcode《04. Triton GEMM Tutorial | Triton 矩阵乘法 (GEMM) 与自动调优 (Autotune)》-->
> **算法视角**：算子优化先要还原成 GEMM、reduce 或 elementwise 的块级数据流；Triton/CUDA 章节里的 tile、mask、program id 和 autotune，都是在控制同一数学操作的访存复用和并行粒度。
>
> 延伸阅读：[04. Triton GEMM Tutorial | Triton 矩阵乘法 (GEMM) 与自动调优 (Autotune)](https://datawhalechina.github.io/llm-algo-leetcode/03_CUDA_and_Triton_Kernels/04_Triton_GEMM_Tutorial.html)

<!-- [融合：复杂度分析] 来源：llm-algo-leetcode《04. Triton GEMM Tutorial | Triton 矩阵乘法 (GEMM) 与自动调优 (Autotune)》-->
| 操作 | 时间复杂度 | 空间复杂度 | 说明 |
|------|-----------|-----------|------|
| GEMM $C=A B$ | $O(MNK)$ | $O(MK+KN+MN)$ | tile 的意义是提高 SRAM / register 复用，减少 HBM 往返；参考：[对应章节](https://datawhalechina.github.io/llm-algo-leetcode/03_CUDA_and_Triton_Kernels/04_Triton_GEMM_Tutorial.html) |
| reduce / norm / softmax | $O(N)$ | $O(N)$ 或一行工作集 | 这类算子常是 bandwidth-bound，融合可以减少读写次数。 |

`MatMul` 更像数学操作或框架算子名：给定两个张量，按最后两个维度做矩阵乘，可能带 batch 维、broadcast、transpose 或不同 backend lowering。

`GEMM` 是 BLAS 语境里的 General Matrix-Matrix Multiplication，标准形式更接近：

\[
C \leftarrow \alpha A B + \beta C
\]


其中 \(A\)、\(B\)、\(C\) 是二维 dense matrices，\(\alpha\)、\(\beta\) 是标量。深度学习里的 bias、activation、dequant、requant、residual 等，通常属于 GEMM kernel 的 **epilogue fusion** 或周边融合，不是 GEMM 这个数学定义本身。

二者关系可以这样记：

| 概念 | 所在层级 | 典型含义 |
| --- | --- | --- |
| `Linear` | 模型层 | \(y=xW+b\)，把 hidden 表示投影到新维度 |
| `MatMul` | 数学/API 层 | 两个矩阵或 batched tensors 相乘 |
| `GEMM` | 库/kernel 层 | 高性能 dense 矩阵-矩阵乘，常见形式是 \(C \leftarrow \alpha AB+\beta C\) |
| `BMM` / batched GEMM | 库/kernel 层 | 一批小矩阵乘，常见于 attention 的 QK/PV 路径 |
| fused GEMM | 工程实现层 | GEMM 主计算后融合 bias、activation、scale、quant/dequant 等 epilogue |

因此，**不是所有 MatMul 都是 GEMM**：向量乘、稀疏乘法、带复杂 broadcast 的高维张量乘、attention 的 fused kernel、某些 tiny shape 的专用 kernel，都可能不走普通 GEMM 路径。反过来，很多 `Linear` 层虽然代码里写的是 `matmul`，运行时会被 reshape、pack、dispatch 到 GEMM 或 fused GEMM kernel。

## Linear 层在做什么

线性层最常见的形式是：

\[
y = xW + b
\]


如果输入是：

\[
x \in \mathbb{R}^{B \times L \times D}
\]


权重是：

\[
W \in \mathbb{R}^{D \times D_{\text{out}}}
\]


输出就是：

\[
y \in \mathbb{R}^{B \times L \times D_{\text{out}}}
\]


直觉上，Linear 层是在问：当前每个 token 的 \(D\) 维表示，应该被重新组合成什么新的 \(D_{\text{out}}\) 维表示。

!!! note "实现细节：权重矩阵方向可能和公式不同"
    上面的公式用的是教学里常见的 \(xW+b\) 写法，所以 \(W\) 写成 \(D \times D_{\text{out}}\)。PyTorch 的 `nn.Linear(in_features, out_features)` 通常把权重存成 `[out_features, in_features]`，前向等价于 \(xW^\top+b\)。这只是存储布局和 API 约定不同，不改变 Linear、MatMul、GEMM 的层级关系。

## 为什么 Transformer 到处都是 Linear

一个 Transformer block 里常见的 Linear 包括：

| 位置 | 作用 | 典型维度变化 |
| --- | --- | --- |
| Q 投影 | 生成 query | \(D \rightarrow D\) |
| K 投影 | 生成 key | \(D \rightarrow D\) |
| V 投影 | 生成 value | \(D \rightarrow D\) |
| Attention 输出投影 | 合并多头结果 | \(D \rightarrow D\) |
| MLP 升维 | 扩张中间表达 | \(D \rightarrow 4D\) 或更大 |
| MLP 降维 | 压回 hidden size | \(4D \rightarrow D\) |

所以当你听到“优化 Transformer 性能”时，很多时候其实是在优化这些 Linear 对应的 MatMul/GEMM 执行路径，而不是在改 Linear 的数学定义。

## MLP 为什么通常先升维再降维

Transformer 里的 MLP 常写成：

\[
\text{MLP}(x)=W_2 \sigma(W_1x)
\]


其中 \(W_1\) 把维度升高，激活函数 \(\sigma\) 引入非线性，\(W_2\) 再把维度压回去。

这像一个“临时扩展工作台”：

1. 先把表示展开到更大的中间空间；
2. 在中间空间里做非线性组合；
3. 再把结果压回模型主干维度。

如果没有 MLP，Transformer 只有 attention 的信息混合能力，逐 token 的非线性变换会弱很多。

## Linear 如何落到 GEMM

以一个 `Linear` 为例：

\[
x \in \mathbb{R}^{B \times L \times D},\qquad W \in \mathbb{R}^{D \times D_{\text{out}}}
\]


这里 \(x\) 仍是三维输入张量，\(W\) 是二维权重矩阵；运行时通常不会直接把三维形状交给普通 GEMM，而会先展平。

框架常把前两维合并：

\[
X_{\text{flat}} \in \mathbb{R}^{(B L) \times D}
\]


这行式子在说：把 \(B\) 个 batch 和每个 batch 的 \(L\) 个 token 合并成 \(BL\) 行，每一行仍是一个 \(D\) 维 token 表示。

再执行：

\[
Y_{\text{flat}} = X_{\text{flat}} W + b,\qquad
Y_{\text{flat}} \in \mathbb{R}^{(B L) \times D_{\text{out}}}
\]


这行式子就是二维 GEMM 视角下的 Linear：左边有 \(BL\) 行输入 token，右边权重把每行从 \(D\) 维投影到 \(D_{\text{out}}\) 维，bias \(b\) 再加到输出维度上。

最后 reshape 回：

\[
y \in \mathbb{R}^{B \times L \times D_{\text{out}}}
\]


这个二维 dense matrix multiplication 很适合落到 GEMM。若 \(B L\) 很小，例如 decode 阶段每次只来少量 token，它仍然可能走 GEMM family，但性能画像会变成小 \(M\)、skinny GEMM、batched GEMM 或专用 kernel 问题。

## GEMM 为什么是硬件热路径

GPU、TPU、NPU 等加速器都非常擅长规则 dense 矩阵-矩阵乘。大模型性能优化里，很多问题最终会变成：

1. 矩阵形状是否适合硬件 tile；
2. 数据是否连续；
3. dtype 是否命中 Tensor Core 或专用低精度路径；
4. 是否能把小矩阵合并成更大的 batch GEMM；
5. 量化后是否仍有对应 INT4/FP8/FP4 kernel。

这也是为什么 [算子与编译器](../operators/index.md) 章节会反复讨论 GEMM、layout、tile 和 kernel。

## 一个最小伪代码

```text
function Linear(x, W, b):
    # x: [B, L, D]
    # W: [D, D_out]
    # y: [B, L, D_out]
    # API 层写作 matmul；runtime 可能 lower 到 GEMM/fused GEMM。
    return matmul(x, W) + b

function MLP(x):
    h = Linear(x, W_up, b_up)      # D -> 4D
    h = GELU(h)
    y = Linear(h, W_down, b_down)  # 4D -> D
    return y
```

## 和量化有什么关系

权重量化最常压的就是 Linear 层权重，因为它们占参数量大头。常见形式包括：

- `W8A16`：权重 INT8，激活 FP16/BF16。
- `W4A16`：权重 INT4，激活 FP16/BF16。
- `W8A8`：权重和激活都 INT8。
- `FP8 W8A8`：权重和激活走 FP8 或近似 FP8 路径。

如果量化格式没有高效 GEMM / batched GEMM / fused dequant MatMul kernel，模型文件虽然变小，推理可能并不会更快。这个问题会在 [量化路线图](../quantization/index.md) 和 [数值、显存与运行时](numerics-memory-and-runtime-basics.md) 里继续展开。

## 本页结论

Linear 是表示投影，MLP 是逐 token 非线性变换，MatMul 是数学/API 操作，GEMM 是常见的底层 dense 矩阵-矩阵执行核心。理解这几层的边界，能帮助你把 Transformer、量化、算子优化和推理性能连成一条线。
