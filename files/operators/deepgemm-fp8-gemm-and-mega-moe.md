# DeepGEMM里的低精度矩阵乘与专家批量

DeepGEMM提供面向特定GPU路径的矩阵乘及相关kernel。理解它先从输入shape、低精度数据与scale布局开始，再看多个专家如何组织。本文依据commit `5081f851`的README与Python导出，接口描述限定在这一版本。

## 普通GEMM先确定矩阵方向

该版本的 `fp8_gemm_nt` 对应使用A与B转置的乘法，逻辑上A为M×K、B为N×K，输出M×N。调用名中的nt描述矩阵方向，不应把B误当作已按K×N存放的普通行优先矩阵。

README还规定scale布局。SM90路径使用FP32尺度，SM100相关路径使用打包UE8M0尺度，左侧尺度有转置和TMA对齐要求。实际调用前要核对data tensor与scale tensor两份布局。

输入转换和转置由调用方负责，README明确提醒辅助PyTorch处理可能较慢。因此单独GEMM的吞吐不包含全部量化准备成本。

## 先写一张 shape contract

把一次调用压缩成四行，能先排除一半问题。下面是逻辑维度，不是对某个版本内部 stride 的完整描述。

| 对象 | 逻辑 shape | 要同时记录 |
| --- | --- | --- |
| A | `[M, K]` | dtype、行 stride、量化分组方向 |
| B | `[N, K]` | `nt` 语义、列或行的存储方式 |
| C | `[M, N]` | 输出 dtype、累加精度 |
| scale | 随 A/B 的分组而定 | scale 轴、转置、对齐和格式 |

`fp8_gemm_nt` 的 `nt` 表示逻辑乘法使用 (A B^T)。若 (M=3,K=4,N=5)，A 的逻辑 shape 是 `[3,4]`，B 是 `[5,4]`，结果是 `[3,5]`。把 B 先物理转成 `[4,5]` 并不能自动满足接口，调用约定和实际 stride 仍需对照固定版本源码。

每一次实验都把这四项写进日志。shape 相同而 scale 轴错了，kernel 可能照常运行并给出数值；这类错误比直接的维度异常更难发现。

## 先把量化误差和布局错误分开

用一个标量组说明排查顺序。若某一组 FP8 值的缩放因子为 (s)，量化和反量化可抽象为

\[
q=\operatorname{clip}\left(\operatorname{round}(x/s),q_{\min},q_{\max}\right),
\qquad
\hat x=sq.
\]

这不是 DeepGEMM 某个硬件路径的完整格式定义，只是为了得到可比较的参考值。先在 CPU 或高精度张量上计算 \(\hat A\hat B^T\)，再与 kernel 输出比较：

```text
reference = matmul(dequantize(A_q, scale_A),
                   dequantize(B_q, scale_B).T)
kernel_out = deep_gemm_fp8(A_q, B_q, scale_A, scale_B)
error = max_abs(kernel_out - reference)
```

如果 reference 已经和 FP32 结果有误差，那是量化误差；如果 kernel 与 reference 的差异随转置、padding 或 scale stride 改变，优先检查布局和元数据。不要一开始就用端到端 loss 归因，那里还混入了路由、激活和通信。

FP8 的 scale 并非一个可以随便 broadcast 的常数。分组大小、scale 的行列含义和硬件要求决定了每个元素使用哪个尺度。调试时打印一个小矩阵的元素坐标、所属分组和最终取到的 scale，比只打印 scale 的均值更有用。

## 多个专家可以沿M合并

假设两个专家的输入宽K、输出宽N相同，分别收到3和5个token。它们的M不同，W形状相同。Contiguous grouped模式把各专家token段放在同一个大输入中，另用元数据标识段与专家。

该版本M-grouped接口要求N/K固定，并有每专家段M对齐要求。不能仅因为名字叫grouped就假定M/N/K都可任意不同。Padding可能增加实际计算，尤其每专家token很少时，需要纳入收益分析。

一个教学布局可以写成

```text
expert 0: rows [0:3)   M0 = 3
expert 1: rows [3:8)   M1 = 5
tokens:   [e0_t0, e0_t1, e0_t2, e1_t0, ...]
```

kernel 完成后，输出仍按这段顺序排列；MoE 层还要依据路由索引把它们放回原 token，并乘上 router weight。只核对 grouped GEMM 的连续输出，不能证明 dispatch 和 combine 的顺序正确。至少构造一个 token 被两个专家选中、一个专家收到零 token、一个专家接近容量上限的例子。

若对齐要求把 (M_0=3) 补到 4，padding 行必须有明确的有效计数，并在 combine 时被排除。把 padding 当成真实 token 会污染输出，漏掉 padding 则可能读到未初始化内存。记录每个专家的有效 M、填充 M 和最终写回范围，才能解释小 batch 下的性能和数值。

## Decode为什么需要masked布局

CUDA Graph场景希望图结构和缓冲区稳定，而每个专家实际收到的token数量会变化。Masked grouped接口用有效计数等元数据限定计算范围，让固定缓冲区表达动态有效工作。

这与attention的causal mask不是同一个含义。一个描述专家段中哪些行有效，另一个描述query可以看到哪些key，不能混用概念。

## MegaMoE把哪些操作放在一起

该commit的README描述MegaMoE融合并重叠EP dispatch、两次专家线性层、SwiGLU及EP combine，并要求多进程与对称内存。调用前需要准备路由索引、路由权重、输入数据与scale，并按所需布局转换专家权重。

这些前置操作、缓冲分配和权重转换仍有成本。融合kernel不意味着整个MoE模型没有其他工作，也不能把单机普通GEMM的条件直接套到MegaMoE。

## 怎样读性能数字

先确认计时包含的是已准备数据的kernel，还是含quantize、转置、dispatch和combine的完整专家层。再固定每专家token分布、top-k、dtype、设备数量与拓扑。平均token数相同，负载偏斜和padding也可能产生不同时间。

源码和README在持续演进。这里的接口地图用于读懂固定版本，接入前应对照实际安装版本的函数签名和测试，不能将本文当作永久API保证。

## 基准至少要有三条时间线

同一个 shape 建议同时测三种边界：

| 时间线 | 包含内容 | 能回答什么 |
| --- | --- | --- |
| 纯 kernel | 已准备好的 data 与 scale 到输出 | kernel 本身的吞吐和稳定性 |
| 单层 | quantize、转置、packing、kernel | 低精度层是否真的省时间 |
| 专家全链路 | dispatch、grouped GEMM、SwiGLU、combine、通信 | MoE 系统是否获得端到端收益 |

每条时间线都要预热 JIT，并记录 p50、p95、有效 token 数、专家负载方差和 padding 比例。平均 M 一样时，负载集中在一个专家和均匀分散在多个专家，通常会选择不同配置。一个“快 20%”的数字如果没有说明时间线和 token 分布，无法迁移到另一台机器。

可以把结果卡片写成

```text
commit: 5081f851
gpu: <architecture and count>
shape: M=..., N=..., K=...
expert_tokens: [3, 5, 0, 8]
dtype: fp8 data / fp32 or packed scale
timed_region: kernel | layer | full_moe
warmup: ...
padding_rows: ...
max_abs_error_vs_reference: ...
```

这张卡片把数值正确性、shape 条件和性能边界放在一起，避免用一个吞吐数字覆盖接口差异。

![DeepSeek-V3 FP8 framework](../assets/images/paper-deep-dives/technical-reports/deepseek-v3/figure-6-fp8-framework.png){ width="860" }

<small>图源：[DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)，Figure 6。原图表达 FP8 mixed precision 训练框架中的高精保留、scale 和累加路径；读图重点是把 DeepGEMM / FP8 GEMM 看成“低精度格式、scale layout、硬件搬运和 kernel 形状”共同设计的问题。它不能证明 DeepGEMM 在所有 GEMM 上优于通用库。</small>

## 对照源码与延伸阅读

- [固定commit的接口、scale、grouped和MegaMoE说明](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/README.md)
- [该版本实际Python导出](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/__init__.py)

## 先把 DeepGEMM 当成系统契约阅读

固定版本、GPU 架构、A/B 方向、scale shape、dtype、专家 token 分布和计时边界，再比较普通 GEMM、grouped GEMM 与 MegaMoE。首先用小矩阵和高精度参考验证结果，随后分别加入 packing、dispatch、padding、combine 和多进程通信。单个 kernel 吞吐不能替代完整专家层成本。

**来源与改编说明**：[DeepGEMM 固定 commit README](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/README.md) 是接口和 scale 布局的主要来源；本文没有运行 GPU benchmark，因此性能判断只保留条件和测量方法。自测时构造两个专家，一个收到 0 个 token，另一个收到全部 token，说明 masked grouped 路径需要验证什么。

想把 grouped GEMM 放回完整路由流程，可以继续看[MoE 路由与稀疏 kernel](moe-routing-and-sparse-kernels.md)；低精度的 scale、累加和回退边界见[低精度与量化算子](low-precision-and-quantized-kernels.md)。
