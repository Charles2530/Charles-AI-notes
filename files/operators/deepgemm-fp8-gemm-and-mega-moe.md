# DeepGEMM里的低精度矩阵乘与专家批量

DeepGEMM提供面向特定GPU路径的矩阵乘及相关kernel。理解它先从输入shape、低精度数据与scale布局开始，再看多个专家如何组织。本文依据commit `5081f851`的README与Python导出，接口描述限定在这一版本。

## 普通GEMM先确定矩阵方向

该版本的 `fp8_gemm_nt` 对应使用A与B转置的乘法，逻辑上A为M×K、B为N×K，输出M×N。调用名中的nt描述矩阵方向，不应把B误当作已按K×N存放的普通行优先矩阵。

README还规定scale布局。SM90路径使用FP32尺度，SM100相关路径使用打包UE8M0尺度，左侧尺度有转置和TMA对齐要求。实际调用前要核对data tensor与scale tensor两份布局。

输入转换和转置由调用方负责，README明确提醒辅助PyTorch处理可能较慢。因此单独GEMM的吞吐不包含全部量化准备成本。

## 多个专家可以沿M合并

假设两个专家的输入宽K、输出宽N相同，分别收到3和5个token。它们的M不同，W形状相同。Contiguous grouped模式把各专家token段放在同一个大输入中，另用元数据标识段与专家。

该版本M-grouped接口要求N/K固定，并有每专家段M对齐要求。不能仅因为名字叫grouped就假定M/N/K都可任意不同。Padding可能增加实际计算，尤其每专家token很少时，需要纳入收益分析。

## Decode为什么需要masked布局

CUDA Graph场景希望图结构和缓冲区稳定，而每个专家实际收到的token数量会变化。Masked grouped接口用有效计数等元数据限定计算范围，让固定缓冲区表达动态有效工作。

这与attention的causal mask不是同一个含义。一个描述专家段中哪些行有效，另一个描述query可以看到哪些key，不能混用概念。

## MegaMoE把哪些操作放在一起

该commit的README描述MegaMoE融合并重叠EP dispatch、两次专家线性层、SwiGLU及EP combine，并要求多进程与对称内存。调用前需要准备路由索引、路由权重、输入数据与scale，并按所需布局转换专家权重。

这些前置操作、缓冲分配和权重转换仍有成本。融合kernel不意味着整个MoE模型没有其他工作，也不能把单机普通GEMM的条件直接套到MegaMoE。

## 怎样读性能数字

先确认计时包含的是已准备数据的kernel，还是含quantize、转置、dispatch和combine的完整专家层。再固定每专家token分布、top-k、dtype、设备数量与拓扑。平均token数相同，负载偏斜和padding也可能产生不同时间。

源码和README在持续演进。这里的接口地图用于读懂固定版本，接入前应对照实际安装版本的函数签名和测试，不能将本文当作永久API保证。

![DeepSeek-V3 FP8 framework](../assets/images/paper-deep-dives/technical-reports/deepseek-v3/figure-6-fp8-framework.png){ width="860" }

<small>图源：[DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)，Figure 6。原图表达 FP8 mixed precision 训练框架中的高精保留、scale 和累加路径；读图重点是把 DeepGEMM / FP8 GEMM 看成“低精度格式、scale layout、硬件搬运和 kernel 形状”共同设计的问题。它不能证明 DeepGEMM 在所有 GEMM 上优于通用库。</small>

## 对照源码与延伸阅读

- [固定commit的接口、scale、grouped和MegaMoE说明](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/README.md)
- [该版本实际Python导出](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/__init__.py)
