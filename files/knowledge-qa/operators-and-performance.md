# 算子正确性与性能问答

优化一个算子，首先要知道它为哪些输入计算什么结果，然后再量时间。数学等价、浮点容差、注册行为和端到端收益是不同层次的检查。

## Linear的M、N、K到底是哪几个轴

输入`[B,L,D_in]`可以视作`[M,K]`，其中M=B×L、K=D_in，权重按数学方向为`[K,N]`，N=D_out。训练与prefill的M通常较大，小batch decode的M较小；同一层可能需要不同kernel。

PyTorch权重常存`[N,K]`，前向使用转置。打印一个M×N结果并不能证明转置正确，先用非方阵和可手算数值核对。见[Linear与GEMM](../foundations/linear-layers-mlp-and-gemm.md)。

## FLOPs少了，为什么时间反而增加

减少计算可能同时增加临时内存、kernel launch、布局转换或通信。一个算术强度低的操作主要受搬运限制，删掉一些乘法几乎无效；一个融合版本若寄存器压力过高，也可能出现spill和并发下降。

先估算FLOPs与必须读取写入的字节，再看profiler中的实际带宽、执行时间和访存。峰值FLOP/s只是上限，不是任意shape都能达到的常数。见[Roofline与benchmark](../operators/roofline-benchmarking-and-case-studies.md)。

## Tiling为什么比直接并行三重循环更快

矩阵乘的同一块A和B会被多个输出重复使用。将它们加载到更近的存储后复用，可以减少昂贵的全局读取。tile大小还决定寄存器、shared memory和并行块数量，所以越大并非越好。

以2×2输出tile为例，一小段K上的两行A与两列B可产生四个输出的贡献；若每个输出完全独立从全局内存读，会重复搬相同数据。真实Tensor Core布局更复杂，但复用逻辑相同。见[CUDA内存层次](../operators/cuda-programming-model-and-memory.md)。

## FlashAttention的exact是不是逐bit相同

Exact通常指计算同一数学attention，没有通过稀疏或低秩近似改目标。分块softmax、归约顺序和累加格式不同，浮点结果仍可能与朴素实现存在小误差。

应按dtype与规模设置合理绝对/相对容差，并覆盖大logits、mask、不同长度与反向梯度。把容差设到足以吞掉明显错误不算验证，要求所有格式逐bit相同也可能不合理。见[Attention与融合](../operators/gemm-attention-and-fused-kernels.md)。

## Persistent GEMM减少的是哪种开销

普通tiled GEMM也可以只有一次kernel launch。Persistent实现让较少的program或CTA在一次launch内持续处理多个tile，可能改善调度、局部复用和资源利用；不能笼统说它把“每tile一次launch”消除了。

收益受矩阵尺寸、设备、tile调度与缓存影响。与同样单次launch的基线比较，才能判断优势来自哪里。见[高级kernel模式](../operators/advanced-kernel-patterns-and-shape-specialization.md)。

## `opcheck`通过就证明梯度正确了吗

不证明。框架注册检查关注schema、fake tensor、autograd注册等契约；数学梯度需用合适的`gradcheck`、参考实现或解析例子验证。两者都重要，但不能替代。

此外前向对一个连续输入正确，不代表非连续stride、尾块、零长度或重复索引也正确。测试应围绕算子允许的输入范围，明确拒绝不支持的情况。见[自定义算子与框架集成](../operators/custom-op-development-and-framework-integration.md)。

## 怎么避免测到编译、异步返回或缓存假象

先预热，使编译和初始化按预定方式完成；用设备事件或正确同步量执行时间；重复多次并记录分布。输入是否重复、缓存是否命中、时钟状态与并发负载也会影响结果。

端到端优化还要检查调用是否真的命中新kernel。Amdahl定律给出简单上界：原耗时30%的部分加速2倍，理想总加速为1/(0.7+0.3/2)≈1.176。它假设其余成本不变，新增转换开销会让收益更低。见[性能回归与维护](../operators/testing-regression-and-maintenance.md)。
