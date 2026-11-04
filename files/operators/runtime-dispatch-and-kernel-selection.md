# Runtime怎样判断当前输入能用哪个kernel

假设已有一个只支持连续BF16矩阵、K为32倍数的快GEMM。Dispatch首先要判断这个实现能否正确处理输入，随后才比较它是否比通用实现更快。把这两个判断混在一起，容易让某个漂亮shape上的优化进入不适用的输入。

## 一组可检查的分流条件

下面是逻辑示意，`fast_gemm`和`generic_gemm`代表已实现的调用入口。

```python
def choose_path(shape, dtype, stride, supported_device):
    m, n, k = shape
    legal = supported_device and dtype == 'bf16' and stride == 'contiguous' and k % 32 == 0
    profitable = m >= 32 and n >= 128
    return 'fast' if legal and profitable else 'generic'
```

`legal`表达实现契约，`profitable`表达经过测量得到的性能区间。示例阈值仅用于展示两者分工，并非推荐硬编码到某个GPU。

## 同shape的输入也可能需要不同实现

转置视图和连续矩阵可有相同shape但不同stride。量化矩阵还多出scale布局、group size和packing。Attention需要区分causal、滑动窗口和有效长度。若这些条件影响地址或数学定义，就不能只按M/N/K分流。

一种做法是在wrapper转换输入，使其符合快路径。转换本身需要计入成本，有时通用stride-aware实现反而更快。

## 缓存key与调优key各回答什么

编译缓存key用于复用特定代码生成结果，必须区分会改变编译语义的条件。调优key用于复用某类输入的候选优劣，可以按性能特征组织。框架会自动处理一部分dtype和特化参数，但自定义缓存仍需明确剩余假设。

若每个请求长度都生成新变体，会增加首次编译与缓存数量；如果把不同语义粗暴合成一个key，则有正确性风险。Shape bucket可以减少变体，但需要保持mask和真实有效长度。

## Fallback应该返回同一个函数

通用实现必须保持dtype约定、广播、mask和梯度语义。快路径不支持时回退，结果才仍然可解释。回退发生频率还影响性能，不能只在日志中记录一次“fast kernel加载成功”。

对真实流量统计每桶命中路径、转换耗时和首次编译耗时，再比较完整层。若最常见输入从未满足合法条件，继续调快路径tile无法改善服务。

## 一个新变体怎样进入选择器

先验证边界shape与非连续输入是否正确分流，再在相同设备和计时范围下比较候选，最后用实际调用频率加权。保留通用路径使新硬件、稀有shape和编译失败有明确处理方式，避免在请求中隐式换用语义不同的计算。

![vLLM runtime components](../assets/images/paper-figures/inference/vllm-system-overview-padded.png){ width="860" }

<small>图源：[vLLM 官方博客](https://vllm.ai/blog/2023-06-20-vllm)。原图表达 runtime 不只是模型 forward，还包含 scheduler、KV manager、allocator 和 worker；读图重点是把 kernel selection 放回 runtime dispatch：同一个算子在不同请求桶、cache 状态和 batch 形态下可能走不同路径。它不能证明 vLLM 的 dispatch 策略适用于所有框架。</small>

## 对照源码与延伸阅读

- [矩阵乘wrapper、autotune key和候选配置](https://github.com/triton-lang/triton/blob/cd2e01c6686d6503802e6cdfac56d5a5dfdb27a1/python/tutorials/03-matrix-multiplication.py)
- [架构条件决定候选是否可用](https://github.com/triton-lang/triton/blob/cd2e01c6686d6503802e6cdfac56d5a5dfdb27a1/python/tutorials/09-persistent-matmul.py)
