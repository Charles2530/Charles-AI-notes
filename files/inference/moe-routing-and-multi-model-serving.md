# 专家路由与请求路由分别选择什么

MoE根据token隐藏向量选专家，服务路由根据整个请求选择模型或实例。两者都分配计算，但输入、状态与质量保证不同。先分别算一次，就能看清“只激活部分参数”不等于所有成本按比例下降。

## 一个token选中两个专家

设四个专家的router概率为 `[0.5,0.3,0.15,0.05]`，选择前两个。若模型规定在选中集合重新归一化，权重变成0.625和0.375；若不归一化，则保留0.5和0.3。两种模型定义会得到不同输出。

假设选中专家标量输出为2和6，归一化结果为3.5。实际专家输出是向量，按同样权重逐元素合并。具体门控与共享专家规则需读模型实现。

## 一批token决定专家计算的shape

T个token、top-k为k时，约有Tk条token-expert记录。它们按专家重排后，各专家的token数就是其GEMM的M维。平均负载小，不代表最热门专家也小。

若8个token、top-2产生16条记录，其中某专家收到10条，另外几个总共6条，计算与通信会不均。是否丢弃超额token、动态分配或padding，取决于capacity策略；dropless实现也可能承担尾部延迟。

未激活专家参数通常仍需在某处驻留或加载，所以激活参数量更小不自动等于整模型显存同样小。专家并行还增加dispatch与combine通信。

![DeepSeek-V3 expert load balance](../assets/images/paper-deep-dives/technical-reports/deepseek-v3/figure-9-expert-load.png){ width="760" }

<small>图源：[DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)。原图表达 MoE expert load 在训练中的均衡状态；读图重点是把路由看成质量、成本、延迟和负载之间的动态分配问题。它不能推出任意 MoE 或多模型路由都会自然均衡，线上仍要按 expert、设备和请求桶观测热点。</small>

## 请求级路由先看整项任务

小模型可先处理简单请求，复杂或不确定的请求升级大模型。假设小模型成本1、大模型10、升级率30%，不复用计算时平均成本4。若小模型耗时0.2秒、大模型1秒，串行升级请求至少增加到1.2秒，再加队列和路由判断。

若路由器错把困难题交给小模型且未升级，质量会受损。若过于保守，大量重复尝试又增加成本。因此需要在固定任务集上测路由误判、最终成功率与完整等待。

## 服务状态可能不能跨模型复用

不同权重、tokenizer或结构的KV通常不能直接共享。切换模型可能重新prefill，adapter和缓存身份也需核对。将升级成本只记为大模型decode，会漏掉重新构建上下文的费用。

## 两种路由怎样一起测

MoE层记录每专家token、重排、通信和GEMM时间；请求路由记录每模型调用、升级、重试与任务结果。前者解释一条模型路径内部的不均衡，后者解释请求选择和重复计算。两套指标可以关联，但不应合成一个笼统“路由命中率”。


## 继续读

路由策略落到服务端后，继续看[服务 runtime 的 batch 与 fallback](serving-runtimes-vllm-sglang-and-tensorrt-llm.md)、[成本模型与容量规划](cost-modeling-and-slo-design.md)以及[FP8 与 MoE kernel](../operators/deepgemm-fp8-gemm-and-mega-moe.md)。

## 对照源码与延伸阅读

- [专家矩阵的grouped调度](https://github.com/triton-lang/triton/blob/cd2e01c6686d6503802e6cdfac56d5a5dfdb27a1/python/tutorials/08-grouped-gemm.py)
- [请求状态、缓存身份与调度接口](https://github.com/vllm-project/vllm/blob/fa3b9d1377219247060d3c8b49eae199c9e7e154/vllm/v1/core/sched/scheduler.py)

## MoE 的负载均衡损失为何存在

如果 router 总是把 token 送到少数专家，热门专家的 GEMM、显存和通信会成为关键路径，其他专家则空闲。Switch Transformer 这类方法会加入辅助负载均衡项，使 token 分配和 router 概率不要长期集中。抽象地写，若 (f_i) 是分配到专家 (i) 的 token 比例、(p_i) 是 router 对专家 (i) 的平均概率，可以用

\[
L_{\text{balance}}=N\sum_{i=1}^{N}f_i p_i
\]

作为一种额外信号，其中 (N) 是专家数。它鼓励 (f_i) 与 (p_i) 更均匀，但不能保证每个 batch 都完全均衡，也会与主任务 loss 发生权衡。具体系数、top-k、capacity 和丢弃策略必须按模型实现核对。

## Capacity、dropless 与 token dropping

设一个 batch 有 (T) 个 token、专家数 (N)、top-k 为 (k)，平均每个专家收到 (Tk/N) 条 token。若 capacity factor 为 (c)，一个简单的容量上限近似为

\[
C=\left\lceil c\frac{Tk}{N}\right\rceil.
\]

当热门专家超过 (C) 时，系统可能丢弃 token、把它送到备用专家、padding 到更大 shape，或采用 dropless 但承受不规则 GEMM。四种策略对应不同的质量、显存和尾延迟边界。报告“top-2 MoE”还不够，至少要写 `capacity_factor`、溢出 token 比例和是否丢弃。

## 路由实验应该观察哪些层

不要只画平均 expert load。一个可用的 MoE trace 至少包括：

| 层 | 指标 | 能回答的问题 |
| --- | --- | --- |
| router | 每 expert token 数、概率熵、top-k 重叠 | 路由是否集中 |
| dispatch | 重排时间、通信 bytes、padding/丢弃 | 路由开销是否抵消稀疏收益 |
| expert GEMM | 每专家 M、kernel 时间、空闲率 | shape 不均是否造成尾延迟 |
| combine | 回填时间、输出 norm | token 是否按原位置恢复 |
| 训练/服务 | loss、质量、p50/p99、OOM | 负载均衡是否值得代价 |

同一模型在短 batch 和长上下文中可能有不同的专家分布；同一专家平均负载正常，也可能在某些 prompt 桶中持续热点。因此应按 token 数、任务类型和序列长度分桶。

## MoE 路由与请求路由的组合案例

考虑一个小模型先处理请求，困难请求升级到大模型；大模型内部又是 top-2 MoE。一次请求的成本可拆成：

```text
请求路由判断
  -> 小模型 prefill/decode
  -> （可选）升级并重新 prefill
  -> 大模型 router / dispatch / expert GEMM / combine
  -> 质量判定与重试
```

如果升级模型使用不同 tokenizer 或权重，KV 不能直接复用。若路由器误判，重试可能比直接使用大模型更贵；若 MoE 热门专家造成尾延迟，外层请求路由的“便宜模型”也可能无法满足 SLO。实验应把外层升级率和内层 expert 热点同时记录，而不是合并成一个命中率。

## 一个可复现的路由对照

固定模型、输入集合、总 token 和质量阈值，至少比较：

1. dense baseline。
2. MoE，关闭或降低均衡损失。
3. MoE，启用均衡损失和相同容量。
4. MoE，改变 capacity 或 dropless 策略。

每组记录激活 FLOPs 只是第一步；还要记录专家参数驻留、dispatch 通信、实际 kernel、GPU 显存和 p99。稀疏理论计算量下降，不自动等于端到端延迟下降。

## 来源与改编说明

- [Switch Transformers](https://arxiv.org/abs/2101.03961)提供稀疏路由与辅助负载均衡损失的原始解释；公式在不同实现中可能有缩放或变体。
- [DeepSeek-V3 技术报告](https://arxiv.org/abs/2412.19437)用于对照大规模 MoE 的专家负载、路由和系统约束；报告中的硬件与训练设置不能直接外推。
- [Triton grouped GEMM 教程](https://github.com/triton-lang/triton/blob/cd2e01c6686d6503802e6cdfac56d5a5dfdb27a1/python/tutorials/08-grouped-gemm.py)用于理解按专家 shape 分组的 kernel 入口。

## 自测

1. (T=1024,N=8,k=2,c=1.25) 时，平均容量上限约是多少？超过容量的 token 有哪些处理选择？
2. 为什么专家平均负载均衡仍可能出现 p99 尾延迟？
3. 外层请求升级和内层专家路由分别记录哪些事件？
4. 如果 MoE 激活 FLOPs 下降但速度不变，你会先检查 dispatch、GEMM shape、通信还是请求排队？请写出证据顺序。
