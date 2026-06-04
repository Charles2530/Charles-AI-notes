# MoE 路由与稀疏 Kernel

稀疏计算和 `MoE`（Mixture of Experts）常被看成“模型结构创新”，但一旦真正训练或部署，问题很快就会变成 kernel 与通信系统问题。原因很简单：
**稠密模型的热点大多是规则大矩阵，稀疏和 MoE 的热点则往往是 token 重排、路由、gather/scatter、all-to-all 和不均匀负载。**

这使得稀疏系统的工程难点不在公式，而在如何把 token 送到正确专家、如何让热门专家不成为热点、如何让重排和通信不把省下来的 FLOPs 吃掉，以及如何让路由 kernel 和主计算 kernel 协同。

![DeepSeek-V3 expert load balance](../assets/images/paper-deep-dives/technical-reports/deepseek-v3/figure-9-expert-load.png){ width="760" }

<small>图源：[DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)。原图表达 expert load 的均衡状态；本站读法是把 sparse kernel 性能先放回 token-to-expert 分布、grouped GEMM 和 all-to-all 账本。它不能证明路由天然均衡，也不能替代线上 expert 热点监控。</small>

## 为什么稀疏 kernel 和稠密 kernel 不一样

稠密 kernel 往往假设数据布局规则、工作量均匀，并且可以用大块 GEMM 吃满硬件。稀疏 / MoE 场景则相反：token 分布不均，某些 expert 很热、某些很闲，计算前要先 route 和重排，工作负载还会随每批输入变化。

这意味着稀疏系统的优化目标不只是“把 GEMM 做快”，更是“把不规则性控制住”。

## Top-k Routing 的核心代价

一个常见 MoE 流程是 router 为每个 token 产生 expert 分数，选出 top-k expert，将 token dispatch 到对应 expert，expert 各自计算后再把结果 combine 回去。真正的代价通常出现在 token permutation、dispatch / combine、all-to-all、负载不均和小批量 expert GEMM。

## MoE token 流程图

```mermaid
flowchart LR
    A["tokens [B, L, D]"] --> B["router logits"]
    B --> C["top-k experts"]
    C --> D["capacity / drop / pad"]
    D --> E["token permutation"]
    E --> F["all-to-all dispatch"]
    F --> G["expert GEMM"]
    G --> H["all-to-all combine"]
    H --> I["unpermute"]
    I --> J["residual stream"]
```

这张图说明 MoE 性能瓶颈不只在 `expert GEMM`。路由、容量、重排、通信、合并和反重排都在热路径里。只优化专家矩阵乘，而不看 dispatch 和 all-to-all，常常会得到“FLOPs 少了，系统没快”的结果。

### 3.1 一个 token 数字账

假设 batch 内共有 `B*L=8192` 个 token，`top-k=2`，expert 数为 64。理论上每个 expert 平均接收：

\[
\frac{8192\times2}{64}=256
\]

个 token。这个 `256` 是平均负载，不是每个 expert 实际收到的数量。工程上常用 capacity factor 给每个 expert 预留容量：

\[
capacity \approx \frac{B L \times topk}{N_{\text{expert}}}\times factor
\]

所以 capacity 设为 `320` 相当于 factor 约 `1.25`，设为 `640` 相当于 factor `2.5`。但真实 router 可能不均匀：热门 expert 收 520 个 token，冷 expert 只收 80 个。若 capacity 是 320，热门 expert 会 drop 或 reroute；若 capacity 是 640，padding、空槽和小批量 GEMM shape 会浪费。这个超参同时影响模型质量、kernel shape 和 all-to-all 消息大小。

## Token Permutation 为什么是主痛点

Router 选完 expert 后，token 往往需要按 expert 分组重排。
这一步本质上是一个复杂的索引、scatter、gather 问题。

### 3.1 为什么它会很慢

它慢在访问不连续、写入位置动态，常要配合 prefix sum 或计数，也很难像 GEMM 那样规则向量化。

在很多 MoE 系统里，真正先成为瓶颈的并不是 expert matmul，而是重排与通信。

## Capacity Factor 与 kernel 的关系

MoE 里常用 capacity factor 控制每个 expert 最多接收多少 token。
这看似是训练稳定性超参，但实际上也影响 kernel 路径：容量越宽松，负载可能更不均；容量越严格，drop token 风险更高；expert batch 大小分布又会直接决定后续 GEMM 形状。

因此它不是纯模型超参，也是系统超参。

## All-to-All 是什么，为什么难

在多卡 expert 并行里，不同 token 可能要被发往不同 GPU 上的 expert。
这通常意味着 all-to-all 风格的通信：每个 rank 都向多个 rank 发 token，也从多个 rank 收 token。它比 all-reduce 更复杂，因为消息大小不均、负载可能严重偏斜、与 token permutation 耦合很深，也更难重叠。

## 稀疏系统为什么容易出现“理论省 FLOPs，实际不快”

因为虽然只激活部分 expert，但新增了 router、token 重排、dispatch / combine、all-to-all、小而碎的 expert batch 和热门 expert 拥堵等成本。

只要这些额外成本没有被控制住，稀疏计算的收益就会被抵消。

## 服务态 MoE 的额外问题

在线推理里，MoE 比训练更麻烦，因为 batch 更小、请求更不规则、每步 decode token 少，热 expert 还可能放大 tail latency，多租户也会让路由分布更复杂。因此 MoE 推理服务特别依赖 expert placement、路由缓存、dispatch kernel 和通信调度。

## 稀疏 attention 与 block-sparse 路线

除了 expert 稀疏，还有一类稀疏 kernel 出现在 block-sparse attention、局部窗口 attention、检索增强 attention 和动态剪枝路径中。这些方法试图减少不必要计算，但会引入更复杂的索引、block map、稀疏布局管理、边界条件和实际有效算子粒度问题。

如果 block 设计和硬件不匹配，稀疏反而可能比稠密更慢。

视频 DiT 里的 [SLA / SLA2](../paper-deep-dives/foundations/sla-sparse-linear-attention.md) 是一个典型例子：SLA 不是只做 block-sparse attention，而是把 critical blocks 交给 sparse exact attention，把 marginal blocks 交给 linear attention 补偿；SLA2 又把路由改成可学习的 \(R(Q,K)\)，并用 QAT 给 sparse branch 加 low-bit attention。

## MoE 验收指标

| 指标 | 看什么 | 失败信号 |
| --- | --- | --- |
| expert load | 每个 expert 的 token 数分布 | 热 expert 长期拥堵 |
| dispatch time | permutation + all-to-all 时间 | GEMM 快但端到端不快 |
| drop / reroute rate | capacity 是否过紧 | 质量掉点或训练不稳 |
| expert GEMM shape | 每个 expert 的 batch 大小 | 小而碎，Tensor Core 吃不满 |
| tail latency | 最慢 expert / rank 拖尾 | P99 被少数 rank 拉坏 |

服务态还要单独看租户和 prompt 类型是否改变 expert 分布。一个 MoE 模型在离线 benchmark 上平衡，不代表线上流量也平衡；热门业务路径可能持续打到少数 expert，变成系统热点。

## 稀疏 kernel 的优化重点

与稠密 kernel 相比，稀疏 kernel 更应优先关注索引和重排开销、局部性、token / block 分组、热点负载均衡和通信路径。

也就是说，稀疏系统优化的第一优先级往往不是“更多 Tensor Core”，而是“更少无意义搬运与更少不均衡”。

## 直觉例子

如果稠密模型像每个零件都要走同一条大装配线，那么 MoE 更像每个零件会被动态分发到不同的专业工位。好处是主装配线不用让所有工位同时开工，但坏处是零件分拣、转运和工位排队本身会变成新的瓶颈。MoE 系统真正难的地方，不在工位上的加工，而在整个分拣与物流系统是否高效。

## 读完以后怎么判断

稀疏、MoE 与路由 kernel 的核心，不是“把矩阵变稀疏”这么简单，而是如何在保持模型容量优势的同时，控制 token 重排、all-to-all、负载不均和小批次 expert 计算带来的系统开销。
如果没有这层 kernel 与通信视角，MoE 很容易停留在“理论参数效率很高”的纸面优势，而无法转化成真实训练和推理服务系统里的可观收益。

## 工程收束

MoE 路由与稀疏 kernel 要同时看 top-k 路由、token dispatch、expert 负载、combine 路径和稀疏访存。只盯 expert matmul 会忽略重排成本，负载不均也会把尾延迟拉坏；上线前应把 dispatch / gather 一并 profile，监控 expert load，并为 router 建立回归。

## 相关阅读与下一步

- 外部材料：[NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)。
- 外部材料：[Triton 文档](https://triton-lang.org/main/index.html)。
- 外部材料：[CUTLASS 文档](https://docs.nvidia.com/cutlass/)。
- 站内下一步：[算子与编译器专题](index.md)。
- 站内下一步：[CUDA 编程模型与内存层次](cuda-programming-model-and-memory.md)。
- 站内下一步：[Triton 编程模型与自动调优](triton-programming-model-and-autotuning.md)。
