# MoE 路由与稀疏 Kernel

稀疏计算和 `MoE`（Mixture of Experts）常被看成“模型结构创新”，但一旦真正训练或部署，问题很快就会变成 kernel 与通信系统问题。原因很简单：  
**稠密模型的热点大多是规则大矩阵，稀疏和 MoE 的热点则往往是 token 重排、路由、gather/scatter、all-to-all 和不均匀负载。**

这使得稀疏系统的工程难点不在公式，而在如何把 token 送到正确专家、如何让热门专家不成为热点、如何让重排和通信不把省下来的 FLOPs 吃掉，以及如何让路由 kernel 和主计算 kernel 协同。

!!! note "初学者先抓住"

    MoE 的系统难点不在“专家层公式”，而在 token 怎么被路由、重排、通信、计算再合并。稀疏节省了部分 FLOPs，但引入了不规则访问、负载不均和 all-to-all。

!!! example "有趣例子：分拣中心"

    包裹送到正确仓库只是第一步。如果某个仓库爆满、传送带堵住、回收路径绕远，整体效率仍然很差。MoE 的 token dispatch 和 combine 也是这个问题。

## 1. 为什么稀疏 kernel 和稠密 kernel 不一样

稠密 kernel 往往假设数据布局规则、工作量均匀，并且可以用大块 GEMM 吃满硬件。稀疏 / MoE 场景则相反：token 分布不均，某些 expert 很热、某些很闲，计算前要先 route 和重排，工作负载还会随每批输入变化。

这意味着稀疏系统的优化目标不只是“把 GEMM 做快”，更是“把不规则性控制住”。

## 2. Top-k Routing 的核心代价

一个常见 MoE 流程是 router 为每个 token 产生 expert 分数，选出 top-k expert，将 token dispatch 到对应 expert，expert 各自计算后再把结果 combine 回去。真正的代价通常出现在 token permutation、dispatch / combine、all-to-all、负载不均和小批量 expert GEMM。

## 3. Token Permutation 为什么是主痛点

Router 选完 expert 后，token 往往需要按 expert 分组重排。  
这一步本质上是一个复杂的索引、scatter、gather 问题。

### 3.1 为什么它会很慢

它慢在访问不连续、写入位置动态，常要配合 prefix sum 或计数，也很难像 GEMM 那样规则向量化。

在很多 MoE 系统里，真正先成为瓶颈的并不是 expert matmul，而是重排与通信。

## 4. Capacity Factor 与 kernel 的关系

MoE 里常用 capacity factor 控制每个 expert 最多接收多少 token。  
这看似是训练稳定性超参，但实际上也影响 kernel 路径：容量越宽松，负载可能更不均；容量越严格，drop token 风险更高；expert batch 大小分布又会直接决定后续 GEMM 形状。

因此它不是纯模型超参，也是系统超参。

## 5. All-to-All 是什么，为什么难

在多卡 expert 并行里，不同 token 可能要被发往不同 GPU 上的 expert。  
这通常意味着 all-to-all 风格的通信：每个 rank 都向多个 rank 发 token，也从多个 rank 收 token。它比 all-reduce 更复杂，因为消息大小不均、负载可能严重偏斜、与 token permutation 耦合很深，也更难重叠。

## 6. 稀疏系统为什么容易出现“理论省 FLOPs，实际不快”

因为虽然只激活部分 expert，但新增了 router、token 重排、dispatch / combine、all-to-all、小而碎的 expert batch 和热门 expert 拥堵等成本。

只要这些额外成本没有被控制住，稀疏计算的收益就会被抵消。

## 7. 服务态 MoE 的额外问题

在线推理里，MoE 比训练更麻烦，因为 batch 更小、请求更不规则、每步 decode token 少，热 expert 还可能放大 tail latency，多租户也会让路由分布更复杂。因此 MoE 推理服务特别依赖 expert placement、路由缓存、dispatch kernel 和通信调度。

## 8. 稀疏 attention 与 block-sparse 路线

除了 expert 稀疏，还有一类稀疏 kernel 出现在 block-sparse attention、局部窗口 attention、检索增强 attention 和动态剪枝路径中。这些方法试图减少不必要计算，但会引入更复杂的索引、block map、稀疏布局管理、边界条件和实际有效算子粒度问题。

如果 block 设计和硬件不匹配，稀疏反而可能比稠密更慢。

视频 DiT 里的 [SLA / SLA2](../paper-deep-dives/foundations/sla-sparse-linear-attention.md) 是一个典型例子：SLA 不是只做 block-sparse attention，而是把 critical blocks 交给 sparse exact attention，把 marginal blocks 交给 linear attention 补偿；SLA2 又把路由改成可学习的 \(R(Q,K)\)，并用 QAT 给 sparse branch 加 low-bit attention。

## 9. 稀疏 kernel 的优化重点

与稠密 kernel 相比，稀疏 kernel 更应优先关注索引和重排开销、局部性、token / block 分组、热点负载均衡和通信路径。

也就是说，稀疏系统优化的第一优先级往往不是“更多 Tensor Core”，而是“更少无意义搬运与更少不均衡”。

## 10. 一个形象比喻

如果稠密模型像每个零件都要走同一条大装配线，那么 MoE 更像每个零件会被动态分发到不同的专业工位。好处是主装配线不用让所有工位同时开工，但坏处是零件分拣、转运和工位排队本身会变成新的瓶颈。MoE 系统真正难的地方，不在工位上的加工，而在整个分拣与物流系统是否高效。

## 11. 小结

稀疏、MoE 与路由 kernel 的核心，不是“把矩阵变稀疏”这么简单，而是如何在保持模型容量优势的同时，控制 token 重排、all-to-all、负载不均和小批次 expert 计算带来的系统开销。  
如果没有这层 kernel 与通信视角，MoE 很容易停留在“理论参数效率很高”的纸面优势，而无法转化成真实训练和推理服务系统里的可观收益。

## 工程收束

MoE 路由与稀疏 kernel 要同时看 top-k 路由、token dispatch、expert 负载、combine 路径和稀疏访存。只盯 expert matmul 会忽略重排成本，负载不均也会把尾延迟拉坏；上线前应把 dispatch / gather 一并 profile，监控 expert load，并为 router 建立回归。
