# 稀疏、MoE 与路由 Kernel

稀疏计算和 `MoE`（Mixture of Experts）常被看成“模型结构创新”，但一旦真正训练或部署，问题很快就会变成 kernel 与通信系统问题。原因很简单：  
**稠密模型的热点大多是规则大矩阵，稀疏和 MoE 的热点则往往是 token 重排、路由、gather/scatter、all-to-all 和不均匀负载。**

这使得稀疏系统的工程难点，不在公式，而在：

1. 如何把 token 送到正确专家；
2. 如何让热门专家不成为热点；
3. 如何让重排和通信不把节省下来的 FLOPs 吃掉；
4. 如何让路由 kernel 和主计算 kernel 协同。

## 1. 为什么稀疏 kernel 和稠密 kernel 不一样

**稠密 kernel 往往假设**：

1. 数据布局规则；
2. 工作量较均匀；
3. 可用大块 GEMM 吃满硬件。

而稀疏 / MoE 场景下，常见特点是：

1. token 分布不均；
2. 某些 expert 很热，某些很闲；
3. 需要先 route，再重排，再算；
4. 工作负载随每批输入变化。

这意味着稀疏系统的优化目标不只是“把 GEMM 做快”，更是“把不规则性控制住”。

## 2. Top-k Routing 的核心代价

**一个常见 MoE 流程是**：

1. router 为每个 token 产生 expert 分数；
2. 选 top-k expert；
3. 将 token dispatch 到对应 expert；
4. expert 各自计算；
5. 再把结果 combine 回去。

**真正的代价通常出现在**：

1. token permutation；
2. dispatch / combine；
3. all-to-all；
4. 负载不均；
5. 小批量 expert GEMM。

## 3. Token Permutation 为什么是主痛点

Router 选完 expert 后，token 往往需要按 expert 分组重排。  
这一步本质上是一个复杂的索引、scatter、gather 问题。

### 3.1 为什么它会很慢

因为：

1. 访问不连续；
2. 写入位置动态；
3. 常要配合 prefix sum 或计数；
4. 很难像 GEMM 那样规则向量化。

在很多 MoE 系统里，真正先成为瓶颈的并不是 expert matmul，而是重排与通信。

## 4. Capacity Factor 与 kernel 的关系

MoE 里常用 capacity factor 控制每个 expert 最多接收多少 token。  
这看似是训练稳定性超参，但实际上也影响 kernel 路径：

1. 容量越宽松，负载可能更不均；
2. 容量越严格，drop token 风险更高；
3. expert batch 大小分布直接决定后续 GEMM 形状。

因此它不是纯模型超参，也是系统超参。

## 5. All-to-All 是什么，为什么难

在多卡 expert 并行里，不同 token 可能要被发往不同 GPU 上的 expert。  
这通常意味着 all-to-all 风格的通信：

1. 每个 rank 都向多个 rank 发 token；
2. 每个 rank 也从多个 rank 收 token。

这比 all-reduce 更复杂，因为：

1. 消息大小不均；
2. 负载可能严重偏斜；
3. 与 token permutation 耦合很深；
4. 更难重叠。

## 6. 稀疏系统为什么容易出现“理论省 FLOPs，实际不快”

因为虽然只激活部分 expert，但新增成本包括：

1. router；
2. token 重排；
3. dispatch / combine；
4. all-to-all；
5. 小而碎的 expert batch；
6. 热门 expert 拥堵。

只要这些额外成本没有被控制住，稀疏计算的收益就会被抵消。

## 7. 服务态 MoE 的额外问题

在线推理里，MoE 比训练更麻烦，因为：

1. batch 更小；
2. 请求更不规则；
3. 每步 decode token 少；
4. 热 expert 可能放大 tail latency；
5. 多租户会让路由分布更复杂。

**因此 MoE 服务系统特别依赖**：

1. expert placement；
2. 路由缓存；
3. dispatch kernel；
4. 通信调度。

## 8. 稀疏 attention 与 block-sparse 路线

除了 expert 稀疏，还有一类稀疏 kernel 出现在：

1. block-sparse attention；
2. 局部窗口 attention；
3. 检索增强 attention；
4. 动态剪枝路径。

这类方法试图减少不必要计算，但也会引入：

1. 更复杂的索引；
2. block map；
3. 稀疏布局管理；
4. 边界条件和实际有效算子粒度问题。

如果 block 设计和硬件不匹配，稀疏反而可能比稠密更慢。

## 9. 稀疏 kernel 的优化重点

与稠密 kernel 相比，稀疏 kernel 更应优先关注：

1. 索引和重排开销；
2. 局部性；
3. token / block 分组；
4. 热点负载均衡；
5. 通信路径。

也就是说，稀疏系统优化的第一优先级往往不是“更多 Tensor Core”，而是“更少无意义搬运与更少不均衡”。

## 10. 一个形象比喻

如果稠密模型像每个零件都要走同一条大装配线，那么 MoE 更像每个零件会被动态分发到不同的专业工位。好处是主装配线不用让所有工位同时开工，但坏处是零件分拣、转运和工位排队本身会变成新的瓶颈。MoE 系统真正难的地方，不在工位上的加工，而在整个分拣与物流系统是否高效。

## 11. 小结

稀疏、MoE 与路由 kernel 的核心，不是“把矩阵变稀疏”这么简单，而是如何在保持模型容量优势的同时，控制 token 重排、all-to-all、负载不均和小批次 expert 计算带来的系统开销。  
如果没有这层 kernel 与通信视角，MoE 很容易停留在“理论参数效率很高”的纸面优势，而无法转化成真实训练和服务系统里的可观收益。
