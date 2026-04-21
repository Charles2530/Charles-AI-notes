# Profiling、调试与数值稳定性

很多 kernel 优化工作最后并不是败在“不会写更快代码”，而是败在三件事上：

1. 没有正确测量，优化方向从一开始就错了；
2. 没有可靠验证，性能上去了但结果悄悄错了；
3. 没有处理数值稳定性，速度和精度之间出现了不可接受的偏差。

因此真正成熟的算子工程，不是“会写 kernel”，而是“会建立一条从热点识别、到性能定位、到正确性验证、到数值验收的完整闭环”。

## 1. 为什么 profiling 比直觉更重要

在 AI 系统里，直觉经常会误导人。例如：

1. 你以为 GEMM 是瓶颈，实际热点是 layout transform；
2. 你以为 decode 很慢是模型太大，实际是 kernel launch 太碎；
3. 你以为量化没提速是算法问题，实际是 dequant kernel 带宽受限；
4. 你以为 occupancy 太低，实际真正卡在共享内存 bank conflict。

这也是为什么任何 serious optimization 都应从 profile 开始，而不是从改代码开始。

## 2. Profiling 的三层视角

**通常需要同时看三层**：

1. **系统级**：请求流、prefill/decode、通信、I/O、CPU/GPU 重叠；
2. **kernel 级**：单个 kernel 的时间、launch 次数、形状分布；
3. **微架构级**：带宽、occupancy、warp stall、Tensor Core 利用率、cache 行为。

### 2.1 为什么只看一种 profile 不够

只看系统级，你知道某一段慢，但不知道为什么慢；  
只看 kernel 级，你知道哪个 kernel 慢，但不知道是不是被调度策略放大；  
只看微架构级，又容易掉进局部最优而忽略全局瓶颈。

真正有效的工作流，是这三层来回切换。

## 3. 常见 profiling 工具各自看什么

### 3.1 系统级时间线工具

例如 `Nsight Systems`、某些 serving trace、PyTorch profiler 时间线视图。  
**适合回答**：

1. GPU 是否真的忙；
2. CPU 发 kernel 是否有空洞；
3. 通信与计算是否重叠；
4. kernel launch 是否过碎；
5. prefill 与 decode 时间分别占多少。

### 3.2 微架构与 kernel 指标工具

例如 `Nsight Compute`。  
**适合回答**：

1. 这个 kernel 是算力受限还是带宽受限；
2. occupancy 如何；
3. register、shared memory 使用如何；
4. warp stall 原因是什么；
5. Tensor Core 是否真的被有效利用。

### 3.3 框架级 profiler

例如 `PyTorch Profiler`。  
**适合回答**：

1. 哪些 Python / C++ / kernel 调用最耗时；
2. 图捕获前后有什么差异；
3. 自定义 op、Triton kernel、库调用在整体路径上各占多少；
4. 是否有很多细碎框架开销。

## 4. 做 microbenchmark 的基本原则

很多 kernel 优化论文或博客会给出巨大加速比，但工程里最容易出问题的正是微基准方法本身。较稳妥的做法包括：

1. 充分 warmup；
2. 固定 shape 与 dtype；
3. 区分单 kernel 时间与端到端时间；
4. 固定输入布局；
5. 用多个 batch / sequence / head dim 组合测试；
6. 报告平均值和尾部结果。

### 4.1 为什么 warmup 很重要

**因为第一次运行往往掺杂**：

1. JIT 编译；
2. kernel cache 填充；
3. memory pool 初始化；
4. graph capture 前置成本；
5. page fault 或缓存预热。

如果不区分 warmup，很多 benchmark 没有参考意义。

### 4.2 为什么只测“最优 shape”会误导

真实系统中 shape 往往不固定。  
一个只在某一组整齐 shape 下极快的 kernel，到了：

1. batch 变化；
2. 序列变化；
3. head dim 变化；
4. 量化 group 变化；

就可能明显退化。  
因此 benchmarking 要尽量覆盖真实分布，而不是只报最漂亮的一点。

## 5. 关键性能指标应该怎么看

### 5.1 时间类

1. 单次 kernel latency；
2. 吞吐；
3. launch 次数；
4. TTFT、TPOT；
5. 训练中的 step time。

### 5.2 资源类

1. SM 利用率；
2. Tensor Core 利用率；
3. DRAM throughput；
4. shared memory 使用；
5. register 使用；
6. occupancy。

### 5.3 结构类

1. warp stall 原因；
2. branch divergence；
3. cache hit/miss；
4. load/store efficiency；
5. memory transaction 粒度。

这些指标不是都要追到极致，而是要和算子类型对应着看。

## 6. Roofline 在实战里怎么用

Roofline 不只是教材图，它非常适合作为“先验判断器”。  
如果一个 kernel 的算术强度低，而 DRAM throughput 已接近上限，那么继续做算术优化收益可能很小；  
如果 Tensor Core 利用率很低，而理论上它应是 GEMM 型热点，则说明 kernel 结构可能还有明显浪费。

### 6.1 用 Roofline 避免错误优化

它最有价值的地方，是防止你在错误方向上投入太多时间。例如：

1. bandwidth-bound 的 norm kernel，不该优先去想更复杂 MMA；
2. compute-bound 的大 GEMM，不该只盯着 allocator；
3. 主要由 launch 碎片化导致的 decode kernel，不该只从单 kernel FLOPs 出发。

## 7. 数值正确性验证

任何优化都必须经过正确性验证。最基础的做法包括：

1. 与参考实现逐元素对比；
2. 测多种 shape；
3. 测随机输入和边界输入；
4. 测不同 dtype；
5. 测训练与推理路径是否一致。

### 7.1 为什么“看起来差不多”远远不够

很多数值 bug 在小样本上不明显，但在：

1. 长序列；
2. 大 batch；
3. 极值输入；
4. 多步迭代；

下会被放大。  
例如一个归一化 kernel 某些边界块少算了一个元素，单次看起来误差很小，但多层叠加后就可能明显偏航。

## 8. 数值稳定性为什么是算子设计的一等公民

很多 AI 算子不是“算出来就行”，而是要在低精度、长序列和大动态范围下保持稳定。常见问题包括：

1. softmax overflow / underflow；
2. 方差计算的数值误差；
3. 低精度累加损失；
4. 量化 scale 过大或过小；
5. 原子加法和归约顺序引入的不稳定。

### 8.1 Online softmax 的稳定性逻辑

**softmax 一般会写成**：

$$
\mathrm{softmax}(x)_i
=
\frac{e^{x_i - m}}{\sum_j e^{x_j - m}},
\quad
m = \max_j x_j.
$$

减去最大值是为了避免指数爆炸。  
在分块计算里，维护在线最大值与归一化项，同样是为了兼顾性能与稳定性。

### 8.2 Welford 与统计量计算

对于方差、均值等统计量，很多高性能实现会采用更稳定的在线算法，而不是简单先平方再减均值。  
这类策略在 layernorm、batchnorm、rmsnorm 相关实现里都很重要。

## 9. 低精度下该如何思考精度

**现代 AI 系统常见数据类型包括**：

1. FP32；
2. FP16；
3. BF16；
4. FP8；
5. INT8 / INT4。

**不同类型的风险不同**：

1. FP16 范围较窄，容易 overflow / underflow；
2. BF16 范围较宽但尾数较少；
3. FP8 更依赖 scaling 与校准；
4. INT 系列依赖量化尺度和累加精度设计。

### 9.1 低精度 kernel 的关键判断

1. 输入能否低精度存储；
2. 累加是否需要更高精度；
3. 输出是否需要回写更高精度；
4. scale / zero-point 放在哪一级；
5. fusion 后是否引入额外误差路径。

这也是为什么“混合精度”常常比“纯低精度”更现实。

## 10. Race condition 与同步错误

很多自定义 kernel 的 bug 不是数学公式错，而是同步错：

1. 少了 `__syncthreads()`；
2. warp 级假设超出了实际范围；
3. 异步拷贝尚未完成就使用数据；
4. 原子更新和归约顺序不合理；
5. 多 block 写同一输出却缺少正确同步语义。

**这些错误有个共同特点**：  
可能在小数据、少运行次数下几乎不出现，但在高并发或不同 GPU 上突然显形。

## 11. Debug kernel 时的常见策略

### 11.1 先缩小问题规模

先在最小 shape 上复现，让你能：

1. 打印中间值；
2. 对照参考实现；
3. 缩小越界范围；
4. 看清边界块行为。

### 11.2 先保正确，再追性能

一个常见错误是边写边优化，最终同时失去正确性与可读性。更稳妥的顺序通常是：

1. 先写朴素但对的版本；
2. 再逐步引入向量化、shared memory、warp primitive、fusion；
3. 每一步都保留可验证基线。

### 11.3 对 Triton 与 CUDA 的调试差异

Triton 的优势是表达更短，往往更容易验证块逻辑；  
CUDA 的优势是你能更细地控制执行，但调试复杂同步和越界时通常更痛苦。  
无论哪一种，保留参考实现都极其关键。

## 12. 端到端 profile 与单 kernel profile 的关系

有时单 kernel 优化带来 `30%` 提升，但端到端几乎不变。常见原因包括：

1. 它在整体路径里只占很小比例；
2. 新 kernel 增加了数据准备或转换开销；
3. 调度或通信成了新瓶颈；
4. decode 场景下 tail latency 被别的路径主导。

**这再次说明**：单 kernel profile 是必要的，但不充分。

## 13. 性能报告应该怎么写才可信

一个较可信的性能报告通常应包括：

1. GPU 型号；
2. shape 分布；
3. dtype；
4. baseline；
5. warmup 与重复次数；
6. 平均值与波动范围；
7. 端到端与单 kernel 两种视角；
8. 正确性误差阈值。

否则所谓“提速 2 倍”很容易因为 benchmark 条件不同而失去意义。

## 14. 优化循环的最佳实践

**一个成熟的优化循环大致如下**：

1. 先确定业务热点；
2. 系统级 profile 定位瓶颈；
3. kernel 级 profile 分解时间；
4. 微架构级 profile 看资源与 stall；
5. 设计改动；
6. 正确性与数值验证；
7. 再做端到端回归。

这套流程虽然不花哨，但能避免大量无效劳动。

## 15. 一个 Attention kernel 的典型排查例子

假设你发现一个 decode attention kernel 很慢，较稳妥的排查顺序可能是：

1. 看系统时间线，确认瓶颈确实在 attention；
2. 看是否大量小 kernel 和 layout transform 混在一起；
3. 看 Nsight Compute，判断是 bandwidth-bound 还是 launch-bound；
4. 看 paged KV 访问是否导致访存不连续；
5. 看 head dim / block size / page size 是否匹配；
6. 再决定是优化 kernel 本体，还是先优化调度与布局。

如果一开始就重写 kernel，很可能方向错了。

## 16. 一个形象比喻

优化 kernel 就像改赛车。你不能只换一个更强的引擎就期待整车更快，还必须看轮胎抓地、传动系统、散热、赛道条件和计时方式。Profiling 是测速和遥测，正确性验证是安全检测，数值稳定性则像发动机在高转速下会不会爆缸。只有这些都建立起来，所谓“跑得更快”才是可信结论。

## 17. 小结

Profiling、调试与数值稳定性，不是 kernel 优化的附属流程，而是主体。没有 profile，你不知道瓶颈在哪里；没有正确性验证，你不知道优化是否可信；没有数值稳定性控制，你的性能改进可能只是用错误换来的假象。

真正成熟的算子工程，最终比拼的不是谁能写出最炫的 kernel，而是谁能建立一条可靠、可复现、可扩展的优化闭环。这条闭环一旦建立，CUDA、Triton、CUTLASS、图编译器这些工具才会真正变成系统能力，而不是零散技巧。

## 18. Nsight Systems 和 Nsight Compute 应该怎么配合

很多团队会在两个工具之间来回切换，却没有形成稳定流程。更高效的方式通常是：

1. **先用 Nsight Systems 看“什么时候慢”**；
2. **再用 Nsight Compute 看“为什么这个 kernel 慢”**。

### 18.1 Nsight Systems 更擅长回答什么

1. **CPU 调度是否跟不上**；
2. **kernel launch 是否太碎**；
3. **不同 stream 是否真正重叠**；
4. **NCCL、数据搬运和计算谁在卡关键路径**；
5. **真实请求链路里 prefill、decode、post-processing 谁占主导**。

### 18.2 Nsight Compute 更擅长回答什么

1. **这个 kernel 是 compute-bound 还是 memory-bound**；
2. **warp stall 主要来自什么**；
3. **访存 transaction 是否健康**；
4. **Tensor Core / vectorized load / shared memory 使用是否合理**；
5. **寄存器、occupancy、L2 命中和指令构成如何**。

### 18.3 一个常见错误流程

一上来就对某个 kernel 做深度分析，却没先确认它是否真在关键路径上。这样很容易优化一个热点排序第三或第四的算子，最后端到端几乎没变化。

## 19. 先做时间线，再做指标解释

如果一个系统“变慢了”，最先要建立的是时间线，而不是立即盯某个百分比指标。

### 19.1 一条更实用的分析顺序

1. **把请求或训练 step 切成阶段**；
2. **看每个阶段时间是否稳定**；
3. **找出真正占壁钟时间的热点 kernel 或同步点**；
4. **再下钻到该 kernel 的硬件指标**。

### 19.2 为什么顺序不能反过来

因为某个 kernel 的 `dram utilization` 看着不高，不代表它需要被优先优化；它可能只占总时间的 3%。时间线先行，是为了避免局部最优。

## 20. Kernel 指标最容易被误读的地方

### 20.1 `occupancy`

高 occupancy 不等于快，低 occupancy 也不自动等于问题。它只能说明资源驻留情况，不能直接说明是否把带宽和算力都用好。

### 20.2 `SM utilization`

很多监控里的 GPU utilization 只是粗粒度“是否忙”，并不等于真正的算力利用。某些 memory-bound kernel 也会显示 GPU 很忙。

### 20.3 `dram throughput`

带宽高并不一定是好事，可能说明你在做很多无用搬运；带宽低也不一定是坏事，可能说明你的瓶颈在指令依赖或 launch。

### 20.4 `tensor core utilization`

它是重要信号，但也不能孤立看。某些服务态 decode 或轻量融合算子，本来就不是靠 Tensor Core 吃饭。

## 21. 用 Roofline 把指标串起来

Roofline 的价值不在于画一张漂亮图，而在于帮助你把两个问题接起来：

1. **这个 kernel 的算术强度大致在哪**；
2. **它离带宽上限或算力上限还有多远**。

### 21.1 实际使用时怎么做

1. **先估算读写字节数和 FLOPs**；
2. **算出 arithmetic intensity 的大致范围**；
3. **结合 profile 判断它更像哪一侧受限**；
4. **再决定是改数据流、改 tile，还是改 fusion**。

### 21.2 为什么这比只看一个 profiler 页面更好

因为它能把“指标现象”转成“优化方向”。否则你可能知道带宽不高、occupancy 也一般，但还是不知道该动哪里。

## 22. 数值问题排查最好做成固定矩阵

数值异常最怕“偶发”。真正成熟的做法是把测试矩阵固定下来，而不是每次临时想几个 shape。

### 22.1 建议至少覆盖

1. **极小 shape**；
2. **典型 shape**；
3. **极端长序列或大批量 shape**；
4. **边界非对齐 shape**；
5. **不同 dtype**；
6. **随机值、全零、全常数、超大值、超小值、异常值混入**；
7. **训练态和推理态两条路径**。

### 22.2 为什么边界 shape 特别重要

很多 kernel 的 correctness bug 和数值 bug 都只在尾块出现，例如：

1. **mask 漏了一行或一列**；
2. **最后一块 reduction 累加精度不足**；
3. **反量化尾部没对齐**；
4. **某些 block 只对齐到半个向量宽度**。

## 23. 服务系统里的 profile 和离线 microbenchmark 不一样

在线推理或 agent 系统里，性能问题往往跨越多层。单独看 kernel benchmark 常常不够。

### 23.1 服务态应额外关心什么

1. **TTFT 与 TPOT 的分解**；
2. **CPU runtime 调度时间**；
3. **KV cache 分配和回收时间**；
4. **外部工具或 RPC 对关键路径的干扰**；
5. **混合长度请求下 kernel 形状的演化**。

### 23.2 为什么要把 trace 和业务请求绑定

如果只看 GPU trace，你经常不知道慢的是哪类请求。更稳妥的做法是把 request id、模型路由、长度分布、cache 命中等元数据写进 trace 标记中。

## 24. 性能回归最好做成“差分实验”

当某次改动让系统慢了，不要直接看最新 profile。更高效的方法是做**before/after 差分**。

### 24.1 差分实验至少要固定

1. **输入 shape 集合**；
2. **随机种子**；
3. **数据布局和权重版本**；
4. **GPU 型号与驱动版本**；
5. **编译缓存状态**。

### 24.2 差分时重点比较什么

1. **kernel 时间结构变化**；
2. **新增的中间 kernel 或同步**；
3. **寄存器和 occupancy 变化**；
4. **访存 transaction 和 L2 行为变化**；
5. **端到端是否真变差，而不是单点 benchmark 波动**。

## 25. 数值稳定性的典型事故模式

### 25.1 低精度累加误差被长序列放大

在 attention、norm、optimizer 或长 reduction 场景中，序列越长、块越多，累加误差越容易积累。

### 25.2 Reorder 改变了归约顺序

很多“只是优化了 layout / tile”的改动，实际上也在改变求和路径，从而改变误差分布。

### 25.3 量化元数据不匹配

scale、group size、zero-point 或 block mapping 有一个偏移，结果就会表现成“偶尔精度差”，但根因其实是索引错位。

### 25.4 训练推理两套 kernel 不一致

某些模型在训练时用高精度参考路径，推理时换成量化或 fused kernel，最终线上数值偏移才第一次暴露。

## 26. Debug 一个疑难 kernel 的实战顺序

1. **先做最小可复现**，剥离框架和大图影响；
2. **和参考实现逐元素比对**；
3. **缩小到第一个出错的 tile / block / token**；
4. **对该点打印中间统计量或做分阶段核对**；
5. **再看这个错误是否只在特定 shape / dtype / alignment 出现**；
6. **最后才回到性能优化版本继续迭代**。

### 26.1 为什么别一边 debug 一边继续融合

因为融合一多，中间态消失，问题定位难度会指数上升。很多时候应先退回较朴素但正确的版本，把正确性边界摸清，再逐步把优化加回去。

## 27. 把 profiling 和 correctness 放在同一张发布单里

每次上线关键 kernel，建议发布单至少包含：

1. **目标 shape family**；
2. **端到端收益**；
3. **最重要的 profiler 证据**；
4. **正确性测试覆盖范围**；
5. **数值误差阈值**；
6. **已知退化场景与回滚条件**。

### 27.1 为什么这能显著降低线上事故

因为很多事故不是没人测，而是性能团队、模型团队、推理团队各自只看自己的一半。把 profile 和 correctness 绑在同一张单子上，能迫使团队承认：**快和对，必须一起交付**。

## 28. 小结

Profiling、调试与数值稳定性不是三份附属工作，而是同一个闭环：**先证明慢在哪里，再解释为什么慢；先证明结果对，再证明它在边界上也稳定。** 只有把时间线、硬件指标、差分实验和固定测试矩阵做成长期流程，团队才能持续演进复杂算子，而不是靠运气保持正确。
