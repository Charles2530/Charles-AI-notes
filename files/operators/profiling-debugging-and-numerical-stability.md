# Profiling、调试与数值稳定性

很多 kernel 优化最后不是败在“不会写更快代码”，而是败在三件事上：没有正确测量，优化方向从一开始就错了；没有可靠验证，性能上去了但结果悄悄错了；没有处理数值稳定性，速度和精度之间出现不可接受的偏差。

成熟算子工程不是“会写 kernel”，而是建立从热点识别、性能定位、正确性验证到数值验收的完整闭环。

!!! note "初学者先抓住"
    Profiling 必须先于优化。否则你可能把时间花在看起来重要、实际不在热路径的 kernel 上。正确流程是先定位瓶颈，再改实现，再验证数值和端到端收益。

!!! example "有趣例子：修水管"
    家里水压低，不应该先随便换水龙头。要先看是总阀、水管堵塞、热水器还是某个弯头漏水。Profiling 就是这套排查流程，避免优化错位置。

## Profiling 先于优化

AI 系统里的直觉经常误导人：

1. 以为 GEMM 是瓶颈，实际热点是 layout transform；
2. 以为 decode 慢是模型太大，实际是 kernel launch 太碎；
3. 以为量化没提速是算法问题，实际是 dequant kernel 带宽受限；
4. 以为 occupancy 太低，实际卡在 shared memory bank conflict。

任何 serious optimization 都应从 profile 开始，而不是从改代码开始。优化前要明确：问题发生在系统调度、单 kernel、内存访问、微架构利用率，还是数值回退路径。

## 三层 Profiling 视角

| 层级 | 关注点 | 常用工具 |
| --- | --- | --- |
| 系统级 | 请求流、prefill/decode、通信、I/O、CPU/GPU 重叠 | Nsight Systems、serving trace、PyTorch profiler timeline |
| Kernel 级 | 单个 kernel 时间、launch 次数、shape 分布、调用路径 | PyTorch Profiler、框架 trace、自定义日志 |
| 微架构级 | 带宽、occupancy、warp stall、Tensor Core、cache 行为 | Nsight Compute、硬件计数器 |

只看系统级，知道哪段慢但不知道为什么；只看 kernel 级，知道哪个 kernel 慢但不知道是否被调度放大；只看微架构级，容易局部最优。有效工作流是在三层之间来回切换。

## Microbenchmark 要可信

微基准很容易夸大收益。较稳妥的原则是：

1. 充分 warmup，排除 JIT、memory pool、cache 和 graph capture 前置成本；
2. 固定 shape、dtype、layout 和输入分布；
3. 区分单 kernel 时间与端到端时间；
4. 覆盖真实 shape 分布，而不是只测最漂亮的整齐形状；
5. 报告平均值、p95/p99 和长尾 shape；
6. 明确是否包含编译、autotune、cache miss 和 fallback。

真实系统中 shape 往往不固定。一个 kernel 在某个整齐 shape 下极快，到了 batch、sequence、head dim、量化 group 变化时可能明显退化。优化结论必须按真实 workload 分桶报告。

## 关键指标怎么读

| 指标类 | 看什么 | 常见解释 |
| --- | --- | --- |
| 时间 | latency、throughput、launch 次数、TTFT、TPOT、step time | 判断端到端是否真的变快 |
| 资源 | SM 利用率、Tensor Core、DRAM throughput、shared memory、register、occupancy | 判断算力或带宽是否被用起来 |
| 结构 | warp stall、branch divergence、cache hit/miss、load/store efficiency | 判断为什么没跑满 |
| 数值 | 误差、溢出、归约差异、低精度 scale | 判断是否安全可替换 |

Roofline 是很好的先验判断器。若 kernel 算术强度低且 DRAM throughput 已接近上限，继续做算术优化收益很小；若理论上应是 Tensor Core 热点但 Tensor Core 利用率很低，说明 tile、layout 或 dtype 路径可能有问题。

## 正确性验证

任何优化都必须先通过正确性验证。基础做法包括：

1. 与参考实现逐元素对比；
2. 覆盖多种 shape、dtype、layout 和边界 mask；
3. 测随机输入、极值输入、全零、全同值、非连续 stride；
4. 测训练与推理路径；
5. 测多步迭代误差是否放大；
6. 固定 seed 和输入，保证可复现。

“看起来差不多”远远不够。一个归一化 kernel 某些边界块少算一个元素，单次误差可能很小，多层叠加后会明显偏航。低精度、softmax、attention、norm、optimizer kernel 都应有更严格的误差阈值和边界样本。

## 数值稳定性是一等公民

AI 算子不仅要算得快，还要在低精度、长序列和大动态范围下稳定。常见问题包括：

1. softmax overflow / underflow；
2. 方差计算误差；
3. FP16/FP8 累加损失；
4. quant scale 过大或过小；
5. 原子加法和归约顺序导致非确定性；
6. fused kernel 改变了高精度边界。

Softmax 通常写成：

\[
\mathrm{softmax}(x)_i =
\frac{e^{x_i-m}}{\sum_j e^{x_j-m}},\quad m=\max_j x_j
\]

减去最大值是为了避免指数爆炸。FlashAttention 这类分块实现还要维护在线最大值与归一化项，既减少 I/O，又保持数值精确。LayerNorm、RMSNorm、variance 计算则需要关注 Welford、FP32 accumulation 或其他稳定统计方法。

## 低精度验收

不同低精度类型风险不同：

| 类型 | 主要风险 | 验收重点 |
| --- | --- | --- |
| FP16 | 动态范围窄，易 overflow/underflow | loss scale、FP32 accumulate、极值输入 |
| BF16 | 范围大但尾数少 | 长期误差、归约和 optimizer 路径 |
| FP8 | 强依赖 scaling 和校准 | amax、scale 粒度、溢出/下溢、敏感层 |
| INT8/INT4 | 依赖量化尺度和校准数据 | scale、zero point、outlier、累加精度 |

低精度 kernel 不能只看单层误差。要看长序列、多层堆叠、训练反向、端到端任务指标和异常 bucket。尤其是 FP8/INT 相关 kernel，scale 的读取、更新、广播和融合边界都可能成为真实 bug 来源。

## 调试闭环

一个实用闭环是：

1. 用系统 trace 找到真实热点；
2. 用 kernel profile 判断瓶颈类型；
3. 用 microbenchmark 验证单点优化；
4. 用参考实现做正确性和数值对齐；
5. 用真实 workload 分桶验证端到端收益；
6. 把 shape、dtype、误差阈值和性能基线写入回归测试；
7. 记录硬件、驱动、编译器、框架和 kernel 版本。

算子优化的基本纪律是：没有 profiling，不谈优化；没有正确性，不谈性能；没有端到端验证，不谈上线收益。
