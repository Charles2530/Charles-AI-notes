# Runtime怎样判断当前输入能用哪个kernel

同一个 GEMM 接口可能收到连续张量、转置 view、量化打包权重和不同长度的 batch。它们的 shape 甚至可以相同，地址和数学约定却不同。运行时选择 kernel 时，第一件事是确认候选能不能正确处理输入，第二件事才是问它是否值得走快路径。两个问题混在一起，最容易把一个漂亮 benchmark 变成线上错误。

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

## 继续读

dispatch 规则需要和实现生命周期一起维护，可以继续看[自定义 op 与框架集成](custom-op-development-and-framework-integration.md)、[测试与回归](testing-regression-and-maintenance.md)和[shape bucket 建模](workload-modeling-and-shape-buckets.md)。

## 对照源码与延伸阅读

- [矩阵乘wrapper、autotune key和候选配置](https://github.com/triton-lang/triton/blob/cd2e01c6686d6503802e6cdfac56d5a5dfdb27a1/python/tutorials/03-matrix-multiplication.py)
- [架构条件决定候选是否可用](https://github.com/triton-lang/triton/blob/cd2e01c6686d6503802e6cdfac56d5a5dfdb27a1/python/tutorials/09-persistent-matmul.py)

## 用合法性矩阵保护 dispatch

为每个候选 kernel 建一张小矩阵，行是 dtype、layout、stride、mask、架构和 shape 边界，列是“可执行”“结果正确”“值得走快路径”。例如连续矩阵的 $K$ 是 128 的倍数，可能满足全部三列；非连续 stride 也许结果正确但需要额外 copy，最后一列应由实测决定。矩阵中的每个格子都对应一个测试或 benchmark，不能只依赖常见 shape。

```text
case_id | dtype | layout | shape | mask | legal | correct | profitable | path
```

`legal=false` 时必须安全回退；`correct=false` 的 case 应直接阻止发布；`profitable=false` 时可以保留实现，但不能让它污染线上快路径。把这三类状态分开，能避免“编译成功”被误读成“适合所有输入”。

## 加权性能比单个峰值更可靠

对候选 kernel 的选择可以使用真实调用分布加权：

\[
T_{weighted}=\sum_b q_b\,T_b,
\]

其中 $q_b$ 是 shape bucket 在目标流量中的比例，$T_b$ 包含转换、首次编译、kernel 执行和同步。离线基准中最快的 shape，如果在线只占 1%，不应成为整体收益的主要证据。应至少报告 warm cache 与 cold start 两种结果，以及 p50 和 p95。

## 发布前的回退演练

在真实流量或录制请求回放中，主动覆盖三类情况：快路径不支持的稀有 shape、编译缓存失效、运行时错误或设备降级。确认回退函数保持 dtype、mask、广播和梯度语义，并在日志中记录 `path=fast|generic`、转换耗时和失败原因。若回退比例超过预设阈值，应该回到 shape bucket 或输入规范设计，而不是继续增加特化分支。

## 自测

1. 两个输入 shape 相同但 stride 不同，为什么不能共享同一个 dispatch key？
2. 新 kernel 在 warm cache 下快 2 倍，但首次编译 800 ms，什么流量场景下它可能仍然不划算？
3. 快路径结果与通用路径只在非连续输入上不同，先查 legality、layout 还是模型精度？

答案应分别指向地址语义、请求分布与冷启动比例、输入 layout 的正确性测试。把这些问题加入 CI，才能让运行时优化持续可维护。

## 选择器应当像一份接口合同

一个可维护的 dispatch 表至少记录四类字段：

| 字段 | 要回答的问题 | 失败时的动作 |
| --- | --- | --- |
| legality | 这个 kernel 是否能读懂 dtype、layout、stride 和 mask | 直接回退 |
| correctness | 在边界 shape、空输入和非连续 view 上是否与参考实现一致 | 阻止发布 |
| profitability | 把转换、编译和同步算进去后是否更快 | 保留实现但不选它 |
| observability | 线上能否知道命中了哪条路径、付出了多少额外成本 | 补 trace 或计数 |

这张表比一个布尔值 `use_fast_kernel` 更有用。它把“不能执行”“能执行但结果错”“结果对却不划算”分开，也让回归测试和 benchmark 有明确的归属。

## 实现依据

候选配置和 shape 特化参考 [Triton 矩阵乘教程](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html) 与 [Persistent Matmul 教程](https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html)。运行时 fallback 的设计原则对照 [PyTorch 自定义算子教程](https://docs.pytorch.org/tutorials/advanced/python_custom_ops.html)，因为自定义算子需要同时说明 schema、设备支持和参考实现。vLLM 图只用于说明一个服务 runtime 里选择器处在什么位置；本文的合法性矩阵、阈值和加权公式是本站整理的工程模板。

自测时给同一 shape 构造一个连续 view 和一个转置后再 reshape 的 view。让两条路径都返回结果，再把 `stride`、转换时间和输出误差写入同一条记录。若只比较 kernel 时间，无法知道 wrapper 是否把收益吃掉。
