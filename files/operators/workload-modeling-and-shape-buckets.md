# 怎样从shape分布选出值得优化的路径

平均序列长度不能代表kernel看到的工作。一个请求会触发许多层和许多decode迭代，同一个请求的shape还在变化。建模时需要连接请求分布、实际调用次数与累计时间。

## 一百分之一的请求可以占一半时间

教学设定中，99个短请求各花1毫秒，1个长请求花99毫秒。长请求只占1%，却占总时间的一半。若目标是总计算成本，长请求值得关注；若目标是普通请求P50，结论可能不同。

再设长请求包含很多decode步，就会多次访问增长中的KV。只按请求入口记一次长度，会丢掉真实kernel调用分布。应记录每次调用的M/N/K、query/key长度、head结构、dtype与stride。

<!-- systems-workflow-visual -->
```mermaid
flowchart LR
  R["100个请求"] --> S["99短请求 · 各1ms"]
  R --> L["1长请求 · 99ms"]
  S --> A["累计99ms"]
  L --> B["累计99ms"]
  A --> C["各占总时间50%"]
  B --> C
```

请求频率和累计成本是两种权重，优化优先级取决于目标指标。

## Bucketing会同时改变调度与计算量

把长度33和63都放进64桶，可以复用执行shape，但有效token共96，分配槽位128，利用率75%。若分别按实际长度执行，padding少了，动态shape和调度开销却可能增加。

对于dense attention，按长度平方计的配对计算更敏感。两条真实长度的平方和为5058，若都按64完整计算则为8192；是否真的支付这些padding计算，还要看mask和变长kernel实现。

## 优化权重应使用累计时间

对桶b记录调用次数 $n_b$、基准延迟 $t_b$，简化串行总成本为 $\sum_b n_bt_b$。新路径比较的是 $\sum_b n_bt'_b$，而非每桶加速倍数的算术平均。

例如90次100微秒与10次2毫秒，总共29毫秒。前一桶快2倍，后一桶慢10%，新总量26.5毫秒，整体只快约9.4%。只汇报高频桶“快2倍”会掩盖代价。

## 桶必须包含会改变语义的条件

相同shape但不同stride，可能需要不同地址计算；相同长度但不同mask，可能采用不同attention实现；相同FP8数据但不同scale布局，也未必兼容。桶可以为统计合并一些相近输入，dispatch却不能省略正确性条件。

建立候选桶后，保留每桶代表性真实输入与边界输入。特别检查桶边界前后、空输入、尾块和非连续视图。真实流量迁移后重新计算频率与累计时间，旧热点不应永久占据特化预算。

![vLLM serving system overview](../assets/images/paper-figures/inference/vllm-system-overview-padded.png){ width="860" }

<small>图源：[vLLM 官方博客](https://vllm.ai/blog/2023-06-20-vllm)。原图表达请求调度、KV 管理和 worker 执行之间的关系；读图重点是把 shape bucket 看成 runtime 调度和 kernel 选择的前置条件。它不能证明某一组 bucket 适合所有业务，bucket 必须从真实流量统计中来。</small>

## 继续读

shape bucket 设计之后，继续看[kernel 成本模型](kernel-cost-models-and-selection-heuristics.md)、[runtime dispatch](runtime-dispatch-and-kernel-selection.md)和[性能反模式](performance-anti-patterns-and-failure-modes.md)。

## 对照源码与延伸阅读

- [按M/N/K选择配置并测量的实际教程](https://github.com/triton-lang/triton/blob/cd2e01c6686d6503802e6cdfac56d5a5dfdb27a1/python/tutorials/03-matrix-multiplication.py)
- [不同shape上的persistent候选](https://github.com/triton-lang/triton/blob/cd2e01c6686d6503802e6cdfac56d5a5dfdb27a1/python/tutorials/09-persistent-matmul.py)

## 从 trace 建立 shape 数据集

不要手工挑“看起来典型”的 M/N/K。在线或离线回放时，为每次 kernel 调用保存请求级关联信息：模型层、请求 ID、batch、有效长度、padding、dtype、layout、stride、设备、前后同步点和耗时。随后按语义条件分桶，再按累计时间排序。

建议保留三份样本：高频样本、累计时间最高的样本、边界/失败样本。只有第一份，容易优化到平均请求却忽略长尾；只有第二份，容易过度特化极少见的大请求。第三份用于验证 padding、空输入、奇数维度和非连续视图仍能安全回退。

## 何时新增一个 bucket

新增 bucket 应同时满足两个条件：它包含会改变正确性或候选配置的条件，而且在目标流量中有足够累计成本。若只是 shape 数值略有不同，但同一 kernel 和同一 tile 已经稳定，继续切桶会增加 dispatch 与缓存复杂度。

可以用一个简单门槛筛选：

```text
benefit = traffic_share * (baseline_time - candidate_time)
cost = implementation_time + compile_cache_cost + maintenance_risk
```

这个式子不是精确投资回报率，却能强迫记录“流量占比”和“工程成本”。候选路径还必须通过 correctness matrix；性能收益不足或边界不完整时，保留通用实现通常更稳妥。

## 变化后的流量要重新采样

上下文长度、batching 策略、模型版本和用户请求分布都会改变热点。发布新模型后，至少重新录制一段带 warmup、稳态和长尾的 trace，比较 bucket 频率、累计时间和回退比例。旧的 autotune 结果可以作为起点，但不能把旧流量的最优配置永久写死。

## 把 trace 变成可回放的 workload

每个 shape bucket 最少保留一条真实输入和一条边界输入，并保存模型层、请求阶段、有效长度、padding、dtype、layout、stride、设备和计时边界。回放时先用固定输入确认输出和 mask，再在相同 warmup、并发和同步条件下比较候选路径。这样性能差异可以回到某个 bucket，而不是停留在一张平均曲线。

回放结果最好同时给三种权重：按调用次数、按累计时间、按用户目标指标。高频小请求可能影响 P50，少量长请求可能决定 GPU 成本，最慢请求又可能决定 P99。没有说明权重时，“整体更快”这句话没有明确分母。

**来源与改编说明**：shape、tile 和基准边界参考 Triton 矩阵乘与 persistent matmul 教程，服务态 shape 和 KV 调度的背景参考 vLLM 官方博客。页面中的 99 个短请求与 1 个长请求是构造算例，真实 bucket 应来自可回放 trace。
