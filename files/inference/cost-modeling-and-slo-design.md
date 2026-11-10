# 一小时GPU费用怎样变成成功任务成本

假设一组GPU每小时费用20元，某小时完成10000个请求，其中8000个满足任务质量要求。平均每请求成本0.002元，每成功任务成本0.0025元。两个分母都有效，但回答不同问题。

## Token成本先固定统计范围

若这一小时生成360万输出token，输出口径每百万token成本为 $20/3.6\approx5.56$ 元。这个小时费用已包含处理输入、排队期间资源驻留等开销，不能再把同一GPU费用重复加进另一项。

输入和输出token计算成本不同，cache命中又会减少部分输入计算。报告token成本时注明总token、输入token还是输出token，以及是否含闲置容量、CPU与网络。

## SLO把“完成”进一步缩成“可用完成”

若8000个正确请求中只有7000个满足延迟要求，每个SLO内成功任务成本约0.00286元。为了提高原始吞吐而增加排队，可能降低这个可用完成量。

TTFT、TPOT、整请求时间与任务成功率应分别定义。若回答有N个token，平均TPOT常按首token到末token的时长除以N-1；N小于2时需要另行处理。

## 一条长请求有哪些不同成本

32k prompt会支付prefill计算，并在后续生成期间持续占KV。按32层、8个KV头、头维128、BF16计算，主体KV约4GiB。长输出还延长驻留时间并增加decode轮数。

因此同样总token数，长输入短输出和短输入长输出也可能有不同成本曲线。Prefix命中可减少prefill，不能消除后续KV读取；权重量化与KV量化又分别作用于不同预算。

## 质量与重试怎样改变经济性

教学设定中，方案A单次成本1、成功率0.9，方案B成本0.8、成功率0.6。若每次独立重试且成功率不变，期望成功成本分别约1/0.9和0.8/0.6，B反而更贵。

真实重试不一定独立，可能保留同一错误或升级模型，所以应直接记录任务链的总调用和最终结果。不能仅凭单次token便宜判断用户任务成本。

## 分位数不能直接相加

队列P99加prefill P99通常不等于TTFT P99，因为两项极端值可能发生在不同请求。应按同一请求合计阶段时间，再求分位数；异步重叠时还需使用实际关键路径。

若SLO规定99%请求满足阈值，一万请求的允许失败数是100。这个计数可以用于错误预算，但还应说明时间窗口、超时与取消是否计入分母。

## 用约束筛选配置

先排除不满足质量与延迟的配置，再比较可用完成量和成本。容量、batch、量化与路由都可能改变分布，需用同一到达过程与长度集合复测。价格和实测性能随环境变化，本文数字只用于统一单位与分母。

![vLLM serving system overview](../assets/images/paper-figures/inference/vllm-system-overview-padded.png){ width="860" }

<small>图源：[vLLM 官方博客：Easy, Fast, and Cheap LLM Serving with PagedAttention](https://vllm.ai/blog/2023-06-20-vllm)。原图表达 scheduler、KV cache manager、allocator 和 GPU worker 的关系；读图重点是把成本拆到 queue、prefill、decode、KV 与调度层，而不是只算单 token 价格。它不能证明 vLLM 在所有 workload 下成本最低。</small>

## 对照源码与延伸阅读

- [请求执行与输出产生的真实边界](https://github.com/vllm-project/vllm/blob/fa3b9d1377219247060d3c8b49eae199c9e7e154/vllm/v1/engine/core.py)

## 把每个请求记录成一条时间线

成本和 SLO 只能在同一个请求的事件被串起来时计算。建议至少保留以下时间戳和计数：

```text
request_id, arrival_ts, queue_start_ts, prefill_start_ts,
first_token_ts, last_token_ts, finish_ts,
input_tokens, output_tokens, cached_input_tokens,
queue_ms, prefill_ms, decode_ms, total_ms,
status, retry_count, quality_pass
```

阶段关系可以写成

\[
T_{\text{total}}=T_{\text{queue}}+T_{\text{prefill}}+T_{\text{decode}}+T_{\text{postprocess}}.
\]

若第一个输出 token 在 prefill 后立即产生，TTFT 近似为 \(T_{\text{queue}}+T_{\text{prefill}}+T_{\text{first-decode}}\)；TPOT 则应使用首 token 后的生成时间和实际新增 token 数。对于只输出一个 token 的请求，TPOT 没有正常的 \(N-1\) 分母，应单独标注。

不要把平均输入长度乘以平均输出长度来估计某个请求的耗时。prefill 更接近输入 token 的批量计算，decode 反复读取已有 KV 并追加新 KV，二者对 batch、长度和内存带宽的敏感性不同。

## 一次可复用的 benchmark 协议

1. **固定请求集**：保存 prompt、输入 token 数、最大输出长度、停止条件和质量判定器的版本。
2. **固定服务状态**：记录模型 commit、运行时版本、GPU 型号、量化配置、并行方式和 KV 配置。
3. **预热**：丢弃预热请求，但记录预热次数和是否触发编译、CUDA graph 或内存池分配。
4. **复现到达过程**：分别测 closed-loop（下一个请求等上一个完成）和 open-loop（按时间戳注入），不能用前者冒充线上排队表现。
5. **保存逐请求记录**：至少保存 TTFT、TPOT、总时延、输入/输出 token、错误、取消、重试和质量结果。
6. **按请求求分位数**：先求每条请求的关键路径，再计算 p50/p95/p99；不要把各阶段的 p99 相加。
7. **重复与置信区间**：不同 seed 或不同到达片段重复测量，并报告样本数、时间窗口和异常请求处理规则。

一个最小的 CSV 可以长这样：

```text
request_id,input_tokens,output_tokens,ttft_ms,total_ms,
queue_ms,prefill_ms,decode_ms,status,quality_pass
r001,2048,128,84.2,612.5,11.0,48.7,528.0,ok,1
r002,8192,64,211.4,503.8,80.1,119.2,304.5,ok,1
r003,2048,128,420.3,1100.2,347.0,49.1,681.8,timeout,0
```

第三条请求的总时延上升主要来自排队和重试路径。只比较 decode kernel 的平均时间，会漏掉真正决定用户体验的部分。

## 把 SLO 写成可检查的事件

“低延迟”必须拆成条件。例如：

| SLO | 分母 | 目标 | 失败处理 |
| --- | --- | --- | --- |
| TTFT | 所有收到的请求 | p95 ≤ 300 ms | 超时和取消仍进入分母 |
| TPOT | 至少生成 2 个 token 的成功请求 | p95 ≤ 40 ms/token | 单 token 请求单独统计 |
| 成功完成 | 所有业务任务 | ≥ 98% | 质量失败、工具失败和超时都算失败 |
| 成本 | 业务任务 | 每个 SLO 内成功任务 ≤ 0.003 元 | 重试链按最终任务合并 |

如果质量判定需要外部 evaluator，应把 evaluator 版本、延迟和失败本身记录下来。一个更快但质量判定器漏检的服务，不能据此宣称 SLO 改善。

### 从到达率估计容量，而不是只看单请求吞吐

令到达率为 (lambda)，平均服务时间为 (S)，并发资源数为 (c)，粗略利用率为

\[
\rho\approx\frac{\lambda S}{c}.
\]

当 (ho) 接近 1 时，哪怕平均服务时间没有变化，队列等待也可能急剧上升。这个关系用于筛选压测区间，不是完整的多阶段服务模型；真实系统还要把请求长度、batch 调度、共享前缀、KV 容量和错误重试分开测量。

建议画三张图：

- 到达率—TTFT p95，观察队列何时开始非线性增长。
- 并发—有效 tokens/s，观察吞吐是否只是把延迟转移给排队。
- 输入长度桶—KV 峰值和错误率，观察长上下文是否触发新的瓶颈。

## PagedAttention 应放在哪个成本项里

vLLM 的 PagedAttention 解决的是 KV cache 的管理和分配问题，结合 scheduler、block allocator 与 GPU worker，使不同请求的 KV 可以按块管理。它能减少连续内存分配带来的浪费，并提高并发利用率，但不能保证所有 workload 的端到端成本最低。

阅读 vLLM 材料时，先把收益拆成三种可能：

1. **容量收益**：同一显存容纳更多活动序列。
2. **排队收益**：可调度的请求更多，等待时间降低。
3. **计算收益**：kernel 或 batch 形状改变，实际 tokens/s 提升。

只有第三种才直接是算力加速；前两种可能表现为更低的 TTFT 或更高的可用完成量。实验中应同时记录 KV 使用量、活动序列、排队时长和 GPU 利用率。

## 成本决策表

| 观察 | 可能原因 | 下一项最小实验 |
| --- | --- | --- |
| 显存下降，p95 不变 | 原瓶颈在排队或 CPU | 固定到达率，比较 queue/prefill/decode 分段 |
| 平均吞吐上升，p99 变差 | batch 或排队更激进 | 固定质量和 p99 阈值，扫描并发上限 |
| weight-only 量化省显存但 tokens/s 不变 | 运行时仍走 dequant 或非融合 kernel | 检查实际 kernel、权重布局和算子回退 |
| prefix 命中后 TTFT 下降，长输出成本不变 | 只减少了 prefill | 分开报告 cached input、decode 和总成本 |
| 单请求很快，线上成本很高 | 到达率和重试造成排队 | 用生产到达时间戳回放 open-loop workload |

## 来源与改编说明

- [vLLM PagedAttention 官方博客](https://vllm.ai/blog/2023-06-20-vllm)用于解释 KV cache、block allocator、scheduler 和 GPU worker 的关系；文中案例不能外推为所有硬件和请求集的统一 benchmark。
- [vLLM 源码中的请求执行入口](https://github.com/vllm-project/vllm/blob/fa3b9d1377219247060d3c8b49eae199c9e7e154/vllm/v1/engine/core.py)用于核对请求生命周期边界。
- [Hugging Face Transformer KV cache 文档](https://huggingface.co/docs/transformers/en/kv_cache)适合补缓存对象、增长和实现接口；本文的成本公式仍需在目标运行时测量。

## 自测

1. 为什么 queue p99、prefill p99 和 decode p99 不能直接相加得到总 p99？
2. 一个请求输入 8k token、输出 4 token，另一个输入 512 token、输出 256 token；为什么不能只按总 token 数比较成本？
3. 如果 weight-only 量化降低了显存但 tokens/s 不变，下一步应检查哪些运行时证据？
4. 请设计一个压测矩阵，能区分“KV 容量增加”与“kernel 计算加速”。

具体请求阶段的 trace 见[推理服务系统](serving-systems.md)，缓存与投机带来的额外状态见[缓存、路由与投机解码](caching-routing-and-speculative.md)。先把阶段边界对齐，再讨论 SLO 是否改善。
