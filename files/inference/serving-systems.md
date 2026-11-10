# 一条生成请求怎样被推理引擎执行

把请求 A 的 prompt 设成 `[8,2,5]`，最多生成两个 token。它最后收到 `[7,1]`，模型只做了一次 prompt prefill 和一次 decode；服务端却经历了入队、KV 分配、采样、流式发送和结束回收。排查这类请求时，最容易把“已经发给用户”误记成“已经写进 KV”，或者把排队时间算进模型 forward。本页沿着 A 的两轮状态变化追一遍 vLLM，再看并发和长上下文怎样改变这条路径。

下面的函数和字段按 vLLM commit `fa3b9d13` 的 V1 引擎描述。源码版本持续变化，示意数字只用于理解状态；这里没有进行 GPU 性能测试。

```mermaid
flowchart LR
    Q[请求进入队列] --> S[scheduler 选择 token]
    S --> K[KV cache 分配或回收]
    K --> F[prefill / decode forward]
    F --> P[采样或验证]
    P --> O[流式输出]
    O --> U[更新已计算长度、KV 和结束原因]
    U -->|仍需生成| S
    U -->|EOS、长度上限或取消| R[释放或保留可复用块]
```

这条路径把用户可见输出和内部状态更新放在同一条时间线上。首 token 已经发出时，它通常还没有作为下一轮输入写入 KV；取消请求也要等状态机处理，不能只关闭 HTTP 连接。

## 入队时，模型还没有计算 prompt

设请求 A 的 prompt token 为 `[8,2,5]`，要求最多生成两个 token。刚入队时，已知 token 数为 3，已经计算的 token 数为 0，输出为空。它还需要模型权重、采样设置、位置编号、KV 分配以及停止条件。

引擎的 [EngineCore.step](https://github.com/vllm-project/vllm/blob/fa3b9d1377219247060d3c8b49eae199c9e7e154/vllm/v1/engine/core.py#L674)把一次推进拆成 `scheduler.schedule`、`model_executor.execute_model`、必要的 `sample_tokens`，以及 `scheduler.update_from_output`。这些是内部函数，不是建议应用直接调用的稳定公共 API。

请求排队时，用户已经在等待，但 GPU 可能还没处理它。鉴权、路由、tokenization 与网络也发生在模型计算之外。测慢请求应保留这些阶段的起止点，避免把所有时间都记为模型耗时。

## 调度器决定这一轮处理哪些 token

在 [Scheduler.schedule](https://github.com/vllm-project/vllm/blob/fa3b9d1377219247060d3c8b49eae199c9e7e154/vllm/v1/core/sched/scheduler.py#L699) 的注释中，调度逻辑以 `num_computed_tokens` 与需要处理的 token 数描述请求，而非为 prefill 与 decode 各写一套完全独立状态机。它给请求分配本轮 token 预算，让已经计算的位置追上待处理位置。

先忽略投机、多模态和异步 lookahead。对 A 来说，需要新算 $3-0=3$ 个 token。如果本轮预算足够，便安排完整 prompt；若只剩 2 个，可先计算前两个，下一轮继续。长 prompt 的 chunked prefill 就可以用这种“尚有位置未处理”的状态理解。

安排计算还必须获得 KV 空间。源码先调用 `kv_cache_manager.allocate_slots`，分配失败时会进入等待或抢占等分支。不能先完成 prefill、到最后才想起初始 KV 无处存放。被抢占的请求可能以后重算，这也是仅看输出吞吐会漏掉的额外工作。

## Prefill 处理三个位置，首个输出来自最后位置

Worker 对 `[8,2,5]` 做前向。每个位置产生各层 K/V，因果 attention 让位置 0 只关注自身、位置 1 关注前两个、位置 2 关注三个。最后位置的 logits 给出下一个 token 分布，假设采样得到 7。

这时可以向用户发出首个输出 7。无需为了“首个 token”再对完整 prompt 做一次额外 decode。Prompt 的 3 个位置已经写入 KV；刚采样的 7 还没有作为输入经过模型，所以它通常尚无自己的完整 K/V。

```text
已知 token        [8, 2, 5, 7]
已计算的前缀      [8, 2, 5]
已提交输出        [7]
下一轮需处理      [7]
```

已知序列长度、已计算长度和已分配容量是三个不同量。分页可能提前分配一整块，投机可能计算未被接受的候选，异步调度可能存在尚未回收的结果，因此生产实现还需要更多状态。

## Decode 读取历史，再把 7 写入 KV

下一轮只把新 token 7 送入模型，使用其正确位置编号。每层生成当前 query、key、value；query 读取历史 `[8,2,5]` 的 K/V，同时按实现处理当前 token，再得到新的隐藏表示和 logits。假设这次采样得到 1。

输出已是 `[7,1]`，达到两个 token 的上限，请求结束。一般无需为了给最后采样的 1 建 KV 再运行一轮，因为没有后续预测需要它。若要继续会话，具体复用边界由下一请求的前缀匹配与缓存策略决定。

[update_from_output](https://github.com/vllm-project/vllm/blob/fa3b9d1377219247060d3c8b49eae199c9e7e154/vllm/v1/core/sched/scheduler.py#L2260)负责把 worker 结果映射回请求并更新状态。该版本还处理已取消请求、异步旧结果、KV 接收失败以及投机拒绝后的已计算长度回退。一次 forward 返回张量，只是请求推进过程的一部分。

## 用一张小表对齐一次 step 的状态

把 A 请求写成 trace，可以直接看到“已经输出”和“已经写入 KV”发生在不同时间。表中数字只描述状态变化，不对应 vLLM benchmark。

| 轮次 | 本轮调度 token | `num_computed_tokens` 更新前 | 本轮模型得到的下一个 token | 输出序列 | 更新后状态 |
| --- | --- | ---: | ---: | --- | --- |
| 0 | `[8, 2, 5]` | 0 | 7 | `[7]` | prompt 已写入 KV，7 尚未作为输入计算 |
| 1 | `[7]` | 3 | 1 | `[7, 1]` | 7 的 KV 已写入，达到长度上限 |

如果把第一轮的 `num_computed_tokens` 直接记成 4，就会把刚采样的 7 错当成已经参与过一次 forward 的输入。这个差一位的状态会影响下一轮的位置编号、KV 分配和 prefix cache 命中。排查实现时，建议同时记录已计算 token 数、输出 token 数和当前逻辑序列长度，三列分别对应不同事件；具体字段名要以所固定的源码版本为准。

加入请求 B 后，表格可以继续说明 chunked prefill。假设 B 的 prompt 有 5 个 token，而本轮剩余预算只能容纳 2 个，那么第一轮先把 B 的前两个位置写入 KV，B 没有立即产生用户可见输出；下一轮再处理余下三个位置，最后一个位置的 logits 才能采样首 token。把“prompt 已经入队”写成“prompt 已经完成 prefill”，会把排队、分块和首 token 等待混成一个时间段。

这张表还给出一个简单的日志格式。生产 trace 至少保留请求 ID、调度轮次、计划 token 数、实际计算 token 数、已接受输出数、KV block 数和结束原因。遇到取消、抢占或投机拒绝时，再记录一次状态回退前后的数值。只保存最终文本，无法判断是调度没有给到 token，还是模型算了但输出处理丢了结果。

## 首 token 和后续间隔应怎样量

设请求在 $t_0$ 发出，首 token 到达为 $t_1$，最后 token 到达为 $t_N$，输出 N 个 token。

$$
\mathrm{TTFT}=t_1-t_0,\qquad
\mathrm{TPOT}=\frac{t_N-t_1}{N-1}\quad(N>1).
$$

若首 token 等待 300 ms，之后 100 个 token 平均间隔 40 ms，最后 token 约在 4.3 秒到达。单 token 输出的 TPOT 不适用。网络若一次发送多个 token，客户端测到的是 chunk 时间，不能据此声称知道每个 token 的精确时间戳。

vLLM 官方指标把这几个量分开暴露。`time_to_first_token_seconds` 对应 TTFT，`inter_token_latency_seconds` 记录相邻输出事件的时间差，`request_time_per_output_token_seconds` 则按一次请求计算

$$
\frac{\text{端到端时延}-\text{TTFT}}{\text{输出 token 数}-1}.
$$

下面是一条本站构造的流式 trace。客户端在 300 ms 收到 `7`，350 ms 收到包含 `8,9` 的一个 chunk，410 ms 收到 `10`。

| 输出事件 | 到达时间 | chunk 内 token | 相邻输出事件间隔 |
| --- | ---: | --- | ---: |
| 1 | 300 ms | `[7]` | — |
| 2 | 350 ms | `[8,9]` | 50 ms |
| 3 | 410 ms | `[10]` | 60 ms |

请求一共输出 4 个 token，因此 TTFT 为 300 ms，request-level TPOT 为 $(410-300)/(4-1)\approx36.7$ ms。客户端只能从这条 trace 得到 50 ms 和 60 ms 两个输出事件间隔；它不知道 `8` 和 `9` 在 350 ms 内各自何时生成。若把 50 ms 当成一个 token 的 TPOT，结论就已经超出了日志能证明的范围。[↗](https://docs.vllm.ai/en/stable/design/metrics/)

服务端还应单独记录排队、prefill、采样与发送。两段工作的 P95 不能简单相加得到总 P95，因为最慢的请求可能不是同一批；总时延分位数应由逐请求时间计算。

## 三条请求怎样共享一次迭代

设 A、B、C 分别还需生成 2、5、3 个 token。第一轮三条都参与；第二轮 A 完成；第三轮可以安排新请求 D 的工作，同时继续 B、C。它们无需等最长的 B 完成后再一起换批。

新请求 D 仍有 prefill 工作，调度器要在 token 预算与 KV 空间内安排它。如果一段长 prefill 连续占据设备 200 ms，已在流式输出的 B 可能出现明显停顿。切块为调度器提供插入其他工作的机会，代价是更多调度和可能较低的单块效率。

![vLLM serving system overview](../assets/images/paper-figures/inference/vllm-system-overview-padded.png){ width="900" }

<small>图源：[vLLM PagedAttention blog](https://vllm.ai/blog/2023-06-20-vllm)。这张图说明 scheduler、KV cache manager、CPU/GPU block allocator 和 worker 的关系。</small>

图中的 scheduler、KV manager 与 worker 对应这三种职责。持续批处理提高资源利用的同时，也可能改变单轮耗时和请求等待。大 batch 的总 token/s 提高，不代表每条请求的 token 间隔都缩短。

## KV 从几个 token 变成显存上限

对普通全注意力 Transformer，L 层、$H_{kv}$ 个 K/V heads、每 head 维度 d、每元素 b 字节，一个缓存 token 占

$$
m_{token}=2L H_{kv} d b.
$$

取 L=32、$H_{kv}=8$、d=128、BF16 的 b=2，结果为 128 KiB/token。32,768-token 单请求占 4 GiB，四条并发约占 16 GiB；若每条再生成 4,096 token，还会增加合计 2 GiB。权重、workspace、块表和临时状态需要另算。

这个公式描述逻辑总量。TP 下各卡可能分片或复制 K/V，滑动窗口、MLA 与混合模型也有不同布局，应按实际实现重新计算，不能把总量机械除以卡数。

PagedAttention 将逻辑连续的 K/V 放进固定大小的物理块。每块容纳 16 token 时，34-token 请求需要三块，末块暂时空出 14 个位置。分页减少连续预留与外部碎片，但仍有尾块浪费和索引成本。

## 请求结束与取消都要回收状态

A 完成后，块的活动引用可以释放；如果 prefix cache 选择保留公共前缀，块仍可能作为可复用状态驻留。另一条请求还引用的块不能提前释放，因此“请求结束”与“物理块立即全部回收”不总是一回事。

取消也需要进入引擎状态机。仅关闭 HTTP 流，worker 仍可能继续生成；立即释放正在使用的 GPU 块，又可能让执行中的 kernel 读到被重用的内存。对照版本的调度器会忽略已结束请求的过期输出，并有延迟释放相关处理，具体时机依异步与传输配置而变。

验证服务生命周期时，至少应覆盖正常 EOS、最大输出长度、排队时取消、执行中取消和缓存失效。它们直接影响资源是否能持续被新请求使用。

## 优化应落在这条执行路径的具体位置

共享前缀缓存减少重复 prefill，仍需后续 decode 读取历史 KV；投机解码用草稿和并行验证减少目标模型的串行轮次，还要回滚拒绝分支；量化减少某些张量载荷，收益取决于 kernel 与质量；拆分 prefill/decode 增加独立排程能力，也增加 KV 传输。

如果一个 Agent 任务总共 6 秒，其中工具占 4 秒、模型占 2 秒，把模型加速两倍后总耗时仍是 5 秒。先从 trace 找关键路径，再沿 [缓存与投机解码](caching-routing-and-speculative.md)、[GPU 执行](gpu-kernels-batching-and-memory-systems.md)或 [RAG 与工具系统](rag-agents-and-long-context-systems.md)继续分析。

## 用三个小题检查是否真的读懂了请求生命周期

1. 目标模型有 32 层、8 个 KV head、每个 head 维度 128，使用 BF16。一个 token 的逻辑 KV 大小是多少？如果 prompt 有 32,768 个 token，单请求的理论缓存量是多少？先按正文公式计算，再说明权重、workspace 和块表为什么不能放进这个数字里。
2. 某请求从发出到首 token 用时 300 ms，随后收到 100 个 token，最后一个 token 在 4.26 s 到达。分别计算 TTFT 和 TPOT，并说明客户端按网络 chunk 计时为什么可能得到不同的间隔。
3. 一个请求已经输出两个 token，第三个 token 还在 worker 执行时被取消。日志至少要留下哪些状态，才能判断 KV 是延迟释放、立即释放还是被 prefix cache 保留？把请求状态、执行轮次、KV block 引用和取消原因写成一行记录，再说明还缺哪一项时间戳。

第一题的结果应为 128 KiB/token 和约 4 GiB 的逻辑 KV。第二题的 TPOT 分母应取 99，因为首 token 没有一个前置的生成间隔。第三题没有唯一的字段名，重点是把取消事件、正在执行的 kernel、引用计数和最终回收事件分开记录。若只能回答“请求结束了”，还没有对上引擎状态机。

## 固定版本、源码入口与原始机制

- [vLLM EngineCore](https://github.com/vllm-project/vllm/blob/fa3b9d1377219247060d3c8b49eae199c9e7e154/vllm/v1/engine/core.py)，固定 commit `fa3b9d13`，从 `step` 追调度、执行、采样与状态更新。
- [vLLM V1 Scheduler](https://github.com/vllm-project/vllm/blob/fa3b9d1377219247060d3c8b49eae199c9e7e154/vllm/v1/core/sched/scheduler.py)，同一 commit，查看 token 预算、KV 分配、输出处理与拒绝回退。
- [PagedAttention 论文](https://arxiv.org/abs/2309.06180)解释分页管理，论文结果限定于其硬件和请求设置。
- [Orca](https://www.usenix.org/conference/osdi22/presentation/yu)解释 iteration-level scheduling 的系统背景。

## 源码版本和指标口径

我按 vLLM V1 的 [EngineCore](https://github.com/vllm-project/vllm/blob/fa3b9d1377219247060d3c8b49eae199c9e7e154/vllm/v1/engine/core.py) 和同一 commit 的 Scheduler 追了一遍状态更新，再用 [PagedAttention 博客](https://vllm.ai/blog/2023-06-20-vllm)与[论文](https://arxiv.org/abs/2309.06180)补上 KV 分页的背景。正文里的 A 请求、token 数和显存算例是为了读状态而写的，不是 vLLM 的 benchmark，也没有在本机声称测过 GPU 性能。源码持续变化，函数名和字段只对这个 commit 负责。

自测时给 A 增加一个共享前缀请求，画出两个请求的 KV 块引用、释放和回收时点。然后解释为什么 prefix cache 省掉了部分 prefill，decode 仍然要读取历史 KV。
