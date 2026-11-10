# GPU性能异常的分层排查

这个页面是诊断索引。先固定模型、输入shape、dtype和软件版本，再比较异常是否出现在单卡kernel、完整单卡流程，还是多卡执行中。没有这个分界，同一种“吞吐下降”可以被错误解释成完全不同的问题。

## 从最小复现确定范围

| 观察 | 可以支持的假设 | 下一项对照 |
| --- | --- | --- |
| 相同单卡kernel也变慢 | 设备状态、编译或执行路径变化 | 固定输入比较时钟、资源与实际kernel |
| 单kernel正常，完整层慢 | 转换、分配、同步、launch或图路径增加 | 同时记录wrapper与设备时间线 |
| 单卡正常，多卡慢 | 负载不均、通信或映射变化 | 对齐rank计算结束与collective完成时刻 |
| 只有部分shape慢 | 特化、尾块、stride或fallback | 比较桶边界及实际命中路径 |

这些是调查方向，不是由症状直接确定根因。单卡正常并不能排除多卡时的功率、共享带宽或计算负载变化。

<!-- systems-workflow-visual -->
```mermaid
flowchart TD
  S["固定输入后复现异常"] --> A{"单卡kernel也慢?"}
  A -->|"是"| K["执行路径、资源与设备状态"]
  A -->|"否"| B{"完整单卡流程慢?"}
  B -->|"是"| W["转换、分配、同步与调度"]
  B -->|"否"| R["逐rank比较进入通信前后时刻"]
  R --> D["负载不均或链路对照"]
```

每个分支只缩小调查范围，仍需通过控制变量确认根因。

## 带宽判断需要匹配实际数据流

先计算输入输出的最低字节量，再看缓存、重复读取、padding与转换是否增加了物理流量。接近HBM上限的kernel可优先考虑减少搬运；远低于上限时，还可能受不连续访问、并行度、依赖或小任务启动限制。

不要把逻辑GB/s与硬件计数器流量混为一谈。反复基准同一小tensor可能主要命中缓存，必须说明测量工作集与缓存条件。

## 先算这段 kernel 值不值得优化

假设一次完整 step 用时 100 ms，其中目标 kernel 占 20 ms。即使它理想地加速 2 倍，step 也只会变成 90 ms，整体提升约 11.1%。如果 kernel 只占 5 ms，同样的 2 倍加速只能把 step 从 100 ms 降到 97.5 ms。这个估算不是性能结论，却能在写新特化前提醒你先确认它是否位于关键路径。

可以把每次候选实现记成一行：

| 范围 | 原耗时 | 新耗时 | 关键路径占比 | 伴随变化 |
| --- | ---: | ---: | ---: | --- |
| kernel | 20 ms | 10 ms | 20% | 寄存器增加、occupancy 下降 |
| 完整层 | 34 ms | 25 ms | 34% | layout copy 减少 |
| 单卡请求 | 100 ms | 90 ms | 100% | 仍受排队和采样影响 |

表里的“关键路径占比”必须来自同一条 trace，不能拿 kernel 的独立 benchmark 直接填入。若完整层只快 9 ms，说明 wrapper、转换或同步仍占据剩余时间；继续调 tile 之前，先查这些阶段有没有把收益吃掉。

## 对多卡先找最早落后的事件

如果某rank进入collective前已经迟到，检查其数据、计算和设备状态；如果进入时间相近而交换阶段拖长，再检查拓扑和实际链路。记录rank到GPU映射及GPU到NIC/NUMA关系，换设备位置做对照可以帮助定位。

温度、功率、时钟、ECC和驱动信息是上下文证据。出现相关指标异常后，仍需确认它与性能变化在时间和设备上对应，不能仅凭一条告警解释所有退化。

## 一次排查应留下可重复的最小记录

保留固定输入、旧新版本、计时范围、各rank关键时刻、实际kernel与设备状态。改变一个因素后，用相同工作负载复测，并检查数值输出。

如果怀疑FP8路径没有生效，先看cast/dequant和GEMM；如果怀疑KV瓶颈，先看真实历史长度和KV读取；如果怀疑JIT，区分首请求与预热后。这些分支对应[量化kernel](low-precision-and-quantized-kernels.md)、[服务attention](serving-attention-and-kv-kernels.md)和[dispatch](runtime-dispatch-and-kernel-selection.md)。

![Nsight Systems GPU starvation](../assets/images/tools/nsight-systems/gpu-starvation.png){ width="820" }

<small>图源：[NVIDIA Nsight Systems 文档](https://docs.nvidia.com/nsight-systems/)。原图表达 GPU 时间线中 CPU/GPU 等待和空洞如何暴露系统瓶颈；读图重点是把硬件排查从“猜 kernel 慢”改成先看带宽、occupancy、launch、同步和拓扑。它不能单独定位根因，只提供缩圈入口。</small>

## 对照源码与延伸阅读

- [可固定shape与配置的矩阵乘基准](https://github.com/triton-lang/triton/blob/cd2e01c6686d6503802e6cdfac56d5a5dfdb27a1/python/tutorials/03-matrix-multiplication.py)
- [实际配置和JIT观察入口](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/README.md)

## 一次排查的证据包

把排查结果固定成一个目录，而不是只贴一张 profiler 截图：

```text
repro/
  workload.json       # shape、dtype、batch、长度
  env.txt             # driver、CUDA、runtime、commit
  topology.txt        # GPU/NIC/NUMA 映射
  reference.json      # 数值 checksum 与容差
  trace.nsys-rep      # CPU/GPU 时间线
  counters.csv        # 带宽、SM、occupancy、功率、时钟
  conclusion.md       # 假设、改变的变量、结论边界
```

如果没有环境和输入记录，下一次运行即使现象相同，也无法确认是同一问题。结论文件应明确哪些字段被改变、哪些仍未验证，以及结果只覆盖单卡、某个 shape 还是完整服务。

## 先修复错误，再解释性能

性能异常伴随数值差异时，优先保存最小输入并完成参考对照。错误 kernel 可能因为提前退出、少做计算或错误 mask 而“更快”；把它当成性能提升会污染后续判断。多卡问题还要确认 collective 的归约语义和全局分母没有变化。

## 自测

给出“单 kernel 变慢 20%，完整服务不变”的 trace，列出至少三个可能解释，并指定下一项最小对照。一个合理答案应包括：该 kernel 不在关键路径、wrapper/排队变化抵消、或测量边界不一致；不能只写“换更大 tile”。

**来源与改编说明**：时间线、设备状态与矩阵乘入口参考 NVIDIA Nsight Systems、Triton 和固定版本 DeepGEMM README。页面的证据包结构是工程模板，不代表单一 profiler 能自动定位根因。
