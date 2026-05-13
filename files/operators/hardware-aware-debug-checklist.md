# 硬件感知排查清单

当一个 kernel 或训练/推理路径性能异常时，最容易犯的错误是只在软件层找原因。实际上很多问题只有带着硬件感知去排查，才能快速定位。

!!! note "初学者先抓住"

    性能排查先判断瓶颈类型：是算不动、搬不动、同步慢、launch 多，还是真实 workload 没命中优化路径。不同瓶颈对应完全不同动作，不能只靠“再优化代码”解决。

!!! example "有趣例子：水管、工人和仓库"

    工人很多但水管太细，产线仍然慢；仓库离得太远，工人也会等料。GPU 里对应的就是算力、带宽、缓存、拓扑和调度要一起看。

## 1. 先问是不是带宽问题

如果：

1. DRAM 吞吐很高；
2. 算术强度不高；
3. Tensor Core 利用率也不高；

那就更应该先检查数据流、layout、fusion 和访问模式，而不是继续堆算力优化。

## 2. 再问是不是寄存器和 occupancy 问题

如果：

1. register 使用过高；
2. occupancy 很低；
3. 出现 spill；

那么问题可能来自 tile 过大、融合过度或中间值管理不当。

## 3. 再问是不是拓扑和通信问题

如果单卡 microbenchmark 很快，但多卡系统仍慢，就要怀疑：

1. TP/DP 分组不合理；
2. 高频通信跨了慢链路；
3. overlap 没有真正发生；
4. 某些 rank 成为拖尾点。

## 4. 再问是不是 workload 分布问题

如果 benchmark 很好看，线上却收益一般，通常要回到：

1. shape bucket 是否真实；
2. 高频路径是否真被命中；
3. fallback 比例是否过高；
4. prefill 和 decode 是否被混在一起评估。

## 5. 小结

**硬件感知排查的核心是**：不要把所有问题都解释成算法或框架问题。很多性能瓶颈最终都要回到带宽、寄存器、拓扑和 workload 分布这些更底层也更真实的约束上。只要先从这些约束出发，定位通常会快很多。

## 工程收束

硬件感知排查要同时看频率、温度、ECC、PCIe / NVLink 和 NUMA 映射。拓扑退化容易被误判成 kernel 退化，坏卡也可能被误判成模型不稳；上线前应固定探针基准，保留故障前后硬件日志，做节点隔离与复验，并按机型建立健康基线。

## 真实排查案例：多卡吞吐突然掉一截

**输入症状**：同一份推理服务配置，在 8 卡机器上从约 1.0x 基线掉到 0.72x；单卡 microbenchmark 没有明显退化，业务侧表现为长上下文请求 P99 抖动。

**关键指标**：GPU util 仍高，但部分 rank 的 SM active 明显低；NVLink / PCIe counters 不均衡；decode 阶段每步耗时出现周期性尖峰；同一 batch 中尾部 rank 经常晚 10-20ms 完成。

**Nsight / trace 观察**：Nsight Systems 时间线上，计算 kernel 之间夹着不规则的 NCCL wait；慢 rank 的通信跨到 PCIe 路径，和预期 NVLink 邻接关系不一致；CPU 侧调度没有新增同步点。

**判断**：问题不在单个 attention 或 GEMM kernel，而在拓扑映射和并行分组。某次换机后 GPU ordinal 与物理拓扑不一致，TP 分组跨了慢链路，导致局部通信拖住全局 step。

**修复**：重新生成拓扑感知 rank mapping，把高频 TP 通信放回 NVLink 邻接组；启动时打印并保存 `nvidia-smi topo -m`、rank-to-device 映射和 NCCL 选路日志；灰度期按 rank 记录 step time 分布。

**反例**：如果单卡 kernel 也同步变慢、DRAM 吞吐接近上限、没有 rank 间拖尾，那么优先怀疑 kernel / layout / dtype 路径，而不是拓扑。拓扑问题的典型特征是“局部 rank 拖尾 + 通信 wait + 单卡基准正常”。
