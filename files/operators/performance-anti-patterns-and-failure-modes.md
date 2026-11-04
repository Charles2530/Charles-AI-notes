# 可以算清的性能误判

下面都是为解释机制构造的例子，没有声称发生在某个真实生产系统。每个例子给出局部数字为何不足，以及还需哪一项证据。

## 一个算子快一倍，整步只快一点

若该算子只占整步10%，其余90%不变，算子提速2倍后总时间为原来的95%，整体加速约1.053倍。这个结果完全符合预期。

需要先测算子在关键路径中的占比，再决定优化价值。若新实现还新增2%的转换成本，总节省会进一步减到3%。

## 省了一次launch，却增加更多内存流量

融合前两个kernel各使用小量寄存器，融合后同时保留更多中间值。若因此spill，新增local memory访问可能超过省掉的中间写回。

不能只看kernel数量下降。检查寄存器、spill、实际读写与完整层时间，再缩小融合范围或针对部分shape启用。这个因果假设需要编译和profile证据，不是所有融合变慢的统一解释。

## 调优了最常见的调用，却错过最贵的调用

90次100微秒和10次2毫秒，总时间29毫秒。前者调用最多却只占9毫秒。只按调用次数选热点，可能把主要成本留在后者。

应按累计时间与目标指标加权，同时区分吞吐、P50和P99。不同目标可以产生不同优先级。

## “相同shape”里藏着一次复制

一个快kernel只接受连续输入，wrapper对转置视图执行 `contiguous()`。单测已经连续的tensor看不到复制；真实模型每层都经过转置，复制就反复发生。

比较时记录stride，并把转换放进同一计时边界。可选择支持原layout、让上游直接产出目标layout，或在转换成本高的桶里保留通用实现。

## 平均负载不高，最慢的设备却决定整步时间

四个 rank 的计算时间分别为 4、4、4、12 毫秒。若下一阶段必须等所有 rank，计算阶段至少需要 12 毫秒；平均值 6 毫秒隐藏了三个 rank 的等待。忽略通信时，这一阶段的有效利用率为 24/(4×12)=50%。理想均分成每个 6 毫秒可能把关键路径减半，但重分配新增的通信与调度必须一起计时。

Attention 的有效计算量还与 mask 有关。同样多的 token，落在因果三角形不同位置的分块可能拥有不同数量的有效 query-key 对。下图可用来追踪每个设备实际接到哪些计算块，再对照 trace 中谁最后完成。

![MagiAttention ring attention load balance 原论文图](../assets/images/paper-deep-dives/foundations/magiattention/figure-2-ring-attention-load-balance.png){ width="860" }

<small>图源：[MagiAttention 官方文档](https://sandai-org.github.io/MagiAttention/docs/main/index.html) 与 [GitHub](https://github.com/SandAI-org/MagiAttention)，Figure 2。原图意：不同 attention 切分策略会导致计算块在设备间分布不均，长上下文训练需要显式处理 mask 与负载均衡。</small>

## Trace里出现overlap，却没有减少等待

假设计算和通信单独各需10个单位，同时执行因带宽竞争各变成18。虽然色块重叠，完成时间18只比串行20略好。若调度还增加额外同步，甚至可能更慢。

检查最后依赖完成时间与资源竞争，而非只观察两条stream同时有工作。通信的排队时间还可能包含等待其他rank到达。

![DeepSeek-V3 DualPipe 原论文图](../assets/images/paper-deep-dives/technical-reports/deepseek-v3/figure-5-dualpipe.png){ width="900" }

<small>图源：[DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)，Figure 5。原论文图意：DualPipe 通过更细粒度地组织前向、反向、通信与 pipeline 阶段，减少流水线气泡并提高通信和计算重叠。</small>

## 低比特文件更小，首token更慢

如果每次请求新增激活量化、权重解包或布局转换，prefill的计算收益可能不足以抵消这些步骤。也可能只是首次JIT混进了测试。

先比较冷启动与稳态，再区分权重预处理和每请求操作，最后检查实际GEMM路径。质量下降则走另一条数值对照链，不能用同一种性能解释覆盖所有问题。

每次优化记录“减少的成本”和“新增加的成本”，并把假设放回实际调用分布。这样即使结果不如预期，也能得到下一步应查什么。


## 对照源码与延伸阅读

- [输入layout、候选与基准边界](https://github.com/triton-lang/triton/blob/cd2e01c6686d6503802e6cdfac56d5a5dfdb27a1/python/tutorials/03-matrix-multiplication.py)
- [高级调度的资源和架构约束](https://github.com/triton-lang/triton/blob/cd2e01c6686d6503802e6cdfac56d5a5dfdb27a1/python/tutorials/09-persistent-matmul.py)
