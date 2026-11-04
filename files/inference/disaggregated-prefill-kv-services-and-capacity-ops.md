# Prefill完成以后，KV怎样交给另一组GPU

Prefill一次处理许多输入token，decode每轮生成少量token。分开部署可以减少两类工作互相干扰，但下一阶段必须拿到正确KV。是否划算，要比较减少的等待与新增传输。

## 先算一条请求要交接多少状态

设32层、8个KV头、头维128、BF16、32768个prompt token，主体KV为

$$
2\times32\times8\times128\times32768\times2
=4\ \text{GiB}.
$$

若有效传输带宽假设为25GB/s，单向传输下界约0.172秒。这里GiB按 $2^{30}$ 字节，GB/s按十进制，单位不同已显式换算。还需加队列、布局转换、元数据与协议开销。

## 交接的不只是一个字节数组

接收端需要模型与adapter身份、token/位置约定、已计算长度、层与KV头布局、dtype/scale以及页映射。两侧张量并行方式不同，还可能要重新分片或排列。

错误的模型身份可能产生数值合法但语义错误的KV；错误的长度会让未初始化槽位进入attention。传输完成与状态可提交需要同一个明确边界。

## 哪段计算可以和传输重叠

按层或按块流式传输可能隐藏部分时间，但decode能开始哪部分计算，取决于数据依赖和实现。不能把全部4GiB传输时间直接从prefill时间里减掉，除非实际时间线证明它被覆盖。

若网络与其他通信共享资源，独立传输基准也可能高估实际带宽。应在目标并行与负载下测完整交接。

## 两个池分别按哪种需求扩容

Prefill池主要承受每秒输入token及其长度分布，decode池承受输出token、活跃请求KV和逐步调度。两个池的服务率并不由同一个tokens/s数字决定。

教学设定中，每秒2个请求、平均输入8000、输出200，对应每秒16000输入token和400输出token。实际需要多少GPU，必须用各池在该长度和batch分布下的实测能力，再考虑尾延迟与冗余。

## 一次传输慢不能解释所有后续token都慢

KV交接通常主要影响进入decode前的等待。若每个后续token持续变慢，应继续检查decode池负载、KV layout、远程读取或并行配置，而不能只归因到一次初始传输。

比较合并与拆分部署时固定硬件总预算和请求分布，同时报告TTFT、TPOT、网络流量、KV驻留和质量。只有减少的干扰与排队超过新增交接成本，拆分才在这个配置下有价值。

![PagedAttention KV block mapping](../assets/images/paper-figures/inference/vllm-pagedattention.png){ width="820" }

<small>图源：[Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)。原图展示 logical KV blocks 到 physical KV blocks 的映射；读图重点是：当 prefill、decode 和 KV 被拆开后，KV 不再只是模型内部状态，而是要被命名、转移、复用和回收的系统资源。这张图不能证明解耦 prefill 一定有收益，收益仍取决于长短请求分布和 KV 转移成本。</small>

## 对照源码与延伸阅读

- [KV身份、容量与调度边界的参考实现](https://github.com/vllm-project/vllm/blob/fa3b9d1377219247060d3c8b49eae199c9e7e154/vllm/v1/core/sched/scheduler.py)
