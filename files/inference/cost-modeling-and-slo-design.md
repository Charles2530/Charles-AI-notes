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
