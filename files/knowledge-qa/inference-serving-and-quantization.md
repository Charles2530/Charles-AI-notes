# 推理服务与量化问答

服务变慢可能发生在排队、prefill、decode或结果传输阶段。先把一次请求拆开，才能判断缓存、批处理和量化应作用在哪里。

## 吞吐提高，为什么用户反而等得更久

批处理把更多请求放在同一轮计算，提高设备利用率，却可能让单条请求等待凑批或让每轮变长。每秒输出token更多，与首token更早出现是两项指标。

如果请求在0秒到达、0.3秒开始计算、0.5秒收到首token，TTFT是0.5秒，不能只报告0.2秒prefill执行时间。队列时间、冷启动和网络是否计入，要在协议中明确。见[推理服务流程](../inference/serving-systems.md)。

## TPOT的分母为什么容易算错

若一条请求输出5个token，首个在0.5秒，最后一个在0.9秒，后续共有4个间隔，平均TPOT为(0.9-0.5)/4=0.1秒。把总请求时间除以5，会把排队和首token混进decode指标。

单token响应没有后续间隔，TPOT应按约定标为不可用或单独处理。流式批量发送时，客户端看到的间隔又可能混入缓冲，不等于每次GPU迭代时间。见[成本与SLO](../inference/cost-modeling-and-slo-design.md)。

## 权重量化以后，为什么长上下文仍然OOM

权重量化主要减参数存储与读取，KV随层数、长度、KV头数、batch和每元素字节增长。32层、8个KV头、每头128维、BF16、长度32768的一条序列，原始KV约4GiB；4条就是16GiB。

若OOM来自大量长请求，4-bit权重不会自动把KV也变成4-bit。还需检查KV格式、分页、prefix共享、最大并发和调度预算。见[数值与运行时](../foundations/numerics-memory-and-runtime-basics.md)。

## Paged KV为什么省碎片，却不一定省数学需要的KV

分页把逻辑token映射到固定大小物理块，允许按需分配与复用，不要求每条序列预占完整最大长度。尾页仍可能有空位，block table也占空间。它优化分配与访问，并不自动删除必要历史。

Prefix cache进一步复用相同前缀的已计算状态，但匹配条件不只可见文本，还可能包括模型、适配器和位置语义。错误复用会改变答案，命中率高也不代表所有延迟都能消失。见[缓存与路由](../inference/caching-routing-and-speculative.md)。

## 量化的平均误差很小，为什么某类问题掉分

同样的权重误差经不同输入方向放大不同，长尾激活或少数敏感层也可能主导任务退化。校准集如果只含短问答，未必覆盖长代码、视觉token或动作输出。

先对固定输入比较分层激活与logits，再看任务组、长度和异常范围。PTQ方法、实际kernel和数据格式必须一致；只用fake quant验证数学误差，不能证明部署kernel正确。见[校准与异常值](../quantization/activation-outliers-and-calibration-strategies.md)。

## 投机解码接受率高就一定更快吗

假设一次草稿产生4个候选，平均接受3个，看起来很好，但草稿成本、目标验证成本、KV维护和额外同步都要计入。若原本batch已很大、目标验证效率下降，收益可能很小。

严格接受与补偿规则可维持目标采样分布；随意拒绝、只取argmax或用不同温度，可能改变保证。接受率描述算法内部事件，不是答案正确率。见[多token预测与投机](../training/mtp-and-speculative-decoding.md)。

## 一个更快的kernel为什么没有改善服务P99

P99可能由排队、长请求、通信或少数慢路径决定。单个算子占请求时间10%，即使无限加速，其他部分不变时整体最多约1/0.9=1.111倍。还需确认优化是否命中真实shape和dtype。

使用端到端trace把请求阶段与设备活动对齐，再比较固定负载下的分位数和吞吐。平均kernel时间不能替代尾延迟测量。见[可观测性与在线评测](../inference/observability-and-online-evaluation.md)。

## 快速自测

拿一条请求画出 queue、prefill、decode、工具和输出传输的时间线，再分别标出权重、KV、prefix cache 和 speculative decoding 改变了哪一段。最后写出一个会让平均延迟下降、P99 却上升的负载条件。具体实验入口见[推理服务](../inference/serving-systems.md)和[量化验收](../quantization/evaluation-and-deployment-checklist.md)。

**来源与改编说明**：服务阶段和分页 KV 对照 [vLLM 文档](https://docs.vllm.ai/)、[PagedAttention 论文](https://arxiv.org/abs/2309.06180)；量化接口再看 [PyTorch AO](https://pytorch.org/ao/stable/) 和 [TensorRT-LLM 文档](https://nvidia.github.io/TensorRT-LLM/)。算例中的长度、时间和显存数字用于手算，不是线上测量。
