# 世界模型效率技术对照

这页用于在已知瓶颈后查找方法。每行区分机制、已有证据与当前系统还需要测的量。论文在其任务中的收益、本站可运行的数值算例、尚未执行的工程设计是不同证据，不能合成一项“全链路已验证”的结论。

![Video world model taxonomy](../assets/images/paper-deep-dives/world-models/towards-video-world-models/figure-2-world-model-taxonomy.jpg)

<small>图源：[Towards Video World Models](https://www.xunhuang.me/blogs/world_model.html)，Figure 2。它提供模型用途的分类视角，不提供本页技术组合的实测性能。</small>

## 真实交互或训练预算受限

| 候选机制 | 改变哪笔工作 | 可追溯材料 | 在自己的任务中仍需比较 |
| --- | --- | --- | --- |
| RSSM 与 imagined behavior learning | 用模型轨迹提供部分策略更新信号 | [Dreamer 方法](https://arxiv.org/abs/1912.01603)、[状态与行为学习解释](rssm-dreamer-and-planning.md) | 固定真实交互预算的回报，以及新增 GPU 计算与模型偏差 |
| 表征与动力学分工 | 改变视觉训练和时序训练的目标与更新组织 | [MWM](https://arxiv.org/abs/2206.14244)、[机制说明](masked-jepa-and-latent-prediction.md) | 相同数据/计算下的接触、遮挡和任务结果 |
| 针对失败补采 | 调整采样分布与可获得监督 | [瓶盖失败的数据流程](data-engines-and-self-improvement.md)，构造案例 | 独立目标桶收益、主分布退化、采样成本与选择偏差 |
| 固定日志再训练 | 复用真实转移 | [离线 RL](../reinforcement-learning/offline-reinforcement-learning.md) | 动作支持与离线评测偏差，不能凭日志数量推出泛化 |

训练损失更低和真实交互更少不是同一指标。若把算力换成数据效率，报告应同时列两者；如果组合方法改变了训练步数，也要说明比较预算。

## 状态表示或显存受限

| 候选机制 | 直接机制 | 可追溯材料 | 关键限制 |
| --- | --- | --- | --- |
| Visual resampler | 将多 patch 映射为少量 token | [视觉连接器](../vlm/visual-tokenizers-and-connectors.md)、[fixture 的 6.70 倍 token 账](world-model-efficient-training-experiment-report.md) | fixture 没有训练 encoder；token 比不是速度比 |
| JEPA-style prediction | 在目标表征中预测，改变对像素 decoder 的依赖 | [V-JEPA](https://arxiv.org/abs/2404.08471)、[V-JEPA 2](https://arxiv.org/abs/2506.09985) | 表征可预测不等于已学动作后果；需要对应动作数据与接口 |
| Activation checkpointing | 保留较少激活，反向重新计算 | [训练系统](../training/distributed-training-and-checkpointing.md) | 重算时间、随机性、循环状态与 mask 语义 |
| 参数/优化器分片 | 将训练状态分布到多设备 | [分布式训练](../training/distributed-training-and-checkpointing.md) | 通信、保存恢复与实际峰值显存 |
| 低比特训练 | 用不同数值格式存储或计算 | [低比特与数值](../training/low-bit-training-and-numerics.md) | scale、累加、稳定性和任务退化需要单独测 |

压缩是否划算，取决于被省掉的部分是否位于主要成本上。相关的 Amdahl 算例见[效率路线](efficient-training-roadmap.md)。

## 候选规划超过等待时间

| 候选机制 | 可以省掉什么 | 来源与推导 | 本地必须验证什么 |
| --- | --- | --- | --- |
| 相同前缀 KV 共享 | 候选分叉前的重复存储与计算 | [24 MiB 前缀与 16 分支算例](rollout-serving-quantization-kernel-case.md) | 条件真正相同、动作分支隔离、缓存版本与实测延迟 |
| KV 量化/合并 | 缓存大小及部分访存 | [KVSlimmer](https://arxiv.org/abs/2603.00907)、[部署解释](rollout-serving-quantization-kernel-case.md) | 身份记忆、接触事件、候选排序；文本结果不能直接迁移 |
| 权重量化 | 参数存储、访存及支持硬件上的低精度计算 | [SmoothQuant](https://arxiv.org/abs/2211.10438) | 校准分布、转换开销、风险/动作读出与尾延迟 |
| 少步、因果视频生成 | 生成步骤与交互组织 | [CausVid](https://arxiv.org/abs/2412.07772) | 动作条件预测与闭环用途，生成速度不证明物理准确 |
| Packing、mask 与 attention 调度 | 无效计算、负载不均及通信等待 | [MagiAttention 文档](https://sandai-org.github.io/MagiAttention/docs/main/index.html) | 与基准相同的可见关系、硬件/shape 和端到端性能 |
| 有界队列与期限处理 | 过期观察、排队和迟到结果 | [观测年龄](../thinking-exploration/embodied-inference-observation-age-1009-2059.md) | 丢帧是否遗漏事件，超时回退与实际任务后果 |

## 怎样辨认一行证据的分量

本站 mini-chain 真实执行的是 JSONL 指标计算，它没有训练模型、生成视频或执行机器人。它支持核对 token 总数、分桶 ECE 与 top-1 选择规则；不支持某项优化改善了任务成功的结论。其 `training-config.yaml` 也未被 evaluator 读取。

论文图与原始代码能够解释方法，复现性能还需要匹配数据、模型、硬件与预算。对于自己的系统，一项技术是否保留，应由成本变化和对应任务结果共同决定。沿一条失败定位下一实验，可直接查[开发与评测](development-roadmap-data-simulation-and-evaluation.md)。
