# 世界模型高效训练效率技术覆盖矩阵

> **这篇回答的问题。** 如何理解“世界模型高效训练效率技术覆盖矩阵”背后的核心机制、适用边界和下一步阅读路径。

这页是全站地图，不替代各专题正文。它只回答一个问题：**当你说某项技术让世界模型训练更高效时，它到底省哪类成本，证据在哪里，还不能证明什么。**

![Video world model taxonomy](../assets/images/paper-deep-dives/world-models/towards-video-world-models/figure-2-world-model-taxonomy.jpg)

<small>图源：`Towards Video World Models` Figure 2（可从[论文标题 arXiv 检索](https://arxiv.org/search/?query=Towards+Video+World+Models&searchtype=title)核对）。原图表达 internal world model、external world model、video world model、simulation platform 和 neural 3D/4D 的位置关系；本站读法是：效率技术必须先落到“表示、训练系统、推理、验证”中的某个成本层。它不能证明某项技术已经具备闭环控制收益，只提供分类坐标。</small>

## 数据效率

| 效率技术 | 解决的成本 | 站内入口 | 原始证据 | 缺口 / 不能证明 |
| --- | --- | --- | --- | --- |
| Dreamer / RSSM latent imagination | 真实环境交互、策略试错 | [RSSM、Dreamer 与规划](rssm-dreamer-and-planning.md) | [DreamerV3, Figure 3(a)/(b), Figure 6](https://arxiv.org/abs/2301.04104) | 不证明高分辨率视频模拟器天然可规划 |
| Masked World Models | 视觉控制样本、像素重建压力 | [Masked / JEPA 与潜变量预测](masked-jepa-and-latent-prediction.md) | [MWM, Figure 1, Figure 4](https://arxiv.org/abs/2206.14244) | 不证明 masked 表征天然有动作因果 |
| 数据引擎 / hard negatives | 真实失败采集、人工排查 | [世界模型数据引擎](data-engines-and-self-improvement.md) | [mini-chain Toy Fixture](action-conditioned-video-world-model-training-case.md) | Toy fixture 不是生产模型效果 |
| Offline-online hybrid | 在线探索成本、长尾样本覆盖 | [端到端案例](action-conditioned-video-world-model-training-case.md) | 站内 schema 与 failure replay | 需要真实重训日志或线上灰度才能证明收益 |

## 表示效率

| 效率技术 | 解决的成本 | 站内入口 | 原始证据 | 缺口 / 不能证明 |
| --- | --- | --- | --- | --- |
| V-JEPA / JEPA-style latent prediction | 像素 decoder、视频 token 冗余 | [Masked / JEPA 与潜变量预测](masked-jepa-and-latent-prediction.md) | [V-JEPA, Figure 3, Table 1/4/5](https://arxiv.org/abs/2404.08471) | 原始 V-JEPA 不含 action、reward、done |
| Visual tokenizer / resampler | 多相机长视频 token | [视觉 Tokenizer 与连接器](../vlm/visual-tokenizers-and-connectors.md) | 站内 token 成本账与 VLM/VLA 接口 | 压缩后是否保留接触和小目标要单独验 |
| MWM decoupled representation | 表征学习与 dynamics 相互拖累 | [Masked / JEPA 与潜变量预测](masked-jepa-and-latent-prediction.md) | [MWM, Figure 1](https://arxiv.org/abs/2206.14244) | 不能替代 closed-loop planning 证据 |
| Depth Anything 3 / geometry state | 几何状态估计、深度/相机底座 | [Depth Anything 3 专题](../paper-deep-dives/embodied-ai/depth-anything-3.md) | [Depth Anything 3](https://arxiv.org/abs/2511.10647) | 几何估计不等于动力学和策略安全 |

## 训练系统效率

| 效率技术 | 解决的成本 | 站内入口 | 原始证据 | 缺口 / 不能证明 |
| --- | --- | --- | --- | --- |
| Activation checkpointing | 长轨迹激活显存 | [训练](../training/index.md) | 通用训练系统实践 | 增加重算，需确认 recurrent state 和 mask 一致 |
| Sequence packing | padding、无效 attention | [高效训练路线图](efficient-training-roadmap.md) | 站内 4 路相机 token 账 | 不能让不同 episode 互相可见 |
| FSDP / ZeRO | optimizer、gradient、parameter state | [训练](../training/index.md) | [ZeRO 专题](../paper-deep-dives/foundations/zero.md) | 通信和 checkpoint I/O 可能成为新瓶颈 |
| MagiAttention | 异构 mask 长上下文通信和负载不均 | [MagiAttention 专题](../paper-deep-dives/foundations/magiattention.md) | [MagiAttention docs / benchmark](https://sandai-org.github.io/MagiAttention/docs/main/index.html) | 只证明系统吞吐，不证明 dynamics 更准 |
| Low-bit training / FP8 | 显存、带宽、通信 payload | [低比特训练](../training/low-bit-training-and-numerics.md) | 站内低比特与技术报告专题 | reward/risk/action head 需要精度保护 |

## 推理效率

| 效率技术 | 解决的成本 | 站内入口 | 原始证据 | 缺口 / 不能证明 |
| --- | --- | --- | --- | --- |
| KV cache / GQA | rollout decode 显存和带宽 | [推理成本建模](../inference/cost-modeling-and-slo-design.md) | [高效训练路线图 KV 账](efficient-training-roadmap.md) | KV 变小不保证候选动作排序不变 |
| KVSlimmer | 长上下文 KV 合并和 decoder 开销 | [KVSlimmer 专题](../paper-deep-dives/inference/kvslimmer.md) | [KVSlimmer](https://arxiv.org/abs/2603.00907) | 文本长上下文收益不能直接外推到动作分叉 |
| Low-bit KV | KV 显存、HBM 带宽 | [量化](../quantization/index.md) | 站内量化与 KV cache 页面 | risk / action head 仍需回归测试 |
| CausVid / DMD | 少步视频 rollout、streaming latency | [CausVid 专题](../paper-deep-dives/diffusion/causvid.md) | CausVid paper result | 视频延迟下降不等于 planning utility |
| SLO / batching | 端到端延迟、并发容量 | [推理成本建模](../inference/cost-modeling-and-slo-design.md) | 系统日志或压测 | P95/P99 变好不证明模型更安全 |

## 验证效率

| 效率技术 | 解决的成本 | 站内入口 | 原始证据 | 缺口 / 不能证明 |
| --- | --- | --- | --- | --- |
| Action sensitivity | 无效视频 demo、动作不敏感模型 | [世界模型评测](evaluation-and-failure-modes.md) | [mini-chain 报告样例](world-model-efficient-training-experiment-report.md) | 单一反事实测试不能覆盖所有风险 |
| Closed-loop gain | open-loop 指标和真实决策脱节 | [世界模型评测](evaluation-and-failure-modes.md) | Dreamer / V-JEPA 2-AC 等闭环设置 | 成功率需要配合失败桶和成本账 |
| Risk calibration | near-miss 漏报、安全错判 | [不确定性与风险敏感规划](uncertainty-and-risk-aware-planning.md) | mini-chain risk ECE 示例 | 小样本 ECE 不能证明部署安全 |
| Failure replay | 失败不可复盘、数据回流低效 | [端到端案例](action-conditioned-video-world-model-training-case.md) | Toy fixture failure replay | 需要真实视频、预测、动作和人工标签 |

## 读完怎么用

1. 如果你在做方案设计，先从“解决的成本”定位路线。
2. 如果你在写博客或论文笔记，至少引用“原始证据”和“缺口 / 不能证明”两列。
3. 如果你在做实验，进入 [完整实验报告样例](world-model-efficient-training-experiment-report.md)，按数据、配置、系统账、评测和失败归因写证据链。
