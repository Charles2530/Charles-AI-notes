# 世界模型：从内部状态到可决策的未来

世界模型是一组关于环境变化的学习接口：压缩历史形成 belief，预测动作条件下的未来状态，生成未来观测，估计奖励与终止，或为策略提供可想象的训练轨迹。判断一项工作是否真正服务世界建模，关键看预测对象能否支持它声称的消费方。

## 世界模型的统一接口

给定历史 \(h_t=(o_{\le t},a_{<t})\)，状态模型先得到内部状态：

\[
z_t\sim q_\phi(z_t\mid h_t).
\]

动作条件动力学预测：

\[
\hat z_{t+1}\sim p_\theta(z_{t+1}\mid z_t,a_t).
\]

根据用途，模型还可以包含观测、奖励、终止和风险头：

\[
\hat o_t\sim p_\theta(o_t\mid z_t),\quad
\hat r_t=r_\theta(z_t,a_t),\quad
\hat d_t=d_\theta(z_t,a_t).
\]

这里的 \(z_t\) 不要求等于真实物理状态；它只需保留下游任务所需的信息。用于视频生成的状态可能强调可见细节，用于控制的状态更强调动作相关性、奖励和可预测性。完整基础见[状态、动力学、观测与决策](beginner-world-models-primer.md)。

## 三类建模对象

| 路线 | 主要预测对象 | 是否要求像素生成 | 典型用途 |
| --- | --- | --- | --- |
| latent dynamics | belief、下一 latent、reward、continuation | 否，decoder 可选 | MPC、actor-critic、latent imagination |
| predictive representation | 被遮挡或未来片段的目标表征 | 否 | 表征学习、物理推理、动作条件规划 |
| generative simulation | 未来帧、视频 latent、交互环境 | 通常是 | 可视化仿真、合成 rollout、agent 训练 |

对应阅读：

- [RSSM、Dreamer 与规划](rssm-dreamer-and-planning.md)：latent dynamics 怎样进入 MPC 和 actor 学习。
- [Masked / JEPA 与潜变量预测](masked-jepa-and-latent-prediction.md)：预测表示怎样避免对不可预测像素投入全部容量。
- [生成式模拟与视频世界模型](generative-simulation-and-video.md)：视频底座何时能够承担动作条件 rollout。
- [WM / WAM / VAM](wm-wam-vam-and-action-conditioned-modeling.md)：动作放在条件、输出或 latent 接口的不同位置意味着什么。

三类路线可以组合。一个系统可以用视频 autoencoder 定义 latent，用 Transformer 预测动作条件 latent，再用 decoder 生成可见未来，同时用 reward head 给规划器评分。

## 从预测到决策

世界模型进入决策有三种主要方式：

1. **在线规划。**从当前 belief 展开多个候选动作序列，用预测 reward、constraint 和 terminal 选择第一步，再读取真实观测重规划。见 [planning as inference](planning-as-inference-and-latent-actions.md)。
2. **想象中训练策略。**先从真实历史得到 latent，随后在模型中 rollout，训练 actor 与 critic，部署时由 actor 直接输出动作。见 [RSSM 与 Dreamer](rssm-dreamer-and-planning.md)。
3. **生成训练与评测环境。**模型产生可交互未来、反事实或 hard cases，供 agent 学习和测试。见[模拟器、反事实与合成 Rollout](simulators-counterfactuals-and-synthetic-rollouts.md)。

预测误差必须按消费方评测：像素误差小不代表候选动作排序正确；短期 latent 准确不代表长时 rollout 校准；平均未来合理不代表危险尾部没有被漏掉。对应方法见[世界模型评测与失效模式](evaluation-and-failure-modes.md)与[不确定性及风险规划](uncertainty-and-risk-aware-planning.md)。

## 应用与工程

| 分支 | 主要问题 | 入口 |
| --- | --- | --- |
| 机器人与驾驶 | 状态、动作、几何和风险怎样进入实际闭环 | [机器人与自动驾驶应用](applications-in-robotics-and-autonomous-driving.md) |
| VLA 结合 | world model 怎样影响候选动作，而非只生成旁路视频 | [VLA 世界模型全链路](vla-world-model-full-chain-case.md) |
| 动作条件视频 | 数据 schema、action alignment、loss 与 rollout 怎样连接 | [端到端训练案例](action-conditioned-video-world-model-training-case.md) |
| 数据飞轮 | 失败、near-miss、反事实和主动采样怎样回流 | [数据引擎与自我改进](data-engines-and-self-improvement.md) |
| 开发路线 | 数据、几何、动作、仿真和评测以什么顺序建立 | [开发路线](development-roadmap-data-simulation-and-evaluation.md) |
| 训练效率 | 表示、长序列、系统和验证成本怎样拆账 | [高效训练路线图](efficient-training-roadmap.md) |
| 技术选择 | 不同效率方法实际覆盖了哪一类瓶颈 | [效率技术覆盖矩阵](efficiency-technique-coverage-matrix.md) |
| rollout 服务 | 量化、KV、kernel 与交互延迟怎样共同约束部署 | [Rollout Serving 案例](rollout-serving-quantization-kernel-case.md) |
| 实验报告 | 数据、系统账、评测和失败归因怎样形成可复现实验 | [完整实验报告样例](world-model-efficient-training-experiment-report.md) |

工程上应先写清三张表：状态与动作 schema、训练目标与时间对齐、下游消费方与失败门禁。模型规模和架构选择应在这些接口之后讨论。

## 按问题进入

| 问题 | 建议阅读 |
| --- | --- |
| 世界模型和视频预测、仿真器、策略的边界 | [世界模型基础](beginner-world-models-primer.md) |
| 为什么部分可观测环境需要 belief state | [RSSM、Dreamer 与规划](rssm-dreamer-and-planning.md) |
| 为什么 JEPA 选择预测表示而不是像素 | [Masked / JEPA](masked-jepa-and-latent-prediction.md) |
| 动作条件为什么是“可交互”的最低门槛 | [WM / WAM / VAM](wm-wam-vam-and-action-conditioned-modeling.md) |
| 世界模型怎样用于规划而不是只展示未来 | [Planning as Inference](planning-as-inference-and-latent-actions.md) |
| 合成 rollout 怎样避免模型错误自我强化 | [模拟器与合成 Rollout](simulators-counterfactuals-and-synthetic-rollouts.md) |
| 多步预测怎样表达不确定性和尾部风险 | [风险感知规划](uncertainty-and-risk-aware-planning.md) |
| 从一个动作条件视频模型开始怎样安排里程碑 | [端到端训练案例](action-conditioned-video-world-model-training-case.md) |
| 训练和 serving 成本从哪里来 | [高效训练路线图](efficient-training-roadmap.md) → [Rollout Serving](rollout-serving-quantization-kernel-case.md) |
