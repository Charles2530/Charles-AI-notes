# 世界模型专题：预测、想象与可行动环境

世界模型不等于“视频生成更逼真”。更准确地说，它试图学习环境状态如何随时间、动作和不确定性变化，并把这种预测用于规划、控制、仿真或数据生成。读这一章时，可以把模型分成三类：只预测表征的模型，能生成未来观测的模型，以及能被策略或 Agent 用来做反事实推演的模型。

这一章会参考 DeepMind、Meta、学术博客和中文公众号的讲解方式，把 Genie、Dreamer、V-JEPA、WAM/VAM 等路线放到同一张问题图里比较：学什么状态，需不需要动作，能不能交互，评测是否证明了可行动性。

## 从概念进入

| 你想弄清的问题 | 建议先读 |
| --- | --- |
| 世界模型和视频生成、仿真器、VLA 的边界在哪里 | [从零理解世界模型](beginner-world-models-primer.md) |
| RSSM、latent imagination 和 Dreamer 如何训练策略 | [RSSM、Dreamer 与规划](rssm-dreamer-and-planning.md) |
| JEPA、masked prediction 为什么强调 latent 表征 | [Masked / JEPA 与潜变量预测](masked-jepa-and-latent-prediction.md) |
| WM、WAM、VAM 为什么都在争动作条件建模 | [WM / WAM / VAM 与动作条件建模](wm-wam-vam-and-action-conditioned-modeling.md) |

## 从应用进入

| 目标 | 建议先读 |
| --- | --- |
| 用视频生成做环境 rollout 和合成数据 | [生成式仿真与视频世界模型](generative-simulation-and-video.md) |
| 给机器人或自动驾驶做规划、反事实和风险评估 | [机器人与自动驾驶应用](applications-in-robotics-and-autonomous-driving.md) |
| 理解 simulator、counterfactual、synthetic rollout 的差别 | [仿真器、反事实与合成 Rollout](simulators-counterfactuals-and-synthetic-rollouts.md) |
| 把世界模型接入 VLA 全链路 | [VLA 世界模型全链路案例](vla-world-model-full-chain-case.md) |

## 从工程进入

| 工程问题 | 建议先读 |
| --- | --- |
| 数据引擎和自我改进闭环如何组织 | [数据引擎与自我改进](data-engines-and-self-improvement.md) |
| 训练、部署、量化和 kernel 哪些环节最贵 | [高效训练路线图](efficient-training-roadmap.md)、[Rollout Serving、量化与 Kernel 案例](rollout-serving-quantization-kernel-case.md) |
| 评测怎么避免“视频好看但不能规划” | [世界模型评测与失效模式](evaluation-and-failure-modes.md) |
| 不确定性和风险如何进入规划 | [不确定性与风险感知规划](uncertainty-and-risk-aware-planning.md) |

读世界模型时，最重要的边界是：预测准确不等于可控制，生成清晰不等于可交互，短期 rollout 稳定不等于长期规划可靠。每篇文章都会尽量把这三层拆开。
