# VLA 阅读入口：把视觉、语言和动作接成闭环

VLA 的核心不是“多模态模型再加一个动作头”。真正困难的是把视觉证据、语言目标、动作表示、低层控制、数据回收和安全接管放进同一个闭环。一个模型如果只会说出任务意图，却不能稳定输出可执行动作；或者离线评测很高，却不能在真实环境里恢复错误，都还不能算可靠的 VLA 系统。

这一章会参考优秀公众号、博客和论文综述的讲法，从任务链路而不是模型名开始讲。事实部分以 RT-2、Open X-Embodiment、ALOHA/ACT、SayCan、pi0/GR 系列等论文和官方材料为准，每篇页面都有对应来源台账。

## 先理解动作接口

| 你想弄清的问题 | 建议先读 |
| --- | --- |
| VLA 的动作 token、连续控制、末端执行器命令有什么差别 | [动作表示与控制接口](action-representation-and-interfaces.md) |
| 视觉语言模型怎样接到策略学习和机器人数据 | [VLA 数据与策略学习](data-and-policy-learning.md) |
| 为什么 ACT、action chunking 和 latent skill 会影响稳定性 | [动作分块、层级策略与潜在技能](action-chunking-hierarchical-policies-and-latent-skills.md) |

## 再看数据、评测和闭环

| 工程问题 | 建议先读 |
| --- | --- |
| 跨机器人数据、benchmark 和数据引擎如何组织 | [VLA 评测与数据引擎](benchmarks-and-data-engines.md) |
| 模型失败后如何检测、恢复和人工接管 | [闭环恢复与失败分析](closed-loop-recovery-and-failure-analysis.md) |
| 真实部署时安全边界、权限和回滚如何设计 | [VLA 部署与安全](deployment-and-safety.md) |
| 和具身系统、世界模型如何衔接 | [具身智能阅读入口](../embodied-ai/index.md)、[世界模型阅读入口](../world-models/index.md) |

## 读 VLA 时抓什么

每篇 VLA 工作都建议用六个问题检查：它使用什么观测和动作空间；数据来自远程操作、仿真、真实机器人还是混合数据；模型输出是否被低层控制器改写；评测是否包含长时程任务和失败恢复；安全策略是否只停留在文本描述；部署时是否有日志、回放和人工接管。
