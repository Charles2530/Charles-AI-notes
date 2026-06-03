# 世界模型评测与失效模式来源台账

这份台账服务于 [世界模型评测：不要让好看的未来骗过决策系统](../world-models/evaluation-and-failure-modes.md)。正文不把世界模型评测写成视频指标清单，而是按消费方组织：状态编码器、视频模拟器、planner、risk gate、data engine 和 policy evaluator 分别需要不同证据。

## 使用方式

第一类来源是经典控制世界模型。PlaNet 和 DreamerV3 支撑 open-loop / imagined rollout / planning 的基本接口：模型不是只预测像素，而是要提供可被 planner 或 actor 消费的 latent future、reward、continuation 或 value 信号。

第二类来源是视频生成评测。FVD、VBench、FVD bias 论文用于说明视频质量指标有价值，但不能外推为动作效用。正文把它们放在“画面尺”里，而不是拿它们判断机器人任务是否能成功。

第三类来源是具身世界模型评测。World-in-World、WoW-World-Eval、WorldArena 和 WorldArena 2.0 代表 2025-2026 年评测从 open-loop 视觉质量走向 closed-loop task utility、functional utility、multimodal interaction 和 cross-platform performance。正文使用这些来源时，重点是“先定义消费方，再定义指标”。

第四类来源是动作条件和风险校准。DreamZero 用 WAM 失败案例提醒：未来视频和未来动作可能一起错，并且错得很连贯；CausalVQA 和 V-JEPA 2 benchmark 提醒评测要覆盖 counterfactual、hypothetical、anticipation、planning 等物理因果问题；Guo et al. calibration paper 支撑风险分数按桶校准的基本口径。

中文技术媒体、公众号和社区材料只学习讲法：先讲为什么“视频好看”会骗过决策，再讲新基准如何把评价对象拉回机器人执行。具体数字、论文结论和模型能力仍回到 arXiv、官方博客、项目页或代码仓库核对。

## 写作边界

不要把 FVD、VBench、视觉质量或官方 demo 直接写成世界模型可用于规划。它们最多说明画面和时序质量。若正文声称模型可用于动作选择，必须补 action sensitivity、candidate ranking、closed-loop success、failure replay 或 cost per success。

不要把 closed-loop 成功率写成所有能力都强。闭环结果仍受任务、平台、动作接口、低层控制器、仿真真实性、安全层和数据分布影响。正文要保留“该 benchmark 设置下”的边界。

不要把中文媒体的产业判断当成强证据。它们适合学习问题意识和叙事结构；涉及性能、相关系数、成功率、模型规格、延迟和安全结论时，必须回到原论文或官方来源。

## 图片使用

本页遵守用户约束：不使用 image2，不生成新图。正文只复用已有论文图，并删除了质量报告会标为 medium、且第一眼不够清晰的 PlaNet open-loop 图。

| 图片 | 本地路径 | 来源 | 用途 |
| --- | --- | --- | --- |
| DreamZero failure case | `../assets/images/paper-deep-dives/world-models/dreamzero/figure-16-failure-case.png` | [World Action Models are Zero-shot Policies](https://arxiv.org/abs/2602.15922) Figure 16 | 说明 WAM 失败要同时检查生成未来、未来动作、真实执行和安全层。 |

## 来源表

| 来源 | 类型 | 支撑正文哪部分 | 证据边界 |
| --- | --- | --- | --- |
| [Learning Latent Dynamics for Planning from Pixels](https://arxiv.org/abs/1811.04551) | 论文 | Open-loop / multi-step latent dynamics、planning from pixels 的经典接口。 | 经典控制任务；不代表现代视频世界模型评测充分。 |
| [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104) | 论文 | DreamerV3 imagined rollout、actor/value 和多任务 control 证据。 | 证明 Dreamer-style control world model；不等于视频模拟器指标。 |
| [Towards Accurate Generative Models of Video](https://arxiv.org/abs/1812.01717) | 论文 | FVD 作为视频生成分布质量指标的入口。 | FVD 是视频质量指标，不直接衡量动作 utility。 |
| [VBench](https://arxiv.org/abs/2311.17982) | 论文 / GitHub | 视频生成质量拆成 subject consistency、motion smoothness、temporal flickering 等维度。 | 评测视频生成质量；不证明 closed-loop planner 可用。 |
| [On the Content Bias in Fréchet Video Distance](https://arxiv.org/abs/2404.12391) | 论文 | 支撑“单一 FVD 可能偏向帧质量，不足以衡量 temporal / physical utility”。 | 是对 FVD 偏差的分析；不替代任务评测。 |
| [World-in-World](https://arxiv.org/abs/2510.18135) | 论文 | 视觉质量不保证任务成功；closed-loop platform 和 task success 作为核心评测。 | 仍受四个环境、统一 planner 和动作 API 设置限制。 |
| [WoW-World-Eval](https://arxiv.org/abs/2601.04137) | 论文 | 感知、规划、预测、泛化、执行与 IDM Turing Test 评测具身视频世界模型。 | 视频基础模型作为具身先验的 benchmark；不等于真实部署全覆盖。 |
| [WorldArena](https://arxiv.org/abs/2602.08971) | 论文 / Benchmark | 感知质量、functional utility、人类偏好、data engine / policy evaluator / action planner。 | benchmark 结果要按模型、任务和模拟器设置理解。 |
| [WorldArena 2.0](https://arxiv.org/abs/2605.17912) | 论文 / Benchmark | 截至 2026-06-03 核到的最新扩展：vision-only -> visuotactile，offline -> interactive RL environment，simulator-only -> simulated and real-world robotic settings。 | 2026-05-18 新提交论文；后续版本和 leaderboard 需再核。 |
| [World Action Models are Zero-shot Policies](https://arxiv.org/abs/2602.15922) | 论文 / 图源 | WAM、joint video-action prediction、closed-loop control 和失败案例。 | 论文结果来自作者设置；不写独立复现。 |
| [CausalVQA](https://arxiv.org/abs/2506.09943) | Benchmark 论文 | counterfactual、hypothetical、anticipation、planning、descriptive 五类物理因果问题。 | VQA 形式评估推理；不直接等同机器人执行成功。 |
| [Meta V-JEPA 2 blog](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/) | 官方博客 | 物理推理 benchmark 和 zero-shot robot planning 的能力形态。 | 官方解释；具体实验边界回论文和项目页。 |
| [Genie 3](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/) | 官方博客 | 交互式世界模型能力形态与 limitations：动作空间、交互对象、真实地点、文本、持续时间。 | 官方 research preview；不能外推为机器人闭环控制。 |
| [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599) | 论文 | confidence / probability calibration 和温度缩放背景，支撑风险桶校准。 | 分类校准基础来源；世界模型风险校准仍需任务化定义。 |
| [智源社区：World-in-World 中文页](https://hub.baai.ac.cn/paper/ed6ea002-a8f0-4b22-9a61-a92ce16ed88d) | 中文论文解读 | 学习中文讲法：从“视觉逼真”转向“具身实用性”。 | 二次解读；事实回 arXiv。 |
| [智源社区：WoW-World-Eval 中文页](https://hub.baai.ac.cn/paper/7cf6a004-2ca9-4118-87bf-6a93e24e2528) | 中文论文解读 | 学习把感知、规划、预测、泛化、执行组织成读者可读的能力框架。 | 二次解读；事实回 arXiv。 |
| [量子位 / 智源：世界模型真正的考场在机器人执行里](https://hub.baai.ac.cn/view/54116) | 中文技术媒体 / 公众号转载 | 学习“视频榜单和机器人执行错位”的开篇方式，以及用行业问题引出评测对象。 | 行业材料；具体模型数字和 claim 需追原文。 |

## 后续维护

新增世界模型评测来源时，先标清它评测的是哪一层：画面质量、动作可控性、闭环成功、风险校准、数据增益、policy evaluation 相关性，还是部署成本。若来源只给总分，不给样本、协议、动作接口和失败回放，正文只能把它当作阅读入口，不能当强证据。
