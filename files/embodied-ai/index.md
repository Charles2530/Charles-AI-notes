# 具身智能专题：把模型放回真实闭环

具身智能的难点不在“让模型看懂一张图”，而在让感知、语言、动作、控制、数据回收和安全约束在真实时间里闭环。一个机器人任务会同时暴露视觉误差、动作接口、环境随机性、用户指令歧义和硬件限制，所以读这一章时要始终把模型能力放回具体任务里看。

这里会借鉴高质量公众号和博客的讲法：少堆概念，多围绕一个任务拆链路。论文和官方资料负责给事实，中文讲解材料负责提醒我们怎样把复杂系统讲得能被读者跟上。

## 从任务闭环进入

| 你想弄清的问题 | 建议先读 |
| --- | --- |
| 具身智能需要先补哪些概念 | [具身智能从零路线](beginner-learning-path.md) |
| 一个厨房、桌面或移动操作任务如何跑通闭环 | [一个任务跑通具身闭环](closed-loop-task-walkthrough.md) |
| 任务类型和 benchmark 为什么会影响模型结论 | [任务分类与评测基准](task-taxonomy-and-benchmarks.md) |
| 家庭机器人项目常见工作流和失败模式是什么 | [家庭机器人工作流与失败模式](household-robotics-workflows-and-failure-patterns.md) |
| 机器人数据里的 `observation.state` 和 `action` 到底是什么 | [机器人 State / Action 数据](state-action-robot-datasets.md) |

## 从感知和数据进入

| 环节 | 建议先读 |
| --- | --- |
| 相机、深度、标定和机器人视觉如何接起来 | [相机、深度与机器人视觉](cameras-depth-and-robot-vision.md) |
| 双目、cost volume 和 3D 估计为什么会影响抓取 | [Stereo Matching 与 Cost Volume](stereo-matching-and-cost-volume.md) |
| ASSET 式感知、抓取、轨迹链路如何拆解 | [ASSET 感知、抓取与轨迹 Pipeline](asset-perception-grasp-and-trajectory-pipeline.md) |
| Sim2Real、合成数据和数据引擎怎样补真实数据不足 | [Sim2Real 与具身数据引擎](sim2real-and-data-engines.md) |

## 从系统落地进入

| 目标 | 建议先读 |
| --- | --- |
| 规划、控制和安全约束如何接到策略输出 | [规划、控制与安全](planning-control-and-safety.md) |
| Isaac Sim、RoboTwin 等工具在项目里负责什么 | [Isaac Sim 与 RoboTwin](isaacsim-and-robotwin.md) |
| 机器人数据如何被记录、回放和可视化 | [Rerun 机器人数据可视化](rerun-robotics-data-visualization.md) |
| 部署模式、安全 case 和人工接管如何设计 | [具身部署模式与安全案例](deployment-patterns-and-safety-cases.md) |

具身智能的阅读原则是：不要只问模型“会不会”，还要问它在什么传感器、什么动作频率、什么安全边界、什么恢复策略下会不会。
