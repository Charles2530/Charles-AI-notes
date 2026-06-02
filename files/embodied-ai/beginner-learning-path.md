# 具身智能从零路线

具身智能不是“给机器人接一个大模型”。它是让模型在真实或仿真环境里感知、决定、行动、观察后果，并在失败后恢复。语言和视觉只是入口，动作、控制、数据、评测和安全才决定系统能不能落地。

## 核心问题

一个具身系统至少包含：

```text
传感器 -> 感知表示 -> 任务理解 -> 策略/规划 -> 控制器 -> 机器人执行 -> 环境反馈
```

如果只讲 VLM 或 VLA，容易忽略后半段：动作接口、低层控制、接触物理、延迟、标定、安全和失败恢复。

## 从一个抓取任务看闭环

假设任务是“把桌上的杯子拿起来”。系统需要看到杯子，估计位置和姿态，判断可抓区域，规划夹爪路径，执行接近和闭合，检测是否抓稳，失败时重试或换抓点。

这里任何一环都可能失败：相机标定错，深度噪声大，夹爪碰到桌沿，模型没理解“杯子把手”，轨迹规划穿过障碍，或者抓起后杯子滑落。具身智能的难点在于错误会真实改变世界，不是重新生成一段文本就结束。

## RT-1/RT-2：从视觉语言到动作

RT-1 和 RT-2 是理解 VLA 的好入口。RT-1 更强调机器人数据和动作序列建模；RT-2 强调把 web-scale VLM 的知识迁移到机器人动作。

![RT-1 overview](../assets/images/paper-figures/embodied-ai/rt1-figure-1-overview.png){ width="760" }

<small>图源：[RT-1: Robotics Transformer for Real-World Control at Scale](https://arxiv.org/abs/2212.06817)，Figure 1。原图表达机器人从图像和语言指令输入，输出离散化动作 token。本站用它说明：具身模型必须把感知和语言落到连续闭环控制，而不是只做图文理解。</small>

![RT-2 overview](../assets/images/paper-figures/vla/rt2-overall-wide.png){ width="920" }

<small>图源：[RT-2](https://arxiv.org/abs/2307.15818)，Figure 1。原图表达 VLM 训练和机器人动作 token 的结合。本站用它说明：web 知识迁移很有价值，但动作 token、控制频率和机器人数据仍然是落地瓶颈。</small>

## 数据比模型名更先卡住

机器人数据难在三个地方。第一，采集贵：真实机器人慢、会坏、要人工重置。第二，分布窄：一个实验室、一种夹爪、一种相机位置的数据很难覆盖真实世界。第三，接口不统一：不同机器人动作空间、相机标定、任务脚本和成功标准都不同。

![Open X-Embodiment dataset](../assets/images/paper-figures/embodied-ai/open-x-embodiment-figure-1-dataset.png){ width="760" }

<small>图源：[Open X-Embodiment](https://arxiv.org/abs/2310.08864)，Figure 1。原图表达跨机器人、跨任务的大规模 embodied dataset。本站用它说明：具身智能需要数据规模，但更需要统一动作、观测、任务和评测接口。</small>

大数据集不是自动答案。多机器人数据混在一起，如果动作语义、坐标系和时间戳不对齐，模型会学到混乱接口。数据工程要和控制接口一起设计。

## 仿真和 Sim2Real

仿真能便宜地产生数据、测试失败、覆盖危险场景。但仿真和真实世界之间有 gap：纹理、光照、接触、摩擦、传感器噪声、执行器延迟都可能不同。

Sim2Real 的关键不是“仿真越真实越好”这么简单，而是确定哪些差异会伤害任务。视觉识别任务可能更怕纹理和光照；抓取任务更怕接触和摩擦；导航任务更怕地图、动态障碍和定位误差。

## 世界模型和 VLA 怎么接

世界模型可以帮助具身系统在执行前预测后果：这个抓取会不会碰撞？这个推门动作会不会卡住？这个路径会不会进入危险区域？但它必须动作条件化，并且输出能被策略、规划器或风险检查器使用。

如果只是生成“看起来合理”的未来视频，而不能区分不同动作的后果，它对机器人控制帮助有限。具身世界模型真正有价值的证据是 closed-loop success、near-miss recall、恢复能力和真实执行成本下降。

## 学习顺序

| 阶段 | 先弄清什么 | 推荐读法 |
| --- | --- | --- |
| 1 | 传感器和动作接口 | 相机、深度、坐标系、控制频率。 |
| 2 | VLA 基本结构 | RT-1、RT-2、action token、action chunk。 |
| 3 | 数据和评测 | Open X-Embodiment、任务成功率、失败桶。 |
| 4 | 闭环和恢复 | replanning、safety checker、human intervention。 |
| 5 | 世界模型 | action-conditioned rollout、risk prediction、sim data gate。 |

## 常见误解

| 误解 | 更准确的说法 |
| --- | --- |
| 具身智能等于 VLA | VLA 只是策略层，还需要感知、控制、数据和安全。 |
| 仿真数据越多越好 | 没有 sim2real 校验，仿真偏差会进入策略。 |
| 机器人失败就是模型没学会 | 标定、控制器、夹爪、延迟和任务脚本都可能是根因。 |
| 成功率一个数字就够 | 还要看失败类型、恢复次数、碰撞、耗时和人工介入。 |

## 外部精读

- [RT-1](https://arxiv.org/abs/2212.06817)：真实机器人 Transformer policy 的早期重要工作。
- [DeepMind RT-2 blog](https://deepmind.google/blog/rt-2-new-model-translates-vision-and-language-into-action/)：VLM 到动作的直观介绍。
- [Open X-Embodiment](https://arxiv.org/abs/2310.08864)：理解跨机器人数据问题。
- [ALOHA / ACT](https://arxiv.org/abs/2304.13705)：理解 action chunk 和低成本双臂数据。
- [RoboTwin](https://arxiv.org/abs/2409.02920)：看仿真 benchmark 和双臂任务评测。
