# 机器人 `state/action` 数据来源台账

这份台账服务于 [机器人数据里的 `state` 和 `action`](../embodied-ai/state-action-robot-datasets.md)。正文目标不是做机器人数据格式综述，而是帮助读者在 LeRobot、CALVIN、robomimic 或自采 parquet 中判断 `observation.state` 与 `action` 的真实语义。

## 使用方式

第一类来源是数据格式。LeRobotDataset v3.0 官方博客和 Hugging Face Robotics Course 支撑正文关于 parquet、video、metadata、feature schema、stats、episode 边界和 normalization 的说法。正文使用 LeRobot v3 pillars 图说明：不能只看单行 parquet，要看 metadata 如何给一行数据定位。

第二类来源是 observation/action schema。robomimic observations 文档支撑 observation keys、modalities、low_dim/RGB/depth 的区分；robomimic action configuration 文档支撑 action components、拼接顺序和 normalization 的讨论。

第三类来源是 action space 与 controller。CALVIN README 支撑 absolute Cartesian pose、relative Cartesian displacement 和 joint action 的分类；robosuite controllers 文档支撑 controller 对 policy action 的解释；MoveIt Pro IK 文档支撑 Cartesian 末端目标需要通过 IK 转成 joint states。

第四类来源是 VLA 动作接口。RT-2 官方博客支撑 action tokens 的图示与讲法：动作可以写成 token，但每个 token 仍然对应具体控制量、坐标系和执行协议。

## 写作边界

正文不把 `observation.state` 简化成“关节角”。很多数据集会把末端位姿、夹爪、关节位置、速度、力传感器或其他本体状态拼在一起；具体含义必须看 schema 和采集代码。

正文不把 `action` 简化成 delta。`action` 可以是绝对关节目标、关节增量、速度、末端绝对位姿、末端相对位姿、力矩、action chunk 或离散 token。是否执行 `state + action` 必须由 action space 和 control mode 决定。

正文不把同帧 `state == action` 直接判断为字段错误。它也可能来自 post-action alignment、下一帧目标被 shift、归一化域不一致、采样频率高导致相邻帧变化很小，或 episode 边界比较错误。

正文里的 Kai0 判断来自用户提供的系统笔记，只作为排查口径示例。最终项目文档应以实际 parquet、metadata、控制脚本和误差分布验证为准。

## 图片使用

| 图片 | 本地路径 | 来源 | 用途 |
| --- | --- | --- | --- |
| LeRobotDataset v3 pillars | `../assets/images/embodied-ai/robot-state-action-data/lerobot-v3-pillars.png` | [Hugging Face LeRobotDataset v3.0 blog](https://huggingface.co/blog/lerobot-datasets-v3) | 说明 tabular data、visual data、metadata 的三层组织。 |
| CALVIN sensors | `../assets/images/embodied-ai/robot-state-action-data/calvin-sensors.png` | [CALVIN GitHub README](https://github.com/mees/calvin) | 说明机器人 observation 可以同时包含多视角、多模态和本体状态。 |
| RT-2 action string | `../assets/images/embodied-ai/robot-state-action-data/rt2-action-string.png` | [Google DeepMind RT-2 blog](https://deepmind.google/blog/rt-2-new-model-translates-vision-and-language-into-action/) | 说明 VLA action token 仍然对应具体控制量。 |

## 来源

| 来源 | 类型 | 本站使用方式 | Checked Date |
| --- | --- | --- | --- |
| [LeRobotDataset v3.0 blog](https://huggingface.co/blog/lerobot-datasets-v3) | 官方博客 / 图源 | 支撑 v3 数据三支柱、feature schema、stats、episode metadata 和多 episode parquet。 | 2026-06-15 |
| [LeRobotDataset docs source](https://github.com/huggingface/lerobot/blob/main/docs/source/lerobot-dataset-v3.mdx) | 官方文档源码 | 交叉核对 v3 设计和目录结构。 | 2026-06-15 |
| [Hugging Face Robotics Course: LeRobotDataset](https://huggingface.co/learn/robotics-course/unit1/3) | 官方课程 | 支撑面向初学者的多模态、时间序列、episodic 数据解释。 | 2026-06-15 |
| [robomimic Multimodal Observations](https://robomimic.github.io/docs/tutorials/observations.html) | 官方文档 | 支撑 observation keys、modalities、groups 和 low_dim/RGB/depth 的区分。 | 2026-06-15 |
| [robomimic Action Configuration](https://robomimic.github.io/docs/tutorials/action_configs.html) | 官方文档 | 支撑 action components、顺序、normalization 和 action config 的解释。 | 2026-06-15 |
| [CALVIN GitHub README](https://github.com/mees/calvin) | 官方仓库 / 图源 | 支撑 sensory observations 与三类 action space。 | 2026-06-15 |
| [CALVIN paper](https://arxiv.org/abs/2112.03227) | 论文 | 作为 CALVIN benchmark 的论文入口。 | 2026-06-15 |
| [robosuite Controllers](https://robosuite.ai/docs/modules/controllers.html) | 官方文档 | 支撑 controller 对 action space、delta、policy frequency 和 joint torque 的解释。 | 2026-06-15 |
| [MoveIt Pro IK docs](https://docs.picknik.ai/how_to/robotics_applications/computing_inverse_kinematics/) | 官方文档 | 支撑 IK 定义：由末端目标位姿求 joint states。 | 2026-06-15 |
| [RT-2 blog](https://deepmind.google/blog/rt-2-new-model-translates-vision-and-language-into-action/) | 官方博客 / 图源 | 支撑 action string、VLA 和 action token 的直观解释。 | 2026-06-15 |
| [RT-2 paper](https://arxiv.org/abs/2307.15818) | 论文 | 作为 VLA action tokenization 和机器人控制的论文入口。 | 2026-06-15 |

## 后续维护

如果后续加入某个具体数据集，例如 Kai0、DROID、Open X 或 RoboTwin，应优先补充四项内容：`meta/info.json` 中 feature schema、`meta/stats.json` 中 normalization、真实控制脚本中的 controller/action mode、以及按 episode 分组后的 `action_t` 与 `state_t/state_{t+1}` 误差分布。没有这四项，只能写成假设，不能写成确定结论。
