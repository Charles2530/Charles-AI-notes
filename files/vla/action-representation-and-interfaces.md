# VLA 动作表示与控制接口

VLA 最容易被讲成“把语言、图像和动作都 token 化”。这句话太粗。真正的问题是：动作到底用什么坐标系、什么频率、什么粒度表示，模型输出后由谁执行，执行失败时如何反馈给策略。

## 核心问题

机器人动作不是一句文本。它可能是末端位姿、关节角、速度、夹爪开合、轨迹 waypoint、action chunk，甚至是高层技能名。动作表示一旦选错，模型看起来会“懂任务”，但落到真实控制时会抖、慢、撞、抓不稳。

一个 VLA 闭环可以拆成：

```text
视觉/语言/状态 -> policy -> action representation -> controller -> robot -> new observation
```

动作表示处在 policy 和 controller 之间，是语义理解能否变成物理执行的接口。

## 动作可以长什么样

| 动作形式 | 例子 | 适合场景 | 风险 |
| --- | --- | --- | --- |
| 离散 token | `move_left`, `grasp`, `open_drawer` | 高层任务和游戏式环境 | 粒度粗，难落到连续控制。 |
| 末端位姿 delta | \(\Delta x,\Delta y,\Delta z,\Delta R\) | 机械臂 manipulation | 坐标系和尺度错会直接失败。 |
| 关节控制 | joint position / velocity | 底层控制或仿真 | 学习难度高，跨机器人差。 |
| waypoint / trajectory | 多个未来点 | 长动作、平滑控制 | 需要时间对齐和控制器跟踪。 |
| action chunk | 一次输出短窗口动作 | 减少频繁推理，提高连贯性 | 错了会连续错几步。 |
| latent skill | 高层技能向量 | 层级策略 | 可解释性和安全边界弱。 |

## RT-2：把动作写进语言 token

RT-2 的关键设计是把机器人动作离散化成 token，让 VLM 可以像生成文本一样生成动作。这个设计很大胆，因为它把互联网视觉语言知识和机器人控制接到同一个模型接口上。

![RT-2 overall 原论文图](../assets/images/paper-figures/vla/rt2-overall-wide.png){ width="920" }

<small>图源：[RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control](https://arxiv.org/abs/2307.15818)，Figure 1。原图表达 RT-2 将 web-scale VLM 训练和机器人动作 token 输出连接起来。本站用它说明：VLA 的核心不是“多模态输入”，而是让动作进入同一个可生成接口。</small>

这张图要从右往左读：最终机器人执行的是动作，不是文本答案。中间的关键桥梁是 action tokenization。模型能把视觉和语言知识迁移到机器人任务上，但动作 token 的离散化、频率和控制接口决定了它能执行到什么程度。

## ACT：动作分块为什么有用

Action Chunking with Transformers 的直觉是：很多 manipulation 任务不是一步一步孤立决策，而是一小段连续动作。一次输出多个未来动作，可以减少每步推理延迟，也让轨迹更连贯。

![ALOHA ACT algorithm](../assets/images/paper-figures/vla/aloha-act-algorithm.png){ width="720" }

<small>图源：[Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705)，ACT 方法图。原图表达 policy 根据当前观测输出一段 action sequence，并用 temporal ensembling 平滑执行。本站用它说明：action chunk 是短时控制接口，不是完整任务计划。</small>

Action chunk 的代价也明显：如果前几步已经偏了，后面 chunk 可能继续把错误执行下去。因此工程上要配合重规划频率、执行监控、chunk overlap 和安全控制器。

## 坐标系比模型名更重要

动作表示一定要回答坐标系：动作是在 robot base frame、camera frame、end-effector frame，还是 world frame？单位是米、像素、归一化值，还是离散 bin？旋转用 Euler、quaternion、axis-angle，还是 6D rotation？

很多 VLA 失败不是模型“不懂语言”，而是动作接口不对齐：训练数据里动作在一种坐标系，部署机器人在另一种坐标系；相机外参变了但模型不知道；夹爪开合的归一化方向反了；控制频率从 10Hz 变成 30Hz 后动作尺度没调。

## 频率和延迟决定闭环质量

VLA policy 不可能无限快。假设模型推理 200ms，控制器 50Hz，机器人执行动作又有延迟，那么一次 policy 输出和真实世界反馈之间已经跨过很多低层控制周期。

这就是为什么有些系统用 action chunk，有些系统用高层 planner + 低层 controller，有些系统把视觉 encoder 缓存或蒸馏到更小模型。动作表示不是独立模块，它和推理延迟、控制频率、传感器刷新、fail-safe 一起决定闭环。

## 常见误解

| 误解 | 更准确的说法 |
| --- | --- |
| 把动作 token 化就成了 VLA | 还要解决坐标系、频率、控制器和安全边界。 |
| 动作越底层越通用 | 底层动作跨机器人更难，数据需求更大。 |
| action chunk 只是在提速 | 它同时改变了误差传播和闭环反馈频率。 |
| VLA 可以跳过控制器 | 真实机器人仍需要低层控制、限幅、碰撞检查和恢复逻辑。 |

## 外部精读

- [DeepMind RT-2 blog](https://deepmind.google/blog/rt-2-new-model-translates-vision-and-language-into-action/)：理解 VLM 到动作的路线。
- [RT-2 paper](https://arxiv.org/abs/2307.15818)：看 action tokenization 和迁移实验。
- [ALOHA / ACT](https://arxiv.org/abs/2304.13705)：理解 action chunking。
- [Open X-Embodiment](https://arxiv.org/abs/2310.08864)：看跨机器人数据和动作接口问题。
- [Octo](https://arxiv.org/abs/2405.12213)：看通用机器人策略的数据和 action 表达。
