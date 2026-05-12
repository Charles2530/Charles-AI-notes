# 具身智能从零学习路线

这一页是具身智能章节的第一入口。它不从论文名开始，而是从一个更朴素的问题开始：

**机器人要在真实世界完成任务，到底需要哪些能力？**

答案不是“一个更大的模型”，而是一条闭环：

```text
看见世界 -> 理解任务 -> 选择动作 -> 预测后果 -> 执行控制 -> 判断成败 -> 回流数据
```

只要这条链里有一段不稳，系统都会失败。π0.5、DreamZero、LingBot-World、Dreamer、RoboTwin、cuRobo 这些名字，都是在强化这条链里的不同位置。

## 1. 先记住一个核心差别：静态智能 vs 具身智能

普通 VLM / LLM 多数时候回答的是静态问题：

```text
输入: 一张图 + 一个问题
输出: 一句话
```

具身智能回答的是闭环问题：

```text
输入: 当前观测 + 机器人状态 + 任务目标
输出: 一个动作
动作改变世界
系统重新观测
再输出下一个动作
```

所以具身智能最关键的不是“看懂了没有”，而是：

1. 看懂以后能不能动；
2. 动了以后世界会不会按预期变化；
3. 如果变化不对，能不能恢复；
4. 如果有危险，能不能停下或拒绝。

这就是为什么具身智能一定会同时涉及视觉、语言、动作、物理、控制、安全和数据回流。

## 2. 六个基本问题

任何具身系统都可以按六个问题拆开：

| 问题 | 初学者理解 | 典型模块 | 论文 / 项目对应 |
| --- | --- | --- | --- |
| 我看见了什么 | 图像里有哪些物体、它们在哪里、多大、什么状态 | VLM、检测、分割、pose estimation、RGB-D | [相机、深度与机器人视觉](cameras-depth-and-robot-vision.md)、资产/感知管线 |
| 我要做什么 | 任务目标是什么，下一步子任务是什么 | LLM/VLM planner、high-level policy | π0.5 的 high-level subtask |
| 我能怎么动 | 动作怎么表示，输出几步，给哪个控制器 | action token、action chunk、末端位姿、关节角 | π0.5 的 FAST + flow action expert |
| 做了会怎样 | 执行动作后未来状态、视频、风险会怎样 | world model、WAM、risk model | DreamZero、LingBot-World、Dreamer |
| 能不能安全执行 | 路径是否碰撞，力是否过大，是否越界 | IK、cuRobo、STOMP、MPC、safety filter | 规划控制页 |
| 成功了吗 | 任务是否达成，失败在哪一步 | success checker、task progress、benchmark | RoboTwin / 自建任务脚本 |

如果一篇论文没有告诉你它在回答哪一个问题，就很容易读乱。

## 3. 四类模型先分清楚

### VLM：看懂当前画面

VLM 主要回答：

\[
\text{answer} \sim p(y \mid image, text)
\]

它能识别物体、关系、文字、场景语义。但 VLM 本身不一定能控制机器人，因为它通常不输出连续动作，也不处理执行后的物理后果。

### VLA：从视觉语言到动作

VLA 主要回答：

\[
a_{t:t+H} \sim \pi(o_{\le t}, q_t, \ell)
\]

它输入视觉、机器人状态和语言，输出未来一小段动作。π0.5 就是这条路线的代表：先预测高层子任务，再用 action expert 输出连续动作块。

### 世界模型：预测动作后果

世界模型主要回答：

\[
o_{t+1:t+H} \sim p(o_{t+1:t+H} \mid o_{\le t}, a_{t:t+H-1})
\]

它的核心是 what-if：如果执行这个动作，未来会怎样。LingBot-World 是偏视频世界模拟器的路线，Dreamer 是偏 latent dynamics + RL 的路线。

### WAM：同时预测未来和动作

WAM 主要回答：

\[
(o_{t+1:t+H}, a_{t:t+H}) \sim p(o, a \mid history, goal)
\]

DreamZero 是这条路线的代表。它把未来视频和未来动作一起生成，让动作必须和未来视觉变化对齐。这样它比普通世界模型更接近 policy，也比普通 VLA 更强调动作后果。

## 4. π0.5 应该怎么读

[π0.5](../paper-deep-dives/vla/pi05.md) 最适合作为 VLA 入门论文。读它时不要先陷进模型细节，先抓住三件事。

### 第一，它解决的是开放世界长任务

普通桌面 VLA 可能是：

```text
pick up the red cube
```

π0.5 面对的是：

```text
clean the kitchen
clean the bedroom
```

这类任务要持续几分钟，需要机器人自己判断下一步该处理什么对象。所以 π0.5 不只是输出动作，还要输出 high-level subtask。

### 第二，它把动作分成两个层级

```text
high-level subtask: pick up the plate
low-level action chunk: 未来约 1 秒的连续机器人动作
```

这个分层很重要。高层子任务解决“语义上该做什么”，action chunk 解决“身体怎么动”。一个模型如果只会动作，但不知道下一步该处理哪个对象，长任务会乱；如果只会任务分解，但动作不稳，也无法落地。

### 第三，它依赖异构数据

π0.5 不只靠目标机器人自己的数据，而是混合：

- web 多模态数据：提供开放世界物体和场景知识；
- 跨机器人数据：提供更多低层操作先验；
- 移动操作数据：贴近目标机器人；
- high-level subtask：训练任务分解；
- verbal instruction：让人类像教新员工一样指导机器人。

所以 π0.5 的核心启发是：**VLA 不是只收更多动作轨迹，而是要把语义、任务分解和动作控制放进同一个训练体系。**

## 5. DreamZero 应该怎么读

[DreamZero](../paper-deep-dives/world-models/dreamzero.md) 最适合理解 WAM。它和 π0.5 的区别是：

| 问题 | π0.5 | DreamZero |
| --- | --- | --- |
| 主要目标 | 从视觉语言输出动作 | 同时生成未来视频和未来动作 |
| 强调什么 | 高层子任务 + 连续 action chunk | video-action alignment |
| 典型风险 | 长任务子任务错、动作执行错 | 未来视频错，动作也跟着错 |
| 系统需求 | 控制、安全、success checker | 实时推理、真实观测刷新、安全层 |

DreamZero 的关键循环是：

```text
当前图像 + 语言 + 机器人状态
-> 生成未来视频和动作
-> 执行动作
-> 拿到真实新观测
-> 用真实观测替换模型预测上下文
-> 继续下一段
```

读 DreamZero 时要抓住一个矛盾：它让模型学会“动作导致什么视觉变化”，这很强；但如果未来视频预测错，它可能会非常一致地执行错误计划。因此 WAM 不能绕过安全层。

## 6. LingBot-World 和 Dreamer 放在哪里

LingBot-World 和 Dreamer 都是世界模型，但用途不同：

| 系统 | 世界模型输出 | 主要用途 |
| --- | --- | --- |
| LingBot-World | 动作条件未来视频 | 交互模拟、反事实视频、数据生成 |
| Dreamer | latent state、reward、continue | 在想象中训练策略 |
| DreamZero | 未来视频 + 未来动作 | WAM 直接作为 policy |

所以世界模型不等于“视频生成”。真正要问的是：它预测的未来有没有被用于决策、风险评估、训练或数据回流。

## 7. 你的仿真 pipeline 应该放在哪里

你现在做的：

```text
3D 资产 -> 物理属性 -> 抓取轴 -> 场景 -> success checker -> 轨迹 GT
```

它是具身智能的数据底座。

| 你的工作 | 支撑什么能力 |
| --- | --- |
| 资产和纹理 | 让视觉模型看到不同物体和尺度 |
| 物理属性 | 让仿真接触、质量和摩擦更接近真实 |
| 抓取轴 | 给 planner 和策略提供可执行抓取候选 |
| 场景布置 | 生成 clean / random 训练和评测分布 |
| success checker | 自动判断轨迹是否可用 |
| 轨迹 GT | 给 VLA 做行为克隆，给 WAM 做 video-action 对齐 |

这条 pipeline 不是“传统方法”，也不是和 π0.5 / DreamZero 脱节。前沿 VLA 和 WAM 都需要可靠数据。没有资产、状态、动作和成功判定，模型训练出来也很难解释。

如果你想知道 Isaac Sim、RoboTwin、cuRobo 在这条 pipeline 里分别干什么，读 [Isaac Sim 与 RoboTwin：仿真和数据生成工具链](isaacsim-and-robotwin.md)。那一页把仿真世界、任务组织、轨迹规划和自动判定拆开解释。

## 8. 一个任务贯穿所有概念

以“把圆柱体放入尺寸匹配的罐子”为例：

```text
看见: 圆柱体和多个罐子
理解: 需要比较圆柱直径和罐口大小
选择: 选最匹配的罐子
抓取: 选择圆柱体的 grasp pose
规划: 生成从当前位置到抓取点、再到罐口的轨迹
执行: 控制夹爪闭合、移动、释放
判定: 圆柱是否在正确罐子里，夹爪是否打开，物体是否稳定
回流: 如果失败，记录是识别错、尺寸错、抓取错、规划错还是判定错
```

这就是具身智能的完整闭环。你可以用 [从一个任务跑通具身闭环](closed-loop-task-walkthrough.md) 把这条链详细走一遍。

如果你对“机器人怎么知道物体有多远、多大、在 3D 空间哪里”还不清楚，先读 [相机、深度与机器人视觉](cameras-depth-and-robot-vision.md)。那一页用图解释单目、双目、RGB-D、深度图、点云、腕部相机和固定相机。

## 9. 推荐学习顺序

如果你现在还觉得散，按这个顺序读：

1. 本页：先建立最小概念框架。
2. [相机、深度与机器人视觉](cameras-depth-and-robot-vision.md)：先弄清单目、双目、RGB-D 和机器人相机长什么样。
3. [Isaac Sim 与 RoboTwin](isaacsim-and-robotwin.md)：理解仿真、任务组织、轨迹规划和数据生成工具链。
4. [VLA、WAM 与世界模型系统地图](vla-wam-world-model-system-map.md)：把 π0.5、DreamZero、LingBot-World 放到同一张图里。
5. [从一个任务跑通具身闭环](closed-loop-task-walkthrough.md)：用具体任务把概念串起来。
6. [资产、感知、抓取与轨迹数据管线](asset-perception-grasp-and-trajectory-pipeline.md)：理解你的项目 pipeline 每一步在做什么。
7. [规划、控制与安全](planning-control-and-safety.md)：理解为什么模型输出动作后还要接控制和 safety filter。
8. [π0.5](../paper-deep-dives/vla/pi05.md)：理解 VLA 如何做开放世界长任务。
9. [DreamZero](../paper-deep-dives/world-models/dreamzero.md)：理解 WAM 如何把未来视频和动作绑在一起。
10. [LingBot-World](../paper-deep-dives/world-models/lingbot-world.md)：理解视频生成模型如何变成交互世界模拟器。

## 10. 读懂的检查标准

读完这一组页面后，你应该能回答：

1. VLM、VLA、world model、WAM 的区别是什么？
2. π0.5 为什么需要 high-level subtask？
3. DreamZero 为什么要同时生成视频和动作？
4. 为什么 success checker 要比大规模生成轨迹更早写？
5. 为什么 action chunk 不是完整 trajectory？
6. 为什么模型输出动作后仍要经过 planner、controller 和 safety filter？
7. 你的资产/轨迹 pipeline 能给 VLA、WAM 和传统 planner 分别提供什么？

如果这七个问题都能答清楚，具身智能这部分的主线就基本建立起来了。
