# 具身智能基础：观测、动作与反馈闭环

具身系统通过身体与环境交换信息和作用：传感器提供有限、延迟且带噪的观测，策略提出动作，规划与控制把动作变成物理执行，环境返回新的状态。能力来自整条反馈链，而不是单独一个视觉模型、语言模型或动作网络。

具身智能的系统坐标包括：任务与运行边界怎样定义；观测、状态和动作怎样组成 episode；VLA、policy、planner、controller 和 safety layer 各自负责什么；时间同步、坐标系和接触为什么是模型接口的一部分；数据与评测怎样回到闭环。

这里以机器人操作和移动操作为主要例子。自动驾驶、无人机与虚拟 agent 共享反馈控制框架，但传感器、动力学和安全要求不同。

## 前置知识

- [动态系统、控制与状态空间](../prerequisite-math/dynamic-systems-and-control.md)：状态、动态与反馈。
- [几何、视觉与三维空间](../prerequisite-math/geometry-vision-and-3d.md)：坐标、投影和位姿。
- [Agent、环境与训练闭环](../reinforcement-learning/rl-agent-environment-loop.md)：观测、动作、奖励与 episode。
- [世界模型与具身智能总入口](../world-models-and-embodied-ai/index.md)：策略、世界模型和控制器的系统关系。

## 核心机制

### 先定义任务、成功状态与运行边界

“抓取杯子”不是完整任务定义。至少要写清：

- 初始场景、目标物体和可接受目标区域；
- 机器人、夹具、相机和控制接口；
- 成功判定与最大 horizon；
- 碰撞、掉落、超力、越界和人工接管条件；
- 允许出现的人、物体、光照和布局范围；
- 失败后是重试、恢复、拒绝还是停机。

这些条件构成任务分布与 operational design domain（ODD）。若训练、评测和部署的 ODD 不一致，总成功率无法说明泛化范围。

### 观测是传感器记录，状态是决策表示

在时刻 \(t\)，系统可能接收：

\[
o_t =
\{
I_t^{1:K},\ d_t,\ q_t,\ \dot q_t,\ f_t,\ \ell
\},
\]

包括多视角图像、深度、关节状态、力觉和语言条件。观测具有不同频率与延迟，需要通过时间戳、缓存或插值对齐。

策略使用的状态可以是当前观测，也可以是历史编码或 belief：

\[
\hat s_t = e(o_{\le t},a_{<t}).
\]

遮挡、物体被机械臂挡住、接触状态和任务进度都可能要求历史。把最后一帧图像直接视为完整状态，会让策略在部分可观测任务中反复丢失上下文。

### state 和 action 字段是一份控制协议

机器人数据中的 `state` 可能表示关节角、末端位姿、夹爪状态或它们的组合；`action` 可能表示：

- 绝对关节目标；
- 关节或末端增量；
- 速度；
- 力矩；
- 阻抗或力控制目标；
- 技能与参数。

动作必须同时带上坐标系、单位、归一化、控制周期和时间语义。同一个六维向量若分别表示世界坐标绝对位姿和末端局部增量，学习问题完全不同。字段解析见[机器人 State / Action 数据](state-action-robot-datasets.md)。

### 策略、规划与控制是三层接口

策略或 VLA 输出：

\[
a_{t:t+H}\sim\pi_\theta(\cdot\mid \hat s_t,\ell).
\]

它可能是一段末端轨迹目标、动作块或技能。随后：

1. planner 检查运动学、碰撞和可达性，形成时间参数化参考；
2. safety layer 对越界、危险接触和未知状态进行否决或投影；
3. controller 以更高频率计算硬件命令，使真实状态跟踪参考。

高层策略通常运行在数 Hz 到几十 Hz，底层控制可能更快。推理延迟若接近动作周期，模型看到的状态和动作真正生效时的状态已经不同。系统必须明确 observation timestamp、inference finish、command dispatch 和 actuator response。

### 闭环执行不断重新读取动作后果

闭环不是把长动作序列一次性发完，而是在执行中读取新观测：

```text
observe and synchronize sensors
estimate state and task progress
predict action or action chunk
check feasibility and safety
execute a bounded prefix
observe the consequence
detect success, deviation, or failure
continue, recover, ask for help, or stop
```

动作块可以提高时间一致性并降低有效 horizon，但块太长会减少纠错机会。执行中应支持 receding horizon：只执行计划前缀，再根据最新状态更新。

### 接触让机器人任务不同于纯视觉任务

抓取、插入、开门和工具使用依赖接触。图像可能看起来接近目标，但夹爪力、接触法向、摩擦或对象约束不正确。只用视觉终态判定会漏掉：

- 虚抓或夹持不稳；
- 物体已接触但没有插入；
- 门把手跟随机械臂移动却未解锁；
- 目标完成但发生过危险碰撞。

success checker 应结合视觉、几何、本体状态、力觉和任务状态机。

### VLA 只覆盖策略层

VLA 把视觉、语言和状态映射到动作，擅长复用语义表示与跨任务数据。它不自动解决：

- 相机标定与坐标变换；
- action schema 和机器人适配；
- 碰撞检测与轨迹可达性；
- 高频控制和硬件实时性；
- success checking、恢复与安全案例。

因此 VLA 应通过明确协议接入其上下游。完整路线见[VLA 专题](../vla/index.md)。

### 世界模型提供动作后果预测

世界模型可以在执行前比较候选动作：

\[
\hat z_{t+1:t+H}
\sim p_\phi(\cdot\mid z_t,a_{t:t+H}).
\]

它可用于 MPC、风险估计、反事实或合成数据。世界模型预测的是后果，不负责保证底层轨迹可执行；它和 planner、controller、safety layer 仍需分工。

### 数据单元是 episode，不是独立帧

一个可用 episode 至少要对齐：

- 多视角传感器与 calibration；
- 机器人 state、commanded action 和实际执行状态；
- 语言任务、阶段或技能标签；
- success、failure、termination 与人工接管；
- 软件、模型、硬件和环境版本。

若只保留成功轨迹，策略难以学习如何识别偏离和恢复。数据引擎需要把 near-miss、拒绝、碰撞前兆、恢复成功与恢复失败作为独立资产。

### 评测必须回到闭环

离线 action error、视频指标和单帧识别可以诊断模块，但最终需要 episode 指标：

- 任务成功率与分阶段成功率；
- 扰动后恢复率和平均恢复次数；
- 碰撞、超力、越界与急停；
- 推理延迟、控制频率和动作平滑性；
- 人工接管、拒绝和无效尝试；
- 新物体、新布局、新任务和新 embodiment 的泛化。

实验室、仿真和部署结果应分开报告。相同任务名在不同初始条件和 success checker 下不可直接比较。

## 常见误解与失效边界

**把具身智能等同于 VLA。**VLA 是策略层；完整系统还包含感知、状态、规划、控制、安全、硬件和数据闭环。

**认为真实机器人只是在仿真策略后面接驱动。**视觉、动力学、延迟、摩擦、线缆和安全分布都会变化，需要 system identification、randomization 和真实 hard set。

**只看成功演示。**固定初始状态下的成功无法证明对扰动、失败恢复和长期运行的能力。

**忽略 commanded action 与 executed action 的差异。**控制器饱和、碰撞投影和通信延迟会让实际动作偏离策略输出，训练与评测都应记录两者。

**用单一成功率比较系统。**任务定义、horizon、重置、人工帮助和判定器不同，数字没有可比性。

**认为更大模型自动解决接触和实时性。**模型容量不能替代正确的坐标、频率、力觉、控制与安全接口。

## 检查理解

1. observation、state、policy action 和 control command 分别是什么？
2. 为什么动作向量必须同时说明坐标系、单位和时间周期？
3. 动作块怎样改善一致性，又怎样减少闭环纠错机会？
4. VLA 输出为什么不能直接绕过 planner 与 safety layer？
5. commanded action 与 executed action 的差异怎样进入数据和评测？
6. 为什么具身评测必须报告恢复与安全事件？

## 外部材料

- 原始来源：[RT-1](https://arxiv.org/abs/2212.06817)、[RT-2](https://arxiv.org/abs/2307.15818)、[Open X-Embodiment](https://arxiv.org/abs/2310.08864)、[DROID](https://arxiv.org/abs/2403.12945)、[OpenVLA](https://arxiv.org/abs/2406.09246)。
