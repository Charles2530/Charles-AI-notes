# 具身智能：把感知、策略与物理执行接成闭环

具身智能系统在真实或模拟环境中持续接收观测、选择动作、承受动作后果并更新内部状态。它的能力不能只由单帧理解、离线动作误差或成功演示定义；传感器、坐标系、控制频率、接触、延迟、安全和恢复共同决定一个策略能否运行。

## 具身系统的统一接口

一个任务 episode 可以写成：

\[
o_t \rightarrow \hat s_t \rightarrow a_{t:t+H}
\rightarrow u_t \rightarrow o_{t+1},
\]

其中 \(o_t\) 是多模态观测，\(\hat s_t\) 是状态估计或历史表示，\(a_{t:t+H}\) 是策略提出的动作块，\(u_t\) 是控制器最终发送给硬件的命令。任务条件、语言、目标图像和记忆也可以进入策略。

必须区分三种动作：

- policy action：VLA 或策略网络输出的末端位姿、关节目标、动作 token 或技能；
- planned trajectory：满足碰撞和运动学约束的时间参数化轨迹；
- control command：按控制周期发送的速度、位置、力矩或阻抗目标。

基础接口见[观测、动作与反馈闭环](beginner-learning-path.md)。

## 系统闭环

一条完整链路通常包含：

1. 任务解析与运行边界；
2. 传感器同步、标定与状态估计；
3. 策略或 VLA 产生动作候选；
4. world model、planner 或 value function 比较后果；
5. trajectory planner、controller 与 safety layer 执行；
6. success checker、异常检测和恢复状态机判断结果；
7. 日志、失败片段和人工接管进入数据引擎。

[闭环任务拆解](closed-loop-task-walkthrough.md)沿一个任务展开这些阶段；[VLA、WAM 与世界模型系统图](vla-wam-world-model-system-map.md)说明策略、预测器、规划器和控制器的职责边界。

## 感知与状态

| 问题 | 文章 |
| --- | --- |
| 相机内外参、深度与机器人坐标怎样连接 | [相机、深度与机器人视觉](cameras-depth-and-robot-vision.md) |
| 相机、末端、基座和物体坐标怎样组成可验证的变换链 | [坐标变换与手眼协调](coordinate-transforms-and-hand-eye-calibration.md) |
| 关节角怎样通过 FK 和 DH 参数变成末端位姿 | [机器人运动学与 DH 参数](robot-kinematics-and-dh-parameters.md) |
| 双目怎样从对应关系恢复深度，cost volume 为什么昂贵 | [Stereo Matching 与 Cost Volume](stereo-matching-and-cost-volume.md) |
| episode 中 state/action 字段实际表示什么 | [机器人 State / Action 数据](state-action-robot-datasets.md) |
| 3D 资产、抓取位姿和轨迹生成怎样形成数据生产线 | [资产到轨迹 Pipeline](asset-perception-grasp-and-trajectory-pipeline.md) |
| 多模态时序日志怎样回放和对齐 | [Rerun 数据可视化](rerun-robotics-data-visualization.md) |

感知输出必须落到统一时间轴和坐标系。像素检测准确但外参错误、深度尺度漂移或状态延迟，都会直接变成动作误差。

## 策略、规划与控制

VLA 和行为策略回答“根据当前观测应该做什么”，规划与控制回答“这个动作能否安全、及时地实现”。两层可以联合优化，但部署接口仍需明确。

- [VLA 专题](../vla/index.md)覆盖动作表示、策略学习、动作块和闭环恢复。
- [规划、控制与安全](planning-control-and-safety.md)覆盖 MPC、轨迹跟踪与安全投影。
- [家庭机器人流程与失败排查](household-robotics-workflows-and-failure-patterns.md)展示开放环境中的跨模块耦合。
- [人机协作与交互评测](human-robot-interaction-and-evaluation.md)处理共享空间、意图、可预期性和接管。

策略输出若绕过控制与安全层，离线效果再好也不能直接成为执行命令。反过来，过度保守的安全层也可能让任务永远无法完成，因此需要同时报告安全违规、拒绝率和任务收益。

## 数据、评测与部署

| 分支 | 主要问题 | 入口 |
| --- | --- | --- |
| 任务与 benchmark | 成功条件、组合泛化和隐藏捷径怎样定义 | [任务谱系与评测](task-taxonomy-and-benchmarks.md) |
| Sim2Real | 视觉、动力学和任务分布差怎样分开处理 | [Sim2Real 与数据引擎](sim2real-and-data-engines.md) |
| 仿真工具链 | 资产、任务、轨迹和自动判分怎样组织 | [Isaac Sim 与 RoboTwin](isaacsim-and-robotwin.md) |
| VLA 路线 | episode、模型和评测怎样放在同一张路线图 | [VLA 数据、模型与评测](vla-data-model-and-evaluation-roadmap.md) |
| 部署 | ODD、人工接管、监控和安全案例怎样形成证据 | [部署模式与安全案例](deployment-patterns-and-safety-cases.md) |

仿真、实验室和真实部署结果应分开报告。一个总成功率不足以定位问题，还需按阶段、场景、物体、动作类型、恢复次数、延迟和安全事件分桶。

## 按问题进入

| 问题 | 建议阅读 |
| --- | --- |
| 一个具身系统最少有哪些接口 | [具身智能基础](beginner-learning-path.md) |
| 从看到物体到真实抓取经过哪些坐标转换 | [相机与机器人视觉](cameras-depth-and-robot-vision.md) → [坐标变换与手眼协调](coordinate-transforms-and-hand-eye-calibration.md) |
| 关节状态、末端位姿、FK 和 IK 怎样连接 | [机器人运动学与 DH 参数](robot-kinematics-and-dh-parameters.md) → [资产到轨迹 Pipeline](asset-perception-grasp-and-trajectory-pipeline.md) |
| action 是绝对位姿、增量、速度还是控制量 | [机器人 State / Action 数据](state-action-robot-datasets.md) |
| VLA 输出后为什么仍需要 planner 和 controller | [规划、控制与安全](planning-control-and-safety.md) |
| 任务失败后怎样恢复并生成训练数据 | [闭环任务拆解](closed-loop-task-walkthrough.md) → [VLA 闭环恢复](../vla/closed-loop-recovery-and-failure-analysis.md) |
| 应该选择哪类 benchmark | [任务谱系与评测](task-taxonomy-and-benchmarks.md) |
| 仿真数据怎样迁移到真实机器人 | [Sim2Real 与数据引擎](sim2real-and-data-engines.md) |
| 家庭与人机协作场景为什么不能只测成功率 | [家庭机器人失败模式](household-robotics-workflows-and-failure-patterns.md) → [人机交互评测](human-robot-interaction-and-evaluation.md) |

返回[世界模型与具身智能总入口](../world-models-and-embodied-ai/index.md)。
