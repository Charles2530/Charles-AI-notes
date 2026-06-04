# 具身智能与 VLA QA

这一页单独整理具身智能、VLA、机器人数据和闭环部署的高频问题。这里的易错点通常不是概念不会背，而是把 VLM 能力、视频生成能力、离线指标或 demo 结果过度外推到真实机器人执行。

## 基础定义与系统边界

### Q：具身智能和普通多模态 AI 的关键区别是什么？

**面试回答。** 普通多模态 AI 多数停留在感知和问答：看图、读视频、回答问题。具身智能要让智能体在物理或仿真环境中感知、决策、行动，并承受动作带来的后果。它的闭环是 observation -> policy -> action -> environment -> new observation。

**追问展开。** 面试时可以说：具身智能多了动作、时间、物理约束、安全和反馈。错误不只是文本答错，而可能导致碰撞、抓取失败或任务中断。  
**易错点 / 边界。** 不要把“能理解图片”当成“能执行任务”。感知能力只是具身系统的一部分。  
**继续读。** [具身智能从零路线](../embodied-ai/beginner-learning-path.md) / [任务分类与 Benchmark](../embodied-ai/task-taxonomy-and-benchmarks.md)

### Q：VLA 是什么？和 VLM 的关键差别是什么？

**面试回答。** VLA 是把视觉、语言和动作接在一起的模型或系统：输入图像/视频和语言指令，输出机器人动作或动作 token。VLM 输出文本，VLA 输出会改变物理世界的动作，所以必须处理控制频率、动作空间、本体差异和安全约束。

**追问展开。** 面试中可以说 VLA 比 VLM 多了两个难点：动作接口和闭环执行。动作一旦执行会改变环境，下一步观察又会变，所以错误会累积。  
**易错点 / 边界。** 不要把“VLM 能解释图片”当成“VLA 能控制机器人”。文本语义和物理执行之间还有巨大鸿沟。  
**继续读。** [VLA 动作表示与控制接口](../vla/action-representation-and-interfaces.md) / [RT-2：视觉语言模型接到机器人动作](../paper-deep-dives/embodied-ai/rt2.md)

### Q：VLA、Policy、Planner 和 World Model 各自负责什么？

**面试回答。** Policy 负责从当前状态输出动作；Planner 负责搜索或选择一段动作计划；World Model 负责预测动作后果；VLA 是一种把视觉、语言和动作接起来的策略或系统形态。它们可以组合：VLA 给候选动作，world model 评估未来，planner 选择更好的动作。

**追问展开。** 面试里可以用一句话区分：VLA 问“做什么动作”，world model 问“做了以后会怎样”，planner 问“哪组动作最好”。  
**易错点 / 边界。** 不要把所有机器人模型都叫 VLA，也不要把能输出动作的模型自动称为世界模型。  
**继续读。** [VLA、WAM 与世界模型系统图](../embodied-ai/vla-wam-world-model-system-map.md) / [WM / WAM / VAM：动作条件建模](../world-models/wm-wam-vam-and-action-conditioned-modeling.md)

## 动作表示与控制接口

### Q：动作 token 为什么不是随便离散化几个数字？

**面试回答。** 动作 token 要表达连续控制量，例如末端位姿、关节角、夹爪状态、速度或 delta action。离散化方式会影响精度、平滑性、训练难度和跨机器人泛化。动作 token 不是普通文本 token，它直接连接控制系统。

**追问展开。** 如果粒度太粗，控制不准；太细，序列变长、学习困难、错误累积。不同机器人本体动作空间不同，还需要标准化或适配层。  
**易错点 / 边界。** Token 数更少不一定更好；最终要看真实执行成功率、动作平滑和失败恢复。  
**继续读。** [VLA 动作表示与控制接口](../vla/action-representation-and-interfaces.md) / [VLA 动作分块、层级策略与潜在技能](../vla/action-chunking-hierarchical-policies-and-latent-skills.md)

### Q：绝对动作、相对动作和末端位姿动作怎么选？

**面试回答。** 绝对动作直接给目标关节或位姿，表达清晰但对标定和本体差异敏感；相对动作输出当前位置的增量，适合闭环纠偏；末端位姿动作更贴近任务空间，如移动到某个 6DoF pose，再由低层控制器转成关节命令。

**追问展开。** 面试回答可以按任务说：精细操作常需要相对动作和高频反馈，长距离移动可能用高层 waypoint，跨机器人泛化则要考虑动作规范化和控制接口抽象。  
**易错点 / 边界。** 动作空间不是模型输出格式问题，而是控制系统接口问题。错误动作表示会让学习和部署都变难。  
**继续读。** [VLA 动作表示与控制接口](../vla/action-representation-and-interfaces.md) / [规划、控制与安全](../embodied-ai/planning-control-and-safety.md)

### Q：Action chunking 为什么在机器人策略里常见？

**面试回答。** Action chunking 一次预测一段动作序列，而不是每个控制周期都重新推理。它能降低推理频率，提高动作平滑性，并让模型在较长时间范围内表达意图。ACT、diffusion policy 和许多 VLA 路线都用类似思想。

**追问展开。** Chunk 长度是 tradeoff：长 chunk 推理更省、动作更连贯；短 chunk 反馈更及时，适合动态环境。Temporal ensembling 还可以把多个 chunk 的重叠预测平滑融合。  
**易错点 / 边界。** Chunking 不是越长越好。接触、避障和动态目标需要频繁闭环纠错。  
**继续读。** [VLA 动作分块、层级策略与潜在技能](../vla/action-chunking-hierarchical-policies-and-latent-skills.md) / [闭环恢复与失败分析](../vla/closed-loop-recovery-and-failure-analysis.md)

### Q：Diffusion Policy 为什么适合连续动作？

**面试回答。** 连续机器人动作常有多峰分布：同一个任务可能有多条可行轨迹。用 MSE 回归动作容易平均多种模式，得到不可执行的中间轨迹；diffusion policy 把动作序列当生成对象，可以表达多峰轨迹分布。

**追问展开。** 面试里可以说 diffusion policy 生成的是 action trajectory，而不是图像。它通过条件去噪从噪声动作序列生成可执行动作序列，适合复杂操作和多解任务。  
**易错点 / 边界。** Diffusion policy 不是天然闭环安全；采样步数、推理延迟、chunk 长度和失败恢复都要工程化处理。  
**继续读。** [VLA 数据与策略学习](../vla/data-and-policy-learning.md) / [VLA 动作分块、层级策略与潜在技能](../vla/action-chunking-hierarchical-policies-and-latent-skills.md)

### Q：层级策略和 latent skill 解决什么？

**面试回答。** 层级策略把长任务拆成高层意图和低层控制。高层决定子目标或技能，如“抓起杯子”“移动到桌边”；低层负责连续控制。Latent skill 则把一段动作压缩成可复用技能表示，降低长时程决策难度。

**追问展开。** 这对长任务很重要，因为直接预测几百步低层动作很难。层级结构能减少搜索空间，也更容易结合语言规划和低层控制器。  
**易错点 / 边界。** Skill 可复用不等于自动泛化；技能边界、终止条件和失败恢复都需要评估。  
**继续读。** [VLA 动作分块、层级策略与潜在技能](../vla/action-chunking-hierarchical-policies-and-latent-skills.md) / [规划、控制与安全](../embodied-ai/planning-control-and-safety.md)

## 数据、训练与 Sim2Real

### Q：具身智能为什么总卡在数据闭环？

**面试回答。** 机器人数据采集贵、慢、危险且异构；失败样本和长尾场景尤其难收集。具身智能系统要持续采集任务、发现失败、标注或自动修复、再训练、再评测，形成数据闭环。没有数据闭环，模型很难持续覆盖真实环境变化。

**追问展开。** Web-scale 图文数据能提供语义先验，但接触、动作、力、失败恢复和安全边界必须来自机器人或高质量仿真/真实轨迹。  
**易错点 / 边界。** 不要以为扩大 VLM 数据就能自然解决机器人控制；动作数据和闭环反馈是另一类资源。  
**继续读。** [具身智能从零路线](../embodied-ai/beginner-learning-path.md) / [VLA 数据与策略学习](../vla/data-and-policy-learning.md)

### Q：机器人数据为什么比普通视觉数据更难规模化？

**面试回答。** 普通视觉数据可以从网页和视频大规模收集，机器人数据需要真实设备、动作执行、传感器同步、标定、安全控制和任务标注。不同机器人本体、相机位置、动作空间和环境差异还会让数据难以直接合并。

**追问展开。** 典型数据包括 teleoperation、human demonstration、robot autonomous rollout、simulation rollout、failure replay。每类数据都有偏差：人类示范质量高但贵，自动 rollout 有失败样本但风险高，仿真便宜但有 sim2real gap。  
**易错点 / 边界。** 数据量大不等于可用；动作、观测和任务标签必须对齐，且要能覆盖失败和恢复。  
**继续读。** [VLA 数据与策略学习](../vla/data-and-policy-learning.md) / [资产、感知、抓取与轨迹流水线](../embodied-ai/asset-perception-grasp-and-trajectory-pipeline.md)

### Q：Open X-Embodiment 这类跨机器人数据集难在哪里？

**面试回答。** 跨机器人数据集把不同机器人、传感器、动作空间、任务和环境的数据混在一起，规模更大，但异构性也更强。核心难点是动作标准化、观察对齐、任务语义统一和 embodiment gap。

**追问展开。** 面试里可以说：同样是“向前移动”，不同机械臂或移动底盘的动作维度和物理效果不同。模型需要学到可迁移的高层意图，同时适配低层控制接口。  
**易错点 / 边界。** 数据来源多不等于泛化一定强；如果没有本体适配和动作接口设计，混合数据可能互相干扰。  
**继续读。** [VLA 评测与数据引擎](../vla/benchmarks-and-data-engines.md) / [Sim2Real 与具身数据引擎](../embodied-ai/sim2real-and-data-engines.md)

### Q：Sim2Real 为什么难？

**面试回答。** Sim2Real 难在仿真和真实世界之间存在 domain gap：视觉纹理、光照、相机噪声、接触物理、摩擦、延迟、标定和执行器误差都不同。仿真中成功的策略，真实世界可能因为很小的物理偏差失败。

**追问展开。** 常见缓解包括 domain randomization、真实数据微调、系统辨识、混合仿真/真实数据、失败回放和安全 fallback。  
**易错点 / 边界。** 更逼真的渲染不等于解决 Sim2Real；接触、动力学和传感器误差同样关键。  
**继续读。** [Sim2Real 与具身数据引擎](../embodied-ai/sim2real-and-data-engines.md) / [任务分类与 Benchmark](../embodied-ai/task-taxonomy-and-benchmarks.md)

### Q：仿真数据什么时候有用，什么时候会误导？

**面试回答。** 仿真数据在安全探索、长尾场景、可控变量消融、自动标注和大规模预训练中很有用。它会误导的情况是物理、视觉、接触或任务分布和真实环境差距过大，策略学到了仿真 artifact。

**追问展开。** 面试可以说仿真最好服务于“覆盖和预训练”，真实数据服务于“校准和验证”。如果任务依赖精细接触，仿真到真实的 gap 会更明显。  
**易错点 / 边界。** 仿真 benchmark 成绩高不能直接代表真实机器人成功率；必须做 real-world eval 或至少做 domain gap 分析。  
**继续读。** [Sim2Real 与具身数据引擎](../embodied-ai/sim2real-and-data-engines.md) / [IsaacSim 与 RoboTwin](../embodied-ai/isaacsim-and-robotwin.md)

## 感知、几何与状态

### Q：机器人为什么需要深度、点云或 3D 几何？

**面试回答。** 机器人不仅要知道“是什么”，还要知道“在哪里、离我多远、能不能抓、会不会碰撞”。RGB 图像提供语义和纹理，深度、点云、相机位姿和 3D 重建提供空间状态，帮助抓取、导航和避障。

**追问展开。** 对抓取任务，2D 识别杯子不够，还需要估计杯子的位置、姿态、可抓区域和障碍物。几何感知是策略和规划的状态输入。  
**易错点 / 边界。** 几何感知不等于控制成功；它只是状态层，动作后果和闭环恢复仍要策略或世界模型处理。  
**继续读。** [相机、深度与机器人视觉](../embodied-ai/cameras-depth-and-robot-vision.md) / [双目匹配与 Cost Volume](../embodied-ai/stereo-matching-and-cost-volume.md)

### Q：双目、单目深度和 RGB-D 各有什么取舍？

**面试回答。** 双目深度利用视差估计几何，适合有纹理和标定可靠的场景；单目深度从单张图预测相对或 metric depth，灵活但尺度和泛化难；RGB-D 传感器直接给深度，工程方便但受距离、材质、光照和噪声限制。

**追问展开。** 面试时可以按部署说：真实机器人常需要相机标定、深度置信度、遮挡处理和多视角融合。选择传感器不是只看模型分数，还要看环境和硬件。  
**易错点 / 边界。** 深度图看起来合理不代表可用于精细操作；边缘、透明物体、反光物体和近距离接触都很难。  
**继续读。** [相机、深度与机器人视觉](../embodied-ai/cameras-depth-and-robot-vision.md) / [双目匹配与 Cost Volume](../embodied-ai/stereo-matching-and-cost-volume.md)

### Q：几何感知和世界模型是什么关系？

**面试回答。** 几何感知提供状态层，例如 depth、camera pose、point cloud、occupancy 或 3D reconstruction；世界模型还要预测这些状态如何随动作变化。前者回答“世界现在是什么结构”，后者回答“动作之后世界会怎么变”。

**追问展开。** 机器人抓取、导航和操作需要几何状态，但只知道物体在哪里还不够，还要知道接触、可达性、稳定性和动作后果。  
**易错点 / 边界。** 几何估计不等于动力学建模。看清世界只是控制世界的前置条件。  
**继续读。** [相机、深度与机器人视觉](../embodied-ai/cameras-depth-and-robot-vision.md) / [VLA、WAM 与世界模型系统图](../embodied-ai/vla-wam-world-model-system-map.md)

### Q：视觉伺服和端到端 VLA 有什么关系？

**面试回答。** 视觉伺服用视觉误差直接驱动控制，例如让目标在图像中靠近期望位置；端到端 VLA 则从视觉和语言直接输出动作。二者可以结合：VLA 给高层目标或粗动作，视觉伺服/传统控制器负责精细闭环。

**追问展开。** 面试里可以说，传统控制提供稳定性和可解释约束，VLA 提供语义理解和任务泛化。工程系统常是混合架构，而不是纯端到端。  
**易错点 / 边界。** 不要把端到端当成排斥控制理论；真实机器人通常需要低层控制器、安全约束和状态估计。  
**继续读。** [规划、控制与安全](../embodied-ai/planning-control-and-safety.md) / [VLA 动作表示与控制接口](../vla/action-representation-and-interfaces.md)

## 评测、闭环恢复与部署

### Q：机器人闭环评测应该看什么？

**面试回答。** 至少看任务成功率、失败类型、恢复能力、动作平滑性、安全事件、跨场景泛化和每次成功成本。最好保留视频、状态、动作、传感器和模型日志，能复盘每一次失败。

**追问展开。** 离线 loss 只能说明模型拟合数据，open-loop 预测只能说明短期预测质量；闭环评测才接近“策略执行是否成功”。  
**易错点 / 边界。** 单个 demo 视频不能证明平均成功率；真实机器人评测要有任务分布、试次数、失败统计和安全说明。  
**继续读。** [VLA 评测与数据引擎](../vla/benchmarks-and-data-engines.md) / [VLA 闭环恢复与失败分析](../vla/closed-loop-recovery-and-failure-analysis.md)

### Q：为什么失败恢复比单次成功更重要？

**面试回答。** 真实环境中动作误差、遮挡、滑动、碰撞和用户干预都很常见。一个只能在理想状态下成功的策略不够可靠；具身系统需要发现失败、重新感知、调整计划或安全停止。

**追问展开。** 面试可以举例：抓杯子没抓稳时，系统要检测到状态变化，重新规划抓取，而不是继续执行原 chunk。闭环恢复体现的是系统鲁棒性。  
**易错点 / 边界。** 成功率统计如果不报告失败原因和恢复策略，很难指导改进。  
**继续读。** [VLA 闭环恢复与失败分析](../vla/closed-loop-recovery-and-failure-analysis.md) / [家庭机器人流程与失败模式](../embodied-ai/household-robotics-workflows-and-failure-patterns.md)

### Q：VLA 评测为什么不能只看离线 action accuracy？

**面试回答。** 离线 action accuracy 衡量模型是否模仿数据里的动作，但同一个状态可能有多条可行动作，离线标签也不一定唯一。真实机器人关心执行是否成功、是否安全、是否能恢复，而不是是否复制某条示范轨迹。

**追问展开。** 更好的评测要包含 closed-loop success、trajectory quality、intervention count、failure replay 和 cross-scene generalization。  
**易错点 / 边界。** 行为克隆 loss 低不代表策略好；分布偏移会让小错误在闭环里累积。  
**继续读。** [VLA 评测与数据引擎](../vla/benchmarks-and-data-engines.md) / [VLA 数据与策略学习](../vla/data-and-policy-learning.md)

### Q：真实机器人部署为什么要安全门禁？

**面试回答。** 机器人动作有物理后果，错误可能损坏设备、伤人或破坏环境。部署时要有速度/力限制、碰撞检测、workspace 约束、急停、权限控制、人工确认和 fallback policy。

**追问展开。** VLA 输出动作前后都可以加安全层：动作前做可达性和碰撞检查，动作中做力/视觉监控，动作后检查任务状态。  
**易错点 / 边界。** 不要把模型对齐等同于机器人安全。物理安全必须由系统和控制层共同保障。  
**继续读。** [VLA 部署与安全](../vla/deployment-and-safety.md) / [规划、控制与安全](../embodied-ai/planning-control-and-safety.md)

### Q：如果面试让你设计一个具身智能系统，你会怎么回答？

**面试回答。** 我会先明确任务、场景、机器人本体、传感器和动作空间；再设计感知状态、策略模型、低层控制、安全约束和数据闭环。最小系统可以是 `perception -> state/geometry -> policy/VLA -> safety filter -> controller -> logging/eval`。

**追问展开。** 数据侧要有示范、失败、仿真和真实回放；评测侧要有 closed-loop success、失败分类、延迟、成本和安全事件；部署侧要有 fallback 和监控。  
**易错点 / 边界。** 不要只回答“用一个 VLA 大模型”。具身系统是模型、控制、数据、评测、安全和运维的组合。  
**继续读。** [具身智能从零路线](../embodied-ai/beginner-learning-path.md) / [部署模式与安全案例](../embodied-ai/deployment-patterns-and-safety-cases.md)

### Q：多模态和具身方向的强 claim 应该怎么写才稳？

**面试回答。** 先标清证据类型：论文主表、消融、系统吞吐、官方 demo、真实闭环还是本站推断。再说明它能支持什么，不能证明什么。比如“官方 demo 展示能力形态”不能写成“平均成功率已验证”。

**追问展开。** 面试或写作中可以用一句模板：这条证据在某数据、某任务、某硬件或某 demo 设置下支持某个结论，但不能外推到跨平台、真实闭环或生产安全。  
**易错点 / 边界。** 最大误区是把 demo、open-loop 指标、system throughput、closed-loop success 混成同一种证据。  
**继续读。** [全站证据与复现状态标准](../references/evidence-and-reproducibility-standard.md) / [Claim Ledger](../references/claim-ledger.md)
