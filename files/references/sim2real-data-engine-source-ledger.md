# Sim2Real 与具身数据引擎来源台账

这份台账服务于 [Sim2Real 与具身数据引擎](../embodied-ai/sim2real-and-data-engines.md)。正文不把 Sim2Real 写成“仿真画面更像真实照片”，而是围绕一条数据闭环讲：仿真如何覆盖变量，system identification 如何校准中心，真实部署日志如何暴露长尾，失败样本如何筛选/标注/门禁，hard set 如何验证下一轮模型是否真的补上短板。

## 使用方式

第一类来源是 domain randomization 和 sim2real 经典工作。Domain Randomization 论文、OpenAI dexterous manipulation、RoboTwin / Isaac Sim / Isaac Lab 文档用于解释：仿真不只是渲染，而是资产、物理、传感器、动作、任务、轨迹、success checker 和日志导出的系统。正文使用这类来源时，要明确随机化的是视觉、动力学、相机、控制延迟还是任务扰动。

第二类来源是 system identification 和物理校准。MuJoCo / Isaac / robotics sim 文档、机器人系统论文和真实轨迹回放资料用于支撑“把仿真中心调准，再围绕中心随机化”的讲法。正文不要把 domain randomization 与 system ID 写成二选一；它们通常分别解决中心偏差和长尾不确定性。

第三类来源是真实机器人数据引擎。Open X-Embodiment、DROID、BridgeData、RoboCat、π0.5、RT-X 和站内 VLA 数据台账用于解释：真实机器人数据的样本单位是轨迹和事件，而不是单张图片；metadata 必须包含 embodiment、相机、动作接口、时间戳、成功信号、失败标签和人工接管。

第四类来源是 hard set 与评测闭环。VLA benchmark、world model evaluation、Sim2Real benchmark、RoboTwin、SimplerEnv 等资料用于支撑“平均成功率不够，必须按失败桶验收”。正文写评测时，要把透明/反光、接触动力学、相机扰动、长任务阶段和人类干扰拆开。

第五类来源是中文长文和行业资料。智源、量子位、机器之心、Datawhale、NVIDIA/Isaac 中文教程、RoboTwin 中文文档适合学习如何用中文把产业痛点、仿真系统和数据闭环讲清楚。强事实、实验结果、数据规模和安全结论仍以论文、官方文档、项目页和代码仓库为准。

## 写作边界

不要把 photorealistic rendering 等同于 Sim2Real。视觉真实只是观测分布的一部分；接触、摩擦、质量、控制延迟、相机外参、任务扰动和人类干扰同样会造成真实失败。

不要把仿真数据写成免费真数据。仿真可以低成本生成轨迹和标签，但如果 success checker、碰撞网格、资产尺度或动力学参数错了，生成越多越会放大偏差。

不要把线上日志直接倒进训练集。日志必须经过选择、标注、门禁和 hard set 分流；否则数据引擎会退化成脏数据回灌。

不要只报告平均成功率。Sim2Real 的价值在长尾，应该按 hard bucket 报告透明/反光、接触、遮挡、外参扰动、长任务阶段和人工干扰下的变化。

不要把跨机器人数据混合写成“更多 embodiment 自动泛化”。不同机器人动作语义、控制频率、gripper 定义、相机视角和安全边界都需要 metadata 和接口对齐。

## 来源

| 来源 | 类型 | 本站使用方式 | 关联问题 |
| --- | --- | --- | --- |
| [Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World](https://arxiv.org/abs/1703.06907) | 论文 | 支撑随机化仿真参数以覆盖真实差异的经典思路。 | domain randomization |
| [Learning Dexterous In-Hand Manipulation](https://arxiv.org/abs/1808.00177) | 论文 | 支撑大规模随机化、自动域随机化和真实灵巧手迁移案例。 | sim2real 工程 |
| [RoboTwin 2.0 docs](https://robotwin-platform.github.io/doc/) | 官方文档/图源 | 支撑任务、资产、expert trajectory、domain randomization 和 success checker 组织方式。 | 仿真数据工厂 |
| [Isaac Sim documentation](https://docs.isaacsim.omniverse.nvidia.com/) | 官方文档 | 支撑传感器、物理、USD 资产、headless 仿真和机器人任务生成。 | 仿真平台 |
| [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) | 官方文档/代码 | 支撑可复现机器人 RL 环境、domain randomization 和 manager-based workflow。 | 仿真训练 |
| [Open X-Embodiment / RT-X](https://robotics-transformer-x.github.io/) | 项目页/论文/图源 | 支撑跨机器人数据异构、动作口径和多任务数据混合。 | 跨 embodiment |
| [DROID](https://droid-dataset.github.io/) | 项目页/论文 | 支撑 in-the-wild 真实机器人数据、遥操作采集、多场景和长轨迹覆盖。 | 真实数据引擎 |
| [BridgeData](https://rail-berkeley.github.io/bridgedata/) | 项目页/论文 | 支撑多环境机器人数据、离线策略学习和真实数据覆盖。 | robot dataset |
| [RoboCat](https://deepmind.google/discover/blog/robocat-a-self-improving-robotic-agent/) | 官方博客/论文入口 | 支撑机器人 self-improvement loop：收集新任务、再训练、扩展能力。 | 自改进 |
| [π0.5](https://arxiv.org/abs/2504.16054) | 论文/图源 | 支撑 web data、cross-embodiment data、multi-environment data 的数据消融读法。 | 数据 recipe |
| [SimplerEnv](https://simpler-env.github.io/) | 项目页/论文 | 支撑真实-仿真评测连接和策略转移评估。 | sim2real benchmark |
| [Rerun](https://rerun.io/) | 工具/官方文档 | 支撑机器人日志可视化、失败回放和多传感器数据复盘。 | 数据可观测性 |
| [智源社区具身专题](https://hub.baai.ac.cn/view/41216) | 中文会议观点整理 | 学习中文长文如何把真实数据、世界模型和产业约束放在一起讲。 | 中文讲法 |
| [Datawhale 大模型算法教程](https://datawhalechina.github.io/llm-algo-leetcode/) | 中文教程 | 学习公式、变量、机制和例子逐层解释的组织方式。 | 中文讲法 |

## 后续写作提醒

写 Sim2Real 时，优先用失败来组织段落：看错、抓空、打滑、插不进去、任务阶段错、人工干扰、上线回流污染。每个失败都要能映射到一个处理动作：随机化、system ID、补真实数据、改 success checker、补 hard set、改控制器或安全层。不要只写“扩大数据集”。

图片优先用论文、官方文档和项目页图。图注必须说明：原图证明什么，本站要读出什么机制，不能证明什么边界。
