# VLA 评测与数据引擎来源台账

这份台账服务于 [VLA 评测与数据引擎](../vla/benchmarks-and-data-engines.md)。正文不把 benchmark 写成分数表，而是围绕一条闭环讲：episode 里要记录什么，为什么总成功率不够，失败如何进入 hard set，数据回流怎样避免污染，仿真、实验室和真实部署证据如何分层。

## 使用方式

第一类来源是跨机器人数据与通用策略。Open X-Embodiment / RT-X、DROID、BridgeData、Octo、OpenVLA 和 π0.5 用于说明机器人数据的样本单位是 episode / trajectory，metadata 必须包含 embodiment、相机、动作接口、时间戳、成功信号、失败标签和部署版本。

第二类来源是 benchmark 与 real-to-sim。RoboTwin、RoboTwin 2.0、SimplerEnv 和 Sim2Real 台账用于支撑“仿真、实验室、真实部署要分层报告”。正文使用这类来源时，不能把仿真相关性外推成真实部署等价；它只能降低迭代成本和暴露一批系统性失败。

第三类来源是数据闭环和失败回放。VLA 数据策略台账、世界模型评测台账、Sim2Real 台账和站内证据标准用于解释 hard set、replay set、安全集、训练回流和线上监控应该分开。正文要强调训练污染和平均成功率掩盖风险。

第四类来源是中文技术长文和公众号生态。Datawhale Every-Embodied、机器之心、量子位、智源、PaperWeekly 等材料用于学习中文讲法：从一个现场失败案例切入，再拆 benchmark、数据回流、仿真验证和真实部署。强 claim、成功率、数据规模和安全结论仍回到论文、项目页、官方文档或代码仓库。

## 写作边界

不要把总成功率当成成熟度。VLA 评测要同时报告首次成功率、最终成功率、恢复率、人工接管率、阶段完成率和失败桶。

不要把线上日志直接倒回训练集。日志必须经过筛选、去重、标注、分流和门禁；hard set 不能被训练污染。

不要把仿真 benchmark 当成真实部署证据。仿真适合做扰动扫描、消融和可复现对比；真实接触、传感器噪声、人类干扰和硬件漂移仍要用实机验证。

不要把跨机器人数据混合写成“越多越好”。若动作坐标系、控制频率、gripper 语义和成功定义不一致，数据量越大也可能越像接口噪声。

## 图片使用

本页没有使用 image2 生成图片。正文只使用已有 Open X-Embodiment 数据总览图；旧 `openx-data-vis.jpg` 曾被图片脚本提示裁切过紧，因此改用站内已验证更清晰的归档版本。

| 图片 | 文件 | 原始来源 | 正文用途 | 使用边界 |
| --- | --- | --- | --- | --- |
| Open X-Embodiment 数据总览 | `../assets/images/paper-figures/embodied-ai/open-x-embodiment-figure-1-dataset.png` | [Open X-Embodiment / RT-X](https://robotics-transformer-x.github.io/) | 说明跨机构、跨机器人、跨任务数据汇聚时必须先核对观测、动作、任务、结果和部署语义。 | 不用它证明通用策略已经解决跨机器人部署；正文只借它解释数据与评测口径。 |

## 来源

| 来源 | 类型 | 本站使用方式 | 关联问题 |
| --- | --- | --- | --- |
| [Open X-Embodiment / RT-X](https://robotics-transformer-x.github.io/) | 项目页 / 论文 / 图源 | 支撑跨机器人数据、动作标准化、embodiment 差异和 RT-X 评测。 | 跨 embodiment |
| [DROID](https://droid-dataset.github.io/) | 项目页 / 论文 | 支撑 in-the-wild 真实机器人数据、多场景、多任务和长轨迹采集。 | 真实数据 |
| [BridgeData V2](https://arxiv.org/abs/2308.12952) | 论文 / 数据集 | 支撑多环境机器人数据、离线策略学习和数据规模化。 | 数据集 |
| [Octo](https://octo-models.github.io/) | 项目页 / 论文 | 支撑通用机器人策略评测、多数据集、多任务条件和 action spaces。 | generalist policy |
| [OpenVLA](https://openvla.github.io/) | 项目页 / 论文 | 支撑开源 VLA 微调、跨任务评测和部署边界。 | 开源 VLA |
| [π0.5](https://arxiv.org/abs/2504.16054) | 论文 / 技术报告 | 支撑 open-world VLA、真实家庭评测、环境扩展、数据和语言消融。 | 真实部署 |
| [RoboTwin](https://arxiv.org/abs/2409.02920) | 论文 / benchmark | 支撑双臂 benchmark、real-to-sim 和 synthetic data 的早期路线。 | 仿真 benchmark |
| [RoboTwin 2.0](https://arxiv.org/abs/2506.18088) | 论文 / benchmark | 支撑强 domain randomization、双臂任务和多 embodiment benchmark 的后续路线。 | 仿真数据工厂 |
| [SimplerEnv](https://simpler-env.github.io/) | 项目页 / 论文 | 支撑 real-to-sim policy evaluation、仿真与真实表现相关性和评测边界。 | sim2real benchmark |
| [VLA 数据与策略学习来源台账](vla-data-policy-learning-source-ledger.md) | 本站台账 | 复用行为克隆、分布偏移、失败回放和闭环评测边界。 | policy learning |
| [Sim2Real 与数据引擎来源台账](sim2real-data-engine-source-ledger.md) | 本站台账 | 复用 domain randomization、真实日志回流、hard set 和门禁边界。 | 数据闭环 |
| [世界模型评测来源台账](world-model-evaluation-source-ledger.md) | 本站台账 | 复用 action sensitivity、closed-loop success、failure replay 和 cost per success 证据标准。 | 评测证据 |
| [Datawhale Every-Embodied](https://github.com/datawhalechina/every-embodied) | 中文开源教程 / 公众号生态 | 学习具身智能中文教程如何把数据、仿真、复现和部署路线组织成学习路径。 | 中文讲法 |
| [外部精读来源台账](external-reading-ledger.md) | 本站台账 | 汇总机器之心、量子位、智源、PaperWeekly、Datawhale 等中文技术材料的讲法边界。 | 讲法参考 |

## 后续维护

新增 VLA benchmark、real-to-sim benchmark 或机器人数据集时，先标清它证明的是哪一层证据：离线动作拟合、失败回放、仿真闭环、实验室真机、目标环境部署，还是安全门禁。若来源只给平均成功率，缺少 episode metadata、失败桶、人工接管、阶段完成率和数据版本，正文不能把它写成稳定部署结论。
