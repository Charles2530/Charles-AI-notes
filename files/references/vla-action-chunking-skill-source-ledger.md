# VLA 动作分块、层级策略与潜在技能来源台账

这份台账服务于 [动作分块、层级策略与潜在技能](../vla/action-chunking-hierarchical-policies-and-latent-skills.md)。正文不把 action chunk、hierarchical policy 和 latent skill 写成三组名词，而是围绕时间尺度讲：逐步动作为什么抖，chunk 怎样提供短时连贯性，option/skill 为什么必须有终止条件，潜在技能怎样压缩重复行为，最后怎样用闭环和失败恢复验收。

## 使用方式

第一类来源是 action chunk 和短时序动作生成。ACT、Diffusion Policy、Octo 和 VLA 动作接口台账用于解释 chunk horizon、receding horizon、temporal ensembling、多峰动作和 diffusion decoder。正文写这部分时，必须强调 chunk 是短时控制接口，不是让机器人开环执行完整任务。

第二类来源是层级强化学习和技能选择。Options 框架提供启动集合、内部策略和终止函数这套基本语言；SayCan 用语言相关性和 affordance/value function 组合说明高层技能不能只按语言合理性选择。正文把它们作为结构化思想来源，不把它们写成现代 VLA 的唯一实现。

第三类来源是潜在技能和动作 token。Play-LMP、Behavior Transformer / VQ-BeT、FAST、Octo、π0.5 用于支撑“低层动作需要被压缩成可复用、可选择、可部署的中间表示”。正文应提醒：latent code 不自动可解释，action token 不自动可执行，必须看重构、复用、语言对齐、终止和失败恢复。

第四类来源是中文技术长文和公众号生态。Datawhale Every-Embodied、机器之心、量子位、智源、PaperWeekly 等材料用于学习中文讲法：先用真实机器人任务把问题抛出来，再按时间尺度、接口、失败模式和评测证据拆解。事实、图源、成功率和模型细节仍回论文、项目页、官方博客或代码仓库。

## 写作边界

不要把 action chunk 写成开环计划。部署时通常执行 chunk 的前几步，再重观察、重规划，并用 safety filter 或 controller 约束动作。

不要把技能写成标签名。一个可执行技能至少要说明何时能启动、内部怎样控制、何时终止、失败时怎样退出。

不要把 latent skill 写成天然可解释。若没有语言、子目标或终止条件约束，latent code 可能只是压缩后的动作噪声，不能直接当成可组合技能。

不要把层级结构当成一定优于端到端。若任务短、动作单峰、控制器稳定，普通 BC 或短 chunk 可能足够；层级越多，接口错配和信用分配问题也越多。

## 图片使用

本页没有使用 image2 生成图片。图片均来自已有论文、项目页或站内归档资产；图注必须说明原图机制和本站读法，不能把图外推成通用部署证据。

| 图片 | 文件 | 原始来源 | 正文用途 | 使用边界 |
| --- | --- | --- | --- | --- |
| ACT 方法图 | `../assets/images/paper-figures/vla/aloha-act-algorithm-2x.png` | [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705) | 说明 action chunk、CVAE/Transformer 和 temporal ensembling 是短时控制接口。 | 不证明所有精细操作都应使用 ACT；硬件、示范、控制频率和闭环恢复仍决定部署效果。 |
| SayCan 方法图 | `../assets/images/paper-figures/embodied-ai/saycan-figure-3-llm-affordance.png` | [Do As I Can, Not As I Say / SayCan](https://say-can.github.io/) | 说明高层技能选择需要同时考虑语言相关性和当前场景可执行性。 | 不把 SayCan 写成现代 VLA 的完整方案；它主要服务层级技能选择的直觉。 |

## 来源

| 来源 | 类型 | 本站使用方式 | 关联问题 |
| --- | --- | --- | --- |
| [ALOHA / ACT](https://arxiv.org/abs/2304.13705) | 论文 / 图源 | 支撑 action chunk、CVAE、temporal ensembling 和低成本双臂精细操作。 | 动作分块 |
| [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) | 项目页 / 论文 | 支撑动作序列作为条件生成对象、多峰连续控制和 receding horizon。 | 多峰动作 |
| [Options 框架](http://incompleteideas.net/papers/SPS-aij.pdf) | 论文 | 提供 option 的启动集合、内部策略和终止函数定义。 | 层级策略 |
| [SayCan](https://say-can.github.io/) | 项目页 / 论文 / 图源 | 支撑语言技能选择必须结合 affordance 和 value function 的讲法。 | 技能选择 |
| [Play-LMP](https://arxiv.org/abs/1903.01973) | 论文 | 支撑从 play data 中学习 latent plan / latent skill 的思路。 | 潜在技能 |
| [VQ-BeT](https://arxiv.org/abs/2403.03181) | 论文 | 支撑用 VQ latent action 和 behavior transformer 表达多峰行为。 | 行为 token |
| [Octo](https://octo-models.github.io/) | 项目页 / 论文 | 支撑通用机器人策略、多数据集、多任务条件、action chunk 和 diffusion decoder。 | 通用 policy |
| [FAST](https://arxiv.org/abs/2501.09747) | 论文 | 支撑把高频连续动作压缩成更短 action token 序列。 | action tokenizer |
| [π0.5](https://arxiv.org/abs/2504.16054) | 论文 / 技术报告 | 支撑 high-level subtask、异构 co-training 和低层 action expert 的现代 VLA 读法。 | open-world VLA |
| [VLA 动作接口来源台账](vla-action-interface-source-ledger.md) | 本站台账 | 复用动作坐标系、action token、chunk 和低层控制器边界。 | 动作接口 |
| [VLA 数据与策略学习来源台账](vla-data-policy-learning-source-ledger.md) | 本站台账 | 复用行为克隆、分布偏移、失败回放和闭环评测边界。 | policy learning |
| [Datawhale Every-Embodied](https://github.com/datawhalechina/every-embodied) | 中文开源教程 / 公众号生态 | 学习具身智能中文教程如何把硬件、仿真、数据、模型和部署路线串成路径。 | 中文讲法 |
| [外部精读来源台账](external-reading-ledger.md) | 本站台账 | 汇总机器之心、量子位、智源、PaperWeekly、Datawhale 等中文技术材料的讲法边界。 | 讲法参考 |

## 后续维护

新增技能学习或 action tokenizer 工作时，先判断它改变了哪一层：短时 action chunk、技能选择、潜在 code、语言子任务、世界模型 rollout，还是低层 controller。若来源只展示 demo 或平均成功率，缺少技能边界、终止条件、失败恢复和真实机器人闭环证据，正文只能把它作为能力形态参考，不能写成稳定部署结论。
