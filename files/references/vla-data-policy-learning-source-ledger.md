# VLA 数据与策略学习来源台账

这份台账服务于 [VLA 数据与策略学习](../vla/data-and-policy-learning.md)。正文不把 VLA 学习写成“视觉模型 + 语言模型 + 动作头”，而是围绕一条闭环链讲：轨迹数据如何定义，行为克隆为什么只是入口，分布偏移为什么会滚雪球，action chunk 和扩散策略分别解决什么，跨机器人数据为什么首先是接口对齐问题，最后如何用真实闭环证据验收。

## 使用方式

第一类来源是行为克隆、离线 RL 和分布偏移。DAgger、offline RL tutorial、机器人模仿学习资料用于解释：训练时看到专家状态，部署时看到模型自己造成的状态；如果数据里没有偏差状态和恢复动作，模型不能凭空学会纠错。正文写这部分时，不要把“更多成功示范”写成万能解，而要强调失败样本、恢复轨迹、人工纠正和在线数据聚合。

第二类来源是动作表示和 action chunk。RT-1、RT-2、ALOHA/ACT、FAST 和动作接口来源台账用于解释动作 token、连续动作、chunk horizon、temporal ensembling 和低层 controller。正文要区分 action chunk 与开环计划：chunk 是短时控制接口，仍需要重观察、重规划和安全层。

第三类来源是多峰连续动作建模。Diffusion Policy、ACT、Diffusion Policy 的项目页和论文图用于解释为什么 MSE 会把多条合理轨迹平均成危险中间动作，以及扩散策略怎样把动作序列当作条件生成对象。正文不要把扩散策略写成“比 BC 更高级”；它的价值只在多峰动作、接触、软体和连续控制任务中更明显，推理延迟和控制器配合仍是边界。

第四类来源是跨机器人和通用策略。RT-X/Open X-Embodiment、Octo、OpenVLA、π0/π0.5、DROID 和 BridgeData 用于支撑“规模化数据不是把轨迹倒进模型”。写这类工作时，重点检查观测、动作、任务、结果和部署语义是否对齐，而不是只看数据量、模型参数量或 demo 数量。

第五类来源是中文技术长文和公众号/行业报告。Datawhale Every-Embodied、智源、量子位、机器之心等材料适合学习中文讲法：先给一个机器人任务失败案例，再沿数据、动作接口、训练目标和闭环评测拆开。强 claim、成功率、数据规模和安全结论仍必须回到论文、官方项目页、代码仓库或可复查报告。

## 写作边界

不要把 VLA 数据问题写成“样本不够”。很多失败来自样本异构：相机外参不同、gripper 语义不同、动作频率不同、任务语言粒度不同、成功定义不同、失败状态没有记录。正文应把这些写成可检查的账本。

不要把 offline action error 当成策略成功。动作误差低只说明模型在专家分布附近拟合得好，不说明它能在自己造成的偏差状态里恢复。VLA 页面必须把 offline、replay、simulator、real robot 四层证据分开。

不要把 action tokenization 当成 VLA 的充分条件。离散 token 让动作接入 Transformer/LLM 训练栈，但真实机器人仍要处理坐标系、限幅、控制频率、精细控制和安全停止。

不要把 diffusion policy 写成万能策略。扩散动作建模适合多峰连续动作，但推理成本、噪声步数、闭环频率、低层 controller 和 safety filter 会决定是否能部署。

不要把跨机器人数据混合写成“越多越好”。如果同一个 action 维度在不同 embodiment 中含义不同，模型学到的是接口噪声。RT-X、Octo、OpenVLA、π0.5 等工作都应按数据口径、动作口径、评测桶和部署约束来读。

## 图片使用

本页没有使用 image2 生成图片；图片都来自已有论文、项目页或站内已归档资产。图注必须说明原图想证明什么，以及本站正文借它解释什么机制，不能用图去暗示论文没有证明的泛化能力。

| 图片 | 文件 | 原始来源 | 正文用途 | 使用边界 |
| --- | --- | --- | --- | --- |
| Open X-Embodiment 数据总览 | `../assets/images/paper-figures/embodied-ai/open-x-embodiment-figure-1-dataset.png` | [Open X-Embodiment / RT-X](https://robotics-transformer-x.github.io/) | 解释跨机构、跨机器人、跨任务数据混合为什么首先是观测、动作和任务标签对齐问题。 | 不能仅凭这张图推断通用策略一定能跨机器人部署。 |
| ACT 方法图 | `../assets/images/paper-figures/vla/aloha-act-algorithm-2x.png` | [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705) | 解释 action chunk、CVAE/Transformer 输入和 temporal ensembling 如何服务短时连续控制。 | 不把 ACT 图写成所有精细操作的唯一方案；部署仍依赖硬件、示范质量和闭环频率。 |
| Diffusion Policy 输入输出图 | `../assets/images/paper-figures/vla/diffusion-policy-input-output-padded.jpg` | [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) | 解释把未来动作序列当作条件生成对象，适合多峰连续动作。 | 不把扩散策略写成普遍优于 BC/ACT；推理延迟和控制器配合仍是关键边界。 |

## 来源

| 来源 | 类型 | 本站使用方式 | 关联问题 |
| --- | --- | --- | --- |
| [DAgger](https://arxiv.org/abs/1011.0686) | 论文 | 支撑 covariate shift、数据聚合和专家纠正为什么能缓解滚雪球误差。 | 分布偏移 |
| [Offline Reinforcement Learning Tutorial](https://arxiv.org/abs/2005.01643) | 综述/教程 | 支撑离线 RL 中分布外动作和 Q-value 虚高的风险。 | 离线 RL |
| [RT-1](https://robotics-transformer1.github.io/) | 项目页/论文 | 支撑多任务机器人示范、动作 token 和真实机器人 Transformer policy。 | 行为克隆规模化 |
| [RT-2 blog](https://deepmind.google/blog/rt-2-new-model-translates-vision-and-language-into-action/) | 官方博客 | 支撑 web-scale VLM 知识如何通过动作 token 接到机器人。 | VLA 语义迁移 |
| [RT-2 paper](https://arxiv.org/abs/2307.15818) | 论文 | 核对 co-fine-tuning、action token、seen/unseen/emergent evaluation 和失败边界。 | VLA 证据 |
| [Open X-Embodiment / RT-X](https://robotics-transformer-x.github.io/) | 项目页/论文/图源 | 支撑跨机器人数据、动作标准化、embodiment 差异和 RT-X 策略评测。 | 数据异构 |
| [ALOHA / ACT](https://tonyzhaozh.github.io/aloha/) | 项目页/论文/图源 | 支撑 action chunk、CVAE、temporal ensembling 和低成本双臂精细操作。 | 动作分块 |
| [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) | 项目页/论文/图源 | 支撑动作扩散、多峰连续轨迹和 receding horizon 控制。 | 多峰动作 |
| [Octo](https://octo-models.github.io/) | 项目页/论文 | 支撑通用机器人策略、多数据集、多任务条件、多 action spaces 和开源模型。 | 通用 policy |
| [OpenVLA](https://openvla.github.io/) | 项目页/论文 | 支撑开源 VLA、Prismatic VLM、robot action fine-tuning 和跨任务评测。 | 开源 VLA |
| [π0](https://www.physicalintelligence.company/blog/pi0) | 官方博客 | 支撑 vision-language-action flow matching policy、真实机器人动作生成和系统边界。 | VLA policy |
| [π0.5](https://arxiv.org/abs/2504.16054) | 论文/技术报告 | 支撑 open-world VLA、异构 co-training、语义子任务和真实家庭泛化证据。 | open-world VLA |
| [DROID](https://droid-dataset.github.io/) | 项目页/论文 | 支撑真实机器人数据采集、跨场景、多任务和长轨迹数据引擎。 | 数据覆盖 |
| [BridgeData](https://rail-berkeley.github.io/bridgedata/) | 项目页/论文 | 支撑多环境、多任务机器人数据与离线策略学习。 | 数据集 |
| [FAST action tokenizer](https://arxiv.org/abs/2501.09747) | 论文 | 支撑高频连续动作压缩成更短离散 token 序列。 | action tokenizer |
| [Datawhale Every-Embodied](https://github.com/datawhalechina/every-embodied) | 中文开源教程/公众号生态 | 学习把硬件、仿真、数据采集、VLA 复现和部署路线组织成新手能跟上的学习路径。 | 中文讲法 |
| [外部精读来源台账](external-reading-ledger.md) | 本站台账 | 汇总机器之心、量子位、智源、PaperWeekly、Datawhale 等中文技术材料的讲法边界。 | 中文讲法 |

## 后续写作提醒

写 VLA 数据与策略时，优先从失败模式反推方法：多峰轨迹对应扩散或 mixture，短时抖动对应 chunk 和 temporal ensembling，分布偏移对应数据聚合和失败回放，跨机器人泛化对应动作口径和观测口径，语义泛化对应 VLM/web 数据与机器人数据的 co-training。不要让“模型更大、数据更多、demo 更漂亮”替代这个因果链。

图源优先选择论文、项目页和官方博客图。每张图下方必须说明：原图在证明什么，本站正文要读出什么机制，不能证明什么边界。
