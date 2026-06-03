# VLA 动作表示与控制接口来源台账

这份台账服务于 [VLA 动作表示与控制接口](../vla/action-representation-and-interfaces.md)。正文不把 VLA 写成“图像、语言、动作都 token 化”，而是围绕一条执行链讲：policy 输出什么，动作在哪个坐标系，如何离散或连续建模，是否输出 action chunk，controller 如何执行，失败如何回到闭环。

## 使用方式

第一类来源是 action tokenization。RT-1、RT-2 和 Open X-Embodiment / RT-X 支撑正文关于离散动作 token、7 维 gripper-frame 动作和跨机器人动作口径标准化的讨论。正文使用 RT-2 Figure 1 说明 web-scale VLM 知识只有通过动作接口才能进入真实机器人。

第二类来源是 action chunk。ALOHA / ACT 支撑一次输出未来动作序列、temporal ensembling 和低成本双臂操作的讲法。正文使用 ACT 方法图说明 action chunk 是短时控制接口，不是完整任务计划。

第三类来源是连续动作分布。Diffusion Policy 支撑“动作也可以是条件生成分布”的解释，尤其是多峰动作、连续 6DoF 控制和 receding horizon。Octo 用于说明通用机器人策略需要同时支持多种传感器、任务条件和 action spaces。

第四类来源是现代 action tokenizer 和空间动作接口。FAST 用于讨论高频连续动作如何被压成更短离散 token；SpatialVLA、π0.5、OpenVLA 和站内深读材料用于提醒：动作 token 不是终点，还要处理 3D 空间、动作专家、部署安全和低层控制。

中文和博客材料主要学习讲法：先把 policy/controller 中间的接口讲清，再讲 RT-1/RT-2、ACT、Diffusion Policy 等不同路线。事实判断以论文、官方博客和项目页为准。用户要求广泛参考高质量博客和公众号时，本页把智源、量子位、机器之心、Datawhale 等材料用于中文读者入口和结构参考，不把它们当作动作接口的强证据。

## 写作边界

正文不把动作 tokenization 当成 VLA 的充分条件。Tokenization 统一训练接口，但真实机器人仍要解决坐标系、频率、限幅、低层控制和安全。

正文不把 action chunk 当成开环计划。Chunk 解决短时连贯性和推理延迟，但必须配合重观察、重规划、temporal ensembling 和 safety filter。

正文不把 diffusion policy 当成“扩散模型更高级”。它的价值在于表达多峰连续动作分布和 receding horizon 控制；是否适合取决于任务、延迟和控制接口。

正文不把跨机器人数据混合说成直接堆轨迹。不同机器人必须对齐观测、动作坐标、单位、时间戳和 gripper semantics，否则泛化会被接口噪声吞掉。

## 图片使用

本页遵守用户约束：不使用 image2，不生成新图。只复用仓库已有论文图，并修正旧版正文中 RT-2 图片路径不匹配的问题：原先使用的是结果柱状图，现改为 RT-2 Figure 1 overview。

| 图片 | 本地路径 | 来源 | 用途 |
| --- | --- | --- | --- |
| RT-2 overview | `../assets/images/paper-deep-dives/embodied-ai/rt2/figure-1-overview.png` | [RT-2](https://arxiv.org/abs/2307.15818) Figure 1 | 说明 VQA / VLM 训练、robot action tokens 和闭环机器人控制如何连成 VLA。 |
| ACT algorithm | `../assets/images/paper-figures/vla/aloha-act-algorithm-2x.png` | [ALOHA / ACT](https://arxiv.org/abs/2304.13705) Figure 2 | 说明 action chunk 与 temporal ensembling 是短时控制接口。 |
| Diffusion Policy input-output | `../assets/images/paper-figures/vla/diffusion-policy-input-output-padded.jpg` | [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) | 说明动作序列可以作为条件生成分布，并接到 receding horizon control。 |

## 来源

| 来源 | 类型 | 本站使用方式 | 关联主题 |
| --- | --- | --- | --- |
| [RT-1](https://robotics-transformer1.github.io/) | 项目页 / 论文 | 支撑真实机器人 Transformer policy、动作 token 和大规模机器人数据。 | action token |
| [RT-2 blog](https://deepmind.google/blog/rt-2-new-model-translates-vision-and-language-into-action/) | 官方博客 | 支撑 VLM 知识迁移到机器人动作的直观讲法。 | VLA |
| [RT-2 paper](https://arxiv.org/abs/2307.15818) | 论文 / 图源 | 使用 Figure 1 说明 web-scale VLM 与 robot action token 输出的连接。 | action token |
| [Open X-Embodiment / RT-X](https://robotics-transformer-x.github.io/) | 项目页 / 论文 | 支撑跨机器人数据、7 维 gripper-frame 动作和动作口径标准化。 | 跨 embodiment |
| [ALOHA / ACT](https://arxiv.org/abs/2304.13705) | 论文 / 图源 | 使用 ACT 方法图说明 action chunking、action sequence 和 temporal ensembling。 | action chunk |
| [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) | 项目页 / 论文 / 图源 | 使用 input-output 图说明连续动作序列、条件扩散策略和 receding horizon。 | 连续动作分布 |
| [Octo](https://octo-models.github.io/) | 项目页 / 论文 | 支撑通用机器人策略、不同 observation/action spaces 和 diffusion decoding。 | generalist policy |
| [FAST](https://arxiv.org/abs/2501.09747) | 论文 | 支撑高频连续动作压缩成离散 token 的现代 VLA action tokenizer。 | action tokenizer |
| [SpatialVLA](https://arxiv.org/abs/2501.15830) | 论文 | 补充 3D 空间表征、adaptive action grids 和动作坐标接口。 | 空间动作 |
| [OpenVLA](https://openvla.github.io/) | 项目页 / 论文 | 作为开源 VLA 底座对照，提醒动作接口和评测闭环不能只看模型大小。 | VLA 底座 |
| [外部精读来源台账](external-reading-ledger.md) | 站内台账 | 记录智源、量子位、机器之心、Datawhale 等中文讲法参考，以及 RT-1/RT-2/ACT/FAST 等一手来源入口。 | 讲法参考 |

## 后续维护

新增 VLA 或机器人策略论文时，先标清动作接口属于哪一层：高层技能、末端位姿、关节命令、action chunk、连续动作分布还是 action tokenizer。若来源只展示 demo 或成功率，缺少坐标系、单位、频率、controller 和失败回放，正文只能把它作为能力形态参考，不能写成可部署证据。
