# 一步生成与蒸馏来源台账

这份台账服务于 [一步生成、蒸馏与整流](../diffusion/distillation.md)。正文不把少步扩散写成“采样器少跑几次”或“某某方法更快”的清单，而是围绕一个核心问题讲：当扩散模型失去几十次小纠错机会时，训练端怎样补回轨迹、分布、细节和条件控制。

## 使用方式

第一类来源是轨迹蒸馏。Progressive Distillation 负责支撑“把教师两步复合成学生一步”的讲法。正文使用它时，要把学生目标解释成教师采样轨迹的压缩，而不是把采样步数直接砍掉。

第二类来源是一致性路线。Consistency Models 和 Latent Consistency Models 支撑“同一 PF-ODE 或教师轨迹上的不同噪声点应回到同一终点”。正文使用它们时，要强调终点一致适合少步推理，但不自动保证所有细节、强条件控制和局部编辑都保住。

第三类来源是分布匹配蒸馏。DMD、DMD2 和 Phased DMD 支撑正文里的主线：一步或少步学生没有足够长的中间轨迹可以逐点模仿，因此需要用最终分布、fake score、GAN critic、阶段训练和真实推理链路模拟来约束学生。DMD 系列图只用来解释训练信号和阶段职责，不用来证明所有任务都能无损一步生成。

第四类来源是整流与速度场。Rectified Flow、Flow Matching 和 Rectified Diffusion 支撑“重画生成路径”这一类方法。正文要把它和 teacher-student 蒸馏分开：前者更像改变从噪声到数据的运输路径，后者更像训练一个学生去继承已有教师。

第五类来源是中英文长文讲法。Lil'Log、Yang Song 的 score 讲解、The Illustrated Stable Diffusion、Hugging Face 的扩散博客，以及机器之心、PaperWeekly 等中文公众号式长文适合学习解释节奏：先把读者真正困惑的问题摆出来，再用图、公式和反例逐层推进。事实、图源、实验结果和版本语义仍以论文、官方项目页和代码仓库为准。

## 写作边界

不要把 few-step / one-step 写成“删除采样步”。少步学生每一步跨越更大噪声区间，训练端必须承担原本在线纠错的工作。

不要只比较 NFE 或漂亮样例。少步蒸馏至少要按步数、条件、视觉细节、分布覆盖、视频运动和端到端延迟拆桶评估。

不要把 DMD 写成 DDIM 跳步到一步。DMD 的核心是 distribution matching：学生最终样本分布要接近目标分布，而不是中间轨迹逐点跟教师一样。

不要把 Consistency / LCM 写成万能加速器。它们适合快速预览和少步生成，但强 prompt、复杂空间关系、ControlNet、文字和局部编辑仍可能变弱。

不要把 Rectified Flow 写成“路径越直一定越好”。少步生成需要路径、时间采样、条件控制和数据几何一起成立；几何直线不等于任务保真。

不要把视频或世界模型蒸馏只看成画质问题。少步学生可能把运动压慢、把动作敏感性压掉，必须单独看 motion bucket、action bucket 和长时 rollout。

## 来源

| 来源 | 类型 | 本站使用方式 | 关联问题 |
| --- | --- | --- | --- |
| [Progressive Distillation](https://arxiv.org/abs/2202.00512) | 论文 | 支撑教师两步复合成学生一步、逐级减半步数的轨迹蒸馏讲法。 | 轨迹压缩 |
| [Consistency Models](https://arxiv.org/abs/2303.01469) | 论文 | 支撑同一轨迹不同噪声点映射到一致终点的少步生成路线。 | 终点一致 |
| [Latent Consistency Models](https://arxiv.org/abs/2310.04378) | 论文 | 支撑把一致性蒸馏接到 latent diffusion 和 Stable Diffusion 生态。 | LCM |
| [DMD](https://arxiv.org/abs/2311.18828) | 论文 / 图源 | 使用 Figure 2 和 Figure 3 解释 distribution matching、real/fake score 和 regression anchor。 | 一步分布匹配 |
| [DMD project page](https://tianweiy.github.io/dmd/) | 项目页 | 核对 DMD 图、模型入口和任务示例，不把项目样例当泛化证明。 | 图源核对 |
| [DMD2](https://arxiv.org/abs/2405.14867) | 论文 / 图源 | 使用 Figure 3 解释 fake score function、GAN loss 和训练-推理错配处理。 | 少步稳定性 |
| [DMD2 project page](https://tianweiy.github.io/dmd2/) | 项目页 | 核对 1/4-step、LoRA / UNet、demo 和代码入口。 | 工程入口 |
| [Phased DMD](https://arxiv.org/abs/2510.27684) | 论文 / 图源 | 使用 Figure 1 解释少步蒸馏仍要保留 SNR 阶段职责。 | 阶段蒸馏 |
| [Phased DMD project page](https://x-niper.github.io/projects/Phased-DMD/) | 项目页 | 核对 phased training 图和视频生成示例，提醒视频评测要看运动。 | 视频少步 |
| [Rectified Flow](https://arxiv.org/abs/2209.03003) | 论文 | 支撑把噪声到数据的运输路径变得更适合粗步积分。 | 速度场 |
| [Flow Matching](https://arxiv.org/abs/2210.02747) | 论文 | 支撑连续速度场和条件路径学习的背景。 | 生成路径 |
| [Rectified Diffusion](https://arxiv.org/abs/2410.07303) | 论文 | 支撑 rectified 路线不只是“画直线”，还要考虑扩散路径和采样质量。 | 路径重设 |
| [DPM-Solver++](https://arxiv.org/abs/2211.01095) | 论文 | 作为不用重训时的少步 solver 对照，帮助正文区分采样器加速与蒸馏。 | solver 对照 |
| [Lil'Log: What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) | 英文技术博客 | 借鉴从 DDPM、score、SDE 到 guidance 的递进讲法。 | 教学结构 |
| [Yang Song score blog](https://yang-song.net/blog/2021/score/) | 作者博客 | 借鉴 score、Langevin dynamics 和连续时间扩散的直觉解释。 | score 直觉 |
| [The Illustrated Stable Diffusion](https://jalammar.github.io/illustrated-stable-diffusion/) | 可视化技术博客 | 借鉴“图像生成 pipeline 先整体再局部”的图文节奏。 | 图文讲法 |
| [Hugging Face Stable Diffusion blog](https://huggingface.co/blog/stable_diffusion) | 技术博客 / 代码讲解 | 借鉴从组件、latent space、scheduler 到推理代码的解释顺序。 | 工程讲法 |

## 后续写作提醒

写少步扩散时，段落最好围绕失败机制组织：大结构漂、细节软、条件弱、模式坍缩、视频变慢、端到端延迟没有线性下降。每个失败都要接到一个训练信号或评测桶上，不要只写方法名。

图片优先来自论文、项目页和官方技术博客。图注必须说明三件事：原图表达什么，本站正文要读出什么机制，它不能证明什么边界。
