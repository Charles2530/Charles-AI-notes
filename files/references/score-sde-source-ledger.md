# Score Matching 到 SDE 来源台账

这份台账服务于 [Score Matching 到 SDE](../diffusion/score-matching-sde-and-probability-flow.md)。正文不把 DDPM、Score SDE、Probability Flow ODE、DPM-Solver 写成名词清单，而是围绕一条方向场主线讲：score 是概率密度的坡度，噪声预测怎样转换成 score，反向 SDE 为什么需要 score，Probability Flow ODE 为什么能接上确定性采样器。

## 使用方式

第一类来源是 score 直觉。Yang Song 的作者博客和 Lil'Log 用更适合学习的顺序解释 score function、Langevin dynamics、多噪声扰动、DDPM 和 SDE。正文借鉴它们“先讲方向，再讲采样”的方式。

第二类来源是基础论文。DDPM 支撑离散前向加噪、噪声预测和反向链；Score SDE 支撑连续时间 SDE、reverse-time SDE、probability flow ODE 和 likelihood / latent encoding；DDIM 支撑确定性采样与跳步。

第三类来源是数值求解器和少步采样。DPM-Solver / DPM-Solver++、EDM 和 consistency / rectified-flow 相关材料帮助正文说明：采样器不是魔法名字，而是有限 NFE 下如何积分方向场。

第四类来源是中文解释。科学空间的“生成扩散模型漫谈”系列，尤其“得分匹配 = 条件得分匹配”，用于校准中文表达：DDPM 常见训练目标更准确地说是条件 score matching，而不是直接监督不可知的边缘 score。

图像使用只选择 Score SDE 原论文图，避免把不同论文的采样路径混在一张读法里。Figure 1 用来讲正向加噪和反向 score 生成，Figure 3 用来讲 probability flow ODE 的确定性轨迹。

## 写作边界

正文不把噪声预测和 score 完全等同。二者有符号和尺度关系，且不同参数化下会变成 \(x_0\)、\(v\) 或 velocity field。

正文不把 SDE 写成炫技数学。SDE 的价值在于统一 DDPM、SMLD、reverse process、probability flow ODE、likelihood 和 solver。

正文不把 ODE sampler 说成必然优于 SDE sampler。ODE 可确定、可编码 latent、可用高阶 solver，但质量和稳定性仍取决于模型、路径、步数和 guidance。

正文不把少步采样说成简单跳步。少步会放大方向场误差和数值误差，常需要专用 solver、蒸馏、consistency 或重新训练目标。

## 来源

| 来源 | 类型 | 本站使用方式 | 关联主题 |
| --- | --- | --- | --- |
| [Yang Song score blog](https://yang-song.net/blog/2021/score/) | 作者博客 | 借鉴 score function、Langevin dynamics、多噪声扰动和 SDE 的直觉讲法。 | score 直觉 |
| [Lil'Log: What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) | 技术博客 | 借鉴从 DDPM 到 score / SDE / guidance 的递进结构。 | 教学结构 |
| [DDPM](https://arxiv.org/abs/2006.11239) | 论文 | 支撑前向加噪、噪声预测、反向扩散链和训练目标。 | 离散扩散 |
| [DDIM](https://arxiv.org/abs/2010.02502) | 论文 | 支撑确定性采样、非马尔可夫反向过程和跳步。 | ODE 采样 |
| [Score SDE](https://arxiv.org/abs/2011.13456) | 论文 / 图源 | 使用 Figure 1 和 Figure 3 解释 reverse SDE 与 probability flow ODE。 | SDE / ODE |
| [DPM-Solver](https://arxiv.org/abs/2206.00927) | 论文 | 支撑 diffusion ODE 的专用高阶 solver 视角。 | 少步采样 |
| [DPM-Solver++](https://arxiv.org/abs/2211.01095) | 论文 | 支撑 guided diffusion 下 data prediction solver 更稳的讨论。 | 少步采样 |
| [EDM](https://arxiv.org/abs/2206.00364) | 论文 | 支撑噪声尺度、preconditioning、solver 和 NFE 是同一设计空间。 | 工程采样 |
| [Flow Matching](https://arxiv.org/abs/2210.02747) | 论文 | 支撑速度场和 ODE path 的生成视角。 | 速度场 |
| [Rectified Flow](https://arxiv.org/abs/2209.03003) | 论文 | 支撑从噪声到数据学习更直路径的对照。 | Rectified Flow |
| [科学空间：得分匹配 = 条件得分匹配](https://spaces.ac.cn/archives/9509) | 中文长文 | 借鉴中文推导方式，强调 DDPM 常见目标与条件 score matching 的关系。 | 中文讲法 |
