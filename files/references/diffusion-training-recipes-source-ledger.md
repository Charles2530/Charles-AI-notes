# 扩散训练配方与排障来源台账

这份台账服务于 [扩散训练配方与排障](../diffusion/practical-training-recipes-and-failure-analysis.md)。正文不把训练写成孤立开关罗列，而是围绕失败归因讲：数据、表示空间、SNR bucket、条件接口、优化稳定性、真实采样链路和任务指标怎样共同决定扩散训练是否可用。

## 使用方式

第一类来源是基础训练目标。DDPM 和 Improved DDPM 支撑噪声预测、反向链、learned variance、cosine schedule 和 timestep 贡献差异。

第二类来源是噪声权重与采样设计。EDM 和 Min-SNR weighting 支撑 SNR bucket、preconditioning、loss weighting、sampler 和 NFE 必须一起看的排障方式。

第三类来源是表示空间和模型规模。Latent Diffusion Models 支撑 VAE latent 训练的成本与质量边界；DiT 支撑 forward GFLOPs、模型规模和 FID 之间的计算权衡，图像使用 DiT Figures 8-9 的组合图。

第四类来源是条件与 guidance。Classifier-Free Diffusion Guidance 支撑 condition dropout 和 CFG 的训练基础；正文只用它解释为什么条件分支和无条件分支都需要在训练中出现。

第五类来源是少步蒸馏训练稳定性。DMD2 支撑 fake critic、regression anchor、brightness / convergence 震荡和 distribution matching loss 不能按普通 MSE 曲线理解。

第六类来源是动作扩散和闭环评测。Diffusion Policy 用于提醒：当扩散输出动作轨迹时，视觉质量或轨迹 MSE 不是最终目标，closed-loop success、碰撞和恢复能力才是关键。

正文中的经验判断只作为排障路径，不写成固定超参建议。不同数据规模、分辨率、latent tokenizer、condition interface 和硬件预算会改变合适设置；每次引用配方时都要连到可观测失败和验证指标。

## 写作边界

不要把训练配方写成通用最佳实践。每个配方项都要能解释一种失败或一种成本取舍。

不要只看平均 loss。必须按 SNR bucket、条件类型、真实 sampler、NFE、CFG、视频时间一致性或任务成功率拆开。

不要把 VAE / tokenizer 当成前处理。latent 压缩、重建质量、时间压缩和 token 数会直接决定扩散主干能学到什么。

不要把少步蒸馏 loss 当成普通监督回归。distribution matching 里 generator、fake critic 和 teacher score 是动态系统。

不要把动作扩散的 MSE 当成成功率。多峰动作平均会让轨迹平滑但不可执行，必须看闭环。

## 来源

| 来源 | 类型 | 本站使用方式 | 关联问题 |
| --- | --- | --- | --- |
| [DDPM](https://arxiv.org/abs/2006.11239) | 论文 | 支撑噪声预测目标、反向采样和训练基础。 | 基础训练 |
| [Improved DDPM](https://arxiv.org/abs/2102.09672) | 论文 | 支撑 learned variance、cosine schedule 和 timestep 贡献差异。 | schedule |
| [EDM](https://arxiv.org/abs/2206.00364) | 论文 | 支撑噪声尺度、preconditioning、loss weighting、sampler 和 NFE 的共同设计。 | 训练/采样接口 |
| [Min-SNR Weighting](https://arxiv.org/abs/2303.09556) | 论文 | 支撑不同时间步优化冲突和 SNR bucket 诊断。 | loss weighting |
| [Latent Diffusion Models](https://arxiv.org/abs/2112.10752) | 论文 | 支撑 latent space 训练、VAE 压缩成本和质量边界。 | 表示空间 |
| [DiT](https://arxiv.org/abs/2212.09748) | 论文 / 图源 | 使用 Figures 8-9 说明 forward GFLOPs、训练计算和生成质量的权衡。 | 计算预算 |
| [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) | 论文 | 支撑 condition dropout、有条件/无条件分支和 CFG 训练基础。 | 条件接口 |
| [DMD2](https://arxiv.org/abs/2405.14867) | 论文 / 图源 | 使用 Appendix Figure 7/8 解释 fake critic 更新、亮度震荡和收敛稳定性。 | 分布匹配 |
| [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) | 项目页 / 论文 | 支撑动作扩散评估必须看 closed-loop control，而不是只看轨迹回归。 | 动作扩散 |

## 后续写作提醒

写训练配方时，先从失败现象开始：糊、主体错、prompt 不听、细节坏、视频闪烁、动作成功率下降、少步学生变钝、端到端延迟超标。每个现象都要能追到数据、SNR、条件、优化、采样链路或任务指标之一。

图片优先来自论文、官方项目页和高质量技术博客。图注必须说明：原图表达什么，本站正文要读出什么机制，它不能证明什么边界。
