# 噪声日程与参数化来源台账

这份台账服务于 [噪声日程与参数化](../diffusion/noise-schedules-and-parameterization.md)。正文不把 schedule 写成一串 \(\beta_t\) 配置，也不把 \(\epsilon\)、\(x_0\)、score、\(v\) 写成符号表，而是围绕一个问题讲：噪声区间怎样分配训练难度，输出坐标怎样影响优化和采样稳定性。

## 使用方式

第一类来源是基础前向过程。DDPM 支撑 \(\bar\alpha_t\)、闭式加噪、\(\epsilon\)-prediction 和反向链的基础写法。

第二类来源是 schedule 与 loss weighting。Improved DDPM 支撑 cosine schedule、learned variance 和 VLB timestep 贡献图；Min-SNR weighting 支撑不同时间步优化冲突和按 SNR 调整 loss 权重的讨论。

第三类来源是参数化。Progressive Distillation 支撑少步采样压力下新参数化与 distillation 的关系；Common Diffusion Noise Schedules and Sample Steps are Flawed 支撑 zero terminal SNR、v-prediction、sampler 起点和 CFG rescale 等训练-推理错配边界。

第四类来源是工程设计空间。EDM 支撑噪声尺度 \(\sigma\)、preconditioning、loss weighting、solver、stochasticity 和 NFE 必须一起看的写法；Diffusers scheduler 文档用于核对工程库里的 scheduler、prediction type 和 timestep spacing 语义。

第五类来源是讲法参考。The Annotated Diffusion Model 和 Lil'Log 用代码或递进解释帮助校准中文讲法；事实、图源和边界仍以论文和官方文档为准。

第六类来源是中文直觉与随机性边界。科学空间的 DDPM 漫谈用于帮助解释前向加噪里数据系数和噪声系数为什么要一起看，尤其是 \(\alpha_t^2+\beta_t^2=1\) 如何让多步高斯噪声合并为闭式采样；“零推导理解 Diffusion 和 Flow Matching”系列用于提供圆弧 / 直线的统一路径直觉；CPS 论文用于正式支撑“随机性注入必须保持 scheduler 指定的数据系数和总噪声水平”。

图像使用三类：本站自绘的 DDPM 数据/噪声系数几何图、Improved DDPM Figure 2 和 EDM Figure 2。自绘图用来讲 \(x_t\) 如何由数据方向和噪声方向组合；Improved DDPM 图用来讲不同 timestep 的训练压力；EDM 图用来讲采样设计空间。图注必须提醒读者：这些图不能单独证明某个 schedule 或 sampler 普遍最优。

## 写作边界

不要把线性、cosine 或 EDM schedule 写成固定优劣排名。它们服务不同噪声参数化、loss weighting、模型输出坐标和采样器。

不要把 SNR 只当公式。SNR 是理解训练难度、loss weighting、少步误差和视频高低噪声职责分工的核心语言。

不要把 \(v\)-prediction 写成万能更稳。它是坐标选择，仍然依赖 schedule、loss weighting、sampler、guidance 和训练数据。

不要把训练 schedule 和 sampling schedule 混为一谈。训练覆盖哪些噪声水平，推理从哪些噪声水平走过，是两个相互约束但不同的问题。

不要只用平均 loss 评估 schedule。要按 SNR/timestep bucket、条件类型、sampler、NFE、guidance、细节和视频运动拆开看。

## 来源

| 来源 | 类型 | 本站使用方式 | 关联问题 |
| --- | --- | --- | --- |
| [DDPM](https://arxiv.org/abs/2006.11239) | 论文 | 支撑闭式加噪、\(\bar\alpha_t\)、\(\epsilon\)-prediction 和反向采样基础。 | 前向过程 |
| [Improved DDPM](https://arxiv.org/abs/2102.09672) | 论文 / 图源 | 使用 Figure 2 解释 VLB timestep 贡献，支撑 cosine schedule 和 learned variance。 | schedule |
| [Progressive Distillation](https://arxiv.org/abs/2202.00512) | 论文 | 支撑少步采样压力下的参数化选择和 progressive distillation 背景。 | 参数化 / 少步 |
| [EDM](https://arxiv.org/abs/2206.00364) | 论文 / 图源 | 使用 Figure 2 解释 \(\sigma\)、preconditioning、solver、stochasticity 和 NFE 的设计空间。 | 工程设计 |
| [Min-SNR Weighting](https://arxiv.org/abs/2303.09556) | 论文 | 支撑不同 timestep / SNR 任务冲突和 loss weighting 的讨论。 | 训练权重 |
| [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/abs/2305.08891) | 论文 | 支撑 zero terminal SNR、v-prediction、sampler 起点和 CFG rescale 的训练-推理错配提醒。 | schedule 错配 |
| [Diffusers Scheduler Overview](https://huggingface.co/docs/diffusers/en/api/schedulers/overview) | 官方文档 | 核对 scheduler、prediction type 和 timestep spacing 在工程实现中的语义。 | 工程接口 |
| [The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion) | 技术博客 / 代码讲解 | 借鉴 forward process、schedule、loss 和采样代码如何对齐的解释顺序。 | 讲法参考 |
| [Lil'Log: What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) | 技术博客 | 借鉴从 DDPM 到 score / SDE / sampling 的递进讲法。 | 讲法参考 |
| [科学空间：生成扩散模型漫谈（一）：DDPM = 拆楼 + 建楼](https://kexue.fm/archives/9119) | 中文技术博客 | 借鉴“拆楼 / 建楼”、系数平方和、高斯噪声合并和低方差噪声预测的解释方式；正文事实仍以 DDPM 为准。 | 讲法参考 / 系数 |
| [零推导理解 Diffusion 和 Flow Matching](https://zhuanlan.zhihu.com/p/11228697012) | 中文技术博客 | 用正交轴和圆弧 / 直线建立 Diffusion 与 Flow Matching 的统一路径直觉。 | 讲法参考 |
| [系数保持采样：Flow Matching 随机性注入的正确打开方式](https://zhuanlan.zhihu.com/p/1948388095151026330) | 中文技术博客 | 作为 Flow Matching 随机性注入与系数保持问题的中文阅读入口。 | 随机性 / Flow |
| [Coefficients-Preserving Sampling for Reinforcement Learning with Flow Matching](https://arxiv.org/abs/2509.05952) | 论文 | 正式支撑系数保持采样、总噪声水平和 Flow-SDE 随机性伪影的讨论。 | 随机性 / 系数保持 |

## 后续写作提醒

写噪声日程时，优先围绕失败机制组织：高噪声语义弱、低噪声细节脏、中噪声结构不稳、少步跨度过大、CFG 过曝、zero terminal SNR 错配、视频运动变慢、动作条件不敏感。每个失败都要能映射到 SNR bucket、参数化、loss weighting、sampling schedule 或 condition interface 之一。

图片优先来自论文、官方项目页和高质量技术博客。图注必须说明：原图表达什么，本站正文要读出什么机制，它不能证明什么边界。
