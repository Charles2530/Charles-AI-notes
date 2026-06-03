# 一致性模型与 Rectified Flow 来源台账

这份台账服务于 [一致性模型与 Rectified Flow](../diffusion/consistency-models-and-rectified-flow.md)。正文不把 Consistency、LCM、Flow Matching、Rectified Flow、Rectified Diffusion 写成少步生成名词表，而是围绕一条区分来讲：一致性模型学习终点映射，Flow / Rectified 路线学习路径上的速度场。

## 使用方式

第一类来源是一致性模型。Consistency Models 支撑“同一 PF-ODE 轨迹上的不同噪声点应映射到一致输出”；Latent Consistency Models 和 LCM 项目页支撑把这个目标接进 latent diffusion、Stable Diffusion 和 LoRA 生态。正文使用它们时，要强调终点一致与细节保真之间的张力。

第二类来源是 Flow Matching。Flow Matching 论文和 Meta guide 支撑“先定义概率路径，再回归路径速度”的统一表述。正文使用它们时，要把 \(v_\theta(x,t,c)\) 写成 ODE 右边的速度场，而不是普通扩散里的输出坐标。

第三类来源是 Rectified Flow 和 Rectified Diffusion。Rectified Flow 论文、官方实现和图源支撑路径整流、reflow、少步 Euler 采样；Rectified Diffusion 支撑写作边界：直线或低曲率不是唯一目标，路径必须和数据几何、条件控制、局部细节和采样质量一起看。

第四类来源是现代系统实践。Scaling Rectified Flow Transformers、Wan 技术报告和站内 Wan 专题用于说明 Flow Matching / velocity 风格目标已经进入高分辨率图像、视频 DiT 和多模态生成底座。正文不要把这些实践写成“传统扩散已经被淘汰”，更准确的讲法是：训练目标、主干、tokenizer、solver 和条件注入在共同迁移。

第五类来源是讲法参考。Lil'Log、The Illustrated Stable Diffusion、Hugging Face Diffusers / Stable Diffusion blog、Oxen 的 Rectified Flow Transformer 长文，以及机器之心、PaperWeekly 等中文长文适合学习解释节奏：先讲读者困惑，再用一个公式或图抓住机制，最后讲边界。事实和实验结论仍以论文、官方文档、项目页和代码仓库为准。

## 写作边界

不要把 Consistency / LCM 写成“更快的采样器”。它们通常需要新的训练或蒸馏目标，核心对象是跨噪声层级一致的终点映射。

不要把 Flow Matching 写成“扩散换名字”。Flow Matching 的模型输出直接被定义为 ODE 速度场；扩散 \(v\)-prediction 则常是输出参数化，采样器还要换算到更新规则。

不要把 Rectified Flow 写成“路径越直越好”。路径、时间采样、数据几何、条件注入、局部细节和少步 solver 都会影响质量。

不要把少步结果只按 FID 或样例图判断。需要按真实步数、CFG、prompt 难度、ControlNet/参考图、视频运动、局部细节和服务延迟拆开。

不要把视频里的 Flow velocity 当成物理运动速度。它是 video latent 的生成方向，画面中人物或相机怎么动是生成内容，不是 ODE 变量本身。

## 来源

| 来源 | 类型 | 本站使用方式 | 关联问题 |
| --- | --- | --- | --- |
| [Consistency Models](https://arxiv.org/abs/2303.01469) | 论文 | 支撑 PF-ODE 轨迹上不同时间点映射一致、distillation 与 training from scratch。 | 终点一致 |
| [Latent Consistency Models](https://arxiv.org/abs/2310.04378) | 论文 | 支撑把 consistency distillation 接进 latent diffusion 和少步文生图。 | LCM |
| [LCM project page](https://latent-consistency-models.github.io/) | 项目页 | 核对 LCM 生态、少步示例和 LoRA 接入方式；不把样例图当泛化证明。 | 工程入口 |
| [Flow Matching](https://arxiv.org/abs/2210.02747) | 论文 | 支撑连续概率路径、目标速度和 ODE 采样框架。 | 速度场 |
| [Meta Flow Matching Guide and Code](https://ai.meta.com/research/publications/flow-matching-guide-and-code/) | 官方研究页 | 作为统一写法和代码入口参考，帮助正文区分路径、速度、采样。 | 工程讲法 |
| [Rectified Flow](https://arxiv.org/abs/2209.03003) | 论文 | 支撑线性路径、reflow 和少步生成的基本机制。 | 路径整流 |
| [Rectified Flow official repo](https://github.com/gnobitab/RectifiedFlow) | 官方实现 / 图源 | 使用 `github_misc/intro_two_gauss.png` 解释两高斯路径整流。 | 图源 |
| [Rectified Diffusion](https://arxiv.org/abs/2410.07303) | 论文 | 支撑“straightness is not your need”的边界提醒。 | 路径边界 |
| [Scaling Rectified Flow Transformers](https://arxiv.org/abs/2403.03206) | 论文 / 技术报告 | 支撑 SD3 路线中 rectified flow、DiT 和高分辨率文生图实践。 | 系统实践 |
| [Wan technical report](https://arxiv.org/abs/2503.20314) | 技术报告 | 支撑视频 VAE、Video DiT 和 Flow Matching / velocity 目标的现代视频生成实践。 | 视频扩散 |
| [Lil'Log: What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) | 技术博客 | 借鉴从 DDPM、score、SDE 到 guidance 的递进讲法。 | 教学结构 |
| [The Illustrated Stable Diffusion](https://jalammar.github.io/illustrated-stable-diffusion/) | 可视化技术博客 | 借鉴图文解释 pipeline 的节奏。 | 图文讲法 |
| [Hugging Face Stable Diffusion blog](https://huggingface.co/blog/stable_diffusion) | 技术博客 / 代码讲解 | 借鉴从组件、scheduler、latent space 到推理代码的解释顺序。 | 工程讲法 |
| [Oxen: How Flux and Rectified Flow Transformers Work](https://www.oxen.ai/blog/arxiv-dive-how-flux-and-rectified-flow-transformers-work/) | 技术博客 | 借鉴把 SD3 / Flux 类 rectified-flow transformer 讲给工程读者的方式。 | 现代长文 |

## 后续写作提醒

写这条线时，先问“模型输出是什么”再问“步数是多少”。如果输出是终点估计，就讲一致性、边界条件和 teacher 轨迹；如果输出是速度场，就讲路径、目标速度、ODE solver 和 time schedule；如果输出是最终样本分布的纠偏梯度，就转去 DMD / DMD2 / Phased DMD。

图片只选能解释机制的图。结果样例可以作为项目页入口，不应替代机制说明。图注必须写清：图源、原图表达什么、本站要读出什么、它不能证明什么。
