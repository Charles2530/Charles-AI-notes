# 条件控制与 Guidance 来源台账

这份台账服务于 [条件控制与 Guidance](../diffusion/guidance-and-conditioning.md)。正文不把条件控制写成 prompt 技巧清单，而是围绕一个区分讲：conditioning 是条件怎样进入模型每一步去噪，guidance 是采样时怎样改变或放大条件方向。

## 使用方式

第一类来源是 guidance 机制。Diffusion Models Beat GANs 支撑 classifier guidance 的外部分类器梯度讲法；Classifier-Free Diffusion Guidance 支撑 condition dropout、无条件/有条件预测差额和 guidance weight 的解释；GLIDE 和 Imagen 支撑文本到图像系统里 guidance 与文本条件、编辑、thresholding 的工程关系。

第二类来源是条件入口。ControlNet、T2I-Adapter 和 IP-Adapter 分别支撑结构条件、轻量 adapter 和参考图条件的区别。正文使用它们时，不把边缘、深度、姿态、参考图都简化成“更长 prompt”，而是强调它们的信息形状、注入层级和作用范围不同。

第三类来源是采样稳定性。DPM-Solver++ 支撑高 guidance 与少步 ODE solver 的稳定性讨论；Imagen 支撑高 guidance 下 thresholding 的动机。正文使用这些材料时，把 guidance scale 解释为采样方向场变化，而不是单纯质量滑块。

第四类来源是讲法参考。Lil'Log、Hugging Face Diffusers 文档和 Stable Diffusion 相关技术博客用于校准“从公式到采样动作”的解释顺序。正文中的事实、公式和边界仍以论文、官方文档和代码为准。

图像使用 Classifier-Free Diffusion Guidance 原论文 Figure 1。它只用来说明 guidance weight 会强化条件方向，不能被当成“scale 越大越好”的证据。

## 写作边界

不要把 conditioning 和 guidance 混成一件事。模型看见条件的方式，和采样时放大条件方向的方式，是两个不同设计层。

不要把 CFG scale 写成质量旋钮。它改变 prompt adherence、自然度、多样性和数值稳定性的权衡，少步采样时尤其容易过冲。

不要把 negative prompt 写成事实过滤器。它参与 CFG 的基准分支，更像从某个方向拉开，而不是成图后的绝对禁令。

不要把 ControlNet、T2I-Adapter、IP-Adapter 都写成 prompt 变体。结构条件、轻量控制和参考图条件的输入形状、注入位置和失败模式不同。

不要把视频或世界模型 guidance 只看成单帧画质问题。时间一致性、动作敏感性、首帧保真、长时 rollout 和服务成本都要单独评估。

## 来源

| 来源 | 类型 | 本站使用方式 | 关联问题 |
| --- | --- | --- | --- |
| [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233) | 论文 | 支撑 classifier guidance、噪声鲁棒分类器梯度和 guidance 改变采样轨迹。 | classifier guidance |
| [GLIDE](https://arxiv.org/abs/2112.10741) | 论文 | 支撑文本到图像 diffusion、classifier-free guidance、文本条件编辑和系统背景。 | 文本条件 |
| [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) | 论文 / 图源 | 使用 Figure 1 说明 guidance weight 对类别条件样本的影响，支撑 condition dropout 和 CFG 公式。 | CFG |
| [Imagen](https://arxiv.org/abs/2205.11487) | 论文 | 支撑大规模文生图系统、高 guidance、thresholding 和文本编码器影响。 | 高 guidance |
| [ControlNet](https://arxiv.org/abs/2302.05543) | 论文 | 支撑结构条件通过额外控制分支注入预训练扩散模型。 | 空间控制 |
| [ControlNet GitHub](https://github.com/lllyasviel/ControlNet) | 官方仓库 | 核对结构条件类型、模型入口和工程生态；不把 gallery 当泛化证明。 | 工程入口 |
| [T2I-Adapter](https://arxiv.org/abs/2302.08453) | 论文 | 支撑轻量 adapter 把额外条件接入文生图模型。 | 轻量控制 |
| [IP-Adapter](https://arxiv.org/abs/2308.06721) | 论文 | 支撑图像 prompt / 参考图条件与文本条件解耦的讲法。 | 参考图条件 |
| [Prompt-to-Prompt](https://arxiv.org/abs/2208.01626) | 论文 | 作为 attention 控制和文本编辑的相邻背景，提醒 prompt 控制也有空间和词级接口。 | 文本编辑 |
| [DPM-Solver++](https://arxiv.org/abs/2211.01095) | 论文 | 支撑 guided diffusion 下少步 solver 和 thresholding 稳定性讨论。 | 少步采样 |
| [Hugging Face Diffusers guiders docs](https://huggingface.co/docs/diffusers/en/api/modular_diffusers/guiders) | 官方文档 | 核对 guidance scale、classifier-free guidance 和实际 pipeline 接口表述。 | 工程接口 |
| [Lil'Log: What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) | 技术博客 | 借鉴从 DDPM、score、classifier guidance 到 CFG 的递进解释方式。 | 讲法参考 |

## 后续写作提醒

写条件控制时，段落最好围绕冲突和失败机制组织：prompt 贴合但自然度下降、结构图保住但语义变弱、参考图保身份却压掉风格、negative prompt 误伤细节、少步高 CFG 过冲、视频强 guidance 闪烁或动作不敏感。每个失败都要能回到条件入口、采样方向或评测桶之一。

图片优先来自论文、官方项目页和高质量技术博客。图注必须说明：原图表达什么，本站正文要读出什么机制，它不能证明什么边界。
