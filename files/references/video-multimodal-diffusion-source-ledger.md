# 视频与多模态扩散来源台账

这份台账服务于 [视频与多模态扩散](../diffusion/video-and-multimodal-diffusion.md)。正文不把视频扩散写成“图像扩散加时间轴”，也不把 Wan、Sora、CausVid、动作扩散和世界模型塞成一张模型清单，而是围绕五个接口讲：视频 tokenizer、时空主干、条件持续注入、Flow / 少步采样、评测与世界模型边界。

## 使用方式

第一类来源是早期视频扩散和 latent video diffusion。Video Diffusion Models、Imagen Video、Make-A-Video、Latent Video Diffusion、Stable Video Diffusion 和 Lumiere 支撑“视频不能逐帧独立生成”的基本事实：时空结构、长程一致性和高分辨率成本会共同改变模型设计。

第二类来源是现代视频基础模型系统。Wan 技术报告、Wan 官方仓库、CogVideoX、HunyuanVideo、Movie Gen 和 Sora 技术报告用于解释现代视频系统怎样把 tokenizer、Video DiT、Flow Matching、条件接口、数据过滤、并行训练和推理优化绑在一起。正文使用这些来源时，不把某个 demo 当作通用世界理解证明。

第三类来源是视频少步和流式生成。CausVid、AnyFlow、Phased DMD、Self Forcing、Diffusion Forcing 支撑“视频每一步都贵，所以少步、因果化和流式 rollout 会成为系统问题”。正文只在边界处引用它们，避免把视频基础页变成少步蒸馏专题。

第四类来源是评测。VBench、VBench-I2V、human preference studies、Wan-Bench 和站内证据标准用于说明视频生成必须拆分单帧质量、时间一致性、运动合理性、条件遵循、长时稳定和服务成本。自动指标不能替代人评，更不能直接替代世界模型 closed-loop utility。

第五类来源是多模态和控制连接。Diffusion Policy、RT-X / Open X-Embodiment、DreamZero、LingBot-World 等工作用于解释扩散思想如何迁移到动作轨迹、机器人策略和世界模拟器。正文只把它们作为边界连接，具体机制放到 VLA 和世界模型专题。

第六类来源是讲法参考。Lil'Log、The Illustrated Stable Diffusion、Hugging Face Diffusers / Stable Diffusion blog、Oxen 的 flow transformer 长文，以及机器之心、PaperWeekly、量子位等中文长文适合学习如何把“系统接口 + 失败模式 + 评测边界”讲清楚。事实、实验结果和版本语义仍以论文、技术报告、官方文档和代码仓库为准。

## 写作边界

不要把视频扩散写成逐帧图像扩散。关键是对象身份、运动、镜头、条件和事件顺序必须跨帧自洽。

不要把 tokenizer 当成前处理。视频 VAE 的时间压缩、空间压缩、因果性、chunk cache 和重建指标会直接决定后端模型能看见什么。

不要把 Flow velocity 当成视频中物体的物理速度。它是 video latent 在生成 ODE 里的移动方向，不是 optical flow。

不要把 Sora / Wan 这类视频生成模型直接等同于可闭环世界模型。没有动作条件、因果 rollout、风险评测和 closed-loop utility，就只能保守说它们提供视频世界先验。

不要只看 demo、FID、FVD 或 VBench 总分。应拆 temporal quality、motion degree、identity consistency、I2V consistency、text alignment、长视频漂移、P50/P95 latency 和显存峰值。

## 来源

| 来源 | 类型 | 本站使用方式 | 关联问题 |
| --- | --- | --- | --- |
| [Video Diffusion Models](https://arxiv.org/abs/2204.03458) | 论文 | 支撑 space-time U-Net 和把图像扩散扩到视频的早期路线。 | 时空 U-Net |
| [Imagen Video](https://arxiv.org/abs/2210.02303) | 论文 | 支撑级联视频扩散、文本到高清视频和多阶段生成系统。 | 级联视频 |
| [Make-A-Video](https://arxiv.org/abs/2209.14792) | 论文 | 支撑图像-文本知识迁移到视频生成的路线。 | 数据迁移 |
| [Latent Video Diffusion Models](https://arxiv.org/abs/2211.13221) | 论文 | 支撑在 video latent 上生成以降低高分辨率视频成本。 | latent video |
| [Stable Video Diffusion](https://arxiv.org/abs/2311.15127) | 论文 / 技术报告 | 支撑大数据 latent video diffusion 和 I2V 基线。 | I2V |
| [Lumiere](https://arxiv.org/abs/2401.12945) | 论文 / 项目 | 支撑一次性生成完整时长视频和 Space-Time U-Net 的取舍。 | 时序一致性 |
| [Sora technical report](https://openai.com/index/video-generation-models-as-world-simulators/) | 官方技术报告 | 支撑“视频生成作为世界模拟器”的讨论，同时作为 open-loop / closed-loop 边界提醒。 | 世界先验 |
| [Wan technical report](https://arxiv.org/abs/2503.20314) | 技术报告 / 图源 | 使用 Wan-VAE 和 Wan DiT 图解释现代视频系统接口。 | 视频基础模型 |
| [Wan2.1 GitHub](https://github.com/Wan-Video/Wan2.1) | 官方仓库 | 核对模型、任务接口、推理和开源生态入口。 | 工程入口 |
| [Wan2.2 GitHub](https://github.com/Wan-Video/Wan2.2) | 官方仓库 | 核对 MoE、高低噪声专家、TI2V-5B、Animate 和 S2V 等后续路线。 | 版本边界 |
| [CogVideoX](https://arxiv.org/abs/2408.06072) | 论文 | 作为开放视频 DiT / T2V-I2V 系统的对照来源。 | 视频 DiT |
| [HunyuanVideo](https://arxiv.org/abs/2412.03603) | 技术报告 | 支撑大规模开源视频生成系统的对照。 | 开源视频 |
| [Movie Gen](https://ai.meta.com/research/movie-gen/) | 官方研究页 | 支撑视频、音频和编辑能力统一到生成系统的例子。 | 多模态生成 |
| [VBench](https://arxiv.org/abs/2311.17982) | 论文 / Benchmark | 支撑视频评测必须拆 temporal quality、motion、alignment 等维度。 | 视频评测 |
| [VBench project](https://vchitect.github.io/VBench-project/) | 项目页 | 核对指标维度、榜单和评测入口。 | benchmark |
| [CausVid](https://arxiv.org/abs/2412.07772) | 论文 | 支撑从 bidirectional 视频扩散到 causal / streaming 视频生成的系统问题。 | 流式视频 |
| [AnyFlow](https://arxiv.org/abs/2510.08271) | 论文 | 支撑 any-step 视频扩散和 flow-map 蒸馏的后训练路线。 | 少步视频 |
| [Diffusion Forcing](https://arxiv.org/abs/2407.01392) | 论文 | 支撑视频预测、世界模型和序列扩散之间的连接。 | 序列生成 |
| [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) | 项目页 / 论文 | 支撑扩散思想迁移到动作序列生成，但评测目标变成 closed-loop control。 | 动作扩散 |
| [The Illustrated Stable Diffusion](https://jalammar.github.io/illustrated-stable-diffusion/) | 可视化技术博客 | 学习从 pipeline 和 latent space 解释复杂扩散系统的图文节奏。 | 讲法参考 |
| [Hugging Face Stable Diffusion blog](https://huggingface.co/blog/stable_diffusion) | 技术博客 | 学习从组件、scheduler、latent 和代码入口解释生成系统。 | 工程讲法 |

## 后续写作提醒

写视频扩散时，优先围绕失败机制组织段落：闪烁、变脸、动作变慢、条件漂移、长视频崩塌、首帧不保真、音画不同步、动作不敏感、延迟过高。每个失败都要能映射到 tokenizer、时空主干、条件接口、噪声日程、少步蒸馏或评测桶。

图片优先使用论文、技术报告、官方项目页和高质量技术博客。图注必须说明：图源是谁，原图表达什么，本站正文要读出什么机制，它不能证明什么边界。不要把 demo gallery 当作机制图。
