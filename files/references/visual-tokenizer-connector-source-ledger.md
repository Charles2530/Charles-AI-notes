# 视觉 Tokenizer 与连接器来源台账

这份台账服务于 [视觉 Tokenizer 与连接器](../vlm/visual-tokenizers-and-connectors.md)。正文不把 connector 写成组件名列表，而是围绕一条信息链讲：视觉 token 从哪里来，为什么多相机视频会吃掉上下文，connector 怎样选择证据，哪些压缩会伤害 VQA、VLA 和 world model。

## 使用方式

第一类来源是视觉 token 的起点。Vision Transformer 用 patchification 说明图片怎样变成 token 序列；Qwen2-VL / Qwen2.5-VL、Gemini 2.5 等技术报告用于提醒读者：视觉 token 数和动态分辨率决定长视频、文档和 UI 场景能否真的进入上下文。

第二类来源是高维输入压缩。Perceiver 和 TokenLearner 支撑“少量 latent / learned token 主动抽取信息”的讲法。正文借用它们的思想解释：压缩不应该被理解成平均池化，而是一个任务相关的信息选择过程。

第三类来源是 VLM connector。BLIP-2 的 Q-Former、Flamingo 的 Perceiver Resampler 和 gated cross-attention、LLaVA 的 projector + visual instruction tuning 帮助正文区分三条常见接法。正文使用 BLIP-2 Figure 2 作为图源，说明 query token 学到什么取决于训练目标。

第四类来源是 VLA / world model 边界。Octo、RT-2、SpatialVLA、V-JEPA 2 和本站世界模型材料用于提醒：VQA 成功不等于动作、接触、遮挡记忆和未来预测成功。连接器若只按语义问答压缩，可能会丢掉机器人最需要的局部几何证据。

中文和报告式材料主要用于学习讲解顺序：先给 token 成本小账，再解释信息瓶颈，最后落到失败案例和验收证据。事实主轴仍以论文、官方博客、项目页和稳定技术报告为准。

## 写作边界

正文不把视觉 tokenizer 简化成“降 token 数”。Tokenizer 同时改变空间分辨率、局部证据、时序记忆和后续 attention 成本。

正文不把 connector 当作无害 projector。Projector、Q-Former、resampler 和 cross-attention 都会决定哪些证据能被 LLM、policy 或 world model 使用。

正文不把 VQA 指标外推成 VLA 能力。VQA 主要验证语义和语言回答；VLA 还要验证坐标、接触、动作敏感性、遮挡记忆和闭环成功。

正文不把 token compression ratio 当作唯一收益。压缩必须同时报告 step time、显存、局部证据、动作分叉和 closed-loop 任务结果。

## 图片使用

本页没有使用 image2 生成图片。正文只使用已有 BLIP-2 Q-Former 图，作为 connector 信息瓶颈的机制图。

| 图片 | 文件 | 原始来源 | 正文用途 | 使用边界 |
| --- | --- | --- | --- | --- |
| BLIP-2 Q-Former objectives | `../assets/images/paper-figures/vlm/blip2-figure-2-qformer-objectives.jpg` | [BLIP-2](https://arxiv.org/abs/2301.12597) Figure 2 | 说明 query token 怎样从冻结视觉特征中抽取证据，训练目标决定保留什么。 | 不把 VQA / caption 能力外推成动作、接触或世界模型能力。 |

## 来源

| 来源 | 类型 | 本站使用方式 | 关联主题 |
| --- | --- | --- | --- |
| [Vision Transformer](https://arxiv.org/abs/2010.11929) | 论文 | 支撑 patch token 数公式和分辨率 / patch size 的成本解释。 | 视觉 token 起点 |
| [Perceiver](https://arxiv.org/abs/2103.03206) | 论文 | 支撑少量 latent array 从高维输入读取信息的压缩思想。 | 高维输入压缩 |
| [TokenLearner](https://arxiv.org/abs/2106.11297) | 论文 | 支撑学习少量 informative tokens 的图像/视频压缩路线。 | token 选择 |
| [BLIP-2](https://arxiv.org/abs/2301.12597) | 论文 / 图源 | 使用 Figure 2 解释 Q-Former、query token 和训练目标决定信息瓶颈。 | VLM connector |
| [Flamingo](https://arxiv.org/abs/2204.14198) | 论文 | 支撑 Perceiver Resampler、gated cross-attention 和 interleaved visual-text context。 | Cross-attention |
| [LLaVA](https://llava-vl.github.io/) | 项目页 / 论文 | 支撑 projector + visual instruction tuning 的简单强路线。 | Projector |
| [Hugging Face VLMs](https://huggingface.co/blog/vlms) | 技术博客 | 借鉴 vision encoder、connector、language model 三段式解释。 | 教学结构 |
| [Qwen2-VL blog](https://qwenlm.github.io/blog/qwen2-vl/) | 官方博客 | 支撑 dynamic resolution、M-ROPE 和不同视觉输入 token 预算的讨论。 | 动态分辨率 |
| [Gemini 2.5 technical report](https://storage.googleapis.com/deepmind-media/gemini/gemini_v2_5_report.pdf) | 技术报告 | 支撑 video token 数、长上下文视频时长和 tokenizer 成本的报告式读法。 | 视频 token |
| [Octo](https://octo-models.github.io/) | 项目页 / 论文 | 支撑机器人策略中视觉 token、动作 chunk、多机器人数据和闭环评测的边界。 | VLA |
| [RT-2](https://arxiv.org/abs/2307.15818) | 论文 | 支撑 VLM 到 action token 的接口提醒。 | VLA |
| [SpatialVLA](https://arxiv.org/abs/2501.15830) | 论文 | 支撑 3D 位置编码、空间 token 和机器人几何证据。 | 空间证据 |
| [OpenBMB BLIP-2 论文速读](https://hub.baai.ac.cn/view/25807) | 中文公众号转载 | 借鉴“背景 -> 方法 -> 训练目标 -> 局限”的中文讲法，不作为核心事实替代。 | 中文讲法 |
| [AI Insight 多模态统一架构综述](https://www.ai-insight.org/reports/custom_multimodal-unified-architecture-2026) | 中文报告 | 借鉴从 CLIP / BLIP-2 / LLaVA 到统一多模态架构的路线梳理。 | 报告式讲法 |
