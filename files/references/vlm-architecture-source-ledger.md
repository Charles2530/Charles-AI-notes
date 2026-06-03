# VLM 架构来源台账

这份台账服务于 [VLM 架构](../vlm/architecture-and-training.md)。正文不把 VLM 写成模型名列表，而是围绕一条证据链讲：视觉 token 如何产生，连接器如何压缩证据，LLM 如何消费视觉 token，视频状态和动作接口为什么会把普通 VLM 推向 VLA / 世界模型。

## 使用方式

第一类来源是图文语义对齐。CLIP 论文、OpenAI CLIP 介绍和 The Illustrated CLIP 帮助本站解释 dual encoder、contrastive learning、相似度矩阵和 zero-shot classifier。本站使用 CLIP Figure 1 说明静态语义对齐的能力边界。

第二类来源是视觉连接器。BLIP-2、Flamingo、LLaVA 和 Hugging Face VLM 教程帮助本站解释 projector、Q-Former、Perceiver Resampler、cross-attention 和 visual instruction tuning。本站使用 BLIP-2 Figure 2 说明 query-based connector 是信息瓶颈。

第三类来源是视频状态和世界模型接口。V-JEPA、V-JEPA 2 和本站世界模型深读材料帮助本站说明：静态图文语义不能直接外推到动作后果；视频 latent prediction 可以学习时空状态，但还需要 action conditioning、planner 和闭环评测。

第四类来源是 VLA / 具身智能材料。RT-2、Octo、SpatialVLA、机器人数据和控制接口文章帮助本站提醒读者：VLM 会描述图片，不代表视觉状态已经适合机器人执行。坐标、接触、遮挡、夹爪、相机外参和动作 token 都是额外接口。

中文讲法主要学习结构：先给一个读者能感受到的任务差异，例如“识别杯子”和“抓起杯子”，再把视觉 token、connector、LLM、状态和动作接口串起来。事实和图片以论文、官方文档、项目页和稳定博客为准。

## 中文与报告式材料

这一页没有把中文材料当作论文证据替代品，而是用来校准讲法：好的公众号文章和中文报告通常先回答“这篇工作为什么出现”，再按方法、训练、结果、局限展开。本站正文采用这种顺序，但结论仍以论文和官方项目页为准。

- [OpenBMB BLIP-2 论文速读](https://hub.baai.ac.cn/view/25807)：智源社区转载的公众号式论文速读，把 BLIP-2 讲成“冻结单模态模型 + 轻量 Q-Former + 两阶段训练”，适合作为中文读者的第一层直觉。
- [智源社区 LLaVA 条目](https://hub.baai.ac.cn/view/25500)：保留了 LLaVA 原文、项目页和视觉指令微调的四个要点，可用于核对中文表达是否偏离论文主张。
- [AI Insight 多模态统一架构综述](https://www.ai-insight.org/reports/custom_multimodal-unified-architecture-2026)：用发展阶段和三段式架构串起 CLIP、BLIP-2、LLaVA、InternVL、Qwen-VL，适合补“从单篇论文到路线图”的背景。

## 写作边界

正文不把 CLIP 说成完整 VLM。CLIP 是强图文语义对齐底座，但不学习动作、接触、遮挡记忆和动作后果。

正文不把 connector 当成无害适配层。Projector、Q-Former、resampler 和 cross-attention 都会决定哪些视觉证据能穿过瓶颈；VQA 成功不等于 VLA 成功。

正文不把 VLM、VLA、world model 混成一个层级。VLM 生成回答，VLA 输出动作，world model 预测动作后的状态；三者需要不同训练目标和评测证据。

正文不把视频状态等同于完整世界模型。V-JEPA-style latent prediction 能学习时空表征，但如果没有动作条件、reward/risk、规划接口和闭环任务，只能作为状态 encoder 候选。

## 图片使用

本页没有使用 image2 生成图片。图片均来自论文或站内已归档资产；图注必须说明原图机制和本站读法，不能把图外推成动作、规划或部署能力。

| 图片 | 文件 | 原始来源 | 正文用途 | 使用边界 |
| --- | --- | --- | --- | --- |
| CLIP 预训练与 zero-shot | `../assets/images/paper-figures/contrastive-learning/clip-figure-1.png` | [CLIP](https://arxiv.org/abs/2103.00020) Figure 1 | 说明图文对比学习怎样建立静态语义空间。 | 不证明动作、接触、遮挡记忆或未来后果建模。 |
| BLIP-2 Q-Former | `../assets/images/paper-figures/vlm/blip2-figure-2-qformer-objectives.jpg` | [BLIP-2](https://arxiv.org/abs/2301.12597) Figure 2 | 说明 Q-Former / query connector 是信息瓶颈。 | 不证明 query token 一定保留 VLA 所需局部几何。 |
| V-JEPA 训练流程 | `../assets/images/paper-deep-dives/world-models/v-jepa/figure-3-training-procedure.png` | [V-JEPA](https://arxiv.org/abs/2404.08471) Figure 3 | 说明视频 latent prediction 和时空状态学习。 | 不把 action-free latent prediction 写成完整 action-conditioned world model。 |

## 来源

| 来源 | 类型 | 本站使用方式 | 关联主题 |
| --- | --- | --- | --- |
| [CLIP](https://arxiv.org/abs/2103.00020) | 论文 / 图源 | 使用 Figure 1 说明图文对比学习和 zero-shot 分类；强调静态语义对齐边界。 | 图文对齐 |
| [OpenAI CLIP blog](https://openai.com/research/clip) | 官方博客 | 补充 CLIP 如何用自然语言引用视觉概念。 | 图文对齐 |
| [The Illustrated CLIP](https://jalammar.github.io/illustrated-clip/) | 可视化博客 | 借鉴相似度矩阵和 prompt classifier 的图解讲法。 | 中文/可视化讲法 |
| [Hugging Face: Vision Language Models Explained](https://huggingface.co/blog/vlms) | 技术博客 | 支撑 vision encoder、modality projector、text decoder / LLM 三段式讲解。 | VLM 总览 |
| [BLIP-2](https://arxiv.org/abs/2301.12597) | 论文 / 图源 | 使用 Q-Former objectives 图说明 connector 是信息瓶颈。 | 连接器 |
| [Flamingo](https://arxiv.org/abs/2204.14198) | 论文 | 支撑 Perceiver Resampler、gated cross-attention 和 interleaved visual-text context。 | Cross-attention |
| [LLaVA](https://llava-vl.github.io/) | 项目页 / 论文 | 支撑 projector + visual instruction tuning 路线。 | 指令微调 |
| [V-JEPA](https://arxiv.org/abs/2404.08471) | 论文 / 图源 | 使用 training procedure 图说明 latent video prediction 和视频状态学习。 | 视频状态 |
| [V-JEPA 2](https://arxiv.org/abs/2506.09985) | 论文 | 补充从视频表征到 action-conditioned planning 的边界。 | 世界模型 |
| [RT-2](https://arxiv.org/abs/2307.15818) | 论文 | 支撑 VLM 到 action token / VLA 的接口提醒。 | VLA |
| [OpenBMB BLIP-2 论文速读](https://hub.baai.ac.cn/view/25807) | 中文公众号转载 | 借鉴“问题背景 -> 方法结构 -> 训练阶段 -> 结果局限”的讲法。 | 中文讲解 |
| [智源社区 LLaVA 条目](https://hub.baai.ac.cn/view/25500) | 中文资料页 | 核对 LLaVA 的视觉指令微调表达和原始项目链接。 | 中文讲解 |
| [AI Insight 多模态统一架构综述](https://www.ai-insight.org/reports/custom_multimodal-unified-architecture-2026) | 中文报告 | 借鉴从 CLIP 到 VLM/VLA 路线图的报告式组织。 | 路线背景 |
