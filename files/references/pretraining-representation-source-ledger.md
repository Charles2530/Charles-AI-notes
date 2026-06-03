# 预训练与表示学习来源台账

这页记录预训练目标、表示学习、对比学习、masked modeling、denoising 和行为克隆相关来源。它服务的写作原则是：不要把 loss 名字写成知识点，要说明这个目标让模型保留什么信息、忽略什么信息，以及它和最终任务接口是否一致。

## 使用方式

写预训练目标文章时，先把来源分成四类。

第一类是语言和序列建模。BERT、GPT/next-token、Decision Transformer 这类来源帮助本站解释“预测目标如何把数据变成序列接口”。正文重点不是复述公式，而是说明 teacher forcing、masked prediction、return conditioning 或动作 token 会让模型在哪些状态分布上学习。

第二类是视觉和多模态表征。MAE、SimCLR、CLIP、V-JEPA 这类来源帮助本站解释“表示为什么会保留某些不变性”。对比学习里的正负样本、图像增强、图文配对和 mask 策略，会直接决定表示是否保留小物体、文字、空间位置、时序和动作相关信息。

第三类是生成建模目标。VAE、DDPM、Score SDE 和 latent diffusion 帮助本站说明“重建、潜变量和去噪目标不是同一回事”。正文要区分 pixel reconstruction、latent reconstruction、noise prediction、score matching 和采样器，不把它们笼统写成生成模型 loss。

第四类是行为和控制目标。RT-1、RT-2、Decision Transformer、Diffusion Policy 和 Open X-Embodiment 帮助本站说明“模仿专家动作”和“会闭环恢复”之间的距离。行为克隆可以扩大监督学习规模，但分布外恢复、探索、风险和长时规划需要额外机制。

## 写作边界

预训练指标不能直接等价为下游能力。CLIP zero-shot 强，不代表会预测动作后果；MAE 重建好，不代表表示可规划；V-JEPA latent prediction 好，不代表已经是 action-conditioned world model；BC loss 低，不代表部署时偏离轨迹还能恢复；扩散样本好，不代表少步实时控制稳定。

正文里可以用博客学习讲法，但关键机制和图优先回论文、官方项目页或官方博客。中文材料适合学习读者入口和术语表达；性能、实验设置和强 claim 仍回原始来源。

## 文章模板

每个目标函数小节建议按同一条线写：先说模型看到什么，再说它预测什么；接着说明这个预测目标奖励哪些信息，允许丢掉哪些信息；最后说明下游任务如果需要额外变量，应该接什么机制。

一个好段落应该能回答：目标变量是什么、正负样本或 mask 怎么构造、模型有没有捷径、表示会形成什么不变性、评测指标是否覆盖最终接口。一个弱段落通常只写“某方法提升表征质量”，却没有说清这个质量是检索、生成、重建、预测未来还是控制成功。

## 来源

| 来源 | 类型 | 本站学什么 | 落到页面 |
| --- | --- | --- | --- |
| [BERT](https://arxiv.org/abs/1810.04805) | 论文 | masked language modeling 和 next sentence prediction 如何塑造双向文本表征。 | 预训练目标、masked modeling |
| [MAE](https://arxiv.org/abs/2111.06377) | 论文 | 高比例 mask、非对称 encoder/decoder 和像素重建如何训练视觉表征。 | 预训练目标、视觉表征 |
| [SimCLR](https://arxiv.org/abs/2002.05709) | 论文 | 数据增强、batch negatives 和 InfoNCE 如何定义视觉不变性。 | 对比学习、表示学习 |
| [CLIP](https://arxiv.org/abs/2103.00020) | 论文 | 图文配对和批内对比如何形成开放类别语义接口。 | 预训练目标、VLM |
| [The Illustrated CLIP](https://jalammar.github.io/illustrated-clip/) | 可视化博客 | 用相似度矩阵和 zero-shot classifier 讲清 CLIP 的直觉。 | 预训练目标、VLM 入门 |
| [V-JEPA](https://arxiv.org/abs/2404.08471) | 论文 | latent video prediction 如何避免逐像素重建，并形成视频表征。 | 预训练目标、世界模型 |
| [A Path Towards Autonomous Machine Intelligence](https://openreview.net/forum?id=BZ5a1r-kVsf) | 路线论文 | JEPA、world model、actor 和 cost 模块如何在 agent 架构里分工。 | JEPA、世界模型边界 |
| [DDPM](https://arxiv.org/abs/2006.11239) | 论文 | 前向加噪与反向去噪如何把生成建模变成 denoising objective。 | 预训练目标、扩散训练 |
| [Score SDE](https://arxiv.org/abs/2011.13456) | 论文 | score field、reverse SDE 和 probability flow ODE 如何统一扩散采样。 | 扩散目标、score matching |
| [RT-1](https://robotics-transformer1.github.io/) | 项目页 / 论文 | 大规模机器人示范如何被组织成动作 token 预测和多任务策略学习。 | behavior cloning、VLA |
| [Decision Transformer](https://arxiv.org/abs/2106.01345) | 论文 | 把 offline RL 轨迹当序列建模，说明 return conditioning 和行为数据分布的关系。 | behavior cloning、RL 序列建模 |
