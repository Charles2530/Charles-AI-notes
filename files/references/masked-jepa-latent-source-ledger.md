# Masked / JEPA 与潜变量预测来源台账

这份台账服务于 [Masked / JEPA 与潜变量预测](../world-models/masked-jepa-and-latent-prediction.md)。正文的目标不是把 MAE、MWM、I-JEPA、V-JEPA、V-JEPA 2 写成模型名清单，而是把它们放到同一条接口线上：先问预测目标保留了什么状态，再问动作、目标、风险和闭环证据怎样接上去。

## 使用方式

第一类来源是 masked reconstruction。MAE 用于说明“遮住大部分 patch，再重建像素”为什么是强自监督信号；正文同时强调 pixel reconstruction 只是在塑造表征，不能直接等价为可规划状态。

第二类来源是 visual control bridge。MWM 用于连接 masked representation 和 model-based RL：视觉状态可以先学，latent dynamics 再在表示空间里接动作。正文使用 MWM 时不外推成“所有 masked model 都优于 RSSM”，只保留论文设置下 decoupled representation + dynamics 的证据。

第三类来源是 JEPA 主线。AMI position paper、I-JEPA 和 V-JEPA 用于说明 joint embedding prediction 的目标：预测目标区域或未来区域的表示，而不是生成目标像素。正文使用这类来源时必须写清：原始 JEPA / V-JEPA 多数是 action-free representation learning，不是完整 action-conditioned planner。

第四类来源是 V-JEPA 2 及 2026 后续。V-JEPA 2 research page 和论文支撑 action-free 预训练到 V-JEPA 2-AC latent planning 的两阶段讲法；V-JEPA 2.1、VLA-JEPA 和 latent video prediction 评测论文只作为前沿进展提示，默认标为 Paper Only，不能写成跨平台真实部署结论。

第五类来源是高质量博客、官方博客、中文公众号和技术长文。它们主要用于学习讲解节奏：先给读者一个混淆点，再用接口、图和反例分层拆开。事实、图源、实验数字和强 claim 仍以论文、官方研究页、官方博客或项目页为准；版权不清或链接不稳定的公众号图不进入正文。

## 写作边界

不要把“预测 embedding”写成“已经理解物理世界”。embedding prediction 可以保留对象和运动线索，但它是否保留接触、风险、目标状态和动作敏感变量，需要下游评测证明。

不要把 action-free video pretraining 写成 robot planning。V-JEPA-style 模型可以是状态层候选；要规划真实动作，必须补 action-conditioned dynamics、goal / cost、planner、controller、真实观测刷新和安全边界。

不要只引用表征 benchmark。Kinetics、Something-Something、Epic-Kitchens、VQA 等能说明表示有信息；世界模型页面还要问 counterfactual action、goal-conditioned ranking、closed-loop success、failure replay 和 latency。

不要把前沿 2026 结果当作稳定工程结论。V-JEPA 2.1、VLA-JEPA 和 latent video prediction robustness 结果适合说明研究方向，但本站默认保守处理复现、平台泛化和部署安全。

## 图片使用

本页没有使用 image2，也没有生成新图片；正文只复用了仓库里已经存在的论文图，并在图注中保留图源和读图方法。

| 正文图片 | 本地路径 | 原始来源 | 本站用途 |
| --- | --- | --- | --- |
| MWM framework | `../assets/images/paper-deep-dives/world-models/masked-world-models/figure-1-mwm-framework.png` | [Masked World Models for Visual Control](https://arxiv.org/abs/2206.14244) Figure 1 | 说明 masked visual representation 与 latent dynamics 解耦。 |
| V-JEPA training procedure | `../assets/images/paper-deep-dives/world-models/v-jepa/figure-3-training-procedure.png` | [V-JEPA](https://arxiv.org/abs/2404.08471) Figure 3 | 说明 context encoder、target encoder 和 predictor 的 joint embedding prediction。 |

## 来源

| 来源 | 类型 | 本站使用方式 | 关联问题 |
| --- | --- | --- | --- |
| [MAE](https://arxiv.org/abs/2111.06377) | 论文 | 支撑 masked pixel reconstruction、高比例 mask 和非对称 encoder-decoder 的基础讲法。 | masked reconstruction |
| [Masked World Models for Visual Control](https://arxiv.org/abs/2206.14244) | 论文 / 图源 | 使用 Figure 1 说明 masked visual autoencoder 与 latent dynamics 解耦。 | MWM |
| [A Path Towards Autonomous Machine Intelligence](https://openreview.net/pdf?id=BZ5a1r-kVsf) | position paper / 路线图 | 支撑 JEPA、world model、actor、cost 和 planner 在 agent 架构中的分工。 | JEPA 路线 |
| [I-JEPA](https://arxiv.org/abs/2301.08243) | 论文 | 支撑从 context block 预测 target block representation，而不是预测像素。 | image JEPA |
| [V-JEPA](https://arxiv.org/abs/2404.08471) | 论文 / 图源 | 使用 Figure 3 说明 latent video feature prediction；支撑 action-free 表征边界。 | video JEPA |
| [Meta V-JEPA blog](https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/) | 官方博客 | 学习官方如何把 V-JEPA 解释成通向 advanced machine intelligence 的表征路线。 | 官方讲法 |
| [V-JEPA 2 research page](https://ai.meta.com/research/publications/v-jepa-2-self-supervised-video-models-enable-understanding-prediction-and-planning/) | 官方研究页 / 图源 | 支撑 action-free 预训练、少量机器人视频 post-training 和 V-JEPA 2-AC zero-shot planning 的官方表述。 | V-JEPA 2-AC |
| [V-JEPA 2 paper](https://arxiv.org/abs/2506.09985) | 论文 | 支撑 candidate action sequence、goal representation、MPC 和 robot planning 证据；Figure 7 作为正文讲法参考，但未在正文复用。 | latent planning |
| [V-JEPA 2.1](https://arxiv.org/abs/2603.14482) | 论文 | 记录 2026 年 dense predictive loss、deep self-supervision、image/video tokenizer 和 real-robot grasping 结果；默认 Paper Only。 | 前沿进展 |
| [Latent Video Prediction Learns Better World Models](https://arxiv.org/abs/2605.15618) | 论文 | 记录 latent-prediction 模型在 corruption、occlusion、fine-grained contact 和 temporal direction 上的系统评测；默认 Paper Only。 | 鲁棒性评测 |
| [VLA-JEPA](https://arxiv.org/abs/2602.10098) | 论文 | 作为 VLA 方向的前沿参考，说明 leakage-free latent state prediction 如何接 action-head fine-tuning；默认 Paper Only。 | VLA / world model |
| [全站外部阅读台账](external-reading-ledger.md) | 站内台账 | 记录高质量博客、官方长文、中文公众号和技术媒体的讲法学习边界。 | 写作方法 |
| [机器之心](https://www.jiqizhixin.com/) | 中文技术媒体 / 公众号生态 | 学习“背景 -> 方法 -> 实验 -> 局限”的中文论文讲解节奏，不作为强事实唯一来源。 | 中文讲法 |
| [量子位](https://www.qbitai.com/) | 中文技术媒体 / 公众号生态 | 学习从产业问题和路线对比切入的写法，事实回原论文或官方材料核对。 | 中文讲法 |
| [智源社区](https://hub.baai.ac.cn/) | 中文社区 / 编译转载 | 学习技术长文和编译材料如何组织中文背景，转载内容追到原文。 | 中文讲法 |
| [PaperWeekly](https://zhuanlan.zhihu.com/paperweekly) | 中文论文解读 / 公众号生态 | 学习论文速读如何把贡献、方法图和实验表拆成读者能跟上的顺序；图和数字仍回原文。 | 中文讲法 |
| [Datawhale 教程矩阵](https://github.com/datawhalechina/so-large-lm) | 中文开源教程 | 学习把概念、公式、代码和学习路径放在同一页的组织方式。 | 中文讲法 |
| [colah: Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) | 经典技术博客 | 学习用一个核心直觉贯穿长文，而不是把公式散放成符号表。 | 博客讲法 |
| [Distill: Feature Visualization](https://distill.pub/2017/feature-visualization/) | 可视化技术博客 | 学习图如何成为论证对象；每张图都要说明读者该看什么。 | 博客讲法 |

## 后续写作提醒

维护这页时，每新增一个 masked / JEPA 工作，都要补四个槽位：预测对象是什么，mask 或 target 怎样构造，动作是否进入模型，下游证据停在哪一层。只写“学到强表征”太宽；本站需要说明这个强表征能否被 dynamics、planner、risk head 或真实闭环消费。
