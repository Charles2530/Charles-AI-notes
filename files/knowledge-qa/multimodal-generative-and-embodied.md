# 多模态与生成模型 QA

这一页回答 VLM、视觉连接器、扩散模型、视频生成和多模态评测中的高频问题。世界模型和具身智能已经拆到独立页面：读世界模型请看 [世界模型 QA](world-models.md)，读 VLA 与机器人闭环请看 [具身智能与 VLA QA](embodied-ai-and-vla.md)。

## VLM 与多模态表示

### Q：VLM 和纯文本 LLM 的核心差别是什么？

**面试回答。** 纯文本 LLM 的输入主要是 token 序列；VLM 需要先把图像、视频或其他模态编码成视觉 token 或 embedding，再通过 projector、resampler、cross-attention 等连接器送入语言模型。典型链路是：image/video -> vision encoder -> visual tokens -> connector -> LLM。

**追问展开。** 面试里要强调多模态难点不只是“多一个 encoder”，而是不同模态的粒度、坐标、时间和信息密度不同。图像 token 可能代表 patch，文本 token 代表子词，二者对齐并不天然。  
**易错点 / 边界。** 能回答图片问题不等于有精确 OCR、定位、几何、时序或动作能力；这些要分任务评测。  
**继续读。** [VLM 架构：视觉表征、连接器与记忆](../vlm/architecture-and-training.md) / [视觉 Tokenizer 与连接器](../vlm/visual-tokenizers-and-connectors.md)

### Q：CLIP 式对比学习学到了什么？没学到什么？

**面试回答。** CLIP 用图文对比学习把匹配图像和文本拉近，把不匹配样本拉远。常见目标是 batch 内 contrastive loss：

\[
\mathcal{L}=-\log\frac{\exp(\operatorname{sim}(I,T^+)/\tau)}
{\sum_j \exp(\operatorname{sim}(I,T_j)/\tau)}
\]

它学到的是开放词汇语义对齐和图文检索表示。  
**追问展开。** CLIP 强在“这张图和哪段文本更匹配”，所以适合 zero-shot 分类、检索和语义初始化。  
**易错点 / 边界。** CLIP 不直接学习像素级定位、3D 几何、动作后果或长视频因果；相似度高不等于机器人能执行。  
**继续读。** [VLM 架构：视觉表征、连接器与记忆](../vlm/architecture-and-training.md) / [CLIP 论文](https://arxiv.org/abs/2103.00020)

### Q：视觉 tokenizer 和 connector 各自做什么？

**面试回答。** 视觉 tokenizer/encoder 把图像或视频压成视觉 token，决定视觉信息保留的粒度；connector 把这些视觉 token 映射到 LLM 可读的表示空间，决定语言模型如何读取视觉证据。一个常见结构是 ViT encoder + linear projector 或 Q-Former。

**追问展开。** 面试时可以说：视觉 token 太多会占上下文和计算，太少会丢细节；connector 太弱会导致视觉信息进不去，太强又可能训练成本高。  
**易错点 / 边界。** 不要把视觉 encoder 的 ImageNet 或 CLIP 分数直接等同于 VLM 最终能力；连接方式和指令数据同样关键。  
**继续读。** [视觉 Tokenizer 与连接器](../vlm/visual-tokenizers-and-connectors.md) / [BLIP-2](https://arxiv.org/abs/2301.12597)

### Q：Projector、Q-Former 和 resampler 的差别是什么？

**面试回答。** Projector 通常是线性层或 MLP，把视觉特征直接映射到 LLM hidden size；Q-Former 用一组 learnable query 通过 cross-attention 从视觉特征中抽取少量证据；resampler 更泛化，目标是把大量视觉 token 压成固定或更小数量的 tokens。三者都在解决同一个问题：视觉信息太多，LLM 不能无成本全读。

**追问展开。** Projector 简单便宜，但压缩和选择能力有限；Q-Former / Perceiver resampler 可以主动查询视觉证据，更适合控制 token 数和跨模态对齐。回答时要把 connector 说成“证据选择和压缩层”，不是普通维度变换。  
**易错点 / 边界。** 更强 connector 不一定带来更强 VLM；如果视觉 encoder、训练数据和指令任务不匹配，connector 也只能压缩错误证据。  
**继续读。** [视觉 Tokenizer 与连接器](../vlm/visual-tokenizers-and-connectors.md) / [多模态推理基础](../foundations/multimodal-cot-and-reasoning-basics.md)

### Q：为什么多模态模型需要控制视觉 token 数？

**面试回答。** 图像和视频 token 密度远高于文本。如果把所有 patch、帧和相机视角都原样放进上下文，attention、KV cache、显存和延迟会迅速爆炸。视觉 token 控制本质上是在质量和成本之间做预算分配。

\[
L_{\text{vision}}=\text{frames}\times\text{patches per frame}\times\text{views}
\]

**追问展开。** 常见做法包括降采样、patch merging、resampler、局部 attention、关键帧选择、动态 token pruning 和多尺度表示。面试时可以补一句：省 token 不能只看平均 VQA，必须看 OCR、小目标、空间关系和长视频记忆是否受损。  
**易错点 / 边界。** token 越少不一定越好；压缩可能优先丢掉细节，而这些细节正是机器人、OCR、图表和医学影像任务需要的。  
**继续读。** [多模态 CoT 与推理基础](../foundations/multimodal-cot-and-reasoning-basics.md) / [数值、显存与运行时基础](../foundations/numerics-memory-and-runtime-basics.md)

### Q：VLM 的空间理解为什么难？

**面试回答。** 文本 token 序列没有天然 2D/3D 几何结构，普通视觉 encoder 也未必保留相机内参、深度、尺度、遮挡和物体关系。空间理解需要模型知道“在哪里”“多远”“相对于谁”，这比识别“是什么”更难。

**追问展开。** 如果任务是机器人抓取、导航或测距，模型需要 depth、pose、point cloud、cost volume 或显式 3D 表示辅助。只靠图文对齐很难稳定处理精确几何。  
**易错点 / 边界。** VQA 答对“桌上有杯子”不代表能判断杯子可抓取姿态。语义识别和几何可执行性要分开。  
**继续读。** [相机、深度与机器人视觉](../embodied-ai/cameras-depth-and-robot-vision.md) / [双目匹配与 Cost Volume](../embodied-ai/stereo-matching-and-cost-volume.md)

### Q：多模态评测为什么不能只看 VQA 分数？

**面试回答。** VQA 主要测图文问答，不能覆盖 OCR、grounding、空间关系、视频记忆、时序因果、幻觉、安全、工具调用和机器人动作。多模态系统要按能力拆分评测，而不是用一个综合榜单替代所有结论。

**追问展开。** 面试里可以举例：模型可能 VQA 分很高，但在小字 OCR、计数、坐标定位或多帧追踪上失败。评测应该包含任务分桶和失败样本分析。  
**易错点 / 边界。** 榜单分数高只能说明该 benchmark 设置下表现好，不能证明真实应用可靠。  
**继续读。** [多模态评估与失败模式](../vlm/multimodal-evaluation-and-failure-modes.md) / [GPT-4o System Card：Omni 安全与多模态评测](../paper-deep-dives/technical-reports/gpt-4o-system-card.md)

### Q：视频 VLM 为什么比单图 VLM 更难？

**面试回答。** 视频多了时间维度，模型要处理帧间运动、事件顺序、物体身份保持和长期记忆。单图 VLM 只需要解释一个静态状态，视频 VLM 还要回答“什么时候发生”“先后关系是什么”“对象是否持续存在”等问题。

**追问展开。** 工程上会遇到帧采样、视频 token 数爆炸、长上下文记忆和跨帧 grounding 问题。常见策略是抽帧、时序池化、视频 tokenizer、memory bank 或分层摘要。  
**易错点 / 边界。** 抽几帧能答题不等于模型理解连续运动；长视频评测要看时序、定位和事件级失败。  
**继续读。** [视频表示与记忆](../vlm/video-representation-and-memory.md) / [多模态评估与失败模式](../vlm/multimodal-evaluation-and-failure-modes.md)

### Q：多模态 CoT 为什么容易被误解？

**面试回答。** 多模态 CoT 的目标是让模型把视觉证据、文本问题和中间推理组织起来，例如先定位物体，再比较关系，再回答。它可以改善可解释性和复杂任务表现，但推理文本本身不保证真实。

**追问展开。** 面试时要区分“模型真的使用了视觉证据”和“生成了一段看起来合理的解释”。更稳的做法是结合 grounding、引用框、可视化注意力或工具验证。  
**易错点 / 边界。** 解释流畅不代表视觉证据正确；多模态 hallucination 常常来自模型语言先验压过图像证据。  
**继续读。** [多模态 CoT 与推理基础](../foundations/multimodal-cot-and-reasoning-basics.md) / [多模态推理来源台账](../references/multimodal-reasoning-source-ledger.md)

### Q：OCR、图表理解和 grounding 为什么常常比普通 VQA 难？

**面试回答。** 普通 VQA 很多问题可以靠图像整体语义和语言先验回答；OCR、图表和 grounding 需要精确读取局部证据、坐标关系和细粒度对象。它们要求模型保留高分辨率细节，并把答案绑定到图像区域或文本位置。

**追问展开。** OCR 受分辨率、字体、旋转和压缩影响；图表理解还要解析坐标轴、图例和数值；grounding 要把语言短语对应到具体 box、mask 或点。回答时要强调“看懂大意”和“定位证据”是两种能力。  
**易错点 / 边界。** 多模态模型解释得像真的，不代表它真的读到了字或定位到了目标；最好用可视化引用、区域标注或可执行工具验证。  
**继续读。** [多模态评估与失败模式](../vlm/multimodal-evaluation-and-failure-modes.md) / [VLM 架构：视觉表征、连接器与记忆](../vlm/architecture-and-training.md)

## 扩散与生成模型

### Q：扩散模型为什么要加噪再去噪？

**面试回答。** 扩散模型先定义一个前向加噪过程，把真实数据逐步变成接近高斯噪声；再训练模型学习反向去噪，从噪声恢复数据。DDPM 中常见前向形式是：

\[
q(x_t|x_0)=\mathcal{N}(\sqrt{\bar{\alpha}_t}x_0,(1-\bar{\alpha}_t)I)
\]

模型通常预测噪声 \(\epsilon\)、score 或 velocity。  
**追问展开。** 面试时要说清：加噪过程是人为定义的训练桥梁，去噪网络学的是每个噪声水平下如何往数据分布方向走。  
**易错点 / 边界。** 去噪能力不等于理解物理因果；图像/视频自然不代表可以做规划。  
**继续读。** [扩散训练与表示](../diffusion/training.md) / [DDPM](https://arxiv.org/abs/2006.11239)

### Q：Latent Diffusion 为什么要在 latent space 里生成？

**面试回答。** Latent Diffusion 先用 VAE 或 tokenizer 把高分辨率图像压到 latent，再在 latent 上做扩散采样，最后解码回像素。这样可以大幅降低空间分辨率和计算成本，让 U-Net 或 DiT 在更小的张量上建模。

**追问展开。** 它省的是采样过程中的 FLOPs、显存和带宽，但代价是生成质量受 VAE 压缩上限影响。细小文字、纹理、几何边界和视频时序都可能被 latent 表示限制。  
**易错点 / 边界。** 在 latent 里生成不代表模型学到了语义 latent；很多 latent 仍是空间网格，只是比像素更紧凑。  
**继续读。** [扩散训练与表示](../diffusion/training.md) / [视频与多模态扩散](../diffusion/video-and-multimodal-diffusion.md)

### Q：DDPM、Score SDE 和 Flow Matching 的关系怎么理解？

**面试回答。** 它们都在学习从简单分布到数据分布的生成路径，只是数学表述不同。DDPM 是离散时间加噪/去噪，Score SDE 用连续随机微分方程描述噪声过程，Flow Matching 学习把噪声样本沿确定性向量场搬到数据样本。

**追问展开。** 可以把问题统一成“学一个方向场”：score 指向高密度方向，velocity/flow 指向从源分布到目标分布的路径。采样器则是在这个场上数值积分。  
**易错点 / 边界。** 不要只背方法名。真正影响工程的是参数化、噪声日程、solver、训练稳定和采样步数。  
**继续读。** [Score Matching 到 SDE](../diffusion/score-matching-sde-and-probability-flow.md) / [一致性模型与 Rectified Flow](../diffusion/consistency-models-and-rectified-flow.md)

### Q：Consistency Model、Rectified Flow 和普通 diffusion 的核心区别是什么？

**面试回答。** 普通 diffusion 通常依赖多步去噪；Consistency Model 试图让模型在不同噪声点上直接映射到一致的干净样本或同一轨迹端点；Rectified Flow 则把生成看成从噪声到数据的连续直线路径或速度场学习。它们都想减少采样步数，但约束方式不同。

**追问展开。** 可以这样区分：diffusion 学多噪声水平的去噪方向，consistency 强调同一轨迹不同时间的一致输出，rectified flow 强调速度场和路径拉直。工程上要看 NFE、训练稳定、样本质量和是否需要蒸馏。  
**易错点 / 边界。** 少步或一步生成不是免费午餐；步数少了，模型每一步承担的分布搬运更大，失败可能表现为细节丢失、结构漂移或时序不稳。  
**继续读。** [一致性模型与 Rectified Flow](../diffusion/consistency-models-and-rectified-flow.md) / [一步生成、蒸馏与整流](../diffusion/distillation.md)

### Q：Classifier-free guidance 为什么会提升条件一致性？

**面试回答。** CFG 同时训练有条件和无条件预测，采样时把方向推向更符合条件的区域。常见写法是：

\[
\hat{\epsilon}=\epsilon_{\text{uncond}}+
s(\epsilon_{\text{cond}}-\epsilon_{\text{uncond}})
\]

其中 \(s\) 是 guidance scale。  
**追问展开。** \(s\) 越大，模型越强调条件，例如文本 prompt；但过大可能牺牲多样性或造成过饱和。  
**易错点 / 边界。** CFG 不是免费增强质量。它提高条件一致性的同时可能降低自然度、多样性和稳定性。  
**继续读。** [条件控制与 Guidance](../diffusion/guidance-and-conditioning.md) / [噪声日程与参数化](../diffusion/noise-schedules-and-parameterization.md)

### Q：少步蒸馏为什么能加速扩散采样？

**面试回答。** 原始扩散模型可能需要几十到上百步采样，少步蒸馏把多步 teacher 的生成路径压缩到少量 step 的 student 中。DMD、consistency model、rectified flow 等路线都在降低 NFE，让生成更快。

**追问展开。** 面试里要说清 tradeoff：少步生成减少延迟，但 student 要学会跨更大噪声区间的变化，训练更难，也可能损失细节和多样性。  
**易错点 / 边界。** 少步指标好不等于所有场景好；视频生成还要看时序一致性，世界模型还要看动作敏感和闭环收益。  
**继续读。** [一步生成、蒸馏与整流](../diffusion/distillation.md) / [DMD：一步扩散蒸馏](../paper-deep-dives/diffusion/dmd.md)

### Q：扩散蒸馏里的 teacher/student 到底对齐什么？

**面试回答。** 扩散蒸馏通常让少步 student 对齐多步 teacher 的生成行为。对齐对象可能是最终样本、噪声预测、score / velocity、分布统计、判别器反馈或 teacher 轨迹中的中间状态。不同方法的关键差别就在“让 student 学什么信号”。

**追问展开。** Progressive distillation 更像逐步缩短采样链；consistency / LCM 强调跨时间一致；DMD 类方法用 distribution matching 或判别信号让 student 少步接近 teacher 分布。回答时不要只说“压缩步数”，要说清监督信号来自哪里。  
**易错点 / 边界。** Student 在 benchmark 上接近 teacher，不代表所有 prompt、视频长度、motion 和安全场景都无损；蒸馏常会牺牲多样性或长尾细节。  
**继续读。** [一步生成、蒸馏与整流](../diffusion/distillation.md) / [DMD2：更稳的少步分布匹配蒸馏](../paper-deep-dives/diffusion/dmd2.md)

### Q：视频生成和图像生成最大的新增困难是什么？

**面试回答。** 视频比图像多了时间维度。模型不仅要生成单帧质量，还要保持物体身份、运动连续性、相机轨迹、遮挡关系和长程一致性。一个视频看起来好，意味着空间质量和时间质量都要过关。

**追问展开。** 训练上会引入 3D attention、temporal modules、video DiT、latent video tokenizer 或分层生成；评测上要看 temporal consistency、motion quality 和 failure cases。  
**易错点 / 边界。** 视频自然度不是世界模型能力。没有动作输入和闭环验证时，它仍只是 open-loop 生成。  
**继续读。** [视频与多模态扩散](../diffusion/video-and-multimodal-diffusion.md) / [扩散训练配方与失败分析](../diffusion/practical-training-recipes-and-failure-analysis.md)

### Q：视频扩散为什么特别关心时序一致性和 motion prior？

**面试回答。** 图像生成只要单帧合理，视频生成还要让相邻帧和长时间跨度都合理。同一个物体不能突然变形、消失或身份漂移，运动速度、相机轨迹和遮挡也要符合物理直觉。Motion prior 是模型对“合理运动模式”的先验。

**追问展开。** 时序一致性可以来自 3D/temporal attention、光流或轨迹约束、视频 tokenizer、分层生成、前一帧条件和专门的视频数据。面试时要强调：视频生成失败常不是单帧画质差，而是时间上不守恒。  
**易错点 / 边界。** 视频连贯不等于可交互世界模型；没有动作条件和反事实验证时，模型可能只是生成自然视频先验。  
**继续读。** [视频与多模态扩散](../diffusion/video-and-multimodal-diffusion.md) / [CausVid：流式自回归视频扩散](../paper-deep-dives/diffusion/causvid.md)

### Q：DiT 和 U-Net 扩散骨干有什么取舍？

**面试回答。** U-Net 通过卷积和多尺度 skip connection 处理图像局部结构，早期扩散模型常用；DiT 把 latent patch 当 token，用 Transformer 建模全局依赖，更容易随数据和算力扩展。现代视频和多模态生成常偏向 DiT 或混合结构。

**追问展开。** DiT 的优势是统一 token 建模和 scaling，缺点是 attention 成本高；U-Net 局部归纳偏置强、工程成熟，但全局建模和多模态扩展不如 Transformer 自然。  
**易错点 / 边界。** 不能简单说 DiT 全面替代 U-Net；小模型、小数据或强局部结构任务里 U-Net 仍可能更经济。  
**继续读。** [扩散训练与表示](../diffusion/training.md) / [视频与多模态扩散](../diffusion/video-and-multimodal-diffusion.md)

### Q：生成模型评估为什么很难？

**面试回答。** 生成模型输出不是单一正确答案，而是分布。图像和视频要同时看质量、多样性、条件一致性、时序稳定、审美、人类偏好和安全。单个指标很难覆盖这些维度。

**追问展开。** FID、FVD、CLIP score、人类偏好、VBench 等指标各自只覆盖一部分。生成模型论文通常需要定量、定性、消融和失败案例一起看。  
**易错点 / 边界。** 不要把单个分数写成“模型更好”的充分证据；生成质量和下游可用性也不是一回事。  
**继续读。** [扩散路线选择与方法对照](../diffusion/comparison-table.md) / [全站证据与复现状态标准](../references/evidence-and-reproducibility-standard.md)
