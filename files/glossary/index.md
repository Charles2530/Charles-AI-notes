# 术语表与易混概念

这里用短定义帮助查词。遇到同一个缩写时，先确认它属于数据、目标、模型、数值还是运行时；再沿正文链接看计算和适用条件。一个术语可能在不同论文中有不同具体实现。

## 先分清这些容易混用的对象

| 容易混淆 | 区别 | 展开 |
| --- | --- | --- |
| logit 与概率 | logit是未归一化分数；softmax或sigmoid之后才是所定义的概率 | [分类器算例](../foundations/symbols-and-minimal-math.md) |
| likelihood 与 posterior | 前者把参数或隐藏解释作为可变对象看观测概率；后者在给定观测后对隐藏对象归一化 | [潜变量与下界](../prerequisite-math/latent-likelihood-and-generative-objectives.md) |
| attention mask 与 loss mask | 前者决定读取关系，后者决定哪些目标计入损失 | [打包与Mask](../foundations/positional-encoding-masks-and-context.md) |
| latent、KV与递推状态 | 都是中间表示，但来源、保存时间与用途不同 | [Mamba与Attention](../foundations/mamba-and-hybrid-ssm-transformers.md) |
| 参数、FLOPs与延迟 | 参数量计存储对象，FLOPs计运算，延迟还取决于搬运、并行和调度 | [数值与运行成本](../foundations/numerics-memory-and-runtime-basics.md) |
| confidence 与成功率 | 前者是模型给出的分数或概率，后者是指定样本上的事件频率 | [概率校准](../prerequisite-math/sampling-estimation-and-calibration.md) |
| 仿真与反事实 | 仿真执行模型设定；对现实的反事实判断还需要因果假设与验证 | [模拟与反事实](../world-models/simulators-counterfactuals-and-synthetic-rollouts.md) |

## 数据、泛化与生成基础

| 术语 | 含义 |
| --- | --- |
| `Autoregressive Generation` | 自回归生成，每一步根据已有前缀预测下一个 token，并把生成结果追加回上下文。 |
| `Beam Search` | 解码时保留多条高概率候选序列并继续扩展，常用于翻译等结构较固定任务；开放式生成中可能更模板化。 |
| `Behavior Cloning / BC` | 行为克隆，用专家示范数据训练策略模仿动作，常写成 \(-\log \pi_\theta(a\mid o,l)\)。 |
| `Contrastive Learning` | 对比学习，让匹配样本表示更近、不匹配样本更远；CLIP 和 InfoNCE 是常见代表。 |
| `Data Leakage` | 数据泄漏，训练或调参过程看到了本应保留给验证、测试或线上判断的数据。 |
| `Dataset` | 数据集，不只是文件集合，还包括来源、样本定义、标签规则、split、metadata 和版本。 |
| `Deduplication` | 去重，移除完全重复或近重复样本，降低训练偏置和 train/test 污染风险。 |
| `Distribution Shift / OOD` | 分布偏移或分布外输入，指训练分布与测试、部署或真实使用分布不一致。 |
| `Greedy Decoding` | 贪心解码，每步选择当前概率最大的 token；不保证整条序列概率最大，也不包含采样。并列分数与底层数值实现仍可能影响输出。 |
| `Label` | 标签或监督信号，可以是类别、下一个 token、动作、reward、偏好排序或伪标签。 |
| `Logits` | softmax 前的未归一化分数。解码、分类和蒸馏通常先看 logits，再看概率。 |
| `Masked Modeling` | 遮挡建模，把输入一部分 mask 掉，让模型预测被遮挡的 token、像素或 latent 表示。 |
| `Next-token Prediction` | 下一个 token 预测，自回归语言模型的核心预训练目标。 |
| `Sample` | 数据样本，训练和评测的最小语义单位，可以是一段文本、一张图、一个 clip 或一条机器人轨迹。 |
| `Top-k Sampling` | 只在概率最高的 \(k\) 个 token 中采样。\(k\) 固定，可能不适应不同上下文的分布形状。 |
| `Top-p / Nucleus Sampling` | 选择累计概率达到 \(p\) 的最小 token 集合再采样，候选集合大小会随上下文动态变化。 |

## 训练与优化

| 术语 | 含义 |
| --- | --- |
| `Activation` | 神经网络中间层输出。量化时 activation 往往比权重更难压低比特，因为动态范围更不稳定。 |
| `AdamW` | Transformer 训练中常用优化器，把 weight decay 与梯度更新解耦。 |
| `DPO` | Direct Preference Optimization，常用偏好对齐方法，直接利用偏好对优化策略分布。 |
| `ELBO` | 证据下界，不超过观测的 log 边缘似然。内部 KL 比较近似后验与先验；下界间隙则是近似后验到真实后验的 KL，见[两种 KL](../prerequisite-math/latent-likelihood-and-generative-objectives.md)。 |
| `Feature Distillation` | 特征蒸馏，让 student 对齐 teacher 的中间 hidden states、feature maps 或 attention/relation 结构，而不只学最终输出。 |
| `GRPO` | Group Relative Policy Optimization，用同一输入的多次采样做组内相对奖励，常用于可验证奖励后训练。 |
| `Knowledge Distillation / KD` | 知识蒸馏，用 teacher model 的输出分布、中间表示、rationale 或生成轨迹训练 student model，常用于压缩、提速和行为迁移。 |
| `Logit Distillation` | Logit 蒸馏，让 student 对齐 teacher 的 pre-softmax logits 或 temperature softmax 分布，保留类别或 token 间相对偏好。 |
| `LoRA` | Low-Rank Adaptation，用低秩增量替代全量参数更新的参数高效微调方法。 |
| `NAS / Neural Architecture Search` | 神经架构搜索，在候选 layer、通道、kernel、head 或 block 组合中寻找质量和成本更好的结构；站内入门见 [模型压缩、剪枝与 NAS](../foundations/model-compression-pruning-and-nas-basics.md)。 |
| `PPO` | 一族限制策略更新幅度的策略优化方法。常用 clipped surrogate 截平部分有利方向的目标；它不硬性保证每个概率比或 KL 都处于给定区间，见[策略梯度与 PPO](../reinforcement-learning/policy-gradient-actor-critic-ppo-grpo.md)。 |
| `QAT` | Quantization-Aware Training，在训练中模拟量化误差，让模型适应低比特约束。 |
| `PTQ` | Post-Training Quantization，训练完成后再进行量化。 |
| `Rationale Distillation` | Rationale 蒸馏，用 teacher 生成的解释、CoT 或中间步骤作为额外监督，帮助 student 学任务过程而不只是最终标签。 |
| `RLHF` | Reinforcement Learning from Human Feedback，用人类偏好或 reward model 优化模型行为。 |
| `RLVR` | Reinforcement Learning with Verifiable Rewards，用规则或任务指标等可验证奖励优化模型。 |
| `SFT` | Supervised Fine-Tuning，用高质量示范数据教模型遵循任务格式和行为。 |
| `Soft Target` | teacher 产生的软目标分布，例如每个类别或 token 的概率；相比 one-hot hard label，它保留了相似类别之间的相对关系。 |
| `Structured Pruning` | 结构化剪枝，按通道、filter、head、block 或 layer 删除结构，使 dense kernel 能直接跑更小 shape，通常比单 weight 稀疏更容易兑现 latency 收益。 |
| `Teacher / Student` | 蒸馏语境里的教师模型和学生模型。teacher 通常能力更强但更慢或更贵，student 通过学习 teacher 信号换取低成本部署。 |
| `Temperature` | softmax 中控制分布尖锐程度的温度参数；蒸馏里较高 temperature 会让 teacher 分布更软，暴露更多非最大类信息。 |
| `Muon` | 面向矩阵权重的优化器路线：先积累 momentum，再用 Newton-Schulz 等方式近似正交化更新方向。 |
| `Distributed Muon` | Muon 在分布式训练中的实现形态，核心难点是正交化通常需要看到完整矩阵 momentum，而不是单个 optimizer shard。 |
| `Update RMS` | 参数更新张量的均方根尺度，用于比较不同参数组或形状的更新幅度。具体缩放规则是优化器实现的一部分。 |
| `Newton-Schulz` | 常用于矩阵逆平方根或正交化近似的迭代方法，Muon 用它把 momentum 映射到更均衡的矩阵更新方向。 |

## 长上下文与分布式 Attention

| 术语 | 含义 |
| --- | --- |
| `Context Parallelism / CP` | 把长序列上下文切到多个 GPU 或 ranks 上共同计算 attention 的并行方式，重点瓶颈通常在 KV 通信、负载均衡和 overlap。 |
| `GQA` | Grouped-Query Attention，多组 query heads 共享较少的 key/value heads。它能降低 KV 成本，但会让 head-sharding 方法更受可整除约束影响。 |
| `FlashAttention` | 分块计算 softmax attention，通过局部归约与重算减少显存读写。“exact”表示没有引入稀疏或低秩近似，不表示不同运算顺序下逐 bit 相同，见[Attention 与融合算子](../operators/gemm-attention-and-fused-kernels.md)。 |
| `FA3 / FA4` | FlashAttention 的不同实现代际。硬件、dtype、mask、前反向支持随版本变化，使用前应核对具体 release 与构建条件，见[长上下文与 FlashAttention](../operators/long-context-and-flashattention-evolution.md)。 |
| `MagiAttention` | SandAI 的分布式 attention 方案，用 FFA、dispatch solver、GroupCast/GroupReduce 和 overlap solver 支持超长上下文与异构 mask 训练。 |
| `AttnSlice` | MagiAttention 中的 mask 表达单元，形如 `(QRange, KRange, MaskType)`，把复杂 attention mask 拆成可调度的 2D 子块。 |
| `FFA` | Flex-Flash-Attention，MagiAttention 中基于 FlashAttention-3 思路扩展的灵活 mask attention kernel。 |
| `varlen` | Variable length 的缩写，指 batch 或 packed data 中样本长度不同。长上下文训练里 varlen 会显著影响 mask area 和 rank 负载。 |
| `Block-causal mask` | 以块表达可见性的attention mask，常允许块内特定交互、块间只读历史。块内是否双向、各块是否属于同一样本，需由任务定义，不能与所有packed mask等同。 |
| `Patch-and-Pack / PnP` | 把不同尺寸或不同长度样本切成 patch 后 pack 到长序列中的数据组织方式，可提升吞吐，但会制造更复杂的 attention mask。 |
| `Dispatch solver` | 根据 mask area 把 query chunks 分配到 CP ranks 的调度器，目标是让有效 attention 计算量尽量均衡。 |
| `CalcMeta / CommMeta` | MagiAttention 中的运行元数据：前者描述本 rank 要算哪些 AttnSlice，后者描述需要向哪些 ranks 收发 KV 或 dKV。 |
| `GroupCast / GroupReduce` | MagiAttention 的 group communication primitives：只把 KV 发送给需要它的 ranks，并把 backward 中的 partial dKV reduce 回 owner rank。 |
| `AlltoAll-v` | 可变长度 all-to-all 通信，每个 rank 发给其他 ranks 的数据量可以不同。MagiAttention 用它作为 group collectives 的原型实现。 |
| `overlap_degree` | MagiAttention 中控制多阶段计算/通信重叠粒度的超参数。过小隐藏不了通信，过大可能增加调度和 kernel 开销。 |
| `Ulysses` | DeepSpeed 的长序列并行方案，主要通过 head-sharded all-to-all 做 sequence parallel attention。 |
| `Ring Attention` | 序列块沿 ring 在 ranks 间流动的长上下文 attention 方案，扩展性强但在异构 mask 下可能有冗余通信和负载不均。 |
| `SLA / SLA2` | Sparse-Linear Attention系列，组合稀疏计算与线性补偿等设计。路由、训练和量化细节按具体方法区分，不能把后续版本组件归给所有版本。 |

## 推理与推理服务系统

| 术语 | 含义 |
| --- | --- |
| `Continuous Batching` | 连续批处理，允许请求在 decode 过程中动态加入和退出，提高 GPU 利用率。 |
| `KV Cache` | Transformer 推理时缓存历史 token 的 key/value 张量，长上下文服务的核心显存瓶颈之一。 |
| `Prefix Cache` | 复用完全匹配前缀的 KV，减少重复 prefill。匹配需包含 token、模型及影响 KV 的适配器、位置与多模态条件；命中率与缓存容量决定实际收益。 |
| `Speculative Decoding` | 草稿模型或草稿路径先生成候选 token，再由目标模型并行验证的推理加速方法。 |
| `TTFT` | Time To First Token，从定义好的请求起点到首个输出 token 的耗时。客户端与服务端起点不同会产生不同数字，报告需说明是否包含网络和排队。 |
| `TPOT` | Time Per Output Token。常见请求级定义为首token之后的生成时间除以后续输出token数；只有一个输出token时该定义无分母。不同系统的聚合与计时口径需核对。 |
| `vLLM` | LLM serving runtime，常与 paged attention、continuous batching、KV 管理一起讨论。 |
| `SGLang` | 面向结构化生成和 agent workload 的 serving/runtime 生态。 |

## 量化与数值

| 术语 | 含义 |
| --- | --- |
| `AWQ` | Activation-aware Weight Quantization，量化时考虑 activation 对权重误差的放大。 |
| `BF16` | 16位浮点，指数位与FP32相同，正规数范围相近，但尾数更短。较大范围可缓解部分溢出，不能保证更准确或自动消除训练不稳定；微小更新仍可能舍入丢失。 |
| `FP4` | 一类4位浮点表示，具体指数、尾数、共享scale及硬件路径由格式规定。仅写FP4无法确定数值范围和可用kernel，见[数值格式](../quantization/symbols-and-minimal-math.md)。 |
| `FP8` | 8 位浮点格式，常见 E4M3、E5M2 等变体，依赖 scale 策略和硬件支持。 |
| `GPTQ` | 典型后训练权重量化方法，通过二阶近似减少量化误差。 |
| `NVFP4` | NVIDIA定义的一种低比特格式及缩放方案。使用时要同时确认元素编码、block scale、额外缩放与对应硬件算子，不能与任意4位格式互换。 |
| `Microscaling` | 微缩放低精度格式，让一小块元素共享 scale，再用 FP8/FP4 等低比特元素表示，降低局部动态范围压力。 |
| `Fake Quantization` | 训练中先量化再反量化回高精度 tensor，用高精度 GEMM 模拟低比特误差，常用于 QAT。 |
| `STE` | Straight-Through Estimator，为不可导或离散前向指定替代反向。常见量化STE在一定范围内把梯度近似直通；通常是有偏的梯度代理，并非离散运算的真实导数。 |
| `Attn-QAT` | 面向 attention 的 4-bit QAT 方法，同时关注 forward 量化、FlashAttention backward 重算和 softmax 梯度中的低精度一致性。 |
| `SageAttention` | 针对attention推理的量化加速系列。不同版本的位宽、平滑、累加及硬件实现不同，不应把某一版本的FP4路径当作整个系列的共同定义。 |
| `Q/K smoothing` | 降低Q/K量化困难的平移或重缩放设计。必须检查对QK乘积、softmax不变性或补偿项的影响，不能任意平滑后认为结果不变。 |
| `MXFP8` | Microscaling FP8，每块配缩放因子，常用于更细粒度低精度训练或推理。 |
| `Outlier` | 激活或权重中的极端值，会显著影响量化和低精度训练稳定性。 |
| `Scale` | 量化中的缩放因子，用于把原始数值映射到低比特表示范围。 |

## 扩散与生成模型

| 术语 | 含义 |
| --- | --- |
| `CFG` | Classifier-Free Guidance，用条件分支和无条件分支组合增强条件对齐。 |
| `DDPM` | Denoising Diffusion Probabilistic Models，通过前向加噪与学习的反向过程建模数据分布，具体训练目标与方差约定需按实现核对。 |
| `DDIM` | Denoising Diffusion Implicit Models，在相容预测模型上构造不同采样路径；可选择更少时间步和确定性设置，质量仍需随采样配置评估。 |
| `DMD` | Distribution Matching Distillation，一步扩散蒸馏代表路线之一。 |
| `DMD2` | 分布匹配蒸馏的后续方法，修改教师/生成分布估计与训练方式等设计；具体机制见[蒸馏](../diffusion/distillation.md)，效果以原文设置为界。 |
| `DPM-Solver` | 利用扩散ODE结构设计的数值求解器系列。评价时同时比较函数调用数、引导设置与生成质量，不能只看步数。 |
| `Heun` | 二阶 predictor-corrector 数值求解器，常用于扩散采样。 |
| `LCM` | Latent Consistency Models，少步扩散采样路线之一。 |
| `Phased DMD` | 将 DMD 分布匹配阶段化的方法，目标是更稳地保住结构与细节。 |

## 表征、检索与多模态

| 术语 | 含义 |
| --- | --- |
| `CLIP` | Contrastive Language-Image Pre-training，图文对齐和多模态检索代表方法。 |
| `Embedding` | 向量表示。检索、VLM 对齐和状态建模中，embedding 空间几何很重要。 |
| `Hard Negative` | 困难负样本，看起来很像但关键语义不同的负样本。 |
| `InfoNCE` | 对一组候选定义的对比目标，通过归一化分数提高正配对的概率。负样本构成、温度和正对定义会影响学习信号。 |
| `MoCo` | Momentum Contrast，通过动量编码器和队列维护大量负样本。 |
| `BYOL` | Bootstrap Your Own Latent，不显式依赖负样本的自监督方法。 |
| `RAG` | Retrieval-Augmented Generation，用检索结果增强生成模型。 |

## 世界模型、具身与控制

| 术语 | 含义 |
| --- | --- |
| `Action` | agent 或控制器执行的动作，在 LLM 中可对应 token，在机器人中可对应末端位姿或关节控制量。 |
| `Advantage` | 某个动作相对当前状态平均预期有多好，常用于 PPO/GRPO 等策略更新。 |
| `Bellman Equation` | 把当前价值拆成当前奖励和下一状态价值的递推关系，是 value learning 和 model-based planning 的基础。 |
| `Camera Intrinsics / K / 内参` | 针孔相机内参矩阵 K 通常包含焦距、主点及可选skew，用于相机坐标到像素的投影。畸变系数属于另外的镜头模型参数，通常不包含在3×3的K内，见[几何与三维](../prerequisite-math/geometry-vision-and-3d.md)。 |
| `Camera Extrinsics / [R\|t] / 外参` | 相机坐标系和 world / robot base / gripper 坐标系之间的刚体变换；回答“这台相机在哪里、朝哪看”。 |
| `BP-1 / BP-2` | Stereo bad-pixel 指标，统计 disparity 误差超过 1 或 2 像素的比例；越低越好。 |
| `ConvGRU` | 把卷积和 GRU 门控结合的 recurrent module，常用于视频、光流或 stereo refinement，在空间特征图上迭代更新 hidden state。 |
| `Cost Volume` | 双目匹配中按像素和 disparity 候选组织的匹配代价或匹配特征，常见形状类似 \(C \times D \times H \times W\)。 |
| `D1` | KITTI stereo 常用 bad disparity 指标，统计超过阈值的坏像素比例；越低越好。 |
| `Depth Anything / DA` | 单目深度 foundation model 路线，重点是开放域相对深度和细节泛化；站内专题见 [DA1 讲解](../paper-deep-dives/embodied-ai/depth-anything.md)。 |
| `Depth Anything 2 / DA2` | 单目深度方法，用合成数据训练教师并以真实图像伪标签训练学生。深度模式、尺度与泛化需按具体模型和评测区分，见[DA2](../paper-deep-dives/embodied-ai/depth-anything-v2.md)。 |
| `Depth Anything 3 / DA3` | 把 DA/DA2 的深度能力扩展到任意视角几何，输出 depth、ray、camera 和可融合 3D 表示；站内专题见 [DA3 讲解](../paper-deep-dives/embodied-ai/depth-anything-3.md)。 |
| `Disparity` | 双目左右图中同一个 3D 点的横向像素位移。已知焦距 \(f\) 和 baseline \(B\) 时，可用 \(z=fB/d\) 直观看深度关系。 |
| `Ray Map` | 每个像素对应的相机射线表示，通常包含 ray origin 与 ray direction。DA3 用 depth + ray 恢复 3D 点和相机几何。 |
| `Stereo Rectification` | 双目校正，把左右图变换到同一水平扫描线，使匹配点主要沿横向 disparity 搜索；是 stereo matching 的关键输入假设。 |
| `VGGT` | Visual Geometry Grounded Transformer，feed-forward 3D 几何模型，直接预测 camera parameters、depth maps、point maps 和 3D point tracks；站内专题见 [VGGT 讲解](../paper-deep-dives/embodied-ai/vggt.md)。 |
| `World Model` | 对环境观测或状态演化的预测模型。若用于比较候选动作的后果，需有相应动作条件或干预接口及验证；单纯视频预测不自动具备规划能力。 |
| `RSSM` | Recurrent State-Space Model，Dreamer 等 latent world model 的核心结构之一。 |
| `MPC` | Model Predictive Control，滚动优化动作序列的经典规划方法。 |
| `Policy` | 从状态到动作的规则或概率分布，记作 \(\pi(a\mid s)\)。 |
| `Reward` | 环境、规则或评测器给出的反馈信号，用来定义什么行为更好。 |
| `Return` | 从当前时刻开始累计的折扣奖励，是 value function 要估计的目标。 |
| `WAM` | World-Action Model，强调动作和世界未来的联合建模。 |
| `VAM` | Video-Action Model，强调视频表示与动作建模的统一。 |
| `Near-miss` | 接近失败但尚未造成事故的样本，常用于风险训练和数据回流。 |
| `Counterfactual` | 反事实，关于已发生情境中若采取另一干预会怎样的判断。需要说明结构假设与保持不变的背景；仅改变一个相关输入并重新预测，不自动识别因果后果。 |

## 框架与部署工具

| 术语 | 含义 |
| --- | --- |
| `CUDA` | NVIDIA GPU 编程模型和运行时。 |
| `Triton` | 面向张量块 kernel 的 DSL 和 codegen 工具。 |
| `CUTLASS` | CUDA 高性能线性代数和 GEMM 类 kernel 模板库。 |
| `CuTe` | CUTLASS 中强调 layout algebra 和 tile 组合表达的抽象层。 |
| `ONNX Runtime / ORT` | 跨平台推理 runtime，可通过不同 Execution Provider 对接多类硬件。 |
| `TensorRT` | NVIDIA 面向深度学习推理的优化和部署 runtime，常用于图优化、kernel 选择、低精度和固定 shape 场景。 |
| `TensorRT-LLM` | NVIDIA 面向 LLM 推理优化的 runtime / kernel 生态。 |
| `NCCL` | NVIDIA Collective Communications Library，分布式训练和推理通信基元。 |
| `DeepEP` | DeepSeek开源的专家并行通信库，涉及token dispatch与combine。支持的拓扑、缓冲区和通信模式需按具体版本核对。 |
| `LightLLM` | 用于大语言模型推理服务的开源框架。性能需在给定模型、硬件、长度和并发下测量，不能由框架名字直接比较。 |
