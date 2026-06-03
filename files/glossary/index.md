# 术语表

这一页把知识库中反复出现的术语做统一整理。目标不是百科定义，而是在本站语境里说明它主要指什么、最容易和什么概念混淆。

**初学者先抓住。**

术语表不是要背诵，而是用来防止“同一个词在不同层意思不同”。例如 checkpoint 在训练里是恢复资产，KV cache 在推理里是显存主角，sampler 在扩散和数据管线里又各有语境。

| 使用场景 | 建议做法 |
| --- | --- |
| 读论文时遇到缩写 | 先确认基本含义，再回到对应专题页判断它改的是训练、推理、系统还是评测哪一层 |
| 排查系统问题 | 先区分术语属于训练、推理、量化、runtime 还是部署工具，避免把不同层的问题混在一起 |
| 扩写新页面 | 新术语优先补到这里，再从正文链接回来，减少各页重复解释 |

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
| `Greedy Decoding` | 贪心解码，每一步选择概率最高的 token，稳定快速但容易重复或卡在局部最优。 |
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
| `ELBO` | Evidence Lower Bound，变分推断目标，在扩散模型和世界模型中常出现。 |
| `Feature Distillation` | 特征蒸馏，让 student 对齐 teacher 的中间 hidden states、feature maps 或 attention/relation 结构，而不只学最终输出。 |
| `GRPO` | Group Relative Policy Optimization，用同一输入的多次采样做组内相对奖励，常用于可验证奖励后训练。 |
| `Knowledge Distillation / KD` | 知识蒸馏，用 teacher model 的输出分布、中间表示、rationale 或生成轨迹训练 student model，常用于压缩、提速和行为迁移。 |
| `Logit Distillation` | Logit 蒸馏，让 student 对齐 teacher 的 pre-softmax logits 或 temperature softmax 分布，保留类别或 token 间相对偏好。 |
| `LoRA` | Low-Rank Adaptation，用低秩增量替代全量参数更新的参数高效微调方法。 |
| `NAS / Neural Architecture Search` | 神经架构搜索，在候选 layer、通道、kernel、head 或 block 组合中寻找质量和成本更好的结构；站内入门见 [模型压缩、剪枝与 NAS](../foundations/model-compression-pruning-and-nas-basics.md)。 |
| `PPO` | Proximal Policy Optimization，限制新旧策略差距的策略优化方法，是 RLHF 经典算法。 |
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
| `Update RMS` | 参数更新的均方根尺度。Muon 论文用它把不同形状矩阵的更新尺度拉齐，并尽量复用 AdamW 学习率。 |
| `Newton-Schulz` | 常用于矩阵逆平方根或正交化近似的迭代方法，Muon 用它把 momentum 映射到更均衡的矩阵更新方向。 |

## 长上下文与分布式 Attention

| 术语 | 含义 |
| --- | --- |
| `Context Parallelism / CP` | 把长序列上下文切到多个 GPU 或 ranks 上共同计算 attention 的并行方式，重点瓶颈通常在 KV 通信、负载均衡和 overlap。 |
| `GQA` | Grouped-Query Attention，多组 query heads 共享较少的 key/value heads。它能降低 KV 成本，但会让 head-sharding 方法更受可整除约束影响。 |
| `FlashAttention` | IO-aware 的 exact attention kernel 路线，通过 tiling 和重算减少 \(O(N^2)\) attention matrix 显存写回。 |
| `FA3 / FA4` | FlashAttention-3/4。FA3 主要面向 Hopper，FA4/CuTe 路线开始覆盖 Hopper 与 Blackwell 的新版实现。 |
| `MagiAttention` | SandAI 的分布式 attention 方案，用 FFA、dispatch solver、GroupCast/GroupReduce 和 overlap solver 支持超长上下文与异构 mask 训练。 |
| `AttnSlice` | MagiAttention 中的 mask 表达单元，形如 `(QRange, KRange, MaskType)`，把复杂 attention mask 拆成可调度的 2D 子块。 |
| `FFA` | Flex-Flash-Attention，MagiAttention 中基于 FlashAttention-3 思路扩展的灵活 mask attention kernel。 |
| `varlen` | Variable length 的缩写，指 batch 或 packed data 中样本长度不同。长上下文训练里 varlen 会显著影响 mask area 和 rank 负载。 |
| `Block-causal mask` | 以块为单位的因果注意力 mask，常用于视频或 packed sequence，使某些块只能看历史块，某些块可作为全局块。 |
| `Patch-and-Pack / PnP` | 把不同尺寸或不同长度样本切成 patch 后 pack 到长序列中的数据组织方式，可提升吞吐，但会制造更复杂的 attention mask。 |
| `Dispatch solver` | 根据 mask area 把 query chunks 分配到 CP ranks 的调度器，目标是让有效 attention 计算量尽量均衡。 |
| `CalcMeta / CommMeta` | MagiAttention 中的运行元数据：前者描述本 rank 要算哪些 AttnSlice，后者描述需要向哪些 ranks 收发 KV 或 dKV。 |
| `GroupCast / GroupReduce` | MagiAttention 的 group communication primitives：只把 KV 发送给需要它的 ranks，并把 backward 中的 partial dKV reduce 回 owner rank。 |
| `AlltoAll-v` | 可变长度 all-to-all 通信，每个 rank 发给其他 ranks 的数据量可以不同。MagiAttention 用它作为 group collectives 的原型实现。 |
| `overlap_degree` | MagiAttention 中控制多阶段计算/通信重叠粒度的超参数。过小隐藏不了通信，过大可能增加调度和 kernel 开销。 |
| `Ulysses` | DeepSpeed 的长序列并行方案，主要通过 head-sharded all-to-all 做 sequence parallel attention。 |
| `Ring Attention` | 序列块沿 ring 在 ranks 间流动的长上下文 attention 方案，扩展性强但在异构 mask 下可能有冗余通信和负载不均。 |
| `SLA / SLA2` | Sparse-Linear Attention 及后续版本，针对视频 DiT attention，把少数关键块精确计算，其余部分用 linear compensation、learnable routing 和 QAT 适配。 |

## 推理与推理服务系统

| 术语 | 含义 |
| --- | --- |
| `Continuous Batching` | 连续批处理，允许请求在 decode 过程中动态加入和退出，提高 GPU 利用率。 |
| `KV Cache` | Transformer 推理时缓存历史 token 的 key/value 张量，长上下文服务的核心显存瓶颈之一。 |
| `Prefix Cache` | 缓存重复 prompt 前缀，避免反复 prefill。模板化 agent 和企业问答中价值很高。 |
| `Speculative Decoding` | 草稿模型或草稿路径先生成候选 token，再由目标模型并行验证的推理加速方法。 |
| `TTFT` | Time To First Token，首 token 延迟。 |
| `TPOT` | Time Per Output Token，生成阶段每 token 延迟。 |
| `vLLM` | LLM serving runtime，常与 paged attention、continuous batching、KV 管理一起讨论。 |
| `SGLang` | 面向结构化生成和 agent workload 的 serving/runtime 生态。 |

## 量化与数值

| 术语 | 含义 |
| --- | --- |
| `AWQ` | Activation-aware Weight Quantization，量化时考虑 activation 对权重误差的放大。 |
| `BF16` | 16 位浮点格式，指数范围接近 FP32，训练稳定性通常好于 FP16。 |
| `FP4` | 4 位浮点格式，能利用 Blackwell 等硬件的低比特 tensor core，但动态范围和离散值很少，对 attention 这类 heavy-tailed activation 更敏感。 |
| `FP8` | 8 位浮点格式，常见 E4M3、E5M2 等变体，依赖 scale 策略和硬件支持。 |
| `GPTQ` | 典型后训练权重量化方法，通过二阶近似减少量化误差。 |
| `NVFP4` | NVIDIA FP4/microscaling 路线，通常结合小 block、共享 scale 和 FP8 scale，用于 Blackwell 上的 FP4 GEMM 或 attention。 |
| `Microscaling` | 微缩放低精度格式，让一小块元素共享 scale，再用 FP8/FP4 等低比特元素表示，降低局部动态范围压力。 |
| `Fake Quantization` | 训练中先量化再反量化回高精度 tensor，用高精度 GEMM 模拟低比特误差，常用于 QAT。 |
| `STE` | Straight-Through Estimator，量化等不可导操作中常用的梯度近似，backward 近似把量化当作恒等映射。 |
| `Attn-QAT` | 面向 attention 的 4-bit QAT 方法，重点不是只量化 forward，而是修正 FlashAttention backward 重算和 softmax 梯度中的低精度一致性。 |
| `SageAttention` | 面向 attention inference 的低比特加速路线，常结合 Q/K smoothing、FP4 tensor core 和 attention probability 量化等技巧。 |
| `Q/K smoothing` | 量化 attention 时对 Q/K 分布做平滑或重缩放，目标是缓解 outlier 对 \(QK^\top\) 的破坏。 |
| `MXFP8` | Microscaling FP8，每块配缩放因子，常用于更细粒度低精度训练或推理。 |
| `Outlier` | 激活或权重中的极端值，会显著影响量化和低精度训练稳定性。 |
| `Scale` | 量化中的缩放因子，用于把原始数值映射到低比特表示范围。 |

## 扩散与生成模型

| 术语 | 含义 |
| --- | --- |
| `CFG` | Classifier-Free Guidance，用条件分支和无条件分支组合增强条件对齐。 |
| `DDPM` | Denoising Diffusion Probabilistic Models，现代扩散模型主流路线起点。 |
| `DDIM` | Denoising Diffusion Implicit Models，不重训模型也能减少采样步数的重要方法。 |
| `DMD` | Distribution Matching Distillation，一步扩散蒸馏代表路线之一。 |
| `DMD2` | DMD 的改进版本，重点改善一步蒸馏的模糊和训练不稳。 |
| `DPM-Solver` | 面向 diffusion ODE 的高阶专用求解器，用于少步高质量采样。 |
| `Heun` | 二阶 predictor-corrector 数值求解器，常用于扩散采样。 |
| `LCM` | Latent Consistency Models，少步扩散采样路线之一。 |
| `Phased DMD` | 将 DMD 分布匹配阶段化的方法，目标是更稳地保住结构与细节。 |

## 表征、检索与多模态

| 术语 | 含义 |
| --- | --- |
| `CLIP` | Contrastive Language-Image Pre-training，图文对齐和多模态检索代表方法。 |
| `Embedding` | 向量表示。检索、VLM 对齐和状态建模中，embedding 空间几何很重要。 |
| `Hard Negative` | 困难负样本，看起来很像但关键语义不同的负样本。 |
| `InfoNCE` | 图文/样本匹配常用目标，要求模型把正确匹配项排到候选最前。 |
| `MoCo` | Momentum Contrast，通过动量编码器和队列维护大量负样本。 |
| `BYOL` | Bootstrap Your Own Latent，不显式依赖负样本的自监督方法。 |
| `RAG` | Retrieval-Augmented Generation，用检索结果增强生成模型。 |

## 世界模型、具身与控制

| 术语 | 含义 |
| --- | --- |
| `Action` | agent 或控制器执行的动作，在 LLM 中可对应 token，在机器人中可对应末端位姿或关节控制量。 |
| `Advantage` | 某个动作相对当前状态平均预期有多好，常用于 PPO/GRPO 等策略更新。 |
| `Bellman Equation` | 把当前价值拆成当前奖励和下一状态价值的递推关系，是 value learning 和 model-based planning 的基础。 |
| `Camera Intrinsics / K / 内参` | 相机自身的成像参数，包括 \(f_x,f_y,c_x,c_y\) 和畸变；回答“这台相机怎么把相机坐标投到像素”。 |
| `Camera Extrinsics / [R\|t] / 外参` | 相机坐标系和 world / robot base / gripper 坐标系之间的刚体变换；回答“这台相机在哪里、朝哪看”。 |
| `BP-1 / BP-2` | Stereo bad-pixel 指标，统计 disparity 误差超过 1 或 2 像素的比例；越低越好。 |
| `ConvGRU` | 把卷积和 GRU 门控结合的 recurrent module，常用于视频、光流或 stereo refinement，在空间特征图上迭代更新 hidden state。 |
| `Cost Volume` | 双目匹配中按像素和 disparity 候选组织的匹配代价或匹配特征，常见形状类似 \(C \times D \times H \times W\)。 |
| `D1` | KITTI stereo 常用 bad disparity 指标，统计超过阈值的坏像素比例；越低越好。 |
| `Depth Anything / DA` | 单目深度 foundation model 路线，重点是开放域相对深度和细节泛化；站内专题见 [DA1 讲解](../paper-deep-dives/embodied-ai/depth-anything.md)。 |
| `Depth Anything 2 / DA2` | DA 的更强版本，常作为单目深度基线或 teacher 参考；它用 synthetic teacher 和 pseudo-labeled real images 提升细节与鲁棒性，本身主要解决单图深度，不直接给出多视角相机几何；站内专题见 [DA2 讲解](../paper-deep-dives/embodied-ai/depth-anything-v2.md)。 |
| `Depth Anything 3 / DA3` | 把 DA/DA2 的深度能力扩展到任意视角几何，输出 depth、ray、camera 和可融合 3D 表示；站内专题见 [DA3 讲解](../paper-deep-dives/embodied-ai/depth-anything-3.md)。 |
| `Disparity` | 双目左右图中同一个 3D 点的横向像素位移。已知焦距 \(f\) 和 baseline \(B\) 时，可用 \(z=fB/d\) 直观看深度关系。 |
| `Ray Map` | 每个像素对应的相机射线表示，通常包含 ray origin 与 ray direction。DA3 用 depth + ray 恢复 3D 点和相机几何。 |
| `Stereo Rectification` | 双目校正，把左右图变换到同一水平扫描线，使匹配点主要沿横向 disparity 搜索；是 stereo matching 的关键输入假设。 |
| `VGGT` | Visual Geometry Grounded Transformer，feed-forward 3D 几何模型，直接预测 camera parameters、depth maps、point maps 和 3D point tracks；站内专题见 [VGGT 讲解](../paper-deep-dives/embodied-ai/vggt.md)。 |
| `World Model` | 面向决策的环境预测模型，学习世界如何随动作、时间和目标变化。 |
| `RSSM` | Recurrent State-Space Model，Dreamer 等 latent world model 的核心结构之一。 |
| `MPC` | Model Predictive Control，滚动优化动作序列的经典规划方法。 |
| `Policy` | 从状态到动作的规则或概率分布，记作 \(\pi(a\mid s)\)。 |
| `Reward` | 环境、规则或评测器给出的反馈信号，用来定义什么行为更好。 |
| `Return` | 从当前时刻开始累计的折扣奖励，是 value function 要估计的目标。 |
| `WAM` | World-Action Model，强调动作和世界未来的联合建模。 |
| `VAM` | Video-Action Model，强调视频表示与动作建模的统一。 |
| `Near-miss` | 接近失败但尚未造成事故的样本，常用于风险训练和数据回流。 |
| `Counterfactual` | 反事实条件变化，用于比较“如果当时换个动作会怎样”。 |

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
| `DeepEP` | DeepSeek 开源的 expert-parallel communication library。MagiAttention 的 native group collectives 借鉴了其中的分层通信和去重思路。 |
| `LightLLM` | 偏轻量、高性能和易扩展的 LLM serving framework。 |
