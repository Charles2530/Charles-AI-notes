# 参考文献总表

这一页不是简单罗列论文名，而是把全站内容按研究问题整理成可长期维护的索引。  
阅读时建议先按主题找到入口，再回到对应章节做细读。

!!! tip "使用方式"
    如果你刚入门，优先看每节的“必读起点”。如果你准备做课题或工程复现，优先看“系统 / Benchmark / 开源实现”。如果某篇论文名字后面写了“工程意义”，说明它不只是理论参考，还会影响训练、推理、评测或部署决策。

## 1. 扩散模型与生成模型

对应站内章节：[扩散模型](../diffusion/index.md)。

### 1.1 必读起点

1. [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)  
   `DDPM` 的经典起点，建立“逐步加噪、逐步去噪”的基本框架。
2. [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)  
   `DDIM` 解释为什么采样路径可以非马尔可夫化，是少步采样和确定性采样的重要基础。
3. [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456)  
   把扩散过程放到 `SDE / ODE` 视角，是理解连续时间扩散、概率流 ODE 和求解器的关键论文。
4. [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364)  
   `EDM` 把噪声参数化、采样器、预条件等设计拆开，是工程调参和复现实验很值得反复看的论文。

### 1.2 快速采样、求解器与少步蒸馏

1. [DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps](https://arxiv.org/abs/2206.00927)  
   从数值积分角度设计扩散采样器，适合连接站内“求解器视角”内容。
2. [DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models](https://arxiv.org/abs/2211.01095)  
   更贴近 classifier-free guidance 等实际生成配置。
3. [Progressive Distillation for Fast Sampling of Diffusion Models](https://arxiv.org/abs/2202.00512)  
   少步蒸馏的经典路线：把多步 teacher 压成少步 student。
4. [Consistency Models](https://arxiv.org/abs/2303.01469)  
   把不同噪声水平映射到同一数据点，适合理解一步/少步生成的另一条路线。
5. [Latent Consistency Models](https://arxiv.org/abs/2310.04378)  
   `LCM` 是把 consistency 思路推向 Stable Diffusion 生态的实用论文。
6. [Improved Distribution Matching Distillation for Fast Image Synthesis](https://arxiv.org/abs/2405.14867)  
   `DMD2` 值得放在少步蒸馏里看，关注分布匹配和高质量少步生成。

### 1.3 Rectified Flow、Rectified Diffusion 与视频扩散

1. [Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003)  
   `Rectified Flow` 关注把生成路径“拉直”，降低求解难度。
2. [InstaFlow: One Step is Enough for High-Quality Diffusion-Based Text-to-Image Generation](https://arxiv.org/abs/2309.06380)  
   少步/一步生成与 rectified flow 结合的重要实例。
3. [Rectified Diffusion: Straightness Is Not Your Need in Rectified Flow](https://arxiv.org/abs/2410.07303)  
   适合补充 `Rectified Diffusion` 术语：重点不是简单追求路径直，而是重新分析整流目标、采样效率与生成质量之间的关系。
4. [Video Diffusion Models](https://arxiv.org/abs/2204.03458)  
   视频扩散的早期代表，适合作为时序生成入口。
5. [Imagen Video: High Definition Video Generation with Diffusion Models](https://arxiv.org/abs/2210.02303)  
   多级级联视频生成路线，对理解高分辨率视频生成系统很有帮助。

## 2. VLM 与多模态模型

对应站内章节：[VLM](../vlm/index.md)。

### 2.1 视觉语言预训练与指令微调

1. [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)  
   `CLIP` 是图文对齐、对比学习、多模态检索和后续 VLM 的基础。
2. [Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198)  
   早期强 VLM 代表，适合理解视觉编码器和语言模型之间的桥接。
3. [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/abs/2301.12597)  
   `Q-Former` 是连接视觉特征和 LLM 的经典结构。
4. [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485)  
   `LLaVA` 让开源 VLM 进入指令微调时代，是理解 VLM 数据合成和对话微调的起点。
5. [PaLM-E: An Embodied Multimodal Language Model](https://arxiv.org/abs/2303.03378)  
   多模态语言模型进入具身控制和机器人任务的重要节点。

### 2.2 文档、图表、OCR 与 Grounding

1. [LayoutLM: Pre-training of Text and Layout for Document Image Understanding](https://arxiv.org/abs/1912.13318)  
   文档理解中“文字 + 坐标 + 布局”的基础路线。
2. [Pix2Struct: Screenshot Parsing as Pretraining for Visual Language Understanding](https://arxiv.org/abs/2210.03347)  
   适合连接截图理解、表格理解和结构化输出。
3. [Donut: Document Understanding Transformer without OCR](https://arxiv.org/abs/2111.15664)  
   OCR-free 文档理解代表。
4. [ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning](https://arxiv.org/abs/2203.10244)  
   图表问答 benchmark，适合评估“看懂图”和“会推理”是否同时成立。
5. [Referring Expression Comprehension Survey / RefCOCO 系列](https://arxiv.org/abs/1608.00272)  
   Grounding 和指代表达理解的长期基准入口。

## 3. VLA、机器人策略与具身智能

对应站内章节：[VLA](../vla/index.md)、[具身智能](../embodied-ai/index.md)。

### 3.1 机器人基础模型与 VLA

1. [Do As I Can, Not As I Say: Grounding Language in Robotic Affordances](https://arxiv.org/abs/2204.01691)  
   `SayCan` 代表“语言模型规划 + affordance 评分”的早期路线。
2. [RT-1: Robotics Transformer for Real-World Control at Scale](https://arxiv.org/abs/2212.06817)  
   把机器人策略学习推向大规模真实数据的代表工作。
3. [RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control](https://arxiv.org/abs/2307.15818)  
   `VLA` 概念的重要入口，把网页视觉语言知识迁移到机器人动作。
4. [Open X-Embodiment: Robotic Learning Datasets and RT-X Models](https://arxiv.org/abs/2310.08864)  
   多机器人、多任务数据聚合的重要数据工程论文。
5. [OpenVLA: An Open-Source Vision-Language-Action Model](https://arxiv.org/abs/2406.09246)  
   开源 VLA 代表，适合结合代码和数据做复现。
6. [Octo: An Open-Source Generalist Robot Policy](https://arxiv.org/abs/2405.12213)  
   通用机器人策略的开源入口。

### 3.2 模仿学习、动作分块与 Diffusion Policy

1. [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705)  
   `ALOHA / ACT` 方向入口，适合理解动作分块和双臂操作数据。
2. [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137)  
   把扩散模型用于连续动作策略学习的经典论文。
3. [RoboCat: A Self-Improving Generalist Agent for Robotic Manipulation](https://arxiv.org/abs/2306.11706)  
   自我改进和机器人数据闭环路线的重要参考。
4. [BridgeData V2: A Dataset for Robot Learning at Scale](https://arxiv.org/abs/2308.12952)  
   机器人策略学习的数据基础设施参考。

### 3.3 Benchmark 与评测

1. [RLBench: The Robot Learning Benchmark & Learning Environment](https://arxiv.org/abs/1909.12271)  
   桌面操作任务 benchmark。
2. [Meta-World: A Benchmark and Evaluation for Multi-Task and Meta Reinforcement Learning](https://arxiv.org/abs/1910.10897)  
   多任务操作策略学习常用基准。
3. [BEHAVIOR-1K: A Benchmark for Embodied AI with 1,000 Everyday Activities and Realistic Simulation](https://arxiv.org/abs/2403.09227)  
   家庭和日常活动场景的具身评测入口。
4. [CALVIN: A Benchmark for Language-Conditioned Policy Learning for Long-Horizon Robot Manipulation Tasks](https://arxiv.org/abs/2112.03227)  
   长程语言条件机器人操作常用评测。

## 4. 世界模型、训练与规划

对应站内章节：[世界模型](../world-models/index.md)。

### 4.1 必读起点：latent dynamics 与 imagined rollout

1. [World Models](https://arxiv.org/abs/1803.10122)  
   把“压缩观测、学习动态、在想象中规划”讲清楚的经典起点。
2. [Learning Latent Dynamics for Planning from Pixels](https://arxiv.org/abs/1811.04551)  
   `PlaNet / RSSM` 路线入口：从像素学习 latent dynamics，再在 latent space 中规划。
3. [Dream to Control: Learning Behaviors by Latent Imagination](https://arxiv.org/abs/1912.01603)  
   `Dreamer` 的核心：在 learned world model 的想象轨迹中训练 actor-critic。
4. [Mastering Atari with Discrete World Models](https://arxiv.org/abs/2010.02193)  
   `DreamerV2`，离散 latent 对复杂视觉控制任务很关键。
5. [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104)  
   `DreamerV3`，展示同一套世界模型和策略学习配方跨多个 domain 的泛化能力。

### 4.2 控制、MPC 与规划质量

1. [Temporal Difference Learning for Model Predictive Control](https://arxiv.org/abs/2203.04955)  
   `TD-MPC` 把 TD learning 和 latent-space MPC 结合。
2. [TD-MPC2: Scalable, Robust World Models for Continuous Control](https://arxiv.org/abs/2310.16828)  
   更系统地讨论 scalable world models for control。
3. [IRIS: Transformers are Sample-Efficient World Models](https://arxiv.org/abs/2209.00588)  
   用 Transformer 世界模型解决 Atari，适合理解离散 token 世界模型。
4. [Masked World Models for Visual Control](https://arxiv.org/abs/2206.14244)  
   结合 masked modeling 和 visual control，是世界模型表征学习的代表。

### 4.3 视频生成、交互式模拟与具身世界模型

1. [I-JEPA: Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture](https://arxiv.org/abs/2301.08243)  
   预测表征而非像素，适合理解“好世界模型不一定要重建像素”的观点。
2. [V-JEPA: Video Joint Embedding Predictive Architecture](https://arxiv.org/abs/2404.08471)  
   视频预测表征，对世界模型训练和视频理解都有参考价值。
3. [Learning Interactive Real-World Simulators](https://arxiv.org/abs/2310.06114)  
   `UniSim` 代表把真实世界交互建模成可模拟环境。
4. [Genie: Generative Interactive Environments](https://arxiv.org/abs/2402.15391)  
   从视频中学可交互世界模型，适合连接“视频生成模型”和“可控世界模型”。
5. [Video Language Planning](https://arxiv.org/abs/2310.10625)  
   连接视频预测、语言规划和长程任务分解。

### 4.4 自动驾驶、数据闭环与大规模仿真

1. [GAIA-1: A Generative World Model for Autonomous Driving](https://arxiv.org/abs/2309.17080)  
   自动驾驶生成式世界模型代表。
2. [DriveDreamer: Towards Real-world-driven World Models for Autonomous Driving](https://arxiv.org/abs/2309.09777)  
   自动驾驶场景下的 controllable driving world model。
3. [World Model on Million-Length Video and Language with Blockwise RingAttention](https://arxiv.org/abs/2402.08268)  
   长视频世界模型训练和长序列系统能力的参考。
4. [Waymo Open Dataset](https://waymo.com/open/)  
   自动驾驶研究的核心数据集入口。
5. [nuScenes](https://www.nuscenes.org/)  
   多传感器自动驾驶数据集，适合世界模型、感知与规划评测。

### 4.5 世界模型 Benchmark 与开放难题

1. [Procgen Benchmark](https://arxiv.org/abs/1912.01588)  
   用于测试泛化和分布外表现，适合检验 world model 是否只是记住训练环境。
2. [VBench: Comprehensive Benchmark Suite for Video Generative Models](https://arxiv.org/abs/2311.17982)  
   视频生成评测参考，但要注意：视觉质量不等于规划质量。
3. [MineDojo: Building Open-Ended Embodied Agents with Internet-Scale Knowledge](https://arxiv.org/abs/2206.08853)  
   开放式具身任务 benchmark，可用于讨论 long-horizon world model。
4. [Crafter: Open-Ended Reinforcement Learning in a Minecraft-like World](https://arxiv.org/abs/2109.06780)  
   轻量、可控、适合快速测试的开放式世界环境。

### 4.6 世界模型训练效率与系统优化

这里的参考文献只收“让训练更高效”的工作，不把普通 world model 训练目标、loss 设计或模型谱系重复列入。阅读时重点看它们优化的是哪一种效率：真实交互步数、训练 wall-clock、显存/序列计算、rollout 复用率，还是把视频基础模型低成本改造成可交互模拟器。

#### 样本效率：更少真实环境交互

1. [Model-Based Reinforcement Learning for Atari](https://arxiv.org/abs/1903.00374)  
   `SimPLe` 是早期代表：用视频预测模型生成短期模拟轨迹，让 policy 在 learned model 里训练，从而减少 Atari 真实交互量。
2. [Mastering Atari Games with Limited Data](https://arxiv.org/abs/2111.00210)  
   `EfficientZero` 重点是有限数据下的 sample-efficient RL，把 learned model、self-supervised consistency、value prefix 和 MCTS 结合起来。
3. [EfficientZero V2: Mastering Discrete and Continuous Control with Limited Data](https://arxiv.org/abs/2403.00564)  
   把 EfficientZero 推到离散/连续动作、视觉/低维输入的统一框架，适合看有限数据设置下 world model + planning 的通用化。

#### Transformer world model：并行训练与 token 效率

1. [Transformers are Sample-Efficient World Models](https://arxiv.org/abs/2209.00588)  
   `IRIS` 用离散 autoencoder 把像素压成 tokens，再用 autoregressive Transformer 建模动态，是 tokenized world model 的重要入口。
2. [Transformer-based World Models Are Happy With 100k Interactions](https://arxiv.org/abs/2303.07109)  
   `TWM` 关注 Transformer world model 在 Atari 100k 低数据场景下的训练效率，适合和 IRIS、STORM 对比。
3. [STORM: Efficient Stochastic Transformer based World Models for Reinforcement Learning](https://arxiv.org/abs/2310.09615)  
   把 stochastic latent 和 Transformer 序列建模结合，论文直接报告了单卡训练时间，是 wall-clock 维度很值得看的工作。
4. [Efficient World Models with Context-Aware Tokenization](https://arxiv.org/abs/2406.19320)  
   `Delta-IRIS / Δ-IRIS` 重点在 tokenization：减少不必要 token 和长序列计算，是“世界模型训练效率”里很直接的一类优化。
5. [Learning Transformer-based World Models with Contrastive Predictive Coding](https://arxiv.org/abs/2503.04416)  
   `TWISTER` 用 action-conditioned CPC 改善 Transformer world model 表征学习，重点看表示质量和样本效率的关系。

#### 连续控制：可扩展模型与数据复用

1. [TD-MPC2: Scalable, Robust World Models for Continuous Control](https://arxiv.org/abs/2310.16828)  
   `TD-MPC2` 关注 decoder-free latent world model、TD learning 与 MPC 的结合，重点看多任务数据、模型规模和固定超参如何提升训练可扩展性。

#### 长上下文训练：降低序列长度带来的系统成本

1. [World Model on Million-Length Video And Language With Blockwise RingAttention](https://arxiv.org/abs/2402.08268)  
   `LWM` 用 Blockwise RingAttention 处理百万级 video-language token，是长视频世界模型训练系统的直接参考。
2. [Long-Context State-Space Video World Models](https://arxiv.org/abs/2505.20171)  
   用 state-space memory 处理长时依赖，并结合 local attention 保留局部视觉细节，重点是让长时 rollout 成本不要随历史长度线性失控。

#### 视频世界模型实时化：从离线生成训练到 streaming rollout

1. [Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion](https://arxiv.org/abs/2407.01392)  
   用每帧独立噪声水平把 full-sequence diffusion 和 next-token prediction 接起来，适合看视频模型如何获得更自然的 causal rollout 训练形式。
2. [From Slow Bidirectional to Fast Autoregressive Video Diffusion Models](https://arxiv.org/abs/2412.07772)  
   `CausVid` 把预训练 bidirectional video diffusion 改造成 causal autoregressive video diffusion，并通过 distillation 和 KV cache 降低交互延迟。
3. [MAGI-1: Autoregressive Video Generation at Scale](https://arxiv.org/abs/2505.13211)  
   从 chunk-wise autoregressive video generation 出发讨论大规模训练和部署基础设施，重点看 causal video foundation model 如何保持固定峰值推理成本。
4. [Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion](https://arxiv.org/abs/2506.08009)  
   把 inference-time autoregressive rollout 的状态分布带进训练，重点解决 teacher forcing 和自回归推理之间的分布错位。
5. [Autoregressive Adversarial Post-Training for Real-Time Interactive Video Generation](https://arxiv.org/abs/2506.09350)  
   `AAPT` 关注少步、实时、交互式视频生成的 post-training，是把视频生成模型推向交互世界模拟器的重要效率方向。

!!! note "为什么世界模型文献要单独强调训练与评测"
    世界模型的核心难点不只是“能不能生成未来帧”，而是生成的 latent 是否能支持下游规划、控制、反事实推理和泛化。重建 loss、next-frame prediction loss、FID 或视频视觉质量指标都只能回答一部分问题。更有价值的 benchmark 应该把下游任务泛化、rollout 误差累积、规划质量和交互控制成功率放到核心位置。

## 5. 对比学习与表征学习

对应站内章节：[对比学习](../contrastive-learning/index.md)。

### 5.1 图文对齐与通用对比学习

1. [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)  
   `SimCLR` 是现代对比学习的基本范式入口。
2. [Momentum Contrast for Unsupervised Visual Representation Learning](https://arxiv.org/abs/1911.05722)  
   `MoCo` 用队列和动量编码器解决大规模负样本问题。
3. [Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning](https://arxiv.org/abs/2006.07733)  
   `BYOL` 是非对比自监督的重要入口。
4. [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294)  
   `DINO` 适合连接 ViT、自蒸馏和视觉表征。
5. [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)  
   `CLIP` 也是多模态对比学习的中心论文。

### 5.2 评测与失败模式

1. [An Empirical Study of Training Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.02057)  
   适合对照不同自监督训练策略。
2. [Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere](https://arxiv.org/abs/2005.10242)  
   `alignment / uniformity` 是解释对比学习几何结构的常用框架。

## 6. 量化、数制与低比特训练

对应站内章节：[量化](../quantization/index.md)。

### 6.1 低比特 LLM 综述与基础数制

1. [A Survey of Low-bit Large Language Models: Basics, Systems, and Algorithms](https://arxiv.org/abs/2409.16694)  
   推荐作为量化专题的总入口：把数值格式、算法、系统和训练路线放在同一张地图里。
2. [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)  
   `E4M3 / E5M2` 的基础论文，适合理解 FP8 为什么比 INT8 更自然地适配训练和推理。
3. [Microscaling Data Formats for Deep Learning](https://arxiv.org/abs/2310.10537)  
   `MXFP / MXINT` 微缩放格式入口，是 `MXFP8 / MXFP4` 的基础。
4. [Ascend HiFloat8 Format for Deep Learning](https://arxiv.org/abs/2409.16626)  
   `HiFloat8 / HIFP8` 方向入口，适合和 FP8、MXFP8 对比动态范围和尾数分配。
5. [OCP Microscaling Formats MX v1.0 Specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)  
   `MX` 格式标准参考。

### 6.2 PTQ、权重量化与激活离群值

1. [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339)  
   解释大模型 activation outlier 为什么会破坏朴素 INT8。
2. [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)  
   权重量化经典方法，适合理解二阶信息和逐列量化。
3. [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438)  
   把激活难量化问题转移到权重侧，是 W8A8 路线的关键论文。
4. [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978)  
   激活感知权重量化，工程生态中非常常见。
5. [OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models](https://arxiv.org/abs/2308.13137)  
   面向校准和端到端误差控制的 PTQ 路线。
6. [QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks](https://arxiv.org/abs/2402.04396)  
   旋转、码本和低比特权重量化的重要参考。

### 6.3 QLoRA、低资源微调与量化训练

1. [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)  
   量化底座 + LoRA 微调的核心论文。
2. [ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers](https://arxiv.org/abs/2206.01861)  
   结合分层量化和系统部署的早期代表。
3. [FP8-LM: Training FP8 Large Language Models](https://arxiv.org/abs/2310.18313)  
   FP8 训练 LLM 的早期系统路线，重点是梯度、优化器状态和缩放策略。
4. [COAT: Compressing Optimizer States and Activation for Memory-Efficient FP8 Training](https://arxiv.org/abs/2410.19313)  
   重点关注 optimizer states 和 activation，适合作为“低比特训练不只是压权重”的参考。
5. [Scaling FP8 Training to Trillion-Token LLMs](https://arxiv.org/abs/2409.12517)  
   讨论大规模 FP8 训练稳定性，尤其是 SwiGLU outlier 和长训练后期风险。
6. [To FP8 and Back Again: Quantifying Reduced Precision Effects on LLM Training Stability](https://arxiv.org/abs/2405.18710)  
   低精度训练稳定性评估参考，提醒不要只看短跑 loss。
7. [μnit Scaling: Simple and Scalable FP8 LLM Training](https://arxiv.org/abs/2502.05967)  
   静态缩放和参数化改造路线，适合关注“降低动态 scale 管理复杂度”的读者。
8. [Recipes for Pre-training LLMs with MXFP8](https://arxiv.org/abs/2506.08027)  
   MXFP8 预训练配方，适合和传统 FP8 训练对比。

### 6.4 FP4、MXFP4、NVFP4 与前沿低比特训练

1. [Optimizing Large Language Model Training Using FP4 Quantization](https://arxiv.org/abs/2501.17116)  
   FP4 训练框架代表，重点是可微量化估计、outlier clamp 与 mixed precision。
2. [Training LLMs with MXFP4](https://arxiv.org/abs/2502.20586)  
   MXFP4 训练路线，适合对比 Microscaling 和随机 Hadamard 变换如何处理 outlier。
3. [Quartet: Native FP4 Training Can Be Optimal for Large Language Models](https://arxiv.org/abs/2505.14669)  
   面向原生 FP4 硬件和 kernel 的训练路线，强调 scaling law、梯度无偏和 Blackwell 上的实现。
4. [Pretraining Large Language Models with NVFP4](https://arxiv.org/abs/2509.25149)  
   `NVFP4` 预训练代表，关注 1x16 / 16 元素微块、FP8 scale 和长 token 训练稳定性。
5. [Bridging the Gap Between Promise and Performance for Microscaling FP4 Quantization](https://arxiv.org/abs/2509.23202)  
   对 `MXFP4 / NVFP4` PTQ 和硬件 kernel 的系统分析，提醒 FP4 格式需要方法与 kernel 配套。
6. [Four Over Six: More Accurate NVFP4 Quantization with Adaptive Block Scaling](https://arxiv.org/abs/2512.02010)  
   `4/6` adaptive block scaling，适合理解 NVFP4 中“缩到 6 不一定最优”的问题。
7. [Quartet II: Accurate LLM Pre-Training in NVFP4 by Improved Unbiased Gradient Estimation](https://arxiv.org/abs/2601.22813)  
   `NVFP4` 预训练的后续路线，重点是更好的无偏梯度估计。

!!! note "低比特训练的核心阅读线"
    如果你要研究 FP8/FP4 训练，不要只按 bit 数排序。更好的阅读顺序是：先看 `FP8 Formats` 和 `Microscaling` 理解数制，再看 `FP8-LM / COAT / Scaling FP8` 理解 activation 与 optimizer 状态，再看 `MXFP8 / MXFP4 / NVFP4 / Quartet` 理解 4-bit 训练为什么必须同时改数制、scale、rounding、kernel 和稳定性评测。

### 6.5 量化运行时、ONNX 与可视化工具

1. [ONNX Runtime Quantization Documentation](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)  
   ONNX 量化实践入口，适合模型导出、校准、静态/动态量化和部署链路。
2. [ONNX Runtime GitHub](https://github.com/microsoft/onnxruntime)  
   官方实现与 issue 入口。
3. [Netron](https://github.com/lutzroeder/netron)  
   模型图可视化工具，适合检查 ONNX 节点、权重量化节点和算子路径。
4. [Qualcomm AI Hub Documentation](https://app.aihub.qualcomm.com/docs/)  
   移动端和 Qualcomm 设备部署入口。
5. [Qualcomm AI Engine Direct SDK / QNN](https://www.qualcomm.com/developer/software/qualcomm-ai-engine-direct-sdk)  
   `QNN` 路线入口，`QNN_DLC` 相关部署常需要结合 SDK 文档、转换工具和目标芯片支持矩阵看。

## 7. MTP、投机解码与推理加速

对应站内章节：[MTP 与投机解码](../training/mtp-and-speculative-decoding.md)、[缓存、路由与投机执行](../inference/caching-routing-and-speculative.md)。

### 7.1 MTP 与多 token 预测

1. [Blockwise Parallel Decoding for Deep Autoregressive Models](https://proceedings.neurips.cc/paper_files/paper/2018/file/c4127b9194fe8562c64dc0f5bf2c93bc-Paper.pdf)  
   可以看作多 token 并行预测和验证的早期思想源头。
2. [ProphetNet: Predicting Future N-gram for Sequence-to-Sequence Pre-training](https://arxiv.org/abs/2001.04063)  
   未来 n-gram 预测目标，适合理解“让模型提前规划未来 token”的训练动机。
3. [Better & Faster Large Language Models via Multi-token Prediction](https://arxiv.org/abs/2404.19737)  
   Meta MTP 路线：共享 trunk + 多个 future-token heads，强调 sample efficiency 和自投机推理收益。
4. [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)  
   DeepSeek-V3 把 MTP 写进训练目标，同时采用 MoE、MLA 和高效训练系统，是工程化 MTP 的重要参考。
5. [DeepSeek-V3 GitHub](https://github.com/deepseek-ai/DeepSeek-V3)  
   查看模型发布、配置和实现细节。
6. [Multi-Token Prediction Needs Registers](https://arxiv.org/abs/2505.10518)  
   `MuToR` 用 register token 做多未来监督，优点是尽量减少架构侵入。
7. [L-MTP: Leap Multi-Token Prediction Beyond Adjacent Context for Large Language Models](https://arxiv.org/abs/2505.17505)  
   跨步 MTP 路线，重点是非相邻未来 token、look-backward 和推理步长压缩。

### 7.2 标准投机解码与自投机解码

1. [Fast Inference from Transformers via Speculative Decoding](https://proceedings.mlr.press/v202/leviathan23a.html)  
   经典投机解码入口：小 draft model 提案，大 target model 并行验证，并保持输出分布一致。
2. [Draft & Verify: Lossless Large Language Model Acceleration via Self-Speculative Decoding](https://arxiv.org/abs/2309.08168)  
   自投机解码路线，不依赖额外小模型，而是利用模型内部路径做 draft。
3. [LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding](https://aclanthology.org/2024.acl-long.681/)  
   Meta 的 early exit + self-speculative 路线，和 `Logit Lens`、中间层可解码性关系紧密。
4. [SpecExec: Massively Parallel Speculative Decoding for Interactive LLM Inference on Consumer Devices](https://arxiv.org/abs/2406.02532)  
   关注消费级设备和 offloading 场景下的大规模草稿树验证。
5. [OPT-Tree: Speculative Decoding with Adaptive Draft Tree Structure](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00735/128189)  
   自适应 draft tree 路线，适合和 EAGLE-2 动态树对照。

### 7.3 EAGLE 系列与特征级 draft

1. [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077)  
   从 token draft 转向 feature draft，重点是特征级自回归和不确定性处理。
2. [EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees](https://arxiv.org/abs/2406.16858)  
   用上下文感知动态草稿树提升接受长度。
3. [EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](https://arxiv.org/abs/2503.01840)  
   引入 training-time test 和多层特征融合，解决 feature constraint 带来的数据扩展瓶颈。
4. [EAGLE GitHub](https://github.com/SafeAILab/EAGLE)  
   代码和模型入口。

### 7.4 量化 + 投机解码

1. [QSpec: Speculative Decoding with Complementary Quantization Schemes](https://arxiv.org/abs/2410.11305)  
   低精度 `W4A4` 负责快速草拟，高精度 `W4A16` 负责验证，是“量化和投机解码耦合”的核心参考。
2. [QSpec GitHub](https://github.com/hku-netexplo-lab/QSpec)  
   复现入口。
3. [An Introduction to Speculative Decoding for Reducing Latency in AI Inference](https://developer.nvidia.com/blog/an-introduction-to-speculative-decoding-for-reducing-latency-in-ai-inference/)  
   NVIDIA 官方博客，适合工程入门和向非算法同学解释 draft/verify 机制。
4. [TorchSpec](https://github.com/torchspec-project/TorchSpec)  
   PyTorch 原生 speculative decoding 训练框架，适合追踪工程实现。

!!! note "MTP 与投机解码的阅读重点"
    `MTP` 是训练目标或模型结构；`speculative decoding` 是推理算法。二者可以互相增强，但不能互相替代。阅读时要同时盯三类指标：主模型质量、多步预测接受率、端到端 latency / throughput。只看 future-token accuracy 或平均 speedup 都容易误判。

## 8. 训练系统、分布式与大规模训练

对应站内章节：[训练](../training/index.md)。

### 8.1 模型并行、数据并行与优化器状态

1. [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)  
   Tensor Parallel 和大模型训练系统入口。
2. [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)  
   ZeRO optimizer state partitioning，是 DeepSpeed 路线基础。
3. [DeepSpeed: System Optimizations Enable Training Deep Learning Models with Over 100 Billion Parameters](https://arxiv.org/abs/2207.00032)  
   DeepSpeed 系统论文入口。
4. [PyTorch FSDP Documentation](https://pytorch.org/docs/stable/fsdp.html)  
   FSDP 官方文档，工程实践必看。
5. [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965)  
   Pipeline parallelism 基础。
6. [PipeDream: Generalized Pipeline Parallelism for DNN Training](https://arxiv.org/abs/1806.03377)  
   Pipeline 调度、bubble 和权重版本问题参考。

### 8.2 Attention、长序列与显存优化

1. [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)  
   IO-aware attention 的核心论文。
2. [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)  
   更实用的高性能 attention kernel。
3. [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608)  
   面向 Hopper 和低精度的 attention 演进。
4. [Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889)  
   长上下文 sequence parallel / ring attention 的重要参考。
5. [Sequence Parallelism: Long Sequence Training from System Perspective](https://arxiv.org/abs/2105.13120)  
   长序列训练并行化入口。

### 8.3 MoE、路由与大规模稀疏训练

1. [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668)  
   大规模 MoE 和自动分片入口。
2. [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961)  
   Switch Transformer 是 MoE 训练与路由常用起点。
3. [Mixtral of Experts](https://arxiv.org/abs/2401.04088)  
   开源 MoE 模型路线参考。
4. [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434)  
   DeepSeek 系 MoE、MLA 和训练成本优化参考。

## 9. 推理系统与服务运行时

对应站内章节：[推理](../inference/index.md)。

### 9.1 Serving runtime 与调度

1. [vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180)  
   PagedAttention、KV cache 管理和 continuous batching 的核心论文。
2. [vLLM GitHub](https://github.com/vllm-project/vllm)  
   当前最重要的开源 serving runtime 之一。
3. [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104)  
   面向结构化 LLM 程序、RadixAttention 和 runtime 编排。
4. [SGLang GitHub](https://github.com/sgl-project/sglang)  
   工程入口。
5. [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)  
   NVIDIA GPU 上的高性能 LLM 推理栈。
6. [LightLLM](https://github.com/ModelTC/lightllm)  
   轻量级 LLM 推理服务框架。
7. [llama.cpp](https://github.com/ggml-org/llama.cpp)  
   CPU/边缘/本地推理生态入口，`GGUF` 和量化格式很值得参考。

### 9.2 KV cache、长上下文与分层内存

1. [PagedAttention / vLLM](https://arxiv.org/abs/2309.06180)  
   KV cache 分页管理入口。
2. [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving](https://arxiv.org/abs/2401.09670)  
   Prefill/Decode 解耦服务代表。
3. [SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills](https://arxiv.org/abs/2308.16369)  
   Chunked prefill 和 decode piggybacking 参考。
4. [LMCache](https://github.com/LMCache/LMCache)  
   KV cache 复用和跨请求缓存工程入口。

## 10. 算子、Kernel 与编译器

对应站内章节：[算子与编译器](../operators/index.md)。

### 10.1 GPU 编程、GEMM 与 Attention Kernel

1. [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)  
   CUDA 编程和内存层次基础。
2. [CUTLASS](https://github.com/NVIDIA/cutlass)  
   NVIDIA GEMM、tensor core 和 CuTe 编程入口。
3. [Triton](https://github.com/triton-lang/triton)  
   高性能 Python-like GPU kernel 编程框架。
4. [FlashAttention](https://arxiv.org/abs/2205.14135)  
   attention kernel 的必读系统论文。
5. [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)  
   DeepSeek 的高性能 GEMM kernel 库，包含 FP8、FP4、MoE grouped GEMM 等方向。

### 10.2 编译器、图优化与 Roofline

1. [TVM: An Automated End-to-End Optimizing Compiler for Deep Learning](https://arxiv.org/abs/1802.04799)  
   深度学习编译器入口。
2. [MLIR: A Compiler Infrastructure for the End of Moore's Law](https://arxiv.org/abs/2002.11054)  
   多层 IR 和编译基础设施参考。
3. [Roofline: An Insightful Visual Performance Model for Multicore Architectures](https://dl.acm.org/doi/10.1145/1498765.1498785)  
   性能分析基本模型，适合判断算子是算力瓶颈还是带宽瓶颈。
4. [NVIDIA Tensor Core Evolution from Volta to Blackwell](https://newsletter.semianalysis.com/p/nvidia-tensor-core-evolution-from-volta-to-blackwell)  
   适合理解硬件数制和 tensor core 演进，但建议和官方规格交叉核对。

## 11. 对齐、后训练与评测

对应站内章节：[预训练、微调与对齐](../training/pretraining-finetuning-alignment.md)。

### 11.1 RLHF、DPO 与偏好优化

1. [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)  
   InstructGPT / RLHF 经典论文。
2. [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)  
   `DPO` 是后训练偏好优化的重要简化路线。
3. [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)  
   AI feedback 和规则化对齐入口。
4. [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2204.05862)  
   Anthropic HH-RLHF 路线参考。

### 11.2 Benchmark 与评测

1. [MMLU: Measuring Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300)  
   通用知识与多任务理解基准。
2. [GSM8K: Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168)  
   数学推理基础 benchmark。
3. [HumanEval: Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374)  
   代码生成评测常用入口。
4. [HELM: Holistic Evaluation of Language Models](https://arxiv.org/abs/2211.09110)  
   综合评测框架，适合理解模型评测覆盖面。
5. [MT-Bench / Chatbot Arena](https://arxiv.org/abs/2306.05685)  
   对话模型评测入口。

## 12. 官方工具与开源项目索引

### 12.1 训练与分布式

1. [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
2. [DeepSpeed](https://github.com/microsoft/DeepSpeed)
3. [PyTorch FSDP](https://pytorch.org/docs/stable/fsdp.html)
4. [NVIDIA Transformer Engine](https://github.com/NVIDIA/TransformerEngine)
5. [FlashAttention](https://github.com/Dao-AILab/flash-attention)

### 12.2 推理与量化

1. [vLLM](https://github.com/vllm-project/vllm)
2. [SGLang](https://github.com/sgl-project/sglang)
3. [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
4. [LightLLM](https://github.com/ModelTC/lightllm)
5. [llama.cpp](https://github.com/ggml-org/llama.cpp)
6. [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ)
7. [AutoAWQ](https://github.com/casper-hansen/AutoAWQ)
8. [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes)
9. [ONNX Runtime](https://github.com/microsoft/onnxruntime)
10. [Netron](https://github.com/lutzroeder/netron)

### 12.3 世界模型、机器人与具身

1. [Open X-Embodiment](https://robotics-transformer-x.github.io/)
2. [OpenVLA](https://github.com/openvla/openvla)
3. [Octo](https://github.com/octo-models/octo)
4. [RoboCasa](https://github.com/robocasa/robocasa)
5. [ManiSkill](https://github.com/haosulab/ManiSkill)
6. [Habitat-Lab](https://github.com/facebookresearch/habitat-lab)
7. [Isaac Lab](https://github.com/isaac-sim/IsaacLab)

## 13. 后续维护建议

1. 每次新增一个技术页时，在本页补一个对应“必读起点”和一个“工程实现入口”。
2. 对 2025-2026 年快速出现的低比特训练、投机解码、世界模型 benchmark，优先保留 arXiv / 官方 GitHub / 官方文档链接。
3. 避免把参考文献页变成无序书单：每个条目最好说明它回答什么问题。
4. 对未复现、未验证、只有宣传材料的内容，建议放到“待核验”小节，不要和已成熟基础文献混在一起。
