# 参考文献总表

这一页不是简单罗列论文名，而是把全站内容按研究问题整理成可长期维护的索引。  
阅读时建议先按主题找到入口，再回到对应章节做细读。

<!-- [补充] -->
| 读者 | 建议读法 | 工程落点 |
| --- | --- | --- |
| 初学者 | 先抓“参考文献总表”要解决的问题、关键术语和最小例子；公式只看变量方向。 | 能说清输入、输出和常见失败信号。 |
| 工程师 | 直接对照结构、指标、成本和边界条件，必要时回到前置页补概念。 | 落到数据、算力、延迟、显存、评测和回滚检查。 |


参考文献页适合查源头，不适合当第一本教材顺序读。初学者先在专题页建立问题地图，再回来找必读论文；工程读者则优先核对系统实现、benchmark、代码和复现状态。

| 读法 | 先做什么 | 关键入口 |
| --- | --- | --- |
| 10 分钟读法 | 先确认证据等级，再查具体论文 | [本站判断原则](site-judgement-principles.md)、[Claim Ledger](claim-ledger.md) |
| 深入读法 | 回到专题页核对 Figure/Table、setting 和结论边界 | [论文专题讲解](../paper-deep-dives/index.md)、[全站证据标准](evidence-and-reproducibility-standard.md) |
| 实验/工程读法 | 只摘 claim、source、evidence、repro 和 cannot prove | [全站效率技术覆盖矩阵](site-efficiency-coverage-matrix.md)、[图片与公式质量标准](image-and-formula-quality-standard.md) |

| 常见失败模式 | 先查什么 |
| --- | --- |
| 证据陈旧 | 前沿 claim 必须写核验日期和官方来源 |
| 复现状态误写 | 没有第三方证据时不写 Independent Reproduced |
| 站内推断越界 | Site Inference 只能作为路线假设，不能当论文结论 |
| 图片或公式难读 | 图源、图解、公式符号解释要贴近正文，而不是集中塞到页面末尾 |

!!! note "合并提示"
    **info · 最小验收指标**

    | 要证明什么 | 最小指标 | 不能替代什么 |
    | --- | --- | --- |
    | 引用可追踪 | 论文/报告/仓库/项目页直链和版本日期 | 站内转述 |
    | 复现状态可信 | Paper Only、Author Code、Official Demo 分开 | 官方代码当独立复现 |
    | 外推边界清楚 | 每条前沿 claim 有 Cannot Prove | 只列参考文献 |

    **failure · 最佳反例**

    引用最多不等于证据最强。一个页面列了很多论文，但没有 Figure/Table、实验设置、Repro Status 和 Cannot Prove，读者仍然无法判断 claim 能支撑到哪里。

## 如何使用这张索引

参考文献页不是学习起点，而是查证和扩展入口。建议按下面顺序使用：

| 目标 | 用法 | 回到哪里 |
| --- | --- | --- |
| 建立主题主线 | 先看每节“必读起点”，只抓问题定义和方法转折 | 对应主题总览页 |
| 做方案设计 | 找“系统 / Benchmark / 开源实现”条目，核对工程约束 | 训练、推理、量化、算子和评测页面 |
| 做论文笔记 | 从对应专题页确定精读论文，再在这里补相关工作 | [论文专题讲解](../paper-deep-dives/index.md) |
| 做复现或调研 | 优先选有代码、数据、benchmark 或官方实现的条目 | [训练评测与消融设计](../training/evaluation-and-ablation-methodology.md) |

不要把这页当成从头到尾阅读的书单。更高效的方式是：遇到具体问题时回来查源头，再把结论写回对应主题页的机制、评测或工程边界里。

## 扩散模型与生成模型

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
   更贴近 classifier-free guidance 等实际生成配置；站内专题见 [DPM-Solver++ 讲解](../paper-deep-dives/diffusion/dpm-solver-plus-plus.md)。
3. [Progressive Distillation for Fast Sampling of Diffusion Models](https://arxiv.org/abs/2202.00512)  
   少步蒸馏的经典路线：把多步 teacher 压成少步 student；也常作为理解 \(v\)-prediction / velocity 参数化的入口之一。
4. [Consistency Models](https://arxiv.org/abs/2303.01469)  
   把不同噪声水平映射到同一数据点，适合理解一步/少步生成的另一条路线。
5. [Latent Consistency Models](https://arxiv.org/abs/2310.04378)  
   `LCM` 是把 consistency 思路推向 Stable Diffusion 生态的实用论文。
6. [Improved Distribution Matching Distillation for Fast Image Synthesis](https://arxiv.org/abs/2405.14867)  
   `DMD2` 值得放在少步蒸馏里看，关注分布匹配和高质量少步生成。

### 1.3 Rectified Flow、Rectified Diffusion 与视频扩散

1. [Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003)  
   `Rectified Flow` 关注把生成路径“拉直”，降低求解难度。
2. [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)  
   `Flow Matching` 直接拟合连续生成路径上的速度场，是理解 velocity、ODE 采样和现代 flow-style 视频生成的重要论文。
3. [InstaFlow: One Step is Enough for High-Quality Diffusion-Based Text-to-Image Generation](https://arxiv.org/abs/2309.06380)  
   少步/一步生成与 rectified flow 结合的重要实例。
4. [Rectified Diffusion: Straightness Is Not Your Need in Rectified Flow](https://arxiv.org/abs/2410.07303)  
   适合补充 `Rectified Diffusion` 术语：重点不是简单追求路径直，而是重新分析整流目标、采样效率与生成质量之间的关系。
5. [Video Diffusion Models](https://arxiv.org/abs/2204.03458)  
   视频扩散的早期代表，适合作为时序生成入口。
6. [Imagen Video: High Definition Video Generation with Diffusion Models](https://arxiv.org/abs/2210.02303)  
   多级级联视频生成路线，对理解高分辨率视频生成系统很有帮助。
7. [Wan: Open and Advanced Large-Scale Video Generative Models](https://arxiv.org/abs/2503.20314)  
   `Wan2.1` 技术报告，适合理解开源视频 foundation model 的 VAE、DiT、Flow Matching、多任务和系统训练设计；站内专题见 [Wan2.1 深度讲解](../paper-deep-dives/diffusion/wan.md)。
8. [Wan2.1 官方 GitHub](https://github.com/Wan-Video/Wan2.1)  
   开源模型、推理脚本和任务接口入口，覆盖 T2V、I2V、editing、T2I 和 V2A 等任务。
9. [Wan2.2 官方 GitHub](https://github.com/Wan-Video/Wan2.2)  
   Wan2.2 的 MoE、高低噪声专家、A14B、TI2V-5B、Animate 和 S2V 等模型入口，适合连接视频扩散与世界模型专题。

## VLM/VLA 与多模态到动作模型

对应站内章节：[VLM/VLA](../vlm/index.md)。

### 2.1 视觉语言预训练与时序表征

1. [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)  
   `CLIP` 是图文对齐和后续 VLM 的基础。
2. [Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198)  
   早期强 VLM 代表，适合理解视觉编码器和语言模型之间的桥接。
3. [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/abs/2301.12597)  
   `Q-Former` 是连接视觉特征和 LLM 的经典结构。
4. [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485)  
   `LLaVA` 让开源 VLM 进入指令微调时代，是理解 VLM 数据合成和对话微调的起点。
5. [V-JEPA: Latent Video Prediction for Visual Representation Learning](https://openreview.net/forum?id=WFYbBOEOtv)
   视频 latent prediction 是从静态 VLM 走向世界模型前置状态的重要路线。
6. [PaLM-E: An Embodied Multimodal Language Model](https://arxiv.org/abs/2303.03378)
   多模态语言模型进入具身控制和机器人任务的重要节点。

## VLA、机器人策略与具身智能

对应站内章节：[VLM/VLA](../vlm/index.md)、[具身智能](../embodied-ai/index.md)。

### 3.1 机器人基础模型与 VLA

1. [Do As I Can, Not As I Say: Grounding Language in Robotic Affordances](https://arxiv.org/abs/2204.01691)  
   `SayCan` 代表“语言模型规划 + affordance 评分”的早期路线。
2. [RT-1: Robotics Transformer for Real-World Control at Scale](https://arxiv.org/abs/2212.06817)  
   把机器人策略学习推向大规模真实数据的代表工作。
3. [RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control](https://arxiv.org/abs/2307.15818)  
   `VLA` 概念的重要入口，把网页视觉语言知识迁移到机器人动作；站内专题见 [RT-2 讲解](../paper-deep-dives/embodied-ai/rt2.md)。
4. [Open X-Embodiment: Robotic Learning Datasets and RT-X Models](https://arxiv.org/abs/2310.08864)  
   多机器人、多任务数据聚合的重要数据工程论文。
5. [OpenVLA: An Open-Source Vision-Language-Action Model](https://arxiv.org/abs/2406.09246)  
   开源 VLA 代表，适合结合代码和数据做复现。
6. [Octo: An Open-Source Generalist Robot Policy](https://arxiv.org/abs/2405.12213)  
   通用机器人策略的开源入口。
7. [π0.5: a Vision-Language-Action Model with Open-World Generalization](https://arxiv.org/abs/2504.16054)  
   Physical Intelligence 的开放世界 VLA 系统，用异构 co-training、FAST 动作 token、flow matching action expert 和 high-level subtask prediction 在未见家庭中执行长任务；站内专题见 [π0.5 讲解](../paper-deep-dives/embodied-ai/pi05.md)。

### 3.2 模仿学习、动作分块与 Diffusion Policy

1. [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705)  
   `ALOHA / ACT` 方向入口，适合理解动作分块和双臂操作数据。
2. [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137)  
   把扩散模型用于连续动作策略学习的经典论文。
3. [RoboCat: A Self-Improving Generalist Agent for Robotic Manipulation](https://arxiv.org/abs/2306.11706)  
   自我改进和机器人数据闭环路线的重要参考。
4. [BridgeData V2: A Dataset for Robot Learning at Scale](https://arxiv.org/abs/2308.12952)  
   机器人策略学习的数据基础设施参考。

### 3.3 3D 感知、深度与几何底座

1. [Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data](https://arxiv.org/abs/2401.10891)
   DA 系列起点，用大规模未标注数据、strong perturbation 和 DINOv2 feature alignment 构建开放域单目深度 foundation model；站内专题见 [Depth Anything 讲解](../paper-deep-dives/embodied-ai/depth-anything.md)。
2. [Depth Anything V2](https://arxiv.org/abs/2406.09414)
   DA2 强化 synthetic teacher、pseudo-labeled real images 和模型尺度，是理解 DA3 teacher-student 路线的前置文献；站内专题见 [Depth Anything V2 讲解](../paper-deep-dives/embodied-ai/depth-anything-v2.md)。
3. [VGGT: Visual Geometry Grounded Transformer](https://arxiv.org/abs/2503.11651)
   feed-forward 3D 几何路线，直接预测 camera parameters、depth maps、point maps 和 3D point tracks；站内专题见 [VGGT 讲解](../paper-deep-dives/embodied-ai/vggt.md)。
4. [Depth Anything 3: Recovering the Visual Space from Any Views](https://arxiv.org/abs/2511.10647)
   从任意视角恢复一致 depth、ray、camera 和 3D geometry，并把 DA/DA2 单目深度能力推进到多视角几何；站内专题见 [Depth Anything 3 讲解](../paper-deep-dives/embodied-ai/depth-anything-3.md)。
5. [ByteDance-Seed/Depth-Anything-3](https://github.com/ByteDance-Seed/depth-anything-3)
   官方实现与模型仓库，包含 model zoo、Python API、CLI、benchmark pipeline 和 DA3-Streaming。

### 3.4 Benchmark 与评测

1. [RLBench: The Robot Learning Benchmark & Learning Environment](https://arxiv.org/abs/1909.12271)  
   桌面操作任务 benchmark。
2. [Meta-World: A Benchmark and Evaluation for Multi-Task and Meta Reinforcement Learning](https://arxiv.org/abs/1910.10897)  
   多任务操作策略学习常用基准。
3. [BEHAVIOR-1K: A Benchmark for Embodied AI with 1,000 Everyday Activities and Realistic Simulation](https://arxiv.org/abs/2403.09227)  
   家庭和日常活动场景的具身评测入口。
4. [CALVIN: A Benchmark for Language-Conditioned Policy Learning for Long-Horizon Robot Manipulation Tasks](https://arxiv.org/abs/2112.03227)  
   长程语言条件机器人操作常用评测。

### 3.5 具身数据与可视化工具

1. [Rerun Documentation: What is Rerun?](https://rerun.io/docs/overview/what-is-rerun)
   面向 Physical AI 的多模态时序数据层，适合记录和回放 RGB、depth、point cloud、transforms、text、scalar 和 tensor；站内专题见 [Rerun 讲解](../embodied-ai/rerun-robotics-data-visualization.md)。

## 世界模型、训练与规划

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
3. [V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning](https://arxiv.org/abs/2506.09985)  
   把 V-JEPA 表征推进到 action-conditioned planning，是 masked / JEPA 路线接回世界模型控制主线的重要论文。
4. [Learning Interactive Real-World Simulators](https://arxiv.org/abs/2310.06114)  
   `UniSim` 代表把真实世界交互建模成可模拟环境。
5. [Genie: Generative Interactive Environments](https://arxiv.org/abs/2402.15391)  
   从视频中学可交互世界模型，适合连接“视频生成模型”和“可控世界模型”。
6. [Video Language Planning](https://arxiv.org/abs/2310.10625)  
   连接视频预测、语言规划和长程任务分解。
7. [Genie 3 官方页面](https://deepmind.google/models/genie/)
   官方 demo 适合观察实时探索、长时环境一致性和文本到可交互世界的系统形态。
8. [NVIDIA Cosmos 官方页面](https://www.nvidia.com/en-us/ai/cosmos/)
   官方 demo 适合理解 world foundation models 在 physical AI、预测、推理和数据管线中的平台化定位。
9. [LingBot-World](https://arxiv.org/abs/2601.20540)
   前沿视频世界模型系统，适合关注视频底座、动作注入、因果化、长时记忆和实时交互的工程路线。
10. [LingBot-World 官方 GitHub](https://github.com/robbyant/lingbot-world)
   README 提供 Video Demo、Real-Time Interactive Demo 和推理脚本，适合把论文图和可运行系统接口对应起来。
11. [DreamZero: World Action Models are Zero-shot Policies](https://arxiv.org/abs/2602.15922)
   WAM / joint video-action prediction 路线，适合核对未来视频、动作 chunk、真实观测刷新和实时闭环控制的证据边界。

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
5. [A Comprehensive Survey on World Models for Embodied AI](https://arxiv.org/abs/2510.16732)
   用 functionality、temporal modeling 和 spatial representation 三轴梳理具身世界模型，适合作为世界模型文献地图入口；站内专题见 [具身世界模型综述](../paper-deep-dives/world-models/world-models-embodied-ai-survey.md)。

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

## 强化学习、后训练与世界模型

对应站内章节：[强化学习](../reinforcement-learning/index.md)。

### 5.1 大模型 RLVR 与后训练技术报告

1. [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)  
   用 GRPO、rule-based reward 和大规模 RLVR 训练长 CoT reasoning model 的关键技术报告；站内专题见 [DeepSeek-R1 讲解](../paper-deep-dives/technical-reports/deepseek-r1.md)。
2. [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)  
   Qwen3 把 thinking / non-thinking 双模式、thinking budget、GRPO、general RL 和 strong-to-weak distillation 合到同一个开源模型家族；站内专题见 [Qwen3 讲解](../paper-deep-dives/technical-reports/qwen3.md)。
3. [Gemini 2.5: Pushing the Frontier with Advanced Reasoning, Multimodality, Long Context, and Next Generation Agentic Capabilities](https://arxiv.org/abs/2507.06261)  
   Gemini 2.5 把 native multimodal sparse MoE、teacher distillation、TPUv5p/Pathways、SFT/RM/RL、dynamic thinking、长上下文和 agentic / multimodal 后训练合到同一条闭源前沿模型路线；站内专题见 [Gemini 2.5 讲解](../paper-deep-dives/technical-reports/gemini-2-5.md)。
4. [Kimi K2: Open Agentic Intelligence](https://arxiv.org/abs/2507.20534)  
   Kimi K2 用 MuonClip 跑稳 1T MoE 预训练，并把 agentic data synthesis、tool-use SFT、RLVR、self-critique reward 和 RL rollout 基础设施合成一条开放 agent 模型路线；站内专题见 [Kimi K2 讲解](../paper-deep-dives/technical-reports/kimi-k2.md)。
5. [Qwen3.5-Omni Technical Report](https://arxiv.org/abs/2604.15804)  
   Qwen3.5-Omni 用 Thinker-Talker、Hybrid MoE、AuT、ARIA、RVQ/MTP/Code2Wav 和 256K context 构建可流式语音交互的全模态 agent；站内专题见 [Qwen3.5-Omni 讲解](../paper-deep-dives/technical-reports/qwen35-omni.md)。

### 5.2 经典强化学习基础方法

1. [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
   `DQN` 是深度强化学习的经典起点，适合理解 value learning、replay buffer 和 target network。
2. [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)
   `A3C` 适合理解并行 actor、异步采样、entropy regularization 和现代 RL 系统中 rollout worker 的来源。
3. [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
   `GAE` 是 PPO/actor-critic 中常见的 advantage 估计方法，重点看 bias-variance tradeoff。
4. [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
   `PPO` 是现代 RLHF 和很多 actor-critic 训练栈的核心算法入口。
5. [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290)
   `SAC` 适合理解 off-policy actor-critic、maximum entropy objective、reward scale 和探索稳定性。
6. [Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems](https://arxiv.org/abs/2005.01643)
   `Offline RL` 适合理解固定数据集训练、behavior policy、support constraint、trajectory stitching 和机器人数据复用。
7. [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/abs/2106.01345)
   `Decision Transformer` 适合理解 return-conditioned trajectory modeling，以及为什么 VLA / 世界模型可以把状态、动作和回报组织成 Transformer 序列。
8. [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
   `GRPO` 的重要来源之一，适合理解组内相对奖励如何减少 critic 依赖。

### 5.3 LLM/VLM 后训练与 verl

1. [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
   `InstructGPT` 是 SFT、reward model 和 PPO/RLHF 三步流程的基础论文。
2. [HybridFlow: A Flexible and Efficient RLHF Framework](https://arxiv.org/abs/2409.19256)
   `HybridFlow / verl` 适合理解大模型 RLHF 为什么需要 dataflow、worker、rollout backend 和资源映射。
3. [A Framework for Training Large Language Models for Code Generation via Proximal Policy Optimization](https://i.cs.hku.hk/~cwu/papers/gmsheng-NL2Code24.pdf)
   适合从代码生成任务看 PPO 后训练的数据、奖励和系统约束。

### 5.4 强化学习与世界模型

1. [Learning Latent Dynamics for Planning from Pixels](https://arxiv.org/abs/1811.04551)
   `PlaNet` 是从像素学习 RSSM latent dynamics 并做 latent-space planning 的关键论文。
2. [Dream to Control: Learning Behaviors by Latent Imagination](https://arxiv.org/abs/1912.01603)
   `Dreamer` 把 world model 变成 imagined actor-critic 的训练环境。
3. [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104)
   `DreamerV3` 适合理解跨任务鲁棒 world model RL 的稳定化设计。
4. [TD-MPC2: Scalable, Robust World Models for Continuous Control](https://arxiv.org/abs/2310.16828)
   `TD-MPC2` 连接 decoder-free latent world model、TD learning 和 MPC。
5. [RLVR-World: Training World Models with Reinforcement Learning](https://arxiv.org/abs/2505.13934)
   `RLVR-World` 展示如何用可验证奖励直接优化语言和视频世界模型的任务指标。

## 量化、数制与低比特训练

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
4. [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)  
   671B MoE、MLA、MTP、FP8 mixed precision、DualPipe 和 SFT/RL 后训练的完整系统报告；站内专题见 [DeepSeek-V3 讲解](../paper-deep-dives/technical-reports/deepseek-v3.md)。
5. [COAT: Compressing Optimizer States and Activation for Memory-Efficient FP8 Training](https://arxiv.org/abs/2410.19313)  
   重点关注 optimizer states 和 activation，适合作为“低比特训练不只是压权重”的参考。
6. [Scaling FP8 Training to Trillion-Token LLMs](https://arxiv.org/abs/2409.12517)  
   讨论大规模 FP8 训练稳定性，尤其是 SwiGLU outlier 和长训练后期风险。
7. [To FP8 and Back Again: Quantifying Reduced Precision Effects on LLM Training Stability](https://arxiv.org/abs/2405.18710)  
   低精度训练稳定性评估参考，提醒不要只看短跑 loss。
8. [μnit Scaling: Simple and Scalable FP8 LLM Training](https://arxiv.org/abs/2502.05967)  
   静态缩放和参数化改造路线，适合关注“降低动态 scale 管理复杂度”的读者。
9. [Recipes for Pre-training LLMs with MXFP8](https://arxiv.org/abs/2506.08027)  
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
8. [Attn-QAT: 4-Bit Attention With Quantization-Aware Training](https://arxiv.org/abs/2603.00040)\
   把 FP4 attention 的问题推进到 forward/backward/kernel 一致性，重点看 FlashAttention backward 重算、softmax 梯度和 real quant inference kernel 如何对齐；站内专题见 [Attn-QAT 讲解](../paper-deep-dives/foundations/attn-qat.md)。
9. [SageAttention3: Microscaling FP4 Attention for Inference and An Exploration of 8-Bit Training](https://arxiv.org/abs/2505.11594)\
   FP4 attention inference 和 8-bit attention training 的重要对照，适合理解 Q/K smoothing、FP4 tensor core、attention probability quantization 与 Attn-QAT 的差异。

!!! note "低比特训练的核心阅读线"
    如果你要研究 FP8/FP4 训练，不要只按 bit 数排序。更好的阅读顺序是：先看 `FP8 Formats` 和 `Microscaling` 理解数制，再看 `FP8-LM / COAT / Scaling FP8` 理解 activation 与 optimizer 状态，再看 `MXFP8 / MXFP4 / NVFP4 / Quartet` 理解 4-bit 训练为什么必须同时改数制、scale、rounding、kernel 和稳定性评测；如果目标是 low-bit attention，还要单独看 `SageAttention3` 和 `Attn-QAT`，因为 fused attention 的 backward 与 softmax 路径会引入普通 GEMM QAT 没有的约束。

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

## MTP、投机解码与推理加速

对应站内章节：[MTP 与投机解码](../training/mtp-and-speculative-decoding.md)、[缓存、路由与投机解码](../inference/caching-routing-and-speculative.md)。

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

## 训练系统、分布式与大规模训练

对应站内章节：[训练](../training/index.md)。

### 8.1 模型并行、数据并行与优化器状态

1. [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)  
   Tensor Parallel 和大模型训练系统入口。
2. [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)  
   ZeRO optimizer state partitioning，是 DeepSpeed 路线基础。
3. [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982)  
   把矩阵正交化优化器 Muon 扩展到 5.7T token LLM 预训练，重点看 weight decay、update RMS matching、Distributed Muon 以及和 AdamW 的边界；站内专题见 [Muon 讲解](../paper-deep-dives/foundations/muon-is-scalable.md)。
4. [DeepSpeed: System Optimizations Enable Training Deep Learning Models with Over 100 Billion Parameters](https://arxiv.org/abs/2207.00032)  
   DeepSpeed 系统论文入口。
5. [PyTorch FSDP Documentation](https://pytorch.org/docs/stable/fsdp.html)  
   FSDP 官方文档，工程实践必看。
6. [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965)  
   Pipeline parallelism 基础。
7. [PipeDream: Generalized Pipeline Parallelism for DNN Training](https://arxiv.org/abs/1806.03377)  
   Pipeline 调度、bubble 和权重版本问题参考。

### 8.2 Attention、长序列与显存优化

1. [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)  
   IO-aware attention 的核心论文。
2. [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)  
   更实用的高性能 attention kernel。
3. [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608)  
   面向 Hopper 和低精度的 attention 演进。
4. [Sequence Parallelism: Long Sequence Training from System Perspective](https://arxiv.org/abs/2105.13120)\
   长序列训练并行化入口。
5. [DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models](https://arxiv.org/abs/2309.14509)\
   通过 head-sharded all-to-all 支持极长序列训练，是理解 context parallelism 和 head divisibility 约束的重要起点。
6. [Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889)\
   长上下文 sequence parallel / ring attention 的重要参考，适合和 MagiAttention 中的 zero-redundant communication 对照。
7. [USP: A Unified Sequence Parallelism Approach for Long Context Generative AI](https://arxiv.org/abs/2405.07719)\
   结合 Ulysses 与 Ring 思路的统一 sequence parallelism 路线，适合理解混合 CP 策略。
8. [LoongTrain: Efficient Training of Long-Sequence LLMs with Head-Context Parallelism](https://arxiv.org/abs/2406.18485)\
   Head-context parallelism 代表工作，适合和 Ulysses、USP、MagiAttention 一起看长序列训练的并行边界。
9. [FlexAttention: A Programming Model for Generating Optimized Attention Kernels](https://arxiv.org/abs/2412.05496)\
   PyTorch 灵活 attention kernel 路线，适合理解“可表达复杂 mask”和“保持 kernel 性能”之间的工程张力。
10. [FlashMask: Efficient and Rich Mask Extension of FlashAttention](https://arxiv.org/abs/2410.01359)\
   面向 rich mask 支持的 FlashAttention 扩展，是 MagiAttention 相关 mask-flexible kernel 讨论的重要背景。
11. [MagiAttention: A Distributed Attention Towards Linear Scalability for Ultra-Long Context, Heterogeneous Mask Training](https://sandai-org.github.io/MagiAttention/docs/main/blog/magi_attn.html)\
   SandAI 的分布式 attention 系统博客，重点看 AttnSlice、FFA、dispatch solver、GroupCast/GroupReduce 和 multi-stage overlap；站内专题见 [MagiAttention 讲解](../paper-deep-dives/foundations/magiattention.md)。
12. [Long-Context Attention Benchmark](https://sandai-org.github.io/MagiAttention/docs/main/blog/cp_benchmark.html)\
   MagiAttention 官方 CP benchmark，提供 H100/B200、varlen causal、GQA、TFLOPs/s/GPU 和 distributed baselines 的对照设置。
13. [SLA: Beyond Sparsity in Diffusion Transformers via Fine-Tunable Sparse-Linear Attention](https://arxiv.org/abs/2509.24006) / [SLA2: Sparse-Linear Attention with Learnable Routing and QAT](https://arxiv.org/abs/2602.12675)\
   视频 DiT attention 加速路线，把 sparse exact attention、linear compensation、learnable routing 和 QAT 接到同一条工程链路；站内专题见 [SLA / SLA2 讲解](../paper-deep-dives/foundations/sla-sparse-linear-attention.md)。
14. [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)\
   Selective SSM 和 hardware-aware scan 的核心论文；站内基础讲解见 [Mamba 与混合 SSM 架构](../foundations/mamba-and-hybrid-ssm-transformers.md)。
15. [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060)\
   Mamba-2 和 state space duality 入口，适合理解 attention 与 SSM 的统一视角。
16. [Jamba: A Hybrid Transformer-Mamba Language Model](https://arxiv.org/abs/2403.19887) / [Jamba-1.5](https://arxiv.org/abs/2408.12570)\
   混合 Transformer-Mamba-MoE 架构代表，重点看 layer ratio、MoE 和长上下文部署。
17. [Nemotron-H: A Family of Accurate and Efficient Hybrid Mamba-Transformer Models](https://arxiv.org/abs/2504.03624)\
   NVIDIA 的 hybrid Mamba-Transformer family，重点看以推理成本为目标的架构替换、蒸馏压缩和 FP8 训练 recipe。
18. [Mamba-3: Improved Sequence Modeling using State Space Principles](https://arxiv.org/abs/2603.15569)\
   2026 年的 SSM 改进工作，关注 state tracking、complex-valued state update 和 MIMO formulation。

### 8.3 MoE、路由与大规模稀疏训练

1. [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668)  
   大规模 MoE 和自动分片入口。
2. [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961)  
   Switch Transformer 是 MoE 训练与路由常用起点。
3. [Mixtral of Experts](https://arxiv.org/abs/2401.04088)  
   开源 MoE 模型路线参考。
4. [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434)  
   DeepSeek 系 MoE、MLA 和训练成本优化参考。
5. [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)  
   DeepSeek-V3 把 auxiliary-loss-free load balancing、node-limited routing、64-way EP、DualPipe 和 FP8 训练放到 671B MoE 规模；站内专题见 [DeepSeek-V3 讲解](../paper-deep-dives/technical-reports/deepseek-v3.md)。
6. [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)  
   Qwen3 同时发布 dense 与 MoE 模型，MoE 采用 128 total experts / 8 activated experts，并用 global-batch load balancing loss；站内专题见 [Qwen3 讲解](../paper-deep-dives/technical-reports/qwen3.md)。
7. [Kimi K2: Open Agentic Intelligence](https://arxiv.org/abs/2507.20534)  
   Kimi K2 采用 1.04T total / 32.6B activated 的 ultra-sparse MoE，使用 384 total experts / 8 active experts，并通过 sparsity scaling law 选择 sparsity 48；站内专题见 [Kimi K2 讲解](../paper-deep-dives/technical-reports/kimi-k2.md)。

### 8.4 实验追踪与训练治理工具

1. [Weights & Biases](https://wandb.ai/site)
   AI developer platform，覆盖 experiment tracking、artifacts、tables、sweeps、reports、registry 等训练治理入口；站内专题见 [W&B 讲解](../training/wandb-experiment-tracking-and-governance.md)。
2. [W&B Experiments](https://docs.wandb.ai/models/track)
   记录 metrics、hyperparameters、system metrics 和 model artifacts 的官方文档入口。
3. [W&B Artifacts](https://docs.wandb.ai/models/artifacts)
   用 run 的输入/输出追踪和版本化数据集、模型与评测产物。

## 推理系统与服务运行时

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
5. [KVSlimmer: Theoretical Insights and Practical Optimizations for Asymmetric KV Merging](https://arxiv.org/abs/2603.00907)
   用 Q/K/V 投影矩阵谱分布解释 KV 非对称，并把 Key merge 的 exact Hessian coupling 化简成 forward-only 闭式权重；站内专题见 [KVSlimmer 讲解](../paper-deep-dives/inference/kvslimmer.md)。

## 算子、Kernel 与编译器

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
5. [FlashAttention-4 / CuTe implementation](https://github.com/Dao-AILab/flash-attention/tree/main/flash_attn/cute)\
   FlashAttention 在 CuTeDSL / Blackwell 方向的实现入口，适合理解 MagiAttention `FFA_FA4` backend 的上游生态。
6. [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)\
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

## 对齐、后训练与评测

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

## 官方工具与开源项目索引

### 12.1 训练与分布式

1. [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
2. [DeepSpeed](https://github.com/microsoft/DeepSpeed)
3. [PyTorch FSDP](https://pytorch.org/docs/stable/fsdp.html)
4. [NVIDIA Transformer Engine](https://github.com/NVIDIA/TransformerEngine)
5. [FlashAttention](https://github.com/Dao-AILab/flash-attention)
6. [MagiAttention](https://github.com/SandAI-org/MagiAttention)\
   分布式 attention / context parallel backend，适合追踪 FFA、dispatch solver、native group collectives 和 Megatron/FSDP 集成。
7. [DeepEP](https://github.com/deepseek-ai/DeepEP)\
   Expert-parallel communication library，也适合作为 group communication、RDMA transfer de-duplication 和层次化通信优化的实现参考。

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
11. [SageAttention](https://github.com/thu-ml/SageAttention)\
   attention 低比特推理 kernel 入口，适合和 Attn-QAT 的 training-aware attention quantization 对照。

### 12.3 世界模型、机器人与具身

1. [Open X-Embodiment](https://robotics-transformer-x.github.io/)
2. [OpenVLA](https://github.com/openvla/openvla)
3. [Octo](https://github.com/octo-models/octo)
4. [RoboCasa](https://github.com/robocasa/robocasa)
5. [ManiSkill](https://github.com/haosulab/ManiSkill)
6. [Habitat-Lab](https://github.com/facebookresearch/habitat-lab)
7. [Isaac Lab](https://github.com/isaac-sim/IsaacLab)

### 12.4 Qualcomm 端侧 AI 与 XR 工具链

1. [Qualcomm AI Hub Workbench Documentation](https://workbench.aihub.qualcomm.com/docs/)  
   官方云端编译、量化、profile、inference 和部署文档；站内调研见 [Qualcomm XR2 Gen 2 / QCS8450](../thinking-exploration/qualcomm-xr2-gen2.md)。
2. [Qualcomm AI Hub](https://aihub.qualcomm.com/)  
   Qualcomm 端侧模型、Workbench 和应用入口，适合查模型兼容性和设备 profile。
3. [Qualcomm AI Hub Models](https://github.com/qualcomm/ai-hub-models)  
   开源模型库，包含导出脚本、scorecard、perf/numerics 数据和设备目标。
4. [QIDK](https://github.com/quic/qidk)  
   QNN、SNPE、AIMET 和 Android 示例仓库，适合看本地部署和模型 enablement。
5. [Snapdragon XR2 Gen 2 Platform Product Brief](https://docs.qualcomm.com/doc/87-73689-1/)  
   XR2 Gen 2 官方产品简介，包含显示、相机、AI、CPU、GPU、连接和安全规格。

## 后续维护建议

1. 每次新增一个技术页时，在本页补一个对应“必读起点”和一个“工程实现入口”。
2. 对 2025-2026 年快速出现的低比特训练、投机解码、世界模型 benchmark，优先保留 arXiv / 官方 GitHub / 官方文档链接。
3. 避免把参考文献页变成无序书单：每个条目最好说明它回答什么问题。
4. 对未复现、未验证、只有宣传材料的内容，建议放到“待核验”小节，不要和已成熟基础文献混在一起。

!!! tip "下一站"
    - 按导航顺序继续：[本站判断原则](site-judgement-principles.md)。
