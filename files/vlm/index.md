# VLM/VLA 路线图：从看懂到行动

本章沿着一条通往世界模型的主线展开：先让模型把视觉和语言对齐，再把视觉压成可用状态，随后把状态接到动作策略，最后自然过渡到动作条件的未来预测。

!!! tip "基础知识入口"
    这条线会反复用到 `Embedding`、`Cross-Attention`、视频 token、动作 token、行为克隆、动作块和闭环评测。第一次读可以先看 [Transformer 与 Attention](../foundations/transformer-attention-and-tokenization.md)、[概率与潜变量模型](../foundations/probability-latent-variables-and-generative-models.md) 和 [优化与训练入门](../foundations/optimization-and-training-basics.md)。

## 1. 主线问题

VLM 通常回答：

\[
p_\theta(y \mid x_v, x_t)
\]

也就是给定视觉输入 \(x_v\) 和文本输入 \(x_t\)，输出回答、框、标签或工具调用。VLA 进一步回答：

\[
\pi_\theta(a_t \mid o_{\le t}, l)
\]

也就是给定历史观测 \(o_{\le t}\) 和语言指令 \(l\)，输出动作 \(a_t\)。世界模型再往前走一步，回答：

\[
p_\theta(o_{t+1:t+H}, r, d \mid o_{\le t}, a_{t:t+H-1}, l)
\]

也就是预测动作之后的未来观测、奖励、终止、风险或其他可用于规划的信号。

| 层级 | 核心问题 | 本章对应内容 |
| --- | --- | --- |
| VLM | 当前视觉和语言如何对齐 | CLIP、BLIP-2、视觉连接器 |
| 视频表征 | 视觉状态如何带上时间和运动 | V-JEPA、长视频/latent 表征 |
| VLA | 视觉语言状态如何变成动作 | RT-1/RT-2/RT-X、动作 token、行为克隆 |
| 闭环策略 | 出错后如何恢复，动作如何安全执行 | action chunk、Diffusion Policy、ACT、恢复与安全 |
| 世界模型 | 动作会让未来如何分叉 | DreamZero、LingBot-World、WAM/VAM |

!!! note "先把 VLM/VLA 放进世界模型路线里"
    本章不是把所有 VLM 应用逐个展开，而是沿着“视觉语义 -> 时序状态 -> 动作策略 -> 动作后果预测”推进。判断一项内容是否该放在这里，标准不是它是否热门，而是它是否能帮助回答：模型如何从看懂当前画面，走向在动作条件下预测未来。

## 从论文专题回填的多模态知识点

论文专题把 VLM/VLA 从静态图文对齐推进到了实时语音、全模态 agent 和动作世界模型。回看本章时，可以按下面几条线补知识点：

| 专题来源 | 多模态 / VLA 层面的补充 | 应接到本章哪里 |
| --- | --- | --- |
| [GPT-4o System Card](../paper-deep-dives/technical-reports/gpt-4o-system-card.md) | 端到端 omni model 让 audio 不再只是 ASR/TTS 外挂；语音身份、声音一致性、实时安全分类器也变成模型系统接口 | [VLM 架构：视觉表征、连接器与记忆](architecture-and-training.md)、[推理服务系统](../inference/serving-systems.md) |
| [Qwen3.5-Omni](../paper-deep-dives/technical-reports/qwen35-omni.md) | Thinker-Talker、AuT、ARIA、RVQ/MTP/Code2Wav 说明全模态 agent 常需要把理解、规划、语音生成和流式输出拆成协同模块 | [VLM 架构：视觉表征、连接器与记忆](architecture-and-training.md)、[VLA 闭环恢复与失败分析](../vla/closed-loop-recovery-and-failure-analysis.md) |
| [π0.5](../paper-deep-dives/embodied-ai/pi05.md) | heterogeneous co-training、FAST 动作离散化、flow matching action expert 和 high-level subtask prediction 说明开放世界 VLA 要同时吃 web 语义、机器人数据和任务分解信号 | [数据与策略学习](../vla/data-and-policy-learning.md)、[动作表示与控制接口](../vla/action-representation-and-interfaces.md) |
| [DreamZero](../paper-deep-dives/embodied-ai/dreamzero.md)、[LingBot-World](../paper-deep-dives/world-models/lingbot-world.md) | 动作不只是 policy 输出，也可以作为未来视频和世界状态的条件变量；这把 VLA 推向 WAM/VAM | [动作分块与层级策略](../vla/action-chunking-hierarchical-policies-and-latent-skills.md)、[世界模型路线图](../world-models/index.md) |

## 2. VLM：先建立静态语义空间

![CLIP pre-training and zero-shot transfer](../assets/images/paper-figures/contrastive-learning/clip-figure-1.png){ width="830" }

<small>图源：[Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)，Figure 1。原论文图意：用图像编码器和文本编码器把图文对拉到同一表示空间，并把类别文本 prompt 当作 zero-shot 分类器。</small>

!!! note "图解：CLIP 是静态对齐，不是世界动态"
    CLIP 把图片和文本拉到同一 embedding 空间，让模型知道“这张图和这句话是否匹配”。这对 VLM 是起点：没有语义对齐，就谈不上语言指令和视觉状态交互。但 CLIP 只处理当前图文相似性，不知道对象如何运动，也不知道动作会造成什么后果。

## 3. 连接器：把视觉压成可消费状态

![BLIP-2 Q-Former objectives](../assets/images/paper-figures/vlm/blip2-figure-2-qformer-objectives.jpg){ width="920" }

<small>图源：[BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/abs/2301.12597)，Figure 2。原论文图意：冻结图像编码器，用查询式连接模块抽取视觉信息，并分别服务 image-text contrastive learning、image-grounded text generation 和 image-text matching。</small>

!!! note "图解：Q-Former 的关键是信息瓶颈"
    原始图像 patch 太多，LLM 不能无限接收。Q-Former 用少量查询 token 从视觉特征中抽取任务相关信息。这个问题会一直延续到世界模型：你必须决定哪些视觉信息进入状态 \(z_t\)，哪些细节可以丢掉。VLM 按当前语言任务压缩视觉，世界模型则要按未来可预测性和决策价值压缩视觉。

## 4. 视频表征：从一张图走向状态

世界模型需要的不是“图像 caption”，而是可持续更新的状态。视频自监督表征是从 VLM 走向世界模型的关键中间层。

![V-JEPA overview](../assets/images/paper-deep-dives/world-models/v-jepa/figure-1-overview.png){ width="920" }

<small>图源：[V-JEPA: Latent Video Prediction for Visual Representation Learning](../paper-deep-dives/world-models/v-jepa.md)，Figure 1。原论文图意：V-JEPA 在视频 latent space 中预测 masked spatio-temporal regions 的目标表示，用 frozen evaluation 验证 motion-based 与 appearance-based 任务。</small>

!!! note "图解：V-JEPA 为什么比普通 VLM 更靠近世界模型"
    V-JEPA 不重建像素，而是在 latent space 预测被遮挡的时空区域表示。它逼模型学习对象、运动和上下文连续性。它还不是完整世界模型，因为缺少动作条件、奖励和规划接口；但它回答了世界模型的前置问题：怎样得到一个能表达视频动态的视觉状态。

## 5. VLA：把视觉语言状态接到动作

![RT-1-X and RT-2-X action token interface](../assets/images/paper-figures/vla/openx-rt1x-rt2x-model.jpg){ width="920" }

<small>图源：[Open X-Embodiment: Robotic Learning Datasets and RT-X Models](https://arxiv.org/abs/2310.08864)，Figure 2。原论文图意：RT-1-X 与 RT-2-X 都以图像和文本指令为输入并输出离散 end-effector actions；RT-2-X 把动作表示成另一种语言 token，与视觉语言数据一起训练。</small>

!!! note "图解：动作 token 是 VLM 到 VLA 的接口"
    RT-X 图展示了一个直接桥接方式：输入仍是图像和语言，输出变成离散化的末端执行器动作。这样可以复用 Transformer 的序列建模能力，也方便把机器人数据和视觉语言数据放到同一训练范式里。代价是连续控制被量化，所以还需要低层控制器、动作平滑和安全过滤。

VLA 章节保留的重点是四件事：

| 重点 | 为什么保留 |
| --- | --- |
| [数据与策略学习](../vla/data-and-policy-learning.md) | 行为克隆、covariate shift 和动作生成是 VLA 的基础 |
| [动作表示与控制接口](../vla/action-representation-and-interfaces.md) | 动作 token、连续控制、坐标系决定策略能否执行 |
| [动作分块与层级策略](../vla/action-chunking-hierarchical-policies-and-latent-skills.md) | 真实机器人通常需要一段短技能，而不是单步动作 |
| [闭环恢复与安全](../vla/closed-loop-recovery-and-failure-analysis.md) | VLA 的价值在于执行后还能根据反馈修正 |

常见模型可以按“离世界模型有多近”这样放：

| 模型族 | 它主要解决什么 | 到世界模型还缺什么 |
| --- | --- | --- |
| CLIP / ALIGN | 图文语义对齐，建立静态 embedding 空间 | 时间、动作和未来预测 |
| BLIP-2 / Flamingo | 让语言模型消费视觉信息并进行多模态回答 | 动作接口和闭环反馈 |
| V-JEPA / video latent prediction | 从视频中学时序表征和可预测 latent | 动作条件、reward、risk 和 planner 接口 |
| RT-1 / RT-2 / RT-X / OpenVLA | 把视觉语言状态映射到机器人动作 | 显式建模“动作之后会怎样” |
| Diffusion Policy / ACT / π0 / π0.5 | 用 action chunk、扩散/flow 或连续动作专家生成更稳定动作，并开始引入高层子任务推断 | 反事实未来、风险校准和长期记忆 |
| DreamZero / WAM / VAM | 联合建模未来视频与动作 | 真实闭环收益、失败恢复和安全评测仍需验证 |

## 6. 从 VLA 到世界模型：动作后果必须被建模

VLA 主要学习“当前应该做什么”。世界模型还要学习“做了以后会怎样”。这两者的区别很关键：

| 问题 | VLA / policy | 世界模型 / dynamics |
| --- | --- | --- |
| 输出 | 动作、动作块、技能调用 | 未来状态、奖励、终止、风险 |
| 训练信号 | 示范动作、成功轨迹、偏好或奖励 | 下一状态、未来视频、反事实、risk label |
| 主要失败 | 动作偏、接口不稳、闭环恢复差 | 未来不对动作敏感、rollout 漂移、风险低估 |
| 典型消费方 | 控制器、安全层、机器人执行器 | planner、policy、数据引擎、风险模块 |

![Joint video and action prediction](../assets/images/paper-deep-dives/world-models/dreamzero/figure-2-joint-video-action.png){ width="920" }

<small>图源：[DreamZero: World Action Models are Zero-shot Policies](../paper-deep-dives/embodied-ai/dreamzero.md)，Figure 2。原论文图意：展示 DreamZero 同时生成未来视频和动作，并强调预测动作与生成视频在未见任务上保持对齐。</small>

!!! note "图解：DreamZero 展示了 VLA 和世界模型的合流点"
    普通 VLA 可以只预测动作，普通视频模型可以只预测未来画面。DreamZero 把未来视频和未来动作一起建模，让动作必须解释视觉变化，视觉预测也必须对动作敏感。这就是通向 WAM/VAM 的关键：动作不再只是最终输出，而是未来世界分叉的条件变量。

## 7. 技术博客视角：几条代表路线怎么接起来

论文图能解释模型结构，技术博客和项目页更适合观察系统目标。把几个代表系统放在一起看，可以更清楚地看到 VLM/VLA 如何过渡到世界模型。

| 系统 | 公开材料中的定位 | 对本章的启发 |
| --- | --- | --- |
| [OpenVLA](https://openvla.github.io/) | 开源 VLA generalist policy，展示真实机器人 rollout、语言 grounding、OOD 任务和失败恢复案例 | VLA 的第一步是把 VLM 表示变成可执行策略；OpenX 预训练能提升跨任务适应，但困难语义泛化仍依赖更强互联网预训练和机器人数据混合 |
| [π0](https://www.physicalintelligence.company/blog/pi0) | Physical Intelligence 的 generalist robot policy，把互联网规模 VLM 预训练、多机器人数据和连续动作输出结合 | 说明 VLA 不只是“动作 token 化”：高频 dexterous control 可能更适合连续动作专家或 flow matching / diffusion 式动作生成 |
| [π0.5](../paper-deep-dives/embodied-ai/pi05.md) | Physical Intelligence 的 open-world VLA，用 heterogeneous co-training 和 high-level subtask prediction 在未见家庭中执行长任务 | 说明开放世界泛化需要把 web 语义、跨机器人动作数据、目标机器人数据和人类 verbal instruction 合到同一训练配方 |
| [Genie 3](https://deepmind.google/models/genie/) | Google DeepMind 将其定位为可实时探索的 interactive world model | 这条路线从可交互世界生成出发，重点是实时性、动作影响、世界一致性和长时记忆 |
| [NVIDIA Cosmos](https://www.nvidia.com/en-us/ai/cosmos/) | 面向 physical AI 的 world foundation model 平台，包含世界生成、视频数据处理、评估和 post-training 框架 | 说明世界模型不是单个视频模型，而会变成数据处理、合成数据、评估、后训练和仿真工具链 |

!!! note "难点解释：为什么 OpenVLA / π0 还不是完整世界模型"
    OpenVLA 和 π0 这类系统主要输出动作，它们的核心评价是动作是否完成任务、是否能泛化和恢复。Genie 3、Cosmos、LingBot-World 这类路线则更强调生成或模拟未来世界。真正的世界模型接口要把两者接起来：动作既是策略输出，也是未来预测的条件；未来预测不仅给人看，还要被 planner、policy、risk module 或数据引擎消费。

!!! example "有趣例子：从会开车到会预判交通"
    一个 VLA 像新手司机：看到红灯会刹车，看到行人会避让。世界模型像有经验的司机：它会预判旁车是否要变道、雨天刹车距离是否变长、如果现在加速会不会卡在路口。前者强调“现在做什么”，后者强调“做了以后会发生什么”。

## 8. 推荐阅读顺序

如果目标是理解世界模型，建议这样读：

1. [VLM 架构：视觉表征、连接器与记忆](architecture-and-training.md)：理解 CLIP/InfoNCE 静态对齐、视觉 token、连接器和视频 latent。
2. [数据与策略学习](../vla/data-and-policy-learning.md)：理解 VLA 为什么首先是策略学习问题。
3. [动作表示与控制接口](../vla/action-representation-and-interfaces.md)：理解动作如何被 token 化、归一化和执行。
4. [动作分块、层级策略与潜在技能](../vla/action-chunking-hierarchical-policies-and-latent-skills.md)：理解短技能和动作 chunk。
5. [VLA 闭环恢复与失败分析](../vla/closed-loop-recovery-and-failure-analysis.md)：理解策略如何进入真实闭环。
6. [强化学习总览](../reinforcement-learning/index.md)：理解 reward、policy、PPO/GRPO 和 imagined rollout。
7. [世界模型路线图](../world-models/index.md)：进入动作条件未来预测、imagined rollout 和风险规划。

读完本章后，判断是否可以进入世界模型，只看一个问题：你是否能清楚区分“模型输出动作”和“模型预测动作后果”。
