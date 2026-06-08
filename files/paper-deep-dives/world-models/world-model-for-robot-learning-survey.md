---
paper_date: 2026-04
paper_title: "World Model for Robot Learning: A Comprehensive Survey"
paper_topic: 世界模型
---

# World Model for Robot Learning：机器人学习世界模型综述

!!! info "论文信息"
    **论文题名：** World Model for Robot Learning: A Comprehensive Survey。<br>
    **作者：** Bohan Hou、Gen Li、Jindou Jia、Tuo An、Xinying Guo、Sicong Leng、Haoran Geng、Yanjie Ze、Tatsuya Harada、Philip Torr、Oier Mees、Marc Pollefeys、Zhuang Liu、Jiajun Wu、Pieter Abbeel、Jitendra Malik、Yilun Du、Jianfei Yang。<br>
    **机构：** Nanyang Technological University、University of California, Berkeley、Stanford University、The University of Tokyo、University of Oxford、Microsoft、ETH Zurich、Princeton University、Harvard University。<br>
    **时间 / 主题：** 2026-04；世界模型、机器人学习、VLA、视频世界模型、learned simulator。<br>
    **arXiv / 官方报告：** arXiv：[2605.00080](https://arxiv.org/abs/2605.00080)；PDF：[arxiv.org/pdf/2605.00080](https://arxiv.org/pdf/2605.00080)。<br>
    **GitHub / 项目：** GitHub：[NTUMARS/Awesome-World-Model-for-Robotics-Policy](https://github.com/NTUMARS/Awesome-World-Model-for-Robotics-Policy)；项目页：[ntumars.github.io/wm-robot-survey](https://ntumars.github.io/wm-robot-survey/)。<br>
    **元数据来源与核验口径：** 来源：[arXiv](https://arxiv.org/abs/2605.00080)、论文 PDF、官方 GitHub / 项目页；Checked Date：2026-06-08；Repro Status：Survey / official resource list reviewed, independent reproduction not claimed。

这篇不是提出一个新模型，而是把 2023 到 2026 年快速膨胀的机器人世界模型论文整理成一张“政策学习中心”的地图。它最有用的地方，是把世界模型从一个容易泛化过头的词，重新压回三个机器人问题：

1. 世界模型怎样和 robot policy 耦合；
2. 世界模型怎样作为 learned simulator 参与 RL、评估和规划；
3. 视频生成模型怎样逐步变成 action-controllable、structure-aware、foundation-scale 的 robotic video world model。

如果读 LingBot-World 时你关心的是“一个视频底座怎么改造成交互世界模拟器”，读这篇综述时要关心的是“这些路线在机器人学习里分别占哪一格”。它不是 recipe，却很适合作为读新论文的坐标系。

## 它的贡献是什么

| 维度 | 贡献 |
| --- | --- |
| 节省的成本 | 节省的是阅读、选型和评测成本。它不提供单一训练加速方法，而是把 policy、simulator、video WM、dataset、benchmark 的证据链拆开 |
| 核心机制 | 用 policy-centric taxonomy 组织文献：world model for policy、world model as simulator、world model for robotic video generation |
| 对世界模型主线的意义 | 明确区分“视觉上合理的未来”和“对动作、评估、规划有用的未来” |
| 主要风险 | 综述表格不是统一复现实验；不同论文 benchmark、数据、动作空间和闭环协议差异很大，不能按表格直接排总榜 |
| 应接到本站哪里 | [世界模型评测与失效模式](../../world-models/evaluation-and-failure-modes.md)、[WM / WAM / VAM：动作条件建模](../../world-models/wm-wam-vam-and-action-conditioned-modeling.md)、[VLA、WAM 与世界模型系统图](../../embodied-ai/vla-wam-world-model-system-map.md) |

## 证据等级与外推边界

这篇综述的强证据来自分类框架、原论文归纳表、数据集属性表和 benchmark 汇总；弱证据在于它不重新跑实验，也不统一所有模型的训练条件。因此读它时要把“survey conclusion”和“single model result”分开。

| 论文结论 | 证据来源 | 证据等级 | 可外推到世界模型高效训练 | 不能直接外推 |
| --- | --- | --- | --- | --- |
| 机器人世界模型应以 action usefulness 为中心 | 定义、Figure 1、evaluation discussion | Survey synthesis | 评测不能只看视频质量，要看 action sensitivity、planning utility、policy ranking | 不能证明某个具体世界模型已经可靠控制真实机器人 |
| policy coupling 已从 decoupled 走向 unified / MoE / latent | Figure 2、Table 1、Sec. 3 方法族 | Literature taxonomy | 选型时先问未来预测是否在 inference 激活、动作和视频是否共享 backbone | 不能说明 shared backbone 一定优于 decoupled |
| learned simulator 正在从训练环境扩展到评估器 | Figure 5、Sec. 4 | Functional taxonomy | 世界模型可以提供 reward、termination、policy ranking、candidate filtering | simulator hallucination 会污染 reward 和评估信号 |
| robotic video WM 的重点从 imagination 转向 controllable foundation interface | Figure 6、Table 2、Sec. 5 | Literature taxonomy | 视频底座训练要补动作、结构、长时一致和交互数据 | 不能把普通 text-to-video 能力等同于机器人世界模型 |
| 数据集要按 transition richness 而不是规模单独判断 | Dataset tables、Sec. 7 | Survey synthesis | 训练数据要记录动作、语言、3D/multiview、接触和长 horizon | 表格没有统一数据质量审计，也不等于推荐排序 |

## 论文位置

本站已有一篇 [Embodied World Model Survey](world-models-embodied-ai-survey.md)，它更像具身世界模型的大范围地图，覆盖 POMDP、latent dynamics、3D/空间表示和具身智能系统。`World Model for Robot Learning` 更偏机器人 policy：它把 VLA、video world model、world model as simulator 和 robot benchmark 放到一起。

这篇综述的核心判断是：在机器人里，世界模型不是“知道世界知识”的大模型，也不是“会生成未来视频”的模型，而是一个能让动作选择更可靠的预测结构。它可以输出未来图像、latent state、reward、termination、value、progress、risk 或符号状态；但这些输出必须服务 control、planning、simulation、evaluation 或 data generation。

## 定义：什么才算机器人世界模型

论文采用一个宽但不泛的定义：

$$
p(x_{t+1:t+H} \mid x_t, a_{t:t+H-1}, l)
$$

这里 \(x_t\) 是被建模的状态，可以是视觉观测、latent state、结构化物理状态或符号状态；\(a_{t:t+H-1}\) 是低层动作序列；\(l\) 是语言指令或高层任务目标。论文特别把语言也视为 high-level action，因为语言指定“未来应该实现什么”。

如果状态直接用视频观测表示，公式变成：

$$
p(v_{t+1:t+H} \mid o_t, a_{t:t+H-1}, l)
$$

这解释了为什么最近很多视频生成模型会被放进世界模型讨论里。关键不在于它生成视频，而在于这个视频是否 action-conditioned、long-horizon consistent、physically plausible，并且能被 policy、planner 或 evaluator 消费。

**最容易误读的一点。**
一个模型生成的未来看起来真实，并不意味着它是机器人世界模型。机器人关心的是同一历史下换一个动作，未来是否发生可解释的分叉；如果模型对动作不敏感，它只是强视频先验，不是可用于控制的世界模型。

## Figure 1：综述组织结构

![World Model for Robot Learning survey organization 原论文图](../../assets/images/paper-deep-dives/world-models/world-model-robot-learning-survey/figure-1-survey-organization.png){ width="920" }

<small>图源：`World Model for Robot Learning: A Comprehensive Survey`，Figure 1。原论文图意：整篇综述按三个主轴组织：Section 3 讨论 world models 如何和 robot policies 耦合；Section 4 讨论 world models 作为 simulators；Section 5 讨论 robotic video world models 的能力演进。</small>

**这张图要按功能读。**
上半部分是 world model for policy：模型可能接收 instruction、language、observation，然后通过 future observation 或 internal predictive state 辅助 action。中间是 world model as environment：policy 给动作，world model rollout 并返回 evaluation / reward。下半部分是 robotic video：从 language-conditioned 和 action-conditioned 的视频未来，发展成更可控、更结构化的世界模型。

它支撑的是分类框架，不是证明路线优劣。读图时最重要的是三种“使用方式”不同：policy coupling 是把预测放进动作生成，simulator 是把预测放进交互环境，robotic video WM 是把视频生成变成可控未来接口。

## Figure 2：时间演化

![Temporal evolution of representative works 原论文图](../../assets/images/paper-deep-dives/world-models/world-model-robot-learning-survey/figure-2-temporal-evolution.png){ width="920" }

<small>图源：`World Model for Robot Learning: A Comprehensive Survey`，Figure 2。原论文图意：上支展示 world model for policy 从 IDM-style 走向 single-backbone、MoE/MoT、unified VLA 和 latent world modeling；下支展示 world model as simulator 从 validation / candidate evaluation 走向 RL、post-training 和 policy-world-model co-optimization。</small>

**时间线不是替代关系。**
图里“后出现”的方法不一定淘汰前面的 decoupled 方法。比如 IDM-style 的 `predict future -> inverse dynamics -> action` 仍然很清楚、可解释、模块化；single-backbone 和 MoE/MoT 的价值在于减少未来预测和动作生成之间的 representational gap；latent world modeling 则试图跳过昂贵像素生成，直接预测更适合控制的未来表示。

这张图真正有用的是提示两个趋势：第一，世界模型越来越贴近 action generation；第二，世界模型越来越像一个可训练、可评估、可修正的 learned environment，而不是离线生成器。

## 主线一：World Model for Policy

论文把 world-model-based policies 看成对同一个联合分布的不同查询：

$$
p(o_{t+1:t+k}, a_{t+1:t+k} \mid o_t, l)
$$

由这个联合分布可以得到：

| Query | Formulation | 读法 |
| --- | --- | --- |
| Policy Model | \(p(a_{t+1:t+k}\mid o_t,l)\) | 直接问下一段动作是什么 |
| Passive World Model | \(p(o_{t+1:t+k}\mid o_t,l)\) | 只按历史和任务想象未来 |
| Controllable World Model | \(p(o_{t+1:t+k}\mid o_t,a_{t+1:t+k})\) | 给定候选动作，预测后果 |
| Inverse Dynamics Model | \(p(a_{t+1:t+k}\mid o_{t:t+k})\) | 给定状态转变，反推动作 |

这套视角解释了为什么 policy 和 world model 会耦合：policy 不是只从当前观测直接回归动作，也可以先引入未来观测、未来 latent、value 或 progress 作为辅助变量。

### Table 1：Architectural paradigms for world-model-based policies

| Paradigm | Representative Work | Future Generation at Inference | Backbone | Coupling Style |
| --- | --- | --- | --- | --- |
| IDM-style | UniPi | Explicit video rollout | VGM | Decoupled |
| IDM-style | VidMan | Explicit video rollout | VGM | Decoupled |
| IDM-style | Vidar | Explicit video rollout | VGM | Decoupled |
| IDM-style | Gen2Act | Explicit human-video rollout | VGM | Decoupled |
| IDM-style | VPP | Latent predictive features | VGM | Decoupled |
| IDM-style | Video2Act | Latent predictive features | VGM | Decoupled |
| IDM-style | MimicVideo | Latent visual plan | VGM | Decoupled |
| IDM-style | TC-IDM | Structured execution plan | VGM | Decoupled |
| IDM-style | LVP | Visual plan | VGM | Decoupled |
| IDM-style | Say-Dream-ACT | Video prompt | VGM | Decoupled |
| Single-backbone | UVA | Joint latent prediction | VGM | Shared backbone |
| Single-backbone | UWA | Joint diffusion process | VGM | Shared backbone |
| Single-backbone | VideoVLA | Joint video rollout | VGM | Shared backbone |
| Single-backbone | VideoPolicy | Video policy substrate | VGM | Shared backbone |
| Single-backbone | Cosmos Policy | Parallel action/state/value outputs | VGM | Shared backbone |
| Single-backbone | DreamZero | Chunk-wise joint rollout | VGM | Shared backbone |
| Single-backbone | UD-VLA | Synchronous denoising | VGM | Shared backbone |
| Single-backbone | GigaWorld-Policy | Optional visual branch | VGM | Shared backbone |
| MoE/MoT | GE-Act | Latent visual guidance | VGM | Expert fusion |
| MoE/MoT | Motus | Expert rollout | VGM | MoT fusion |
| MoE/MoT | LingBot-VA | Visual predictive context | VGM | MoT fusion |
| MoE/MoT | BagelVLA | Single-step visual foresight | VGM | MoT fusion |
| MoE/MoT | Fast-WAM | Train-time video, test-time skipped | VGM | MoT fusion |
| MoE/MoT | LDA-1B | Latent dynamics only | VGM | Expert fusion |
| MoE/MoT | FRAPPE | Latent representation alignment | VGM | Parallel experts |
| MoE/MoT | DiT4DiT | Latent video guidance | VGM | Expert fusion |
| Unified VLA | GR-1 | Future image prediction | UMM | Joint co-training |
| Unified VLA | UP-VLA | Future image prediction | UMM | Joint co-training |
| Unified VLA | WorldVLA | Future image (mainly train-time) | UMM | Joint co-training |
| Unified VLA | DreamVLA | Structured world knowledge | UMM | Joint co-training |
| Unified VLA | UniVLA | Latent world modeling | UMM | Joint co-training |
| Unified VLA | CoWVLA | Latent dynamics | UMM | Joint co-training |
| Unified VLA | F1 | Visual foresight | UMM | Unified MoT |
| Unified VLA | InternVLA-A1 | Latent foresight | UMM | Unified MoT |
| Unified VLA | HALO | Visual subgoal prediction | UMM | Unified multi-expert |
| Unified VLA | TriVLA | Episodic dynamics | UMM | Multi-system |
| Latent-space WM | FLARE | Latent alignment | MLLM | Latent internalization |
| Latent-space WM | VLA-JEPA | Latent target prediction | MLLM | Latent internalization |
| Latent-space WM | JEPA-VLA | Predictive embeddings | MLLM | Latent internalization |
| Latent-space WM | WoG | Future condition only | MLLM | Latent internalization |
| Latent-space WM | DIAL | Latent visual foresight | MLLM | Latent internalization |

<small>表源：`World Model for Robot Learning: A Comprehensive Survey`，Sec. 3 Table。原表英文列名保留。`VGM` 指 video generation model，`UMM` 指 unified multimodal model，`MLLM` 指 multimodal large language model。</small>

**这张表怎么用。**
不要把它当排行榜。它回答的是工程选型问题：未来生成在 inference 时是否真的发生？视频、动作、语言是否共享 backbone？视频分支是完整生成、latent guidance，还是只做训练时辅助？这直接影响训练成本、推理延迟和失败形态。

### Figure 3：三类 video-backbone policy 耦合

![Policy paradigms 原论文图](../../assets/images/paper-deep-dives/world-models/world-model-robot-learning-survey/figure-3-policy-paradigms.png){ width="920" }

<small>图源：`World Model for Robot Learning: A Comprehensive Survey`，Figure 3。原论文图意：展示 IDM-style、single-backbone-style 和 MoT-style 三种用世界模型支撑 policy 的架构范式。</small>

**训练细节要看耦合位置。**
IDM-style 通常先训练或适配视频预测模型，再训练 inverse dynamics / policy head 从预测未来中恢复动作。它的优点是模块清楚，缺点是未来视频一旦 action-inconsistent，后面的 action decoder 会跟着错。

single-backbone-style 把未来视觉 latent 和动作 latent 放进同一生成主干。论文用一个通用形式表示：

$$
\mathbf{x}=[z^{v};z^{a}], \qquad
\hat{y}=f_\theta(\tilde{\mathbf{x}}_\tau,o_t,l,\tau)
$$

这里 \(z^v\) 是未来视觉表示，\(z^a\) 是未来动作表示，\(\tilde{\mathbf{x}}_\tau\) 是被噪声、mask 或离散扰动破坏后的联合表示。目标可以是 diffusion noise、flow matching velocity 或 masked token。训练信号逼迫模型在同一动态过程中同时解释“未来世界怎样变”和“动作应该怎样出”。

MoT-style 保留视频专家和动作专家，但用 shared attention、cross attention 或 interleaved sequence 反复交换信息。它的关键不是“多了几个专家”，而是承认视频预测和动作控制的频率、尺度和优化目标不同。一个实用经验是：视频分支不一定要在部署时完整生成画面；很多系统只需要它在训练时提供 predictive latent，或在推理时给 action branch 中间特征。

### Figure 4：MLLM 内部化世界模型

![UMM and latent-space VLA 原论文图](../../assets/images/paper-deep-dives/world-models/world-model-robot-learning-survey/figure-4-umm-latent-vla.png){ width="860" }

<small>图源：`World Model for Robot Learning: A Comprehensive Survey`，Figure 4。原论文图意：展示两条 MLLM-based 路线：unified VLA 在同一模型里联合处理 observation、language、action 和未来辅助输出；latent-space WM 则把未来动态压进 compact world representation 或 latent prediction。</small>

**这里最像 JEPA 路线。**
Unified VLA 不一定显式生成未来视频，但它会用 future image prediction、visual foresight、structured world knowledge 或 latent future target 给 action learning 加预测压力。latent-space WM 更进一步：未来帧只作为 latent target，不要求模型重建像素。这对训练效率很重要，因为机器人动作未必需要完整未来画面，可能只需要对象位置、运动趋势、接触风险或 subgoal embedding。

从训练角度看，latent 路线最怕 leakage。未来帧可以产生监督目标，但不能在推理路径里被模型“偷看”。VLA-JEPA 这类方法强调 leakage-free state prediction，本质上是在保证训练时的未来监督不会变成部署时不存在的信息捷径。

## 主线二：World Model as Simulator

世界模型作为 simulator 时，它不只是给 policy 一个辅助特征，而是替代环境参与 rollout：

$$
(\hat{o}_{t+1}, \hat{r}_t, \hat{d}_t) \sim p_\phi(\cdot \mid o_{\le t}, a_{\le t}, l)
$$

这里 \(\hat{o}_{t+1}\) 是想象出来的下一观测，\(\hat{r}_t\) 是 reward 或 progress 信号，\(\hat{d}_t\) 是 termination / done。policy 可以在这个 learned simulator 里做 RL：

$$
J(\theta)=\mathbb{E}_{\hat{\tau}\sim(\pi_\theta,p_\phi)}
\left[\sum_t \gamma^t \hat{r}_t\right]
$$

论文还把近年的 VLA post-training 放进 PPO / GRPO-style 视角：

$$
\mathcal{L}_{\mathrm{RL}}(\theta)
=-\mathbb{E}_t
\left[
\min(r_t(\theta)\hat{A}_t,
\mathrm{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t)
\right]
$$

这不是说所有方法都用同一套实现，而是说明它们共享一个目标：用 learned world simulator 产生 imagined transitions 和 feedback，让 policy 少依赖昂贵、危险、难 reset 的真实机器人交互。

### Figure 5：RL 和 validation 两种 simulator 用法

![World model for RL and validation 原论文图](../../assets/images/paper-deep-dives/world-models/world-model-robot-learning-survey/figure-5-rl-vs-validation.png){ width="860" }

<small>图源：`World Model for Robot Learning: A Comprehensive Survey`，Figure 5。原论文图意：左侧是 world model as learned simulator for RL，右侧是 world model as validation / evaluator，用 imagined consequences 评分候选动作。</small>

**RL 路线和 evaluator 路线不要混写。**
RL 路线把 world model 当训练环境。World-Env、VLA-RFT、WMPO、World4RL、World-Gymnast、PlayWorld、RehearseVLA、RISE、GigaBrain 等方法都属于这个大方向，只是 reward、termination、action head 和优化器兼容性不同。关键训练细节是：simulator 必须输出足够可信的 state transition 和 reward-like feedback，否则 policy 会学会利用模拟器漏洞。

Evaluator 路线不一定更新 policy。它让 policy 先提出多个候选动作或候选轨迹，world model rollout 后按成功概率、progress、risk、value 或 policy-ranking fidelity 进行选择。GPC、IRASim、World-in-World、DreamPlan、WorldEval、WorldArena 等工作更接近这一类。它们的核心指标不是视频漂亮，而是候选排序是否跟真实执行一致。

论文还强调第二层进展：world model 本身要和 policy co-evolve。World-VLA-Loop、VLAW、WoVR 这类方法把 policy failure rollouts 回流给 simulator，再用改进后的 simulator 产生更好的 imagined data。抽象成：

$$
\phi^{k+1}\leftarrow \mathrm{UpdateWM}(\phi^k,D_{\mathrm{real}}\cup D_{\mathrm{policy}}(\pi_{\theta^k})),
\qquad
\theta^{k+1}\leftarrow \mathrm{UpdatePolicy}(\theta^k,\hat{D}(\phi^{k+1}))
$$

这里的风险也最明显：如果 simulator 长时幻觉、动作不敏感或 reward head 偏了，policy optimization 会被错误未来牵着走。

## 主线三：Robotic Video World Model

视频世界模型是这篇综述最贴近 LingBot-World 的部分。论文把 robotic video generation 的演进分成四级：

1. Video generation as imagination for policy learning；
2. Action-controllable video world models；
3. Structure-aware generation with interaction and geometry priors；
4. Foundation video world models。

### Figure 6：Robotic video world models

![Robotic video world models 原论文图](../../assets/images/paper-deep-dives/world-models/world-model-robot-learning-survey/figure-6-video-world-models.png){ width="920" }

<small>图源：`World Model for Robot Learning: A Comprehensive Survey`，Figure 6。原论文图意：以 robotic video world model 为核心，展示它从 task-conditioned imagination，到 action conditioning，再到 structure priors 和 foundation-scale video backbone 的能力递进。</small>

**这张图支撑的是能力递进。**
最左侧的 imagination 主要补 supervision：生成未来执行片段、视觉计划或合成数据。中间的 action-controllable 才开始回答“给定动作会发生什么”。再往右的 structure-aware 通过 mask、depth、normal、多视角、identity cue 或 3D flow 强化物理与交互一致性。最右侧 foundation-scale 则把强视频先验变成可复用的 world-model interface。

### Table 2：Robotic video WM capability regimes

| Group | Method | Task-cond. | Action-cond. | Structure-aware | Foundation-scale | Main use |
| --- | --- | --- | --- | --- | --- | --- |
| Imagination-Based | UniPi | ✓ | -- | -- | ✓ | Plan. |
| Imagination-Based | Video Language Planning | ✓ | -- | -- | ✓ | Plan. |
| Imagination-Based | Dreamitate | -- | -- | -- | ✓ | Plan. |
| Imagination-Based | RoboDreamer | ✓ | -- | -- | -- | Plan. |
| Imagination-Based | ManipDreamer | ✓ | -- | ✓ | -- | Plan. |
| Imagination-Based | DreMa | -- | ✓ | ✓ | -- | Data |
| Imagination-Based | PhysWorld | ✓ | -- | ✓ | -- | Plan. |
| Imagination-Based | DreamGen | ✓ | -- | -- | ✓ | Data |
| Action-Controllable | IRASim | -- | ✓ | -- | -- | Plan. |
| Action-Controllable | RoboEnvision | ✓ | -- | -- | -- | Plan. |
| Action-Controllable | RoboMaster | -- | ✓ | ✓ | -- | Data |
| Action-Controllable | Ctrl-World | -- | ✓ | ✓ | -- | Eval. |
| Action-Controllable | EnerVerse-AC | -- | ✓ | ✓ | -- | Eval. |
| Action-Controllable | Interactive World Simulator | -- | ✓ | -- | -- | Sim. |
| Action-Controllable | EVA | ✓ | -- | -- | -- | Eval. |
| Structure-Aware | Mask2IV | ✓ | -- | ✓ | -- | Data |
| Structure-Aware | TesserAct | ✓ | -- | ✓ | -- | Sup. |
| Structure-Aware | RoboVIP | ✓ | -- | ✓ | -- | Data |
| Foundation Video WM | Vid2World | -- | ✓ | -- | ✓ | Sim. |
| Foundation Video WM | Genie Envisioner | ✓ | ✓ | -- | ✓ | Sim. |
| Foundation Video WM | DreamDojo | -- | ✓ | -- | ✓ | Sim. |
| Foundation Video WM | WoW | -- | -- | -- | ✓ | Plan. |
| Foundation Video WM | UnifoLM-WMA-0 | ✓ | ✓ | -- | -- | Sim. |
| Foundation Video WM | Cosmos Predict 2.5 | ✓ | -- | -- | ✓ | Sim. |
| Foundation Video WM | GigaWorld-0 | -- | -- | ✓ | ✓ | Data |
| Foundation Video WM | ABot-PhysWorld | -- | ✓ | -- | ✓ | Sim. |

<small>表源：`World Model for Robot Learning: A Comprehensive Survey`，Sec. 5 Table。原表英文列名保留。Checkbox 表示原论文明确支持或强调该能力。</small>

**训练细节怎么补。**
Imagination-Based 方法通常从 task / language 条件生成未来视觉片段，再通过 inverse dynamics、latent action modeling 或 visual plan 给 policy 增加监督。它的训练风险是未来片段可能符合语义，却不符合机器人可执行动作。

Action-Controllable 方法要把低层动作插进时序生成路径。IRASim 用 frame-level action conditioning 对齐动作和帧；Ctrl-World、EnerVerse-AC 等更关心 action-faithful rollout 是否可用于 evaluation。这里的核心训练样本不再是被动视频，而是 \((o_t,a_t,o_{t+1})\) 或多步 \((o_{\le t},a_{t:t+H},o_{t+1:t+H})\) transition。

Structure-Aware 方法给模型额外结构变量，例如 actor/object trajectories、mask、depth、normal、多视角 identity。它的意义是把“接触、遮挡、物体身份、空间一致性”这些变量显式暴露出来，降低纯 RGB 生成的歧义。

Foundation Video WM 方法则接近 LingBot-World 的工程思路：先继承大规模视频底座，再通过动作、结构、机器人数据和后训练把它改造成可交互 world model。这里最重要的训练问题是避免 catastrophic forgetting，同时让动作真正进入 dynamic path，而不是停在 prompt 层。

## 数据：世界模型训练到底缺什么

论文强调，robotic world model 的数据不是普通 video corpus，而是 agent-environment transitions。一个样本最好同时包含观测、动作、语言、任务进度、embodiment constraint、接触或物理交互信号。数据规模当然重要，但更关键的是这些数据是否覆盖“动作导致世界变化”的因果结构。

### Table 3：Core attributes of representative datasets/resources

| Name | Year | Source | X-Emb. | Act. | Obs./3D | Lang. | M/C |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RoVid-X | 2026 | Real/Robot video | partial | partial | partial | ✓ | × |
| Open X-Embodiment (OXE) | 2024 | Real | ✓ | ✓ | partial | partial | partial |
| DROID | 2024 | Real | × | ✓ | partial | ✓ | partial |
| BridgeData V2 | 2023 | Real | × | ✓ | partial | ✓ | partial |
| AgiBot World | 2025 | Real | partial | ✓ | partial | ✓ | partial |
| Galaxea Open-World Dataset | 2025 | Real | partial | ✓ | partial | ✓ | partial |
| Humanoid Everyday | 2025 | Real | partial | ✓ | ✓ | ✓ | ✓ |
| RoboMIND 2.0 | 2025 | Real+Sim | ✓ | ✓ | partial | ✓ | ✓ |
| FastUMI-100K | 2025 | Real | partial | ✓ | ✓ | ✓ | partial |
| BRMData | 2024 | Real | partial | ✓ | ✓ | partial | partial |
| UMI | 2024 | Real | partial | ✓ | partial | partial | × |
| MV-UMI | 2025 | Real | ✓ | ✓ | ✓ | partial | × |
| ActiveUMI | 2025 | Real | partial | ✓ | ✓ | partial | × |
| TWIST2 | 2025 | Real | partial | ✓ | partial | × | × |
| DexWild | 2025 | Human+Robot | ✓ | partial | partial | × | × |
| EgoMimic | 2025 | Human+Robot | partial | partial | ✓ | × | × |
| PHSD / In-N-On | 2025 | Human ego | × | partial | partial | × | × |
| UniHand | 2025 | Human video | partial | partial | partial | ✓ | × |
| UniHand 2.0 | 2026 | Human+Robot+VLM | ✓ | ✓ | partial | ✓ | × |
| Hoi! | 2025 | Human+Robot | ✓ | ✓ | ✓ | × | ✓ |
| FreeTacMan | 2025 | Robot-free | partial | ✓ | ✓ | × | ✓ |
| Humanoid Visual-Tactile-Action | 2025 | Real | × | ✓ | partial | × | ✓ |
| VTDexManip | 2025 | Human tactile | × | partial | partial | × | ✓ |
| RH20T | 2023 | Real | partial | ✓ | partial | ✓ | ✓ |
| RH20T-P | 2025 | Real | partial | ✓ | partial | ✓ | ✓ |
| RoboTwin 2.0 | 2025 | Sim | ✓ | ✓ | partial | ✓ | partial |
| Action100M | 2026 | Web video | × | partial | × | ✓ | × |

<small>表源：`World Model for Robot Learning: A Comprehensive Survey`，Table "Core attributes of representative datasets/resources for embodied world model training"。原表英文列名保留。`X-Emb.` 是 cross-embodiment coverage；`Act.` 是 explicit action supervision or aligned action proxy；`Obs./3D` 是 multiview/depth/LiDAR/3D 等观测支持；`Lang.` 是 language/task conditioning；`M/C` 是 multimodal or contact-rich signals。</small>

### Table 4：Dataset relevance to world-modeling capabilities

| Name | General Traj. | Long-Horizon | X-Emb. Scaling | Human Prior | Contact / Physics | Synth. / Recipe |
| --- | --- | --- | --- | --- | --- | --- |
| RoVid-X | partial | partial | partial | × | partial | × |
| Open X-Embodiment (OXE) | ✓ | partial | ✓ | × | partial | × |
| DROID | ✓ | partial | × | × | partial | × |
| BridgeData V2 | ✓ | partial | × | × | partial | × |
| AgiBot World | ✓ | partial | partial | × | partial | × |
| Galaxea Open-World Dataset | ✓ | partial | partial | × | partial | × |
| Humanoid Everyday | ✓ | ✓ | partial | × | ✓ | × |
| RoboMIND 2.0 | ✓ | ✓ | ✓ | × | ✓ | partial |
| FastUMI-100K | ✓ | ✓ | partial | × | partial | × |
| BRMData | ✓ | ✓ | partial | × | partial | × |
| UMI | partial | partial | partial | ✓ | × | × |
| MV-UMI | partial | partial | ✓ | ✓ | × | × |
| ActiveUMI | partial | partial | partial | ✓ | × | × |
| TWIST2 | partial | partial | partial | × | partial | × |
| DexWild | partial | partial | ✓ | ✓ | × | × |
| EgoMimic | × | partial | partial | ✓ | × | × |
| PHSD / In-N-On | × | partial | × | ✓ | × | × |
| UniHand | × | partial | partial | ✓ | × | × |
| UniHand 2.0 | partial | partial | ✓ | ✓ | × | ✓ |
| Hoi! | partial | partial | ✓ | partial | ✓ | × |
| FreeTacMan | partial | partial | partial | partial | ✓ | × |
| Humanoid Visual-Tactile-Action | partial | partial | × | × | ✓ | × |
| VTDexManip | × | partial | × | ✓ | ✓ | × |
| RH20T | partial | partial | partial | × | ✓ | × |
| RH20T-P | partial | partial | partial | × | ✓ | × |
| RoboTwin 2.0 | partial | partial | ✓ | × | partial | ✓ |
| Action100M | × | partial | × | partial | × | ✓ |

<small>表源：`World Model for Robot Learning: A Comprehensive Survey`，Table "Relevance of representative datasets/resources to embodied world-modeling capabilities"。原表英文列名保留。`✓` 表示 strong relevance，`partial` 表示 partial relevance，`×` 表示 limited or no direct relevance。</small>

**数据表的工程读法。**
大规模成功示范只能教模型“常见任务怎么做”，不一定教它失败恢复、接触临界点和决策敏感变化。一个可靠 world model 数据栈至少要覆盖四类分布：成功轨迹、失败与恢复、动作反事实、接触/几何密集片段。当前数据 landscape 的短板正是后两类稀缺。

## Benchmark：别把视觉指标当闭环能力

论文把 embodied world model evaluation 分成三类：

| Evaluation category | 关注点 | 典型问题 |
| --- | --- | --- |
| Action-conditioned generation and open-loop predictive quality | 给定历史和动作，未来是否语义正确、时序一致、动作敏感 | 视频看起来合理，但动作是否真的控制了未来 |
| Closed-loop task utility and policy evaluation | 世界模型放进 planner / evaluator / simulator 后是否提升 decision utility | imagined rollout 的排序、value 和真实执行是否一致 |
| Physical consistency, controllability, and executability diagnostics | 生成未来是否物理可信、动作可恢复、可执行 | 能否通过 IDM recovery、physical-law 或 execution-oriented checks |

这个分类很适合本站主线：视觉 realism 是最弱的证据，action-conditioned consistency 是中等证据，closed-loop task utility 和 policy-ranking fidelity 才更接近机器人世界模型的实用价值。

### Table 5：LIBERO standard 4-suite results

| Group | Method | Spatial | Object | Goal | Long | Avg |
| --- | --- | --- | --- | --- | --- | --- |
| Decoupled | UniPi | -- | -- | -- | 0.0 | -- |
| Decoupled | MimicVideo | 94.2 | 96.8 | 90.6 | 94.0 | 93.9 |
| Decoupled | Say-Dream-ACT | 99.4 | 99.2 | 98.6 | 95.4 | 98.1 |
| Single-backbone | UVA | -- | -- | -- | 90.0 | -- |
| Single-backbone | VideoPolicy | -- | -- | -- | 94.0 | -- |
| Single-backbone | Cosmos Policy | 98.1 | 100.0 | 98.2 | 97.6 | 98.5 |
| Single-backbone | UD-VLA | 94.1 | 95.7 | 91.2 | 89.6 | 92.7 |
| MoE / MoT | Motus | 96.8 | 99.8 | 96.6 | 97.6 | 97.7 |
| MoE / MoT | LingBot-VA | 98.5 | 99.6 | 97.2 | 98.5 | 98.5 |
| Unified VLA | RynnVLA-002 | 99.0 | 99.8 | 96.4 | 94.4 | 97.4 |
| Unified VLA | DreamVLA | 97.5 | 94.0 | 89.5 | 89.5 | 92.6 |
| Unified VLA | UniVLA | 96.5 | 96.8 | 95.6 | 92.0 | 95.2 |
| Unified VLA | Unified VLA | 95.4 | 98.8 | 93.6 | 94.0 | 95.5 |
| Unified VLA | CoWVLA | 97.2 | 97.8 | 94.6 | 92.8 | 95.6 |
| Unified VLA | F1 | 98.2 | 97.8 | 95.4 | 91.3 | 95.7 |
| Unified VLA | TriVLA | 91.2 | 93.8 | 89.8 | 73.2 | 87.0 |
| Latent-space WM | VLA-JEPA | 96.2 | 99.6 | 97.2 | 95.8 | 97.2 |
| Latent-space WM | JEPA-VLA | 97.2 | 98.0 | 95.6 | 94.8 | 96.4 |

<small>表源：`World Model for Robot Learning: A Comprehensive Survey`，Table "Representative results on the LIBERO standard 4-suite benchmark"。原表英文列名保留。`Avg` 是四个 LIBERO suite 平均成功率；`--` 表示原论文未按标准 Spatial/Object/Goal/Long 协议直接报告。</small>

### Table 6：RoboTwin, CALVIN, and SIMPLER-style results

| Group | Method | RT-A | RT-B | C-A | C-D | S-G | S-W | S-O |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Decoupled | UniPi | -- | -- | 0.92 | -- | -- | -- | -- |
| Decoupled | VidMan | -- | -- | 3.42 | -- | -- | -- | -- |
| Decoupled | Vidar | 65.8 | 17.5 | -- | -- | -- | -- | -- |
| Decoupled | VPP | -- | -- | 4.33 | -- | -- | -- | -- |
| Decoupled | Video2Act | 54.6 | 54.1 | -- | -- | -- | -- | -- |
| Decoupled | MimicVideo | -- | -- | -- | -- | -- | -- | 46.9/56.3 |
| Single-backbone | VideoVLA | -- | -- | -- | -- | 73.1/62.8 | 53.1 | 63.0 |
| Single-backbone | UD-VLA | -- | -- | -- | 4.64 | -- | 62.5 | -- |
| MoE / MoT | Motus | 88.7 | 87.0 | -- | -- | -- | -- | -- |
| MoE / MoT | LingBot-VA | 92.9 | 91.6 | -- | -- | -- | -- | -- |
| MoE / MoT | LingBot-VLA | 88.6 | 86.7 | -- | -- | -- | -- | -- |
| MoE / MoT | BagelVLA | 75.3 | 20.9 | 4.41 | -- | -- | -- | -- |
| MoE / MoT | FRAPPE | 57.5 | 25.5 | -- | -- | -- | -- | -- |
| Unified VLA | GR-1 | -- | -- | 3.06 | 4.21 | -- | -- | -- |
| Unified VLA | UP-VLA | -- | -- | 4.08 | 4.42 | -- | -- | -- |
| Unified VLA | DreamVLA | -- | -- | 4.44 | -- | -- | -- | -- |
| Unified VLA | Unified VLA | -- | -- | 4.41 | 4.63 | -- | 69.8 | -- |
| Unified VLA | CoWVLA | -- | -- | 4.21 | 4.47 | 60.9 | 76.0 | -- |
| Unified VLA | F1 | -- | -- | -- | -- | -- | -- | 72.9 |
| Unified VLA | InternVLA-A1 | 89.4 | 87.0 | -- | -- | -- | -- | -- |
| Unified VLA | HALO | 80.5 | 26.4 | -- | -- | -- | -- | -- |
| Unified VLA | TriVLA | -- | -- | 4.37 | -- | -- | -- | -- |
| Latent-space WM | VLA-JEPA | -- | -- | -- | -- | 65.2 | 57.3 | -- |
| Latent-space WM | JEPA-VLA | 73.5 | 17.7 | -- | -- | -- | -- | -- |
| Latent-space WM | WoG | -- | -- | -- | -- | 69.4 | 63.5 | -- |

<small>表源：`World Model for Robot Learning: A Comprehensive Survey`，Table "Representative results on RoboTwin, CALVIN, and SIMPLER-style benchmarks"。原表英文列名保留。`RT-A` / `RT-B` 是 RoboTwin 两种设置；`C-A` / `C-D` 是 CALVIN ABCD / ABCDD；`S-G`、`S-W`、`S-O` 是 SIMPLER-style 的 Google Robot、WidowX 和 other reported setups。</small>

**结果表要谨慎读。**
这些数字说明多个 paradigm 都能拿到强结果，不能说明某一种架构绝对胜出。LIBERO 的 Long suite 更能区分长时任务能力；RoboTwin、CALVIN、SIMPLER-style 的结果更受 embodiment、动作空间和协议影响。因此它们适合看趋势，不适合把不同论文硬排一个总榜。

## 训练细节和工程启发

如果要把这篇综述落成训练方案，可以抽出五条原则。

| 训练问题 | 综述给出的启发 | 工程检查项 |
| --- | --- | --- |
| 未来预测是否对动作敏感 | controllable world model 要建模 \(p(o_{t+1:t+k}\mid o_t,a_{t+1:t+k})\) | 固定历史，替换候选动作，看未来 latent / reward / risk 是否分叉 |
| 视频分支是否必须推理时启用 | 很多方法把未来预测作为 train-time auxiliary 或 latent guidance | 记录部署路径：完整 video rollout、partial denoising、latent features，还是 test-time skipped |
| 训练数据是否是 transition-rich | 数据应包含 action、long horizon、cross-embodiment、3D/multiview、contact-rich signals | 数据 schema 不只存视频，还要存动作、语言、状态、失败、恢复、接触和时间同步 |
| simulator reward 是否可信 | RL in learned world model 会放大 reward head 和 termination head 的误差 | 用真实 replay、hard negative、policy ranking fidelity 检查 reward exploit |
| 评测是否功能化 | open-loop visual realism 不等于 control utility | 至少补 action sensitivity、closed-loop success、policy ranking、executability diagnostics |

效率上，这篇综述反复指向一个现实判断：完整像素级视频 rollout 很贵，也不一定必要。MimicVideo、LingBot-VA、Fast-WAM、LeWorldModel 等路线都在不同程度上说明，policy 可能只需要 motion dynamics、latent foresight 或 train-time world-model regularization。真正要优化的是 cost per useful decision，而不是单纯让视频更清晰。

## 局限与挑战

论文最后把挑战收束到六类：

| Challenge | 具体含义 | 对项目的提醒 |
| --- | --- | --- |
| Causal Conditioning Gaps | 未来预测更受历史和任务意图影响，而不是待执行动作影响 | action token 要进入 dynamic path，并做反事实动作评测 |
| Efficiency Bottlenecks | 共同预测视频和动作、或扩散视频 rollout 会显著增加训练和推理成本 | 优先尝试 latent prediction、partial denoising、train-time-only WM |
| Multi-Modal Perception Bottlenecks | 仅靠 vision/proprioception 很难捕捉摩擦、刚度、接触稳定性 | 接触任务要引入 tactile、force、dense proprioception 或物理状态 |
| Classical Control Integration | MPC 需要大量 rollout，高容量 world model 很难实时优化 | learned dynamics 要和稳定性、鲁棒控制、低层 controller 协同 |
| Symbolic Structure Integration | 像素/latent rollout 长时会漂，符号和关系结构可提升组合泛化 | 长任务需要 object-centric、relational、predicate 或 occupancy abstraction |
| Open Challenges in Evaluation Metrics | 缺少统一 metrics；视觉真实和功能价值经常脱钩 | 建议固定一组 task success、policy-ranking fidelity、executability diagnostics |

最值得记住的是 causal conditioning gap。很多系统看起来有 world model head，但未来其实主要由历史画面和语言意图决定，动作只是弱条件。这样的模型可以辅助表征学习，却不能可靠做 candidate action evaluation。

## 阅读结论

这篇综述的成熟度是“前沿路线图 + 官方维护资源列表”，不是可复现单模型 recipe。它最可复用的是分类框架：先判断一个方法是 world model for policy、world model as simulator，还是 robotic video world model；再判断未来生成是否 inference-active、是否 action-conditioned、是否结构化、是否有 closed-loop utility 证据。

对世界模型高效训练来说，它给出的最大启发是：不要把训练目标写成“生成更好看的未来视频”。更稳的目标是训练一个可被 policy 消费的预测接口，能在动作分叉、长 horizon、接触风险、reward / termination 和 policy ranking 上提供可靠信号。下一步最需要补的证据，是统一协议下的 action sensitivity、closed-loop task utility、simulator exploitation analysis 和跨 embodiment 复现。

