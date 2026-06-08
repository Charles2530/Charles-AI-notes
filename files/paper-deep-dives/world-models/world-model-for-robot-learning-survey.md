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

这篇综述要解决的不是“再列一批世界模型论文”，而是给机器人学习里的世界模型划清三种用途：**放进 policy、变成 simulator、作为 video world model 提供可控未来**。读它最重要的收获，是以后看到一篇新工作时，能先问清楚：它到底在帮助动作生成、帮助模拟训练，还是只是在生成更像真的未来视频。

一句话概括：**机器人世界模型的价值不在于未来画面是否漂亮，而在于预测是否能改变动作选择。**

## 先抓主线

这篇论文的所有分类都可以压到一个判断链路：

```text
当前观测 / 语言 / 动作
  -> 预测未来状态、未来视频、reward、termination、value 或 latent
  -> policy / planner / evaluator 使用这些预测
  -> 动作选择、训练数据或评估结论发生变化
```

如果最后一步没有发生，模型最多只是 video predictor 或 representation learner，还不能强称为机器人学习里的 actionable world model。

| 问题 | 论文对应章节 | 要判断的核心 |
| --- | --- | --- |
| 世界模型怎样帮助 policy 出动作 | Sec. 3 World Model for Policy | 预测未来是 decoupled、shared backbone、expert fusion，还是 latent internalization |
| 世界模型怎样替代环境 | Sec. 4 World Model as Simulator | 它是在 imagined rollout 里训练 policy，还是在执行前评估候选动作 |
| 视频生成怎样变成机器人世界模型 | Sec. 5 Robotic Video Generation | 它是否 task-conditioned、action-conditioned、structure-aware、foundation-scale |
| 数据和评测怎么支撑这些 claim | Sec. 7 Benchmarks, Datasets, and Results | 数据是否包含 transition，评测是否看 closed-loop utility |

所以这篇要按“功能角色”读，不要按论文发布时间或表格行数读。

## 证据边界

综述的强项是组织文献和统一问题语言；弱项是它不重新跑实验，也不统一所有论文的训练条件。下面这张表是读全文的证据口径。

| 论文结论 | 证据来源 | 证据等级 | 可以吸收什么 | 不能直接推出什么 |
| --- | --- | --- | --- | --- |
| 机器人世界模型应以 action usefulness 为中心 | 定义、Figure 1、evaluation discussion | Survey synthesis | 评测不能只看视觉质量，要看 action sensitivity、planning utility、policy ranking | 某个具体模型已经可靠控制真实机器人 |
| policy coupling 正从 decoupled 走向 unified / MoE / latent | Figure 2、Figure 3、Table 1 | Literature taxonomy | 选型时先看未来预测是否 inference-active，以及动作和视频在哪里耦合 | shared backbone 一定优于 decoupled |
| learned simulator 正从训练环境扩展到 evaluator | Figure 5、Sec. 4 | Functional taxonomy | world model 可以提供 reward、termination、policy ranking、candidate filtering | simulator 的 reward 或 ranking 一定可信 |
| robotic video WM 的重点从 imagination 转向 controllable interface | Figure 6、Table 2 | Literature taxonomy | 视频底座训练要补动作、结构、长时一致和交互数据 | 普通 text-to-video 能力等同于机器人世界模型 |
| 数据集要按 transition richness 判断 | Dataset tables、Sec. 7 | Survey synthesis | 数据 schema 要记录动作、语言、3D/multiview、接触、长 horizon | 表格就是数据集推荐榜 |

## 定义：世界模型不是视频生成器

论文采用的是 robotics-centered definition。它不要求世界模型一定预测像素，也不要求一定是 Dreamer 那类 latent dynamics。它要求预测必须对机器人决策有用。

通用形式是：

$$
p(x_{t+1:t+H} \mid x_t, a_{t:t+H-1}, l)
$$

这里 \(x_t\) 是被建模的状态，可以是 RGB 视频、latent state、3D/occupancy、reward/value 相关状态，甚至 object relation / predicate 这类符号状态；\(a_{t:t+H-1}\) 是动作序列；\(l\) 是语言指令或高层任务目标。论文特别把语言也看成 high-level action，因为语言指定“未来应该实现什么”。

如果状态落到视频空间，就得到：

$$
p(v_{t+1:t+H} \mid o_t, a_{t:t+H-1}, l)
$$

这也是为什么视频生成模型会进入世界模型讨论。但分界线很清楚：**视频模型生成 plausible future；机器人世界模型要生成 action-faithful future。**

固定历史画面，如果输入“向左移动”和“向右移动”得到几乎同一个未来，那它可能是强视频先验，却不是可用于控制的动作条件世界模型。

## 总图：三种用法，不是一条流水线

![World Model for Robot Learning survey organization 原论文图](../../assets/images/paper-deep-dives/world-models/world-model-robot-learning-survey/figure-1-survey-organization.png){ width="920" }

<small>图源：`World Model for Robot Learning: A Comprehensive Survey`，Figure 1。原论文图意：整篇综述按三个主轴组织：Sec. 3 讨论 world models 如何和 robot policies 耦合；Sec. 4 讨论 world models 作为 simulators；Sec. 5 讨论 robotic video world models 的能力演进。</small>

这张图不是在说三条路线按顺序发生，而是在说一个世界模型可能有三种身份。

| 身份 | 输入 | 输出 | 下游使用者 | 主要风险 |
| --- | --- | --- | --- | --- |
| World Model for Policy | observation、language、history，有时有 future target | action、future latent、future image、subgoal | policy head / action decoder | 未来预测和动作生成脱节 |
| World Model as Simulator | observation、candidate actions、instruction | imagined observation、reward、done、value | RL algorithm / evaluator / planner | simulator hallucination 污染训练或排序 |
| Robotic Video World Model | observation、task、action、structure priors | future video / visual state | data engine / planner / policy evaluator | 视觉真实但动作不可信 |

把这三种身份分开，文章就顺了：先看模型怎样进入 policy，再看怎样替代环境，最后看视频模型怎样变得可控。

![Temporal evolution of representative works 原论文图](../../assets/images/paper-deep-dives/world-models/world-model-robot-learning-survey/figure-2-temporal-evolution.png){ width="920" }

<small>图源：`World Model for Robot Learning: A Comprehensive Survey`，Figure 2。原论文图意：上支展示 world model for policy 从 IDM-style 走向 single-backbone、MoE/MoT、unified VLA 和 latent world modeling；下支展示 world model as simulator 从 validation / candidate evaluation 走向 RL、post-training 和 policy-world-model co-optimization。</small>

Figure 2 的重点不是“新方法替代旧方法”，而是两个收敛趋势：policy 侧越来越把预测和动作生成绑紧；simulator 侧越来越把世界模型放进训练、评估和自我修正循环里。

## 主线一：世界模型进入 Policy

policy 侧的核心问题是：**未来预测到底在哪里影响动作？**

论文把 policy、world model、inverse dynamics 看成同一个联合分布的不同查询：

$$
p(o_{t+1:t+k}, a_{t+1:t+k} \mid o_t, l)
$$

| Query | Formulation | 直觉 |
| --- | --- | --- |
| Policy Model | \(p(a_{t+1:t+k}\mid o_t,l)\) | 直接从当前观测和指令出动作 |
| Passive World Model | \(p(o_{t+1:t+k}\mid o_t,l)\) | 不显式给动作，只按任务想象未来 |
| Controllable World Model | \(p(o_{t+1:t+k}\mid o_t,a_{t+1:t+k})\) | 给定候选动作，预测后果 |
| Inverse Dynamics Model | \(p(a_{t+1:t+k}\mid o_{t:t+k})\) | 给定状态变化，反推动作 |

这个视角很有用，因为很多论文看起来完全不同，本质上只是把未来预测放在了 policy 的不同位置。

![Policy paradigms 原论文图](../../assets/images/paper-deep-dives/world-models/world-model-robot-learning-survey/figure-3-policy-paradigms.png){ width="920" }

<small>图源：`World Model for Robot Learning: A Comprehensive Survey`，Figure 3。原论文图意：展示 IDM-style、single-backbone-style 和 MoT-style 三种用世界模型支撑 policy 的架构范式。</small>

### 1. Decoupled / IDM-style：先想象，再反推动作

这类方法先用世界模型生成未来观测或未来 latent：

$$
\hat{o}_{t+1:t+H}=\mathcal{W}(o_t,l)
$$

再用 inverse dynamics 或 policy head 预测动作：

$$
\pi(a_{t+1:t+H'} \mid o_t,l,\Phi(\hat{o}_{t+1:t+H}))
$$

它的优点是模块清楚：world model 负责“未来会怎样”，policy 负责“怎么做过去”。问题也很直接：如果生成未来不受动作约束，IDM 会从一个错误或平均化的未来里恢复动作。

训练上常见做法是先训练或适配 video predictor，再冻结或轻量适配它，最后训练 action decoder。VPP、Video2Act、MimicVideo 这类工作会进一步把 pixel rollout 压成 latent predictive features，减少完整视频生成成本。

### 2. Single-backbone：未来视觉和动作一起生成

single-backbone 方法不再把 world model 和 policy 分成两个模块，而是把未来视觉表示 \(z^v\) 和动作表示 \(z^a\) 放进同一个生成过程：

$$
\mathbf{x}=[z^v;z^a], \qquad
\hat{y}=f_\theta(\tilde{\mathbf{x}}_\tau,o_t,l,\tau)
$$

这里 \(\tilde{\mathbf{x}}_\tau\) 可以是加噪 diffusion latent、flow matching 中间状态，或 masked token。目标 \(y\) 可能是 diffusion noise、velocity field 或 token target。训练压力来自联合预测：模型必须同时解释“未来世界怎样变”和“动作应该怎样出”。

这条路线的关键收益是减少 representational gap。Cosmos Policy 把 action、future state、value 作为额外 latent frames；DreamZero 做 chunk-wise joint denoising；GigaWorld-Policy 让视觉分支在推理时可选。这些设计都说明一个事实：完整视频不是最终目的，真正重要的是可被动作生成使用的 predictive structure。

### 3. MoE / MoT：视频专家和动作专家深度交互

MoE/MoT-style 不完全共享所有参数，而是保留视频、动作、语言等专家分支，通过 shared attention、cross attention 或 interleaved sequence 交换信息：

$$
(\mathbf{h}^{v}_{\ell+1},\mathbf{h}^{a}_{\ell+1})
=
\mathcal{F}^{mix}_{\ell}(\mathbf{h}^{v}_{\ell},\mathbf{h}^{a}_{\ell};o_t,l)
$$

它的动机是视频预测和动作控制的频率、尺度、监督目标不同。强行完全共享 backbone 可能让两边互相拖累；完全拆开又让信息交互太浅。LingBot-VA、Motus、BagelVLA、Fast-WAM 等方法都在这条中间路线上做取舍。

### 4. Unified VLA / Latent-space WM：把未来预测内化进表征

![UMM and latent-space VLA 原论文图](../../assets/images/paper-deep-dives/world-models/world-model-robot-learning-survey/figure-4-umm-latent-vla.png){ width="860" }

<small>图源：`World Model for Robot Learning: A Comprehensive Survey`，Figure 4。原论文图意：展示两条 MLLM-based 路线：unified VLA 在同一模型里联合处理 observation、language、action 和未来辅助输出；latent-space WM 则把未来动态压进 compact world representation 或 latent prediction。</small>

Unified VLA 不一定在线生成未来视频，但会用 future image prediction、visual foresight、structured world knowledge 或 latent future target 给 action learning 加预测压力。Latent-space WM 则更像 JEPA 路线：未来帧只产生 latent target，不要求像素重建。

这一类最关键的训练细节是防止 leakage。未来帧可以做监督目标，但推理时不能被模型偷看。VLA-JEPA 这类方法强调 leakage-free state prediction，本质上是在保证 future supervision 真正变成内部 predictive state，而不是训练捷径。

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

这张表的用法很简单：不要横向比“谁更强”，而是先看 **Future Generation at Inference**。如果未来预测只在训练时出现，它更像 representation regularizer；如果推理时也 rollout，它就是 latency 和 simulator reliability 的核心成本。

## 主线二：世界模型作为 Simulator

simulator 侧的核心问题是：**世界模型是否能替代环境给 policy 反馈？**

最一般形式是：

$$
(\hat{o}_{t+1},\hat{r}_t,\hat{d}_t)
\sim p_\phi(\cdot \mid o_{\le t},a_{\le t},l)
$$

这里 \(\hat{o}_{t+1}\) 是想象下一观测，\(\hat{r}_t\) 是 reward / progress，\(\hat{d}_t\) 是 termination。policy 可以在 learned simulator 里最大化 imagined return：

$$
J(\theta)=
\mathbb{E}_{\hat{\tau}\sim(\pi_\theta,p_\phi)}
\left[\sum_t \gamma^t \hat{r}_t\right]
$$

对现代 VLA post-training，论文把它概括到 PPO / GRPO-style objective：

$$
\mathcal{L}_{RL}(\theta)=
-\mathbb{E}_t
\left[
\min(r_t(\theta)\hat{A}_t,
\mathrm{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t)
\right]
$$

这个公式不是说所有工作实现完全一样，而是把共同点说清：真实机器人交互贵、慢、危险、难 reset，所以大家试图在 learned world simulator 里产生 rollouts 和 feedback。

![World model for RL and validation 原论文图](../../assets/images/paper-deep-dives/world-models/world-model-robot-learning-survey/figure-5-rl-vs-validation.png){ width="860" }

<small>图源：`World Model for Robot Learning: A Comprehensive Survey`，Figure 5。原论文图意：左侧是 world model as learned simulator for RL，右侧是 world model as validation / evaluator，用 imagined consequences 评分候选动作。</small>

### Simulator 有两种功能

| 功能 | 做什么 | 代表方向 | 最关键的风险 |
| --- | --- | --- | --- |
| Reinforcement Training | policy 在 learned simulator 里 rollout、收 reward、更新参数 | World-Env、VLA-RFT、WMPO、World4RL、World-Gymnast、PlayWorld、RehearseVLA | policy exploit simulator，reward / done head 被钻空子 |
| Evaluation / Validation | policy 先提出候选动作，world model rollout 后排序、过滤或估值 | GPC、IRASim、World-in-World、DreamPlan、WorldEval、WorldArena | 候选排序和真实执行排序不一致 |

这里有一个很重要的区别：RL route 会改变 policy 参数，validation route 不一定训练 policy，它可能只是 test-time selection 或 offline policy ranking。因此两条路线的证据也不同。RL 要看真实任务成功率是否提升；validation 要看 policy-ranking fidelity、value fidelity、failure detection 是否跟真实世界一致。

论文还强调了一个更前沿的 loop：world model 和 policy 共同演化。

$$
\phi^{k+1}\leftarrow \mathrm{UpdateWM}
(\phi^k,D_{real}\cup D_{policy}(\pi_{\theta^k}))
$$

$$
\theta^{k+1}\leftarrow
\mathrm{UpdatePolicy}(\theta^k,\hat{D}(\phi^{k+1}))
$$

直觉是：policy 失败轨迹暴露 simulator 盲区；更好的 simulator 再生成更有用的 imagined data。World-VLA-Loop、VLAW、WoVR 都在靠近这个方向。工程上要特别小心，因为一旦 simulator 对长 horizon、接触或动作分叉不可靠，policy 会沿着错误反馈越走越偏。

## 主线三：视频世界模型如何变成机器人世界模型

视频世界模型部分最容易写成流水账，因为方法很多。更清楚的读法是看四个能力台阶。

![Robotic video world models 原论文图](../../assets/images/paper-deep-dives/world-models/world-model-robot-learning-survey/figure-6-video-world-models.png){ width="920" }

<small>图源：`World Model for Robot Learning: A Comprehensive Survey`，Figure 6。原论文图意：以 robotic video world model 为核心，展示它从 task-conditioned imagination，到 action conditioning，再到 structure priors 和 foundation-scale video backbone 的能力递进。</small>

| 台阶 | 解决的问题 | 训练信号 | 成熟后的用途 |
| --- | --- | --- | --- |
| Imagination-Based | 用未来视频补监督或视觉计划 | task / language conditioned video，inverse dynamics，latent action | data augmentation、visual planning |
| Action-Controllable | 给定动作后未来要跟动作走 | \((o_t,a_t,o_{t+1})\)、frame-level action conditioning、多步 transition | policy evaluation、model-based planning |
| Structure-Aware | RGB 不足以表达接触、遮挡和几何 | mask、depth、normal、multi-view、3D flow、object trajectory | contact-aware data、physically consistent rollout |
| Foundation Video WM | 复用大规模视频底座并接到机器人 | video pretraining、action adaptation、post-training、distillation | reusable simulator、data engine、planning substrate |

这一段的主线是：**从“生成一个像任务的视频”，走向“给定动作和结构，生成可用于控制的未来”。**

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

表里最该看的不是方法数量，而是列名。`Task-cond.` 只能说明模型知道任务目标；`Action-cond.` 才开始接近控制；`Structure-aware` 说明模型额外看见几何、接触或对象结构；`Foundation-scale` 说明它可能继承大规模视频先验。一个机器人世界模型最好不是只占一列，而是能把动作、结构和长期记忆同时接起来。

## 数据：训练世界模型缺的不是视频，而是转移

论文的 dataset 部分可以用一句话概括：**世界模型训练数据不是 video collection，而是 agent-environment transition collection。**

理想数据样本不只是帧序列，还要包含动作、语言、embodiment 信息、任务阶段、失败和恢复、接触或力觉、多视角 / 3D 线索。否则模型学到的是“常见未来”，不是“动作后果”。

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

两张数据表应该合起来读。Table 3 看原始属性：有没有动作、多视角、语言、接触信号。Table 4 看训练用途：能不能支撑 general trajectory pretraining、long-horizon modeling、cross-embodiment scaling、human-prior transfer、contact / physics modeling。当前最大短板不是成功轨迹太少，而是失败恢复、动作反事实和接触临界样本太少。

## 评测：从 open-loop 到 decision utility

论文把 embodied world model evaluation 分成三层。层级越往后，越接近真正机器人价值。

| Evaluation category | 关注点 | 典型问题 | 证据强度 |
| --- | --- | --- | --- |
| Action-conditioned generation and open-loop predictive quality | 给定历史和动作，未来是否语义正确、时序一致、动作敏感 | 视频看起来合理，但动作是否真的控制了未来 | 中 |
| Closed-loop task utility and policy evaluation | 世界模型放进 planner / evaluator / simulator 后是否提升 decision utility | imagined rollout 的排序、value 和真实执行是否一致 | 高 |
| Physical consistency, controllability, and executability diagnostics | 生成未来是否物理可信、动作可恢复、可执行 | 能否通过 IDM recovery、physical-law 或 execution-oriented checks | 高，但协议更难统一 |

这也是为什么论文结果表不能当简单榜单。它们来自不同 benchmark 和不同 reporting protocol，更适合看设计趋势。

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

LIBERO 表最值得看的不是最高平均分，而是 Long suite。很多方法在 Spatial / Object 上已经很高，Long 更能暴露长时动作一致性、子任务衔接和错误恢复问题。

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

RoboTwin、CALVIN、SIMPLER-style 更碎，不能和 LIBERO 一起排绝对名次。它们更适合提醒我们：world model 方法对 embodiment、动作空间、仿真随机化和任务协议都很敏感。一个方法在单一 benchmark 上强，不等于它已经学到可迁移的物理世界模型。

## 从综述落到训练方案

如果要把这篇综述转成项目 checklist，最有价值的是下面五个训练问题。

| 训练问题 | 综述给出的判断 | 工程检查项 |
| --- | --- | --- |
| 未来预测是否真的受动作控制 | controllable world model 要建模 \(p(o_{t+1:t+k}\mid o_t,a_{t+1:t+k})\) | 固定历史，替换候选动作，看 future latent / reward / risk 是否分叉 |
| 视频分支推理时是否必要 | 很多方法只把 future prediction 作为 train-time auxiliary 或 latent guidance | 明确部署路径：full video rollout、partial denoising、latent features、test-time skipped |
| 数据是否 transition-rich | 世界模型需要 action、long horizon、cross-embodiment、3D/multiview、contact-rich signals | 数据 schema 存动作、语言、状态、失败、恢复、接触、时间同步 |
| simulator feedback 是否可信 | RL in learned world model 会放大 reward / termination 错误 | 用真实 replay、hard negative、policy ranking fidelity 检查 reward exploit |
| 评测是否功能化 | open-loop realism 不能代表 control utility | 至少补 action sensitivity、closed-loop success、policy ranking、executability diagnostics |

一个很实际的结论是：完整像素级视频 rollout 经常太贵，也不一定必要。MimicVideo、LingBot-VA、Fast-WAM、LeWorldModel 等路线都指向同一件事：policy 很多时候需要的是 motion dynamics、latent foresight 或 train-time world-model regularization，而不是每次都生成高清未来视频。

## 局限与挑战

论文最后的挑战可以按“为什么世界模型还不能放心进机器人闭环”来理解。

| Challenge | 本质问题 | 对项目的提醒 |
| --- | --- | --- |
| Causal Conditioning Gaps | 未来预测更受历史和任务意图影响，而不是待执行动作影响 | action token 要进入 dynamic path，并做反事实动作评测 |
| Efficiency Bottlenecks | 共同预测视频和动作、或扩散视频 rollout 会显著增加成本 | 优先尝试 latent prediction、partial denoising、train-time-only WM |
| Multi-Modal Perception Bottlenecks | 仅靠 vision/proprioception 难以捕捉摩擦、刚度、接触稳定性 | 接触任务要引入 tactile、force、dense proprioception 或物理状态 |
| Classical Control Integration | MPC 需要大量 rollout，高容量 world model 很难实时优化 | learned dynamics 要和稳定性、鲁棒控制、低层 controller 协同 |
| Symbolic Structure Integration | 像素/latent rollout 长时会漂，符号和关系结构可提升组合泛化 | 长任务需要 object-centric、relational、predicate 或 occupancy abstraction |
| Open Challenges in Evaluation Metrics | 缺少统一 metrics；视觉真实和功能价值经常脱钩 | 建议固定 task success、policy-ranking fidelity、executability diagnostics |

其中最关键的是 causal conditioning gap。很多系统有 world model head，但未来主要由历史画面和语言意图决定，动作只是弱条件。这样的模型可以辅助表征学习，却不能可靠做 candidate action evaluation。

## 阅读结论

这篇综述的成熟度是“前沿路线图 + 官方维护资源列表”，不是可复现单模型 recipe。它最值得复用的是三步判断法：第一，看世界模型是进入 policy、替代 simulator，还是作为 video WM；第二，看动作、未来、结构和 reward/value 在训练图里在哪里耦合；第三，看证据是 open-loop 视觉、closed-loop success，还是 policy-ranking / executability。

对世界模型高效训练来说，它给出的最大启发是：不要把目标写成“生成更好看的未来视频”。更稳的目标是训练一个可被 policy 消费的预测接口，能在动作分叉、长 horizon、接触风险、reward / termination 和 policy ranking 上提供可靠信号。下一步最需要补的证据，是统一协议下的 action sensitivity、closed-loop task utility、simulator exploitation analysis 和跨 embodiment 复现。

