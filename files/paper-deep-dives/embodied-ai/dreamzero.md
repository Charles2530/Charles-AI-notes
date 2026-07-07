---
title: DreamZero：世界动作模型为什么可以做零样本策略
paper_date: 2026-02
paper_title: "World Action Models are Zero-shot Policies"
paper_topic: 具身智能 / VLA
---

# DreamZero：世界动作模型为什么可以做零样本策略

!!! info "论文信息"
    **论文题名：** World Action Models are Zero-shot Policies。<br>
    **作者：** Seonghyeon Ye、Yunhao Ge、Kaiyuan Zheng、Shenyuan Gao、Sihyun Yu、George Kurian、Suneel Indupuru、You Liang Tan、Chuning Zhu、Jiannan Xiang 等（共 36 人）。<br>
    **机构：** NVIDIA。<br>
    **时间 / 主题：** 2026-02；具身智能。<br>
    **arXiv / 官方报告：** arXiv：[2602.15922](https://arxiv.org/abs/2602.15922)；官方材料：[dreamzero0.github.io/](https://dreamzero0.github.io/)。<br>
    **GitHub / 项目：** GitHub：[dreamzero0/dreamzero](https://github.com/dreamzero0/dreamzero)；项目页：[dreamzero0.github.io/](https://dreamzero0.github.io/)。<br>
    **元数据来源与核验口径：** 来源：[arXiv](https://arxiv.org/abs/2602.15922)；[官方 / 项目材料](https://dreamzero0.github.io/)；[官方 GitHub](https://github.com/dreamzero0/dreamzero)；Checked Date：2026-06-16；Repro Status：Paper / official materials / official GitHub reviewed, independent reproduction not claimed。

## 原论文故事线

### 研究背景

DreamZero 属于 World Action Model（WAM）路线，背景是 VLA 模型在语义泛化和物理动作泛化之间的落差。VLA 借助 VLM 预训练可以理解物体、语言和场景，能把“把可乐放到某个位置”这类语义指令接到已学动作上；但面对未见过的物理技能，例如解开绳结、使用新工具或在新环境中执行复杂接触运动，它往往缺少“动作怎样改变世界”的动态知识。

视频生成模型提供了另一种先验：它们不是只识别当前画面，而是学习世界随时间如何变化。WAM 的核心想法是把 future video prediction 和 action prediction 放在同一个模型里，让策略不仅输出动作，还要解释动作对应的未来视觉后果。DreamZero 就是在这个背景下提出的 14B robot foundation model：用预训练视频扩散 backbone 继承世界物理动态，再通过机器人数据把动态先验接到动作控制上。

### 问题（Challenge）

论文要解决的问题是：**如何让机器人策略在未见任务、未见环境和新 embodiment 上泛化，而不依赖每个任务的大量重复示教。** 难点在于，机器人数据昂贵且窄，VLA 的语言/视觉先验不能直接给出精确空间运动、接触动力学和电机控制。只拟合动作标签容易学到训练分布内模板，一旦任务需要新 motion，就会失效。

另一个挑战是实时闭环。WAM 如果基于大型视频扩散模型，一边生成未来视频一边输出动作，计算成本很高。真实机器人控制不能只追求想象质量，还要在低延迟下稳定执行。DreamZero 因此同时要证明 WAM 的泛化价值和系统可部署性。

### Finding

作者的 finding 是：**视频可以作为世界演化的 dense representation；当模型联合预测未来视频和动作时，它学到的不是单纯动作标签映射，而是“动作-后果”对齐的物理动态先验。** 这个洞见把策略学习从“看当前状态出动作”翻转成“预测世界如何变化，并让动作与这种变化一致”。

这个 finding 能解决泛化挑战，是因为未见任务的关键往往不是新物体名字，而是新动作后果。公开视频、人类视频或其他机器人的 video-only demonstration 虽然没有目标机器人的动作标签，但包含了物体和场景变化。WAM 可以先学这些视觉未来，再通过少量当前机器人数据学习 implicit inverse dynamics，把视觉未来映射到自己的动作空间。

### 方法

DreamZero 建在预训练 video diffusion backbone 上，模型同时预测 future video latent 和 action chunk。输入包括历史视觉观测、机器人 proprioception 和语言指令；输出包括未来视频和连续动作。论文把这种模型称为 World Action Model，而不是普通 Video Action Model，因为视频只是当前使用的世界建模目标，未来也可扩展到触觉、力反馈或 latent dynamics。

训练上，DreamZero 系统探索 heterogeneous robot data、diverse non-repetitive data 和 cross-embodiment transfer。它使用自回归视频扩散架构做 long-horizon world-action modeling，并通过模型和系统优化让 14B 模型能以 7Hz 做实时闭环控制。对于跨 embodiment，论文引入 video-only demonstrations：其他机器人或人类视频不提供当前机器人的动作，但通过联合建模和少量 play data，模型可以把视觉动态迁移到新 embodiment。

### 结论

实验显示 DreamZero 相比 state-of-the-art VLAs，在真实机器人新任务和新环境泛化上获得超过 2 倍的平均 task progress 提升。系统优化使 14B autoregressive video diffusion model 可以在 7Hz 下闭环控制。跨 embodiment 实验中，使用其他机器人或人类的 video-only demonstrations，在仅 10-20 分钟数据下使 unseen task performance 相对提升超过 42%；新 embodiment 适配中，仅 30 分钟 play data 即可迁移，并保持 zero-shot generalization。

论文也给出边界：DreamZero 当前更像 System 1 短时策略，visual memory 大约 6 秒，长时复杂执行仍可能需要 System 2 planner 或更长上下文 WAM；高精度装配、插钥匙等亚厘米任务仍继承行为克隆的局限；高自由度 embodiment 需要更多 play data 来学习 implicit inverse dynamics。结论不是“视频模型直接解决机器人”，而是“联合世界预测和动作预测能显著提高物理泛化”。

### 关键术语

- **World Action Model, WAM（世界动作模型）**：联合预测未来世界状态和动作的机器人策略模型。
- **Vision-Language-Action, VLA（视觉语言动作模型）**：以视觉和语言为条件输出机器人动作的模型。
- **Future World State（未来世界状态）**：模型预测的未来视频或其他世界表征，用来刻画动作后果。
- **Implicit Inverse Dynamics Model（隐式逆动力学模型）**：从期望视觉变化中隐式推断本体动作的能力。
- **Cross-Embodiment Transfer（跨具身迁移）**：把来自其他机器人或人类视频的动态知识迁移到当前机器人。
- **Play Data（自由探索数据）**：机器人在新 embodiment 上采集的少量非任务化交互数据，用于适配动作空间。

DreamZero 这篇论文要回答一个比“视频生成模型能不能预测未来”更尖锐的问题：**如果模型一边预测未来画面，一边预测产生这些画面的机器人动作，它能不能直接变成闭环机器人 policy？**

这就是论文所谓的 World Action Model，简称 WAM。普通 VLA 学的是“当前图像和语言 -> 动作”；WAM 学的是“当前世界状态和语言 -> 未来世界怎样变化，以及机器人要做哪段动作”。两者都输出动作，但学习信号完全不同。VLA 容易把动作当成标签回归；WAM 被迫解释动作的后果，因此更接近“我知道这样动会把世界推到哪里”。

## 一句话核心

DreamZero 的核心不是把视频模型拿来当想象模块，而是把视频扩散模型改造成一个联合去噪器：

\[
p_\theta(v_{t+1:t+H}, a_{t:t+K} \mid o_{\le t}, s_{\le t}, l)
\]

其中 \(v_{t+1:t+H}\) 表示未来视频 latent，\(a_{t:t+K}\) 表示未来一段 action chunk，\(o_{\le t}\) 是历史视觉观测，\(s_{\le t}\) 是机器人本体状态，\(l\) 是语言指令。这个式子最重要的部分不是条件很多，而是输出同时包含 video 和 action：动作不再只是独立回归目标，而必须和未来画面变化一致。

![DreamZero overview](../../assets/images/paper-deep-dives/world-models/dreamzero/figure-1-overview.png){ width="920" }

<small>图源：`World Action Models are Zero-shot Policies`，Figure 1。原图表达 DreamZero 的路线：从预训练视频扩散模型继承世界动态先验，再通过视频-动作联合建模支持多样机器人数据、未见任务泛化、跨具身迁移和少样本适配。本站读法：先看它没有把 video prediction 当辅助任务，而是把 video 和 action 都放进 policy 输出。</small>

## VLA 缺的不是语言，而是动作后果

很多 VLA 论文强调语义泛化：模型见过“杯子”“桌子”“把 A 放到 B”这类语言和视觉概念，所以能把指令映射到动作。但机器人操作的困难不只在语义。一个策略需要知道“夹爪往左 3 厘米后，物体是否会被碰倒”“拉抽屉时把手和手腕轨迹会怎样变化”“从这个角度抓会不会滑落”。

这类知识更像物理动态，而不是语言知识。VLM 的先验能在语义上告诉机器人“该做什么”，但缺少与几何结构、物理动力学和底层电机控制对齐的“怎么精确做”。DreamZero 的批评点在这里：如果训练目标只让模型拟合动作标签，它可以在训练集附近学到动作模板，却不一定学到动作如何改变世界。相反，如果模型必须预测未来视频，它就要关心物体、手、相机、接触和遮挡的连续变化。

可以把 VLA 和 WAM 的目标粗略写成：

\[
\text{VLA}: \quad a_t = f_\theta(o_t, s_t, l)
\]

\[
\text{WAM}: \quad (v_{t+1:t+H}, a_{t:t+K}) = g_\theta(o_{\le t}, s_{\le t}, l)
\]

其中第一行表示单纯从当前观测回归动作，第二行表示同时预测未来世界和一段动作。这里的差异不是公式复杂度，而是训练时模型有没有被要求解释“动作之后会看到什么”。DreamZero 的整个贡献都围绕第二行展开。

![Joint video and action prediction](../../assets/images/paper-deep-dives/world-models/dreamzero/figure-2-joint-video-action.png){ width="920" }

<small>图源：`World Action Models are Zero-shot Policies`，Figure 2。原图表达模型同时生成未来视频和动作。本站读法：观察动作轨迹和预测画面是否互相支持，尤其是未见任务中模型是否能生成合理的动作后果。</small>

## 和先生成视频再取动作的差别

在 DreamZero 之前，已经有一类工作证明视频生成模型可以为机器人提供运动线索。常见做法是先生成未来视频，再用额外模块把视频转成可执行动作。

| 路线 | 做法 | 主要边界 |
| --- | --- | --- |
| 逆动力学模型 | 给定当前帧和生成的目标帧，反推需要执行的动作 | 依赖单独训练的 inverse dynamics model，视频和动作可能错位 |
| 光流 / 稠密对应 | 从生成视频里估计像素、关键点或末端执行器运动，再映射到控制指令 | 更偏几何线索，仍需要额外转换和控制接口 |
| 轨迹预测作高层规划 | 生成未来轨迹或关键点，让下层 controller 负责精确执行 | 视频只给方向，不直接约束低层动作 |

这三类方法的共同点是 video generation 和 action generation 被拆开：视频模型先“想象”，动作模块再“翻译”。DreamZero 的关键变化，是让同一个自回归 DiT 同时去噪未来视频 latent 和动作 latent。这样第 \(i\) 段动作不再只是从视频后处理出来，而是在训练时就要和第 \(i\) 段视觉变化对齐。

## 架构：把 Wan2.1 视频扩散模型改成动作模型

DreamZero 使用 Wan2.1-I2V-14B-480P 作为 backbone。这个选择很重要：论文并没有从头训练一个机器人 transformer，而是复用大视频模型已经学到的视觉动态、物体外观、相机运动和场景变化先验。

输入端可以拆成三路。第一路是 visual context，也就是机器人当前或过去几帧看到的多视角图像 / 视频，先由 VAE encoder 压到 latent 空间；第二路是 language instruction，经 text encoder 编成任务条件；第三路是 proprioceptive state，包括关节角、末端执行器位置、夹爪开合和速度等，由 state encoder 接入 backbone。输出端则分成 video decoder 和 action decoder，分别恢复未来画面和动作块。

为了让视频模型理解机器人，DreamZero 只增加几类机器人相关模块：

- `state encoder`：把本体状态编码进模型。
- `action encoder`：训练时把带噪动作 latent 放入联合去噪。
- `action decoder`：把去噪后的 action latent 解码为机器人 action chunk。
- 多视角输入拼接：把多摄像头画面拼成一个视频帧，避免重写 backbone。

这也解释了 DreamZero 和 Wan、Diffusion Forcing 的关系。Wan 提供的是强视频 latent 与 DiT 动态先验；DreamZero 把这个视频底座接上状态、动作和闭环真实观测；Diffusion Forcing 则从训练范式上提醒我们，未来 token 的不确定性不应该全一样。三者合起来看，路线不是“视频越真越像世界模型”，而是：视频底座要有动作接口，采样/训练要表达不确定性，真实执行时还要用观测回写打断错误想象。

![DreamZero architecture](../../assets/images/paper-deep-dives/world-models/dreamzero/figure-4-architecture.png){ width="920" }

<small>图源：`World Action Models are Zero-shot Policies`，Figure 4。原图表达 DreamZero 架构：视觉上下文、语言和本体状态进入自回归 DiT，训练时联合去噪视频与动作 latent，推理时用真实观测刷新上下文。本站读法：把这张图看成“视频扩散模型如何多出动作接口”，而不是普通 VLA 架构图。</small>

训练样本被切成 chunk。每个 chunk 内，模型看到历史条件帧和机器人状态，然后对未来视频 latent 与动作 latent 做 flow matching。一个简化的训练目标可以写成：

\[
\mathcal{L}
= \mathbb{E}_{t, \epsilon}
\left[
\lambda_v \lVert u_v - \hat{u}_{v,\theta} \rVert_2^2
+
\lambda_a \lVert u_a - \hat{u}_{a,\theta} \rVert_2^2
\right]
\]

其中 \(u_v\) 和 \(u_a\) 分别表示视频 latent 与动作 latent 的目标速度场，\(\hat{u}_{v,\theta}\) 和 \(\hat{u}_{a,\theta}\) 是模型预测，\(\lambda_v,\lambda_a\) 是两类损失的权重。这里要读懂的是“同一个模型同时学两个去噪方向”：视频去噪要求它理解未来画面，动作去噪要求它把这种未来变化落到机器人控制上。

训练配置也值得记住。论文在 AgiBot 和 DROID 两套数据上都训练 100K steps，global batch size 为 128；更新所有 DiT blocks、state encoder、action encoder 和 action decoder，但冻结 text encoder、image encoder 和 VAE。动作侧过滤 idle actions，并默认使用 relative joint positions。这个配方说明 DreamZero 不是简单给 Wan 加一个 LoRA 动作头，而是在保留视觉 / 文本编码器稳定性的同时，让 DiT 主干真正吸收机器人动作接口。

## 为什么要自回归，而不是一次性生成整段视频

离线视频生成可以一次性生成几十帧，但机器人闭环控制不能这样。真实执行时，机器人每走一步都会得到新观测；如果模型继续相信自己上一轮生成的未来画面，误差会越来越大。DreamZero 因此采用 autoregressive WAM：生成一个 action chunk 后，系统等待真实观测回来，再把真实观测写回上下文。

这有两个直接好处。

第一，KV cache 能复用历史条件帧，让大模型推理不必每次从头算。第二，真实观测会替换模型想象的画面，把错误从闭环中截断。也就是说，DreamZero 不是让机器人活在生成视频里，而是每个控制周期都用真实世界纠偏。

自回归还有一个对齐层面的好处：它按真实时间顺序生成视频和动作，能减轻“第几帧对应第几个动作”“语言指令对应哪段视觉变化”“动作片段是否服务当前指令”这类 modality alignment 问题。相比一次性双向生成整段序列，自回归让每个 chunk 的视觉上下文、动作输出和真实观测刷新更容易挂在同一条时间线上。

![DreamZero attention strategy](../../assets/images/paper-deep-dives/world-models/dreamzero/figure-14-attention-strategy.png){ width="860" }

<small>图源：`World Action Models are Zero-shot Policies`，Figure 14。原图表达训练和推理时的注意力策略。本站读法：重点看推理侧，条件帧 KV 被缓存，生成的 future frame 不是长期信念，真实观测会回写到上下文中。</small>

## 实时化：WAM 的难点是 14B 视频模型要跑闭环

DreamZero 最容易被低估的部分是系统实现。一个 14B 视频扩散模型天然不适合机器人控制：朴素生成一个 action chunk 需要数秒，而真实机器人需要接近 5-10Hz 的响应频率。论文把这个问题称作 `reactivity gap`。最终系统通过约 38 倍推理加速，让模型在 2 台 GB200 上以约 7Hz 进行闭环控制；这已经能跑真实机器人，但仍比很多轻量 VLA 的消费级 GPU 频率更贵。

它的优化分三层：

| 层级 | 做法 | 解决的问题 |
| --- | --- | --- |
| 系统层 | asynchronous execution、CFG parallelism、DiT caching | 让感知、生成和执行流水化；conditional / unconditional CFG forward 可并行；相邻 velocity 方向相近时复用缓存，把平均 DiT steps 从 16 降到 4 |
| 实现层 | torch.compile、CUDA Graphs、kernel / scheduler 优化、NVFP4 量化 | 降低每步 DiT 推理成本；Blackwell 上权重和激活用 NVFP4，QKV / Softmax 等敏感操作保留更高精度 |
| 模型层 | DreamZero-Flash、少步去噪、decoupled noise schedules | 减少扩散步数，同时保住动作质量 |

这张加速表的读法不是“用了很多工程技巧”，而是看瓶颈迁移：单靠缓存、编译和量化只能把延迟压下一部分；真正让 WAM 进入闭环的是少步化模型本身。

## DreamZero-Flash：为什么动作和视频要用不同噪声日程

少步扩散容易出现一个问题：视频生成还没有充分去噪，动作已经必须可执行。标准 DreamZero 里 video 和 action 使用同一个 denoising timestep，少步推理时容易出现 train-test mismatch：训练时动作依赖较干净的视频上下文，部署时视频 token 还很 noisy，动作却已经要落到机器人上。

DreamZero-Flash 的做法是 decoupled noise schedules：动作 timestep 仍然均匀采样，视频 timestep 则偏向高噪声状态，例如写成

\[
t_{\text{video}} = 1 - \eta,\quad \eta \sim \operatorname{Beta}(7, 1)
\]

这样模型在训练时经常要从 noisy visual context 里预测 clean action。直觉上，视频 latent 可以保持更高噪声，因为它主要提供未来动态方向；动作 latent 则需要更快靠近可执行值。

![Decoupled noise schedules](../../assets/images/paper-deep-dives/world-models/dreamzero/figure-5-decoupled-noise.png){ width="860" }

<small>图源：`World Action Models are Zero-shot Policies`，Figure 5。原图表达 coupled 和 decoupled noise schedule 的区别。本站读法：动作不是等视频完全生成好再预测，而是在有噪视觉上下文里尽快得到可执行 chunk。</small>

这背后的直觉很实用：机器人不需要每个控制周期都生成一段电影级未来视频，它需要从粗糙但方向正确的未来动态里抽出下一段动作。论文在 table bussing 上报告：标准 DreamZero 从 4-step 降到 1-step 时 task progress 从约 83% 掉到 52%；DreamZero-Flash 的 1-step 版本约 74%，延迟从约 350ms 降到 150ms。这个数字说明 Flash 的价值不是“视频更好看”，而是让动作在极少步去噪下仍然可用。

## 数据：为什么 diverse data 对 WAM 特别重要

DreamZero 使用两类机器人数据：AgiBot G1 的约 500 小时 teleoperation data，以及 DROID-Franka 这类公开异质机器人数据。AgiBot 数据覆盖 22 个独特环境，包括 homes、restaurants、supermarkets、coffee shops、offices 等；论文统计约 7.2K episodes，单条 episode 平均约 4.4 分钟、约 42 个 subtasks。它强调的不是“同一个任务做很多遍”，而是在真实场景里收集有用行为的长尾组合。

对 VLA 来说，多样但非重复的数据可能带来动作分布噪声；对 WAM 来说，这些数据还有一个额外价值：即使动作标签不完美，未来视频仍然告诉模型世界怎样变化。因此 WAM 更可能从 heterogeneous data 里学习可迁移动态。

论文的消融也支持这个方向：diverse data、14B model scale、自回归闭环结构都不是装饰。尤其是模型规模，视频动态先验和机器人动作接口都依赖大 backbone 的表达能力；如果模型太小，联合预测很容易退化成普通动作回归。

| 消融问题 | 结果读法 | 含义 |
| --- | --- | --- |
| diverse data vs. repetitive data | 同样 500 小时，PnP Easy task progress 从约 33% 到 50% | WAM 需要多样状态-动作对应来学习隐式 inverse dynamics |
| 5B vs. 14B WAM | 14B 约 50%，5B 约 21% | 视频生成质量会传导到动作执行，scale 不只是语义能力 |
| VLA scale-up | 5B / 14B VLA 在同样 diverse data 上接近 0% | 只放大 VLA 不等于能吃下异质非重复数据 |
| bidirectional vs. autoregressive | task progress 接近，但 AR 运动更平滑，推理因 KV cache 更快 | 自回归的优势主要体现在时间一致性和部署路径 |

## 实验应该怎样读

DreamZero 的实验不应只看平均分，而要按四个问题读。

第一，seen tasks 的新环境和新物体能不能泛化。论文报告 AgiBot G1 上 DreamZero 平均 task progress 为 62.2%，超过最佳 pretrained VLA baseline 的 27.4%，并且是在不同机器人、不同环境、不同物体上做 out-of-the-box 评测。这个结果说明 WAM 从多样数据中学到的不是单个环境模板。

![Seen task evaluation](../../assets/images/paper-deep-dives/world-models/dreamzero/figure-8-seen-task.png){ width="920" }

<small>图源：`World Action Models are Zero-shot Policies`，Figure 8。原图表达 seen tasks 在新环境和新物体设置下的表现。本站读法：看 DreamZero 是否在每类任务上都提升，而不是只靠某个容易任务拉高平均值。</small>

第二，训练中没出现过的 tasks 能不能有非平凡进展。论文覆盖 untying shoelaces、ironing、painting with a brush、shaking hands 等任务。在 AgiBot G1 上，DreamZero unseen tasks 平均 task progress 为 39.5%，高于 pretrained VLA baseline 的 16.3%；在 DROID-Franka 上，DreamZero 对 unseen verbs 达到 49% task progress / 22.5% success rate，高于 GR00T N1.6 等 baseline。论文还做了 100+ free-form prompting 测试，但那部分更像展示能力边界，不能和结构化 benchmark 混成同一种证据。

![Zero-shot generalization](../../assets/images/paper-deep-dives/world-models/dreamzero/figure-9-zero-shot.png){ width="920" }

<small>图源：`World Action Models are Zero-shot Policies`，Figure 9。原图表达 unseen tasks 的 zero-shot generalization。本站读法：这组图最接近论文标题中的 zero-shot policy，但仍要记住评测任务来自作者设定的机器人平台和任务集。</small>

第三，post-training 会不会破坏开放泛化。论文在 shirt folding、fruit packing、table bussing 三个下游任务上继续训练，分别收集约 33 小时、12 小时和 40 小时数据，并每个任务训练 50K steps。结果显示任务表现提高，同时保留一定环境泛化。这个结论比“后训练涨分”更重要，因为很多机器人 policy 一微调就会过拟合实验室环境。

![Posttraining results](../../assets/images/paper-deep-dives/world-models/dreamzero/figure-10-posttraining.png){ width="920" }

<small>图源：`World Action Models are Zero-shot Policies`，Figure 10。原图表达 post-training 后的任务结果。本站读法：关注后训练是否把模型变成窄任务专家，还是仍能保留 WAM 的泛化优势。</small>

第四，cross-embodiment transfer 是否真的来自视频动态。论文用 robot-to-robot 和 human-to-robot 的 video-only demonstration 改善未见任务表现：YAM 机器人约 20 分钟、人类第一视角约 12 分钟，都是只用视频目标，不用跨本体动作标签；之后与原 AgiBot 预训练数据按 1:1 混合继续训练 10K steps。结果从 DreamZero baseline 的 38.3% 提升到 Human2Robot 54.3%、Robot2Robot 55.4%。这个实验很有价值，因为它绕开了动作空间不一致的问题：视频可以跨具身共享，但动作标签不能直接共享。

![Cross-embodiment transfer](../../assets/images/paper-deep-dives/world-models/dreamzero/figure-11-cross-embodiment.png){ width="920" }

<small>图源：`World Action Models are Zero-shot Policies`，Figure 11。原图表达跨具身迁移，包括机器人到机器人和人到机器人。本站读法：把它看成 WAM 路线最有想象力的证据，前提是任务和场景仍在论文覆盖范围内。</small>

第五，few-shot embodiment adaptation 是否只是换个机器人微调。论文把 AgiBot G1 上预训练的 DreamZero 迁移到新 YAM 机器人，只用约 30 分钟 play data、55 条 trajectories、11 个 unique tasks。值得注意的是，它不是宣称任何新本体都能 30 分钟适配，而是在两个双臂平行夹爪机器人之间展示了很高的数据效率；作者也把失败主要归因于视频预测错误，而不是动作提取本身。

## 边界：WAM 不是万能机器人大脑

DreamZero 的强主张很吸引人，但边界也要写清楚。

第一，它仍然依赖真实机器人数据。视频 backbone 提供动态先验，但动作接口、控制频率、末端执行器习惯、状态编码都要靠机器人数据对齐。第二，它的 closed-loop control 依赖很强的系统工程；没有低延迟推理，WAM 只能是离线想象器。第三，视频预测失败会直接诱导错误动作，因为动作和未来画面在模型里是耦合的。第四，当前模型更像短时 System 1 policy，论文披露的视觉上下文约 6.6 秒，长时任务仍可能需要 planner 或更长上下文。第五，高精度装配、插钥匙这类 sub-centimeter 任务没有被充分证明，宽泛泛化和精细接触并不是同一个能力。

![Failure case](../../assets/images/paper-deep-dives/world-models/dreamzero/figure-16-failure-case.png){ width="920" }

<small>图源：`World Action Models are Zero-shot Policies`，Figure 16。原图表达 generated video 与 executed action 的失败配对。本站读法：当未来视频计划错了，动作也会沿着错误计划执行，这正是 WAM 比普通 VLA 更强也更危险的地方。</small>

所以更准确的结论是：DreamZero 证明了“视频世界模型 + 动作联合建模 + 闭环真实观测回写”可以成为一条可工作的机器人 policy 路线；它还没有证明一个 WAM 可以替代所有 VLA、规划器和低层控制器。

## 证据链快照

| 论文主张 | 主要证据 | 读数边界 |
| --- | --- | --- |
| WAM 联合预测未来视频和动作 | Figure 1/2/4、video-action diffusion objective | 视频-动作一致不等于安全可执行 |
| 预训练视频模型可作为机器人动态底座 | Wan2.1 backbone、机器人状态/action 模块 | 动作接口仍依赖真实机器人数据对齐 |
| 自回归 + KV refresh 支持闭环 | attention strategy、真实观测刷新 | 实时性高度依赖系统优化和硬件 |
| DreamZero-Flash 降低推理成本 | decoupled noise schedule、Flash table bussing 实验 | 少步化可能影响预测质量，需要任务评测 |
| diverse data 比重复示范更适合 WAM | 500 小时 diverse vs. repetitive 消融 | 说明数据分布重要，不等于数据规模已充分探索 |
| zero-shot / cross-embodiment 有启发 | seen / zero-shot / cross-embodiment / YAM adaptation 图表 | 不等于未见硬件或开放世界任务都可靠 |

## 阅读结论

DreamZero 最重要的知识点是“动作和未来画面要共同去噪”。普通 VLA 可以把动作当标签回归，旧的视频规划路线常常先生成视频再额外提取动作；WAM 则要求动作必须在生成过程中解释未来世界怎样变化，这让它更接近 action-conditioned world model。真正让论文成立的不是一个大标题，而是五个条件同时出现：Wan2.1 这类视频动态先验，多样真实机器人数据，video-action joint diffusion objective，自回归 + KV refresh 的闭环路径，以及 DreamZero-Flash / 系统优化带来的实时性。读完这篇后，应该能区分三件事：VLM 看到什么，VLA 现在做什么，WAM 做了以后世界会怎样变。

## 外部精读

- [World Action Models are Zero-shot Policies](https://arxiv.org/abs/2602.15922)：论文原文，重点读方法、系统加速和真实机器人实验。
- [DreamZero project page](https://dreamzero0.github.io/)：项目页有演示视频，适合核对 zero-shot、post-training 和 cross-embodiment 的实际效果。
- [Wan 2.1 technical report](https://arxiv.org/abs/2503.20314)：理解 DreamZero 为什么选择 Wan2.1-I2V-14B 作为视频扩散 backbone。
- [Diffusion Forcing](https://arxiv.org/abs/2407.01392)：作为序列扩散训练范式对照，理解 per-token uncertainty 和 causal rollout 为什么重要。
- [DROID dataset](https://droid-dataset.github.io/)：理解 DreamZero 为什么强调异质机器人数据。
- [π0.5: a Vision-Language-Action Model with Open-World Generalization](https://www.physicalintelligence.company/blog/pi05)：作为 VLA 路线对照，帮助区分 semantic generalization 和 world-action modeling。

## 相关阅读与下一步

- 外部材料：[论文标题 arXiv 检索](https://arxiv.org/search/?query=dreamzero&searchtype=title)。
- 外部材料：[Semantic Scholar 标题检索](https://www.semanticscholar.org/search?q=dreamzero)。
- 外部材料：[Papers with Code 检索](https://paperswithcode.com/search?q=dreamzero)。
- 站内下一步：[论文精读专题](../index.md)。
- 站内下一步：[论文专题写作规范](../writing-standard.md)。
- 站内下一步：[外部精读来源台账](../../references/external-reading-ledger.md)。
