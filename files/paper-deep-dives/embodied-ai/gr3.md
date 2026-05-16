---
paper_date: 2025-07
paper_title: "GR-3 Technical Report"
paper_topic: 具身智能 / VLA
---

# GR-3：少样本长时程 VLA

!!! info "论文信息"
    - 论文：`GR-3 Technical Report`
    - 链接：[arXiv:2507.15493](https://arxiv.org/abs/2507.15493)
    - 项目页：[ByteDance Seed GR-3](https://seed.bytedance.com/en/GR3)
    - 关键词：VLA、Qwen2.5-VL、flow matching、Action DiT、task status、few-shot human trajectories、ByteMini


!!! info "事实版本"
    | Checked Date | Official Source | Repro Status | Notes |
    | --- | --- | --- | --- |
    | 2026-05-16 | [arXiv:2507.15493](https://arxiv.org/abs/2507.15493) | Paper Only | 未见本站记录的第三方独立复现；按论文/官方材料保守引用。 |

### Evidence Snapshot

| Claim | Direct Source | Figure/Table/Setting | Evidence Type | Repro Status | Can Support | Cannot Prove |
| --- | --- | --- | --- | --- | --- | --- |
| `GR-3 Technical Report` 的核心结果在论文设置下成立 | [arXiv:2507.15493](https://arxiv.org/abs/2507.15493) | 主实验、架构图或 benchmark setting | Paper Result | Paper Only | 支撑本页对该方法机制、任务设置和局部收益的介绍 | 不证明跨数据分布、跨硬件、真实部署或第三方独立复现成立 |
| 本页把方法接回世界模型/效率主线 | 本站综合判断 | 成本账、失败边界与相邻专题链接 | Site Inference | Site Inference | 支撑阅读路线、实验设计和工程优先级假设 | 不能替代原论文 claim，也不能证明 closed-loop 或生产收益 |

!!! warning "读数边界"
    本页若引用结果表，只能按论文 benchmark、数据集、硬件和评测协议解释；需额外确认是否 closed-loop、是否真实机器人、是否系统吞吐、是否独立复现，不能把单项分数直接外推为规划收益或部署安全。

GR-3 是一篇很适合放进具身智能专题的技术报告，因为它把 VLA 从“看图出动作”继续推向三个更真实的问题：严格语言跟随、少样本新物体适配，以及长时程/灵巧任务执行。

它最值得记的一点是：**VLA 不只要输出 action chunk，还要知道任务是在进行、结束，还是根本无效。**

## 论文位置

GR-3 接在 π0 / π0.5、GR-2 之后看最顺。GR-2 重点是视频知识迁移；π0.5 重点是异构数据和高层子任务；GR-3 更强调任务状态、VLM co-training、few-shot human trajectory adaptation 和真实双手移动机器人部署。

![GR-3 teaser](../../assets/images/paper-deep-dives/embodied-ai/gr3/figure-1-teaser.jpeg){ width="920" }

<small>Figure source: `GR-3 Technical Report`, Figure 1. 原论文图意：展示 GR-3 在新物体、新环境、抽象指令、长时程任务和灵巧衣物操作上的能力。</small>

!!! note "这张图怎么读"
    **输入输出**：输入是少量机器人数据、视觉语言上下文和任务条件，输出是长时程操作策略。  
    **效率机制**：通过数据配方和模型结构提高少样本真实机器人任务覆盖。  
    **对主线意义**：它服务 VLA 的数据效率与开放任务泛化讨论。  
    **不能证明什么**：少样本结果不能证明所有新 embodiment 或安全恢复都解决。

## 模型结构

GR-3 使用 `Qwen2.5-VL-3B-Instruct` 作为 VLM 主干，并接 flow matching Action DiT。总规模约 `4B parameters`。

![GR-3 model architecture](../../assets/images/paper-deep-dives/embodied-ai/gr3/figure-3-model-arch.jpg){ width="920" }

<small>Figure source: `GR-3 Technical Report`, Figure 3. 原论文图意：GR-3 同时在 robot trajectories 和 vision-language data 上 co-train；左侧用 flow-matching objective 训练动作，右侧用 next-token prediction 保持视觉语言能力。</small>

| Component | GR-3 choice | Why it matters |
| --- | --- | --- |
| VLM backbone | Qwen2.5-VL-3B-Instruct | 保留强视觉语言理解和 grounding |
| Action model | flow matching Action DiT | 输出连续 action chunk |
| Conditioning | current robot state + VLM KV cache | 让动作生成利用 VLM 表征 |
| Action sequence | K-length action chunk as tokens | 建模短时动作依赖 |
| Attention | causal mask in Action DiT | 保持动作块内部时间结构 |
| Efficiency | Action DiT has about half the VLM layers and uses later-layer KV cache | 降低推理成本 |
| Stability | extra RMSNorm after attention and FFN linear layers | 提升训练稳定性和 instruction following |

GR-3 不只是把 VLM 输出接到动作头。它让动作 DiT 读取 VLM 后半层 KV cache，并通过 flow matching timestep / AdaLN 等方式生成动作块。这让视觉语言理解可以更直接地调制动作生成。

## 任务状态：in progress / terminate / invalid

GR-3 的一个关键设计，是把 task status 作为辅助监督的附加动作维度：

| Status | Meaning | System behavior |
| --- | --- | --- |
| `in progress` | task is being executed | continue rolling out action chunks |
| `terminate` | task has been successfully completed | stop and return to safe posture |
| `invalid` | instruction cannot be completed under current observation | refuse execution or ask for clarification |

论文指出，策略可能利用多视角里的虚假相关性，而不真正关注语言条件。加入 task status 后，模型被迫理解指令和当前场景是否匹配。训练时会随机把语言指令替换成无效指令，并让模型预测 `invalid`，但不监督动作块其他维度。

这对真实机器人非常关键：桌上没有刀时，“put the knife into the basket” 不应该触发机器人乱抓；它应该判断任务无效。

## 训练 recipe

GR-3 训练由三类数据共同构成：

1. robot trajectory data for imitation learning；
2. web-scale vision-language data for co-training；
3. few-shot human trajectory data from VR for adaptation。

![GR-3 data recipe](../../assets/images/paper-deep-dives/embodied-ai/gr3/figure-4-data.jpg){ width="920" }

<small>Figure source: `GR-3 Technical Report`, Figure 4. 原论文图意：展示 robot trajectory data、vision-language data 和 human trajectory data 的收集与 co-training 配方。</small>

### Robot trajectory + vision-language co-training

GR-3 把机器人轨迹和视觉语言数据联合训练。VL 数据覆盖 image captioning、visual question answering、image grounding 和 interleaved grounded image captioning。这样做的目的不是让机器人“会聊天”，而是让动作 DiT 在动作预测时仍能利用 VLM 的开放视觉语言能力。

### Few-shot human trajectory adaptation

论文用 `PICO 4 Ultra Enterprise` 收集人类轨迹。PDF 笔记里记录了一个很实用的数字：人类轨迹可达到约 `450 trajectories/hour`，高于机器人遥操作约 `250 trajectories/hour`。人类轨迹缺少腕部视图、关节状态和夹爪状态，因此训练时填充空白腕部图像，并用手部轨迹监督。

这个设计说明：新物体适配未必只能重收昂贵机器人轨迹，人类 VR 轨迹可以作为快速适配信号。

### Training tricks

| Trick | Role |
| --- | --- |
| Extra RMSNorm | 稳定 Action DiT 训练，并显著提升 instruction following |
| Multi-sampled flow timesteps per VLM forward | 加速训练，降低重复 VLM 前向开销 |
| Co-training with VL and robot data | 保住视觉语言泛化，同时学习动作 |
| Task status auxiliary dimension | 学会拒绝无效任务和判断终止 |

## ByteMini 机器人系统

GR-3 部署在 ByteMini 双手移动机器人上。

![ByteMini robot](../../assets/images/paper-deep-dives/embodied-ai/gr3/figure-5-bytemini.jpg){ width="920" }

<small>Figure source: `GR-3 Technical Report`, Figure 5. 原论文图意：展示 ByteMini 的机器人规格、多相机视角和 wrist sphere joint 的运动范围。</small>

论文强调全身顺应性控制和全身遥操作。策略 rollout 时，GR-3 用预测的 action chunk 控制 `19 DoF`，并加入 pure pursuit 和 trajectory optimization 以减少抖动、保持路径点和轨迹之间的平滑过渡。

## 实验：从抓取到长时程桌面清理

GR-3 的实验覆盖三类任务。

### Generalizable pick-and-place

训练数据约 `35K robot trajectories`，覆盖 `101 objects`，总计约 `69 hours`。Unseen Objects 设置下，对 45 个未见对象每个最多收集 10 条人类轨迹；450 条人类轨迹总时长约 30 分钟，再 co-train 20K steps。

![GR-3 pick-and-place results](../../assets/images/paper-deep-dives/embodied-ai/gr3/figure-7-pick-place-results.jpg){ width="920" }

<small>Figure source: `GR-3 Technical Report`, Figure 7. 原论文图意：展示 GR-3 在 basic、unseen environments、unseen objects 和 few-shot human trajectory settings 下的 pick-and-place 结果。</small>

### Long-horizon table bussing

这部分最能体现 task status 的价值。任务包括 Flat Setting 和 Instruction-Following Setting；invalid trials 要求模型在 `10 seconds` 内不操纵任何对象才算成功。论文记录该任务约 `101 hours` 机器人轨迹。

![GR-3 table bussing](../../assets/images/paper-deep-dives/embodied-ai/gr3/figure-8-table-bussing.jpg){ width="920" }

<small>Figure source: `GR-3 Technical Report`, Figure 8. 原论文图意：展示 table bussing 的任务设置、Flat / Instruction-Following / Invalid settings 和实验结果。</small>

### Dexterous cloth manipulation

衣物任务约 `116 hours` 机器人轨迹，要求模型处理柔体、衣架和晾衣架等复杂接触。

![GR-3 cloth manipulation](../../assets/images/paper-deep-dives/embodied-ai/gr3/figure-10-cloth-results.jpg){ width="920" }

<small>Figure source: `GR-3 Technical Report`, Figure 10. 原论文图意：展示衣物操作任务的阶段拆解、成功/失败流向和不同设置下的结果。</small>

## 训练细节要点

| Detail | GR-3 choice | Why it matters |
| --- | --- | --- |
| VLM backbone | Qwen2.5-VL-3B-Instruct | 强视觉语言 grounding |
| Total size | about 4B parameters | 比早期小 VLA 更接近 foundation policy |
| Action objective | flow matching Action DiT | 连续 action chunk 生成 |
| Task status | in progress / terminate / invalid | 让模型学会停止和拒绝 |
| Few-shot adaptation | PICO VR human trajectories | 快速适配新物体和新设置 |
| Pick-and-place data | 35K trajectories, 101 objects, 69 hours | 泛化抓取基础 |
| Table bussing data | about 101 hours | 长时程任务和 invalid refusal |
| Cloth data | about 116 hours | 柔体和灵巧操作 |
| Stabilization | RMSNorm after attention / FFN linear | 改善训练稳定和指令跟随 |

## 局限与启发

GR-3 很强，但它也暴露了 VLA 走向真实机器人的成本：需要机器人数据、VL co-training、人类轨迹适配、全身控制器、任务状态标签和大量真实评测。模型本身只是系统的一层。

它最值得复用的工程经验是三条：

1. 任务状态要进入训练和部署；
2. 人类轨迹可以作为新设置快速适配桥梁；
3. VLA action chunk 必须接控制器和平滑优化，不能直接当电机命令。

## 参考链接

1. [arXiv:2507.15493](https://arxiv.org/abs/2507.15493)
2. [GR-3 project page](https://seed.bytedance.com/en/GR3)
3. [ar5iv HTML and figures](https://ar5iv.labs.arxiv.org/html/2507.15493)
