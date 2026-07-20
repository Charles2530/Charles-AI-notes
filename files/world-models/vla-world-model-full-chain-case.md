# VLA 到世界模型：一条机器人闭环链路怎么搭

VLA、世界模型、训练系统和评测经常被分开讲，但真实项目里它们必须接成一条链：机器人数据给出观测和动作，VLA 产生候选动作，世界模型预测动作后果，评测系统判断这些预测是否真的改善闭环决策。

做“机器人数据 + VLA 动作 + 世界模型预测 + 闭环评测”方案时，每一层都要讲清输入、输出、证据和失败边界。

## 数据和动作先对齐

VLA / 世界模型项目最常见的失败，不是模型不够大，而是视频和动作日志没有对齐。视频里看得到状态变化，但动作日志缺单位、坐标系、频率或控制模式；VLA 能输出动作 token，但世界模型不知道这些 token 如何改变未来状态。

![RT-2 overall architecture](../assets/images/paper-figures/vla/rt2-overall-wide.png){ width="920" }

<small>图源：[RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control](https://arxiv.org/abs/2307.15818)，整体结构图。原图展示视觉语言模型如何扩展到机器人动作 token 输出。读图重点：RT-2 的启发是把 web-scale VLM 知识接到动作接口上，但世界模型还必须知道动作 token、坐标系和控制频率怎样进入 dynamics。</small>

一个最小 episode schema 应同时记录：图像或多相机视频、语言指令、机器人本体状态、动作 chunk、控制频率、坐标系、reward / done、失败类型、人工接管、near-miss 和恢复片段。只存成功 demonstration 会让模型误以为所有动作都通向成功；只存视频不存动作，会让世界模型无法做反事实。

## 数据覆盖决定泛化上限

真实机器人数据昂贵，所以数据集不能只追求干净。它要覆盖不同机器人、不同场景、不同任务、失败和恢复过程。多 embodiment 数据的价值，是让模型看到同一语义目标可以对应不同身体和控制接口。

![Open X-Embodiment dataset overview](../assets/images/paper-figures/vla/openx-data-vis.jpg){ width="920" }

<small>图源：[Open X-Embodiment: Robotic Learning Datasets and RT-X Models](https://arxiv.org/abs/2310.08864)，数据集可视化图。原图展示多机器人、多任务、多场景数据聚合。读图重点：多数据集聚合能扩大覆盖，但不能自动解决动作空间不一致、标注噪声和跨机器人安全迁移。</small>

数据账本至少要能回答：哪些任务成功，哪些失败，失败是碰撞、遮挡、滑落、错目标、不可达还是人工接管；哪些动作导致 near-miss；哪些场景属于长尾；哪些样本适合进入再训练，哪些只能做人审或评测。

## 先压缩状态，再学动作条件 dynamics

四路相机、16 帧历史、RGBD、多模态 token 和语言上下文很快会把 context 撑爆。工程上通常先把视觉压成 latent 或表征，再在该状态上学习动作条件转移。

![MWM framework](../assets/images/paper-deep-dives/world-models/masked-world-models/figure-1-mwm-framework.png){ width="920" }

<small>图源：[Masked World Models for Visual Control](https://arxiv.org/abs/2206.14244)，Figure 1。原图展示 masked visual representation learning 与 latent dynamics 的解耦。读图重点：状态层不是为了重建所有像素，而是保留对控制有用的视觉结构、小物体、接触和动作后果。</small>

状态压缩要被测试，而不是只报告 compression ratio。压缩后，小物体是否还在，接触状态是否可读，遮挡后的目标是否保持身份，不同动作是否造成可区分未来，risk head 是否还能识别危险。否则压缩只是把关键信息一起扔掉。

![V-JEPA 2 action-conditioned world model](../assets/images/paper-deep-dives/world-models/v-jepa2/figure-action-conditioned-world-model.png){ width="920" }

<small>图源：[V-JEPA 2](https://arxiv.org/abs/2506.09985)，Figure 7。原图展示 action-conditioned 模块如何在 JEPA 表征空间中做目标图像规划。读图重点：JEPA/latent 表征适合降低像素重建压力，但仍要补动作条件、reward/risk 和闭环评测。</small>

## 世界模型要影响动作选择

全链路里，世界模型最有价值的输出不是“未来视频”，而是“哪个候选动作更可能成功、更安全、更可恢复”。因此实验报告必须写出候选动作、预测成功率、预测风险、真实结果和最终选择。

一个最小决策账可以这样组织：

```text
history observation
  -> VLA proposes K action chunks
  -> world model rolls out each candidate
  -> risk / success / reward heads score candidates
  -> planner chooses action
  -> real execution logs outcome and failure type
```

这里的核心指标是 candidate ranking agreement：世界模型预测的 top-1 安全成功动作，是否和真实安全成功动作一致。没有这项，所谓“世界模型帮助规划”就很难成立。

## 失败必须进入报告

一个合格的全链路报告至少有四张账。

| 账本 | 最小字段 | 为什么重要 |
| --- | --- | --- |
| 数据账 | episode、camera、action、reward、done、failure_type | 没有字段契约就无法复盘 |
| 状态账 | raw tokens、compressed tokens、关键状态探针 | 证明压缩省了什么、丢了什么 |
| 决策账 | candidate action、predicted success/risk、chosen action | 证明模型影响了动作选择 |
| 失败账 | slip、wrong_target、occlusion、collision、recovery | 证明下一轮数据该补哪里 |

失败账不是附录。世界模型最需要的往往不是更多成功样本，而是边界样本：快碰撞但没碰、差一点滑落、遮挡导致错抓、语言理解正确但动作不可达、成功 checker 漏判。它们决定下一轮训练是否真的补上系统短板。

## 交付顺序

第一步，确定动作 schema：坐标系、频率、chunk 长度、归一化、单位和控制器接口。第二步，确定状态压缩：视觉 token、几何状态、mask、episode boundary 和关键状态探针。第三步，训练 VLA 或动作生成器，让它能产生候选动作而不是只输出单一步动作。第四步，训练世界模型预测 reward、risk、done 或未来 latent。第五步，评测 action sensitivity、candidate ranking、risk ECE、failure replay 和 closed-loop success。第六步，把失败样本回流到数据桶。

这条链路最怕跳步。没有动作 schema，VLA 输出无法被世界模型消费；没有状态探针，压缩是否丢掉接触和小目标没人知道；没有候选动作账，世界模型是否改善决策没人能证明；没有失败回放，下一轮训练只能盲目加数据。

## 外部材料

- [RT-2 paper](https://arxiv.org/abs/2307.15818) 与 [DeepMind RT-2 blog](https://deepmind.google/discover/blog/rt-2-new-model-translates-vision-and-language-into-action/)：理解 VLM 如何被训练成 VLA。
- [Open X-Embodiment](https://arxiv.org/abs/2310.08864)：理解多机器人、多任务数据为什么是 VLA 泛化的重要来源。
- [Masked World Models for Visual Control](https://arxiv.org/abs/2206.14244)：理解表征学习和 latent dynamics 为什么可以解耦。
- [V-JEPA 2](https://arxiv.org/abs/2506.09985)：理解 JEPA 表征如何接 action-conditioned planning。
- [DreamerV3](https://arxiv.org/abs/2301.04104)：理解 imagined rollout 如何服务策略学习。

- [World Models 论文](https://arxiv.org/abs/1803.10122)。
- [DeepMind Genie 2](https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/)。
- [Meta V-JEPA 2](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/)。
