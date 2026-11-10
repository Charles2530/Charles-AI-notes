---
title: 技术地图
hide:
  - navigation
  - toc
---

<div class="atlas-hero" markdown="1">

# AI 研究与工程地图

同一个模型，loss 在降，显存却先爆了；同一个 kernel，单测变快，端到端请求却没有变化。论文和代码真正难对上的地方，往往就在这些小断点里。

我把每个专题都从一个能检查的断点开始写：输入是什么，中间量变成什么，哪里会失败。公式、代码和来源跟在这个断点后面，论文结论、本站算例和本地运行结果也分开记。你可以从眼前的报错进入，也可以从基础一路读到训练、推理和具身系统。

<div class="atlas-hero-actions">
  <a href="foundations/">从基础知识进入</a>
  <a href="paper-deep-dives/">查找论文精读</a>
</div>

</div>

## 按技术问题查阅

<div class="atlas-topic-matrix">
  <a class="atlas-topic-card" href="foundations/"><strong>基础知识</strong><p>数据怎样变成预测，损失怎样变成梯度。把数学、张量与模型计算连起来。</p><small>数据 · 概率 · 张量 · 自动微分</small></a>
  <a class="atlas-topic-card" href="model-architectures/"><strong>模型结构</strong><p>从线性层与卷积到 attention，理解每个模块的运算以及模块之间的配合。</p><small>MLP · CNN · Transformer · MoE · SSM</small></a>
  <a class="atlas-topic-card" href="training/"><strong>训练</strong><p>数据提供什么信号，目标如何驱动参数更新，训练过程如何扩展和保持稳定。</p><small>优化 · 预训练 · 微调 · 分布式训练</small></a>
  <a class="atlas-topic-card" href="inference/"><strong>推理与效率</strong><p>一条请求怎样得到输出，缓存、调度、量化与算子分别改变哪些开销。</p><small>解码 · KV cache · 量化 · GPU 计算</small></a>
  <a class="atlas-topic-card" href="diffusion/"><strong>生成模型</strong><p>怎样学习数据分布，噪声预测与速度预测怎样通过采样器变成图像和视频。</p><small>潜变量 · 扩散 · Flow Matching · 蒸馏</small></a>
  <a class="atlas-topic-card" href="reinforcement-learning/"><strong>强化学习</strong><p>动作改变未来时，如何从交互数据估计价值并改进策略。</p><small>MDP · 价值学习 · 策略梯度 · 离线 RL</small></a>
  <a class="atlas-topic-card" href="world-models-and-embodied-ai/"><strong>多模态与具身</strong><p>视觉与语言怎样形成表示，预测、决策和控制怎样连接到环境。</p><small>VLM · 世界模型 · VLA · 机器人</small></a>
  <a class="atlas-topic-card" href="paper-deep-dives/"><strong>论文精读</strong><p>回到方法的具体改动、关键公式和实验，理解已有结论与实现条件。</p><small>方法机制 · 训练推理 · 实验证据</small></a>
</div>

## 从一个具体疑问开始

| 当前的问题 | 直接阅读 | 进一步关联 |
| --- | --- | --- |
| 为什么矩阵维度对了，模型结果仍然错 | [张量与轴的含义](foundations/tensors-shapes-and-computation-graphs.md) | [线性层怎样组合特征](foundations/linear-layers-mlp-and-gemm.md) |
| loss 下降时，模型究竟发生了什么变化 | [梯度与参数更新](foundations/optimization-and-training-basics.md) | [自动微分怎样计算梯度](foundations/autograd-activation-checkpointing-and-memory.md) |
| Q、K、V 究竟做了哪些计算 | [一次 attention 的完整计算](foundations/transformer-attention-and-tokenization.md) | [位置与可见性](foundations/positional-encoding-masks-and-context.md) |
| 模型为什么每次可能给出不同回答 | [生成与解码](foundations/generation-and-decoding-basics.md) | [推理中的状态与缓存](inference/caching-routing-and-speculative.md) |
| 模型文件更小，为什么运行没有更快 | [量化对象与误差](quantization/ptq-gptq-awq-smoothquant.md) | [算子成本与选择](operators/kernel-cost-models-and-selection-heuristics.md) |
| 给图像加噪，为什么能学会生成 | [扩散的核心机制](diffusion/foundations-and-core-intuition.md) | [公式对应到 DDPM 代码](diffusion/minimal-ddpm-implementation.md) |
| 一段视频怎样从噪声变成连续画面 | [视频生成的完整过程](diffusion/video-and-multimodal-diffusion.md) | [VAE 的时间压缩](diffusion/video-vae-1029-1246.md) · [Video DiT](diffusion/video-dit-1029-1246.md) |
| 当前奖励与长期价值有什么区别 | [价值函数与 Bellman](reinforcement-learning/mdp-value-bellman.md) | [从样本估计价值](reinforcement-learning/dynamic-programming-monte-carlo-td.md) |
| 能预测未来画面，怎样才能帮助机器人行动 | [世界模型的基本对象](world-models/beginner-world-models-primer.md) | [RSSM、Dreamer 与规划](world-models/rssm-dreamer-and-planning.md) |

## 先把一件小事算清楚

想知道模型如何学习，可以先算[两个样本的一次梯度更新](foundations/optimization-and-training-basics.md)。想理解生成，可以先算[一个VAE的采样、损失与梯度](foundations/probability-latent-variables-and-generative-models.md)，再进入扩散或视频。想理解条件计算，可以跟着[四个token经过两个MoE专家](foundations/moe-and-large-model-architecture-basics.md)，看路由如何变成真实分组与合并。

正文里的手算和小程序只负责把机制讲明白。训练效果、GPU 速度和机器人闭环属于另外三类证据，需要各自的实验。能把程序跑通，是开始检查的条件，不是结论。

本站参考外部教程时遵循[来源、改编与证据边界](roadmap/source-and-adaptation-guidelines-1110-0106.md)，将教材、官方文档、论文、源码、图片和本地实验分开记录。

这里有一件事需要先说清楚。本站不是把某篇博客整篇搬过来的转载站。正文通常根据论文、官方文档、源码和少量教程重新组织，公式推导与小算例由本站重写，论文原图和第三方代码另记许可。历史记录显示，有 101 个页面曾吸收过 Datawhale 算法教程的基础或复杂度讲解块；目前的长句级重合检查没有发现已缓存来源中的精确复制，但它不能代替近似翻译、段落顺序和许可证的逐页复核。历史页面经历过批量辅助生成，不能因此把每一页都称作已经逐句人工审校的成稿。读到句式过整齐、段落像提纲的地方，应当把它看作待改稿，并沿页面来源回到原文核对。

## 知识之间怎样连接

数学描述对象及其变化，模型结构把这些运算组合成预测函数，训练用数据改变函数的参数，推理把学习到的函数用于新输入。生成模型和强化学习进一步规定了输出如何形成，以及模型怎样处理未来和动作。多模态与具身系统则把这些机制接到图像、语言、空间与物理执行。

读公式遇到困难时，可以从[数学工具](prerequisite-math/index.md)查当前需要的概念；需要区分相近名词时，查[术语表](glossary/index.md)和[知识 QA](knowledge-qa/foundations-and-transformer.md)。[工具与查阅](technical-reference/index.md)还收录可运行示例和探索文章。

## 把知识变成研究记录

读懂一个机制以后，把问题写成[实验合同](training/research-loop-from-hypothesis-to-evidence-1123-2351.md)。`server_monitor` 记录运行和观测，PaperPilot 保存论文证据与笔记版本，条件、反例和下一步实验再写回对应专题。

## 三种进入方式

如果只有半小时，挑一个[小算例](foundations/optimization-and-training-basics.md)或[可运行示例](assets/examples/reinforcement-learning/README.md)，先把输入、目标和更新顺序复算一遍。如果正在读论文，从[论文精读](paper-deep-dives/index.md)进入，回到正文中的公式、图表和代码提交。如果准备做实验，直接从[研究闭环](training/research-loop-from-hypothesis-to-evidence-1123-2351.md)写 `question_id`、对照、预算和停止条件，再把结果回写到对应专题。
