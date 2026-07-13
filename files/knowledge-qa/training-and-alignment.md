# 训练与对齐 QA

这一页回答预训练、微调、对齐、优化器、稳定性、分布式训练和实验治理中的高频问题。面试回答重点说清训练目标、数据流、公式含义和工程取舍。

## 预训练与训练基础

### Q：大语言模型预训练到底在学什么？

**面试回答。** 主流 decoder-only LLM 的预训练目标是 next-token prediction：给定前面的 token，预测下一个 token。训练时模型不是显式学习“知识库”，而是在最大化训练语料的似然，压缩语言模式、事实片段、代码结构、推理模板和世界统计规律。

\[
\mathcal{L}_{\text{LM}}=-\sum_{t=1}^{T}\log p_\theta(x_t\mid x_{<t})
\]

**追问展开。** 这也是为什么预训练可以用大规模无标注文本：标签就是序列里的下一个 token。模型能力来自目标函数、数据规模、模型容量和优化过程共同作用，而不是某一个神秘模块。  
**易错点 / 边界。** 预训练学到的是训练分布中的统计结构，不保证事实最新、价值对齐、工具可靠或推理必然正确。  
**继续读。** [预训练目标与表示学习](../foundations/pretraining-objectives-and-representation-learning.md) / [预训练、微调与对齐](../training/pretraining-finetuning-alignment.md)

### Q：为什么数据质量比单纯堆 token 更重要？

**面试回答。** 数据决定模型看到什么、重复什么、忽略什么。低质量重复数据会浪费计算并放大噪声；污染数据会让评测失真；覆盖不足会导致长尾任务和领域任务失败。训练数据治理通常包括去重、过滤、质量打分、来源配比、敏感内容处理和 benchmark contamination 检查。

**追问展开。** 面试时可以把数据质量拆成四层：语料是否干净，分布是否覆盖目标能力，比例是否匹配训练阶段，评测是否没有泄漏。大模型训练不是“数据越多越好”，而是“有效 token 越多越好”。  
**易错点 / 边界。** 高质量不等于只保留教科书文本；代码、数学、多语言、长上下文、工具轨迹和错误修复样本都可能是关键数据。  
**继续读。** [数据质量、去重与治理](../training/data-quality-dedup-and-governance.md) / [数据与数据集基础](../foundations/data-and-dataset-basics.md)

### Q：Scaling law 在训练决策里怎么用？

**面试回答。** Scaling law 用经验公式描述 loss 随模型参数量 \(N\)、数据量 \(D\)、计算量 \(C\) 的变化趋势。它的价值是帮助估算预算：给定计算资源，应该扩大模型、增加数据，还是延长训练。一个常见直觉是 loss 会按幂律下降，但边际收益递减。

\[
L(N,D) \approx L_\infty + aN^{-\alpha}+bD^{-\beta}
\]

**追问展开。** 面试时不要把 scaling law 说成精确定律。它更像实验经济学工具：指导小规模 sweep、预算规划和是否值得放大。  
**易错点 / 边界。** 新数据配方、新 tokenizer、新架构、后训练和推理时计算都会改变外推；最终能力还要靠真实评测。  
**继续读。** [Scaling Laws 与实验经济学](../training/scaling-laws-and-experiment-economics.md) / [评测与消融方法](../training/evaluation-and-ablation-methodology.md)

### Q：学习率 warmup 为什么重要？

**面试回答。** 训练初期参数、激活和优化器动量还不稳定，如果一开始就用大学习率，容易出现 loss spike、梯度爆炸或低精度溢出。Warmup 让学习率从小逐步升到目标值，给优化器状态和网络激活一个稳定过渡。

**追问展开。** 常见 schedule 是 linear warmup + cosine decay。回答时可以说：warmup 解决“起步稳定性”，decay 解决“后期收敛精细化”。  
**易错点 / 边界。** warmup 不能修复坏数据、错误 loss、错误初始化或 kernel bug；它只是优化稳定性的一个部件。  
**继续读。** [目标函数、优化器与 LR 日程](../training/objectives-optimizers-and-schedules.md) / [训练稳定性与失败排查](../training/stability-numerics-and-failure-triage.md)

### Q：AdamW 和 SGD 的关键差异是什么？

**面试回答。** SGD 主要沿当前梯度方向更新，Adam 维护一阶动量和二阶方差估计，对每个参数自适应调整步长。AdamW 把 weight decay 从梯度更新中解耦，避免 L2 正则和自适应学习率耦合导致的行为偏差。

\[
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\quad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
\]

**追问展开。** LLM 训练常用 AdamW 或变体，是因为大规模非凸优化里自适应步长更稳。Muon 等新优化器则试图改善矩阵参数更新方向和 scaling 表现。  
**易错点 / 边界。** 优化器不能脱离 batch size、学习率、数据、精度和模型结构单独比较。  
**继续读。** [目标函数、优化器与 LR 日程](../training/objectives-optimizers-and-schedules.md) / [Muon：LLM 预训练优化器](../paper-deep-dives/foundations/muon-is-scalable.md)

### Q：混合精度训练为什么能省显存和提速？

**面试回答。** 混合精度用 FP16/BF16/FP8 存储或计算部分张量，减少显存占用和内存带宽，并利用 Tensor Core 加速矩阵乘法。为了稳定训练，通常保留 FP32 master weights、FP32 optimizer states 或更高精度累加。

**追问展开。** FP16 动态范围小，常需要 loss scaling；BF16 指数位更多，更不容易溢出；FP8 依赖更严格的 scale、校准和硬件支持。面试时要强调“低精度省的是存储和带宽，但稳定性靠混合策略”。  
**易错点 / 边界。** 低精度可能带来 overflow、underflow、梯度噪声和层归一化误差；上线前必须看 loss、梯度和质量回归。  
**继续读。** [低比特训练与数值格式](../training/low-bit-training-and-numerics.md) / [FP8 与混合精度推理](../quantization/fp8-and-mixed-precision-serving.md)

### Q：训练 loss 突然 NaN 应该怎么排查？

**面试回答。** 我会先按“数据、优化、数值、分布式、算子”五类排查。数据上查异常样本和标签；优化上查学习率、warmup、梯度范数；数值上查混合精度溢出、loss scale、除零；分布式上查是否只在某个 rank 出现；算子上查最近改过的 kernel 或 fused op。

**追问展开。** 面试里可以补一个工程流程：复现最小 batch，固定 seed，记录 first bad step，dump input 和梯度范数，逐步关闭新特性。不要直接跳到“调小学习率”这种单点操作。  
**易错点 / 边界。** NaN 不是一个原因，而是一类症状；不定位根因就继续训练，后续 checkpoint 可能不可用。  
**继续读。** [训练稳定性与失败排查](../training/stability-numerics-and-failure-triage.md) / [低比特训练与数值格式](../training/low-bit-training-and-numerics.md)

### Q：Ablation 为什么是训练论文和项目里的关键证据？

**面试回答。** Ablation 的作用是回答“到底是哪一部分带来了收益”。主实验只能说明完整系统有效，不能说明数据、架构、loss、优化器或系统组件分别贡献多少。好的 ablation 会移除或替换一个因素，并保持其他条件尽量不变。

**追问展开。** 常见 ablation 包括去掉模块、换数据配比、改 loss 权重、换 scale、换推理预算、去掉系统优化。面试时可以强调：ablation 不是为了凑表，而是为了验证 claim。  
**易错点 / 边界。** Ablation 结论只在当前 setting 下成立；小模型上的贡献不一定能外推到大模型或生产流量。  
**继续读。** [评测与消融方法](../training/evaluation-and-ablation-methodology.md)

## 微调与参数高效训练

### Q：全量微调和 LoRA 的取舍是什么？

**面试回答。** 全量微调更新所有参数，容量和自由度最大，但显存、计算和灾难性遗忘风险更高。LoRA 冻结原权重，只学习低秩增量，把权重更新写成：

\[
W' = W + \Delta W,\quad \Delta W = BA,\quad \operatorname{rank}(\Delta W)\le r
\]

其中 \(A\) 和 \(B\) 是可训练低秩矩阵。  
**追问展开。** LoRA 的优势是成本低、易合并、易切换，适合领域适配和多任务 adapter。选择 rank、target modules、学习率和数据质量非常关键。  
**易错点 / 边界。** LoRA 不一定能学会大幅偏离基座的新能力；如果任务需要改动底层知识或推理方式，全量微调或继续预训练可能更合适。  
**继续读。** [预训练、微调与对齐](../training/pretraining-finetuning-alignment.md) / [低比特训练与数值格式](../training/low-bit-training-and-numerics.md)

### Q：QLoRA 为什么能把微调显存降下来？

**面试回答。** QLoRA 把基座模型权重量化成 4-bit 存储，训练时冻结基座，只训练 LoRA 增量；前向计算时再反量化参与计算。这样主要显存不再被全精度基座权重和优化器状态占用，而是集中在 LoRA 参数、激活和少量状态上。

**追问展开。** 面试时可以补 NF4、double quantization 和 paged optimizer 的直觉：前者适合近似正态分布权重，后者进一步压缩 scale 和优化器内存峰值。  
**易错点 / 边界。** QLoRA 省显存不代表无损；量化误差、rank、数据质量和 target modules 都会影响最终质量。  
**继续读。** [低比特训练与数值格式](../training/low-bit-training-and-numerics.md) / [QLoRA 与量化训练](../quantization/qlora-and-quantized-training.md)

### Q：继续预训练、SFT 和偏好优化分别解决什么？

**面试回答。** 继续预训练主要让模型吸收新领域分布和知识风格；SFT 让模型学会指令格式和期望回答方式；偏好优化让模型在多个可行回答中更偏向人类或规则更喜欢的回答。三者顺序通常是“能力和领域适配 -> 指令跟随 -> 偏好和安全行为”。

**追问展开。** 如果企业有大量领域语料但缺指令数据，先继续预训练可能更合适；如果模型不会按格式答题，SFT 更直接；如果模型能回答但风格、安全或帮助性不稳，偏好优化更重要。  
**易错点 / 边界。** 这些阶段可能互相干扰；过度 SFT 或偏好优化会导致遗忘、模板化和 reward hacking。  
**继续读。** [预训练、微调与对齐](../training/pretraining-finetuning-alignment.md) / [偏好数据与对齐陷阱](../training/preference-data-and-alignment-pitfalls.md)

### Q：SFT 数据应该怎么构造才可靠？

**面试回答。** SFT 数据要覆盖目标任务、格式、难度、边界条件和拒答场景。每条样本最好有清晰 instruction、必要上下文和高质量 response；如果是多轮对话，还要保持角色、状态和历史一致。高质量 SFT 数据比数量更重要。

**追问展开。** 面试中可以补：SFT 数据要避免答案幻觉、过长模板、格式污染、重复样本和 benchmark 泄漏。对于代码、数学和工具任务，最好加入可验证结果或执行轨迹。  
**易错点 / 边界。** SFT 只模仿数据分布，不会自动判断数据是否正确；坏答案会被模型学进去。  
**继续读。** [预训练、微调与对齐](../training/pretraining-finetuning-alignment.md) / [偏好数据与对齐陷阱](../training/preference-data-and-alignment-pitfalls.md)

## RLHF、DPO 与 GRPO

### Q：RLHF 的三个核心阶段是什么？

**面试回答。** 经典 RLHF 包括三步：先用 SFT 得到能跟随指令的初始策略；再用人类偏好对训练 reward model；最后用 PPO 等 RL 方法优化策略，让模型生成更高 reward 的回答。可以把 reward model 看成把人类偏好压成一个可优化标量。

**追问展开。** PPO 阶段通常还会加 KL penalty，防止策略偏离 SFT 模型太远：

\[
\max_\theta \mathbb{E}[r_\phi(y)-\beta \operatorname{KL}(\pi_\theta(\cdot|x)\|\pi_{\text{ref}}(\cdot|x))]
\]

**易错点 / 边界。** Reward model 会错，策略会钻空子；RLHF 提升偏好不等于事实性、数学能力或安全性全面提升。  
**继续读。** [Policy Gradient、Actor-Critic、PPO 与 GRPO](../reinforcement-learning/policy-gradient-actor-critic-ppo-grpo.md) / [偏好数据与对齐陷阱](../training/preference-data-and-alignment-pitfalls.md)

### Q：DPO 和 PPO 式 RLHF 有什么区别？

**面试回答。** PPO 式 RLHF 显式训练 reward model，再用强化学习优化策略；DPO 直接用偏好对优化模型，让 chosen answer 的概率相对 rejected answer 更高。DPO 的目标来自把 reward 和 policy 的关系解析化，因此不需要单独训练 reward model 和在线 RL rollout。

\[
\mathcal{L}_{\text{DPO}}=-\log\sigma\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)}
-\beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)
\]

**追问展开。** 面试回答可以说 DPO 更简单、更稳定、工程成本更低；PPO 更灵活，适合复杂 reward 和在线探索。  
**易错点 / 边界。** DPO 强依赖偏好数据质量；对工具调用、长程任务和过程奖励，它不一定足够。  
**继续读。** [预训练、微调与对齐](../training/pretraining-finetuning-alignment.md) / [偏好数据与对齐陷阱](../training/preference-data-and-alignment-pitfalls.md)

### Q：GRPO 为什么常和推理模型训练一起讨论？

**面试回答。** GRPO 用同一问题的多条采样回答构成组，按组内相对 reward 估计 advantage，减少对 value model 的依赖。对于数学、代码、逻辑这类可规则验证任务，可以用答案正确性、格式和过程信号作为 reward，强化更好的推理轨迹。

**追问展开。** 直觉上，GRPO 不问“这个回答绝对值是多少”，而问“同一组回答里哪个更好”。这适合规则奖励明确、可以批量采样比较的场景。  
**易错点 / 边界。** 如果 reward 只看最终答案或格式，模型可能学会投机；开放式任务和不可验证任务需要更谨慎。  
**继续读。** [Policy Gradient、Actor-Critic、PPO 与 GRPO](../reinforcement-learning/policy-gradient-actor-critic-ppo-grpo.md) / [DeepSeek-R1：RL 激发推理能力](../paper-deep-dives/technical-reports/deepseek-r1.md)

### Q：Reward hacking 是什么？

**面试回答。** Reward hacking 指模型找到提高 reward 的捷径，但这个捷径并不符合真实目标。比如 reward 偏好长答案，模型就写更长但不更正确的回答；reward 偏好某种格式，模型就迎合格式而不提升推理。

**追问展开。** 解决思路包括更好的 reward 数据、对抗样本、规则校验、多维评估、KL 约束和人工审查。面试时要强调 reward 是 proxy，不是目标本身。  
**易错点 / 边界。** 只看 reward 曲线上升非常危险；必须同时看真实任务指标、失败样本和安全行为。  
**继续读。** [偏好数据与对齐陷阱](../training/preference-data-and-alignment-pitfalls.md) / [观测与在线评估](../inference/observability-and-online-evaluation.md)

### Q：后训练数据飞轮为什么需要 judge model？它有什么风险？

**面试回答。** 后训练数据飞轮会持续收集线上失败、生成候选回答、筛选高价值样本，再导出到 SFT、DPO、reward model 或回归集。Judge model 的作用是用较低成本给候选打分、分类失败、发现难例和辅助人审，把数据筛选从纯人工扩展成半自动流程。

**追问展开。** 一个稳妥说法是：judge 不是最终真理，而是数据路由器。它可以先粗筛、聚类、排序，再让规则、程序验证或人工复核关键样本。好的数据飞轮要记录 judge 版本、评分维度、误判样本和线上回归。  
**易错点 / 边界。** judge 一旦偏，飞轮会放大偏差：错误样本被当成高质量，DPO/RM 学到错误偏好，离线分数变好但线上体验变差。高风险任务不能让 judge 完全替代人工和真实指标。  
**继续读。** [后训练数据引擎与 Judge 模型](../training/post-training-data-engines-and-judge-models.md) / [RAG、Agent 与评估 QA](rag-agent-and-evaluation.md)

## 分布式训练与实验治理

### Q：数据并行、张量并行和流水线并行分别切什么？

**面试回答。** 数据并行复制模型、切 batch，不同 GPU 处理不同样本后同步梯度；张量并行切单层矩阵或 attention/MLP 内部计算；流水线并行按层切模型，让不同 stage 处理不同 micro-batch。大模型训练通常把三者组合起来。

**追问展开。** 取舍可以按瓶颈讲：模型放不下时需要张量/流水线并行，batch 太大或吞吐要求高时用数据并行，显存状态太大时用 ZeRO/FSDP。  
**易错点 / 边界。** 并行会引入通信、气泡、负载不均和 checkpoint 复杂度；不是 GPU 越多越线性加速。  
**继续读。** [分布式训练与 Checkpoint](../training/distributed-training-and-checkpointing.md) / [Megatron-LM：Tensor Model Parallel](../paper-deep-dives/foundations/megatron-lm.md)

### Q：ZeRO 和 FSDP 主要省哪里？

**面试回答。** 训练显存主要包括参数、梯度、优化器状态和激活。ZeRO/FSDP 的核心是把参数、梯度和优化器状态分片到多个 GPU，而不是每张卡都保存完整副本。ZeRO stage 越高，分片对象越多，显存越省，但通信更复杂。

**追问展开。** 可以粗略说：AdamW 优化器状态通常比参数本身还大，因为要存 \(m\)、\(v\) 等状态。ZeRO-1 分 optimizer states，ZeRO-2 再分 gradients，ZeRO-3 连 parameters 也分。  
**易错点 / 边界。** 省显存不等于更快；all-gather、reshard、offload 和 checkpoint 可能增加 step time。  
**继续读。** [分布式训练与 Checkpoint](../training/distributed-training-and-checkpointing.md) / [ZeRO：为什么数据并行会浪费这么多显存](../paper-deep-dives/foundations/zero.md)

### Q：Activation checkpointing 为什么能省显存？

**面试回答。** 反向传播需要前向激活来计算梯度，如果全部保存会占大量显存。Activation checkpointing 只保存部分边界激活，反向时重新计算中间激活，用额外计算换显存。

**追问展开。** 这适合深层 Transformer 和长序列训练，因为激活显存随 batch、seq length、hidden size 和层数增长很快。面试时可以说它是 compute-memory tradeoff。  
**易错点 / 边界。** checkpoint 粒度太细会明显变慢；和分布式通信、自定义算子、随机层一起用时要保证可复现和正确性。  
**继续读。** [自动微分、激活检查点与显存](../foundations/autograd-activation-checkpointing-and-memory.md) / [分布式训练与 Checkpoint](../training/distributed-training-and-checkpointing.md)

### Q：为什么 checkpoint 不只是保存权重？

**面试回答。** 真正可恢复训练需要保存模型参数、优化器状态、scheduler、随机数状态、数据加载进度、并行切分信息、混合精度状态和 tokenizer/config。只保存权重只能用于推理或重新微调，不能保证训练从同一步无缝继续。

**追问展开。** 分布式训练里 checkpoint 还涉及 shard 格式、rank 映射、对象存储吞吐和容灾恢复时间。大模型训练 checkpoint 是系统工程问题，不只是 `torch.save(model)`。  
**易错点 / 边界。** checkpoint 能恢复不代表结果 bitwise 一致；异步数据加载、非确定性 kernel 和集群变化都可能导致曲线轻微漂移。  
**继续读。** [分布式训练与 Checkpoint](../training/distributed-training-and-checkpointing.md) / [集群运维与实验管理](../training/cluster-operations-and-experiment-management.md)

### Q：如何判断一次训练是否值得继续放大？

**面试回答。** 我会先看小规模实验是否满足四个条件：loss 曲线稳定下降，验证集和关键能力指标有趋势，吞吐和显存符合预算，失败样本没有暴露系统性数据或实现问题。再用 scaling 估算放大后收益是否值得成本。

**追问展开。** 放大前最好做 ablation：数据配比、学习率、模型宽深、batch、精度和评测集。如果小规模已经 NaN、数据污染或评测不稳定，放大只会放大问题。  
**易错点 / 边界。** 小模型趋势不能保证大模型结果，但小模型失败通常是大模型失败的预警。  
**继续读。** [模型训练从零入门](../training/beginner-model-training-primer.md) / [Scaling Laws 与实验经济学](../training/scaling-laws-and-experiment-economics.md)

### Q：训练实验为什么需要 W&B 或类似追踪系统？

**面试回答。** 大模型实验成本高、变量多，实验追踪系统用来绑定配置、代码版本、数据版本、指标、日志、artifact 和样本输出。没有这些记录，曲线好坏都很难复盘，也很难知道哪个改动导致了效果变化。

**追问展开。** 好的实验记录至少包含 git commit、训练参数、seed、数据快照、硬件、吞吐、loss、评测结果和失败样本。它服务的是可复现和可比较，而不只是画图。  
**易错点 / 边界。** Dashboard 不能替代实验设计；指标如果定义错，记录得再完整也会误导。  
**继续读。** [W&B 实验追踪与治理](../training/wandb-experiment-tracking-and-governance.md) / [评测与消融方法](../training/evaluation-and-ablation-methodology.md)
