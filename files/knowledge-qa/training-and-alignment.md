# 训练与对齐 QA

预训练、微调、对齐、优化器、稳定性、分布式训练和实验治理的高频问题，需要说清训练目标、数据流、公式含义和工程取舍。

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

### Q：为什么数据集要定义成契约，而不只是文件集合？

**面试回答。** 可复现的数据集至少要一起定义输入、训练目标、metadata、来源与权限、processor、split、过滤规则和版本。模型真正消费的是处理后的样本分布，同一批原始文件经过不同 tokenizer、裁剪、抽帧、chunking 或同步策略，会变成不同训练数据。

**追问展开。** 文本要记录 tokenizer 和清洗规则，视觉数据要记录 resize/crop，RAG 要记录 chunk 与文档版本，机器人轨迹要记录坐标系、频率和传感器对齐。Metadata 还负责按来源、语言、设备和任务分桶定位退化。  
**易错点 / 边界。** 只保存训练代码和数据路径不能复现实验；raw snapshot、processor、split 和 sampling policy 都会改变结果。  
**继续读。** [数据与数据集基础](../foundations/data-and-dataset-basics.md) / [数据质量、去重与治理](../training/data-quality-dedup-and-governance.md)

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

### Q：重参数化、REINFORCE 和 STE 分别在解决什么？

**面试回答。** 三者都处理普通链式法则无法直接穿过的节点。重参数化把连续随机样本写成参数的可微变换与独立噪声，常用于 VAE；REINFORCE 用回报乘 log-probability 梯度处理离散采样；STE 在前向保留 round、argmax 或 one-hot 等硬操作，反向借用近似梯度。

**追问展开。** 重参数化通常方差较低，但要求可微采样路径；REINFORCE 较通用但方差大，常配 baseline；STE 易于工程实现，但通常有偏。选择时要同时看部署前向是否真实、梯度偏差、方差和目标是否可验证。  
**易错点 / 边界。** 能传回梯度只说明训练链路接通，不证明梯度估计准确，也不证明代理目标和部署目标一致。  
**继续读。** [梯度估计、离散变量与训练信号](../foundations/gradient-estimators-and-training-signals.md) / [采样、估计与校准](../prerequisite-math/sampling-estimation-and-calibration.md)

### Q：Stop-gradient 和 EMA teacher 为什么常一起出现？

**面试回答。** Stop-gradient 阻止 target 分支被当前 matching loss 直接更新，EMA teacher 则用 student 参数的指数滑动平均缓慢更新 target 网络。这样可以给 student 提供变化较慢的学习目标，降低两条分支同时追逐造成的不稳定和表示坍缩风险。

**追问展开。** JEPA、BYOL 和部分蒸馏方法都会使用这种结构。防坍缩通常还需要 predictor asymmetry、masking、负样本或 variance/covariance 约束，EMA teacher 只负责目标稳定性。  
**易错点 / 边界。** Teacher 稳定不代表 teacher 正确。它也可能稳定地保留偏差，仍要通过独立任务评测和表示诊断验证。  
**继续读。** [梯度估计、离散变量与训练信号](../foundations/gradient-estimators-and-training-signals.md) / [预训练目标与表示学习](../foundations/pretraining-objectives-and-representation-learning.md)

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

### Q：为什么评测平均分还要配置信区间和分桶？

**面试回答。** 平均分把不同任务、长度、语言和风险样本压成一个数，可能掩盖关键桶退化；有限测试样本还会带来统计波动。置信区间描述当前估计的不确定性，分桶指标则说明变化发生在哪里。

**追问展开。** 成功率可用 Wilson 区间，复杂指标可用 bootstrap。模型对比还要固定评测协议、随机性和样本版本，并优先报告成对差异，避免把采样噪声写成真实提升。  
**易错点 / 边界。** 置信区间窄只说明估计稳定，不说明测试集代表部署分布；数据污染和切分错误会让稳定结果同样失真。  
**继续读。** [数据划分与评测指标](../foundations/data-splits-metrics-and-evaluation-basics.md) / [采样、估计与校准](../prerequisite-math/sampling-estimation-and-calibration.md)

### Q：Packing 为什么能提高吞吐，又为什么容易引入数据泄漏？

**面试回答。** Packing 把多个短样本拼进一个定长序列，减少 padding，让更多 token 真正参与训练。风险在于 attention mask、position id 或 label mask 配错后，后一个样本可能看到前一个样本的答案，或者模型在不该计算 loss 的 prompt 和 padding 上学习。

**追问展开。** 验证 packing 不能只看吞吐和 loss。应构造 toy batch，逐项可视化样本边界、attention mask、position id 与监督位置，并比较 pack 前后的有效 token 数和单样本 loss。  
**易错点 / 边界。** Tokens/s 上升不等于有效训练吞吐上升；边界泄漏会让离线 loss 虚假变好。  
**继续读。** [输入管线、Packing 与吞吐](../training/input-pipelines-packing-and-throughput-governance.md) / [数据与数据集基础](../foundations/data-and-dataset-basics.md)

### Q：数据配比和课程学习为什么也是训练目标的一部分？

**面试回答。** 训练目标是不同样本梯度的加权平均，采样概率会直接改变哪些能力得到更多更新。课程学习进一步改变训练阶段顺序，让模型先学较稳定或较基础的模式，再进入更长上下文、更难推理或更稀有的数据。

**追问展开。** 设计 data mixture 时应记录每类数据的采样权重、有效 loss token、重复曝光和阶段切换，并用能力桶观察新增数据是否提升目标能力、是否冲刷旧能力。  
**易错点 / 边界。** 按文件数或样本数配比常会误导，因为不同样本长度、mask 和重复率对应的有效训练信号不同。  
**继续读。** [Scaling、课程学习与数据配比](../training/scaling-curriculum-and-data-mixture.md) / [数据质量、去重与治理](../training/data-quality-dedup-and-governance.md)

### Q：Global batch、micro-batch 和 gradient accumulation 怎么连接？

**面试回答。** Micro-batch 是单次前向和反向在每张设备上处理的样本数；gradient accumulation 把多个 micro-step 的梯度累积后再更新；global batch 是一次 optimizer step 覆盖的所有设备和累积步的数据总量。常见关系是 `global batch = micro-batch × data-parallel size × accumulation steps`。

**追问展开。** Global batch 改变梯度噪声和学习率配方，micro-batch 主要受显存约束，accumulation 用更多时间换更大有效 batch。比较实验时还要统一有效 token 数，因为变长样本下只看样本数不够。  
**易错点 / 边界。** 增大 accumulation 不会提高数据并行通信带来的扩展效率，也不能自动复现更大 micro-batch 的 kernel 利用率。  
**继续读。** [模型训练的完整链路](../training/beginner-model-training-primer.md) / [分布式训练与 Checkpoint](../training/distributed-training-and-checkpointing.md)

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

### Q：知识蒸馏可以蒸馏什么，应该怎样验收？

**面试回答。** Student 可以学习 teacher 的 soft logits、hidden states、attention relation、rationale、动作轨迹或生成分布。不同信号传递的知识不同，student 容量和层映射也会限制能保留多少。

**追问展开。** 验收要同时看 student 与 teacher 的一致率、独立 gold eval、失败分桶、校准以及真实延迟、吞吐和显存。扩散蒸馏还要看少步采样下的多样性、条件一致性和长尾质量。  
**易错点 / 边界。** Student 学得像 teacher 不等于 student 正确；teacher 的事实错误、偏见和脆弱推理也会被迁移。  
**继续读。** [模型蒸馏入门](../foundations/model-distillation-basics.md) / [扩散蒸馏](../diffusion/distillation.md)

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

### Q：Context parallel 和 expert parallel 分别解决什么？

**面试回答。** Context parallel 沿序列维切分长上下文的 activation 和 attention 计算，主要解决单卡放不下长序列的问题。Expert parallel 把 MoE 专家分散到不同设备，让 token 经过 router 后通过 all-to-all 发送到对应专家，主要解决专家参数与计算分布问题。

**追问展开。** 两者节省的对象不同，也引入不同通信。Context parallel 依赖序列块间的 attention 通信；expert parallel 对 token 路由均衡、网络拓扑和 all-to-all 更敏感。  
**易错点 / 边界。** 并行维度越多，配置空间、通信开销和 checkpoint 恢复复杂度越高；应从显存与 profiler 证据反推组合。  
**继续读。** [分布式训练与 Checkpoint](../training/distributed-training-and-checkpointing.md) / [MoE 与大模型架构](../foundations/moe-and-large-model-architecture-basics.md)

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
