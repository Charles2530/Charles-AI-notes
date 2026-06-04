# 模型训练从零入门：把数据、目标和更新压力对齐

> **这篇回答的问题。** 如何理解“模型训练从零入门”背后的核心机制、适用边界和下一步阅读路径。

训练不是“把数据喂给模型，然后等 loss 下降”。更准确地说，训练是在控制三件事：模型看见什么数据，用什么目标判断对错，用多大的更新压力修改参数。只要这三件事没有对齐，loss 曲线再顺，也可能只是模型学会了错误捷径。

可以先把训练看成一条生产线：

```text
data distribution
  -> sampling / packing
  -> batch
  -> forward
  -> loss
  -> backward
  -> optimizer step
  -> eval / checkpoint
```

这条线里每一环都会改变模型最终学到什么。数据决定“世界长什么样”，loss 决定“什么叫错”，batch 和学习率决定“每一步改多猛”，评测和 checkpoint 决定“这个实验能不能相信、能不能恢复、能不能比较”。

## 先把一次 Step 看清楚

最小监督学习目标可以写成：

\[
\min_\theta \; \mathbb{E}_{(x,y)\sim \mathcal D}[\ell(f_\theta(x),y)]
\]

这里 \(\theta\) 表示模型参数，\(\mathcal D\) 表示训练数据分布，\(x\) 表示输入，\(y\) 表示目标，\(f_\theta(x)\) 表示模型预测，\(\ell\) 表示 loss。训练就是不断从 \(\mathcal D\) 里抽样，让模型预测，再用 loss 把错误变成梯度，最后更新 \(\theta\)。

代码层面，一步训练像这样：

```text
batch = next(dataloader)
pred = model(batch.inputs)
loss = loss_fn(pred, batch.targets)
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

这几行不是口号。`next(dataloader)` 决定模型这一刻看见哪个数据桶；`model(...)` 决定当前参数如何解释输入；`loss_fn(...)` 决定哪些错误被放大；`backward()` 把错误沿计算图传回参数；`optimizer.step()` 才真正改变模型。很多训练故障并不在模型结构，而在这几行之间的语义没有对齐。

LLM 预训练只是这个框架的一个特例：输入是一段 token，目标通常是下一个 token。扩散模型可能预测噪声、velocity 或 score；VLA 行为克隆可能预测连续动作、离散动作 token 或动作 chunk；偏好训练让 chosen response 比 rejected response 更可能出现。形式不同，本质都是在定义“什么样的输出应该被奖励，什么样的输出应该被惩罚”。

## 数据不是素材，是梯度来源

模型不是在“所有知识”上训练，而是在实际采样出来的分布上训练。一个训练集里代码、数学、中文、网页噪声、长上下文样本、机器人失败轨迹各占多少，都会改变梯度的平均方向。

大模型训练通常不是一次性把数据洗好就结束，而是阶段化的数据工程：

| 阶段 | 数据在提供什么信号 | 最容易漏看的问题 |
| --- | --- | --- |
| Pretraining | 语言、知识、代码、模式和通用表示 | 去重、污染、混合比例、低质网页 |
| Mid-training | 长上下文、推理、多语言、领域或模态能力 | 新数据是否冲刷底座能力 |
| SFT | 指令格式、任务示范和交互入口 | 模板过拟合、回答风格变窄 |
| Preference / RL | 偏好、规则、可验证奖励和安全边界 | reward hacking、过度拒答、长尾退化 |
| Distillation | teacher 行为、分布或中间表示 | 复制 teacher 错误、丢失探索性 |

Hugging Face 的 Smol Training Playbook 和 SmolLM3 复盘值得参考，因为它们把训练写成数据、规模、评测和失败复盘的组合，而不是单个 loss 名字。

对训练来说，真正重要的不是“样本条数”，而是有效 token 曝光：

\[
N_{\text{eff}}=\sum_{\text{batch}}\text{valid tokens after packing and masking}
\]

这里 \(N_{\text{eff}}\) 表示真正参与 loss 的 token 数。padding、被 mask 的位置、坏样本、重复样本和没有参与 loss 的区域，都不应该被当成有效训练信号。长文本 packing 可以提高吞吐，但如果 attention mask 写错，会让样本边界泄漏；如果长度桶采样偏了，短样本可能被过度曝光，长样本反而学不到；如果只记录原始数据比例，不记录实际参与 loss 的 token 比例，实验结论会很虚。

一个现实例子：你给机器人策略加入 10,000 条“接触失败后恢复”的轨迹，但这些轨迹很长、被 packing 截断，又只占总 token 的 0.3%。训练 loss 可能几乎看不出变化，闭环恢复能力也不会明显提高。不是模型不懂恢复，而是恢复场景没有形成足够训练压力。

## Loss 定义“错在哪里”

以 next-token 语言建模为例，交叉熵目标是：

\[
\mathcal L_{\text{CE}}=-\sum_{t=1}^{L}\log p_\theta(x_t\mid x_{<t})
\]

这里 \(L\) 表示序列长度，\(x_t\) 表示第 \(t\) 个真实 token，\(p_\theta(x_t\mid x_{<t})\) 表示模型在历史 token 条件下给真实下一个 token 的概率。真实 token 概率越低，惩罚越大。

这个 loss 很适合训练语言模型的统计预测能力，但它不会自动保证模型会遵守指令、诚实拒答、稳定使用工具，或者在机器人闭环里安全执行动作。所以训练经常分阶段，不是因为后面的 loss 更“高级”，而是因为训练信号换了。

| 信号 | 它主要改变什么 | 典型副作用 |
| --- | --- | --- |
| Pretraining CE | 底座表示、知识、语言和代码能力 | 学到数据噪声和错误模式 |
| SFT | 任务格式、指令遵循和示范行为 | 模板化、风格变窄 |
| Preference / RL | 输出排序、规则偏好和可验证行为 | reward hacking、过拟合偏好模型 |
| Distillation | teacher 行为或分布 | 复制 teacher 错误，压窄分布 |
| Domain adaptation | 领域语言和任务模式 | 通用能力被冲刷 |

平均 loss 下降也不等于关键能力提高。假设世界模型训练中有三类数据桶：

| Bucket | Token Share | Old Loss | New Loss | Closed-loop |
| --- | ---: | ---: | ---: | ---: |
| 静态背景 | `72%` | `0.180` | `0.158` | `+0.3%` |
| 物体移动 | `18%` | `0.246` | `0.241` | `+0.1%` |
| 接触与遮挡 | `10%` | `0.310` | `0.337` | `-6.8%` |

平均 loss 会下降，但真正影响机器人成功率的接触与遮挡桶变差。优化器没有犯错，它只是忠实优化了你给它的加权目标。训练报告必须有分桶评测、失败样本回放和下游指标。

## Batch 和学习率决定更新压力

Batch 不是越大越好，它改变的是梯度估计。小 batch 噪声大，可能帮助探索，也可能让曲线抖动；大 batch 噪声小，吞吐更好，但学习率、warmup 和泛化行为都要重新匹配。

分布式训练里要区分三层 batch：

```text
micro-batch:      单卡一次 forward/backward 能放下多少
accumulation:     累积几个 micro-batch 再更新一次
global batch:     所有卡、所有累积合起来的有效更新规模
```

如果 8 张卡，每卡 micro-batch 是 2，gradient accumulation 是 16，那么一次 optimizer step 的样本数是：

\[
8\times2\times16=256
\]

这里的 256 只表示样本数。对 LLM 和多模态模型，最好继续换算成 tokens per update。两次实验如果 global batch 都是 256，但一个平均 512 token，另一个平均 4096 token，它们的训练压力和系统负载完全不同。

最小参数更新可以写成：

\[
\theta \leftarrow \theta-\eta\nabla_\theta\mathcal L
\]

这里 \(\nabla_\theta\mathcal L\) 表示 loss 对参数的梯度，\(\eta\) 表示学习率。真实训练里，AdamW、Momentum、Muon 等优化器会用历史梯度、自适应尺度或矩阵结构修正这一步。学习率决定这一步有多大，warmup 是为了让参数、激活、优化器状态和混合精度统计先进入稳定区，decay 则是在后期减少震荡。

比较两个 run 时，优先对齐 consumed tokens、有效 batch、学习率日程、数据版本、tokenizer、packing 策略和评测脚本。只对齐 `step` 数，常常是在比较两种不同训练。

## 系统先算显存账和时间账

训练态显存不只存参数，还要存梯度、优化器状态、激活、中间 buffer 和通信缓存。AdamW 通常要保存一阶动量、二阶动量和可能的 master weights；反向传播还需要保留或重算激活。一个模型推理能放下，不代表训练能放下。

![ZeRO memory optimization stages](../assets/images/paper-figures/training/zero-memory-stages.png){ width="860" }

<small>图源：[ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)，Figure 1。本站用这张图说明 optimizer states、gradients、parameters 在普通数据并行里为什么会重复保存，以及 ZeRO 三个阶段分别切走哪部分状态；没有用 image2 或其他生成式工具作图。</small>

读这张图时，先看普通 data parallel 为什么浪费：每张卡都有完整参数、梯度和 optimizer state。ZeRO Stage 1 先切 optimizer state，Stage 2 再切 gradient，Stage 3 连参数也切。显存省下来了，但训练中会多出 gather、scatter、通信 overlap 和 checkpoint 分片管理。

训练慢也不一定是 GPU 算不动。数据管线、通信、pipeline bubble、checkpoint I/O、kernel launch、CPU/GPU 同步都可能拖后腿。

![GPipe pipeline parallelism](../assets/images/paper-figures/training/gpipe-pipeline-parallelism.png){ width="860" }

<small>图源：[GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965)，Figure 2(c)。本站用这张图说明 micro-batch 如何填充 pipeline bubble，让不同 accelerator 同时处理不同模型分段；没有用 image2 或其他生成式工具作图。</small>

GPipe 图里的 micro-batch 很关键。没有 micro-batch，第一段算完才轮到第二段，很多设备会等待；micro-batch 把流水线填起来。但 micro-batch 太多又会增加激活保存、调度复杂度和恢复语义。系统优化不是把所有并行开关打开，而是让模型大小、显存、通信拓扑、batch 语义和 checkpoint 目标共同闭合。

## Scaling Law 是预算分配问题

训练预算不是只堆参数。Chinchilla 的核心提醒是：固定计算量下，参数量和训练 token 数要配平。模型太大但 token 不够，会欠训练；模型太小但 token 很多，容量会成为瓶颈。

![Chinchilla IsoFLOP curves](../assets/images/paper-figures/training/chinchilla-isoflop-curves.png){ width="920" }

<small>图源：[Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)，Figure 4。本站用这张图说明固定 FLOP 预算下，不同模型大小会得到不同最终 loss，曲线谷底对应更合适的参数/token 配比；没有用 image2 或其他生成式工具作图。</small>

这张图不要求读者记住某个固定比例，因为数据质量、架构、tokenizer、目标函数和后训练阶段都会改变实际选择。它真正有用的地方是改变问题问法：不要只问“模型要不要更大”，而要问“这笔算力应该分给参数、token、上下文长度、数据清洗、评测和失败复盘中的哪一块”。

## Checkpoint 是证据链，不只是备份

可信 checkpoint 不只是 `model.safetensors`。如果要恢复训练或解释结果，它至少要回答：来自哪份代码、哪份配置、哪份数据 manifest、哪个 tokenizer、哪个 sampler 状态、哪个 optimizer state、哪个 scheduler 位置、哪个 RNG 状态、已经消耗多少 token。

只保存权重，通常只能推理，不能证明训练可以无缝续跑。恢复后最常见的问题是：学习率跳到错误位置，数据重复或跳样，gradient accumulation 边界错位，loss scaler 重置，分布式分片映射变化，packing 的 token 边界恢复不准。

可靠做法是把 checkpoint 当作实验资产：保存模型、optimizer、scheduler、RNG、sampler、tokenizer、data manifest、配置和 eval artifacts。恢复后做短跑一致性检查，看若干 step 内的 loss、grad norm、吞吐、consumed tokens 和关键评测是否在可解释范围内。

## 训练故障先按症状分桶

训练事故通常不是“模型坏了”这么笼统。先按症状分桶，排查会快很多。

| 症状 | 先怀疑什么 | 需要看的证据 |
| --- | --- | --- |
| loss spike + grad norm 暴涨 | 学习率过大、异常 batch、低精溢出、梯度爆炸 | batch 样本、grad norm、overflow、LR |
| train loss 降，eval loss 不降 | 过拟合、数据泄漏、目标错位、评测分布不匹配 | train/eval bucket、contamination、eval script |
| 吞吐很高，质量不变 | 系统变快了，训练信号没变强 | tokens/s、有效 token、分桶指标 |
| 恢复训练后曲线偏移 | optimizer/scheduler/RNG/sampler 状态没恢复 | checkpoint manifest、consumed tokens、短跑对齐 |
| 某个能力突然退化 | 数据混合或 loss 权重改坏了长尾桶 | 数据桶曝光、失败样本、下游评测 |

Karpathy 的训练 recipe 仍然值得读，是因为它强调先让一个小模型、小数据、小 batch 的实验完全跑通，再逐步增加复杂度。成熟训练不是凭经验调几个超参，而是让每个异常都有可观察证据。

## 一次训练改动是否可信

一个训练改动声称“更快、更稳、更好”，至少要同时交代三类证据。

| 证据 | 看什么 | 缺失时的风险 |
| --- | --- | --- |
| 质量证据 | train/eval loss、任务 benchmark、分桶指标、失败样本、人工/闭环评估 | 只优化了代理目标 |
| 稳定性证据 | loss spike、NaN/Inf、grad norm、activation 分布、overflow、恢复后一致性 | 短跑正常，长跑崩 |
| 成本证据 | tokens/s、MFU、显存峰值、通信等待、checkpoint pause、data loader stall | 质量提高但不可负担 |

这三类证据缺一类，结论都会变轻。只报质量，可能成本不可接受；只报吞吐，可能模型没变好；只报短跑稳定，可能长跑数据阶段切换后才崩。

## 最后判断

训练的核心不是“让数字下降”，而是让数据分布、目标函数和更新压力共同服务真实能力。数据决定模型看见什么，loss 决定哪些错误被惩罚，batch 与学习率决定每一步压力，optimizer 和数值策略决定训练能不能稳定走下去，eval 与 checkpoint 决定实验是否可信。

读任何训练报告时，先沿着这条链问：有效 token 是什么，loss 在优化什么，更新压力是否稳定，系统账是否闭合，质量证据是否覆盖关键失败桶。如果这些问题都能回答，训练才从一条曲线变成一套可以复盘、可以恢复、可以继续改进的工程过程。

## 外部精读

- [PyTorch: Optimization Loop](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)：用官方最小训练循环理解 `loss.backward()`、`optimizer.step()` 和 `zero_grad()`。
- [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/)：训练排查经典博客，适合学习如何从小实验逐步走到可信训练。
- [Smol Training Playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook)：把训练目标、数据、评测和成本放到同一份工程手册里看。
- [SmolLM3: smol, multilingual, long-context reasoner](https://huggingface.co/blog/smollm3)：官方训练复盘，适合理解多阶段数据与后训练如何组成 recipe。
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)：理解参数、token 和计算预算如何配平。
- [ZeRO](https://arxiv.org/abs/1910.02054)：理解大模型训练显存账。
- [GPipe](https://arxiv.org/abs/1811.06965)：理解流水线并行为什么需要 micro-batch。
- [Datawhale: VRAM Calculation and ZeRO](https://datawhalechina.github.io/llm-algo-leetcode/01_Hardware_Math_and_Systems/06_VRAM_Calculation_and_ZeRO.html)：中文显存估算补充读物，适合把参数、梯度、优化器状态拆成手算账。

## 相关阅读与下一步

- 外部材料：[PyTorch Distributed 文档](https://pytorch.org/docs/stable/distributed.html)。
- 外部材料：[DeepSpeed ZeRO 教程](https://www.deepspeed.ai/tutorials/zero/)。
- 外部材料：[Weights & Biases 文档](https://docs.wandb.ai/)。
- 站内下一步：[训练系统专题](index.md)。
- 站内下一步：[分布式训练与 Checkpoint](distributed-training-and-checkpointing.md)。
- 站内下一步：[评测与 Ablation 方法论](evaluation-and-ablation-methodology.md)。
