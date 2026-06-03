# 模型训练从零入门

训练不是“把数据喂给模型然后等 loss 下降”。更准确地说，训练是在控制三件事：**模型看见什么数据，用什么目标判断对错，用多大的更新压力改参数**。如果这三件事不清楚，loss 曲线再漂亮，也可能只是模型学会了错误捷径。

一条训练生产线可以写成：

```text
数据分布 -> 采样与 packing -> batch -> forward -> loss -> backward -> optimizer step -> eval/checkpoint
```

这条线里每一环都会改变模型最终学到什么。数据决定“世界长什么样”，loss 决定“什么叫错”，batch 和学习率决定“每一步有多大压力”，评测和 checkpoint 决定“这个实验能不能相信、能不能恢复、能不能比较”。

## 先看一次训练 step

最小监督学习目标可以写成：

\[
\min_\theta \; \mathbb{E}_{(x,y)\sim \mathcal{D}}[\ell(f_\theta(x), y)]
\]

这行公式很短，但含义很完整。 \(\theta\) 是模型参数，\(\mathcal{D}\) 是训练数据分布，\(x\) 是输入，\(y\) 是目标，\(f_\theta(x)\) 是模型预测，\(\ell\) 是 loss。训练就是不断从 \(\mathcal{D}\) 里抽样，让模型预测，再用 loss 把错误变成梯度，最后更新 \(\theta\)。

代码层面，一步训练像这样：

```text
batch = dataloader.next()
pred = model(batch.inputs)
loss = loss_fn(pred, batch.targets)
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

这不是伪装成代码的口号。`dataloader.next()` 决定这一刻模型看见哪个数据桶；`model(...)` 决定当前参数如何解释输入；`loss_fn(...)` 决定哪些错误被放大；`backward()` 把错误沿计算图传回每个参数；`optimizer.step()` 才真正改变模型。很多训练故障并不在模型结构，而在这几行之间的语义没有对齐。

LLM 预训练只是这个框架的一个特例：输入是一段 token，目标通常是下一个 token。扩散模型可能预测噪声、velocity 或 score；VLA 行为克隆可能预测连续动作、离散动作 token 或一个动作 chunk；偏好训练则让 chosen response 比 rejected response 更可能出现。形式不同，但本质都是在定义“什么样的输出应该被奖励，什么样的输出应该被惩罚”。

## 数据不是素材，而是训练分布

新手最容易低估数据。模型不是在“所有知识”上训练，而是在你实际采样出来的分布上训练。一个训练集里有多少代码、数学、中文、网页噪声、机器人失败轨迹、长上下文样本，都会改变梯度的平均方向。

大模型训练通常不是一次性把数据洗好就结束，而是阶段化的数据工程。预训练阶段会关注覆盖、去重、质量过滤和 contamination；mid-training 可能加入长上下文、推理、代码或多语言数据；SFT 会加入指令格式和任务示范；偏好/RL 阶段会加入人类或模型反馈。Hugging Face 的 SmolLM3 训练复盘和 Smol Training Playbook 值得参考，因为它们把训练写成数据、规模、评测和失败复盘的组合，而不是单个 loss 名字。

对训练来说，真正重要的不是“样本条数”，而是有效 token 曝光：

\[
N_{\text{eff}} = \sum_{\text{batch}} \text{valid tokens after packing and masking}
\]

这里的有效 token 排除了 padding、被 mask 掉的位置、坏样本、重复样本和没有参与 loss 的区域。长文本 packing 可以提高吞吐，但如果 attention mask 写错，就会让样本边界泄漏；如果长度桶采样偏了，短样本可能被过度曝光，长样本反而学不到；如果数据 mixture 只记录原始比例，不记录实际参与 loss 的 token 比例，实验结论会很虚。

一个很现实的例子：你给机器人策略加入了 10,000 条“接触失败后恢复”的轨迹，但这些轨迹很长、被 packing 截断、又只占总 token 的 0.3%。训练 loss 可能几乎看不出变化，闭环恢复能力也不会明显提高。不是模型“不懂恢复”，而是恢复场景没有形成足够的训练压力。

## Batch 决定每一步看见多少证据

Batch 不是越大越好，它改变的是梯度估计。小 batch 噪声大，可能帮助探索，也可能让曲线抖动；大 batch 噪声小，吞吐更好，但学习率、warmup 和泛化行为都要重新匹配。

分布式训练里要区分三层 batch：

```text
micro-batch:      单卡一次 forward/backward 能放下多少
accumulation:     累积几个 micro-batch 再更新一次
global batch:     所有卡、所有累积合起来的有效更新规模
```

如果 8 张卡，每卡 micro-batch 是 2，gradient accumulation 是 16，那么一次 optimizer step 的样本数是 \(8 \times 2 \times 16 = 256\)。但对 LLM 和多模态模型，样本数还不够，最好继续换算成 tokens per update。两次实验如果 global batch 都是 256，但一个平均 512 token，另一个平均 4096 token，它们的训练压力和系统负载完全不同。

比较两个 run 时，优先对齐 consumed tokens、有效 batch、学习率日程、数据版本、tokenizer、packing 策略和评测脚本。只对齐 `step` 数，常常是在比较两种不同训练。

## Loss 定义“错在哪里”

以 next-token 语言建模为例，交叉熵目标是：

\[
\mathcal{L}_{\text{CE}} = -\sum_{t=1}^{L}\log p_\theta(x_t \mid x_{<t})
\]

它的意思是：每个位置都问模型“你给真实下一个 token 分配了多少概率”。真实 token 概率越低，惩罚越大。这个 loss 很适合训练语言模型的统计预测能力，但它不会自动保证模型会遵守指令、诚实拒答、稳定使用工具，或者在机器人闭环里安全执行动作。

所以训练经常分阶段：

```text
pretraining:  学语言、知识、代码、模式和通用表示
mid-training: 强化长上下文、推理、多语言、领域或模态能力
SFT:          学任务格式、指令遵循和示范行为
preference/RL:把输出推向更符合偏好、规则或可验证奖励的区域
distillation:把 teacher 的行为或分布迁移给 student
```

每个阶段不是“更高级的 loss”，而是换了训练信号。预训练的文本似然能给底座能力；SFT 能给格式和任务入口；偏好/RL 能改变回答排序；蒸馏能迁移强模型行为。它们也各有副作用：SFT 可能过拟合模板，偏好训练可能 reward hacking，蒸馏可能复制 teacher 错误，领域数据可能冲刷通用能力。

判断 loss 时要回到任务。平均 loss 下降可能来自大桶样本变好，但长尾风险变差。假设世界模型训练中：

| Bucket | Token Share | Old Loss | New Loss | Closed-loop |
| --- | ---: | ---: | ---: | ---: |
| 静态背景 | `72%` | `0.180` | `0.158` | `+0.3%` |
| 物体移动 | `18%` | `0.246` | `0.241` | `+0.1%` |
| 接触与遮挡 | `10%` | `0.310` | `0.337` | `-6.8%` |

平均 loss 会下降，但真正影响机器人成功率的接触与遮挡桶变差。优化器没有犯错，它只是忠实优化了你给它的加权目标。这就是为什么训练报告必须有分桶评测、失败样本回放和下游指标。

## Backward 与 optimizer 把错误变成更新

反向传播用链式法则计算每个参数对 loss 的影响。最小更新公式是：

\[
\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}
\]

\(\nabla_\theta \mathcal{L}\) 是 loss 对参数的梯度，\(\eta\) 是学习率。这行公式说的是：沿着让 loss 下降的方向迈一步。真实训练里，AdamW、Momentum、Muon 等优化器会用历史梯度、自适应尺度或矩阵结构修正这一步。

学习率决定这一步有多大。训练早期常用 warmup，是因为参数、激活、优化器状态和混合精度统计还不稳定，直接用峰值学习率容易 loss spike 或 overflow。后期 decay 是为了在较好区域减少震荡。学习率日程必须和 effective batch、数据阶段、loss 权重、gradient clipping、混合精度一起看。

很多训练事故的症状并不是“loss 不下降”，而是：

```text
loss spike + grad norm 暴涨        -> 学习率过大、异常 batch、低精溢出或梯度爆炸
train loss 降，eval loss 不降      -> 过拟合、数据泄漏、目标错位或评测分布不匹配
吞吐很高，但质量不变              -> 系统变快了，训练信号没有变强
恢复训练后曲线偏移                -> optimizer/scheduler/RNG/sampler 状态没有恢复一致
```

这也是 Karpathy 的训练 recipe 仍然值得读的原因：它强调先让一个小模型、小数据、小 batch 的实验完全跑通，再逐步增加复杂度。成熟训练不是凭经验调几个超参，而是让每个异常都有可观察的证据。

## 训练系统先算三张账

第一张是显存账。训练态显存不只存参数，还要存梯度、优化器状态、激活、中间 buffer 和通信缓存。AdamW 通常要保存一阶动量、二阶动量和可能的 master weights；反向传播还需要保留或重算激活。一个模型推理能放下，不代表训练能放下。

![ZeRO memory optimization stages 原论文图](../assets/images/paper-figures/training/zero-memory-stages.png){ width="860" }

<small>图源：[ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)，Figure 1。原图表达 ZeRO 把普通数据并行中重复保存的 optimizer states、gradients、parameters 分阶段切到不同 data-parallel ranks。本站使用它说明：省显存不是一个开关，而是决定哪些训练状态常驻、哪些分片、哪些按需 gather。</small>

读这张图时，先看“普通 data parallel 为什么浪费”：每张卡都有完整参数、梯度和 optimizer state。ZeRO Stage 1 先切 optimizer state，Stage 2 再切 gradient，Stage 3 连参数也切。显存省下来了，但训练中会多出 gather、scatter、通信 overlap 和 checkpoint 分片管理。

第二张是时间账。训练慢可能不是 GPU 算不动，而是数据管线、通信、pipeline bubble、checkpoint I/O、kernel launch、CPU/GPU 同步在拖后腿。

![GPipe pipeline parallelism 原论文图](../assets/images/paper-figures/training/gpipe-pipeline-parallelism.png){ width="860" }

<small>图源：[GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965)，Figure 2(c)。原图表达把 mini-batch 切成 micro-batch 后，不同 accelerator 可以同时处理不同模型分段。本站使用它说明：pipeline parallelism 主要处理设备空转和时间排布，而不是改变训练目标。</small>

这张图里的 micro-batch 很关键。没有 micro-batch，第一段算完才轮到第二段，很多设备会等待；micro-batch 把流水线填起来。但 micro-batch 太多又会增加激活保存、调度复杂度和恢复语义。系统优化不是把所有并行开关打开，而是让模型大小、显存、通信拓扑、batch 语义和 checkpoint 目标共同闭合。

第三张是质量账。吞吐提升、显存降低、MFU 变高，都不能自动证明模型变好。系统改动必须回到 validation、benchmark、分桶评测、人工评估、闭环任务或线上 shadow 指标。训练工程的目标不是“跑得很满”，而是“在可信评测上，以可复现方式，把模型推向更好的能力区域”。

## Scaling law 是预算分配问题

训练预算不是只堆参数。Chinchilla 的核心提醒是：固定计算量下，参数量和训练 token 数要配平。模型太大但 token 不够，会欠训练；模型太小但 token 很多，容量会成为瓶颈。

![Chinchilla IsoFLOP curves 原论文图](../assets/images/paper-figures/training/chinchilla-isoflop-curves.png){ width="920" }

<small>图源：[Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)，Figure 4。原图表达在固定 FLOP 预算下，不同模型大小有不同最终 loss，曲线谷底对应 compute-optimal 的参数/token 配比。本站使用它说明：训练设计是一笔预算分配，参数、token、数据质量和训练步数必须一起看。</small>

这张图不要求读者记住某个固定比例，因为不同数据质量、架构、tokenizer、目标函数和后训练阶段都会改变实际选择。它真正有用的地方是改变问题问法：不要只问“模型要不要更大”，而要问“这笔算力应该分给参数、token、上下文长度、数据清洗、评测和失败复盘中的哪一块”。

## Checkpoint 是证据链，不只是备份

一个可信 checkpoint 不只是 `model.safetensors`。如果要恢复训练或解释结果，它至少要回答：来自哪份代码、哪份配置、哪份数据 manifest、哪个 tokenizer、哪个 sampler 状态、哪个 optimizer state、哪个 scheduler 位置、哪个 RNG 状态、已经消耗多少 token。

只保存权重，通常只能推理，不能证明训练可以无缝续跑。恢复后最常见的问题是：学习率跳到错误位置，数据重复或跳样，gradient accumulation 边界错位，loss scaler 重置，分布式分片映射变化，packing 的 token 边界恢复不准。

可靠做法是把 checkpoint 当作实验资产：保存模型、optimizer、scheduler、RNG、sampler、tokenizer、data manifest、配置和 eval artifacts。恢复后做短跑一致性检查，看若干 step 内的 loss、grad norm、吞吐、consumed tokens 和关键评测是否在可解释范围内。

## 一次训练改动是否可信

一个训练改动声称“更快、更稳、更好”，至少要同时交代三类证据。

质量证据包括 train/eval loss、任务 benchmark、分桶指标、失败样本和人工/闭环评估。稳定性证据包括 loss spike、NaN/Inf、grad norm、activation 分布、overflow、恢复后一致性。成本证据包括 tokens/s、MFU、显存峰值、通信等待、checkpoint pause、数据 loader stall 和每个有效 token 的成本。

这三类证据缺一类，结论都会变轻。只报质量，可能成本不可接受；只报吞吐，可能模型没变好；只报短跑稳定，可能长跑数据阶段切换后才崩。

## 常见误解

**误解一：loss 下降就代表模型更好。**  
Loss 是代理目标上的信号，不等于真实任务成功。看平均 loss 之前，要先看数据桶、评测分布和下游指标。

**误解二：GPU 利用率高就是训练高效。**  
设备忙不等于 Tensor Core、通信、数据管线和 checkpoint 都高效。训练性能要看 tokens/s、MFU、stall、通信等待和端到端成本。

**误解三：batch 改大只是为了速度。**  
Batch 会改变梯度噪声、学习率语义、warmup 需求和泛化行为。换卡数、换 accumulation 或换 packing，都可能改变训练。

**误解四：checkpoint 只是防断电。**  
Checkpoint 是恢复语义和实验审计的证据链。不能恢复 optimizer、scheduler、sampler 和数据进度，就不能保证续训曲线可信。

## 阅读结论

训练的核心不是“让数字下降”，而是让数据分布、目标函数和更新压力共同服务真实能力。数据决定模型看见什么，loss 决定哪些错误被惩罚，batch 与学习率决定每一步压力，optimizer 和数值策略决定训练能不能稳定走下去，eval 与 checkpoint 决定实验是否可信。读任何训练报告时，先沿着这条链问：有效 token 是什么，loss 在优化什么，更新压力是否稳定，系统账是否闭合，质量证据是否覆盖关键失败桶。

## 外部精读

- [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/)：训练排查经典博客，适合学习如何从小实验逐步走到可信训练。
- [Smol Training Playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook)：把训练目标、数据、评测和成本放到同一份工程手册里看。
- [SmolLM3: smol, multilingual, long-context reasoner](https://huggingface.co/blog/smollm3)：官方训练复盘，适合理解多阶段数据与后训练如何组成 recipe。
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)：理解参数、token 和计算预算如何配平。
- [ZeRO](https://arxiv.org/abs/1910.02054)：理解大模型训练显存账。
- [GPipe](https://arxiv.org/abs/1811.06965)：理解流水线并行为什么需要 micro-batch。
- [Datawhale: VRAM Calculation and ZeRO](https://datawhalechina.github.io/llm-algo-leetcode/01_Hardware_Math_and_Systems/06_VRAM_Calculation_and_ZeRO.html)：中文显存估算补充读物，适合把参数、梯度、优化器状态拆成手算账。
