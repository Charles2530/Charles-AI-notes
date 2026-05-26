# 模型训练从零入门

这页给研究入门读者一条更顺的路线：先看清一次训练循环，再读公式、阶段、论文图和系统账。你不需要一开始就懂 Megatron、ZeRO 或 RLHF，只要抓住一个问题：**模型怎样从一批数据里得到错误信号，再把这个信号变成参数更新**。

<!-- [补充] -->
| 读者 | 建议读法 | 工程落点 |
| --- | --- | --- |
| 初学者 | 先抓“模型训练从零入门”要解决的问题、关键术语和最小例子；公式只看变量方向。 | 能说清输入、输出和常见失败信号。 |
| 工程师 | 直接对照结构、指标、成本和边界条件，必要时回到前置页补概念。 | 落到数据、算力、延迟、显存、评测和回滚检查。 |


!!! note "合并提示"
    **note · 读法定位**

    这页先回答“模型训练从零入门”在「训练」里的位置：它解决什么局部问题，依赖哪些前置，最后会影响哪类工程或研究判断。
    **前置**：先知道 loss、optimizer、batch、checkpoint 和评测集的基本含义。 必要时先回 [训练入口](index.md)、[基础知识](../foundations/index.md) 或 [术语表](../glossary/index.md)。
    **主线关系**：把目标函数、数据、分布式系统、评测回流和实验治理串起来，看一次训练为什么成功、为什么不稳、为什么贵。

    **note · 本页读法**

    遇到公式时，先不要急着推导。先问四件事：谁是输入，谁是目标，模型输出了什么，loss 在惩罚什么。公式下面的“符号含义”和“白话读法”就是为这个准备的。

<!-- [补充] -->
> **踩坑原因**：这类工程问题通常不是单点公式错，而是 shape、资源预算、并发路径和观测指标之间有一个假设没对齐；先把触发条件、监控信号和回滚边界拆开查。

## 一次训练在做什么

最小训练循环可以写成：

```text
取一批数据 batch
模型做预测
用 loss 衡量预测和目标的差距
反向传播计算每个参数该怎么改
优化器按学习率更新参数
定期评测和保存 checkpoint
```

把它写成一个通用公式，就是经验风险最小化：

\[
\min_\theta
\mathcal{J}(\theta)
=
\mathbb{E}_{(x,y)\sim \mathcal{D}}
\left[\ell(f_\theta(x),y)\right].
\]

<!-- [补充] -->
| 符号 | 含义 | 取值范围 |
| --- | --- | --- |
| \(\theta\) | 模型或策略参数 | 实值张量，形状由模型结构决定 |
| \(x\) | 样本、状态或 latent 变量 | 张量空间，形状由数据表示决定 |
| \(y\) | 公式中的变量或中间量 | 按上下文定义 |
| \(f\) | 公式中的变量或中间量 | 按上下文定义 |


**白话读法**：在训练集里不断抽样，让模型预测，再把预测和目标比较；我们要找一组参数 \(\theta\)，让平均错误 \(\mathcal{J}\) 尽量小。

!!! tip "先记住一个核心句"
    Loss 不是模型能力本身，而是训练时给模型的压力。训练能不能成功，取决于这个压力是否真的对应你想要的能力。

## 数据与 batch：训练先从“看见什么”开始

训练不是一次看完整个互联网，而是每步看一小批样本。设一个 batch 为：

\[
B=\{(x_i,y_i)\}_{i=1}^{m}
\]

<!-- [补充] -->
| 符号 | 含义 | 取值范围 |
| --- | --- | --- |
| \(B\) | 公式中的变量或中间量 | 按上下文定义 |
| \(x_i\) | 样本、状态或 latent 变量 | 张量空间，形状由数据表示决定 |
| \(y_i\) | 公式中的变量或中间量 | 按上下文定义 |
| \(i\) | 时间步、索引或计数 | 整数索引或归一化时间区间，见正文 |
| \(m\) | 公式中的变量或中间量 | 按上下文定义 |


实际训练常用 batch 平均损失：

\[
\widehat{\mathcal{L}}_B(\theta)
=
\frac{1}{m}
\sum_{i=1}^{m}
\ell(f_\theta(x_i),y_i).
\]

<!-- [补充] -->
| 符号 | 含义 | 取值范围 |
| --- | --- | --- |
| \(\theta\) | 模型或策略参数 | 实值张量，形状由模型结构决定 |
| \(m\) | 公式中的变量或中间量 | 按上下文定义 |
| \(i\) | 时间步、索引或计数 | 整数索引或归一化时间区间，见正文 |
| \(f\) | 公式中的变量或中间量 | 按上下文定义 |
| \(x_i\) | 样本、状态或 latent 变量 | 张量空间，形状由数据表示决定 |
| \(y_i\) | 公式中的变量或中间量 | 按上下文定义 |


**白话读法**：这一步不可能看完整个数据集，所以用当前 batch 的平均错误近似整体错误。batch 太小，信号很抖；batch 太大，训练成本和泛化行为都会变。

Chinchilla 的图能帮你理解“数据不是越多越好，参数也不是越大越好”，而是要在固定算力下配平。

![Chinchilla IsoFLOP curves 原论文图](../assets/images/paper-figures/training/chinchilla-isoflop-curves.png){ width="920" }

<small>图源：[Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)，Figure 4。原论文图意：固定 FLOP 预算下，不同模型大小和训练 token 数会得到不同最终 loss，曲线谷底对应更合适的参数/token 配比。</small>

!!! note "图解：这张图在回答什么"
    横向看，它在问：同样算力下，参数多一点还是训练 token 多一点更划算？初学者容易只盯参数量，但这张图提醒你：如果 token 不够，大模型会欠训练；如果模型太小，再多 token 也会容量不足。训练计划要同时写清模型规模、训练 token、数据质量和计算预算。

## 模型预测：\(f_\theta(x)\) 是“当前版本的答案”

语言模型最常见的输出是下一个 token 的概率分布：

\[
p_\theta(x_t \mid x_{<t})
\]

<!-- [补充] -->
| 符号 | 含义 | 取值范围 |
| --- | --- | --- |
| \(\theta\) | 模型或策略参数 | 实值张量，形状由模型结构决定 |
| \(p\) | 概率、转移分布或预测量 | 概率分布或任务定义的张量 |
| \(x_t\) | 样本、状态或 latent 变量 | 张量空间，形状由数据表示决定 |
| \(x_{<t}\) | 样本、状态或 latent 变量 | 张量空间，形状由数据表示决定 |


**白话读法**：给定前文，模型给每个候选 token 一个概率。训练时希望正确 token 的概率越来越高。

多模态、扩散、世界模型和 VLA 也一样有“预测”这一步，只是输出对象变了：可能是图像噪声、视觉 latent、未来状态、动作 chunk 或奖励风险。

## Loss：告诉模型“错在哪里”

自回归语言模型常用交叉熵：

\[
\mathcal{L}_{\text{CE}}
=
-
\sum_{t=1}^{T}
\log p_\theta(x_t\mid x_{<t}).
\]

<!-- [补充] -->
| 符号 | 含义 | 取值范围 |
| --- | --- | --- |
| \(\theta\) | 模型或策略参数 | 实值张量，形状由模型结构决定 |
| \(t\) | 时间步、索引或计数 | 整数索引或归一化时间区间，见正文 |
| \(T\) | 时间步、索引或计数 | 整数索引或归一化时间区间，见正文 |
| \(p\) | 概率、转移分布或预测量 | 概率分布或任务定义的张量 |
| \(x_t\) | 样本、状态或 latent 变量 | 张量空间，形状由数据表示决定 |
| \(x_{<t}\) | 样本、状态或 latent 变量 | 张量空间，形状由数据表示决定 |


**白话读法**：从第 1 个位置到第 \(T\) 个位置，模型每次都要猜下一个 token。猜对且概率高，loss 小；正确 token 概率低，loss 大。

扩散模型则常用噪声预测损失：

\[
\mathcal{L}_{\text{diff}}
=
\mathbb{E}_{x_0,\epsilon,t}
\left[
\|\epsilon-\epsilon_\theta(x_t,t,c)\|_2^2
\right].
\]

<!-- [补充] -->
| 符号 | 含义 | 取值范围 |
| --- | --- | --- |
| \(\epsilon\) | 公式中的变量或中间量 | 按上下文定义 |
| \(\epsilon_\theta\) | 模型或策略参数 | 实值张量，形状由模型结构决定 |
| \(x_0\) | 样本、状态或 latent 变量 | 张量空间，形状由数据表示决定 |
| \(t\) | 时间步、索引或计数 | 整数索引或归一化时间区间，见正文 |
| \(x_t\) | 样本、状态或 latent 变量 | 张量空间，形状由数据表示决定 |
| \(c\) | 公式中的变量或中间量 | 按上下文定义 |


**白话读法**：把干净样本加噪后交给模型，让模型猜噪声是什么；猜得越接近真实噪声，loss 越小。

!!! warning "常见误区：loss 下降不等于能力一定变好"
    如果训练集里简单样本占 90%，平均 loss 下降可能只是简单样本更好了；真正重要的长尾任务、接触动作、代码测试或安全边界反而可能退化。所以训练结论必须配合分桶评测。

## 梯度与反向传播：错误怎样传回参数

参数更新的最小形式是：

\[
\theta_{t+1}
=
\theta_t
-
\eta_t
\nabla_\theta \mathcal{L}(\theta_t).
\]

<!-- [补充] -->
| 符号 | 含义 | 取值范围 |
| --- | --- | --- |
| \(\theta_{t+1}\) | 模型或策略参数 | 实值张量，形状由模型结构决定 |
| \(\theta_t\) | 模型或策略参数 | 实值张量，形状由模型结构决定 |
| \(\eta_t\) | 公式中的变量或中间量 | 按上下文定义 |
| \(t\) | 时间步、索引或计数 | 整数索引或归一化时间区间，见正文 |


**白话读法**：梯度告诉模型“往哪边改会更少犯错”，学习率告诉模型“一次改多少”。学习率太大可能震荡或炸掉，太小则训练很慢。

Loss landscape 图适合建立优化直觉：训练不是沿一条平路走，而是在复杂地形里找低谷。

![Loss landscape visualization 原论文图](../assets/images/paper-figures/foundations/loss-landscape-figure-resnet56.png){ width="820" }

<small>图源：[Visualizing the Loss Landscape of Neural Nets](https://arxiv.org/abs/1712.09913)，loss landscape visualization。原论文图意：通过二维切片可视化网络周围的 loss surface，帮助比较不同结构或训练配置下优化地形的平滑程度。</small>

!!! note "图解：这张图在回答什么"
    图里的高低起伏可以理解成 loss 地形。优化器不是直接知道全局最低点在哪里，只能根据局部梯度一步步走。残差结构、归一化、学习率、batch、初始化和优化器都会改变这条路是否好走。

长序列训练还会遇到梯度爆炸或消失。Pascanu 的 RNN 图很经典：错误信号沿时间反传时，可能被反复放大。

![Pascanu et al. Figure 6 exploding gradients](../assets/images/paper-figures/foundations/pascanu13-figure-6-exploding-gradients.png){ width="700" }

<small>图源：[On the difficulty of training recurrent neural networks](https://proceedings.mlr.press/v28/pascanu13.html)，Figure 6。原论文图意：单隐藏单元 RNN 的 error surface 中存在高曲率墙；普通梯度下降可能因为梯度过大跳过稳定区域。</small>

!!! note "图解：这张图在回答什么"
    如果梯度突然特别大，模型一步更新就可能越过稳定区域，表现成 loss spike、NaN 或训练发散。长上下文、视频、RNN、低精度和异常 batch 都可能触发类似问题。

梯度裁剪是常见刹车：

![Pascanu et al. Figure 7 gradient clipping success](../assets/images/paper-figures/foundations/pascanu13-figure-7-gradient-clipping-success.png){ width="920" }

<small>图源：[On the difficulty of training recurrent neural networks](https://proceedings.mlr.press/v28/pascanu13.html)，Figure 7。原论文图意：在 temporal order 任务上，带 clipping 和 regularization 的训练策略更能维持长序列训练成功率。</small>

!!! note "图解：这张图在回答什么"
    Clipping 不是让模型更聪明，而是防止异常大梯度把训练轨迹冲坏。使用它时要监控触发频率：如果每步都在裁剪，可能说明学习率、数据或数值路径本身有问题。

## Optimizer 与 LR：决定“怎么走”和“走多快”

实际训练通常不用最朴素的梯度下降，而是用 AdamW、Muon、Adafactor 等优化器。可以抽象成：

\[
\theta_{t+1}
=
\theta_t
-
\eta_t u_t.
\]

<!-- [补充] -->
| 符号 | 含义 | 取值范围 |
| --- | --- | --- |
| \(\theta_{t+1}\) | 模型或策略参数 | 实值张量，形状由模型结构决定 |
| \(\theta_t\) | 模型或策略参数 | 实值张量，形状由模型结构决定 |
| \(\eta_t\) | 公式中的变量或中间量 | 按上下文定义 |
| \(t\) | 时间步、索引或计数 | 整数索引或归一化时间区间，见正文 |
| \(u_t\) | 公式中的变量或中间量 | 按上下文定义 |


**白话读法**：optimizer 负责把当前梯度和历史信息整理成一个更稳的更新方向；LR schedule 决定每一步迈多大。

常见学习率日程是 warmup + decay：

\[
\eta_t =
\begin{cases}
\eta_{\max}\frac{t}{T_w}, & t\le T_w\\
\eta_{\max}f(t), & t>T_w
\end{cases}
\]

<!-- [补充] -->
| 符号 | 含义 | 取值范围 |
| --- | --- | --- |
| \(\eta_t\) | 公式中的变量或中间量 | 按上下文定义 |
| \(\eta_{\max}\) | 公式中的变量或中间量 | 按上下文定义 |
| \(T_w\) | 时间步、索引或计数 | 整数索引或归一化时间区间，见正文 |
| \(t\) | 时间步、索引或计数 | 整数索引或归一化时间区间，见正文 |
| \(f\) | 公式中的变量或中间量 | 按上下文定义 |


**白话读法**：训练刚开始模型和优化器状态都不稳，先小步走；稳定后走到最大步长；后期再慢慢收敛。

低精度训练把“走得快”和“走得稳”的矛盾放大了。FP8 图的重点不是“8 bit 一定可以”，而是要看收敛曲线是否接近高精度基线。

![FP8 training loss curves 原论文图](../assets/images/paper-figures/foundations/fp8-formats-figure-1-training-loss.png){ width="630" }

<small>图源：[FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)，Figure 1。原论文图意：比较不同规模 GPT 模型在 BF16 与 FP8 训练下的 loss/perplexity 曲线，展示 FP8 在合适 scaling 与配置下可以接近 BF16 收敛行为。</small>

!!! note "图解：这张图在回答什么"
    数值格式不能只看 bit 数。FP8 能不能用，要看 loss spike、梯度、scale 饱和率、NaN/Inf 和最终评测。低精度训练的核心是省显存和带宽，但不能把收敛质量省掉。

## 训练阶段：预训练、SFT、对齐分别学什么

大模型通常不是一次训完，而是分阶段塑形：

| 阶段 | 训练信号 | 常见目标 | 主要解决 |
| --- | --- | --- | --- |
| 预训练 | 大规模文本/代码/多模态 token | next-token 或自监督目标 | 能力边界、知识和表示 |
| SFT | 人写或强模型生成的高质量示范 | 条件交叉熵 | 指令遵循、格式、任务流程 |
| 偏好/RL | 人类排序、reward model、verifier | DPO、PPO、GRPO 等 | 倾向选择、风险边界、工具流程 |

InstructGPT 的流程图是理解后训练的好入口：

![InstructGPT RLHF pipeline 原论文图](../assets/images/paper-figures/training/instructgpt-rlhf-pipeline.png){ width="920" }

<small>图源：[Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)，Figure 2。原论文图意：展示 demonstration data、comparison data、reward model 和 PPO policy optimization 的三步流程。</small>

!!! note "图解：这张图在回答什么"
    Step 1 是示范学习：模型模仿人写的好答案。Step 2 是偏好建模：人比较多个回答，训练 reward model。Step 3 才是 RLHF：模型自己采样回答，reward model 打分，再用 PPO 更新 policy。三种数据不是同一回事，混在一起会把训练阶段讲乱。

偏好/RL 里常见一个目标：

\[
\max_\theta
\mathbb{E}_{y\sim \pi_\theta(\cdot\mid x)}
\left[
r_\phi(x,y)
-
\beta \mathrm{KL}(\pi_\theta\|\pi_{\text{ref}})
\right].
\]

<!-- [补充] -->
| 符号 | 含义 | 取值范围 |
| --- | --- | --- |
| \(\pi_\theta\) | 模型或策略参数 | 实值张量，形状由模型结构决定 |
| \(\phi\) | 概率、转移分布或预测量 | 概率分布或任务定义的张量 |
| \(\beta\) | 公式中的变量或中间量 | 按上下文定义 |
| \(\pi_\theta\\) | 模型或策略参数 | 实值张量，形状由模型结构决定 |
| \(\pi_{\text{ref}\) | 策略或概率分布 | 非负概率分布，总和为 1 |
| \(y\) | 公式中的变量或中间量 | 按上下文定义 |


**白话读法**：让当前模型更常生成高奖励回答，同时用 KL 项限制它别离参考模型太远。

PPO 的 clip 图解释了为什么“奖励高”也不能让模型一步改太猛。

![PPO clipped surrogate objective 原论文图](../assets/images/paper-figures/training/ppo-clipped-surrogate.png){ width="760" }

<small>图源：[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)，Figure 1。原论文图意：展示 clipped surrogate objective 中单个 timestep 项如何随概率比率 \(r\) 变化；当 advantage 为正或负时，clip 会限制策略概率变化带来的收益。</small>

!!! note "图解：这张图在回答什么"
    \(r\) 是新策略相对旧策略的概率比率。若某个回答比预期好，模型想提高它的概率；若比预期差，模型想降低它的概率。clip 限制概率变化带来的收益，防止模型为了 reward model 的局部偏好突然跑偏。

## 分布式与 checkpoint：训练为什么变成系统工程

单卡训练时，参数、梯度、优化器状态和激活都放在同一张 GPU 上。大模型训练时，这些状态会爆显存，所以需要并行和分片。

ZeRO 图说明普通数据并行的冗余在哪里：

![ZeRO memory optimization stages 原论文图](../assets/images/paper-figures/training/zero-memory-stages.png){ width="860" }

<small>图源：[ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)，Figure 1。原论文图意：比较普通数据并行和 ZeRO-DP 三个阶段的单设备模型状态显存，展示优化器状态、梯度和参数分片如何降低单卡常驻状态。</small>

!!! note "图解：这张图在回答什么"
    普通数据并行会在每张卡上重复保存完整参数、梯度和优化器状态。ZeRO 逐步把优化器状态、梯度和参数分散到不同 data-parallel rank 上。它省的是单卡常驻显存，代价是通信、gather/scatter 和 checkpoint 恢复语义更复杂。

GPipe 图说明流水线并行为什么要 micro-batch：

![GPipe pipeline parallelism 原论文图](../assets/images/paper-figures/training/gpipe-pipeline-parallelism.png){ width="860" }

<small>图源：[GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965)，Figure 2(c)。原论文图意：将 mini-batch 拆成多个 micro-batch，使不同 accelerator 能同时处理不同 micro-batch 的不同模型分段，并在末尾同步应用梯度。</small>

!!! note "图解：这张图在回答什么"
    Pipeline parallelism 的问题不是“能不能切模型”，而是切完后设备会不会空等。micro-batch 用来填流水线，但它也牵动激活保存、反向顺序、梯度累积和优化器更新时机。

Checkpoint 不只是保存权重。成熟训练恢复至少要保存：

| 状态 | 为什么要保存 |
| --- | --- |
| model parameters | 模型本体 |
| optimizer states | AdamW 等优化器的历史统计 |
| scheduler state | 学习率当前走到哪一步 |
| scaler / dtype state | 混合精度缩放和数值状态 |
| dataloader position | 恢复后不要悄悄换数据顺序 |
| parallel manifest | 分片参数怎样重新拼回同一模型 |

## 评测：训练结论怎样才可信

训练中最危险的误判是“loss 降了，所以模型好了”。更稳的判断至少要看：

| 要证明什么 | 应看什么 | 反例 |
| --- | --- | --- |
| 模型真的学到了 | validation、downstream、分桶指标 | train loss 降但 eval 不涨 |
| 关键能力没退 | 数学、代码、长上下文、格式、安全等 anchor eval | 平均分涨但长尾桶掉 |
| 系统更高效 | step time、MFU、峰值显存、通信 trace | 单张速度截图 |
| 实验可复现 | 数据版本、config、seed、checkpoint manifest | 恢复后曲线不连续 |

可以把一次训练结论写成一句话：

```text
在相同数据版本、模型规模、评测桶和恢复语义下，
新 recipe 是否用更少资源得到不退化或更好的能力？
```

!!! note "合并提示"
    **note · 下一步怎么读**

    读完这页后，建议按顺序进入 [目标函数、优化器与 LR 日程](objectives-optimizers-and-schedules.md)、[预训练、微调与对齐](pretraining-finetuning-alignment.md)、[论文图解索引](page-by-page-visual-guide.md)、[分布式训练与 Checkpoint](distributed-training-and-checkpointing.md)。如果公式还不稳，回看 [优化与训练入门](../foundations/optimization-and-training-basics.md)。

    **tip · 下一站**

    - 回到本专题入口：[训练](index.md)，确认这页在整条路线中的位置。
    - 按导航顺序继续：[论文图解索引](page-by-page-visual-guide.md)。
    - 概念或符号卡住时，先查 [术语表](../glossary/index.md)，再回到当前页。
