# 模型压缩、剪枝与 NAS：别只看参数少了多少

这篇文章只回答一个问题：把一个强模型变小、变快、变便宜时，为什么“参数少了”不等于“部署真的快了”。

模型压缩不是一种单独算法，而是一组约束下的再设计。你有一个原模型 \(M\)，想得到一个替代模型 \(M'\)。它要在任务质量上足够接近 \(M\)，同时降低某个真实成本：显存、权重体积、带宽、端到端 latency、P99、能耗、租用成本，或者移动端可安装大小。

可以把压缩目标粗略写成：

\[
\min_{M'}\ \mathcal{L}_{task}(M') + \lambda C(M')
\]

这里 \(\mathcal{L}_{task}\) 是任务损失，\(C(M')\) 是你真正关心的成本，\(\lambda\) 是质量和成本之间的权重。这个式子重要不在数学形式，而在提醒你：如果 \(C\) 写成参数量，你会得到小模型；如果 \(C\) 写成某张 GPU 上的 P99 latency，你会得到完全不同的设计。

## 先问压缩的是哪条成本链

常见方法经常被放在一张表里，但它们改的是不同位置。

| 方法 | 主要改什么 | 最常降低什么 | 最容易误判什么 |
| --- | --- | --- | --- |
| 蒸馏 | 训练信号和学生模型行为 | 小模型质量、训练样本效率 | student 可能继承 teacher 偏差 |
| 量化 | 权重、activation、KV 的数值格式 | 显存、带宽、部分 kernel latency | 没有 kernel 支持时只是“存得小” |
| 剪枝 | 删除 weight、channel、head、block、layer | 参数、MACs、部分 dense shape | 稀疏不一定被硬件利用 |
| NAS | 搜索结构、宽度、kernel、block 组合 | 指定平台 latency / accuracy trade-off | proxy 指标不等于真实端到端 |
| Runtime / kernel | 编译、融合、layout、batch、cache | P50 / P99 / throughput | 离开目标 shape 或硬件就失效 |

Deep Compression 的经典路线是 pruning、trained quantization、Huffman coding 三段式压缩。它特别适合说明“压缩”最初很大一部分是在解决存储和带宽：把大模型从 off-chip DRAM 压到更便宜的存储层，可能比单纯减少 FLOPs 更重要。但现代 GPU/LLM/VLM 场景里，还要追问压缩后的结构是否能被 Tensor Core、TensorRT、FlashAttention、稀疏 kernel 或移动端 NPU 真的吃到。

## 参数、MACs 和 latency 不是同一个指标

先把三个最常被混用的指标拆开：

| 指标 | 它测什么 | 读论文时的边界 |
| --- | --- | --- |
| Params | 权重数量或权重存储 | 不等于每次推理都触达；MoE、cache、embedding 会让口径复杂 |
| MACs / FLOPs | 某个输入 shape 下的乘加 | 不含 memory movement、kernel launch、layout 转换、dequant、同步 |
| Latency | 真实硬件上跑完请求的时间 | 依赖 batch、shape、compiler、runtime、cache、driver、数据搬运 |

MACs 降了但 latency 不降，常见原因很具体：kernel 太碎、shape 不对齐 Tensor Core、稀疏模式不被硬件支持、dequant 或 gather/scatter 吃掉收益、算子融合失败、batch 太小导致 launch overhead 占主导、或者瓶颈本来就在 memory bandwidth 而不是乘加。

所以压缩论文里的强证据不是“参数少了 50%”，而是同硬件、同 batch、同输入 shape、同 runtime 下的端到端 latency 和模块分解。最好还能报告 P99，因为线上服务的用户体验常被尾延迟决定。

## 剪枝先分清稀疏和结构

剪枝的直觉是：模型里有些参数、通道或模块对最终质量贡献较小，可以删除，再训练恢复精度。问题在于“删掉”有很多层级。

| 剪枝粒度 | 删除对象 | 部署含义 |
| --- | --- | --- |
| Unstructured pruning | 单个 weight | 稀疏率高，但普通 dense GEMM 不会自动变快 |
| Semi-structured pruning | 固定模式，例如 2:4 | 如果硬件和 runtime 支持，可以换到稀疏 Tensor Core 路径 |
| Structured pruning | channel、filter、head、block | 改变 dense tensor shape，更容易变成真实加速 |
| Layer / block dropping | 整层或整块 | 降低路径长度，但质量恢复更难 |

Lottery Ticket Hypothesis 说明过一个重要现象：大网络里可能存在很小的可训练子网络。但这不是部署结论。它告诉你过参数化里有冗余和初始化结构，不等于随便剪出来的稀疏模型就能在你的硬件上更快。

NVIDIA Ampere/Hopper 的 2:4 structured sparsity 是另一个很好的反例：它不是任意稀疏，而是每四个连续权重里至少两个为零。这个约束看似更死板，但正因为模式固定，硬件才能存非零值和少量 metadata，并让稀疏 Tensor Core 处理。部署世界经常如此：自由度少一点，硬件收益反而更真实。

## 结构化剪枝为什么要看依赖图

如果你剪掉卷积层的第 17 个输出通道，后续消费这个特征图的层也要同步删除对应输入通道。残差、concat、attention head、recurrent state 和多输出 head 会让这种依赖变复杂。

常见依赖如下：

| 结构 | 不能随便剪的原因 |
| --- | --- |
| residual add | 两条分支输出 channel 必须一致 |
| concat | 后续层输入 channel 要按拼接结果重算 |
| attention head | Q/K/V 和 output projection 通常要按 head 成组处理 |
| recurrent state | \(h_{t-1}\) 的通道会被下一步 \(h_t\) 消费 |
| fixed output head | 类别数、mask 通道、disparity 输出格式不能改 |

DepGraph 这类结构化剪枝工作强调的就是 dependency graph：先自动找出哪些参数必须作为一组被剪，再对这组参数评估重要性。否则剪枝代码可能 shape 不报错，但语义已经错了；或者直接 shape 报错，说明你剪掉了某个下游还在使用的接口。

Fast-FoundationStereo 的 refinement pruning 图很适合作为部署型剪枝案例。

![Fast-FoundationStereo refinement pruning](../assets/images/paper-deep-dives/inference/fast-foundation-stereo/figure-5-refinement-pruning.png){ width="800" }

<small>图源：[Fast-FoundationStereo](https://arxiv.org/abs/2512.11130)，Figure 5。原图表达 refinement module 里的 ConvGRU 存在 recurrent dependency，剪枝时要区分可剪通道、固定通道和依赖关系。本站使用这张图说明：结构化剪枝不是简单按通道重要性排序，还要保证迭代模块前后接口一致。</small>

图里最关键的是 recurrent dependency。ConvGRU 会反复更新 hidden state，上一轮输出的 hidden channel 下一轮还要继续用。若只按某一层局部重要性剪掉通道，后续迭代可能接口不一致。真正可部署的剪枝流程通常是：构建依赖图，按组评估重要性，剪掉结构组，短期恢复训练，再测真实 runtime。

## Taylor 重要性只是一阶近似

剪枝需要估计“删谁伤害最小”。一种常见方法是一阶 Taylor 近似。对参数 \(\theta_i\)，可以写成：

\[
\Delta \mathcal{L}_i
\approx
\left|
\frac{\partial \mathcal{L}}{\partial \theta_i}\theta_i
\right|
\]

这里 \(\Delta \mathcal{L}_i\) 表示删掉第 \(i\) 个参数后损失可能增加多少，\(\frac{\partial \mathcal{L}}{\partial \theta_i}\) 是训练样本上 loss 对该参数的梯度，\(\theta_i\) 是参数值本身。直觉是：如果一个参数很大，而且 loss 对它很敏感，删掉它更危险；如果乘积很小，它可能更冗余。

实际工程里不会只剪单个 weight。结构化剪枝会把一个 channel、head 或 block 里的分数聚合起来，再按依赖组排序。这里有两个边界：第一，Taylor 是局部一阶近似，剪完后其他参数会重新适应；第二，剪枝后的质量常常取决于 retraining，而不是剪完那一刻的 loss。

## NAS 搜索的是结构，不是魔法模型

NAS 的核心不是“自动发现神奇架构”，而是在给定约束下搜索结构组合。例如你可以让搜索空间包含不同 kernel size、channel width、depth、attention head 数、block 类型和 expansion ratio，然后用成本模型约束 latency。

一个平台感知 NAS 目标可以写成：

\[
\max_{a\in\mathcal{A}} Acc(a)
\quad
\text{s.t.}
\quad
Latency_h(a)\le \tau
\]

这里 \(a\) 是候选架构，\(\mathcal{A}\) 是搜索空间，\(Acc(a)\) 是验证集质量，\(Latency_h(a)\) 是在目标硬件 \(h\) 上测到或预测的延迟，\(\tau\) 是延迟预算。MnasNet 这类工作的重要性就在于直接把移动端真实 latency 放进目标，而不是只用 FLOPs proxy。

OFA 的思路则是先训练一个 supernet，再从里面抽出适配不同设备和预算的 subnet。它的价值是把“为每个设备重新训练一个网络”的成本降下来；边界是 supernet 训练、progressive shrinking、子网采样和最终校准本身也会引入工程复杂度。

## Blockwise NAS 把搜索变成可承受的组合优化

Fast-FoundationStereo 里 cost filtering 的 channel dimension 已经不大，直接结构剪枝收益有限，论文改用 blockwise NAS：每个 block 提供多个候选结构，先让候选 block 局部蒸馏 teacher block，再测候选替换后的误差变化和耗时变化，最后用 ILP 选组合。

可以把选择问题写成：

\[
\min \sum_{i,c} \Delta m_i^c e_i^c
\quad
\text{s.t.}
\quad
\sum_{i,c} \Delta t_i^c e_i^c \leq \Delta \tau
\]

这里 \(i\) 是第几个 block，\(c\) 是该 block 的候选结构，\(\Delta m_i^c\) 是选择候选 \(c\) 后的误差变化，\(\Delta t_i^c\) 是耗时变化，\(e_i^c\) 是 0/1 决策变量，\(\Delta \tau\) 是总 latency 预算。读作：在总耗时约束内，为每个 block 选一个候选，让误差增加尽量小。

![Fast-FoundationStereo blockwise search](../assets/images/paper-deep-dives/inference/fast-foundation-stereo/figure-8-blockwise-search.png){ width="780" }

<small>图源：[Fast-FoundationStereo](https://arxiv.org/abs/2512.11130)，Figure 8。原图比较不同 latency budget 下 searched candidate 和 random candidate 的精度/速度。本站使用这张图说明：NAS 的价值不是搜索空间听起来很大，而是在明确预算下找到比随机选择更好的结构组合。</small>

这类 blockwise proxy 有代价：局部误差不一定完全可加，候选之间可能有交互，最终仍要端到端 retrain 和评测。但它给了一个清晰工程模式：先把昂贵的全局搜索拆成局部候选，再用可解释的成本约束组合。

## 同一模型里常常要混用多种压缩

真正的高效模型很少只靠一种技术。Fast-FoundationStereo 的总体框架很典型：不同瓶颈对应不同工具。

![Fast-FoundationStereo framework](../assets/images/paper-deep-dives/inference/fast-foundation-stereo/figure-3-framework-padded.png){ width="920" }

<small>图源：[Fast-FoundationStereo](https://arxiv.org/abs/2512.11130)，Figure 3。原图表达 FoundationStereo pipeline 包含 feature extraction、cost filtering 和 disparity refinement，论文分别用 backbone distillation、cost filtering blockwise search、refinement pruning 加速。本站使用这张图说明：部署型压缩要先做瓶颈拆解，再为每段选择不同方法。</small>

这张图的读法是先找热路径，而不是先找“用了什么算法”。Feature backbone 重，适合用 distillation 换成轻量 student；cost filtering 的结构空间复杂，但直接剪枝收益有限，适合 blockwise NAS；refinement 是反复迭代模块，有结构冗余，适合 dependency-aware structured pruning；最后再接 TensorRT/runtime 验证。它的核心经验是：压缩策略应该跟瓶颈形态匹配，而不是拿同一种方法扫全模型。

## 一个可执行的压缩判断流程

压缩前先写清目标资源。如果目标是移动端安装包，Deep Compression 式存储压缩、权重量化和编码很重要；如果目标是 LLM decode TPOT，KV cache、attention kernel、batching 和 speculative decoding 可能比剪枝更关键；如果目标是机器人实时感知，端到端 P99、输入分辨率、TensorRT 支持和失败场景更关键。

可以按这条顺序判断：

| 步骤 | 要问的问题 | 为什么 |
| --- | --- | --- |
| 1. 定义瓶颈 | 是权重、activation、KV、算力、带宽、P99 还是能耗？ | 不同瓶颈对应不同压缩方法 |
| 2. 固定口径 | 硬件、batch、shape、runtime、精度格式是否固定？ | 防止 benchmark 漂移 |
| 3. 选择方法 | 量化、蒸馏、剪枝、NAS、kernel 哪个正中瓶颈？ | 避免为了压缩而压缩 |
| 4. 保留训练恢复 | 是否需要 retrain、KD、QAT、校准或继续训练？ | 压缩后质量通常要恢复 |
| 5. 测端到端 | 是否报告模块分解、P50/P99、throughput、内存峰值？ | 局部快不等于整体快 |
| 6. 看失败分桶 | OOD、长尾、小目标、透明材质、长上下文是否掉点？ | 压缩常先伤难例 |

## 常见误解

| 误解 | 更准确的说法 |
| --- | --- |
| 参数少就一定快 | 只有当计算、带宽或 runtime 热路径真的减少，才会快 |
| 稀疏率高就一定能加速 | 稀疏模式必须被硬件、kernel 和 runtime 支持 |
| 剪枝就是按重要性排序删除 | 结构化剪枝还要满足 dependency graph |
| NAS 找到的是通用最优架构 | 它通常只在某个搜索空间、数据、硬件和预算下最优 |
| MACs 是最公平指标 | MACs 方便比较，但不能替代真实 latency 和能耗 |
| 压缩只会影响平均精度 | 长尾、OOD、校准、安全边界和失败模式常先被压坏 |

## 阅读结论

模型压缩的核心不是“把模型变小”，而是“把模型变成目标平台愿意执行的形态”。剪枝要看结构依赖，量化要看数值分布和 kernel，NAS 要看搜索空间和真实 latency，蒸馏要看 teacher 信号和 student 容量，runtime 要看端到端热路径。

读任何压缩、剪枝或 NAS 论文时，先画三件事：原模型的热路径，压缩方法改动的位置，目标硬件实际省下的成本。只要这三件事没对齐，参数表再漂亮也可能只是纸面压缩。

## 外部精读

- [Deep Compression](https://arxiv.org/abs/1510.00149)：理解 pruning、trained quantization、Huffman coding 如何共同降低存储和带宽。
- [The Lottery Ticket Hypothesis](https://arxiv.org/abs/1803.03635)：理解稀疏子网络和剪枝现象，但不要把它直接等同于部署加速。
- [DepGraph: Towards Any Structural Pruning](https://arxiv.org/abs/2301.12900)：理解 dependency graph 为什么是结构化剪枝的核心。
- [NVIDIA Structured Sparsity](https://developer.nvidia.com/blog/structured-sparsity-in-the-nvidia-ampere-architecture-and-applications-in-search-engines/)：理解 2:4 稀疏为什么要匹配硬件和 TensorRT / cuSPARSELt 路径。
- [MnasNet](https://arxiv.org/abs/1807.11626)：理解平台感知 NAS 为什么要把真实 latency 放进目标。
- [Once-for-All](https://arxiv.org/abs/1908.09791)：理解 supernet 和多设备子网选择。
- [PyTorch Pruning Tutorial](https://docs.pytorch.org/tutorials/intermediate/pruning_tutorial.html)：看工程上 mask、parameter reparameterization 和 pruning API 如何落地。
- [Fast-FoundationStereo](https://arxiv.org/abs/2512.11130)：看蒸馏、blockwise NAS、结构化剪枝如何在同一个实时感知系统里配合。
