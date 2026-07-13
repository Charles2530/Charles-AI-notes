# 高效 AI 的状态账本：少搬状态，比少算一次更重要

> **本文回答的问题。** 为什么参数更少、bit 更低、attention 更稀疏、draft 接受更长，最后却不一定带来同等幅度的端到端加速？

过去我理解高效 AI，首先会数 FLOPs：矩阵乘少了多少，attention block 跳过多少，模型调用减少几次。连续读完这一组训练与推理论文后，我更在意另一张账：一次训练或生成里，系统保存了哪些状态，这些状态被复制几份，何时跨设备搬运，又在什么时刻才真正需要完整物化。

这里的“状态”范围很宽。参数、梯度和 Adam 一二阶矩是状态；attention 的 Q/K/V、反向重算中间量和长上下文 KV cache 也是状态；投机解码的候选树、stereo 的 cost volume、量化 scale 与 packed weight，同样会占据内存、带宽和调度位置。

我现在会把高效系统的目标写成一句更具体的话：

> **少搬、少复制、延后物化，只在真正的消费者到来时保留足够精度的状态。**

“少算”仍然重要。只是很多系统真正省下来的时间，来自计算前后那条更长的数据路径。反过来，一项算法即使显著减少理论 FLOPs，只要制造了更多格式转换、全量同步、临时缓存或无效候选，收益就可能停在局部 benchmark 里。

## 15 篇材料分别在移动什么

下表先把 15 篇文章放进同一张状态账本。表中的实验数字和系统结论属于论文作者或官方材料报告；“我的读法”是本文的跨文综合，不是这些论文共同证明的定律。

| 分组 | 材料 | 主要状态 | 怎样减少无效移动或复制 | 新增代价 / 我的读法 |
| --- | --- | --- | --- | --- |
| 训练状态 | [ZeRO](../paper-deep-dives/foundations/zero.md) | 参数、梯度、optimizer states | 在 DP 组内分片，当前层需要时再 gather | 常驻副本变少，通信、checkpoint 和状态生命周期更复杂 |
| 训练计算 | [Megatron-LM](../paper-deep-dives/foundations/megatron-lm.md) | MLP、attention、vocab 的 tensor shards | 按 column/row/head 切分，把同步推迟到必要边界 | 切分是否有效取决于 GEMM shape、非线性位置和通信域 |
| 长上下文 | [RingAttention](../paper-deep-dives/world-models/ringattention.md) | 长序列 Q/K/V blocks | query 留在本地，KV 沿 ring 轮转，避免每卡持有全序列 | 需要足够大的 block 和高带宽互联，长上下文本身仍要付 FLOPs |
| 异构注意力 | [MagiAttention](../paper-deep-dives/foundations/magiattention.md) | heterogeneous mask、KV/dKV、dispatch metadata | 按有效 mask area 分配计算，只向需要者 GroupCast KV | solver、metadata、group collectives 与 overlap 调参进入系统 |
| 低精注意力 | [Attn-QAT](../paper-deep-dives/foundations/attn-qat.md) | Q/K/V/P、反向重算量、量化 scale | 让 forward、backward recomputation 与 inference kernel 共用低精语义 | 为了正确部署，训练时反而要保留高精辅助状态 (O') |
| 优化器 | [Muon](../paper-deep-dives/foundations/muon-is-scalable.md) | 矩阵动量、正交化更新 | 矩阵参数只保留一份 momentum，按矩阵几何更新 | 完整矩阵正交化需要 DP gather；少状态不等于零通信 |
| 稀疏计算 | [SLA / SLA2](../paper-deep-dives/foundations/sla-sparse-linear-attention.md) | attention blocks、router、linear compensation | critical 精确算，marginal 线性补，negligible 跳过 | 路由、近似误差、微调和低比特 kernel 共同决定收益 |
| 性能观测 | [GPU Utilization](../paper-deep-dives/foundations/gpu-utilization-misleading.md) | kernel timeline、SM、tensor pipe、通信等待 | 不直接搬状态；负责识别哪些搬运和 kernel 没产生有效 FLOPs | 100% utilization 只能证明 GPU 忙，不能证明状态流动有效 |
| KV 组织 | [PagedAttention](../paper-deep-dives/inference/vllm-pagedattention.md) | 每个请求持续增长的 KV cache | logical block 映射到 physical block，按需分配并共享前缀 | block table、copy-on-write 和 scheduler 成为正确性边界 |
| KV 内容 | [KVSlimmer](../paper-deep-dives/inference/kvslimmer.md) | Key 与 Value cache entries | 合并更同质的 Key，尽量保留异质 Value；权重只用 forward 变量 | 平均分之外仍要检查关键证据是否被局部合并抹掉 |
| Draft 接口 | [EAGLE](../paper-deep-dives/inference/eagle.md) | target hidden feature、draft tokens、candidate KV | 复用 target 已算出的 feature，不再运行一个完整小 LLM | target 必须暴露稳定 feature，draft 开销仍要由接受长度摊销 |
| Draft 调度 | [EAGLE-2](../paper-deep-dives/inference/eagle-2.md) | draft tree nodes 与 confidence | 只扩展 path value 高的节点，再动态重排验证树 | confidence 失准时会把验证预算投错分支 |
| Draft 训练 | [EAGLE-3](../paper-deep-dives/inference/eagle-3.md) | draft 自己生成的递归状态 | 训练时就把自生成状态喂回模型，减少深层候选浪费 | 仍需 tree mask、临时 KV 和 target verification 共同正确 |
| 实时视觉 | [Fast-FoundationStereo](../paper-deep-dives/inference/fast-foundation-stereo.md) | feature pyramid、cost volume、ConvGRU hidden state | backbone 蒸馏、cost filtering 搜索、refinement 剪枝分别处理 | 搜索成本、伪标签门禁、相机标定与 TensorRT 路径仍在账内 |
| 数值格式 | [Low-bit LLM Survey](../paper-deep-dives/inference/low-bit-llm-survey.md) | packed weights、activation、KV、scale、zero point | 用低 bit 减少常驻字节与 HBM / 通信流量 | Q/DQ、scale 读取、layout 和硬件 kernel 可能吃掉节省的时间 |

把这些论文并排后，可以看到一个很稳定的分工：ZeRO、PagedAttention 处理“存几份”；Megatron、Ring、Magi 处理“在哪算、往哪传”；Attn-QAT、low-bit survey 处理“以什么格式传”；SLA、KVSlimmer、EAGLE-2 处理“哪些内容值得传”；EAGLE-3 和 Fast-FoundationStereo 继续追问“训练出来的状态是否真能沿部署路径被消费”。

## 先给状态分工，再谈压缩

同样占 1GB 的张量，职责可能完全不同。对它们使用同一种压缩策略，通常会把系统问题伪装成一个统一 bitwidth 或统一 sparsity 参数。

| 状态职责 | 典型对象 | 真正的消费者 | 常见误判 |
| --- | --- | --- | --- |
| 长期知识 | 模型参数、embedding、expert weights | 每次 forward 的算子 | 只看模型文件大小，不看每 token 激活和权重搬运 |
| 训练记忆 | gradient、momentum、variance、master weight | optimizer step 与故障恢复 | 只看参数 dtype，不算 optimizer state 和 checkpoint |
| 临时计算证据 | activation、Q/K/V、softmax 统计、cost volume | backward、refinement 或下一融合算子 | 以为临时张量寿命短，所以移动成本不重要 |
| 历史记忆 | LLM KV、长视频 KV、recurrent hidden state | 后续 decode 或 rollout | 只看单请求容量，不看并发、共享和释放时机 |
| 候选状态 | draft tree、beam、检索结果、规划分支 | 昂贵 verifier 或 target model | 候选越多看似覆盖越好，却可能放大验证与临时 KV |
| 数值元数据 | scale、zero point、block map、router mask | kernel、allocator、scheduler | 元数据很小，就误以为它不会改变热路径 |

SLA 和 KVSlimmer 给了两个很直观的反例。SLA 观察到，高权重 attention blocks 仍可能是高秩结构，应该精确计算；大量中间权重单个不大，却会累计影响输出，适合 linear compensation；真正可跳过的是贡献极小的部分。KVSlimmer 则把 K 和 V 拆开：Key 更像寻址，Value 更像内容。相邻 Key 可合并，不代表相邻 Value 也能平均。

压缩前，我最先问“这个状态一旦出错，谁会受影响”，然后才问“它能压到几 bit”。如果消费者是 softmax 反向、长文档证据检索或机器人深度边界，一次局部误差可能沿后续状态链放大；如果消费者只是低价值候选排序，近似空间会大得多。

## 训练系统先消除副本，再安排同步边界

ZeRO 最清楚地展示了“副本”有多贵。mixed-precision Adam 的 model states 粗略可到每参数 16 bytes；普通数据并行让每张卡都长期保存完整参数、梯度和 optimizer states。ZeRO 依次分片 optimizer、gradient 和 parameter，把“每卡完整拥有”改成“DP 组共同拥有，计算时短暂恢复”。

Megatron-LM 做的是另一层重排。MLP 第一层按输出维切，GeLU 留在本地；第二层按输入维切，到 block 输出才 all-reduce。attention 按 head 切，也把同步放在 output projection 之后。这里没有删掉 Transformer 的数学计算，收益来自中间 activation 不必在每个非线性前来回聚合。

RingAttention 把同一思路推到序列维度：query block 留在设备上，KV blocks 在设备环上逐站经过。每个 query 最终仍看到全序列，省掉的是“先把全部 KV 复制到每张卡再算”的冲动。

![RingAttention method 原图](../assets/images/localized-remote/paper-deep-dives/world-models/ringattention/ringattention-method-原图.png){ width="860" }

<small>图源：[Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889)，Figure 2。原图上半部分展示 query block 留在本设备、KV block 沿 ring 传递，下半部分展示 blockwise attention 与 feedforward 的组织。本文用它说明“流动状态”可以替代“复制状态”；这张图不能证明长上下文数据一定有用，也不能证明慢互联下通信总能被隐藏。</small>

MagiAttention 又给 Ring 的规则通信加了一层 workload awareness。heterogeneous mask 下，并非每个 rank 都需要每段 KV。AttnSlice、dispatch solver 和 GroupCast/GroupReduce 让数据移动感知有效 mask area，减少把无用 KV 绕完整个 ring 的情况。

Muon 提供了一个有意思的反例：它为了对完整矩阵动量做正交化，需要额外 gather。单看通信字节，它未必总比 AdamW 简单；但矩阵参数只保存一份 momentum，并尝试用更好的更新几何减少达到同 loss 所需的训练 FLOPs。必要的搬运可以保留，应该先删掉的是那些没有产生足够训练价值的搬运。

## 推理系统把缓存和候选变成可调度资源

PagedAttention 的贡献很像一次内存观念更新。KV cache 不再是一条必须连续增长的大数组，而是一组 logical blocks。物理 block 可以分散存放、按需分配、请求结束后立即回收；共享 prompt 的多个候选先共用 block，真正分叉时才 copy-on-write。

这项设计没有减少每个历史 token 的 K/V 内容，也没有降低 attention 读取历史的理论成本。它先提高同一显存里可容纳的 active tokens，再让 continuous batching 把容量变成吞吐。内存抽象和 scheduler 必须一起工作。

KVSlimmer 继续处理“block 里面装什么”。它保留 Value 的异质信息，主要合并相邻 Key，并把原本需要 Hessian/gradient 的权重化成 forward-only 计算。PagedAttention 管物理组织，KVSlimmer 改内容表示；两者作用层不同，也可以叠加。叠加后必须增加 lost-evidence 测试，因为容量增长不应以丢掉少数关键 token 为代价。

EAGLE 系列处理的是另一类临时状态：尚未被 target 接受的未来。

```text
EAGLE：复用 target feature，降低提出一个候选的成本
EAGLE-2：按 confidence / path value，只物化更值得验证的树节点
EAGLE-3：训练时见到自己的递归状态，提高深层候选被接受的概率
```

三篇连起来后，投机解码更像一个状态分配器。draft 负责便宜地产生候选状态，tree policy 决定物化哪些分支，target verification 决定哪些 token 和 KV 可以提交。拒绝分支必须被清理，兄弟节点不能互相偷看，临时 KV 也要进入显存账本。平均接受长度只是算法指标，P95 latency、峰值 KV 和 batch throughput 才能说明系统是否闭合。

Fast-FoundationStereo 把这种按职责分配方法带到实时视觉。feature backbone 保存的是视觉先验，适合蒸馏；cost filtering 操作小通道 4D cost volume，直接剪枝收益有限，所以用分块搜索；ConvGRU refinement 有重复迭代和通道冗余，更适合结构化剪枝。论文作者报告 3090 上 runtime 从 496ms 降到 49ms，TensorRT 路径约 21ms。这个结果来自三段热路径共同缩短，也依赖 rectified 输入、伪标签过滤和部署 backend。

## 状态格式必须和部署计算语义一致

状态移动不只有“多少字节”，还包括 dtype、layout 和重算语义。低比特 survey 反复提醒：INT4 checkpoint 变小，可能只是 host 或 HBM 中少搬了权重；如果 kernel 先把权重 dequant 成 FP16，再进入普通 MatMul，算力路径并没有一起变短。Weight-only、W&A 和 KV quantization 应该分开验收。

Attn-QAT 把这个问题推进到 fused attention 内部。FlashAttention 为省显存，会在 backward 重算概率并使用高精度 softmax 恒等式。forward 一旦换成 FP4，backward 若仍假设原来的高精度中间量，梯度语义就会错位。Attn-QAT 因此在训练中额外保留高精度辅助输出 `O'`，并让重算的量化路径与 inference kernel 对齐。

这个例子改变了我对“省状态”的直觉。为了最终少搬 FP4 attention，训练时可能要暂时多保存一个高精度状态。多出来的 buffer 不是浪费，它在保护训练和部署对同一算子的理解。真正该删的是没有消费者、重复保存或无法转成端到端收益的状态，不是所有高精度中间量。

SLA2 的 low-bit sparse branch 也需要放在这条边界上读。SLA2 专题页描述了低比特 forward 与高精 backward 的 QAT 路径；Attn-QAT 则表明，某些 FlashAttention-style backward 不能简单沿用高精假设。两者模型、格式和 kernel 不同，不能据此判定冲突已经发生；如果未来把 SLA2 routing、FP4 attention 和分布式长序列 kernel 合并，backward recomputation 应单独做数值回归，而不是默认组件可直接拼装。

EAGLE-3 和 Fast-FoundationStereo 也在处理同一种接口问题。EAGLE-3 让 draft 训练时看到部署中的自生成状态；Fast-FoundationStereo 的搜索目标直接记录候选 block 的真实 runtime，并给出 TensorRT 路径。算法图里更小的张量，只有进入真实 runtime 后才算完成加速。

## 每次加速都会把瓶颈推给下一层

作者报告的几个数字很能说明问题。

| 局部改进 | 作者报告 | 下一处压力 |
| --- | --- | --- |
| GPU 一直有工作 | Trainy 案例中 GPU utilization 为 100%，MFU 仍约 20%；优化后约 38% | softmax、memory-bound kernel、通信和并行布局 |
| SLA attention kernel | forward attention 相对 FlashAttention2 约 13.7x | 端到端视频生成约 2.2x，VAE、MLP 和采样循环成为大头 |
| ZeRO state sharding | 每卡 model-state memory 可随 DP degree 大幅下降 | all-gather、reduce-scatter、checkpoint 与恢复 |
| Ring / Magi 长上下文 | 单卡不必持有全部序列 KV | interconnect、block size、mask solver 和最慢 rank |
| PagedAttention | 碎片和前缀复制减少，可放入更多 active tokens | scheduler、block table、临时 speculative KV；历史读取仍然存在 |
| KVSlimmer | cache 和 decoder 开销低于反传式 merge baseline | 关键证据丢失、layer/head 压缩比例与非局部冗余 |
| Low-bit | 权重、activation 或 KV 字节下降 | scale、packing、Q/DQ、寄存器压力和 kernel availability |

所以我不把“瓶颈迁移”理解成优化失败。它反而是成功优化后的正常现象。问题在于，团队是否提前知道下一层是谁，并用端到端指标确认局部收益确实传了下去。

GPU utilization 那篇文章给了一个很实用的顺序：先看设备是否 idle，再看 MFU、step time 或 tokens/s，然后用 SM、tensor pipe、memory bandwidth 和 timeline 找具体阶段。推理侧也应采用相同思路：先看 TTFT、TPOT、吞吐和 P95/P99，再追 KV block、draft、Q/DQ 或 attention kernel。只看一张“100%”表盘，很容易把忙于搬运误读成高效计算。

## 我会怎样审计一个高效 AI 方案

下面是我从这 15 篇材料里整理出的状态审计表。它不是标准 benchmark，只是比单独数 FLOPs 更接近系统真实路径。

| 问题 | 要查的对象 | 失败信号 |
| --- | --- | --- |
| 谁拥有状态 | replicated、sharded、shared、copy-on-write | 每个 rank / candidate 长期保存同一份内容 |
| 状态活多久 | 常驻、按层、按 step、按请求、按 phase | 为极少使用的最大长度或分支提前预留 |
| 谁消费状态 | backward、下一 token、verifier、planner、refinement | 压缩指标正常，但关键消费者的任务质量下降 |
| 以什么格式存在 | BF16、FP4、INT4、packed block、linear compensation | Q/DQ 与 layout 转换占据热路径，或训练/部署语义不同 |
| 沿哪条路径移动 | HBM、NVLink、InfiniBand、PCIe、CPU、磁盘 | 通信无法被计算覆盖，allocator / scheduler 出现空洞 |
| 失败后怎样恢复 | rejected KV、checkpoint shard、cache block、视觉伪标签 | 状态污染下一个请求，或恢复后 loss / 输出不连续 |
| 端到端是否获益 | MFU、step time、TTFT、TPOT、P95、任务成功率 | kernel/FLOPs 很漂亮，用户侧延迟或质量没有改善 |

我会再补一张很朴素的账：

```text
状态账本 = 常驻状态 + 临时状态 + 缓存状态 + 共享状态 + 重算状态
移动账本 = HBM 读写 + 设备互联 + CPU/GPU 搬运 + 格式转换 + 分配/回收
```

如果一定要用一个式子概括我的判断，我会写成：

\[
\text{有效效率}
\approx
\frac{\text{被任务消费者真正使用的状态更新}}
{\text{计算}+\text{搬运}+\text{复制}+\text{临时物化}+\text{恢复成本}}.
\]

这不是论文里的正式指标。它只是提醒我：一次更少的矩阵乘，如果换来更多无效状态、错误精度或昂贵恢复，未必更高效；一次额外 gather 或高精 buffer，如果显著提高收敛、接受率或部署正确性，也可能很划算。

## 证据边界

本文的事实底稿来自站内 15 篇专题页及其对应论文或官方材料。文中的 16 bytes/parameter、MFU、kernel speedup、runtime、context length 和 benchmark 结果都属于作者报告，我没有把它们当成独立复现。不同论文使用的模型、硬件、数据、runtime 和 workload 不一致，数字不能直接横向排名。

“高效 AI 更像状态移动问题”“按需物化优于长期完整持有”“瓶颈会沿状态链迁移”是我的跨文综合。它们适合用来检查系统设计，还不是已经有统一实验证明的理论。具体项目仍应测自己的形状、流量、互联、任务质量与恢复路径。

## 相关阅读与下一步

- [扩散模型加速：每删一步，都要重新安排误差](diffusion-speedup-redistributes-error.md)
- [世界模型该预测什么：答案取决于谁消费未来](world-model-future-consumers.md)
- [具身智能的可纠错状态链](embodied-ai-correctable-state-chain.md)
- [前沿模型正在变成受约束的闭环](model-capability-is-a-constrained-loop.md)
