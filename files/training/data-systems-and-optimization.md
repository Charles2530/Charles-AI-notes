# 训练数据系统与吞吐优化

很多人把训练问题理解成“调优化器”。这只对了一小部分。在大模型训练里，优化器当然重要，但真正决定结果的往往是整套训练系统：数据如何清洗和混合，batch 如何组织，序列如何 packing，学习率如何调度，dataloader 能否持续喂满 GPU，checkpoint 与容错是否可靠。

<!-- [补充] -->
| 读者 | 建议读法 | 工程落点 |
| --- | --- | --- |
| 初学者 | 先抓“训练数据系统与吞吐优化”要解决的问题、关键术语和最小例子；公式只看变量方向。 | 能说清输入、输出和常见失败信号。 |
| 工程师 | 直接对照结构、指标、成本和边界条件，必要时回到前置页补概念。 | 落到数据、算力、延迟、显存、评测和回滚检查。 |


!!! note "读法定位"
    这页先回答“训练数据系统与吞吐优化”在「训练」里的位置：它解决什么局部问题，依赖哪些前置，最后会影响哪类工程或研究判断。
    **前置**：先知道 loss、optimizer、batch、checkpoint 和评测集的基本含义。 必要时先回 [训练入口](index.md)、[基础知识](../foundations/index.md) 或 [术语表](../glossary/index.md)。
    **主线关系**：把目标函数、数据、分布式系统、评测回流和实验治理串起来，看一次训练为什么成功、为什么不稳、为什么贵。

这页把“优化”和“数据系统”放在一起看，因为现实里它们几乎总是耦合的。更细的数据配方讨论见 [Scaling、课程学习与数据配比](scaling-curriculum-and-data-mixture.md)，运维侧见 [集群运维与实验治理](cluster-operations-and-experiment-management.md)。

!!! note "合并提示"
    **note · 页面边界**

    本页是训练数据生产线的总览：把数据治理、有效 token、batch、packing、吞吐和 checkpoint 放到同一张系统图里。具体的 packing、tokenizer、streaming 细节放在 [输入管线、Packing 与吞吐](input-pipelines-packing-and-throughput-governance.md)；数据质量和去重细节放在 [数据质量、去重与治理](data-quality-dedup-and-governance.md)；数据配比和课程学习放在 [Scaling、课程学习与数据配比](scaling-curriculum-and-data-mixture.md)。

    **note · 初学者先抓住**


    大模型训练不是“模型在 GPU 上孤立学习”，而是一条数据生产线不断把样本、token、mask、batch 和状态送进模型。优化器只能处理已经送到它面前的信号；如果数据管线慢、乱、重复或恢复不一致，训练结论就会被系统问题污染。

    **note · 关键判断：有效 token 比步数更重要**

    两个实验都跑了 10 万步，不代表它们学到的东西一样多。若一个实验 padding 很多、坏样本很多，真正参与 loss 的有效 token 少；另一个实验 packing 好、样本质量高，它每一步的学习信号更密。比较训练时应看有效 token、数据曝光和吞吐，而不只是 global step。

    **tip · 主线定位：降低数据成本、token 成本和训练成本**


    数据系统页在“世界模型高效训练”里负责把原始样本变成有效训练信号。它降低的是数据成本、token 成本和训练成本：同样 GPU-hour 下让更多高价值 token 进入 loss，同时避免坏样本、重复样本、padding 和恢复错位污染结论。

<!-- [补充] -->
> **踩坑原因**：这类工程问题通常不是单点公式错，而是 shape、资源预算、并发路径和观测指标之间有一个假设没对齐；先把触发条件、监控信号和回滚边界拆开查。

## 有效 Token 是核心资源

设单步训练耗时为 \(T_{\text{step}}\)，有效 token 数为 \(N_{\text{eff}}\)，训练吞吐可以写成：

\[
\text{Throughput} = \frac{N_{\text{eff}}}{T_{\text{step}}}
\]

<!-- [补充：符号说明] -->
| 符号 | 含义 | 取值范围 / 常见值 |
| --- | --- | --- |
| \(N_{}\) | 公式中的变量或中间量 | 按上下文定义 |
| \(T_{}\) | 时间步、索引或计数 | 整数索引或归一化时间区间，见正文 |


这里的 \(N_{\text{eff}}\) 不是名义 batch token，而是扣掉 padding、无效样本、mask 区域、坏样本之后真正参与学习的 token。训练系统真正优化的资源单位，往往不是“步数”，而是有效 token。

低吞吐会直接压缩研究空间：实验次数更少、训练时长被迫缩短、消融规模变小，数据配比和阶段切换也更难充分探索；一旦失败，重跑成本会迅速变成研究瓶颈。

因此吞吐不是单纯运维指标，而是训练方法学的一部分。

## 数据链路

<!-- [融合：算法基础] 来源：llm-algo-leetcode《09. SFT Training Loop | 监督微调训练框架: 数据构造与 Loss Masking (SFT Training Loop)》-->
> **算法视角**：训练循环可以拆成 batch 构造、前向、loss mask、反向和 optimizer step；SFT / 训练循环章节特别提醒，哪些 token 参与 loss 往往和模型结构一样影响结果。
>
> 延伸阅读：[09. SFT Training Loop | 监督微调训练框架: 数据构造与 Loss Masking (SFT Training Loop)](https://datawhalechina.github.io/llm-algo-leetcode/02_PyTorch_Algorithms/09_SFT_Training_Loop.html)

<!-- [融合：复杂度分析] 来源：llm-algo-leetcode《09. SFT Training Loop | 监督微调训练框架: 数据构造与 Loss Masking (SFT Training Loop)》-->
| 操作 | 时间复杂度 | 空间复杂度 | 说明 |
|------|-----------|-----------|------|
| 单步训练 | 约为前向的 2 到 3 倍 | 参数、梯度、优化器状态、激活共同占用显存 | 实际成本还取决于 batch token、mask、checkpoint 和并行策略；参考：[对应章节](https://datawhalechina.github.io/llm-algo-leetcode/02_PyTorch_Algorithms/09_SFT_Training_Loop.html) |

一个完整训练数据系统通常从原始采集、解析清洗、去重过滤开始，经过质量打分、重采样、多源混合和分桶 packing，再进入 shard 分布式读取、在线监控与回流。每一层都会改变最终进入模型的 token 分布。

这条链路比很多模型结构改动更稳定地决定上限。常见训练失败并不是架构错，而是数据质量失控、去重没做好、多源混合比例偏、dataloader 吞吐跟不上、packing 差导致 GPU 空转，或后训练数据和预训练底座严重错位。

数据系统要记录的不只是“用了哪个数据集”，还包括过滤器版本、质量打分器版本、去重粒度、sampler 状态、packing 版本和实际曝光比例。否则训练结果很难复现。

## 优化器与学习率

AdamW 仍是 Transformer 训练中最常见的默认选择，原因很现实：对不同尺度参数较稳，对 noisy gradient 适应性好，工程实现成熟，大多数经验都围绕它建立。

但优化器只是局部。真实训练里更常见的问题是学习率日程、warmup、batch 和数据顺序不匹配。一个典型 schedule 是 warmup + decay：

\[
\eta_t =
\begin{cases}
\eta_{\max}\frac{t}{T_w}, & t \le T_w \\
\eta_{\max}\cdot f(t), & t > T_w
\end{cases}
\]

<!-- [补充：符号说明] -->
| 符号 | 含义 | 取值范围 / 常见值 |
| --- | --- | --- |
| \(\eta_t\) | 公式中的变量或中间量 | 按上下文定义 |
| \(\eta_{\max}\) | 公式中的变量或中间量 | 按上下文定义 |
| \(T_w\) | 时间步、索引或计数 | 整数索引或归一化时间区间，见正文 |
| \(t\) | 时间步、索引或计数 | 整数索引或归一化时间区间，见正文 |
| \(f\) | 公式中的变量或中间量 | 按上下文定义 |


Warmup 不只是数学上更平滑，也能让混合精度、分布式梯度统计、dataloader 和异步管线进入稳定区间。很多早期发散看起来像“优化器问题”，实际是过大的初始学习率叠加数据和系统抖动。

评估优化配置时，至少要把 loss spike、grad norm、参数范数、学习率变化点、batch 调整、数据源曝光、吞吐和系统异常时间线放在一起看。单独看 loss，很容易把数据抖动或系统 stall 误判成优化器问题。

## Batch、Packing 与 Sampler

有效 batch 可写成：

\[
B_{\text{eff}} = B_{\text{micro}} \times N_{\text{accum}} \times N_{\text{data-parallel}}
\]

<!-- [补充：符号说明] -->
| 符号 | 含义 | 取值范围 / 常见值 |
| --- | --- | --- |
| \(B_{}\) | 公式中的变量或中间量 | 按上下文定义 |
| \(N_{}\) | 公式中的变量或中间量 | 按上下文定义 |


但在长文本、代码、多模态和长视频场景里，只看样本数 batch 不够。更重要的是 tokens per update、packed token utilization、mask 比例和不同长度桶的曝光。

Packing 的目标是减少 padding 浪费，但它也会改变训练分布。坏 packing 可能引入样本边界错误、attention mask 泄漏、短样本过度曝光、长样本系统性丢弃，甚至让不同数据源在 packed batch 中的比例失真。

Sampler 也不是简单 shuffle。大规模训练需要追踪数据源采样比例、shard 顺序、恢复位置、重复样本、跳样、多 worker 随机性，以及 checkpoint 恢复后的数据游标。

一旦 sampler 状态丢失，训练可能看起来正常继续，实际已经重复读样或跳过关键数据。

## 数据质量与混合

数据系统本身也在做优化。去重、过滤、打分和重采样会决定每个 token 的学习价值。

| 环节 | 目标 | 常见风险 |
| --- | --- | --- |
| 去重 | 降低重复和评测污染 | 粒度过粗伤害合法模板，过细挡不住近重复 |
| 过滤 | 去掉乱码、低质、错配样本 | 过滤器偏置导致长尾丢失 |
| 打分 | 提高高价值样本曝光 | 过拟合打分器偏好 |
| 混合 | 控制能力结构 | 某源过强导致能力漂移 |
| 回流 | 把失败和长尾写回训练 | 线上噪声污染训练集 |

数据混合本质上是隐形目标函数。一个代码模型若自然语言比例过高，代码能力会被冲淡；若全是代码，又可能不会解释和对话。一个企业助手若内部规范和事故复盘权重过低，就会长期停留在通用助手，而不是懂内部系统的助手。

### 真实场景：回流数据进来了，指标为什么没涨

假设一个世界模型团队把线上失败回流加入训练，期望提高遮挡和接触恢复能力。数据日报显示新增 `1.8M` clips，但训练后主评测几乎不变：

```text
raw_reflow_clips=1.8M
valid_after_decode=1.42M
valid_after_action_align=0.96M
kept_after_near_duplicate=0.41M
used_in_training_exposure=2.7%
contact_recovery_bucket_exposure=0.4%
```

问题不在“回流没用”，而在回流样本从采集到实际曝光一路缩水。若只看原始 clips 数，会高估数据增量；真正进入 loss 的 contact recovery token 只有 `0.4%` 曝光，几乎无法改变策略。

修复链条应按数据生产线拆：

```text
症状：加入回流数据后平均指标不变，高风险桶仍差
观察：decode 通过率、action 对齐通过率、近重复过滤、bucket 曝光、有效 token 占比
判断：高价值样本在过滤和 sampler 阶段被稀释
修复：给回流样本独立 bucket、单独质量阈值、最小曝光配额、按失败类型做 replay eval
反例：强行提高回流比例可能引入线上噪声和分布偏移，主分布 success 会掉
边界：回流数据只有在标签、动作对齐和失败类型可解释时才是资产；否则只是更贵的噪声
```

## Dataloader 与存储路径

训练吞吐不只由 GPU kernel 决定。数据路径慢，会直接让 GPU 等数据。

常见瓶颈包括对象存储带宽不足、小文件过多、在线解压或解析太慢、预处理没有离线化、本地缓存命中率低、多作业争用共享文件系统，以及 checkpoint 写入和训练读取互相干扰。

更稳的做法是把数据路径拆成三层：原始数据湖、训练前处理后的可训练数据层、集群本地缓存或高速读取层。训练作业不应直接依赖原始对象存储的随机小文件访问。

监控 dataloader 时，不要只看平均吞吐，还要看 loader stall、worker 重启、缓存命中率、数据源曝光和不同 rank 的 step time 漂移。

## Checkpoint 与容错

Checkpoint 不只是保存模型权重，还要保存完整训练状态：模型参数、优化器、scheduler、RNG、scaler 或低精度状态、sampler 与数据游标、配置快照、代码版本，以及并行拓扑相关状态都应能被恢复。

恢复训练要证明等价：恢复后 loss、吞吐、数据曝光和随机性应落在预期范围内。否则训练可能在表面上继续，实际已经进入不可比较状态。

Checkpoint 路径也要和数据读取路径分开规划。异步写入、热/冷归档、恢复演练和 checkpoint 校验都属于训练系统质量的一部分。

## 诊断清单

当训练效果或吞吐异常时，建议按下面顺序排查：

| 排查层 | 重点看什么 |
| --- | --- |
| 有效 token | padding、mask、坏样本、packing 利用率是否异常 |
| 数据质量 | 重复、过滤、错配、多源比例是否变化 |
| Sampler | 是否跳样、重复读样、恢复游标错误 |
| 优化状态 | 学习率、warmup、grad norm、loss scaling 是否异常 |
| Batch 结构 | micro batch、accumulation、DP、长度桶是否变化 |
| 数据路径 | loader stall、缓存命中、对象存储和文件系统是否抖动 |
| Checkpoint | 恢复是否等价，状态是否完整 |
| 系统联动 | 异常是否与节点、网络、kernel 或通信时间线重合 |

训练系统的核心目标，是把高质量 token 以稳定、可复现、可解释的方式送进优化过程。只有数据链路、batch 组织、优化日程和容错机制一起成立，训练曲线才值得被拿来做方法结论。

!!! tip "下一站"
    - 回到本专题入口：[训练](index.md)，确认这页在整条路线中的位置。
    - 按导航顺序继续：[数据质量、去重与治理](data-quality-dedup-and-governance.md)。
    - 概念或符号卡住时，先查 [术语表](../glossary/index.md)，再回到当前页。
