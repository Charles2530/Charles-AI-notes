# 训练稳定性、数值异常与故障排查

大模型训练里最贵的失败，往往不是“效果差一点”，而是训练跑到中后期突然 `nan`、loss spike、恢复后曲线漂移、低精度路径悄悄失真，最后一整段 GPU 小时白白损失。稳定性排障的目标，是把这些高代价问题拆成可观测、可定位、可复现的流程。

这页关注训练故障的 triage 顺序。低精度训练细节可继续看 [低比特训练与数值体系](low-bit-training-and-numerics.md)，平台侧恢复流程见 [训练集群运维、实验管理与成本治理](cluster-operations-and-experiment-management.md)。

!!! note "初学者先抓住"
    训练排障不要从“调参”开始，而要先分类：是立刻爆炸、中途尖刺、缓慢漂移、恢复后失真，还是低精度隐性退化。症状不同，最该查的层完全不同。

!!! example "有趣例子：看病先分诊"
    `nan` 像突然高烧，loss spike 像间歇性剧痛，恢复后曲线漂移像手术后指标变了。医生不会一上来只开止痛药，训练排障也不能一上来只调小学习率。

## 先分类，再排障

训练不稳定不是一种现象，至少可以分成几类：

| 类型 | 表现 | 高概率根因 |
| --- | --- | --- |
| 立即爆炸 | 起跑不久就 `nan/inf` | 输入异常、mask 错、学习率过大、低精度初始化问题 |
| 中途尖刺 | 长时间正常后 loss spike | 数据切换、rare sample、低精度边界、恢复状态不完整 |
| 缓慢漂移 | 不挂但质量越来越怪 | 数据分布漂移、mixture 失真、schedule 不匹配 |
| 恢复后失真 | checkpoint 后曲线不连续 | sampler/RNG/scheduler/optimizer 状态未恢复 |
| 低精度隐性退化 | 没有报错但长期变差 | 累加精度、scale 统计、fused kernel 数值误差 |

不先分类，团队很容易把所有问题都归咎于“学习率太大”。大量稳定性问题本质上是数据、恢复、并行或实现错误，盲目调参只是在给错误路径打补丁。

## 常见故障源

| 路径 | 常见问题 | 排查信号 |
| --- | --- | --- |
| 数据 | tokenizer 损坏、空样本、极端长样本、label mask 错、多模态错配 | 样本 ID、长度桶、数据源曝光、坏样本率 |
| 优化器与 schedule | warmup 太短、batch 变化但 LR 未改、阶段切换未重配 | LR、grad norm、参数范数、loss spike 时间点 |
| 低精度 | loss scaling、FP8 scale、fused kernel 累加路径、master weight 不一致 | scaler、amax、溢出计数、逐层极值 |
| 分布式 | 某 rank 先坏、all-reduce 后扩散、pipeline 状态错位 | rank 级 loss/grad、通信错误、stage 时间线 |
| Checkpoint | optimizer moment、RNG、sampler、scheduler 状态缺失 | 恢复前后曲线、数据游标、global step |

排障时应先问：这是新数据、新 checkpoint、新并行配置、新 kernel 还是新精度路径引入的。变更点往往比症状本身更有信息量。

## Nan、Inf 与 Loss Spike

`nan/inf` 不一定来自“模型太大”。更常见的是输入里有非法值、mask 逻辑错误、某层输出异常放大、低精度范围溢出，或某个 rank 先坏后通过同步扩散。

建议按阶段缩圈：

1. 输入检查：token、mask、label、图像/视频张量是否有非法值；
2. 前向检查：逐层 activation min/max、norm、`nan/inf` 首次出现位置；
3. 反向检查：逐层 grad min/max、grad norm、异常 rank；
4. 更新检查：optimizer state、weight norm、loss scaler、FP8 scale；
5. 数据定位：记录触发 batch、样本 ID、长度桶和数据源。

Loss spike 尤其要看时间点。若 spike 正好出现在 curriculum 切换、数据版本切换、恢复点、LR 变化点或新 shape bucket 出现时，优先排查这些外部事件，而不是直接调小学习率。

## BF16、FP16、FP8 的故障画像

不同精度路径的故障画像并不相同，不能统一叫“低精度不稳定”。

| 精度 | 常见问题 | 排查重点 |
| --- | --- | --- |
| FP16 | 动态范围不够，容易溢出 | loss scale、grad clip、累加精度、恢复点 scaler |
| BF16 | 指数范围大，不常炸，但可能悄悄退化 | reduction 累加、fused optimizer、长期质量漂移 |
| FP8 | 校准链路和 scale 策略更复杂 | amax、per-tensor/per-token scale、溢出/下溢、敏感层回退 |

FP8 训练还要特别关注 activation outlier、scale 更新滞后、非线性层数值范围、FlashAttention 或 fused kernel 内部低精度路径。很多 FP8 问题不是一步爆炸，而是先出现局部极值和 loss sharpness 变化，再在后期放大。

低精度排障的基本原则是：先确认是数值路径问题，再决定是提高某些操作精度、修改 scale 策略、回退敏感层，还是修正 kernel。

## 恢复后漂移

Checkpoint 恢复后曲线漂移是一类单独问题。它危险在于训练表面上还能继续，但统计已经变了。

常见原因包括：

1. sampler 状态没恢复；
2. RNG 状态变化；
3. scheduler 和 global step 对不上；
4. optimizer moment 或 scaler 丢失；
5. 并行配置变化；
6. 数据快照、packer 或 tokenizer 版本变化。

恢复训练必须做一致性校验：恢复后若干 step 的 loss、grad norm、吞吐、数据曝光和长度桶分布应落在预期范围内。对于高成本训练，建议定期做恢复演练，而不是等事故发生才第一次验证恢复链路。

## 推荐 Triage 顺序

一个高效排障顺序是：

1. 确认异常类型：立即爆炸、中途 spike、缓慢漂移、恢复后漂移、低精度退化；
2. 对齐时间线：数据切换、LR 变化、checkpoint、kernel 更新、并行配置变化；
3. 缩小阶段：输入、前向、反向、更新、通信、恢复；
4. 缩小范围：层、rank、shape bucket、数据源、样本 ID；
5. 最小复现：固定 seed、固定 batch、固定 shape，缩小到可重复触发；
6. 决定修复：改数据、改实现、改精度策略、改 schedule，而不是默认改超参；
7. 加回归：把触发样本、shape 和断言写入测试或 playbook。

“先缩圈”比“先调参”更重要。能被最小复现的问题才容易变成团队资产。

## 监控与 Playbook

建议至少记录：

1. loss、grad norm、参数范数；
2. grad scaler、FP8 amax/scale、溢出计数；
3. 每层 activation 和 gradient 极值；
4. 每个 rank 的异常计数；
5. 数据源、长度桶和样本 ID 曝光；
6. checkpoint 写入、加载和恢复事件；
7. kernel、runtime、编译器和并行配置版本。

每次高价值故障都应沉淀为 playbook：

1. 症状是什么；
2. 触发条件是什么；
3. 根因是什么；
4. 如何监控提前发现；
5. 修复方式是什么；
6. 是否需要回归测试；
7. 是否影响已有实验结论。

稳定性能力不是找一个万能超参，而是把高代价故障拆成可观察、可缩圈、可回归的问题。模型越大、训练越长、分布式栈越复杂，这套 triage 能力就越像训练系统的安全壳。
