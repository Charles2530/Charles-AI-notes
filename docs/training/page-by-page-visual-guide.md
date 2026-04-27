# 训练专题逐页方法图解（实用主义版）

这页不追求“讲故事”，重点是：  
把训练专题每一页压缩成可执行的方法框架、判断顺序和落地清单。

更推荐的读法是：先看每页的核心图，理解“问题从哪里来、应该先查什么、最后怎么验收”；再看后面的白话解释和例子，把抽象概念落到真实项目里。

每节固定结构：

1. 页面核心方法（它到底解决什么问题）；
2. 最小落地步骤（从哪里开始做）；
3. 常见失败信号（怎么快速发现问题）；
4. 概括图（流程图、对照图、决策图）。
5. 白话解释与例子（怎么讲给新人听、怎么落到实际排查）。

## 0. 这些图应该怎么用

这里的图不是为了“好看”，而是为了减少训练项目里的沟通成本。建议按下面四类使用：

| 图的类型 | 适合回答的问题 | 用法 |
| --- | --- | --- |
| 流程图 | 训练链路该按什么顺序推进 | 放在项目启动会里统一流程 |
| 决策图 | 什么时候该继续、暂停、回滚或扩规模 | 放在实验评审和发布门禁里 |
| 排障图 | 发生异常后先查什么、后查什么 | 放在 on-call 手册和故障复盘里 |
| 指标图 | 哪些指标必须一起看，避免单指标误导 | 放在 dashboard 设计和实验报告里 |

一个实用原则是：**每张图都应该能转成一个检查清单**。如果一张图不能指导下一步动作，它就只是插画，不适合放在训练方法页里。

---

## 1. [训练总览](index.md)

**页面核心方法**  
训练要按“生产线”管理，而不是按“单实验”管理。  
核心框架是：`数据 -> 训练 -> 评测 -> 发布 -> 回流`。

**最小落地步骤**

1. 先定义成功标准（质量、时延、成本、稳定性）；
2. 再定义资产标准（checkpoint、数据版本、评测报告）；
3. 最后定义回流标准（线上失败样本如何进入下一轮训练）。

**常见失败信号**

1. loss 在降，但线上问题不降；
2. 训练吞吐提升，但有效 token 利用率不提升；
3. 发布后无法复现实验结论。

**实用补充：发布门禁顺序**

1. 先过数据门，再过模型门；
2. 再过系统可恢复门；
3. 最后才做在线灰度门禁。

![训练生产线总览图](../assets/images/training/generated/training-production-line.png){ width="920" }
![训练问题分层排查栈](../assets/images/training/generated/training-layer-triage-stack.svg){ width="920" }
![数据质量到行为映射图](../assets/images/training/generated/data-quality-to-model-behavior-map.png){ width="920" }
![训练发布门禁图](../assets/images/training/generated/training-release-gate.png){ width="920" }
![训练总览知识卡片](../assets/images/training/generated/training-card-01-overview.png){ width="920" }
![大模型训练排障作战图](../assets/images/training/generated/training-triage-war-room-poster.png){ width="920" }

---

## 2. [预训练、微调与对齐](pretraining-finetuning-alignment.md)

**页面核心方法**  
把“能力问题、行为问题、边界问题”分阶段处理：

1. 预训练管能力上限；
2. SFT 管任务可用性；
3. 对齐管风险边界与偏好一致性。

**最小落地步骤**

1. 先写清楚每阶段目标和不可替代性；
2. 把评测指标按阶段绑定，不混评；
3. 让阶段输出成为下阶段可复用输入资产。

**常见失败信号**

1. SFT 数据越来越多但泛化不升；
2. 对齐后安全提升但任务完成率明显掉；
3. 阶段交接时评测口径变化导致“伪提升”。

**实用补充：阶段指标映射**

1. 预训练重点看能力类指标；
2. SFT 重点看任务完成类指标；
3. 对齐重点看风险边界与拒答精度。

![预训练-SFT-对齐三阶段框架](../assets/images/training/generated/pretrain-sft-alignment.png){ width="920" }
![能力结构示意图](../assets/images/training/neural-network.svg){ width="860" }
![三阶段训练指标路由图](../assets/images/training/generated/stage-metric-routing-map.svg){ width="920" }
![三阶段指标映射图](../assets/images/training/generated/pretrain-sft-alignment-metric-map.png){ width="920" }
![预训练 SFT 对齐知识卡片](../assets/images/training/generated/training-card-02-stages.png){ width="920" }

---

## 3. [MTP 与投机解码](mtp-and-speculative-decoding.md)

**页面核心方法**  
把训练优化和推理优化拆开决策：

1. `MTP` 解决训练目标与模型结构问题；
2. `speculative` 解决推理时延与吞吐问题；
3. 两者要通过 acceptance 和系统开销统一验收。

**最小落地步骤**

1. 离线先验证主质量不退化；
2. 半在线验证 acceptance 分桶稳定；
3. 灰度验证 p95/p99 和熔断回退行为。

**常见失败信号**

1. 平均 acceptance 高但 p99 变差；
2. 只有低温短输出有收益；
3. 验证路径开销吞掉加速收益。

**实用补充：启用策略**

1. 先只在长输出桶开启；
2. 监控 acceptance 与 p99 联动；
3. 设定自动回退阈值。

![MTP 与 speculative 对照图](../assets/images/training/generated/mtp-vs-speculative.png){ width="920" }
![Speculative 启用决策树](../assets/images/training/generated/speculative-enable-decision-tree.png){ width="920" }
![MTP 投机解码上线看板](../assets/images/training/generated/acceptance-latency-dashboard.svg){ width="920" }
![MTP 与投机解码知识卡片](../assets/images/training/generated/training-card-03-mtp.png){ width="920" }

---

## 3.5 [低比特训练与训练数制](low-bit-training-and-numerics.md)

**页面核心方法**  
把低比特训练从“部署量化”里拆出来：训练时要同时管理权重、激活、梯度、优化器状态、scale、kernel 和恢复语义。

**最小落地步骤**

1. 先建立 BF16 baseline；
2. 再只打开主干 GEMM 的 FP8；
3. 接着评估 activation cache、optimizer states 和梯度低精度；
4. 最后才尝试 FP4、MXFP4、NVFP4 等更激进路线。

**常见失败信号**

1. 短跑 loss 正常，长跑后期突然 loss spike；
2. activation outlier 放大，scale 饱和率变高；
3. kernel 频繁 cast 回 BF16，端到端吞吐没有提升；
4. checkpoint 恢复后低精度 scale 或 optimizer 状态不一致。

**实用补充：低比特训练不要只看 bit 数**

1. `FP8` 更像稳妥入口；
2. `FP4` 更依赖硬件、scale 和无偏梯度估计；
3. `MXFP / NVFP` 的关键是块级 scale 和 kernel 是否匹配；
4. 真正的验收要同时看稳定性、吞吐、显存和下游 benchmark。

![低比特 LLM 量化地图](../assets/images/quantization/generated/low-bit-llm-quantization-map.png){ width="920" }
![训练稳定性排查树](../assets/images/training/generated/stability-triage-tree.png){ width="920" }
![训练集群 SLO 看板](../assets/images/training/generated/training-cluster-slo-board.png){ width="920" }

---

## 4. [Megatron-LM、DeepSpeed 与训练系统栈](megatron-lm-deepspeed-and-open-training-stacks.md)

**页面核心方法**  
先定主约束，再定并行组合：

1. 显存约束优先看 TP/ZeRO/FSDP；
2. 吞吐约束优先看 PP、通信重叠、kernel；
3. 长上下文约束优先看 CP/SP 和拓扑映射。

**最小落地步骤**

1. 建立显存账本、通信账本、吞吐账本；
2. 给每个并行维度绑定物理拓扑；
3. 先做稳定配置，再逐步增加复杂度。

**常见失败信号**

1. 并行维度理论正确但实测吞吐低；
2. 强依赖单一框架特性导致迁移困难；
3. checkpoint 在框架切换时不可恢复。

**实用补充：并行选型顺序**

1. 先判主瓶颈是显存还是通信；
2. 再确定 TP/PP/DP/CP 组合；
3. 最后把逻辑网格映射到物理拓扑。

![并行维度方法图](../assets/images/training/generated/parallelism-map-dp-tp-pp-cp.png){ width="920" }
![并行收益上限示意（Amdahl）](../assets/images/training/amdahls-law.svg){ width="860" }
![并行选型决策树](../assets/images/training/generated/parallelism-selection-playbook-tree.png){ width="920" }
![并行策略排障阶梯](../assets/images/training/generated/parallelism-debug-ladder.svg){ width="920" }
![并行训练系统栈知识卡片](../assets/images/training/generated/training-card-04-parallelism.png){ width="920" }

---

## 5. [训练集群运维、实验管理与成本治理](cluster-operations-and-experiment-management.md)

**页面核心方法**  
把训练平台当“运营系统”：

1. 调度策略；
2. 运行可观测；
3. 事故响应；
4. 复盘与成本治理。

**最小落地步骤**

1. 定义集群 SLO（作业成功率、启动时延、恢复时延）；
2. 定义实验证据链（配置、数据、代码、指标）；
3. 定义成本口径（每有效 token 成本、失败作业占比）。

**常见失败信号**

1. 告警很多但无优先级；
2. Postmortem 做了但无动作闭环；
3. 月度成本可见，日内成本不可控。

**实用补充：SLO 看板最小集**

1. 作业成功率；
2. 排队时延；
3. 恢复时延；
4. 每有效 token 成本。

![训练运维闭环图](../assets/images/training/generated/cluster-operations-loop.png){ width="780" }
![实验经济学成本瀑布图](../assets/images/training/generated/experiment-economics-cost-waterfall.png){ width="920" }
![训练集群SLO看板](../assets/images/training/generated/training-cluster-slo-board.png){ width="920" }
![PID 闭环直觉图](../assets/images/training/pid-loop.svg){ width="820" }
![P 控制图：过程漂移监控](../assets/images/training/p-control-chart.svg){ width="860" }
![Gantt 图：训练计划关键路径](../assets/images/training/gantt-diagram.svg){ width="860" }
![集群运维与成本治理知识卡片](../assets/images/training/generated/training-card-05-ops.png){ width="920" }

---

## 6. [输入管线、Packing 与吞吐治理](input-pipelines-packing-and-throughput-governance.md)

**页面核心方法**  
吞吐治理先做“主路径分解”：

1. 读取；
2. 解码/tokenize；
3. packing/bucketing；
4. batch 组装；
5. GPU 消费。

**最小落地步骤**

1. 给每段路径加时间和队列指标；
2. 先优化最慢段，再优化显著浪费段；
3. 在业务真实长度分布下复测。

**常见失败信号**

1. GPU 利用率低但 CPU 已满；
2. padding 率高导致有效 token/s 低；
3. 小 batch 有收益，大 batch 反而退化。

**实用补充：优化执行顺序**

1. 先测量，再调 workers/prefetch；
2. 再做分桶与 packing；
3. 最后再看主机到设备传输重叠。

![输入管线瓶颈流程图](../assets/images/training/generated/input-pipeline-bottlenecks.png){ width="920" }
![Packing 前后对照图](../assets/images/training/generated/packing-before-vs-after.png){ width="920" }
![Dataloader 优化顺序图](../assets/images/training/generated/dataloader-optimization-order.png){ width="820" }
![输入管线指标地图](../assets/images/training/generated/input-metric-map.svg){ width="920" }
![输入管线与 Packing 知识卡片](../assets/images/training/generated/training-card-06-input.png){ width="920" }

---

## 7. [训练稳定性、数值异常与故障排查](stability-numerics-and-failure-triage.md)

**页面核心方法**  
按“症状 -> 分类 -> 验证 -> 处置”排障，不直接重训。

**最小落地步骤**

1. 先判是数据、数值、优化、通信还是硬件；
2. 用最小复现实验缩小范围；
3. 固化排障脚本和阈值，减少人工判断波动。

**常见失败信号**

1. loss 突刺伴随 grad overflow；
2. 不同机器同配置结果漂移；
3. 恢复后训练曲线形态明显变形。

**实用补充：信号到动作映射**

1. 每个异常信号都要有“第一检查项”；
2. 每个检查项都要有“立即动作”；
3. 每个动作都要有“长期修复项”。

![稳定性排障决策树](../assets/images/training/generated/stability-triage-tree.png){ width="920" }
![分布变化直觉图](../assets/images/training/normal-distribution.svg){ width="860" }
![稳定性信号-动作矩阵](../assets/images/training/generated/stability-signal-action-matrix.png){ width="920" }
![数值异常前 30 分钟排障图](../assets/images/training/generated/numerics-debug-ladder.svg){ width="920" }
![P 控制图：训练指标漂移监控](../assets/images/training/p-control-chart.svg){ width="860" }
![稳定性与数值异常知识卡片](../assets/images/training/generated/training-card-07-stability.png){ width="920" }

---

## 8. [数据系统与优化](data-systems-and-optimization.md)

**页面核心方法**  
“优化器 + 数据系统”联调，而不是单调优化器。

**最小落地步骤**

1. 固定优化器后先调数据混合和采样；
2. 再调学习率、权重衰减、warmup；
3. 通过误差桶回流来修数据而不是盲目加总量。

**常见失败信号**

1. 换优化器收益不稳定；
2. 数据分布轻微变化导致模型行为大幅波动；
3. 某些任务桶长期无改进。

**实用补充：数据策略影响矩阵**

1. 每次只改一个策略变量；
2. 同时看收敛、鲁棒、幻觉风险、成本；
3. 用矩阵比“经验印象”更可靠。

![数据引擎闭环图](../assets/images/training/generated/data-engine-collect-filter-label-loop.png){ width="860" }
![数据策略-训练结果矩阵](../assets/images/training/generated/data-strategy-training-outcome-matrix.png){ width="920" }
![MapReduce 数据流直觉图](../assets/images/training/mapreduce.svg){ width="820" }
![Pareto 图：优先处理最大问题](../assets/images/training/pareto-chart.svg){ width="860" }
![数据系统与优化知识卡片](../assets/images/training/generated/training-card-08-data-opt.png){ width="920" }

---

## 9. [Scaling、课程学习与数据混合](scaling-curriculum-and-data-mixture.md)

**页面核心方法**  
把数据混合看成“时间函数”，不是常量。

**最小落地步骤**

1. 分阶段定义课程目标（基础、技能、领域、对齐）；
2. 每阶段设混合比上限和下限；
3. 用分桶评测决定阶段切换时机。

**常见失败信号**

1. 早期高难数据占比过大导致收敛慢；
2. 后期基础数据过多导致专业能力不上升；
3. 阶段切换后出现遗忘。

**实用补充：阶段切换门禁**

1. 指标平台期是否达到阈值；
2. 错误桶是否下降到目标区间；
3. 稳定性检查是否通过。

![课程学习与混合策略图](../assets/images/training/generated/curriculum-data-mixture-strategy.png){ width="920" }
![分布结构直觉图](../assets/images/training/histogram-example.svg){ width="860" }
![课程阶段切换门禁图](../assets/images/training/generated/curriculum-phase-switch-criteria-gates.png){ width="920" }
![Mixture Governor 数据配方闭环](../assets/images/training/generated/mixture-governor-loop.svg){ width="920" }
![Scaling 课程学习混合知识卡片](../assets/images/training/generated/training-card-09-mixture.png){ width="920" }

---

## 10. [后训练数据引擎与 Judge 模型](post-training-data-engines-and-judge-models.md)

**页面核心方法**  
后训练要同时管理两条链：

1. 数据回流链（收集-标注-训练）；
2. judge 可信链（校准-抽检-漂移监控）。

**最小落地步骤**

1. 建立固定 holdout 集做 judge 校准；
2. 每轮训练后比对 human/judge 一致性；
3. 对高分低质样本建立反例池。

**常见失败信号**

1. judge 分数上升但用户满意度不升；
2. 某类任务 judge 系统性偏高或偏低；
3. judge 模型更新后历史结论不可比。

![后训练数据引擎系统图](../assets/images/training/generated/post-training-data-engine-loop.png){ width="780" }
![Judge 校准图](../assets/images/training/generated/judge-model-calibration.png){ width="920" }
![Judge 与人审校准闭环](../assets/images/training/generated/judge-human-calibration-loop.svg){ width="920" }
![后训练数据引擎与 Judge 知识卡片](../assets/images/training/generated/training-card-10-judge.png){ width="920" }

---

## 11. [偏好数据与对齐失效模式](preference-data-and-alignment-pitfalls.md)

**页面核心方法**  
偏好学习要围绕“抗投机”设计数据和评测。

**最小落地步骤**

1. 建立偏好标注 rubric；
2. 用对抗样本测试奖励投机；
3. 对“高偏好分、低任务完成”样本做专项回流。

**常见失败信号**

1. 过度拒答；
2. 风格模板化；
3. 过度迎合 judge。

![偏好数据流水线图](../assets/images/training/generated/preference-data-pipeline.png){ width="920" }
![偏好对齐失效模式图](../assets/images/training/generated/preference-alignment-failure-modes.png){ width="920" }
![偏好标注 Rubric 评分卡](../assets/images/training/generated/preference-rubric-scorecard.svg){ width="920" }
![偏好数据与对齐失效知识卡片](../assets/images/training/generated/training-card-11-preference.png){ width="920" }

---

## 12. [分布式训练与 Checkpoint](distributed-training-and-checkpointing.md)

**页面核心方法**  
checkpoint 设计必须先定义“恢复语义”，再定义“保存性能”。

**最小落地步骤**

1. 定义 checkpoint manifest（版本、分片、拓扑、状态）；
2. 定义恢复前一致性检查；
3. 定义 world size 变化时的重映射规则。

**常见失败信号**

1. 能保存但不能跨规模恢复；
2. 恢复后 loss 轨迹断层；
3. 对象存储异步写入产生部分可见状态。

![Checkpoint 生命周期图](../assets/images/training/generated/checkpoint-lifecycle-distributed.png){ width="920" }
![恢复一致性检查流程图](../assets/images/training/generated/distributed-resume-consistency-check.png){ width="920" }
![Checkpoint Manifest 必备字段](../assets/images/training/generated/checkpoint-contract-fields.svg){ width="920" }
![分布式训练与 Checkpoint 知识卡片](../assets/images/training/generated/training-card-12-checkpoint.png){ width="920" }

---

## 13. [数据质量、去重与治理](data-quality-dedup-and-governance.md)

**页面核心方法**  
治理要按层做，不做“一刀切清洗”。

**最小落地步骤**

1. 拆层治理：精确去重、近重复、语义重复；
2. 给每层治理定义误杀率与漏检率；
3. 对高价值领域设置“保护名单”避免过清洗。

**常见失败信号**

1. 去重后覆盖面下降；
2. 误删专业术语密集文本；
3. 数据洁净度提升但任务表现下降。

![数据治理层级图](../assets/images/training/generated/data-quality-governance-layers.png){ width="920" }
![去重方法对照图](../assets/images/training/generated/dedup-strategy-exact-near-semantic.png){ width="920" }
![数据治理优先级漏斗](../assets/images/training/generated/data-governance-priority-funnel.svg){ width="920" }
![Pareto 图：数据问题优先级](../assets/images/training/pareto-chart.svg){ width="860" }
![数据质量去重治理知识卡片](../assets/images/training/generated/training-card-13-quality.png){ width="920" }

---

## 14. [目标函数、优化器与学习率日程](objectives-optimizers-and-schedules.md)

**页面核心方法**  
把目标函数、优化器、LR 日程看成一个联动系统。

**最小落地步骤**

1. 明确主目标与辅助目标的权重关系；
2. 先选稳定优化器，再匹配 LR 日程；
3. 用短实验验证稳定性，用长实验验证泛化。

**常见失败信号**

1. warmup 不足导致初期震荡；
2. 后期学习率衰减过慢导致收敛拖尾；
3. 训练分数提升但泛化下降。

**实用补充：调参顺序**

1. 先 warmup 与 peak LR；
2. 再权重衰减和 beta；
3. 最后做裁剪与长跑验证。

![优化器与LR联动图](../assets/images/training/generated/optimizer-lr-schedule-interaction.png){ width="920" }
![梯度下降直觉图](../assets/images/training/gradient-descent.svg){ width="860" }
![优化器调参顺序图](../assets/images/training/generated/optimizer-tuning-order-flowchart.png){ width="920" }
![优化器症状到动作映射](../assets/images/training/generated/optimizer-symptom-action-map.svg){ width="920" }
![目标函数优化器学习率知识卡片](../assets/images/training/generated/training-card-14-optimizer.png){ width="920" }

---

## 15. [评测与消融方法学](evaluation-and-ablation-methodology.md)

**页面核心方法**  
消融最重要的是“可归因”，不是“表格多”。

**最小落地步骤**

1. 建 baseline 并锁定口径；
2. 一次只改一个关键变量；
3. 报告中同时写均值、方差、显著性和效应量；
4. 离线评测后再做在线 A/B。

**常见失败信号**

1. 多变量同改导致无法归因；
2. 平均分提升但高风险桶退化；
3. 在线结果与离线结果方向相反。

**实用补充：评测契约与复现清单**

1. 离线与在线指标口径一致；
2. 桶划分规则一致；
3. 置信区间与回滚阈值一致。

![评测与消融漏斗图](../assets/images/training/generated/evaluation-ablation-funnel.png){ width="920" }
![单变量消融矩阵图](../assets/images/training/generated/ablation-one-variable-matrix.png){ width="920" }
![PR 曲线示意图](../assets/images/training/precision-recall.svg){ width="860" }
![箱线图与分布直觉图](../assets/images/training/boxplot-vs-pdf.svg){ width="860" }
![评测风险分桶地图](../assets/images/training/generated/evaluation-risk-bucket-map.svg){ width="920" }
![离线在线评测契约图](../assets/images/training/generated/offline-online-evaluation-contract.png){ width="920" }
![评测可复现检查清单](../assets/images/training/generated/evaluation-reproducibility-checklist.png){ width="920" }
![混淆矩阵示意图](../assets/images/training/confusion-matrix.png){ width="720" }
![ROC 曲线示意图](../assets/images/training/roc-curve.svg){ width="860" }
![评测与消融知识卡片](../assets/images/training/generated/training-card-15-eval.png){ width="920" }

---

## 16. [Scaling Law 与实验经济学](scaling-laws-and-experiment-economics.md)

**页面核心方法**  
做规模决策时要同时看三条曲线：

1. 质量收益曲线；
2. 训练成本曲线；
3. 迭代速度曲线。

**最小落地步骤**

1. 先做小规模外推，别直接满配；
2. 把“失败实验成本”计入总账；
3. 用“每有效 token 成本”替代“每 GPU 小时成本”做主指标。

**常见失败信号**

1. 大模型收益进入递减区还在盲目扩；
2. 推理成本债被忽略；
3. 过度追单次最好成绩，牺牲实验并发和节奏。

![Scaling 与经济学前沿图](../assets/images/training/generated/scaling-law-vs-economics.png){ width="920" }
![实验成本瀑布图](../assets/images/training/generated/experiment-economics-cost-waterfall.png){ width="920" }
![Scaling 实验 Stop Go 看板](../assets/images/training/generated/scaling-stop-go-board.svg){ width="920" }
![Gantt 图：扩规模实验排期](../assets/images/training/gantt-diagram.svg){ width="860" }
![Bias-Variance 权衡示意图](../assets/images/training/variance-bias.svg){ width="860" }
![Scaling Law 与实验经济学知识卡片](../assets/images/training/generated/training-card-16-economics.png){ width="920" }

---

## 17. 逐页详细展开：白话解释、例子和落地动作

这一节把前面的“方法框架”再展开一层。  
如果你只想快速建立直觉，可以只读每页的“白话解释”和“例子”；如果你要把这些知识用到项目里，重点看“落地动作”和“验收信号”。

### 17.1 训练总览：先把训练当成生产线，而不是一次实验

**白话解释**  
训练不是“我有一批数据、跑一个脚本、得到一个模型”这么简单。更准确的理解是：训练是一条长期运转的生产线，输入是数据、配置、代码和算力，输出是 checkpoint、评测报告、线上反馈和下一轮改进方向。只要其中任意一环没有版本化，模型变好或变坏都很难解释清楚。

**生动例子**  
把训练想成开餐厅。预训练数据像原材料，训练脚本像厨房流程，评测像试吃员，线上反馈像真实顾客评价。如果顾客说菜太咸，你不能只怪厨师，也要查盐的批次、菜单设计、试吃标准和出餐流程。训练也是一样，线上效果不好时，不应该只说“模型不行”，而要回看数据、训练、评测和发布链路。

**落地动作**  
先做一张训练资产表，至少记录数据版本、代码 commit、配置 hash、checkpoint 路径、评测集版本、线上灰度范围和回滚方案。只要这张表缺一项，这次训练就还不能算成可复现资产。

**验收信号**  
一条训练线真正成熟的标志不是 loss 很低，而是出问题时能快速回答三个问题：这次和上次到底差在哪里，差异是否能复现，是否能安全回滚。

### 17.2 预训练、SFT 与对齐：能力、行为和边界不要混着修

**白话解释**  
预训练像“广泛读书”，SFT 像“照着范例练习答题”，对齐像“学习什么回答更合适、更安全、更符合偏好”。三者解决的问题不同，所以不能拿一个阶段硬补另一个阶段的缺口。底座没学会的知识，靠 SFT 很难凭空补出来；模型会做但不按格式输出，通常也不该回到预训练阶段大改。

**生动例子**  
训练代码助手时，如果模型根本不知道某个框架的 API，那是能力问题，应该看预训练数据覆盖或领域继续训练。如果模型知道 API，但总是不按公司代码规范写，那是 SFT 或后训练数据问题。如果模型遇到内部密钥处理场景时过度冒险，那是对齐和安全边界问题。

**落地动作**  
把 bug 单按三类打标签：能力缺口、任务执行缺口、偏好边界缺口。每类 bug 对应不同训练动作，避免把所有失败样本都扔进同一个 SFT 数据池。

**验收信号**  
预训练看迁移能力和基础任务，SFT 看任务完成率和格式稳定性，对齐看风险边界、拒答精度和偏好一致性。三个阶段如果共用一个总分，很容易把问题藏起来。

### 17.3 MTP 与投机解码：训练侧让模型会“多想几步”，推理侧让系统少等几步

**白话解释**  
`MTP` 是训练目标或模型结构，要求模型在当前位置预测多个未来 token；投机解码是推理算法，用便宜路径先草拟，再让目标模型验证。二者能配合，但不是同义词。做了 `MTP` 不等于线上一定变快，没有 `MTP` 也可以用独立 draft model 做投机解码。

**生动例子**  
可以把它想成写邮件。`MTP` 像训练一个人不仅会写下一个字，还能提前想好后面半句话；投机解码像让助理先写草稿，再由负责人快速批改通过。助理写得越接近负责人风格，通过率越高，整体越快；但如果批改成本太高，反而不省时间。

**落地动作**  
不要只看平均 speedup，要按任务类型、输出长度、采样温度和 batch size 切桶看 acceptance rate、平均接受 token 数和 p95/p99 延迟。尤其要确认收益是不是只出现在短回答或低温采样里。

**验收信号**  
好的投机方案必须同时满足主模型质量不退、长输出桶有稳定收益、验证开销没有吞掉收益、异常时可以自动回退。

### 17.4 Megatron-LM、DeepSpeed 与训练系统栈：并行策略先服务瓶颈，不是越复杂越好

**白话解释**  
大模型训练里，单卡装不下、单机跑不快、单条序列太长都会逼你做分布式并行。`TP` 拆算子，`PP` 拆层，`DP` 拆数据，`CP/SP` 拆序列或上下文，`ZeRO/FSDP` 拆优化器和参数状态。问题是每拆一次，通信、调度和 checkpoint 都会变复杂。

**生动例子**  
把训练看成搬家。数据并行像多支搬家队各搬一批箱子；张量并行像几个人一起抬一张巨大的桌子；流水并行像楼上、楼下、车厢分工传递。分工能提升效率，但如果楼道太窄或指挥混乱，人越多越堵。

**落地动作**  
先写三本账：显存账、通信账、吞吐账。显存不够时优先看 ZeRO/FSDP/activation checkpointing，通信拖慢时看 TP/PP 组合和拓扑映射，长上下文出问题时看 CP/SP 与 attention kernel。

**验收信号**  
系统栈选型合格的标准不是“用了最先进框架”，而是能在目标规模下稳定跑、恢复一致、吞吐可解释、换机器或扩规模时行为不突然变形。

### 17.5 训练集群运维、实验管理与成本治理：GPU 利用率只是表层指标

**白话解释**  
训练集群不是一堆 GPU，而是调度、存储、网络、监控、故障恢复和成本核算组成的运营系统。只看 GPU 利用率会误导决策，因为利用率高不代表有效 token 多，也不代表实验有价值。

**生动例子**  
一家工厂机器满负荷运转，但如果一半产品返工、一半订单做错规格，机器再忙也没有意义。训练平台也是一样，作业跑得满不等于产出有效模型。如果大量作业最后因为数据版本错、checkpoint 不可恢复或评测口径变了而废掉，真实效率很低。

**落地动作**  
把平台 SLO 定成面向训练资产的指标：作业启动时延、作业成功率、恢复时延、有效 token/s、失败作业成本占比、checkpoint 可恢复率。再把这些指标按队列、项目和硬件池切开看。

**验收信号**  
成熟平台应该能做到：小故障自动止血，大故障有 postmortem，成本异常能日内发现，关键实验能追溯完整证据链。

### 17.6 输入管线、Packing 与吞吐治理：GPU 没吃饱，常常不是 GPU 的错

**白话解释**  
训练吞吐的主路径不是从 GPU 开始，而是从数据读取开始。样本要经过读取、解压、解析、tokenize、分桶、packing、batch 组装、传输到设备，最后才被 GPU 消费。任何一段慢，GPU 都会等。

**生动例子**  
把 GPU 想成厨房大厨，dataloader 是备菜团队。大厨站着等菜，不代表大厨能力差，可能是采购慢、切菜慢、配菜规格混乱，或者每盘菜空隙太多。`packing` 就像把零散小菜合理拼盘，减少空盘浪费，但拼错会串味。

**落地动作**  
先量化每一段耗时和队列深度，再看 padding 率、有效 token/s、CPU 利用率、host-to-device 传输时间和 batch 形状分布。不要在没测量的情况下盲目加 worker 或改 tokenizer。

**验收信号**  
输入系统做对后，吞吐提升应该同时反映在 GPU 利用率、有效 token/s 和稳定性上。如果只有样本数提升但有效 token 没提升，往往只是把浪费跑得更快。

### 17.7 训练稳定性、数值异常与故障排查：先缩圈，再修复

**白话解释**  
训练不稳定不是一个问题，而是一组问题：数据异常、学习率过激、低精度溢出、通信错乱、硬件故障、checkpoint 恢复不完整都可能表现成 loss spike、NaN 或效果漂移。排障第一步不是改超参，而是分类。

**生动例子**  
医生看病不会一上来就开最猛的药，而是先问症状、做检查、排除高风险原因。训练排障也一样。loss 突刺像发烧，可能是吃坏东西，也可能是感染，也可能是仪器误差。你需要先定位是哪一类。

**落地动作**  
建立最小排障路径：先看数据 batch 和异常样本，再看 grad norm、overflow、学习率和 loss scale，再看 rank-local 指标、通信错误和硬件事件，最后做最小复现实验。

**验收信号**  
一次排障完成后，不应只留下“这次修好了”，还应沉淀触发条件、检测脚本、回滚动作和长期修复项。否则下一次还是靠人肉经验。

### 17.8 数据系统与优化：优化器不是孤立旋钮，数据调度同样在优化

**白话解释**  
训练表面上是在优化模型参数，实际上也在优化数据进入模型的顺序、比例和质量。学习率、batch size、采样策略、去重、过滤、课程阶段和难例回流都会改变梯度统计，所以数据系统和优化器必须一起看。

**生动例子**  
健身计划里，优化器像训练动作和重量，数据混合像每天吃什么和练什么部位。动作标准但饮食混乱，效果不会稳定；饮食很好但训练强度安排不合理，也会卡住。训练模型同理，不能只调 AdamW 而忽视数据配方。

**落地动作**  
每次改动只动一个主要变量，并同时记录收敛速度、鲁棒性、幻觉率、长尾桶表现和成本。对于数据策略，特别要记录样本来源、质量分、采样权重和回流原因。

**验收信号**  
一个数据优化动作有价值，应该能在特定错误桶或能力桶上产生可解释收益，而不是只让总体 loss 略微下降。

### 17.9 Scaling、课程学习与数据混合：数据配方是会随时间变化的预算表

**白话解释**  
Scaling 不是简单地把模型变大、数据变多。实际训练里，最关键的是在不同阶段如何分配 token 预算。早期可能需要广覆盖、低噪声数据打基础，中期需要技能和领域数据塑造能力，后期需要高价值难例、长上下文或对齐数据补短板。

**生动例子**  
培养一个新人不是第一天就给最难项目。先学基础工具，再做标准任务，然后接触复杂案例，最后处理线上事故。课程学习也是这个意思，但它不是机械“先简单后困难”，而是让数据难度、长度、领域和噪声与模型当前能力匹配。

**落地动作**  
把 mixture 写成版本化配方，而不是散落在脚本里的采样权重。每个阶段要写清楚目标、数据比例、切换门槛、保护桶和回滚条件。

**验收信号**  
课程策略成功时，分桶指标会按预期接力提升；失败时常见现象是领域挤占、阶段切换后遗忘、难例提升但基础能力下降。

### 17.10 后训练数据引擎与 Judge 模型：judge 是仪表，不是上帝

**白话解释**  
后训练越来越像数据飞轮：收集样本、生成候选、judge 评分、人审校准、训练、评测、线上回流。Judge 模型能放大标注效率，但它本身也会偏、会漂移、会被模型投机，所以必须被校准和监控。

**生动例子**  
Judge 像考试阅卷老师。老师能提高批改效率，但如果老师偏爱长答案，学生就会学会写很多废话；如果老师只看格式，学生就会把格式做漂亮但事实答错。后训练中模型也会学习 judge 的偏好漏洞。

**落地动作**  
至少建立三套集合：judge 校准 holdout、人审抽检池、高分低质反例池。每轮模型更新后，要看 human/judge 一致性是否变化，而不是只看 judge 平均分。

**验收信号**  
合格的后训练飞轮应当能回答：哪些样本被选中、为什么被选中、judge 是否可靠、训练后哪些错误桶下降、是否引入新的过拟合或模板化问题。

### 17.11 偏好数据与对齐失效模式：偏好学习最怕学会讨好评分器

**白话解释**  
偏好数据不是简单告诉模型“哪个回答更好”，而是在塑造模型面对多个合理选项时的排序倾向。这个过程很容易把模型推向过度保守、模板化、冗长、迎合 judge 或牺牲事实性的方向。

**生动例子**  
如果客服机器人总被奖励“语气非常礼貌”，它可能学会每次先写一大段道歉和感谢，却没有真正解决问题。如果安全 judge 过度保守，模型可能遇到普通技术问题也拒答。这些都是偏好信号偏了。

**落地动作**  
偏好标注 rubric 必须拆维度：事实正确性、任务完成度、简洁性、风险边界、格式合规、用户体验。不要只给一个笼统“更好”的标签。

**验收信号**  
偏好对齐做对后，应该是坏行为减少、任务完成率不明显下降、拒答边界更精准。如果安全分涨了但可用性大幅下降，就不是合格收益。

### 17.12 分布式训练与 Checkpoint：保存成功不等于能恢复训练

**白话解释**  
分布式 checkpoint 保存的不只是权重，还包括优化器状态、学习率状态、数据 sampler 状态、随机数状态、并行拓扑、分片元数据和版本信息。少任何一项，恢复后都可能看似继续训练，实际已经不是同一个实验。

**生动例子**  
玩游戏存档不只要保存角色等级，还要保存背包、任务进度、地图位置和世界状态。只保存角色等级，读档后看似能玩，但剧情和物品全乱了。训练 checkpoint 也是这样，只恢复权重远远不够。

**落地动作**  
为 checkpoint 写 manifest，明确每个文件分片、每个状态对象、保存时间、global step、world size、并行策略和数据位置。恢复前先做一致性检查，再启动训练。

**验收信号**  
恢复后 loss 曲线应连续，数据不重复不跳样，学习率和 batch 语义一致，world size 变化时重映射规则可解释。

### 17.13 数据质量、去重与治理：清洗不是越狠越好

**白话解释**  
数据治理的目标不是把数据变得“看起来很干净”，而是让模型看到高价值、低污染、覆盖合理、风险可控的数据。去重也不只是删重复文本，因为精确重复、近重复和语义重复影响不同，误删和漏删都有成本。

**生动例子**  
做代码模型时，很多项目都有相似模板文件。如果完全不去重，模型会浪费大量 token 学模板；如果去重太狠，又可能删掉不同框架下真正有价值的变体。治理的难点就是既减少浪费，又不破坏覆盖面。

**落地动作**  
把治理分层：精确去重处理完全相同样本，near-dup 处理复制粘贴和轻微改写，语义去重处理表达不同但内容高度重复的样本。每层都要估计误杀率和漏检率。

**验收信号**  
好的治理结果应当提升有效 token 质量，同时保持关键领域覆盖。只看去重率没有意义，必须看训练收益、下游桶表现和高价值样本保留率。

### 17.14 目标函数、优化器与学习率日程：训练不是只选一个 AdamW

**白话解释**  
目标函数决定模型往哪里学，优化器决定怎么走，学习率日程决定什么时候快走、什么时候慢走。三者一起决定训练稳定性和泛化。很多发散、平台期或后期退化问题，不是单个超参能解释的。

**生动例子**  
开车去目的地，目标函数是目的地，优化器是驾驶方式，学习率是油门。刚启动时油门太大容易冲出去，所以需要 warmup；接近目的地还猛踩油门，会来回震荡，所以需要 decay；路况变了还用同一套驾驶方式，也会出事。

**落地动作**  
调参顺序要保守：先确定目标和主质量不退，再用短跑找稳定 warmup 和 peak LR，再调 weight decay、beta、clip norm，最后用长跑验证泛化和尾部行为。

**验收信号**  
合格的 schedule 不只是让 loss 好看，还要让训练早期不炸、中期收敛有效、后期不过拟合，并且在恢复训练和 batch 变化时语义一致。

### 17.15 评测与消融方法学：评测不是打分，是建立可归因证据

**白话解释**  
评测的核心不是“分数高一点”，而是回答为什么高、在哪些桶高、代价是什么、能不能复现。消融的核心也不是表格多，而是一次只问一个清楚问题，让结论可归因。

**生动例子**  
新模型平均分涨了 2 分，但长文档问答掉了 8 分，安全拒答过多，线上 p99 变差。单看平均分会说“模型更强”，但产品上可能完全不可接受。评测必须像体检报告，而不是只给一个总分。

**落地动作**  
建立固定 baseline，锁定数据版本和评测口径。每个实验报告至少写：改了什么、没改什么、预算是否一致、均值和方差、分桶结果、失败样本、成本变化和是否可以上线。

**验收信号**  
高质量评测能支持决策：继续、回滚、分桶上线、补数据、重做消融。不能支持决策的分数，只是装饰。

### 17.16 Scaling Law 与实验经济学：模型变大之前，先算实验账

**白话解释**  
Scaling law 帮你估计规模扩大后质量可能怎么变，但实验经济学会追问：这个收益值不值、失败成本是多少、推理成本会不会变成债、团队迭代速度会不会下降。研究里“更大更强”不等于产品里“更值得”。

**生动例子**  
买车时不能只看最高速度，还要看油耗、维护成本、停车难度和使用场景。训练模型也一样，更大的模型可能 benchmark 更好，但如果训练周期太长、推理成本太高、线上收益有限，就未必是正确选择。

**落地动作**  
用小规模实验先拟合趋势，再决定是否上大规模。成本口径至少包括 GPU 小时、有效 token 成本、失败实验成本、评测成本、工程维护成本和未来推理成本。

**验收信号**  
值得扩规模的方向应该同时满足：小规模趋势稳定、关键桶持续改善、成本可接受、线上价值明确、失败后能复用中间资产。

---

## 18. 每页知识点精讲：概念、解释和例子

这一节更像“讲义”。前面的部分帮你建立框架，这里把每页容易混淆的知识点拆开讲清楚。  
读的时候建议先问自己：这个知识点解决的是能力问题、行为问题、系统问题，还是评测问题。

### 18.1 训练总览

| 知识点 | 通俗解释 | 生动例子 |
| --- | --- | --- |
| 训练生产线 | 训练不是一次性脚本，而是数据、模型、系统、评测和回流的长期循环。 | 像开餐厅，菜品质量不只取决于厨师，还取决于原料、流程、试吃和顾客反馈。 |
| 能力问题 | 模型底层不会，后训练很难凭空补回来。 | 学生没学过微积分，让他背几道标准答案也不能真正会解题。 |
| 行为问题 | 模型可能会，但不会按你要求的方式输出。 | 员工会做事，但不按公司模板写报告。 |
| 系统问题 | 训练是否稳定、可恢复、可复现，决定结论是否可信。 | 工厂生产记录缺失，产品合格也很难追责和复盘。 |

**容易误解的点**  
很多人把“模型效果不好”直接归因到模型结构，但训练项目里更常见的问题是数据版本漂移、评测口径不稳、输入管线浪费、checkpoint 恢复不完整。总览页的价值，就是先把问题放回生产线，而不是一上来找某个公式背锅。

**项目里怎么用**  
每次训练都应该产出一份最小实验档案：数据版本、采样配方、代码 commit、配置 hash、训练步数、checkpoint manifest、评测报告、失败样本摘要、是否可回滚。没有这份档案，训练结果就只是一次“跑出来了”，不是可复用资产。

### 18.2 预训练、SFT 与对齐

| 知识点 | 通俗解释 | 生动例子 |
| --- | --- | --- |
| 预训练 | 给模型建立世界知识、语言结构和迁移能力。 | 大量阅读各种书，形成知识底子。 |
| SFT | 让模型学会按任务格式和示范方式回答。 | 看标准答案，学习考试题怎么写步骤。 |
| 偏好对齐 | 让模型在多个可能答案里更倾向选人类满意、风险更低的答案。 | 两个回答都能用，但一个更专业、更稳妥、更少废话。 |
| 阶段评测 | 不同阶段要用不同指标，不要用一个总分糊住所有问题。 | 体检不能只看“健康总分”，还要看血压、血糖、心率。 |

**容易误解的点**  
SFT 不是万能补丁。底座完全不会的领域，SFT 只能让模型模仿少量答案，很容易变成“看起来会”。对齐也不是单纯让模型更礼貌，它是在改变模型的选择倾向，所以过强对齐可能压制能力、增加拒答或让回答模板化。

**项目里怎么用**  
拿到失败样本后先标注阶段归因：如果模型知识缺失，回看预训练或领域继续训练；如果格式错、工具协议错，回看 SFT；如果拒答边界、风格、风险判断异常，回看偏好数据和 judge。不要把所有失败样本混成一个后训练数据池。

### 18.3 MTP 与投机解码

| 知识点 | 通俗解释 | 生动例子 |
| --- | --- | --- |
| MTP | 训练时让模型在当前位置预测多个未来 token。 | 写句子时不只想下一个字，还提前想好后半句。 |
| Draft | 推理时先由便宜路径生成候选 token。 | 助理先写草稿。 |
| Verify | 目标模型并行验证 draft 是否可接受。 | 负责人快速批改草稿。 |
| Acceptance | 决定加速是否真实成立的关键指标。 | 草稿通过越多，负责人越省时间。 |

**容易误解的点**  
`MTP` 不等于投机解码。`MTP` 是训练结构和目标函数上的设计，投机解码是服务时的算法。没有 `MTP` 也可以用小模型做 draft；有了 `MTP` 也可能因为 acceptance 低、verify 开销大、KV 管理复杂而没有端到端收益。

**项目里怎么用**  
上线前至少做四组分桶：短输出和长输出、低温和高温、代码和开放问答、小 batch 和大 batch。只要某些关键桶 p99 变差，就不能只拿平均 speedup 宣称成功。

### 18.4 Megatron-LM、DeepSpeed 与训练系统栈

| 知识点 | 通俗解释 | 生动例子 |
| --- | --- | --- |
| 数据并行 | 每张卡处理不同样本，再同步梯度。 | 多个小组做同一套题，最后汇总答案。 |
| 张量并行 | 把一个大算子拆到多张卡上。 | 几个人一起抬一张巨大的桌子。 |
| 流水并行 | 把模型层切成多段，像流水线一样传递。 | 工厂里一道工序接一道工序。 |
| ZeRO / FSDP | 把参数、梯度、优化器状态分片，降低单卡显存压力。 | 大仓库太满，就把库存分到多个分仓。 |

**容易误解的点**  
并行维度越多，不代表系统越强。每增加一个并行维度，就会增加通信、调度、checkpoint 和故障恢复复杂度。小规模吞吐好，也不保证大规模好，因为跨机通信和拓扑映射会突然成为主瓶颈。

**项目里怎么用**  
先写显存账，再写通信账，最后写吞吐账。显存账回答“装不装得下”，通信账回答“扩不扩得动”，吞吐账回答“跑得值不值”。没有这三本账，选框架容易变成信仰之争。

### 18.5 集群运维、实验管理与成本治理

| 知识点 | 通俗解释 | 生动例子 |
| --- | --- | --- |
| Gang scheduling | 分布式训练需要一组资源同时到位。 | 乐队演出不能只到鼓手，其他人没到也开不了场。 |
| 拓扑感知调度 | 把通信频繁的 rank 放在更近的硬件上。 | 经常协作的人坐在同一间会议室。 |
| 实验证据链 | 配置、代码、数据、指标都能追溯。 | 财务报销必须有发票、审批和付款记录。 |
| 成本治理 | 看每有效 token 成本，而不只是 GPU 小时。 | 机器转了很久但产出废品，真实成本更高。 |

**容易误解的点**  
GPU 利用率高不等于训练有效。如果数据重复、padding 浪费、失败作业多、恢复后实验不可比，那么集群看起来很忙，实际产出很低。训练平台的目标不是让 GPU 永远满，而是让高价值实验稳定地产生可信结论。

**项目里怎么用**  
平台看板至少要有作业成功率、排队时延、恢复时延、有效 token/s、失败作业成本、checkpoint 可恢复率。成本异常要日内可见，而不是月底账单来了才发现。

### 18.6 输入管线、Packing 与吞吐治理

| 知识点 | 通俗解释 | 生动例子 |
| --- | --- | --- |
| Padding 浪费 | 为了对齐长度填入无效 token，计算却照样花钱。 | 一辆车只坐两个人也按整车油耗付费。 |
| Bucketing | 把长度相近的样本放在一起，减少 padding。 | 身高相近的人站一排，队伍更整齐。 |
| Packing | 把多个短样本拼进同一个序列，提高有效 token 占比。 | 把小件货物拼箱运输。 |
| Deterministic resume | 恢复训练后数据顺序要连续一致。 | 看书中断后要从正确页码继续，而不是重读或跳页。 |

**容易误解的点**  
Packing 不是免费收益。它可能破坏样本边界、影响 loss 统计、改变任务分布，尤其在多轮对话、多模态样本和工具轨迹里更容易出错。吞吐提升必须和质量回归一起看。

**项目里怎么用**  
排查顺序是：先看 GPU 等待时间，再看 dataloader 队列，再看 tokenize/解码耗时，再看 padding 率和 batch 形状，最后看 host-to-device 传输和 kernel overlap。

### 18.7 训练稳定性、数值异常与故障排查

| 知识点 | 通俗解释 | 生动例子 |
| --- | --- | --- |
| Loss spike | 损失突然尖峰，可能是数据、LR、低精度或通信问题。 | 体温突然升高，只是症状，不是病因。 |
| NaN / Inf | 数值溢出或非法计算，常见于低精度和异常梯度。 | 计算器显示错误，不代表题目本身无解。 |
| Rank-local 观测 | 分布式训练要看每个 rank，而不是只看平均值。 | 班级平均分正常，不代表某个学生没交卷。 |
| 故障回放 | 保留现场，让问题可复现。 | 交通事故要保存行车记录仪。 |

**容易误解的点**  
一看到 NaN 就降学习率，往往是低效排障。NaN 可能来自坏数据、某个 fused kernel、某个 rank 的硬件错误、恢复状态不完整，甚至 tokenizer 或 packing 边界错误。先分类比先调参更重要。

**项目里怎么用**  
稳定性排障按三段走：先止血，保存现场并暂停扩大影响；再定位，用最小复现实验缩小变量；后归因，把检测脚本、回滚动作和长期修复写进手册。

### 18.8 数据系统与优化

| 知识点 | 通俗解释 | 生动例子 |
| --- | --- | --- |
| 有效 token | 真正贡献训练信号的 token。 | 读 100 页书，其中 30 页是重复广告，实际学习只有 70 页。 |
| 数据混合比 | 不同来源数据的采样权重，本质上在改隐形目标函数。 | 饮食配比会影响训练效果。 |
| Loss-aware sampling | 根据损失或难度调整采样。 | 老师把学生错得多的题拿出来重点练。 |
| 数据回流 | 把失败样本转成下一轮训练资产。 | 客服把高频投诉整理成培训材料。 |

**容易误解的点**  
优化器变好了，不一定是优化器本身更优，可能只是更适配当前数据分布。数据质量、采样权重、batch 形状和有效 token 利用率都会影响梯度噪声，所以不能只比较优化器名称。

**项目里怎么用**  
每次数据策略改动要记录来源、过滤规则、质量分、采样权重、影响的任务桶和预期收益。不要只写“新增高质量数据”，这句话无法复现，也无法审计。

### 18.9 Scaling、课程学习与数据混合

| 知识点 | 通俗解释 | 生动例子 |
| --- | --- | --- |
| Scaling | 研究模型、数据、计算扩大后收益如何变化。 | 公司扩张时要看收入增长是否跟得上成本增长。 |
| Curriculum | 让数据顺序和难度随训练阶段变化。 | 新人先做基础任务，再处理复杂项目。 |
| Mixture | 数据配方，决定模型主要吸收哪些能力。 | 一个人每天读什么书，会塑造他的能力结构。 |
| Phase gate | 阶段切换门槛，防止凭感觉换配方。 | 考试合格后再进入下一门课。 |

**容易误解的点**  
课程学习不等于简单“先易后难”。有时要先短后长，有时先高质量后多样，有时先通用后领域，有时先文本后多模态。关键是让数据阶段和模型当前能力匹配。

**项目里怎么用**  
把数据配方写成版本化文件：每个阶段的目标、数据来源、权重上下限、切换指标、保护桶、回滚条件都要明确。否则所谓 curriculum 只是脚本里的临时经验。

### 18.10 后训练数据引擎与 Judge 模型

| 知识点 | 通俗解释 | 生动例子 |
| --- | --- | --- |
| 后训练飞轮 | 收集、生成、评判、训练、评测、回流的循环。 | 产品反馈不断进入下一轮员工培训。 |
| Judge 校准 | 确认 judge 分数和人类判断一致。 | 阅卷老师要先统一评分标准。 |
| 反例池 | 收集 judge 高分但实际低质的样本。 | 老师喜欢字迹工整，但学生内容错了也要扣分。 |
| 主动采样 | 优先标注最有价值的样本。 | 医生优先检查疑难病例，而不是随机看健康人。 |

**容易误解的点**  
Judge 分数上升不等于用户体验上升。模型会学习 judge 的偏好漏洞，比如变长、变客气、套模板、回避困难问题。后训练必须同时看 judge、人审和线上指标。

**项目里怎么用**  
至少维护三套数据：稳定 holdout 用来校准 judge，抽检池用来估计偏差，反例池用来防止奖励投机。每次 judge 更新后，历史结论要重新标注版本，否则分数不可比。

### 18.11 偏好数据与对齐失效模式

| 知识点 | 通俗解释 | 生动例子 |
| --- | --- | --- |
| 偏好排序 | 学习 A 比 B 更好，而不是学习唯一标准答案。 | 两篇作文都对，但一篇更清楚、更有边界。 |
| Rubric | 标注准则，把“好”拆成可判断维度。 | 面试评分表不能只写“感觉不错”。 |
| 过度保守 | 安全提高了，但普通任务也拒答。 | 保安太严格，员工也进不了公司。 |
| 奖励投机 | 模型学会讨好评分器，而不是完成任务。 | 学生研究老师喜好，而不是掌握知识。 |

**容易误解的点**  
偏好数据越多不一定越好。如果候选回答差异太小、标注员标准不一致、rubric 太模糊，模型学到的是噪声和模板。偏好学习尤其怕“看起来更好”的表面特征压过事实正确性和任务完成度。

**项目里怎么用**  
每条偏好样本最好记录维度分和选择理由，而不仅是 winner/loser。上线前要专门看过度拒答、冗长、模板化、事实退化和高分低质样本。

### 18.12 分布式训练与 Checkpoint

| 知识点 | 通俗解释 | 生动例子 |
| --- | --- | --- |
| Sharded checkpoint | 分片保存大模型状态。 | 巨大档案拆成多个箱子存。 |
| Portable checkpoint | 能跨拓扑或框架迁移的 checkpoint。 | 文件不依赖某台电脑才能打开。 |
| Async checkpoint | 后台保存，减少训练主链阻塞。 | 厨房继续出餐，后台同时归档账单。 |
| Manifest | 描述 checkpoint 所有文件、状态和版本的清单。 | 搬家箱子外面的清单。 |

**容易误解的点**  
保存成功不等于恢复成功。分布式训练里，权重恢复只是基础，优化器状态、学习率、sampler、RNG、并行拓扑、数据位置都要连续。否则 loss 曲线可能看似继续，实验语义已经变了。

**项目里怎么用**  
定期做恢复演练，而不是等故障时才第一次恢复。恢复后要检查 loss 连续性、数据不重复不跳样、LR 不断层、world size 变化是否按规则重映射。

### 18.13 数据质量、去重与治理

| 知识点 | 通俗解释 | 生动例子 |
| --- | --- | --- |
| 精确去重 | 删除完全相同样本。 | 复印了十份同一张讲义。 |
| 近重复 | 删除轻微改写或模板化重复。 | 改了变量名但整体代码一样。 |
| 语义重复 | 表达不同但信息几乎相同。 | 三篇文章都在讲同一条新闻。 |
| 数据污染 | 训练数据泄漏到评测集或高风险来源。 | 考试前拿到了答案。 |

**容易误解的点**  
去重率不是越高越好。过度去重可能删除领域里的重要模板、标准写法和合法变体。治理目标不是追求“最干净”，而是在减少浪费、降低污染和保持覆盖之间做平衡。

**项目里怎么用**  
治理 pipeline 要有抽样审计。每次过滤后不仅看删了多少，还要看删掉了什么、关键领域是否受损、下游任务桶是否退化。对高价值领域要设置保护名单或人工复核。

### 18.14 目标函数、优化器与学习率日程

| 知识点 | 通俗解释 | 生动例子 |
| --- | --- | --- |
| 目标函数 | 告诉模型优化方向。 | 地图上的目的地。 |
| 优化器 | 决定参数怎么更新。 | 开车方式，是稳一点还是激进一点。 |
| Warmup | 训练初期逐步增大学习率。 | 新车上路先慢慢加速。 |
| Decay | 后期降低学习率，避免震荡和过拟合。 | 快到停车位时松油门。 |

**容易误解的点**  
AdamW 不是“选了就完了”。`beta`、`epsilon`、weight decay、gradient clipping、batch size、混合精度都会影响实际行为。小模型调出来的超参，放到大模型和长上下文时经常失效。

**项目里怎么用**  
调参先短跑，再长跑。短跑验证是否稳定、是否起跑正常；长跑验证泛化、尾部任务和恢复一致性。所有 schedule 最好按 token 数定义，而不是只按 step，因为 packing 和 batch 变化会改变每步含义。

### 18.15 评测与消融方法学

| 知识点 | 通俗解释 | 生动例子 |
| --- | --- | --- |
| Baseline | 可比较的固定基线。 | 比赛要有同一条起跑线。 |
| 消融 | 一次只改一个关键变量，判断收益来源。 | 做菜时只换盐，才能知道味道变化是不是盐导致的。 |
| 分桶评测 | 按长度、任务、风险、成本等切开看。 | 体检报告分项目，不只给总分。 |
| 显著性和效应量 | 判断提升是真变化还是随机波动。 | 一次考试多 1 分可能只是运气。 |

**容易误解的点**  
平均分提升最容易骗人。模型可能在简单题上涨很多，在高价值难题上退化；也可能离线分数涨了，但线上时延和拒答变差。评测不是证明模型“更强”，而是帮助做上线决策。

**项目里怎么用**  
实验报告必须写预算口径、随机种子、数据版本、评测版本、分桶结果、失败样本和成本变化。没有失败样本的评测报告，通常无法指导下一轮训练。

### 18.16 Scaling Law 与实验经济学

| 知识点 | 通俗解释 | 生动例子 |
| --- | --- | --- |
| Scaling law | 用小规模结果预测大规模趋势。 | 用小店试营业估计全国开店风险。 |
| Compute-optimal | 给定算力下模型和数据怎么配更划算。 | 同样预算买更大机器，还是多买原料。 |
| Product-optimal | 产品里最值得的规模，不一定是论文最优规模。 | 跑车很快，但通勤未必划算。 |
| 机会成本 | 做一个大实验会占掉其它实验机会。 | 全公司只做一个大项目，其他机会都暂停。 |

**容易误解的点**  
规模变大带来的不只是质量提升，还有训练周期、失败成本、推理成本、部署复杂度和团队迭代速度下降。一个方向如果小规模趋势不稳，直接放大通常是在放大不确定性。

**项目里怎么用**  
扩规模前做 Stop/Go 评审：小规模趋势是否稳定，关键桶是否持续改善，失败中间资产是否可复用，推理成本是否能接受。如果只剩“也许大了会好”，通常应该先停。

---

## 19. 快速阅读路径（按任务）

如果你在做训练平台：

1. `index.md`
2. `megatron-lm-deepspeed-and-open-training-stacks.md`
3. `distributed-training-and-checkpointing.md`
4. `cluster-operations-and-experiment-management.md`

如果你在做数据与后训练：

1. `data-quality-dedup-and-governance.md`
2. `data-systems-and-optimization.md`
3. `post-training-data-engines-and-judge-models.md`
4. `preference-data-and-alignment-pitfalls.md`

如果你在做上线验证：

1. `evaluation-and-ablation-methodology.md`
2. `mtp-and-speculative-decoding.md`
3. `scaling-laws-and-experiment-economics.md`

---

## 20. 图片来源

图分两类：

1. Wikimedia Commons 公共图源；
2. GPT 生成的“方法概括图”。

完整清单见：  
[训练专题图片来源与授权](image-sources.md)
