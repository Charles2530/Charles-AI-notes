# 训练总览

<div class="atlas-hero">
  <div class="atlas-kicker">Training Systems</div>
  <p class="atlas-lead">这一专题围绕大模型训练的完整生产线展开：从预训练、SFT、对齐，到 Megatron-LM、DeepSpeed、输入管线、稳定性、checkpoint 与评测方法学。</p>
  <div class="atlas-chip-row">
    <span class="atlas-chip">预训练</span>
    <span class="atlas-chip">系统栈</span>
    <span class="atlas-chip">数据配方</span>
    <span class="atlas-chip">稳定性</span>
  </div>
</div>

## 专题定位

<div class="atlas-meta-grid">
  <div>
    <strong>核心问题</strong>
    <p>如何在给定数据、模型和算力预算下，把训练做成稳定、可恢复、可解释且能迁移到真实任务的系统。</p>
  </div>
  <div>
    <strong>适合读者</strong>
    <p>适合正在做预训练、后训练、数据治理、分布式训练或训练基础设施的人。</p>
  </div>
  <div>
    <strong>阅读方式</strong>
    <p>建议先读总览与系统栈页，再进入数据、优化器、并行、评测和回流体系。</p>
  </div>
</div>

## 推荐入口

<div class="atlas-card-grid">
  <a class="atlas-card" href="pretraining-finetuning-alignment.md">
    <strong>预训练、微调与对齐</strong>
    <p>先建立训练阶段分工，明确哪些问题该回看底座，哪些问题该回看 SFT 与偏好对齐。</p>
  </a>
  <a class="atlas-card" href="megatron-lm-deepspeed-and-open-training-stacks.md">
    <strong>Megatron-LM、DeepSpeed 与训练系统栈</strong>
    <p>理解 TP、PP、ZeRO、FSDP、CP 等系统栈如何决定训练规模上限。</p>
  </a>
  <a class="atlas-card" href="input-pipelines-packing-and-throughput-governance.md">
    <strong>输入管线、Packing 与吞吐治理</strong>
    <p>快速定位 GPU 为什么吃不满，以及有效 token 利用率为什么经常比显存更早出问题。</p>
  </a>
</div>

**训练的核心是**：给定数据、模型和算力预算，如何让损失稳定下降，并最终迁移到真实任务。

**一个统一抽象是经验风险最小化**：

\[
\min_\theta \mathbb{E}_{(x,y)\sim \mathcal{D}}[\ell(f_\theta(x), y)]
\]

但在大模型时代，训练从来不只是目标函数。真正影响结果的，是一整套系统：

1. 数据清洗；
2. 采样混合；
3. 优化器与学习率；
4. batch 组织与 packing；
5. 分布式训练与容错；
6. 后训练与对齐；
7. 算子、kernel 与编译栈。

也就是说，训练不是“把样本喂给模型”这么简单，而是把能力目标、数据生产线、优化过程和系统资源组织成一条长期运转的生产线。

## 1. 训练要解决的不是一个问题，而是三层问题

### 能力问题

模型能不能学到足够强的基础表示。

### 行为问题

模型会不会按任务要求输出。

### 系统问题

训练能不能稳定、可复现、高吞吐地跑完。

**这三层分别对应后面常见的**：

1. 预训练；
2. SFT / 对齐；
3. 数据、优化与分布式系统。

### 1.1 为什么这三层必须拆开看

因为很多问题表面相似，但成因完全不同。比如“模型表现不好”可能来自：

1. 基础表示没学够；
2. 后训练行为边界没立住；
3. 数据混合比有偏；
4. 训练中 packing、去重或 checkpoint 恢复导致隐藏偏差；
5. 分布式训练配置和 kernel 路径让你根本没有高效看到足够有效 token。

如果不先拆层，优化方向很容易跑偏。

## 2. 一个直观例子：训练代码模型

真正影响效果的往往不只是 loss 公式，还有：

1. GitHub 代码去重是否干净；
2. 文档和代码比例是否合理；
3. 长文件是否被截断；
4. 指令数据是否会破坏代码风格；
5. loader 是否能高效处理大文件；
6. 分布式训练是否允许足够长上下文。

这说明“训练效果”常常是数据工程和优化工程共同作用的结果。

### 2.1 再看一个多模态例子

训练文档 VLM 时，效果不仅取决于模型架构，还取决于：

1. OCR 是否稳定；
2. 文本与图像是否对齐；
3. 表格与图表是否被正确保留；
4. 合成数据是否引入模板偏差；
5. 后训练是否过度追求对话流畅而损伤结构准确性；
6. 多模态 batch 和 packing 是否高效。

这说明训练并不是“换个 backbone”就能解释的单变量问题。

## 3. 从大图景看训练流程

**很多现代模型训练可以概括成**：

\[
\text{Pretraining} \rightarrow \text{SFT} \rightarrow \text{Preference Alignment}
\]

**同时整条链路又依赖**：

\[
\text{Data} + \text{Optimization} + \text{Systems}
\]

也就是说，训练不是一个阶段，而是一条生产线。

### 3.1 这条生产线为什么越来越长

因为模型不再只是“学会预测下一个 token”，而是被要求：

1. 会对话；
2. 会遵循格式；
3. 会调用工具；
4. 会拒答高风险问题；
5. 会在特定领域表现稳定；
6. 会在真实系统里以合理成本运行。

这些能力无法只靠单一预训练阶段自然长出来。

## 4. 一个够用的判断框架

当模型出现问题时，可以先问它属于哪一层：

1. **能力边界不够**：多半回看预训练；
2. **输出格式和行为不对**：多半回看 SFT；
3. **风格、保守性、拒答边界异常**：多半回看对齐；
4. **loss、吞吐、稳定性异常**：多半回看优化和数据系统；
5. **大规模训练成本太高**：多半回看并行、checkpoint、kernel 与通信。

### 4.1 再往下追问三件事

对每个问题，再继续问：

1. 这是数据问题，还是目标函数问题？
2. 这是训练阶段问题，还是评测口径问题？
3. 这是模型能力问题，还是系统实现问题？

很多“模型不行”的结论，经不起这三问。

## 5. 为什么训练里最难的是资源分配

训练系统最终总会落回“有限资源如何分配”：

1. 算力分给更大模型还是更长训练；
2. 数据预算分给更多样本还是更高质量标注；
3. 系统工程时间分给吞吐优化还是评测重建；
4. 后训练时间分给偏好对齐还是工具使用；
5. 算子优化时间分给主干 GEMM、attention，还是 loader 和数据系统。

这也是为什么训练方向总和实验经济学、数据治理和服务成本紧紧绑定。

## 6. 为什么这组文档要拆成多页

因为训练里至少有四块逻辑明显不同：

### 训练阶段逻辑

预训练、SFT、DPO 等，是“模型学什么”。

### 优化与系统逻辑

batch、packing、学习率、checkpoint、并行、恢复，是“训练怎么跑”。

### 数据与评测逻辑

去重、混合、回流、ablation、judge、数据版本，是“训练凭什么可信”。

### 算子与实现逻辑

kernel、通信、Triton、CUDA、量化与编译栈，是“训练为什么能跑得动”。

把这几块混在一起，会让很多工程判断失真。

## 7. 训练文档在本知识库里的位置

这里的训练章节不是只讲神经网络教科书，而是专门围绕基础模型和系统化训练问题组织。你会看到：

1. 预训练、微调与对齐之间的关系；
2. `Megatron-LM`、`DeepSpeed`、`ZeRO`、TP/PP/CP 等训练系统栈如何支撑大规模训练；
3. 集群运维、实验管理与成本治理如何决定研究效率和复现可信度；
4. 输入管线、packing 与 token 利用率如何决定真实吞吐；
5. 训练稳定性、低精度路径和故障排查如何避免高代价失败；
6. 数据系统与优化如何影响最终能力；
7. scaling、课程学习与数据混合如何塑造能力结构；
8. 分布式训练、checkpoint 与容错为何是主问题；
9. 后训练数据引擎和 judge 如何影响行为质量；
10. 评测与消融为什么决定训练结论是否可信；
11. 这些训练决策最终如何落到算子、编译器和推理服务上。

## 8. 和“算子与编译器”专题是什么关系

新补进去的算子专题，并不是和训练专题平行无关的一组低层笔记。它解决的是训练里一个经常被忽略的问题：

1. 为什么某些训练吞吐上不去；
2. 为什么某些量化路径训练不稳；
3. 为什么分布式 overlap 做了仍然不快；
4. 为什么长上下文训练最后卡在 attention 和内存系统。

也就是说，训练专题回答“为什么要这样训”，算子专题回答“为什么这套训练系统能跑起来”。

## 9. 推荐阅读顺序

**建议先读**：

1. [预训练、微调与对齐](pretraining-finetuning-alignment.md)
2. [Megatron-LM、DeepSpeed 与开放训练系统栈](megatron-lm-deepspeed-and-open-training-stacks.md)
3. [集群运维、实验管理与成本治理](cluster-operations-and-experiment-management.md)
4. [输入管线、Packing 与吞吐治理](input-pipelines-packing-and-throughput-governance.md)
5. [训练稳定性、数值异常与故障排查](stability-numerics-and-failure-triage.md)
6. [数据系统与优化](data-systems-and-optimization.md)
7. [Scaling、课程学习与数据混合](scaling-curriculum-and-data-mixture.md)
8. [目标函数、优化器与学习率日程](objectives-optimizers-and-schedules.md)
9. [分布式训练与 Checkpoint](distributed-training-and-checkpointing.md)
10. [评测与消融方法学](evaluation-and-ablation-methodology.md)

如果你关注部署前的数据闭环，再继续看：

1. [数据质量、去重与治理](data-quality-dedup-and-governance.md)
2. [后训练数据引擎与 Judge 模型](post-training-data-engines-and-judge-models.md)
3. [Scaling Law 与实验经济学](scaling-laws-and-experiment-economics.md)

如果你开始关心训练为什么会被硬件和 kernel 塑形，就继续看：

1. [算子与编译器总览](../operators/index.md)

## 10. 一个总判断

训练不是“把数据喂给模型”那么简单，而是把能力目标、数据结构、优化过程和系统资源组织成一条长期运转的生产线。真正成熟的训练体系，不只是能把 loss 跑下去，而是能解释为什么它会下去、为什么它值得相信，以及为什么它最终能转化成真实系统中的能力与收益。

而一旦规模足够大，这条生产线就会自然延伸到更底层的问题：并行、通信、kernel、量化、编译器、缓存和部署。理解训练，最终一定会理解系统；理解系统，也反过来会帮助你重新理解训练。
