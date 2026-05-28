# Megatron、DeepSpeed 与训练栈




<!-- [补充] -->
| 读者 | 建议读法 | 工程落点 |
| --- | --- | --- |
| 初学者 | 先抓“Megatron、DeepSpeed 与训练栈”要解决的问题、关键术语和最小例子；公式只看变量方向。 | 能说清输入、输出和常见失败信号。 |
| 工程师 | 直接对照结构、指标、成本和边界条件，必要时回到前置页补概念。 | 落到数据、算力、延迟、显存、评测和回滚检查。 |

!!! note "合并提示"
    **note · 读法定位**

    这页先回答“Megatron、DeepSpeed 与训练栈”在「训练」里的位置：它解决什么局部问题，依赖哪些前置，最后会影响哪类工程或研究判断。
    **前置**：先知道 loss、optimizer、batch、checkpoint 和评测集的基本含义。 必要时先回 [训练入口](index.md)、[基础知识](../foundations/index.md) 或 [术语表](../glossary/index.md)。
    **主线关系**：把目标函数、数据、分布式系统、评测回流和实验治理串起来，看一次训练为什么成功、为什么不稳、为什么贵。

    **info · 事实版本**

    | Checked Date | Official Source | Repro Status | Notes |
    | --- | --- | --- | --- |
    | 2026-05-16 | 原论文 / 官方材料 | Paper Only | 未见本站记录的第三方独立复现；按论文/官方材料保守引用。 |

### Evidence Snapshot

| Claim | Direct Source | Figure/Table/Setting | Evidence Type | Repro Status | Can Support | Cannot Prove |
| --- | --- | --- | --- | --- | --- | --- |
| `Megatron、DeepSpeed 与训练栈` 的核心结果在论文设置下成立 | 原论文 / 官方材料 | 主实验、架构图或 benchmark setting | Paper Result | Paper Only | 支撑本页对该方法机制、任务设置和局部收益的介绍 | 不证明跨数据分布、跨硬件、真实部署或第三方独立复现成立 |
| 本页把方法接回世界模型/效率主线 | 本站综合判断 | 成本账、失败边界与相邻专题链接 | Site Inference | Site Inference | 支撑阅读路线、实验设计和工程优先级假设 | 不能替代原论文 claim，也不能证明 closed-loop 或生产收益 |

!!! warning "读数边界"
    本页若引用结果表，只能按论文 benchmark、数据集、硬件和评测协议解释；需额外确认是否 closed-loop、是否真实机器人、是否系统吞吐、是否独立复现，不能把单项分数直接外推为规划收益或部署安全。

![Megatron hybrid parallelism](../assets/images/paper-deep-dives/foundations/megatron-lm/figure-8-hybrid-mp-dp.png){ width="780" }

<small>图源：`Megatron-LM`，Figure 8。原论文图意：把 tensor model parallel 与 data parallel 组合，说明单层计算切分和数据并行副本可以叠加。</small>

!!! note "这张图怎么读"
    **输入输出**：输入是 Transformer 层、GPU rank 和数据 batch，输出是 tensor-parallel 组内计算与 data-parallel 组间同步的训练拓扑。  
    **效率机制**：TP 降低单层矩阵和词表投影的单卡压力，DP 保持数据吞吐，两者共同决定显存和通信账。  
    **对主线意义**：世界模型长轨迹训练遇到显存/通信瓶颈时，要先把并行拓扑和状态分片讲清楚。  
    **不能证明什么**：这张系统结构图不能证明任意模型质量提升，也不能替代端到端吞吐、恢复和收敛评测。


开放训练系统栈要解决的问题不是“选一个最流行框架”，而是在给定模型规模、硬件拓扑、数据吞吐、显存预算、容错要求和团队能力后，组织出一条可长期维护的训练路径。Megatron-LM、Megatron Core、DeepSpeed、ZeRO、FSDP、Accelerate、Ray、Torch Distributed 等组件各自覆盖不同层面，真正难的是组合边界和状态语义。

本页和 [分布式训练与 Checkpoint](distributed-training-and-checkpointing.md) 分工如下：那一页讲并行拓扑、通信、checkpoint 和恢复资产；本页讲训练系统栈的定位、选型和工程组织，避免把所有并行细节再展开一遍。

!!! note "合并提示"
    **note · 初学者先抓住**


    训练系统栈不是“框架排行榜”，而是把模型、显存、通信、数据和状态管理拼成一条能长期运行的路径。Megatron、DeepSpeed、FSDP、Ray、Slurm 等工具各管一部分，选型时先找主瓶颈，再决定该引入哪一层复杂度。

    **note · 难点解释：为什么不能只按 GPU 数量选框架**


    同样是 128 张卡，模型层宽、序列长度、MoE 路由、网络拓扑和 checkpoint 策略不同，最佳并行方式会完全不同。框架能提供能力，但不能替团队决定 rank mapping、通信域、数据恢复和状态导出语义。

<!-- [补充] -->
> **踩坑原因**：这类工程问题通常不是单点公式错，而是 shape、资源预算、并发路径和观测指标之间有一个假设没对齐；先把触发条件、监控信号和回滚边界拆开查。

## 先用两篇论文建立底座

Megatron-LM 和 ZeRO 适合成对读。前者从 Transformer layer 内部出发，问的是“单层 MLP、attention、词表 logits 太大时，怎么把 GEMM 切到多张 GPU 上”；后者从数据并行状态出发，问的是“每张卡都完整复制 optimizer states、gradients、parameters，显存为什么会被浪费掉”。

| 论文 | 先解决哪张账 | 核心动作 | 对训练栈选型的启发 |
| --- | --- | --- | --- |
| [Megatron-LM](../paper-deep-dives/foundations/megatron-lm.md) | 单层计算和参数太大 | MLP 用 column-parallel + row-parallel GEMM，attention 按 heads 切 Q/K/V，词表 loss 避免大 logits all-gather | 当 hidden size、attention heads 或 vocab projection 太大时，优先考虑 TP，并把高频通信放在节点内高速互联 |
| [ZeRO](../paper-deep-dives/foundations/zero.md) | DP 重复模型状态太贵 | 在 DP ranks 间分片 optimizer states、gradients、parameters，并处理 activation、buffer 和 fragmentation | 当优化器状态、梯度和参数副本占满显存时，优先考虑 ZeRO/FSDP，而不是盲目加 TP/PP |

这两个问题经常同时出现，但层级不同。Megatron-LM 让一个模型实例能被多卡共同计算；ZeRO 让多个 data-parallel 副本不再重复保存整份训练状态。现代训练栈里的 `TP + PP + DP + ZeRO/FSDP`，本质上就是把这两张账叠起来。

一个实用心法是：**先拆显存，再拆通信，再拆代码所有权**。显存里要分清 parameters、gradients、optimizer states、activations、temporary buffers 和 fragmentation；通信里要分清 TP layer 内 all-reduce、DP reduce-scatter/all-gather、PP 点对点、CP attention 通信和 checkpoint I/O；代码所有权里要分清模型构件、运行时、数据进度和导出格式分别由谁负责。

## 一、训练系统栈到底在解决什么

<!-- [融合：算法基础] 来源：llm-algo-leetcode《06. VRAM Calculation and ZeRO | 显存计算与 ZeRO 优化 (VRAM Calculation & ZeRO)》-->
> **算法视角**：ZeRO、张量并行、流水线并行和通信 overlap 都是在切分“状态、矩阵或 micro-batch”；收益来自显存和计算分摊，代价会转成 all-gather、reduce-scatter、all-reduce 或 pipeline bubble。
>
> 延伸阅读：[06. VRAM Calculation and ZeRO | 显存计算与 ZeRO 优化 (VRAM Calculation & ZeRO)](https://datawhalechina.github.io/llm-algo-leetcode/01_Hardware_Math_and_Systems/06_VRAM_Calculation_and_ZeRO.html)

<!-- [融合：复杂度分析] 来源：llm-algo-leetcode《06. VRAM Calculation and ZeRO | 显存计算与 ZeRO 优化 (VRAM Calculation & ZeRO)》-->
| 操作 | 时间复杂度 | 空间复杂度 | 说明 |
|------|-----------|-----------|------|
| ZeRO 状态分片 | 计算量基本不变 | 被分片状态约 $O(\Psi/N_d)$ | $\Psi$ 为参数量，$N_d$ 为 data-parallel degree；参考：[对应章节](https://datawhalechina.github.io/llm-algo-leetcode/01_Hardware_Math_and_Systems/06_VRAM_Calculation_and_ZeRO.html) |
| 分布式通信 | 取决于消息大小与拓扑 | 通信 buffer 需额外预算 | 显存省下后，瓶颈常转移到带宽、延迟和 overlap 能力。 |

一个大模型训练栈至少要同时管理计算、显存、通信、数据和状态资产。计算层涉及 GEMM、Attention、MoE、activation recompute 和混合精度；显存层要容纳参数、梯度、优化器状态、激活和通信 buffer；通信层处理 DP、TP、PP、CP、EP、ZeRO / FSDP 的 collective；数据层负责输入吞吐、packing、随机顺序和断点恢复；状态资产则包括 checkpoint、配置、日志、指标、导出和后训练接口。

可以把训练系统栈看成三层：

| 层级 | 代表组件 | 主要职责 |
| --- | --- | --- |
| 模型并行层 | Megatron-LM、Megatron Core | Transformer 构件、TP/PP/CP/EP、通信重叠、MoE |
| 运行时内存层 | DeepSpeed、ZeRO、FSDP | 状态分片、offload、pipeline runtime、state dict |
| 编排与资产层 | Torch Distributed、Ray、Slurm/K8s、实验平台 | 作业调度、日志指标、数据状态、checkpoint 生命周期 |

选型时先问“主约束是什么”，再问“哪个工具合适”。如果主约束是超大 hidden size，TP 和 Megatron 的价值更高；如果主约束是优化器状态放不下，ZeRO/FSDP 更关键；如果主约束是数据喂不满，再换训练框架也不会解决根因。

## 二、Megatron 路线：并行拓扑和 GPU 亲和

Megatron-LM 的影响力来自两个方面：一是把大规模 Transformer 的 TP/PP/DP 等并行方法工程化，二是长期围绕 NVIDIA GPU、通信重叠和大模型训练吞吐优化。Megatron Core 则把其中更底层的 Transformer 构件和并行原语拆成可组合模块。

### 它适合什么

Megatron 路线更适合超大稠密 Transformer 或 MoE，尤其是需要 TP、PP、CP、EP 混合并行、追求高 MFU 和硬件利用率，并且团队能处理 rank mapping、process group、通信重叠和 kernel 亲和的场景。

它不一定适合小团队快速训中小模型。Megatron 的收益通常来自更强的系统设计能力；如果数据管线、实验管理和恢复流程还不稳定，过早引入复杂并行反而会降低迭代速度。

### Megatron Core 的意义

Megatron Core 的价值不只是“把脚本库模块化”。更重要的是，它把 Transformer block、并行线性层、sequence/context parallel、MoE、通信 overlap、混合精度和 checkpoint 这些构件变成更可复用的积木。对组织而言，这意味着训练栈可以从“复制一套脚本”逐步走向“维护一套可组合训练基础设施”。

从 Megatron-LM 原论文看，TP 的关键不是“任意切矩阵”，而是避开会强制同步的位置：MLP 第一层按列切，让 GeLU 可以本地做；第二层按行切，只在 block 输出处 all-reduce。Attention 则利用 head 级并行，把 Q/K/V projection 切到不同 ranks，再在 output projection 后同步。这个设计把单个 Transformer layer 的通信压缩到少数几个 `all-reduce`，也解释了为什么 TP size、attention heads、hidden size 和 vocab padding 会直接影响吞吐。

## 三、DeepSpeed、ZeRO 与 FSDP：状态分片和运行时组织

DeepSpeed 的代表能力是 ZeRO、offload、pipeline engine 和训练运行时配置。ZeRO 的核心是把 DP 中重复保存的优化器状态、梯度和参数逐步分片，从而降低单卡显存压力。

FSDP 的思想相近，但更贴近 PyTorch 原生生态，强调按 module 包装、按需 gather 和原生 state dict 集成。二者不是简单 API 风格差异，而是系统气质不同：

| 维度 | DeepSpeed / ZeRO | FSDP |
| --- | --- | --- |
| 生态定位 | 训练运行时平台，配置化能力强 | PyTorch 原生分布式能力 |
| 优势 | ZeRO/offload 成熟，工程配套多 | 原生集成好，定制模型更自然 |
| 主要风险 | 运行时黑盒感更强，配置组合复杂 | wrap、state dict、通信策略要自己管细 |
| 适合场景 | 状态显存是主瓶颈，需要快速组合能力 | 更重视原生生态、代码可控和框架一致性 |

ZeRO 论文给了一个非常有用的估算：mixed precision Adam 训练 \(\Psi\) 个参数时，fp16 parameters、fp16 gradients、fp32 master parameters、Adam momentum 和 Adam variance 合起来大约是 \(16\Psi\) bytes。也就是说，1.5B 参数的 fp16 权重看似只有 3GB，训练态模型状态却至少约 24GB，还没算 activation 和 buffer。ZeRO Stage 1/2/3 分别切 optimizer states、gradients、parameters，真正省的是“每个 DP rank 都长期保存一整份”的冗余。

### Offload 的边界

Offload 能把参数、梯度或优化器状态放到 CPU/NVMe，降低 GPU 显存压力，但它不是银弹。它适合“显存是硬瓶颈且吞吐要求可接受”的场景；如果 PCIe、CPU 内存或 NVMe 成为瓶颈，训练会从 GPU-bound 变成 I/O-bound。

判断 offload 是否值得，应同时看 GPU 显存是否真的无法通过 checkpointing、分片或 batch 调整解决，offload 后 step time 增长是否可接受，恢复和 checkpoint 是否更复杂，以及训练目标只是跑通实验还是长期高效预训练。

## 四、混合并行、长上下文、MoE 与低精度训练

训练系统栈的复杂度往往来自多种需求叠加，而不是单一技术本身。

### 混合并行

现实系统常组合 TP、PP、DP、ZeRO/FSDP、CP、EP。常见直觉是：高频通信放在高速拓扑域，粗粒度复制放在外层，状态分片放在 DP 组内。但这只是起点，真实配置还要看模型层宽、层数、序列长度、micro-batch、网络拓扑和 checkpoint 策略。

Process group 设计就是训练系统的网络骨架。并行维度越多，rank mapping、通信组、状态分片和恢复 manifest 越需要统一设计。否则训练能跑起来，但一旦出错很难定位。

### 长上下文训练

长上下文会把训练栈推向新的耦合点：attention 激活和通信压力上升，CP / SP 与 attention kernel 强绑定，activation checkpointing 代价被放大，数据 packing 和恢复一致性更难，评测与采样成本也更高。

因此长上下文不是“把 max length 调大”，而是数据、kernel、并行、checkpoint 和评测共同变化。

### MoE

MoE 会让训练栈从并行工程升级成路由工程。除了专家并行，还要管理 token dispatch、负载均衡、容量因子、drop 策略、专家热点、all-to-all 和 checkpoint 分片。很多 MoE 训练栈难复制，不是因为专家层公式复杂，而是因为路由、通信、数据和恢复全都耦合。

### FP8 与低精度

低精度训练不只是某个 kernel 的 dtype 变化。FP8/MXFP/NVFP/HiFloat 会影响 scaling、activation 保存、optimizer states、通信、checkpoint 和数值稳定性。相关内容放在 [低比特训练与数值格式](low-bit-training-and-numerics.md)；训练栈层面要关注的是：低精度状态如何保存、恢复、导出，以及是否与后训练和推理数制一致。

## 五、不同规模下的策略差异

训练栈选型强烈依赖规模。

| 规模区间 | 优先目标 | 常见策略 |
| --- | --- | --- |
| 8-32 卡 | 简单、可复现、快速迭代 | FSDP/ZeRO、少量 TP，先保证数据和 checkpoint 稳 |
| 64-256 卡 | 拓扑和状态管理开始成为主问题 | 混合 DP/TP/PP，系统化通信重叠和 sharded checkpoint |
| 512 卡以上 | 组织能力和工程治理成为核心 | 明确 owner、故障演练、自动化验收、资产兼容 |
| 多模态/长上下文/MoE | 所有瓶颈同时放大 | CP/EP、专用数据管线、专用 checkpoint 和评测闭环 |

小规模最重要的是不要过度复杂化。能用简单栈稳定复现，就不要为了“看起来先进”提前引入过多并行维度。大规模则相反，很多问题必须提前工程化，否则一次故障或一次不可恢复 checkpoint 就会消耗大量预算。

### 选型框架

可以按主约束选择：参数和优化器状态放不下时，先看 ZeRO / FSDP、activation checkpointing 和低精度 optimizer；单层太大时，看 TP 和 Megatron-style parallel layers；模型层数太深时，看 PP、stage 切分和 micro-batch；序列太长时，看 CP / SP、长上下文 attention kernel 和 packing；MoE 通信重时，看 EP、专家放置、dispatch / combine 和 all-to-all；数据喂不满时，先修数据系统，不要先换训练框架；恢复成本高时，优先 checkpoint manifest、异步保存和短跑恢复验证。

## 六、资产兼容：开放训练栈最容易被低估的部分

开放训练栈真正缺的往往不是功能，而是资产兼容。训练产物要能流向继续预训练、SFT、偏好优化、评测、推理转换、量化和归档。只在当前脚本里能恢复，不等于资产可复用。

### 状态语义

一个训练状态通常包括模型权重、optimizer states、scheduler 和 loss scaler、RNG / sampler / packing / consumed tokens、并行拓扑、分片策略、数据版本、代码版本、tokenizer、配置、特征开关和低精度 scaling 信息。

当 ZeRO、FSDP、TP、PP、CP、EP 组合后，状态语义会变得复杂。checkpoint 到底是训练态、推理态、portable 态还是中间态，必须明确标注。否则后训练和推理团队拿到产物后，会花大量时间猜“这些 shard 到底怎么拼”。

### 训练栈接口

一个成熟训练栈应提供稳定接口：能从训练 checkpoint 导出 full weights，能让 full weights 进入 SFT 或评测，能从训练态恢复并继续消耗正确数据，能记录 low precision scale、tokenizer 和数据版本，能支持关键指标与 trace 对齐，也能对失败 checkpoint 做不可用标记，避免误用。

这部分看起来不如并行算法“高级”，但长期维护价值更高。

## 七、验收清单

训练系统栈的真正验收标准不是“能跑一个 step”，而是能跑快、能跑稳、能恢复、能复用、能维护：吞吐、MFU、数据输入和通信重叠要达到目标；loss、grad norm、NaN、OOM、NCCL 和 I/O 失败要能定位；checkpoint 要能保存、校验、恢复和迁移，数据进度要连续；产物要能进入 SFT、评测、推理、量化和归档；配置、owner、版本、指标和实验记录也要清楚。

最小代码接口可以长这样：

<!-- [补充] -->
下面代码块优先看 shape、缓存、状态更新或资源申请相关行；这些行通常决定故障是语义、并发还是容量问题。
```python
def build_training_stack(cfg):
    model = build_model(cfg.model)
    model = apply_parallelism(model, cfg.parallelism)
    optimizer = build_optimizer(model, cfg.optimizer)
    runtime = build_runtime(model, optimizer, cfg.runtime)
    checkpointer = build_checkpointer(cfg.checkpoint, cfg.parallelism)
    return runtime, checkpointer
```

这段伪代码的重点是分层：模型构建、并行包装、优化器、运行时和 checkpoint 不应混成一团。只有边界清楚，后续替换 Megatron、DeepSpeed、FSDP、低精度策略或 checkpoint 后端时，系统才不会整体失控。

!!! tip "下一站"
    - 回到本专题入口：[训练](index.md)，确认这页在整条路线中的位置。
    - 按导航顺序继续：[分布式训练与 Checkpoint](distributed-training-and-checkpointing.md)。
    - 概念或符号卡住时，先查 [术语表](../glossary/index.md)，再回到当前页。
