# Megatron-LM、DeepSpeed 与开放训练系统栈

开放训练系统栈要解决的问题不是“选一个最流行框架”，而是在给定模型规模、硬件拓扑、数据吞吐、显存预算、容错要求和团队能力后，组织出一条可长期维护的训练路径。Megatron-LM、Megatron Core、DeepSpeed、ZeRO、FSDP、Accelerate、Ray、Torch Distributed 等组件各自覆盖不同层面，真正难的是组合边界和状态语义。

本页和 [分布式训练与 Checkpoint](distributed-training-and-checkpointing.md) 分工如下：那一页讲并行拓扑、通信、checkpoint 和恢复资产；本页讲训练系统栈的定位、选型和工程组织，避免把所有并行细节再展开一遍。

!!! note "初学者先抓住"

    训练系统栈不是“框架排行榜”，而是把模型、显存、通信、数据和状态管理拼成一条能长期运行的路径。Megatron、DeepSpeed、FSDP、Ray、Slurm 等工具各管一部分，选型时先找主瓶颈，再决定该引入哪一层复杂度。

!!! note "难点解释：为什么不能只按 GPU 数量选框架"

    同样是 128 张卡，模型层宽、序列长度、MoE 路由、网络拓扑和 checkpoint 策略不同，最佳并行方式会完全不同。框架能提供能力，但不能替团队决定 rank mapping、通信域、数据恢复和状态导出语义。

## 一、训练系统栈到底在解决什么

一个大模型训练栈至少要同时管理五类资源：

1. **计算**：GEMM、Attention、MoE、activation recompute、混合精度；
2. **显存**：参数、梯度、优化器状态、激活、通信 buffer；
3. **通信**：DP、TP、PP、CP、EP、ZeRO/FSDP 的 collective；
4. **数据**：输入吞吐、packing、随机顺序、断点恢复；
5. **状态资产**：checkpoint、配置、日志、指标、导出和后训练接口。

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

Megatron 路线更适合：

1. 超大稠密 Transformer 或 MoE；
2. 需要 TP、PP、CP、EP 混合并行；
3. 追求高 MFU 和硬件利用率；
4. 团队能处理 rank mapping、process group、通信重叠和 kernel 亲和；
5. 希望参考大规模训练的主流系统实现。

它不一定适合小团队快速训中小模型。Megatron 的收益通常来自更强的系统设计能力；如果数据管线、实验管理和恢复流程还不稳定，过早引入复杂并行反而会降低迭代速度。

### Megatron Core 的意义

Megatron Core 的价值不只是“把脚本库模块化”。更重要的是，它把 Transformer block、并行线性层、sequence/context parallel、MoE、通信 overlap、混合精度和 checkpoint 这些构件变成更可复用的积木。对组织而言，这意味着训练栈可以从“复制一套脚本”逐步走向“维护一套可组合训练基础设施”。

## 三、DeepSpeed、ZeRO 与 FSDP：状态分片和运行时组织

DeepSpeed 的代表能力是 ZeRO、offload、pipeline engine 和训练运行时配置。ZeRO 的核心是把 DP 中重复保存的优化器状态、梯度和参数逐步分片，从而降低单卡显存压力。

FSDP 的思想相近，但更贴近 PyTorch 原生生态，强调按 module 包装、按需 gather 和原生 state dict 集成。二者不是简单 API 风格差异，而是系统气质不同：

| 维度 | DeepSpeed / ZeRO | FSDP |
| --- | --- | --- |
| 生态定位 | 训练运行时平台，配置化能力强 | PyTorch 原生分布式能力 |
| 优势 | ZeRO/offload 成熟，工程配套多 | 原生集成好，定制模型更自然 |
| 主要风险 | 运行时黑盒感更强，配置组合复杂 | wrap、state dict、通信策略要自己管细 |
| 适合场景 | 状态显存是主瓶颈，需要快速组合能力 | 更重视原生生态、代码可控和框架一致性 |

### Offload 的边界

Offload 能把参数、梯度或优化器状态放到 CPU/NVMe，降低 GPU 显存压力，但它不是银弹。它适合“显存是硬瓶颈且吞吐要求可接受”的场景；如果 PCIe、CPU 内存或 NVMe 成为瓶颈，训练会从 GPU-bound 变成 I/O-bound。

判断 offload 是否值得，应同时看：

1. GPU 显存是否真的无法通过 checkpointing、分片或 batch 调整解决；
2. offload 后 step time 增长是否可接受；
3. 恢复和 checkpoint 是否更复杂；
4. 训练目标是跑通实验，还是长期高效预训练。

## 四、混合并行、长上下文、MoE 与低精度训练

训练系统栈的复杂度往往来自多种需求叠加，而不是单一技术本身。

### 混合并行

现实系统常组合 TP、PP、DP、ZeRO/FSDP、CP、EP。常见直觉是：高频通信放在高速拓扑域，粗粒度复制放在外层，状态分片放在 DP 组内。但这只是起点，真实配置还要看模型层宽、层数、序列长度、micro-batch、网络拓扑和 checkpoint 策略。

Process group 设计就是训练系统的网络骨架。并行维度越多，rank mapping、通信组、状态分片和恢复 manifest 越需要统一设计。否则训练能跑起来，但一旦出错很难定位。

### 长上下文训练

长上下文会把训练栈推向新的耦合点：

1. attention 激活和通信压力上升；
2. CP/SP 与 attention kernel 强绑定；
3. activation checkpointing 代价被放大；
4. 数据 packing 和恢复一致性更难；
5. 评测和采样成本也更高。

因此长上下文不是“把 max length 调大”，而是数据、kernel、并行、checkpoint 和评测共同变化。

### MoE

MoE 会让训练栈从并行工程升级成路由工程。除了专家并行，还要管理 token dispatch、负载均衡、容量因子、drop 策略、专家热点、all-to-all 和 checkpoint 分片。很多 MoE 训练栈难复制，不是因为专家层公式复杂，而是因为路由、通信、数据和恢复全都耦合。

### FP8 与低精度

低精度训练不只是某个 kernel 的 dtype 变化。FP8/MXFP/NVFP/HiFloat 会影响 scaling、activation 保存、optimizer states、通信、checkpoint 和数值稳定性。相关内容放在 [低比特训练与训练数制](low-bit-training-and-numerics.md)；训练栈层面要关注的是：低精度状态如何保存、恢复、导出，以及是否与后训练和推理数制一致。

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

可以按主约束选择：

1. **参数和优化器状态放不下**：先看 ZeRO/FSDP、activation checkpointing、低精度 optimizer；
2. **单层太大**：看 TP 和 Megatron-style parallel layers；
3. **模型层数太深**：看 PP、stage 切分和 micro-batch；
4. **序列太长**：看 CP/SP、长上下文 attention kernel 和 packing；
5. **MoE 通信重**：看 EP、专家放置、dispatch/combine 和 all-to-all；
6. **数据喂不满**：先修数据系统，不要先换训练框架；
7. **恢复成本高**：优先 checkpoint manifest、异步保存和短跑恢复验证。

## 六、资产兼容：开放训练栈最容易被低估的部分

开放训练栈真正缺的往往不是功能，而是资产兼容。训练产物要能流向继续预训练、SFT、偏好优化、评测、推理转换、量化和归档。只在当前脚本里能恢复，不等于资产可复用。

### 状态语义

一个训练状态通常包括：

1. 模型权重；
2. optimizer states；
3. scheduler 和 loss scaler；
4. RNG、sampler、packing 和 consumed tokens；
5. 并行拓扑、分片策略、数据版本和代码版本；
6. tokenizer、配置、特征开关和低精度 scaling 信息。

当 ZeRO、FSDP、TP、PP、CP、EP 组合后，状态语义会变得复杂。checkpoint 到底是训练态、推理态、portable 态还是中间态，必须明确标注。否则后训练和推理团队拿到产物后，会花大量时间猜“这些 shard 到底怎么拼”。

### 训练栈接口

一个成熟训练栈应提供稳定接口：

1. 从训练 checkpoint 导出 full weights；
2. 从 full weights 进入 SFT 或评测；
3. 从训练态恢复并继续消耗正确数据；
4. 记录 low precision scale、tokenizer、数据版本；
5. 支持关键指标和 trace 对齐；
6. 对失败 checkpoint 做不可用标记，避免误用。

这部分看起来不如并行算法“高级”，但长期维护价值更高。

## 七、验收清单

训练系统栈的真正验收标准不是“能跑一个 step”，而是：

1. **能跑快**：吞吐、MFU、数据输入、通信重叠达到目标；
2. **能跑稳**：loss、grad norm、NaN、OOM、NCCL 和 I/O 失败可定位；
3. **能恢复**：checkpoint 可保存、校验、恢复、迁移，数据进度连续；
4. **能复用**：产物可进入 SFT、评测、推理、量化和归档；
5. **能维护**：配置、owner、版本、指标和实验记录清楚。

最小代码接口可以长这样：

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
