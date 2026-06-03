# 分布式训练与 Checkpoint 来源台账

这份台账服务于 [分布式训练与 Checkpoint](../training/distributed-training-and-checkpointing.md)。正文参考高质量技术博客和中文公众号常用讲法：先把训练长跑画成数据、前后向、优化器、通信和 checkpoint 五条流，再解释 DP / TP / PP / CP / EP、ZeRO / FSDP、pipeline bubble、分布式 checkpoint 和恢复演练。事实、API 和强性能 claim 仍以论文与官方文档为准。

## 使用方式

第一类来源是 pipeline 和混合并行论文。GPipe 支撑 micro-batch、pipeline stage、bubble 和理想利用率的讲法。Megatron-LM 2021 大规模训练论文支撑 tensor、pipeline、data parallelism 组合以及 interleaved pipeline 的扩展边界。DeepSeek-V3、Kimi K2 等技术报告可作为训练系统共设计案例，但正文只引用其“系统约束”讲法，不把单个 schedule 当作通用最优。

第二类来源是状态分片。ZeRO 论文支撑 mixed precision Adam 约 \(16\Psi\)、ZeRO Stage 1/2/3 分别切 optimizer states、gradients、parameters 的显存账。PyTorch FSDP 和 DeepSpeed ZeRO 文档用于核对 full sharding、all-gather、reshard、state dict、offload 和运行时 API 的当前语义。

第三类来源是分布式 checkpoint。PyTorch Distributed Checkpoint 文档支撑多 rank 并行保存/加载、load-time resharding、async save、staging、planner、state dict getter/setter 和格式转换。Megatron Core dist checkpointing 文档支撑“一个并行配置保存、另一个并行配置加载”的训练资产语义，并补充 distributed optimizer checkpoint 格式的边界。

第四类来源是中文教程和工程长文。Datawhale VRAM / ZeRO 教程、OneFlow / 智源系统长文适合学习中文讲法：先算状态显存，再看并行维度，再讨论通信和恢复。事实、API 版本和强性能 claim 仍以论文与官方文档为准。

## 写作边界

正文不重复训练栈页的框架分工。Megatron、DeepSpeed、FSDP 的工具定位只作为背景；本页重点是拓扑、通信、checkpoint 和恢复一致性。

正文不把 checkpoint 等同于 model checkpoint。Activation checkpointing 是 step 内重计算策略；training checkpoint 是跨 step 的恢复资产。两者都叫 checkpoint，但省的账和风险完全不同。

正文不把 checkpoint load 成功当成恢复成功。恢复还必须验证数据进度、RNG、LR / scheduler、loss scaler、optimizer states、consumed tokens、parallel topology 和 shard manifest。

正文不把 async checkpoint 写成无风险优化。异步保存需要 staging 边界、最终提交 manifest、后台失败告警、对象存储一致性和定期恢复演练。

正文没有使用 image2 或其他生成式工具作图。所有图片都来自仓库已有资产，并在正文标出原始来源。

## 图片使用

| 图片 | 原始来源 | 本站使用方式 |
| --- | --- | --- |
| `zero-memory-stages.png` | [ZeRO](https://arxiv.org/abs/1910.02054) Figure 1 | 说明普通 data parallel 复制 optimizer states、gradients、parameters，ZeRO 三阶段分别分片哪些状态。 |
| `gpipe-pipeline-parallelism.png` | [GPipe](https://arxiv.org/abs/1811.06965) Figure 2(c) | 说明 micro-batch 如何填充 pipeline stage 并减少 bubble。 |

## 来源

| 来源 | 类型 | 本站使用方式 | 关联主题 |
| --- | --- | --- | --- |
| [GPipe](https://arxiv.org/abs/1811.06965) | 论文 / 图源 | 使用 pipeline parallelism 图解释 micro-batch、stage 和 bubble。 | PP |
| [Megatron-LM 2021](https://arxiv.org/abs/2104.04473) | 论文 | 支撑 TP / PP / DP 混合并行和大规模集群扩展边界。 | Hybrid parallel |
| [ZeRO](https://arxiv.org/abs/1910.02054) | 论文 / 图源 | 支撑 \(16\Psi\) 显存账和状态分片边界。 | State sharding |
| [PyTorch Distributed Checkpoint](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html) | 官方文档 | 支撑多 rank 保存/加载、load-time resharding、async save 和 state dict 接口。 | DCP |
| [PyTorch async checkpoint recipe](https://docs.pytorch.org/tutorials/recipes/distributed_async_checkpoint_recipe.html) | 官方教程 | 支撑异步 checkpoint 的 staging、后台写入和训练主链关系。 | async checkpoint |
| [Megatron Core dist_checkpointing package](https://docs.nvidia.com/megatron-core/developer-guide/0.16.1/api-guide/core/dist_checkpointing.html) | 官方文档 | 支撑不同并行配置间加载、distributed optimizer checkpoint 格式和 safe loading。 | Megatron checkpoint |
| [PyTorch FSDP](https://docs.pytorch.org/docs/stable/fsdp.html) | 官方文档 | 支撑 full sharding、state dict、reshard 和参数生命周期。 | FSDP |
| [DeepSpeed ZeRO documentation](https://deepspeed.readthedocs.io/en/latest/zero3.html) | 官方文档 | 支撑 ZeRO-3、offload、参数 gather 和分片状态保存边界。 | ZeRO runtime |
| [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) | 技术报告 | 支撑 DualPipe、bubble 与通信重叠作为系统案例。 | pipeline overlap |
| [Kimi K2 Technical Report](https://arxiv.org/abs/2507.20534) | 技术报告 | 支撑 checkpoint engine 和训练 / 推理参数更新系统案例。 | checkpoint engine |
| [Datawhale VRAM Calculation and ZeRO](https://datawhalechina.github.io/llm-algo-leetcode/01_Hardware_Math_and_Systems/06_VRAM_Calculation_and_ZeRO.html) | 中文教程 | 学习用手算显存账解释参数、梯度、优化器状态和 ZeRO 分片。 | 中文讲法 |
