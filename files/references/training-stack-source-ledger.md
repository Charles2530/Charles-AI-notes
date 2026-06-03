# Megatron、DeepSpeed 与训练栈来源台账

这份台账服务于 [Megatron、DeepSpeed 与训练栈](../training/megatron-lm-deepspeed-and-open-training-stacks.md)。正文不把训练栈写成框架排行榜，而是按计算账、状态账、激活账和资产账解释：Megatron 主要切 Transformer 层内计算，ZeRO/FSDP 主要切 data-parallel 训练状态，DeepSpeed 主要提供运行时和内存平台，checkpoint/export 决定训练资产能否复用。

## 使用方式

第一类来源是 Megatron 论文和 Megatron Core 文档。Megatron-LM 原论文支撑 MLP column parallel / row parallel、attention heads 切分、vocab parallel 和通信点选择。2021 Megatron-LM 大规模训练论文支撑 TP、PP、DP 组合、interleaved pipeline 和跨集群扩展边界。Megatron Core 文档用于把论文概念对应到现代训练库：TP、PP、DP、EP、CP、distributed optimizer、MoE、dist checkpointing 和 Transformer 构件。

第二类来源是 ZeRO、DeepSpeed 和 FSDP。ZeRO 论文支撑 mixed precision Adam 约 \(16\Psi\)、Stage 1/2/3 分别切 optimizer states、gradients、parameters 的显存账。DeepSpeed Megatron 教程和 ZeRO 文档支撑 Megatron + ZeRO 的工程组合、ZeRO-3、offload 和运行时 API。PyTorch FSDP 文档和 FSDP blog 支撑 full sharding、按需 all-gather/reshard、state dict 和 PyTorch 原生生态边界。

第三类来源是分布式训练与 checkpoint 文档。Megatron Core dist checkpointing、FSDP state dict、DeepSpeed checkpoint 和 PyTorch distributed 文档帮助正文强调：训练 checkpoint 不是单个权重文件，而是模型、优化器、调度器、RNG、data progress、topology、shard manifest、低精度 scale 和导出路径的组合资产。

第四类来源是中文教程和工程长文。Datawhale 的 VRAM/ZeRO 教程、OneFlow/智源转载的大模型系统长文适合学习“先算显存账、再选工具”的中文讲法。事实、API 语义、性能 claim 和图片来源仍回论文、官方文档和稳定技术博客核对。

## 写作边界

正文不把 Megatron、DeepSpeed、FSDP 写成互斥替代品。Megatron-style TP/PP/CP/EP 是模型计算和并行拓扑层；ZeRO/FSDP 是训练状态分片层；DeepSpeed 是运行时和内存平台；它们可以组合，但组合后 checkpoint、optimizer ownership、rank mapping 和导出语义必须明确。

正文不把 OOM 归因于“模型太大”一个词。参数、梯度、optimizer states、activations、temporary buffers、communication buckets、fragmentation 和 checkpoint staging 都要分开。不同 OOM 来源对应不同工具。

正文不把 ZeRO/FSDP 说成减少计算量。它们减少的是 DP ranks 间长期常驻状态冗余，代价是 all-gather、reduce-scatter、reshard、元数据和 checkpoint 复杂度。

正文不只看“能跑一个 step”。训练栈成熟度要看 step time、MFU、P95 checkpoint time、恢复时间、数据进度连续性、full weight 导出、SFT/评测/推理复用和故障演练。

## 来源

| 来源 | 类型 | 本站使用方式 | 关联主题 |
| --- | --- | --- | --- |
| [Megatron-LM](https://arxiv.org/abs/1909.08053) | 论文 / 图源 | 使用 Figure 3a 解释 MLP column/row parallel 和通信点。 | Tensor parallel |
| [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473) | 论文 | 支撑 TP/PP/DP 组合、pipeline schedule 和大规模扩展取舍。 | Hybrid parallel |
| [ZeRO](https://arxiv.org/abs/1910.02054) | 论文 / 图源 | 支撑 \(16\Psi\) 显存账和 Stage 1/2/3 状态分片图。 | State sharding |
| [DeepSpeed Megatron tutorial](https://www.deepspeed.ai/tutorials/megatron/) | 官方教程 | 支撑 Megatron 与 ZeRO 在工程训练栈中的组合。 | Megatron + ZeRO |
| [DeepSpeed ZeRO documentation](https://deepspeed.readthedocs.io/en/latest/zero3.html) | 官方文档 | 支撑 ZeRO-3、offload、参数 gather、初始化和运行时 API。 | ZeRO runtime |
| [Megatron Core documentation](https://docs.nvidia.com/megatron-core/developer-guide/latest/index.html) | 官方文档 | 支撑现代 Megatron Core 的 TP/PP/DP/EP/CP、MoE、distributed optimizer 和 checkpoint 模块。 | Megatron Core |
| [Megatron Core tensor parallel API](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/tensor_parallel.html) | 官方文档 | 支撑 column/row parallel、vocab parallel、parallel cross entropy 等现代接口。 | TP API |
| [PyTorch FSDP documentation](https://docs.pytorch.org/docs/stable/fsdp.html) | 官方文档 | 支撑 full sharding、state dict、参数生命周期和 PyTorch 原生边界。 | FSDP |
| [PyTorch FSDP blog](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/) | 官方博客 | 补充 FSDP API 的设计动机和工程入口。 | FSDP 讲法 |
| [Datawhale VRAM Calculation and ZeRO](https://datawhalechina.github.io/llm-algo-leetcode/01_Hardware_Math_and_Systems/06_VRAM_Calculation_and_ZeRO.html) | 中文教程 | 学习用手算显存账解释参数、梯度、优化器状态和 ZeRO 分片。 | 中文讲法 |
