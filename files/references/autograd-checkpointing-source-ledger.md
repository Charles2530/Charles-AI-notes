# 自动微分、激活显存与重计算来源台账

这页记录自动微分、activation checkpointing、训练显存账和重计算策略相关来源。它服务的写作原则是：不要把 forward、backward、activation、checkpoint 写成词汇表，而要说明 backward 公式为什么需要 forward 中间值，以及每种省显存手段到底省的是哪一张账。

## 使用方式

写自动微分与显存文章时，先把来源分成四类。

第一类是自动微分和 saved tensors。PyTorch autograd 文档、JAX autodiff/remat 文档和经典 backprop 教材帮助本站解释：自动微分不是“倒放 forward”，而是每个算子的 backward rule 读取 forward 保存的输入、输出或辅助状态。正文里公式要落到具体例子，例如矩阵乘的梯度为什么需要输入 activation，GELU/dropout/attention 为什么还要保存额外信息。

第二类是 activation checkpointing / rematerialization。Sublinear Memory、Revolve、PyTorch checkpoint、JAX checkpoint 和 TensorFlow recomputation 资料帮助本站说明“少存一点、多算一遍”的精确折中。正文要避免把 checkpointing 写成压缩算法；它减少 activation residency，但会增加重算、随机数状态管理和调度复杂度。

第三类是训练显存账。ZeRO、FSDP、Megatron-LM、DeepSpeed 文档和 Hugging Face performance guide 帮助本站把参数、梯度、optimizer state、activation、temporary buffers 分开。读者遇到 OOM 时，应该先定位是哪张账爆了，而不是盲目同时打开 ZeRO、checkpointing、offload 和低精度。

第四类是长上下文、多模态和 kernel 来源。FlashAttention、sequence/context parallel、video diffusion / VLA 训练报告帮助本站说明：activation 压力不只来自层数，也来自 token 数、帧数、视觉分辨率、attention saved tensors 和 kernel workspace。checkpointing 的收益会被 fused kernel、IO-aware attention、compiler remat 和并行边界共同改变。

## 写作边界

activation checkpointing 和 model checkpoint 必须分开写。前者是 step 内部的中间激活保存/重算策略，后者是训练资产和恢复语义。ZeRO/FSDP 省的是长期模型状态冗余，activation checkpointing 省的是中间激活 residency，FlashAttention 省的是 attention 里的 IO 和 softmax 中间矩阵；这些手段可以组合，但不能互相替代。

中文工程博客适合学习讲法，尤其是如何从 OOM、显存峰值、GPU trace 和配置开关进入问题。涉及 API 参数、版本行为和强性能 claim 时，正文仍回 PyTorch/JAX/Hugging Face 官方文档或原论文核对。

## 来源

| 来源 | 类型 | 本站学什么 | 落到页面 |
| --- | --- | --- | --- |
| [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174) | 论文 | 把部分 activation 从保存改成 backward 前重算，形成显存/计算折中。 | activation checkpointing |
| [Revolve](https://dl.acm.org/doi/10.1145/347837.347846) | 论文 | 最优 checkpoint scheduling 的经典来源，说明重计算有调度问题。 | rematerialization 边界 |
| [PyTorch Autograd mechanics](https://pytorch.org/docs/stable/notes/autograd.html) | 官方文档 | saved tensors、计算图和 autograd function 如何服务 backward。 | 自动微分机制 |
| [PyTorch Activation Checkpointing](https://pytorch.org/docs/stable/checkpoint.html) | 官方文档 | reentrant/non-reentrant、RNG state 和 checkpoint API 的当前语义。 | checkpointing 实践 |
| [JAX gradient checkpointing](https://docs.jax.dev/en/latest/gradient-checkpointing.html) | 官方文档 | `jax.checkpoint` / remat 如何用 policy 控制保存和重算。 | remat 视角 |
| [TensorFlow Recompute Grad](https://www.tensorflow.org/api_docs/python/tf/recompute_grad) | 官方文档 | TensorFlow 中用 recompute decorator 换显存的接口。 | 框架对比 |
| [ZeRO](https://arxiv.org/abs/1910.02054) | 论文 | 参数、梯度和 optimizer state 在数据并行中的重复保存问题。 | 训练显存账 |
| [PyTorch FSDP](https://docs.pytorch.org/docs/stable/fsdp.html) | 官方文档 | 参数分片、all-gather、reshard 和 state dict 语义。 | 长期状态分片 |
| [Hugging Face Transformers performance guide](https://huggingface.co/docs/transformers/perf_train_gpu_one) | 官方文档 | batch size、mixed precision、gradient checkpointing 和训练实践入口。 | OOM 排查 |
| [Megatron-LM](https://arxiv.org/abs/1909.08053) | 论文 | 大模型训练中 activation、parallelism 和 checkpointing 与系统吞吐相关。 | 训练系统背景 |
| [FlashAttention](https://arxiv.org/abs/2205.14135) | 论文 | IO-aware attention 如何减少 attention memory traffic 和中间矩阵保存。 | attention 显存 |
| [OneFlow 官方博客](https://blog.csdn.net/OneFlow_Official) | 中文工程博客 | 学习用显存账、kernel、并行策略和工程 trace 串起系统文章的写法。 | 中文讲法 |
| [Datawhale LLM 算法教程](https://datawhalechina.github.io/llm-algo-leetcode/) | 中文工程教程 | 公式、PyTorch 代码和复杂度同页对照的讲解方式。 | 中文讲法 |
