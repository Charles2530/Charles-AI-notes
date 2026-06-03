# 模型训练从零入门来源台账

这份台账服务于 [模型训练从零入门](../training/beginner-model-training-primer.md)。正文参考高质量技术博客和中文公众号常用的讲法：先从“loss 下降为什么不等于训练成功”这个困惑进入，再把数据、loss、batch、optimizer、显存、时间、checkpoint 和评测拆成几张账。事实、公式、论文图和系统边界以官方文档、论文和一手工程复盘为准。

## 使用方式

第一类来源是训练循环和排查方法。PyTorch Optimization Loop 支撑最小训练 step 的代码语义；Karpathy 的训练 recipe 支撑从小实验、过拟合小 batch、检查 loss/grad 到逐步扩容的排查顺序。

第二类来源是现代训练 recipe。Hugging Face Smol Training Playbook 和 SmolLM3 复盘支撑正文里“训练不是单个 loss 名字，而是数据、规模、评测、成本和失败复盘组合”的讲法。

第三类来源是训练预算和系统论文。Chinchilla 支撑参数/token/compute 预算配平；ZeRO 支撑 optimizer states、gradients、parameters 的显存分片；GPipe 支撑 micro-batch 填充 pipeline bubble 的图解。

第四类来源是中文讲法参考。Datawhale 显存估算与 ZeRO 教程，以及其他中文技术博客、知乎专栏和微信公众号，只用于学习如何把参数、梯度、优化器状态、显存和分布式训练讲成可手算的账。稳定性不足、版权不清或没有原始出处的图和结论不进入正文。

## 写作边界

正文不把 loss 下降当作训练成功。Loss 是代理目标，必须和数据桶、评测分布、失败样本和下游指标一起看。

正文不把 GPU 利用率当作训练效率。训练效率要看 tokens/s、MFU、显存峰值、通信等待、checkpoint pause、数据管线和每个有效 token 成本。

正文不把 batch 改大写成单纯加速。Batch 会改变梯度噪声、学习率语义、warmup 需求和泛化行为。

正文不把 checkpoint 写成普通备份。Checkpoint 是恢复语义和实验审计证据链，必须包含 optimizer、scheduler、RNG、sampler、数据进度和配置。

正文没有使用 image2 或其他生成式工具作图。所有图片都来自仓库已有资产，并在正文标出原始来源。

## 图片使用

| 图片 | 原始来源 | 本站使用方式 |
| --- | --- | --- |
| `zero-memory-stages.png` | [ZeRO](https://arxiv.org/abs/1910.02054) Figure 1 | 说明 optimizer states、gradients、parameters 在普通 data parallel 中的冗余，以及 ZeRO 三阶段分片。 |
| `gpipe-pipeline-parallelism.png` | [GPipe](https://arxiv.org/abs/1811.06965) Figure 2(c) | 说明 micro-batch 如何填充 pipeline bubble。 |
| `chinchilla-isoflop-curves.png` | [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) Figure 4 | 说明固定 FLOP 预算下参数量和 token 数的配平问题。 |

## 来源

| 来源 | 类型 | 本站使用方式 | 关联主题 |
| --- | --- | --- | --- |
| [PyTorch: Optimization Loop](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html) | 官方教程 | 支撑最小训练循环、loss、backward、optimizer step 和 zero_grad 的代码语义。 | 训练 step |
| [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/) | 工程博客 | 支撑从小实验到可信训练的排查顺序和训练异常诊断思路。 | 训练排查 |
| [Smol Training Playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook) | 官方工程手册 | 支撑数据、评测、规模、成本和失败复盘共同组成训练 recipe 的讲法。 | 训练 recipe |
| [SmolLM3](https://huggingface.co/blog/smollm3) | 官方训练复盘 | 支撑多阶段数据、长上下文、多语言和后训练组合的现代小模型训练案例。 | 训练复盘 |
| [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) | 论文 / 图源 | 使用 Figure 4 解释参数、token 和 FLOP 预算配平。 | Scaling law |
| [ZeRO](https://arxiv.org/abs/1910.02054) | 论文 / 图源 | 使用 Figure 1 解释训练状态显存冗余和 ZeRO 三阶段分片。 | 显存 |
| [GPipe](https://arxiv.org/abs/1811.06965) | 论文 / 图源 | 使用 Figure 2(c) 解释 pipeline parallelism 和 micro-batch。 | 并行训练 |
| [Datawhale: VRAM Calculation and ZeRO](https://datawhalechina.github.io/llm-algo-leetcode/01_Hardware_Math_and_Systems/06_VRAM_Calculation_and_ZeRO.html) | 中文教程 | 借鉴把参数、梯度、优化器状态和 ZeRO 显存账拆成手算过程的中文讲法。 | 中文讲法 |
