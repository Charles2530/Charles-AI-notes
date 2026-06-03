# 优化与训练基础来源台账

这份台账服务于 [目标函数、优化器与学习率日程](../training/objectives-optimizers-and-schedules.md)，也服务于 [优化与训练入门](../foundations/optimization-and-training-basics.md) 中相同主题的基础解释。正文参考高质量技术博客和中文公众号常用讲法：先从“loss 写出来为什么还不够”切入，再把目标函数、偏好/RL 信号、优化器、学习率日程、稳定项和验收证据拆成训练压力链。事实、公式、论文图和 API 行为以论文与官方文档为准。

## 使用方式

第一类来源是训练过程和调试方法。Karpathy 的 training recipe、Google tuning playbook 和 PyTorch / Hugging Face 文档帮助本站学习“先验证数据和训练闭环，再加复杂度”的写法。正文要强调 init loss、小 batch overfit、baseline、分桶评测和恢复一致性，而不是只列 optimizer 名字。

第二类来源是目标函数与对齐信号。InstructGPT 支撑 SFT、偏好排序、reward model 和 PPO 的数据链；DPO 支撑 chosen / rejected pair 到 closed-form preference objective 的讲法；DeepSeek-R1 支撑 GRPO / RLVR 的组内相对优势解释。

第三类来源是优化机制。SGD、Momentum、Adam、AdamW、Adafactor、PyTorch optimizer 文档和 Hugging Face schedule 文档帮助本站解释更新方向 \(u_t\) 如何由当前梯度、历史统计、二阶矩、weight decay 和 schedule 构成。写 AdamW 时必须说明 decoupled weight decay，而不是笼统说“AdamW 比 Adam 好”。

第四类来源是稳定性和证据链。Loss Landscape、Pascanu gradient clipping、warmup / scheduler 文档和低精度训练资料帮助本站解释 learning rate、初始化、梯度路径、clip threshold、overflow 和 loss spike 的关系。正文要把 clipping 写成限制单步更新幅度，而不是修复目标函数。

第五类来源是讲法参考。中文博客、知乎专栏、微信公众号、OneFlow / 智源等中文技术长文只用于学习结构化表达：用 loss spike、梯度范数、训练日志和配置变更验收讲工程排查。涉及算法定义、API 行为和强 claim 时，正文仍回论文、官方文档或公开技术报告核对。

## 写作边界

正文不把目标函数写成客观真理。Loss 是训练压力的代理，必须和数据桶、真实任务、分桶评测和下游指标一起看。

正文不把 DPO、PPO、GRPO 写成互斥排行榜。它们使用的信号、rollout 成本、稳定项和适用任务不同；页面只解释信号如何进入梯度，不替代 RLHF / RLVR 专题。

正文不把 AdamW 写成默认魔法。AdamW 的关键是自适应更新与 weight decay 解耦；它是否更好仍取决于模型、数据、batch、学习率和正则目标。

正文不把 LR schedule 写成孤立旋钮。Warmup、decay、数据阶段、loss 阶段、effective batch、checkpoint resume 和混合精度状态必须一起看。

正文没有使用 image2 或其他生成式工具作图。所有图片都来自仓库已有资产，并在正文标出原始来源。

## 图片使用

| 图片 | 原始来源 | 本站使用方式 |
| --- | --- | --- |
| `instructgpt-labeler-ranking.png` | [InstructGPT](https://arxiv.org/abs/2203.02155) Appendix Figure 19(b) | 说明偏好数据是多个回答的相对排序，不是 token 级标准答案。 |
| `figure-grpo.png` | [DeepSeek-R1](https://arxiv.org/abs/2501.12948) Supplementary Fig. 1 | 说明 PPO 和 GRPO 的 advantage 来源差异。 |
| `loss-landscape-figure-resnet56.png` | [Visualizing the Loss Landscape of Neural Nets](https://arxiv.org/abs/1712.09913) | 说明 optimizer 在复杂 loss surface 中按梯度、历史统计和学习率逐步移动。 |

## 来源

| 来源 | 类型 | 本站使用方式 | 关联主题 |
| --- | --- | --- | --- |
| [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/) | 技术博客 | 训练是 leaky abstraction；先看数据、baseline、init loss、小 batch overfit，再逐步复杂化。 | 训练调试流程 |
| [Google Deep Learning Tuning Playbook](https://developers.google.com/machine-learning/guides/deep-learning-tuning-playbook) | 官方指南 | 把调参拆成目标、搜索空间、训练长度、评估噪声和证据链。 | 调参方法 |
| [InstructGPT](https://arxiv.org/abs/2203.02155) | 论文 / 图源 | 使用 Appendix Figure 19(b) 解释偏好排序数据；支撑 SFT、reward model 和 PPO 数据链。 | RLHF |
| [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) | 论文 | 支撑 chosen / rejected pair、reference model 和 \(\beta\) 偏好强度的 DPO 公式。 | 偏好学习 |
| [DeepSeek-R1](https://arxiv.org/abs/2501.12948) | 技术报告 / 图源 | 使用 Supplementary Fig. 1 解释 GRPO 与 PPO 的 advantage 来源差异。 | RLVR / GRPO |
| [Visualizing the Loss Landscape of Neural Nets](https://arxiv.org/abs/1712.09913) | 论文 / 图源 | 支撑 optimizer 在复杂 loss surface 中逐步移动的直觉。 | loss landscape |
| [On the difficulty of training recurrent neural networks](https://proceedings.mlr.press/v28/pascanu13.html) | 论文 | 支撑梯度爆炸、梯度裁剪和长序列训练困难的经典来源。 | gradient clipping |
| [Adam](https://arxiv.org/abs/1412.6980) | 论文 | 支撑一阶矩、二阶矩和 bias correction 形成自适应更新。 | Adam |
| [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) | 论文 | 支撑 AdamW 把 weight decay 从 Adam 自适应梯度更新中解耦。 | AdamW |
| [PyTorch AdamW](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html) | 官方文档 | 核对当前工程 API、参数含义和 AdamW 算法描述。 | optimizer 语义 |
| [PyTorch gradient clipping](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html) | 官方文档 | 核对 global norm clipping 的 API 入口和工程语义。 | clipping 实践 |
| [Hugging Face optimization schedules](https://huggingface.co/docs/transformers/main_classes/optimizer_schedules) | 官方文档 | 核对 warmup、linear、cosine、Adafactor 和 schedule API 的工程入口。 | LR schedule |
| [DeepSpeed ZeRO](https://arxiv.org/abs/1910.02054) | 论文 | 支撑 optimizer state、gradient 和 parameter 显存账影响训练配置。 | 训练状态账 |
| [OneFlow 官方博客](https://blog.csdn.net/OneFlow_Official) | 中文工程博客 | 学习用训练日志、系统瓶颈和优化器配置串起大模型训练文章。 | 中文讲法 |
| [智源社区](https://hub.baai.ac.cn/) | 中文社区 / 编译转载 | 学习中文技术长文的结构化讲解；强事实追回原文。 | 中文入口 |
