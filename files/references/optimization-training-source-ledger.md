# 优化与训练基础来源台账

这页记录目标函数、mini-batch 梯度、优化器、学习率日程、梯度裁剪和训练证据链相关来源。它服务的写作原则是：不要把 SGD、AdamW、warmup、clipping 写成术语清单，而要说明 loss 如何通过 noisy gradient 和 optimizer update 变成一次参数更新，以及为什么平均 loss 会误导真实任务判断。

## 使用方式

写优化文章时，先把来源分成四类。

第一类是训练过程和调试方法。Karpathy 的 training recipe、Google tuning playbook 和 Hugging Face/Transformers 文档帮助本站学习“先验证数据和训练闭环，再加复杂度”的写法。正文要强调 init loss、小 batch overfit、baseline、分桶评测和恢复一致性，而不是只列 optimizer 名字。

第二类是优化机制来源。SGD、Momentum、Adam、AdamW、Adafactor 和 PyTorch optimizer 文档帮助本站解释更新方向 \(u_t\) 如何由当前梯度、历史统计、二阶矩和 weight decay 构成。写 AdamW 时必须说明 decoupled weight decay，而不是笼统说“AdamW 比 Adam 好”。

第三类是稳定性来源。Loss Landscape、Pascanu gradient clipping、warmup/scheduler 文档和低精度训练资料帮助本站解释 learning rate、初始化、梯度路径、clip threshold、overflow 和 loss spike 的关系。正文要把 clipping 写成限制单步更新幅度，而不是修复目标函数。

第四类是证据链来源。评测、ablation、分桶、closed-loop 和系统吞吐资料帮助本站说明训练收益要同时看质量、稳定性和成本。平均 loss、单张曲线或单个 demo 不能证明训练 recipe 更好。

## 写作边界

优化基础页不负责展开所有后训练目标，也不负责替代分布式训练、低比特训练和 RLHF 专题。它只讲清共同骨架：目标函数给压力，batch 给梯度估计，学习率给步长，optimizer 改方向，稳定项限制坏更新，证据链验证是否服务真实任务。

中文博客和微信公众号适合学习讲法，尤其是用 loss spike、梯度范数和训练日志讲工程排查；涉及算法定义、API 行为和强 claim 时，正文仍回论文、官方文档或公开技术报告核对。

## 来源

| 来源 | 类型 | 本站学什么 | 落到页面 |
| --- | --- | --- | --- |
| [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/) | 技术博客 | 训练是 leaky abstraction；先看数据、baseline、init loss、小 batch overfit，再逐步复杂化。 | 训练调试流程 |
| [Google Deep Learning Tuning Playbook](https://developers.google.com/machine-learning/guides/deep-learning-tuning-playbook) | 官方指南 | 把调参拆成目标、搜索空间、训练长度、评估噪声和证据链。 | 调参方法 |
| [Visualizing the Loss Landscape of Neural Nets](https://arxiv.org/abs/1712.09913) | 论文 | loss landscape 可视化如何帮助理解优化地形和结构稳定性。 | loss landscape |
| [On the difficulty of training recurrent neural networks](https://proceedings.mlr.press/v28/pascanu13.html) | 论文 | 梯度爆炸、梯度裁剪和长序列训练困难的经典来源。 | gradient clipping |
| [Adam](https://arxiv.org/abs/1412.6980) | 论文 | 一阶矩、二阶矩和 bias correction 如何形成自适应更新。 | Adam |
| [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) | 论文 | AdamW 把 weight decay 从 Adam 自适应梯度更新中解耦。 | AdamW |
| [PyTorch AdamW](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html) | 官方文档 | 当前工程 API、参数含义和 AdamW 算法描述。 | optimizer 语义 |
| [PyTorch gradient clipping](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html) | 官方文档 | global norm clipping 的 API 入口和工程语义。 | clipping 实践 |
| [Hugging Face optimization schedules](https://huggingface.co/docs/transformers/main_classes/optimizer_schedules) | 官方文档 | warmup、linear、cosine、Adafactor 和 schedule API 的工程入口。 | LR schedule |
| [DeepSpeed ZeRO](https://arxiv.org/abs/1910.02054) | 论文 | optimizer state、gradient 和 parameter 显存账影响训练配置。 | 训练状态账 |
| [OneFlow 官方博客](https://blog.csdn.net/OneFlow_Official) | 中文工程博客 | 学习用训练日志、系统瓶颈和优化器配置串起大模型训练文章。 | 中文讲法 |
| [智源社区](https://hub.baai.ac.cn/) | 中文社区 / 编译转载 | 学习中文技术长文的结构化讲解；强事实追回原文。 | 中文入口 |
