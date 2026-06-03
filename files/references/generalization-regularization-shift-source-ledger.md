# 泛化、正则化与分布偏移来源台账

## 使用方式

这份台账服务于 [泛化、正则化与分布偏移](../foundations/generalization-regularization-and-distribution-shift.md)。正文的目标不是罗列正则化技巧，而是把训练风险、验证证据、测试/部署风险和分布偏移区分清楚。

第一类来源是泛化现象论文。`Rethinking Generalization` 用来说明深度网络可以记忆随机标签但仍能在真实标签上泛化；`Deep Double Descent` 用来说明现代容量曲线不能用简单的“模型越大越过拟合”概括。

第二类来源是正则化方法论文。Dropout、AdamW、Mixup、Label Smoothing 这些来源帮助本站解释每个方法限制了哪种捷径，而不是把它们写成固定 checklist。

第三类来源是分布偏移和 benchmark 论文。WILDS、Dataset Cartography、Model Selection under Distribution Shift 等资料用来支撑“平均分不够，必须按环境、来源、时间和风险分桶”的写法。

第四类来源是中文讲法和官方教程。《动手学深度学习》适合补 train/validation 曲线、欠拟合、过拟合、weight decay 和 dropout 的直觉；Google Machine Learning Crash Course 可作为验证集、测试集和泛化概念的英文入门对照。

## 写作边界

正文不把 train-val gap 当成自动诊断。gap 是信号，根因可能是过拟合，也可能是 split 难度不同、验证集污染、近重复泄漏或部署分布变化。

正文不把正则化写成越多越好。每个正则化方法都要同时说明它想限制什么、可能伤害什么，以及为什么需要分桶验证。

正文不把 OOD 写成一个抽象标签。必须落到输入偏移、标签偏移、条件关系变化、时间偏移和选择偏差这些可检查的分布差异。

## 来源

| 来源 | 类型 | 本站使用方式 | 关联主题 |
| --- | --- | --- | --- |
| [Understanding Deep Learning Requires Rethinking Generalization](https://arxiv.org/abs/1611.03530) | 论文 | 说明深度网络的记忆能力和泛化表现之间不能用传统容量直觉直接等同。 | 泛化现象 |
| [Deep Double Descent](https://arxiv.org/abs/1912.02292) | 论文 | 使用 Figure 1 和 double descent 结果解释现代容量曲线。 | double descent |
| [WILDS: A Benchmark of in-the-Wild Distribution Shifts](https://arxiv.org/abs/2012.07421) | 论文 | 支撑“真实世界分布偏移需要按环境/group 评测”的写法。 | 分布偏移 |
| [Distributionally Robust Neural Networks](https://arxiv.org/abs/1911.08731) | 论文 | 补充分布鲁棒优化和 worst-group 风险视角。 | group robustness |
| [Dataset Cartography](https://arxiv.org/abs/2009.10795) | 论文 | 支撑按训练动态分析 easy/ambiguous/hard 样本的想法。 | 数据分桶 |
| [Dropout](https://jmlr.org/papers/v15/srivastava14a.html) | 论文 | 解释 dropout 随机屏蔽激活、减少 co-adaptation 的原始动机。 | dropout |
| [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) | 论文 | 支撑 AdamW 与 L2 penalty 在自适应优化器中不等价的说明。 | weight decay |
| [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412) | 论文 | 解释 mixup 通过线性插值样本和标签改变经验风险。 | augmentation |
| [When Does Label Smoothing Help?](https://arxiv.org/abs/1906.02629) | 论文 | 解释 label smoothing 对过度自信、表示和蒸馏的影响边界。 | label smoothing |
| [ImageNet-trained CNNs are biased towards texture](https://arxiv.org/abs/1811.12231) | 论文 | 说明模型可能利用纹理、背景等捷径，需要分桶和增强验证。 | shortcut learning |
| [Google ML Crash Course: Overfitting](https://developers.google.com/machine-learning/crash-course/overfitting/overfitting) | 官方教程 | 参考 train/test、复杂度、泛化概念的入门解释方式。 | 官方教程 |
| [动手学深度学习：模型选择、欠拟合和过拟合](https://zh.d2l.ai/chapter_multilayer-perceptrons/underfit-overfit.html) | 中文教材 | 学习面向中文读者讲 train/validation 曲线与模型选择的方式。 | 中文讲法 |
| [动手学深度学习：权重衰减](https://zh.d2l.ai/chapter_multilayer-perceptrons/weight-decay.html) | 中文教材 | 补 weight decay 作为 L2 正则的基础推导。 | weight decay |
| [动手学深度学习：暂退法](https://zh.d2l.ai/chapter_multilayer-perceptrons/dropout.html) | 中文教材 | 补 dropout 的中文直觉和代码视角。 | dropout |
