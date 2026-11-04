# 数学工具

这里把数学放回具体计算。矩阵乘法解释特征怎样组合，导数解释参数怎样影响损失，概率解释未知结果如何平均，动力学解释状态怎样随时间变化。每篇都包含可复算的数值或推导，可以从当前公式涉及的对象进入。

## 数组、方向与变化

| 当前不清楚的问题 | 入口 | 正文里实际计算什么 |
| --- | --- | --- |
| 一个矩阵保留哪些输入差别 | [线性代数](linear-algebra-and-geometry.md) | 两个样本经同一个矩阵投影，秩如何丢掉方向 |
| 局部变化率怎样形成一次更新或轨迹 | [微积分](calculus-and-newton-leibniz.md) | 回归梯度、Euler误差、SDE噪声尺度 |
| 向量函数怎样反向求导 | [Jacobian](differential-jacobian-and-gradient-flow.md) | 同一二维映射的JVP、VJP和差分检查 |
| 为什么保留90%谱能量仍可能丢任务 | [谱与二阶结构](spectral-second-order-and-low-rank.md) | 截断SVD、Hessian方向、Fisher与KL |
| 同一局部窗口怎样改变信号 | [卷积与频域](convolution-frequency-and-multiscale.md) | 四点信号、相位、循环滤波与混叠 |

只想先读懂公式记号，可以从[一个分类器的全部符号](../foundations/symbols-and-minimal-math.md)开始。需要写张量代码时，再对照[轴、广播和存储](../foundations/tensors-shapes-and-computation-graphs.md)。

## 分布、目标与有限样本

| 当前的问题 | 入口 | 关键计算 |
| --- | --- | --- |
| 为什么测试比例改变会影响成功率 | [概率与期望](probability-distributions-and-expectation.md) | 联合表、条件概率、分层权重 |
| CE和KL为何数值不同却可能梯度相同 | [信息论](information-theory-and-divergence.md) | 同一p/q上的熵、交叉熵和KL |
| 80/100次成功到底有多可信 | [采样与校准](sampling-estimation-and-calibration.md) | Wilson区间、重要性权重、概率校准 |
| 看不到隐藏原因，怎样训练生成器 | [潜变量目标](latent-likelihood-and-generative-objectives.md) | 两个隐藏状态的后验、ELBO与精确间隙 |
| 两组数据都胜出，汇总却落后 | [因果与偏移](causality-counterfactual-and-distribution-shift.md) | 分层反转、调整权重与识别条件 |

概率期望需要明确抽样来源，损失需要明确归约单位。公式里的一个平均号，可能对应样本、token、轨迹或模型自己的输出，这些对象不能互换。

## 优化、选择与时间

| 当前的问题 | 入口 | 关键计算 |
| --- | --- | --- |
| 学习率为何能让负梯度也增大loss | [优化方法](optimization-and-gradient-methods.md) | 二次曲面的稳定步长、Adam和约束 |
| Top-k、采样和专家路由差在哪里 | [离散选择](discrete-selection-topk-and-routing.md) | 分数、集合、权重、容量与搜索 |
| 标签右移和因果mask怎么对齐 | [序列与位置](sequence-position-and-masking.md) | 同一短序列的输入、目标和缓存 |
| 带伞观测如何改变天气预测 | [马尔可夫过程](stochastic-processes-and-markov.md) | 转移矩阵、多步分布和belief更新 |
| 奖励怎样变成策略更新 | [RL目标](value-policy-and-rl-objectives.md) | 两步return、TD、GAE和clipping |
| 计划遇到误差后怎样修正 | [动态系统与控制](dynamic-systems-and-control.md) | 两步MPC、反馈稳定、可观测性 |

## 空间与计算成本

[几何与三维](geometry-vision-and-3d.md)从一个点的投影与反投影解释相机、位姿和深度。[数值与复杂度](numerics-complexity-and-estimation.md)用同一矩阵乘法算FLOPs、字节与性能下界。[缩放曲线](scaling-laws-and-experimental-curves.md)解释如何从有限实验估计资源收益，以及为什么不能随意外推。

这些工具分别约束不同结论。几何往返正确不代表标定无误，复杂度低不保证运行快，拟合曲线漂亮也不保证超出观测范围后仍成立。正文会把相应条件放在推导旁边。
