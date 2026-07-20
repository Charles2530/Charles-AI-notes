# 前置数学知识

前置数学知识帮助你把 AI 文章里的公式翻译回机制：谁在变化，谁是随机的，谁被优化，误差怎样传播，以及这些因素如何影响质量、显存、延迟或闭环行为。

1. 如果想系统学习，按下面五组从前往后读。
2. 如果已经在某篇博客或论文里卡住，直接用“从下游反查”找到最短补课路径。

## 数学语言与变化

这一组先回答“对象放在哪个空间里”和“输入动一点，输出怎样变”。它是 attention、自动微分、优化、量化和低秩方法的共同起点。

| 你要解决的问题 | 核心工具 | 页面 |
| --- | --- | --- |
| embedding、矩阵乘、投影、范数和低秩发生在哪个空间 | 向量、矩阵、内积、秩 | [线性代数与几何直觉](linear-algebra-and-geometry.md) |
| SVD、谱能量、Newton-Schulz、Hessian 和 Fisher 怎样描述方向结构 | 特征值、奇异值、Taylor、二阶敏感度 | [谱分解、二阶近似与低秩结构](spectral-second-order-and-low-rank.md) |
| 梯度、积分、链式法则、ODE/SDE 和指数积分器怎样连接 | 导数、积分、连续动力学 | [微积分与牛顿-莱布尼茨公式](calculus-and-newton-leibniz.md) |
| 向量函数的扰动怎样传播，backward 为什么算 VJP | Jacobian、JVP、VJP、局部线性化 | [微分、Jacobian 与梯度传播](differential-jacobian-and-gradient-flow.md) |

## 概率、统计与信息

这一组回答“数据和未来有多不确定”“有限样本结论有多可信”以及“两个分布差多少”。

| 你要解决的问题 | 核心工具 | 页面 |
| --- | --- | --- |
| 随机变量、条件概率、贝叶斯、方差和协方差怎么连起来 | 分布、期望、协方差矩阵 | [概率论、分布与期望](probability-distributions-and-expectation.md) |
| Monte Carlo、置信区间、bootstrap、importance weight 和 calibration 怎样读 | 偏差、方差、Wilson、ESS、ECE | [采样、估计与校准](sampling-estimation-and-calibration.md) |
| 熵、交叉熵、KL、互信息和 perplexity 为什么能写进 loss | 信息量和分布距离 | [信息论、熵与分布距离](information-theory-and-divergence.md) |
| 一串随机状态怎样组成 trajectory，马尔可夫假设在说什么 | 转移概率、随机过程、MDP | [随机过程与马尔可夫性](stochastic-processes-and-markov.md) |

## 优化与学习目标

这一组把“模型应该学什么”变成可计算目标，再解释连续更新、离散选择、生成目标和长期收益分别怎样优化。

| 你要解决的问题 | 核心工具 | 页面 |
| --- | --- | --- |
| loss 怎样变成参数更新，momentum、Adam 和约束在改什么 | 梯度下降、曲率、正则化 | [优化、梯度与约束](optimization-and-gradient-methods.md) |
| prior、posterior、Jensen、ELBO、energy、score 和 flow 怎样统一 | 潜变量、下界、EBM、方向场 | [潜变量、似然与生成模型目标](latent-likelihood-and-generative-objectives.md) |
| top-k、beam、MCTS、router 和 load balancing 怎样分配选择预算 | 离散搜索、gating、熵与容量 | [离散选择、Top-k 与稀疏路由](discrete-selection-topk-and-routing.md) |
| return、value、Bellman、lambda-return、PPO 和 GRPO 怎样支撑决策 | 价值递推、advantage、策略梯度 | [价值函数、策略梯度与强化学习目标](value-policy-and-rl-objectives.md) |

## 时序、控制与空间

这一组把孤立样本接成序列、轨迹和三维世界，服务 Transformer、世界模型、VLA 和机器人视觉。

| 你要解决的问题 | 核心工具 | 页面 |
| --- | --- | --- |
| token 顺序、RoPE、causal mask、temperature 和 KV cache 怎样读 | 序列索引、位置、可见性 | [序列、位置编码与注意力掩码](sequence-position-and-masking.md) |
| 状态怎样演化，MPC/CEM 怎样用世界模型选择动作 | 状态空间、稳定性、闭环控制 | [动态系统、控制与状态空间](dynamic-systems-and-control.md) |
| kernel、Fourier、DCT、卷积定理、SNR 和多尺度为什么一起出现 | 信号、频率、噪声尺度 | [卷积、频域与多尺度表示](convolution-frequency-and-multiscale.md) |
| 相机内外参、深度、四元数、对极几何、位姿和点云怎样连接 | 投影几何、SE(3)、多视角约束 | [几何、视觉与三维空间](geometry-vision-and-3d.md) |

## 研究判断与工程

最后一组不直接定义模型层，而是帮助判断实验能否外推、曲线是否可信、方法在机器上是否划算。

| 你要解决的问题 | 核心工具 | 页面 |
| --- | --- | --- |
| correlation、intervention、counterfactual、OOD 和 train-test mismatch 怎样区分 | 因果语言、分布偏移 | [因果、反事实与分布偏移](causality-counterfactual-and-distribution-shift.md) |
| scaling law、log-log、边际收益、Pareto 和 ablation 怎样读 | 幂律、外推、实验曲线 | [缩放律、幂律与实验曲线](scaling-laws-and-experimental-curves.md) |
| Big-O、FLOPs、bytes、浮点误差、条件数和 benchmark 怎样落到系统 | 数量级、数值稳定性、成本估算 | [数值计算、复杂度与估算](numerics-complexity-and-estimation.md) |

## 一张图上的主依赖

~~~mermaid
flowchart LR
  A["线性代数"] --> B["微积分"]
  B --> C["Jacobian / JVP / VJP"]
  A --> D["谱分解"]
  C --> E["自动微分与优化"]
  C --> F["误差传播"]
  D --> G["Hessian / Fisher"]
  F --> H["量化与压缩"]
  G --> H

  P["概率论"] --> Q["采样与估计"]
  P --> R["信息论"]
  P --> S["随机过程"]
  P --> T["潜变量与生成目标"]
  B --> T
  T --> U["扩散与世界模型"]
  S --> V["价值、RL 与控制"]
  V --> W["VLA 与具身闭环"]

  X["序列与掩码"] --> E
  Y["卷积与频域"] --> Z["几何与三维"]
  Z --> W
~~~

这张图只保留主干。线性代数和概率论是两条起点：前者描述表示和变化方向，后者描述不确定性和分布。微积分与 Jacobian 把局部变化接到自动微分和误差传播；随机过程、价值函数与控制把一步预测接成长期决策；卷积、频域和几何把像素接到三维状态。

## 从下游专题反查

| 正在读的方向 | 建议先补的数学 |
| --- | --- |
| 训练、优化、量化与模型压缩 | [线性代数](linear-algebra-and-geometry.md) → [Jacobian](differential-jacobian-and-gradient-flow.md) → [谱、Hessian 与 Fisher](spectral-second-order-and-low-rank.md) → [数值与复杂度](numerics-complexity-and-estimation.md) |
| 扩散、Flow Matching 与少步生成 | [微积分与 ODE/SDE](calculus-and-newton-leibniz.md) → [概率](probability-distributions-and-expectation.md) → [潜变量、score 与 flow](latent-likelihood-and-generative-objectives.md) → [卷积、SNR 与尺度](convolution-frequency-and-multiscale.md) |
| 强化学习、Dreamer 与世界模型 | [概率](probability-distributions-and-expectation.md) → [随机过程](stochastic-processes-and-markov.md) → [价值与策略](value-policy-and-rl-objectives.md) → [动态系统与控制](dynamic-systems-and-control.md) |
| VLM、VLA、3D 视觉与机器人 | [线性代数](linear-algebra-and-geometry.md) → [序列与掩码](sequence-position-and-masking.md) → [三维几何](geometry-vision-and-3d.md) → [控制](dynamic-systems-and-control.md) → [因果与分布偏移](causality-counterfactual-and-distribution-shift.md) |
| 推理系统、MoE、算子与性能 | [离散选择](discrete-selection-topk-and-routing.md) → [数值与复杂度](numerics-complexity-and-estimation.md) → [采样与校准](sampling-estimation-and-calibration.md) → [实验曲线](scaling-laws-and-experimental-curves.md) |
