# 潜变量、似然与生成模型目标

这篇回答一个问题：latent、prior、posterior、likelihood、ELBO、score matching、denoising loss 和 flow matching 为什么会同时出现在 VAE、扩散、世界模型和表征学习论文里。

生成模型不是只问“标签是什么”，而是问“数据是从哪个分布、哪个隐藏状态、哪条生成过程里来的”。这会让公式里同时出现可见变量 \(x\)、潜变量 \(z\)、噪声状态 \(x_t\)、先验、后验和各种下界或匹配目标。

## 潜变量：看不见但解释观测的因素

潜变量模型常写成：

\[
z\sim p(z),\quad x\sim p_\theta(x\mid z)
\]

这里 \(z\) 是隐藏原因，\(x\) 是观测。生成方向是先抽 \(z\)，再由 decoder 或生成模型产生 \(x\)。

在图像里，\(z\) 可能编码姿态、光照、身份和风格；在世界模型里，\(z_t\) 可能编码当前 belief state；在视频扩散里，latent 可能是压缩后的图像网格，而不是语义变量。读论文时不要把 latent 当成固定含义，要问它服务重建、生成、预测、压缩还是规划。

## 先验和后验：生成前相信什么，观察后相信什么

先验：

\[
p(z)
\]

表示还没看到当前样本前，对隐藏变量的默认分布。

后验：

\[
p_\theta(z\mid x)
\]

表示看到样本 \(x\) 后，对隐藏变量 \(z\) 的更新信念。根据贝叶斯公式：

\[
p_\theta(z\mid x)
=
\frac{p_\theta(x\mid z)p(z)}
{p_\theta(x)}
\]

难点在分母：

\[
p_\theta(x)
=
\int p_\theta(x\mid z)p(z)\,dz
\]

高维神经生成模型里，这个积分通常不可算，所以需要近似推断。

## 似然：模型解释数据的能力

最大似然训练希望：

\[
\max_\theta \sum_i \log p_\theta(x_i)
\]

直觉是让真实数据在模型分布下更可能。但很多生成模型不会直接计算完整 \(p_\theta(x)\)。自回归模型用条件概率连乘；VAE 用 ELBO；扩散用加噪去噪目标近似分布学习；flow matching 学连续路径的速度场。

所以看到 likelihood 时要问：它是显式可算的，还是被某个下界、代理 loss 或采样路径替代了。

## ELBO：把不可算似然换成可优化下界

VAE 引入近似后验：

\[
q_\phi(z\mid x)
\]

先把边际似然乘除同一个 \(q_\phi(z\mid x)\)：

\[
p_\theta(x)
=
\int
q_\phi(z\mid x)
\frac{
p_\theta(x,z)
}{
q_\phi(z\mid x)
}
\,dz
\]

也就是：

\[
\log p_\theta(x)
=
\log
\mathbb E_{q_\phi(z\mid x)}
\left[
\frac{
p_\theta(x,z)
}{
q_\phi(z\mid x)
}
\right]
\]

因为 \(\log\) 是凹函数，Jensen 不等式给出：

\[
\log
\mathbb E_q[R]
\ge
\mathbb E_q[\log R]
\]

所以：

\[
\log p_\theta(x)
\ge
\mathbb E_{q_\phi(z\mid x)}
\left[
\log
\frac{
p_\theta(x,z)
}{
q_\phi(z\mid x)
}
\right]
\equiv
\mathcal L_{\mathrm{ELBO}}(x)
\]

“lower bound” 就来自这一步：先在 log 外取期望，再把 log 移到期望里面，会得到不超过真实 log likelihood 的量。把联合分布 \(p_\theta(x,z)=p_\theta(x\mid z)p(z)\) 展开，ELBO 常写成：

\[
\mathcal{L}_{\text{ELBO}}(x)
=
\mathbb{E}_{q_\phi(z\mid x)}
\left[
\log p_\theta(x\mid z)
\right]
-
D_{\mathrm{KL}}
\left(
q_\phi(z\mid x)\,\|\,p(z)
\right)
\]

第一项让 latent 能解释当前样本，第二项把每个样本的 posterior 拉回统一 prior。KL 太弱，latent 空间可能碎掉，采样差；KL 太强，decoder 可能忽略 latent，出现 posterior collapse。

训练时还要从 \(q_\phi(z\mid x)\) 采样。对高斯 posterior，可以写成：

\[
\epsilon\sim\mathcal N(0,I),
\qquad
z
=
\mu_\phi(x)
+
\sigma_\phi(x)\odot\epsilon
\]

这一步把随机性搬到与参数无关的 \(\epsilon\)，让 reconstruction term 的梯度能沿 \(z\) 回到 encoder。ELBO 本身说明“优化什么下界”，重参数化说明“这个随机目标怎样反向传播”；更完整的训练信号比较见[梯度估计、离散变量与训练信号](../foundations/gradient-estimators-and-training-signals.md)。

![DDPM graphical model](../assets/images/paper-figures/diffusion/ddpm-figure-2-graphical-model.png){ width="820" }

<small>图源：`Denoising Diffusion Probabilistic Models`，Figure 2，本站已本地化。原图展示扩散模型的正向加噪链和反向生成链。本站在这里用它说明：生成模型常把直接建模 \(p(x)\) 改写成一条包含 latent/noise states 的逐步过程。</small>

## Energy-Based Model：先学兼容性，再决定怎样归一化和搜索

有些模型不直接输出规范化概率，而是学习能量：

\[
E_\theta(x,y)
\]

给定输入 \(x\)，兼容的 \(y\) 应该有较低能量，不兼容的 \(y\) 应该有较高能量。如果要把它写成条件概率，可以定义：

\[
p_\theta(y\mid x)
=
\frac{
\exp(-E_\theta(x,y))
}{
Z_\theta(x)
}
\]

其中：

\[
Z_\theta(x)
=
\int
\exp(-E_\theta(x,y'))
\,dy'
\]

叫 partition function。分子只给出未归一化相对权重，分母把所有可能 \(y'\) 的权重加起来，才让概率积分为 1。高维连续输出里 \(Z_\theta(x)\) 往往很难计算，这正是 EBM 灵活但训练和采样不轻松的地方。

EBM 的推理可以是：

\[
y^*
=
\arg\min_y E_\theta(x,y)
\]

也可以通过 sampling 找多个低能量解。它适合表达“一种输入对应多个合理输出”：不必把多种未来平均成一个点，只要给不同合理区域较低能量。

但 energy function 不是训练 loss 的同义词。只降低真实样本能量，会让所有位置都变低，形成 energy collapse。训练必须同时塑造不合理区域，常见路线包括 negative / contrastive samples，或对表示方差、协方差和 latent capacity 加约束。H-JEPA 里出现的 latent EBM 就建立在这个通用视角上；论文特有架构留到[H-JEPA 精读](../paper-deep-dives/world-models/h-jepa.md)展开。

可以先这样区分三类对象：

| 对象 | 学到什么 | 推理时做什么 |
| --- | --- | --- |
| 显式 likelihood model | 规范化 \(p_\theta(x)\) 或条件概率 | 直接算概率或按分解采样 |
| EBM | 未归一化兼容性 \(E_\theta(x,y)\) | 优化或采样寻找低能量状态 |
| Score-based model | 数据空间里的 \(\nabla_x\log p_t(x)\) | 沿 SDE / ODE 或采样器更新样本 |

## Score：概率密度地形的方向

score 定义为：

\[
\nabla_x \log p_t(x)
\]

它表示在噪声水平 \(t\) 下，样本 \(x\) 往哪个方向移动会让 log density 上升。扩散模型看起来在做噪声预测或 denoising MSE，本质上是在不同噪声水平学习这个概率方向。

如果模型学到的 score 不准，采样路径就会偏；如果少步蒸馏让某些噪声阶段退化，生成会出现结构、纹理或动态上的问题。DMD、DMD2、Score SDE、Phased DMD 都能从这个角度读。

## Flow matching：学习从噪声到数据的速度场

Flow matching 常把生成看成连续路径：

\[
\frac{dx_t}{dt}
=
v_\theta(x_t,t)
\]

这里 \(dx_t/dt\) 表示样本状态随连续时间的瞬时变化率，\(v_\theta(x_t,t)\) 表示模型在当前位置和时间给出的速度方向。

训练目标让模型速度 \(v_\theta\) 接近某个目标速度 \(u_t\)：

\[
\mathcal{L}_{\text{FM}}
=
\mathbb{E}_{t,x_t}
\left[
\|v_\theta(x_t,t)-u_t\|^2
\right]
\]

这和 score matching 的读法相近：模型不是直接输出最终样本，而是在每个时间点告诉采样器下一步往哪里走。Rectified Flow、flow matching action head、连续时间视频生成都在使用这类语言。

## 世界模型里的 latent：要服务决策

世界模型里的 latent state 常写成：

\[
p_\theta(z_{t+1}\mid z_t,a_t)
\]

这里 latent 不只是压缩图像。它要保留未来预测、动作后果、reward、risk 和 planning 需要的信息。

![PlaNet latent dynamics](../assets/images/paper-deep-dives/world-models/planet/figure-2-latent-dynamics.png){ width="820" }

<small>图源：`PlaNet`，Figure 2，本站已本地化。原图比较不同 latent dynamics 设计。本站在这里用它说明：世界模型里的 latent state 既要压缩历史，也要表达可 rollout 的不确定未来。</small>

## 常见误读

| 误读 | 更稳的理解 |
| --- | --- |
| latent 就是压缩向量 | latent 可能是生成原因、belief state、噪声状态或动作相关状态 |
| 重建好就代表 latent 好 | 重建可能保留纹理，却丢掉动作、接触和风险 |
| likelihood 高就一定样本好 | likelihood、视觉质量和下游决策价值不完全等价 |
| 低 energy 就已经是规范化概率 | 还需要 partition function；很多 EBM 推理直接使用相对能量 |
| energy function 就是训练时直接最小化的 loss | 训练要塑造正负区域的能量差，不能把所有位置一起压低 |
| denoising loss 只是像素误差 | 它常在近似学习分布方向或生成路径 |

## 读完以后怎么判断

看到 latent、prior、posterior、ELBO、energy、score 或 flow matching，先问：随机变量是什么，生成链从哪里开始，推断链看到什么，训练目标是在逼近 likelihood、下界、兼容性、方向场还是下游 reward，评测是否覆盖采样质量和决策价值。这个问题链会支撑 VAE、EBM、DDPM、Score SDE、DMD、Dreamer、PlaNet、V-JEPA、JEPA 和视频世界模型论文。

## 相关阅读与下一步

- 站内下一步：[概率、潜变量与生成模型](../foundations/probability-latent-variables-and-generative-models.md)。
- 站内下一步：[Score Matching 到 SDE](../diffusion/score-matching-sde-and-probability-flow.md)。
- 论文精读：[PlaNet](../paper-deep-dives/world-models/planet.md)。
- 论文精读：[H-JEPA](../paper-deep-dives/world-models/h-jepa.md)。
- 论文精读：[Score SDE](../paper-deep-dives/diffusion/score-sde.md)。
