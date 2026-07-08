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

ELBO 可以写成：

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

![DDPM graphical model](../assets/images/paper-figures/diffusion/ddpm-figure-2-graphical-model.png){ width="820" }

<small>图源：`Denoising Diffusion Probabilistic Models`，Figure 2，本站已本地化。原图展示扩散模型的正向加噪链和反向生成链。本站在这里用它说明：生成模型常把直接建模 \(p(x)\) 改写成一条包含 latent/noise states 的逐步过程。</small>

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
| denoising loss 只是像素误差 | 它常在近似学习分布方向或生成路径 |

## 读完以后怎么判断

看到 latent、prior、posterior、ELBO、score 或 flow matching，先问：随机变量是什么，生成链从哪里开始，推断链看到什么，训练目标是在逼近 likelihood、下界、方向场还是下游 reward，评测是否覆盖采样质量和决策价值。这个问题链会支撑 DDPM、Score SDE、DMD、Dreamer、PlaNet、V-JEPA、JEPA 和视频世界模型论文。

## 相关阅读与下一步

- 站内下一步：[概率、潜变量与生成模型](../foundations/probability-latent-variables-and-generative-models.md)。
- 站内下一步：[Score Matching 到 SDE](../diffusion/score-matching-sde-and-probability-flow.md)。
- 论文精读：[PlaNet](../paper-deep-dives/world-models/planet.md)。
- 论文精读：[Score SDE](../paper-deep-dives/diffusion/score-sde.md)。
