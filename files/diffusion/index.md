# 扩散模型路线图

扩散模型可以看成一类“先破坏，再恢复”的生成模型。训练时，模型不断看到被噪声污染的数据；推理时，模型学习把纯噪声一步步还原成目标样本。

DDPM 原论文里的 graphical model 很适合建立第一层直觉：左到右的 \(q\) 是人为定义的前向加噪链，右到左的 \(p_\theta\) 是模型要学习的反向生成链。读扩散模型时，先把这两条方向分清，后面的噪声预测、采样器、蒸馏和整流才不会混在一起。

![DDPM graphical model 原论文图](../assets/images/paper-figures/diffusion/ddpm-figure-2-graphical-model.png){ width="620" }

<small>图源：[Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)，Figure 2。原论文图意：DDPM 使用固定的正向扩散过程 \(q\) 将数据逐步推向噪声，再学习参数化反向过程 \(p_\theta\) 从噪声逐步生成数据。</small>

!!! note "图解：DDPM 图里的两条方向"
    左到右的 \(q\) 是前向加噪链：它不需要模型学习，只是人为定义的破坏过程，把 \(x_0\) 一步步推向接近高斯噪声的 \(x_T\)。右到左的 \(p_\theta\) 才是模型要学习的反向去噪链：在每个噪声强度下，估计怎样从 \(x_t\) 回到更干净的 \(x_{t-1}\)。训练目标回答“怎样估计噪声或干净样本”，采样器回答“推理时沿什么路径走回去”，两者不是同一层问题。

!!! tip "基础知识入口"
    如果你对 `UNet`、`DiT`、`score`、`ELBO`、`SDE/ODE` 或 `Cross-Attention` 还不熟，可以先看 [卷积与特征提取](../foundations/convolution-and-feature-extraction.md)、[Transformer 与 Attention](../foundations/transformer-attention-and-tokenization.md) 和 [概率与潜变量模型](../foundations/probability-latent-variables-and-generative-models.md)。这些是扩散模型后续章节反复使用的公共概念。

!!! note "放到世界模型主线里怎么理解"
    扩散模型本身不是世界模型，它首先是一类生成分布的方法。但视频世界模型常会借用扩散/flow 的视频生成能力，再补动作条件、时间因果性、长期记忆和实时推理。读扩散专题时，主线应放在“生成路径、条件控制、少步蒸馏和视频扩散如何成为世界模拟器底座”，而不是只停在图像生成效果。

## 先按三组读

| 分组 | 核心问题 | 先看页面 |
| --- | --- | --- |
| 机制与训练 | 加噪、去噪、score、SDE/ODE、噪声日程和 conditioning 到底在学什么 | 演进脉络、训练、Score/SDE、噪声日程、Guidance |
| 采样、蒸馏与视频 | 怎么少步生成，如何把图像扩散推到视频和多模态 | 推理采样、蒸馏、Consistency / Rectified Flow、视频扩散 |
| 实践与对照 | 训练失败、配方取舍和路线差异如何判断 | 实战配方、方法对照表 |

## 一个统一的数学骨架

设原始样本为 \(x_0 \sim q_{\text{data}}(x)\)。前向加噪过程写成：

\[
q(x_t \mid x_{t-1}) = \mathcal{N}\left(x_t; \sqrt{1-\beta_t}\,x_{t-1}, \beta_t I\right)
\]

把它展开到任意时刻 \(t\)，可以直接写成：

\[
q(x_t \mid x_0) = \mathcal{N}\left(x_t; \sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)I\right)
\]

其中：

\[
\alpha_t = 1 - \beta_t,\qquad \bar{\alpha}_t = \prod_{s=1}^{t}\alpha_s
\]

这意味着我们可以一次采样到任意噪声时刻：

\[
x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon,\qquad \epsilon \sim \mathcal{N}(0, I)
\]

这条式子非常重要，因为它把“长链加噪”变成了“直接构造带噪样本”的监督学习问题。

## 为什么它有效

直觉上，扩散模型把一个很难直接学的高维分布，拆成很多个更容易学的局部恢复问题。噪声很大时，模型先学粗结构；噪声变小时，模型再补细节和纹理。这有点像修复一张被雾化的照片：先判断画面里大概是人、树、房子，还是猫、桌子、杯子，再慢慢把轮廓、边缘和纹理补回来。

## 四个最重要的视角

理解扩散模型时，最好不要只盯某一篇论文，而是始终从四个视角来读：

### 1. 建模视角

数据如何被逐步加噪、又如何被逐步恢复。

### 2. 参数化视角

这一视角关心模型到底在预测什么。DDPM 主线常用噪声预测，直接估计加入到 \(x_t\) 里的高斯噪声 \(\epsilon\)，实现简单且训练稳定；原图预测直接回归干净样本 \(x_0\)，直觉清楚但高噪声区更难；score 预测估计 \(\nabla_x \log p_t(x)\)，也就是把样本推向高概率数据区域的方向；velocity 预测则估计信号与噪声混合路径上的速度 \(v\)，常用于 latent diffusion、Flow Matching 风格训练和少步采样，尺度更平衡。

### 3. 求解器视角

推理过程能否被看成 <a class="term-tip" href="score-matching-sde-and-probability-flow/#6-probability-flow-ode" data-tip="ODE：Ordinary Differential Equation，常微分方程。扩散里常指不再额外注入随机噪声、沿 probability flow 连续轨迹求解。">ODE</a>/<a class="term-tip" href="score-matching-sde-and-probability-flow/#4-sde" data-tip="SDE：Stochastic Differential Equation，随机微分方程。扩散里常指带随机噪声项的连续时间加噪或反向生成过程。">SDE</a> 的数值积分问题。

!!! tip "术语快释"
    `ODE` 更像沿着一条确定路线开车，给定起点和方向场后轨迹可重复；`SDE` 更像开车时还不断受到随机风扰动，同一方向场下也可能走出不同轨迹。扩散采样器把这两类连续过程离散成有限步，因此会出现 `Euler`、`Heun`、`DPM-Solver` 等求解器。

### 4. 加速视角

这一视角关心速度提升究竟来自哪里。更好的采样器是在同一模型上用更准确的数值更新减少采样误差；更少的步数是直接压缩反向过程长度，用更少网络前向换速度；蒸馏把多步教师的能力迁移到学生模型上，将推理成本转移到离线训练；路径整流则把噪声到数据的轨迹训练得更直，让粗步长也更可用。

这四个视角串起来，基本就能覆盖从 DDPM 到 DMD2 / Rectified Diffusion 的主线。

## 三类核心问题

理解扩散模型时，可以一直抓住三个问题：

### 1. 训练什么

最常见的是预测噪声 \(\epsilon_\theta(x_t, t)\)。除此之外，也可以直接回归无噪声样本 \(x_0\)，估计当前噪声分布的对数密度梯度 score \(\nabla_{x_t} \log q(x_t)\)，或用 velocity \(v\) 在混合坐标里同时编码信号和噪声方向。这里要注意，`v-prediction` 是扩散参数化，`Flow Matching` 是直接拟合连续速度场的训练框架，二者经常相关但不是同一个概念。

### 2. 怎么采样

采样可以理解为解一个逆向 SDE 或 probability flow ODE，因此会出现一批求解器。`DDIM` 改变采样路径，在不重训模型的前提下减少步数；`Euler` 是简单快速的一阶积分方法，但低步数时误差更明显；`Heun` 用二阶校正先预测再修正，通常比 Euler 更平滑；`DPM-Solver` 则利用扩散 ODE 结构做专用高阶求解，常用于少步高质量采样。

### 3. 怎么加速

主要有两条路线：一种不重训模型，只换求解器；另一种重新训练学生模型，把几十步压到几步甚至一步，例如 `DMD2`、`Phased DMD`、`Rectified Diffusion`。

如果再细一点，可以把整条路线拆成四类方法：

| 类别 | 代表方法 | 本质 |
| --- | --- | --- |
| 原始建模 | DDPM | 学习逐步去噪的反向过程 |
| 少步采样 | DDIM | 改采样轨迹而不重训 |
| 数值求解器 | Euler、Heun、DPM-Solver | 把采样视作 ODE/SDE 求解 |
| 一步生成、蒸馏与整流 | DMD、DMD2、Phased DMD、Rectified Diffusion | 重训以实现极少步或一步生成 |

## 一个生动例子：从噪声里“雕刻”柴犬头像

假设我们要生成一张“戴红围巾的柴犬正脸照片”。在 \(t=T\) 时，输入只是纯高斯噪声，什么也看不出来；前几步去噪后，模型先决定圆形头部、两只耳朵和正脸构图；中间阶段逐渐出现“狗脸”“围巾”“背景虚化”等语义；后期阶段再修胡须、毛发、眼球高光和围巾褶皱。

**扩散模型的强项就在这里**：它天然适合把“全局语义 -> 中观结构 -> 局部细节”拆成多个阶段来学。

### 再换一个例子：城市夜景海报

如果提示词是“下雨的赛博朋克街道，远处有霓虹招牌和出租车”，扩散过程通常会先确定透视结构和大块亮暗区域，再形成道路、楼体、招牌和车辆轮廓，最后补雨丝、反光、灯牌字体和局部高光。

这也是为什么扩散模型在图像生成里常被认为比一次性生成更容易兼顾结构与细节。

## 为什么扩散模型会有这么多采样器

因为训练目标通常定义的是一个连续或近连续的去噪场，而真正出图时必须把这条轨迹离散成有限步。  
于是你会看到 `DDIM` 这类早期少步采样方法，`Euler / Heun` 这类数值积分器，以及 `DPM-Solver / DPM-Solver++` 这类扩散 ODE 专用高阶求解器。

也就是说，采样器多并不是因为大家随便发明新名字，而是因为“怎么走回去”本来就是一个数值求解问题。

## 为什么后来又会转向蒸馏和一步生成

因为光靠更好的 solver，通常很难把几十步压到一步。  
于是研究开始进一步问：能不能让学生直接学会更短的生成映射，能不能不模仿整条轨迹而直接对齐最终分布，能不能把生成路径改得更适合少步离散。Progressive Distillation 逐级把教师采样步数减半，Consistency / LCM 学习不同时间点映射到同一干净结果的一致性，DMD / DMD2 / Phased DMD 用分布匹配蒸馏追求一步或极少步高质量生成，Flow Matching / Rectified Flow / Rectified Diffusion 则直接拟合噪声到数据路径上的速度场，并重塑生成路径，让少步 ODE 积分更容易。

## 读扩散模型时最容易混淆的几件事

### 训练目标和采样器不是一回事

同一个训练好的模型，往往能搭配多种采样器。

### 少步采样和一步蒸馏不是一回事

前者通常是推理侧优化，后者通常需要重训学生模型。

### 质量与速度始终在拉扯

采样步数越少，系统越快；但想保住细节、多样性和文本对齐，难度也会急剧上升。

## 学习路径与阶段检查

扩散专题最容易混在一起的是“训练目标、采样轨迹、条件控制、少步蒸馏”。建议按下面四段读：

| 阶段 | 先读 | 读完要能回答 |
| --- | --- | --- |
| 1. 建模起点 | [扩散模型发展脉络](evolution.md)、[扩散训练与表示](training.md) | 前向加噪、反向去噪、ELBO、噪声预测和 \(x_0/v/score\) 参数化分别在解决什么 |
| 2. 连续视角 | [Score Matching 到 SDE](score-matching-sde-and-probability-flow.md)、[噪声日程与参数化](noise-schedules-and-parameterization.md) | 为什么同一个扩散模型可以被看成 SDE、ODE 或 score field |
| 3. 推理与控制 | [条件控制与 Guidance](guidance-and-conditioning.md)、[采样与推理加速](inference.md) | CFG、solver、步数、随机性和文本对齐之间如何权衡 |
| 4. 加速与变体 | [一步生成、蒸馏与整流](distillation.md)、[一致性模型与 Rectified Flow](consistency-models-and-rectified-flow.md)、[视频与多模态扩散](video-and-multimodal-diffusion.md) | 少步采样、一步蒸馏、Flow Matching / rectified 路线、Wan 系列视频底座各自改的是哪一层 |

如果目标是过渡到世界模型，读完第 4 阶段后直接接 [视频与多模态扩散](video-and-multimodal-diffusion.md) 里的 Wan 系列说明、[Towards Video World Models 专题](../paper-deep-dives/world-models/towards-video-world-models.md) 和 [LingBot-World 专题](../paper-deep-dives/world-models/lingbot-world.md)。如果只是快速定位方法，先看 [扩散方法对照表](comparison-table.md)。如果要做训练或部署决策，最后一定补 [扩散训练配方与排障](practical-training-recipes-and-failure-analysis.md)：扩散系统的很多失败不是公式错，而是数据、噪声日程、采样器和评测口径没有对齐。

## 从论文专题回填的扩散知识点

专题页已经把图像、视频、少步蒸馏和世界模型路线连了起来。回到扩散总览时，可以把它们放回下面四个问题：

| 专题来源 | 回填到扩散章节的知识点 | 应接到本章哪里 |
| --- | --- | --- |
| [DMD](../paper-deep-dives/diffusion/dmd.md)、[DMD2](../paper-deep-dives/diffusion/dmd2.md) | 少步/一步生成不是只换 solver，而是用 distribution matching、fake critic、regression loss 或 GAN 监督重新训练学生 | [一步生成、蒸馏与整流](distillation.md)、[一致性模型与 Rectified Flow](consistency-models-and-rectified-flow.md) |
| [Phased DMD](../paper-deep-dives/diffusion/phased-dmd.md) | 高低 SNR 区间承担不同生成职责，few-step 蒸馏可以按噪声子区间拆 expert，以保住视频运动和图像多样性 | [噪声日程与参数化](noise-schedules-and-parameterization.md)、[一步生成、蒸馏与整流](distillation.md) |
| [Wan2.1](../paper-deep-dives/diffusion/wan.md) | 视频 foundation model 的质量来自 VAE、Video DiT、Flow Matching、分阶段图文视频训练、Context Parallel 和后训练的系统组合 | [视频与多模态扩散](video-and-multimodal-diffusion.md)、[扩散训练配方与排障](practical-training-recipes-and-failure-analysis.md) |
| [CausVid](../paper-deep-dives/diffusion/causvid.md)、[Self Forcing](../paper-deep-dives/world-models/self-forcing.md) | 双向视频扩散要变成流式世界模型，必须处理 causalization、KV cache、self-rollout 和 train-test gap | [采样与推理加速](inference.md)、[视频与多模态扩散](video-and-multimodal-diffusion.md) |
| [SLA / SLA2](../paper-deep-dives/foundations/sla-sparse-linear-attention.md) | 视频 DiT 的 attention 可以拆成 sparse exact、linear compensation 和 skip；这说明扩散加速也可能来自 attention 结构和低比特 kernel | [视频与多模态扩散](video-and-multimodal-diffusion.md)、[扩散训练配方与排障](practical-training-recipes-and-failure-analysis.md) |

## 快速代码示例

```python
import torch

def cfg_eps(eps_uncond, eps_cond, scale=6.0):
    # classifier-free guidance
    return eps_uncond + scale * (eps_cond - eps_uncond)

@torch.no_grad()
def ddim_step(x_t, alpha_t, alpha_prev, eps):
    # 一个简化版 DDIM 更新
    x0 = (x_t - (1 - alpha_t).sqrt() * eps) / alpha_t.sqrt()
    return alpha_prev.sqrt() * x0 + (1 - alpha_prev).sqrt() * eps
```

这段代码把两个常见推理组件放在一起：`cfg_eps` 展示了 **Classifier-Free Guidance** 如何混合条件/无条件噪声预测，`ddim_step` 展示了单步 **DDIM** 更新。实践中你可以先固定步数，再调 `scale` 找到质量与多样性的平衡点。

*[ODE]: Ordinary Differential Equation，常微分方程；在扩散里常指没有额外随机噪声项的 probability flow 轨迹。
*[SDE]: Stochastic Differential Equation，随机微分方程；在扩散里常指带随机噪声项的连续时间加噪或反向生成过程。
