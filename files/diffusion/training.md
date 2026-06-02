# 扩散训练与表示

扩散模型训练的核心不是“让模型从噪声里变出图片”这句口号，而是学会在不同噪声强度下判断样本应该往哪里去。训练时我们主动把干净样本加噪，模型学习反向去噪或预测 score；采样时再从噪声一步步走回数据分布。

## 核心问题

为什么扩散模型要先把好图弄脏？因为直接学习复杂数据分布很难，但学习“给定一张被加噪的图，如何稍微修正它”更稳定。把很多噪声等级串起来，就得到从纯噪声到样本的生成过程。

最常见 DDPM 前向加噪可以写成：

\[
q(x_t \mid x_0)=\mathcal{N}(\sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)I)
\]

这行表示：从干净样本 \(x_0\) 出发，按时间步 \(t\) 混入不同强度的高斯噪声。\(\bar{\alpha}_t\) 越小，样本越接近纯噪声。

## 训练目标：预测噪声只是一个坐标选择

DDPM 常用训练目标是预测噪声 \(\epsilon\)：

\[
\mathcal{L}=\mathbb{E}_{x_0,\epsilon,t}\left[\|\epsilon-\epsilon_\theta(x_t,t)\|_2^2\right]
\]

白话说：随机选一张真图、一个噪声、一个时间步，把真图加噪成 \(x_t\)，然后让网络猜“我刚才加进去的噪声是什么”。猜得越准，就越能在采样时把噪声从图里拿掉。

不过预测 \(\epsilon\) 不是唯一选择。模型也可以预测 \(x_0\)、score 或 velocity \(v\)。这些参数化会影响训练稳定性、采样器设计和不同噪声区间的权重。

![DDPM graphical model 原论文图](../assets/images/paper-figures/diffusion/ddpm-figure-2-graphical-model.png){ width="620" }

<small>图源：[Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)，Figure 2。原图表达前向过程逐步加噪，反向过程逐步去噪。本站用它说明：训练时知道加噪轨迹，采样时只保留学到的反向去噪模型。</small>

看这张图时，左到右是训练构造的前向 noising，右到左是生成时要学的 reverse denoising。扩散模型的优雅之处在于，前向过程固定可控，学习任务被拆成很多局部去噪问题。

## Noise schedule 决定每段训练难度

Noise schedule 决定每个时间步加多少噪。太快加到纯噪声，模型很难学中间结构；太慢则采样步数和训练资源会被拉长。改 schedule 本质上是在改变不同噪声区间的学习权重。

![Improved DDPM VLB terms 原论文图](../assets/images/paper-figures/diffusion/improved-ddpm-figure-2-vlb-terms.png){ width="660" }

<small>图源：[Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672)，Figure 2。原图表达不同时间步对 variational lower bound 的贡献差异。本站用它说明：噪声日程不是装饰参数，它决定训练信号在低噪声和高噪声区域如何分布。</small>

低噪声区更接近细节修复，高噪声区更接近语义和全局结构重建。图像、视频、音频、多模态扩散对这两端的需求不同，所以 schedule 和 loss weighting 经常要跟任务一起调。

## 采样：把局部去噪串成轨迹

训练学到的是单步或局部去噪能力。采样时从 \(x_T\sim\mathcal{N}(0,I)\) 开始，按时间步反复调用模型，逐渐得到样本。DDPM 的采样随机性更强，DDIM、DPM-Solver、ODE/SDE sampler 会改变轨迹和步数。

![DDPM progressive generation 原论文图](../assets/images/paper-figures/diffusion/ddpm-figure-14-progressive-generation.jpg){ width="670" }

<small>图源：[DDPM](https://arxiv.org/abs/2006.11239)，Figure 14。原图表达从噪声到图像的逐步生成过程。本站用它说明：采样不是一次性解码，而是在不同噪声尺度上连续修正结构和细节。</small>

少步采样和蒸馏的困难也在这里：如果你把几十步压成几步，模型不再只是做“轻微修正”，而要跨越更大的分布距离。这就是 consistency model、rectified flow、DMD 等方法要重新设计训练目标的原因。

## 扩散训练里的表示问题

在 latent diffusion 里，扩散不直接跑在像素空间，而跑在 VAE latent 上。这样 token 数和空间尺寸都更小，训练和采样更便宜。代价是 VAE 会限制可表达细节，也会引入压缩误差。

在 DiT 里，U-Net 被 Transformer block 替代，图像 latent 被切成 patch token。它和 Transformer 的关系非常直接：扩散负责训练目标和采样过程，Transformer 负责 token 间建模。

对视频扩散和世界模型来说，表示问题更尖锐：时间、动作、相机、场景几何和长时一致性都要进入条件或 latent。生成好看的短视频，不等于模型理解了动作后果。

## 常见误解

| 误解 | 更准确的说法 |
| --- | --- |
| 扩散模型就是不断去噪 | 训练目标、噪声日程、参数化、采样器和表示空间共同决定效果。 |
| 预测噪声是唯一正确目标 | \(\epsilon\)、\(x_0\)、score、velocity 各有权衡。 |
| 少步采样只是少跑几步 | 少步会改变每一步跨越的分布距离，需要蒸馏或新目标。 |
| 视频扩散自然就是世界模型 | 没有动作条件、状态接口和闭环评测时，它更像视频生成器。 |

## 外部精读

- [Lil'Log: What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)：扩散直觉和 SDE 统一视角。
- [DDPM](https://arxiv.org/abs/2006.11239)：看前向/反向过程和基本训练目标。
- [Improved DDPM](https://arxiv.org/abs/2102.09672)：看 noise schedule 和 likelihood 细节。
- [Score-Based Generative Modeling through SDEs](https://arxiv.org/abs/2011.13456)：把扩散放进连续时间 SDE 框架。
- [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)：理解 latent diffusion 为什么省算力。
