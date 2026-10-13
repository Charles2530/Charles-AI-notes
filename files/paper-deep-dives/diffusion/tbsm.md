---
paper_date: 2026-07
paper_title: "Three-Body Scattering for Generative Modeling"
paper_topic: 扩散模型
---

# TBSM：用三体散射把分布距离变成一步监督

!!! info "论文信息"
    **论文题名：** Three-Body Scattering for Generative Modeling。<br>
    **作者：** Peng Sun、Zhenglin Cheng、Deyuan Liu、Jun Xie、Xinyi Shang、Tao Lin。<br>
    **机构：** Westlake University、Zhejiang University、University College London。<br>
    **时间 / 主题：** 2026-07；生成模型与一步视觉生成。<br>
    **arXiv / 代码：** [2607.18198](https://arxiv.org/abs/2607.18198)；[github.com/sp12138/TBSM](https://github.com/sp12138/TBSM)。<br>
    **元数据与核验口径：** 来源为 arXiv HTML v1；Checked Date：2026-09-07；Repro Status：Paper / official code link reviewed, independent reproduction not claimed。

## 原论文故事线

### 背景与问题

一步生成通常依赖三类监督。GAN 用 critic 做对抗比较，扩散和 Flow 用时间索引的噪声、score 或 velocity target，自回归模型按顺序预测 token。TBSM 想提供第四种接口：从一个 proper distributional energy 出发，为每个生成样本构造可以直接回归的 sample-level target，不需要 path teacher、adversarial discriminator、噪声时间表或有序解码。

### 核心发现

论文从 energy distance 出发。它等价于一种距离核 MMD，包含生成到真实的 attraction 和生成到生成的 self-interaction。对一个生成 projectile，只采样一个同条件 real source 和一个独立 generated source，就能得到一个三体事件。这个 inter-source bearing 减去 intra-source bearing 的条件期望，等于 energy-distance 的 Wasserstein gradient-flow velocity。于是一个 batch 的 $B$ 个 projectile 只需要 $O(B)$ 个常数规模事件，每个事件都能形成 detached regression target。

### 三体散射公式

对条件 $c$，生成 projectile $x_p\sim P_\theta(\cdot|c)$，真实 source $x_r\sim Q(\cdot|c)$，独立生成 source $x_s\sim P_\theta(\cdot|c)$。定义

\[
b_r=\frac{x_r-x_p}{\lVert x_r-x_p\rVert},\qquad
b_s=\frac{x_s-x_p}{\lVert x_s-x_p\rVert},
\]

三体散射向量为

\[
\widehat v_{\text{scat}}=b_r-b_s.
\]

第一项把 projectile 拉向同条件真实样本，第二项把它从一个生成样本推开，避免纯 attraction 带来的收缩和模式坍缩。论文将 “three-body” 限定为一个 projectile 与两个 source 的有符号交互，并不声称它是物理学中的 Newtonian dynamics。

## 从 energy distance 到 MMD

TBSM 的总体目标是

\[
\mathcal F(\theta)=\mathbb E_c\left[\frac12D_E^2(P_\theta(\cdot|c),Q(\cdot|c))\right],
\]

其中

\[
D_E^2(P,Q)=2\mathbb E\lVert x_p-x_r\rVert
-\mathbb E\lVert x_p-x_s\rVert
-\mathbb E\lVert x_r-x_r'\rVert.
\]

欧氏空间是 strong negative type，因此在有限一阶矩下 energy distance 非负，且为零当且仅当 $P=Q$。它也可以写成距离核诱导的平方 MMD。这个关系很适合放回 MMD 视角理解：RDM 使用正定高斯核的 kernel mean embedding，TBSM 选择 energy distance 对应的距离核，并把其梯度拆成一个个带符号的粒子事件。

区别在于训练估计器。全量 MMD 常见实现要计算 batch 内核矩阵或近似全量 cross-term；TBSM 每个 projectile 只配一个 real source 和一个 detached generated source，得到线性数量的 sample-level loss。代价是单事件噪声较大，因此论文再训练一个 tracker 估计给定 projectile 和 condition 时的散射向量条件期望。

## Frozen-target regression

三体事件形成 detached target

\[
x_{\text{target}}=\operatorname{sg}\left[x_p+\widehat v_{\text{scat}}\right].
\]

生成器优化 $\lVert x_p-x_{\text{target}}\rVert_2^2$。由于 target 和 source 分支停止梯度，当前参数的输出梯度就是 $-\widehat v_{\text{scat}}$；取条件期望后，它等于总体 energy objective 梯度的 pullback。这个回归 loss 的数值本身不是 energy distance 的估计，而是 energy-gradient 的局部 surrogate。

论文还引入两个参数。$\lambda$ 控制 generated-source 排斥项，$\rho$ 控制瞬时散射和 tracker field 的混合

\[
\widehat v_\lambda=b_r-\lambda b_s,\qquad
\widehat v_{\text{mix}}=(1-\rho)\widehat v_\lambda+\rho v_{\text{trk}}.
\]

在 $\lambda=1$ 时对应 proper energy-distance endpoint。若 $\lambda<1$，生成内部 dispersion 项的权重被削弱，目标通常会偏向更低的生成分散度，因此论文把它当作经验调参，不宣称仍是同一个 proper objective。

## Tracked scattering

瞬时散射每个事件只用一对 source，方差较大。tracker $v_\phi(\tilde x,c)$ 学习这些 noisy vectors 的条件期望。算法同时更新 generator 与 tracker：

1. 生成 projectile 和 detached generated source。
2. 采样 real source，计算 $b_r,b_s$。
3. 得到 $\widehat v_\lambda$，并在 real-projectile 线段上采样 query point $\tilde x$。
4. 用 tracker 输出与瞬时向量混合，形成 generator 的 detached target。
5. 用瞬时向量监督 tracker。

$\rho=0,\lambda=1$ 是 instant scattering，最接近 Drift-like 粒子动力学；$\rho=1,\lambda=1$ 是 fully tracked energy-distance endpoint。tracker 不是 discriminator，它回归的是向量场，论文也给出其偏差与方差分解。

## 表征空间与模型接口

ImageNet-256 实验使用冻结 image representations，包括 ResNet-18、SigLIP2-B 和 MAE-B。三者联合时，representation-field ablation 的 FID 为 8.29，优于单一 encoder 的 13.47、21.87 和 16.18。主 benchmark 同时使用 JiT、PixelDiT 与 DiT。JiT 和 PixelDiT 在像素空间工作，DiT 使用 SD-VAE latent。

所有 ImageNet TBSM 结果都使用 PyTorch 和 AdamW。除了 FD-loss refinement 外，ImageNet generator 都从已有 diffusion 或 flow matching 多步 checkpoint 初始化，因此结果主要说明“把已有 generator 改造成一步模型”的能力，不能等同于随机初始化从头训练的结论。每项指标使用 50K 生成样本，FID、FDr^6 和 IS 分别衡量分布距离、六个表征空间的 normalized FD 与单图质量。

## 与扩散、Drifting 和 GAN 的设计关系

| 方法 | 每个样本的监督 | 是否需要时间路径 | 关键分布信号 |
| --- | --- | --- | --- |
| Diffusion / Flow | denoising、score 或 velocity target | 需要 | 按 $t$ 的局部路径监督 |
| DMD | real score 与 fake score 差 | 需要 teacher 的噪声 score 接口 | reverse-KL 近似梯度 |
| Drifting | batch 级 kernelized attraction-repulsion field | 不需要 | 零漂移 fixed point |
| RDM | frozen encoder 上的 MMD 估计器 | 不需要 | 生成与真实 feature law 的距离 |
| TBSM | real projectile、real source、generated source 的三体事件 | 不需要 | energy distance 的 Wasserstein gradient |

论文的 design map 还说明，去掉 generated-source 分支会变成单纯 fake-to-real attraction，形式上更接近 GAN-like displacement；加入 noisy input、clean target 和时间条件后，才会接近 perceptual flow matching 或 pixel diffusion 的监督。相似的向量形式不代表目标完全相同，输入、target、是否有时间路径和是否 detach 都会改变训练问题。

## 实验结论

| 模型与空间 | NFE | FID | FDr^6 | IS |
| --- | ---: | ---: | ---: | ---: |
| TBSM，PixelDiT-XL，pixel | 1 | 2.23 | 7.15 或 7.59 | 210.0 或 230.9 |
| TBSM，DiT-XL/2，latent | 1 | 1.63 | 6.76 | 243.0 |
| TBSM，DiT-XL/4，ImageNet-512 | 1 | 1.92 | 未在主表给出 | 未在主表给出 |

TBSM 在 latent 与 pixel 两个空间都能达到有竞争力的一步结果。$\lambda$ 从 1.0 降到 0.9 时，JiT-B 的 FID 从 3.35 降到 2.92，IS 从 205.3 升到 228.7；作者强调这只是经验上类似 CFG 的 trade-off，$\lambda$ 实际控制的是 generated-source repulsion，不是 classifier-free guidance。

## 局限与证据边界

理论收敛需要 exact $\lambda=1$ field、链式法则正则性、斜率条件和相对 tracker 误差控制，有限 generator 的结论还需要 smooth objective、realizability、gradient dominance、bounded variance 和 diminishing steps。论文明确说这些条件没有覆盖一般神经网络的 AdamW/SGD 训练。

表示空间的 properness 只作用于 encoder 投影后的分布。随机初始化的 ImageNet 规模训练尚未被充分测试，主结果大量依赖多步 checkpoint 初始化。强条件场景中一条 caption 只有一张真实图时，条件 energy objective 可能过度贴近这一个配对样本，除非有增强、跨条件共享统计结构或额外正则化。

## 关键术语

- **Energy Distance（能量距离）**：由跨分布距离减去两侧内部距离构成的 proper distributional discrepancy。
- **Projectile / Source（三体事件中的投射粒子与源）**：projectile 是可微的生成样本，real source 与 generated source 构造 signed interaction。
- **Bearing Vector（方位向量）**：从 projectile 指向 source 的单位向量。
- **Tracked Scattering（跟踪散射）**：用在线向量场 tracker 估计瞬时散射的条件期望。
- **Frozen Target Regression（冻结目标回归）**：把散射后的 projectile detach 成 target，用普通 MSE 拉动生成器输出。

## 阅读结论

TBSM 把“分布距离”翻译成了每个生成样本都能执行的局部动作。MMD 解释了 energy distance 的分布匹配背景，三体事件解释了为什么需要同时 attraction 和 self-repulsion，tracker 解释了如何把高方差单事件变成更平滑的 field。它的实验结果说明一步生成可以拥有比 GAN 更细的分布几何监督，但理论保证、随机初始化扩展和强条件场景仍是开放问题。

## 外部精读

- [arXiv HTML v1](https://arxiv.org/html/2607.18198v1)
- [官方代码](https://github.com/sp12138/TBSM)
- [Energy distance 与距离核 MMD 的关系](https://arxiv.org/html/2607.18198v1#S2.F1)
