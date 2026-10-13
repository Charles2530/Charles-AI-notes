---
paper_date: 2026-02
paper_title: "Generative Modeling via Drifting"
paper_topic: 扩散模型
---

# Drifting：把生成分布的迭代搬到训练阶段

!!! info "论文信息"
    **论文题名：** Generative Modeling via Drifting。<br>
    **作者：** Mingyang Deng、He Li、Tianhong Li、Yilun Du、Kaiming He。<br>
    **机构：** MIT、Harvard University。<br>
    **时间 / 主题：** 2026-02，v2；扩散模型与一步生成。<br>
    **arXiv / 项目页：** [2602.04770](https://arxiv.org/abs/2602.04770)；[Generative Modeling via Drifting](https://lambertae.github.io/projects/drifting/)。<br>
    **代码：** 项目页提供 JAX 代码、预训练权重与 toy demo。<br>
    **元数据与核验口径：** 来源为 arXiv HTML、项目页；Checked Date：2026-09-07；Repro Status：Paper / official project materials reviewed, independent reproduction not claimed。

## 原论文故事线

### 问题从哪里来

扩散和 Flow Matching 把一个复杂的 pushforward map 拆成很多个推理时变换。每个时间步只移动一点，样本从噪声逐渐到数据，因此质量高但 NFE 很多。Drifting 的出发点是，神经网络训练本身已经在反复更新参数。既然 $f_{i+1}$ 相对 $f_i$ 会让同一个噪声的输出移动，就可以直接设计“样本应该怎样移动”的训练规则，把分布演化放到训练时间。

### 核心发现

论文定义一个依赖真实分布 $p$ 和当前生成分布 $q$ 的 drifting field $V_{p,q}(x)$。它满足反对称性 $V_{p,q}=-V_{q,p}$，所以在 $p=q$ 时漂移为零。训练的目标是生成器输出贴近一个由当前输出加上漂移得到的 stop-gradient target。优化器不断让网络追随这个 target，等价于让 pushforward 分布在训练过程中向数据分布移动；推理时只需要一次网络前向。

### 漂移场与 kernel

论文用正样本 $y^+\sim p$ 和负样本 $y^-\sim q$ 构造

\[
V(x)=V_p^+(x)-V_q^-(x),
\]

其中每个 $V^+$ 是对真实样本的 kernel 加权 mean-shift，$V^-$ 是对生成样本的 kernel 加权 mean-shift。kernel 采用

\[
k(x,y)=\exp\left(-\frac{\lVert x-y\rVert_2}{\tau}\right),
\]

实践中用对 $y$ 的 softmax 归一化，并额外对 batch 中的 $x$ 做归一化。真实样本吸引生成点，生成样本之间相互排斥。反对称性是防止只吸引到一个模式的关键结构。

### 固定点训练目标

理想更新是

\[
f_{\theta_{i+1}}(\epsilon)
\leftarrow
f_{\theta_i}(\epsilon)+V_{p,q_{\theta_i}}(f_{\theta_i}(\epsilon)).
\]

论文把它改成

\[
\mathcal L=\mathbb E_\epsilon\left[
\left\|f_\theta(\epsilon)-\operatorname{stopgrad}\left(f_\theta(\epsilon)+V(f_\theta(\epsilon))\right)\right\|_2^2
\right].
\]

stop-gradient 把本轮的漂移 target 冻结下来，避免直接对依赖 $q_\theta$ 的分布场反传。loss 数值等于漂移场范数的平方，但优化实现是回归到当前 drifted target，而不是把分布本身写成可微对象。

## 与 MMD 的关系

Drifting 不是简单地把 MMD loss 原样实现。MMD 用核均值嵌入的距离比较 $p$ 和 $q$，其梯度可以写成真实端吸引与生成端排斥的组合。Drifting 选择一个归一化的、逐粒子可计算的 field，并把样本更新规则作为固定点回归目标。论文附录也给出 MMD drifting field 的关系，说明在适当 kernel 和归一化下两者有共同的 attraction-repulsion 几何。

工程上两者的差异更重要。Drifting 每步从当前 batch 重建正负参考，估计器成本和方差受 batch 大小影响；RDM 把全量真实参考用 Nyström 预计算并冻结，再用大 batch 做精确生成内部 repulsion。Drifting 因此更像“在线粒子动力学”，RDM 更像“稳定的全量参考 MMD 优化”。

## 表征空间与模型

Drifting 的漂移可以在像素空间，也可以在任意 feature space 中计算

\[
\mathbb E\left[\left\|\phi(x)-\operatorname{stopgrad}\left(\phi(x)+V(\phi(x))\right)\right\|_2^2\right].
\]

论文主要用自监督预训练的 ResNet-style encoder，在多个 stage 和空间位置计算多尺度漂移；也测试了 latent-MAE。若生成器输出 SD-VAE latent，先用 VAE decoder 得到像素，再送入 feature encoder。feature encoder 只在训练时使用，推理时不增加 NFE。

ImageNet-256 的 latent 版本采用 32×32×4 SD-VAE latent，DiT-like patch size 2、adaLN-zero 条件注入。pixel-space 版本直接输出 256×256×3，使用 patch size 16。论文的强模型从头训练，并报告 latent L/2 的 1-NFE FID 1.54、pixel-space L/16 的 FID 1.61。

## Classifier-Free Guidance 的改写

Drifting 把 CFG 放进训练时的负样本分布。给定类别 $c$，正样本来自 $p_{data}(\cdot|c)$，负样本混合当前 $q_\theta(\cdot|c)$ 与无条件真实分布

\[
\tilde q(\cdot|c)=(1-\gamma)q_\theta(\cdot|c)+\gamma p_{data}(\cdot|\varnothing)
\]

这对应生成分布近似

\[
q_\theta(\cdot|c)=\alpha p_{data}(\cdot|c)-(\alpha-1)p_{data}(\cdot|\varnothing),
\qquad \alpha=1/(1-\gamma)
\]

推理时输入一次网络并指定 $\alpha$，不需要像标准 CFG 那样在每个时间步分别跑有条件和无条件模型。论文报告最佳 FID 在 $\alpha=1.0$，也就是通常所说的无 CFG 设置；更高 guidance 会在 FID 和 IS 之间产生 trade-off。

## 实验与结论

| 任务 | 设置 | 结果 |
| --- | --- | --- |
| ImageNet-256 latent | Drifting L/2，1 NFE | FID 1.54，IS 258.9 |
| ImageNet-256 pixel | Drifting L/16，1 NFE | FID 1.61，IS 307.5 |
| Pixel-space 对比 | 87G FLOPs | FID 1.61，优于 StyleGAN-XL 的 FID 2.30 / 1574G FLOPs |
| Robotics control | Drifting Policy 1 NFE 对 Diffusion Policy 100 NFE | 多数任务成功率相当或更高，ToolHang visual 与 PushT state 等任务仍有差异 |

toy 2D 实验从中间、远离目标和单模态坍缩三种初始化出发，展示 $q$ 在训练过程中向双模态 $p$ 漂移。作者把它作为抗 mode collapse 的直觉证据。ImageNet 表格则说明 Drifting 的单步结果已经接近甚至超过部分多步 diffusion/flow 基线，但比较同时受到模型规模、VAE、训练步数和评估协议影响。

## 论文位置

Drifting 与 DMD 的监督来源不同。DMD 用 teacher real score 和 online fake score 差，目标是 reverse-KL 方向；Drifting 直接从真实与生成样本计算 kernelized field，不需要 teacher score 或噪声时间表。它与 RDM 都使用 frozen representation 和 attraction-repulsion，但 RDM 直接优化 MMD 估计器，Drifting 通过 zero-drift equilibrium 组织训练。TBSM 则把 energy-distance 的场改写为每个 projectile 与一个 real source、一个 generated source 的三体事件，并可用 tracker 降低随机场噪声。

## 局限与证据边界

论文明确承认 $V=0\Rightarrow p=q$ 对任意 field 并不成立，kernelized construction 的反向可识别性依赖额外的非退化假设。实验上的零漂移与质量相关性很强，但不构成对一般神经网络 SGD 的收敛证明。高维结果依赖大规模训练、特征 encoder、kernel temperature、正负样本分配和从头训练资源。机器人实验沿用 Diffusion Policy 协议，能说明单步策略的可行性，不能直接说明长期闭环稳定性或真实机器人安全。

## 关键术语

- **Pushforward Distribution（推前分布）**：把先验噪声经过网络 $f$ 映射后得到的输出分布 $f_\#p_{prior}$。
- **Drifting Field（漂移场）**：根据真实和生成分布，为每个生成样本给出移动向量的函数。
- **Zero-Drift Equilibrium（零漂移平衡）**：生成分布与目标匹配时，所有样本的 field 为零的固定点。
- **Mean-Shift Vector（均值漂移向量）**：按 kernel 相似度加权的邻域平均方向。
- **Stop-Gradient Fixed-Point Regression（停止梯度固定点回归）**：把当前样本加 drift 后的结果冻结为 target，再训练网络追随。

## 阅读结论

Drifting 的概念贡献是重新安排迭代发生的时间。扩散模型在推理时反复移动样本，Drifting 在训练时反复移动网络的 pushforward 分布。MMD 提供了理解它的统计背景，kernelized field 提供了粒子级训练信号，stop-gradient 则让这个信号可以用普通回归实现。它适合单步图像和低延迟策略，但“零漂移”与“真实分布完全匹配”之间仍有理论和表示空间边界。

## 外部精读

- [arXiv HTML v2](https://arxiv.org/html/2602.04770v2)
- [项目页与代码](https://lambertae.github.io/projects/drifting/)
- [MMD 的经典综述与两样本检验](https://www.jmlr.org/papers/v13/gretton12a.html)
