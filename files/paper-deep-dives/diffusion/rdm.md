---
paper_date: 2026-07
paper_title: "Representation Distribution Matching for One-Step Visual Generation"
paper_topic: 扩散模型
---

# RDM：用表征分布匹配训练一步视觉生成器

!!! info "论文信息"
    **论文题名：** Representation Distribution Matching for One-Step Visual Generation。<br>
    **作者：** Lan Feng、Wuyang Li、Éloi Zablocki、Matthieu Cord、Alexandre Alahi。<br>
    **机构：** EPFL、Valeo.ai、Sorbonne Université。<br>
    **时间 / 主题：** 2026-07；扩散模型与一步视觉生成。<br>
    **arXiv / 项目页：** [2607.02375](https://arxiv.org/abs/2607.02375)；[alan-lanfeng.github.io/rdm](https://alan-lanfeng.github.io/rdm/)。<br>
    **代码：** 论文项目页提供 GitHub 链接。<br>
    **元数据与核验口径：** 来源为 arXiv HTML、论文项目页；Checked Date：2026-09-07；Repro Status：Paper / project page reviewed, independent reproduction not claimed。

## 原论文故事线

### 背景与问题

一步生成器把随机噪声一次映射成图像，推理成本很低，但训练时需要一个足够可靠的分布级监督。GAN 依赖在线判别器，DMD 依赖 real/fake score 和 teacher 轨迹，Fréchet distance loss 只匹配均值和协方差，单一视觉 encoder 又可能被生成器利用。论文把这条路线统一命名为 **Representation Distribution Matching（RDM，表征分布匹配）**，并追问两个问题：分布距离应该怎样从有限 batch 中估计，应该在哪些表征空间里匹配。

### 核心发现

论文的关键发现不是“把 MMD 加进 loss”这么简单，而是 MMD 的估计器决定了它能否扩展到高维视觉生成。作者发现，经典 MMD 配上精确的 batch 内 repulsion、对全量真实数据一次性压缩并冻结的 Nyström attraction，以及足够大的新鲜生成 batch 后，可以成为强的一步生成目标。与此同时，任何单一 encoder 都会给生成器留下可利用的盲点，因此需要一组不同视觉 encoder，并用受约束的权重控制最难满足的表征空间。

### 总体方案

设一步生成器 $g_\theta(z)$ 的输出分布为 $p_\theta$，冻结 encoder 为 $\phi$。RDM 优化

\[
\mathcal L(\theta)=\mathcal D\big(\phi_*p_\theta,\phi_*p_{\mathrm{data}}\big)，
\]

其中 $\mathcal D$ 可以是 MMD、Fréchet distance 或其他分布距离。iRDM 选用高斯核的平方 MMD，并在十个训练 encoder 上联合优化。ImageNet 真实训练集的参考 embedding 先用 4096 个 k-means landmarks 做 Nyström 压缩，得到固定的 kernel mean embedding；每一步只对新生成 batch 计算梯度。

## MMD 在 RDM 中具体做什么

对特征 $u=\phi(x)$，高斯核为

\[
k(u,v)=\exp\left(-\frac{\lVert u-v\rVert_2^2}{2\sigma_\phi^2}\right)。
\]

平方 MMD 的三项分别是生成生成、生成真实和真实真实的核期望。训练时真实真实项与生成器参数无关，可以预计算；生成生成项保留 batch 内精确 pairwise repulsion；生成真实项如果每步随机抽全量真实样本会有较大参考噪声，因此论文用 Nyström 特征近似并冻结全训练集的平均 embedding。这样既保留分布距离的结构，也把 attraction 的计算降到 $O(Bm)$，其中 $B$ 是生成 batch，$m=4096$ 是 landmark 数。

这解释了 RDM 与 Drifting 的联系。Drifting 也可以看成把 kernelized MMD 的梯度写成逐样本的吸引减排斥向量，只是它采用每个 batch 重建的正负参考与归一化漂移场。RDM 选择直接优化 MMD 估计器，并通过固定全量参考和大 batch 降低估计噪声。两者数学上相近，训练接口和工程行为不同。

## 两条设计轴

### 轴一：怎样比较分布

论文比较了 biased MMD、全 pairwise exact MMD、random Fourier feature MMD、Nyström MMD、Fréchet loss、Sliced-Wasserstein loss 和 Drifting field。一个容易忽略的结论是，写下相同的总体距离不代表得到相同的训练行为。有限 batch 估计器决定参考是否漂移、方差有多大、生成器可以利用哪些漏洞。

在 ImageNet-256 的受控实验中，mmdx，也就是 batch 内精确 repulsion 加 Nyström cross-term，排名最好；其后是 random-feature MMD、exact full cross-term、Fréchet、Sliced-Wasserstein 和 faithful Drifting port。这个结果只说明在该 warm-start、encoder 和预算下，估计器选择影响很大，不应解读为 MMD 在所有任务上必然优于其他距离。

### 轴二：在哪些表征空间比较

单一 encoder 会被 gaming。论文展示了只匹配 DINOv2 时，DINOv2 距离可以低于真实 validation floor，但图像仍会保留明显伪影。iRDM 使用十个 encoder 训练，另外四个 encoder 只用于评估，包括 Inception、ConvNeXt、MAE、CLIP、DINOv3、PE-Core、SigLIP2、AIMv2、Web-SSL DINO 和 DreamSim 等训练表征，并保留 DINOv2、SigLIP v1、C-RADIOv3-L 与 FLUX VAE 做 held-out 检查。

多 encoder 并非简单平均。论文用 gated proportional Lagrangian controller，把权重集中到当前最难满足且还没有达到真实 floor 的 encoder；已经满足的空间退出竞争。这样做的目标是减少某个表征空间先饱和后被继续优化的浪费。

## 条件生成中的 joint matching

文生图不能只匹配边缘图像分布 $p(x)$，还要匹配图像与文本的联合分布 $p(x,c)$。否则生成器可以输出很像真实照片的图像，却不执行 prompt。iRDM 因此在 FLUX.2 [klein] 的 post-training 中使用 joint image-text objective，把 prompt fidelity 纳入分布匹配。

论文从四步 FLUX.2 [klein] 出发，以 batch size 10240、180 steps 和约 90 H200 GPU-hours 做 post-training，并把约 300K 个 teacher generations 一次性整理成冻结参考。一步 iRDM 的 GenEval overall 为 0.826，高于四步模型的 0.794；PickScore 为 22.76，高于 22.58。相同 teacher 的 DMD2 一步基线为 0.804 和 22.36。

## 关键实验

| Setting | 结果 | 说明 |
| --- | --- | --- |
| ImageNet-256，iRDM | $\mathrm{SW}_{r^{14}}=1.30$ | 14 个 encoder 的 Sliced-Wasserstein 比值，真实 validation draw 归一化为 1 |
| ImageNet，PickScore | 71.2% 胜过 pMF-H FD-SIM | 评价指标没有参与训练，是独立的人类偏好代理 |
| iRDM 对 held-out real photos | 63.6% 偏好 | 只能说明 PickScore 的偏好结果，不能证明真实照片质量被普遍超越 |
| FLUX.2 [klein]，四步到一步 | GenEval 0.826 vs 0.794；PickScore 22.76 vs 22.58 | joint matching 的文生图 post-training |

主评估 $\mathrm{SW}_{r^{14}}$ 与训练 MMD 使用不同估计器，因此作者用它来检查是否只是把训练 loss 做低。四个 encoder 被完全留作 held-out generalization check。论文还报告了 kernel-MMD 版本的评估，整体排序大体一致，但中间模型会有交换。

## 训练细节与复现边界

- ImageNet-256 从公开的 pMF-H FD-SIM checkpoint post-train 4000 steps，learning rate $1.6\times10^{-6}$，生成 batch size $N=5120$。
- 十个训练 encoder 使用各自 median heuristic bandwidth，真实端使用 1.28M ImageNet training images 的 4096-landmark Nyström reference。
- 生成 batch 是关键变量，论文的最佳点在 2048 以上，显存通过 gradient caching 处理。
- 文生图实验从 FLUX.2 [klein] 四步 checkpoint 开始，180 steps、learning rate $2.83\times10^{-6}$、batch size 10240。
- 论文没有声称独立复现；结果依赖预训练 checkpoint、encoder panel、reference curation 和大 batch 资源。

## 论文位置

RDM 把“一步生成”从 trajectory distillation 的问题改写成 feature-distribution matching。它与 DMD 的主要差别是没有 online teacher、fake score 或 paired trajectory；与 Drifting 的主要差别是直接设计 MMD 估计器，而不是以零漂移 fixed point 作为训练接口；与 TBSM 的主要差别是 TBSM 使用 energy distance 的样本级三体事件和 online tracker。

## 局限与风险

表征空间的 properness 只约束 $\phi(x)$ 的分布。encoder 忽略的像素细节、长尾语义或条件组合不会自动被约束。大 batch 和多 encoder 让训练成本从推理端转回训练端。单一 MMD 数字不能替代 held-out encoder、prompt alignment、FID、PickScore 和定性检查；论文自己的 gaming 例子正说明了这一点。

## 关键术语

- **Representation Distribution Matching（表征分布匹配）**：在冻结预训练 encoder 的特征空间匹配生成与真实分布。
- **Kernel Mean Embedding（核均值嵌入）**：把分布映射成 RKHS 中的平均特征，MMD 比较的是两个平均嵌入的距离。
- **Nyström Approximation（Nyström 近似）**：用少量数据依赖的 landmarks 近似核矩阵，压缩全量真实参考。
- **Joint Image-Text Matching（图文联合匹配）**：匹配 $p(x,c)$，让 prompt 对齐进入分布目标。
- **Metric Gaming（指标投机）**：生成器降低某个 encoder 的距离，却没有获得对应的真实视觉质量。

## 阅读结论

RDM 的价值在于把 MMD 从一个“理论上能比较分布、实践上不够稳”的老目标，变成由估计器、batch、reference 和 encoder panel 共同定义的现代训练系统。它提示扩散模型研究者，分布匹配的难点不只在距离公式，还在有限样本统计和表征选择。一步推理来自直接学习最终输出分布，扩散路径与 solver 已经退到初始化或 teacher 角色。

## 外部精读

- [arXiv HTML](https://arxiv.org/html/2607.02375)
- [RDM 项目页](https://alan-lanfeng.github.io/rdm/)
- Gretton et al. [A Kernel Two-Sample Test](https://www.jmlr.org/papers/v13/gretton12a.html)
