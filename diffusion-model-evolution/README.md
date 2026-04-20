# 扩散模型的发展脉络：从 DDPM 到 DMD、DMD2、Phased DMD 与 rDM

## 说明

这份笔记把扩散模型的发展放在一条连续主线上来看：

1. `DDPM` 解决“怎么把生成建模做稳、做强”。
2. `DDIM`、`Euler`、`Heun`、`DPM-Solver` 解决“怎么把采样做快”。
3. `DMD`、`DMD2`、`Phased DMD`、`rDM` 这类方法继续追问“能不能把几十步甚至几百步，压到几步甚至一步”。

术语说明：

- 你写的 `phrased DMD`，本文按 2025-10-31 的论文 **Phased DMD** 来理解。
- `rDM` 在文献里有歧义。本文默认按“**Rectified Diffusion / Rectified Flow** 这条快速生成路线”来讲；另外，大写 `RDM` 也常指 **Relay Diffusion Model**，那是另一条偏分辨率级联的路线，不是这里的重点。

---

## 一句话总览

扩散模型的发展，本质上是在做三件事：

- 把生成过程写成一个更容易训练的“逐步去噪”问题。
- 把“逐步去噪”重新解释成 `SDE/ODE` 的数值求解问题，从而引入 `Euler`、`Heun`、`DPM-Solver` 这类求解器。
- 再进一步，用蒸馏、分布匹配、整流流把多步采样压缩成少步甚至一步。

如果只记一条主线，可以记成：

```text
DDPM
  -> DDIM
  -> Score SDE / Probability Flow ODE
  -> Euler / Heun / DPM-Solver / DPM-Solver++
  -> Progressive Distillation / Consistency / LCM
  -> DMD / DMD2 / Phased DMD
  -> Rectified Flow / Rectified Diffusion (rDM 语境)
```

---

## 时间线

| 时间 | 方法 | 解决的问题 | 代表意义 |
| --- | --- | --- | --- |
| 2020-06 | DDPM | 训练稳定、生成质量高，但采样慢 | 扩散模型主流化起点 |
| 2020-10 | DDIM | 不重训模型的前提下减少采样步数 | 把扩散采样从“随机马尔可夫链”改成可确定的非马尔可夫过程 |
| 2020-11 | Score SDE / Probability Flow ODE | 给扩散一个连续时间统一视角 | 采样开始被理解为 SDE/ODE 数值积分 |
| 2022-02 | Progressive Distillation | 把多步模型逐级蒸馏成少步模型 | “少步采样”路线成形 |
| 2022-06 | EDM | 系统整理噪声参数化、采样设计 | `Euler`、`Heun` 在扩散里被系统化使用 |
| 2022-06 | DPM-Solver | 面向 diffusion ODE 的高阶专用求解器 | 10 到 20 步高质量采样成为现实 |
| 2022-11 | DPM-Solver++ | 改善 classifier-free guidance 下的稳定性 | 工业界和开源工具链广泛采用 |
| 2023-03 | Consistency Models | 直接学习一步或少步映射 | 少步/一步生成进一步成熟 |
| 2023-09 | Rectified Flow / InstaFlow | 学习更直的生成路径 | 一步生成开始具备较强可用性 |
| 2023-11 | DMD | 用分布匹配做一步扩散蒸馏 | 从“轨迹模仿”转向“分布对齐” |
| 2024-05 | DMD2 | 改善 DMD 的模糊与不稳定 | 一步生成质量继续逼近甚至超过教师 |
| 2024-10 | Rectified Diffusion | 重新解释“整流”为什么有效 | 把 rectified 路线和传统 diffusion 更紧密地接上 |
| 2025-10 | Phased DMD | 将分布匹配分阶段进行 | 在复杂生成任务里兼顾质量与多样性 |

---

## 第一阶段：DDPM 把扩散模型做成主流

`DDPM`（Denoising Diffusion Probabilistic Models）是现代扩散模型真正爆发的起点。它的核心思想是：

- 正向过程逐步往数据里加高斯噪声。
- 反向过程学习逐步去噪。
- 训练时把难的似然问题，转成神经网络预测噪声或 score 的监督学习问题。

常见写法是：

```text
x_t = alpha_t * x_0 + sigma_t * eps
```

模型学习在给定 `x_t` 和时间 `t` 的情况下预测噪声 `eps`，再逐步还原 `x_0`。

`DDPM` 的价值：

- 训练稳定。
- 生成质量高。
- 与 score matching、变分推断建立了联系。

`DDPM` 的代价：

- 采样通常需要几百到上千步。
- 真正部署时延迟很高。

所以，扩散模型接下来的几乎所有工作，都在回答同一个问题：**如何减少步数**。

---

## 第二阶段：DDIM 让“少步采样”第一次真正可用

`DDIM`（Denoising Diffusion Implicit Models）最重要的贡献，不是换了训练目标，而是换了**采样路径**：

- 它证明可以在与 `DDPM` 兼容的训练框架下，构造非马尔可夫的反向过程。
- 当随机项取特定形式时，采样可以变成近似确定性的映射。
- 因而可以在不重训模型的前提下，大幅减少采样步数。

这件事很关键，因为它带来两个后果：

1. 扩散采样不再只能理解成“老老实实一步一步反马尔可夫链”。
2. 采样开始越来越像“沿着一条连续轨迹做离散积分”。

从今天回头看，`DDIM` 是从离散扩散走向 `ODE solver` 时代的桥梁。

---

## 第三阶段：Score SDE 与 Probability Flow ODE 统一了扩散视角

`Score-Based Generative Modeling through SDEs` 把扩散模型放进连续时间框架里：

- 可以把生成写成逆时间 `SDE`。
- 同时存在一个对应的 `probability flow ODE`，它和 `SDE` 共享相同边缘分布。

这一步的理论意义非常大：

- 从此以后，采样可以直接借用数值分析里的 ODE/SDE 求解器。
- `DDIM` 可以被理解成这条 ODE 路线上的一种低阶离散化。
- 后面的 `Euler`、`Heun`、`DPM-Solver`，都能放进统一框架里比较。

也就是说，扩散模型从“概率图模型里的反向链”变成了“神经网络定义的微分方程求解问题”。

---

## 第四阶段：Euler、Heun、DPM-Solver 把扩散采样变成数值求解

这一阶段的关键词是：**同一个训练好的模型，换更好的求解器，就能更快出图**。

### 1. Euler

`Euler` 是一阶显式方法，特点是：

- 每一步只做一次网络评估，开销低。
- 实现简单，鲁棒性好。
- 步数少时误差相对大，但在实际工具链里很常用。

在很多 Stable Diffusion 工具中，`Euler` 系列采样器常被当成速度和质量的平衡点。

### 2. Heun

`Heun` 可以理解为带校正的二阶方法，常见形式是 predictor-corrector：

- 先预测一步。
- 再用预测点修正斜率。

它相比 Euler 的优点是：

- 同样步数下，通常误差更小。
- 在中低步数下常能带来更稳的细节质量。

代价是：

- 通常每步需要更多计算。

### 3. DPM-Solver

`DPM-Solver` 的思路不是直接套通用 ODE solver，而是利用 diffusion ODE 的特殊结构，构造专用的高阶求解器：

- 把线性部分尽量解析处理。
- 对神经网络项做更合适的高阶近似。
- 目标是在 `10` 到 `20` 步附近依然维持高质量。

它的重要性在于：

- 说明扩散采样的加速，不一定靠重新训练模型。
- 纯推理端的数值方法本身就能吃掉大量步数。

### 4. DPM-Solver++

`DPM-Solver++` 是更面向实际使用的版本，尤其关注 `classifier-free guidance` 场景：

- 对 guided sampling 更稳。
- 在高 guidance 下更不容易发散或失真。
- 实际开源生态里采用度很高。

### 这一阶段的本质

这几类方法都没有改变扩散模型的根本训练范式，它们做的是：

- 把“1000 步离散反推”改写成“少量高质量数值积分步”。
- 通过更聪明的离散化，让扩散模型在不蒸馏、不重训的情况下先快一大截。

所以，从研究脉络上看：

- `DDIM` 是第一代“少步采样”。
- `Euler/Heun` 是通用数值积分器思路进入扩散。
- `DPM-Solver` 是面向 diffusion ODE 的专用高阶求解器。

---

## 第五阶段：从“更好的求解器”走向“直接蒸馏少步模型”

当大家发现求解器再优化，仍然难把几十步压到一步时，研究开始转向蒸馏。

### 1. Progressive Distillation

`Progressive Distillation` 的策略很直接：

- 让学生模型学会“教师两步合成的一步”。
- 再不断重复，把步数从大到小压缩。

它的重要性在于：

- 证明扩散模型可以系统地被蒸馏到少步。
- 为后面的 Consistency、LCM、DMD 一类工作铺路。

### 2. Consistency Models / LCM

这条路线的目标是：

- 直接学习“不同噪声层级下的一致映射”。
- 使模型天然适合一步或少步采样。

这类方法说明，少步生成不一定非要严格重演原始反向轨迹，也可以直接学一个更短的生成算子。

---

## 第六阶段：DMD 把一步蒸馏改成“分布匹配”

`DMD`（Distribution Matching Distillation）是一步扩散蒸馏的重要转折点。

在它之前，很多少步方法更像在做：

- 轨迹蒸馏。
- 输出回归。
- 或者局部一致性约束。

而 `DMD` 的关键变化是：

- 不再只逼学生去模仿某条教师轨迹。
- 转而直接优化“学生分布”和“真实数据分布/教师诱导分布”的差距。

粗略理解，`DMD` 使用两个 diffusion score 网络来构造训练信号：

- 一个刻画真实数据侧的 score。
- 一个刻画生成样本侧的 score。

二者的差可以近似形成分布匹配梯度，从而训练一个一步生成器。

`DMD` 的意义：

- 让“一步扩散”不再完全受限于逐点回归。
- 速度极快，适合实时或交互式生成。

`DMD` 的问题：

- 训练稳定性要求高。
- 一步模型容易出现模糊、模式塌缩或细节不足。
- 为了稳住训练，原始版本还需要额外的回归正则。

所以，`DMD` 是“方向非常对”，但第一版还不够成熟。

---

## 第七阶段：DMD2 让一步扩散更像真正可用的方案

`DMD2` 可以看成对 `DMD` 的系统修正。它抓住了 DMD 的主要痛点：

- 原始 DMD 过度依赖回归损失。
- 图像容易偏糊。
- 一步生成器和“假样本侧”估计器之间的协同还不够强。

`DMD2` 的关键改进通常概括为：

- 去掉原始 `DMD` 中效果不理想的回归约束。
- 用更合理的双时间尺度更新去训练生成器和“假样本侧”网络。
- 引入 `GAN loss` 来缓解模糊问题。
- 不只支持一步，也可以扩展到多步。

这使它比 `DMD` 更接近实战：

- 质量更高。
- 一步结果更锐。
- 在一些基准上能逼近甚至超过教师采样器。

如果说 `DMD` 是证明“分布匹配的一步扩散蒸馏能做”，那 `DMD2` 就是在回答“怎么把它做得更稳、更强”。

---

## 第八阶段：Phased DMD 把分布匹配从“一次做完”改成“分阶段做”

`Phased DMD` 的直觉是：一步就把整个生成过程蒸馏完，难度可能太高。

因此它把问题拆开：

- 按 `SNR` 或时间区间把生成过程分成多个 phase。
- 每个 phase 内分别做分布匹配和分数匹配。
- 让教师知识分阶段地传给学生，而不是一次性压缩。

这样做的动机非常清楚：

- 复杂生成任务里，不同噪声阶段承担的语义不一样。
- 早期阶段更偏全局结构。
- 后期阶段更偏纹理和细节。

如果把这些阶段全部硬压成一个一步映射，优化会非常困难；分 phase 训练更接近“逐段蒸馏”。

因此，`Phased DMD` 可以看成：

- 继承 `DMD` 的“分布匹配”思想。
- 同时吸收“多阶段蒸馏/课程学习”的工程直觉。

它代表了 DMD 家族的一个自然演化方向：**不是只问一步够不够快，而是问怎样在很少步下保住质量与多样性**。

---

## 第九阶段：rDM 语境下的 Rectified Flow / Rectified Diffusion

如果把你说的 `rDM` 理解成 `Rectified Diffusion` 这条路线，那么它和 DMD 家族的关系是“目标相近，路径不同”。

### 1. Rectified Flow 的直觉

`Rectified Flow` 希望学习更“直”的生成路径：

- 从噪声到数据的传输轨迹越直，越容易用少量步数逼近。
- 如果轨迹足够直，甚至一步 Euler 离散也可能够用。

这和传统扩散的想法不同：

- 传统扩散更强调逐步去噪的概率建模。
- rectified 路线更像在做生成传输路径的重构。

### 2. InstaFlow 的作用

`InstaFlow` 把 rectified flow 路线真正推到文生图实战里，证明：

- 通过 reflow 和蒸馏，可以把大模型压缩到一步。
- 一步生成在真实文生图系统中是可以跑起来的。

### 3. Rectified Diffusion 的再解释

`Rectified Diffusion` 的一个重要观点是：

- 少步甚至一步生成之所以有效，不一定只是因为“轨迹更直”。
- 更关键的可能是：构造了更适合一阶离散近似的路径与配对方式。

这让 rectified 路线和传统 diffusion 之间的边界变得没有那么硬：

- 它不再像是“另一套完全不同的生成范式”。
- 更像是在说：传统扩散也能被重新整形成适合少步求解的路径。

从研究史上看，这条路线和 `DMD/DMD2/Phased DMD` 在同一个方向上会师：

- 都在追求极少步采样。
- 但 DMD 家族更偏“蒸馏 + 分布匹配”。
- Rectified 家族更偏“重新设计生成路径，让少步离散天然有效”。

---

## 这些方法之间到底是什么关系

可以用下面这张关系表来理解：

| 类别 | 代表方法 | 是否需要重新训练 | 核心思想 | 典型目标 |
| --- | --- | --- | --- | --- |
| 原始扩散 | DDPM | 是 | 学习逐步去噪的反向过程 | 高质量生成 |
| 非马尔可夫少步采样 | DDIM | 否 | 改采样路径，不改主训练框架 | 从上千步降到几十步 |
| 数值求解器 | Euler、Heun、DPM-Solver | 否 | 把采样看作 ODE/SDE 离散求解 | 用更少步获得更好质量 |
| 逐级蒸馏 | Progressive Distillation | 是 | 学教师多步到学生少步 | 从几十步压到几步 |
| 一致性建模 | Consistency、LCM | 是 | 直接学一步/少步映射 | 快速推理 |
| 分布匹配蒸馏 | DMD、DMD2、Phased DMD | 是 | 不只模仿轨迹，直接对齐分布 | 一步或极少步高质量生成 |
| 路径整流 | Rectified Flow、Rectified Diffusion | 是 | 学更适合少步离散的生成路径 | 一步或极少步生成 |

最核心的区别是：

- `DDIM / Euler / Heun / DPM-Solver` 是 **推理加速**。
- `DMD / DMD2 / Phased DMD / Rectified Diffusion` 是 **模型重训后的结构性加速**。

前者更像“给同一台发动机换更好的变速箱”，后者更像“直接换发动机结构”。

---

## 从今天回头看，这条发展线说明了什么

### 1. 扩散模型的第一性问题不是“能不能生成”，而是“采样太慢”

`DDPM` 之后，大家已经知道扩散模型能生成高质量样本。真正限制它大规模落地的，是推理步数。

### 2. 先有数值分析视角，后有极少步蒸馏视角

研究顺序大致是：

- 先把扩散统一到 `ODE/SDE`。
- 再引入 `Euler`、`Heun`、`DPM-Solver` 这类 solver。
- 最后发现 solver 再好，也有极限，于是转向蒸馏和路径重构。

### 3. 一步生成并不是“把原来 50 步硬砍成 1 步”这么简单

到了 `DMD2`、`Phased DMD`、`Rectified Diffusion` 这一阶段，研究者已经不再满足于简单压缩步数，而是在重新思考：

- 学生到底应该模仿教师的什么？
- 是模仿轨迹、模仿分布，还是重设计路径？
- 怎样在少步下仍然保住多样性、锐度和文本对齐？

这也是为什么这些新方法常常会引入：

- 分布匹配。
- 对抗损失。
- 分阶段训练。
- 更适合少步积分的轨迹设计。

---

## 如果按“方法选择”来理解

### 1. 你有一个现成扩散模型，不想重训

优先看：

- `DDIM`
- `Euler`
- `Heun`
- `DPM-Solver / DPM-Solver++`

这是最直接的推理侧优化。

### 2. 你愿意重训，希望做到 1 到 8 步

优先看：

- `Progressive Distillation`
- `Consistency Models / LCM`
- `Rectified Flow / Rectified Diffusion`

### 3. 你追求一步生成，而且能接受更复杂训练

重点看：

- `DMD`
- `DMD2`
- `Phased DMD`

这条路线最像“高质量一步扩散”的正面攻坚。

---

## 结论

扩散模型的发展，不是若干孤立论文的堆叠，而是一条非常清晰的优化链：

- `DDPM` 建立了强大的逐步去噪生成框架。
- `DDIM` 打开了少步采样的大门。
- `Score SDE / ODE` 统一视角后，`Euler`、`Heun`、`DPM-Solver` 把采样问题正式变成数值求解问题。
- 当纯求解器优化逼近上限后，研究进入蒸馏与路径重构阶段。
- `DMD` 把一步蒸馏提升到“分布匹配”层面。
- `DMD2` 修补了 DMD 的稳定性和清晰度问题。
- `Phased DMD` 进一步把一步/少步蒸馏拆成阶段化学习。
- `rDM` 语境下的 `Rectified Diffusion` 则从另一侧说明：若能把生成路径设计得更适合少步离散，极少步甚至一步生成就会更自然。

所以，今天看扩散模型，最好不要把 `DDPM`、`DDIM`、`DPM-Solver`、`DMD2`、`Rectified Diffusion` 看成平行方法，而要把它们看成同一条技术路线上的不同阶段：

- 先解决“能生成”。
- 再解决“能快”。
- 最后解决“极少步下还能高质量地快”。

---

## 参考文献

1. Ho, Jain, Abbeel. *Denoising Diffusion Probabilistic Models*. arXiv:2006.11239. <https://arxiv.org/abs/2006.11239>
2. Song, Meng, Ermon. *Denoising Diffusion Implicit Models*. arXiv:2010.02502. <https://arxiv.org/abs/2010.02502>
3. Song et al. *Score-Based Generative Modeling through Stochastic Differential Equations*. arXiv:2011.13456. <https://arxiv.org/abs/2011.13456>
4. Salimans, Ho. *Progressive Distillation for Fast Sampling of Diffusion Models*. arXiv:2202.00512. <https://arxiv.org/abs/2202.00512>
5. Karras et al. *Elucidating the Design Space of Diffusion-Based Generative Models*. arXiv:2206.00364. <https://arxiv.org/abs/2206.00364>
6. Lu et al. *DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps*. arXiv:2206.00927. <https://arxiv.org/abs/2206.00927>
7. Lu et al. *DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models*. arXiv:2211.01095. <https://arxiv.org/abs/2211.01095>
8. Song et al. *Consistency Models*. arXiv:2303.01469. <https://arxiv.org/abs/2303.01469>
9. Luo et al. *Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference*. arXiv:2310.04378. <https://arxiv.org/abs/2310.04378>
10. Liu et al. *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow*. arXiv:2209.03003. <https://arxiv.org/abs/2209.03003>
11. Liu et al. *InstaFlow: One Step is Enough for High-Quality Diffusion-Based Text-to-Image Generation*. arXiv:2309.06380. <https://arxiv.org/abs/2309.06380>
12. Yin et al. *One-step Diffusion with Distribution Matching Distillation*. arXiv:2311.18828. <https://arxiv.org/abs/2311.18828>
13. Yin et al. *Improved Distribution Matching Distillation for Fast Image Synthesis*. arXiv:2405.14867. <https://arxiv.org/abs/2405.14867>
14. Wang et al. *Rectified Diffusion: Straightness Is Not Your Need in Rectified Flow*. arXiv:2410.07303. <https://arxiv.org/abs/2410.07303>
15. Yan et al. *Phased DMD: Distilling Generative Video Diffusion Models via Distribution Matching and Score Matching*. arXiv:2510.27684. <https://arxiv.org/abs/2510.27684>
