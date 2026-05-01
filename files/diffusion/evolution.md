# 扩散模型发展脉络

这页把扩散模型的发展放在一条连续主线上看：`DDPM` 解决如何稳定训练高质量生成模型，`DDIM / ODE solver` 解决如何更快采样，`Consistency / LCM / DMD / Rectified Flow` 继续追问能否把几十步甚至几百步压到几步或一步。

!!! note "初学者先抓住"
    扩散模型发展的主线不是“论文名字越来越多”，而是同一个问题被不断追问：怎样从噪声生成数据，并且生成得更快、更稳、更可控。先把“训练稳定”和“采样加速”分开，后面的路线就不容易混。

!!! example "有趣例子：修复一张被雾化的照片"
    DDPM 像一位修图师每次只擦掉一点雾，虽然慢但稳；DDIM 和 ODE solver 像找到更聪明的擦拭路线；DMD、Consistency 和 Rectified Flow 则像训练一位新修图师，让他几下就能完成原来几十下的效果。

## 统一记号

本文统一使用：

\[
x_t=\alpha_t x_0+\sigma_t\epsilon,\qquad \epsilon\sim\mathcal{N}(0,I)
\]

在 DDPM 文献中常见等价写法是：

\[
x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon
\]

可以理解为真实信号 \(x_0\) 和噪声 \(\epsilon\) 的按时刻混合。扩散模型学习的是如何从噪声状态一步步回到数据分布。

## 一条主线

扩散模型的发展主要围绕三件事：

1. 把生成过程写成更容易训练的逐步去噪问题；
2. 把逐步去噪解释成 SDE/ODE 数值求解，引入 Euler、Heun、DPM-Solver；
3. 用蒸馏、分布匹配、整流流把多步采样压缩成少步甚至一步。

一个简化路线是：

```text
DDPM
  -> DDIM
  -> Score SDE / Probability Flow ODE
  -> Euler / Heun / DPM-Solver / DPM-Solver++
  -> Progressive Distillation / Consistency / LCM
  -> DMD / DMD2 / Phased DMD
  -> Rectified Flow / Rectified Diffusion
```

!!! note "常见误区：新方法不是简单替代旧方法"
    这条时间线更像一组不同代价分配方式，而不是“后来的名字自动淘汰前面的名字”。`DDIM` 主要改采样路径，`DPM-Solver` 主要改数值求解，`Consistency / LCM` 更依赖新的训练或蒸馏目标，`DMD` 关注生成分布对齐。读论文时先问“它动的是训练、采样、蒸馏还是分布匹配”，比先记年份更有用。

## 时间线

| 时间 | 方法 | 解决的问题 | 代表意义 |
| --- | --- | --- | --- |
| 2020 | DDPM | 训练稳定、生成质量高但采样慢 | 扩散模型主流化起点 |
| 2020 | DDIM | 不重训减少采样步数 | 从马尔可夫链走向确定性路径 |
| 2020 | Score SDE / Probability Flow ODE | 连续时间统一视角 | 采样变成微分方程求解 |
| 2022 | EDM | 系统整理噪声参数化和采样设计 | 工程采样器设计成熟 |
| 2022 | DPM-Solver / DPM-Solver++ | 高阶 diffusion ODE 求解 | 10 到 20 步高质量采样 |
| 2023 | Consistency / LCM | 学习一步或少步一致映射 | 少步生成进入实用区间 |
| 2023 | DMD | 分布匹配式一步蒸馏 | 从轨迹模仿转向分布对齐 |
| 2024 | DMD2 / Rectified Diffusion | 改善一步生成质量和稳定性 | 少步/一步生成继续逼近教师 |
| 2025 | Phased DMD | 分阶段分布匹配 | 在复杂任务中兼顾质量和多样性 |

## DDPM 到 DDIM

`DDPM` 的核心是正向逐步加噪，反向学习逐步去噪。训练时通常预测噪声、\(x_0\) 或 velocity，把难的生成建模转成监督学习式的去噪问题。

它的价值是训练稳定、质量高、与 score matching 和变分推断建立联系；代价是采样常需几百到上千步，部署时延迟很高。

`DDIM` 的关键贡献是改变采样路径，而不是重训模型。它证明可以在兼容 DDPM 训练框架下构造非马尔可夫反向过程，并在特定设置下近似确定性采样。这让扩散采样从“逐步反马尔可夫链”走向“沿连续轨迹做离散积分”。

## ODE Solver 时代

Score SDE 和 probability flow ODE 把扩散模型放进连续时间框架。采样可以被理解为逆时间 SDE 或确定性 ODE 的数值求解。

这带来一个直接工程收益：同一个训练好的模型，换更好的求解器就能更快出图。

| 求解器 | 特点 |
| --- | --- |
| Euler | 一阶、简单、每步便宜，低步数误差较大 |
| Heun | 二阶 predictor-corrector，更稳但每步更贵 |
| DPM-Solver | 针对 diffusion ODE 设计的高阶求解器 |
| DPM-Solver++ | 改善 classifier-free guidance 下的稳定性 |
| EDM sampler | 系统整理噪声参数化和采样细节 |

这一阶段的核心不是改变模型训练，而是把采样当作数值分析问题。

## 蒸馏与一致性

少步采样进一步依赖蒸馏和一致性学习。Progressive Distillation 把多步模型逐级压缩成更少步；Consistency Models 和 LCM 直接学习不同噪声水平之间的一致映射，让模型能在很少步内到达数据分布。

这类方法的核心取舍是：

1. 步数更少，速度更快；
2. 训练目标更复杂；
3. 可能损失多样性或细节；
4. 对 teacher、数据和噪声日程更敏感。

少步模型的验收不能只看单张视觉效果，还要看多样性、文本对齐、结构稳定和不同 guidance 强度下的表现。

## DMD 与分布匹配

`DMD`（Distribution Matching Distillation）代表了一条从轨迹模仿转向分布对齐的路线。它不只要求 student 逐步模仿 teacher 的去噪轨迹，而是更直接地让 student 生成分布对齐 teacher / data 分布。

`DMD2` 继续改善一步蒸馏中的模糊、训练不稳和细节不足；`Phased DMD` 则把分布匹配分阶段进行，试图在复杂生成任务中兼顾结构、细节和多样性。

这条路线的吸引力很明确：如果能用一步或极少步达到多步扩散质量，推理成本会大幅下降。但它也更容易暴露 teacher bias、模式坍缩和分布覆盖不足。

## Rectified Flow 与收口判断

Rectified Flow / Rectified Diffusion 试图学习更直的生成路径，把噪声到数据的传输路径整流，从而减少采样步数。它和 DMD、consistency、LCM 都指向同一个工程目标：少步甚至一步生成。

理解扩散演化时，不必把每个方法割裂成独立名词。更实用的判断是：

1. 它改变训练目标，还是只改变采样器；
2. 它减少步数的代价是什么；
3. 它是否保持多样性、结构和条件对齐；
4. 它是否在真实分辨率、真实 prompt 和真实延迟预算下成立。

扩散模型的发展主线，就是在质量、步数、稳定性和多样性之间不断重新分配代价。
