# Score Matching、SDE 与 Probability Flow

如果只把扩散模型理解成“往图像里加噪声，再一步步去噪”，就会错过它真正的数学骨架。扩散模型之所以能够从 DDPM 发展到 DDIM、DPM-Solver、Rectified Flow 和 Consistency，一条关键主线就是：它本质上是在学习数据分布的 score，并把生成过程重写成随机微分方程（SDE）或常微分方程（ODE）上的数值问题。本页把这条骨架系统串起来。

## 1. 什么是 score

对一个分布 $p(x)$，其 score 定义为

$$
s(x) = \nabla_x \log p(x).
$$

直观上，score 告诉你：在当前点附近，往哪个方向走会让概率密度增加得最快。若把数据分布想成一片山谷地形，score 就像每个位置上的“上坡方向箭头”。

对于复杂高维图像分布，我们无法直接写出 $\log p(x)$，但可以学习其梯度场。只要这个梯度场足够准确，就能逐步把噪声样本拉回高概率区域。

## 2. 从去噪到 score matching

在连续扰动下，令

$$
x_t = \alpha_t x_0 + \sigma_t \epsilon,\quad \epsilon \sim \mathcal{N}(0, I),
$$

则带噪分布为 $p_t(x)$。扩散训练的关键事实是：预测噪声 $\epsilon$、预测干净样本 $x_0$ 和预测 score 这三件事本质上可以互相转换。一个常见关系是

$$
\nabla_x \log p_t(x)
=
-\frac{1}{\sigma_t}\epsilon_\theta(x_t, t)
$$

或在等价参数化下差一个时间依赖系数。也就是说，很多看似“噪声回归”的训练，其实在学习时间条件的 score field。

## 3. Score Matching 的训练直觉

原始 score matching 的目标是让模型输出的梯度场接近真实梯度场。因为直接访问真实 score 困难，扩散模型利用带噪过程，把难问题变成对已知高斯扰动的回归。于是常见损失

$$
\mathcal{L}
=
\mathbb{E}_{x_0,\epsilon,t}
\left[
\|\epsilon-\epsilon_\theta(x_t,t)\|^2
\right]
$$

既是噪声回归，也是 score matching 的一种可训练实现。

## 4. 连续时间视角下的 SDE

把离散扩散步数无限细化，可以得到正向 SDE：

$$
dx = f(x,t)\,dt + g(t)\,d w_t,
$$

其中 $w_t$ 是 Wiener process。这个方程描述“数据如何逐渐被随机扰动推向简单噪声分布”。不同扩散家族，如 VP、VE、sub-VP，本质上对应不同的 $f$ 和 $g$ 设计。

反向生成则由 reverse-time SDE 给出：

$$
dx = \left[f(x,t) - g(t)^2 \nabla_x \log p_t(x)\right]dt + g(t)\,d\bar{w}_t.
$$

一旦有了 score，反向过程就被确定了。

## 5. 为什么 SDE 视角重要

**它至少带来三件事**：

1. 统一 DDPM、score-based model、连续时间采样器的表述；
2. 让采样器设计变成数值积分问题；
3. 为 Probability Flow ODE、DPM-Solver 和高阶方法提供理论基础。

换句话说，SDE 视角把“生成图片”从经验技巧变成了微分方程求解问题。

## 6. Probability Flow ODE

**有一个非常重要的事实**：与某个 SDE 对应，存在一个 ODE，它在边缘分布层面与该 SDE 一致。这就是 probability flow ODE：

$$
\frac{dx}{dt}
=
f(x,t) - \frac{1}{2} g(t)^2 \nabla_x \log p_t(x).
$$

它没有随机噪声项，因此给定初始点时轨迹是确定的。DDIM 可以被视为这条思路的一种离散近似，而很多现代高阶求解器正是在这个 ODE 上做数值优化。

## 7. DDPM 与 DDIM 在这条线上的位置

### 7.1 DDPM

更接近逆向马尔可夫链采样，每一步都引入随机性。优点是理论自然、生成多样性强；缺点是步数多。

### 7.2 DDIM

把采样路径改写成近似确定性的更新。优点是更快，更适合少步采样；缺点是当步数压得太低时，误差容易积累。

从 SDE/ODE 视角看，两者不是完全不同的模型，而是对同一连续过程的不同离散化方式。

## 8. 高阶求解器为什么更快

如果生成是 ODE 积分问题，那么 Euler、Heun、Runge-Kutta、DPM-Solver 等本质上是在问：用同样少的步数，怎样更准确地逼近真实轨迹。低阶方法每步只用局部一阶信息，高阶方法则利用更多局部结构，减少全局积分误差。

因此很多“采样加速”其实不是模型本身变了，而是求解器更聪明了。

## 9. 为什么 Probability Flow ODE 适合蒸馏

一旦你把生成轨迹看成确定性流，就更容易做：

1. 轨迹蒸馏
2. 一步或少步 distillation
3. Consistency 映射
4. Rectified Flow 的路径整流

因为目标从“每一步注入随机噪声的随机过程”变成了“从起点走到终点的可近似轨迹”。

## 10. 例子：从雾中找路

可以把数据生成想象成从大雾里的山脚走回城市中心。score 是每个位置上的“往市中心更近”的箭头；SDE 是你一边走一边被风吹扰动；Probability Flow ODE 则是假设你只按照平均最合理方向前进，不再额外被风随机推来推去。DDPM 像每一步都要考虑风，DDIM 和高阶 ODE 求解器则更像在更准确地拟合一条可重复的导航路线。

## 11. 对后续研究的意义

理解这条数学线索之后，再看：

1. DPM-Solver 为什么能少步保持质量；
2. Rectified Flow 为什么强调路径更直；
3. Consistency 为什么试图跨时间学习同一终点映射；
4. 动作扩散和视频扩散为什么也能接上这些工具；

都会更容易放到同一张图里，而不是把它们看成互不相干的方法名。

## 12. 小结

扩散模型真正强大的地方，不只是“会去噪”，而是它把复杂分布生成重写成 score field 学习和 SDE/ODE 求解问题。理解 score matching、reverse-time SDE 与 probability flow ODE，等于掌握了现代扩散方法大部分加速、蒸馏和整流思路的共同母语。 
