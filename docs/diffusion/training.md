# 扩散模型的训练与表示

**这一页回答三个问题**：

1. 前向过程到底怎么定义。
2. 反向过程到底学什么。
3. 为什么“预测噪声”会变成一个有效训练目标。

## 1. 前向过程：把数据逐步推向高斯分布

**前向链定义为**：

\[
q(x_{1:T} \mid x_0) = \prod_{t=1}^{T} q(x_t \mid x_{t-1}),\qquad q(x_t \mid x_{t-1}) = \mathcal{N}\left(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I\right)
\]

当 \(T\) 足够大时，\(x_T\) 会近似标准高斯。

**更实用的是闭式重参数化**：

\[
x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon
\]

这样训练时无需真的迭代加噪 \(t\) 次。

### 例子：给一张猫照片加噪

若 \(t\) 很小，猫的轮廓仍清晰，只是有些颗粒。

若 \(t\) 中等，能看出“这大概是一只猫”，但眼睛和毛发细节消失。

若 \(t\) 接近 \(T\)，图像已经接近白噪声，肉眼几乎无法识别。

## 2. 反向过程：学习从噪声往回走

**理想的反向过程写作**：

\[
p_\theta(x_{0:T}) = p(x_T)\prod_{t=1}^{T} p_\theta(x_{t-1}\mid x_t)
\]

其中：

\[
p_\theta(x_{t-1}\mid x_t)=\mathcal{N}(x_{t-1}; \mu_\theta(x_t,t), \Sigma_\theta(x_t,t))
\]

很多实现里固定协方差，只学习均值；而均值又可以通过噪声预测网络来表达。

## 3. 从 ELBO 到噪声预测

**标准训练目标来源于变分下界**：

\[
\mathcal{L}_{\text{vlb}} = \mathbb{E}_q \left[-\log p_\theta(x_0 \mid x_1) + \sum_{t=2}^{T} D_{\text{KL}}\left(q(x_{t-1}\mid x_t, x_0)\,\|\,p_\theta(x_{t-1}\mid x_t)\right) + D_{\text{KL}}\left(q(x_T\mid x_0)\,\|\,p(x_T)\right)\right]
\]

在 Ho et al. 的参数化下，这个目标可简化为加权的噪声回归：

\[
\mathcal{L}_{\epsilon} = \mathbb{E}_{x_0,\epsilon,t}\left[\left\|\epsilon - \epsilon_\theta(x_t,t)\right\|_2^2\right]
\]

**这里的直觉是**：

- \(x_t\) 已知。
- 生成它时注入的噪声 \(\epsilon\) 也已知。
- 那么网络只要学会“这张带噪图里混入了多少噪声”，就能反推出更干净的样本。

## 4. 三种常见参数化

### 噪声预测

\[
\hat{\epsilon} = \epsilon_\theta(x_t,t)
\]

优点是实现简单、训练稳定。

### 原图预测

\[
\hat{x}_0 = x_{0,\theta}(x_t,t)
\]

对图像恢复直觉更强，但不同噪声区间的尺度差异更敏感。

### 速度预测

**定义**：

\[
v = \alpha_t \epsilon - \sigma_t x_0
\]

网络预测 \(v_\theta(x_t,t)\)。这类参数化在 latent diffusion 和少步采样场景里常更稳。

## 5. 从噪声预测恢复 \(x_0\)

如果模型预测了 \(\hat{\epsilon}\)，那么：

\[
\hat{x}_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t}\hat{\epsilon}}{\sqrt{\bar{\alpha}_t}}
\]

这个式子很像“信号减去估计噪声，再按信噪比做重标定”。

### 例子：老照片修复

想象 \(x_t\) 是一张很脏的老照片：

- 左边人物脸部还在，但有雪花噪点。
- 背景窗框模糊，颜色发灰。

如果网络能正确估计噪声，就等价于判断“哪些像素波动是噪声，哪些才是真实轮廓”。于是恢复后的 \(\hat{x}_0\) 会先出现脸部轮廓，再补窗框纹理，最后补肤色和光照。

## 6. 条件生成与 classifier-free guidance

**条件扩散通常学习**：

\[
\epsilon_\theta(x_t, t, c)
\]

其中 \(c\) 可以是文本、类别或图像条件。`classifier-free guidance` 的常见写法为：

\[
\hat{\epsilon}_{\text{cfg}} = \epsilon_\theta(x_t,t,\varnothing) + s\left(\epsilon_\theta(x_t,t,c)-\epsilon_\theta(x_t,t,\varnothing)\right)
\]

这里 \(s\) 是 guidance scale。

**直觉上**：

- 无条件分支负责“生成一张合理图像”。
- 条件分支负责“朝着提示词方向偏移”。
- 两者差值越大，说明条件带来的方向越明确。

### 例子：提示词“雨夜中的霓虹便利店”

若 \(s\) 太小，图像可能只是普通街景。

若 \(s\) 合理，画面会更聚焦于“便利店、霓虹、雨夜反光”。

若 \(s\) 过大，可能出现颜色过饱和、结构变形、局部过度锐化。

## 7. 训练时真正难的不是公式，而是分布覆盖

工程上，扩散模型训练的核心难点通常不在于公式本身，而在于：

- 数据质量是否足够高。
- 文本标注是否覆盖语义细节。
- 时间采样策略是否均衡。
- 网络容量与噪声日程是否匹配。

因此，好的扩散训练往往是“目标函数 + 数据系统 + 训练稳定性”一起作用的结果。
