# 扩散训练与表示

**这一页回答三个问题**：

1. 前向过程到底怎么定义。
2. 反向过程到底学什么。
3. 为什么“预测噪声”会变成一个有效训练目标。

DDPM 原论文的 graphical model 可以先帮你把训练问题定位清楚：训练不是直接从 \(x_T\) 一口气生成 \(x_0\)，而是在很多时间步上学习局部反向转移。后续的噪声预测、\(x_0\) 预测和 v-prediction，都是在这个骨架上选择不同参数化。

![DDPM graphical model 原论文图](../assets/images/paper-figures/diffusion/ddpm-figure-2-graphical-model.png){ width="620" }

<small>图源：[Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)，Figure 2。原论文图意：DDPM 将数据逐步加噪形成 \(x_1,\ldots,x_T\)，再学习反向链 \(p_\theta(x_{t-1}\mid x_t)\)。</small>

!!! note "图解：训练页里的 DDPM 图要看局部反向转移"
    图里的每个 \(p_\theta(x_{t-1}\mid x_t)\) 都是在学习一个局部去噪转移，而不是一次从 \(x_T\) 直接跳到 \(x_0\)。训练时随机抽一个时间步 \(t\)，构造带噪样本 \(x_t\)，再让网络预测噪声、干净样本或 velocity。如果你只是入门，先把 `epsilon prediction` 当作默认坐标系；如果你在看 latent diffusion、少步采样或强 guidance，再重点理解 `v-prediction` 为什么常更稳。

!!! note "难点解释：为什么可以直接采样任意时刻"
    前向加噪虽然写成一步步马尔可夫链，但高斯噪声的叠加仍然是高斯。因此训练时不必真的从 \(x_0\) 加噪到 \(x_t\) 走 \(t\) 次，可以用闭式公式一次构造 \(x_t\)。这就是扩散训练能高效做随机时间步监督的关键。

!!! example "有趣例子：调一杯奶茶"
    \(x_0\) 像原味茶，\(\epsilon\) 像奶和冰。不同 \(t\) 只是不同配比：前期茶味浓，后期奶冰越来越多。模型训练就是随机拿一杯不同混合比例的奶茶，让它猜“里面加了多少噪声”。

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

噪声预测之所以长期成为主流，不只是因为它最早出现，而是因为它把训练目标写成了非常统一的监督形式。网络不必直接恢复完整图像，只需要回答“这张带噪图里混进来了什么噪声”。这种目标在实现、调参和稳定性上都相对友好。

### 原图预测

\[
\hat{x}_0 = x_{0,\theta}(x_t,t)
\]

对图像恢复直觉更强，但不同噪声区间的尺度差异更敏感。

`x_0` 预测看上去更接近最终任务，因为网络直接去猜干净样本本身。但也正因为如此，它会在高噪声区间承受更大的回归压力。离真实图像越远的样本，直接回归原图越容易受到尺度和细节缺失影响。

### 速度预测

**定义**：

\[
v = \alpha_t \epsilon - \sigma_t x_0
\]

网络预测 \(v_\theta(x_t,t)\)。这类参数化在 latent diffusion 和少步采样场景里常更稳。

`v` 参数化更像是在学习“当前状态该往哪个混合方向移动”。它同时吸收了信号项和噪声项的信息，因此在不少 latent diffusion、少步采样和 guidance 路线里会显得更平衡。也正因为如此，近年的很多工程实现更愿意把它当成默认更舒服的训练坐标系。

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

这段公式的真正工程意义在于：训练阶段模型已经同时见过“有条件”和“无条件”两种世界，因此推理阶段才能用两者差异来人为放大条件作用。换句话说，`CFG` 虽然在推理时起作用，但它的前提其实是在训练里就把这两个分支准备好了。

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

更直白地说，扩散训练里真正贵的错误通常不是“公式写错了”，而是**模型见到的世界不够对**。caption 太粗、结构标注太稀、长尾风格太少、多对象关系覆盖不够、分辨率分布失衡，都会在采样结果里以一种“模型似乎差一点”的方式出现。实际上问题并不一定在模型，而在训练分布本身。

这也是为什么成熟的扩散训练往往离不开数据清洗、条件质量检查、噪声时间采样设计和专门的长尾回流。目标函数只定义了方向，真正决定模型学到多少内容的，仍然是数据和覆盖。

## 快速代码示例

```python
import torch
import torch.nn.functional as F

def train_step(unet, scheduler, x0, text_embed):
    bsz = x0.size(0)
    t = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), device=x0.device)
    noise = torch.randn_like(x0)
    x_t = scheduler.add_noise(x0, noise, t)

    noise_pred = unet(x_t, t, encoder_hidden_states=text_embed).sample
    loss = F.mse_loss(noise_pred, noise)  # epsilon-prediction
    return loss
```

这段代码对应标准的扩散训练单步：随机采样时间步 `t`，把干净样本加噪得到 `x_t`，再让网络回归真实噪声。它体现了最核心的 **epsilon-prediction** 训练目标，便于和 `v-pred`、`x0-pred` 做等价对比。

## 工程收束

扩散训练要按噪声目标、数据分辨率、采样器耦合、条件控制和计算预算一起设计。最容易踩坑的是只抄默认配方、训练和推理目标割裂、高分辨率代价被低估；更稳的验收方式是区分 base model 与 downstream control，按分辨率建预算，对条件信号做专项 ablation，并固定参数化、条件、分辨率和训练-推理一致性四组对照。
