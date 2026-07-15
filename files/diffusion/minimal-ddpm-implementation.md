# 最小 DDPM 实现：让公式、训练与采样逐行对齐

这一实现只保留 DDPM 闭环中不可缺少的部分：噪声日程、任意时间步加噪、带时间条件的去噪网络、噪声预测损失、单步反向转移和完整采样链。它的目标不是复刻生产级训练框架，而是让每条核心公式都能在一份可测试的 PyTorch 代码里找到对应位置。

## 本文解决什么问题

本文把 [DDPM 数学与训练](training.md) 中的六个对象映射到代码，并解释张量形状、时间下标和最终一步的边界处理。配套实现支持两种验证方式：

- 单元测试分别检查 schedule、闭式加噪、网络 shape、参数更新和采样可复现性；
- `smoke` 命令用随机张量完成一次优化器更新和一条短采样链，不下载数据集。

完整代码见 [`minimal_ddpm.py`](../assets/examples/diffusion/minimal_ddpm.py)，测试见 [`test_minimal_ddpm.py`](../assets/examples/diffusion/test_minimal_ddpm.py)。

## 前置知识

- [扩散模型基础](foundations-and-core-intuition.md)：概率路径、局部预测与 sampler 的职责。
- [DDPM 数学与训练](training.md)：前向边缘、反向均值和噪声预测目标。
- PyTorch 的 BCHW 张量、`nn.Module`、反向传播和优化器。

本文使用 0-based 时间下标：`t=0` 是训练日程中的第一个带噪状态，`t=T-1` 是噪声最强的状态。公式文献常使用 \(1,\ldots,T\)，对照代码时需要把下标整体平移。

## 核心机制

### 1. 预计算噪声日程

`make_linear_schedule` 先构造 \(\beta_t\)，再一次性缓存：

\[
\alpha_t=1-\beta_t,\qquad
\bar\alpha_t=\prod_{s=1}^{t}\alpha_s,\qquad
\tilde\beta_t=
\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t.
\]

代码同时保存 \(\sqrt{\bar\alpha_t}\) 与 \(\sqrt{1-\bar\alpha_t}\)，避免每个 batch 重复计算。`DiffusionSchedule` 不是可学习模块；它描述前向路径以及教学版 DDPM sampler 使用的后验方差。

线性 beta 日程便于对应原始 DDPM，但不代表它是所有数据、分辨率和参数化下的最佳选择。生产系统还常使用 cosine、log-SNR 或 \(\sigma\)-space 日程。

### 2. 用闭式公式构造任意 \(x_t\)

`q_sample` 实现：

\[
x_t=\sqrt{\bar\alpha_t}x_0+
\sqrt{1-\bar\alpha_t}\epsilon.
\]

`_extract` 根据 batch 中每个样本自己的时间步，从一维 schedule 中取出系数，并 reshape 成 `[B, 1, 1, 1]`。广播后，同一张图的所有通道和像素使用相同噪声系数，但噪声张量 \(\epsilon\) 的每个元素仍独立采样。

这个函数解释了为什么训练不需要真的执行 \(t\) 次前向转移：高斯线性链允许直接得到任意边缘 \(q(x_t\mid x_0)\)。

### 3. 去噪器必须读取时间

`TinyDenoiser` 是一层下采样、一层上采样的 U-Net 形状网络。它包含：

- 正弦时间嵌入与 MLP；
- 在残差块中注入的时间投影；
- 编码器到解码器的 skip connection；
- 与输入相同 shape 的噪声预测头。

网络接口是：

```python
predicted_noise = model(xt, t)  # [B, C, H, W]
```

这份网络刻意很小，只负责验证扩散接口。真实图像系统会增加多尺度 block、attention、条件注入和更大的通道数；DiT 则把主干改成 Transformer，但仍必须接收当前状态和时间。

### 4. 一次训练更新

`diffusion_loss` 对应最常见的简化目标：

\[
\mathcal L_{\text{simple}}
=\mathbb E\left[
\lVert\epsilon-\epsilon_\theta(x_t,t)\rVert_2^2
\right].
\]

训练循环只需四步：

```python
t = torch.randint(0, num_steps, (batch_size,))
loss = diffusion_loss(model, x0, t, schedule)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

输入数据应先缩放到与训练配置一致的范围，例如 `[-1, 1]`。随机图像可以验证梯度和 shape，却不能让模型学到有意义的数据分布；要得到可辨识样本，必须接入真实数据、重复训练并保存 EMA 或 checkpoint。

### 5. 单步反向转移

`p_sample` 先用噪声预测计算反向均值：

\[
\mu_\theta(x_t,t)=
\frac{1}{\sqrt{\alpha_t}}
\left(
x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}
\epsilon_\theta(x_t,t)
\right).
\]

当 \(t>0\) 时，再加入方差为 \(\tilde\beta_t\) 的高斯噪声；当 `t == 0` 时直接返回均值。最后一步若仍加入随机噪声，会把已经恢复的样本再次污染。

实现允许 batch 中出现不同时间步，因而随机项使用逐样本 `nonzero_mask`。生产 sampler 通常让整个 batch 共享一个时间步，但显式处理 mask 能把边界条件写清楚。

### 6. 从 \(x_T\) 执行完整采样链

`sample_loop` 从标准高斯初始化状态，然后倒序遍历整个日程：

```python
xt = torch.randn(shape, generator=generator)
for step in reversed(range(num_steps)):
    t = torch.full((batch_size,), step, dtype=torch.long)
    xt = p_sample(model, xt, t, schedule, generator)
```

局部预测只有被 sampler 连成闭环后才构成生成模型。`seed` 同时控制初始噪声和各反向步的随机项，使测试可以验证完整链路是否可复现。

### 7. 运行验证

在仓库根目录执行：

```bash
python -m venv .venv-diffusion
source .venv-diffusion/bin/activate
python -m pip install -r files/assets/examples/diffusion/requirements.txt
python -m pytest files/assets/examples/diffusion/test_minimal_ddpm.py -q
python files/assets/examples/diffusion/minimal_ddpm.py smoke
```

最后一条命令会输出 JSON，其中包括本次 MSE、采样张量 shape，以及结果是否全部为有限值。环境安装、影响范围和清理方式见配套 [`README`](../assets/examples/diffusion/README.md)。

## 常见误解与失效边界

**把 smoke test 当成训练结果。**它只证明 forward、backward 和 sampler 能连通；随机张量不包含可学习的数据结构。

**用很少的训练日程评价 DDPM 质量。**测试中的 4 到 8 步用于降低执行成本，不足以代表原始 DDPM 的生成设置。少步生成通常需要更合适的路径、solver 或蒸馏。

**只把模型移到 GPU。**`x0`、`t` 和 schedule 必须与模型处在同一设备；实现会主动拒绝 schedule 与采样状态跨设备。

**把 posterior variance 当成唯一正确选择。**这里采用 \(\tilde\beta_t\) 的固定方差版本。Improved DDPM、learned variance 与其他 sampler 的方差约定不同。

**认为通过 shape 测试就代表数学正确。**shape 只能排除接口错误；因此测试还单独验证闭式加噪值、参数确实更新、`t=0` 无随机项以及固定 seed 的完整轨迹一致。

## 检查理解

1. `_extract` 为什么要把 `[B]` reshape 成 `[B, 1, 1, 1]`？
2. `q_sample` 为什么能一次构造任意时间步，而 `sample_loop` 仍需逐步执行？
3. 如果 `p_sample` 在 `t=0` 加入噪声，会出现什么结果？
4. 为什么网络输出 shape 必须与采样噪声相同？
5. smoke test 能证明哪些性质，不能证明哪些性质？

## 相关阅读与下一步

- 网络扩展：[扩散网络架构与 Stable Diffusion](architecture-and-stable-diffusion.md)。
- 更快采样：[采样与推理加速](inference.md)。
- 时间权重与输出坐标：[噪声日程与参数化](noise-schedules-and-parameterization.md)。
- 原始来源：[DDPM](https://arxiv.org/abs/2006.11239)、[Improved DDPM](https://arxiv.org/abs/2102.09672)。
- 返回：[扩散模型专题](index.md)。
