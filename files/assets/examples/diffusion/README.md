# 最小 DDPM 示例

这组文件对应博客文章《最小 DDPM 实现》，用于验证 DDPM 的前向加噪、噪声预测训练和反向采样链。

## 隔离环境

在仓库根目录执行：

```bash
python -m venv .venv-diffusion
source .venv-diffusion/bin/activate
python -m pip install -r files/assets/examples/diffusion/requirements.txt
```

安装只影响 `.venv-diffusion`。不再需要时可先退出环境，再删除该目录：

```bash
deactivate
rm -rf .venv-diffusion
```

## 验证

运行单元测试：

```bash
python -m pytest files/assets/examples/diffusion/test_minimal_ddpm.py -q
```

运行不需要数据集的端到端 smoke test：

```bash
python files/assets/examples/diffusion/minimal_ddpm.py smoke
```

命令完成一次优化器更新和一条短采样链，输出类似：

```json
{"loss": 1.29, "sample_shape": [2, 1, 8, 8], "finite": true}
```

它验证代码路径与数值有限性，不代表模型已经学会真实数据分布。

## 文件与公式对应

`minimal_ddpm.py` 包含以下六个部分：

1. `DiffusionSchedule` 与 `make_linear_schedule` 预计算 \(\beta_t\)、\(\alpha_t\)、\(\bar\alpha_t\) 和后验方差。
2. `q_sample` 使用闭式边缘
   \[
   x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon
   \]
   一次构造任意时间步的带噪状态。
3. `TinyDenoiser` 使用时间嵌入和一层 U-Net 形状结构预测噪声，输入输出均为 BCHW 张量。
4. `diffusion_loss` 计算预测噪声与真实噪声之间的均方误差。
5. `p_sample` 把噪声预测换算成 DDPM 反向均值，并在非最终步加入后验随机项。
6. `sample_loop` 从标准高斯开始，倒序执行整条反向链。

`test_minimal_ddpm.py` 不只检查函数能否运行，还验证：

- beta 位于合法区间且累计信号系数单调下降；
- 闭式加噪结果与手工计算一致，且不修改输入；
- 非法时间步能够被明确拒绝；
- denoiser 保持图像 shape；
- 一次反向传播确实更新模型参数；
- `t=0` 的反向步不再加入随机噪声；
- 固定 seed 时完整采样链可复现；
- 命令行 smoke test 能完成一次训练更新与完整短链采样。

## smoke 参数

默认命令使用 8 个扩散步、两张单通道 8×8 随机图像：

```bash
python files/assets/examples/diffusion/minimal_ddpm.py smoke
```

可以改变测试规模：

```bash
python files/assets/examples/diffusion/minimal_ddpm.py smoke \
  --steps 16 \
  --image-size 16 \
  --batch-size 4
```

`image-size` 必须是至少为 4 的偶数，因为示例网络包含一次 2 倍下采样与上采样。增加 `steps` 会线性增加 smoke 命令中采样阶段的网络前向次数。

输出字段含义：

| 字段 | 含义 |
| --- | --- |
| `loss` | 随机 batch 上一次噪声预测 MSE，仅用于确认 loss 有限 |
| `sample_shape` | 完整反向链最终输出的 BCHW shape |
| `finite` | 输出是否全部为有限浮点数 |

随机数据没有稳定结构，因此 loss 不需要下降到特定阈值，生成结果也不应被解释成图像质量指标。

## 接入真实训练前需要补什么

这份代码刻意不包含数据下载、长时间训练、日志和 checkpoint。要接入 MNIST、CIFAR-10 或自定义数据，至少需要补充：

- 把数据转换成 BCHW 浮点张量，并统一缩放范围；
- 为每个 batch 随机采样时间步；
- 重复调用 `diffusion_loss` 与优化器；
- 按固定间隔运行真实 `sample_loop`，而不只观察训练 MSE；
- 保存模型权重、优化器状态、schedule 配置和随机种子；
- 视训练规模加入 EMA、混合精度、梯度裁剪和分布式数据加载。

若修改 prediction type、noise schedule 或反向方差，训练目标和 `p_sample` 必须成对修改。只替换其中一端，即使张量 shape 完全一致，也会让 sampler 错误解释模型输出。

## 已知边界

- 线性 beta schedule 适合公式对照，不代表最佳训练配方。
- 一层小型 U-Net 只用于验证接口，不具备高分辨率建模能力。
- 示例使用固定后验方差，没有实现 learned variance。
- sampler 是原始 ancestral DDPM 形式，没有实现 DDIM、DPM-Solver 或 CFG。
- smoke test 在 CPU 上即可运行；它不测 GPU 性能、显存或混合精度稳定性。

完整推导、边界条件和后续扩展见博客正文：

- [最小 DDPM 实现](../../../diffusion/minimal-ddpm-implementation.md)
- [DDPM 数学与训练](../../../diffusion/training.md)
- [扩散网络架构与 Stable Diffusion](../../../diffusion/architecture-and-stable-diffusion.md)
