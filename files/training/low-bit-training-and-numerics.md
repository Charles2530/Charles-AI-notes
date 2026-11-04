# 一个微小更新怎样在低精度训练里消失

设参数为1，优化器希望减去0.001。若把结果0.999直接写成BF16，它可能又舍入回1。训练代码已经执行step，参数却没有发生预期变化。理解低精度训练，需要沿表示、乘法、累加、梯度和写回逐段检查。

## 指数范围与有效精度承担不同作用

浮点数可表示的最大幅度主要受指数位影响，相邻值的间距受有效尾数精度影响。FP16有较窄范围，BF16范围较宽但尾数更短。范围足够大，并不保证能分辨微小差值。

BF16在1下方的相邻值约为0.99609375，0.999更接近1。使用更高精度主权重或更新状态，可以先积累这些微小变化，再生成计算所需的低精度副本。具体框架是否维护主副本，需要查优化器实现。

## 低精度操作数不等于低精度累加

矩阵乘会对K个乘积求和。即使每个操作数可表示，长归约也可能积累舍入误差。常见路径以低精度存储或乘法、高一些的精度累加，再转换输出。

为定位误差，可固定输入比较三种路径，先只降低输入精度、保持参考累加，再替换真实kernel，最后加入输出转换。这样能分开输入近似、累加顺序和输出舍入的影响。

![DeepSeek-V3 FP8 framework 原论文图](../assets/images/paper-deep-dives/technical-reports/deepseek-v3/figure-6-fp8-framework.png){ width="920" }

<small>图源：[DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)，Figure 6。原图展示 DeepSeek-V3 的 FP8 mixed precision training framework：GEMM 相关路径进入 FP8，部分状态、累加、embedding、MoE dispatch/combination 和 optimizer 相关路径保留更高精度或特殊处理。</small>

![DeepSeek-V3 FP8 accumulation 原论文图](../assets/images/paper-deep-dives/technical-reports/deepseek-v3/figure-7-fp8-accumulation.png){ width="880" }

<small>图源：[DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)，Figure 7。原图展示细粒度量化、tile/group scale 与提高累加精度的设计；读图时重点看 FP8 元素只负责输入表示，GEMM 的部分和最终累加仍要由更安全的精度路径保护。</small>

## Scale把数值送进合适范围

设反量化约定为 $x\approx s x_q$。量化时除以s，再cast；恢复时乘s。若某块大值抬高s，普通值可能失去分辨率；若s太小，大值可能溢出或截断。

Per-tensor、per-channel和per-block粒度改变了哪些数共用尺度。动态scale还增加统计与转换成本；历史scale会引入跨步状态。框架若存倒数尺度，公式中的乘除位置也应相应调整。

## Loss scaling只处理特定路径

FP16反向中一些小梯度可能下溢。把loss乘以S，会让梯度也乘S；在optimizer step前除回S，理想算术下保持原梯度。若已出现overflow，scaler可能跳过更新并调整尺度。

裁剪前必须先unscale，否则裁剪阈值作用在放大后的梯度上。nanoGPT的循环按 `unscale_`、clip、scaler.step、scaler.update组织。Loss scaling不能修复已经溢出的前向softmax，也不能恢复错误mask删除的信息。

![FP8 training loss curves 原论文图](../assets/images/paper-figures/foundations/fp8-formats-figure-1-training-loss.png){ width="700" }

<small>图源：[FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)，Figure 1。原图比较 BF16 与 FP8 路线在 GPT/BERT 类模型上的 loss 或 perplexity 曲线；读图时重点不是“FP8 也能训”，而是格式选择、scale 策略、累加精度和敏感算子保护共同成立时，曲线才可能贴近 BF16。</small>

## 误差会通过训练反馈

推理只用当前权重计算输出，训练还用近似梯度改变下一步权重。一个无偏小误差、持续偏差和偶发非有限值，会产生不同长期行为。因此一次forward对齐不足以证明训练方案稳定。

固定小数据做多步对照，记录loss、梯度范数、参数更新范数与跳步次数，再进行中断恢复。若FP8尺度历史或优化器状态影响更新，也应纳入保存。最后用独立评测判断训练结果，不能只依据短期loss曲线接近。

更低位宽的格式通常需要更强的范围、布局与kernel协同。应从已验证的局部路径逐步扩展，不把整个训练图简单改成一种dtype。


## 对照源码与延伸阅读

- [混合精度训练循环与unscale顺序](https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/train.py)
- [低精度矩阵乘的data/scale接口](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/README.md)
