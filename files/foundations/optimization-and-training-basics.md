# 优化与训练入门

训练不是“把模型多跑几轮”。训练是一条闭环：**数据给出样本，模型给出预测，loss 定义什么叫错，梯度告诉参数往哪改，optimizer 决定怎么迈步，评测判断这个方向是否真的有用。** 这篇只讲清这条闭环，后面的分布式训练、低比特训练、RLHF、扩散训练和世界模型训练都建立在它上面。

## 为什么训练像在地形里走路

神经网络有大量参数。每一组参数都会对应一个 loss 值。把这些 loss 想成高度，就得到 loss landscape：训练就是在高维地形里找低处。

![Loss landscape visualization 原论文图](../assets/images/paper-figures/foundations/loss-landscape-figure-resnet56.png){ width="820" }

<small>图源：[Visualizing the Loss Landscape of Neural Nets](https://arxiv.org/abs/1712.09913)。原图表达：用 filter normalization 等方法可视化网络周围的 loss surface，比较不同结构或训练配置下地形是否平滑。本站读法：optimizer 不是瞬间跳到最优点，而是在这个地形里沿局部梯度一步步移动；结构、归一化、初始化、batch 和学习率都会改变这条路是否好走。</small>

这张图能解释很多训练现象：

| 现象 | 在地形里的直觉 | 工程症状 |
| --- | --- | --- |
| 学习率太大 | 一步跨过谷底，甚至跳出稳定区域 | loss spike、震荡、NaN |
| 学习率太小 | 每步太短，卡在平台区 | loss 很慢，算力浪费 |
| 梯度噪声大 | 每个 batch 指的方向都不太一样 | 曲线抖动、复现实验方差大 |
| 目标函数错 | 地形低处不对应真实任务成功 | loss 降，线上/闭环指标不升 |

## 标准训练循环

最小训练循环只有四步：

```text
output = model(input)       # 前向：算预测
loss = loss_fn(output, y)   # 目标：算错误
loss.backward()             # 反向：把错误变成梯度
optimizer.step()            # 更新：用梯度改参数
```

对应的最小公式是：

\[
\theta \leftarrow \theta-\eta\nabla_\theta\mathcal{L}
\]

这里 \(\theta\) 是模型参数，\(\mathcal{L}\) 是 loss，\(\nabla_\theta\mathcal{L}\) 是 loss 对参数的梯度，\(\eta\) 是学习率。这行公式在说：沿着让 loss 下降的方向，小步更新参数。

真实训练循环会多很多行：

```text
for step, batch in enumerate(dataloader):
    output = model(batch.input)
    loss = loss_fn(output, batch.target)

    loss.backward()
    clip_grad_norm(model.parameters())
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    if step % eval_interval == 0:
        run_eval()
    if step % save_interval == 0:
        save_checkpoint()
```

这些额外行不是装饰。gradient clipping 防止异常 batch 把参数冲坏；scheduler 控制不同阶段的步长；eval 防止训练 loss 欺骗你；checkpoint 保证恢复后曲线可比。

## Loss 定义“模型错在哪里”

Loss 是训练方向的定义。模型会忠实优化 loss，但 loss 不一定忠实代表你的真实目标。

\[
\mathcal{L}=\ell(f_\theta(x),y)
\]

这行公式里，\(f_\theta(x)\) 是模型预测，\(y\) 是目标，\(\ell\) 是衡量预测和目标差距的函数。不同任务的“错”完全不同：

| 场景 | 模型在学什么 | 常见 loss | 真实验收 |
| --- | --- | --- | --- |
| 分类 | 类别概率 | cross entropy | accuracy、calibration、错误类别 |
| 语言模型 | 下一个 token 分布 | next-token CE | perplexity、任务能力、长上下文稳定性 |
| 扩散模型 | 去噪方向或 score | noise/velocity/score loss | FID、偏好评测、采样步数、条件控制 |
| VLA 行为克隆 | 观测到动作 | action CE、MSE、diffusion action loss | closed-loop success、恢复能力 |
| 世界模型 | 未来状态和任务信号 | reconstruction、latent prediction、reward/done loss | action sensitivity、rollout drift、规划收益 |
| RLHF/GRPO | 高奖励轨迹更可能 | policy loss、KL、reward/value loss | pass rate、reward hacking、人工评估 |

最常见的误判是：loss 下降就以为模型更好。比如 action-conditioned video world model 的平均 loss 下降，可能只是静态背景变得更像；真正影响抓取成功的接触、遮挡、反光 bucket 反而变差。Loss 只是代理目标，必须和分桶评测、闭环成功率、风险召回一起看。

## Backprop：错误怎样传回参数

反向传播用链式法则计算每个参数对 loss 的影响。对于深层模型，可以把它理解成：后面层的错误信号一层层往前传，告诉每个参数“你对这个错误贡献了多少”。

深层网络或长序列模型里，这个过程会反复乘 Jacobian：

\[
\frac{\partial L}{\partial h_t}
=
\frac{\partial L}{\partial h_T}
\prod_{k=t+1}^{T}
\frac{\partial h_k}{\partial h_{k-1}}
\]

如果这些乘积长期小于 1，梯度会逐步变小，早期层几乎收不到有效信号，这就是梯度消失；如果长期大于 1，梯度会被放大，导致 loss spike、NaN 或参数被冲坏，这就是梯度爆炸。

![Pascanu et al. Figure 6 exploding gradients](../assets/images/paper-figures/foundations/pascanu13-figure-6-exploding-gradients.png){ width="700" }

<small>图源：[On the difficulty of training recurrent neural networks](https://proceedings.mlr.press/v28/pascanu13.html)，Figure 6。原图表达：单隐藏单元 RNN 的 error surface 中存在高曲率墙，普通梯度下降可能因为梯度过大跳过稳定区域。本站读法：gradient clipping 的作用不是“修正目标”，而是在异常大梯度出现时限制单步更新幅度。</small>

![Pascanu et al. Figure 7 gradient clipping success](../assets/images/paper-figures/foundations/pascanu13-figure-7-gradient-clipping-success.png){ width="920" }

<small>图源：[On the difficulty of training recurrent neural networks](https://proceedings.mlr.press/v28/pascanu13.html)，Figure 7。原图表达：在 temporal order 任务上，序列长度变长后，不同初始化和训练策略成功率差异明显。本站读法：长序列训练失败往往不是单个模块坏，而是梯度路径、初始化、学习率和裁剪共同决定。</small>

排查时不要只看 loss：

| 症状 | 更像什么 | 需要一起看 |
| --- | --- | --- |
| loss 突然尖刺、grad norm 暴涨、随后 NaN | 梯度爆炸或异常 batch | global/per-layer grad norm、LR、overflow、数据 bucket |
| loss 长期平台、早期层 grad norm 接近 0 | 梯度消失或模型容量不足 | per-layer grad norm、activation 分布、初始化、结构 |
| train loss 降，eval 不升 | 过拟合或目标错位 | validation bucket、数据泄漏、业务指标 |
| 平均 loss 降，关键任务变差 | 长尾 bucket 被大桶淹没 | per-bucket loss、failure replay、risk recall |

## Optimizer 决定怎么迈步

梯度只告诉你局部方向，optimizer 决定实际怎么走。

| Optimizer | 机制直觉 | 适合场景 | 风险 |
| --- | --- | --- | --- |
| SGD | 直接沿梯度走 | 简单模型、强正则、可解释 baseline | 调参敏感，收敛慢 |
| SGD + Momentum | 保留历史方向，减少来回摆动 | 视觉模型、稳定大 batch | 动量过大可能冲过低谷 |
| Adam | 对不同参数自适应步长 | Transformer、稀疏梯度、默认起点 | 泛化和权重衰减要小心 |
| AdamW | 把 weight decay 从 Adam 更新中解耦 | LLM 和现代预训练常用 | 仍需调 LR、warmup、decay |
| Muon | 对矩阵权重做动量正交化 | 大规模预训练实验路线 | 工程复杂，适用边界需实测 |

AdamW 的重要性在于 decoupled weight decay：权重衰减不再和 Adam 的自适应归一化纠缠在一起。大模型训练里，AdamW 仍是稳健默认项；Muon 等新优化器值得作为实验假设，但不能只看短跑 loss，要看最终 eval、稳定性、吞吐和恢复语义。

## Scheduler 和 warmup 保护训练早期

大模型训练常见学习率过程是：

```text
warmup:   从 0 或很小的 LR 逐步升高
plateau:  保持一段高 LR 快速学习
decay:    后期逐步降低 LR 精修参数
```

Warmup 的作用是防止训练初期参数、激活和优化器状态还没稳定时被大步更新冲坏。Cosine decay 或 linear decay 则让训练从“快速探索”转向“细致收敛”。

如果 warmup 太短，常见症状是前几百/几千 step 就出现 loss spike 或 overflow；如果 decay 太快，模型还没学够就进入低学习率；如果 peak LR 太低，训练稳定但浪费算力。

## 混合精度：省算力，但要保护敏感路径

混合精度用 BF16/FP16/FP8 等低精度加速计算和节省显存，但保留必要的高精度状态。它带来的不是“免费加速”：

| 低精路径 | 省什么 | 可能伤什么 |
| --- | --- | --- |
| 权重低精 | 显存、带宽 | 小权重更新、量化误差 |
| 激活低精 | 显存、通信 | attention score、norm、softmax 稳定性 |
| optimizer state 分片/低精 | 单卡显存 | 恢复语义、更新精度 |
| KV cache 低精 | 推理显存和带宽 | 长上下文一致性、候选排序 |

低精训练和推理都要把质量回归同表报告。只说“显存降了”不够，要看 loss、eval、关键 bucket、NaN/overflow、long-context 和闭环指标有没有变。

## 为什么 loss 下降不等于模型可用

假设一个世界模型新 recipe 的平均训练 loss 从 `0.214` 降到 `0.198`，但分桶后是：

| Bucket | Token Share | Baseline Loss | New Loss | Closed-loop |
| --- | ---: | ---: | ---: | ---: |
| easy static scene | `72%` | `0.180` | `0.158` | `+0.3%` |
| moving object | `18%` | `0.246` | `0.241` | `+0.1%` |
| contact / occlusion | `10%` | `0.310` | `0.337` | `-6.8%` |

平均 loss 下降，是因为 easy bucket 占比大；真正影响机器人任务的 contact/occlusion bucket 变差。优化器没有“犯错”，它只是忠实优化了你给它的加权目标。

正确排查链是：

```text
症状：loss 更低，但闭环成功率或风险召回下降
指标：per-bucket loss、grad norm contribution、数据曝光、closed-loop success、risk recall
机制：平均目标被大桶主导，稀有高价值样本梯度不足
修复：重加权、hard bucket oversampling、auxiliary risk loss、按桶设置评测门槛
边界：过度重加权会牺牲主分布稳定性，甚至过拟合少量异常样本
```

## 一个训练改动是否可信

读训练日志或论文消融时，至少看这几类证据：

| 证据 | 说明什么 | 不足 |
| --- | --- | --- |
| train loss | 代理目标是否被优化 | 不代表泛化和任务成功 |
| eval loss / benchmark | 静态验证集是否提升 | 可能和真实任务错位 |
| per-bucket metric | 哪类样本变好或变差 | 分桶设计可能漏掉关键失败 |
| grad/activation stats | 优化是否稳定 | 不直接证明任务能力 |
| throughput / MFU | 系统是否更高效 | 不证明质量 |
| closed-loop success | 任务是否真的更好 | 成本高，受场景分布影响 |

一个训练方法声称“更稳、更快、更好”时，最好把质量、稳定性和成本放到同一张表里。只报 loss，容易忽略任务；只报吞吐，容易忽略质量；只报 demo，容易忽略失败率。

## 和后续专题的关系

- [预训练目标与表示学习](pretraining-objectives-and-representation-learning.md)：不同 loss 到底在逼模型学什么。
- [泛化、正则化与分布偏移](generalization-regularization-and-distribution-shift.md)：为什么训练集下降不等于可泛化。
- [训练生产线入门](../training/beginner-model-training-primer.md)：把本页闭环扩展成数据、训练、评测和发布流程。
- [训练稳定性](../training/stability-numerics-and-failure-triage.md)：排查 loss spike、NaN、梯度异常。
- [低比特训练与数值](../training/low-bit-training-and-numerics.md)：混合精度、FP8 和数值保护。

## 外部精读

- [Visualizing the Loss Landscape of Neural Nets](https://arxiv.org/abs/1712.09913)：理解 loss landscape 和结构/归一化对优化地形的影响。
- [On the difficulty of training recurrent neural networks](https://proceedings.mlr.press/v28/pascanu13.html)：梯度爆炸、梯度裁剪和长序列训练问题的经典论文。
- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)：AdamW 的来源。
- [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/)：Karpathy 写的训练排查经典博客，适合学习工程化训练心法。
- [Hugging Face Performance and Scalability docs](https://huggingface.co/docs/transformers/performance)：训练/推理性能、混合精度和显存优化的实用文档。

## 本页结论

训练的核心不是“让 loss 变小”，而是让一个代理目标通过梯度和优化器稳定地改善真实任务。Loss 定义方向，gradient 传递错误，optimizer 控制步伐，scheduler 和数值策略保护训练过程，eval 和分桶指标防止你被平均 loss 欺骗。
