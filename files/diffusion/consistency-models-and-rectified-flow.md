# Consistency Models 与 Flow：终点映射和局部速度的边界

Consistency Models、LCM、Flow Matching 和 Rectified Flow 都能服务少步生成，但它们学习的对象不同。一致性路线学习轨迹上不同时间点到同一终点的映射；Flow 路线学习当前位置的局部速度，再由 ODE solver 积分出终点。两者可以共享教师轨迹或相同端点分布，不能因此视为同一个训练目标。

读一致性路线时，先分清三个问题：一致性函数的约束是什么；LCM 如何把一致性蒸馏接入 latent diffusion；它与 Flow Matching、Rectified Flow 和普通 sampler 加速分别处在哪一层。

Flow 的条件速度、边缘速度、ODE 与 reflow 推导见 [Flow Matching 与 Rectified Flow](flow-matching-and-rectified-flow.md)。

## 前置知识

- [扩散模型基础](foundations-and-core-intuition.md)：模型、路径和 sampler 的分工。
- [采样与推理加速](inference.md)：多步轨迹、NFE 与数值误差。
- [Flow Matching 与 Rectified Flow](flow-matching-and-rectified-flow.md)：局部速度和 ODE 生成接口。

## 核心机制

### 一致性函数学习同轨迹的共同终点

设 \(x_t\) 与 \(x_s\) 位于同一条 probability flow ODE 或教师轨迹上。一致性函数 \(f_\theta\) 希望满足：

\[
f_\theta(x_t,t)\approx f_\theta(x_s,s).
\]

它不是要求两个带噪状态相同，而是要求它们映射到同一个数据端结果。边界条件提供锚点：

\[
f_\theta(x_0,0)=x_0.
\]

没有边界锚点，“所有时间输出一致”并不能保证输出落在正确数据样本上。实际参数化常把 skip connection 和可学习分支组合起来，使边界条件由结构保证，而不只依靠损失学出。

### 训练来源：独立训练与教师蒸馏

Consistency Models 可以采用两类训练方式：

- consistency training 从数据与噪声构造相邻时间点，并通过数值近似形成一致性目标；
- consistency distillation 使用已训练的 diffusion/score 教师，在教师 ODE 轨迹上产生配对状态。

一个抽象的自蒸馏损失是：

\[
\mathcal L_{\mathrm{cons}}
=
\mathbb E\left[
d\!\left(
f_\theta(x_{t_{n+1}},t_{n+1}),
\operatorname{sg}\!\left(f_{\theta^-}(\hat x_{t_n},t_n)\right)
\right)
\right],
\]

其中 \(\hat x_{t_n}\) 由教师或数值一步从 \(x_{t_{n+1}}\) 得到，\(\theta^-\) 常表示 EMA target network，\(\operatorname{sg}\) 表示 stop-gradient。距离函数、时间离散和参数化会影响稳定性，不能把上式当成所有实现逐项相同的唯一配方。

### 一步生成与少步生成

从高噪声状态直接调用一次 \(f_\theta(x_T,T)\) 可以得到一步样本；也可以在若干下降的时间点之间交替加入受控噪声并调用一致性函数，形成少步生成。少步模式让模型获得中途纠错机会，通常在延迟与质量之间更容易取得平衡。

“支持一步”不等于“一步与教师完全等价”。跨度增大后，语义、局部结构、文字、条件控制和分布覆盖都可能退化。验收必须固定真实步数、条件强度和端到端延迟，而不是只比较 sampler 循环次数。

### LCM：latent diffusion 上的一致性蒸馏

Latent Consistency Models 把一致性蒸馏应用到 latent diffusion：使用预训练 latent diffusion 模型形成教师 ODE 轨迹，让学生学习少步一致映射。外层系统仍包含文本编码器、latent 表示和 VAE decoder；改变的是 latent 生成主干的训练与采样接口。

LCM-LoRA 进一步把蒸馏得到的加速能力表示为 LoRA 参数，使它能接入一部分已有模型生态。兼容性仍依赖基础权重、prediction type、scheduler、LoRA 合并方式和条件模块，不能把任意 LCM-LoRA 当作通用的四步开关。

### 与普通 solver 加速的区别

| 路线 | 是否重训 | 学习对象 | 推理方式 |
| --- | --- | --- | --- |
| DDIM / DPM-Solver | 通常不重训 | 沿用原模型输出 | 用更合适的离散路径或高阶 solver |
| Progressive distillation | 需要学生训练 | 模仿教师多个小步的结果 | 逐轮减少采样步数 |
| Consistency / LCM | 需要训练或蒸馏 | 同轨迹各时间点的共同终点 | 一步或少步一致性调用 |
| Flow Matching | 训练速度场 | 概率路径的局部速度 | ODE solver 积分 |
| Rectified Flow | 训练并可 reflow | 更易有限步积分的速度场 | 少步 ODE 积分 |

solver 加速主要重排数值误差；Consistency 改变模型的监督对象；Flow Matching 改变路径速度的训练接口。只看最终 NFE 会掩盖这些差异。

### 与 Flow 路线可以怎样组合

Flow 模型给出局部速度，Consistency 模型给出跨时间终点映射。二者可以组合：例如以 Flow ODE 生成教师轨迹，再蒸馏一个一致性或其他少步学生。此时“教师轨迹来自 Flow”不代表学生仍在预测 Flow velocity。

Rectified Flow 通过路径与 coupling 设计降低有限步积分难度；Consistency 通过函数约束减少抵达终点所需的调用次数。前者关注轨迹几何，后者关注跨时间映射。它们都可能降低步数，也都不能单独保证少步条件保持与分布覆盖。

### 评测应对准不同失效模式

一致性与 LCM 需要重点检查：

- 一步、二步、四步下质量是否单调且稳定；
- 长 prompt、文字、身份和局部控制是否被压平；
- CFG 或 guidance scale 改变时是否仍处于训练支持范围；
- 多随机种子的覆盖率是否低于教师；
- 实际端到端延迟是否被文本编码、VAE 解码或额外条件模块主导。

Flow 与 Rectified Flow 还要检查时间 bucket 误差、路径曲率、solver 与时间网格，以及同一 NFE 下的积分误差。两类方法都不能只用少数视觉样例下结论。

## 常见误解与失效边界

**把 consistency 当成输出恒等。**一致性约束只在同一生成轨迹上成立，并依赖正确的边界条件；不同初始噪声不应被压成同一结果。

**把 LCM 当成换 scheduler。**LCM 权重经过一致性蒸馏，必须与对应 prediction type 和时间配置一起使用。

**认为一步模型没有轨迹。**它在一次函数调用中近似了跨时间映射；训练目标仍依赖噪声时间和教师或自生成轨迹。

**把 Flow velocity 与一致性输出互换。**局部速度需要积分，终点映射直接预测数据端表示。张量 shape 可能相同，语义和 sampler 接口不同。

**只比较模型前向次数。**不同方法的单次前向成本、CFG 分支、分辨率、VAE 和后处理可能不同；NFE 不是端到端延迟。

## 检查理解

1. 一致性约束为什么还需要边界条件？
2. consistency distillation 中教师轨迹承担什么作用？
3. LCM 与只给原 DDPM 换一个四步 scheduler 有什么根本差别？
4. Flow ODE 作为教师时，学生为什么不一定预测 velocity？
5. 一步模型应怎样检查质量与分布覆盖，而不只看单张结果？

## 外部材料

- 原始来源：[Consistency Models](https://arxiv.org/abs/2303.01469)、[Latent Consistency Models](https://arxiv.org/abs/2310.04378)、[Flow Matching](https://arxiv.org/abs/2210.02747)、[Rectified Flow](https://arxiv.org/abs/2209.03003)。
