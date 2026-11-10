# 一致性模型怎样把多步轨迹压成少数几次调用

把一个扩散模型的采样步数从 50 改成 4，程序通常还能运行，结果却可能只剩下轮廓。这个失败很容易被误判成“采样器没调好”。真正的问题往往更早发生了：原模型学的是某个时间点的局部去噪方向，少步采样却要求它跨过一大段轨迹。**一致性模型重新训练一个函数，让同一条轨迹上的不同状态直接指向同一个低噪声端点。**

这句话里的“直接”有严格边界。它不等于把普通 DDPM 的 `scheduler.step` 调用次数改小，也不等于把网络输出的噪声乘一个更大的步长。下面先用一个能手算的轨迹把三种输出分开，再看一致性蒸馏、无教师训练、一步与多步采样各自需要什么条件。

## 先辨认网络输出的语义

下面三个函数都可以返回和输入 `x` 同样的张量形状，却不能互换。

| 函数 | 输入 | 输出代表什么 | 调用者怎样使用 |
| --- | --- | --- | --- |
| ε 模型 | \(x_t,t\) | 当前状态中估计的噪声 | 代入 DDPM、DDIM 或 ODE 求解器 |
| 速度模型 | \(x_t,t\) | 轨迹在当前时间的局部速度 | 乘以步长并积分 |
| 一致性函数 | \(x_t,t\) | 同一条轨迹在最小噪声端的估计 | 直接作为端点，或重新加噪后再调用 |

例如考虑一条一维 probability-flow ODE

\[
\frac{\mathrm dx}{\mathrm dt}=\frac{x}{2(1+t)},
\qquad p_t=\mathcal N(0,1+t).
\]

从 \(t=3,x=2\) 沿逆时间走到 \(t=0\)，终点是

\[
x(0)=\frac{2}{\sqrt{1+3}}=1.
\]

此时在 \(t=3\) 的局部速度是 \(2/[2(1+3)]=0.25\)，端点函数的输出却是 1。把 1 当成速度，从 3 一步走到 0 会得到 \(2-3\times1=-1\)；把 0.25 当成端点又会得到 0.25。两个错误的张量形状都对，只有变量的含义错了。

同一条轨迹经过 \(t=1\) 时有 \(x=\sqrt2\)，因此端点映射

\[
f(x,t)=\frac{x}{\sqrt{1+t}}
\]

满足

\[
f(2,3)=f(\sqrt2,1)=f(1,0)=1.
\]

这就是一致性的核心。输入状态不同，映射后的端点相同。另一条从 \(-2\) 出发的轨迹应映到 \(-1\)，所以“输出都一样”不是目标。

```mermaid
flowchart LR
    XT[高噪声状态 x_T] --> SOLVER[教师求解器 / 共享噪声轨迹]
    SOLVER --> XN[中间状态 x_tn]
    SOLVER --> XN1[相邻状态 x_tn+1]
    XN --> F1[学生端点 f_theta]
    XN1 --> F2[EMA 目标端点 f_theta-]
    F1 --> LOSS[同一轨迹端点距离]
    F2 --> LOSS
    B[低噪声边界 f(x,epsilon) 约等于 x] --> LOSS
```

图中的两条状态来自同一条轨迹，损失才有“应该指向同一端点”的含义。若两个时间点各自重新采样独立噪声，连接关系就断了；若删掉边界项，恒零函数又会成为便宜的解。

## 一致性条件为什么还需要边界

直接最小化同轨迹状态之间的差异，会出现恒零解

\[
f_\theta(x,t)\equiv0.
\]

它对任意两个时间都满足一致性，却没有生成能力。需要给低噪声端一个边界约束。原始 Consistency Models 使用最小噪声 ε 而不是把数值上的 \(t=0\) 当作可训练端点，要求

\[
f_\theta(x,\epsilon)\approx x.
\]

常见的预条件化写成

\[
f_\theta(x,t)=c_{\mathrm{skip}}(t)x+c_{\mathrm{out}}(t)F_\theta(x,t),
\]

并让 \(c_{\mathrm{skip}}(\epsilon)=1\)、\(c_{\mathrm{out}}(\epsilon)=0\)。这样网络在边界处天然接近恒等映射，参数学习集中在离开边界后的修正。不同实现对输入缩放、噪声坐标和系数的具体定义不同，不能只看这个抽象形式就复现一个 checkpoint。

边界条件解决的是坏解，不能替模型补足数据覆盖。若高噪声端没有训练样本，或条件编码没有进入网络，函数仍可能在训练时间点之间输出看似平滑但语义错误的结果。

## 一致性蒸馏怎样从教师得到监督

设噪声层级为

\[
\epsilon=t_0<t_1<\cdots<t_N=T.
\]

教师模型可以预测局部方向，数值求解器把 \(x_{t_{n+1}}\) 走一步得到较低噪声状态

\[
\hat x_{t_n}=\Psi_{\text{teacher}}(x_{t_{n+1}},t_{n+1},t_n).
\]

学生网络和目标网络分别估计两个状态的端点，损失可写成

\[
\mathcal L_{\mathrm{CD}}
=\mathbb E\left[
\lambda(t_n)d\left(
 f_\theta(x_{t_{n+1}},t_{n+1}),
 \operatorname{sg}\left(f_{\theta^-}(\hat x_{t_n},t_n)\right)
 \right)\right].
\]

这里的 `sg` 是 stop-gradient，\(\theta^-\) 通常由学生参数的 EMA 更新。目标分支不参与当前梯度，避免两个输出一起移动而把损失降到没有意义的位置。

教师一步的误差、求解器离散误差、学生网络误差和目标网络滞后会叠加在一起。可以用三个对照拆开它们：

1. 在解析的一维轨迹上用精确端点，检查损失实现本身
2. 换成一阶 Euler 教师，观察步长减半后误差是否下降
3. 固定教师与学生，只替换 EMA 更新频率，检查目标滞后是否造成振荡

如果第一项都不能收敛，问题在时间索引、边界或距离函数；如果只有第二项随步长变化，先修教师求解器，调 guidance 没有帮助。

## 无教师训练并不是“随机取两个时间点”

Consistency Training 不依赖预训练 score 教师，而是利用同一干净样本和同一份噪声构造相邻状态。以加性噪声坐标为例

\[
x_{t_n}=x_0+t_nz,\qquad
x_{t_{n+1}}=x_0+t_{n+1}z,\qquad z\sim\mathcal N(0,I).
\]

两个状态共享 \(z\)，差值为

\[
x_{t_{n+1}}-x_{t_n}=(t_{n+1}-t_n)z.
\]

若误用两份独立噪声，时间差中会混入一个与步长无关的随机项，学生看到的就不再是稳定的近邻关系。这个细节在代码中只差一次 `randn_like` 的位置，却会让训练目标完全变样。

这里的 \(t\) 是噪声标准差坐标，和前面 probability-flow 例子的时间并非同一个量。原论文还规定了时间离散、权重和边界处理。用共享噪声能说明 CT 的配对直觉，却不能据此宣称任意噪声表都等价于论文训练。

## 一步采样和多步采样各自做了什么

一步采样从高噪声状态调用一次一致性函数

\[
\hat x_\epsilon=f_\theta(x_T,T).
\]

多步采样会在两次函数调用之间把端点估计重新送回某个中间噪声层。对加性噪声路径，一个教学形式是

\[
\hat x_\epsilon=f_\theta(x_t,t),
\qquad
x_s=\hat x_\epsilon+\sqrt{s^2-\epsilon^2}\,z,
\quad s<t.
\]

实际实现还会处理预条件化、截断噪声和条件输入。伪代码中的关键是“先预测端点，再按下一个噪声层重新加噪”，而不是把网络输出当作速度直接积分。

```python
# 仅展示数据流，sigmas 按从高到低排列，预条件化需与训练实现一致
x = torch.randn(batch, channels, height, width, device=device) * sigmas[0]
for sigma, next_sigma in zip(sigmas[:-1], sigmas[1:]):
    x_end = consistency_model(x, sigma, condition)
    if next_sigma > sigma_min:
        noise = torch.randn_like(x_end)
        x = x_end + torch.sqrt(next_sigma**2 - sigma_min**2) * noise
    else:
        x = x_end
```

这段代码不是可直接替换 Diffusers scheduler 的实现。不同模型约定的输入是离散 timestep、连续 sigma 还是经过变换的时间；重加噪系数也必须和训练路径一致。官方 OpenAI 实现的 `launch.sh` 为多步采样显式指定时间点，Diffusers 的 `ConsistencyModelPipeline` 也把 `timesteps` 作为模型接口的一部分。照抄一个时间表到另一个 checkpoint，往往比少调用几次更快失败。

## 它和 Flow Matching、Rectified Flow 的边界

| 方法 | 学习对象 | 少步推理的来源 | 常见误解 |
| --- | --- | --- | --- |
| Flow Matching | 条件速度场 \(v_\theta(x_t,t)\) | 训练出容易积分的路径，推理仍需数值步 | 把速度输出当成最终样本 |
| Rectified Flow | 重新配对后的速度场 | 让路径更接近直线，减少积分误差 | 任何数据配对都自动得到直线 |
| Consistency Distillation | 轨迹端点映射 | 教师轨迹提供跨时间监督 | 只减少 scheduler 步数就等于蒸馏 |
| Consistency Training | 无教师的端点映射 | 共享噪声构造局部一致性约束 | 随机抽两个时间点即可训练 |

Flow Matching 的网络输出是局部速度，Consistency Model 的网络输出是端点。两者可以组成教师和学生，却不能只换一个类名。Rectified Flow 重新定义 coupling 和速度目标，是否真的适合少步积分还要测路径曲率、求解器误差与端到端质量。

## 一个不依赖图像质量的最小排查

先不要用 FID 或肉眼挑图。用前面的解析轨迹做一个一维回放，至少检查四件事：

```python
import numpy as np

def endpoint(x, t):
    return x / np.sqrt(1.0 + t)

states = [(2.0, 3.0), (np.sqrt(2.0), 1.0), (1.0, 0.0)]
values = np.array([endpoint(x, t) for x, t in states])
print(values)                         # [1. 1. 1.]
print(np.max(np.abs(values - 1.0)))   # 接近 0

velocity = 2.0 / (2.0 * (1.0 + 3.0))
wrong = 2.0 - 3.0 * endpoint(2.0, 3.0)
print(velocity, wrong)                 # 0.25, -1.0
```

然后把 `endpoint` 换成待测网络，记录以下字段：

| 字段 | 目的 |
| --- | --- |
| `sigma_grid` | 确认训练和推理使用同一噪声坐标 |
| `boundary_error` | 检查 \(f(x,\epsilon)\approx x\) |
| `same_path_error` | 比较同一轨迹两个时间的端点输出 |
| `teacher_step` | 记录教师求解器步长和方向 |
| `seed` 与 `condition_id` | 让失败样本可回放 |
| `decode_ms`、`model_calls` | 区分少调用带来的真实端到端收益 |

一个能说明“端点映射学到了”的最小结果，应同时给出边界误差、同轨迹误差和不同轨迹的分离度。只有一张清晰的样本图，无法区分网络真的收敛、VAE 解码掩盖了误差，还是随机种子碰巧有利。

## 部署时最容易漏掉的成本

一致性模型减少的是去噪网络调用次数。文本编码器、控制模块、VAE 解码、GPU kernel 启动和数据传输仍然存在。若原系统 20 次 UNet 调用占总延迟的 70%，把调用改成 4 次的理想上限约为

\[
\frac{1}{0.3+0.7\times4/20}\approx2.5\text{ 倍}.
\]

实际收益还会受 batch、分辨率、编译缓存和 CPU 调度影响。报告速度时至少同时记录模型调用次数、端到端延迟、显存峰值和质量指标；只报告“每张图 4 步”无法说明系统是否更快。

官方 Consistency Models 仓库提供 CD、CT、单步和多步脚本，并在 ImageNet-64、LSUN 等设置中报告 FID、Precision、Recall 和 Inception Score。那些数字属于论文的数据集、模型和评测协议，不能直接当作本站机器上的结果。本站的解析轨迹和 smoke 检查只验证数据流与边界，不声称训练出了同等质量的图像。

## 这页参考了什么

- [Consistency Models](https://arxiv.org/abs/2303.01469)给出一致性函数、边界参数化、Consistency Distillation 和 Consistency Training
- [Consistency Models 官方实现](https://github.com/openai/consistency_models/tree/e32b69ee436d518377db86fb2127a3972d0d8716)提供训练、单步/多步采样和评测脚本
- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)用于区分速度场与端点映射
- [Rectified Flow](https://arxiv.org/abs/2209.03003)用于说明重新配对路径与少步积分的关系
- [Diffusers ConsistencyModelPipeline](https://huggingface.co/docs/diffusers/main/en/api/pipelines/consistency_models)用于核对当前推理接口和 `timesteps` 约定

公式的符号、解析轨迹、数值例子、排查字段和伪代码是本站重新组织的教学材料。官方仓库中的代码、权重和论文图没有复制进本站。真正接入某个 checkpoint 时，应以它的模型卡、噪声坐标和推理脚本为准。

读完后可以回到[扩散模型训练](training.md)核对噪声目标，再看[扩散蒸馏](distillation.md)比较轨迹回归与分布匹配。若你关心部署延迟，继续看[采样器与推理](inference.md)，把模型调用次数和 VAE、调度、传输成本一起记入实验表。
