# 卷积、频域与多尺度表示

这篇回答一个问题：卷积、kernel、filter、Fourier/DCT、频率、SNR、timestep 和多尺度金字塔为什么会同时出现在视觉、扩散、动作 chunk 和世界模型论文里。

很多 AI 论文表面在讲不同对象：图像局部纹理、视频运动、机器人动作序列、扩散噪声阶段。数学上它们常在处理同一个问题：信号里哪些变化是局部的，哪些变化是全局的；哪些变化是低频趋势，哪些变化是高频细节；模型应该在哪个尺度上做判断。

## 卷积：用同一个局部模板扫过信号

一维离散卷积可以写成：

\[
(x*k)_t
=
\sum_{i=-m}^{m}x_{t-i}k_i
\]

二维图像卷积可以写成：

\[
(X*K)_{u,v}
=
\sum_{i,j}X_{u-i,v-j}K_{i,j}
\]

这里 \(K\) 是 kernel 或 filter。直觉上，卷积在每个位置问同一个问题：附近像不像某种局部模式。边缘、角点、纹理、深度局部结构、动作短期平滑，都可以用这种局部模板直觉理解。

卷积的关键性质是参数共享。同一个 kernel 在不同位置重复使用，所以它天然适合处理“同一种局部模式可能出现在任何位置”的信号。

## Filter：保留一些变化，压掉另一些变化

filter 不是只能指卷积层。更抽象地说，filter 是一个选择器：保留某些成分，压低另一些成分。

| filter 类型 | 保留什么 | 常见场景 |
| --- | --- | --- |
| 低通 filter | 平滑、慢变化、整体趋势 | 动作 chunk、视频运动趋势、去噪 |
| 高通 filter | 边缘、突变、细节 | 图像边缘、接触瞬间、异常检测 |
| 带通 filter | 某个频率范围 | 音频、周期运动、振动信号 |

读论文里的 filtering、smoothing、temporal aggregation 时，可以先问：它在保留哪种变化，牺牲哪种变化。

## Fourier 和 DCT：把信号换到频率坐标

离散 Fourier 变换可以抽象成：

\[
X_k
=
\sum_{n=0}^{N-1}
x_n
\exp\left(-2\pi i\frac{kn}{N}\right)
\]

它把时间或空间上的信号 \(x_n\)，表示成不同频率成分 \(X_k\)。低频表示慢变化，高频表示快变化或细节。

DCT 可以理解为只使用 cosine 基的一类频域表示。动作 chunk 压缩里用 DCT 的直觉是：一段连续动作通常有平滑趋势，不需要逐时刻自回归生成每个动作维度；先保留少数低频系数，就能用更短的 token 序列表达主要运动。

这也是为什么 FAST 一类动作 tokenization 方法会谈 DCT，视频模型会谈时空频率，扩散和去噪会谈不同噪声水平负责不同尺度。

## 多尺度：同一个对象在不同分辨率下看

图像和视频既有全局布局，也有局部细节。多尺度方法会把信号分成粗到细的层级：

1. 粗尺度看整体结构、长距离关系和大运动。
2. 中尺度看对象、局部布局和阶段变化。
3. 细尺度看纹理、边缘、接触和微小误差。

在视觉模型里，这可能表现为 feature pyramid、patch merging、多分辨率 decoder；在世界模型里，可能表现为 latent dynamics 负责高层未来，pixel decoder 负责可视细节；在机器人里，可能表现为高层 subgoal 和低层控制频率分开。

## SNR 和 timestep：扩散里的尺度坐标

扩散模型里的 timestep 不只是循环编号，它对应噪声强度。常见信噪比可以写成：

\[
\operatorname{SNR}(t)
=
\frac{\alpha_t^2}{\sigma_t^2}
\]

\(\alpha_t\) 表示信号保留强度，\(\sigma_t\) 表示噪声强度。高 SNR 时，样本还有清晰细节；低 SNR 时，样本更接近噪声，模型更需要恢复全局结构。

![Wan SNR phases](../assets/images/paper-deep-dives/diffusion/wan/wan22-moe-snr.png){ width="820" }

<small>图源：`Wan2.2` 相关论文图，本站已本地化于视频扩散专题。原图展示 denoising timestep 与 SNR 阶段，以及不同 expert 在不同噪声区间的验证曲线。本站在这里用它说明：timestep 可以被读成噪声尺度，不同阶段承担的生成职责不同。</small>

Phased DMD 和 Wan2.2 这类论文把去噪过程分阶段，本质上是在承认：不同 SNR 区间不是同一个难度。低 SNR 阶段更像全局结构和运动，高 SNR 阶段更像局部细节和纹理。

## 常见误读

| 误读 | 更稳的理解 |
| --- | --- |
| 卷积只是旧式视觉网络组件 | 卷积代表局部共享模板，很多 dense head 和几何 encoder 仍在用 |
| 频域只是信号处理术语 | DCT、Fourier、低频趋势和高频细节会直接影响动作 token、视频和去噪 |
| timestep 只是采样步数 | 在扩散里它通常对应噪声强度和任务阶段 |
| 分辨率越高越好 | 高分辨率更贵，也可能把模型注意力拉向无关细节 |

## 读完以后怎么判断

看到 convolution、kernel、frequency、DCT、SNR、timestep 或 multi-scale，先问：模型在处理哪个信号，局部模板是什么，低频和高频分别承载什么，当前尺度服务全局结构还是局部细节。这个问题链会支撑 Depth Anything、MapAnything、Fast-WAM、Wan、Phased DMD 和动作 tokenization 论文。

## 相关阅读与下一步

- 站内下一步：[相机、深度与机器人视觉](../embodied-ai/cameras-depth-and-robot-vision.md)。
- 站内下一步：[双目匹配与 Cost Volume](../embodied-ai/stereo-matching-and-cost-volume.md)。
- 论文精读：[Wan 视频生成](../paper-deep-dives/diffusion/wan.md)。
- 论文精读：[Phased DMD](../paper-deep-dives/diffusion/phased-dmd.md)。
