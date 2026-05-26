# 双目匹配与 Cost Volume 入门

双目匹配（stereo matching）解决的是一个很具体的问题：给定同一时刻、已经校正的左右两张图，估计每个像素在左右图中的水平位移，也就是 disparity，再把 disparity 转成深度。

<!-- [补充] -->
| 读者 | 建议读法 | 工程落点 |
| --- | --- | --- |
| 初学者 | 先抓“双目匹配与 Cost Volume 入门”要解决的问题、关键术语和最小例子；公式只看变量方向。 | 能说清输入、输出和常见失败信号。 |
| 工程师 | 直接对照结构、指标、成本和边界条件，必要时回到前置页补概念。 | 落到数据、算力、延迟、显存、评测和回滚检查。 |


!!! note "合并提示"
    **note · 读法定位**

    这页先回答“双目匹配与 cost volume”在「具身智能」里的位置：它把相机几何里的 baseline / disparity，接到深度网络里的 feature、cost volume、filtering 和 refinement。  
    **前置**：先看 [相机、深度与机器人视觉](cameras-depth-and-robot-vision.md)，知道双目深度公式、内参、外参和标定。视觉网络不熟时回 [卷积与视觉特征](../foundations/convolution-and-feature-extraction.md)。  
    **主线关系**：读 [Fast-FoundationStereo](../paper-deep-dives/inference/fast-foundation-stereo.md) 时，本页负责解释 stereo matching pipeline、cost volume 和评测指标。

    **note · 初学者先抓住**

    双目匹配不是“直接预测一张深度图”这么简单。它先在左右图之间为每个像素搜索可能的匹配位移，把这些候选位移堆成 cost volume，再用网络判断哪一个 disparity 最可信。现代 stereo 网络的大部分计算，正是花在 cost volume 的构造、过滤和细化上。

## 一句话流程

典型 stereo matching pipeline 可以写成：

```text
left / right rectified images
  -> feature extraction
  -> disparity search and cost volume
  -> cost filtering / regularization
  -> initial disparity
  -> refinement and upsampling
  -> dense disparity / depth
```

这条线里有两个世界同时存在：

| 层次 | 关心什么 | Fast-FoundationStereo 对应位置 |
| --- | --- | --- |
| 相机几何 | 左右相机、baseline、rectification、disparity 到 depth | 输入假设和 zero-shot stereo 任务 |
| 深度网络 | feature、cost volume、3D conv / Transformer、refinement | backbone、cost filtering、ConvGRU refinement |
| 系统部署 | latency、MACs、TensorRT、真实场景泛化 | 实时 foundation stereo model |

## 符号卡

<!-- [补充] -->
| 符号 / 术语 | 含义 | 取值范围 |
| --- | --- | --- |
| \(I_L, I_R\) | 左图和右图 | 图像张量 |
| \((u,v)\) | 图像像素坐标，\(u\) 是横向位置，\(v\) 是纵向位置 | 图像宽高范围内的整数或亚像素坐标 |
| \(d\) | disparity，左右图同一点的横向像素位移 | 非负像素位移 |
| \(D\) | 最大搜索 disparity，也就是候选位移范围 | 正整数或预设搜索上界 |
| \(f\) | 相机焦距，通常来自内参 | 正实数 |
| \(B\) | 左右相机 baseline | 正实数，单位随标定数据 |
| \(z\) | 深度，常用 \(z = fB / d\) 近似理解 | 正实数，\(d>0\) 时有效 |
| `rectification` | 把左右图校正到同一水平扫描线，方便只沿横向搜索 | 几何预处理步骤 |
| `cost volume` | 每个像素、每个 disparity 候选的匹配代价或匹配特征 | 张量，形状由分辨率和 \(D\) 决定 |
| `refinement` | 在初始 disparity 基础上迭代修正边界、遮挡和细节 | 后处理或网络模块 |

## Rectified Stereo Pair 是什么

真实双目相机的左右镜头不可能天然完美平行，镜头也有畸变。标定和校正的目标，是把左右图变成 rectified stereo pair：同一个 3D 点在左右图中落在同一条水平扫描线上。

校正前，匹配点可能沿任意 epipolar line 出现；校正后，匹配点通常只需要沿同一行横向搜索：

```text
left:   pixel at (u, v)
right:  candidate pixels at (u - d, v), d = 0 ... D
```

这一步很关键，因为它把二维匹配问题压成一维搜索。没有 rectification，网络不仅要判断“像不像”，还要处理复杂的几何搜索空间；有了 rectification，cost volume 可以沿 disparity 维度组织。

!!! warning "输入假设不能忽略"
    很多 stereo 模型默认输入已经 rectified / undistorted，左右图顺序也不能交换。如果相机标定漂移、左右不同步、畸变没处理好，模型输出坏掉不一定是网络能力问题，而是输入几何假设被破坏。

## Disparity Search 怎么看

对左图像素 \((u,v)\)，双目匹配会在右图同一行搜索 \((u-d,v)\)。如果某个 \(d\) 让左右局部特征最像，这个 \(d\) 就是该像素的 disparity 候选。

直觉上：

| 现象 | 几何含义 |
| --- | --- |
| disparity 大 | 物体通常更近 |
| disparity 小 | 物体通常更远 |
| disparity 接近 0 | 可能是远处、天空，或没有可靠匹配 |
| disparity 不连续 | 可能是物体边界、遮挡或深度突变 |

深度公式常写成：

\[
z = \frac{fB}{d}
\]

<!-- [补充] -->
| 符号 | 含义 | 取值范围 |
| --- | --- | --- |
| \(z\) | 样本、状态或 latent 变量 | 张量空间，形状由数据表示决定 |
| \(d\) | 公式中的变量或中间量 | 按上下文定义 |


它说明一个残酷事实：远处 \(d\) 很小，1 个像素的 disparity 误差会放大成很大的深度误差。所以 stereo benchmark 常常同时看近处边界、远处区域、遮挡区域和低纹理区域。

## Cost Volume 是什么

cost volume 可以理解为一张“像素到候选 disparity 的评分表”。对每个左图位置 \((u,v)\)，模型比较左图特征 \(F_L(u,v)\) 和右图不同位移的特征 \(F_R(u-d,v)\)，得到每个 \(d\) 的匹配信息。

常见形状可以粗写成：

\[
\mathbf{V} \in \mathbb{R}^{C \times D \times H \times W}
\]

<!-- [补充] -->
| 符号 | 含义 | 取值范围 |
| --- | --- | --- |
| \(C\) | 公式中的变量或中间量 | 按上下文定义 |
| \(D\) | 公式中的变量或中间量 | 按上下文定义 |
| \(H\) | 公式中的变量或中间量 | 按上下文定义 |
| \(W\) | 公式中的变量或中间量 | 按上下文定义 |


或在下采样后写成：

\[
\mathbf{V} \in \mathbb{R}^{C \times \frac{D}{s} \times \frac{H}{s} \times \frac{W}{s}}
\]

<!-- [补充] -->
| 符号 | 含义 | 取值范围 |
| --- | --- | --- |
| \(C\) | 公式中的变量或中间量 | 按上下文定义 |
| \(s\) | 状态、序列长度或分数 | 状态空间/正整数/实数标量，依上下文而定 |


这里 \(C\) 是 cost volume 的通道数，\(D\) 是 disparity 候选数，\(H,W\) 是空间分辨率，\(s\) 是下采样倍率。

cost volume 的构造方式有多种：

| 构造方式 | 直觉 | 取舍 |
| --- | --- | --- |
| concatenation | 把左右特征拼起来，让后续网络学匹配 | 表达力强，计算较重 |
| correlation | 直接算左右特征相似度 | 更轻，但信息更压缩 |
| group-wise correlation | 分组算相似度 | 在表达力和成本之间折中 |
| feature difference / absolute difference | 看左右特征差异 | 简单，常和其他方式组合 |

!!! note "为什么 cost volume 贵"
    它多了一维 disparity。图像是 \(H \times W\)，cost volume 变成 \(D \times H \times W\)。如果还保留通道 \(C\)，后续 3D convolution 或 attention 的计算和显存都会明显上升。

## Cost Filtering 在做什么

原始 cost volume 很嘈杂。纹理重复、弱纹理、反光、透明、遮挡和光照差异都会让局部匹配出错。cost filtering / cost regularization 的作用，是利用空间上下文和几何一致性修正这些局部错误。

常见模块包括：

| 模块 | 作用 |
| --- | --- |
| 3D convolution | 在 disparity、height、width 三个维度上传播上下文 |
| 3D hourglass | 下采样聚合大范围上下文，再上采样恢复分辨率 |
| axial / planar convolution | 用更结构化的卷积扩大感受野，减少计算 |
| Disparity Transformer | 沿 disparity 或空间维度建模长程关系 |
| feature-guided volume excitation | 用左图 unary features 引导 cost volume 哪些区域更重要 |

Fast-FoundationStereo 里 cost filtering 是三大瓶颈之一。论文选择 blockwise NAS，是因为 cost volume 的通道本来就不大，直接剪枝未必能拿到明显硬件收益；替换成更合适的 block 反而更有效。

## Refinement 为什么还需要

cost filtering 通常先给出一个 initial disparity，但这张图还可能有几个问题：

| 问题 | 常见表现 |
| --- | --- |
| 边界被抹平 | 物体边缘的 disparity 过渡太宽 |
| 遮挡区域不稳定 | 左图看到、右图看不到的位置匹配混乱 |
| 细小结构丢失 | 线缆、把手、薄边缘不清楚 |
| 上采样误差 | 低分辨率 disparity 放大后出现块状或错位 |

refinement module 会把初始 disparity、图像特征和 hidden state 结合起来，迭代修正输出。很多现代 stereo 网络使用 ConvGRU 或类似 recurrent update block：每一轮都基于上一轮 disparity 做小步更新。

这也解释了 Fast-FoundationStereo 为什么专门剪 refinement GRU。迭代模块会重复调用，哪怕单轮不算最大瓶颈，累积 latency 也很可观；但 recurrent hidden state 有通道依赖，所以不能随便乱剪。

## Benchmark 和指标怎么读

Stereo 论文常见数据集：

| Benchmark | 关注点 |
| --- | --- |
| Middlebury | 高分辨率、室内、细节和遮挡，常用于看精细 stereo 质量 |
| ETH3D | 真实多视角 / 双目场景，纹理和几何更接近真实采集 |
| KITTI 2012 / 2015 | 自动驾驶场景，路面、车辆、远距离和动态物体更多 |

常见指标：

| 指标 | 含义 | 怎么读 |
| --- | --- | --- |
| EPE | End-Point Error，平均 disparity 误差 | 越低越好，但平均值可能掩盖局部坏点 |
| BP-1 / BP-2 | bad pixel，误差超过 1 或 2 像素的比例 | 越低越好，阈值越小越严格 |
| D1 | KITTI 常用 bad disparity 指标 | 越低越好，常用于自动驾驶 stereo |

读 zero-shot stereo 结果时，重点不是“哪个数据集最高”，而是模型有没有在未微调的真实场景中同时守住多类分布：室内、户外、远距离、弱纹理、反光/透明和遮挡。

## 和 Fast-FoundationStereo 怎么接

读 Fast-FoundationStereo 时，可以把术语按这张表对应：

| 论文术语 | 本页对应概念 |
| --- | --- |
| feature backbone | 左右图特征提取器，为 cost volume 提供 \(F_L,F_R\) |
| cost volume | 每个像素、每个 disparity 候选的匹配表示 |
| cost filtering | 对 cost volume 做上下文推理，输出 initial disparity |
| 3D hourglass / APC | cost filtering 的结构选择 |
| Disparity Transformer | cost filtering 里的长程 disparity 推理模块 |
| refinement ConvGRU | 迭代修正 initial disparity 的 recurrent module |
| pseudo-label stereo pairs | 用 teacher 给真实双目图生成 dense disparity 监督 |
| BP-1 / BP-2 / D1 | 判断 zero-shot stereo 质量的坏点比例指标 |

!!! tip "下一站"
    - 几何基础不稳，回看 [相机、深度与机器人视觉](cameras-depth-and-robot-vision.md)。
    - 视觉 feature 不稳，回看 [卷积与视觉特征](../foundations/convolution-and-feature-extraction.md)。
    - 压缩方法不稳，继续看 [模型压缩、剪枝与 NAS 入门](../foundations/model-compression-pruning-and-nas-basics.md)。
    - 论文主线，进入 [Fast-FoundationStereo](../paper-deep-dives/inference/fast-foundation-stereo.md)。
