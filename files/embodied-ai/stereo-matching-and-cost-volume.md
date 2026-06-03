# 双目匹配与 Cost Volume：从左右图到可用深度

这篇文章只回答一个问题：双目模型为什么不是“看一张图猜深度”，而是先在左右图之间做匹配，再把匹配证据组织成 cost volume。

双目匹配输入是同一时刻的左右相机图像，输出通常是 dense disparity map。Disparity 是同一个 3D 点在左右图上的横向像素位移；知道相机焦距和 baseline 后，disparity 可以转成深度。它和单目深度最本质的差别是：双目有明确几何约束，模型不是纯粹从纹理和语义中猜距离，而是在两张图之间寻找对应关系。

## 双目先靠几何缩小搜索空间

两台相机看到同一个 3D 点 \(P\)。左图里它投影到 \(x\)，右图里投影到 \(x'\)。如果相机已经标定并校正，左右图上的对应点会落在同一条水平扫描线上，匹配问题就从二维搜索变成一维搜索：

```text
left pixel:  (u, v)
right candidates: (u - d, v), d = 0 ... D
```

这里 \((u,v)\) 是图像坐标，\(d\) 是候选 disparity，\(D\) 是最大搜索范围。Rectification 的价值就在这里：它把“右图里哪里像这个点”限制到同一行。OpenCV 的 stereo 教程也用同样的几何直觉解释 depth map：先找左右图对应点，再由 disparity 推深度。

![Stereo camera geometry](../assets/images/localized-remote/embodied-ai/cameras-depth-and-robot-vision/stereo-camera-model-wikimedia.jpg){ width="640" }

<small>图源：[Wikimedia Commons: Stereo-camera-model.jpg](https://commons.wikimedia.org/wiki/File:Stereo-camera-model.jpg)。原图表达双目相机的焦距、baseline、左右投影点和深度关系。本站使用这张图说明：双目深度不是神经网络凭空生成的，它有明确的三角测量前提。</small>

深度公式常写成：

\[
z=\frac{fB}{d}
\]

这里 \(z\) 是深度，\(f\) 是焦距，\(B\) 是左右相机 baseline，\(d=x-x'\) 是 disparity。这个式子有两个后果。第一，物体越近，左右视差通常越大；物体越远，视差越小。第二，远处的 \(d\) 很小，1 个像素误差会被放大成很大的深度误差。所以双目系统对标定、同步、畸变校正和亚像素估计都很敏感。

## Cost Volume 是“所有候选位移的证据表”

如果只看一个像素，模型不知道哪个 \(d\) 最可信。Cost volume 的做法是：对每个左图像素，沿右图同一行枚举多个 disparity 候选，把每个候选的匹配证据都存下来。

给左右图提取特征：

\[
F_L, F_R \in \mathbb{R}^{C\times H\times W}
\]

这里 \(C\) 是特征通道数，\(H,W\) 是特征图分辨率。对左图位置 \((u,v)\) 和候选位移 \(d\)，比较 \(F_L(u,v)\) 与 \(F_R(u-d,v)\)。如果使用 correlation，可以粗略写成：

\[
V(d,u,v)=
\langle F_L(u,v), F_R(u-d,v)\rangle
\]

这里 \(V(d,u,v)\) 是该像素在候选 disparity \(d\) 下的相似度，\(\langle\cdot,\cdot\rangle\) 是向量点积。分数越高，表示左右特征越像。很多网络不会只存一个相关分数，而是拼接左右特征或分组相关，因此 volume 可能带通道维：

\[
V\in \mathbb{R}^{C_v\times D\times H\times W}
\]

这里 \(C_v\) 是 cost volume 通道数。关键是多出来的 \(D\) 维：图像特征是 \(H\times W\)，cost volume 变成 \(D\times H\times W\)。这解释了为什么 stereo 网络常常比普通语义分割更吃显存，尤其在高分辨率和大 disparity 范围下。

常见构造方式可以这样理解：

| 构造方式 | 它保留什么 | 代价 |
| --- | --- | --- |
| Concatenation | 左右特征都保留，让后续网络学习匹配 | 表达力强，volume 更重 |
| Correlation | 直接给相似度分数 | 更轻，但压缩了信息 |
| Group-wise correlation | 分组算相似度 | 在表达力和成本之间折中 |
| Difference / absolute difference | 显式看左右特征差 | 简单，常与其他方式组合 |

GC-Net 的历史意义在于把 rectified stereo 的几何结构放进深度网络：先构造 cost volume，再用 3D convolution 学上下文。PSMNet 继续强调多尺度上下文和 stacked hourglass 正则化。RAFT-Stereo 则把问题推向 recurrent update：不一次性回归完，而是反复查询 correlation / cost 信息并迭代更新 disparity。

## 原始匹配证据会在这些地方坏掉

Cost volume 只是候选证据，不等于正确深度。双目最常见的失败区很具体：

| 区域 | 为什么难 |
| --- | --- |
| 弱纹理墙面、天空、桌面 | 左右局部 patch 都差不多，没有稳定匹配 |
| 重复纹理 | 多个候选位移都很像，容易选错周期 |
| 反光、透明、玻璃 | 左右视角看到的反射内容不一致 |
| 遮挡边界 | 左图看到的点，右图可能被前景挡住 |
| 细线和薄结构 | 下采样后信息丢失，边界容易被抹平 |
| 未校正或不同步输入 | 几何假设被破坏，网络再强也在错输入上工作 |

这也是为什么传统 stereo 会有 matching cost、cost aggregation、disparity optimization、post-processing 一整套流程；深度学习版本只是把其中很多步骤变成可学习模块。OpenCV 的 StereoBM / StereoSGBM 参数也能看到这些老问题：纹理阈值、speckle filtering、候选 disparity 数、uniqueness ratio，本质上都在处理“不可靠匹配”。

## Cost Filtering 在清理候选证据

Cost filtering / cost regularization 的任务是让一个像素的判断参考周围像素、相邻 disparity 和更大范围的几何上下文。它不是凭空生成深度，而是在问：某个候选 disparity 是否被邻域、边界和整体场景支持。

| 模块 | 直觉 |
| --- | --- |
| 3D convolution | 同时沿 \(D,H,W\) 三个维度传播证据 |
| 3D hourglass | 先看大范围上下文，再回到细粒度候选 |
| Spatial pyramid pooling | 把局部匹配放进多尺度场景上下文 |
| Transformer / long-range reasoning | 让远处结构、重复纹理和大平面互相约束 |
| Cascade / coarse-to-fine volume | 先粗略定位，再缩小候选范围做高分辨率修正 |

GC-Net 和 PSMNet 代表的是“构造 4D cost volume，再用 3D 网络正则化”的路线。Cascade cost volume 一类方法提醒我们，完整高分辨率 volume 太贵，可以先在粗尺度估计，再逐级缩小搜索范围。FoundationStereo 则把 monocular foundation prior 和 long-range context reasoning 接进 stereo，解决的是另一个痛点：传统 stereo 网络在固定 benchmark 上很强，但跨真实场景 zero-shot 泛化不稳。

## Refinement 是把初始视差变成可用视差

Cost filtering 通常会给出 initial disparity，但这个结果还可能边界糊、遮挡乱、细节丢失或上采样错位。Refinement module 会结合图像特征、当前 disparity 和 hidden state，反复做小步修正。

可以把 recurrent refinement 简化成：

\[
d_{t+1}=d_t+\Delta d_t
\]

这里 \(d_t\) 是第 \(t\) 轮的 disparity，\(\Delta d_t\) 是 update block 预测的修正量。RAFT-Stereo、FoundationStereo 和 Fast-FoundationStereo 都可以从这个角度读：网络不是一次性输出最终答案，而是在 cost / correlation 证据上迭代对齐。

Refinement 的价值集中在边界、遮挡和小结构上。它的代价也明显：如果 update block 要跑很多轮，latency 会累积；如果 hidden state 通道多，显存和计算也会增长。这就是 Fast-FoundationStereo 为什么把 refinement ConvGRU 当成单独热路径来剪枝，而不是只压 backbone。

## FoundationStereo / Fast-FoundationStereo 怎么拆这条链

现代 stereo foundation model 不是只把网络做大。FoundationStereo 强在三件事：用大规模合成数据和自筛选提高覆盖，用 monocular vision foundation prior 帮助跨域，用更强 cost filtering 和 refinement 提高匹配鲁棒性。Fast-FoundationStereo 则问另一个问题：这条链如何实时化。

![Fast-FoundationStereo pipeline](../assets/images/paper-deep-dives/inference/fast-foundation-stereo/figure-3-framework-padded.png){ width="920" }

<small>图源：[Fast-FoundationStereo](https://arxiv.org/abs/2512.11130)，Figure 3。原图表达 stereo pipeline 包含 feature backbone、cost filtering 和 disparity refinement，并分别用 distillation、blockwise NAS、structured pruning 加速。本站使用这张图说明：双目模型的核心链条不是一个黑箱，而是“左右图特征 -> cost volume -> cost filtering -> refinement”。</small>

这张图应该从上到下读。最上面是通用 stereo pipeline：左右图进入 feature backbone，构造 cost volume，经过 cost filtering 得到初始 disparity，再 refinement 输出最终 disparity。下面三块对应三类瓶颈：backbone 负责提特征，cost filtering 在 volume 上推理，refinement 做迭代修正。Fast-FoundationStereo 的工程判断是：不同瓶颈冗余形态不同，所以 feature backbone 用蒸馏，cost filtering 用 blockwise NAS，refinement 用结构化剪枝。

## Benchmark 看的是不同失败面

Stereo 论文常同时报告 Middlebury、ETH3D、KITTI、Scene Flow 等数据集，因为它们不是同一种难度。

| Benchmark | 主要检验什么 |
| --- | --- |
| Scene Flow | 大规模合成监督，适合训练和看平均 EPE |
| KITTI 2012 / 2015 | 自动驾驶场景，远距离、车辆、道路、动态物体 |
| Middlebury | 高分辨率室内物体，细节、遮挡、边界和小结构 |
| ETH3D | 真实采集场景，纹理和几何更接近实际相机 |
| In-the-wild stereo | 跨设备、跨场景、反光/透明/弱纹理，考验 zero-shot |

常见指标也要分开读：

| 指标 | 含义 | 容易漏掉什么 |
| --- | --- | --- |
| EPE | 平均 disparity 误差 | 平均值可能掩盖局部坏点 |
| Bad pixel / BP-1 / BP-2 | 误差超过 1 或 2 像素的比例 | 不同阈值严格度不同 |
| D1 | KITTI 常用 outlier 指标 | 更贴近自动驾驶 benchmark 口径 |
| Runtime / FPS | 推理速度 | 必须看硬件、分辨率、batch、runtime |
| Memory | volume 和 refinement 显存 | 高分辨率部署常先卡内存 |

Middlebury v3 明确提供多个 disparity accuracy measures，包括不同阈值的 bad pixel、平均绝对误差、RMS 和误差分位数。这个设计很合理：stereo 不是只要平均误差低，还要看边界、遮挡和长尾区域坏到什么程度。对机器人来说，少数坏点如果落在抓取边缘、桌沿或透明物体上，后果会比平均指标暗示的更严重。

## 读 stereo 论文时先画三张图

第一张图是几何图：相机是否标定、左右图是否 rectified、baseline 和焦距是否已知、输出是 disparity 还是 depth。没有这张图，后面所有网络结构都缺输入假设。

第二张图是 cost volume 图：特征分辨率是多少，最大 \(D\) 多大，volume 是 concat 还是 correlation，后续用 3D conv、hourglass、cascade 还是 Transformer 处理。它决定显存、速度和模型能看到多少候选证据。

第三张图是错误图：低纹理、重复纹理、遮挡、反光、透明、远距离和细小结构分别怎么处理。只看 leaderboard 排名，很容易错过模型在真实闭环里的失败点。

## 阅读结论

双目匹配的核心不是“深度估计网络”，而是“几何约束下的对应点搜索”。Rectification 把搜索压到同一行，disparity 给出可三角测量的深度，cost volume 保存所有候选位移的证据，cost filtering 用上下文清理证据，refinement 把初始视差修成可用结果。

读 FoundationStereo、Fast-FoundationStereo、RAFT-Stereo、PSMNet 或任何 stereo 模型时，先问：输入几何假设是什么，cost volume 怎么构造，哪里最耗显存和 latency，哪些失败区域被评测覆盖，输出 disparity 最后如何进入机器人、AR 或自动驾驶的控制链。

## 外部精读

- [OpenCV: Depth Map from Stereo Images](https://docs.opencv.org/4.x/dd/d53/tutorial_py_depthmap.html)：用最小代码和公式理解 disparity 到 depth。
- [LearnOpenCV: Introduction to Epipolar Geometry and Stereo Vision](https://learnopencv.com/introduction-to-epipolar-geometry-and-stereo-vision/)：适合补 epipolar geometry、对应点和深度直觉。
- [GC-Net](https://arxiv.org/abs/1703.04309)：理解 cost volume 加 3D convolution 的深度 stereo 路线。
- [PSMNet](https://arxiv.org/abs/1803.08669)：理解 spatial pyramid pooling、3D hourglass 和 cost volume regularization。
- [RAFT-Stereo](https://arxiv.org/abs/2109.07547)：理解 recurrent update 和 multilevel correlation / cost 证据。
- [FoundationStereo](https://arxiv.org/abs/2501.09898)：理解 zero-shot stereo 为什么需要大规模数据、单目先验和长程 cost filtering。
- [Fast-FoundationStereo](https://arxiv.org/abs/2512.11130)：理解一个强 stereo foundation model 如何被拆成实时系统。
- [Middlebury Stereo Evaluation v3](https://vision.middlebury.edu/stereo/eval3/MiddEval3-newFeatures.html)：理解 bad pixel、平均误差和分位数指标。
