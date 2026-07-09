# 几何、视觉与三维空间

这篇回答一个问题：相机内参、外参、深度、视差、姿态、四元数、点云、对极几何、SE(3) 和多视角几何为什么会成为具身智能和 3D 视觉论文的共同数学语言。

语言模型主要处理 token 序列，机器人和世界模型还要处理空间。空间不是一张图片那么简单：同一个物体会因为相机位置、焦距、遮挡和深度不同而投影成不同像素。几何知识就是把像素、相机和真实三维位置接起来。

## 针孔相机：三维点怎样落到二维图像

最简相机投影可以写成：

\[
\begin{bmatrix}
u\\v\\1
\end{bmatrix}
\sim
K
\begin{bmatrix}
X/Z\\Y/Z\\1
\end{bmatrix}
\]

其中 \((X,Y,Z)\) 是相机坐标系中的三维点，\((u,v)\) 是像素坐标，\(K\) 是相机内参矩阵。

常见内参矩阵是：

\[
K=
\begin{bmatrix}
f_x & 0 & c_x\\
0 & f_y & c_y\\
0 & 0 & 1
\end{bmatrix}
\]

这里 \(f_x,f_y\) 是焦距按像素尺度表示后的值，\(c_x,c_y\) 是主点位置。直觉上，内参描述“这台相机怎么把光线投到像素平面上”。

## 外参：相机在世界里的位置和朝向

外参把世界坐标变成相机坐标：

\[
X_{\text{cam}}
=
R X_{\text{world}}+t
\]

其中 \(R\) 是旋转矩阵，\(t\) 是平移向量。机器人多相机系统里，每个相机都有自己的外参；如果外参错了，不同视角的点云、深度和动作坐标就会对不上。

在 VLA 或具身数据集里，`calibration` 字段通常就是在记录这些内外参和坐标变换。

## 双目视差：深度来自左右图像的位移

双目几何里，同一个三维点会投影到左右相机的不同像素位置。左右位置差叫视差 \(d\)，深度可以粗略写成：

\[
Z=\frac{fB}{d}
\]

这里 \(Z\) 是深度，\(f\) 是焦距，\(B\) 是左右相机 baseline，\(d\) 是视差。

![Stereo camera model](../assets/images/localized-remote/embodied-ai/cameras-depth-and-robot-vision/stereo-camera-model-wikimedia.jpg){ width="680" }

<small>图源：[Wikimedia Commons: Stereo camera model](https://commons.wikimedia.org/wiki/File:Stereo-camera-model.jpg)，本站已用于相机与双目基础页。原图展示左右相机、baseline、投影点和深度关系。本站在这里用它说明：深度估计不是凭空猜距离，而是在几何约束下从视差反推三维位置。</small>

公式 \(Z=fB/d\) 的读法很直接：baseline 越大、焦距越大，同样视差对应的深度越远；视差越大，说明点离相机越近。Fast-FoundationStereo 这类论文关心 cost volume 和实时双目匹配，本质上就是在更稳、更快地估计 \(d\)。

## 位姿和 SE(3)：刚体在三维空间里的运动

三维刚体位姿通常由旋转和平移组成：

\[
T=
\begin{bmatrix}
R & t\\
0 & 1
\end{bmatrix}
\in SE(3)
\]

这里 \(R\in SO(3)\) 表示旋转，\(t\in\mathbb{R}^3\) 表示平移，\(SE(3)\) 表示三维刚体变换群。

VGGT、MapAnything、D4RT、Depth Anything 3 这类论文常要估计 camera pose、depth、point map 或 track。读这些结果时，要分清模型是在预测像素深度、相机位姿、点云坐标，还是跨帧轨迹。

## 四元数：机器人姿态常用的旋转表示

旋转矩阵直观，但有 9 个数，还要满足正交约束。机器人动作接口和 VLA 数据集里，经常用四元数表示末端姿态：

\[
q=(w,x,y,z)
\]

单位四元数满足：

\[
\|q\|_2=1
\]

它可以表示三维旋转。直觉上，\(w\) 控制旋转角度，\((x,y,z)\) 控制旋转轴方向。相比 Euler angle，四元数不容易遇到万向节锁；相比旋转矩阵，它更紧凑。

读机器人论文时要记住一个细节：\(q\) 和 \(-q\) 表示同一个旋转。因此有些 rotation loss 会写成：

\[
\min\left(\|q-\hat q\|_2,\|q+\hat q\|_2\right)
\]

这不是公式炫技，而是在处理四元数的双重表示。RT-2、\(\pi_0\) 系列、ACT、Diffusion Policy、MapAnything 这类动作或位姿预测论文里，看到 quaternion、axis-angle、6D rotation representation 时，都要先问：它表示的是相机姿态、物体姿态，还是机器人末端执行器姿态。

如果目标是机器人控制接口，优先看部署系统实际接收哪种姿态表示；如果目标是神经网络回归，连续性、归一化约束和 loss 是否处理等价表示，往往比人类可读性更重要。

## 点云：把深度图还原成三维点

如果知道像素 \((u,v)\)、深度 \(Z\) 和相机内参，可以反投影到三维：

\[
X=\frac{(u-c_x)Z}{f_x},
\qquad
Y=\frac{(v-c_y)Z}{f_y}
\]

这说明深度图不是最终目的，它可以变成三维点云，进而用于抓取、避障、重建和空间记忆。

在机器人里，RGB 负责语义，depth / point cloud 负责几何；只看 RGB，模型可能知道“这是杯子”，但不知道杯口朝向、距离和可抓取位置。

## 对极几何：两张图之间的像素约束

双目和多视角重建不只靠 \(Z=fB/d\)。更一般地，如果同一个三维点在两张图里的齐次像素坐标是 \(x_1,x_2\)，它们满足对极约束：

\[
x_2^\top F x_1=0
\]

这里 \(F\) 是 fundamental matrix。它的意思是：第一张图里的点 \(x_1\)，在第二张图里不会随便落在任意位置，而是必须落在一条极线附近：

\[
l_2=F x_1
\]

如果相机内参已知，还会看到 essential matrix：

\[
E=[t]_\times R
\]

其中 \(R,t\) 是两台相机之间的相对旋转和平移，\([t]_\times\) 表示由平移向量构成的叉乘矩阵。VGGT、Fast-FoundationStereo、MapAnything 和 SfM/MVS 相关论文里的 epipolar geometry，本质上是在用这种约束缩小匹配搜索空间。

对极几何的价值是：它不要求一开始就知道深度，而是先告诉你“匹配点应该去哪条线上找”。深度、位姿和三维点可以在后续通过三角化、PnP 或 bundle adjustment 进一步求出来。

## 多视角一致性：不同相机要说同一个世界

多视角系统最怕各相机各说各话。一个三维点从相机 A 变到相机 B，需要经过外参变换；同一个物体在不同视角下的 depth、mask、track 应该互相一致。

这也是为什么 embodied AI 论文会强调 calibration、multi-view、depth、normal、trajectory 和 3D flow。它们不是额外传感器名词，而是在给模型更强的空间约束。

两类常见名字也可以先这样读：PnP 是用 3D 点和 2D 像素对应来估计相机位姿；ICP 是用两组点云反复找对应、估计刚体变换，让它们对齐。它们都不是神经网络专属方法，但会成为 3D foundation model 和机器人系统里的几何验算工具。

## 常见误读

| 误读 | 更稳的理解 |
| --- | --- |
| 深度就是模型直接回归一个距离 | 很多深度来自相机几何、视差或多视角约束 |
| RGB 已经足够表达空间 | 遮挡、接触、尺度和可抓取性常需要几何信号 |
| 位姿只是可视化信息 | 位姿决定不同帧、不同相机、机器人动作能否对齐 |
| 四元数只是四个普通回归数 | 它必须表示单位旋转，且 \(q\) 和 \(-q\) 等价 |
| 双目深度只靠一条 \(Z=fB/d\) | 更一般的多视角匹配还依赖对极约束和位姿关系 |
| 3D 指标好就能控制 | 控制还需要动作接口、闭环反馈和安全约束 |

## 读完以后怎么判断

看到 depth、pose、camera、quaternion、epipolar、3D、point map 或 stereo，先问：坐标系是什么，内参外参是否已知，旋转怎样表示，匹配点受什么几何约束，深度来自视差还是学习先验，多视角是否一致，几何结果最后是否服务动作和闭环。这样读 Fast-FoundationStereo、VGGT、Depth Anything、SpatialVLA 和机器人世界模型会更稳。

## 相关阅读与下一步

- 站内下一步：[相机、深度与机器人视觉](../embodied-ai/cameras-depth-and-robot-vision.md)。
- 站内下一步：[双目匹配与 Cost Volume](../embodied-ai/stereo-matching-and-cost-volume.md)。
- 论文精读：[Fast-FoundationStereo：实时 Zero-Shot 双目匹配](../paper-deep-dives/inference/fast-foundation-stereo.md)。
- 论文精读：[VGGT：前向恢复相机、深度、点云与轨迹](../paper-deep-dives/embodied-ai/vggt.md)。
