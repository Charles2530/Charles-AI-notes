# 坐标变换与手眼协调：把相机观测接到机器人动作

视觉模型可以在图像里找到物体，但机器人控制器不能直接执行“向图片左边移动”。视觉估计给出的位姿只有在坐标系、变换方向、单位、时间戳和标定版本都明确时，才能变成规划与控制目标。

## 本章回答什么

从相机观测到机器人动作的最小空间接口包括：怎样表示旋转和平移；怎样组合、求逆和检查一条坐标变换链；eye-in-hand 与 eye-to-hand 有什么不同；手眼标定解决哪个未知变换；为什么同一个末端增量在局部坐标和世界坐标中会产生不同轨迹。

读完以后，应能把“相机看到了物体”展开成一条可验证的链：

~~~text
相机中的物体位姿
-> 相机与末端之间的标定
-> 末端与基座之间的运动学
-> 基座中的目标位姿
-> planner / controller
~~~

## 前置知识

- [几何、视觉与三维空间](../prerequisite-math/geometry-vision-and-3d.md)：旋转、SE(3)、相机外参与点云。
- [相机、深度与机器人视觉](cameras-depth-and-robot-vision.md)：像素、深度、内参与外参。
- [机器人 State / Action 数据](state-action-robot-datasets.md)：末端位姿、绝对动作和相对动作的字段语义。

## 核心机制

### 坐标系名称必须对应一条变换方向

机器人系统里常见的 frame 包括：

| 坐标系 | 常见含义 | 主要来源 |
| --- | --- | --- |
| world / map | 场景或地图的全局参考 | 建图、仿真或任务定义 |
| base | 机器人本体的固定参考 | 机器人模型 |
| gripper / end-effector | 末端执行器或法兰 | 正运动学 |
| camera | 相机光学坐标 | 相机标定与安装 |
| object | 物体自身参考 | 6D pose 或资产标注 |

这里用 \(^{A}T_B\) 表示“把 \(B\) 坐标系中的坐标变换到 \(A\) 坐标系”。因此：

\[
\bar p_A = {}^{A}T_B \bar p_B.
\]

上标是目标坐标系，下标是来源坐标系。camera_to_base、base_T_camera 这类变量名若没有明确约定，可能分别指同一个变换，也可能方向相反；可靠代码应把数学记号写进注释、类型或字段文档。

### 旋转矩阵与齐次变换

三维刚体位姿由旋转和平移组成：

\[
{}^{A}T_B=
\begin{bmatrix}
{}^{A}R_B & {}^{A}t_B\\
0 & 1
\end{bmatrix},
\qquad
\bar p =
\begin{bmatrix}
x & y & z & 1
\end{bmatrix}^{\top}.
\]

其中 \(R\in SO(3)\) 改变坐标轴方向，\(t\in\mathbb{R}^3\) 表示 \(B\) 原点在 \(A\) 中的位置。合法旋转矩阵满足：

\[
R^\top R=I,\qquad \det R=1.
\]

以绕 \(z\) 轴旋转为例：

\[
R_z(\theta)=
\begin{bmatrix}
\cos\theta & -\sin\theta & 0\\
\sin\theta & \cos\theta & 0\\
0 & 0 & 1
\end{bmatrix}.
\]

若 \(B\) 相对 \(A\) 绕 \(z\) 轴旋转 \(90^\circ\)，且 \(B\) 原点在 \(A\) 中为 \((2,1,0)\)，那么 \(B\) 中的点 \((1,0,0)\) 会落在 \(A\) 中的 \((2,2,0)\)：局部 \(x\) 轴先转向全局 \(y\) 轴，再叠加原点平移。

### 组合与求逆形成可检查的坐标链

从 \(C\) 经过 \(B\) 到 \(A\) 时，变换按坐标链相乘：

\[
{}^{A}T_C = {}^{A}T_B {}^{B}T_C.
\]

相邻的 \(B\) 上下标能够“消去”，这是检查乘法方向最直接的方法。若需要反向变换：

\[
({}^{A}T_B)^{-1}
=
{}^{B}T_A
=
\begin{bmatrix}
R^\top & -R^\top t\\
0 & 1
\end{bmatrix}.
\]

变换不是普通平移向量，不能只把 \(t\) 取负。逆平移还要经过 \(R^\top\)，因为它必须先回到反向坐标轴中。

### Eye-in-hand 与 eye-to-hand

Eye-in-hand 把相机安装在夹爪或腕部。相机跟随机械臂移动，近距离观察和接触校正更直接，但相机外参依赖末端安装刚度。物体在基座中的位姿为：

\[
{}^{\mathrm{base}}T_{\mathrm{object}}
=
{}^{\mathrm{base}}T_{\mathrm{gripper}}
{}^{\mathrm{gripper}}T_{\mathrm{camera}}
{}^{\mathrm{camera}}T_{\mathrm{object}}.
\]

~~~mermaid
flowchart LR
  Base[base frame] -->|正运动学| Gripper[gripper frame]
  Gripper -->|手眼标定| Camera[camera frame]
  Camera -->|视觉位姿| Object[object frame]
  Object -->|目标位姿| Planner[planner / controller]
~~~

Eye-to-hand 把相机固定在机器人外部。相机通常能看到更大工作区，但机器人或人员可能造成遮挡。此时要估计的是相机与 base/world 的固定关系；它和 eye-in-hand 使用的观测组合与未知外参不同，不能只把同一组矩阵换个变量名。

### 手眼标定为什么写成 AX=XB

Eye-in-hand 中，相机相对夹爪的安装变换在多次采集之间保持固定，记为 \(X\)。机器人从一个姿态移动到另一个姿态时，编码器和正运动学给出 gripper motion \(A\)，相机观察标定板得到 camera motion \(B\)。两条路径描述同一次刚体运动，因此满足：

\[
AX=XB.
\]

这条方程的意义不是让控制循环每一步都求一次 \(X\)，而是利用多组机器人姿态和标定板观测，离线估计固定的相机—夹爪外参。运动姿态若过少、旋转轴变化不足、标定板检测噪声大或时间戳错位，解可能不稳定。

OpenCV 的 calibrateHandEye 在 eye-in-hand 设置中接收多组 gripper-to-base 变换 \(^{b}T_g\) 和 target-to-camera 变换 \(^{c}T_t\)，输出 camera-to-gripper 变换 \(^{g}T_c\)。Eye-to-hand 需要按照官方文档重新组织合适的输入方向；把两种安装方式混用会得到数值合法但物理方向错误的结果。

### 局部增量和世界增量的乘法顺序不同

策略输出末端增量时，必须说明增量在哪个 frame 中定义。Body-frame 或局部增量通常右乘：

\[
T_{t+1}^{\text{body}}=T_t\Delta T.
\]

World-frame 或 base-frame 增量通常左乘：

\[
T_{t+1}^{\text{world}}=\Delta T T_t.
\]

右乘表示“沿末端当前朝向移动”，左乘表示“沿固定世界轴移动”。当末端已经旋转时，同一个 \(\Delta x\) 会产生不同轨迹。VLA 动作、遥操作示教和控制器若对这项协议理解不同，就会出现数值尺度正常、运动方向却错误的失败。

### 用 NumPy 做最小数值验证

下面的代码构造飞书笔记中的 \(90^\circ\) 旋转和平移例子，并检查变换、逆变换和旋转矩阵约束：

~~~python
import numpy as np


def rotation_z(theta_deg):
    theta = np.radians(theta_deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def make_transform(rotation, translation):
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


T_A_B = make_transform(rotation_z(90), np.array([2.0, 1.0, 0.0]))
p_B = np.array([1.0, 0.0, 0.0, 1.0])
p_A = T_A_B @ p_B
T_B_A = np.linalg.inv(T_A_B)

assert np.allclose(p_A[:3], [2.0, 2.0, 0.0])
assert np.allclose(T_B_A @ p_A, p_B)
assert np.allclose(T_A_B[:3, :3].T @ T_A_B[:3, :3], np.eye(3))
assert np.isclose(np.linalg.det(T_A_B[:3, :3]), 1.0)
~~~

np.linalg.inv 适合这个教学验证；工程代码若已知输入是刚体变换，也可以显式使用 \(R^\top\) 和 \(-R^\top t\)，同时检查输入是否仍满足刚体约束。

## 常见错误与失效边界

| 错误 | 表面现象 | 应检查什么 |
| --- | --- | --- |
| 把 \(^{A}T_B\) 当成 \(^{B}T_A\) | 物体位置镜像、偏移或方向完全相反 | 点的来源 frame、目标 frame 和逆变换 |
| 混淆 active / passive rotation | 单点例子能对，组合后朝向错误 | 公式是在旋转物体还是切换坐标描述 |
| body-frame 与 world-frame 混用 | 末端旋转后平移方向变化 | 增量左乘还是右乘 |
| 角度当成弧度 | 小角度动作突然大幅旋转 | API 单位与归一化 |
| 米和毫米混用 | 轨迹放大或缩小一千倍 | 数据集、标定和控制器单位 |
| 四元数顺序混用 | 姿态跳变或旋转轴异常 | wxyz、xyzw 和归一化 |
| 忽略时间同步 | 静态准确、运动时持续偏移 | 图像、关节状态与命令时间戳 |
| 标定版本漂移 | 重启、换相机或碰撞后误差增大 | 序列号、安装状态与标定文件版本 |

手眼标定只估计坐标关系，不会自动修复相机内参错误、机械臂运动学误差、末端柔性形变、检测偏差或控制延迟。变换链正确也不代表抓取一定可达或无碰撞，这些仍由运动学、planner 和 controller 处理。

## 工程检查表

1. 为每个矩阵写明来源 frame、目标 frame、单位和时间戳。
2. 检查 \(R^\top R\approx I\)、\(\det R\approx1\) 和齐次矩阵最后一行。
3. 用已知点验证 \(p_A={}^AT_Bp_B\)，再验证 \(T^{-1}(Tp)=p\)。
4. 检查变换连乘时相邻上下标是否匹配。
5. 用静态标定板在工作区不同位置重复测量，观察误差是否随距离或姿态系统变化。
6. 记录相机、夹具、机器人模型、标定文件和软件版本。
7. 在真实执行前用低速、限幅和安全区验证动作方向。

## 检查理解

1. \(^{A}T_B\) 把哪个坐标系中的点变到哪个坐标系？
2. 为什么刚体变换的逆平移不是简单的 \(-t\)？
3. Eye-in-hand 中 \(^{\mathrm{gripper}}T_{\mathrm{camera}}\) 为什么可以作为固定外参？
4. AX=XB 中的 \(A\)、\(B\) 和 \(X\) 分别对应哪类运动或外参？
5. 为什么 \(T_t\Delta T\) 和 \(\Delta T T_t\) 会产生不同轨迹？
6. 变换链数值看起来正常时，还需要检查哪些单位、时间和标定条件？

## 外部材料

- 原始笔记：[机器人基础和控制、手眼协调](https://v11enp9ok1h.feishu.cn/wiki/D5tuwjNfOiWPeakiNEhc88yOnqd?from=from_copylink)。
- 权威接口：[OpenCV calibrateHandEye](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)。
