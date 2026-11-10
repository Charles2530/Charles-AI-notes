# 坐标变换与手眼协调：把相机观测接到机器人动作

视觉模型说物体在图像左边，机器人为什么不能直接把末端向左移？因为这句话没有说明相机、末端、基座和世界坐标之间的关系。只有坐标系、变换方向、单位、时间戳和标定版本都明确，视觉位姿才足以进入规划与控制。


## 坐标链使用的基本对象

- [几何、视觉与三维空间](../prerequisite-math/geometry-vision-and-3d.md)：旋转、SE(3)、相机外参与点云。
- [相机、深度与机器人视觉](cameras-depth-and-robot-vision.md)：像素、深度、内参与外参。
- [机器人 State / Action 数据](state-action-robot-datasets.md)：末端位姿、绝对动作和相对动作的字段语义。

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

固定相机和腕上相机的区别，可以沿同一个杯子点追踪。腕上相机看到的点为 \({}^Cp\)，先通过固定手眼变换 \({}^ET_C\) 进入末端坐标，再通过随关节变化的 \({}^BT_E(q)\) 进入基座坐标

\[
{}^Bp={}^BT_E(q)\,{}^ET_C\,{}^Cp.
\]

机器人换姿态后，杯子在相机里的坐标会变，末端到基座的变换也变；只要杯子真的静止，这两项共同变化后得到的基座点应保持一致。于是可以把“移动机器人后，估计的静止杯子是否也移动”作为标定诊断。不过时间戳错位也会造成这种现象，因此需要先让相机观测与当时关节位姿对齐。

手眼标定进一步处理其中未知的固定变换。一个姿态无法唯一分辨相机安装误差与标定板的位置，多个有足够旋转变化的姿态才提供独立约束。下面从同一块静止标定板在两个姿态下的等式消元，得到熟悉的 AX=XB。

### 从固定标定板推导 AX=XB

设 \(G_i={}^BT_{E_i}\) 是第 \(i\) 个姿态的末端到底座变换，\(C_i={}^CT_{O,i}\) 是标定板到相机的变换，未知安装外参为 \(X={}^ET_C\)。标定板相对底座固定，因此不同拍摄时刻满足

\[
G_iXC_i=G_jXC_j={}^BT_O.
\]

左乘 \(G_j^{-1}\)，右乘 \(C_i^{-1}\)，得到

\[
\underbrace{G_j^{-1}G_i}_{A}X
=X\underbrace{C_jC_i^{-1}}_{B},\qquad AX=XB.
\]

这一步明确了相对运动方向，避免只记住方程却把输入逆错。将刚体矩阵展开，旋转与平移分别满足

\[
R_AR_X=R_XR_B,
\qquad(R_A-I)t_X=R_Xt_B-t_A.
\]

通常先利用多组旋转约束估计 \(R_X\)，再解平移线性系统，或联合优化。单次纯平移时 \(R_A-I=0\)，无法从这项约束确定所有安装平移；多次都绕同一轴旋转，也会留下弱可观测方向。采集姿态需要有足够的旋转轴变化与空间覆盖。

OpenCV 的 `calibrateHandEye` 在 eye-in-hand 设置中接收 gripper-to-base 与 target-to-camera，输出 camera-to-gripper。这些方向对应上面的 \(G_i,C_i,X\)。eye-to-hand 的未知关系和固定条件不同，需要重新写坐标链，不能不加推导地替换变量名。

解出 \(X\) 后，在未参与求解的姿态计算 \(G_iXC_i\)。固定标定板在底座中的估计应基本一致。可以分别统计位置误差与旋转角误差，后者用

\[
e_R=\arccos\left(\operatorname{clip}\left(\frac{\operatorname{tr}(R_1^\top R_2)-1}{2},-1,1\right)\right).
\]

clip 防止浮点误差把反余弦输入推出合法范围。低训练残差只说明拟合了采集姿态，不能代替工作区外的新姿态验证。

### 局部增量和世界增量的乘法顺序不同

策略输出末端增量时，必须说明增量在哪个 frame 中定义。Body-frame 或局部增量通常右乘：

\[
T_{t+1}^{\text{body}}=T_t\Delta T.
\]

World-frame 或 base-frame 增量通常左乘：

\[
T_{t+1}^{\text{world}}=\Delta T T_t.
\]

对于纯平移增量，右乘表示沿末端当前朝向移动，左乘表示沿固定世界轴移动。若左乘增量还含旋转，它也会围绕世界原点旋转当前平移向量；想只在原地改姿态，应单独定义旋转中心。当末端已经旋转时，同一个 \(\Delta x\) 会产生不同轨迹。VLA 动作、遥操作示教和控制器若对这项协议理解不同，就会出现数值尺度正常、运动方向却错误的失败。

### 用 NumPy 做最小数值验证

下面的教学代码构造 \(90^\circ\) 旋转和平移例子，并检查变换、逆变换和旋转矩阵约束：

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


## 外部材料

- 原始笔记：[机器人基础和控制、手眼协调](https://v11enp9ok1h.feishu.cn/wiki/D5tuwjNfOiWPeakiNEhc88yOnqd?from=from_copylink)。
- 权威接口：[OpenCV calibrateHandEye](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)。

继续阅读：[机器人运动学与 DH 参数](robot-kinematics-and-dh-parameters.md)把 base、gripper 和 tool 的变换接到正逆运动学；[相机、深度与机器人视觉](cameras-depth-and-robot-vision.md)补充内参、深度和像素到相机坐标的入口。
