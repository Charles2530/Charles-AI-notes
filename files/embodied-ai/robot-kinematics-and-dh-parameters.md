# 机器人运动学与 DH 参数：从关节状态算到末端位姿

机器人数据中的关节角不是末端位置，视觉模型给出的目标位姿也不能直接变成电机角度。两者之间需要机器人运动学：正运动学把关节状态映射成末端位姿，逆运动学寻找满足目标与约束的关节解，规划和控制再决定怎样安全到达并稳定跟踪。

## 本章回答什么

本文从串联机械臂出发，解释关节空间、任务空间、Forward Kinematics（FK）、Inverse Kinematics（IK）和 Jacobian 的职责边界；用经典 Denavit--Hartenberg（DH）四参数建立相邻连杆变换；再通过二维两连杆和 NumPy 示例验证整条运动学链。

读完以后，应能回答：

1. 已知关节状态 \(q\)，末端执行器在 base frame 中的位姿怎样计算？
2. DH 表中的 \(a_i,\alpha_i,d_i,\theta_i\) 分别描述什么？
3. revolute 与 prismatic 关节把哪个参数作为变量？
4. 为什么经典 DH、改进 DH、URDF 和厂商模型不能按参数名直接混用？
5. 为什么 IK 不是把正运动学矩阵简单求逆？

## 前置知识

- [坐标变换与手眼协调](coordinate-transforms-and-hand-eye-calibration.md)：齐次变换、组合、求逆和 frame 方向。
- [机器人 State / Action 数据](state-action-robot-datasets.md)：关节状态、末端位姿和动作字段。
- [动态系统、控制与状态空间](../prerequisite-math/dynamic-systems-and-control.md)：状态、动力学和反馈控制。

## 核心机制

### 关节空间和任务空间是两种接口

关节空间用：

\[
q=
\begin{bmatrix}
q_1 & q_2 & \cdots & q_n
\end{bmatrix}^{\top}
\]

描述每个关节的位置。对旋转关节，\(q_i\) 通常是角度；对移动关节，\(q_i\) 通常是线位移。任务空间描述末端执行器在某个参考坐标系中的位置和姿态：

\[
{}^{\mathrm{base}}T_{\mathrm{ee}}(q)\in SE(3).
\]

两种表示服务不同接口。编码器和底层控制器更接近关节空间；抓取目标、工具轨迹和视觉位姿通常在任务空间中表达。一个六维末端目标不能与六个关节角逐元素对应，因为多个关节会共同改变同一个末端位置和方向。

### FK、IK、Jacobian、规划与控制各自负责什么

| 模块 | 输入与输出 | 回答的问题 |
| --- | --- | --- |
| FK | \(q\rightarrow{}^{\mathrm{base}}T_{\mathrm{ee}}\) | 当前关节状态让末端在哪里 |
| IK | 目标位姿与约束 \(\rightarrow q\) 候选 | 哪些关节状态可以到达目标 |
| Jacobian | \(\dot q\rightarrow v_{\mathrm{ee}}\) 的局部线性映射 | 关节小速度怎样形成末端速度 |
| planner | 起点、目标、碰撞与限位 \(\rightarrow\) 路径 | 怎样无碰撞地到达 |
| controller | 参考轨迹与反馈 \(\rightarrow\) 硬件命令 | 怎样稳定跟踪 |

FK 是确定的前向计算。IK 可能有多个解、没有解，或解落在关节限位和碰撞约束之外。Jacobian 只在当前姿态附近描述局部变化；planner 与 controller 不能被一个 IK 解替代。

### 经典 DH 用四个参数连接相邻连杆

经典 DH 为每对相邻连杆分配一组参数：

| 参数 | 名称 | 经典 DH 中的作用 |
| --- | --- | --- |
| \(a_i\) | link length | 沿 \(x_i\) 的平移 |
| \(\alpha_i\) | link twist | 绕 \(x_i\) 的旋转 |
| \(d_i\) | link offset | 沿 \(z_{i-1}\) 的平移 |
| \(\theta_i\) | joint angle | 绕 \(z_{i-1}\) 的旋转 |

相邻坐标系的变换顺序必须与所采用的约定一起写明。本文使用经典 DH：

\[
{}^{i-1}T_i
=
R_z(\theta_i)T_z(d_i)T_x(a_i)R_x(\alpha_i).
\]

对 revolute 关节，\(\theta_i\) 通常是变量，其余参数由连杆结构和零位约定给出。对 prismatic 关节，\(d_i\) 通常是变量。实际模型还可能包含固定 offset，不能只根据“旋转关节”四个字就假设零位为零。

### 单节变换怎样写成 4×4 矩阵

令 \(c_{\theta_i}=\cos\theta_i\)、\(s_{\theta_i}=\sin\theta_i\)，经典 DH 单节矩阵为：

\[
{}^{i-1}T_i=
\begin{bmatrix}
c_{\theta_i} & -s_{\theta_i}c_{\alpha_i} & s_{\theta_i}s_{\alpha_i} & a_i c_{\theta_i}\\
s_{\theta_i} & c_{\theta_i}c_{\alpha_i} & -c_{\theta_i}s_{\alpha_i} & a_i s_{\theta_i}\\
0 & s_{\alpha_i} & c_{\alpha_i} & d_i\\
0 & 0 & 0 & 1
\end{bmatrix}.
\]

矩阵的前三列描述新坐标轴在上一坐标系中的方向，最后一列描述新原点的位置。若矩阵与手册结果不一致，先检查采用的是经典 DH 还是改进 DH，而不是立即修改正负号凑结果。

### 多节连乘得到末端位姿

对 \(n\) 节串联机械臂：

\[
{}^{\mathrm{base}}T_{\mathrm{ee}}(q)
=
{}^{0}T_1(q_1)
{}^{1}T_2(q_2)
\cdots
{}^{n-1}T_n(q_n).
\]

相乘顺序表达了从 base 沿连杆树走到末端的坐标链。末端若还安装工具、夹爪指尖或相机，需要继续乘固定工具偏置：

\[
{}^{\mathrm{base}}T_{\mathrm{tool}}
=
{}^{\mathrm{base}}T_{\mathrm{flange}}
{}^{\mathrm{flange}}T_{\mathrm{tool}}.
\]

这也说明手眼标定和正运动学会在同一条系统链中相遇：FK 提供 base-to-gripper，手眼标定提供 gripper-to-camera。

### 二维两连杆先建立可手算直觉

平面两连杆长度分别为 \(l_1,l_2\)，关节角为 \(\theta_1,\theta_2\)。末端位置为：

\[
x=l_1\cos\theta_1+l_2\cos(\theta_1+\theta_2),
\]

\[
y=l_1\sin\theta_1+l_2\sin(\theta_1+\theta_2).
\]

取 \(l_1=l_2=1\)：

- \(\theta_1=90^\circ,\theta_2=0^\circ\) 时，两根连杆同向，末端在 \((0,2)\)。
- \(\theta_1=90^\circ,\theta_2=-90^\circ\) 时，第二根连杆转向全局 \(x\) 轴，末端在 \((1,1)\)。

这个例子说明关节角会通过连杆级联共同决定末端位置。同一个可达点可能存在肘上、肘下等多个 IK 解；距离 base 超过 \(l_1+l_2\) 的目标则不可达。

### 用 NumPy 从 DH 表计算正运动学

下面的实现严格对应本文声明的经典 DH 顺序：

~~~python
import numpy as np


def dh_transform(a, alpha, d, theta):
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def forward_kinematics(dh_rows):
    transform = np.eye(4)
    for a, alpha, d, theta in dh_rows:
        transform = transform @ dh_transform(a, alpha, d, theta)
    return transform


q1, q2 = np.radians([90.0, -90.0])
T_base_ee = forward_kinematics(
    [(1.0, 0.0, 0.0, q1), (1.0, 0.0, 0.0, q2)]
)

assert np.allclose(T_base_ee[:2, 3], [1.0, 1.0])
assert np.allclose(T_base_ee[:3, :3].T @ T_base_ee[:3, :3], np.eye(3))
~~~

输入角度在进入三角函数前已转换为弧度。真实机器人还需要加入厂商定义的零位 offset、工具偏置、基座安装变换和该台机器的标定修正。

### 经典 DH、改进 DH 和 URDF 不能混用

经典 DH 和改进 DH 都能描述串联连杆，但坐标系放置方式和基本变换顺序不同。相同的四列参数放进另一种公式，不会自动得到同一个模型。

URDF 则用 parent/child link、joint origin、joint axis 和 joint type 描述机器人树结构。它可以包含 fixed、continuous、revolute、prismatic 等关节，也能描述分支、视觉网格、碰撞体和惯性；它不是一张 DH 表的 XML 写法。

厂商发布的理想 DH 参数适合说明名义几何，但真实机械臂会因制造和装配误差使用单机标定数据。仿真模型、URDF、控制器和真实机器若不是同一版本，FK 计算看起来正确，末端仍可能产生系统偏差。

## 常见错误与失效边界

| 错误 | 后果 | 更稳的处理 |
| --- | --- | --- |
| 经典 DH 与改进 DH 混用 | 每节误差连续累积 | 记录 frame 放置和矩阵顺序 |
| 度与弧度混用 | 末端姿态明显异常 | 在 API 边界统一转换 |
| 忽略关节零位 offset | 全工作区出现系统偏差 | 用已知 home pose 回归 |
| 把 tool frame 当成 flange frame | 抓取点固定偏移 | 显式加入工具变换 |
| 只验证一个姿态 | 符号错误可能被偶然抵消 | 验证零位、单关节和组合姿态 |
| 把 IK 当成矩阵求逆 | 忽略多解、无解和限位 | 使用带约束 IK 与 seed |
| 忽略奇异位形 | 关节速度骤增或数值不稳定 | 检查 Jacobian 条件与速度限幅 |
| 只检查几何可达 | 轨迹可能碰撞 | 继续经过 planner 和 safety layer |

DH 和 FK 描述几何关系，不包含质量、惯量、摩擦、柔性、驱动延迟或接触力。要预测力矩和动态响应，还需要机器人动力学；要安全执行，还需要轨迹规划、控制和碰撞检查。

## 工程检查表

1. 明确使用经典 DH、改进 DH、URDF 还是厂商专用模型。
2. 为每个关节记录类型、轴方向、单位、零位 offset 和限位。
3. 用零位、单关节 \(90^\circ\) 和多关节组合姿态检查 FK。
4. 检查每个旋转块的正交性、行列式与齐次矩阵最后一行。
5. 对比厂商 SDK、URDF/TF 与自写 FK 的末端结果。
6. 把 flange、tool、gripper、camera 等固定偏置显式放进坐标链。
7. 将不可达、IK 多解、奇异、碰撞和超限分别记录为不同失败类型。
8. 真实执行前使用低速、限幅和安全工作区验证模型方向。

## 检查理解

1. 关节空间 \(q\) 和任务空间位姿分别服务哪些模块？
2. Forward Kinematics、IK、Jacobian、planner 和 controller 的输出有什么不同？
3. 经典 DH 的 \(a_i,\alpha_i,d_i,\theta_i\) 分别对应哪次平移或旋转？
4. revolute 与 prismatic 关节通常把哪个 DH 参数作为变量？
5. 为什么两连杆目标 \((1,1)\) 可能对应多个关节解？
6. 为什么 URDF 不能被视为“换了格式的 DH 表”？
7. FK 数值正确时，真实末端仍可能因哪些标定或执行因素产生偏差？

## 相关阅读与下一步

- 空间接口：[坐标变换与手眼协调](coordinate-transforms-and-hand-eye-calibration.md)。
- 数据语义：[机器人 State / Action 数据](state-action-robot-datasets.md)。
- 任务流水线：[资产、感知、抓取与轨迹 Pipeline](asset-perception-grasp-and-trajectory-pipeline.md)。
- 执行边界：[规划、控制与安全](planning-control-and-safety.md)。
- 原始笔记：[机器人运动学与 DH 参数](https://v11enp9ok1h.feishu.cn/wiki/UU7yw4IlDiiJX8kNWBucuCFJnab?from=from_copylink)。
- 正运动学参考：[Modern Robotics](https://modernrobotics.northwestern.edu/nu-gm-book-resource/4-1-1-product-of-exponentials-formula-in-the-space-frame/)。
- 工业参数参考：[Universal Robots DH 参数](https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/)。
- 模型边界参考：[ROS 2 URDF 可动关节](https://docs.ros.org/en/rolling/Tutorials/Intermediate/URDF/Building-a-Movable-Robot-Model-with-URDF.html)。
- 返回：[具身智能专题](index.md)。
