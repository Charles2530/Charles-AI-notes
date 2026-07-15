# 具身智能坐标变换与运动学双文章设计

## 结论

在具身智能专题下新增两篇互补的体系型文章：

1. `files/embodied-ai/coordinate-transforms-and-hand-eye-calibration.md`：负责跨坐标系的空间表示、变换链和手眼协调。
2. `files/embodied-ai/robot-kinematics-and-dh-parameters.md`：负责从关节状态到末端位姿的正运动学与 DH 建模。

两篇共享统一的变换记号，但不重复完整推导。第一篇回答“不同传感器、机器人和物体坐标怎样对齐”，第二篇回答“关节变化怎样形成末端位姿”。它们共同补齐从视觉估计到机器人执行之间缺失的几何与运动学接口。

## 当前内容审计

仓库已有内容只覆盖了部分前置：

- `files/prerequisite-math/geometry-vision-and-3d.md` 介绍相机外参、SE(3)、四元数和多视角几何。
- `files/embodied-ai/cameras-depth-and-robot-vision.md` 介绍相机内外参、深度反投影和相机安装方式。
- `files/embodied-ai/asset-perception-grasp-and-trajectory-pipeline.md` 简要说明 FK、IK、抓取位姿和轨迹规划的职责。
- `files/embodied-ai/state-action-robot-datasets.md` 介绍末端位姿、相对动作和坐标语义。

现有文章没有系统讲解变换方向、组合与求逆、手眼坐标链、DH 四参数、连杆变换连乘及其数值验证。总览页中的新增段落只能作为脉络摘要，不能替代独立教学文章。

## 方案选择

采用“互补型双文章”，而不是以下方案：

- 不采用完全自包含的双文章，因为两篇都会重复旋转矩阵、齐次变换和坐标记号。
- 不采用按飞书原文直接扩写，因为原文适合快速笔记，尚未覆盖工程约定、失效模式、站内依赖和验证契约。
- 不合并为一篇长文，因为坐标对齐和机器人运动学是两个可独立学习、独立排错的知识单元。

## 信息架构

导航位置保持在“世界模型与具身智能 → 具身智能 → 闭环与状态”中，建议顺序为：

```text
观测、动作与反馈闭环
一个任务跑通具身闭环
机器人 State / Action 数据
相机、深度与机器人视觉
坐标变换与手眼协调
机器人运动学与 DH 参数
双目匹配与 Cost Volume
```

`files/embodied-ai/index.md` 的“感知与状态”和“按问题进入”分别增加两篇文章的入口。`files/world-models-and-embodied-ai/index.md` 的具身智能系统脉络保留两条独立知识点，但入口改为新文章；飞书链接放到对应文章的“来源与延伸阅读”，避免总览表承担来源管理。

## 文章一：坐标变换与手眼协调

文件：`files/embodied-ai/coordinate-transforms-and-hand-eye-calibration.md`

### 教学目标

读完后，读者应能：

1. 解释坐标系记号和变换方向。
2. 用旋转矩阵、平移向量和齐次矩阵表示刚体位姿。
3. 正确组合、求逆和验证一条坐标变换链。
4. 区分 eye-in-hand 与 eye-to-hand 的变换路径。
5. 解释手眼标定解决的未知外参及其工程输入输出。
6. 判断局部坐标增量和世界坐标增量为何产生不同轨迹。

### 建议结构

1. 本章回答什么、前置知识和能力终点。
2. 坐标系不是标签：camera、base、world、end-effector、object 的职责。
3. 统一记号：使用 `^{A}T_B` 表示从 B 到 A 的坐标变换。
4. 旋转矩阵：绕轴旋转、正交约束、行列式和数值检查。
5. 齐次变换：点变换、位姿变换、组合与逆变换。
6. 变换链：camera → object、base → end-effector、end-effector → camera 怎样组合。
7. 手眼协调：eye-in-hand、eye-to-hand、标定对象和 `AX=XB` 的接口级直觉。
8. 动作坐标：body-frame 右乘与 world/base-frame 左乘。
9. 最小 NumPy 示例：构造、组合、求逆并验证 round trip。
10. 常见错误：方向颠倒、主动/被动旋转混淆、米/毫米、角度/弧度、四元数顺序、标定版本漂移。
11. 工程检查表、自测题、来源与下一步。

正文使用一个 eye-in-hand 抓取任务贯穿示例，并用一张 Mermaid 图展示变换链。代码只依赖 NumPy，不扩展为完整 OpenCV/ROS 标定程序。

## 文章二：机器人运动学与 DH 参数

文件：`files/embodied-ai/robot-kinematics-and-dh-parameters.md`

### 教学目标

读完后，读者应能：

1. 区分关节空间与任务空间，以及 FK、IK、Jacobian、规划和控制的职责。
2. 解释经典 DH 的四个参数及旋转关节、移动关节的变量。
3. 写出单节变换和整条串联机械臂的正运动学连乘。
4. 用二维两连杆和一般 DH 表验证末端位姿。
5. 识别经典 DH、改进 DH、URDF 与厂商 SDK 约定不一致的风险。
6. 理解多解、无解、奇异位形和关节限位为何使 IK 不等于“矩阵求逆”。

### 建议结构

1. 本章回答什么、前置知识和能力终点。
2. 关节空间与任务空间：`q`、末端位姿和工具坐标系。
3. FK、IK、Jacobian、planner、controller 的边界。
4. 连杆坐标系与经典 DH 四参数：`a_i`、`alpha_i`、`d_i`、`theta_i`。
5. 单节经典 DH 变换的分解与显式矩阵。
6. 多节连乘：`base_T_ee(q)` 和工具偏置。
7. 旋转关节与移动关节的参数表。
8. 二维两连杆直觉示例：解析位置、可达域和肘上/肘下多解。
9. 最小 NumPy 示例：读取 DH 表、计算 FK、检查零位与已知姿态。
10. 约定边界：经典 DH、改进 DH、URDF、轴方向和零位偏置。
11. IK 失效边界：不可达、奇异、关节限位、碰撞与数值初值。
12. 工程检查表、自测题、来源与下一步。

正文不实现通用 IK 求解器，也不展开刚体动力学、力矩控制或完整 Jacobian 推导；这些内容只给出接口和后续入口。

## 来源与事实边界

两份飞书笔记作为原始知识点来源：

- “机器人基础和控制、手眼协调”：旋转矩阵、齐次变换和 NumPy 点变换示例。
- “机器人运动学与 DH 参数”：FK、标准 DH 四参数和相邻连杆变换。

实现阶段需要再用权威技术资料核对手眼标定方程、DH 约定和旋转/平移顺序。新增文章不得把经典 DH 公式写成适用于所有机器人描述格式的唯一约定，也不得把手眼标定简化成普通外参矩阵抄写。

## 修改范围

### 新增

- `files/embodied-ai/coordinate-transforms-and-hand-eye-calibration.md`
- `files/embodied-ai/robot-kinematics-and-dh-parameters.md`

### 修改

- `files/embodied-ai/index.md`
- `files/world-models-and-embodied-ai/index.md`
- `mkdocs.yml`
- `tests/test_world_models_embodied_curriculum.py`

不修改现有前置数学、相机视觉和资产 Pipeline 文章的主体内容，只增加必要的前后置链接时才做最小编辑。

## 验证与验收

1. 两篇文章均遵循现有教学文章风格：结论先行、公式解释、工程边界、来源和下一步。
2. 两篇文章分别包含至少一个可手算示例和一个 NumPy 验证代码块。
3. 两篇文章相互链接，并链接现有相机视觉、几何前置、State / Action、轨迹 Pipeline 和规划控制文章。
4. 具身智能专题首页、统一总览和 `mkdocs.yml` 均能进入两篇文章。
5. 课程测试验证新增页面存在、导航可达、核心关键词与本地链接完整。
6. `python -m unittest discover -s tests -p 'test_world_models_embodied_curriculum.py' -v` 通过。
7. `mkdocs build --strict` 通过且不产生新增断链提示。

## 非目标

- 不编写完整 ROS TF 教程。
- 不实现 OpenCV 手眼标定采集工具。
- 不实现通用 IK、Jacobian 或动力学库。
- 不复制飞书原文的全部代码和流程图。
- 不重构现有具身智能导航分组之外的章节。
