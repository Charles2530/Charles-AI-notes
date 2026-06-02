# 资产到轨迹：感知、抓取与数据管线

这一页把一个仿真机器人数据集项目拆成具身智能知识点。核心问题不是“生成一批 3D 模型”这么简单，而是把资产、物理属性、视觉观测、抓取标注、任务脚本和专家轨迹接成一条可复用的数据生产线。


可以先把整条线压成一句话：

```text
资产生成/收集 -> 尺寸与物理属性标注 -> 导入仿真 -> 抓取位姿标注
-> 场景和任务布置 -> 成功判定脚本 -> 轨迹规划生成 GT
-> 仿真验证 -> 失败复位和重试 -> 训练/评测数据
```

其中 `GT` 是 ground truth，指可以作为监督信号或评测基准的专家轨迹。对机器人来说，一条 GT 不是一个答案文本，而是一段随时间变化的动作序列。

![SpatialVLA experiment setup 原论文图](../assets/images/paper-deep-dives/embodied-ai/spatialvla/figure-3-experiment-setup.png){ width="720" }

<small>图源：[SpatialVLA](../paper-deep-dives/embodied-ai/spatialvla.md)，Figure 3。原论文图意：展示机器人实验中的多视角相机、机器人本体和任务场景设置，用于说明空间感知、动作网格和跨 embodiment 评测如何连接。</small>

!!! note "图解：资产管线不是离线建模小工具"
    这张图提醒我们，数据管线最终要落到真实相机、机器人本体、任务物体和动作接口上。资产、scale、camera pose、grasp pose 和 trajectory GT 如果不能在这样的实验设置里对齐，就无法稳定支持 VLA、WAM 或传统 planner。

## 硬证据模块：数据管线要能从资产追到失败轨迹

资产到轨迹的页面不能只讲 pipeline 步骤，还要说明每一步如何留下可验证证据。否则训练集看起来很大，失败时却不知道是资产尺度、碰撞网格、相机姿态、抓取标注还是轨迹规划出了问题。

| 证据项 | 最小可复算例子 | 失败案例 | 验收指标 |
| --- | --- | --- | --- |
| 本页解决哪项成本 | 用自动资产/轨迹生成减少人工示范和真机试错 | 自动生成很多轨迹，但真实抓取全滑 | valid trajectory yield、sim validation pass rate |
| 几何一致性 | asset scale、origin、collision mesh、camera pose 版本化 | 视觉 mesh 对，碰撞 mesh 偏，规划穿模 | collision check、reprojection error |
| 抓取证据 | grasp pose、approach direction、pregrasp、成功判定脚本 | 抓取标注可视化正确，但夹爪闭合后滑落 | grasp success、contact event F1 |
| 轨迹 replay | 每条 GT 记录 planner seed、失败重试、controller reject | 只保存最终动作，无法复现失败 | replay determinism、root-cause label |
| world-model 接口 | 资产/轨迹转成 episode schema 和 hard-negative 桶 | 仿真成功样本太干净，真实 near-miss 不会 | sim2real gap、hard replay coverage |

一个硬证据样例应能这样追溯：`asset scale -> collision mesh -> grasp annotation -> planned trajectory -> rendered multi-camera observation -> action chunk -> success checker -> failure label`。只有这条链能回放，资产数据才真正能进入世界模型数据引擎。

## 这套材料里出现了哪些具身知识

| 项目步骤 | 对应的具身智能知识 | 为什么重要 |
| --- | --- | --- |
| 3D 模型和纹理收集 | digital twin asset、视觉 mesh、纹理、类别与尺度 | 仿真里看见的物体必须有几何和外观，策略才能学到视觉与动作的关系 |
| 物理属性标注 | 刚体、质量、摩擦、反弹、碰撞网格 | 同一个动作抓轻泡沫和抓玻璃杯，结果可能完全不同 |
| 导入资产 | 坐标系、单位、origin、scale、collision shape | 尺寸或中心点错了，抓取点、碰撞检测和轨迹规划都会错 |
| 抓取轴标注 | grasp pose、approach direction、pregrasp、夹爪关节空间 | 告诉机械臂从哪里接近、夹爪怎么摆、夹到哪里更稳 |
| 场景布置 | clean / random scene、核心物体、干扰物、domain randomization | 干净场景测基本能力，随机场景测鲁棒性和泛化 |
| 任务状态评估脚本 | success condition、termination、reward、state query | 轨迹是否成功必须能自动判定，否则数据无法规模化生成 |
| cuRobo / STOMP 生成轨迹 | IK、碰撞检测、运动规划、轨迹优化 | 把“抓这个苹果”变成机械臂每个时刻应该怎么动 |
| 轨迹失败后重试 | reset、resample、planner retry、data engine | 真实批量生成中失败很常见，必须自动复位和换随机种子 |

这张表就是从项目材料回到具身智能章节的入口。后面每个概念都会在这条数据线里反复出现。

## 资产不是一张图，而是一组可仿真的对象描述

一个可用于机器人仿真的资产，至少要分成四层：

| 层级 | 例子 | 主要服务谁 |
| --- | --- | --- |
| 视觉外观 | `.obj`、`.usd`、纹理图、材质贴图 | 相机渲染、VLM/VLA 输入 |
| 几何尺寸 | 长宽高、scale、origin、轴向 | 放置、碰撞、抓取、任务判定 |
| 物理属性 | 质量、摩擦、刚体/可变形、玻璃/塑料/金属 | 仿真接触、滑动、搬运稳定性 |
| 操作标注 | 抓取点、抓取轴、可开启轴、功能点 | 轨迹规划、技能库、任务脚本 |

因此类似下面的目录只是开始：

```text
assets/apple/scale_080_080_070/
  info.json
  3d.obj
  textures/
```

更完整的资产包通常还要逐步补上：

```json
{
  "category": "apple",
  "scale_m": [0.08, 0.08, 0.07],
  "mass_kg": null,
  "material_class": "rigid_food",
  "friction": null,
  "visual_mesh": "3d.obj",
  "collision_mesh": null,
  "grasp_annotations": []
}
```

这里把 `mass_kg`、`friction` 先留成接口是合理的。早期可以先跑通视觉、导入、抓取和轨迹生成；等 pipeline 稳定后，再用更细的物理参数做域随机化和 sim2real 校准。

!!! note "难点解释：视觉 mesh 和碰撞 mesh 不是一回事"
    视觉 mesh 负责“看起来像苹果”，可以有很复杂的纹理和三角面。碰撞 mesh 负责“物理上怎么碰撞”，通常要更简单、更稳定，常用凸分解或简化几何。视觉模型好看但碰撞模型破碎，会导致机械臂明明看起来没碰到，仿真却判定穿模或卡住。

## 单目、双目、多视图和 RGB-D 分别是什么

机器人感知里最容易混的是“几张图”和“有没有深度”。它们不是一回事。

| 方式 | 输入 | 能直接得到什么 | 主要风险 |
| --- | --- | --- | --- |
| 单目 RGB | 一台相机的一张或一段 RGB 图像 | 颜色、纹理、2D 位置、语义类别 | 没有直接深度，绝对尺寸和距离容易靠先验猜错 |
| 双目 / stereo | 两台相机同时拍摄，并且已标定相对位姿 | 通过视差估计深度 | 需要精确标定，透明/反光/低纹理物体容易失败 |
| RGB-D | RGB 图 + 深度图 | 每个像素的大致 3D 距离 | 玻璃、黑色高反、金属表面常有深度空洞 |
| 多视图 | 多个角度的图像，可以来自多相机或绕物体拍摄 | 更完整的几何和外观覆盖 | 如果没有相机位姿或尺度，3D 重建会漂 |
| 生成式多视图 | 用图像模型生成前后左右上等视角 | 给图生 3D 模型补视觉条件 | 生成视图可能不几何一致，不能直接当真实测量 |

单目的核心问题是尺度模糊。相机看到一个苹果在图里占 100 像素，它可能是一个近处的小苹果，也可能是远处的大苹果。只有额外信息才能解这个问题，例如已知相机内参、深度传感器、双目视差、桌面平面、物体真实尺寸或多视角重建。

双目深度的直觉公式是：

\[
z \approx \frac{fB}{d}
\]


其中 \(z\) 是深度，\(f\) 是焦距，\(B\) 是两台相机之间的 baseline，\(d\) 是同一个点在左右图里的视差。视差越大，物体越近；视差越小，物体越远。

!!! note "多视图生成 3D 和双目视觉的区别"
    双目视觉是几何测量问题：两台真实相机、同一时刻、已知相机相对位姿，用视差算深度。多视图生成 3D 更像资产生产问题：用多张视角图让 3D 生成模型补全形状和纹理。前者偏在线感知，后者偏离线造资产。二者都用“多张图”，但可靠性和用途不同。

在这类流程里，`RGB 图像 -> 背景消除 -> 文字消除 -> 多视角生成 -> 图生 3D` 属于资产生产；真正部署机器人时，`相机观测 -> 物体检测/分割 -> 6D pose 或点云 -> 抓取/规划` 才是在线感知。

如果你对单目、双目、RGB-D、腕部相机、固定相机和点云还不熟，先读 [相机、深度与机器人视觉](cameras-depth-and-robot-vision.md)。那一页有图解，专门解释这些传感器在机器人上长什么样、为什么双目能估深度、为什么单目有尺度歧义。

## 尺度、坐标系和位姿是仿真资产的生命线

机器人不是只需要知道“这是苹果”，还需要知道它在世界里的位置、朝向和大小。常见的几个词要分清：

| 概念 | 简单解释 | 例子 |
| --- | --- | --- |
| scale | 模型单位到真实米制尺寸的缩放 | `.obj` 里 1 个单位到底是 1m、1cm 还是任意单位 |
| origin / pivot | 物体自身坐标系原点 | 苹果原点在中心，杯子原点在杯底，规划结果会不同 |
| pose | 位置 + 朝向 | 物体在桌上哪里，旋转了多少 |
| frame | 坐标系 | world frame、object frame、gripper frame、camera frame |
| transform | 坐标系之间的变换 | 从 object frame 转到 world frame |

抓取标注、任务判定和轨迹规划都依赖这些量。如果资产 scale 错了，轨迹规划器会以为杯口更大或更小；如果 origin 错了，`把物体上移 5cm` 可能不是沿你以为的方向移动。

!!! example "例子：为什么同样是可乐瓶，还要做大小号"
    如果任务要求“把圆柱体放入罐口尺寸接近的罐子”，模型必须理解相对尺度。只有同类不同尺寸资产，才能测试它是否真的比较了尺寸，而不是只记住“可乐瓶应该放进某个固定容器”。

## 抓取轴标注到底在标什么

抓取不是一个点，而是夹爪相对物体的一组位姿和动作参数。常见标注可以理解为：

\[
T_{\text{object}\rightarrow\text{gripper}} = (R, t)
\]


其中 \(t\) 是夹爪相对物体的位置，\(R\) 是夹爪相对物体的朝向。Isaac 的 grasp yaml 里 `position` 和 `orientation` 就是在描述这件事。

| 字段 | 含义 | 初学者理解 |
| --- | --- | --- |
| grasp position | 夹爪相对物体的位置 | 手伸到物体旁边的哪个位置 |
| grasp orientation | 夹爪相对物体的朝向 | 手掌和手指怎么转 |
| approach direction | 抓取前接近物体的方向 | 从上往下抓，还是从侧面夹 |
| pregrasp | 抓取前的预备位姿 | 先停在物体外侧一点，避免直接撞上 |
| cspace position | 夹爪关节角 | 手指张开多少、闭合多少 |
| confidence | 这个抓取候选的分数 | 越高越值得优先尝试 |

自动生成抓取候选时，系统可能找到一堆物理上可夹住的位置，但这些位置不一定符合任务语义。比如锤子抓锤头也许能把锤子拿起来，但如果任务是“使用锤子敲钉子”，抓锤头就是错误 affordance。这里要分清两种可行性：

| 可行性 | 问题 | 例子 |
| --- | --- | --- |
| 物理可行 | 能不能夹住、不掉、不碰撞 | 从锤头夹起锤子 |
| 任务可行 | 这个抓法是否支持后续任务 | 用锤子时应该抓手柄 |

所以抓取标注最好不要只存一个随机轴，而要存候选集合，并在任务层过滤。早期可以先做手动标注 pipeline：导入物体、移动到夹爪附近、仿真闭合、记录稳定抓取；后续再用自动候选生成加人工筛选提升效率。

![GR-2 bin picking setup 原论文图](../assets/images/paper-deep-dives/embodied-ai/gr2/figure-7-bin-picking-setup.png){ width="760" }

<small>图源：[GR-2](../paper-deep-dives/embodied-ai/gr2.md)，Figure 7。原论文图意：展示 bin picking 真实机器人设置、物体集合和相机/机械臂环境，用来评估机器人在多物体、遮挡和抓取选择中的泛化能力。</small>

!!! note "图解：抓取标注要服务真实任务，而不是只服务夹住"
    bin picking 的难点不是“有一个抓取点”这么简单，而是多个物体遮挡、候选抓法冲突、抓起后是否还能完成后续放置。读这类图时要把 grasp pose、物体实例、相机视角、夹爪几何和 success checker 放在一起看。

## 轨迹、动作和 ground truth 是什么关系

一条机器人轨迹是随时间变化的一串状态和动作。最常见的记录方式包括：

```text
t_0: joint_pos, end_effector_pose, gripper_command
t_1: joint_pos, end_effector_pose, gripper_command
t_2: joint_pos, end_effector_pose, gripper_command
...
```

几个概念不要混：

| 概念 | 说明 | 在机械臂里常见形式 |
| --- | --- | --- |
| action | 某一步要执行的命令 | 关节角增量、末端位姿增量、夹爪开合 |
| waypoint | 轨迹中的关键点 | pregrasp、grasp、lift、place |
| trajectory | 从起点到终点的一整段连续动作 | 一串关节角、速度、夹爪命令 |
| policy | 根据观测输出动作的模型或规则 | VLA、行为克隆策略、RL policy |
| planner | 根据目标和约束生成轨迹的算法 | cuRobo、STOMP、RRT、CHOMP |

一个抓放任务通常可以拆成：

```text
observe object pose
-> move to pregrasp
-> approach grasp pose
-> close gripper
-> lift
-> move to target pose
-> open gripper
-> retreat
```

由 planner 生成并通过仿真验证成功的这段动作序列，就可以作为 ground truth trajectory。它可以用于行为克隆，也可以用于评测 VLA 是否能复现专家动作。

## IK、FK、运动规划和轨迹优化

轨迹生成里最常见的几个底层概念是：

| 名词 | 解决什么问题 | 直觉 |
| --- | --- | --- |
| FK | 已知关节角，算末端在哪里 | 给定胳膊每个关节角，算手在空间的位置 |
| IK | 已知末端目标位姿，求关节角 | 想把手伸到杯子旁边，求肩肘腕怎么摆 |
| collision checking | 判断机器人/物体/环境是否碰撞 | 这条路会不会撞到桌子或其他物体 |
| motion planning | 找一条从当前状态到目标状态的可行路径 | 从 A 到 B 怎么走不碰撞 |
| trajectory optimization | 在可行路径上继续优化平滑、速度、距离和安全裕度 | 路径不只要能走，还要稳、短、平滑 |

`cuRobo` 可以理解为偏 GPU 加速的机器人运动规划和轨迹优化工具链，常用于在已知机器人、障碍物和目标位姿时快速生成可碰撞检查的机械臂轨迹。`STOMP` 是 stochastic trajectory optimization：它会对轨迹加扰动、评估代价、逐步把轨迹推向更低碰撞、更平滑、更接近目标的方向。

!!! note "难点解释：轨迹 GT 不是模型凭空生成的"
    轨迹 GT 通常来自规划器、遥操作、人类示范或已有专家策略。规划器需要知道物体初始位姿、目标位姿、机器人当前关节状态、障碍物和抓取标注。任何一个输入不准，都会导致轨迹不可执行或仿真失败。

## 为什么轨迹生成会失败，以及如何复位重试

批量生成轨迹时，失败不是异常，而是常态。常见失败原因包括：

- IK 无解：目标抓取位姿超出机械臂可达空间。
- 规划失败：有障碍物挡路，找不到无碰撞路径。
- 抓取失败：夹爪闭合后物体滑落或没夹住。
- 放置失败：物体落下后滚动、倾倒或碰撞其他物体。
- 判定失败：动作看起来完成了，但 success script 没有通过。
- 数值失败：仿真接触不稳定、穿模、时间步太大或 collision mesh 过差。

更稳的批量生成逻辑通常是：

```python
for task in tasks:
    for episode_id in range(target_episodes):
        for retry in range(max_retries):
            scene = sample_scene(task)
            grasp = sample_grasp(scene.core_object)
            plan = planner.solve(scene, grasp, task.goal)

            if plan is None:
                continue

            result = simulate(scene, plan)
            if success_checker(result):
                save_episode(scene, plan, result)
                break
        else:
            log_failed_case(task, episode_id)
```

这对应你材料里的“轨迹生成失败后重复生成逻辑”。它不是简单重复同一个规划，而是可以重新采样初始位姿、抓取候选、规划随机种子、干扰物布局或目标位置。

## clean 场景和 random 场景测的不是同一件事

| 场景 | 设置 | 主要测试 |
| --- | --- | --- |
| clean | 一个任务只涉及一个核心物品，干扰少 | 任务定义、资产标注、基础抓取和轨迹规划是否正确 |
| random | 核心物品不变，周围加入随机物品和布局 | 抗干扰、视觉定位、避障和任务判定鲁棒性 |

所以 `50 个 clean 任务 * 50 条轨迹` 更像基础覆盖；`50 个 random 任务 * 500 条轨迹` 更像泛化和鲁棒性数据。前期先用 clean 跑通 pipeline 是对的，因为 random 场景一旦失败，很难判断是资产问题、抓取问题、规划问题、还是任务脚本问题。

## 任务判定脚本在判什么

任务判定脚本不是给模型“打分好不好看”，而是查询仿真状态，判断任务目标是否真的达成。它通常看：

- 物体最终位置和朝向；
- 物体之间的相对关系；
- 物体是否稳定，没有继续滑动或掉落；
- 夹爪是否已经打开；
- 机器人是否没有碰撞或越界；
- 是否在规定时间内完成。

你给的堆叠判定代码，本质逻辑是：

1. 读取所有物体的最终世界坐标；
2. 按 \(z\) 高度排序，确定谁在下面、谁在上面；
3. 用下面物体的位置加上 `stack_offset` 得到理想上层位置；
4. 判断上层物体的 \(x, y, z\) 偏差是否小于阈值；
5. 检查左右夹爪是否打开；
6. 全部满足才算成功。

这类脚本是具身数据集里非常关键的一层。没有它，就无法自动过滤失败轨迹，也无法稳定评估策略。

## 尺寸任务为什么值得单独做

你列的特色任务都围绕“尺度理解”，这和普通抓放任务不同：

| 任务 | 测试的能力 |
| --- | --- |
| 按从小到大排序 | 相对尺寸比较、排序、空间排列 |
| 将圆柱放入尺寸接近的罐子 | 物体尺寸、容器开口尺寸、可放入关系 |
| 根据标签放入对应罐子 | 尺寸感知 + OCR/标签识别 + 分类绑定 |
| 按 5cm 间隔排成直线 | 绝对尺度、距离控制、坐标规划 |
| 从大到小堆叠 | 相对尺寸、重心、垂直稳定性 |
| 从小到大堆叠 | 反常规稳定性测试，容易暴露物理和规划问题 |

如果要加入拒绝类任务，可以让系统学会“没有合适目标时不执行”。例如：圆柱体直径大于所有罐口时，正确行为不是硬塞，而是拒绝、报告不可行或请求人工处理。这能测试 affordance 判断和安全边界。

## 最小可跑通版本建议

早期不要同时追求 100 个资产、50 个任务、自动抓取标注和复杂 random 场景。更稳的顺序是：

1. 选 5 类刚体资产，每类 2 个尺寸、2 个纹理变体。
2. 统一 scale、origin、collision mesh 和 `info.json` 字段。
3. 每个资产手动标 3 到 5 个稳定 grasp pose。
4. 先做 3 个 clean 任务：抓取、放置、按大小排序。
5. 每个任务生成 20 条轨迹，记录失败原因。
6. success checker 先写清楚，再扩大资产和任务数量。
7. clean 成功率稳定后，再加 random 干扰物和多相机观测。

这样做的好处是，每个失败都有可定位的责任层：资产、物理、抓取、规划、仿真、任务脚本或数据记录。等这些层都能解释清楚，再扩到 100 个资产和数万条轨迹，成本会低很多。

## 和 π0.5、DreamZero 的关系

这条资产和轨迹管线不是“传统工程”和“新模型路线”的二选一。它更像底座，给不同模型路线提供不同形式的数据和评测。

| 模型路线 | 需要这条 pipeline 提供什么 | 为什么 |
| --- | --- | --- |
| π0.5 这类 VLA | 多视角观测、语言任务、action chunk、成功/失败标签 | VLA 要学习从视觉语言状态到动作的映射，轨迹 GT 和 task progress 是核心监督 |
| DreamZero 这类 WAM | 对齐的视频、动作和机器人状态序列 | WAM 要同时学习未来画面和未来动作，视频-action 对齐比单纯最终成功更重要 |
| LingBot-World 这类模拟器 | 动作条件下的未来观测、场景状态和交互事件 | 世界模拟器要学习同一历史下不同动作带来不同未来 |
| 经典 planner baseline | 资产 scale、collision mesh、grasp pose、目标位姿、success checker | 没有几何和任务判定，无法稳定生成专家轨迹或验证模型输出 |

所以早期先把 clean 场景、抓取标注、轨迹生成和成功判定跑通，并不会落后于前沿 VLA/WAM 方向。相反，这些工程层越清楚，后面接 π0.5 式 VLA 或 DreamZero 式 WAM 时越容易定位问题。

## 和其他章节的连接

| 继续读 | 你会补上什么 |
| --- | --- |
| [VLA、WAM 与世界模型地图](vla-wam-world-model-system-map.md) | π0.5、DreamZero、LingBot-World 和这条资产/轨迹 pipeline 在同一套系统里如何分工 |
| [一个任务跑通具身闭环](closed-loop-task-walkthrough.md) | 用尺寸排序和容器匹配任务，把资产、抓取、轨迹、在线执行和 success checker 串成完整例子 |
| [相机、深度与机器人视觉](cameras-depth-and-robot-vision.md) | 单目、双目、RGB-D、点云和相机安装方式如何影响感知和抓取 |
| [Isaac Sim 与 RoboTwin](isaacsim-and-robotwin.md) | Isaac Sim、RoboTwin、cuRobo 和 success checker 在数据生成工具链里如何分工 |
| [规划、控制与安全](planning-control-and-safety.md) | 轨迹、MPC、低层控制、安全约束为什么不能被 VLA 替代 |
| [Sim2Real 与具身数据引擎](sim2real-and-data-engines.md) | 物理参数、随机化、失败回流和真实部署之间的关系 |
| [具身任务谱系与评测](task-taxonomy-and-benchmarks.md) | clean/random、尺寸任务、拒绝任务应该怎样组织成评测集 |
| [VLA 动作表示与控制接口](../vla/action-representation-and-interfaces.md) | 动作 token、连续控制、关节空间和末端位姿如何进入模型 |
| [VLA 数据与策略学习](../vla/data-and-policy-learning.md) | 轨迹 GT 如何变成行为克隆或策略学习数据 |
