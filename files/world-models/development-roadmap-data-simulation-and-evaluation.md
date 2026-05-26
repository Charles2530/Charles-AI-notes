# 世界模型开发路线

这页把面向机器人任务的世界模型开发路线整理成一条更完整的教程线：从相机和几何数据开始，到 3D encoder、VLA / diffusion policy、仿真环境、success checker，再到一个能服务真实任务的世界模型闭环。

<!-- [补充] -->
| 读者 | 建议读法 | 工程落点 |
| --- | --- | --- |
| 初学者 | 先抓“世界模型开发路线”要解决的问题、关键术语和最小例子；公式只看变量方向。 | 能说清输入、输出和常见失败信号。 |
| 工程师 | 直接对照结构、指标、成本和边界条件，必要时回到前置页补概念。 | 落到数据、算力、延迟、显存、评测和回滚检查。 |


!!! note "读法定位"
    这页先回答“世界模型开发路线”在「世界模型」里的位置：它解决什么局部问题，依赖哪些前置，最后会影响哪类工程或研究判断。
    **前置**：先知道潜变量、时序预测、VLM/VLA 与训练效率；再看具体论文和工程案例。 必要时先回 [世界模型入口](index.md)、[基础知识](../foundations/index.md) 或 [术语表](../glossary/index.md)。
    **主线关系**：把表征预测、潜变量动力学、规划、rollout 服务和具身/视频应用连成“可模拟、可评估、可行动”的主线。

它的重点不是再解释“世界模型是什么”，而是回答一个更实际的问题：

**如果今天要开发一个面向机器人任务的世界模型，数据、模型、仿真和评测应该怎样接起来？**

![Autonomous vehicle sensors](../assets/images/thinking-exploration/world-models-embodied-ai/autonomous-vehicle-equipment.jpg){ width="920" }

<small>图源：[Wikimedia Commons: Equipment for self-driving car.jpg](https://commons.wikimedia.org/wiki/File:Equipment_for_self-driving_car.jpg)。这张照片适合作为世界模型的直觉入口：真实世界系统不是只看 RGB 图像，而是把相机、深度、位姿、传感器外参、运动规划和风险判定接成一条闭环。</small>

![Simulation approaches for video world models 原论文图](../assets/images/paper-deep-dives/world-models/towards-video-world-models/figure-9-simulation-approaches.jpg){ width="960" }

<small>图源：[Towards Video World Models](https://www.xunhuang.me/blogs/world_model.html)，Figure 9。真实项目里的世界模型往往要在 data、reconstruction、simulation、policy/evaluation 之间循环，而不是孤立训练一个预测器。</small>

!!! note "合并提示"
    **note · 图解：把这张图当成项目路线图读**

    左侧是数据和真实世界来源，中间是重建、生成和模拟方式，右侧才是策略、评测和交互使用。开发时不要从“选一个模型”开始，而要先问：数据字段是否稳定、相机/pose 是否可复算、仿真和 success checker 是否能给模型提供可验证的闭环。

    **note · 先把这页和总览区分开**

    [世界模型路线图](index.md) 讲方法谱系：RSSM、Dreamer、视频世界模型、WAM、数据引擎。本文更像项目路线图：数据字段怎么定，3D 重建 encoder 怎么接，仿真数据怎么生成，成功判定怎么写，最后怎样把模型测试闭环跑起来。

## 本页怎么读

这页按项目落地顺序组织：先定数据契约和几何坐标，再选 3D / VLA / Diffusion 路线，然后接仿真环境、success checker 和模型测试闭环。读的时候不要只记模型名，重点看每一步交付什么可验证产物：字段是否稳定、pose 是否可复算、仿真是否能自动判卷、闭环评测是否真的能暴露失败。

## 硬证据模块：开发路线要有阶段产物和门禁

开发路线页最怕变成愿景列表。每个阶段都应该交付一个能复算、能回放、能失败归因的产物。

| 阶段 | 最小硬证据 | 失败案例 | 验收指标 |
| --- | --- | --- | --- |
| 数据契约 | episode schema、字段版本、单位、坐标系、时间戳 | `cam_pose_list` / `cam_pos_list` 混用导致训练不可复现 | schema pass rate、missing field rate |
| 几何账本 | 内外参、depth、point map、object pose 可复算 | 外参漂移让抓取点偏 3cm | reprojection error、pose error |
| 动作接口 | action chunk 与观测时间轴对齐 | 动作延迟被模型误学成动力学 | timestamp skew、controller reject |
| 世界模型 | 固定 checkpoint 的 rollout 输出和失败 replay | 视频好看但候选动作排序错 | action sensitivity、risk ECE |
| 仿真/评测 | success checker、reset、seed、评测桶固定 | success checker 漏判，模型学错目标 | checker agreement、closed-loop success |

最小项目不必一开始做大，但必须能跑通：`数据样例 -> token/几何统计 -> 训练配置 -> rollout 输出 -> 评测结果 -> 失败回放`。这条链路越早存在，后续模型替换、量化、算子优化和数据回流才有共同验收口径。

## 世界模型项目的四层地基

这条开发路线有一个很强的工程信号：**世界模型不是先选一个大模型就开训，而是先把数据契约、几何坐标、仿真环境和评测口径定住。**

项目可以拆成四层。第一层是数据契约，必须固定相机内外参、机器人状态、目标物体或焊缝、动作、轨迹和 valid 标记，否则字段名、pose 坐标系和 GT 缺失会反复污染训练。第二层是几何表征，用 depth、point map、camera pose、object / seam 3D 表示建立空间账本。第三层是动作条件模型，回答候选动作之后未来观测、状态或轨迹如何变化。第四层是仿真与评测，把 Isaac Sim、RoboTwin 或 ManiSkill 里的任务生成、轨迹执行和 success checker 接成闭环。

一个最小闭环可以写成：

```text
相机标定和 3D 数据
  -> 几何/视觉 encoder
  -> 候选动作或轨迹
  -> 世界模型预测后果
  -> 仿真或几何 checker 判定
  -> 失败样本和 near-miss 回流
```

如果中间任一层没有数据契约，后面的模型会变成“会跑但无法解释”，调参也会很痛苦。

### 前置知识地图：先补齐这五块再谈大模型

前置知识看起来像零散关键词，其实正好对应世界模型的五个底座：相机标定负责把像素、深度和相机位姿放回同一坐标系；6D pose 负责让物体、工件、末端执行器和焊缝有可比较的空间状态；MoGe-2 / DUSt3R 负责在缺少真值深度时补 3D 先验；AR / AE 决定世界模型是在顺序 rollout，还是先把观测压到 latent；Transformer / Attention 则负责让多视图、多帧、多模态 token 对齐。它们的作用不是“先学理论再开工”，而是给 debug 提供坐标轴：当模型预测错了，你能判断错在相机、pose、几何、时序建模，还是动作条件没有进入模型。

## 数据准备：先把 pose、相机和任务对象讲清楚

小雨数据问题很典型：代码需要读取 `cam_pos_list` 和 `seam_pose`，但旧数据里出现 `cam_pose_list` 命名差异且缺少 `seam_pose`，导致模型无法运行。这类问题不是小 bug，而是世界模型项目的根问题：**状态定义不稳定，模型就不知道自己在预测什么。**

建议把每个 episode 的数据契约写成固定字段：

| 字段 | 形状 / 类型 | 含义 | 检查点 |
| --- | --- | --- | --- |
| `cam_pos_list` | `list(tuple(3))` | 相机位置，外参的一部分 | 和 `cam_quat_list` 同步 |
| `cam_quat_list` | `list(tuple(4))` | 相机姿态四元数 | 四元数归一化 |
| `cam_intrinsic` | `(3, 3)` | 相机内参 | 焦距、主点、分辨率匹配 |
| `jointstates` | `(N, 6)` 或具体机器人维度 | 机器人关节状态 | 控制频率对齐 |
| `robot_pose` | `(7,)` | 机器人基座或末端位姿 | 坐标系写清楚 |
| `gongjian_pose` | `(7,)` | 工件位姿 | 与 world frame 对齐 |
| `seam_line_local` | `(20, 3)` | 焊缝局部点 | 与工件坐标系绑定 |
| `seam_line_global` | `(20, 3)` | 焊缝世界坐标点 | 可由 local + 工件 pose 复算 |
| `seam_pose` | `(20, 7)` | 焊缝点位姿 | 模型/评测都要用 |
| `trajectory` | `list(shape(N, 6))` | 轨迹或候选相机路径 | 和 action 对齐 |
| `actions` | `list(shape(K, 7))` | 机械臂或相机动作 | 控制模式写清楚 |
| `valid` | `(M,)` | episode 或轨迹是否可用 | 失败原因要另存 |

!!! tip "实操建议：不要只存成功数据"
    `valid=False` 的样本也有价值，但要记录原因：字段缺失、planner failed、遮挡不足、相机不可达、碰撞、轨迹执行失败、success checker 失败。世界模型后续做数据引擎时，最需要的往往就是这些失败边界。

## 三条模型路线：3D、2D/VLA、Diffusion

主流路线可以先分成三类，这个划分很适合作为选型地图。

| 路线 | 代表技术 | 输入重点 | 输出重点 | 适合什么任务 |
| --- | --- | --- | --- | --- |
| 3D 场景理解 | Depth Anything 3、MapAnything、VGGT、DUSt3R / MoGe 系列 | 多视图图像、相机、深度、点云 | depth、camera、ray map、point map、3D state | 需要可见性、遮挡、尺度和几何可达性的任务 |
| 2D 场景理解 / VLA | π0、π0.5、OpenVLA、GR00T N1、Helix | 图像、语言、本体状态 | 动作 token、动作块、子任务 | 语言到动作、跨场景泛化、桌面操作 |
| Diffusion policy | Diffusion Policy、RDT、DP3 | 视觉条件、机器人状态、噪声动作 | 连续动作序列 | 平滑动作、模仿学习、灵巧操作 |

复现顺序可以更朴素一点：先在 Depth Anything 3、MapAnything、VGGT、VO-DP 这类 3D / 视觉几何模型上验证相机、depth、pose 是否能稳定输出可用几何表示；再用 π0、π0.5、OpenVLA、GR00T N1 检查语言、视觉、本体状态能否输出可执行动作块；随后用 Diffusion Policy、DP3、RDT 看连续动作序列是否平滑、可执行、能跨对象泛化；最后接 LIBERO、DROID、RoboTwin，确认数据格式、任务定义和 success checker 能支持闭环比较。

### 3D 路线：先让模型拥有空间账本

3D 路线的价值在于：很多机器人任务不是“识别物体叫什么”，而是“从哪里看、从哪里伸手、是否被挡住、会不会碰”。Depth Anything 3、MapAnything、VGGT 一类模型值得关注，正是因为它们能把相机、深度和场景几何拉回同一张账本。

[Depth Anything 3](../paper-deep-dives/embodied-ai/depth-anything-3.md) 从任意数量图像和可选 camera pose 中预测一致的 depth / ray，再恢复相机、点云和 3DGS。[MapAnything](../paper-deep-dives/embodied-ai/mapanything.md) 的目标是统一多种 3D 视觉任务：输入一张或多张图像，以及可选的内参、pose、depth 或局部重建，直接回归 metric 3D scene geometry 和 cameras。[VGGT](../paper-deep-dives/embodied-ai/vggt.md) 则把 camera parameters、point maps、depth maps、3D point tracks 放进一个 feed-forward Transformer 里，强调快速直接的几何推断。

对机器人世界模型来说，这类 encoder 不一定直接输出动作，但它能提供空间账本：目标是否在视野里、焊缝是否被看全，目标距离和末端执行器移动量是否匹配，当前视角看不到的区域是否需要换视角，候选相机位姿或机械臂轨迹是否可执行，以及物体高度、姿态、接触关系是否进入任务判定。

### 2D / VLA 路线：让语义和动作先跑起来

2D / VLA 路线更像 policy：给定图像、语言和状态，输出下一段动作。它适合先做桌面操作、移动操作和开放指令泛化，但如果没有世界模型或几何 checker，容易出现“看起来会做，实际碰撞或失败”的问题。

[π0.5](https://www.pi.website/blog/pi05) 的关键启发是异构数据协同训练：web 多模态数据、机器人动作数据、跨 embodiment 数据、高层子任务标签和 verbal instruction 都可以进同一个 VLA 配方。它推理时先生成高层子任务，再用 flow matching action expert 输出低层连续动作块。

### Diffusion 路线：把动作当成一段可采样轨迹

Diffusion Policy 和 RDT 一类方法适合连续动作生成。相比一步动作，动作块能更平滑，也更接近“技能片段”。很多 VLA 工作也强调 action chunk 或未来轨迹，而不是单步 action。

动作扩散模型的关键检查点是：

1. 控制频率是否和数据一致；
2. 预测 horizon 是否匹配任务节奏；
3. 动作归一化是否按单数据集分位数或统一尺度处理；
4. 输出动作是否经过轨迹优化、碰撞过滤和控制器投影；
5. 失败时是否能重规划，而不是一段动作执行到黑。

## 3D 重建 Encoder：数据预处理比模型名更重要

MapAnything 一类 3D 数据转换流程通常会拆成多步，这对任何 3D 世界模型都很关键：

| 步骤 | 做什么 | 为什么重要 |
| --- | --- | --- |
| conversion | 将不同来源数据统一成 WAI 等内部格式 | 后续训练不能每个数据集写一套 loader |
| undistortion | 去除相机畸变 | depth、pose、ray map 才能对齐 |
| rendering | 用 `nvdiffrast` 等工具渲染几何监督 | 给模型提供可控的监督信号 |
| covisibility | 计算共视性 | 多视图训练要知道哪些像素互相看见同一结构 |
| MoGe / monocular depth | 单目深度估计 | 给缺少真深度的数据补几何先验 |
| MVSAnywhere / multiview depth | 多视图深度估计 | 提高跨视角几何一致性 |
| depth consistency confidence | 深度一致性置信度 | 训练时降低错误监督权重 |

MapAnything 这类数据池的组织方式不是只拿一个数据集跑通，而是把合成、室内、室外、物体和机器人仿真数据分层纳入。比如 Aria Synthetic、BlendedMVS、DL3DV-10K、Dynamic Replica、ETH3D、Mapillary、MegaDepth、MVS-Synth、Parallel Domain 4D、ScanNet++ v2、TartanAirV2、Objaverse、HyperSim、Kubric、Omnidata、Habitat 等。真正重要的不是把名字背下来，而是记住这件事：**3D encoder 要吃多源数据，必须同时管理格式状态、下载完整性、相机模型和可用监督类型。**

![Velodyne LiDAR sensor](../assets/images/thinking-exploration/world-models-embodied-ai/velodyne-lidar.jpg){ width="760" }

<small>图源：[Wikimedia Commons: Velodyne High-Def LIDAR.jpg](https://commons.wikimedia.org/wiki/File:Velodyne_High-Def_LIDAR.jpg)。LiDAR、depth、相机 pose、point map 都是在把“图像里有什么”改写成“空间里在哪里”。世界模型要服务机器人决策，就必须把这层空间账本做实。</small>

可以把数据池按监督强度分成三层：

| 数据层 | 典型来源 | 能提供什么 | 主要风险 |
| --- | --- | --- | --- |
| 强几何监督 | 合成数据、仿真数据、带 GT pose/depth 的重建数据 | depth、camera、mask、3D point、object pose | 域差距、材质过于理想、物理不真实 |
| 弱几何监督 | 多视图真实图像、SfM / MVS 可重建数据 | 相机相对位姿、稀疏/半稠密几何 | 失败重建、尺度不稳、动态物体污染 |
| 补全监督 | 单目深度、MoGe / MVSAnywhere 推断、consistency confidence | 覆盖更大真实图像池 | 伪标签偏差会被模型放大 |

这条线最容易被低估。很多项目会先把模型训练脚本跑通，然后才发现数据中的尺度、外参、深度和轨迹不一致。更稳的顺序是：

```text
先做 100 条样本的数据可视化
  -> 再做 10 条 episode 的几何复算
  -> 再跑一个很小的 overfit 训练
  -> 最后才扩大数据集
```

如果一个 3D encoder 连单场景 overfit 都对不齐，大规模训练只会把错误平均得更隐蔽。

## 仿真环境：RoboTwin、Isaac Sim、ManiSkill 各司其职

仿真环境可以压成一句话：仿真不是“有个模拟器”，而是**任务生成、轨迹生成、观测渲染、成功判定和数据保存的流水线**。

| 工具 / 平台 | 在路线中的角色 | 适合关注 |
| --- | --- | --- |
| RoboTwin | 双臂任务、资产标注、专家轨迹、success checker 的组织范式 | 任务模板、功能点、生成式数字孪生、自动判定 |
| Isaac Sim / Isaac Lab | 高保真渲染、USD 资产、传感器、物理仿真、远程 headless 部署 | 相机、depth、RTX Lidar、ROS、合成数据 |
| ManiSkill | 机器人操作 benchmark 和可复现实验环境 | 快速训练、RL / imitation baseline、任务评测 |
| cuRobo / motion planner | 从目标位姿生成可执行关节轨迹 | IK、碰撞、平滑、可达性 |

[Isaac Sim 5.1 文档](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_container.html) 明确把容器安装定位到远程 headless server 或 cloud 场景；这和常见 Docker + WebRTC Streaming Client 部署方式一致。[RoboTwin 2.0 文档](https://robotwin-platform.github.io/doc/) 则更适合理解任务、资产和数据生成管线。

如果把它们放进具体开发阶段，RoboTwin 适合快速搭任务模板和自动评测，Isaac Sim / Isaac Lab 适合视觉数据生成、传感器仿真和部署前验证，ManiSkill 适合快速试 baseline 与 RL / imitation 训练，XiaoYu 焊缝任务适合作为业务闭环验证，Genie Sim / 外部世界模拟则可以作为开放世界交互趋势参考，但不能替代机器人 success rate。

其中 XiaoYu 相关记录特别具体：仿真数据生成偏向 Isaac Sim 5.1，评测可能在 Isaac Sim 4.2 + IsaacLab 1.4.1，或 Isaac Sim 5.1 + IsaacLab main 的 headless 设置里跑。这个细节很朴素，但很关键：训练环境、评测环境和部署环境版本不一致时，资产、物理和传感器差异都可能变成模型误差。

### RoboTwin 生成式数字孪生资产流程

RoboTwin 1.0 的资产生成流程可以理解为一条从单张图到可操作资产的流水线：先输入单张 RGB 参考图，用 GPT-4V 描述物体类别、形状、部件和视觉属性，再让 LLM 扩展同类物体的多样化描述；随后用 SDXL-Turbo 生成不同视角或外观的 2D 图像，用 Rodin 合成几何、法线、线框和纹理；接着用 GPT-4V 辅助判断材质、质量、摩擦并做约 5% 范围随机化，完成 convex decomposition、mesh merge 和 collision 处理；最后标注 functional point / axis，并用图像特征匹配迁移标注，让资产拥有可用于任务和 checker 的操作语义。

这条流程的启发是：世界模型的数据不是“下载 3D 模型就完事”。一个资产必须同时有视觉外观、物理属性、碰撞体、功能点和任务语义，才能进入轨迹生成和 success checker。

## Success Checker：自动判卷器也会“作弊”

RoboTwin 的评测规则很有价值，因为它提醒我们：很多 benchmark 只检查最后状态，可能漏掉中间过程。

常见 success checker 信号包括：

| 信号 | 示例 | 风险 |
| --- | --- | --- |
| 世界坐标 | 物体高度超过阈值、放置到目标平面附近 | 坐标系错误会直接误判 |
| 三维距离 | 接触点和功能点距离小于阈值 | 只看距离可能忽略姿态 |
| 四元数 / 姿态 | 物体方向是否接近目标方向 | 欧拉角阈值容易有边界问题 |
| 接触状态 | 锤子与方块发生 contact | 接触存在不等于用力正确 |
| 夹爪状态 | 任务结束时夹爪打开或闭合 | 夹爪状态不能完全代表抓住物体 |
| 关节状态 | 抽屉、笔记本、开关是否打开到阈值 | 中间是否碰撞可能没记录 |

几个例子很适合做评测设计模板：锤子敲方块不能只看接触，还要看锤子功能点和方块功能点是否对齐、接触方向和碰撞次数是否合理；方块排序要看平面位置、顺序、夹爪释放、中间遮挡和误碰撞；倒垃圾要看小桶高度、垃圾球是否进入目标区间，以及垃圾是否飞出、桶是否倾倒过度；传递麦克风要看接触、交接手闭合、原始手松开、高度、左右侧和双手同步窗口；打开笔记本要看关节打开比例、末端是否仍在铰链附近、是否过力或回弹；扫描物体则要看扫描轴线、距离区间、双手夹紧、可见性和扫描持续时间。

把这些任务再归类，可以得到一张更容易扩展的判定地图：

| 任务组 | 代表任务 | 判定核心 |
| --- | --- | --- |
| 点击 / 开关 | click alarm clock、click bell、turn switch | 正确手臂、按钮/开关接触、关节或状态触发 |
| 抬升 / 倾倒 | lift roller、dump trash、shake bottle | 高度阈值、姿态区间、目标物是否进入容器 |
| 传递 / 协作 | pass cube、pass microphone | 双臂接触窗口、交接手闭合、原手松开、左右侧约束 |
| 悬挂 / 放置 | hang cup、place into basket、place into drawer | 功能点接近、容器范围、夹爪释放、最终稳定 |
| 移动 / 整理 | move can、move medicine、move cards、move stapler | 目标区域、排序关系、干扰物避让 |
| 开合机构 | open laptop、open microwave、open drawer | 关节打开比例、末端轨迹、是否过冲或回弹 |
| 工具 / 扫描 | hammer、scan object、stamp、stack | 功能点对齐、接触方向、持续时间、姿态稳定 |

!!! warning "只看最终状态是不够的"
    “把物体放到抽屉中”如果只检查物体最终在抽屉范围内，可能漏掉抽屉没拉开、物体穿模、机械臂碰撞、夹爪强行挤进去等问题。世界模型评测最好同时记录过程指标：碰撞、接触、速度、遮挡、动作平滑度、风险触发和人工接管。

## 以小雨焊缝任务为例：世界模型到底预测什么

以小雨焊缝观察任务为例，可以把目标写得很具体：用仿真生成相机观测，让模型预测一段相机/机械臂动作轨迹，再用几何/遮挡判断这段轨迹能否把焊缝看全。

这可以整理成一个清晰任务：

```text
输入：
  多视角图像 / depth
  相机内参和外参
  当前机器人关节状态
  工件 pose
  焊缝 seam_line / seam_pose

模型：
  3D encoder 编码场景和焊缝
  trajectory / diffusion head 生成候选相机或机械臂轨迹
  world model / checker 预测可见性、遮挡、碰撞和覆盖率

输出：
  一段相机位姿或机器人动作轨迹
  每个焊缝点的可见性预测
  覆盖率、风险、valid 标记
```

可以先不急着做全生成式视频世界模型，而是从更可控的几何世界模型开始：

\[
\hat{v}_{i,t}, \hat{c}_{t}, \hat{r}_{t} = W_\theta(z_t, a_{t:t+H})
\]

<!-- [补充] -->
| 符号 | 含义 | 取值范围 |
| --- | --- | --- |
| \(\hat{v}_{i,t}\) | value、速度场或向量量 | 实数标量/向量，依上下文而定 |
| \(\hat{c}_{t}\) | 公式中的变量或中间量 | 按上下文定义 |
| \(\hat{r}_{t}\) | 奖励、回报或时间索引相关量 | 实数标量或上下文定义的索引范围 |
| \(\theta\) | 模型或策略参数 | 实值张量，形状由模型结构决定 |
| \(i\) | 时间步、索引或计数 | 整数索引或归一化时间区间，见正文 |
| \(t\) | 时间步、索引或计数 | 整数索引或归一化时间区间，见正文 |


其中 \(\hat{v}_{i,t}\) 表示第 \(i\) 个焊缝点在未来视角下是否可见，\(\hat{c}_{t}\) 表示碰撞或不可达风险，\(\hat{r}_{t}\) 表示轨迹质量或覆盖收益。

这样做的好处是验收明确：是否看全焊缝，是否避免遮挡，轨迹是否可执行，相机是否保持合适距离和角度，以及失败时能否指出是哪段轨迹导致不可见或不可达。

## 评测：不要让离线 loss 冒充闭环能力

世界模型测试至少要分四层：

| 层级 | 指标 | 说明 |
| --- | --- | --- |
| 数据层 | 字段完整率、pose 复算误差、相机标定误差 | 先确认数据可信 |
| 几何层 | depth / point map 误差、覆盖率、可见性 F1 | 检查空间理解 |
| 轨迹层 | 碰撞率、可达率、平滑度、轨迹执行成功率 | 检查动作可执行 |
| 闭环层 | 任务成功率、near-miss、恢复成功率、人工接管率 | 检查系统是否真的有用 |

如果是视频世界模型，还要额外看动作敏感性、长时一致性、因果 rollout 和决策收益：同一状态下换动作，未来是否合理分叉；绕回同一场景时物体是否还在；自回归生成是否越滚越偏；接入 planner 或 policy 后成功率是否真的提高。

[Genie 3](https://deepmind.google/models/genie/) 和 [NVIDIA Cosmos](https://www.nvidia.com/en-us/ai/cosmos/) 展示了外部世界模拟的方向：实时交互、环境一致性、视觉质量和物理 AI 数据生成。但机器人任务里还要回到动作可执行、风险、成功率和真实闭环，不然世界模型容易停在“漂亮模拟器”阶段。

## 一个建议的开发顺序

结合上面的路线，更稳的开发顺序是先冻结数据契约，把字段、shape、坐标系、单位和控制频率全部写进 schema；再做几何可视化，让相机、工件、焊缝、轨迹和点云出现在同一坐标系中；随后实现覆盖率、碰撞、可达性、距离阈值这类最小 checker；接着在小数据上验证 3D encoder 的 depth / point / pose 是否能学；轨迹生成先用规则或 planner 做 baseline，再训练模型模仿；世界模型预测头优先预测可见性、风险和覆盖收益，而不是一上来生成视频；最后把模型输出放回仿真执行，记录失败原因，并把高误差、高遮挡、near-miss 和不可达样本回流训练。

!!! example "把它想成拍电影前的分镜"
    世界模型不是直接喊“生成一部电影”。你要先知道摄影机在哪、演员在哪、道具在哪、哪条轨迹能拍到关键动作、哪段会被挡住。机器人世界模型也是这样：先把几何和动作账本做实，再谈大规模生成。

## 参考资料

- [MapAnything: Universal Feed-Forward Metric 3D Reconstruction](../paper-deep-dives/embodied-ai/mapanything.md)：3D encoder 和统一几何建模路线。
- [Depth Anything 3: Recovering the Visual Space from Any Views](https://arxiv.org/abs/2511.10647)：从任意视角恢复一致 depth、ray、camera 和 3D geometry；站内专题见 [Depth Anything 3 讲解](../paper-deep-dives/embodied-ai/depth-anything-3.md)。
- [VGGT: Visual Geometry Grounded Transformer](../paper-deep-dives/embodied-ai/vggt.md)：feed-forward 3D 属性预测。
- [π0.5: a VLA with Open-World Generalization](https://www.pi.website/blog/pi05)：异构数据协同训练、高层子任务和 flow matching 动作专家。
- [Isaac Sim 5.1 Container Installation](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_container.html)：远程 headless / Docker 部署口径。
- [RoboTwin 2.0 官方文档](https://robotwin-platform.github.io/doc/)：双臂任务、资产、轨迹和评测组织方式。
- [Genie 3](https://deepmind.google/models/genie/) 和 [NVIDIA Cosmos](https://www.nvidia.com/en-us/ai/cosmos/)：世界模型走向可交互模拟和物理 AI 数据生成的代表产品路线。

!!! tip "下一站"
    - 回到本专题入口：[世界模型](index.md)，确认这页在整条路线中的位置。
    - 按导航顺序继续：[模拟器、反事实与合成 Rollout](simulators-counterfactuals-and-synthetic-rollouts.md)。
    - 概念或符号卡住时，先查 [术语表](../glossary/index.md)，再回到当前页。
