# 一个任务跑通具身闭环

这一页用一个具体任务把具身智能串起来。前面的页面讲了闭环、符号、VLA、WAM、世界模型和系统边界；这里把它们放进同一条 episode。


主例子：

```text
将圆柱体放入罐口尺寸接近的罐子
```

这个任务看起来像普通 pick-and-place，但它会同时考察尺寸感知、目标绑定、抓取、路径规划、动作执行、成功判定和失败恢复。

![π0.5 cleaning a new kitchen 原论文图](../assets/images/paper-deep-dives/embodied-ai/pi05/figure-2-cleaning-kitchen.png){ width="920" }

<small>图源：[π0.5](../paper-deep-dives/embodied-ai/pi05.md)，Figure 2。原论文图意：展示机器人在未见过的新厨房中执行多个清理子任务，例如关柜门、把物品放入抽屉、擦拭液体和把餐具放进水槽。</small>

!!! note "图解：长任务要按阶段闭环，而不是一次性输出完整答案"
    图里的每个小任务都可以拆成“看见、选择、抓取、移动、放置、判定”。长任务失败通常不是最后一步才发生，而是在任务绑定、对象选择、抓取姿态、轨迹规划或执行反馈中的某一环开始偏离。walkthrough 的目的，就是把每一阶段的输入、输出和 checker 写清楚。

## 先统一符号

这条任务里会反复用到同一组符号：

在线闭环可以写成：

\[
(o_{\le t}, q_t, \ell)
\rightarrow
\text{state / subtask}
\rightarrow
A_{t:t+H-1}
\rightarrow
\text{safety + controller}
\rightarrow
o_{t+1}
\rightarrow
\text{checker}.
\]


白话读法：先看见并估计状态，再决定下一小段动作，安全过滤后执行，拿到新观测，再由 checker 判断进度。

## 不要先选模型，先定义任务状态

如果状态定义不清，VLA、WAM、planner 和 checker 都会混乱。一个最小状态可以写成：

```json
{
  "objects": [
    {
      "id": "cylinder_0",
      "category": "cylinder",
      "size_m": [0.035, 0.035, 0.08],
      "pose_w": [0.42, -0.10, 0.04, 1, 0, 0, 0],
      "grasp_candidates": ["top_grasp_0", "side_grasp_1"]
    },
    {
      "id": "can_0",
      "category": "container",
      "opening_diameter_m": 0.04,
      "pose_w": [0.30, 0.18, 0.05, 1, 0, 0, 0]
    }
  ],
  "robot": {
    "joint_pos": [],
    "gripper_open": true,
    "eef_pose_w": [0.20, -0.20, 0.30, 1, 0, 0, 0]
  },
  "task": {
    "instruction": "put the cylinder into the closest matching container",
    "target_object": "cylinder_0",
    "target_container": null,
    "phase": "select_container"
  }
}
```

这里有几个重点：

- `pose_w` 表示物体在 world frame 下的位置和朝向；
- `size_m` 和 `opening_diameter_m` 是尺寸任务的关键状态；
- `target_container` 可以先为空，由高层模块根据尺寸匹配算出来；
- `grasp_candidates` 来自手动标注、自动抓取生成或离线筛选；
- `phase` 让 checker 和策略知道任务走到哪一步。

!!! warning "常见错误：把任务状态藏在自然语言里"
    “放进尺寸接近的罐子”听起来很简单，但系统必须知道圆柱直径、罐口直径、物体坐标、抓取候选和当前阶段。若这些状态没有被记录，模型失败后很难判断是看错、选错、抓错还是放错。

## 把任务拆成七步

| 步骤 | 输入 | 模块 | 输出 | checker 关心什么 |
| --- | --- | --- | --- | --- |
| 1. 看见 | \(o_t, q_t\) | 感知、深度、pose | 圆柱和罐子的状态 | 物体是否被正确检测和定位 |
| 2. 绑定目标 | 状态 + \(\ell\) | 规则 / VLM / planner | `target_container` | 尺寸是否匹配，目标是否唯一 |
| 3. 选抓取 | 物体状态 | grasp sampler / affordance | grasp pose | 抓取是否可达、是否稳定 |
| 4. 生成动作 | \(o_{\le t}, q_t, \ell\) | VLA / planner | \(A_{t:t+H-1}\) 或轨迹 | 动作是否朝目标推进 |
| 5. 预测风险 | \(z_t, A_{t:t+H-1}\) | world model / risk head | \(\hat r, \hat c, \hat d\) | 是否可能碰撞、滑落、放错 |
| 6. 执行控制 | action + constraints | controller / safety | robot command | 是否越界、过快、碰撞 |
| 7. 判定回流 | 新观测 + 状态 | success checker | success / failure reason | 是否完成，失败属于哪一类 |

这张表是整页的骨架。任何路线都要能填这几列，只是“模块”不同。

## 第一步：看见和估计状态

输入是：

\[
o_t = \{\text{RGB views}, \text{depth}, \text{robot proprioception}\}.
\]


系统要从 \(o_t\) 中估计：

```text
cylinder pose
cylinder size
container poses
container opening diameters
robot eef pose
gripper state
```

| 输出字段 | 为什么重要 | 失败症状 |
| --- | --- | --- |
| 物体 6D pose | planner 需要知道抓取点和放置点 | 抓取点偏到物体旁边 |
| 尺寸和 scale | 任务要求匹配罐口尺寸 | 放进过大或过小的罐子 |
| 深度 / 点云 | 判断距离、遮挡和碰撞 | 夹爪高度错、撞桌面 |
| 机器人状态 \(q_t\) | 同一画面下手臂姿态可能不同 | 模型输出不可达动作 |

checker 在这一步不判断最终成功，而是判断状态是否足够可信。例如深度缺失、目标被遮挡、罐口尺寸不确定时，可以进入重新观察或请求调整视角，而不是硬执行。

## 第二步：绑定目标和任务阶段

语言指令是：

\[
\ell = \text{``put the cylinder into the closest matching container''}.
\]


目标绑定可以先用规则写清楚：

```python
def choose_container(cylinder, containers):
    radius = cylinder.size_m[0] / 2
    feasible = [
        can for can in containers
        if can.opening_diameter_m > cylinder.size_m[0] + 0.005
    ]
    return min(
        feasible,
        key=lambda can: can.opening_diameter_m - cylinder.size_m[0],
        default=None,
    )
```

这段代码不是最终实现，而是说明“尺寸匹配”必须落到可检查的状态上。

| 可能输出 | 系统动作 |
| --- | --- |
| 找到唯一合适罐子 | 进入抓取阶段 |
| 多个罐子都合适 | 选择 margin 最小或请求澄清 |
| 没有合适罐子 | 拒绝执行或报告不可完成 |
| 尺寸估计不确定 | 重新观察或换视角 |

这里对应 \(r_t\)、\(c_t\)、\(d_t\) 的早期版本：选对目标提高 progress，尺寸不确定提高 risk，无可行目标触发 done / invalid。

## 第三步：生成动作有三种路线

### 6.1 经典 pipeline

```text
检测圆柱和罐子
-> 估计尺寸和 6D pose
-> 规则选择最匹配罐子
-> 抓取标注 + cuRobo 规划
-> 控制器执行
-> success checker 判定
```

| 输入 | 输出 | 优点 | 风险 |
| --- | --- | --- | --- |
| 结构化状态、抓取候选、障碍物 | 完整或分段轨迹 | 可解释、容易 debug、容易加安全约束 | 开放语言和新物体扩展慢 |

适合工业、高精度或任务规则清楚的场景。

### 6.2 π0.5 式 VLA

\[
A_{t:t+H-1} \sim \pi_\theta(o_{\le t}, q_t, \ell).
\]


这行式子在说：策略 \(\pi_\theta\) 根据到当前为止的观测 \(o_{\le t}\)、机器人状态 \(q_t\) 和语言目标 \(\ell\)，采样或输出未来 \(H\) 步动作块 \(A_{t:t+H-1}\)。这里的 \(H\) 是动作块长度，不是任务总长度。

```text
多视角图像 + 机器人状态 + 语言任务
-> high-level subtask: pick up the cylinder
-> action expert 输出短动作块
-> 执行并重新观测
-> high-level subtask: place into the matching container
```

| 输入 | 输出 | 优点 | 风险 |
| --- | --- | --- | --- |
| 图像、状态、语言、历史 | high-level subtask + action chunk | 适合开放任务和长任务 | 仍需 checker、控制器、安全过滤 |

π0.5 式路线把任务分解和动作生成放进模型里，但不意味着模型可以绕过 success checker。

### 6.3 DreamZero 式 WAM

\[
(o_{t+1:t+H}, A_{t:t+H-1})
\sim
p_\theta(o, A \mid o_{\le t}, q_t, \ell).
\]


这行式子在说：WAM 同时预测未来观测片段 \(o_{t+1:t+H}\) 和动作块 \(A_{t:t+H-1}\)。条件里的 \(o_{\le t}\)、\(q_t\)、\(\ell\) 分别表示历史视觉上下文、当前机器人状态和任务语言，\(p_\theta\) 是动作-视频联合生成模型。

```text
当前视觉上下文 + 语言 + 机器人状态
-> 生成未来视频片段和未来动作
-> 执行动作 chunk
-> 用真实观测刷新上下文
-> 继续下一段 video-action prediction
```

| 输入 | 输出 | 优点 | 风险 |
| --- | --- | --- | --- |
| 视觉历史、状态、语言 | 未来视频 + 动作 chunk | 动作和未来视觉互相约束 | 未来视频错时，动作也可能跟错 |

DreamZero 式路线尤其要把 predicted future、issued action、real observation 和 checker 结果并排看。

![DreamZero seen task evaluation 原论文图](../assets/images/paper-deep-dives/world-models/dreamzero/figure-8-seen-task.png){ width="920" }

<small>图源：[DreamZero](../paper-deep-dives/embodied-ai/dreamzero.md)，Figure 8。原论文图意：展示 World-Action Model 在机器人任务上的执行与评测样例，用来观察未来视觉和动作输出是否能转化为真实任务进展。</small>

!!! note "图解：闭环走读时要把预测和真实结果对上"
    WAM 的图不能只看“未来画面像不像”，还要看预测动作执行后真实观测是否朝同一方向变化。实际排查时，把模型预测的未来、下发动作、真实新观测和 success checker 结果并排看，最容易定位失败来自视频预测、动作接口还是执行控制。

## 第四步：安全过滤和控制执行

无论动作来自 VLA、WAM 还是 planner，都应该经过执行接口：

```text
action chunk
-> action adapter
-> collision / workspace / speed / force checks
-> controller
-> robot command
```

对这个任务，安全层至少检查：

| 检查项 | 例子 |
| --- | --- |
| 工作空间 | 末端是否越过桌面边界或机器人可达范围 |
| 碰撞 | 轨迹是否会撞罐子、桌面、其他物体 |
| 速度 / 加速度 | 抓取或放置是否太快 |
| 夹爪状态 | 抓取前是否打开，移动前是否夹稳 |
| 力 / 接触 | 插入罐口时是否卡住或用力过大 |
| 中断条件 | 物体滑落、人手进入、目标消失 |

这里的关键是：模型动作不是最终命令。controller 把动作变成高频控制，safety filter 决定是否执行、缩放、重规划或中止。

## 第五步：success checker 比模型更早写

很多项目会先生成资产和轨迹，最后才写任务判定。这个顺序容易返工。更稳的做法是先写 success checker 草案，因为它定义了“什么叫成功”。

对“把圆柱体放入尺寸匹配的罐子”，checker 至少要判断：

1. 圆柱在目标罐子内部；
2. 目标罐子确实是尺寸匹配的罐子；
3. 夹爪已经打开并离开物体；
4. 圆柱稳定，没有继续滑动或弹出；
5. 过程中没有碰撞、越界或人工接管；
6. 若没有合适罐子，系统应拒绝而不是乱放。

伪代码可以写成：

```python
def check_container_success(state, target_object, target_container):
    obj = state.objects[target_object]
    can = state.objects[target_container]

    inside = is_inside_opening(obj.pose_w, can.pose_w, can.opening_diameter_m)
    size_match = can.opening_diameter_m >= obj.size_m[0] + 0.005
    stable = obj.linear_speed < 0.01 and obj.angular_speed < 0.02
    released = state.robot.gripper_open and distance(state.robot.eef_pose_w, obj.pose_w) > 0.05
    safe = not state.events.collision and not state.events.human_takeover

    return inside and size_match and stable and released and safe
```

这段代码不是最终实现，而是告诉你：任务定义要落到可查询状态和阈值上。否则模型、planner、数据集之间没有共同标准。

## 失败时按责任层排查

具身任务失败后，不要先说“模型不行”。先按责任层排查：

| 失败现象 | 先查什么 | 可能原因 |
| --- | --- | --- |
| 目标物体识别错 | 感知和数据 | 纹理太相似、遮挡、单目深度错误、标签错 |
| 尺寸比较错 | 状态估计和任务绑定 | scale 错、深度错、尺寸字段缺失、规则阈值不合理 |
| 抓取点很奇怪 | grasp annotation / affordance | 抓取轴随机、候选没有任务语义、物体 origin 错 |
| 轨迹规划失败 | planner 和 collision mesh | IK 无解、障碍物太近、碰撞网格过粗或过碎 |
| 仿真执行掉落 | 物理属性和控制 | 摩擦/质量不合理、夹爪力不够、接触模型不稳 |
| 看似成功但判失败 | success checker | 阈值太严、坐标系不一致、夹爪状态读取错 |
| VLA 会短动作但长任务崩 | 高层子任务和记忆 | 没有 task progress、没有状态跟踪、没有失败恢复 |
| WAM 生成视频合理但执行失败 | video-action alignment 和安全层 | 视觉未来缺少物理可执行性，动作没有被控制层约束 |

这张表是具身 debug 的核心。真实系统里大多数问题不是单一模型替换能解决的，而是某个接口没有定义或验证清楚。

## 训练数据应该保存哪些字段

一条 episode 至少建议保存：

| 字段 | 用途 |
| --- | --- |
| `instruction` | 训练语言到动作或语言到子任务 |
| `camera_frames` | 训练 VLA、WAM、视觉感知 |
| `depth_or_pose` | debug 感知和 planner，也可作为 privileged state |
| `robot_state` | 关节角、夹爪、底盘、本体状态 |
| `actions` | 行为克隆、WAM action latent、policy training |
| `subtasks` | π0.5 式 high-level supervision |
| `object_states` | success checker、状态估计、错误分析 |
| `success` / `task_progress` | 过滤数据、训练 reward/judge、评测 |
| `risk` / `cost` | 训练 risk head 和 safety gate |
| `done` / `termination` | 学会停止、失败或拒绝 |
| `failure_reason` | 后续回流和 hard case mining |
| `asset_metadata` | 追踪 scale、质量、材质、碰撞网格、抓取标注版本 |

如果未来想接 DreamZero 这类 WAM，尤其要保证视频、动作和机器人状态严格时间同步。WAM 学的是 video-action 对齐，时间戳漂移会直接破坏监督信号。

## clean 到 random 的升级顺序

不要一开始就上复杂随机场景。更稳的升级顺序：

| 阶段 | 场景设置 | 目标 |
| --- | --- | --- |
| clean-0 | 单圆柱、单罐子、固定相机 | 验证资产、scale、抓取、planner 和 checker |
| clean-1 | 单圆柱、多初始位姿 | 验证 grasp 和 planner 是否泛化 |
| clean-2 | 多个不同尺寸罐子 | 验证尺寸比较和目标绑定 |
| random-1 | 加少量非目标干扰物 | 验证检测、避障和任务筛选 |
| random-2 | 加纹理、光照、位置随机化 | 验证视觉鲁棒性 |
| random-3 | 加失败复位、重试和 hard cases | 生成更接近训练/评测需要的数据 |

这样做可以保证每一步失败都能定位。如果一开始就 random，很容易出现“看起来哪里都错，但不知道先修什么”。

## 和论文路线的关系

| 论文路线 | 这个 walkthrough 对应的位置 |
| --- | --- |
| [π0.5](../paper-deep-dives/embodied-ai/pi05.md) | 把高层子任务和低层 action chunk 串进在线闭环 |
| [DreamZero](../paper-deep-dives/embodied-ai/dreamzero.md) | 把 episode 里的视频、动作和状态变成 WAM 训练信号 |
| [LingBot-World](../paper-deep-dives/world-models/lingbot-world.md) | 把动作条件未来观测用于交互模拟和反事实数据 |
| [Dreamer](../paper-deep-dives/world-models/dreamer.md) | 把状态、奖励和 continue 信号用于 latent imagination |
| [RoboTwin](https://robotwin-platform.github.io/doc/tasks/index.html) | 提供任务组织、资产标注和仿真数据生成的参考 |

最终你应该能把任何具身论文放回这条链路：它到底在改感知、子任务、动作、世界预测、规划控制、数据生成，还是评测回流。

## 最小验收

读完这一页后，你应该能为一个新任务写出：

1. 任务状态 schema；
2. \(o_t, q_t, \ell, A_{t:t+H-1}, r_t, c_t, d_t\) 在任务里的含义；
3. 离线数据生成流程；
4. 在线闭环执行流程；
5. success checker 草案；
6. 失败分类表；
7. clean 到 random 的升级计划。

如果这七项都能写出来，具身系统就不再是一堆论文名，而是一条能被实现、调试和持续改进的闭环。
