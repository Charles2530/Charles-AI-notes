# VLA、WAM 与世界模型系统地图

这一页专门解决一个容易卡住的问题：`VLA`、`世界模型`、`DreamZero`、`π0.5`、`LingBot-World`、`轨迹生成`、`Sim2Real` 到底是什么关系。

先给结论：具身智能不是某一个模型名字，而是一套闭环系统。模型只负责其中一部分，真正落地还要有感知、状态估计、动作接口、规划控制、安全、数据引擎和评测脚本。

## 1. 一张总图：具身系统的七层

可以把具身智能拆成七层：

```text
任务目标 / 语言指令
  -> 感知与状态估计
  -> 高层语义与子任务
  -> 动作表示与策略
  -> 世界模型 / 动作后果预测
  -> 规划、控制与安全
  -> 数据、评测与回流
```

![π0.5 model overview 原论文图](../assets/images/paper-deep-dives/embodied-ai/pi05/figure-3-model-overview.png){ width="920" }

<small>图源：[π0.5](../paper-deep-dives/embodied-ai/pi05.md)，Figure 3。原论文图意：π0.5 把 high-level subtask prediction 和 flow-matching action expert 放进同一套 VLA 推理流程。</small>

!!! note "图解：这张图对应系统地图的哪几层"
    图左侧的多模态输入对应“感知与状态估计”，中间的 high-level prediction 对应“语义与子任务”，右侧的 action expert 对应“动作表示与策略”。它也暴露出 VLA 的边界：图里没有显式展开候选动作未来，也没有替代低层安全控制，所以后面还要接世界模型、planner、controller 和 success checker。

| 层级 | 负责的问题 | 典型对象 | 论文案例 |
| --- | --- | --- | --- |
| 感知与状态估计 | 机器人现在看见什么、物体在哪里、自己在哪里 | RGB、RGB-D、多视角、6D pose、proprioception | π0.5 的 4 路相机输入、DreamZero 的 visual context |
| 高层语义与子任务 | 当前应该处理哪个对象、下一步语义动作是什么 | `pick up the plate`、`open the drawer`、affordance | π0.5 的 high-level subtask prediction |
| 动作表示与策略 | 动作怎么输出，输出多长，给谁执行 | action token、action chunk、末端位姿、关节角、夹爪命令 | π0.5 的 FAST + flow action expert |
| 世界模型 | 如果这样做，未来会怎样 | future video、latent state、reward、risk、continue | Dreamer、LingBot-World |
| WAM | 同时建模未来世界和未来动作 | joint video-action prediction | DreamZero |
| 规划控制与安全 | 动作能不能执行、会不会碰撞、是否越界 | IK、cuRobo、STOMP、MPC、safety filter | 资产/轨迹数据管线、SayCan 式 affordance |
| 数据与评测 | 数据从哪里来、怎么判断成功、失败如何回流 | demos、assets、success checker、task progress | π0.5 的异构数据，DreamZero 的 diverse data |

这张表比“某个模型强不强”更重要。读论文时先问：它主要强化的是哪一层？是更会看、更会拆任务、更会出动作、更会预测未来，还是让数据和评测更可靠？

## 2. 三个核心数学接口

### VLA：从观测和语言直接到动作

VLA 通常学习：

\[
a_{t:t+H} \sim \pi_\theta(o_{\le t}, q_t, \ell)
\]

其中：

- \(o_{\le t}\)：历史视觉观测；
- \(q_t\)：机器人本体状态，例如关节角、夹爪状态、底盘状态；
- \(\ell\)：语言指令；
- \(a_{t:t+H}\)：未来一小段动作，也叫 action chunk。

VLA 的优势是直接：看见当前场景后就输出动作。它的短板是，它不一定显式知道“这个动作执行后未来会怎样”。

### 世界模型：从状态和动作预测未来

世界模型更像内部模拟器：

\[
p_\theta(o_{t+1:t+H}, r, c \mid o_{\le t}, a_{t:t+H-1}, \ell)
\]

它不只是问“现在做什么”，而是问“如果做这些动作，未来观测、奖励、风险和终止状态会怎样”。这条路线适合规划、反事实比较、风险评估和数据生成，但它不一定直接输出可执行动作。

### WAM：把未来世界和未来动作一起建模

DreamZero 代表的 World Action Model 可以写成：

\[
p_\theta(o_{t+1:t+H}, a_{t:t+H} \mid o_{\le t}, q_t, \ell)
\]

它同时生成未来视频和未来动作。直觉上，模型既要想象“世界会变成什么样”，又要给出“哪个动作会导致这个未来”。这就是 DreamZero 为什么比普通视频世界模型更靠近 policy。

## 3. 四条路线怎么区分

| 路线 | 核心问题 | 优势 | 风险 |
| --- | --- | --- | --- |
| 经典 pipeline | 先检测/估计状态，再规划轨迹，再控制执行 | 可解释、容易加安全约束、适合工业和高精度任务 | 语义开放性弱，手工接口多 |
| VLA policy | 观测和语言直接输出动作 | 学习端到端技能，适合跨任务泛化 | 不显式预测未来，失败恢复和安全要额外设计 |
| 世界模型 | 给定动作预测未来 | 能做 what-if、规划、风险评估和数据生成 | 视觉好看不等于动作因果正确 |
| WAM | 联合预测未来视频和动作 | 世界预测和动作输出强绑定，可能直接成为 policy | 推理成本高，视频未来错时动作也会跟着错 |

所以 π0.5、DreamZero、LingBot-World 不是同一类东西：

| 系统 | 更像什么 | 解决的主要问题 |
| --- | --- | --- |
| π0.5 | VLA policy | 新家庭、新物体和长任务中，如何从高层语言目标生成子任务和动作块 |
| DreamZero | WAM policy | 如何让视频基础模型通过 joint video-action prediction 直接输出可执行动作 |
| LingBot-World | interactive world simulator | 如何把视频生成模型改造成动作条件、因果化、少步、实时的世界模拟器 |
| Dreamer | latent world model + RL | 如何在 latent imagination 中训练 actor/value，减少真实交互成本 |

## 4. π0.5：它教会我们怎样理解 VLA

[π0.5](../paper-deep-dives/embodied-ai/pi05.md) 的重点不是“机械臂又会抓东西了”，而是它把 VLA 拆成了高层语义和低层动作两条输出路径。

```text
高层任务: clean the kitchen
  -> high-level subtask: pick up the plate
  -> continuous action chunk: base / arms / grippers move for about 1 second
```

| π0.5 设计 | 放回具身系统里怎么理解 |
| --- | --- |
| heterogeneous co-training | 开放世界泛化不能只靠目标机器人数据，要混合 web 语义、跨机器人数据、移动操作数据和人类指导 |
| high-level subtask prediction | 长任务需要显式或隐式任务分解，否则模型只会做短技能 |
| FAST action tokens | 把动作离散化，方便和文本、bbox、web/VLM 数据一起做大规模预训练 |
| flow matching action expert | 真实控制需要连续动作，不能完全依赖离散 token 自回归输出 |
| task progress rubric | 长任务不能只看最终成功/失败，要看每个阶段完成到哪里 |

π0.5 最适合理解“VLA 是什么”：它把视觉语言状态映射到动作，同时用高层子任务缓解长任务难度。它不等于完整世界模型，因为它主要输出动作，并没有显式展开多个候选动作的未来后果。

!!! note "读 π0.5 时先抓住这个分工"
    `HL` 解决“下一步语义上该做什么”，`action expert` 解决“身体怎么动”。这两个问题不能混在一起。一个模型可能知道该拿盘子，但不会稳定抓；也可能能抓盘子，但不知道清理厨房下一步该抓什么。

## 5. DreamZero：它教会我们怎样理解 WAM

[DreamZero](../paper-deep-dives/embodied-ai/dreamzero.md) 的关键是把视频基础模型改造成 joint video-action model。它不是先预测视频、再用另一个模型反推动作，而是在同一个生成过程中一起预测未来视频和动作。

```text
current visual context + language + robot state
-> future video latents
-> future action chunk
-> execute action
-> refresh context with real observation
```

![DreamZero architecture 原论文图](../assets/images/paper-deep-dives/world-models/dreamzero/figure-4-architecture.png){ width="920" }

<small>图源：[DreamZero](../paper-deep-dives/embodied-ai/dreamzero.md)，Figure 4。原论文图意：DreamZero 在视频生成主干中组织 visual context、language、robot state 和 action prediction，使未来视频和动作共享生成上下文。</small>

!!! note "图解：系统地图里的 WAM 不是普通视频生成器"
    读这张架构图时看四个接口：视觉历史如何进模型，机器人状态如何进模型，动作从哪里解码，执行后如何回到真实观测。只有这四个接口都说清楚，WAM 才能从“会生成未来画面”变成“能进入机器人控制循环”的系统模块。

| DreamZero 设计 | 放回具身系统里怎么理解 |
| --- | --- |
| joint video-action prediction | 动作必须解释未来视觉变化，未来视觉也必须和动作一致 |
| autoregressive WAM | 适合闭环执行和 KV cache，比双向整段生成更接近控制需求 |
| real observation feedback | 执行动作后用真实观测刷新上下文，避免完全依赖自己生成的视频 |
| DreamZero-Flash | 机器人控制有 reactivity gap，少步化和系统优化是能力的一部分 |
| video error -> wrong action | WAM 的动作会忠实跟随错误视频计划，因此需要风险检测和安全层 |

DreamZero 最适合理解“世界模型和 policy 的边界正在变模糊”。传统世界模型是给 planner 用的；DreamZero 让 WAM 自己直接成为 policy。但这也带来新风险：如果未来视频预测错，动作可能不是随机错，而是很一致地执行错误未来。

!!! note "读 DreamZero 时先抓住这个矛盾"
    它比 VLA 多学了未来后果，因此更能利用异质数据和跨任务动态；但它比普通 VLA 更贵、更慢，也更依赖视频预测质量。它不是免费升级，而是把“会想象未来”引入策略后必须付出的系统代价。

## 6. LingBot-World 和 Dreamer：世界模型的两种用途

[LingBot-World](../paper-deep-dives/world-models/lingbot-world.md) 和 [Dreamer](../paper-deep-dives/world-models/dreamer.md) 都是世界模型，但用途不同。

| 路线 | 世界模型输出 | 谁消费这个输出 | 典型用途 |
| --- | --- | --- | --- |
| LingBot-World | 动作条件未来视频 | 人、planner、模拟器、数据引擎 | 交互模拟、反事实视频、训练数据生成 |
| Dreamer | latent state、reward、continue | actor/value | 在想象轨迹里训练策略 |
| DreamZero | 未来视频 + 未来动作 | 机器人控制循环 | WAM 直接当 policy |

LingBot-World 说明视频生成模型要成为世界模拟器，必须补动作条件、长时记忆、因果化和少步推理。Dreamer 说明世界模型不一定要生成像素，只要 latent dynamics 能支持奖励预测和策略训练，也可以非常有用。

因此不要把世界模型狭义理解成“生成未来视频”。更准确的判断标准是：

1. 它是否对动作敏感；
2. 它是否能维持同一个世界状态；
3. 它的输出是否被 planner、policy、risk module 或 data engine 消费；
4. 它是否在闭环中改善任务结果。

## 7. 你那条资产和轨迹 pipeline 放在哪

你整理的流程：

```text
3D 模型+纹理 -> 物理属性 -> 导入资产 -> 抓取轴
-> 场景布置 -> 任务判定脚本 -> cuRobo 轨迹 GT
-> 失败复位重试
```

它不是和 π0.5 / DreamZero 对立的路线，而是在给 VLA、WAM 和传统 planner 提供基础设施：

| 你的 pipeline 组件 | 给模型路线提供什么 |
| --- | --- |
| 资产和纹理 | 视觉输入分布、物体类别、尺度变化 |
| 物理属性 | 接触、摩擦、质量、仿真动力学 |
| 抓取轴标注 | grasp candidates、pregrasp、approach direction |
| 场景布置 | clean / random evaluation、domain randomization |
| 任务判定脚本 | 自动过滤失败、生成 task progress 和 success labels |
| cuRobo / STOMP 轨迹 | 行为克隆 GT、planner baseline、任务可行性验证 |
| 失败复位重试 | 数据引擎、hard case mining、评测稳定性 |

换句话说，π0.5 这类 VLA 需要大量动作轨迹和任务数据；DreamZero 这类 WAM 需要视频和动作对齐轨迹；传统 planner 需要资产、碰撞网格和抓取标注。你的 pipeline 正好是这些路线共同依赖的数据底座。

## 8. 初学者最容易混的六组概念

| 容易混的概念 | 正确区分 |
| --- | --- |
| VLM vs VLA | VLM 主要回答视觉语言问题，VLA 要输出可执行动作 |
| VLA vs world model | VLA 直接给动作，world model 预测动作后果 |
| world model vs WAM | world model 可以只预测未来状态，WAM 同时预测未来状态和动作 |
| action chunk vs trajectory | action chunk 是短窗口动作，trajectory 是完整任务轨迹 |
| planning vs control | planning 找路径或目标序列，control 高频跟踪并处理动力学 |
| success rate vs task progress | success rate 只看最终是否成功，task progress 能看长任务完成到哪一步 |

记住这六组，读 π0.5、DreamZero 和具身数据 pipeline 会清楚很多。

## 9. 推荐阅读顺序

如果你现在对具身智能还没有整体感，建议这样读：

1. 先读 [具身智能从零学习路线](beginner-learning-path.md)，把基本名词和学习顺序立住。
2. 再读本页，建立 VLA、WAM、世界模型和工程 pipeline 的层级关系。
3. 然后读 [从一个任务跑通具身闭环](closed-loop-task-walkthrough.md)，用尺寸排序和容器匹配任务把状态、轨迹、执行和判定连起来。
4. 再读 [资产、感知、抓取与轨迹数据管线](asset-perception-grasp-and-trajectory-pipeline.md)，搞清楚数据从哪里来。
5. 接着读 [VLA 动作表示与控制接口](../vla/action-representation-and-interfaces.md)，理解动作怎么进模型、怎么出模型。
6. 再读 [规划、控制与安全](planning-control-and-safety.md)，理解为什么动作还要经过 planner/controller/safety。
7. 然后读 [π0.5](../paper-deep-dives/embodied-ai/pi05.md)，把 VLA 路线放进真实长任务。
8. 最后读 [DreamZero](../paper-deep-dives/embodied-ai/dreamzero.md) 和 [LingBot-World](../paper-deep-dives/world-models/lingbot-world.md)，理解世界模型路线怎样进入机器人闭环。

读完后，回到一个问题检验是否理解：给定“整理厨房”这个任务，你能否说清楚系统如何看见场景、分解子任务、生成动作、预测风险、执行控制、判定成功，并把失败样本回流到下一轮训练。
