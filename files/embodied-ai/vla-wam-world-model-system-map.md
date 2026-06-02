# VLA、WAM 与世界模型地图

这一页是入门页之后的第二层地图。入门页已经讲了基本符号和四类模型；这里专门回答更工程化的问题：


**VLA、world model、WAM、planner、controller、data engine 到底谁负责什么，彼此怎么接？**

先给结论：具身智能不是某一个模型名字，而是一套闭环系统。模型只负责其中一部分，真正落地还要有感知、状态估计、动作接口、规划控制、安全、数据引擎和评测脚本。

## 一张系统地图

可以把具身系统压成下面这条链：

```text
语言目标 / 任务约束
  -> 观测与状态估计
  -> 高层语义与子任务
  -> 动作表示与策略
  -> 世界模型 / 动作后果预测
  -> planner / controller / safety
  -> success checker / 数据回流
```

对应到符号：

\[
(o_{\le t}, q_t, \ell)
\rightarrow
\text{subtask}_t
\rightarrow
A_{t:t+H-1}
\rightarrow
(\hat z_{t+1:t+H}, \hat r, \hat c, \hat d)
\rightarrow
\text{safe command}
\rightarrow
o_{t+1}.
\]


这条式子表示一次闭环从“看见什么、机器人在哪、目标是什么”开始，经过子任务、动作块和未来风险预测，最后才变成安全控制命令。符号里 \(o_{\le t}\) 是历史观测，\(q_t\) 是当前机器人状态，\(\ell\) 是语言目标，\(\hat r,\hat c,\hat d\) 分别表示预测收益、继续概率和危险信号。

| 环节 | 输入 | 输出 | 主要风险 |
| --- | --- | --- | --- |
| 观测与状态估计 | \(o_t, q_t\) | 物体、机器人、场景状态 | 遮挡、标定错、深度错 |
| 高层语义 | \(o_{\le t}, \ell\) | 子任务、目标对象、任务进度 | 目标绑定错、长任务记忆断 |
| VLA policy | \(o_{\le t}, q_t, \ell\) 或子任务 | \(A_{t:t+H-1}\) | 动作可执行性和恢复不足 |
| world model | \(z_t, A_{t:t+H-1}\) | 未来状态、收益、风险、终止 | 未来好看但不受动作控制 |
| WAM | \(o_{\le t}, q_t, \ell\) | 未来视频 + 动作 chunk | 视频错时动作也跟着错 |
| planner / controller | 动作、目标、约束 | 真实机器人命令 | IK、碰撞、延迟、力控 |
| checker / data engine | 真实新观测、任务脚本 | success、failure、hard case | 判定粗糙，失败无法回流 |

这张表比“某个模型强不强”更重要。读论文时先问：它主要强化的是哪一层？是更会看、更会拆任务、更会出动作、更会预测未来，还是让数据和评测更可靠？

## VLA 的边界：动作从哪里来

VLA 通常学习：

\[
A_{t:t+H-1} \sim \pi_\theta(o_{\le t}, q_t, \ell).
\]


它的接口很直接：观测、机器人状态和语言进来，动作 chunk 出去。这里 \(\pi_\theta\) 表示带参数 \(\theta\) 的策略模型，\(A_{t:t+H-1}\) 表示从当前时刻 \(t\) 开始的 \(H\) 步动作序列。

| VLA 负责 | VLA 不天然负责 |
| --- | --- |
| 从视觉语言状态生成动作 | 严格碰撞检查 |
| 处理跨任务、跨对象的动作泛化 | 高频控制和动力学稳定 |
| 学习人类示范或机器人轨迹 | 明确比较多个候选动作未来 |
| 输出 action token、连续动作或 action chunk | 证明执行后一定成功 |

π0.5 的架构图正好展示了 VLA 的强项和边界。

![π0.5 model overview 原论文图](../assets/images/paper-deep-dives/embodied-ai/pi05/figure-3-model-overview.png){ width="920" }

<small>图源：[π0.5](../paper-deep-dives/embodied-ai/pi05.md)，Figure 3。原论文图意：π0.5 把 high-level subtask prediction 和 flow-matching action expert 放进同一套 VLA 推理流程。</small>

!!! note "图解：这张图对应系统地图的哪几层"
    图左侧的多模态输入对应观测与状态，中间 high-level prediction 对应语义和子任务，右侧 action expert 对应动作表示与策略。图里没有显式展开多个候选动作的未来，也没有替代低层安全控制，所以它需要和 world model、planner、controller、checker 一起看。

π0.5 可以放回系统地图这样理解：

| 设计 | 系统含义 |
| --- | --- |
| heterogeneous co-training | 开放世界泛化不能只靠目标机器人数据 |
| high-level subtask prediction | 长任务需要任务进度和语义拆解 |
| FAST action tokens | 离散动作便于和文本、bbox、web 数据一起预训练 |
| flow matching action expert | 真实控制仍需要连续、平滑的动作块 |
| task progress rubric | 长任务要看阶段进度，不只看最终成功 |

## 世界模型的边界：未来怎么被消费

世界模型通常学习：

\[
p_\theta(z_{t+1:t+H}, r_{t:t+H}, c_{t:t+H}, d_{t:t+H}
\mid z_t, A_{t:t+H-1}, \ell).
\]


它回答的是 what-if：如果执行这段动作，未来会怎样。

| 世界模型输出 | 谁会用 | 用来做什么 |
| --- | --- | --- |
| future video / future latent | 人、planner、policy、数据引擎 | 观察候选未来、做反事实比较 |
| reward / progress | planner、RL actor/value | 选择更接近目标的动作 |
| risk / cost | safety filter、risk module | 避免碰撞、滑落、越界 |
| done / continue | policy、checker | 判断是否结束、失败或继续 |

世界模型的核心不是“生成视频看起来像不像”，而是它的输出是否进入决策。

!!! warning "常见误读：视频世界模型不等于可用世界模型"
    如果固定历史后，无论输入 noop、push、grasp，模型都生成差不多的未来，它就没有学到 action sensitivity。对具身系统来说，动作敏感性比视觉自然度更关键。

一个世界模型最小验收可以这样问：

1. 同一状态下，换不同动作，未来是否合理分叉？
2. 它能否预测风险，而不只是预测像素？
3. 它的 latent 是否能支持 planner 或 policy 选择动作？
4. 它是否用真实观测定期刷新，避免 rollout 漂移？

## WAM 的边界：未来和动作一起生成

DreamZero 代表的 WAM 可以写成：

\[
(o_{t+1:t+H}, A_{t:t+H-1})
\sim
p_\theta(o, A \mid o_{\le t}, q_t, \ell).
\]


它把未来观测和未来动作绑在同一个生成过程中。优势是动作必须解释未来视觉变化；风险是未来视觉错了，动作也可能沿着错误未来走。

![DreamZero architecture 原论文图](../assets/images/paper-deep-dives/world-models/dreamzero/figure-4-architecture.png){ width="920" }

<small>图源：[DreamZero](../paper-deep-dives/embodied-ai/dreamzero.md)，Figure 4。原论文图意：DreamZero 在视频生成主干中组织 visual context、language、robot state 和 action prediction，使未来视频和动作共享生成上下文。</small>

!!! note "图解：WAM 不是普通视频生成器"
    读这张架构图时看四个接口：视觉历史如何进模型，机器人状态如何进模型，动作从哪里解码，执行后如何回到真实观测。只有这四个接口都说清楚，WAM 才能从“会生成未来画面”变成“能进入机器人控制循环”的系统模块。

WAM 和普通 world model 的区别不是“有没有视频”，而是“动作是不是生成目标的一部分”。

| 路线 | 输出 | 常见消费方式 | 风险 |
| --- | --- | --- | --- |
| 动作条件 world model | 给定动作，预测未来 | planner 比较候选动作 | 模型可预测但不直接给动作 |
| VLA policy | 给定观测和语言，输出动作 | controller 执行动作 | 不显式建模动作后果 |
| WAM | 同时输出未来和动作 | 直接进入控制循环 | 未来错时动作也会一致地错 |

DreamZero 失败图很好地说明了 WAM 的双刃剑。

![DreamZero failure case 原论文图](../assets/images/paper-deep-dives/world-models/dreamzero/figure-16-failure-case.png){ width="920" }

<small>图源：[World Action Models are Zero-shot Policies](https://arxiv.org/abs/2602.15922)，Figure 16。原论文图意：展示 WAM 生成视频与真实执行的失败配对；当生成未来偏离任务目标时，执行动作也会沿着错误未来走。</small>

!!! note "图解：这张图支撑的是 WAM 风险判断"
    这不是普通“视频生成失败”的例子，而是动作和未来高度绑定后的失败。绑定成功时，动作更能解释未来；绑定失败时，动作也会忠实执行错误未来。因此 WAM 的验收不能只看视频像不像，还要看 predicted future、issued action、real observation 和 success checker 是否一致。

## Planner、controller 和 safety 的边界

模型动作通常不是最终电机命令。更稳的部署链路是：

```text
VLA / WAM action
-> action adapter
-> planner or controller
-> safety filter
-> robot command
```

| 层级 | 回答的问题 | 典型对象 |
| --- | --- | --- |
| action adapter | 模型动作怎么转成机器人接口 | 坐标系转换、归一化反变换、控制模式 |
| planner | 目标或轨迹是否可达，路径怎么走 | IK、cuRobo、STOMP、采样规划 |
| controller | 每个控制周期怎么跟踪 | PID、MPC、阻抗控制、全身控制 |
| safety filter | 动作是否越界或危险 | 碰撞、速度、力、工作空间、急停 |

SayCan 提供了一个直观例子：语言模型判断 useful，affordance 判断 possible，组合后才选动作。

![SayCan LLM and affordance scores 原论文图](../assets/images/paper-figures/embodied-ai/saycan-figure-3-llm-affordance.png){ width="920" }

<small>图源：[Do As I Can, Not As I Say: Grounding Language in Robotic Affordances](https://arxiv.org/abs/2204.01691)，Figure 3。原论文图意：LLM 给候选技能打“是否符合指令”的分数，value functions 给技能打“当前环境是否可执行”的 affordance 分数，二者组合后选择下一步机器人动作。</small>

!!! note "图解：模型的聪明必须被可执行性约束"
    一个候选动作可能非常符合语言目标，但当前夹爪够不到、路径会碰撞、物体不可抓。planner 和 safety 的作用就是把“语义上正确”投影到“物理上可执行”。

## Data engine 的边界：让系统能进步

VLA、world model 和 WAM 都需要数据，但需要的数据形态不同。

| 数据组件 | 给 VLA | 给 world model / WAM | 给 planner / checker |
| --- | --- | --- | --- |
| 多相机观测 | 视觉条件 | 未来预测和时序建模 | 回放和失败定位 |
| 机器人状态 \(q_t\) | 动作条件 | dynamics condition | 控制和安全约束 |
| action chunk \(A_{t:t+H-1}\) | 行为克隆标签 | action-conditioned future | 执行记录 |
| task progress / reward \(r_t\) | 任务状态监督 | rollout 价值信号 | 分阶段评测 |
| risk / cost \(c_t\) | 避免危险动作 | 风险预测头 | safety 门禁 |
| done \(d_t\) | 学会停止或拒绝 | continue / termination | success checker |
| failure reason | 恢复策略训练 | hard negative | 调试和回流 |

Open X-Embodiment 这张图说明了为什么 data engine 不是“把轨迹堆起来”这么简单。

![Open X-Embodiment dataset overview 原论文图](../assets/images/paper-figures/embodied-ai/open-x-embodiment-figure-1-dataset.png){ width="760" }

<small>图源：[Open X-Embodiment: Robotic Learning Datasets and RT-X Models](https://arxiv.org/abs/2310.08864)，Figure 1。原论文图意：统计 Open X-Embodiment 中不同 robot embodiment、scene、trajectory、skill 和 object category 的分布，展示跨机器人数据的异质性。</small>

!!! note "图解：数据引擎要处理跨本体和接口差异"
    图里的机器人、场景、技能和对象分布都不同。跨数据训练时，动作坐标系、控制频率、相机视角、夹爪形态和任务标签必须被记录清楚，否则模型会把硬件差异误学成任务差异。

如果你的系统已经有下面这条管线：

```text
3D 模型+纹理 -> 物理属性 -> 导入资产 -> 抓取轴
-> 场景布置 -> 任务判定脚本 -> cuRobo 轨迹 GT
-> 失败复位重试
```

它不是和 π0.5 / DreamZero 对立的路线，而是在给它们提供基础设施：

| pipeline 组件 | 给模型路线提供什么 |
| --- | --- |
| 资产和纹理 | 视觉输入分布、物体类别、尺度变化 |
| 物理属性 | 接触、摩擦、质量、仿真动力学 |
| 抓取轴标注 | grasp candidates、pregrasp、approach direction |
| 场景布置 | clean / random evaluation、domain randomization |
| 任务判定脚本 | 自动过滤失败、生成 task progress 和 success labels |
| cuRobo / STOMP 轨迹 | 行为克隆 GT、planner baseline、任务可行性验证 |
| 失败复位重试 | 数据引擎、hard case mining、评测稳定性 |

## 四条路线怎么选

| 路线 | 什么时候优先 | 不适合单独承担什么 |
| --- | --- | --- |
| 经典 pipeline | 工业、高精度、任务结构明确、安全要求高 | 开放语言、开放物体和新场景泛化 |
| VLA policy | 任务多样、语言开放、需要从演示中学技能 | 显式风险预测和严格安全证明 |
| world model + planner | 需要比较候选未来、降低真实试错成本 | 直接输出低层动作 |
| WAM policy | 希望动作和未来视觉一起建模，减少两阶段误差 | 替代安全层和真实观测刷新 |

一个成熟系统常常不是四选一，而是组合：

```text
VLA 产生候选动作
world model 预测未来和风险
planner/controller 投影到可执行命令
success checker 判断结果
data engine 回流失败
```

## 初学者最容易混的六组概念

| 容易混的概念 | 正确区分 |
| --- | --- |
| VLM vs VLA | VLM 主要回答视觉语言问题，VLA 要输出可执行动作 |
| VLA vs world model | VLA 直接给动作，world model 预测动作后果 |
| world model vs WAM | world model 可以只预测未来状态，WAM 同时预测未来状态和动作 |
| action chunk vs trajectory | action chunk 是短窗口动作，trajectory 是完整任务轨迹 |
| planning vs control | planning 找路径或目标序列，control 高频跟踪并处理动力学 |
| success rate vs task progress | success rate 只看最终是否成功，task progress 能看长任务完成到哪一步 |

记住这六组，读 π0.5、DreamZero 和具身数据 pipeline 会清楚很多。

## 继续阅读

建议这样接下去：

1. [一个任务跑通具身闭环](closed-loop-task-walkthrough.md)：用同一套符号走完整 episode。
2. [VLA 动作表示与控制接口](../vla/action-representation-and-interfaces.md)：理解动作怎么进模型、怎么出模型。
3. [规划、控制与安全](planning-control-and-safety.md)：理解模型动作如何落到真实机器人。
4. [资产到轨迹：感知、抓取与数据管线](asset-perception-grasp-and-trajectory-pipeline.md)：搞清楚数据从哪里来。
5. [π0.5](../paper-deep-dives/embodied-ai/pi05.md)、[DreamZero](../paper-deep-dives/embodied-ai/dreamzero.md)、[LingBot-World](../paper-deep-dives/world-models/lingbot-world.md)：把具体论文放回这张地图。

读完后，用一个问题自测：给定“整理厨房”这个任务，你能否说清楚系统如何看见场景、分解子任务、生成动作、预测风险、执行控制、判定成功，并把失败样本回流到下一轮训练。
