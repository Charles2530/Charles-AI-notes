# 具身智能从零路线

这一页是具身智能章节的第一入口。它不是从论文名开始，而是从一个最朴素的任务开始：

<!-- [补充] -->
| 读者 | 建议读法 | 工程落点 |
| --- | --- | --- |
| 初学者 | 先抓“具身智能从零路线”要解决的问题、关键术语和最小例子；公式只看变量方向。 | 能说清输入、输出和常见失败信号。 |
| 工程师 | 直接对照结构、指标、成本和边界条件，必要时回到前置页补概念。 | 落到数据、算力、延迟、显存、评测和回滚检查。 |


!!! note "读法定位"
    这页先回答“具身智能从零路线”在「具身智能」里的位置：它解决什么局部问题，依赖哪些前置，最后会影响哪类工程或研究判断。
    **前置**：先知道 VLA、世界模型、相机/深度和评测集的基本角色。 必要时先回 [具身智能入口](index.md)、[基础知识](../foundations/index.md) 或 [术语表](../glossary/index.md)。
    **主线关系**：把感知、数据、仿真、策略、控制、安全和真实闭环连起来，看机器人系统为什么不能只靠离线指标判断。

**机器人听到“把红杯放到托盘上”，到底要连续做对哪些事？**

它至少要看见杯子和托盘，理解“放到”是一个空间目标，选择抓取方式，输出动作，执行后重新观察，判断杯子是否真的在托盘上。如果杯子滑了、托盘被遮住了、夹爪没夹稳，它还要能停下来、重试或请求帮助。

所以具身智能不是“VLM 接一个机械臂”，而是一条闭环：

```text
看见世界 -> 理解任务 -> 选择动作 -> 预测后果 -> 执行控制 -> 判断成败 -> 回流数据
```

只要这条链里有一段不稳，机器人就会失败。π0.5、DreamZero、LingBot-World、Dreamer、RoboTwin、cuRobo 这些名字，都是在强化这条链里的不同位置。

## 先用一张图建立闭环直觉

RT-1 原论文图很适合作为第一张具身智能图：左边是语言和图像历史，中间是 Transformer policy，右边是机器人动作。注意它不是“看图回答问题”，而是每输出一个动作，世界都会变，下一帧观测又会影响后续动作。

![RT-1 overview 原论文图](../assets/images/paper-figures/embodied-ai/rt1-figure-1-overview.png){ width="760" }

<small>图源：[RT-1: Robotics Transformer for Real-World Control at Scale](https://arxiv.org/abs/2212.06817)，Figure 1。原论文图意：RT-1 接收自然语言指令和图像历史，经过 tokenization 与 Transformer policy，以约 3Hz 输出机器人动作 token，驱动机械臂完成真实任务。</small>

!!! note "图解：第一眼先看循环，不要只看模型框"
    左侧观测进入模型，右侧动作改变环境，新的图像又回到左侧。这就是具身智能和静态 VLM 的根本差别。静态 VLM 的答案通常停在一句话；具身系统的答案会变成物理动作，并产生新的后果。

普通 VLM / LLM 多数时候回答静态问题：

```text
输入: 一张图 + 一个问题
输出: 一句话
```

具身智能回答闭环问题：

```text
输入: 当前观测 + 机器人状态 + 任务目标
输出: 一个动作或一小段动作
动作改变世界
系统重新观测
再输出下一个动作
```

因此具身智能最关键的不是“看懂了没有”，而是：

1. 看懂以后能不能动；
2. 动了以后世界会不会按预期变化；
3. 如果变化不对，能不能恢复；
4. 如果有危险，能不能停下、降级或拒绝。

## 七个基本问题

任何具身系统都可以按七个问题拆开。读论文或看项目时，先问它主要解决哪一格。

| 问题 | 初学者理解 | 常见模块 | 论文 / 项目对应 |
| --- | --- | --- | --- |
| 我看见了什么 | 图像里有哪些物体，它们在哪里、多大、什么状态 | VLM、检测、分割、深度、pose estimation | [相机、深度与机器人视觉](cameras-depth-and-robot-vision.md)、Depth Anything、VGGT |
| 我现在是什么状态 | 机械臂、夹爪、底盘、历史动作处于什么状态 | proprioception、state estimator、memory | 多相机和机器人本体状态输入 |
| 我要做什么 | 任务目标是什么，下一步子任务是什么 | LLM/VLM planner、high-level policy | π0.5 的 high-level subtask |
| 我能怎么动 | 动作怎么表示，输出几步，交给谁执行 | action token、action chunk、末端位姿、关节角 | ACT、π0.5 的 action expert |
| 做了会怎样 | 执行动作后未来状态、视频、风险会怎样 | world model、WAM、risk model | DreamZero、LingBot-World、Dreamer |
| 能不能安全执行 | 路径是否碰撞，力是否过大，是否越界 | IK、motion planner、MPC、safety filter | SayCan、cuRobo、规划控制页 |
| 成功了吗 | 任务是否达成，失败在哪一步，数据如何回流 | success checker、task progress、benchmark | RoboTwin / 自建任务脚本 |

如果一篇论文没有告诉你它在回答哪一个问题，就很容易读乱。

## 符号速查：先别怕公式

具身智能公式的作用，通常只是把“谁输入、谁输出、看多远”写清楚。先记住下面这些符号，后面的 VLM、VLA、世界模型和 WAM 就不会散。

| 符号 | 读法 | 初学者理解 | 例子 |
| --- | --- | --- | --- |
| \(t\) | time step | 当前时刻 | 第 37 帧、当前控制周期 |
| \(o_t\) | observation | 当前传感器观测 | RGB 图像、深度、触觉、关节角读数 |
| \(o_{\le t}\) | observation history | 到当前为止的观测历史 | 最近几帧图像和动作反馈 |
| \(q_t\) | robot state | 机器人本体状态 | 关节角、夹爪开合、底盘速度、末端位姿 |
| \(\ell\) | language instruction | 语言目标 | “把红杯放到托盘上” |
| \(a_t\) | action | 当前动作 | 末端向前 2 厘米、夹爪闭合 |
| \(A_{t:t+H-1}\) | action chunk | 从现在开始的一小段动作 | 未来 1 秒的连续控制量 |
| \(H\) | horizon | 往未来看多少步 | 20Hz 控制下，\(H=20\) 约等于 1 秒 |
| \(\pi_\theta\) | policy | 参数为 \(\theta\) 的策略模型，负责选动作 | VLA policy |
| \(p_\theta\) | probability model | 参数为 \(\theta\) 的概率模型，负责给未来或输出分布建模 | 世界模型、视频生成模型 |
| \(z_t\) | latent state | 模型内部压缩出的世界状态 | 视觉 encoder 或世界模型记忆向量 |
| \(r_t\) | reward / progress | 任务收益或进度 | 抓住了、接近目标、完成一个子任务 |
| \(c_t\) | cost / risk | 风险或代价 | 碰撞、滑落、越界、用力过大 |
| \(d_t\) | done | 任务是否结束 | 成功、失败、超时、中止 |

!!! warning "常见误解：同一个字母在不同论文里可能含义不同"
    有的论文用 \(s_t\) 表示真实状态，有的用 \(z_t\) 表示 latent state；有的把 \(H\) 写成 horizon，有的把 \(H\) 用作图像高度。读公式时先看符号表，不要只按记忆硬套。

一个最小闭环可以写成：

\[
o_{\le t}, q_t, \ell
\rightarrow
\pi_\theta
\rightarrow
A_{t:t+H-1}
\rightarrow
\text{controller}
\rightarrow
o_{t+1}.
\]

<!-- [补充] -->
| 符号 | 含义 | 取值范围 |
| --- | --- | --- |
| \(\pi_\theta\) | 模型或策略参数 | 实值张量，形状由模型结构决定 |
| \(o_{t}\) | 公式中的变量或中间量 | 按上下文定义 |
| \(q_t\) | 公式中的变量或中间量 | 按上下文定义 |
| \(A_{t:t+H-1}\) | 动作、优势或系数 | 动作空间/实数标量，依上下文而定 |
| \(o_{t+1}\) | 公式中的变量或中间量 | 按上下文定义 |


白话读法是：给定历史观测、机器人状态和语言目标，策略输出未来一小段动作，控制器执行后得到新观测。

这个公式说明了闭环接口；它不说明动作一定安全，也不说明任务一定成功。

## 四类模型先分清楚

### 4.1 VLM：看懂当前画面

VLM 主要回答：

\[
y \sim p_\theta(y \mid x_v, x_t)
\]

<!-- [补充] -->
| 符号 | 含义 | 取值范围 |
| --- | --- | --- |
| \(\theta\) | 模型或策略参数 | 实值张量，形状由模型结构决定 |
| \(y\) | 公式中的变量或中间量 | 按上下文定义 |
| \(p\) | 概率、转移分布或预测量 | 概率分布或任务定义的张量 |
| \(x_v\) | 样本、状态或 latent 变量 | 张量空间，形状由数据表示决定 |
| \(x_t\) | 样本、状态或 latent 变量 | 张量空间，形状由数据表示决定 |


**读作什么**：给定图像和文本，模型生成一个回答。

**红杯例子**：输入桌面图和问题“红杯在哪里”，输出“红杯在托盘左侧”。

**它不说明什么**：这个公式不保证机器人会移动，也不说明夹爪执行后世界会怎样。

### 4.2 VLA：从视觉语言到动作

VLA 主要回答：

\[
A_{t:t+H-1} \sim \pi_\theta(o_{\le t}, q_t, \ell)
\]

<!-- [补充] -->
| 符号 | 含义 | 取值范围 |
| --- | --- | --- |
| \(\pi_\theta\) | 模型或策略参数 | 实值张量，形状由模型结构决定 |
| \(A_{t:t+H-1}\) | 动作、优势或系数 | 动作空间/实数标量，依上下文而定 |
| \(o_{t}\) | 公式中的变量或中间量 | 按上下文定义 |
| \(q_t\) | 公式中的变量或中间量 | 按上下文定义 |


**读作什么**：策略根据看见的东西、机器人自身状态和语言目标，输出接下来一段动作。

**红杯例子**：模型输出“末端靠近红杯、夹爪闭合、抬起一点”的连续动作块。

**它不说明什么**：VLA 直接给动作，但不一定显式比较“如果这样做会不会撞、会不会滑、未来是否更好”。

ACT 这张图能帮助理解 action chunk：策略不是每次只吐一个动作点，而是生成一段短动作序列，并在执行时用 temporal ensembling 等方式平滑。

![ACT architecture 原论文图](../assets/images/paper-figures/vla/aloha-act-algorithm.png){ width="720" }

<small>图源：[Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705)，Figure 2。原论文图意：Action Chunking with Transformers (ACT) 用 CVAE 编码动作序列和关节观测，测试时根据多视角图像、关节状态和 latent 生成一段 action sequence，并用 temporal ensembling 平滑执行。</small>

!!! note "图解：action chunk 是短窗口，不是完整任务轨迹"
    看这张图时先找 action sequence。chunk 的价值是保留短时技能结构、减少逐帧抖动；风险是如果第二步已经错了，后面几步可能会继续错。所以真实系统仍要滚动重规划，而不是一次性输出整段任务。

### 4.3 世界模型：预测动作后果

世界模型主要回答：

\[
p_\theta(z_{t+1:t+H}, r_{t:t+H}, c_{t:t+H}, d_{t:t+H}
\mid z_t, A_{t:t+H-1}, \ell)
\]

<!-- [补充] -->
| 符号 | 含义 | 取值范围 |
| --- | --- | --- |
| \(\theta\) | 模型或策略参数 | 实值张量，形状由模型结构决定 |
| \(p\) | 概率、转移分布或预测量 | 概率分布或任务定义的张量 |
| \(z_{t+1:t+H}\) | 样本、状态或 latent 变量 | 张量空间，形状由数据表示决定 |
| \(r_{t:t+H}\) | 奖励、回报或时间索引相关量 | 实数标量或上下文定义的索引范围 |
| \(c_{t:t+H}\) | 公式中的变量或中间量 | 按上下文定义 |
| \(d_{t:t+H}\) | 公式中的变量或中间量 | 按上下文定义 |


**读作什么**：如果从当前状态执行这一段动作，模型预测未来会怎样、收益如何、风险如何、是否结束。

**红杯例子**：如果快速从侧面推红杯，世界模型应该预测滑落或碰撞风险更高；如果从上方夹取并慢速移动，风险更低。

**它不说明什么**：未来预测不是执行保证。真实系统仍要用控制器、安全层和重新观测来纠错。

### 4.4 WAM：同时预测未来和动作

WAM，也就是 World Action Model，主要回答：

\[
(o_{t+1:t+H}, A_{t:t+H-1})
\sim
p_\theta(o, A \mid o_{\le t}, q_t, \ell)
\]

<!-- [补充] -->
| 符号 | 含义 | 取值范围 |
| --- | --- | --- |
| \(\theta\) | 模型或策略参数 | 实值张量，形状由模型结构决定 |
| \(o_{t+1:t+H}\) | 公式中的变量或中间量 | 按上下文定义 |
| \(A_{t:t+H-1}\) | 动作、优势或系数 | 动作空间/实数标量，依上下文而定 |
| \(p\) | 概率、转移分布或预测量 | 概率分布或任务定义的张量 |
| \(o\) | 公式中的变量或中间量 | 按上下文定义 |
| \(A\) | 动作、优势或系数 | 动作空间/实数标量，依上下文而定 |


**读作什么**：模型一边想象未来画面，一边给出导致这个未来的动作。

**红杯例子**：模型生成“夹起红杯并移动到托盘上方”的未来视频，同时输出对应动作。

**它不说明什么**：如果未来视频错了，动作也可能很一致地沿着错误未来走。因此 WAM 尤其需要真实观测刷新、risk checker 和 safety filter。

DreamZero 这条路线正是把未来视频和未来动作绑在一起。

![DreamZero joint video-action 原论文图](../assets/images/paper-deep-dives/world-models/dreamzero/figure-2-joint-video-action.png){ width="920" }

<small>图源：[DreamZero](../paper-deep-dives/embodied-ai/dreamzero.md)，Figure 2。原论文图意：WAM 把未来视觉和未来动作放在同一生成过程中，使动作必须和未来视觉变化对齐。</small>

!!! note "图解：WAM 的关键词是对齐"
    这张图不要只看未来视频是否清晰，而要看动作和视频是否互相解释。动作说“夹起杯子”，未来画面就应该出现杯子被夹起；如果画面没有变化或变化与动作无关，说明模型没有真正学到 action-conditioned future。

## π0.5：把长任务分成高层语义和低层动作

[π0.5](../paper-deep-dives/embodied-ai/pi05.md) 适合作为 VLA 入门论文。它面对的不是单个桌面抓取，而是“clean the kitchen”“clean the bedroom”这类开放世界长任务。

![π0.5 model overview 原论文图](../assets/images/paper-deep-dives/embodied-ai/pi05/figure-3-model-overview.png){ width="920" }

<small>图源：[π0.5](../paper-deep-dives/embodied-ai/pi05.md)，Figure 3。原论文图意：π0.5 同一个 VLA 系统先预测 high-level subtask，再由 action expert 输出低层连续动作，从而把长任务语义和机器人控制接起来。</small>

!!! note "图解：先看高层和低层分工"
    图中 high-level subtask 负责回答“下一步语义上做什么”，action expert 负责回答“身体怎么动”。这两个问题不能混成一个词。一个模型可能知道该拿盘子，但不会稳定抓；也可能能抓盘子，但不知道清理厨房下一步该处理什么。

π0.5 可以记成三句话：

1. 它解决开放世界长任务，需要高层子任务来跟踪进度；
2. 它把动作分成 high-level subtask 和低层连续 action chunk；
3. 它依赖异构数据，包括 web 多模态数据、跨机器人数据、移动操作数据、高层子任务和 verbal instruction。

所以 π0.5 的核心启发是：**VLA 不是只收更多动作轨迹，而是要把语义、任务分解和动作控制放进同一个训练体系。**

## DreamZero：让动作和未来画面互相约束

[DreamZero](../paper-deep-dives/embodied-ai/dreamzero.md) 适合理解 WAM。它的关键循环是：

```text
当前图像 + 语言 + 机器人状态
-> 生成未来视频和动作
-> 执行动作 chunk
-> 拿到真实新观测
-> 用真实观测替换模型预测上下文
-> 继续下一段
```

![DreamZero overview 原论文图](../assets/images/paper-deep-dives/world-models/dreamzero/figure-1-overview.png){ width="920" }

<small>图源：[DreamZero](../paper-deep-dives/embodied-ai/dreamzero.md)，Figure 1。原论文图意：从多源机器人数据训练 world-action model，使模型在执行时同时产生未来视觉和动作，并用真实观测持续闭环刷新。</small>

!!! note "图解：先找三条线"
    第一条线是视觉上下文从哪里来，第二条线是动作 chunk 如何生成，第三条线是真实观测如何回写。没有第三条线，模型会长期活在自己生成的未来里，错误会越滚越大。

DreamZero 和 π0.5 的差别可以压成一张表：

| 问题 | π0.5 | DreamZero |
| --- | --- | --- |
| 主要目标 | 从视觉语言输出动作 | 同时生成未来视频和未来动作 |
| 强调什么 | 高层子任务 + 连续 action chunk | video-action alignment |
| 典型风险 | 子任务错、动作执行错 | 未来视频错，动作也跟着错 |
| 系统需求 | 控制、安全、success checker | 实时推理、真实观测刷新、安全层 |

读 DreamZero 时要抓住一个矛盾：它让模型学会“动作导致什么视觉变化”，这很强；但如果未来视频预测错，它可能会非常一致地执行错误计划。因此 WAM 不能绕过安全层。

## 为什么还需要 planner、controller 和 safety filter

模型输出动作后，真实机器人通常还要经过一层或多层执行接口：

\[
\text{model action}
\rightarrow
\text{planner / controller}
\rightarrow
\text{safe robot command}.
\]

<!-- [补充] -->
| 符号 | 含义 | 取值范围 |
| --- | --- | --- |
| 公式整体 | 本公式表达的变量关系 | 按上下文定义 |


planner 负责找路径或技能序列，controller 负责高频跟踪和动力学约束，safety filter 负责速度、力、碰撞、工作空间和急停边界。

SayCan 的图很适合解释“会说”和“能做”的区别。LLM 给候选技能打“是否符合指令”的分数，affordance / value function 给技能打“当前能否执行”的分数，二者组合后才选下一步。

![SayCan LLM and affordance scores 原论文图](../assets/images/paper-figures/embodied-ai/saycan-figure-3-llm-affordance.png){ width="920" }

<small>图源：[Do As I Can, Not As I Say: Grounding Language in Robotic Affordances](https://arxiv.org/abs/2204.01691)，Figure 3。原论文图意：LLM 给候选技能打“是否符合指令”的分数，value functions 给技能打“当前环境是否可执行”的 affordance 分数，二者组合后选择下一步机器人动作。</small>

!!! note "图解：useful 和 possible 要一起看"
    左边语言模型更像“这个动作对任务有没有用”，右边 affordance 更像“这个动作在当前环境能不能做”。真实机器人需要两者相乘或共同约束：只 useful 可能会撞，只 possible 可能和任务无关。

## 数据和评测：闭环能不能进步

具身智能的数据不是一堆图片，也不是一堆孤立动作。它应该能回放一条 episode：

```text
instruction
-> camera frames + robot state
-> action chunk
-> controller execution
-> new observation
-> success / failure / progress label
```

Open X-Embodiment 这张图提醒我们：跨机器人、跨场景、跨任务的数据非常异质。不同机器人有不同相机、动作空间、夹爪、控制频率和任务分布。

![Open X-Embodiment dataset 原论文图](../assets/images/paper-figures/embodied-ai/open-x-embodiment-figure-1-dataset.png){ width="760" }

<small>图源：[Open X-Embodiment: Robotic Learning Datasets and RT-X Models](https://arxiv.org/abs/2310.08864)，Figure 1。原论文图意：统计 Open X-Embodiment 中不同 robot embodiment、scene、trajectory、skill 和 object category 的分布，展示跨机器人数据的异质性。</small>

!!! note "图解：数据多不等于接口统一"
    这张图的重点不是“数据量大”，而是“数据来源很不一样”。VLA / WAM 要泛化，必须处理跨 embodiment、跨相机、跨动作空间的数据契约；否则模型可能只在某个固定实验室里成功。

一个最小可用数据管线可以写成：

```text
3D 资产 -> 物理属性 -> 抓取候选 -> 场景布置
-> planner 轨迹 -> 仿真或真机执行 -> success checker
-> 失败原因 -> 回流训练集和 hard set
```

这里 success checker 很关键。它定义“什么叫成功”，也决定失败样本能不能被统计和回流。没有 success checker，模型训练出来也很难解释。

## 一个任务贯穿所有概念

以“把圆柱体放入尺寸匹配的罐子”为例：

| 阶段 | 输入 | 输出 | 容易失败在哪里 |
| --- | --- | --- | --- |
| 看见 | \(o_t\)：多视角图像、深度、机器人状态 | 物体位置、尺寸、候选容器 | 深度错、遮挡、尺度不准 |
| 理解 | \(\ell\)：任务语言 + 当前状态 | 目标圆柱和目标罐子 | 尺寸比较错、对象绑定错 |
| 选择动作 | \(o_{\le t}, q_t, \ell\) | \(A_{t:t+H-1}\) | 抓取姿态不稳、动作过长 |
| 预测后果 | \(z_t, A_{t:t+H-1}\) | 未来状态、\(r_t\)、\(c_t\)、\(d_t\) | 低估滑落或碰撞 |
| 执行控制 | action chunk + 安全约束 | 真实机器人命令 | IK 无解、碰撞、力过大 |
| 判断成败 | 新观测 + 状态 | success / failure / progress | checker 阈值错、状态不同步 |
| 回流数据 | 失败 episode | hard case、纠正轨迹 | 没有失败标签，无法改进 |

这就是具身智能的完整闭环。下一页建议读 [VLA、WAM 与世界模型地图](vla-wam-world-model-system-map.md)，它会把这些模块的边界讲清楚；再读 [一个任务跑通具身闭环](closed-loop-task-walkthrough.md)，用同一套符号把任务走完整。

## 推荐学习顺序

如果你现在还觉得散，按这个顺序读：

1. 本页：先建立闭环、符号和四类模型。
2. [相机、深度与机器人视觉](cameras-depth-and-robot-vision.md)：弄清单目、双目、RGB-D、内参、外参和点云。
3. [VLA、WAM 与世界模型地图](vla-wam-world-model-system-map.md)：把 VLA、world model、WAM、planner、controller 和数据引擎放到同一张系统图里。
4. [一个任务跑通具身闭环](closed-loop-task-walkthrough.md)：用具体任务把输入、输出、checker 和失败排查串起来。
5. [资产到轨迹：感知、抓取与数据管线](asset-perception-grasp-and-trajectory-pipeline.md)：理解训练数据从哪里来。
6. [规划、控制与安全](planning-control-and-safety.md)：理解为什么模型输出动作后还要接 planner、controller 和 safety filter。
7. [π0.5](../paper-deep-dives/embodied-ai/pi05.md)：理解 VLA 如何做开放世界长任务。
8. [DreamZero](../paper-deep-dives/embodied-ai/dreamzero.md)：理解 WAM 如何把未来视频和动作绑在一起。
9. [LingBot-World](../paper-deep-dives/world-models/lingbot-world.md)：理解视频生成模型如何变成交互世界模拟器。

## 检查标准

读完这一组页面后，你应该能回答：

1. \(o_t\)、\(q_t\)、\(\ell\)、\(a_t\)、\(A_{t:t+H-1}\)、\(z_t\)、\(r_t\)、\(c_t\)、\(d_t\) 分别是什么？
2. VLM、VLA、world model、WAM 的区别是什么？
3. action chunk 为什么不是完整 trajectory？
4. π0.5 为什么需要 high-level subtask？
5. DreamZero 为什么要同时生成视频和动作？
6. 为什么模型输出动作后仍要经过 planner、controller 和 safety filter？
7. 为什么 success checker 和失败回流要比大规模生成轨迹更早设计？

如果这些问题都能答清楚，具身智能的主线就立住了。

!!! tip "下一站"
    - 回到本专题入口：[具身智能](index.md)，确认这页在整条路线中的位置。
    - 按导航顺序继续：[VLA 数据、模型与评测路线](vla-data-model-and-evaluation-roadmap.md)。
    - 概念或符号卡住时，先查 [术语表](../glossary/index.md)，再回到当前页。
