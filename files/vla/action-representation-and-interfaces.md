# VLA 动作表示与控制接口：模型输出怎样变成机器人动作

> **这篇回答的问题。** 如何理解“VLA 动作表示与控制接口”背后的核心机制、适用边界和下一步阅读路径。

VLA 最容易被讲成“把语言、图像和动作都 token 化”。这句话只讲到训练格式，没有讲到执行。真正要追的是：模型输出的东西，怎样变成机器人在某个坐标系、某个频率、某个安全边界内执行的动作。

机器人动作不是文本答案。它可能是末端位姿增量、关节角、速度、夹爪开合、未来一段轨迹、离散 action token，或者高层技能名。动作接口一旦不清楚，模型看起来会“懂任务”，真实控制时却会抖、慢、撞、抓不稳。

## 动作接口是一份协议

一个最小 VLA 闭环可以拆成：

```text
vision / language / proprioception
  -> policy
  -> action representation
  -> adapter / controller / safety filter
  -> robot
  -> new observation
```

policy 可以是 Transformer、VLM、diffusion policy、flow policy 或层级策略。controller 负责把目标位姿、关节命令或轨迹变成真实电机控制。中间的 action representation 必须说明五件事：

| 接口问题 | 必须说清什么 | 如果不清楚会怎样 |
| --- | --- | --- |
| 坐标系 | base、camera、end-effector、world/task frame | “向左”在模型和机器人里不是同一方向 |
| 变量 | 末端位姿、关节、速度、夹爪、技能 | 模型输出无法被 controller 接住 |
| 时间 | 单步动作、action chunk、技能持续时间 | 控制频率变了，动作尺度也变了 |
| 精度 | 连续值、离散 bin、codebook token、latent | 离散误差或回归噪声会进入执行 |
| 执行 | IK、轨迹跟踪、限幅、碰撞检查、急停 | 模型给出“合理”动作也可能不可执行 |

所以 VLA 的动作层不是一个小尾巴，而是把“理解任务”和“执行任务”接起来的接口层。读论文时，动作协议比模型名更先看。

## 最小动作向量藏着很多假设

机械臂操作里常见动作是末端执行器增量：

\[
a_t=
[\Delta x_t,\Delta y_t,\Delta z_t,\Delta r_t,\Delta p_t,\Delta \psi_t,g_t]
\]

这里前三项是位置增量，中间三项是姿态增量，可以是 roll/pitch/yaw、axis-angle、quaternion 或 6D rotation，\(g_t\) 是夹爪开合。这个向量看起来只有 7 维，但每一项都藏着接口假设：\(\Delta x_t\) 是机器人底座坐标向前，还是相机画面向右？\(g_t=1\) 是张开还是闭合？动作每 100ms 执行一次，还是每 20ms 执行一次？

相对动作常写成：

\[
T_{t+1}=T_t\cdot\Delta T_t
\]

这里 \(T_t\) 是当前末端位姿，\(\Delta T_t\) 是策略输出的相对位姿变化。它的直觉是：模型不直接预测“去世界坐标 \((0.41,0.12,0.28)\)”，而是预测“从当前手的位置向前一点、向下小半格、转一点”。

相对动作通常比绝对动作更容易迁移，因为不同桌面、机器人基座和相机安装带来的全局偏移更少。但它也有边界：状态估计错了会累积，控制频率变了会改变速度，外参错了会把视觉方向映射成错误运动。

## 动作层级不能混在一起

动作可以分成四层。层级不同，模型学到的东西和部署风险都不同。

| 层级 | 例子 | 优势 | 风险 |
| --- | --- | --- | --- |
| 高层技能 | `open drawer`, `pick cup` | 语义清楚，适合规划 | 需要技能库或低层策略接住 |
| 末端位姿 / waypoint | \(\Delta x,\Delta y,\Delta z,\Delta R,g\) | 接近 manipulation 语义，跨机器人相对友好 | 坐标系、IK 和 controller 很关键 |
| 关节位置 / 速度 | \(q_{t+1}\), \(\dot q_t\) | 贴近硬件执行 | 跨 embodiment 难，学习负担大 |
| latent skill / action code | skill vector、action codebook | 可压缩复杂短技能 | 可解释性和安全边界弱 |

RT-2 把动作写成 token，ACT 一次输出短窗口动作，Diffusion Policy 生成未来动作序列，Octo 用 diffusion decoding 支持多模态动作分布。它们不是“动作头”的同义词，而是不同控制接口。

## RT-1 / RT-2：把连续动作离散成 token

RT-1 和 RT-2 的关键不是单纯“机器人用了 Transformer”，而是把动作离散化，让模型可以像预测 token 一样预测动作。

连续动作离散化可以写成：

\[
u_t^j=\operatorname{bin}_j(a_t^j)
\]

\[
p(a_t\mid o_{\le t},l)
\approx
\prod_j p(u_t^j\mid o_{\le t},l,u_t^{<j})
\]

这里 \(a_t^j\) 是动作第 \(j\) 个连续维度，\(\operatorname{bin}_j\) 把它映射到离散桶 \(u_t^j\)，\(o_{\le t}\) 是观测历史，\(l\) 是语言指令。第二行表示：模型不直接回归一个连续向量，而是逐个生成离散动作 token。

![RT-2 overview](../assets/images/paper-deep-dives/embodied-ai/rt2/figure-1-overview.png){ width="920" }

<small>图源：[RT-2](https://arxiv.org/abs/2307.15818)，Figure 1。原图表达 web-scale VQA / VLM 训练和 robot action token 如何被 co-fine-tune 到同一个 VLA 框架里。本文用它说明：VLA 的关键不是“多模态输入”，而是让动作进入可生成接口，并最终被真实机器人执行。</small>

token 化的好处是训练接口统一：文本、视觉和动作都可以进入序列建模。代价是分桶会带来量化误差，精细插接、旋拧、力控和小幅姿态调整可能需要更高分辨率或连续控制头。

Open X-Embodiment / RT-X 进一步说明了接口标准化的重要性。它把不同机器人数据对齐到共同动作口径，例如相对 gripper frame 的 7 维动作。这个细节很关键：跨机器人泛化不是把所有轨迹直接混起来，而是先让观测、动作、单位、坐标和时间戳有可训练的共同语义。

## ACT：一次输出一段短时动作

很多 manipulation 任务不是每一帧都重新决定“下一毫米怎么走”。抓杯子、插电池、拉抽屉、折布都需要短时间连续动作。Action chunk 的做法是让策略一次输出未来 \(H\) 步：

\[
\hat A_{t:t+H-1}=\pi_\theta(o_{\le t},l)
\]

这里 \(\hat A_{t:t+H-1}\) 是未来一段动作，不是单个动作。执行时系统可以每隔一小段时间重新预测，把多个重叠 chunk 平滑融合。

ACT 的 temporal ensembling 可以写成：

\[
\tilde a_t=\sum_{k=0}^{K}w_k\hat a_t^{(t-k)},\qquad \sum_k w_k=1
\]

这里 \(\hat a_t^{(t-k)}\) 表示第 \(t-k\) 次推理对当前时刻 \(t\) 的动作预测，\(w_k\) 是融合权重。白话说，不要让某一次模型输出的噪声立刻造成机械臂抖动，而是把多次预测揉成更平滑的执行命令。

![ALOHA ACT algorithm](../assets/images/paper-figures/vla/aloha-act-algorithm-2x.png){ width="760" }

<small>图源：[Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705)，ACT 方法图。原图表达 policy 根据当前观测输出一段 action sequence，并用 temporal ensembling 平滑执行。本文用它说明：action chunk 是短时控制接口，不是完整任务计划。</small>

Action chunk 的代价也很清楚：如果前几步已经偏了，后面 chunk 可能继续执行错误。因此 chunk 不能开环到底，必须配合重新观察、chunk overlap、安全过滤和失败恢复。它解决的是短时连贯性，不替代闭环。

## Diffusion Policy：动作也可以是一个生成分布

动作不一定只有“回归一个向量”或“生成离散 token”两种形式。Diffusion Policy 把未来动作序列看成要从噪声里生成的对象：

\[
A_{t:t+H-1}\sim p_\theta(A\mid o_{\le t},l)
\]

这里 \(A_{t:t+H-1}\) 是未来 \(H\) 步连续动作序列。模型不是输出单一平均动作，而是学习一个条件动作分布。对多模态任务很重要：同一个杯子可以从左侧抓，也可以从右侧抓；直接回归均值可能得到一个两边都不好的中间动作。

![Diffusion Policy input-output](../assets/images/paper-figures/vla/diffusion-policy-input-output-padded.jpg){ width="880" }

<small>图源：[Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)。原图表达 Diffusion Policy 根据视觉观测和机器人状态生成未来动作序列。本文用它说明：连续动作接口可以用生成式模型表达多峰动作分布，并通过 receding horizon 控制落到闭环执行。</small>

Diffusion Policy 的关键不是“用了扩散模型所以更高级”，而是它把动作分布、连续动作序列和 receding horizon 控制接起来。它仍然需要 controller、动作限幅、执行频率和真实反馈；生成出的 action sequence 不是电机命令的终点，而是下一轮闭环的候选轨迹。

## 坐标系比模型名更重要

动作表示一定要回答坐标系：

| 坐标系 | 动作含义 | 常见风险 |
| --- | --- | --- |
| robot base frame | 相对机器人底座移动 | 机器人安装位置变了要重新对齐 |
| camera frame | 相对相机视角移动 | 相机外参、视角和深度误差会影响动作 |
| end-effector frame | 相对当前手爪移动 | 局部操作自然，但全局路径要另规划 |
| world / task frame | 相对桌面、物体或场景坐标 | 需要可靠标定和物体 pose |

很多 VLA 失败不是语言没听懂，而是动作接口不对齐。训练数据里动作在 gripper frame，部署时 adapter 当成 base frame；相机外参变了但动作转换没更新；夹爪开合归一化方向反了；控制频率从 10Hz 变成 30Hz 后同样的 \(\Delta x\) 变成三倍速度。

旋转也会出问题。Euler angle 可读但有奇异性，quaternion 稳定但有归一化和双覆盖问题，axis-angle 紧凑但大角度不直观，6D rotation 连续性好但需要转换。论文里一句“predict end-effector pose”不够，必须看它如何表示 rotation、如何归一化、如何进入 controller。

## 频率、延迟和控制器决定闭环质量

VLA policy 不可能无限快。假设模型推理 200ms，controller 50Hz，机器人执行还有机械延迟，那么一次 policy 输出到下一次真实反馈之间已经跨过多个低层控制周期。

这会带来三个问题：

| 问题 | 现象 | 常见处理 |
| --- | --- | --- |
| 推理慢 | 机器人等模型，动作断续 | action chunk、小模型蒸馏、异步推理、视觉缓存 |
| 控制频率不匹配 | 动作尺度和速度异常 | 明确定义 action timestep 和 controller interpolation |
| 反馈滞后 | 看到的状态已经过期 | 状态估计、预测补偿、短 horizon 重规划 |

低层 controller 负责轨迹跟踪、阻抗控制、速度/加速度限幅、IK、碰撞检查和急停。VLA 输出的动作最好被看成“候选目标”或“短时轨迹”，不是直接下发电机的最终命令。

## FAST：压缩动作不能丢掉接触时机

FAST 这类现代 action tokenizer 解决的是另一个痛点：机器人动作频率高、连续维度多，如果直接把每个低层动作都塞进自回归序列，token 太多、延迟太高、长程依赖也难学。

FAST 用 frequency-space action sequence tokenization 压缩高频连续动作，让 VLA 更像语言模型一样生成更短的动作 token 序列。它的工程意义不是“动作 token 越少越好”，而是把高频轨迹压到模型能学、能推理、还能还原接触细节的范围内。

压缩动作和压缩文字不一样。文字 token 少几个可能还能读懂，动作 token 少几个可能就是速度、夹爪力度或接触时机丢了。判断 action tokenizer 是否成功，不能只看 token compression ratio，还要看闭环成功率、动作平滑度、接触事件、延迟和失败恢复。

## 读 VLA 动作论文时先问这些

| 问题 | 为什么重要 |
| --- | --- |
| 动作在哪个层级 | 高层技能、末端位姿、关节命令和 latent skill 的泛化边界不同 |
| 坐标系和单位是什么 | 坐标错了，模型再强也会执行错方向 |
| 连续还是离散 | 离散 token 易接序列模型，连续控制精度更自然 |
| 单步还是 chunk | chunk 提升连贯性，也降低反馈频率 |
| 谁负责执行 | controller、IK、安全层和限幅决定动作是否真能落地 |
| 失败如何反馈 | 没有重观察和恢复，小错会被动作接口放大 |
| 是否跨机器人 | 不同 embodiment 的动作空间和传感器不同，不能只混数据 |

这些问题比模型名更重要。一个大 VLA 如果动作接口不清楚，部署风险可能比一个小得多但接口明确的策略更高。

最后看动作链路：模型输出什么，单位是什么，谁执行，多久重规划，失败如何停。只有这条链路清楚，VLA 才不只是会看图说话，而是真的能把意图变成稳定动作。

## 外部精读

- [RT-1](https://arxiv.org/abs/2212.06817)：理解真实机器人 Transformer policy、动作 token 和大规模机器人数据的早期接口。
- [DeepMind RT-2 blog](https://deepmind.google/blog/rt-2-new-model-translates-vision-and-language-into-action/) 与 [RT-2 paper](https://arxiv.org/abs/2307.15818)：理解 VLM 如何通过 action token 接入机器人控制。
- [Open X-Embodiment / RT-X](https://robotics-transformer-x.github.io/)：看跨机器人数据如何统一 gripper-frame 动作口径。
- [ALOHA / ACT](https://arxiv.org/abs/2304.13705)：理解 action chunk 和 temporal ensembling。
- [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)：理解连续动作分布、receding horizon 和视觉条件策略。
- [Octo](https://octo-models.github.io/)：看通用机器人策略如何支持不同观测、动作空间和 diffusion decoding。
- [FAST action tokenizer](https://arxiv.org/abs/2501.09747)：看现代 VLA 如何把高频连续动作压成更高效的离散 token。

## 相关阅读与下一步

- 外部材料：[RT-2 官方博客](https://deepmind.google/discover/blog/rt-2-new-model-translates-vision-and-language-into-action/)。
- 外部材料：[Open X-Embodiment 论文](https://arxiv.org/abs/2310.08864)。
- 外部材料：[ACT 论文](https://arxiv.org/abs/2304.13705)。
- 站内下一步：[VLA 专题](index.md)。
- 站内下一步：[VLA 数据与策略学习](data-and-policy-learning.md)。
- 站内下一步：[VLA 闭环恢复与失败分析](closed-loop-recovery-and-failure-analysis.md)。
