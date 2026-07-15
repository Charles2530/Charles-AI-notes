# 世界模型基础：状态、动力学、观测与决策

世界模型学习环境怎样随时间和动作变化。它不必还原世界的全部细节，而要形成一套对下游消费方足够的内部状态和转移规律：规划器需要比较候选动作，策略学习需要在想象轨迹中估计回报，生成式模拟器需要产生连贯可见的未来，风险系统需要识别不确定和危险尾部。

## 本文解决什么问题

本文建立世界模型的共同接口：观测与状态为何不同；历史怎样形成 belief；动作条件转移怎样定义；随机未来怎样表示；训练目标为什么取决于消费方；latent dynamics、预测式表征和生成式模拟之间是什么关系。

本文讨论连续环境中的学习型世界模型。经典物理模拟器、纯语言环境模型和离散 token 世界模型也可以放入同一框架，但状态、转移核和误差来源不同。

## 前置知识

- [随机过程与马尔可夫性](../prerequisite-math/stochastic-processes-and-markov.md)：状态、转移与部分可观测性。
- [潜变量、似然与生成目标](../prerequisite-math/latent-likelihood-and-generative-objectives.md)：潜变量与变分目标。
- [MDP、价值函数与 Bellman](../reinforcement-learning/mdp-value-bellman.md)：状态、动作、奖励和策略。
- [世界模型与具身智能总入口](../world-models-and-embodied-ai/index.md)：模型在完整系统中的位置。

统一记号：

| 符号 | 含义 |
| --- | --- |
| \(o_t\) | 第 \(t\) 时刻可观测到的传感器数据 |
| \(a_t\) | 在 \(t\) 时刻执行的动作 |
| \(h_t=(o_{\le t},a_{<t})\) | 截至当前的观测与动作历史 |
| \(z_t\) | 模型内部状态或 belief 表示 |
| \(r_t,d_t\) | 奖励与终止/continuation 信号 |
| \(\pi\) | 消费世界模型的策略或规划器 |

## 核心机制

### 观测不等于环境状态

相机图像只记录当前视角，可能受到遮挡、运动模糊和有限视野影响；机器人关节状态不包含物体内部属性；单帧也无法直接判断速度和动作历史。世界模型因此常从历史估计 belief：

\[
z_t\sim q_\phi(z_t\mid o_{\le t},a_{<t}).
\]

\(z_t\) 是模型为预测和决策形成的充分表示，不要求与真实物理状态一一对应。它是否“好”，取决于下游是否能从中得到需要的信息：

- 控制任务需要保留动作相关动力学和奖励因素；
- 视频生成需要保留外观、遮挡和时空一致性；
- 风险规划需要保留不确定性与罕见事件；
- 物理推理需要保留对象、关系和变化规律。

只用单帧编码器得到 feature，并不自动得到具有时间递推能力的状态。

### 动力学回答动作之后会怎样

动作条件转移可以写成：

\[
\hat z_{t+1}\sim p_\theta(z_{t+1}\mid z_t,a_t).
\]

没有动作条件的 \(p(z_{t+1}\mid z_t)\) 可以学习视频时间规律，但无法区分“环境自行变化”和“agent 的动作造成变化”。要做反事实规划，模型至少需要对不同候选动作给出可区分的后果。

动作接口必须包括语义和时间：

- 动作是关节目标、末端增量、速度、力矩还是技能；
- 动作对应哪一个坐标系；
- 它在多长时间内执行；
- 观测与动作时间戳怎样对齐；
- 控制器是否真正实现了数据中记录的命令。

动作错位会让模型学成相关性预测器，而不是可用于控制的动力学。

### 随机状态表示不可确定的未来

同一观测和动作可能对应多个未来：物体摩擦未知、其他 agent 行为不同、传感器噪声或场景本身部分可观测。点预测会把多种未来平均在一起，产生模糊观测或过度自信的 latent。

随机世界模型用分布表达：

\[
p_\theta(z_{t+1}\mid z_t,a_t),
\]

并可通过 stochastic latent、mixture、离散 code、diffusion 或 ensemble 表示不确定性。需要区分：

- aleatoric uncertainty：环境或观测本身的随机性；
- epistemic uncertainty：训练数据不足导致的模型未知。

二者影响规划的方式不同。不可约随机性需要比较未来分布，模型未知则常需要拒绝、探索或回到真实观测。

### 观测、奖励和终止头把状态接到消费方

latent transition 本身无法说明状态是否对任务有用。常见输出头包括：

\[
\hat o_t\sim p_\theta(o_t\mid z_t),\qquad
\hat r_t=r_\theta(z_t,a_t),\qquad
\hat d_t=d_\theta(z_t,a_t).
\]

观测 decoder 提供重建和可视化监督，但会迫使 latent 保留背景纹理等未必与决策有关的信息。reward 与 continuation 直接约束状态保留任务因素，却可能过拟合当前 reward 定义。表示预测目标减少像素负担，但需要证明预测的表示包含动作后果。

因此“是否重建像素”是目标选择，不是世界模型的定义。

### 训练阶段和 rollout 阶段不同

训练时模型通常读取真实历史或 posterior state：

```text
history + current observation -> posterior state z_t
z_t + recorded action a_t -> predict next state / observation / reward
compare with recorded future
```

rollout 时未来真实观测不可用：

```text
initial belief z_t
for candidate action in a planned sequence:
    z_next ~ transition(z, action)
    predict reward / risk / observation
    z = z_next
```

训练的一步预测可以借助每一步真实观测纠错；开放环 rollout 必须消费自己的预测，误差会改变后续状态分布。世界模型评测必须包含多步 rollout、动作反事实和 closed-loop 结果。

### 三条主要路线

#### Latent dynamics 与控制

PlaNet、Dreamer 等路线学习紧凑 latent dynamics。RSSM 结合确定性 recurrent state 与随机 latent，使历史记忆和每一步不确定信息共同决定 belief。PlaNet 在 latent 中在线规划；Dreamer 在 imagined trajectories 中训练 actor 与 critic。

这条路线的目标不是生成最清晰视频，而是用较低成本提供可规划状态、奖励和 continuation。

#### 预测式表征

JEPA 类方法预测目标片段的表示，而不是完整像素。它减少对纹理和不可预测细节的建模压力，强调能够概括对象与动态的 latent。若要进入规划，还需动作条件模型、目标或 value 接口，以及证明 latent 差异能可靠区分动作后果。

#### 生成式世界模拟

视频世界模型继承强生成模型的时空表示与视觉质量，并加入动作或交互条件。它可以作为可见仿真、agent 训练环境或合成数据源。真正的门槛是可控性和闭环一致性：相同初始状态下改变动作应产生符合动作语义的不同未来。

### 世界模型怎样进入决策

世界模型只有被消费时才产生控制价值：

- MPC 展开候选动作序列，按 reward、constraint 和 uncertainty 评分，每次只执行第一段；
- Dreamer 式方法在 latent imagination 中训练 actor，摊销在线规划成本；
- planning as inference 把目标和约束写成对轨迹后验的推断；
- 生成式模拟器提供训练环境、反事实和 hard cases；
- safety layer 使用风险头或 ensemble disagreement 拒绝模型未知区域。

同一个预测模型不一定适合所有消费方。规划器关心候选排序，数据生成器关心覆盖与标签，视频展示关心视觉一致性。

## 常见误解与失效边界

**把视频生成模型直接称为世界模型。**如果未来不受动作控制，或动作变化不能稳定改变后果，它尚不能支持反事实决策。

**认为 latent 越小越接近真实状态。**压缩只减少维度，不保证保留动作、奖励、接触和风险信息。

**用一步 loss 代表长时 rollout。**一步 teacher-forced 预测不断读取真实历史；开放环会进入模型自己制造的状态分布。

**把清晰观测当成正确动力学。**生成模型可以产生视觉合理但物理错误的未来；规划更关心动作排序和约束事件。

**认为世界模型可以替代 controller。**世界模型预测后果，controller 负责在真实动力学和控制周期下跟踪参考；二者接口不同。

**忽略模型利用自己的错误。**规划器可能搜索到预测模型的漏洞，选择在模型中高回报、现实中失败的动作。需要不确定性、约束和真实闭环回放。

## 检查理解

1. 为什么单帧观测通常不足以构成决策状态？
2. 无动作视频预测与动作条件世界模型的反事实能力差在哪里？
3. posterior training 和 prior rollout 分别能看到哪些信息？
4. 像素 reconstruction loss 与控制有用性为什么不是同一指标？
5. 世界模型进入 MPC 和进入 Dreamer actor training 的消费方式有什么不同？
6. 如何检测规划器正在利用模型误差？

## 相关阅读与下一步

- 潜变量控制：[RSSM、Dreamer 与规划](rssm-dreamer-and-planning.md)。
- 表示路线：[Masked / JEPA 与潜变量预测](masked-jepa-and-latent-prediction.md)。
- 可交互生成：[生成式模拟与视频世界模型](generative-simulation-and-video.md)。
- 动作接口：[WM / WAM / VAM](wm-wam-vam-and-action-conditioned-modeling.md)。
- 评测边界：[世界模型评测与失效模式](evaluation-and-failure-modes.md)。
- 原始来源：[World Models](https://arxiv.org/abs/1803.10122)、[PlaNet](https://arxiv.org/abs/1811.04551)、[Dreamer](https://arxiv.org/abs/1912.01603)、[V-JEPA](https://arxiv.org/abs/2404.08471)、[Genie](https://arxiv.org/abs/2402.15391)。
- 返回：[世界模型专题](index.md)。
