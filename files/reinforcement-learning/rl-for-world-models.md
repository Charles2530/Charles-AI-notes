# 世界模型中的强化学习

世界模型不是“会生成未来画面”就够了。对决策系统来说，世界模型真正要回答的是：**如果我在当前状态做这个动作，未来会发生什么，这个未来是否值得执行。** 强化学习提供了三件工具：用 MDP/Bellman 表达长期后果，用 value/reward 判断候选未来，用 policy 或 planner 把高价值未来变成动作。

所以这篇不把 PlaNet、Dreamer、Decision Transformer、DDPO、Flow-GRPO、RLVR-World 和 Genie 当论文名堆起来，而是沿一条主线讲：从真实 transition 到 learned transition，再到 imagined rollout、视频生成后训练和闭环策略优化。

## 世界模型补的是 transition

在 MDP 里，真实环境 transition 是：

\[
P(s_{t+1}\mid s_t,a_t)
\]

它表示：给定当前状态 \(s_t\) 和动作 \(a_t\)，真实环境会产生下一状态 \(s_{t+1}\) 的概率分布。世界模型学习的是一个近似：

\[
\hat P_\theta(s_{t+1}\mid s_t,a_t)
\]

帽子 \(\hat{}\) 很重要：这是模型预测，不是真实世界。世界模型越有用，不是因为它能把未来画得越漂亮，而是因为它的预测能保留决策变量：物体位置、速度、接触、遮挡、任务进度、风险和可恢复性。

很多实用世界模型还会同时预测：

\[
(\hat s_{t+1},\hat r_t,\hat d_t,\hat u_t)=f_\theta(s_t,a_t)
\]

这里 \(\hat r_t\) 是奖励或任务进展，\(\hat d_t\) 是终止/失败信号，\(\hat u_t\) 是不确定性或风险。强化学习需要这些量，是因为 Bellman backup 要把“当前动作的后果”递归成未来价值。

| 世界模型输出 | 对 RL 的作用 | 常见失败 |
| --- | --- | --- |
| 下一状态 \(\hat s_{t+1}\) | 让 planner 或 actor 比较动作后果 | 视觉像但物理错，接触和遮挡丢失 |
| reward \(\hat r_t\) | 给 imagined rollout 提供即时反馈 | reward 与真实任务错位 |
| done/failure \(\hat d_t\) | 判断任务结束、碰撞或不可恢复 | 失败边界过晚或过早触发 |
| uncertainty/risk \(\hat u_t\) | 惩罚模型不确定区域，减少 exploitation | 不确定性未校准，策略过保守或过冒险 |

## RL 到底介入世界模型的哪一层

同样叫“world model + RL”，实际可能在优化四种不同对象：

| RL 介入点 | 被优化的对象 | 典型代表 | 要看的证据 |
| --- | --- | --- | --- |
| 用世界模型训练 policy | actor / planner | Dreamer、PlaNet、MuZero-style planning | closed-loop return、success rate、样本效率 |
| 用 RL 优化生成器 | 视频或图像生成 policy | DDPO、Flow-GRPO、DanceGRPO、视频 RLHF | reward、人评、视频质量、reward hacking |
| 用 RL 优化世界模型本身 | transition predictor | RLVR-World | decoded prediction metric、action sensitivity、下游 planner 收益 |
| 用生成世界训练 agent | agent 在可交互环境中探索 | Genie 2/3 这类 playable world | 动作是否改变环境、环境是否可重复评测、reward 是否可定义 |

这四层不能互相替代。视频生成 RLHF 可能让样本更符合审美和 prompt，却不保证动作因果；Dreamer 可以提升控制回报，却不一定生成高清未来视频；RLVR-World 能把 world model 的预测向可验证指标拉近，但它也需要 reward 覆盖真正重要的状态变量。

## PlaNet：先学 latent dynamics，再在线规划

PlaNet 是理解 model-based RL 的好入口。它不在像素空间里直接规划，而是先把观测编码成 latent belief，再用 RSSM 预测 latent future，最后用 CEM 搜索未来动作序列。

![PlaNet latent dynamics model designs 原论文图](../assets/images/paper-deep-dives/world-models/planet/figure-2-latent-dynamics.png){ width="860" }

<small>图源：[Learning Latent Dynamics for Planning from Pixels](https://arxiv.org/abs/1811.04551)，Figure 2。原图表达：比较 RNN、SSM 和 RSSM 三种 dynamics 结构；RSSM 同时保留 deterministic memory 和 stochastic latent state。本站读法：deterministic state 记历史，stochastic state 表达不确定性，二者合起来更适合部分可观测控制。</small>

PlaNet 的闭环可以写成：

```text
当前观测 -> 编码成 latent belief
  -> 采样多条未来 action sequences
  -> 用 RSSM 预测 latent future 和 reward
  -> 选择预测回报最高的动作序列
  -> 执行第一个动作
  -> 收到新观测后重新规划
```

这里的关键不是“模型会预测未来”，而是“每一步执行后重新观察并重新规划”。这会缓解模型误差累积：如果前一次预测错了，新 observation 可以把 belief 拉回真实状态附近。

PlaNet 的代价也很清楚：执行时要持续搜索。每个控制步都要比较多条候选动作序列，latency 和采样数量会限制它在高频机器人控制或大模型 rollout 服务中的使用。

## Dreamer：把在线搜索换成想象中训练 actor

Dreamer 保留 learned world model，但不再每一步都靠 CEM 在线搜索。它在 world model 里展开 imagined trajectories，用这些轨迹训练 actor 和 critic；执行时 actor 直接输出动作。

![Dreamer algorithm 原论文图](../assets/images/paper-deep-dives/world-models/dreamer/algorithm-1-dreamer.png){ width="820" }

<small>图源：[Dream to Control: Learning Behaviors by Latent Imagination](https://arxiv.org/abs/1912.01603)，Algorithm 1。原图表达：Dreamer 交替进行 dynamics learning、behavior learning 和 environment interaction。本站读法：真实经验主要用于训练 world model，策略更新大量发生在 latent imagination 中。</small>

Dreamer 把训练拆成三层：

| 层 | 学什么 | 用什么数据 |
| --- | --- | --- |
| representation / transition | latent state 和 dynamics | replay buffer 中的真实 observation/action |
| reward / continue | imagined rollout 的即时反馈和终止 | 真实 reward/done 监督 |
| actor / critic | 哪些动作让 imagined return 更高 | world model 中展开的 imagined trajectories |

DreamerV2 的 actor-critic 图更直观：

![DreamerV2 actor-critic 原论文图](../assets/images/paper-deep-dives/world-models/dreamerv2/figure-3-actor-critic.png){ width="860" }

<small>图源：[Mastering Atari with Discrete World Models](https://arxiv.org/abs/2010.02193)，Figure 3。原图表达：从真实 replay state 出发，在 learned world model 中展开 imagined trajectories，并用这些轨迹训练 actor 和 critic。本站读法：world model 提供 Bellman 里的下一状态和 reward，critic 估计未来价值，actor 选择让 imagined return 更高的动作。</small>

Dreamer 的效率来自“真实世界步数”被换成“latent rollout 步数”。如果机器人以 `20Hz` 收集真实交互，`245,760` 个真实 transition 需要：

\[
245{,}760 / 20 = 12{,}288s \approx 3.4h
\]

如果同样数量的 latent transition 在 GPU 上每步 `0.04ms`，只需约：

\[
245{,}760 \times 0.04\text{ms}\approx9.8s
\]

这就是 world model RL 的吸引力。但代价也在同一处：便宜 transition 是模型想象出来的，不是真实世界。如果每一步有 `1%` 的关键接触误判，15 步里至少一次误判的概率约为：

\[
1-0.99^{15}\approx14.0\%
\]

rollout 越长，actor 越可能学会利用模型漏洞。所以实践里常配合短 horizon、value bootstrap、不确定性惩罚、真实 replay 校准和闭环评测。

这就是 model exploitation。比如真实机器人推杯子时，杯子碰到桌沿会卡住；但世界模型没有学好接触边界，想象里杯子可以穿过桌沿继续前进。actor 在 imagined rollout 里会发现“猛推”得到高 reward，于是学出真实环境里会失败的动作。RL 给了世界模型很高的样本效率，也把模型错误变成了可被策略主动利用的漏洞。

## 为什么 open-loop 好看不等于能规划

世界模型有两种完全不同的评价方式：

| 评价方式 | 问的问题 | 不能证明什么 |
| --- | --- | --- |
| open-loop prediction | 未来帧、latent 或状态预测像不像真实数据 | planner 能不能用、动作是否真的影响未来 |
| closed-loop control | 接入策略或规划器后任务是否成功 | 预测图像一定自然、跨平台泛化 |

很多视频世界模型可以生成自然未来，但没有动作条件、reward/done 或 risk head。这样的模型可以是优秀的视频生成器或表征模型，却不能直接称为可规划世界模型。对决策最关键的是 action sensitivity：同一当前状态下，换不同动作，预测未来是否按动作改变。

一个典型失败链是：

```text
训练 loss 下降
-> 未来视频更平滑
-> open-loop 指标更好
-> 不同动作预测差异变小
-> planner 选不出真正更好的动作
-> closed-loop success 不升反降
```

所以世界模型论文至少要区分：是否有动作输入、是否预测 reward/done/risk、是否做 closed-loop eval、是否检查 model exploitation。

## 视频生成里的 RL：把去噪轨迹当成策略

视频生成里的 RL 不是让模型在真实环境中试错，而是把一次生成过程看成一条 trajectory。DDPO 的关键观点是：扩散模型从噪声逐步去噪到样本，这个过程可以被写成多步决策问题；最终视频或图像拿到 reward 后，用 policy gradient 提高高分去噪轨迹的概率。

一个简化的 MDP 写法是：

| RL 对象 | 扩散 / Flow 里的对应物 | 解释 |
| --- | --- | --- |
| state | \((x_t,t,c)\) | 当前 noisy latent、噪声时间和文本/图像/视频条件 |
| action | \(x_{t-\Delta t}\)、噪声项或 velocity 更新 | 采样器在当前步选择的下一 latent |
| policy | denoiser / velocity model + sampler | 给出下一步分布或确定性更新 |
| trajectory | 从噪声到视频的整条采样路径 | 不是机器人轨迹，而是生成轨迹 |
| reward | aesthetic、alignment、motion、OCR、verifier、人类偏好 | 通常在最终样本上打分，信用分配很粗 |

这解释了为什么视觉生成里的 PPO/GRPO 比文本更别扭。LLM 的 token softmax 天然给出 \(\log \pi(a_t\mid s_t)\)；扩散和 Flow Matching 的推理往往由采样器定义，尤其 Flow Matching 常用确定性 ODE。如果要做在线 RL，就必须先让采样器能产生多样候选，并且给每一步 transition 一个可计算的 logprob。

Flow-GRPO 的做法是把 Flow-ODE 转成带随机性的 SDE，用同一 prompt 下多样本 reward 做 GRPO。DanceGRPO 进一步把扩散和 rectified flow 统一到一个视觉生成 RL 框架，覆盖文本生成图像、文本生成视频和图生视频任务。它们共同说明一件事：**视频生成 RL 的 policy 不只是模型权重，还包括噪声初始化、采样方程、KL 约束和 reward model 输入质量。**

LongCat-Video 这类长视频系统把 multi-reward GRPO 放进 post-training：视觉质量、运动质量、文本对齐等 reward 一起参与优化。这对视频基础模型很重要，因为只优化单一审美 reward 容易牺牲运动、身份一致性或 prompt 细节。但它仍然是生成器偏好优化，不自动等于可交互世界模型。若没有动作条件、反事实动作分叉、risk/done 预测和 closed-loop agent 评测，就只能说“朝 world model 方向靠近”，不能说已经具备规划能力。

视频生成 RL 还特别容易出现三类错位：

| 错位 | 表现 | 为什么会被 RL 放大 |
| --- | --- | --- |
| reward 只看单帧美感 | 视频更精致但动作变慢、事件变少 | policy 会偏向容易拿高分的静态未来 |
| reward 不懂物理 | 接触、碰撞、因果状态变化仍然错 | 高 reward 样本可能只是视觉上合理 |
| 随机采样破坏分布 | SDE 注入噪声导致伪影或 ranking 偏差 | advantage 会追逐 reward model 的输入噪声偏差 |

Coefficients-Preserving Sampling 的提醒很实用：给 Flow Matching 注入随机性时，不能随便加噪声；采样过程仍要保持 scheduler 规定的样本系数和总噪声水平，否则 reward model 看到的是过噪 latent 解码出来的伪影。对视频 RLHF 来说，这个问题会和长时一致性、运动幅度、少步采样误差叠加，所以评测不能只看训练 reward 曲线。

## Decision Transformer：把轨迹当序列建模

并不是所有 RL 都必须显式做 Bellman backup。Decision Transformer 把 return-to-go、state、action 交替组织成序列，让 Transformer 在目标回报条件下预测动作。

![Decision Transformer architecture 原论文图](../assets/images/paper-figures/reinforcement-learning/decision-transformer-architecture.png){ width="920" }

<small>图源：[Decision Transformer](https://arxiv.org/abs/2106.01345)，Figure 1。原图表达：把 return-to-go、state、action 交替输入 GPT decoder，用目标回报条件化动作预测。本站读法：轨迹可以被当成序列数据建模，但这并不自动解决分布外动作和真实闭环问题。</small>

这条路线对 VLA 和 action token 很有启发：如果动作、观测和目标都能 token 化，就可以用序列模型学习“给定目标回报时应该怎样行动”。但它依赖数据覆盖。离线数据里没有的动作组合，模型仍然很难可靠外推。

## RLVR-World：用可验证奖励训练世界模型本身

传统 world model 多用 MLE、重建 loss、latent prediction loss 或 reward/done supervised loss。RLVR-World 的问题意识是：这些 surrogate objective 可能和下游任务指标错位。比如视频 MSE 低，不一定代表状态预测更准；画面更自然，不一定代表 action consequence 更可靠。

![RLVR-World concept 原论文图](../assets/images/paper-figures/reinforcement-learning/rlvr-world-concept.png){ width="920" }

<small>图源：[RLVR-World: Training World Models with Reinforcement Learning](https://arxiv.org/abs/2505.13934)，Figure 1。原图表达：左侧是用 MLE 等 surrogate objective 训练 world model，右侧是用 RLVR 直接优化 decoded prediction 的可验证任务指标。本站读法：RL 不只可以训练 policy，也可以用可验证指标反过来训练世界模型预测本身。</small>

这类训练可以抽成：

```text
状态 s + 动作 a
  -> 世界模型采样多个下一状态/未来预测
  -> 解码成结构化状态或视频
  -> 用 ground truth 和任务指标打分
  -> 用 PPO/GRPO 类方法提高高分预测概率
```

这里的关键仍然是 reward 是否真对应任务。如果 reward 只奖励像素相似，模型可能学会输出平均化未来；如果 reward 包含结构化状态、动作敏感性、风险和闭环成功，才更可能服务规划。

把它和视频生成 RLHF 对照，会更清楚：

| 路线 | reward 打在哪里 | 优化目标 | 最容易越界的说法 |
| --- | --- | --- | --- |
| 视频生成 RLHF / GRPO | 最终视频样本或片段 | 人类偏好、清晰度、运动、文本对齐 | “视频更好看，所以能规划” |
| RLVR-World | decoded transition prediction | 状态预测、网页状态、机器人轨迹等可验证指标 | “预测指标更好，所以真实闭环一定更好” |
| Dreamer-style RL | imagined rollout 的 reward/value | policy 在环境里的长期回报 | “能控制，所以视频模拟一定真实” |

RLVR-World 的价值在于把 world model 的训练目标从 MLE 这类 surrogate objective 往任务指标挪了一步。它尤其适合有 verifier 的世界模型：文本游戏状态能被规则检查，网页状态能被 DOM/任务指标检查，机器人轨迹可以用结构化状态或视觉 token 指标检查。但如果 verifier 没覆盖长尾风险，RL 仍然会优化它能看见的局部指标。

## 世界模型中的 RL 有六种增量

| 增量 | 它带来什么 | 主要风险 |
| --- | --- | --- |
| 数据收集闭环 | policy 把数据推向任务相关状态 | 数据分布变窄，失败样本不足 |
| 长期信用分配 | reward/value 把未来结果反馈到早期动作 | value target 错误沿 Bellman 放大 |
| 想象训练 | 用 latent rollout 减少真实试错 | model exploitation |
| 反事实规划 | 比较“如果换动作会怎样” | action sensitivity 不足 |
| 任务指标对齐 | 用可验证 reward 直接优化预测质量 | reward 只覆盖局部指标 |
| 生成器偏好对齐 | 让视频/图像样本更符合人类或模型偏好 | 画质提升掩盖因果、风险或动作失败 |

## 交互式视频世界模型怎样承接 RL

Sora 的技术报告把视频生成称为通向物理世界模拟器的路径：大规模视频模型能表现出一定长程一致性、对象持久性和简单交互模拟；报告同时明确列出物理交互、物体状态变化和长视频一致性的限制。这个边界很关键：Sora 式模型可以提供强视觉先验，但缺少稳定的动作接口和 reward/done/risk 头时，还不是能直接训练 agent 的 RL 环境。

Genie 2/3 把边界往交互式世界推进了一步：环境能响应键盘或鼠标动作，agent 可以在生成世界中获得下一观察。这样的系统更接近 RL 环境，但仍要补三件事才适合严肃训练：同一动作在同一状态下是否可复现，任务 reward 是否可定义和自动计算，模型错误会不会被 agent 系统性利用。

V-JEPA 2-AC 代表另一条更轻的路线：先用大规模视频做 action-free 表征预训练，再用少量机器人轨迹把 latent dynamics 接到动作条件规划。这里 RL 不一定表现为 PPO/GRPO，而是表现为 planner 在 latent future 中比较候选动作。它提醒我们：视频预训练提供动态先验，真正用于行动时还要接上动作数据、goal cost、短 horizon 重规划和真实观测刷新。

可以用下面的栈判断一个视频世界模型是否真的能服务 RL：

```text
视频生成能力
  -> 长时一致性和对象持久性
  -> 动作条件和反事实分叉
  -> reward / value / risk / done 可读出
  -> agent 能闭环行动并重新观测
  -> 真实环境或高保真基准证明任务收益
```

栈越往下，证据越接近决策。只到第一、二层，主要还是视频生成或世界先验；到第三、四层，才开始变成可交互 world model；第五层才是 RL 系统真正关心的收益。

## 和 VLA 的连接

VLA 学的是动作策略：

\[
\pi_\theta(a_t\mid o_{\le t},l)
\]

给定历史观测 \(o_{\le t}\) 和语言指令 \(l\)，输出动作 \(a_t\)。世界模型学的是动作后果：

\[
p_\theta(o_{t+1:t+H},r,d\mid o_{\le t},a_{t:t+H-1},l)
\]

给定历史观测、未来候选动作和语言指令，预测未来观测、奖励和终止信号。RL 把两者接起来：

1. 用 world model 预测候选动作后果。
2. 用 reward/value/risk 判断哪条未来更好。
3. 更新 policy，使它更偏向高回报低风险动作。
4. 用真实闭环数据修正 world model 的盲点。

机器人数据常常来自历史实验、遥操作或离线数据集，不能像游戏一样无限探索。Offline RL 的教训很直接：不要让 value 对数据里没支持的动作胡乱乐观；VLA + world model 也一样，需要 support 约束、失败回放、真实执行校准和安全门禁。

![Offline RL robotic grasping data](../assets/images/paper-figures/reinforcement-learning/offline-rl-qtopt-robot-grasping.png){ width="420" }

<small>图源：[Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems](https://arxiv.org/abs/2005.01643)，Figure 7。原图表达：大规模机器人抓取数据收集与离线/在线训练设置。本站读法：真实机器人数据昂贵，所以世界模型和 offline RL 都必须认真处理数据覆盖与分布外动作。</small>

## 读论文时问这九个问题

| 问题 | 看哪里 |
| --- | --- |
| 模型预测的是 pixel、latent、state，还是 reward/done/risk | 模型结构图、loss、输出头 |
| 动作是否真的进入 transition | action-conditioned ablation、counterfactual action |
| 生成模型的 policy / logprob 是否定义清楚 | sampler、ODE/SDE、old logprob、KL、GRPO objective |
| imagined rollout 多长 | horizon ablation、value bootstrap、error accumulation |
| 是否有 closed-loop eval | 机器人/控制任务 success、cost per success |
| 视频指标是否被误当成世界模型证据 | FVD/VBench、人评、action sensitivity、planner ablation |
| 是否检查 model exploitation | failure replay、uncertainty penalty、真实回放校准 |
| reward 是否和任务一致 | reward definition、verifier、分桶指标 |
| 系统成本是否可接受 | rollout latency、candidate 数、GPU-hour、batch/KV 复用 |

## 外部精读

- [World Models](https://worldmodels.github.io/)：Ha & Schmidhuber 的原始项目页，用很直观的方式说明用内部模型做控制。
- [Learning Latent Dynamics for Planning from Pixels](https://arxiv.org/abs/1811.04551)：PlaNet，理解 latent dynamics + online planning。
- [Dream to Control](https://arxiv.org/abs/1912.01603)：Dreamer，理解 latent imagination 训练 actor-critic。
- [DreamerV3](https://arxiv.org/abs/2301.04104)：理解 world model + actor-critic 如何在多个控制域中用同一配置稳定训练。
- [Decision Transformer](https://arxiv.org/abs/2106.01345)：把 RL 轨迹组织成序列建模问题。
- [Training Diffusion Models with Reinforcement Learning](https://arxiv.org/abs/2305.13301)：DDPO，把扩散去噪过程写成多步决策问题。
- [Flow-GRPO](https://arxiv.org/abs/2505.05470)：把确定性 Flow Matching 采样改造成适合在线 RL 的随机采样过程。
- [DanceGRPO](https://arxiv.org/abs/2505.07818)：统一扩散和 rectified flow 的视觉生成 GRPO，并覆盖视频任务。
- [LongCat-Video Technical Report](https://arxiv.org/abs/2510.22200)：看长视频生成系统如何使用 multi-reward GRPO，但也要保守理解其 world model claim。
- [Coefficients-Preserving Sampling](https://arxiv.org/abs/2509.05952)：理解 Flow-RL 中随机性、scheduler、logprob 和 reward 可靠性的关系。
- [RLVR-World](https://arxiv.org/abs/2505.13934)：用可验证 reward 训练世界模型预测本身。
- [Sora technical report](https://openai.com/index/video-generation-models-as-world-simulators/)：看视频生成与世界模拟器的联系和限制。
- [Genie 2](https://deepmind.google/blog/genie-2-a-large-scale-foundation-world-model/) 与 [Genie 3](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/)：理解 action-controllable / real-time interactive world model 的官方边界。
- [V-JEPA 2](https://ai.meta.com/research/publications/v-jepa-2-self-supervised-video-models-enable-understanding-prediction-and-planning/)：理解 action-free 视频预训练如何接到 action-conditioned latent planning。

## 读完以后怎么判断

强化学习让世界模型从“预测器”变成“决策工具”，也让视频生成器可以直接朝偏好指标优化：transition 预测动作后果，reward/value 衡量未来好坏，policy 或 planner 选择动作，生成器则把采样轨迹变成可优化 policy。真正的难点不在于生成未来，而在于预测必须 action-sensitive、reward 必须任务相关、logprob 和采样器必须定义清楚、rollout horizon 必须受控，最后还要用 closed-loop 证明它真的改善决策。

## 相关阅读与下一步

- 外部材料：[Spinning Up：Policy Optimization](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)。
- 外部材料：[PPO 论文](https://arxiv.org/abs/1707.06347)。
- 外部材料：[Soft Actor-Critic 论文](https://arxiv.org/abs/1801.01290)。
- 站内下一步：[强化学习专题](index.md)。
- 站内下一步：[MDP、价值函数与 Bellman](mdp-value-bellman.md)。
- 站内下一步：[世界模型中的强化学习](rl-for-world-models.md)。
