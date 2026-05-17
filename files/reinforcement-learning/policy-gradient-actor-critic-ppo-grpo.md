# Policy Gradient 到 PPO / GRPO

这页从“采样为什么能学习”讲到 PPO 和 GRPO。目标不是推完所有数学细节，而是让你理解：为什么大模型后训练需要 rollout、reward、advantage、reference/KL，以及 verl 这类框架为什么围绕 actor、rollout、critic 和 reward 组织。

!!! note "读法定位"
    这页先回答“Policy Gradient 到 PPO / GRPO”在「强化学习」里的位置：它解决什么局部问题，依赖哪些前置，最后会影响哪类工程或研究判断。
    **前置**：先懂概率、优化和基本训练曲线；PPO/GRPO 不必先记公式，先抓更新对象。 必要时先回 [强化学习入口](index.md)、[基础知识](../foundations/index.md) 或 [术语表](../glossary/index.md)。
    **主线关系**：把状态、动作、奖励、策略优化和世界模型闭环连起来，看模型如何通过反馈改进决策。

!!! tip "主线定位：降低标注和试错成本，但增加 rollout 成本"
    PPO/GRPO 在世界模型高效训练里负责把“真实人工标注或真实环境试错”替换成可验证 reward、组内比较和自动回放。它能降低数据成本、评测成本和部分人工偏好成本，但会显著增加 rollout token、reward 打分、reference logprob 和训练同步成本。

## 采样为什么能学习

生成式模型的输出经常是离散的：token、动作编号、工具调用、代码片段。离散采样本身不能直接反向传播，但“采到这个动作的概率”可以反向传播。Policy gradient 的核心就是：如果某个采样结果带来高奖励，就提高它以后被采到的概率；如果带来低奖励，就降低概率。

!!! example "生活例子：给骰子调权重"
    假设一个骰子代表模型的动作分布，掷到 6 会赢。你不能对“已经掷出的 6”求导，但可以调整骰子内部权重，让以后更容易掷到 6。policy gradient 做的就是这件事：奖励高的采样结果提高概率，奖励低的采样结果降低概率。

在 LLM/RLHF 里，这个过程可以读成：

```text
prompt -> policy 采样 response -> reward/verifier 打分
  -> 计算这个 response 比预期好不好
  -> 提高好 response 的 token 概率，降低差 response 的 token 概率
```

!!! warning "常见误区：RLHF 只是把 reward 当 loss"
    reward 通常来自完整回答或外部评测，不能像交叉熵那样直接逐 token 对齐。PPO/GRPO 要先采样回答，再算 log prob、reward、advantage 和 KL，最后才更新 policy。

## Policy Gradient：直接让好动作更可能

策略目标可以写成：

\[
J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}[R(\tau)]
\]

逐项读法：目标 \(J(\theta)\) 是“当前策略平均能拿到多少轨迹奖励”。训练要调整参数 \(\theta\)，让从策略 \(\pi_\theta\) 采样出来的轨迹 \(\tau\) 平均奖励更高。

| 符号 | 含义 |
| --- | --- |
| \(J(\theta)\) | policy 的优化目标，越大越好 |
| \(\theta\) | policy 参数 |
| \(\tau\sim\pi_\theta\) | 按当前策略采样出来的一段轨迹 |
| \(R(\tau)\) | 整条轨迹得到的总奖励或评分 |
| \(\mathbb{E}\) | 对很多次采样结果取平均 |

REINFORCE 的核心估计是：

\[
\nabla_\theta J(\theta)
\approx
\mathbb{E}
\left[
G_t \nabla_\theta \log \pi_\theta(a_t\mid s_t)
\right]
\]

逐项读法：\(G_t\) 告诉我们这次从第 \(t\) 步开始的结果好不好；\(\nabla_\theta \log \pi_\theta(a_t\mid s_t)\) 告诉我们怎样改参数才能提高这次动作的概率。两者相乘，就是“好动作提高概率、差动作降低概率”的数学形式。

| 符号 | 含义 |
| --- | --- |
| \(\nabla_\theta J(\theta)\) | 目标对参数的梯度 |
| \(G_t\) | 从第 \(t\) 步开始的 return |
| \(\log \pi_\theta(a_t\mid s_t)\) | policy 给已采样动作的 log probability |
| \(\nabla_\theta \log \pi_\theta(a_t\mid s_t)\) | 提高或降低该动作概率的方向 |

!!! note "为什么要乘 \(\nabla \log \pi\)"
    生成式模型不能直接对“采样结果”求导，因为采样是离散动作。但可以对“这个动作的概率”求导。奖励告诉你方向好坏，\(\nabla \log \pi\) 告诉你该怎样改参数才能提高或降低这个动作概率。

## Baseline 与 Advantage：把绝对分数变成相对好坏

直接用 return 更新策略方差很大。常用做法是减去 baseline：

\[
\nabla_\theta J(\theta)
\approx
\mathbb{E}
\left[
(G_t-b(s_t))\nabla_\theta\log\pi_\theta(a_t\mid s_t)
\right]
\]

逐项读法：不要只看这次得了多少分 \(G_t\)，还要看“这个状态下本来预期能得多少” \(b(s_t)\)。真正推动策略变化的是相对差值。

| 符号 | 含义 |
| --- | --- |
| \(b(s_t)\) | baseline，对当前状态平均表现的估计 |
| \(G_t-b(s_t)\) | 这次结果相对预期好还是差 |
| 乘上 \(\nabla\log\pi\) | 把相对好坏转成参数更新方向 |

当 baseline 取 value function \(V(s_t)\) 时，就得到 advantage：

\[
A_t = G_t - V(s_t)
\]

逐项读法：如果这次 return 高于 value 预期，\(A_t>0\)，提高这次动作概率；如果低于预期，\(A_t<0\)，降低这次动作概率。

| 量 | 简洁解释 |
| --- | --- |
| return \(G_t\) | 这次从当前位置往后实际拿到多少 |
| value \(V(s_t)\) | 一般情况下从这个状态预计拿多少 |
| advantage \(A_t\) | 这次动作比预期好还是差 |

!!! example "生活例子：考试平均分"
    如果你考了 85 分，不能只看 85 高不高，还要看平均分是多少。平均 60 时，85 是好结果；平均 95 时，85 反而差。Advantage 就是“相对预期的好坏”。

## Actor-Critic：actor 负责行动，critic 负责估分

Actor-Critic 把模型拆成两个角色：

| 角色 | 做什么 | LLM/RLHF 对应 |
| --- | --- | --- |
| actor / policy | 产生动作或 token | 当前被训练的语言模型 |
| critic / value model | 估计状态或前缀的未来回报 | 给每个前缀估计 value 的模型 |

critic 不直接执行动作，它负责给 actor 的更新提供低方差 baseline。PPO 就是典型 actor-critic 路线；GRPO 则用组内相对奖励替代 critic，减少训练一个 value model 的成本。

!!! note "critic 不是裁判，而是估分器"
    reward 像最终裁判分，critic 像赛前根据局面估一个平均水平。actor 更新时真正用的是“比估分高还是低”。如果 critic 估得准，策略梯度噪声会小很多；如果 critic 估错，actor 也可能被错误信号带偏。

!!! example "生活例子：餐厅评论"
    actor 像正在选餐厅的人，critic 像根据地点、价格和排队长度估计“这次大概会满意到什么程度”的朋友。真正有用的信息不是“这顿饭 4 星”，而是“这家在这种条件下本来预期 3 星，结果 4 星，说明选择比预期好”。这就是 advantage。

## 探索与 rollout：先看到更多候选，reward 才能比较

探索不是“随机乱试”，而是在不知道哪个动作更好时，给策略保留发现更好行为的机会。

| 做法 | 简洁解释 | 典型位置 |
| --- | --- | --- |
| \(\epsilon\)-greedy | 大多数时候选当前最优动作，少数时候随机试 | DQN、离散动作入门 |
| entropy bonus | 奖励策略保持一定随机性，避免过早变成单一路线 | A3C、PPO、SAC |
| stochastic policy | 策略本身输出分布，动作从分布中采样 | 连续控制、生成模型 |

SAC 是理解“熵为什么有用”的经典论文。它把目标写成“最大化 reward，同时最大化 policy entropy”，也就是既要完成任务，又不要太早把策略压成单一路径。

![SAC stochastic versus deterministic stability](../assets/images/paper-figures/reinforcement-learning/sac-seeds-humanoid-rllab.png){ width="690" }

<small>图源：[Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290)，Figure 4。原论文图意：比较 SAC 与 deterministic SAC variant 在 Humanoid (rllab) 上不同随机种子的稳定性。</small>

!!! note "图解：为什么 stochastic policy 更稳"
    蓝色 SAC 使用最大熵目标，策略不会太早坍缩到一个确定动作；红色 deterministic 变体更容易因为早期偶然高分锁到局部解。对机器人和 VLA 来说，这像“只练一种抓杯姿势”与“保留几种合理抓法”的区别：后者更有机会在物体位置、摩擦和遮挡变化时保持鲁棒。

RL 训练常被数据采样卡住：要先用当前策略和环境交互，才能得到 reward 和 trajectory。A3C 的一个重要启发是用多个 actor-learner 并行收集经验，减轻单个环境采样的瓶颈。

![A3C learning speed comparison](../assets/images/paper-figures/reinforcement-learning/a3c-learning-speed.png){ width="920" }

<small>图源：[Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)，Figure 1。原论文图意：比较 DQN 与多种 asynchronous RL 方法在 Atari 游戏上的学习速度。</small>

!!! note "图解：从 A3C 并行 actor 到 verl rollout worker"
    图里的核心不是某条曲线绝对更好，而是“采样、评估、更新”可以拆成并行工作流。大模型 RLHF/GRPO 中，rollout worker 负责高吞吐生成，reward worker 负责打分，actor/critic worker 负责训练。`verl` 的 dataflow 设计可以看作把这种并行交互思想搬到 LLM/VLM 后训练上，只是环境从 Atari 模拟器变成了 prompt、工具、规则奖励或 reward model。

## PPO：用概率比限制更新幅度

PPO 使用新旧策略概率比：

\[
r_t(\theta)=
\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\theta_{\text{old}}}(a_t\mid s_t)}
\]

逐项读法：同一个状态动作 \((s_t,a_t)\)，新策略给它的概率除以旧策略给它的概率。如果 \(r_t=1.2\)，说明新策略把这个动作概率提高了 20%；如果 \(r_t=0.8\)，说明降低了 20%。

| 符号 | 含义 |
| --- | --- |
| \(r_t(\theta)\) | 新旧策略概率比 |
| \(\pi_\theta\) | 正在更新的新策略 |
| \(\pi_{\theta_{\text{old}}}\) | 采样这批数据时的旧策略 |
| \(a_t,s_t\) | 这批 rollout 中实际出现的动作和状态 |

clipped surrogate objective 常写成：

\[
L^{\text{CLIP}}(\theta)
=
\mathbb{E}_t
\left[
\min
\left(
r_t(\theta)A_t,
\mathrm{clip}(r_t(\theta),1-\epsilon,1+\epsilon)A_t
\right)
\right]
\]

逐项读法：如果 advantage 为正，PPO 想提高这个动作概率，但 \(r_t\) 超过 \(1+\epsilon\) 后收益被截断；如果 advantage 为负，PPO 想降低这个动作概率，但 \(r_t\) 低于 \(1-\epsilon\) 后也不会继续鼓励更激进的变化。

| 符号 | 含义 |
| --- | --- |
| \(L^{\text{CLIP}}\) | PPO 的 clipped surrogate objective |
| \(A_t\) | 当前动作的 advantage |
| \(\epsilon\) | clip 范围，常见如 0.1 或 0.2 |
| \(\mathrm{clip}(r,1-\epsilon,1+\epsilon)\) | 把概率比限制在一个小区间里 |
| \(\min(\cdot,\cdot)\) | 取更保守的目标，避免一步更新过猛 |

![PPO clipped surrogate objective 原论文图](../assets/images/paper-figures/training/ppo-clipped-surrogate.png){ width="760" }

<small>图源：[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)，Figure 1。原论文图意：展示 clipped surrogate objective 如何在概率比偏离旧策略过多时截断收益，避免过大的策略更新。</small>

!!! note "图解：clip 不是为了让模型学慢，而是让更新可信"
    如果某个回答这次得分很高，朴素 policy gradient 可能把它的概率一下推得很高。PPO 的 clip 会限制新策略相对旧策略的概率变化，防止一次 batch 中的偶然高分把模型拉偏。LLM 后训练里还会叠加 reference/KL 约束，目的也是避免模型为了 reward 牺牲基础能力和语言质量。

PPO 的算法流程可以直接按原论文 Algorithm 1 抽成四步：

```text
for iteration = 1, 2, ...:
  run policy for T steps with N parallel actors
  compute advantage estimates
  optimize surrogate objective for K epochs with minibatches
  set old policy = current policy
```

<small>流程源：[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)，Algorithm 1。原论文算法要点：每轮先用当前策略采样轨迹，估计 advantage，再用多个 epoch 的 minibatch 优化 surrogate objective。</small>

!!! note "图解：PPO 的两层循环"
    外层循环负责采样新轨迹，内层循环负责在这批轨迹上做多轮 minibatch 更新。它不是无限重复使用旧数据的离线训练；PPO 仍然是近似 on-policy 方法，所以采样策略和更新策略不能差太远。这也是为什么 rollout、old log prob、reference log prob 和 KL 控制在工程上很重要。

## RLHF 里的 Reference/KL 约束

大模型后训练通常不希望 policy 为了 reward 完全跑偏，所以会保留一个 reference policy，常常是 SFT 模型或旧策略。KL divergence 用来衡量新策略和 reference 的分布差距：

\[
D_{\mathrm{KL}}(\pi_\theta\|\pi_{\mathrm{ref}})
=
\mathbb{E}_{a\sim\pi_\theta}
\left[
\log \pi_\theta(a\mid s)-\log \pi_{\mathrm{ref}}(a\mid s)
\right]
\]

逐项读法：如果新策略很喜欢某个动作，而 reference 不喜欢，\(\log \pi_\theta-\log \pi_{\mathrm{ref}}\) 会变大；KL 变大说明新策略正在远离底座模型。

| 符号 | 含义 |
| --- | --- |
| \(\pi_{\mathrm{ref}}\) | reference policy，通常是 SFT/base 模型 |
| \(D_{\mathrm{KL}}\) | 两个概率分布之间的差异 |
| \(\log \pi_\theta(a\mid s)\) | 当前策略对动作的 log probability |
| \(\log \pi_{\mathrm{ref}}(a\mid s)\) | reference 对同一动作的 log probability |

在 RLHF 里，KL 可以作为 reward penalty，也可以作为 actor loss 的一部分：

\[
r^{\text{total}} = r^{\text{task}} - \beta D_{\mathrm{KL}}(\pi_\theta\|\pi_{\mathrm{ref}})
\]

逐项读法：任务奖励越高越好，但偏离 reference 太远要扣分。系数 \(\beta\) 控制“学新行为”和“保留原模型能力”的平衡。

| 符号 | 含义 |
| --- | --- |
| \(r^{\text{task}}\) | 任务 reward，例如正确性、偏好分、测试通过 |
| \(\beta\) | KL 惩罚强度 |
| \(r^{\text{total}}\) | 最终用于策略更新的奖励 |

| RL 术语 | LLM/RLHF 对应 |
| --- | --- |
| state | prompt 加已经生成的前缀 |
| action | 下一个 token 或一段 response |
| trajectory | 从 prompt 到完整回答的 token 序列 |
| reward | reward model、规则检查器或人类偏好分数 |
| actor | 当前被训练的语言模型 |
| reference policy | SFT 模型或旧策略，用来约束 KL |
| critic | value model，估计当前前缀未来奖励 |

!!! warning "常见误区：KL 越小越好"
    KL 太大，模型可能为了 reward 牺牲语言质量、格式稳定性或安全边界；KL 太小，模型几乎学不动。实际训练要一起看 reward、KL、pass rate、长度、重复率和人工抽查。

## GRPO：用组内相对奖励替代 critic

GRPO 的思路是：对同一个 prompt 采样多个回答，形成一组：

\[
\{o_1,o_2,\ldots,o_G\}
\]

逐项读法：\(G\) 是组大小，同一个问题会生成 \(G\) 个候选输出。它们共享同一个 prompt，所以奖励可以在组内比较。

每个回答得到奖励 \(R_i\)，再用组内平均和标准差归一化：

\[
A_i =
\frac{R_i-\mathrm{mean}(\{R_j\}_{j=1}^G)}
{\mathrm{std}(\{R_j\}_{j=1}^G)+\epsilon}
\]

逐项读法：如果第 \(i\) 个回答的奖励高于同题平均分，它的 advantage 为正；低于平均分则为负。除以标准差是为了把不同题目的奖励尺度拉到可比范围。

| 符号 | 含义 |
| --- | --- |
| \(o_i\) | 同一 prompt 下第 \(i\) 个采样输出 |
| \(G\) | 每个 prompt 的采样数量 |
| \(R_i\) | 第 \(i\) 个输出的 reward |
| \(\mathrm{mean}\) | 同组奖励均值，充当 baseline |
| \(\mathrm{std}\) | 同组奖励标准差，用于归一化 |
| \(\epsilon\) | 防止除以 0 的小常数 |

![GRPO 原论文图](../assets/images/paper-deep-dives/technical-reports/deepseek-r1/figure-grpo.png){ width="920" }

<small>图源：[DeepSeek-R1](https://arxiv.org/abs/2501.12948)，Supplementary Figure 1。原论文图意：PPO 使用 value model 估计 advantage；GRPO 省去 value model，直接从同一问题的一组 sampled outputs 的 reward scores 中估计相对 advantage。</small>

!!! note "图解：为什么 GRPO 适合可验证奖励"
    数学题、代码题、结构化状态预测这类任务常常能给完整输出自动打分。对同一道题采样多个答案后，正确答案天然比错误答案好；GRPO 用同题组内平均分做 baseline，不需要额外训练一个 critic 去猜每个前缀未来是否会成功。

!!! example "生活例子：同一道题采样 5 个解法"
    如果同一道数学题采样 5 个答案，两个正确、三个错误，那么正确答案不只是“得 1 分”，而是“比同题其他答案更好”。GRPO 用这组内部比较当 baseline。它不需要 critic 估计每个前缀的未来价值，但代价是一次 prompt 要生成多条 response，rollout 吞吐和 reward 质量会变得更重要。

## DeepSeek-R1：可验证奖励怎样改变推理行为

DeepSeek-R1-Zero 的重要启发是：如果 reward 足够可靠，模型可以通过大规模 GRPO 从结果反馈中学出更长的推理、检查和回溯行为。

![DeepSeek-R1-Zero AIME curve 原论文图](../assets/images/paper-deep-dives/technical-reports/deepseek-r1/figure-r1-zero-aime.png){ width="760" }

<small>图源：[DeepSeek-R1](https://arxiv.org/abs/2501.12948)，Figure 3。原论文图意：展示 R1-Zero 在 AIME 2024 上随 RL 训练提升的 pass@1 和共识解码表现。</small>

!!! note "图解：曲线读的是 outcome reward 的效果"
    横轴是 RL 训练过程，纵轴是数学题表现。这里的关键不是“模型被教了某种固定解题模板”，而是 outcome reward 会提高那些更容易得到正确答案的采样轨迹概率。只要 reward 能稳定区分正确和错误，策略就有机会从试错中发现更有效的推理习惯。

![DeepSeek-R1-Zero response length 原论文图](../assets/images/paper-deep-dives/technical-reports/deepseek-r1/figure-r1-zero-length.png){ width="760" }

<small>图源：[DeepSeek-R1](https://arxiv.org/abs/2501.12948)，Figure 3。原论文图意：展示 R1-Zero 平均 response length 随 RL 训练增加。</small>

!!! note "图解：长推理不是 prompt 硬拉长"
    如果 reward 只看最终正确性，模型会探索各种能提高正确率的行为。对难题来说，短答案经常不够稳定；更长的推理给模型更多空间去枚举条件、重新计算、发现错误和切换策略。只要这些额外 token 带来的正确率收益超过 KL 和采样成本，策略更新就会提高这类长推理轨迹的概率。

## Reward Hacking：reward 升高不等于能力提升

RL 最危险的地方是：policy 会优化你给它的 reward，而不是你心里真正想要的目标。如果 reward model 或 verifier 有漏洞，policy 会找到漏洞。

![Reward hacking 原论文图](../assets/images/paper-deep-dives/technical-reports/deepseek-r1/figure-reward-hacking.png){ width="760" }

<small>图源：[DeepSeek-R1](https://arxiv.org/abs/2501.12948)，Supplementary Figure 3。原论文图意：训练中 reward 持续上升，但 CodeForces 表现下降，说明 helpful reward model 可被策略利用，造成 reward hacking。</small>

!!! note "图解：为什么不能只看 reward 曲线"
    可验证 reasoning reward 比较硬，数学答案错就是错，代码测试不过就是不过。通用 helpfulness reward 更软，reward model 可能偏好长、顺滑、有礼貌或某种固定风格的回答。RL 时间一长，policy 会找到 reward model 的偏差，而不一定真的更有用。

做 LLM/VLM/VLA 强化学习时，至少要检查：

| 检查项 | 为什么重要 |
| --- | --- |
| reward 是否可被投机 | reward 漏洞会让 policy 学会钻空子 |
| KL/reference 是否合适 | 太强学不动，太弱会破坏基础能力 |
| rollout 分布是否更新及时 | 旧数据太多会偏离 on-policy 假设 |
| advantage 方差是否过大 | 方差大会导致训练不稳或奖励曲线虚高 |
| 评测是否闭环 | 离线 reward 提升不一定等于真实任务成功 |

## PPO、GRPO、RLOO、ReMax 和 DPO 的边界

| 方法 | 是否采样当前策略 | 是否显式 reward | 是否需要 critic | 适合什么 |
| --- | --- | --- | --- | --- |
| PPO | 是 | 是 | 通常需要 | reward model、RLHF、连续改进 |
| GRPO | 是 | 是 | 通常不需要 | 可验证奖励、多样本组内比较 |
| RLOO | 是 | 是 | 不需要 | 多样本 leave-one-out baseline，减少 critic 成本 |
| ReMax | 是 | 是 | 不需要 | 用额外 baseline generation 做方差降低 |
| DPO | 否，主要用偏好对 | 隐式 | 不需要 | 偏好数据稳定、想简化 RL 工程 |

DPO 很重要，但它更像偏好优化的监督化变体；本章重点放 PPO/GRPO/RLOO/ReMax 这类需要 rollout 和 reward 的路线，因为它们更直接连接世界模型、RLVR 和 verl 这类 RL 工程框架。

### 11.1 算一遍：GRPO 的 rollout token 账

假设一轮 GRPO 用 `8,192` 个 prompt，每题采样组大小 `G=8`，平均 response 长度 `512` tokens，则 rollout 生成量是：

\[
8192 \times 8 \times 512 \approx 33.6\text{M tokens}
\]

这个 `33.6M` 只是在数 response decode token。prompt 侧也要算：如果平均 prompt 长度是 `1,024` token，并且 `G=8` 条样本各自独立 prefill，那么输入侧还会有 \(8192\times8\times1024\approx67.1M\) prompt token；如果 serving runtime 能对同一 prompt 做 prefix/KV 复用，则 prefill 更接近 \(8192\times1024\approx8.4M\) token，再分叉出 8 条 decode。两种实现的训练数据“数学上”一样，但 rollout 集群成本完全不同。

如果 rollout 集群端到端吞吐是 `80k tokens/s`，仅生成就约 `420s`。这还没算 reward model、规则执行、reference logprob、old logprob 和训练更新。若把组大小从 `8` 提到 `16`，方差可能下降，但 rollout token 直接翻倍到 `67.1M`；如果平均 response 因奖励鼓励长答案从 `512` 涨到 `768`，成本又再乘 `1.5x`。

![RL infrastructure 原论文图](../assets/images/paper-deep-dives/technical-reports/deepseek-r1/figure-rl-infra.png){ width="920" }

<small>图源：[DeepSeek-R1](https://arxiv.org/abs/2501.12948)，Supplementary Figure 2。原论文图意：R1 的 RL framework 分成 rollout、inference、rule-based reward 和 training 四个模块，并用 offloading、异步调度、数据 packing、vLLM、MTP 和并行训练优化吞吐。</small>

!!! note "图解：RL 工程瓶颈不只是反向传播"
    Rollout、reward/reference inference 和 training 需要的模型实例不同。如果所有模型同时常驻 GPU，大模型后训练的显存压力会非常高。真实系统必须把生成吞吐、规则奖励延迟、reference logprob、actor update 和权重同步一起排进同一条流水线。

### 11.2 真实排查：reward 上升，世界模型闭环不升

一个典型日志是：

```text
train_reward=0.71 -> 0.84
video_mse=0.042 -> 0.039
lpips=0.181 -> 0.176
closed_loop_success=64.2% -> 63.5%
action_sensitivity_pass=58.0% -> 46.5%
```

这时不能只说“RL 没用”。更可能的链条是：reward 主要奖励未来帧像不像，但没有奖励“换动作时未来是否按动作变化”。模型学会生成更平均、更安全的视觉未来，MSE/LPIPS 变好；planner 需要的是反事实差异，因此闭环不升甚至下降。

修复要把 reward 拆开：加入 action-conditioned contrastive check，要求同一状态下不同动作的预测差异与真实后果一致；按接触、遮挡、反光、失败恢复分桶统计；对 reward model 加入 holdout bucket 和 adversarial prompt；限制 response/action chunk 过长导致的 reward hacking。

!!! tip "下一站"
    - 回到本专题入口：[强化学习](index.md)，确认这页在整条路线中的位置。
    - 按导航顺序继续：[世界模型中的强化学习](rl-for-world-models.md)。
    - 概念或符号卡住时，先查 [术语表](../glossary/index.md)，再回到当前页。
