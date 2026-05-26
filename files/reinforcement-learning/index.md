# 强化学习路线图：从奖励到世界模型

强化学习（Reinforcement Learning, RL）研究的是：一个 agent 在环境里行动，环境返回结果和奖励，agent 再把更好的行为变得更可能。它和监督学习最大的差别是，监督学习通常给出标准答案；强化学习更常见的是只告诉你“这段行为最后好不好”。

<!-- [补充] -->
| 读者 | 建议读法 | 工程落点 |
| --- | --- | --- |
| 初学者 | 先抓“强化学习路线图：从奖励到世界模型”要解决的问题、关键术语和最小例子；公式只看变量方向。 | 能说清输入、输出和常见失败信号。 |
| 工程师 | 直接对照结构、指标、成本和边界条件，必要时回到前置页补概念。 | 落到数据、算力、延迟、显存、评测和回滚检查。 |


!!! note "合并提示"
    **note · 读法定位**

    这页先回答“强化学习路线图：从奖励到世界模型”在「强化学习」里的位置：它解决什么局部问题，依赖哪些前置，最后会影响哪类工程或研究判断。
    **前置**：先懂概率、优化和基本训练曲线；PPO/GRPO 不必先记公式，先抓更新对象。 必要时先看本页路线，或回 [基础知识](../foundations/index.md) 或 [术语表](../glossary/index.md)。
    **主线关系**：把状态、动作、奖励、策略优化和世界模型闭环连起来，看模型如何通过反馈改进决策。

    **note · 初学者先抓住**

    RL 不是“多一个 loss 名字”，而是一条闭环：观察局面、采样动作、得到奖励、估计这次行为比预期好还是差、更新策略。PPO、GRPO、Dreamer、RLVR 和 verl 都是在这条闭环上做不同的稳定化或工程化。

    **tip · 本专题的读法**

    先读直觉和符号，不要一上来背公式。每个公式都可以问四件事：随机变量是什么、期望对谁取、奖励从哪里来、最后更新的是 policy 还是 value/world model。

<div class="atlas-meta-grid">
  <div>
    <strong>读前先知道</strong>
    <p>基础机器学习、概率采样、loss、梯度和序列生成。</p>
  </div>
  <div>
    <strong>读中抓主线</strong>
    <p>state、action、reward、return、value、advantage、policy update。</p>
  </div>
  <div>
    <strong>读完能判断</strong>
    <p>一个方法到底改的是奖励、采样、baseline、KL 约束、rollout 系统还是世界模型。</p>
  </div>
</div>

## 最小闭环：五个角色

初学者读 RL 公式前，先把下面五个角色分清楚：

| 角色 | 英文与常见符号 | 直觉 | 大模型 / VLA 里的对应 |
| --- | --- | --- | --- |
| 智能体 | agent | 做决定的系统 | LLM、VLM policy、VLA policy、机器人策略 |
| 环境 | environment | 接收动作并返回后果 | prompt 评测器、工具环境、仿真器、真实机器人 |
| 状态 / 观测 | state \(s_t\) / observation \(o_t\) | 当前局面 | prompt、图像、历史动作、latent state |
| 动作 | action \(a_t\) | agent 选择的输出 | token、完整回答、工具调用、动作块、关节命令 |
| 奖励 | reward \(r_t\) | 结果好坏的反馈 | 正确性、偏好分、成功率、碰撞惩罚、预测指标 |

一次 RL 训练可以压成一条循环：

```text
观察局面 -> 采样动作 -> 得到结果和奖励 -> 估计这次行为比预期好不好 -> 更新策略
```

!!! note "为什么一定要采样"
    如果模型永远只输出最可能的答案，你只能看到一种行为的结果，也就不知道其他答案会不会更好。采样让模型暴露出多种候选行为，reward 才能比较“哪一种更值得提高概率”。GRPO 里的 `rollout.n=5` 就是在同一道题或同一状态下采样多次，用组内比较构造 advantage。

同一套语言可以落到不同系统里：

| 场景 | state / observation | action | reward |
| --- | --- | --- | --- |
| Atari | 屏幕像素和历史帧 | 上下左右、开火等按键 | 游戏分数 |
| 数学 RLVR | 题目 prompt 和当前解题前缀 | 下一段推理或完整答案 | 答案正确、格式正确 |
| VLA | 图像、语言指令、关节状态、历史动作 | 动作 token 或 action chunk | 成功率、碰撞、平滑度 |
| 世界模型 | 当前状态、动作和上下文 | 预测下一状态或未来轨迹 | 和真实未来的可验证指标 |

!!! warning "常见误区：reward 就是标签"
    标签通常告诉模型“正确输出是什么”。reward 只告诉模型“这次行为结果怎么样”，而且可能延迟到整段轨迹结束才出现。RL 的难点正来自这里：模型要把最后的结果分摊回前面的动作。

## 符号速查

后面几页会反复出现这些符号。第一次读不用背，先知道它们在一句话里是什么意思。

| 符号 | 读法 | 直觉 |
| --- | --- | --- |
| \(t\) | 时间步 | 当前是第几步 |
| \(s_t\) | state at time \(t\) | 第 \(t\) 步的真实局面 |
| \(o_t\) | observation at time \(t\) | 传感器或模型实际看到的东西 |
| \(a_t\) | action at time \(t\) | 第 \(t\) 步选择的动作 |
| \(r_t\) | reward at time \(t\) | 第 \(t\) 步得到的奖励 |
| \(\pi_\theta(a_t\mid s_t)\) | policy | 参数为 \(\theta\) 的策略，在状态 \(s_t\) 下选择动作 \(a_t\) 的概率 |
| \(P(s_{t+1}\mid s_t,a_t)\) | transition | 环境如何从当前状态转到下一状态 |
| \(\gamma\) | discount factor | 未来奖励折算到现在的权重 |
| \(G_t\) | return | 从第 \(t\) 步开始累计的未来奖励 |
| \(V^\pi(s)\) | value function | 按策略 \(\pi\) 从状态 \(s\) 出发的预期回报 |
| \(Q^\pi(s,a)\) | action-value | 在状态 \(s\) 先做动作 \(a\)，之后按 \(\pi\) 行动的预期回报 |
| \(A^\pi(s,a)\) | advantage | 动作 \(a\) 比当前平均选择好多少 |
| \(\tau\) | trajectory | 一段状态、动作、奖励组成的轨迹 |

!!! note "state 和 observation 不要混"
    `observation` 是看到的画面或文本，`state` 是足以预测未来的局面描述。机器人摄像头只看到杯子外观，这是 observation；杯子的质量、摩擦、遮挡后的接触关系也会影响未来，这是 state。世界模型和记忆模块常常是在把 observation 压成更接近 state 的内部表示。

## 为什么这里需要强化学习

从 VLM/VLA 走向世界模型时，问题会发生变化：

| 阶段 | 主要问题 | 学习信号 |
| --- | --- | --- |
| VLM | 视觉和语言如何对齐 | 图文配对、指令数据、问答标签 |
| VLA | 当前观测下应该做什么动作 | 示范轨迹、行为克隆、成功/失败反馈 |
| 世界模型 | 动作之后未来会怎样 | 下一状态、奖励、终止、风险、可验证指标 |
| RL | 怎样让行为因结果而改进 | reward、return、advantage、policy update |

如果只用监督学习，模型最多学会“模仿数据里出现过的动作”。真实智能体还需要根据长期结果调整短期动作，主动探索更好的策略，在世界模型里先想象后果再决定是否执行，并用可验证奖励把生成模型或世界模型对齐到任务指标。

!!! example "生活例子：学会煎鸡蛋"
    监督学习像看教程：每一步该怎么做都有人示范。强化学习像自己练习：鸡蛋煎糊会扣分，火候刚好会加分，模型逐渐学会“什么时候翻面”和“火太大要调小”。世界模型则更进一步：还没下锅前，先预测如果继续大火会不会糊。

## 三张图建立直觉

### 3.1 Atari：看图像、选动作、拿分数

经典深度强化学习常从 Atari 游戏讲起，因为它把“看图像、选动作、拿分数、长期优化”压缩在一个容易观察的环境里。

| Pong | Breakout | Space Invaders | Seaquest | Beam Rider |
| --- | --- | --- | --- | --- |
| ![](../assets/images/paper-figures/reinforcement-learning/dqn-atari-pong.png) | ![](../assets/images/paper-figures/reinforcement-learning/dqn-atari-breakout.png) | ![](../assets/images/paper-figures/reinforcement-learning/dqn-atari-space-invaders.png) | ![](../assets/images/paper-figures/reinforcement-learning/dqn-atari-seaquest.png) | ![](../assets/images/paper-figures/reinforcement-learning/dqn-atari-beamrider.png) |

<small>图源：[Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)，Figure 1。原论文图意：展示 DQN 训练中使用的 Atari 2600 游戏画面。</small>

!!! note "图解：为什么 Atari 适合做 RL 入门"
    每张图都可以按同一套语言读：屏幕像素是 observation，按键是 action，游戏分数变化是 reward，一局游戏是 trajectory。DQN 的难点是模型只能从图像和分数里学到“什么时候开火、什么时候躲避、什么时候等待”，这正好对应 RL 的长期信用分配问题。

### 3.2 InstructGPT：从偏好到 RLHF

InstructGPT 的流程图是理解 LLM 强化学习后训练的入口：先做 SFT，再训练 reward model，最后用 PPO 让 policy 朝高奖励回答移动。

![InstructGPT RLHF pipeline 原论文图](../assets/images/paper-figures/training/instructgpt-rlhf-pipeline.png){ width="920" }

<small>图源：[Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)，Figure 2。原论文图意：展示 demonstration data、comparison data、reward model 和 PPO policy optimization 的三步流程。</small>

!!! note "图解：RLHF 不等于人类每一步实时教模型"
    人类通常不逐 token 标注“下一步应该生成什么”。先由人写示范做 SFT，再让人比较多个回答训练 reward model，最后让 policy 自己采样回答并用 reward model 打分。PPO 的角色是把“完整回答得分”变成稳定的策略更新，同时用 reference/KL 约束避免模型为了奖励跑偏。

### 3.3 Dreamer：在模型里想象未来

世界模型中的强化学习更像下面这条线：先学一个能 rollout 的内部环境，再在内部想象中训练策略或优化预测目标。

![Components of Dreamer](../assets/images/paper-deep-dives/world-models/dreamer/figure-3-components.png){ width="890" }

<small>图源：[Dream to Control: Learning Behaviors by Latent Imagination](https://arxiv.org/abs/1912.01603)，Figure 3。原论文图意：Dreamer 从经验中学习 latent world model，并在 imagined trajectories 上训练 action model 和 value model。</small>

!!! note "图解：Dreamer 展示了 RL 和世界模型的合流点"
    Dreamer 不把 world model 当作漂亮的视频生成器，而是当作可以在 latent space 中想象未来的训练环境。真实数据训练 dynamics，imagined rollout 训练 actor 和 value。这样做的价值是用更少真实交互获得更多策略学习信号，但前提是 world model 不能被 actor 轻易利用漏洞。

## 继续阅读

| 你现在想解决的问题 | 先读哪页 | 读到什么程度 |
| --- | --- | --- |
| 看懂 state、action、reward、value | [MDP、价值函数与 Bellman](mdp-value-bellman.md) | 能解释 \(G_t,V^\pi,Q^\pi,A^\pi\) 的区别 |
| 看懂 PPO/GRPO/RLHF/RLVR | [Policy Gradient 到 PPO / GRPO](policy-gradient-actor-critic-ppo-grpo.md) | 能解释采样、advantage、KL、clip 和组内比较 |
| 看懂世界模型为什么需要 RL | [世界模型中的强化学习](rl-for-world-models.md) | 能解释 transition、imagined rollout、误差放大和闭环评测 |
| 看懂 verl 训练系统 | [verl 训练流程](verl-code-flow.md) | 能把 prompt、response、reward、advantage、actor update 对到代码变量 |

如果你只有 30 分钟，可以按这个顺序读：

1. 本页的“最小闭环”和“符号速查”；
2. MDP 页第 1-5 节；
3. PPO/GRPO 页第 1-7 节；
4. world-model RL 页第 1-4 节；
5. verl 页第 5-7 节。

## 看到术语时去哪里补

| 术语 | 一句话提醒 | 回看 |
| --- | --- | --- |
| reward / return | 单步反馈 vs 累计未来反馈 | [MDP、价值函数与 Bellman](mdp-value-bellman.md) |
| value / Q-value | 状态价值 vs 状态-动作价值 | [MDP、价值函数与 Bellman](mdp-value-bellman.md) |
| Bellman | 当前价值 = 当前奖励 + 下一状态价值 | [MDP、价值函数与 Bellman](mdp-value-bellman.md) |
| advantage | 这次动作比预期好多少 | [Policy Gradient 到 PPO / GRPO](policy-gradient-actor-critic-ppo-grpo.md) |
| PPO clip | 限制新策略相对旧策略不要变太猛 | [Policy Gradient 到 PPO / GRPO](policy-gradient-actor-critic-ppo-grpo.md) |
| KL / reference | 防止 policy 为了 reward 远离底座模型 | [Policy Gradient 到 PPO / GRPO](policy-gradient-actor-critic-ppo-grpo.md) |
| GRPO | 同一 prompt 多采样，用组内相对分构造 advantage | [Policy Gradient 到 PPO / GRPO](policy-gradient-actor-critic-ppo-grpo.md) |
| imagined rollout | 在 learned world model 里先想象未来 | [世界模型中的强化学习](rl-for-world-models.md) |
| reward hacking | 策略钻奖励漏洞，reward 升但真实任务不升 | [Policy Gradient 到 PPO / GRPO](policy-gradient-actor-critic-ppo-grpo.md) |
| rollout worker | 高吞吐采样回答或轨迹的工程模块 | [verl 训练流程](verl-code-flow.md) |

## 检查标准

| 问题 | 合格理解 |
| --- | --- |
| RL 和监督学习差在哪 | 监督学习模仿给定答案，RL 根据采样行为的奖励改策略 |
| value 和 reward 差在哪 | reward 是当前反馈，value 是从当前状态往后能拿到的期望回报 |
| PPO 为什么常用 | 它限制新旧策略差距，让高方差的策略更新更稳定 |
| GRPO 为什么流行 | 它用同一 prompt 的多次采样做组内基线，减少或去掉 critic 成本 |
| RL 如何帮助世界模型 | 用任务奖励、规划反馈或 imagined rollout，让模型不只拟合数据分布，还服务决策指标 |

!!! note "合并提示"
    **failure · 最佳反例**

    reward 或 verifier 很便宜，但 policy 学会钻规则漏洞，线上失败更多。RL 的效率只有在 reward hacking、失败回放和闭环收益都过关后才可信。

    **tip · 下一站**

    - 按导航顺序继续：[MDP、价值函数与 Bellman](mdp-value-bellman.md)。
    - 概念或符号卡住时，先查 [术语表](../glossary/index.md)，再回到当前页。
