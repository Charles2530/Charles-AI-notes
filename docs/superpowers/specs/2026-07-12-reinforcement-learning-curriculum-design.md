# 强化学习完整课程化设计

## 目标

把现有强化学习专题从“几篇面向特定方向的长文”重构成一条适合深度学习开发者连续学习的课程：

```text
强化学习定义与交互闭环
-> Bandit、MDP 与价值估计
-> Value-based / Policy-based / Actor-Critic
-> Offline / Model-based / Continuous Control
-> LLM、VLA、世界模型与分布式 RL 工程
```

课程首页必须先回答“什么是强化学习”，再给出 agent 与环境的交互闭环、强化学习与监督学习的差别、适用条件和完整学习路线。读者学完课程后，应能解释常见算法的训练信号和假设，独立运行最小实验，并根据任务的数据来源、动作空间和交互成本选择算法路线。

## 目标读者与先修知识

目标读者是已经具备以下能力的深度学习开发者：

- 能使用 Python 和 PyTorch 编写、训练和调试神经网络。
- 理解张量、反向传播、损失函数、优化器和常见训练循环。
- 学过基础概率，但不要求已经掌握随机过程、重要性采样或策略梯度。

课程不从微积分、线性代数或神经网络基础重新讲起。强化学习真正依赖、而读者可能尚未系统掌握的概率知识，补充到现有 `prerequisite-math/` 页面，并在正文第一次使用时链接。

## 当前问题

1. `reinforcement-learning/index.md` 只有简短专题说明和文章链接，没有强化学习定义，也没有解释 agent、environment、action、reward 和 trajectory 如何组成交互闭环。
2. `mkdocs.yml` 的强化学习导航没有显式列出专题首页，读者会直接进入 MDP，缺少第一课。
3. `mdp-value-bellman.md` 同时承担 MDP、Bellman、Q-learning、Offline RL 和世界模型接口，单篇跨度过大。
4. `policy-gradient-actor-critic-ppo-grpo.md` 从 Policy Gradient 一直讲到 PPO、GRPO、视觉生成和 LLM 系统成本，初学者难以判断哪些是通用基础、哪些是现代应用。
5. 现有内容重视世界模型、LLM 和 VLA，但 Bandit、动态规划、Monte Carlo、TD、DQN、SAC 等经典支点缺失，知识链无法连续。
6. 页面之间缺少统一的先修说明、学习目标、自检题、最小练习和下一章入口。

## 教学方案

采用“线性主干 + 现代应用桥接”。理论按依赖顺序只完整解释一次，现代案例在主干对应位置建立连接：

- GridWorld 贯穿 MDP、Bellman、动态规划、TD 和 Q-learning。
- CartPole / Pendulum 连接 DQN、PPO 和 SAC。
- LLM verifier、Dreamer imagined rollout 和机器人闭环案例连接本站重点方向。

不采用经典 RL 与现代 RL 双路线，因为两条路线会重复解释 value、advantage、off-policy 和 policy ratio。也不采用纯项目驱动路线，因为项目容易掩盖算法的估计假设与适用边界。

## 课程信息架构

### 阶段零：建立强化学习语言

1. **强化学习专题首页**
   - 给出准确、可操作的定义。
   - 用一个完整时间步展示 `observation -> action -> reward + next observation`。
   - 比较监督学习、模仿学习和强化学习的数据来源与目标。
   - 说明何时不该使用强化学习。
   - 提供课程地图、方向入口和先修检查。
2. **Agent、环境与训练闭环**
   - observation、state、action、reward、done、episode、trajectory、policy。
   - rollout、训练 batch 和参数更新之间的数据流。
   - 部分可观测、延迟奖励和策略改变数据分布三个核心难点。

### 阶段一：从局部反馈到长期价值

3. **多臂老虎机与探索**
   - exploration 与 exploitation。
   - sample-average、epsilon-greedy、UCB 的最小实现和对比。
   - regret 只讲使用直觉和最小定义，不展开完整理论证明。
4. **MDP、回报、价值函数与 Bellman**
   - MDP、Markov property、trajectory、discounted return。
   - `V`、`Q`、advantage 和 Bellman expectation / optimality。
   - state 与 observation 的区别。
5. **动态规划、Monte Carlo 与 TD**
   - policy evaluation / improvement / iteration。
   - Monte Carlo return、TD target、bootstrap 和 TD error。
   - bias-variance 与 n-step return 的位置。

### 阶段二：经典深度强化学习

6. **Q-learning 与 DQN**
   - SARSA 与 Q-learning 的 on-policy / off-policy 差别。
   - function approximation、replay buffer、target network 和 epsilon schedule。
   - 用 CartPole 或小型离散环境完成可运行实验。
7. **Policy Gradient 与 REINFORCE**
   - score-function estimator、log-prob trick、return-to-go 和 baseline。
   - 离散采样不可导时梯度从哪里来。
   - 用最小 PyTorch 实现展示 trajectory 到 loss 的转换。
8. **Actor-Critic、GAE 与 PPO**
   - actor / critic 职责、advantage estimation、GAE。
   - old policy、probability ratio、clipping、entropy 和 KL。
   - 明确 PPO 的近似 on-policy 数据约束。
9. **连续控制与 SAC**
   - 连续动作分布、reparameterized sampling、maximum entropy RL。
   - twin Q、target value、temperature 和 replay buffer。
   - 使用 Pendulum 一类轻量环境解释连续控制。

### 阶段三：低成本与模型驱动学习

10. **Offline Reinforcement Learning**
    - online、off-policy 和 offline 的数据边界。
    - distribution shift、out-of-distribution action、extrapolation error。
    - behavior constraint 与 conservative value estimation 的直觉。
11. **Model-based RL 与世界模型**
    - learned transition / reward / done model。
    - planning、MPC、Dyna、latent dynamics、PlaNet 和 Dreamer。
    - model bias、compounding error 和 model exploitation。

### 阶段四：现代大模型与具身系统

12. **LLM 强化学习：RLHF、RLVR 与 GRPO**
    - prompt / prefix 作为状态、token 作为动作、response 作为 trajectory。
    - reward model、verifier、reference policy、KL 和 credit assignment。
    - PPO、GRPO、RLOO 和 DPO 的边界；DPO 作为偏好优化对照，不冒充在线 RL。
13. **VLA、机器人与闭环强化学习**
    - POMDP、连续控制、action chunk、真实试错成本和安全约束。
    - demonstration、offline data、simulation、world model 与真实部署数据闭环。
14. **RL 系统工程与 verl**
    - rollout、reward、old / reference logprob、advantage、训练 worker 和权重同步。
    - 吞吐、显存、staleness、prefix reuse 和可复现性。
15. **综合实践与算法选择**
    - 按动作空间、数据来源、交互成本、奖励质量和模型可用性选择路线。
    - 给出失败诊断表和课程综合练习。

## 文件迁移设计

所有现有 URL 保留，不重命名文件。

| 现有文件 | 处理方式 |
| --- | --- |
| `reinforcement-learning/index.md` | 完整重写为专题首页和课程入口 |
| `mdp-value-bellman.md` | 聚焦 MDP、return、value、Q、advantage 和 Bellman；移出 DQN、Offline RL、世界模型的大段内容 |
| `policy-gradient-actor-critic-ppo-grpo.md` | 聚焦 Actor-Critic、GAE 与 PPO；保留 GRPO 摘要和专章入口 |
| `rl-for-world-models.md` | 扩展为 Model-based RL 与世界模型主章 |
| `verl-code-flow.md` | 保留 URL，纳入系统工程阶段并补全输入知识和总数据流 |

新增页面：

```text
files/reinforcement-learning/
├── rl-agent-environment-loop.md
├── multi-armed-bandits-and-exploration.md
├── dynamic-programming-monte-carlo-td.md
├── q-learning-and-dqn.md
├── policy-gradient-and-reinforce.md
├── continuous-control-and-sac.md
├── offline-reinforcement-learning.md
├── llm-rlhf-rlvr-and-grpo.md
├── vla-robotics-and-closed-loop-rl.md
└── practice-and-algorithm-selection.md
```

`mkdocs.yml` 将显式加入“强化学习导读”，并按课程顺序展示全部章节。现有页面中被迁移的内容在新主章落位后，原位置保留简短摘要和链接，避免概念突然消失或出现断链。

## 前置数学补强

不新增数学页面，定点补强以下现有内容：

| 页面 | 补充内容 | 正文使用位置 |
| --- | --- | --- |
| `probability-distributions-and-expectation.md` | 条件期望、全期望公式、随机变量条件化 | value function、Bellman、baseline |
| `sampling-estimation-and-calibration.md` | Monte Carlo 估计、估计方差、control variate / baseline、重要性采样 | MC、Policy Gradient、off-policy correction |
| `stochastic-processes-and-markov.md` | Markov property、state 对历史的压缩、POMDP 入口 | MDP、世界模型、VLA |
| `information-theory-and-divergence.md` | policy entropy、交叉熵、KL 方向与策略约束 | SAC、PPO、RLHF |

正文只保留当前推导所需的最短回顾，并链接数学页。数学页负责定义和推导，强化学习页负责算法中的具体作用。

## 单章教学契约

每一章按以下顺序组织：

1. **本章回答什么**：给出一个具体决策问题和学习目标。
2. **先修知识**：链接前置数学或上一课程章节。
3. **最小任务**：明确环境、状态或观测、动作、奖励和终止条件。
4. **核心直觉**：解释本章方法解决了上一章的什么不足。
5. **数学定义**：逐个解释符号、shape、采样来源、期望对谁取和成立假设。
6. **算法与数据流**：从 rollout / replay data 走到 target、loss 和参数更新。
7. **最小实现**：提供可以独立运行的 NumPy 或 PyTorch 代码。
8. **结果诊断**：解释 reward、value、TD error、entropy、KL 等曲线。
9. **常见失败**：区分实现错误、估计偏差、分布偏移和任务设定问题。
10. **自检与练习**：至少包含概念题、手算或跟踪题、代码改动题。
11. **下一章入口与外部阅读**：说明继续阅读的知识依赖。

章节可以根据主题省略不适用的小节，但不能省略学习目标、数据流、自检和下一章入口。

## 实践资产

实践分三层，控制依赖和运行成本：

### 层一：必做小实验

- k-armed bandit：epsilon-greedy 与 UCB。
- 小型 GridWorld：policy evaluation、value iteration、MC、TD 和 Q-learning。
- 默认只依赖 Python、NumPy 和 Matplotlib。
- 固定随机种子并断言关键数值或趋势，单个实验应在普通 CPU 上快速完成。

### 层二：核心深度 RL 实验

- DQN / CartPole。
- REINFORCE 或 PPO / CartPole。
- SAC / Pendulum。
- 使用 PyTorch 和仓库选定的标准环境依赖；实现保持教学透明，不引入训练框架封装。
- 每个实验输出最小训练指标和评估结果，区分训练时探索策略与评估策略。

### 层三：方向数据流案例

- 简化 LLM GRPO：预先给定 prompt、候选 response、reward 和 logprob，验证 group advantage 与 policy loss。
- 简化 imagined rollout：展示 latent state、action、predicted reward / value 如何构成 actor 更新输入。
- verl：沿现有代码流解释分布式角色与数据字段，不在教程里训练大语言模型。

层三只验证数据接口和目标计算，不承诺大模型或机器人训练结果。

## 内容来源与证据边界

- 算法定义、公式和技术结论优先引用原论文、教材或项目官方文档。
- 外部博客只用于比较教学顺序、例子组织和读者认知负担，不作为关键技术结论的唯一证据。
- 调研中参考的代表性教学结构包括：
  - [Lilian Weng, A (Long) Peek into Reinforcement Learning](https://lilianweng.github.io/posts/2018-02-19-rl-overview/)：从定义、核心概念、MDP 和 Bellman 进入算法家族。
  - [Jannik Zuern, Reinforcement Learning - An Introduction](https://jannik-zuern.medium.com/reinforcement-learning-to-survive-in-a-hostile-environment-3658624a5d83)：按 Concept、Math、Problem、Code、Results 组织具体任务。
  - [Cameron R. Wolfe, PPO for LLMs: A Guide for Normal People](https://cameronrwolfe.substack.com/p/ppo-llm)：先建立 RL 基础对象，再逐字段展开 PPO 与 LLM 数据流。
- 公式中的符号约定在课程内统一；若论文使用不同时间下标、reward 约定或 KL 方向，正文明确指出差异。
- 图片必须使用仓库已有且来源清楚的素材、论文原图或本站自绘图，并保留说明图实际支持的结论。

## 导航与页面体验

- 强化学习导航第一项为“强化学习导读”，直接指向 `reinforcement-learning/index.md`。
- 导航按四个阶段分组，标题使用短名称，页面 H1 保留完整教学含义。
- 首页首屏包含定义、交互闭环和“从哪里开始”，不以算法缩写或站点方向说明开场。
- 长表格在移动端能够横向滚动；长公式、代码块和 Mermaid / 图片不得超出正文区域。
- 每章顶部和底部都提供清晰的前后章链接，但不引入自定义课程运行时或复杂组件。

## 验证与验收标准

### 内容验收

- 首页首段给出强化学习定义，首个主要小节解释 agent-environment loop。
- 课程覆盖 Bandit、MDP、DP、MC、TD、Q-learning、DQN、Policy Gradient、Actor-Critic、GAE、PPO、SAC、Offline RL、Model-based RL、LLM RL、VLA 和系统工程。
- 每章都有学习目标、先修知识、数据流、自检题和下一章入口。
- 一个概念只有一个完整主讲页；重复页面使用摘要和链接。
- 监督学习、模仿学习、offline RL、DPO 与在线 RL 的边界表述清楚。

### 代码验收

- 所有层一实验在固定随机种子下运行成功。
- 层二实验至少提供短时 smoke test，不以必须达到高回报作为构建门槛。
- 目标值、advantage、probability ratio、terminal mask 等关键计算有数值测试。
- 教程代码与正文公式使用一致的变量语义。

### 站点验收

- MkDocs 严格构建通过，新增页面全部进入导航且无孤立课程页。
- 站内链接和本地图片引用通过检查。
- 在桌面和移动视口检查首页、公式密集页、宽表格和代码块，没有遮挡或不可读溢出。
- `/reinforcement-learning/`、MDP、PPO、世界模型和 verl 旧 URL 继续可访问。

## 实施边界

本轮包含课程主干、必要的数学补强、最小实践代码、导航和验证。不包含以下内容：

- 自研 RL 训练框架、统一配置系统或实验追踪平台。
- 大模型、机器人或世界模型的大规模训练。
- 对课程之外的论文精读、训练专题或世界模型专题做批量重写。
- 为覆盖算法名而加入没有教学任务支撑的算法百科。
- 修改与强化学习课程无关的现有工作区改动。

## 实施顺序

1. 重写首页并调整导航，先建立可访问的课程骨架。
2. 补强前置数学链接和四个数学缺口。
3. 完成 Bandit、MDP、DP / MC / TD、DQN 的 value-based 主线与层一实验。
4. 完成 Policy Gradient、Actor-Critic / PPO、SAC 主线与层二 smoke test。
5. 完成 Offline RL、Model-based RL、LLM RL、VLA 和 verl 方向章节。
6. 完成综合选择页、自检题、交叉链接、严格构建和响应式检查。

每一阶段结束时运行对应的最小代码测试和站点构建，避免最后才集中发现公式、链接或依赖问题。
