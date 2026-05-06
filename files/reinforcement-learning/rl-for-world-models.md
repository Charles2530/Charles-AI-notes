# 强化学习如何帮助世界模型

世界模型的核心不是“生成未来帧”，而是“预测动作后果，并让这些预测能改善决策”。强化学习在这里有三种作用：训练策略、训练或校准世界模型、把世界模型接入规划闭环。

## 1. 世界模型补上了 transition

在 MDP 里，真实环境 transition 是：

\[
P(s_{t+1}\mid s_t,a_t)
\]

世界模型学习一个近似：

\[
\hat P_\theta(s_{t+1}\mid s_t,a_t)
\]

如果还预测 reward、done、risk 或 uncertainty：

\[
(\hat s_{t+1}, \hat r_t, \hat d_t, \hat u_t)=f_\theta(s_t,a_t)
\]

就可以在真实执行前先做想象、搜索或风险评估。

!!! note "难点解释：世界模型不是普通视频预测"
    普通视频预测可能只关心画面像不像。世界模型必须对动作敏感：换一个动作，未来应该不同；同一个动作在不同状态下，风险也应该不同。真正服务 RL 的 world model 要预测对决策有用的状态、奖励、终止和不确定性。

## 2. PlaNet：学 latent dynamics，再在线规划

PlaNet 是从像素控制走向现代世界模型 RL 的关键节点。它先学 RSSM latent dynamics，再用 CEM 在 latent space 里搜索未来动作序列。

![PlaNet latent dynamics model designs 原论文图](../assets/images/paper-deep-dives/world-models/planet/figure-2-latent-dynamics.png){ width="860" }

<small>图源：[Learning Latent Dynamics for Planning from Pixels](https://arxiv.org/abs/1811.04551)，Figure 2。原论文图意：比较 RNN、SSM 和 RSSM 三种 dynamics 结构；RSSM 同时保留 deterministic memory 和 stochastic latent state。</small>

!!! note "图解：RSSM 为什么适合规划"
    机器人和控制任务经常是部分可观测的：当前画面不一定包含速度、接触历史和遮挡信息。RSSM 用 deterministic state 记历史，用 stochastic state 表达不确定性。训练时 posterior 可以看真实观测，规划时 prior 只能根据过去状态和动作往前 rollout。这个结构正好对应“执行前先想象未来”的需求。

PlaNet 的规划闭环可以理解为：

```text
当前观测 -> 编码成 latent belief
  -> 采样多条未来 action sequences
  -> 用 RSSM 预测 latent future 和 reward
  -> 选择预测回报最高的动作序列
  -> 执行第一个动作
  -> 收到新观测后重新规划
```

## 3. Dreamer：在想象中训练 actor-critic

PlaNet 每一步都要在线搜索，成本高。Dreamer 的关键变化是：在 learned world model 的 imagined trajectories 上训练 actor 和 value model，执行时 actor 直接给动作。

![Dreamer algorithm 原论文图](../assets/images/paper-deep-dives/world-models/dreamer/algorithm-1-dreamer.png){ width="820" }

<small>图源：[Dream to Control: Learning Behaviors by Latent Imagination](https://arxiv.org/abs/1912.01603)，Algorithm 1。原论文图意：Dreamer 交替进行 dynamics learning、behavior learning 和 environment interaction。</small>

!!! note "图解：RL 在 Dreamer 里发生在哪里"
    Dreamer 先用真实 replay 训练 world model；再固定 world model，从真实 latent states 出发展开 imagined rollout；最后用 imagined reward 和 value bootstrap 训练 actor/critic。这里的强化学习不完全依赖真实环境每一步试错，而是在 world model 中获得大量便宜的训练信号。

Dreamer 的重点是把世界模型变成策略训练器：

| 组件 | 训练信号 | 用途 |
| --- | --- | --- |
| representation / transition | 真实轨迹的观测、动作、reward | 学 latent dynamics |
| reward model | 真实 reward | 给 imagined rollout 打分 |
| value model | imagined return | 估计长时收益 |
| action model | value gradients 或 policy objective | 输出真实执行动作 |

## 4. Decision Transformer：轨迹 token 化的另一条入口

Decision Transformer 给世界模型和 VLA 一个重要启发：轨迹不一定只能通过 Bellman backup 学，也可以被组织成序列建模问题。

![Decision Transformer architecture 原论文图](../assets/images/paper-figures/reinforcement-learning/decision-transformer-architecture.png){ width="920" }

<small>图源：[Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/abs/2106.01345)，Figure 1。原论文图意：把 return-to-go、state、action 交替输入 GPT decoder，用目标回报条件化动作预测。</small>

!!! note "图解：它和世界模型有什么关系"
    世界模型常建模 \(p(s_{t+1:t+H}\mid s_t,a_{t:t+H-1})\)，Decision Transformer 则建模 \(p(a_t\mid \hat R_{\le t},s_{\le t},a_{<t})\)。前者更像“预测动作后果”，后者更像“给定目标回报生成动作”。如果把二者结合起来，可以先用世界模型评估候选未来，再把高回报轨迹作为条件策略学习材料。VLA 中的 action token、action chunk 和 trajectory transformer 都可以从这张图建立直觉。

!!! warning "边界：离线轨迹建模不保证真实闭环"
    Decision Transformer 在离线数据上很自然，但真实机器人和世界模型会遇到分布外状态。模型可能会在数据中没见过的高 return 条件下输出看似合理、实际不可执行的动作。因此读这类论文时，要额外看 closed-loop evaluation、数据覆盖和失败恢复。

## 5. RLVR-World：用可验证奖励训练世界模型本身

PlaNet/Dreamer 主要把 world model 用于规划或策略学习。RLVR-World 进一步强调：世界模型本身的训练目标也可以从 MLE/SFT 转向任务指标。

![RLVR-World concept 原论文图](../assets/images/paper-figures/reinforcement-learning/rlvr-world-concept.png){ width="920" }

<small>图源：[RLVR-World: Training World Models with Reinforcement Learning](https://arxiv.org/abs/2505.13934)，Figure 1。原论文图意：左侧是用 MLE 等 surrogate objective 训练 world model，右侧是用 RLVR 直接优化 decoded prediction 的可验证任务指标。</small>

!!! note "图解：为什么 MLE 和任务指标会错位"
    MLE 让模型提高训练数据序列的似然，但世界模型最终常被下游拿来做 state prediction、web navigation、robot trajectory prediction 或 real2sim evaluation。似然高不一定代表预测状态更准，视频 MSE 低也不一定代表控制相关细节正确。RLVR 的思路是把 decoded prediction 与 ground truth 比较，直接用 Accuracy、F1、MSE、LPIPS、SSIM 等可验证指标作为 reward。

RLVR-World 的训练逻辑可以压缩成：

```text
状态 s + 动作 a
  -> 世界模型采样多组下一状态预测
  -> 解码成结构化状态或视频帧
  -> 用任务指标和 ground truth s' 比较
  -> 把指标作为 reward，用 GRPO/PPO 更新模型
```

!!! warning "关键风险：可验证不等于完全正确"
    可验证奖励降低了 reward model 主观性，但仍可能遗漏真实任务关心的东西。比如视频指标可能奖励像素相似，却没有检查物理一致性；结构化状态 F1 可能忽略长期可执行性。因此 RLVR 适合补齐任务指标错位，但仍需要闭环评测和失败样本回放。

## 6. 强化学习给世界模型带来的五个增量

| 增量 | 具体作用 |
| --- | --- |
| 数据收集闭环 | policy 会把数据推向任务相关状态，而不是只停在随机分布 |
| 长期信用分配 | reward/value 把未来结果反馈到早期动作或预测 |
| 任务指标对齐 | RLVR 可以直接优化 decoded prediction 的任务指标 |
| 规划与反事实 | world model 支持“如果换个动作会怎样”的内部模拟 |
| 风险敏感控制 | reward 可以显式加入碰撞、失败、不可恢复状态和不确定性惩罚 |

## 7. 和 VLA 的连接

VLA 学的是：

\[
\pi_\theta(a_t\mid o_{\le t},l)
\]

世界模型学的是：

\[
p_\theta(o_{t+1:t+H},r,d\mid o_{\le t},a_{t:t+H-1},l)
\]

强化学习把两者连起来：

1. 用 world model 预测 action 的后果；
2. 用 reward/value 判断哪条后果更好；
3. 更新 policy，使它更倾向于高回报动作；
4. 用真实闭环数据修正 world model 的盲点。

机器人场景里，很多真实数据来自历史实验或遥操作日志，不能像游戏一样无限重开。Offline RL 教会我们：历史数据不只是“模仿材料”，也可以通过 value learning 变成决策材料，但必须处理分布外动作和价值高估。

![Offline RL robotic grasping data](../assets/images/paper-figures/reinforcement-learning/offline-rl-qtopt-robot-grasping.png){ width="560" }

<small>图源：[Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems](https://arxiv.org/abs/2005.01643)，Figure 7。原论文图意：展示大规模机器人抓取数据收集，相关工作用超过 500,000 次抓取试验训练视觉抓取策略，并比较纯离线训练与在线微调。</small>

!!! note "图解：为什么 VLA 不能只靠行为克隆"
    行为克隆会学习“人或旧策略在这些画面下怎么做”，但它不会天然知道“换一个动作会不会更稳”。Offline RL / model-based RL 尝试从历史轨迹里估计动作后果：哪些抓取姿势长期成功率高，哪些动作看似接近目标但会导致碰撞或滑落。世界模型进一步把这种动作后果预测显式建出来，让策略可以先在内部比较候选动作。

!!! example "有趣例子：机器人抓杯子"
    VLA 可能学会看到杯子就伸手。世界模型会预测从左侧推可能把杯子推倒，从上方夹取更稳。RL 再把“成功夹起、没有碰撞、动作平滑”变成 reward，让策略在类似状态中更倾向于安全动作。

## 8. 读论文时的检查清单

| 检查项 | 要问的问题 |
| --- | --- |
| 动作条件是否真实生效 | 换动作时，未来预测是否真的变化 |
| reward 是否任务相关 | 优化指标是否接近下游规划或控制目标 |
| rollout horizon 多长 | 长时预测是否只靠短期视觉质量支撑 |
| 是否有闭环验证 | 只做 open-loop prediction，还是进入控制或规划 |
| 是否评估 model exploitation | policy 或 planner 会不会利用模型漏洞拿虚高回报 |
