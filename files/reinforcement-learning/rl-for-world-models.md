# Model-based RL 与世界模型

Model-based RL 显式使用环境模型预测动作后果，再用规划或策略学习选择动作。模型可以是已知规则，也可以从真实 transition 中学习；world model 是 learned model 的一种，通常在 latent space 中预测状态、reward 与终止。统一判断标准是模型是否改善 closed-loop return。

## 模型怎样成为决策工具

核心链条是：已知或 learned transition model 如何进入规划；Dyna 怎样混合真实经验与模型生成经验；MPC 为什么每步只执行计划的第一个动作；PlaNet 与 Dreamer 怎样在 latent dynamics 中规划或训练 actor；model bias、compounding error 与 model exploitation 为什么必须回到真实闭环校准。

LLM 的 GRPO、视频生成和机器人系统只在与这条链路相关时出现，重点是模型怎样成为 RL 的决策工具。

## 先修知识

- [MDP、价值函数与 Bellman](mdp-value-bellman.md)：真实 transition、reward、done 和长期价值。
- [Offline Reinforcement Learning](offline-reinforcement-learning.md)：固定数据的 support 与分布外外推风险。
- [Actor-Critic、GAE 与 PPO](policy-gradient-actor-critic-ppo-grpo.md)：Dreamer 中 actor/critic 的基础。
- [世界模型中的 RSSM、Dreamer 与规划](../world-models/rssm-dreamer-and-planning.md)：更深入的模型结构专题。

## 最小任务

考虑一个小车到达目标的任务。环境模型接收 (s_t,a_t)，输出下一状态、奖励和终止概率：


\[
(\hat s_{t+1},\hat r_t,\hat d_t)=f_\phi(s_t,a_t).
\]

比较两种模型：

| 模型 | 来源 | 能做什么 | 主要误差 |
| --- | --- | --- | --- |
| known model | 物理方程或完整模拟器 | 直接规划、生成 transition | 参数与真实系统不匹配 |
| learned model | 真实交互或离线日志拟合 | 在未知动力学中预测与想象 | 数据覆盖、训练误差与分布偏移 |

最小规划器枚举长度为 $H$ 的候选动作序列，用模型计算：


\[
\hat G=\sum_{k=0}^{H-1}\gamma^k\hat r_{t+k}
+\gamma^H\hat V(\hat s_{t+H}),
\]

选择预测回报最高的序列，但只执行第一个动作，然后接收新观察再规划。验收时同时比较 model prediction error、action sensitivity 和真实 closed-loop return。

## 核心机制

### 从 model-free 到 Dyna

Model-free RL 直接从真实 transition 更新 value 或 policy。Dyna 在同一 replay 中加入 model learning 和 planning：真实 transition 一方面更新 value/policy，另一方面训练 learned model；随后模型从已有状态产生 imagined transition，继续做 Bellman update。


```text
real environment -> real transition -> value/policy update
                         |-> learned transition/reward/done model
                                  |-> imagined transition -> value/policy update
```

模型样本便宜，但不是独立事实。若 learned model 漏掉碰撞，增加 imagined rollout 数量只会更快传播错误。

### Planning 与 MPC

给定 known model 或 learned model，planner 可以用动态规划、树搜索、随机 shooting 或 CEM 比较动作序列。Model Predictive Control（MPC）的关键是 receding horizon：每一步根据最新 observation 重新建立状态或 belief，重新计算候选序列。

MPC 缓解但不消除 compounding error。短 horizon 降低长期模型漂移，却可能看不到延迟收益；长 horizon 看得更远，却更容易被 model bias 支配。终端 value bootstrap、风险惩罚和不确定性阈值都是这个权衡的一部分。

### PlaNet：latent planning

PlaNet 把像素观察编码为 latent belief，用 Recurrent State-Space Model（RSSM）学习 latent dynamics，再通过 CEM 做在线规划。

![PlaNet latent dynamics model designs](../assets/images/paper-deep-dives/world-models/planet/figure-2-latent-dynamics.png){ width="860" }

<small>图源：[Learning Latent Dynamics for Planning from Pixels](https://arxiv.org/abs/1811.04551)，Figure 2。原图比较 RNN、SSM 与 RSSM dynamics；这里用于说明 deterministic memory 与 stochastic latent state 如何共同表示部分可观测控制状态。</small>


```text
latest observation -> latent belief
  -> candidate action sequences
  -> latent dynamics + reward rollout
  -> choose highest predicted return
  -> execute first action -> observe again
```

latent reconstruction 好不等于规划好。真正相关的是：更换 action 时预测是否相应改变，也就是 action sensitivity；预测的 reward、done 与风险是否校准；接入 planner 后 return 是否提高。

### Dreamer：latent imagination 训练 policy

Dreamer 不在每个执行步做昂贵搜索，而是从真实 replay 的 latent state 出发，在 world model 中展开 imagined trajectories，用它们训练 actor 与 critic；部署时 actor 直接输出动作。

![Dreamer algorithm](../assets/images/paper-deep-dives/world-models/dreamer/algorithm-1-dreamer.png){ width="820" }

<small>图源：[Dream to Control](https://arxiv.org/abs/1912.01603)，Algorithm 1。原图展示 dynamics learning、behavior learning 与 environment interaction 的交替；这里强调 imagined rollout 仍以真实 replay 校准。</small>

| 子模型 | 监督信号 | 被谁消费 |
| --- | --- | --- |
| representation + transition | observation/action sequence | imagined rollout |
| reward head | real reward | return target |
| continuation/done head | real episode boundary | discount 与终止判断 |
| actor/critic | imagined reward、continuation 与 bootstrap value | 执行策略 |

### 模型错误如何进入决策

**Model bias** 是 learned model 对真实动力学的系统偏差。多步 rollout 会把每一步偏差带到新的输入分布，形成 **compounding error**。更危险的是 **model exploitation**：policy 或 planner 会主动寻找模型过于乐观的动作，而不是被动承受平均预测误差。

例如模型没学会杯子与桌沿的接触，想象中“高速推”可穿过边界获得高 reward；actor 会利用这个漏洞。常见缓解包括缩短 horizon、真实 value bootstrap、模型集成与 uncertainty penalty、限制动作到数据支持区域，以及持续用 real data 检查失败桶。

### Open-loop prediction 与 closed-loop return

| 证据 | 测量什么 | 不能单独证明什么 |
| --- | --- | --- |
| one-step prediction | 局部 transition 拟合 | 长 horizon 稳定、planner 可用 |
| open-loop rollout | 固定动作序列下的预测误差 | 策略会选择正确动作 |
| counterfactual action test | 不同动作是否产生可辨别后果 | reward/risk 校准正确 |
| model-based closed-loop return | 模型进入 planner/policy 后的任务结果 | 跨环境与真实机器人泛化 |

自然视频可能忽略动作差异，仍有较好的视觉指标；这样的模型是有价值的预测先验，但不能仅凭 open-loop prediction 称为控制模型。决策模型还必须校准 terminal、reward 与 risk：done 太晚会让规划器穿过失败状态，reward 偏置会改变动作排序，风险低估会诱导危险探索。

## 相邻路线的归属边界

### 四条容易混淆的路线

Model-based RL 的责任是让环境模型进入预测、规划或策略学习；相邻工作也可能使用 trajectory、reward 或 GRPO，但优化对象不同。下面只保留迁移桥，不把四条路线重新展开成独立综述。

| 路线 | 课程归属 | 优化对象 | 证据边界 |
| --- | --- | --- | --- |
| RLVR-World | Model-based RL / learned transition | 用 RL 和 verifiable reward 直接优化 learned transition/world predictor 采样的 decoded prediction | verifier 指标改善不能证明 real closed-loop return；仍需 action sensitivity、transition 校准和 planner 评测。[RLVR-World](https://arxiv.org/abs/2505.13934) |
| Decision Transformer | [Offline RL](offline-reinforcement-learning.md) | 对 fixed trajectory 做 return-conditioned sequence modeling，并从固定数据预测 action | 它不是 online environment interaction，仍受数据覆盖和分布外动作限制。[Decision Transformer](https://arxiv.org/abs/2106.01345) |
| DDPO / Flow-GRPO / video generator RL | [LLM 强化学习中的通用 policy objective](llm-rlhf-rlvr-and-grpo.md)；[视频与多模态扩散](../diffusion/video-and-multimodal-diffusion.md) | 用 reward 优化 diffusion/flow generator/sampler 的生成轨迹 | 这是生成器策略优化，不是 learned environment transition，也不是 LLM 算法；DDPO 与 Flow-GRPO 的 logprob/采样器定义应分别核对。[DDPO](https://arxiv.org/abs/2305.13301)、[Flow-GRPO](https://arxiv.org/abs/2505.05470) |
| interactive video worlds | [世界模型评测与失败模式](../world-models/evaluation-and-failure-modes.md)；[VLA 闭环恢复与失败分析](../vla/closed-loop-recovery-and-failure-analysis.md) | 生成可受动作影响的视觉环境，供 agent 交互或评测 | visual playability/open-loop demo 不能证明 real closed-loop，更不能替代真实机器人风险与回报证据。[Sora](https://openai.com/index/video-generation-models-as-world-simulators/)、[Genie 2](https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/) |

## 训练数据流


```text
real transition (observation, action, reward, done)
  -> train representation + transition + reward + continuation/risk heads
  -> start from real replay states
  -> plan with MPC or generate imagined rollout
  -> update value/policy or select the first action
  -> execute in environment and collect fresh real data
  -> audit model error, exploitation and closed-loop return
```

数据闭环中必须把“模型训练指标”和“策略消费指标”分开：

| 层 | 指标示例 |
| --- | --- |
| 模型拟合 | one-step state error、reward error、done AUROC、risk calibration |
| 长期预测 | horizon-bucketed latent/state error、action sensitivity |
| 规划接口 | candidate ranking、planner latency、horizon 与采样预算 |
| 决策结果 | return、success、constraint violation、real environment step |

模型用离线日志预训练时，要继承 Offline RL 的 support 警告；上线收新数据后，又要记录当前 policy 改变了哪些状态分布。两部分缺一不可。

## 常见失败

1. **只优化 reconstruction。** 表征可能保留纹理却丢失速度、接触或小目标，导致 action sensitivity 下降。
2. **把 imagined transition 当成真实样本计数。** 应分别报告 real environment steps 与 model steps。
3. **rollout horizon 越长越好。** 长 horizon 会放大 compounding error，应按 horizon 分桶并做真实轨迹对照。
4. **reward/done/risk head 缺席。** 只有下一帧预测时，planner 没有可靠的任务排序和失败边界。
5. **open-loop 指标代替 closed-loop return。** 动作固定时预测准确，不代表模型能比较策略将主动选择的动作。
6. **模型集成方差等于真实风险。** ensemble disagreement 只能覆盖模型族表达出的认知不确定性，仍可能共同漏掉同一类危险。

## 自检与练习

1. 对同一任务分别画出 Dyna、MPC/PlaNet 与 Dreamer 的 real data 和 imagined data 流向。
2. 为什么 MPC 只执行动作序列的第一步？这能消除哪些误差，不能消除哪些误差？
3. 设计 counterfactual action test：固定当前 observation，改变三个候选动作，检查哪些预测量必须变化。
4. 给 reward、done 和 risk head 各写一个 calibration 指标，并解释它如何影响 planner。
5. 找一个 open-loop 指标改善但 closed-loop return 下降的可能机制。

权威起点：

- [Integrated Architectures for Learning, Planning, and Reacting Based on Approximating Dynamic Programming](https://dl.acm.org/doi/10.1145/122344.122377)：Dyna 的原始框架。
- [Learning Latent Dynamics for Planning from Pixels](https://arxiv.org/abs/1811.04551)：PlaNet 的 latent dynamics 与在线规划。
- [Dream to Control](https://arxiv.org/abs/1912.01603)：Dreamer 的 latent imagination actor-critic。
- [DreamerV3](https://arxiv.org/abs/2301.04104)：统一 world model 与 imagined behavior learning 的后续系统。
