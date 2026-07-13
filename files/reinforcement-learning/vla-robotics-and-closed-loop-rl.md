# VLA、机器人与闭环强化学习

Vision-Language-Action（VLA）模型把视觉观察和语言任务映射为机器人动作。能预测示范动作只是起点：真实机器人还要在部分可观测、连续控制和安全约束下反复执行、重新观察、恢复失败。闭环强化学习关注的是这条执行链最终能否提高回报与成功率，而不是单帧动作误差是否更小。

## 本章回答什么

本章回答：机器人为什么更接近 POMDP 而不是完全可观测 MDP；continuous action 与 action chunk 如何进入策略；demonstration、offline pretraining、simulation、world-model rollout 和 real deployment data 各自提供什么证据；open-loop action prediction、simulator closed-loop success 与 real closed-loop success 为什么不能互相替代；安全约束、恢复和 human intervention 如何进入数据闭环。

## 先修知识

- [Agent、环境与训练闭环](rl-agent-environment-loop.md)：observation、state、trajectory 与 termination。
- [Offline Reinforcement Learning](offline-reinforcement-learning.md)：固定机器人日志的 dataset support 和 distribution shift。
- [Model-based RL 与世界模型](rl-for-world-models.md)：MPC、world-model rollout 与 model exploitation。
- [VLA 动作表示与接口](../vla/action-representation-and-interfaces.md)：坐标系、控制频率和动作 token/chunk。
- [规划、控制与安全](../embodied-ai/planning-control-and-safety.md)：policy 输出之后的 controller 与 safety filter。

## 最小任务

任务是“拿起桌上的杯子放入托盘”。每个控制周期收到多相机图像、关节状态和语言指令，policy 输出未来 $K$ 步的末端位姿与夹爪命令。最小闭环为：


```text
observe -> update history/belief -> VLA proposes action chunk
  -> safety filter/controller -> execute a short prefix
  -> observe outcome -> continue, replan, recover, or stop
```

不要一次执行完整长 chunk 后才检查结果。执行前缀越长，推理调用越少，但环境变化和模型误差越可能积累；前缀越短，反馈及时，但 latency 与动作抖动更突出。

最小验收必须分三层：

| 评测层 | 指标例子 | 能证明什么 | 不能证明什么 |
| --- | --- | --- | --- |
| open-loop action prediction | action L1/L2、chunk likelihood、示范匹配 | held-out 日志上接近记录动作 | policy 能纠错、任务会成功 |
| simulator closed-loop success | success、return、collision、recovery rate | 在该 simulation 动力学和传感器模型中可闭环 | 真实摩擦、延迟与长尾安全 |
| real closed-loop success | 真实任务成功、人工接管、near-miss、cost per success | 在指定真实设置中的闭环表现 | 未测试场景、硬件和人群上的普遍泛化 |

## 核心机制

核心设计同时包含 POMDP belief、continuous action 与 action chunk、demonstration/offline pretraining、simulation/world-model rollout、real deployment data，以及独立于 reward 的 safety constraint、recovery 与 human intervention；缺一项都可能让 distribution shift 在闭环中失控。

### POMDP、history 与 belief

机器人真实 state 包含物体三维姿态、接触力、摩擦、遮挡后物体位置和人的意图；摄像头与 proprioception 只给 observation。因此更合适的形式是 POMDP。策略需要利用历史观察、动作和反馈形成 belief：


\[
b_t=p(s_t\mid o_{\le t},a_{<t}),\qquad
a_t\sim\pi_\theta(a_t\mid b_t,l).
\]

VLA 的视觉 token、历史帧、语言目标和 recurrent/KV memory 都可能承载 belief，但长上下文不保证保留接触事件。应专门测试遮挡、物体滑动和动作执行失败后的状态更新。

### Continuous action 与 action chunk

机器人常见 continuous action 包括关节位置/速度/力矩，或末端平移、旋转与夹爪状态。VLA 可能直接回归连续量，也可能离散化为 action token。无论表示方式，都必须写清坐标系、单位、频率、归一化和 controller 接口。

action chunk 一次预测 $a_{t:t+K-1}$，减少推理频率并表达连贯技能；但真实系统通常只执行短 prefix 后重新观察。chunk 长度、执行前缀与 replanning frequency 是闭环设计参数，不只是模型输出 shape。

### Demonstration 与 offline pretraining

demonstration 提供“人或既有控制器做过什么”，适合 behavior cloning 和 offline pretraining。它能快速获得基础 manipulation skill，避免从危险随机探索开始，却不能提供数据外动作的真实反事实结果。

失败数据同样重要：碰撞、抓空、滑落、人工接管和 recovery trajectory 能定义风险边界。只保留 imitation success 会让策略不知道失败前兆，也无法学习“何时停止或求助”。reward design 与 imitation success 必须分开：模仿标签描述示范动作，reward 描述任务进展、代价和约束，两者可能不一致。

### Simulation、world model 与真实数据

simulation 可以批量生成 closed-loop interaction、随机化物体和测试恢复；world-model rollout 可以更便宜地比较 action consequence；两者都可能有 model bias。策略会主动利用穿模、错误接触或不准确 risk head，因此仿真高回报必须通过 real deployment data 校准。

真实部署数据最昂贵，也最接近目标环境。合理顺序通常是：demonstration/offline pretraining 建立初始策略，simulation 与 world-model rollout 扩展覆盖，受限真实试验验证，再把失败和干预回流。不是所有阶段都必须用 PPO；有些阶段是 imitation、offline RL、MPC 或安全约束优化。

### Safety constraint、recovery 与 human intervention

安全不能只压成一个 reward。碰撞、速度、工作空间、力矩、隐私和人机距离常是硬 constraint，由 controller、safety filter、急停和权限系统执行。reward 可以鼓励效率或平滑，但不能保证稀有危险永不发生。

recovery policy 处理可恢复失败，例如重新定位、松开重抓或回到安全姿态；不可恢复或高风险状态应触发停止与 human intervention。接管不是“无效样本”，而是重要标签：它表明当时 policy confidence、risk estimate 或任务状态不足以安全自治。

### Distribution shift 是闭环产生的

相机、光照、物体、机器人动力学会造成外部 shift；更隐蔽的是 policy-induced shift。策略动作改变后续 observation，微小偏差会把系统带到 demonstration 没覆盖的状态。open-loop action prediction 无法暴露这条反馈链，必须用 closed-loop rollout 和失败桶观察。

## 训练数据流


```text
demonstrations + historical failures
  -> schema/calibration/episode audit
  -> imitation or offline pretraining
  -> simulation + world-model rollout + safety tests
  -> constrained closed-loop deployment with human intervention
  -> bucket success, failure, recovery and near-miss trajectories
  -> relabel reward/risk/termination and retrain
  -> regression gate before the next deployment
```

这个 flywheel 不是“自动采更多数据”就会变好。采样 policy、场景分布、人工接管规则和 reward 版本都会改变数据含义，必须随 trajectory 保存。

| 数据阶段 | 应记录 | 主要用途 |
| --- | --- | --- |
| demonstration | observation、action、task、operator、calibration | imitation/offline pretraining |
| simulation | seed、domain randomization、sim version、collision | closed-loop 压测和覆盖扩展 |
| world-model rollout | model version、uncertainty、candidate actions | 规划与反事实排序，不当真实证据 |
| real deployment data | hardware、latency、intervention、near-miss | 真实分布校准与失败回流 |

训练集、调参集和最终闭环评测场景要分开。重复在同一房间、物体和初始位姿上迭代，会把数据 flywheel 变成测试集过拟合。

## 常见失败

1. **动作回归误差低就宣称机器人成功。** 多模态 demonstration 可有多个正确动作，open-loop 距离也看不到反馈纠错。
2. **simulation success 当 real closed-loop success。** sim-to-real gap 包括视觉、接触、延迟、磨损和人的不可预测行为。
3. **奖励越密越好。** shaping reward 可能鼓励靠近目标却反复碰撞；必须同时监控 constraint violation。
4. **把 imitation success 当 reward design 已解决。** 示范动作可完成任务，但不定义效率、风险、恢复和长期代价。
5. **长 action chunk 全部执行。** 环境变化后旧计划仍继续，容易把可恢复偏差变成碰撞。
6. **只训练成功轨迹。** policy 看不到何时停止、恢复或请求 human intervention。
7. **world-model rollout 当真实机器人小时。** 模型样本只能证明模型内行为，应分别报告 simulated/model/real steps。

## 自检与练习

1. 为杯子放置任务列出 latent state 与可观测 observation，说明 belief 需要哪些历史。
2. 设计 $K=16$ 的 action chunk，但只执行前 4 步；列出重新规划的触发条件。
3. 对 open-loop、simulator closed-loop 和 real closed-loop 各写一个能支持的 claim 与一个不能支持的 claim。
4. 把碰撞、任务进度和人工接管分别放到 reward、constraint 或 termination 中，并解释选择。
5. 设计 failure replay：如何采样抓空、滑落、遮挡和 recovery，避免平均成功样本淹没长尾。

权威起点：

- [RT-2](https://arxiv.org/abs/2307.15818)：把视觉语言表征接到机器人动作 token 的 VLA 代表。
- [Open X-Embodiment](https://arxiv.org/abs/2310.08864)：多机器人 demonstration 数据与跨 embodiment 学习。
- [OpenVLA](https://arxiv.org/abs/2406.09246)：开源 VLA 训练与评测案例。
- [Learning Latent Dynamics for Planning from Pixels](https://arxiv.org/abs/1811.04551)：部分可观测 latent planning 的基础。

## 下一步

- 前一章：[LLM 强化学习：RLHF、RLVR 与 GRPO](llm-rlhf-rlvr-and-grpo.md)，回顾 rollout、reward 与 policy objective。
- 下一章：[RL 系统工程与 verl](verl-code-flow.md)，把生成、评价和更新的数据流扩展到训练系统。
- 方向补充：[VLA 闭环恢复与失败分析](../vla/closed-loop-recovery-and-failure-analysis.md)、[VLA 部署与安全](../vla/deployment-and-safety.md) 和 [具身闭环任务 walkthrough](../embodied-ai/closed-loop-task-walkthrough.md)。
