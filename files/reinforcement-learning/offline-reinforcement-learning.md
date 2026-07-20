# Offline Reinforcement Learning

Offline Reinforcement Learning（Offline RL）只使用已经记录好的固定数据集训练策略，不在训练期间向环境请求新 transition。它适合机器人、医疗流程和推荐系统这类真实探索昂贵或危险的场景，但也把一个问题放到中心：策略一旦选择数据集没有覆盖的动作，价值估计还能不能相信？

## 本章回答什么

本章回答四个问题：Offline RL 与 behavior cloning、off-policy online RL 有什么区别；dataset support 为什么决定策略能学到什么；OOD action 如何造成 extrapolation error 和 value overestimation；离线指标为什么不能替代真实 closed-loop 证据。

主线是一个固定机器人数据集。日志包含相机观测、机械臂动作、奖励和终止标记，训练时不能让机器人试新动作。目标不是“把日志拟合得更好”，而是在不越出可靠数据支持的前提下，提高长期回报。

## 先修知识

- [MDP、价值函数与 Bellman](mdp-value-bellman.md)：理解 transition、return、value 与 Bellman backup。
- [动态规划、Monte Carlo 与 TD](dynamic-programming-monte-carlo-td.md)：理解 bootstrap 为什么会传播估计误差。
- [连续控制与 SAC](continuous-control-and-sac.md)：理解 replay buffer、off-policy update 与连续动作。
- [机器人状态、动作与数据集](../embodied-ai/state-action-robot-datasets.md)：理解机器人日志中的坐标系、频率和 episode boundary。

## 最小任务

假设固定数据集 $\mathcal D$ 来自一个行为策略 $\mu$：


```text
observation -> 7-DoF end-effector action -> reward + next observation + done
```

数据大多是保守抓取，只有少量接近桌沿的失败。比较三种训练方式：

| 方法 | 训练时的数据来源 | 能否主动纠正数据盲区 | 主要风险 |
| --- | --- | --- | --- |
| behavior cloning | fixed dataset 中的示范动作 | 不能 | 复制示范者错误；多步误差累积 |
| off-policy online RL | replay 数据加持续环境交互 | 能，用新 transition 修正 | 在线探索成本与安全风险 |
| offline RL | fixed dataset，不再交互 | 不能 | OOD action 的 value 被错误高估 |

最小验收不是看训练 loss，而是同时报告：数据覆盖、策略动作相对行为数据的偏移、离线评估不确定性，以及进入仿真或真实机器人前的安全门禁。

## 核心机制

在 fixed dataset 条件下，behavior cloning、off-policy online RL 与 Offline RL 的关键差异，是策略遇到 distribution shift、离开日志支持区域后能否通过新交互纠错；下面从这个边界解释各类机制。

### Dataset support 是可信区域

数据由 behavior policy $\mu(a\mid s)$ 产生。若候选策略 $\pi$ 在某个状态选择 $\mu$ 几乎从未选择的动作，就离开了 dataset support。固定数据里没有这个动作的真实后果，Bellman target 却仍可能查询它：


\[
y=r+\gamma(1-d)\max_{a'}Q(s',a').
\]

其中 $d$ 表示真实终止；最大化会偏好被高估的 OOD action。错误 target 再经过 bootstrap 传播，形成 extrapolation error 和 value overestimation。在线 off-policy 方法还能执行该动作并收集反例；Offline RL 没有这条纠错通道。

### 三类保守办法

| 思路 | 做法 | 代表例子 | 边界 |
| --- | --- | --- | --- |
| behavior constraint | 只在行为数据附近选动作 | BCQ | 约束太强会退化为模仿，错过数据中可组合的改进 |
| conservative value estimation | 压低数据外动作的 Q，保留数据内动作价值 | CQL | 保守强度过大时会系统性低估可行改进 |
| avoid OOD action maximization | 只对数据动作拟合 value，再对高 advantage 数据做加权策略学习 | IQL | 仍依赖数据覆盖和 expectile/温度等选择 |

BCQ、CQL、IQL 是准确理解这些机制的短例子，不是算法排行榜。BCQ 通过生成模型和扰动限制候选动作；CQL 在标准 Bellman error 外加入保守目标；IQL 避免直接评价未在数据中出现的动作，并做 advantage-weighted policy extraction。

### Distribution shift 有两层

第一层是 action shift：$\pi(a\mid s)$ 与 $\mu(a\mid s)$ 不同。第二层是 state shift：动作的小偏差在闭环中累积，把机器人带到日志很少覆盖的状态。即使单步动作误差很小，长 horizon 也可能让 observation 分布改变。

因此 behavior cloning 不是 Offline RL 的同义词。BC 直接拟合动作标签；Offline RL 使用 reward 和长期价值寻找比平均行为更好的策略，但必须显式处理 support 外推。两者都无法凭固定日志证明真实部署安全。

### Offline policy evaluation 的边界

重要性采样可写为：


\[
\hat J(\pi)=\frac{1}{N}\sum_{i=1}^{N}
\left(\prod_t\frac{\pi(a_t^i\mid s_t^i)}{\mu(a_t^i\mid s_t^i)}\right)G_i.
\]

其中连乘项是整条 trajectory 的策略概率比；当策略差异和 horizon 增大时，权重方差会迅速上升。若行为概率未记录或 support 不重叠，估计甚至不可用。FQE 等直接价值估计方法会引入函数逼近和 Bellman 外推误差；基于 learned dynamics 的 OPE 还会额外引入 model bias。Offline policy evaluation 可以筛掉明显失败方案，不能把日志之外的真实后果变成事实。

## 训练数据流


```text
固定日志 D
  -> 校验 observation/action/reward/done 与 episode 边界
  -> 统计 state-action coverage、成功/失败桶和行为策略信息
  -> 训练 Q / V / behavior model / policy
  -> 约束或惩罚 dataset support 外的动作
  -> logged data 上做 OPE、切片评测和不确定性检查
  -> 仿真回放与 safety filter
  -> 小流量真实 closed-loop 验证并记录失败
```

机器人数据要特别保留失败、人工接管、碰撞 near-miss 和恢复动作。只保留成功示范会让 reward/value 看不到危险边界，也让任何保守估计失去校准材料。

| 阶段 | 必查字段 | 通过条件 |
| --- | --- | --- |
| 数据审计 | 动作单位、控制频率、terminated/truncated、reward 版本 | transition 语义一致，失败未被过滤 |
| 策略训练 | Q 分布、动作偏移、support 距离 | 没有依靠极端 OOD Q 获得虚假提升 |
| logged data 评估 | OPE 方差、有效样本量、分桶回报 | 结论写清覆盖范围与置信度 |
| 部署前验证 | 仿真成功、约束违反、恢复率 | safety constraint 和人工接管可用 |

## 常见失败

1. **把 logged data 上的高 Q 当成真实 closed-loop 成功。** Q 和 offline policy evaluation 都依赖数据覆盖与建模假设；真实闭环会访问新的状态分布。
2. **只比较平均 return。** 桌沿、遮挡、不同物体和长 horizon 失败可能被均值淹没，应按场景和风险桶拆分。
3. **数据没有记录行为概率，却硬套重要性采样。** 此时比率无法可靠计算，需要明确行为模型估计带来的误差。
4. **保守正则越大越安全。** 过强保守会拒绝数据内少量但有效的高回报行为；安全还需要动作限幅、碰撞检查、急停和 human intervention。
5. **仿真通过就直接全量部署。** sim-to-real distribution shift 仍可能来自摩擦、延迟、传感器噪声和人类行为。

证据边界应写成：logged data evaluation 支持“在这些数据和假设下更有希望”；仿真 closed-loop 支持“在该模拟器动力学下可工作”；只有受控 real closed-loop 才直接支持真实系统回报与风险的结论。

## 自检与练习

1. 为什么标准 Q-learning 在 fixed dataset 上会主动寻找估计误差，而 behavior cloning 通常不会？
2. 为固定抓取日志画出 dataset support：哪些物体、姿态、动作方向和失败类型被覆盖？
3. 比较 BCQ 的 behavior constraint、CQL 的 conservative value estimation 和 IQL 的 policy extraction，各自在哪一步减少 OOD action 风险？
4. 设计一张 deployment checklist，至少包含动作限幅、碰撞检查、恢复策略、人工接管和失败回流。
5. 解释为什么 OPE 置信区间很窄仍不能自动证明真实机器人安全。

权威起点：

- [Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems](https://arxiv.org/abs/2005.01643)：Offline RL 的问题定义、机器人案例与开放问题。
- [Off-Policy Deep Reinforcement Learning without Exploration（BCQ）](https://arxiv.org/abs/1812.02900)：behavior-constrained value learning。
- [Conservative Q-Learning](https://arxiv.org/abs/2006.04779)：保守价值估计。
- [Offline Reinforcement Learning with Implicit Q-Learning](https://arxiv.org/abs/2110.06169)：不直接查询数据外动作的 value learning 与 policy extraction。
