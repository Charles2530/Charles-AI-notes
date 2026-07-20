# 综合实践与算法选择

## 本章回答什么

算法选择要检查任务信息、动作、数据和证据是否满足算法假设。下面用一套紧凑决策流程和三个可执行 capstone 检查：GridWorld 的模型与 terminal mask、Pendulum 的连续控制更新、verifier LLM 的组内 advantage 与策略约束。

完成后应能写出选择理由、运行命令、保存可审计产物，并明确 smoke test、离线指标或 verifier 分数不能证明什么。

## 先修知识

- [Agent、环境与训练闭环](rl-agent-environment-loop.md)：observation、state、action、reward、terminated 与 truncated。
- [Q-learning 与 DQN](q-learning-and-dqn.md)、[Actor-Critic、GAE 与 PPO](policy-gradient-actor-critic-ppo-grpo.md) 和 [连续控制与 SAC](continuous-control-and-sac.md)：离散/连续动作与 on-policy/off-policy 数据。
- [Offline Reinforcement Learning](offline-reinforcement-learning.md) 和 [Model-based RL 与世界模型](rl-for-world-models.md)：fixed dataset support 与 learned dynamics 误差。
- [LLM 强化学习](llm-rlhf-rlvr-and-grpo.md) 和 [verl 训练流程](verl-code-flow.md)：verifier、group objective、版本与 mask 对齐。
- [示例 README](../assets/examples/reinforcement-learning/README.md)：隔离环境、完整命令矩阵和输出契约。

## 最小任务

三个项目都从仓库根目录、激活 `.venv-rl-course` 后运行。命令只验证数据流与至少一次有限更新；验收标准不把单次输出解释为算法优劣或收敛。

### Capstone 1：GridWorld 中比较 DP、TD 与 Q-learning

**输入**：[`gridworld.py`](../assets/examples/reinforcement-learning/gridworld.py) 中确定性的 2×3 GridWorld、uniform policy、已知 transition model 与 seeded Q-learning 交互。

**步骤**：运行命令；把 `policy_value_mean` 归为 DP policy evaluation，把 `optimal_value_start` 与 `greedy_policy` 归为 value iteration control，把 `q_mean` / `q_max` 归为 sampled Q-learning；再检查 `td_target` 与 Q update 在 `terminated=True` 时去掉 bootstrap，而 sampling boundary 不能冒充 true terminal。

```bash
python files/assets/examples/reinforcement-learning/gridworld.py --seed 7
```

**预期产物**：一张 DP/TD/Q 对照表，列出“是否需要模型、target 来源、估计对象、terminal mask”；一份保留上述命名字段的运行日志。

**验收标准**：命令零退出，三个 summary 行存在且有限；能解释 DP 用完整模型期望、TD 用 sampled one-step bootstrap、Q-learning 更新 action value；terminal mask 让真实终止 target 只保留即时 reward。

**失败诊断**：若 terminal 值非零，先核对进入终点的 reward 与 `terminated`；若 Q 摘要变化，先核对 seed 和探索路径，不拿短 run 要求等于 DP 最优值。

### Capstone 2：Pendulum 中比较 SAC 与 PPO

**输入**：Pendulum-v1 的三维 observation、区间 `[-2, 2]` 的连续 action，以及 [`deep_rl_smoke.py`](../assets/examples/reinforcement-learning/deep_rl_smoke.py) 的 SAC 路径。

**步骤**：先写选择理由：SAC 能用 replay 重用昂贵 online interaction，连续 action 由 squashed Gaussian actor 产生；PPO 也能处理连续 action，但依赖更新鲜的 on-policy rollout。运行 SAC smoke，再读取 actor/critic loss、temperature alpha、episode return 与两个参数变化字段。

```bash
python files/assets/examples/reinforcement-learning/deep_rl_smoke.py sac --seed 7 --steps 256
```

**预期产物**：一页 SAC vs PPO 决策记录；包含 `interaction_count`、`completed_episode_return_mean`、`current_partial_episode_return`、`updates`、`critic_loss`、`actor_loss`、`alpha`、`actor_parameter_delta` 和 `critic_parameter_delta` 的日志。

**验收标准**：命令零退出，actor、critic、alpha 与 return 相关字段有限，`updates=193` 且 actor/critic 参数变化为正。smoke 不是收敛证据；负 return 或单个 seed 不能证明 SAC 失败，也不能证明它优于 PPO。

**失败诊断**：参数不变时先查 optimizer 与 detach 边界；alpha 异常时查 tanh/action-scale logp 坐标；没有完整 episode 时区分 partial return 与 completed return，不把 partial 伪装成均值。

### Capstone 3：Verifier LLM 的组内策略目标

**输入**：[`policy_objectives.py`](../assets/examples/reinforcement-learning/policy_objectives.py) 中两个 prompt group、raw reward、policy/reference logp 与 clipped objective。

**步骤**：运行命令；逐组检查 group advantage；确认 format 组是 zero group，arithmetic 组产生正负相对信号；区分 sampled KL 的逐点 log-ratio 与 full expected KL；记录 clipped loss；最后设计一个能骗过格式 verifier 的 reward hacking 样本及隐藏检查。

```bash
python files/assets/examples/reinforcement-learning/policy_objectives.py --seed 7
```

**预期产物**：按 prompt/candidate 保存 raw reward、group advantage、pointwise penalty 的表；一条 `clipped_policy_loss`；一份 verifier 漏洞与隐藏测试清单。

**验收标准**：命令零退出；等 reward 的 zero group advantage 精确为零；非恒定组含负、零、正信号；sampled KL 允许单点为负；clipped loss 有限。能说明这些只验证 objective 语义，不证明 LLM 改善。

**失败诊断**：zero group 出现非零时查标准差阈值；符号反转时查 `new_logp-old_logp` 和 `policy_logp-reference_logp`；reward 上升而隐藏正确率下降时优先按 reward hacking 处理。

## 核心机制

### 先检查六个任务轴

1. **observation/state/POMDP quality**：当前 observation 是否足以近似 Markov state？若不够，先引入历史、belief 或 memory，并把状态估计误差纳入评测。
2. **discrete / continuous action**：有限离散动作可直接比较 action value；连续动作通常需要显式 actor、搜索或规划器。
3. **online interaction**：能否重置环境，交互的成本与安全风险多大？不能安全探索时，不应先选依赖大量在线试错的路线。
4. **fixed dataset**：日志是否覆盖候选策略会选择的动作？support 不足时，离线 value 外推比训练 loss 更值得先查。
5. **reward / verifier**：信号是否与真实目标一致，粒度是 step、token 还是 sequence，是否存在泄漏、长度或格式捷径？
6. **learned model 与 calibration**：若使用 learned model，它在候选 action 和长 horizon 上是否校准？open-loop prediction 好不等于 closed-loop return 好。

### 紧凑决策流

先问任务是否有跨步状态和延迟后果；没有时从 Bandit/监督学习基线开始。有 MDP 结构后按 action 类型分离离散与连续候选，再按 online interaction 是否可用决定 online、off-policy replay 或 Offline RL。只有在模型可用且经 action-conditioned 校准时才把 Model-based RL 加入候选。LLM 若对同一 prompt 可采多条 response 且 verifier 能稳定排序，可评估 GRPO；verifier 不可靠时，先修正信号而不是更换 optimizer。

### 算法选择表

| 算法 | 适用条件 | 数据 | 动作 | 主要风险 | 首个诊断 |
| --- | --- | --- | --- | --- | --- |
| Bandit | 无状态转移或 horizon 为一步，只需探索收益未知的臂 | online action-reward | 有限离散臂 | 非平稳与探索不足 | 每臂 count、均值与 regret |
| DQN | observation 近似 Markov，动作可枚举 | online interaction 加 replay | 有限离散 | value 过估计、replay 分布漂移 | TD target、terminal mask 与 Q 尺度 |
| PPO | 可持续采近似 on-policy rollout，需要稳定限制单次更新 | 新鲜 online rollout | 离散或连续 | policy lag、mask 或 advantage 错位 | ratio、clip fraction、KL 与版本号 |
| SAC | 连续控制且 online 数据昂贵，希望 replay 重用 | online interaction 加 replay | 连续 | logp 坐标、Q 偏差、temperature 失衡 | actor/critic/alpha 更新与 action range |
| Offline RL | 训练期不能交互且 fixed dataset 有一定 support | 固定 trajectory dataset | 离散或连续 | OOD action 与 value 外推 | action coverage、behavior gap 与 OPE 边界 |
| Model-based RL | known/learned dynamics 可用于规划或 imagined rollout | 真实 transition 加模型数据 | 离散或连续 | model bias、compounding error、model exploitation | action-conditioned calibration 与 closed-loop return |
| GRPO | 同 prompt 可采多个 response，reward / verifier 能做组内比较 | online grouped responses | 离散 token 序列 | zero group、长度偏置、reward hacking | 组内 reward 方差、group id、sampled KL |

这张表只筛选假设相容的候选，不给出跨任务优胜排序。最终选择仍需和监督学习、模仿学习、搜索、规则控制及“不训练”基线比较。

## 训练数据流

三个 capstone 共用一份证据链：

| 项目 | 原始数据 | 中间量 | 更新或求解 | 必须保留的证据 |
| --- | --- | --- | --- | --- |
| GridWorld | model 或 sampled transition | Bellman expectation、TD target、Q target | DP sweep 或 tabular update | terminal flag、target、seed、value/Q summary |
| Pendulum | online continuous transition | replay minibatch、twin-Q target、actor logp、alpha | critic/actor/temperature optimizer | action range、terminated/truncated、参数变化、完整与 partial return |
| verifier LLM | prompt-group response | raw reward、group advantage、old/reference logp | clipped policy objective | group id、mask、zero group、sampled KL、reward version |

报告结果时按“输入版本 -> target/advantage -> loss/update -> 独立评测”保存，而不是只贴最终 reward。GridWorld 的已知模型结果可作为 sampled 方法的语义对照；SAC smoke 只证明更新链连通；LLM helper 只证明符号、shape 和 zero-variance 边界。

## 常见失败

| 失败 | 错误结论 | 修正 |
| --- | --- | --- |
| observation 当作完整 state | 网络更大就能恢复缺失信息 | 检查 POMDP，加入 history/belief 并单独评测 |
| 连续 action 硬离散化却不查分辨率 | DQN 一定适合 | 比较动作维数、精度、搜索成本和 actor 路线 |
| 交互不安全仍直接 online 探索 | reward 会自动学会约束 | 先用 simulator、offline data、shield 与人工审批 |
| fixed dataset 不查 support | offline loss 下降代表部署更好 | 检查 OOD action、coverage 和 OPE 假设 |
| verifier 分数当真实目标 | reward 上升就是能力上升 | 隐藏测试、分量指标、人工抽检和反作弊样本 |
| learned model 只看 one-step error | open-loop 好就能控制好 | 做 counterfactual action 与 closed-loop return 评测 |
| smoke 当 convergence | 一次参数变化证明算法有效 | 长训练、多 seed、独立 evaluation 与区间报告 |
| 先定算法再改任务定义 | 调参可以补齐语义缺口 | 先冻结 state/action/data/reward/termination 契约 |

## 自检与练习

1. 为一个只有最终成功信号、不能在线试错的机器人日志任务，依次回答六个任务轴，并列出暂时不能证明的结论。
2. 说明 Pendulum 中 SAC 相对 PPO 的选择理由为何来自 action/data 条件，而不是算法名称的优胜关系。
3. 若 GridWorld 的 `terminated` 被当作 truncation，写出 TD/Q target 会多出的项。
4. 给 GRPO 设计三组 verifier reward：有区分度、zero group、被格式捷径攻击，并给每组首个诊断。
5. 为任一 capstone 写一页实验卡：代码/依赖版本、seed、输入、命令、原始输出、验收、失败边界和下一实验。
