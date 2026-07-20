# LLM 强化学习：RLHF、RLVR 与 GRPO

LLM 强化学习把一段 response 看成由 token action 组成的 trajectory：模型从 prompt 开始逐 token 采样，reward model、规则或 verifier 在结尾打分，再把结果变成 policy update。RLHF、RLVR、PPO 和 GRPO 的差异主要在 reward 来源、advantage 构造与更新约束，不在“是否用了大模型”这个标签。

## 本章回答什么

本章回答：prompt、token、response 和策略分别对应 RL 的什么；RLHF 与 RLVR 的反馈从哪里来；PPO、GRPO 与 RLOO 怎样构造更新信号；group normalization 为什么会产生 cross-sample 依赖和 zero-variance 边界；DPO 为什么是 offline preference optimization 对照，而不是在线环境交互。

公式与可运行 NumPy 例子对应 [policy_objectives.py](../assets/examples/reinforcement-learning/policy_objectives.py)。例子只验证 objective 语义，不声称复现大模型训练或收敛结果。

## 先修知识

- [Actor-Critic、GAE 与 PPO](policy-gradient-actor-critic-ppo-grpo.md)：old logprob、advantage、clipped surrogate 与 reference policy。
- [Model-based RL 与世界模型](rl-for-world-models.md)：区分真实 rollout、模型生成 rollout 与 closed-loop 证据。
- [Agent、环境与训练闭环](rl-agent-environment-loop.md)：trajectory、reward 和 termination 的通用语义。
- [verl 训练流程](verl-code-flow.md)：大规模 rollout、reward 与 actor update 的系统落地。

## 最小任务

运行：


```bash
python files/assets/examples/reinforcement-learning/policy_objectives.py --seed 7
```

程序打印两个 prompt、每个 prompt 三个 candidate 的 raw reward、group advantage、pointwise KL penalty 和一个 finite clipped policy loss。第一个 prompt 的三个 reward 相同，因此 zero-variance group 得到精确零 advantage；第二组才产生正负更新信号。

LLM trajectory 的最小映射如下：

| RL 对象 | LLM 后训练对应物 |
| --- | --- |
| initial observation | prompt、system instruction、工具状态 |
| prefix state | prompt 加已生成 token (y_{<t}) |
| token action | 当前生成 token (y_t) |
| response trajectory | 完整 token 序列 (y_{1:T}) |
| policy | $\pi_\theta(y_t\mid x,y_{<t})$ |
| reward | 人类偏好模型、规则、代码执行器或 verifier 分数 |
| old policy | 生成本批 rollout 时冻结的策略快照 |
| reference policy | 用于 KL 约束的底座或对齐前策略 |

## 核心机制

统一记号是：prompt 初始化 prefix state，模型每步选择 token action，直到形成 response trajectory；reward model 或 verifier 评分，old policy 提供采样比率，reference policy 提供偏移约束。RLHF、RLVR、PPO、GRPO、RLOO 与 DPO 的差异都可以沿这组对象定位，GRPO 还必须显式处理 zero-variance group。

### RLHF、RLVR 与反馈来源

RLHF（Reinforcement Learning from Human Feedback）通常先收集人类偏好比较，训练 reward model，再由 PPO 等方法优化 policy。reward model 把开放式人类判断压成可计算标量，但会继承标注分布、模型泛化和对抗性漏洞。

RLVR（Reinforcement Learning with Verifiable Rewards）在答案可自动检查时使用 verifier，例如数学最终答案、代码单元测试或格式约束。它降低逐条人类标注成本，却不会自动解决错误 verifier、答案泄漏、只看最终结果而忽略推理质量等问题。

两者都需要区分：reward 定义了训练代理目标，真实任务价值仍要由独立评测证明。reward hacking 就是 policy 找到高分但不符合真实意图的 response。

### PPO：旧策略上的 clipped surrogate

对已采样 token 或 response，概率比率为：


\[
r_t(\theta)=\exp\left(\log\pi_\theta(a_t\mid s_t)-
\log\pi_{\text{old}}(a_t\mid s_t)\right).
\]

其中分子策略是当前 policy，分母策略是采样这批 response 的 old policy。代码域中的同一方向是 `new_logp - old_logp`：先取当前策略减旧策略，再 exponentiate，不能反向。

PPO clipped objective 为：


\[
L^{\text{clip}}(\theta)=
\mathbb E\left[\min\left(r_t A_t,
\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t\right)\right].
\]

这里的 $A_t$ 是 advantage，$\epsilon$ 是 ratio 裁剪宽度。代码中的 `clipped_policy_loss` 返回这个 surrogate 的负均值，作为 loss 交给 optimizer 最小化；$L^{\text{clip}}$ 写的是要最大化的正 surrogate，不是 helper 直接返回的正值。正 advantage 时过大的增益被上界裁剪；负 advantage 时过大的概率下降被另一侧限制。old policy 负责定义采样时分布和 ratio，reference policy 负责限制离底座太远，两者不能混为同一个快照。

### GRPO：同一 prompt 内做相对归一化

GRPO 对同一 prompt 采样一组 $G$ 个 response，以组内 reward 构造 advantage。可运行 helper 把 batch 的 last axis 当作同 prompt group：每组 reward 先减去组内均值，再按总体标准差（`ddof=0`）归一化：


\[
A_i=\begin{cases}
\dfrac{R_i-\bar R}{\sigma_R+\epsilon}, & \sigma_R>\epsilon,\\
0, & \sigma_R\le\epsilon.
\end{cases}
\]

这里的 $\epsilon$ 让非零组的标准差略小于 1；它是数值稳定项，不应被解释成新 reward。zero-variance group 的 reward 全相同，没有相对排序信息，`group_advantages` 返回精确零，因此这个例子产生零 policy update 信号而不是除零或虚构方向。

GRPO 没有消除 baseline，而是把 baseline 变成同 prompt 的 group mean。于是一个 sample 的 advantage 依赖同组其他 sample，形成 cross-sample dependency。改变 group size、采样温度、去重或 verifier 分布都会改变 advantage；不能把每条 response 当成完全独立标签。

### RLOO 的边界

RLOO（REINFORCE Leave-One-Out）同样用同 prompt 多个 response 降低方差，但第 $i$ 个样本的 baseline 是其余 $G-1$ 个 reward 的均值，而不是包含自身的组均值/标准差归一化。它提供一个不依赖 critic 的 REINFORCE-style 对照，但 group size 小、reward 高噪声时方差仍可能很大。不要把所有“同 prompt 多采样”方法都统称 GRPO。

### Pointwise log-ratio 不等于 full KL

在从 policy 采样的 token 上，可计算：


\[
k_t=\log\pi_\theta(a_t\mid s_t)-
\log\pi_{\text{ref}}(a_t\mid s_t).
\]

`sampled_forward_kl` 精确返回这个 pointwise sampled log-ratio estimator。单个 $k_t$ 可以为负；只有在规定的采样分布下取期望，才与 expected KL 联系起来。它没有枚举完整词表，也不是 full KL。训练日志应写清是 token sum、response mean、batch mean，还是完整分布上的 KL。

### DPO 是离线偏好优化对照

DPO 使用固定的 chosen/rejected preference pair 和 reference policy，直接优化成对分类形式的目标。它是 offline preference optimization：训练步骤不向在线环境采新 response，也不在每轮通过 reward model/verifier 闭环打分。DPO 可避免显式训练 reward model 与在线 PPO rollout，但仍依赖偏好数据覆盖，部署后的新状态不会自动进入训练集。

所以“DPO 不属于 RL”或“DPO 就是在线 RLHF”都过于粗糙。更准确的课程边界是：它来自 KL-regularized reward optimization 的推导，但实际训练 dataflow 是固定偏好数据上的离线优化，应与在线 rollout -> reward -> update 主线分开。

## 训练数据流


```text
prompt batch
  -> old policy rollout: G responses per prompt
  -> reward model / verifier scores each response
  -> group advantage, RLOO baseline, or critic advantage
  -> current and reference policy logprobs
  -> clipped policy objective + optional KL control
  -> actor update
  -> independent evaluation and failure prompts
```

系统成本常由 rollout 而非反向传播单独决定。每个 prompt 采 $G$ 个长 response 会放大 decode token 数、KV cache、verifier 调用和跨 worker 传输。prefix reuse 可让同组 candidate 共享 prompt prefill/KV，但 response decode 仍各自增长；复用策略也必须保证 old logprob 与实际采样路径一致。

| 数据字段 | 为什么必须保留 |
| --- | --- |
| prompt/group id | 正确做 group normalization 与 cross-sample 审计 |
| response token + mask | 对齐有效 token，避免 padding 进入 loss |
| old logprob | 构造 PPO ratio，确认 rollout 与 update 版本 |
| reference logprob | 计算逐点 KL penalty 或 KL loss |
| raw reward 与各分量 | 发现 verifier 冲突和 reward hacking |
| advantage | 检查 zero-variance group 与异常尺度 |
| response length/termination | 防止长度成为奖励捷径 |

RLHF 还需要 reward model 的训练/验证切分与人工审计；RLVR 需要 verifier 单测、隐藏测试和防泄漏数据。两者都需要保留 reward 版本，避免旧 rollout 被新定义静默重算。

## 常见失败

1. **把 sampled KL 每点都要求非负。** pointwise log-ratio 可以为负；非负性属于同分布上的 expected KL。
2. **zero-variance group 强行加随机噪声。** 这会制造没有 reward 证据的更新方向；应先检查 verifier 是否分辨不了候选或采样多样性不足。
3. **只看平均 reward。** 格式、长度、拒答、正确率和安全分量可能互相抵消，应单独记录。
4. **把 reference policy 当 old policy。** 一个控制偏移，一个定义采样 ratio；版本错配会使 objective 语义失真。
5. **DPO 当作在线闭环。** 固定 preference pair 不会自动暴露当前 policy 的新失败。
6. **verifier 通过等于任务正确。** 公开测试可被模式匹配或泄漏利用，需隐藏测试、人工抽检和 distribution shift 评测。
7. **忽略 rollout cost。** group size 和 response length 提高统计信号时，也线性或更高地增加生成、KV 与 reward 服务成本。

## 自检与练习

1. 手算 rewards `[1, 1, 1]` 与 `[0, 1, 2]` 的 group advantage，并解释第一组为什么零更新。
2. 构造一个负的 pointwise KL sample，解释它为什么不违反 expected KL 非负性。
3. 分别写出 old policy 和 reference policy 在一次 PPO/GRPO 更新中的职责。
4. 比较 GRPO group mean、RLOO leave-one-out baseline 与 PPO critic advantage 的信息来源。
5. 设计一个会被格式 verifier 奖励、却不解决原任务的 reward hacking response，并给出隐藏检查。

权威起点：

- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)：PPO clipped surrogate。
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)：监督微调、reward model 与 PPO 的 RLHF pipeline。
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)：固定 preference pair 的离线偏好目标。
- [DeepSeekMath](https://arxiv.org/abs/2402.03300)：GRPO 的方法来源与数学推理训练设置。
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948)：可验证 reward 与 reasoning-oriented RL 的报告边界。
- [Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback](https://arxiv.org/abs/2402.14740)：RLOO 等 REINFORCE-style 方法。
