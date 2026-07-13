# verl 训练流程：从 rollout 到分布式更新

## 本章回答什么

`verl` 这一章负责回答系统问题：一个 prompt batch 如何经过分布式 rollout、reward、advantage、optimizer update 和 weight sync，最终成为下一版本的生成策略。它不重新推导 PPO 或 GRPO 公式，而是追踪公式依赖的字段由谁产生、在哪个 mask 上对齐，以及 rollout 与 update 是否仍属于同一个 policy version。

学完后应能画出 worker 拓扑和字段流，区分 rollout policy、proximal old policy、current train policy 与 reference policy，解释吞吐与样本新鲜度为何冲突，并用版本、mask、样本标识和服务元数据定位分布式训练故障。本文保留稳定概念；涉及具体字段和 worker ownership 的结论只对应文中锁定的 `e5687fce...` 代码快照，不外推为其他版本的 API。

## 先修知识

- [LLM 强化学习：RLHF、RLVR 与 GRPO](llm-rlhf-rlvr-and-grpo.md)：prompt、token action、verifier、group advantage 与 sampled KL。
- [Actor-Critic、GAE 与 PPO](policy-gradient-actor-critic-ppo-grpo.md)：old logprob、value、returns、clipped surrogate 和 on-policy 边界。
- [分布式训练与 Checkpoint](../training/distributed-training-and-checkpointing.md)：参数切分、通信、checkpoint 与可复现恢复。
- [推理运行时](../inference/serving-runtimes-vllm-sglang-and-tensorrt-llm.md)：prefill、decode、KV cache 与 serving batching。

## 最小任务

把一次最小系统任务写成：输入一个带 `prompt`、`input_ids`、`attention_mask` 和唯一 sample id 的 prompt batch，由带版本号的 actor 执行 distributed rollout，再由 reward/reference/critic 等角色补齐训练字段，完成一次 distributed update，并把新权重同步给下一轮 rollout。

```text
prompt batch@policy_version_id=17
  -> distributed rollout@rollout_version_id=17
  -> response + masks + reward + logp/value
  -> advantage/returns -> minibatch -> distributed update
  -> weight sync -> policy_version_id=18
```

最小验收不是 reward 上升，而是每条 response 能反查 prompt、rollout sampling behavior/version、`rollout_log_probs`、`old_log_probs` 的生成模式、reference version 和 optimizer step。bypass 模式会令 old 与 rollout 相同；decoupled 模式会另外计算 proximal old anchor。若只看最终 loss，字段或策略角色错位也可能产生有限数值。

## 核心机制

### SFT 与 RL 后训练的系统差异

SFT 的 target response 已在数据中，典型 batch 只需 forward、cross entropy、backward 和 optimizer step。RL 后训练必须先让当前 actor 生成 response，再由 reward model、规则或 verifier 评价，最后把评价变成策略更新信号。同一个 actor 因而在生成形态和训练形态之间切换：前者关心 decode batching 与 KV cache，后者关心梯度、optimizer state 和参数切分。

### 四种 policy 角色与字段

下表的字段名与分支语义锁定在 `e5687fce...`。这里最重要的修正是：采样行为、PPO proximal anchor、正在更新的参数和 KL anchor 是四种责任，不能只用“old/current/reference”三个模糊快照概括。

| 策略角色 | 字段 | 快照语义 | 用途 |
| --- | --- | --- | --- |
| rollout policy | `rollout_log_probs` | 实际生成 response 的 sampling behavior/version | 记录 behavior probability，并为 rollout correction 提供采样侧基准 |
| proximal old policy | `old_log_probs` | 每个 data batch 固定的 proximal snapshot | PPO ratio anchor；来源由 bypass 或 recompute 模式决定 |
| current train policy | `pi_theta`；actor loss 内计算 current log probs | 随当前 minibatch update 演进的参数 | PPO ratio 分子，执行 actor update |
| reference policy | `ref_log_prob` | 配置的 KL anchor，通常不随 actor minibatch 更新 | 约束策略偏移；不是 old policy，也不是 sampling behavior |

reference policy 是否启用与 old logprob 模式是正交问题。pinned 源码注释中的“两策略/三策略”只计算 rollout、old、current 这条 PPO/rollout-correction 链，不把可选 reference KL anchor 算入数量。

### Bypass 与 decoupled 两种 old logprob 模式

| 模式 | old_log_probs 来源 | 策略关系 | 修正边界 |
| --- | --- | --- | --- |
| bypass | `old_log_probs = rollout_log_probs` | `pi_rollout = pi_old`，只有 pi_rollout 与 pi_theta 两策略 | 不另算 proximal snapshot；仍需监控当前更新相对 rollout behavior 的 mismatch |
| decoupled / recompute | actor worker 重新计算 proximal old policy 的 `old_log_probs` | pi_rollout、pi_old、pi_theta 可以不同 | rollout correction / importance sampling 按配置处理 sampling-vs-old/current mismatch；不能修复任意 staleness |

这一分支直接见 pinned [`ray_trainer.py` 1533-1536 行](https://github.com/verl-project/verl/blob/e5687fce0516d31e1fdc4580499074a9bd94c751/verl/trainer/ppo/ray_trainer.py#L1533-L1536)：bypass 把 rollout logprob 作为 old anchor，decoupled 则为每个 data batch 重算一次稳定的 proximal old anchor。后续 correction 仅在实现与配置规定的范围内处理分布 mismatch；它不能把任意过期、错版本或字段错位的样本自动变成有效 on-policy 数据。

### Worker 角色与边界

下表锁定 `e5687fce...` 的所有权。worker role 可能 colocate 或由配置启用；表中 role 不承诺独立进程，但明确哪个执行点承担计算。这里的“训练阶段”只是概念流程阶段，不是独立 VERL process role。

| 角色 | 职责 | 产物或边界 |
| --- | --- | --- |
| RayPPOTrainer driver / controller | 通过 RPC 编排 worker calls，并执行 driver-side `compute_advantage` | 合并 batch、`advantages` / `returns`、调度与 metrics；不替 actor/critic 做 backward |
| actor / rollout worker | 执行 rollout、相关 logprob 计算和 `update_actor` | response、`rollout_log_probs` / `old_log_probs`；actor minibatch、backward、optimizer |
| critic worker（可选） | 计算 `values`，并用 `train_mini_batch` 拟合 returns | `values`；critic minibatch、backward、optimizer；无 critic 路线可省略 |
| reward role | 按配置调用规则、reward model 或 colocated/remote reward 路径 | token-level 或 sequence-level score 与 reward metadata |
| reference role | 启用 reference policy 时计算 `ref_log_prob`，placement 可配置 | KL anchor；参数不更新，也不充当 proximal old policy |

直接代码证据是：[`_update_actor` 与 `_update_critic` worker calls](https://github.com/verl-project/verl/blob/e5687fce0516d31e1fdc4580499074a9bd94c751/verl/trainer/ppo/ray_trainer.py#L1332-L1359) 分别调用 actor worker group 的 `update_actor` 和 critic worker group 的 `train_mini_batch`；[`fit()` ownership 注释](https://github.com/verl-project/verl/blob/e5687fce0516d31e1fdc4580499074a9bd94c751/verl/trainer/ppo/ray_trainer.py#L1368-L1373) 明确 driver 通过 RPC 构造 dataflow，轻量 advantage computation 在 driver process 完成。

### HybridFlow：算法图映射到 worker 图

![HybridFlow architecture 原论文图](../assets/images/paper-figures/reinforcement-learning/hybridflow-architecture.png){ width="920" }

<small>图源：[HybridFlow: A Flexible and Efficient RLHF Framework](https://arxiv.org/abs/2409.19256)，原论文图题为 Architecture of HybridFlow。这里仅用它说明算法 dataflow、resource mapping 与 worker communication 是不同层次；具体后端和布局以实际代码版本为准。</small>

HybridFlow 的系统贡献不是重写 PPO 公式，而是把 rollout、reward、reference、critic 和 actor update 等算法节点映射到物理资源。训练并行与推理并行的合适布局往往不同：actor/critic worker 要容纳梯度和 optimizer state，生成侧要容纳 KV cache 并减少逐 token decode 延迟。映射错误会表现为等待、重复搬运或某一服务长时间成为 straggler。

### 配置按“谁生产字段”来读

下面保留的是概念性配置读法，来源于本文原有 `verl.trainer.main_ppo` / Ray trainer 代码流分析。上游配置键会演进，使用时必须以锁定 commit 或安装版本的示例为准，不能把此片段当作当前 upstream API 保证。

```bash
python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  actor_rollout_ref.model.path=Qwen/Qwen3-8B \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=5 \
  actor_rollout_ref.actor.use_kl_loss=True \
  trainer.n_gpus_per_node=8
```

读配置时依次问：prompt 从哪里来，每个 prompt 采多少 response，rollout 使用哪个 sampling behavior，old logprob 走 bypass 还是 recompute，reward/reference 字段由谁补齐，driver 采用哪种 advantage estimator，以及 actor/critic worker 各自执行哪个 optimizer。这样即使其他版本字段变化，责任边界仍可核查。

### Prefix/KV reuse、吞吐与 freshness

同一 prompt 的多候选 rollout 可以做 prefix/KV reuse，减少重复 prefill；它不会消除各 response 的 decode、reward 和传输成本，也不能让不同 policy version 共享不兼容的 cache。cache key 至少应覆盖模型/adapter 版本、tokenized prefix 和影响生成的配置。

提高并发和更大的 rollout queue 通常改善 throughput，却可能拉大采样到更新的延迟，降低 freshness。weight sync 频繁可缩小 policy lag，但会增加通信与 actor 暂停时间；同步稀疏可提高设备利用率，却必须接受并度量 staleness。两者是显式取舍，不存在只调一个 batch size 就同时最优的结论。

![HybridFlow one RLHF iteration 原论文图](../assets/images/paper-figures/reinforcement-learning/hybridflow-one-iteration.png){ width="920" }

<small>图源同上，原论文图题为 3D-HybridEngine workflow in one RLHF iteration。图中训练布局、生成布局、权重聚合与重新切分说明 weight sync 本身就是一次迭代的系统成本；它不证明任意当前版本采用完全相同实现。</small>

## 训练数据流

字段应按阶段单调补齐，而不是用一个笼统的 trainer worker 猜测缺失语义：

| 阶段 | 输入字段 | 输出字段 | 首要边界 |
| --- | --- | --- | --- |
| prompt batch | `prompt`、`input_ids`、`attention_mask`、sample id | tokenized prompt batch | tokenizer/version 与 padding side 固定 |
| rollout | prompt batch、`policy_version_id`、sampling metadata | `response`、`response_mask`、`action_mask`、`rollout_version_id`、`rollout_log_probs` | 只把实际采样 token 当 action，并保留 sampling behavior/version |
| reward | response、有效 token mask、verifier/reward version | token-level reward 或 sequence-level `reward` 及分量 | 分数必须回到同一 sample/token |
| policy fields | response、rollout behavior、模式配置、reference/critic snapshot | `rollout_log_probs`、`old_log_probs`、`ref_log_prob`、PPO 可选 `values` | old 按 bypass/recompute 产生；reference 只作 KL anchor |
| advantage | reward、`values`、终止语义、有效 token mask | driver-side `advantage`、`returns` | driver 执行；PPO/GAE 与 group estimator 不能混用边界 |
| critic update | 完整 batch、`values`、`returns`、masks | critic minibatch -> backward -> optimizer、critic metrics | critic worker 所有；无 critic 路线跳过 |
| actor update | 完整 batch、`old_log_probs`、advantage、correction fields | actor minibatch -> current `pi_theta` loss -> backward -> optimizer | actor worker 所有；current policy 在 minibatch 间变化 |
| weight sync | 新训练权重、目标 rollout workers | 完成 `weight sync` 的 rollout version | 全量确认后才为新样本标新版本 |

LLM 的 `terminated` 常对应真实结束条件，例如 EOS 或任务完成；`truncated` 表示最大生成长度、调度边界等截断。两者都停止继续 decode，但只有算法语义上的 true termination 才能无条件去掉 critic bootstrap。系统应同时保留终止原因，而不是压成一个 `done`。

`attention_mask` 描述输入序列中非 padding token，`response_mask` 描述 response 的有效位置，`action_mask` 进一步限定哪些 token 真正进入 policy loss。prompt token、padding、工具返回或被裁切尾部是否属于 action，必须由数据协议明确。reward 若是 sequence-level，需要说明怎样广播或只放在末 token；若是 token-level，则长度必须和 logp/value 对齐。reward/logp 的 token 偏移一位仍可能得到有限 loss，因此要用 sample id、token id 和 mask 总数做逐层断言。

一次可审计运行还应保存 reproducibility metadata：代码 commit、容器和依赖版本、模型/tokenizer/reward 版本、随机 seed、数据切片、sampling 参数、并行拓扑、batch id、rollout behavior version、old-logprob mode、current actor step、reference version、终止原因与重试计数。seed 不能保证跨硬件或 serving kernel 位级一致，但这些元数据能区分配置变化、策略角色错配和 nondeterministic serving。

## 常见失败

| 失败 | 直接症状 | 首个检查 |
| --- | --- | --- |
| stale logp | PPO ratio 或 correction metrics 异常 | 分别核对 `rollout_log_probs` 的 behavior version、`old_log_probs` 模式和 current actor step |
| wrong mask | prompt/padding 进入 loss，长度越长偏差越大 | 比较 `attention_mask`、`response_mask`、`action_mask` 的 token 计数 |
| reward/logp misalignment | reward 正常但更新方向反常 | 在同一样本上打印 token id、reward 位置和 logp 位置 |
| reference/old confusion | KL 与 PPO ratio anchor 混用 | 核对 `ref_log_prob` 只服务 KL，`old_log_probs` 只服务 proximal ratio |
| duplicate samples | 某些 prompt 权重被无意放大 | 检查全局 sample id、重试次数与去重位置 |
| nondeterministic serving | 同 seed 重跑 response 或 logp 不一致 | 保存 backend/kernel/batching 元数据并先缩到单 worker |
| reward 很快上涨、验证不涨 | verifier 漏洞或长度/格式捷径 | 拆 reward 分量并跑隐藏集和人工抽检 |
| weight sync 部分完成 | worker 间混合 policy version | 同步使用 barrier/确认记录，禁止提前发新版本样本 |

调试顺序沿字段流前进：prompt 和 id，response 与 masks，reward 对齐，rollout/old/current/reference 四种角色，driver-side advantage，critic update，actor update，最后才是跨节点性能。这样能先排除数据协议错误，再分析 placement、communication 和设备利用率。

## 自检与练习

1. 分别说明 bypass 与 decoupled 模式中 `old_log_probs` 的来源，并解释为何只有 bypass 把它直接设为 `rollout_log_probs`。
2. 给一个 prompt 长 4、response 长 3、右侧 padding 2 的样本写出 `attention_mask`、`response_mask` 与 `action_mask`，说明各自进入哪个计算。
3. 若 rollout queue 从 1 扩到 8，列出 throughput、freshness、显存和 weight sync 等待时间应记录的指标。
4. 为 rollout behavior、proximal old、current actor 和 reference 分别设计版本记录，并说明 rollout correction 为什么不能修复任意 staleness。
5. 对 sequence-level verifier reward，说明怎样验证它与 group id、response 和 token logp 属于同一条样本。

继续精读时可对照 [verl GitHub](https://github.com/verl-project/verl)、[verl PPO Ray Trainer docs](https://verl.readthedocs.io/en/latest/workers/ray_trainer.html) 与 [HybridFlow 论文](https://arxiv.org/abs/2409.19256)。前两者用于核对实际安装版本，论文用于理解 dataflow/resource mapping 设计；若三者细节不同，应以锁定代码快照为运行事实并记录 commit。

原页保留的算法与实现背景也可继续使用：[Hugging Face 的 PPO/RLHF implementation details](https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo) 用于核查容易遗漏的工程语义，[Spinning Up 的 Policy Optimization](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html) 和 [PPO 论文](https://arxiv.org/abs/1707.06347) 用于回看 on-policy 目标，[Soft Actor-Critic 论文](https://arxiv.org/abs/1801.01290) 用作 off-policy 系统对照。这些来源解释算法责任，不能替代所用 `verl` commit 的运行事实。

## 下一步

进入 [综合实践与算法选择](practice-and-algorithm-selection.md)，用 GridWorld、Pendulum 和 verifier LLM 三个 capstone 把算法假设、可执行命令与验收证据连起来。需要回看目标定义时返回 [LLM 强化学习](llm-rlhf-rlvr-and-grpo.md)，需要重新选择完整路径时返回 [强化学习课程地图](index.md#完整学习路线)。
