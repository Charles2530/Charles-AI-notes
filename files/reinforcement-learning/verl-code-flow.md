# verl 训练流程：从 rollout 到分布式更新

## 本章回答什么

`verl` 这一章负责回答系统问题：一个 prompt batch 如何经过分布式 rollout、reward、advantage、optimizer update 和 weight sync，最终成为下一版本的生成策略。它不重新推导 PPO 或 GRPO 公式，而是追踪公式依赖的字段由谁产生、在哪个 mask 上对齐，以及 rollout 与 update 是否仍属于同一个 policy version。

学完后应能画出 worker 拓扑和字段流，区分 old policy 与 reference policy，解释吞吐与样本新鲜度为何冲突，并用版本、mask、样本标识和服务元数据定位分布式训练故障。本文保留稳定概念，不把上游易变的配置键或类名当作长期 API；涉及架构的描述以 [HybridFlow 论文](https://arxiv.org/abs/2409.19256) 和文末来源为边界。

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

最小验收不是 reward 上升，而是每条 response 能反查 prompt、采样版本、token mask、reward 版本和 optimizer step；`old_logp` 必须对应实际生成该 response 的版本。若只看最终 loss，字段错位也可能产生有限数值。

## 核心机制

### SFT 与 RL 后训练的系统差异

SFT 的 target response 已在数据中，典型 batch 只需 forward、cross entropy、backward 和 optimizer step。RL 后训练必须先让当前 actor 生成 response，再由 reward model、规则或 verifier 评价，最后把评价变成策略更新信号。同一个 actor 因而在生成形态和训练形态之间切换：前者关心 decode batching 与 KV cache，后者关心梯度、optimizer state 和参数切分。

### Old policy 不等于 reference policy

old policy 是采集当前 rollout 的策略快照，`old_logp` 是 PPO ratio 的分母；reference policy 通常保持不更新，用 `reference_logp` 定义相对底座策略的 KL 约束。两者不是同一个角色：即使训练开始时参数相同，old policy 会随 rollout 批次前移，而 reference policy 可以长期固定。

因此每批数据应记录 `policy_version_id` 与 `rollout_version_id`。若 trainer 已更新到版本 19，却把版本 17 的 response 当作版本 18 的 rollout，重新计算的 logp 不能替代采样时 `old_logp`；这属于 policy lag 或 staleness。系统必须明确允许的版本差、丢弃策略和重试语义，而不是让延迟样本静默混入近似 on-policy 更新。

### Worker 角色与边界

下表是概念角色，不承诺某个 `verl` 发布版必须以同名进程一一部署。角色可以 colocate，也可以跨资源池；实际 placement 与 communication 应以所使用版本的文档、配置和运行日志为准。

| 角色 | 职责 | 产物或边界 |
| --- | --- | --- |
| driver / controller | 调度阶段、资源 placement、失败重试与 communication | batch id、版本推进、worker 生命周期；不生成算法张量 |
| actor / rollout | 按指定 policy version 做 prefill/decode，训练阶段计算 actor loss | response、采样元数据、`old_logp`；生成与训练可能使用不同并行布局 |
| critic（可选） | PPO/GAE 路线估计 value 并拟合 returns | `value`；GRPO 等无 critic 路线可以省略 |
| reward | 调用 reward model、规则或 verifier，并保留分量 | token-level 或 sequence-level reward、reward version |
| reference | 计算 KL 约束所需 logp，参数不更新 | `reference_logp`；不能代替采样 old policy |
| trainer | 切 minibatch、执行 backward/optimizer、生成新权重版本 | optimizer step、训练指标、待同步权重 |

### HybridFlow：算法图映射到 worker 图

![HybridFlow architecture 原论文图](../assets/images/paper-figures/reinforcement-learning/hybridflow-architecture.png){ width="920" }

<small>图源：[HybridFlow: A Flexible and Efficient RLHF Framework](https://arxiv.org/abs/2409.19256)，原论文图题为 Architecture of HybridFlow。这里仅用它说明算法 dataflow、resource mapping 与 worker communication 是不同层次；具体后端和布局以实际代码版本为准。</small>

HybridFlow 的系统贡献不是重写 PPO 公式，而是把 rollout、reward、reference、critic 和 actor update 等算法节点映射到物理资源。训练并行与推理并行的合适布局往往不同：训练要容纳参数、梯度和 optimizer state，生成要容纳 KV cache 并减少逐 token decode 延迟。映射错误会表现为等待、重复搬运或某一服务长时间成为 straggler。

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

读配置时依次问：prompt 从哪里来，每个 prompt 采多少 response，rollout 使用哪个已同步版本，reward 和 reference 字段由谁补齐，advantage 属于 PPO/GAE 还是 group estimator，最后哪个 worker 执行 optimizer。这样即使字段名变化，数据责任仍可核查。

### Prefix/KV reuse、吞吐与 freshness

同一 prompt 的多候选 rollout 可以做 prefix/KV reuse，减少重复 prefill；它不会消除各 response 的 decode、reward 和传输成本，也不能让不同 policy version 共享不兼容的 cache。cache key 至少应覆盖模型/adapter 版本、tokenized prefix 和影响生成的配置。

提高并发和更大的 rollout queue 通常改善 throughput，却可能拉大采样到更新的延迟，降低 freshness。weight sync 频繁可缩小 policy lag，但会增加通信与 actor 暂停时间；同步稀疏可提高设备利用率，却必须接受并度量 staleness。两者是显式取舍，不存在只调一个 batch size 就同时最优的结论。

![HybridFlow one RLHF iteration 原论文图](../assets/images/paper-figures/reinforcement-learning/hybridflow-one-iteration.png){ width="920" }

<small>图源同上，原论文图题为 3D-HybridEngine workflow in one RLHF iteration。图中训练布局、生成布局、权重聚合与重新切分说明 weight sync 本身就是一次迭代的系统成本；它不证明任意当前版本采用完全相同实现。</small>

## 训练数据流

字段应按阶段单调补齐，而不是由 trainer 猜测缺失语义：

| 阶段 | 输入字段 | 输出字段 | 首要边界 |
| --- | --- | --- | --- |
| prompt batch | `prompt`、`input_ids`、`attention_mask`、sample id | tokenized prompt batch | tokenizer/version 与 padding side 固定 |
| rollout | prompt batch、`policy_version_id`、sampling metadata | `response`、`response_mask`、`action_mask`、`rollout_version_id`、采样时 logp | 只把实际采样 token 当 action |
| reward | response、有效 token mask、verifier/reward version | token-level reward 或 sequence-level `reward` 及分量 | 分数必须回到同一 sample/token |
| PPO estimates | response、采样版本、当前 reference/critic snapshot | `old_logp`、`reference_logp`、PPO 可选 `value` | old 来自 rollout；reference 用于 KL |
| advantage | reward、`value`、终止语义、有效 token mask | `advantage`、`returns` | PPO/GAE 与 group estimator 不能混用边界 |
| minibatch | 完整 batch、sample/group id、所有 masks | 打乱后的 `minibatch` | group 统计先完成，切分后仍可追踪原样本 |
| optimizer | minibatch、actor/critic loss、optimizer state | optimizer step、新 `policy_version_id` | 有限 loss 不代表字段对齐正确 |
| weight sync | 新训练权重、目标 rollout workers | 完成 `weight sync` 的 rollout version | 全量确认后才为新样本标新版本 |

LLM 的 `terminated` 常对应真实结束条件，例如 EOS 或任务完成；`truncated` 表示最大生成长度、调度边界等截断。两者都停止继续 decode，但只有算法语义上的 true termination 才能无条件去掉 critic bootstrap。系统应同时保留终止原因，而不是压成一个 `done`。

`attention_mask` 描述输入序列中非 padding token，`response_mask` 描述 response 的有效位置，`action_mask` 进一步限定哪些 token 真正进入 policy loss。prompt token、padding、工具返回或被裁切尾部是否属于 action，必须由数据协议明确。reward 若是 sequence-level，需要说明怎样广播或只放在末 token；若是 token-level，则长度必须和 logp/value 对齐。reward/logp 的 token 偏移一位仍可能得到有限 loss，因此要用 sample id、token id 和 mask 总数做逐层断言。

一次可审计运行还应保存 reproducibility metadata：代码 commit、容器和依赖版本、模型/tokenizer/reward 版本、随机 seed、数据切片、sampling 参数、并行拓扑、batch id、policy/rollout version、终止原因与重试计数。seed 不能保证跨硬件或 serving kernel 位级一致，但这些元数据能区分配置变化、版本错配和 nondeterministic serving。

## 常见失败

| 失败 | 直接症状 | 首个检查 |
| --- | --- | --- |
| stale logp | PPO ratio 漂移、clip fraction 异常 | 对齐 `old_logp` 的 `rollout_version_id` 与 sample id |
| wrong mask | prompt/padding 进入 loss，长度越长偏差越大 | 比较 `attention_mask`、`response_mask`、`action_mask` 的 token 计数 |
| reward/logp misalignment | reward 正常但更新方向反常 | 在同一样本上打印 token id、reward 位置和 logp 位置 |
| reference/old confusion | KL 与 ratio 同时异常或过度约束 | 核对两个 snapshot 的职责、版本和是否更新 |
| duplicate samples | 某些 prompt 权重被无意放大 | 检查全局 sample id、重试次数与去重位置 |
| nondeterministic serving | 同 seed 重跑 response 或 logp 不一致 | 保存 backend/kernel/batching 元数据并先缩到单 worker |
| reward 很快上涨、验证不涨 | verifier 漏洞或长度/格式捷径 | 拆 reward 分量并跑隐藏集和人工抽检 |
| weight sync 部分完成 | worker 间混合 policy version | 同步使用 barrier/确认记录，禁止提前发新版本样本 |

调试顺序沿字段流前进：prompt 和 id，response 与 masks，reward 对齐，old/reference/value 版本，advantage 分布，minibatch 完整性，optimizer，最后才是跨节点性能。这样能先排除数据协议错误，再分析 placement、communication 和设备利用率。

## 自检与练习

1. 解释为什么重新用当前 actor 计算 response logp，不能自动恢复采样时的 `old_logp`。
2. 给一个 prompt 长 4、response 长 3、右侧 padding 2 的样本写出 `attention_mask`、`response_mask` 与 `action_mask`，说明各自进入哪个计算。
3. 若 rollout queue 从 1 扩到 8，列出 throughput、freshness、显存和 weight sync 等待时间应记录的指标。
4. 设计 `policy_version_id` / `rollout_version_id` 检查，使部分 worker 同步失败时不会静默混入旧样本。
5. 对 sequence-level verifier reward，说明怎样验证它与 group id、response 和 token logp 属于同一条样本。

继续精读时可对照 [verl GitHub](https://github.com/verl-project/verl)、[verl PPO Ray Trainer docs](https://verl.readthedocs.io/en/latest/workers/ray_trainer.html) 与 [HybridFlow 论文](https://arxiv.org/abs/2409.19256)。前两者用于核对实际安装版本，论文用于理解 dataflow/resource mapping 设计；若三者细节不同，应以锁定代码快照为运行事实并记录 commit。

原页保留的算法与实现背景也可继续使用：[Hugging Face 的 PPO/RLHF implementation details](https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo) 用于核查容易遗漏的工程语义，[Spinning Up 的 Policy Optimization](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html) 和 [PPO 论文](https://arxiv.org/abs/1707.06347) 用于回看 on-policy 目标，[Soft Actor-Critic 论文](https://arxiv.org/abs/1801.01290) 用作 off-policy 系统对照。这些来源解释算法责任，不能替代所用 `verl` commit 的运行事实。

## 下一步

进入 [综合实践与算法选择](practice-and-algorithm-selection.md)，用 GridWorld、Pendulum 和 verifier LLM 三个 capstone 把算法假设、可执行命令与验收证据连起来。需要回看目标定义时返回 [LLM 强化学习](llm-rlhf-rlvr-and-grpo.md)，需要重新选择完整路径时返回 [强化学习课程地图](index.md#完整学习路线)。
