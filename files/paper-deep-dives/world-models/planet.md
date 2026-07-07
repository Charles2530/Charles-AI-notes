---
paper_date: 2018-11
paper_title: Learning Latent Dynamics for Planning from Pixels
paper_topic: 世界模型
---

# PlaNet：像素规划的潜变量动力学

!!! info "论文信息"
    **论文题名：** Learning Latent Dynamics for Planning from Pixels。<br>
    **作者：** Danijar Hafner、Timothy Lillicrap、Ian Fischer、Ruben Villegas、David Ha、Honglak Lee、James Davidson。<br>
    **机构：** 未在公开元数据中稳定解析；以 arXiv/PDF 或官方页 affiliation block 为准。<br>
    **时间 / 主题：** 2018-11；世界模型。<br>
    **arXiv / 官方报告：** arXiv：[1811.04551](https://arxiv.org/abs/1811.04551)。<br>
    **GitHub / 项目：** GitHub：[github.com/danijar/planet](https://github.com/danijar/planet)。<br>
    **元数据来源与核验口径：** 来源：[arXiv](https://arxiv.org/abs/1811.04551)；[GitHub API / repo](https://github.com/danijar/planet)；Checked Date：2026-06-04；Repro Status：Paper / official materials reviewed, independent reproduction not claimed。

## 原论文故事线

### 研究背景

PlaNet 是 Dreamer 系列之前的关键世界模型论文。经典控制常常假设 dynamics 已知，planner 可以直接在真实动力学或 simulator 里搜索动作；但很多真实任务没有这个条件，例如从像素玩游戏、机器人在复杂环境里操作物体，agent 只看到图像、动作和奖励，不知道真实环境状态，也没有可用的动力学公式。PlaNet 问的就是：能不能先从交互数据学一个 world model，再在这个学出来的模型里 planning。

传统 model-based RL 如果直接在像素空间预测未来，计算很重，而且多步误差会迅速累积；model-free RL 可以避免模型误差，但真实交互样本效率很低。PlaNet 的切入点是保留 planning 的好处，同时把要学的对象从完整图像未来改成低维 latent dynamics。

PlaNet 的研究背景可以概括为一个折中：让世界模型学习一个 latent dynamics，而不是完整像素模拟器；规划时在 latent space 中搜索动作序列，而不是为每个候选动作生成未来图像。

### 问题（Challenge）

论文要解决的问题是：**能否从像素轨迹学习一个足够好的 latent dynamics model，使 agent 直接在 latent space 中规划并完成连续控制任务？** 挑战有四个。第一，像素观测是部分可观测的，单帧图像不能表示速度、接触和隐藏状态。第二，规划需要多步预测，一步重建好不代表多步 rollout 好。第三，planner 会搜索高回报动作，因此容易利用模型错误。第四，如果每个候选未来都解码成像素，CEM planning 的计算成本会很高。

这里还有一个典型的 train-test mismatch。训练时模型常被 posterior 和真实观测不断校正，错误只需要跨一步；规划时却要从当前 belief 出发，连续用 prior 想象未来十几步。一步误差在训练 loss 里可能不显眼，但在 CEM 的多步搜索里会被累计、放大，甚至变成 planner 主动利用的模型漏洞。

这个问题的价值在于，它把世界模型从“视频预测是否好看”转成“latent state 是否足以支持控制”。对于后续机器人和 agent 训练，这个转变很重要。

### Finding

作者的 finding 是：**对控制而言，模型不需要在规划时生成未来图像；只要 latent state 同时保留历史记忆和随机不确定性，并能预测 reward，就可以在 latent space 中完成有效规划。** 这个洞见把像素复杂性和动作搜索复杂性分开了。

这个 finding 解释了 RSSM 的设计。deterministic recurrent path 记住历史，stochastic latent path 表达当前观测和不确定性；observation model 训练 latent 不丢视觉信息，reward model 让 latent 对任务有用；CEM 只 rollout latent 和 reward，不再把每个候选动作序列解码成图像。

### 方法

PlaNet 的输入是真实环境收集的 image observation、action 和 reward 序列。模型使用 Recurrent State-Space Model：encoder 把图像编码到 latent，recurrent hidden state 根据上一 latent 和动作更新，posterior 在训练时结合当前观测，prior 用于没有未来观测的 rollout。训练目标包括 observation reconstruction、reward prediction 和 KL regularization；论文还提出 latent overshooting，用多步 latent consistency 改善长期预测。

控制时，PlaNet 没有单独 policy network。每一步先用当前观测更新 belief/state，再用 Cross-Entropy Method 在 latent space 采样并优化多条候选动作序列；模型预测每条序列的未来 latent 和 reward，选择回报最高序列的第一个动作执行，然后下一个环境步重新规划。

### 结论

实验在六个基于图像的连续控制任务上验证 PlaNet。论文报告 PlaNet 能在远少于 model-free 方法的 episode 数下达到相近或更好表现，并在若干任务上相比 D4PG 有明显样本效率优势。消融显示，RSSM 优于纯 deterministic 或纯 stochastic dynamics，说明历史记忆和随机 latent 的组合对像素控制有价值。

需要补充的是，PlaNet 的规划仍依赖在线 CEM，每一步执行成本较高；这也正是 Dreamer 后续把规划摊销进 actor 的原因。PlaNet 的贡献边界是：它证明了“像素 -> latent dynamics -> latent planning”可行，而不是解决所有真实环境中的模型误差和长期探索。

### 关键术语

- **PlaNet（Planning Network）**：从像素学习 latent dynamics，并用 CEM 在 latent space 规划动作的 model-based RL agent。
- **Recurrent State-Space Model, RSSM（循环状态空间模型）**：结合 deterministic memory 和 stochastic latent 的时序模型，处理部分可观测环境。
- **Latent Overshooting（潜变量超前预测）**：训练模型在 latent space 中对多步未来保持一致，缓解只做一步预测导致的 rollout 误差。
- **Cross-Entropy Method, CEM（交叉熵方法）**：通过反复采样、筛选高回报候选并重拟合分布来优化动作序列的规划算法。
- **Model Predictive Control, MPC（模型预测控制）**：每一步重新规划一段未来动作，但只执行第一个动作。
- **Reward Model（奖励模型）**：从 latent state 预测奖励，使规划可以不解码像素而直接评估动作序列。

这篇论文是 Dreamer 系列之前非常关键的一步。它提出的问题很直接：如果只有像素观测，能否先学习一个 latent dynamics model，再完全在 latent space 里做在线规划，而不是训练 model-free policy。

这件事的吸引力不只是“有模型”。model-based planning 通常更省数据，因为每条交互轨迹可以同时训练 observation、reward 和 transition，而不只是在已采样状态上更新策略或价值函数；它也不必完全依赖 Bellman backup 把远期奖励一层层往前传。任务很长、reward 很稀疏时，这个差异会变得很重要：model-free 方法需要反复交互才能把奖励信号传播到早期动作，而 PlaNet 试图让 learned dynamics 直接给 planner 一个可搜索的未来。

另一个好处是 test-time compute。学好 world model 后，不一定每次都重新训练 policy；决策时可以投入更多计算，采样和评估更多候选动作序列，尝试换来更好的当前动作。PlaNet 后面的限制也来自这里：在线搜索很灵活，但每一步都要付计算成本。

PlaNet 的答案是：

```text
image observation + action + reward trajectories
  -> Recurrent State-Space Model (RSSM)
  -> latent-space CEM planning
  -> execute first action
  -> receive next observation and replan
```

它和 [DreamerV3](dreamerv3.md) 属于同一条“交互轨迹驱动的 latent dynamics 世界模型路线”，但接口不同：PlaNet 每一步用 CEM 做 model-predictive control；Dreamer 则在 world model 的 imagined rollout 上训练 actor-critic，把规划成本摊到 policy 学习里。

## 论文位置

PlaNet 的历史位置可以这样理解：

| 阶段 | 代表 | 关键变化 |
| --- | --- | --- |
| 早期像素 model-based RL | PlaNet | 从像素学习 RSSM，并直接在 latent space 中规划 |
| policy learning from imagination | Dreamer | 不再每步 CEM 搜索，而是在 imagined latent trajectories 上训练 actor-critic |
| 鲁棒跨域 world model RL | DreamerV3 | 固定超参数、离散 latent、symlog/twohot/free bits 等稳定化技巧 |

所以 PlaNet 最值得放进世界模型专题的原因，不是它今天仍是最强控制算法，而是它把后续 Dreamer 路线的两个核心概念打牢了：**RSSM latent state** 和 **action-conditioned latent rollout**。

![Image-based control domains](../../assets/images/paper-deep-dives/world-models/planet/figure-1-domains.png){ width="920" }

<small>Figure source: `Learning Latent Dynamics for Planning from Pixels`, Figure 1. 原论文图注要点：该图展示六个像素控制任务，覆盖固定视角部分可观测、稀疏奖励、接触动力学、较大状态动作空间和平衡控制等挑战。</small>

**PlaNet 任务图要看像素控制覆盖哪些难点。**
**输入输出**：输入是像素观测、动作和 reward，输出是 RSSM belief/state 与 CEM 规划动作。
**效率机制**：规划时只在 latent space rollout reward，避免为每个候选动作生成像素未来。
**对主线意义**：它奠定了“从像素学 latent dynamics 再规划”的世界模型路线。
**不能证明什么**：PlaNet benchmark 不能证明所有真实硬件或长时稀疏任务同样稳定。

## 核心问题

传统 planning 在已知动力学里很强，但未知环境中必须先学 dynamics。困难在于像素输入下的 dynamics learning 很容易失败：

1. 单帧图像不是完整状态，任务本质是 POMDP；
2. 模型如果只会一步预测，规划时多步 rollout 会迅速漂移；
3. planner 会主动寻找模型漏洞，导致 model exploitation；
4. 如果在像素空间生成每个候选未来，计算成本太高。

PlaNet 的设计选择是把视觉复杂性和规划复杂性拆开。视觉重建用于训练 latent state，但规划时不生成图像，只在 latent state 上预测 reward 并评估 action sequence。

这使它的世界模型回答的是：

$$
h_t, s_t, a_t \rightarrow h_{t+1}, s_{t+1}, r_{t+1}
$$

而不是简单回答：

$$
o_t \rightarrow o_{t+1}
$$

这正是 PlaNet 和普通视频预测模型的根本区别。

## 总体训练循环

PlaNet 的 agent 没有 policy network，也没有 value network。它的行为完全来自在线 planning：

```text
用随机动作收集少量 seed episodes
  -> 从 replay data 训练 RSSM world model
  -> 当前观测通过 encoder 得到 latent belief
  -> CEM 在 latent space 搜索未来 action sequence
  -> 执行第一个动作并加入少量探索噪声
  -> 收集新 episode，继续更新 world model
```

每次环境 step 后都会重新规划，所以这是典型的 MPC 风格：规划一个 horizon，但只执行当前动作。新观测回来后，先用 encoder 修正当前 latent belief，再重新搜索动作序列。

这个循环里有两个关键约束：

1. world model 必须能在没有未来图像的情况下多步 rollout；
2. reward head 必须足够准确，因为 planner 只看 latent reward prediction，不看生成图像。

## RSSM World Model

PlaNet 使用 `Recurrent State-Space Model`。RSSM 把状态拆成两部分：

| State component | Role |
| --- | --- |
| deterministic state \(h_t\) | 通过 RNN 记住历史，承载长时记忆 |
| stochastic state \(s_t\) | 表达当前不确定性，支持多未来预测 |

也可以把 latent state 直观理解成两层：deterministic component 是一条确定性隐藏状态轨迹，负责把过去观测和动作压成记忆；stochastic component 是每一步的随机 latent variable，负责表达同一段历史下仍然不确定的当前状态和未来分支。PlaNet 论文强调二者都重要，不是为了架构复杂，而是因为像素控制同时需要记忆和不确定性。

论文对比了三种 latent dynamics 设计。

![Latent dynamics model designs](../../assets/images/paper-deep-dives/world-models/planet/figure-2-latent-dynamics.png){ width="860" }

<small>Figure source: `Learning Latent Dynamics for Planning from Pixels`, Figure 2. 原论文图注要点：该图比较 RNN、SSM 和 RSSM。RNN 只有确定性路径，SSM 只有随机状态，RSSM 同时保留确定性记忆和随机状态，以更稳健地预测多个未来。</small>

**这张结构对比图怎么读。**
PlaNet 的关键不是“用了一个 latent model”，而是选择了 RSSM 这种同时包含确定性状态和随机状态的结构。RNN 只有确定性 hidden state，适合记历史，但不擅长表达同一个过去下可能有多个未来；SSM 有随机状态，但缺少强 recurrent memory，长时部分可观测任务容易丢上下文。

RSSM 把两者拆开：\(h_t\) 负责累计历史和动作，\(s_t\) 负责当前不确定性。训练时 posterior \(q(s_t\mid h_t,o_t)\) 可以看真实观测修正 belief；规划时只能用 prior \(p(s_t\mid h_t)\) 往前 rollout。读这张图时要抓住这个分工：**记忆靠 deterministic path，不确定未来靠 stochastic latent**。

RSSM 的 generative model 可以写成：

$$
h_t = f(h_{t-1}, s_{t-1}, a_{t-1})
$$

$$
s_t \sim p(s_t \mid h_t)
$$

这行式子表示：在给定 deterministic state \(h_t\) 后，从 prior dynamics \(p(s_t\mid h_t)\) 里采样当前 stochastic state \(s_t\)。这里 \(h_t\) 负责历史记忆，\(s_t\) 负责当前不确定性。

$$
o_t \sim p(o_t \mid h_t, s_t), \quad
r_t \sim p(r_t \mid h_t, s_t)
$$

训练时，模型还需要一个 encoder / inference model：

$$
q(s_t \mid h_t, o_t)
$$

直觉上，\(q\) 是看到真实图像后的 posterior belief；\(p\) 是没有未来图像时的 prior dynamics。训练阶段可以用真实图像修正 \(s_t\)，但 planning 阶段只能从当前 belief 出发，用 prior 自己往前滚。

这就是为什么 RSSM 的结构很重要。纯 RNN 容易给 planner 一个确定但错误的未来；纯 SSM 难以记住长时历史；RSSM 用 \(h_t\) 负责记忆，用 \(s_t\) 负责不确定性，兼顾部分可观测和多未来。

## 训练目标

PlaNet 的基本训练目标是变分序列模型的 evidence lower bound。简化理解可以分成三项：

| Loss term | What it trains |
| --- | --- |
| observation reconstruction | latent state 是否包含足够视觉信息 |
| reward prediction | latent state 是否包含决策相关信息 |
| KL regularization | prior dynamics 是否能预测 posterior belief |

写成概念式：

$$
\mathcal{L}
=
-\log p(o_t \mid h_t, s_t)
-\log p(r_t \mid h_t, s_t)
+\mathrm{KL}\left(q(s_t \mid h_t,o_t)\,\|\,p(s_t \mid h_t)\right)
$$

这里的核心矛盾是：标准 KL 主要训练一步 prior。如果模型只在 \(t \rightarrow t+1\) 上准确，但 planning 需要连续 rollout 十几步，那么一小点误差会在 CEM 搜索里被放大。

### Latent Overshooting

为了解决多步预测问题，论文提出 `latent overshooting`。

![Latent overshooting](../../assets/images/paper-deep-dives/world-models/planet/figure-3-latent-overshooting.png){ width="860" }

<small>Figure source: `Learning Latent Dynamics for Planning from Pixels`, Figure 3. 原论文图注要点：该图比较 standard variational bound、observation overshooting 和 latent overshooting。latent overshooting 不解码所有多步预测图像，而是在 latent space 里约束多步 prior 接近对应 posterior。</small>

**为什么 overshooting 放在 latent space。**
标准 ELBO 主要约束一步 prior 接近下一步 posterior，但 PlaNet 的规划要连续想象很多步。一小步模型误差在 CEM 规划中会被放大，所以作者加入 overshooting，让从更早时刻 rollout 多步得到的 prior 也接近后面时刻的 posterior。

直接在 observation space 做 overshooting 要解码每个多步预测图像，成本高，而且像素误差会把模型拉向视觉细节。latent overshooting 只约束 latent belief 的多步一致性，更贴近规划真正需要的 dynamics accuracy。读这张图时可以把它理解成：不是让模型多步“画得像”，而是让模型多步“状态信念滚得准”。

标准目标只约束：

$$
p(s_t \mid s_{t-1}, a_{t-1}) \approx q(s_t \mid o_{\le t}, a_{<t})
$$

latent overshooting 进一步约束多步 prior：

$$
p(s_t \mid s_{t-d}, a_{t-d:t-1}) \approx q(s_t \mid o_{\le t}, a_{<t})
$$

这里的 \(d\) 是 overshooting distance。它的意义是让模型从更早的 latent state 出发，经过多步 transition 后仍能落到真实 posterior 附近。

这和 observation overshooting 的区别很关键。observation overshooting 会把多步预测都解码成图像再做 reconstruction loss，在像素任务中很贵。latent overshooting 只在 latent space 做 KL 约束，避免大量额外图像解码。

需要注意一个细节：论文实验中，RSSM 本身已经比较稳健，latent overshooting 对 RSSM 的主结果不是唯一决定因素；但它明显揭示了 model-based RL 的关键训练压力：**规划需要的是多步 latent dynamics，而不是只会一步重建的视觉模型**。

## Latent Planning

PlaNet 的 planning 使用 `Cross-Entropy Method`。流程如下：

```text
当前 belief: q(s_t | o_{<=t}, a_{<t})
  -> 初始化未来 action sequence 的 Gaussian distribution
  -> 采样 J 条候选 action sequences
  -> 用 RSSM prior rollout 每条候选序列
  -> 累加 reward head 预测的 mean rewards
  -> 选择 top K sequences
  -> 重新拟合 Gaussian
  -> 重复 I 次
  -> 执行当前步 mean action
```

CEM 可以理解成一种“采样、筛选、更新”的黑盒优化方法。它不需要对 action 求梯度，只需要 world model 能给一串动作打分。PlaNet 在模型内部尝试很多未来动作序列，保留预测回报最高的一小批 elite sequences，再用这些 elite sequences 重新拟合下一轮采样分布。多轮之后，planner 取当前步的 mean action 执行；真正进入环境的只有第一个动作，下一步拿到新观测后再重新规划。

论文使用的典型 planner 设置是：

| Hyperparameter | Value | Meaning |
| --- | --- | --- |
| \(H\) | 12 | planning horizon |
| \(I\) | 10 | CEM optimization iterations |
| \(J\) | 1000 | sampled action sequences |
| \(K\) | 100 | elite sequences used for refitting |

最重要的工程点是：candidate action sequences 的评估完全在 latent space 里完成。模型不需要为每条候选序列生成未来图像，只需要预测 latent state 和 reward。这就是 PlaNet 能在每个环境 step 在线规划的原因。

因此 PlaNet 没有直接训练一个 policy 来输出动作，而是把 policy 的角色交给在线 MPC：模型负责预测未来，CEM 负责搜索动作，环境只看到搜索结果的第一个动作。这个接口很干净，也暴露了代价：模型越不准，搜索越容易被误导；动作维度或 horizon 越大，CEM 需要的候选序列就越多。

它也解释了 PlaNet 和视频世界模型的差别。视频世界模型通常重视未来画面质量；PlaNet 只把像素重建当成训练信号，推理时真正服务 planning 的是 latent dynamics 和 reward head。

## 实验结论

PlaNet 在 DeepMind Control Suite 的六个像素控制任务上评估。论文重点不是最终分数绝对最高，而是样本效率：PlaNet 使用 1,000 episodes，从 pixels 训练；A3C 和 D4PG 对照结果使用 100,000 episodes。

![Model comparison](../../assets/images/paper-deep-dives/world-models/planet/figure-4-model-comparison.png){ width="920" }

<small>Figure source: `Learning Latent Dynamics for Planning from Pixels`, Figure 4. 原论文图注要点：该图比较 PlaNet、model-free baselines 和不同 dynamics model 设计；结果强调 RSSM 比纯 GRU 或纯 SSM 更适合从像素学习可规划的 dynamics。</small>

Table 1 reports comparison of PlaNet to model-free algorithms and true-simulator CEM planning.

| Method | Modality | Episodes | Cartpole Swing Up | Reacher Easy | Cheetah Run | Finger Spin | Cup Catch | Walker Walk |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A3C | proprioceptive | 100,000 | 558 | 285 | 214 | 129 | 105 | 311 |
| D4PG | pixels | 100,000 | 862 | 967 | 524 | 985 | 980 | 968 |
| PlaNet (ours) | pixels | 1,000 | 821 | 832 | 662 | 700 | 930 | 951 |
| CEM + true simulator | simulator state | 0 | 850 | 964 | 656 | 825 | 993 | 994 |
| Data efficiency gain PlaNet over D4PG (factor) |  |  | 250 | 40 | 500+ | 300 | 100 | 90 |

<small>表源：`Learning Latent Dynamics for Planning from Pixels`，Table 1。原论文表格要点：该表比较 PlaNet、model-free baselines 和真实 simulator CEM 在 DeepMind Control Suite 六个任务上的分数；重点是 PlaNet 用 `1,000` episodes 从 pixels 学习，就能接近或超过若干使用 `100,000` episodes 的 model-free 结果。</small>

这张表支持两个判断。第一，PlaNet 在 1,000 episodes 下接近或超过强 model-free baseline 的部分结果，尤其 Cheetah Run 上 PlaNet 高于 D4PG。第二，真实 simulator 上的 CEM 上界说明，planner 本身在这些任务中可行，关键瓶颈是 learned dynamics 是否足够准确。

这里的数据效率要按机制读，而不是只看最终分数。PlaNet 从每段 episode 中同时学视觉状态、transition 和 reward，planning 又能把当前模型用于主动收集后续数据；model-free baseline 则主要依赖环境回报通过价值或策略更新逐步传播。论文没有证明所有稀疏奖励任务都能被 learned model 解决，但它说明在这组六个像素控制任务上，latent dynamics + online planning 可以用少得多的环境交互达到有竞争力的表现。

![Agent designs](../../assets/images/paper-deep-dives/world-models/planet/figure-5-agent-designs.png){ width="920" }

<small>Figure source: `Learning Latent Dynamics for Planning from Pixels`, Figure 5. 原论文图注要点：该图比较 PlaNet、random collection 和 random shooting，说明在线数据收集与 CEM 迭代优化都对任务表现有贡献。</small>

Figure 5 的结论更偏系统设计：如果只用 random actions 收集数据，模型很难看到任务相关状态；如果只做 random shooting 而不用 CEM 迭代优化，动作序列搜索也不够强。PlaNet 的效果来自两个闭环同时成立：

1. planning 让数据收集更接近任务分布；
2. 新数据让 world model 更适合后续 planning。

## Open-Loop Prediction

论文还用 open-loop video prediction 检查 latent dynamics 是否真的能往前滚。

![Open-loop predictions](../../assets/images/paper-deep-dives/world-models/planet/figure-9-open-loop-predictions.png){ width="560" .atlas-figure-tall }

<small>Figure source: `Learning Latent Dynamics for Planning from Pixels`, Figure 9. 原论文图注要点：该图展示测试 episode 上的 open-loop video predictions，前 5 列是 reconstructed context frames，后续帧由模型 open-loop 生成，用于观察不同 dynamics model 的长期预测质量。</small>

这张图不应该被理解成“PlaNet 是视频生成模型”。PlaNet 的视频预测主要是诊断工具和训练约束。真正用于控制的是 latent state 和 reward prediction。换句话说，PlaNet 需要视觉预测足够好来塑造状态，但部署时不依赖生成未来视频给 planner 看。

## 和 DreamerV3 的关系

PlaNet 和 DreamerV3 的共同点是：

| 维度 | PlaNet | DreamerV3 |
| --- | --- | --- |
| 世界模型 | RSSM | RSSM 变体 |
| 数据 | environment interaction trajectories | environment interaction trajectories |
| 核心变量 | latent state, action, reward | latent representation, action, reward, continuation |
| rollout 空间 | latent space | latent space |
| 视觉重建 | 用于训练 latent state | 用于训练 latent representation |

它们的关键区别是 action selection：

| 维度 | PlaNet | DreamerV3 |
| --- | --- | --- |
| 控制方式 | 每个 step 用 CEM 在线规划 | 用 imagined rollout 训练 actor-critic |
| 推理成本 | 高，需要大量候选 action rollout | 低，actor 直接输出动作 |
| 策略网络 | 无 | 有 |
| 价值网络 | 无 | 有 |
| 主要瓶颈 | planning latency 和短 horizon 搜索 | model bias 与 actor-critic 稳定性 |

可以把 Dreamer 看成对 PlaNet 的一次关键工程升级：把在线规划变成离线 amortized policy learning。PlaNet 每一步都问“现在搜索哪条 action sequence 最好”；Dreamer 则训练一个 actor，让它学会在类似 latent state 下直接给出动作。

## 和视频世界模型路线的关系

PlaNet 和 LingBot-World 这类视频世界模型都涉及未来预测，但研究目标完全不同。

| 维度 | PlaNet | 视频生成型世界模型 |
| --- | --- | --- |
| 输入数据 | 交互轨迹：observation, action, reward | 大规模视频、文本、图像、动作或相机轨迹 |
| 训练目标 | 重建观测、预测 reward、约束 latent dynamics | 生成未来视频、动作条件续写、长时视觉一致性 |
| 推理输出 | 当前动作 | 未来视频或可交互视觉场景 |
| planning 接口 | CEM 在 latent reward 上搜索 | 通常需要额外 policy/planner 接入 |
| 强项 | 决策相关 dynamics 明确 | 视觉真实感、开放域场景、多样性 |

因此，PlaNet 是 `interaction-trajectory-based latent dynamics world model` 的早期代表。它不追求开放域视觉真实感，而是追求一个能让 planner 使用的低维世界状态。

## 局限与风险

PlaNet 的局限也很清楚。

1. **在线规划成本高**：每个环境 step 都要采样和评估大量 action sequences，难以直接扩展到高频真实机器人控制。
2. **planning horizon 有限**：CEM 在高维长 horizon action space 中搜索困难，太长的 horizon 反而会降低表现。
3. **依赖 reward signal**：PlaNet 需要 reward head 服务规划，不像纯视频模型那样可以只用互联网视频预训练。
4. **模型误差会被 planner 利用**：如果 learned dynamics 有系统性漏洞，CEM 会主动搜索这些漏洞。
5. **实验环境仍偏模拟控制**：DeepMind Control Suite 是重要基准，但还不是开放真实世界交互。
6. **train-test mismatch 没有完全消失**：latent overshooting 缓解多步 prior rollout 的训练压力，但不能保证部署时所有长 horizon 想象都可靠。

这些限制解释了为什么后续 Dreamer 系列会转向 actor-critic imagination：它保留 PlaNet 的 latent world model，但避免每一步都做昂贵的 CEM 搜索。

## 阅读结论

PlaNet 的核心贡献可以概括为三点：

1. 提出用 RSSM 从像素学习 action-conditioned latent dynamics；
2. 证明 planner 可以只在 latent space 里评估 reward，而不必生成未来图像；
3. 用 latent overshooting 强调 world model 训练必须关注多步 rollout，而不是只优化一步重建。

一句话总结：PlaNet 是“世界模型用于控制”的经典原型。它把世界模型从视频预测器推进成一个可供规划器调用的 latent simulator，也为后续 DreamerV3 这类 imagined rollout policy learning 铺好了路线。

PlaNet 的核心证据链是 RSSM posterior 用真实观测校正，prior 在 imagination 中 rollout，再用 CEM 在 latent space 规划动作。DMC 分数、latent overshooting 和 open-loop prediction 图要一起读：它证明 latent planning 有数据效率优势，也暴露在线 CEM 成本高，这正是 Dreamer 后来改成 actor 的原因。

## 相关阅读与下一步

- 外部材料：[论文标题 arXiv 检索](https://arxiv.org/search/?query=Learning+Latent+Dynamics+for+Planning+from+Pixels&searchtype=title)。
- 外部材料：[Semantic Scholar 标题检索](https://www.semanticscholar.org/search?q=Learning+Latent+Dynamics+for+Planning+from+Pixels)。
- 外部材料：[Papers with Code 检索](https://paperswithcode.com/search?q=Learning+Latent+Dynamics+for+Planning+from+Pixels)。
- 站内下一步：[论文精读专题](../index.md)。
- 站内下一步：[论文专题写作规范](../writing-standard.md)。
- 站内下一步：[外部精读来源台账](../../references/external-reading-ledger.md)。
