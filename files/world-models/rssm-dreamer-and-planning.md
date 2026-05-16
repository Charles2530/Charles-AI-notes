# RSSM、Dreamer 与规划

这页讲世界模型最经典的一条路线：从真实交互轨迹里学习一个 latent dynamics，然后在这个模型里“想象未来”，用 imagined rollout 做规划或训练策略。

如果总览页回答“世界模型是什么”，本页回答三个更具体的问题：

1. 为什么要把观测压成 latent state，而不是直接在像素上规划。
2. 为什么 RSSM 要同时有确定性记忆 \(h_t\) 和随机状态 \(z_t\)。
3. Dreamer 如何把 world model 变成 actor-critic 的训练器。

!!! tip "先给结论"
    RSSM/Dreamer 的核心证据不是视频更清晰，而是 latent imagination 能用更少真实环境交互训练出更好的控制策略。它是理解 WAM、VAM、视频世界模型和 action-conditioned planning 的基线。

!!! note "符号承接"
    如果 \(x_t/o_t\)、\(z_t\)、\(h_t\)、\(a_t\)、\(r_t\)、\(c_t/d_t\)、\(p_\theta\)、\(q_\phi\)、\(\pi_\psi\)、\(V_\eta\) 还不熟，先读 [从零理解世界模型](beginner-world-models-primer.md)。

## 1. 为什么从像素转向 latent

真实观测 \(x_t\) 往往很大：图像、视频、多相机、深度、机器人本体状态。直接在像素空间做规划有三个问题：

| 问题 | 为什么麻烦 |
| --- | --- |
| 维度太高 | planner 每评估一个候选动作都要处理高维图像未来 |
| 细节太杂 | 背景纹理、光照、阴影会占用建模容量 |
| 决策信号稀疏 | 规划真正关心的是接触、风险、奖励、终止和目标进展 |

latent dynamics 的想法很朴素：不要把世界逐像素记住，只保留会改变动作选择的信息。一个基础形式是：

$$
p_\theta(z_{t+1}\mid z_t,a_t), \qquad
p_\theta(x_t\mid z_t), \qquad
p_\theta(r_t\mid z_t)
$$

其中 \(z_t\) 是模型学到的潜状态。它不必和真实物理状态逐维对应，但必须足够支持未来预测、奖励估计和动作选择。

## 2. RSSM 的图先看一遍

![PlaNet latent dynamics model designs](../assets/images/paper-deep-dives/world-models/planet/figure-2-latent-dynamics.png){ width="860" }

<small>图源：[Learning Latent Dynamics for Planning from Pixels](https://arxiv.org/abs/1811.04551)，Figure 2。原论文图意：比较 RNN、SSM 和 RSSM 三种 dynamics 设计；RSSM 同时保留 deterministic hidden state 和 stochastic latent state，用于从像素学习可规划的潜空间动力学。</small>

!!! note "图解卡：RSSM 结构对比"
    **输入输出**：从历史观测和动作得到当前 belief，再预测下一 latent、观测和 reward。<br>
    **关键模块**：RNN 只有确定性记忆，SSM 有随机状态但长时记忆弱，RSSM 把 \(h_t\) 与 \(z_t\) 合起来。<br>
    **公式对应**：\(h_t=f_\theta(h_{t-1},z_{t-1},a_{t-1})\)，\(p_\theta(z_t\mid h_t)\)，\(q_\phi(z_t\mid h_t,x_t)\)。<br>
    **容易误读**：RSSM 不是把 VAE 和 RNN 简单拼起来，而是在部分可观测环境中维护可 rollout 的 belief state。

## 3. RSSM 公式拆成五张卡

### 卡 1：确定性记忆 \(h_t\)

$$
h_t = f_\theta(h_{t-1}, z_{t-1}, a_{t-1})
$$

| 读法 | 解释 |
| --- | --- |
| 输入 | 上一步记忆 \(h_{t-1}\)、上一步随机状态 \(z_{t-1}\)、上一步动作 \(a_{t-1}\) |
| 输出 | 当前确定性记忆 \(h_t\) |
| 直觉 | 把过去几帧、过去动作和遮挡前的信息压进一段记忆 |
| 例子 | 雾天开车时，单帧看不清目标，但过去几帧运动趋势能帮助判断前方是不是车 |

\(h_t\) 是 deterministic 的意思不是“世界确定”，而是这部分状态由网络递推得到，不在这一步重新采样。它负责记历史。

### 卡 2：随机状态 \(z_t\)

$$
z_t \sim p_\theta(z_t\mid h_t)
$$

| 读法 | 解释 |
| --- | --- |
| 输入 | 当前记忆 \(h_t\) |
| 输出 | 当前可能 latent 的分布 |
| 直觉 | 同一段历史下，未来可能有多种解释或分支 |
| 例子 | 遮挡后可能有人，也可能没有人；杯子可能已经被夹稳，也可能快要滑落 |

\(z_t\) 用来表达不确定性和多模态未来。世界模型如果只输出一个平均未来，常会把风险抹平。

### 卡 3：prior 和 posterior

训练时，模型既能看到真实观测，也要学会在没有未来观测时自己往前滚。

先验链：

$$
p_\theta(z_t\mid h_t)
$$

后验链：

$$
q_\phi(z_t\mid h_t,x_t)
$$

| 名称 | 什么时候用 | 看到了什么 | 作用 |
| --- | --- | --- | --- |
| prior \(p_\theta\) | planning / imagined rollout | 只看历史和动作 | 自己预测下一 latent |
| posterior \(q_\phi\) | 训练 / 表征更新 | 看历史、动作和真实观测 \(x_t\) | 用真实观测校正 belief |

直觉上，人开车时也会先根据上一秒和自己的动作预测前方，再看到新画面后修正判断。世界模型训练要让 prior 和 posterior 足够接近，否则一旦进入 imagined rollout，模型就会因为看不到真实未来图像而失效。

### 卡 4：序列 ELBO / world model loss

一个常见 world model 目标可以简化写成：

$$
\mathcal{L}_{\text{wm}} =
\sum_{t=1}^{T}
\mathbb{E}_{q_\phi(z_t \mid h_t, x_t)}
\left[
-\log p_\theta(x_t \mid h_t, z_t)
-\log p_\theta(r_t \mid h_t, z_t)
-\log p_\theta(c_t \mid h_t, z_t)
\right]
+
\beta \sum_{t=1}^{T}
D_{\text{KL}}\left(
q_\phi(z_t \mid h_t, x_t)
\;\|\;
p_\theta(z_t \mid h_t)
\right)
$$

| 项 | 在要求什么 | 为什么和决策有关 |
| --- | --- | --- |
| 观测重建 | latent 解释当前观测 | 防止状态丢掉基本场景信息 |
| reward 预测 | latent 读出任务进展 | actor/planner 要知道哪条未来更好 |
| continue / done | latent 读出是否终止 | 撞墙或失败后不能继续假装优化 |
| KL | posterior 不要离 prior 太远 | imagined rollout 只能依赖 prior |

这里的重点不是把公式背下来，而是理解四个责任：看懂观测、读出奖励、知道何时终止、让模型能在没有真实未来观测时自己 rollout。

### 卡 5：imagined return

Dreamer 训练 actor 时，不是每次都去真实环境试错，而是在 learned world model 中展开未来：

$$
a_t \sim \pi_\psi(a_t\mid h_t,z_t)
$$

$$
h_{t+1}, z_{t+1} \sim p_\theta(h_{t+1},z_{t+1}\mid h_t,z_t,a_t)
$$

$$
J(\psi)=
\mathbb{E}_{\pi_\psi,p_\theta}
\left[
\sum_{k=0}^{H-1}\gamma^k r_{t+k}
\right]
$$

| 符号 | 含义 |
| --- | --- |
| \(\pi_\psi\) | actor / policy，根据 latent 输出动作 |
| \(p_\theta\) | world model，用动作推进 latent |
| \(H\) | imagined rollout 的未来步数 |
| \(\gamma\) | 折扣，控制远期奖励权重 |
| \(J(\psi)\) | actor 希望最大化的想象回报 |

这就是 Dreamer 的核心：把真实环境 step 换成 GPU 上的 latent imagination。但它也带来风险：如果 world model 错了，actor 可能学会利用模型漏洞，在 imagined future 里高分，真实环境却失败。

## 4. Dreamer 的训练闭环

![Components of Dreamer](../assets/images/paper-deep-dives/world-models/dreamer/figure-3-components.png){ width="890" }

<small>图源：[Dream to Control](https://arxiv.org/abs/1912.01603)，Figure 3。原论文图意：Dreamer 先从经验数据中把观测和动作编码到 compact latent states 并预测 reward；再在 latent space 中想象 trajectories，训练 action 和 value；最后用真实 episode history 编码当前状态并执行动作。</small>

!!! note "图解卡：Dreamer 的三步"
    **输入输出**：真实 episode 进入 replay，world model 输出 latent/reward/continue，actor/value 在 imagined trajectory 上更新。<br>
    **关键模块**：左侧学习 dynamics，中间学习 behavior，右侧回到环境执行并继续收集数据。<br>
    **公式对应**：world model loss 对应上面的 ELBO 卡，actor update 对应 imagined return 卡。<br>
    **容易误读**：Dreamer 不是单纯“训练一个生成模型再用 RL”，而是让 world model 和 actor/value 交替改进。

![Dreamer algorithm](../assets/images/paper-deep-dives/world-models/dreamer/algorithm-1-dreamer.png){ width="820" }

<small>图源：[Dream to Control: Learning Behaviors by Latent Imagination](https://arxiv.org/abs/1912.01603)，Algorithm 1。原论文图意：Dreamer 交替执行真实环境交互、world model 更新、imagined rollout、actor/value 更新和真实动作执行。</small>

!!! note "图解卡：算法框里的两条数据流"
    **输入输出**：真实经验流更新 replay buffer 和 world model；想象经验流从 replay state 出发展开 rollout 并更新 actor/value。<br>
    **关键模块**：`representation / transition / reward / continue` 属于世界模型，`actor / value` 属于行为学习。<br>
    **公式对应**：真实观测训练 \(q_\phi\) 和 decoder，imagined rollout 主要依赖 \(p_\theta\)。<br>
    **容易误读**：算法图不是“先把模型训完再训策略”的单向流程，而是持续交替。

## 5. DreamerV3 图：world model 和 actor-critic 分开看

![DreamerV3 world model learning](../assets/images/paper-deep-dives/world-models/dreamerv3/x2.png){ width="900" }

![DreamerV3 actor critic learning](../assets/images/paper-deep-dives/world-models/dreamerv3/x3.png){ width="900" }

<small>图源：[DreamerV3: Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104)，Figure 3(a)/(b)。原图意：Figure 3(a) 展示 world model learning，Figure 3(b) 展示 actor-critic 在 imagined latent trajectories 上学习。</small>

!!! note "图解卡：DreamerV3 为什么拆成两张图"
    **输入输出**：第一张图从 replay 经验学习 representation、dynamics、reward 和 continuation；第二张图用这些 latent rollout 训练 actor 与 critic。<br>
    **关键模块**：world model 负责“世界怎么变”，actor-critic 负责“怎么行动更好”。<br>
    **公式对应**：第一张图对应 \(\mathcal{L}_{wm}\)，第二张图对应 imagined return、value target 和 policy objective。<br>
    **容易误读**：DreamerV3 的结果不能直接证明高清视频模拟器可规划；它证明的是 learned latent dynamics 可以支撑行为学习。

## 6. Planning 和 policy learning 的区别

世界模型有两种主要用法。

| 用法 | 形式 | 优点 | 代价 |
| --- | --- | --- | --- |
| 学策略 | 在 imagined rollout 上优化 actor \(\pi_\psi\) | 执行快，适合实时控制 | actor 可能利用模型漏洞 |
| 在线规划 | 搜索动作序列 \(a_{t:t+H-1}\) | 新场景更灵活，可显式比较候选 | 在线优化慢，候选数和 horizon 会放大成本 |

在线规划常写成：

$$
a_{t:t+H-1}^\star =
\arg\max_{a_{t:t+H-1}}
\mathbb{E}\left[
\sum_{k=0}^{H-1}\gamma^k r_{t+k}
\right]
$$

它像 `MPC`：每次只执行前一小段动作，再用新观测刷新状态并重新规划。机器人、自动驾驶和目标图像规划里经常用这种形式。

## 7. 为什么 continue / done 头很重要

很多初学者只盯着观测重建和 reward，但 \(c_t/d_t\) 对规划很关键。如果模型不知道 imagined trajectory 已经撞墙、掉落或任务结束，planner 会继续在一个不该存在的未来里优化。

| 头 | 预测什么 | 坏了会怎样 |
| --- | --- | --- |
| reward head | 当前或未来进展 | 策略朝错误目标优化 |
| continue / done head | 轨迹是否还能继续 | 失败后仍然累计虚假收益 |
| risk head | 碰撞、near-miss、不确定性 | planner 低估危险动作 |

真实系统常会在 Dreamer 式结构上再加 risk、success、contact、visibility、reachability 等头，让 latent 不只服务 reward，也服务安全和恢复。

## 8. 最怕的失败模式

| 失败模式 | 表现 | 先查什么 |
| --- | --- | --- |
| 视觉上像，动力学上错 | 视频合理但接触、摩擦、碰撞不对 | action sensitivity、object state drift |
| 长时误差累积 | H 越长，latent 越偏离真实 | horizon ablation、open-loop vs closed-loop |
| reward 预测不准 | 重建好但任务判断错 | reward head calibration、failure replay |
| 分布外动作 | planner 搜到训练数据没见过的高速/怪异动作 | action prior、behavior constraint、uncertainty |
| model exploitation | actor 在模型里高分，真实失败 | real rollout audit、risk gate、short-horizon refresh |

如果单步 latent 误差为 \(\epsilon\)，多步 rollout 通常会随 horizon 放大：

$$
\|\hat{z}_{t+k}-z_{t+k}\| \uparrow \quad \text{as } k \text{ grows}
$$

所以世界模型不是 horizon 越长越好。实用系统通常会短 horizon 规划、频繁用真实新观测刷新，并让不确定性参与回退逻辑。

## 9. 最小复现实验应该看什么

| 证据项 | 最小要求 |
| --- | --- |
| 样本效率 | return per environment step 或 success per robot hour |
| latent 可规划 | reward/done/risk head 在 H=1/5/15 的误差 |
| 动作敏感性 | 固定历史，替换动作，未来状态和排序合理变化 |
| 闭环收益 | 接入 planner 或 actor 后任务成功率提升 |
| 反例边界 | 哪些 horizon、场景、动作分布下开始失效 |

这套检查也适用于新的视频世界模型和 WAM/VAM 工作：无论模型多大，最后都要回答预测是否改善决策。

## 10. 和后续路线的关系

| 后续路线 | 与 RSSM/Dreamer 的关系 |
| --- | --- |
| Masked / JEPA | 先学更好的 latent 表征，再接 action-conditioned dynamics |
| WAM / VAM | 尝试把动作生成、未来视频和世界预测更紧地联合建模 |
| 视频世界模型 | 更强调可视化和外部模拟，但仍要证明 action sensitivity 和 planning utility |
| 风险规划 | 在 latent rollout 上额外预测 risk、uncertainty、termination |
| 数据引擎 | 用失败 replay、near-miss 和反事实样本修正 world model 盲点 |

## 11. 小结

RSSM 提供了经典世界模型的状态骨架：确定性记忆 \(h_t\) 加随机潜状态 \(z_t\)。Dreamer 则把这套骨架真正接到控制：先从真实经验学习 world model，再在 latent imagination 中训练 actor/value，最后回到真实环境收集新数据。

读完本页后，再看 [Masked / JEPA 与潜变量预测](masked-jepa-and-latent-prediction.md) 会更清楚：为什么很多方法先优化 latent 表征。再看 [WM / WAM / VAM](wm-wam-vam-and-action-conditioned-modeling.md) 会更清楚：新路线到底是在保留 Dreamer 的分层接口，还是把动作和世界未来联合起来。
