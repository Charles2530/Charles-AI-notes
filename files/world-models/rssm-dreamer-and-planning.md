# RSSM、Dreamer 与规划：世界模型怎样变成控制信号

Dreamer 这条线要回答的是：真实环境交互很贵时，能不能先学一个内部环境，再让策略在这个内部环境里练习。RSSM 是这个内部环境的状态骨架，PlaNet 用它在线搜索动作，Dreamer 用它批量想象轨迹来训练 actor-critic。

如果只记一个判断：**posterior 用真实观测校正现在，prior 在没有未来观测时预测未来，planner 或 actor 只能在这个 latent future 上做决策。**

## 为什么单帧图像不够当状态

控制问题里，“看到什么”和“处在什么状态”不是一回事。机器人相机看到杯子在桌上，但单帧图像不告诉你末端速度、刚才是否碰到杯沿、杯子是否被遮挡过；Atari 当前帧能显示球的位置，却不一定显示球的运动方向。

世界模型不能只学：

\[
o_t \rightarrow o_{t+1}
\]

这更像一步视频预测。用于控制时，它至少要学：

\[
(o_{\le t}, a_{<t}, a_t) \rightarrow (s_{t+1}, r_{t+1}, c_{t+1})
\]

\(o_{\le t}\) 是到当前为止的观测历史，\(a_{<t}\) 是历史动作，\(a_t\) 是当前候选动作，\(s_{t+1}\) 是下一步内部状态，\(r_{t+1}\) 是奖励或任务进展，\(c_{t+1}\) 是 continuation，表示 episode 是否继续。

白话说：模型要回答“我刚才经历了什么、现在做这个动作后世界会怎样、这个后果好不好、任务是否结束”。RSSM 的作用，就是把这件事压到一个可以 rollout 的 latent state 里。

## RSSM 把记忆和不确定性分开

RSSM 全称是 Recurrent State-Space Model。它把 state 拆成两部分：

| 组成 | 作用 | 为什么需要 |
| --- | --- | --- |
| deterministic state \(h_t\) | 递归记住历史 latent 和动作 | 单帧看不到速度、遮挡和接触历史。 |
| stochastic state \(z_t\) | 表达当前不确定状态 | 同一段历史下可能存在多个未来，不能全压成平均值。 |

最小递推可以写成：

\[
h_t=f_\theta(h_{t-1}, z_{t-1}, a_{t-1})
\]

\[
z_t^{\text{post}}\sim q_\phi(z_t\mid h_t,o_t),
\qquad
z_t^{\text{prior}}\sim p_\theta(z_t\mid h_t)
\]

第一行用上一时刻的记忆、latent 和动作更新当前记忆。第二行有两条路径：posterior \(q_\phi\) 在训练时能看到真实观测 \(o_t\)，负责把当前图像里的证据并入 state；prior \(p_\theta\) 不看当前观测，只靠历史记忆预测当前 latent，未来想象时真正往前滚的就是它。

![PlaNet latent dynamics model designs](../assets/images/paper-deep-dives/world-models/planet/figure-2-latent-dynamics.png){ width="860" }

<small>图源：[Learning Latent Dynamics for Planning from Pixels](https://arxiv.org/abs/1811.04551)，Figure 2。原图比较 RNN、SSM 和 RSSM 三种 latent dynamics；这里的重点是 RSSM 同时保留 deterministic history 和 stochastic latent，适合在部分可观测环境里把历史记忆、不确定性和未来 rollout 接起来。</small>

RNN 只有确定性记忆，容易把多种可能未来压成一个平均未来。SSM 有随机状态，但缺少强历史记忆。RSSM 把两者合起来：\(h_t\) 负责长时上下文，\(z_t\) 负责当前不确定性。真实观测进入 posterior 校正状态；没有真实观测时，prior 自己往前滚。

## prior 和 posterior 分开，是为了不偷看未来

训练时，模型拥有完整轨迹：图像、动作、奖励和终止信号都在数据里。它可以用当前真实观测 \(o_t\) 得到 posterior：

\[
q_\phi(z_t\mid h_t,o_t)
\]

但规划和 imagined rollout 时，未来观测还没有发生。模型只能从当前 state 和动作出发，用 prior：

\[
p_\theta(z_t\mid h_t)
\]

KL 项真正服务的是这件事：

\[
\mathrm{KL}\left(q_\phi(z_t\mid h_t,o_t)\,\|\,p_\theta(z_t\mid h_t)\right)
\]

这行式子不是“给 VAE 加一个正则”。它在逼一个不看当前图像的 prior 去追一个看过真实图像的 posterior。训练时允许 posterior 看答案，是为了把真实世界状态编码准；同时训练 prior 贴近 posterior，是为了未来想象不要刚离开真实观测就漂移。

如果 KL 太弱，posterior 可能编码很多 prior 追不上的细节，未来 rollout 一离开真实观测就崩。如果 KL 太强，posterior 又可能被压成很好预测但信息不足的 latent，reward、接触和任务物体都丢了。DreamerV2 的 KL balancing、DreamerV3 的 dynamics loss / representation loss 和 free bits，本质上都在调这个矛盾。

## world model loss 训练的是可控制状态

RSSM 通常和 observation、reward、continuation 几个 head 一起训练。简化写法是：

\[
\mathcal L_{\text{wm}}
=
\sum_t
\Big[
-\log p_\theta(o_t\mid h_t,z_t^{\text{post}})
-\log p_\theta(r_t\mid h_t,z_t^{\text{post}})
-\log p_\theta(c_t\mid h_t,z_t^{\text{post}})
+\beta\,\mathrm{KL}(q_\phi(z_t\mid h_t,o_t)\|p_\theta(z_t\mid h_t))
\Big]
\]

观测项让 latent 保留环境结构，reward 项让 latent 对任务目标敏感，continuation 项让模型知道 episode 是否结束，KL 项让 prior 和 posterior 落在同一个可 rollout 的空间。

图像重建不是最终目标。重建图像是一种塑造 state 的信号，真正被 planner 或 actor 消费的是 latent transition、reward、continuation 和 value。一个世界模型可以画出模糊但对控制足够的未来；也可以画出好看的未来，却把 reward 或接触预测错。

## PlaNet：每一步都在 latent 里做 MPC

PlaNet 先用 RSSM 从像素、动作和奖励轨迹中学 latent dynamics，然后每一步用 CEM 在 latent space 搜索动作序列。它没有训练一个长期 policy network，决策时不断重新规划：

```text
当前 observation
  -> posterior 更新 belief state
  -> 采样多条候选 action sequence
  -> 用 RSSM prior rollout 每条候选未来
  -> reward head 估计每条未来的 return
  -> CEM 保留高分序列并重新拟合动作分布
  -> 执行第一步动作
  -> 看到新 observation 后再规划
```

这就是典型 MPC：计划一个 horizon，只执行第一步。新观测回来后，先校正 belief，再重新搜索。

PlaNet 的关键不是生成未来图像，而是避免在像素空间评估每条候选动作序列。候选动作质量主要通过 latent reward prediction 判断，所以它的数据效率高，也暴露了风险：reward head 一旦错，CEM 会积极寻找这个错误。

## Dreamer：把在线规划摊销成 actor

Dreamer 继承 RSSM，但不想每一步都用 CEM 搜索动作序列。它把 PlaNet 的“每次决策临时规划”改成“先在 imagination 里训练一个 actor”。执行时，actor 直接根据 latent state 输出动作。

Dreamer 的训练循环可以拆成三段：

```text
真实环境数据
  -> 训练 RSSM world model
  -> 从真实 posterior state 启动 imagined rollout
  -> actor 在 latent 中出动作
  -> prior dynamics 继续 rollout
  -> reward/value 估计 return
  -> 更新 actor 和 critic
```

想象阶段从真实 posterior state 出发，但未来不再看真实图像：

\[
a_\tau\sim \pi_\eta(a_\tau\mid h_\tau,z_\tau),
\qquad
z_{\tau+1}\sim p_\theta(z_{\tau+1}\mid h_{\tau+1})
\]

\[
\hat r_\tau=r_\theta(h_\tau,z_\tau),
\qquad
\hat c_\tau=c_\theta(h_\tau,z_\tau)
\]

第一组公式表示 actor 根据当前 latent state 选动作，并让 prior 预测下一步 latent；第二组公式表示 reward head 和 continuation head 从同一个 latent state 里读出学习信号。图像 decoder 可以把 imagined state 可视化，但 actor 训练不需要先解码成图像再判断动作好坏。

## lambda return 把短想象接到长期价值

imagined rollout 不能无限长。太短，看不到延迟奖励；太长，prior dynamics 的误差会沿时间累积。Dreamer 用 value network 在 horizon 末尾 bootstrap，并用 \(\lambda\)-return 在短期 reward 和长期 value 之间折中：

\[
R_t^\lambda
=
\hat r_t
+ \gamma \hat c_t
\left((1-\lambda)V_\psi(s_{t+1})+\lambda R_{t+1}^\lambda\right),
\qquad
R_T^\lambda=V_\psi(s_T)
\]

\(s_t=(h_t,z_t)\)，\(\hat r_t\) 是 world model 预测的奖励，\(\hat c_t\) 控制 episode 是否继续，\(\gamma\) 是折扣因子，\(V_\psi\) 是 critic。 \(\lambda\) 越接近 1，越相信更长 rollout；越接近 0，越早用 critic 的 value 收尾。

这行公式的直觉是：Dreamer 不要求 world model 把很远的未来都滚准，而是让短 horizon imagination 负责局部后果，再让 value 估计 horizon 之外的长期价值。效率来自内部 rollout 多、真实交互少；风险也来自同一个地方，一旦 imagined future 在关键状态上错了，actor 会把错未来当训练数据。

![Dreamer horizon ablation](../assets/images/paper-deep-dives/world-models/dreamer/figure-4-horizon.png){ width="760" }

<small>图源：[Dreamer](https://arxiv.org/abs/1912.01603)，Figure 4。原图展示 imagined horizon 对控制表现的影响；这里的重点是 rollout horizon 是偏差和收益的折中，太短看不到长期后果，太长会放大模型误差。</small>

这张图比视频预测图更能解释 Dreamer 的控制逻辑。世界模型在控制里不是一个越长越好的模拟器，而是一个有限可信区间内的训练器。工程项目不能只问“能 rollout 多远”，更要问：rollout 多远之后，reward、done、接触、关键物体和风险还准不准。

## DreamerV2 和 DreamerV3 改了什么

DreamerV2 把这条路线推进到 Atari，关键变化包括 discrete latent、KL balancing、面向离散动作和复杂视觉任务的 actor-critic 细节。它说明 latent world model 不只适合连续控制，也能在强离散控制基准上训练 policy。

DreamerV3 的重点不再是单个环境技巧，而是跨域稳定 recipe。它保留 RSSM 和 imagined actor-critic 的主线，但加入 symlog、symexp two-hot、free bits、return normalization 等数值稳健化设计，让同一套配置覆盖 Atari、DMLab、连续控制、Crafter、Minecraft 等任务。

![DreamerV3 world model learning](../assets/images/paper-deep-dives/world-models/dreamerv3/x2.png){ width="900" }

<small>图源：[Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104)，Figure 3(a)。原图展示 DreamerV3 world model learning：观测被编码成离散 latent representation，recurrent dynamics 在动作条件下预测未来 latent；这里的重点是 DreamerV3 的稳定化技巧仍服务同一目标，让 prior、posterior、reward、continuation 和 reconstruction 在不同任务尺度下共同塑造可决策 state。</small>

这里最值得学的是设计重心的变化：PlaNet 证明 RSSM 能规划，Dreamer 证明 latent imagination 能训练 actor，DreamerV2 证明 discrete world model 能打 Atari，DreamerV3 证明这套 recipe 可以更少依赖单任务调参。

## 每篇论文承担一个问题

| 论文 / 项目 | 它回答的问题 | 对主线的作用 |
| --- | --- | --- |
| [World Models](https://worldmodels.github.io/) | 能否用 VAE + RNN + controller 建立 latent dream 的直觉 | 给“在内部模型里练习”提供早期概念入口。 |
| [PlaNet](https://arxiv.org/abs/1811.04551) | 能否从像素学 RSSM，再用 CEM 在 latent space 规划 | 奠定 RSSM 和 latent MPC。 |
| [Dreamer](https://arxiv.org/abs/1912.01603) | 能否不每步搜索，而在 imagination 里训练 actor-critic | 把规划摊销成 policy。 |
| [DreamerV2](https://arxiv.org/abs/2010.02193) | world model 能否在 Atari 这种强离散基准上训练行为 | 引入 discrete latent 和 KL balancing 经验。 |
| [DayDreamer](https://arxiv.org/abs/2206.14176) | Dreamer 能否直接在真实机器人上在线学习 | 说明路线能离开仿真，但真实硬件仍有安全和数据约束。 |
| [DreamerV3](https://arxiv.org/abs/2301.04104) | 能否用同一套配置跨很多任务稳定工作 | 把 Dreamer 系列升级成更通用的 model-based RL recipe。 |
| [MuZero](https://arxiv.org/abs/1911.08265) | 不重建图像，只学 reward/value/policy 相关 dynamics 是否可行 | 对比“任务特定 latent dynamics”和 Dreamer 的图像重建路线。 |

这个表的用法不是背年份，而是看接口：有的论文在问 state 怎么学，有的在问 action 怎么选，有的在问 policy 怎么从 imagined future 更新，有的在问真实机器人能不能承受在线试错。

## 和视频世界模型、VLA 的边界

Dreamer-style world model 优先学习：

\[
s_t,a_t\rightarrow s_{t+1},r_{t+1},c_{t+1}
\]

视频世界模型更常见的问题是：

\[
o_{\le t},a_{\le t}\rightarrow o_{t+1:t+H}
\]

前者关心可训练策略的内部动力学，后者关心可视未来是否连贯。两者可以互相启发，但证据不能混用：好看的 open-loop 视频不证明 policy 能闭环成功；高回报 RL benchmark 也不证明模型能生成可检查的真实世界视频。

VLA 又多一层接口。Dreamer 的动作通常来自环境定义的连续或离散 action space；真实机器人还要处理相机外参、控制频率、动作限幅、力控、碰撞、失败恢复和安全停止。DayDreamer 说明 Dreamer 可以直接用于多种真实机器人在线学习，但这仍然是受控任务和硬件设置下的证据，不能自动外推成开放家庭机器人已经解决。

## 判断一篇世界模型论文是否讲清楚

读新的世界模型论文时，先画五张账本：

| 账本 | 要问的问题 |
| --- | --- |
| state | state 是像素、token、latent vector、RSSM state，还是显式 3D / object state？是否包含历史、动作和不确定性？ |
| dynamics | 模型预测 \(s_{t+1}\)、图像、reward、done、risk 中的哪些量？是 action-free video prediction，还是 action-conditioned rollout？ |
| consumption | 未来被谁消费？CEM / MPC、actor-critic、value search、VLA policy、risk checker，还是只做人类可视化？ |
| horizon | rollout 多远？多远之后误差开始污染 reward、done、接触和目标物体？是否有 value bootstrap 或不确定性门控？ |
| evidence | 论文给的是 reconstruction、open-loop prediction、benchmark return、real robot success、ablation，还是失败案例？ |

这五张账能防止把“视频预测好看”“latent 表示强”“控制回报高”混成同一种证据。世界模型最终要服务决策，证据也必须沿着状态、动作、rollout、消费方和闭环结果一路追下去。

## 最后判断

RSSM / Dreamer 这条线的知识含量在三点。第一，posterior 和 prior 分工清楚：posterior 用真实观测校正当前 state，prior 才是在未来想象中使用的 dynamics。第二，world model loss 不只是图像重建，而是观测、奖励、continuation 和 KL 一起塑造可控制的 latent state。第三，PlaNet 和 Dreamer 的差别不是有没有世界模型，而是世界模型被怎么消费：PlaNet 用它在线搜索动作，Dreamer 用它训练一个能快速执行的 actor。

这条路线的价值在于减少真实试错，把策略学习从昂贵环境转移到便宜 latent rollout。它的边界也很清楚：model bias 会被 planner 或 actor 放大，视频预测不能替代闭环回报，真实机器人还要额外面对延迟、接触、不可逆失败和安全约束。

继续往下读，可以接 [Masked / JEPA 与潜变量预测](masked-jepa-and-latent-prediction.md)、[WM / WAM / VAM：动作条件建模](wm-wam-vam-and-action-conditioned-modeling.md) 和 [世界模型评测与失效模式](evaluation-and-failure-modes.md)。

## 外部材料

- [PlaNet project page](https://planetrl.github.io/)：先看 latent dynamics、CEM planning 和 planning from pixels 的整体直觉。
- [PlaNet paper](https://arxiv.org/abs/1811.04551)：重点读 RSSM、latent overshooting 和 CEM planning。
- [Dreamer project page](https://danijar.com/project/dreamer/)：理解 latent imagination 如何训练 actor/value。
- [Google Research: Introducing Dreamer](https://research.google/blog/introducing-dreamer-scalable-reinforcement-learning-using-world-models/)：官方博客把 PlaNet 和 Dreamer 的差别讲得很清楚。
- [DreamerV2](https://arxiv.org/abs/2010.02193)：重点看 discrete latent、KL balancing 和 Atari 证据。
- [DreamerV3](https://danijar.com/project/dreamerv3/)：重点看固定配置跨域、Minecraft diamond 和 scaling。
- [DayDreamer](https://danijar.com/project/daydreamer/)：看 Dreamer 路线进入真实机器人时增加了哪些实际约束。
