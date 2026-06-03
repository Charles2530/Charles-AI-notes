# WM / WAM / VAM：动作到底怎样进入世界模型

近两年，世界模型、视频模型和机器人策略越来越容易被写到同一篇论文里：有的给动作预测未来，有的直接从视觉语言输出动作，有的把未来视频和未来动作一起生成。`WM / WAM / VAM` 不是统一标准术语，但很适合作为读论文的接口坐标系。

这页只抓一个问题：**动作在模型里放在哪里，决定了它能不能服务规划和控制。**

不要先问模型名字。先问四件事：输入里有没有动作，输出里有没有未来，换动作后未来会不会变，最后谁消费这个未来。一个模型能生成漂亮视频，不等于能做控制；一个模型能输出动作，也不等于理解动作后果。世界模型用于机器人时，最小门槛是：同一历史下换一条动作，模型预测的未来、风险或成功排序应该随之改变。

## 先把接口摆正

| 路线 | 动作在哪里 | 模型回答的问题 | 主要消费方 | 最容易误读成 |
| --- | --- | --- | --- | --- |
| WM | 动作是条件 | 如果我做 \(a\)，世界会怎样变 | planner、critic、risk checker | 普通视频预测器 |
| VLA policy | 动作是输出 | 当前观测和语言下，现在该做什么 | controller、task executor | 有世界模型的 agent |
| WAM | 动作和未来联合输出 | 目标下，什么动作和什么未来应该一起出现 | policy、trajectory generator、closed-loop controller | 视频模型加动作头 |
| VAM | 视频预测表征服务动作 | 视频模型学到的动态先验能不能帮助选动作 | action model、data engine、robot policy | 生成视频就能控制 |

这张表的用法不是给缩写下定义，而是防止证据错位。WM 要看反事实动作和候选动作排序；VLA 要看真实动作成功率和恢复；WAM 要看视频、动作和闭环执行是否互相支撑；VAM 要看视频预测的中间表征是否真的提升 policy，而不是视频画质是否好看。

## WM：动作作为条件，未来作为评估对象

最清楚的世界模型接口是：

\[
p_\theta(z_{t+1:t+H}, r_{t:t+H}, d_{t:t+H}, u_{t:t+H}
\mid z_t, a_{t:t+H-1}, c)
\]

\(z_t\) 是当前 latent state，\(a_{t:t+H-1}\) 是候选动作序列，\(r\) 是 reward 或任务进展，\(d\) 是 done / continuation，\(u\) 是 risk 或 uncertainty，\(c\) 是语言、目标图像、地图等上下文。这个式子的重点不是概率符号，而是动作在条件里：模型被要求回答“给定这串动作，未来会怎样”。

Dreamer 系列是干净的 WM 基线。它不追求把每个未来像素画清楚，而是在 latent space 里学习 dynamics，再用 imagined rollout 训练 actor 和 critic。读 Dreamer 时要看动作如何进入 transition、reward/value 如何消费未来，而不是把它当成普通视频生成器。

WM 的核心证据是反事实动作。固定同一个 \(z_t\)，替换 \(a_{t:t+H-1}\)，未来状态、reward、risk 或 done 应该合理变化。如果换动作后未来几乎不变，模型只是惯性视频预测器；如果未来会变，但 reward/risk 不变，planner 仍然没有可靠依据。

Genie 2 / Genie 3 这类可交互世界模型把这个接口扩展到更开放的 3D 环境：从图像或提示生成可被人或 AI agent 操作的 playable world。它说明“世界模型”不只限于机器人 latent dynamics，也可以是可交互生成环境。但用于机器人控制时仍要补上机器人动作空间、风险、任务成功、真实观测刷新和安全层。

## VLA：能输出动作，不等于有世界模型

VLA policy 的常见接口更像：

\[
a_t = \pi_\theta(o_{\le t}, l)
\]

\(o_{\le t}\) 是观测历史，\(l\) 是语言指令，输出直接是动作 token、连续控制量或 action chunk。RT-2 这类模型的意义在于把 web-scale 视觉语言知识迁移到机器人动作空间；Open X-Embodiment / RT-X 则说明跨机器人数据需要处理动作口径、相机、embodiment 和任务语义。

但 policy 输出动作，不代表模型内部显式预测了动作后果。一个 VLA 可以很会把“把苹果放进碗里”翻译成动作模板，却不一定能回答“如果我从右侧推，苹果会不会碰倒杯子”。所以本站不把所有 VLA 都称为世界模型。只有当模型显式预测或评估动作后的未来，并让 planner、risk checker 或 policy 消费这个未来时，才进入世界模型证据层。

这不是贬低 VLA。很多真实系统会让 VLA 直接做 policy，再用世界模型或风险模型做前瞻检查。关键是把角色说清：VLA 负责把语义落到动作，WM 负责评估动作后果，低层 controller 负责让动作可执行。

## WAM：动作和未来世界一起生成

WAM 更像把 world model 和 policy 合成一个生成过程。它不只是问“给定动作后未来怎样”，还问“在这个目标下，什么动作和什么未来应该一起出现”。简化接口可以写成：

\[
p_\theta(o_{t+1:t+H}, a_{t:t+K}\mid o_{\le t}, s_{\le t}, g)
\]

\(o_{\le t}\) 是历史观测，\(s_{\le t}\) 是本体状态，\(g\) 是语言或目标条件，模型同时输出未来观测和未来动作。动作不再只是外部输入，而是和未来世界一起被建模。

![DreamZero joint video-action architecture](../assets/images/paper-deep-dives/world-models/dreamzero/figure-4-architecture.png){ width="920" }

<small>图源：[World Action Models are Zero-shot Policies / DreamZero](https://arxiv.org/abs/2602.15922)，Figure 4。本站复用已有论文图，未使用 image2 生成新图。原图表达：训练时联合去噪未来视频 latent 和动作 latent，推理时用真实观测刷新上下文。本站读法：WAM 不是“视频模型旁边加动作头”，而是让动作与未来世界互相约束，并在闭环中不断用真实观测纠偏。</small>

WAM 的吸引力在于，很多机器人动作的合理性只能通过未来来判断。抓杯子不是单帧分类，拉抽屉也不是一句语言到一个动作 token 的翻译；它们要求模型理解接触、遮挡、手爪姿态、目标物移动和失败恢复。把未来视觉和未来动作联合建模，理论上能让模型从视频动态里学到更多物理约束。

但 WAM 的风险也更难排查。动作错了，未来视频可能仍然看起来合理；未来视频错了，动作又可能被错误未来“解释”得好像合理。联合生成会让归因变难：到底是语言理解错、视频动态错、动作接口错，还是闭环刷新太慢？

所以读 WAM 论文时，要把 open-loop 和 closed-loop 分开。open-loop 视频能证明模型会生成某种未来；closed-loop rollout 才证明动作在真实观测刷新后还能继续有效。DreamZero 论文和项目页报告了 7Hz closed-loop control、real robot generalization 和 cross-embodiment transfer，这类证据比单纯展示未来视频更接近 WAM 的核心 claim；但它仍是特定数据、机器人平台和任务集合下的论文证据。

## VAM：视频先验不直接等于策略

VAM 可以理解成“让视频模型的动态表征帮助动作生成”。它通常不要求视频模型自己就是 planner，而是把未来预测过程中的 latent、attention 或 intermediate feature 交给 action model。

\[
\phi_{t:t+H}=F_{\mathrm{video}}(v_{\le t}, g),
\qquad
a_{t:t+K}=G_{\mathrm{policy}}(o_{\le t}, g, \phi_{t:t+H})
\]

\(F_{\mathrm{video}}\) 是视频模型，\(\phi\) 是预测过程中产生的动态表征，\(G_{\mathrm{policy}}\) 是动作模型。这个式子想表达的重点是：视频模型提供的不是最终画出来的视频，而是对未来运动、遮挡、接触和任务阶段的内部表示。

![Video Prediction Policy overview](../assets/images/paper-deep-dives/embodied-ai/vpp/figure-1-overview.png){ width="920" }

<small>图源：[Video Prediction Policy](https://arxiv.org/abs/2412.14803)，Figure 1。本站复用已有论文图，未使用 image2 生成新图。原图展示先训练文本条件视频预测模型，再聚合视频模型内部 predictive visual representations 来输出机器人动作。本站读法：中间表征比最终视频画质更关键；VAM 要证明视频预测学到的动态先验能提升动作成功率。</small>

VPP 的动机很直接：传统视觉 encoder 往往偏静态，能识别物体，却不一定知道下一秒手和物体会怎样相互作用；视频扩散模型为了预测未来帧，可能学到更强的动态表征。VAM 把这种动态表征转给 policy，让动作模型少从零学习物理变化。

这条路线的边界也要讲清。视频好看不等于动作成功，FVD 或视觉质量不等于控制质量。一个合格 VAM 实验应报告动作成功率、真实机器人 rollout、跨场景泛化、失败恢复、动作平滑度和推理延迟。视频预测只是中间机制，机器人是否完成任务才是最终证据。

## V-JEPA 2-AC：在 latent 里比较动作后果

V-JEPA 2-AC 很适合作为 WM 与 VAM 之间的桥。它先用大规模视频/图像做 action-free self-supervised representation，再用少量机器人轨迹后训练 action-conditioned world model。规划时，给当前观测、目标图像和候选动作序列，在 latent space 里 rollout，选让未来表示接近目标表示的动作。

\[
\hat z_{t+1:t+H}=f_\theta(z_t,a_{t:t+H-1}),
\qquad
a^\star=\arg\min_a D(\hat z_{t+H}, z_{\mathrm{goal}})
\]

这个公式的重点是：动作作为候选条件进入模型，未来 latent 作为比较对象，planner 选择代价最低的动作。它比纯 VLA 更强调动作后果，比完整视频生成更节省输出成本。它也提醒我们：action-free 视频预训练可以提供动态先验，但要规划真实动作，仍需要 interaction data 把表示接到 action-conditioned world model。

## 动作放在哪里，失败就在哪里

| 动作位置 | 典型失败 | 怎么验证 |
| --- | --- | --- |
| 输入条件 | 模型忽略动作，未来只按视频惯性滚动。 | 固定历史，替换动作，比较未来 latent、reward、risk。 |
| 输出动作 | 动作像训练集平均行为，不理解后果。 | 真实观测刷新后连续执行，看是否能恢复偏差。 |
| 联合输出 | 视频和动作互相“圆谎”，归因困难。 | 分开评估视频预测、动作成功、失败回放和人审。 |
| prompt 弱条件 | 文本说左转，模型按常见运动继续。 | 构造反事实 prompt 和动作冲突案例。 |
| latent planning | latent 距离不等于任务成功。 | 用真实机器人或 simulator 验证 goal reach、collision、done。 |

这张表是写作和实验的核心。世界模型不是一个标签，而是动作、未来和消费方之间的合同。合同最薄的地方，就是最容易失败的地方。

一个机器人系统还必须处理闭环刷新。无论模型多强，真实环境都会偏离想象：物体滑动、遮挡变化、手爪接触失败、相机延迟、控制器限幅都会改变下一步输入。开环生成两秒动作看起来顺滑，不代表闭环执行安全。靠谱系统通常会短 horizon 执行、重新观测、重新规划，并把失败样本回流给数据引擎。

## 按消费方读证据

| 消费方 | 应该看的证据 | 不能替代它的证据 |
| --- | --- | --- |
| Planner | candidate ranking、MPC / CEM ablation、closed-loop success、cost per success | 只展示未来视频 |
| Policy | action chunk 成功率、真实观测刷新后的稳定性、失败恢复 | 离线 action L2 loss |
| Risk checker | collision recall、near-miss recall、risk calibration | 视频清晰度或 CLIPScore |
| Data engine | hard-negative yield、failure replay usefulness、新任务覆盖 | 合成样本数量 |
| 人审 / 调试 | 可视 rollout、反事实回放、错误归因 | 平均 benchmark 分数 |

论文里最容易偷换的是证据等级。视频 demo 只能说明模型能生成可视未来；offline benchmark 只能说明某些数据分布上预测或动作接近标签；real robot closed-loop 才能说明模型输出在真实噪声、延迟和偏差下仍能完成任务。读 WM/WAM/VAM 论文时，先把证据放到对应消费方，再判断 claim 有没有越界。

## 最后判断

WM / WAM / VAM 的核心不是缩写，而是动作接口。

WM 把动作作为条件，最小证据是反事实动作会改变未来、reward 或 risk。WAM 把动作和未来联合生成，最小证据是真实观测刷新后的闭环执行。VAM 用视频预测表征帮助动作，最小证据是机器人任务成功率和泛化，而不是视频画质。

大规模视频预训练可以提供动态先验，但要服务控制，必须接上动作、目标、风险、延迟和恢复。读相邻内容时，可以接 [RSSM、Dreamer 与规划](rssm-dreamer-and-planning.md)、[Masked / JEPA 与潜变量预测](masked-jepa-and-latent-prediction.md)、[VLA 动作表示与控制接口](../vla/action-representation-and-interfaces.md) 和 [世界模型评测与失效模式](evaluation-and-failure-modes.md)。

## 外部精读

- [DreamerV3](https://arxiv.org/abs/2301.04104)：理解 latent world model 如何通过 imagined future 训练行为。
- [Genie 2 DeepMind blog](https://deepmind.google/blog/genie-2-a-large-scale-foundation-world-model/)：理解 action-controllable world model 和可交互生成环境的边界。
- [Genie 3](https://deepmind.google/models/genie/)：看 Google DeepMind 对交互式世界模型最新入口的官方表述。
- [V-JEPA 2 Meta research page](https://ai.meta.com/research/publications/v-jepa-2-self-supervised-video-models-enable-understanding-prediction-and-planning/)：理解 action-free 视频预训练如何接到 action-conditioned planning。
- [DreamZero / World Action Models](https://dreamzero0.github.io/)：理解 WAM 为什么强调联合预测未来视频和动作，以及为什么要看 closed-loop control。
- [Video Prediction Policy](https://video-prediction-policy.github.io/)：理解 VAM 如何把视频模型内部的 predictive visual representation 接到机器人动作。
- [RT-2 DeepMind blog](https://deepmind.google/blog/rt-2-new-model-translates-vision-and-language-into-action/)：对比 VLA 直接输出动作与 WM/WAM/VAM 预测动作后果之间的差异。
