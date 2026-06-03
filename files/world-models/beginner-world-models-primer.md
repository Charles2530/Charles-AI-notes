# 从零理解世界模型：预测行动后的可用未来

世界模型不是“能生成未来视频”的模型。它真正回答的问题是：**如果智能体在当前局面下执行某个动作，未来会怎样，风险在哪里，下一步还值不值得做。**

这句话里有三个关键词。第一是当前局面，它通常不是单帧图像，而是历史观测、动作和记忆压成的状态。第二是动作，因为世界模型必须支持反事实：同一个起点，向左走、向右走、打开抽屉、停下来，未来应该不同。第三是可用未来，因为预测结果最终要被 planner、policy、risk checker 或 evaluator 消费。

所以入门时先不要问“它能不能生成视频”。更好的问题是：**它是否把 belief state、action、rollout 和决策接口接成了一条闭环。**

## 最小接口

一个决策智能体每一步都在问：

```text
我现在看到这些东西，也记得刚才做过什么。
如果我执行动作 a，会看到什么变化，会得到什么收益，会不会失败？
```

把它写成最小接口，就是：

```text
历史观测 + 历史动作 + 候选动作
-> 未来状态 / 观测 / 奖励 / 风险 / 终止
-> planner 或 policy 选择下一步
```

这个接口能把几类模型分开。

| 模型看起来会做什么 | 为什么还不一定是决策世界模型 |
| --- | --- |
| 生成一段合理未来视频 | 如果动作换了未来不变，它更像视频生成器。 |
| 重建当前图像或补全遮挡 | 如果没有 transition 和 rollout，不能评估候选动作。 |
| 学到强视频表征 | 如果没有 action-conditioned dynamics，只能作为状态编码候选。 |
| 预测 reward 或风险 | 如果状态不稳或动作不敏感，planner 会利用模型漏洞。 |

世界模型的要点不在“未来看起来像不像”，而在“未来能不能帮助做决定”。

## 观测、状态和 belief state

观测 observation 是传感器当下看到的东西：RGB、深度、文本、触觉、关节角、速度、电流。真实状态 state 是足以预测未来的局面：物体位置、速度、遮挡、接触、任务阶段和不可见约束。latent state 是模型学到的压缩表示，用来近似真实状态。

现实里智能体很少直接拿到完整状态。单帧图像看不到速度，看不到杯子背后的遮挡物，也不知道夹爪上一秒是否碰到了物体。因此世界模型通常维护 belief state：给定历史观测和历史动作后，智能体对当前世界状态的内部估计。

\[
b_t \approx p(s_t \mid o_{\le t}, a_{<t})
\]

\(b_t\) 是 belief state，\(s_t\) 是真实但通常不可直接观测的状态，\(o_{\le t}\) 是到当前为止的观测序列，\(a_{<t}\) 是已经执行过的动作。白话说：模型不是只看一张图，而是用“我一路看见了什么、做了什么”来猜当前世界到底处在什么状态。

这就是为什么 PlaNet、Dreamer 和很多机器人世界模型都要用 recurrent state、memory token 或 latent dynamics。它们不是为了把架构写复杂，而是因为部分可观测世界里，记忆本身就是状态的一部分。

## 动作条件是分水岭

普通视频预测可以写成：

\[
p_\theta(o_{t+1}\mid o_{\le t})
\]

它问的是“接下来通常会发生什么”。动作条件世界模型要写成：

\[
p_\theta(o_{t+1}\mid o_{\le t}, a_t)
\]

它问的是“如果我做这个动作，接下来会发生什么”。这就是世界模型和普通视频续写最容易混淆、也最重要的差别。

在机器人里，这个差别非常具体。同一个桌面场景下，夹爪左移和右移应该导致不同接触；推杯子和绕过杯子应该导致不同轨迹；停下来和继续前进应该导致不同风险。如果模型没有学到动作对未来的因果影响，它不能做反事实规划，只能做条件不足的未来猜测。

![State-space video world model](../assets/images/paper-deep-dives/world-models/towards-video-world-models/figure-5-state-space-video-world-model.png){ width="900" }

<small>图源：[Towards Video World Models](https://www.xunhuang.me/blogs/world_model.html)，Figure 5。本站复用已有图，未使用 image2 生成新图。原图表达 observation、latent state、action、rollout 和 policy/planner 的关系；本站用它说明世界模型的关键接口是 state + action -> future，而不是单纯生成一段视频。</small>

读这张图时先找三件事：state 从哪里来，action 在哪里进入，rollout 的输出被谁消费。如果没有消费方，模型最多是演示器；如果 action 不影响未来，模型就无法做反事实规划。

## Rollout：在模型里向未来展开

Rollout 是从当前状态出发，连续预测多步未来：

```text
z_t --a_t--> z_{t+1} --a_{t+1}--> z_{t+2} --...--> z_{t+H}
```

rollout 的好处是便宜。智能体可以先在模型里试很多候选动作序列，再把看起来最好的动作拿到真实世界执行。PlaNet 用 latent planning 评估候选动作序列；Dreamer 则在 latent imagined rollout 里训练 actor 和 critic。

但 rollout 也有危险：模型误差会逐步累积。一步预测很准，不代表十步后还准；视频看起来顺滑，不代表 reward 或接触状态正确；模型越被策略利用，越可能暴露训练数据里没覆盖的角落。

这就是世界模型评测不能只看 open-loop prediction loss 的原因。真正要问的是：多想几步之后，模型是否仍保留任务关键状态；planner 是否因为模型偏差选错动作；策略是否能回到真实环境里闭环成功。

## 三条主线

世界模型今天大致有三种常见形态。它们不是互相替代，而是在回答不同层级的问题。

| 路线 | 核心问题 | 典型代表 | 关键证据 |
| --- | --- | --- | --- |
| latent 控制模型 | 如何把历史压成可规划状态 | World Models、PlaNet、Dreamer | latent rollout、reward/value、控制回报 |
| 生成式交互环境 | 如何生成可交互、可观察的未来 | Genie、Genie 2、视频世界模型 | action-controllable demo、counterfactual、agent eval |
| 预测式表征 / JEPA | 如何学习不依赖像素重建的时空状态 | V-JEPA、V-JEPA 2 | latent prediction、action-conditioned planning、闭环任务 |

### latent 控制模型

PlaNet 和 Dreamer 代表经典 model-based RL 路线。它们不追求生成最高清的视频，而是把观测压进 latent state，让 dynamics、reward 和 policy 在 latent space 中工作。

PlaNet 更偏在线规划：给定当前 belief，搜索候选 action sequence，选择模型预测回报最高的动作。Dreamer 更偏策略学习：在 latent imagined rollout 中训练 actor，执行时直接输出动作。二者共同说明：世界模型的价值在于把未来预测转成策略学习或动作选择信号。

### 生成式交互环境

Genie、Genie 2 和视频世界模拟器更强调外部可见的未来：生成环境、可交互场景、动作后果或视频 rollout。它们的优势是可视化强，容易做人类检查、数据生成和交互原型；难点是长时一致性、动作可控性、物理约束、低延迟和 closed-loop 评测。

DeepMind 在 Genie 2 中把重点放在 action-controllable、playable 3D environments：单张 prompt image 可以生成可被人或 AI agent 用键鼠交互的环境，并强调同一起点下不同动作会产生不同轨迹。这类系统更接近“可玩的世界模拟器”，但读它时仍要区分 demo、open-loop consistency、agent evaluation 和真实机器人闭环。

生成式世界模型的关键边界是：画面可信不等于动力学可信，长视频一致不等于动作因果正确，能被人玩不等于能安全训练真实机器人。它适合做模拟、数据引擎、交互原型和任务评测，但要进入决策闭环，还必须证明动作、状态、风险和奖励接口可用。

### 预测式表征和 JEPA

V-JEPA / V-JEPA 2 代表另一条路线：不一定生成像素，而是在 latent representation 里预测被遮挡或未来的信息。这样的好处是模型可以少花容量在纹理重建上，把重点放在时空结构、物体运动和抽象状态。

V-JEPA 2 的读法很适合初学者：第一阶段从大量视频中学习理解和预测，第二阶段用较少机器人交互视频把 latent world model 接到动作和规划。它说明“强视频表征”和“可规划世界模型”不是同一件事，中间还差 action-conditioned dynamics、goal interface 和闭环验证。

## 证据强弱怎么判断

世界模型论文最容易被误读，因为不同证据回答的是不同问题。

| 证据 | 它能说明什么 | 不能说明什么 |
| --- | --- | --- |
| reconstruction / prediction loss | 模型能否复原或预测训练分布里的观测 | 不证明动作因果和控制收益。 |
| 视频 demo | 画面是否连贯、是否有明显物理错觉 | 不证明 planner 可用，也不证明长尾安全。 |
| action counterfactual | 同一起点下不同动作是否导致不同未来 | 还要看未来是否任务相关、是否可闭环。 |
| reward / done / risk prediction | latent 是否保留任务信号 | 不自动证明策略会利用好这些信号。 |
| planning ablation | 世界模型是否帮助选择动作 | 通常依赖任务、horizon、搜索预算和数据覆盖。 |
| closed-loop success | 模型是否真的改善真实或仿真任务 | 需要看泛化、失败桶、安全边界和复现条件。 |

因此读一篇世界模型论文时，不要先问“它是不是 SOTA”。先问：预测对象是什么，动作怎么进入，rollout 被谁消费，证据停在哪一层。一个视频世界模型可以很强，但仍只证明 open-loop 生成；一个 latent model 画面不好看，却可能对控制很有用。

## 最后判断

世界模型最该记住的不是某个模型名，而是一个接口：**belief state + action -> future useful for decision**。观测要被压成可记忆的状态，动作要真正改变未来，rollout 要能被 planner、policy 或 risk checker 消费，证据要从 open-loop 预测一路追到闭环任务。

这也是为什么世界模型会同时出现三种形态：Dreamer 式 latent 控制模型，Genie 式可交互生成环境，V-JEPA 式预测表征。它们不是互相替代，而是在回答不同层级的问题：如何记住世界，如何预测动作后果，如何让未来服务决策。

继续往下读，可以接 [RSSM、Dreamer 与规划](rssm-dreamer-and-planning.md)、[Masked / JEPA 与潜变量预测](masked-jepa-and-latent-prediction.md)、[WM / WAM / VAM：动作条件建模](wm-wam-vam-and-action-conditioned-modeling.md) 和 [世界模型评测与失效模式](evaluation-and-failure-modes.md)。

## 外部精读

- [World Models](https://worldmodels.github.io/)：经典直觉入口，用 VAE、MDN-RNN 和 controller 展示“在自己的梦里学习”的早期路线。
- [PlaNet](https://arxiv.org/abs/1811.04551)：理解 latent dynamics、planning from pixels 和 latent overshooting。
- [Dreamer](https://arxiv.org/abs/1912.01603)：理解 imagined actor-critic 和 latent imagination。
- [DeepMind Genie 2](https://deepmind.google/blog/genie-2-a-large-scale-foundation-world-model/)：看 action-controllable playable world 的现代官方表述。
- [Meta V-JEPA 2 research page](https://ai.meta.com/research/publications/v-jepa-2-self-supervised-video-models-enable-understanding-prediction-and-planning/)：看 action-free 表征预训练如何接到 action-conditioned planning。
- [Towards Video World Models](https://www.xunhuang.me/blogs/world_model.html)：用系统图梳理视频世界模型、状态空间和交互模拟器之间的关系。
