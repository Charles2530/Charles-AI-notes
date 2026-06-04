# 世界模型 QA

这一页单独整理世界模型方向的高频问题。重点不是背 PlaNet、Dreamer、JEPA、Genie、WAM 这些名字，而是能在面试里讲清：模型预测什么、动作如何进入、效率来自哪里、什么证据能证明它可用于规划，什么证据只能说明视频生成或 demo 形态。

## 基础定义与能力边界

### Q：世界模型到底是什么？

**面试回答。** 世界模型是对环境动态的内部模型，用来预测当前状态在动作影响下会如何演化。最小形式可以写成：

\[
p_\theta(s_{t+1}\mid s_t,a_t)
\]

更完整的模型还会预测 reward、done/continue、risk、未来 observation 或 latent state。它的目标不是只生成自然画面，而是服务规划、控制、反事实推演或样本高效学习。

**追问展开。** 如果面试官问“和生成模型有什么关系”，可以说生成模型关注从条件生成样本，世界模型关注 action-conditioned transition。世界模型可以用生成式方法实现，但不等于所有生成模型都是世界模型。  
**易错点 / 边界。** 最大误区是把“能生成未来视频”当成“世界模型可用于控制”。没有动作条件和闭环评测，只能说明 open-loop prediction 或 video prior。  
**继续读。** [从零理解世界模型](../world-models/beginner-world-models-primer.md) / [世界模型评测与失效模式](../world-models/evaluation-and-failure-modes.md)

### Q：世界模型和普通视频生成模型有什么区别？

**面试回答。** 普通视频生成模型通常学习 \(p(\text{video}\mid \text{text/image/history})\)，目标是生成自然、连贯、符合条件的视频；世界模型要学习动作如何改变未来，至少要建模 \(p(s_{t+1:t+H}\mid s_t,a_{t:t+H})\)。前者强调视觉质量，后者强调可预测、可比较、可用于决策。

**追问展开。** 你可以用一个反例解释：固定当前画面，如果输入“向左转”和“向右转”，模型预测的未来应该不同；如果几乎不变，它可能只是视频先验，不是动作敏感的世界模型。  
**易错点 / 边界。** FVD、VBench、视频自然度和 demo 观感不能证明 planning utility；必须看动作敏感、闭环成功或反事实一致性。  
**继续读。** [生成式仿真与视频](../world-models/generative-simulation-and-video.md) / [视频与多模态扩散](../diffusion/video-and-multimodal-diffusion.md)

### Q：世界模型一般预测 pixel、latent 还是 embedding？

**面试回答。** 三者都可以。Pixel prediction 可解释性强但成本高，容易把容量浪费在纹理；latent prediction 把观测压缩成状态，适合控制和长时 rollout；embedding prediction 更抽象，可能更适合表征学习和下游任务。选择取决于模型服务什么：重建、规划、控制、评估还是表示学习。

**追问展开。** Dreamer 类方法偏 latent state，JEPA 类方法偏 embedding prediction，视频扩散类方法常偏 pixel/latent video generation。真正要问的是预测空间是否保留决策变量，如动作、接触、遮挡、目标和风险。  
**易错点 / 边界。** Latent 更小不等于更适合控制；如果 latent 丢掉动作敏感信息，规划会失败。  
**继续读。** [Masked / JEPA 与潜变量预测](../world-models/masked-jepa-and-latent-prediction.md) / [世界模型评测与失效模式](../world-models/evaluation-and-failure-modes.md)

### Q：状态、观测和 latent state 有什么区别？

**面试回答。** Observation 是传感器看到的原始或近原始数据，如图像、深度、 proprioception；state 是环境真正用于决策的变量，可能包含不可直接观测的信息；latent state 是模型从历史观测和动作中压缩出的内部表示，用来近似决策所需状态。

**追问展开。** 在 POMDP 里，单帧 observation 往往不够，模型需要历史 \(o_{\le t}, a_{<t}\) 推断 belief state。RSSM 这类模型就是把历史压成 latent state，再在 latent 中 rollout。  
**易错点 / 边界。** 不要把一帧图像等同于完整状态。遮挡、速度、接触力、目标意图都可能不在单帧里。  
**继续读。** [RSSM、Dreamer 与规划](../world-models/rssm-dreamer-and-planning.md) / [MDP、价值函数与 Bellman](../reinforcement-learning/mdp-value-bellman.md)

## RSSM、Dreamer 与强化学习

### Q：RSSM 为什么适合做世界模型？

**面试回答。** RSSM 把 deterministic recurrent state 和 stochastic latent state 结合起来，既能保留历史信息，又能表达未来不确定性。它通常用 encoder 把 observation 编到 latent，用 transition model 预测下一个 latent，用 decoder/reward/continue head 做重建和任务信号预测。

\[
z_t \sim q_\phi(z_t\mid h_t,o_t),\quad
h_{t+1}=f_\theta(h_t,z_t,a_t)
\]

**追问展开。** 面试里可以强调 RSSM 的关键是把高维像素控制问题转到 latent dynamics 里，避免每次规划都在像素空间做。  
**易错点 / 边界。** RSSM 不是单纯 autoencoder；如果不建模动作条件 transition、reward 和 continue，就不能支撑控制学习。  
**继续读。** [RSSM、Dreamer 与规划](../world-models/rssm-dreamer-and-planning.md) / [PlaNet：像素规划的潜变量动力学](../paper-deep-dives/world-models/planet.md)

### Q：Dreamer 的核心训练闭环是什么？

**面试回答。** Dreamer 先用真实交互数据训练 latent world model，再在 latent imagination 中 rollout 多步未来，用想象轨迹训练 actor-critic。它的价值是减少真实环境交互成本，把大量策略优化放到可学习的 latent dynamics 中。

\[
z_{t+1}\sim p_\theta(z_{t+1}\mid z_t,a_t),\quad
a_t\sim \pi_\psi(a_t\mid z_t)
\]

**追问展开。** 面试回答要讲清三件事：world model 学 transition/reward/continue，actor 在想象轨迹里选动作，critic 估计 imagined return。Dreamer 不是只做视频预测，而是 model-based RL。  
**易错点 / 边界。** Imagined rollout 如果模型有偏，策略会 exploit model error；所以需要真实数据刷新、短 rollout、正则和闭环评测。  
**继续读。** [RSSM、Dreamer 与规划](../world-models/rssm-dreamer-and-planning.md) / [Dreamer：想象轨迹中训练控制策略](../paper-deep-dives/world-models/dreamer.md)

### Q：PlaNet 和 Dreamer 的差别是什么？

**面试回答。** PlaNet 用 latent dynamics 做 planning，常通过 MPC 在 latent 里搜索动作序列；Dreamer 更进一步，在 latent imagination 中训练显式 actor-critic，让策略学会直接输出动作。简化说，PlaNet 更偏“每次规划”，Dreamer 更偏“用想象训练策略”。

**追问展开。** 这也是从 model predictive control 到 latent policy learning 的区别。MPC 在线优化成本高但灵活；actor 学出来后推理快，但依赖训练分布和 world model 质量。  
**易错点 / 边界。** 不要把所有 RSSM 方法都叫 Dreamer；PlaNet、Dreamer、DreamerV2/V3 在规划、离散 latent、任务规模和训练细节上都有差别。  
**继续读。** [PlaNet：像素规划的潜变量动力学](../paper-deep-dives/world-models/planet.md) / [DreamerV3：latent 里训练策略](../paper-deep-dives/world-models/dreamerv3.md)

### Q：世界模型和 model-free RL 的取舍是什么？

**面试回答。** Model-free RL 直接学习 policy 或 value，不显式建模环境转移；世界模型路线先学 dynamics，再用模型做规划或想象训练。世界模型通常更样本高效，因为可以复用数据做 imagined rollout，但会引入 model bias 和 exploitation 风险。

**追问展开。** 面试可以说：model-free 少一个模型误差来源但真实交互成本高；model-based 省交互但模型错了会带偏策略。机器人和游戏里选择哪条路线，要看数据成本、模型可学性、任务复杂度和安全。  
**易错点 / 边界。** 不要只说 model-based 更省样本。它还要付出训练 dynamics、uncertainty、planning 和验证成本。  
**继续读。** [世界模型中的强化学习](../reinforcement-learning/rl-for-world-models.md) / [MDP、价值函数与 Bellman](../reinforcement-learning/mdp-value-bellman.md)

## JEPA、Masked Prediction 与表示学习

### Q：JEPA / masked prediction 为什么常被拿来谈世界模型？

**面试回答。** JEPA 和 masked prediction 在 latent/embedding 空间预测缺失部分，避免像素重建里的大量低层细节，能学习更抽象的表征。它们可能成为世界模型的状态表示层，让后续 dynamics 不必处理原始像素。

**追问展开。** V-JEPA 这类方法强调预测 target encoder 的 latent representation，而不是重建 RGB。这样能减少纹理细节干扰，更关注语义和结构。  
**易错点 / 边界。** JEPA 表征不等于完整世界模型。没有动作输入、reward/risk 和 closed-loop 验证，就不能说它能规划。  
**继续读。** [Masked / JEPA 与潜变量预测](../world-models/masked-jepa-and-latent-prediction.md) / [V-JEPA：视频潜变量预测表征](../paper-deep-dives/world-models/v-jepa.md)

### Q：Masked reconstruction 和 latent prediction 的差别是什么？

**面试回答。** Masked reconstruction 要恢复被遮挡的像素或 token，目标更偏重细节重建；latent prediction 预测另一个 encoder 产生的表示，目标更偏向语义和结构。前者容易学到低层纹理，后者更可能形成下游可用表征。

**追问展开。** MAE 类方法重建像素，JEPA 类方法预测 representation。世界模型更关心后者是否保留动作相关和任务相关变量。  
**易错点 / 边界。** Latent prediction 更抽象不代表更可控；如果 representation 对动作变化不敏感，它仍不能服务规划。  
**继续读。** [Masked / JEPA 与潜变量预测](../world-models/masked-jepa-and-latent-prediction.md) / [V-JEPA：视频潜变量预测表征](../paper-deep-dives/world-models/v-jepa.md)

### Q：V-JEPA 2-AC 这类 action-conditioned 表征为什么重要？

**面试回答。** 它把 JEPA 表征和动作条件规划接起来，让模型不只是预测视频表征，还能在 latent 空间中评估动作导致的未来变化。这样才更接近可规划世界模型。

**追问展开。** 面试回答可以说：action-conditioned latent planning 的关键是目标图像、当前状态、候选动作之间的关系。模型必须让不同动作产生可区分的 latent 未来。  
**易错点 / 边界。** 有 action-conditioned 论文结果也不代表通用机器人策略已解决；任务设置、平台、目标形式和 closed-loop 评测都要看。  
**继续读。** [Masked / JEPA 与潜变量预测](../world-models/masked-jepa-and-latent-prediction.md) / [Claim Ledger](../references/claim-ledger.md)

## 动作条件、WAM/VAM 与生成式仿真

### Q：Action-conditioned world model 为什么是关键分水岭？

**面试回答。** 控制关心“我做这个动作会怎样”，所以世界模型必须让动作影响未来预测。没有动作条件，模型只能学环境自然演化或视觉先验；有动作条件，模型才能比较候选动作并支持规划。

\[
p_\theta(\tau_{t+1:t+H}\mid s_t,a_{t:t+H})
\]

**追问展开。** 面试里可以说 action conditioning 要能通过反事实测试：同一个初始状态，换动作，未来状态、风险和成功概率应发生合理变化。  
**易错点 / 边界。** 动作作为输入还不够。很多模型可能忽略动作，只靠视觉先验生成“看起来合理”的未来。  
**继续读。** [WM / WAM / VAM：动作条件建模](../world-models/wm-wam-vam-and-action-conditioned-modeling.md) / [世界模型评测与失效模式](../world-models/evaluation-and-failure-modes.md)

### Q：WAM、VAM 和 VLA 的差别怎么说？

**面试回答。** VLA 主要从视觉和语言输出动作，关注 policy；WAM 通常强调 world-action model，即联合建模视频/状态和动作，让模型预测动作与世界变化；VAM 更偏 video-action modeling，把动作和视频生成/预测绑在一起。三者边界不是绝对，但关注点不同：VLA 输出动作，WAM/VAM 更强调动作-后果建模。

**追问展开。** 面试中可以用一句话区分：VLA 问“下一步做什么”，WAM 问“做了以后世界怎样变”，世界模型问“能否用这个预测来规划”。  
**易错点 / 边界。** 不要把名字当能力证明。无论叫 WAM 还是 VAM，都要看动作敏感性、闭环成功率和失败恢复。  
**继续读。** [WM / WAM / VAM：动作条件建模](../world-models/wm-wam-vam-and-action-conditioned-modeling.md) / [VLA、WAM 与世界模型系统图](../embodied-ai/vla-wam-world-model-system-map.md)

### Q：Genie 这类无动作标签方法为什么有启发？

**面试回答。** Genie 类方法试图从视频中学习 latent actions，让模型能从观察数据中发现可控因素。它的启发是：即使没有真实动作标签，也可以通过视频变化学习某种可交互结构。

**追问展开。** 但 latent action 和真实机器人动作不同。latent action 可能对应视频中的抽象变化，不一定能映射到机械臂关节、末端位姿或移动机器人控制。  
**易错点 / 边界。** 无动作标签可交互视频模型不等于可部署机器人策略；要看 latent action 是否可控、可解释、可接真实动作接口。  
**继续读。** [Genie：无动作标签学可交互环境](../paper-deep-dives/world-models/genie.md) / [WM / WAM / VAM：动作条件建模](../world-models/wm-wam-vam-and-action-conditioned-modeling.md)

### Q：为什么视频世界模型要关心因果化和流式生成？

**面试回答。** 真实交互是时间向前发生的，控制系统不能看到未来帧。视频世界模型如果训练时使用双向上下文，推理时做自回归 rollout，就会有训练-推理错配。因果化和流式生成让模型更接近在线决策场景。

**追问展开。** CausVid、Diffusion Forcing、Self Forcing 等路线都在处理视频生成和在线 rollout 的错配。核心问题是：模型能不能边接收新观测、边更新未来预测。  
**易错点 / 边界。** 视频流式更快不等于规划更好；还要看动作条件和闭环任务。  
**继续读。** [生成式仿真与视频](../world-models/generative-simulation-and-video.md) / [CausVid：流式自回归视频扩散](../paper-deep-dives/diffusion/causvid.md)

## 评测、证据与工程落地

### Q：世界模型评测最容易被什么骗？

**面试回答。** 最容易被 open-loop 指标骗。一个模型可以 FVD、loss 或视频自然度很好，但对动作不敏感，规划时不能区分好动作和坏动作。世界模型评测要看 closed-loop success、action sensitivity、反事实一致性、风险校准和失败回放。

**追问展开。** 面试里可以给反例：固定当前画面，换不同动作，未来预测几乎不变；这说明模型学了视觉先验，却没学动作动力学。  
**易错点 / 边界。** 系统吞吐、官方 demo、open-loop 视频都不能替代真实或仿真闭环评测。  
**继续读。** [世界模型评测与失效模式](../world-models/evaluation-and-failure-modes.md) / [证据判断原则](../references/site-judgement-principles.md)

### Q：Action sensitivity 应该怎么测？

**面试回答。** Action sensitivity 测的是模型未来预测是否真正随动作变化。基本做法是固定当前状态和目标，输入多个候选动作，观察未来 latent、视频、reward、risk 或 success score 是否产生合理差异，并和真实环境或高保真仿真对照。

**追问展开。** 更进一步可以看动作排序是否正确：模型认为更好的动作，真实执行成功率是否更高。也可以看 counterfactual consistency：小动作变化是否带来合理的小未来变化。  
**易错点 / 边界。** 只看生成视频有变化不够；变化必须和动作语义一致，并能提升决策。  
**继续读。** [世界模型评测与失效模式](../world-models/evaluation-and-failure-modes.md) / [Claim Ledger](../references/claim-ledger.md)

### Q：Open-loop loss、FVD 和 closed-loop success 分别能证明什么？

**面试回答。** Open-loop loss 说明模型在固定数据轨迹上的预测误差；FVD 说明生成视频分布和真实视频分布的视觉/时序相似度；closed-loop success 说明模型或策略在环境中实际执行任务的成功率。三者证据强度不同。

**追问展开。** 如果目标是规划和控制，closed-loop success 更接近最终目标；open-loop 指标可用于模型调试，但不能替代执行结果。  
**易错点 / 边界。** 不要用 FVD 证明控制能力。它最多说明视频分布相似，不说明动作选择正确。  
**继续读。** [全站证据与复现状态标准](../references/evidence-and-reproducibility-standard.md) / [世界模型评测与失效模式](../world-models/evaluation-and-failure-modes.md)

### Q：世界模型为什么需要不确定性和风险建模？

**面试回答。** 真实环境有部分可观测、随机性和模型误差。世界模型如果只给单一路径预测，策略可能过度相信错误未来；不确定性和风险 head 可以帮助 planner 避开高风险动作，或在不确定时请求真实观测刷新。

**追问展开。** 常见做法包括 ensemble、stochastic latent、分布式预测、risk score、uncertainty calibration。机器人和自动驾驶尤其需要知道“模型不知道什么”。  
**易错点 / 边界。** 平均预测很准不代表风险可控；长尾碰撞、遮挡和接触失败常在平均指标里被淹没。  
**继续读。** [不确定性与风险感知规划](../world-models/uncertainty-and-risk-aware-planning.md) / [世界模型评测与失效模式](../world-models/evaluation-and-failure-modes.md)

### Q：世界模型如何和推理系统、KV cache、量化这些工程问题连接？

**面试回答。** 大规模视频/多模态世界模型推理成本很高，长 rollout 会消耗大量 KV cache、显存和采样步数。系统优化可以降低 rollout latency、并发成本和部署门槛，例如 KV 压缩、少步扩散蒸馏、量化、batching 和缓存。

**追问展开。** 但系统吞吐只证明成本下降，不证明模型预测更准或规划更好。工程优化必须和 action sensitivity、closed-loop success、质量回归同表评估。  
**易错点 / 边界。** 不能因为 rollout 更快就说世界模型更强；速度和任务 utility 是两条证据链。  
**继续读。** [世界模型高效训练实验报告](../world-models/world-model-efficient-training-experiment-report.md) / [Claim Ledger](../references/claim-ledger.md)

### Q：如果面试让你设计一个世界模型系统，你会怎么说？

**面试回答。** 我会先定义任务和状态：输入哪些观测、动作空间是什么、预测 horizon 多长、输出是 latent/reward/risk 还是视频。然后设计数据闭环：收集轨迹、训练 transition/reward/risk、做 open-loop 和 action sensitivity 检查，再接 planner 或 policy 做 closed-loop eval。

**追问展开。** 系统结构可以是 `encoder -> latent dynamics -> reward/risk heads -> planner/policy -> evaluator`。评测上必须报告 cost per success、failure replay、latency 和不确定性。  
**易错点 / 边界。** 不要一上来就说“用一个视频生成大模型”。世界模型设计必须从任务、动作、状态和评测闭环出发。  
**继续读。** [发展路线：数据、仿真与评测](../world-models/development-roadmap-data-simulation-and-evaluation.md) / [VLA、WAM 与世界模型系统图](../embodied-ai/vla-wam-world-model-system-map.md)
