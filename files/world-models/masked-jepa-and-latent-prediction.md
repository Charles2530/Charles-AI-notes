# Masked / JEPA 与潜变量预测：高效世界模型的表征路线

这页专门补上高效世界模型里最容易被轻描淡写的一环：**先学什么样的状态表示，才值得在上面做动作条件 rollout**。`Masked World Models`、`JEPA / V-JEPA` 和 `V-JEPA 2-AC` 都在减少像素级预测的浪费，但它们解决的不是同一个问题。

!!! note "先给结论"
    Masked / JEPA 路线能显著改善表示效率和样本效率，因为它们把训练压力从逐像素复原转向预测更抽象、更稳定的 latent 或 joint embedding。但原始 MWM / V-JEPA 通常不完整包含 `action`、`reward`、`done` 和 closed-loop 消费接口，所以不能直接等同于可规划世界模型。要进入世界模型主线，必须再接 action-conditioned dynamics、风险/奖励头和闭环评测。

## 本页解决哪项成本

| 成本项 | 这页的答案 | 典型技术 |
| --- | --- | --- |
| 像素重建成本 | 不把建模容量花在背景纹理和低价值像素上 | masked feature reconstruction、latent prediction、joint embedding prediction |
| 视觉 token 成本 | 用 encoder / tokenizer / latent state 压缩长视频 | MWM autoencoder、V-JEPA tokenization、video latent |
| 真实交互成本 | 更好的视觉表征让 dynamics 和 policy 用更少样本学习 | MWM + DreamerV2 对照、V-JEPA 2-AC robot planning |
| 动作条件成本 | 把动作从下游 policy 标签提升为 latent dynamics 的条件 | V-JEPA 2-AC、Dreamer/RSSM、WAM |
| 评测成本 | 明确哪些结果是 representation eval，哪些才是 closed-loop world-model eval | Claim Ledger、读数边界、失败边界表 |

## 四种训练目标对照

同样是“预测未来或遮挡部分”，训练目标不同，省下的成本和能支持的结论也不同。

| 目标 | 简化 loss | 输入输出 | 高效机制 | 不能直接证明 |
| --- | --- | --- | --- | --- |
| Pixel reconstruction | \(\mathcal{L}_{pix}=d(D_\theta(z), x)\) | 输入观测或上下文，输出像素 | 监督密集、可视化直接 | 预测像素准不等于能选对动作 |
| Token / latent reconstruction | \(\mathcal{L}_{tok}=d(\hat{z}_M,z_M)\) | 输入未遮挡 token，输出遮挡 token 的 latent | 降低输出维度，减少无关纹理压力 | latent 是否保留 reward/risk/action 仍需评测 |
| Joint-embedding prediction | \(\mathcal{L}_{jepa}=d(P_\theta(E_c(x_{\setminus M})),\text{sg}(E_t(x_M)))\) | context encoder 看可见部分，target encoder 产生目标表征 | 无需像素 decoder，target encoder 提供稳定目标 | 原始 JEPA 不含动作、奖励、终止 |
| Action-conditioned latent dynamics | \(\mathcal{L}_{dyn}=d(\hat{z}_{t+1},z_{t+1})+\lambda_r\ell(\hat{r},r)+\lambda_d\ell(\hat{d},d)\) | 输入当前 latent 和动作，输出未来 latent / reward / done | 在低维状态里比较候选动作，减少真实试错 | 多步漂移和模型漏洞仍会破坏 closed-loop |

可以把它们理解为一条链：`pixel reconstruction` 证明模型能看见，`latent reconstruction` 证明模型能压缩，`joint embedding prediction` 证明模型能预测抽象结构，`action-conditioned latent dynamics` 才开始回答“如果我这样做，会发生什么”。

## Masked World Models：把视觉表征和 dynamics 解耦

`Masked World Models for Visual Control` 的核心设计是：先用 masked convolutional feature autoencoder 学视觉表征，并加入 reward prediction 让表征更贴近任务；再在这个 latent space 上训练 dynamics model。这个设计把“看懂画面”和“学习动作条件状态转移”分开，避免 dynamics 学习被高维像素重建拖住。

![MWM framework](../assets/images/paper-deep-dives/world-models/masked-world-models/figure-1-mwm-framework.png){ width="920" }

<small>图源：[Masked World Models for Visual Control](https://arxiv.org/abs/2206.14244)，Figure 1。原图意：左侧用 masked convolutional features 训练视觉 autoencoder，并预测 reward；右侧在 autoencoder latent space 中训练 latent dynamics，且 dynamics learning 阶段不更新 autoencoder 参数。</small>

!!! note "这张图怎么读"
    **输入输出**：左边输入图像 \(o_t\)，遮挡卷积特征后重建观测并预测 reward；右边输入 latent \(z_t\)、动作历史和 stochastic state，预测下一步 latent。<br>
    **效率机制**：表征学习用 mask 逼迫 encoder 保留全局结构和任务相关信息，dynamics 只在压缩 latent 上学习，不直接承担像素级预测。<br>
    **对世界模型主线的意义**：它把 visual representation 和 latent dynamics 的责任拆开，是“先压缩，再规划”的清晰案例。<br>
    **不能证明什么**：这张框架图不能证明 MWM 在任意机器人、任意动作空间或真实闭环场景中都比 Dreamer 更稳。

![MWM results](../assets/images/paper-deep-dives/world-models/masked-world-models/figure-4-mwm-results.png){ width="920" }

<small>图源：[Masked World Models for Visual Control](https://arxiv.org/abs/2206.14244)，Figure 4。原图意：展示 MWM 与 DreamerV2 在 Meta-world aggregate、RLBench Reach Target 和 RLBench Push Button 上的 success rate 学习曲线。</small>

!!! note "这张图怎么读"
    **输入输出**：横轴是环境交互步数，纵轴是成功率；蓝线是 MWM，黑线是 DreamerV2。<br>
    **效率机制**：如果同等环境步数下蓝线更高，说明 masked representation + latent dynamics 在这些视觉控制设置中提高了样本效率。<br>
    **对世界模型主线的意义**：这是“表征目标会影响控制样本效率”的直接证据，比只展示重建图更接近世界模型评价。<br>
    **不能证明什么**：它不能证明所有 masked model 都具备可规划能力，也不能替代 action sensitivity、reward 校准和真实机器人闭环评测。

!!! warning "读数边界"
    这组曲线是特定 Meta-world / RLBench 实验设置下的 paper result。它支持“MWM 在这些视觉控制任务中相对 DreamerV2 更 sample efficient”，但不支持“masked reconstruction 本身已经解决动作因果”或“所有视频世界模型都应优先用 MWM 结构”。

## V-JEPA：预测 latent，而不是把像素画回来

V-JEPA 的关键转向是：把视频中被遮挡的部分作为 target encoder 表征来预测，而不是用 decoder 重建像素。这样做可以减少对低层纹理的执念，把容量更多放在对象、运动和场景结构上。

![V-JEPA 3D multi-block masking](../assets/images/paper-deep-dives/world-models/v-jepa/figure-2-3d-multi-block-masking.png){ width="860" }

<small>图源：[V-JEPA: Latent Video Prediction for Visual Representation Learning](https://arxiv.org/abs/2404.08471)，Figure 2。原图意：在视频时空体上采样多个 3D block mask，让 context encoder 只能看到部分时空区域。</small>

!!! note "这张图怎么读"
    **输入输出**：输入是一段视频 token grid，mask 会遮住多个时空块，模型要从可见上下文预测被遮挡区域的 target latent。<br>
    **效率机制**：高比例、结构化 mask 迫使模型利用运动和对象关系，而不是复制局部纹理。<br>
    **对世界模型主线的意义**：它解释了为什么 masked video pretraining 可以作为世界模型的状态表示前置训练。<br>
    **不能证明什么**：mask 预测本身不等于理解动作后果，因为这里没有动作条件和 reward/done 输出。

![V-JEPA training procedure](../assets/images/paper-deep-dives/world-models/v-jepa/figure-3-training-procedure.png){ width="920" }

<small>图源：[V-JEPA: Latent Video Prediction for Visual Representation Learning](https://arxiv.org/abs/2404.08471)，Figure 3。原图意：context encoder 处理 masked sequence，predictor 用 learnable mask tokens 预测目标位置，target encoder 处理未遮挡完整输入并提供 latent target。</small>

!!! note "这张图怎么读"
    **输入输出**：context encoder 输出可见 token 的表示，predictor 输出 masked positions 的预测表示，target encoder 给出 stop-gradient 的目标表示。<br>
    **效率机制**：目标是 latent space 的 \(L_1\) 预测，不需要昂贵像素 decoder，也减少对背景纹理的过拟合。<br>
    **对世界模型主线的意义**：这是一种 internal world model 表征路线，适合作为下游 latent dynamics 或 planning head 的输入。<br>
    **不能证明什么**：它不能单独证明模型能做 MPC、actor-critic 或风险规划。

![V-JEPA tokenization](../assets/images/paper-deep-dives/world-models/v-jepa/figure-4-tokenization.png){ width="860" }

<small>图源：[V-JEPA: Latent Video Prediction for Visual Representation Learning](https://arxiv.org/abs/2404.08471)，Figure 4。原图意：展示视频如何被切分成时空 token，并通过 encoder/predictor 进入 latent prediction 目标。</small>

!!! note "这张图怎么读"
    **输入输出**：连续视频先被分块 token 化，再进入 encoder 和 predictor；输出不是像素，而是目标 token 的 latent。<br>
    **效率机制**：tokenization 决定上下文长度，latent prediction 决定输出维度，两者一起控制训练和推理成本。<br>
    **对世界模型主线的意义**：多相机、长 horizon 世界模型首先会被 token 数压垮，V-JEPA 这类路线提供了压缩状态的候选接口。<br>
    **不能证明什么**：token 化更省不等于下游控制必然更好，仍需验证接触、遮挡、风险和动作敏感性是否保留。

## V-JEPA 2-AC：从表征预测走向动作条件规划

`V-JEPA 2-AC` 是这条路线里更接近世界模型主线的一步：它不仅使用视频自监督预训练得到的表征，还训练 action-conditioned predictor，并用 model predictive control 搜索动作序列，使未来 latent 接近目标图像 latent。

![V-JEPA 2-AC planning](../assets/images/paper-deep-dives/world-models/v-jepa2/figure-action-conditioned-world-model.png){ width="920" }

<small>图源：[V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning](https://arxiv.org/abs/2506.09985)，Figure 7。原图意：在 V-JEPA 2-AC 中，对候选动作序列做 model predictive control，让 imagined state representation 在未来 \(T\) 步后接近 goal representation。</small>

!!! note "这张图怎么读"
    **输入输出**：输入是当前观测、目标图像和一段候选动作；模型在 latent space 里 rollout，输出未来表示并用 \(L_1\) 距离和目标表示比较。<br>
    **效率机制**：规划不需要生成完整未来视频，只在表征空间评估动作序列，因此比像素级视频 rollout 更便宜。<br>
    **对世界模型主线的意义**：这是 JEPA-style 表征真正接到 action-conditioned planning 的关键桥梁。<br>
    **不能证明什么**：它仍不自动证明跨机器人、跨任务、长 horizon 风险规划都成立；证据要看 closed-loop robot 设置和动作空间约束。

## 失败边界：什么时候不要把它叫完整世界模型

| 情况 | 为什么不够 | 需要补什么 |
| --- | --- | --- |
| 只有 masked image/video reconstruction | 学到了视觉补全，不一定学到动作后果 | action-conditioned dynamics、反事实动作评测 |
| 只有 JEPA 表征 benchmark | 表征好不等于能规划 | reward/done/risk heads、closed-loop policy 或 MPC |
| 只报告 FVD/VBench/demo | 视频自然度和 planning utility 不是同一指标 | action sensitivity、candidate ranking agreement、cost per success |
| latent 太压缩 | 省 token 但可能丢接触、透明物体、遮挡身份 | task-state probing、failure replay、hard-negative eval |
| mask 或 tokenizer 只在静态数据上验证 | 静态视觉泛化不能覆盖控制分布 shift | online/offline hybrid、真实 rollout 回流 |

## 工程判断：何时选哪条路线

| 需求 | 优先路线 | 原因 |
| --- | --- | --- |
| 机器人/游戏样本效率，已有 reward/done | RSSM / Dreamer / MWM | 直接围绕 latent dynamics 和 control signal 优化 |
| 大规模视频预训练，缺少动作 | V-JEPA / H-JEPA | 先学稳定视频表征，再接动作头 |
| 需要目标图像驱动的规划 | V-JEPA 2-AC / MPC | 能在 latent space 比较候选动作与目标 |
| 需要生成可检查视频反事实 | action-conditioned video world model | 需要外部模拟和人类/系统审查 |
| 需要实时闭环控制 | latent dynamics + 少步/因果 rollout | 优先压延迟和 drift，而不是追求像素质量 |

## Claim Ledger

| Claim | Source | Evidence Type | Can Support | Cannot Prove |
| --- | --- | --- | --- | --- |
| MWM 在论文设置中通过 decoupled representation + dynamics 改善视觉控制样本效率 | [MWM](https://arxiv.org/abs/2206.14244) | Paper Result | masked visual representation 可以服务 model-based visual control | 任意 masked model 都能 closed-loop 规划 |
| V-JEPA 通过 latent target prediction 避免了部分像素重建成本 | [V-JEPA](https://arxiv.org/abs/2404.08471) | Ablation | joint-embedding prediction 是高效视频表征预训练路线 | 原始 V-JEPA 已经包含动作、奖励和终止建模 |
| V-JEPA 2-AC 把 JEPA 表征接到动作条件 latent planning | [V-JEPA 2](https://arxiv.org/abs/2506.09985) | Closed-loop | 表征空间 MPC 可以在特定机器人设置中执行目标驱动任务 | 跨平台通用控制、长时风险规划和真实部署安全性 |
| masked / JEPA 路线适合作为世界模型前置表征 | 本页机制归纳 | Site Inference | 可作为“先压缩，再建模动作”的工程路径 | 单独替代 RSSM、reward head、risk head 或 closed-loop 评测 |

## 接回主线

读完这页后，建议按下面顺序回到主线：

1. 接 [RSSM、Dreamer 与规划](rssm-dreamer-and-planning.md)，看 latent dynamics 如何真正进入 actor-critic 和 imagined rollout。
2. 接 [WM / WAM / VAM 与动作条件建模](wm-wam-vam-and-action-conditioned-modeling.md)，判断动作是条件、输出还是和世界联合生成。
3. 接 [世界模型评测与失效模式](evaluation-and-failure-modes.md)，用 action sensitivity、latent drift 和 closed-loop gain 验证表征是否真的有用。
4. 回到 [世界模型高效训练技术路线图](efficient-training-roadmap.md)，把 masked/JEPA 放进数据、token、显存、rollout 和验证成本账里。
