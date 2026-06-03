# 世界模型评测：不要让好看的未来骗过决策系统

世界模型评测不能只看一步预测误差，也不能只看生成视频是否清晰。一个模型可能短期 rollout 很准，却在闭环里被自己的误差放大；也可能视频看起来很自然，但换动作后未来几乎不变，根本无法服务规划。

这页只回答一个问题：怎样判断一个世界模型真的能改善决策，而不是只会补帧或生成演示。

![PlaNet open-loop predictions 原论文图](../assets/images/paper-deep-dives/world-models/planet/figure-9-open-loop-predictions.png){ width="560" .atlas-figure-tall }

<small>图源：[Learning Latent Dynamics for Planning from Pixels](../paper-deep-dives/world-models/planet.md)，Figure 9。原图展示 PlaNet 从真实历史出发做 open-loop latent dynamics 预测。本站读法：不要只看第一帧像不像，要沿时间看对象身份、空间布局、任务变量和误差漂移。</small>

## 先分清评测对象

世界模型至少有三类用途。第一类是表示世界状态，把历史观测压成 latent 或 belief；第二类是预测动作后果，给定状态和动作预测未来状态、reward、risk 或 done；第三类是服务决策，把预测交给 planner、policy、risk checker 或数据引擎。

不同用途的评测完全不同。只做表示学习时，可以看状态探针、未来预测和关键变量可读性；做动作条件 dynamics 时，必须看反事实动作分叉；服务规划时，必须看 closed-loop success、regret、risk calibration 和 cost per success。

如果论文只展示 open-loop 视频，不应直接推断它能规划。如果论文只展示 closed-loop 成功率，也不应忽略失败是否可回放和归因。

## Open-loop 看漂移，Closed-loop 看反馈

Open-loop 评测是在真实历史之后，让模型自己向前滚一段。它适合离线大规模比较，也适合画 horizon curve。一步预测损失可以写成：

$$
\mathcal L_{1\text{-step}}
=
\mathbb E[\ell(\hat o_{t+1}, o_{t+1})]
$$

这里 \(\hat o_{t+1}\) 是模型预测的下一步观测，\(o_{t+1}\) 是真实下一步，\(\ell\) 是像素、latent、状态或任务变量上的误差。它只检查一步，因此很容易高估模型能力。

多步 rollout 更接近世界模型使用方式：

$$
\mathcal L_{\text{rollout}}(H)
=
\frac{1}{H}\sum_{h=1}^{H}
\mathbb E[\ell(\hat o_{t+h}, o_{t+h})]
$$

这里 \(H\) 是预测 horizon。随着 \(H\) 增大，如果误差快速上升，说明模型只能做局部拟合，不能稳定推演。对机器人和自动驾驶来说，这种漂移会把 planner 带到真实轨迹之外。

Closed-loop 评测更严格。模型的预测会影响下一步动作，下一步动作又改变后续输入。很多模型 open-loop 看起来不错，closed-loop 却失败，因为 planner 会主动寻找模型里的高回报漏洞。

## 反事实动作是最低门槛

动作条件世界模型必须回答一个最小问题：固定同一历史，换动作，未来是否合理分叉。

可以把动作敏感性写成：

$$
\Delta_{\text{act}}
=
d(\hat o_{t+1:t+H}^{(a)}, \hat o_{t+1:t+H}^{(a')})
$$

其中 \(a\) 和 \(a'\) 是两组不同动作，\(\hat o^{(a)}\) 与 \(\hat o^{(a')}\) 是对应未来，\(d\) 可以是 latent 距离、任务事件差异、风险差异或成功概率差异。这个量不是越大越好，而是要在相关变量上合理变化：向左移动应该改变视角和遮挡，不应该让无关物体随机换身份。

反事实评测应至少包含三类样本。普通样本检查模型是否理解常规动作后果；边界样本检查两个候选动作成功率接近时模型是否排序正确；hard negative 检查碰撞、滑落、遮挡、错误抓取等危险未来是否被低估。

## 风险校准要按场景桶看

世界模型用于真实系统时，平均误差不够。高风险场景里，一个低置信但可回退的模型，比一个平均分高却在 near-miss 上过度自信的模型更安全。

风险评测要看五件事：高不确定性是否对应更高真实误差，高风险预测是否对应更高失败率，预测区间是否覆盖真实未来，不同场景桶的校准是否一致，不确定性升高时回退策略是否真的更保守。

Near-miss 要单独建桶。真实事故稀有，但 near-miss 更高频，且对数据回流更有价值。记录时不要只存视频片段，还要保存动作候选、模型预测、真实结果、最小安全距离、接触前风险分数、planner 是否继续推进、是否触发回退或人工接管。

## 失败回放比总分更重要

一个世界模型失败日志应能回答“为什么选错动作”。例如移动操作机器人要把杯子放到上层橱柜，有两个候选动作：先抬手再伸过去，或者先靠近再抬手。真实结果是前者碰到架沿，后者成功；模型却把前者判成低风险。

这条样本不应只记成“失败一次”，而要拆成四个判定：状态表示是否保留杯子、柜门和架沿位置；反事实排序是否把安全动作排在前面；风险分数是否把碰撞动作放进高风险桶；planner 依赖该预测时是否被安全层截停。

如果模型生成的视频更清晰，却仍把危险动作排在安全动作前，它没有通过世界模型评测。世界模型的目标不是生成更好看的失败，而是帮助系统避开失败。

![DreamZero failure case 原论文图](../assets/images/paper-deep-dives/world-models/dreamzero/figure-16-failure-case.png){ width="920" }

<small>图源：[DreamZero](../paper-deep-dives/embodied-ai/dreamzero.md)，Figure 16。原图展示 video-action model 的失败案例。本站读法：WAM 的危险在于错误未来和错误动作可能互相支持，看起来像一条连贯计划；评测必须检查真实观测刷新后能否纠正，以及 safety layer 能否提前截停。</small>

## 失效模式按修复路径分类

世界模型 failure mode 最好按“怎么修”分类，而不是只写现象。

| 失效类型 | 现象 | 更可能的修复方向 |
| --- | --- | --- |
| 表示失败 | 小物体丢失、遮挡后失忆、接触状态不可读 | 改 tokenizer / latent、加关键状态标签或辅助头 |
| 动力学失败 | 短期准，长期漂移，多主体交互崩 | 多步训练、latent rollout 正则、闭环失败回放 |
| 动作条件失败 | 换动作后未来几乎不变 | 增加反事实动作数据、改动作注入方式 |
| 平均化未来 | 多种可能被洗成均值 | 分支采样、随机 latent、扩散或离散 token 生成 |
| 规划漏洞 | planner 找到模型里高回报但现实坏的动作 | adversarial planning test、不确定性惩罚、真实回放 |
| 部署失败 | 太慢、不可监控、不能回退 | 分层使用、低频分析、fallback、资源预算 |

这个表的意义是把评测和下一轮训练连接起来。否则世界模型报告会停在“某指标下降”，却不知道是表示、动力学、动作、风险还是系统预算出了问题。

## 发布门槛怎么写

发布门槛应围绕模型消费方，而不是围绕论文榜单。给 planner 用，就看候选动作排序、closed-loop success、regret 和 cost per success；给 risk checker 用，就看 collision recall、near-miss recall、ECE 和 false negative replay；给 data engine 用，就看 hard-negative yield、失败回放是否对再训练有帮助；给人审用，就看反事实是否可解释。

一个可执行的门槛可以这样写：目标任务桶的 counterfactual ranking 至少提升，near-miss / collision / human takeover 任一高风险桶不能退化，主分布成功率不能明显下降，所有新增失败必须能定位到表示、动作条件、风险或系统预算中的至少一类。

最终判断很简单：如果世界模型不能在闭环里提升动作选择、风险识别或数据回流，它还只是预测模型或生成模型；只有当它改变系统决策质量时，才算通过验收。

## 外部精读

- [PlaNet: Learning Latent Dynamics for Planning from Pixels](https://arxiv.org/abs/1811.04551)：理解 open-loop latent dynamics 和 planning 的经典入口。
- [DreamerV3](https://arxiv.org/abs/2301.04104)：理解 imagined rollout 如何真正训练策略，而不只是预测未来。
- [World Action Models are Zero-shot Policies](https://arxiv.org/abs/2602.15922)：理解 WAM 失败为什么必须同时看未来视频和未来动作。
- [V-JEPA 2 world model blog](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/)：理解物理推理与规划 benchmark 如何从表征走向世界模型评测。
