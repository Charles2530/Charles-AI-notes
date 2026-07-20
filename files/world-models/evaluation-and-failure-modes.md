# 世界模型评测：不要让好看的未来骗过决策系统

评测世界模型时，最容易被一段漂亮视频带偏。视频很清晰、动作很顺、物体看起来也没穿模，但只要换一个候选动作，未来仍然差不多；或者 planner 按它的预测去执行，真实环境里马上碰撞、滑落、卡住。这样的模型可以是很强的视频模型，却还不是可靠的决策模型。

世界模型评测要判断模型是否改善动作选择、风险识别或数据回流，而不只是生成更顺眼的未来。根据消费方选择 open-loop、反事实、closed-loop 和风险分桶指标，才能判断模型是否达到发布门槛。

## 先问谁要用它

同一个世界模型，给不同消费方用，验收口径完全不同。读论文或项目页时，先不要急着看总分，而要先定位它被谁消费。

| 消费方 | 它真正需要什么 | 不够用的指标 |
| --- | --- | --- |
| 状态编码器 | latent 保留目标、接触、遮挡、位姿和任务阶段 | 重建图像清晰 |
| 视频模拟器 | 未来视频一致、可控、能持续交互 | 单帧美学分 |
| 动作 planner | 候选动作排序正确，错误动作被压低 | open-loop FVD |
| risk gate | near-miss、碰撞、滑落、不可达不漏报 | 平均预测误差 |
| data engine | 合成样本能补失败桶，进入训练后有效 | 生成样本数量 |
| policy evaluator | 离线/仿真评估与真实成功率相关 | 演示视频成功 |

因此，一份有效的世界模型评测需要覆盖多组接口测试。接口不同，分数不能互相代替。

## 三把尺子

第一把是画面尺。它回答“生成的视频像不像真实视频”。FVD、VBench、WorldArena 的视觉质量维度都属于这一层。它们能发现抖动、身份漂移、运动不平滑、背景不一致、物理不自然等问题，但不能单独证明模型能帮助机器人完成任务。

第二把是动作尺。它回答“同一历史下，换动作后未来是否合理分叉”。这一步检查动作条件是否生效：模型必须对动作敏感，而且敏感点要落在任务变量上。向左推杯子，未来应改变杯子位置、遮挡和碰撞风险；不应只是随机换背景纹理。

第三把是部署尺。它回答“预测被系统消费之后，真实决策有没有变好”。[World-in-World](https://arxiv.org/abs/2510.18135)、[WorldArena](https://arxiv.org/abs/2602.08971)、[WorldArena 2.0](https://arxiv.org/abs/2605.17912) 和 [WoW-World-Eval](https://arxiv.org/abs/2601.04137) 这类新基准的重要变化，就在于把视觉质量放回功能用途里看：作为 data engine 是否提升 policy，作为 evaluator 是否相关，作为 planner 是否在闭环里提高成功率。

截至 2026-06-03，根据上述基准公开的评测设计可以得到一个趋势判断：世界模型评测前沿正从 `vision-only + offline score` 扩展到 `multimodal + interactive utility + cross-platform`。这是对公开材料的趋势归纳，不代表对各基准结果做过独立复现。

## Open-loop 只看漂移

Open-loop 评测是在真实历史之后，让模型自己向前滚一段。它适合大规模离线比较，也适合找 horizon 变长后的误差漂移。

一步预测损失可以写成：

$$
\mathcal L_{1\text{-step}}
=
\mathbb E[\ell(\hat o_{t+1}, o_{t+1})]
$$

这里 \(\hat o_{t+1}\) 是下一步预测，\(o_{t+1}\) 是真实下一步，\(\ell\) 可以是像素、latent、状态变量或任务变量上的误差。这行公式只问“下一步像不像”，因此很容易高估能力。

多步 rollout 更接近使用方式：

$$
\mathcal L_{\text{rollout}}(H)
=
\frac{1}{H}\sum_{h=1}^{H}
\mathbb E[\ell(\hat o_{t+h}, o_{t+h})]
$$

这里 \(H\) 是预测步数。随着 \(H\) 增大，如果误差快速上升，说明模型只能做局部拟合，不能稳定推演。对机器人和自动驾驶来说，这类漂移会把 planner 带到训练分布之外。

但即使多步 open-loop 很好，也只能说明“沿着真实历史继续预测”还可以。它还没有回答“换动作后会怎样”，也没有回答“planner 会不会利用模型漏洞”。

## 反事实动作是最低门槛

动作条件世界模型必须通过一个最小测试：固定同一段历史，替换动作序列，未来是否按任务逻辑分叉。

可以把动作敏感性写成：

$$
\Delta_{\text{act}}
=
d\left(
\hat y_{t+1:t+H}^{(a)},
\hat y_{t+1:t+H}^{(a')}
\right)
$$

这里 \(a\) 和 \(a'\) 是两组候选动作，\(\hat y\) 可以是未来状态、风险、reward、事件或视频 latent。需要检查的是变化是否落在正确变量上，\(\Delta_{\text{act}}\) 并非越大越好。

反事实测试至少要覆盖三类样本。

| 样本 | 看什么 | 常见坏相 |
| --- | --- | --- |
| 普通动作 | 日常动作后果是否合理 | 换动作后未来几乎不变 |
| 边界动作 | 两个候选动作接近时能否正确排序 | 把更危险动作排到前面 |
| hard negative | 碰撞、滑落、遮挡、错误抓取是否被识别 | 生成一条连贯但不可执行的计划 |

CausalVQA、V-JEPA 2 的物理推理 benchmark、WorldArena 的 controllability / action following 维度，都在往这个方向靠：把“看见了什么”推进到“如果这样做，会发生什么”。

## Closed-loop 看模型会不会骗系统

Closed-loop 评测更严格，因为模型的预测会影响下一步动作，下一步动作又改变后续输入。很多模型 open-loop 看起来不错，closed-loop 失败，是因为 planner 会主动搜索模型里的高回报漏洞。

闭环指标至少要和系统动作绑定：

| 指标 | 读法 |
| --- | --- |
| task success | 真实或仿真执行是否完成任务 |
| regret | 模型选的动作和更好动作之间差多少 |
| candidate ranking agreement | 世界模型排序是否接近真实执行排序 |
| intervention / takeover | 安全层或人工是否频繁接管 |
| cost per success | 每个成功任务消耗多少 rollout、延迟和显存 |
| failure replay pass rate | 已知失败回放后是否不再犯同类错 |

World-in-World 的提醒很直接：视觉质量本身不保证任务成功，可控性更关键。WorldArena 进一步把世界模型放进 data engine、policy evaluator 和 action planner 三种下游任务里看。这个思路值得保留：先定义消费方，再定义指标。

## 风险要按场景桶校准

平均误差最容易掩盖高风险失败。真实系统里，一个低置信但会回退的模型，通常比一个平均分高却在 near-miss 上过度自信的模型更可用。

风险评测建议按桶报告，而不是只报总分：

| 风险桶 | 需要记录什么 |
| --- | --- |
| collision | 最小距离、接触前风险分、planner 是否继续推进 |
| slip / drop | 抓取姿态、接触状态、释放时刻、真实结果 |
| occlusion | 被遮挡对象是否仍保留在 state 中 |
| unreachable | IK / collision checker 是否提前拒绝 |
| human takeover | 接管前模型预测、候选动作和回退触发条件 |
| long-horizon drift | 任务阶段、目标物身份和空间布局何时开始漂 |

校准可以简单地按分数桶看。比如模型把一组样本预测成 80% 成功，那么真实成功率是否接近 80%。如果高置信桶里隐藏大量 near-miss，部署时就需要提高不确定性惩罚、补 hard set，或者让 safety layer 接管。

## 失败回放比总分更重要

一个世界模型失败日志应能回答“为什么选错动作”。例如移动操作机器人要把杯子放到上层柜子，有两个候选动作：先抬手再伸过去，或者先靠近再抬手。真实结果是前者碰到架沿，后者成功；模型却把前者判成低风险。

这条样本不应只记成“失败一次”，而要拆成四个判定：状态表示是否保留杯子、柜门和架沿位置；反事实排序是否把安全动作排在前面；风险分数是否把碰撞动作放进高风险桶；planner 依赖该预测时是否被安全层截停。

如果模型生成的视频更清晰，却仍把危险动作排在安全动作前，它没有通过世界模型评测。世界模型的目标不是生成更好看的失败，而是帮助系统避开失败。

![DreamZero failure case 原论文图](../assets/images/paper-deep-dives/world-models/dreamzero/figure-16-failure-case.png){ width="920" }

<small>图源：[World Action Models are Zero-shot Policies，Figure 16](https://arxiv.org/abs/2602.15922)。原图展示 video-action model 的失败案例。读图重点：WAM 的危险在于错误未来和错误动作可能互相支持，看起来像一条连贯计划；评测必须检查真实观测刷新后能否纠正，以及 safety layer 能否提前截停。</small>

## 失效模式按修复路径分类

failure mode 最好根据“下一步怎么修”来分类，只写现象无法指导后续修改。

| 失效类型 | 现象 | 更可能的修复方向 |
| --- | --- | --- |
| 表示失败 | 小物体丢失、遮挡后失忆、接触状态不可读 | 改 tokenizer / latent、加关键状态标签或辅助头 |
| 动力学失败 | 短期准，长期漂，多主体交互崩 | 多步训练、latent rollout 正则、闭环失败回放 |
| 动作条件失败 | 换动作后未来几乎不变 | 增加反事实动作数据、改动作注入方式 |
| 未来平均化 | 多种可能被洗成均值 | 分支采样、随机 latent、扩散或离散 token 生成 |
| 规划漏洞 | planner 找到模型里高回报但现实坏的动作 | adversarial planning test、不确定性惩罚、真实回放 |
| 风险失准 | 高置信预测里藏着 near-miss | calibration、hard set、保守回退阈值 |
| 部署失败 | 太慢、不可监控、不能回退 | 分层使用、低频分析、fallback、资源预算 |

这个表的意义是把评测和下一轮训练连接起来。否则世界模型报告会停在“某指标下降”，却不知道是表示、动力学、动作、风险还是系统预算出了问题。

## 发布门槛怎么写

发布门槛应由消费方决定，不能直接照搬论文榜单。

| 给谁用 | 最小发布门槛 |
| --- | --- |
| planner | 候选动作排序提升，closed-loop success 不退化，cost per success 可接受 |
| risk gate | collision / near-miss false negative 不增加，高风险桶校准不过度自信 |
| data engine | 合成样本进入训练后改善目标失败桶，不能污染主分布 |
| policy evaluator | 世界模型评估分与真实或高可信仿真成功率有稳定相关 |
| 人审工具 | 反事实未来可解释，失败样本能回放到动作和状态变量 |

一个可执行的门槛可以这样写：目标任务桶的 counterfactual ranking 至少提升；near-miss、collision、human takeover 任一高风险桶不能退化；主分布成功率不能明显下降；所有新增失败必须能定位到表示、动作条件、风险或系统预算中的至少一类。

最终判断很简单：如果世界模型不能在闭环里提升动作选择、风险识别或数据回流，它还只是预测模型或生成模型；只有当它改变系统决策质量时，才算通过验收。

## 自测


先独立回答下面三个迁移题，不要只复述前文结论：

1. 仓储机械臂的世界模型生成质量提高了，但候选动作排序没有变化，碰撞漏报还在增加。它能否作为 planner、risk gate 或 data engine 发布？三种消费方分别还缺什么证据？
2. 两个模型的 open-loop 误差曲线相近：一个模型换动作后只改变背景纹理，另一个会改变目标位置和接触结果。你会怎样设计 counterfactual hard set 和指标，把这两种行为区分开？
3. 一个 policy evaluator 在常见任务上与真实成功率相关，却持续高估遮挡和 near-miss 桶。你会怎样写发布门槛、回退条件和下一轮数据回流规则？
## 外部材料

- [WorldArena 2.0](https://arxiv.org/abs/2605.17912)：理解最新 embodied world model 评测为什么从视觉扩展到触觉、交互式 RL 环境和多平台。
- [World-in-World](https://arxiv.org/abs/2510.18135)：理解 closed-loop world model benchmark 为什么要把任务成功率放到核心位置。
- [WorldArena](https://arxiv.org/abs/2602.08971)：理解感知质量、功能用途和人类偏好如何组合成世界模型评测框架。
- [VBench](https://arxiv.org/abs/2311.17982)：理解视频生成评测维度；使用时要保留“视频质量不等于动作 utility”的边界。
- [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599)：理解风险分数为什么需要校准，而不是只看分类或预测准确率。
