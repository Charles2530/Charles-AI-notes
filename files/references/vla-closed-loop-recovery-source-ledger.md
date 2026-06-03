# VLA 闭环恢复与失败分析来源台账

这份台账服务于 [VLA 闭环恢复与失败分析](../vla/closed-loop-recovery-and-failure-analysis.md)。正文不把恢复写成“再规划一下”，而是围绕检测、归因、恢复、验证和数据回流讲：失败怎样被发现，失败桶怎样帮助定位问题，恢复层级怎样控制风险，episode 日志怎样进入 hard set 和训练回流。

## 使用方式

第一类来源是顺序决策和分布偏移。DAgger、VLA 数据台账和 benchmark 台账用于说明：学习器自己的动作会制造新状态，如果训练数据没有失败和恢复片段，部署时很容易偏一步后滚雪球。

第二类来源是真实机器人 VLA 失败边界。RT-2、RT-1、Open X-Embodiment、π0.5、OpenVLA 和相关深读页用于提醒：语义泛化不等于动作成功，动作成功不等于失败后能恢复。正文使用 RT-2 失败图时，只把它作为 failure dynamics 直觉，不外推成所有 VLA 失败类型。

第三类来源是技能选择、安全和恢复门禁。SayCan、部署安全页、Sim2Real 台账和世界模型评测台账用于支撑 affordance、risk checker、human takeover、failure replay 和安全集的讲法。正文要把“能继续尝试”和“应该继续尝试”分开。

第四类来源是中文技术长文和公众号生态。Datawhale Every-Embodied、机器之心、量子位、智源、PaperWeekly 等材料用于学习中文讲法：先从具体失败链条切入，再拆检测、归因、恢复、评测和数据回流。事实、图片、成功率和安全结论仍回到论文、项目页、官方博客或代码仓库。

## 写作边界

不要把闭环恢复写成无限重试。恢复必须有 budget、升级规则和安全退出。

不要把失败检测完全交给主策略。success checker、risk checker 和 progress checker 应作为可复盘信号进入日志。

不要把最终成功率当作恢复质量。若最终成功依赖危险动作、过度重试或人工接管，部署价值有限。

不要把失败轨迹删掉。失败、near-failure、恢复失败和人工接管是训练风险头、恢复策略和 hard set 的关键资产。

## 图片使用

本页没有使用 image2 生成图片。正文只使用已有 RT-2 失败案例图，作为失败动力学和真实机器人边界的说明。

| 图片 | 文件 | 原始来源 | 正文用途 | 使用边界 |
| --- | --- | --- | --- | --- |
| RT-2 failure dynamics | `../assets/images/paper-deep-dives/embodied-ai/rt2/figure-9-failure-dynamics.png` | [RT-2](https://arxiv.org/abs/2307.15818) Figure 9 | 说明 web-scale VLM 语义迁移仍会在 unseen object dynamics 和真实接触中失败。 | 不把单张图外推成所有 VLA 失败模式；它只服务 failure bucket 和恢复数据的讲法。 |

## 来源

| 来源 | 类型 | 本站使用方式 | 关联问题 |
| --- | --- | --- | --- |
| [DAgger](https://arxiv.org/abs/1011.0686) | 论文 | 支撑学习器自己造成的状态分布会偏离专家分布，失败/恢复数据需要进入训练覆盖。 | 分布偏移 |
| [RT-2](https://arxiv.org/abs/2307.15818) | 论文 / 图源 | 使用失败案例图说明语义泛化和动作执行边界不同。 | VLA 失败 |
| [RT-2 blog](https://deepmind.google/blog/rt-2-new-model-translates-vision-and-language-into-action/) | 官方博客 | 学习如何把 VLM 知识迁移到机器人动作的能力和边界讲给读者。 | 讲法参考 |
| [SayCan](https://say-can.github.io/) | 项目页 / 论文 | 支撑语言技能选择必须结合 affordance / value function，不能只靠语言合理性。 | 技能恢复 |
| [π0.5](https://arxiv.org/abs/2504.16054) | 论文 / 技术报告 | 支撑真实家庭任务、high-level subtask、数据消融和部署边界。 | open-world VLA |
| [OpenVLA](https://openvla.github.io/) | 项目页 / 论文 | 作为开源 VLA 对照，提醒微调和评测仍要看动作接口和闭环。 | 开源 VLA |
| [VLA 数据与策略学习来源台账](vla-data-policy-learning-source-ledger.md) | 本站台账 | 复用行为克隆、失败样本、恢复轨迹和 covariate shift 边界。 | policy learning |
| [VLA 评测与数据引擎来源台账](vla-benchmark-data-engine-source-ledger.md) | 本站台账 | 复用首次/最终成功率、恢复率、hard set、replay set 和数据回流。 | benchmark |
| [Sim2Real 与数据引擎来源台账](sim2real-data-engine-source-ledger.md) | 本站台账 | 复用真实日志、人工接管、hard set 和安全门禁的组织方式。 | 数据闭环 |
| [世界模型评测来源台账](world-model-evaluation-source-ledger.md) | 本站台账 | 复用 action sensitivity、failure replay、risk calibration 和 closed-loop success 证据标准。 | 评测证据 |
| [Datawhale Every-Embodied](https://github.com/datawhalechina/every-embodied) | 中文开源教程 / 公众号生态 | 学习具身智能中文教程如何把数据、仿真、复现和部署路线串成学习路径。 | 中文讲法 |
| [外部精读来源台账](external-reading-ledger.md) | 本站台账 | 汇总机器之心、量子位、智源、PaperWeekly、Datawhale 等中文技术材料的讲法边界。 | 讲法参考 |

## 后续维护

新增恢复相关论文或系统时，先判断它改善的是 detection、attribution、recovery policy、safety gate、human takeover、failure replay，还是数据回流。若来源只报告最终成功率，缺少失败桶、恢复耗时、危险恢复率、人工接管和 hard set 回归，正文不能把它写成部署级恢复能力。
