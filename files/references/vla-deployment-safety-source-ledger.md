# VLA 部署与安全来源台账

这份台账服务于 [VLA 部署与安全](../vla/deployment-and-safety.md)。正文不把部署写成“模型推理加安全裁剪”，而是围绕 ODD、延迟预算、动作适配、controller、safety filter、状态机、人工接管、监控和数据回流讲：模型如何被放进一个可停、可追溯、可审计的系统。

## 使用方式

第一类来源是 VLA 和机器人策略。SayCan、RT-2、OpenVLA、π0.5、VLA 动作接口台账和闭环恢复台账用于说明：高层语义能力必须被 affordance、动作接口、低层控制器和安全层约束。

第二类来源是机器人安全标准和安全案例。ISO 10218-1:2025、ISO 10218-2:2025、ISO/TS 15066:2016、NIST autonomous systems 和具身部署安全页用于提醒：机器人安全需要 ODD、风险评估、人机协作条件、速度/力限制、接管和审计。本站只作学习参考，不替代正式合规评估。

第三类来源是工程部署与数据闭环。VLA 评测台账、Sim2Real 数据引擎台账和世界模型评测台账用于支撑 shadow、gated rollout、hard set、near-miss、safety reject、human takeover 和 rollback 的讲法。

第四类来源是中文技术长文和公众号生态。Datawhale Every-Embodied、机器之心、量子位、智源、PaperWeekly 等材料用于学习中文讲法：从真实场景风险切入，再拆 ODD、控制链路、安全壳、接管和审计。事实、标准条款、成功率和安全结论仍回到官方标准、论文、项目页或代码仓库。

## 写作边界

不要把本页写成合规指南。机器人安全合规取决于地区、行业、硬件、场景和集成方式；正文只提供知识库层面的技术判断框架。

不要把 safety filter 写成动作裁剪。安全层必须能拒绝、降级、急停和记录；静默裁剪可能掩盖系统性风险。

不要只看平均延迟。P95/P99、jitter、controller saturation 和过期观测更接近真实部署风险。

不要把高层语义能力等同于可部署能力。VLA 会理解任务，不代表它知道当前机器人是否能安全执行。

## 图片使用

本页没有使用 image2 生成图片。正文只使用已有 SayCan 方法图，作为高层语言建议需要 affordance 约束的直觉入口。

| 图片 | 文件 | 原始来源 | 正文用途 | 使用边界 |
| --- | --- | --- | --- | --- |
| SayCan grounding language with affordances | `../assets/images/paper-figures/vla/saycan-intro.png` | [Do As I Can, Not As I Say / SayCan](https://arxiv.org/abs/2204.01691) Figure 1 | 说明语言建议需要当前场景的可执行性约束。 | 不把 SayCan 写成完整部署安全方案；它只服务 affordance 和技能约束的讲法。 |

## 来源

| 来源 | 类型 | 本站使用方式 | 关联问题 |
| --- | --- | --- | --- |
| [SayCan](https://arxiv.org/abs/2204.01691) | 论文 / 图源 | 使用 Figure 1 说明语言建议必须被 affordance / value function 接地。 | skill safety |
| [RT-2 blog](https://deepmind.google/blog/rt-2-new-model-translates-vision-and-language-into-action/) | 官方博客 | 学习 web-scale VLM 知识接到机器人动作后的能力叙事和边界。 | VLA deployment |
| [RT-2 paper](https://arxiv.org/abs/2307.15818) | 论文 | 核对 action token、实时推理、seen/unseen/emergent 评测和失败边界。 | VLA evidence |
| [OpenVLA](https://openvla.github.io/) | 项目页 / 论文 | 作为开源 VLA 对照，提醒微调和部署要看动作接口和真实任务。 | 开源 VLA |
| [π0.5](https://arxiv.org/abs/2504.16054) | 论文 / 技术报告 | 支撑 open-world VLA、真实家庭任务、high-level subtask 和部署边界。 | open-world VLA |
| [ISO 10218-1:2025](https://www.iso.org/standard/73933.html) | 官方标准页 | 提醒工业机器人安全要求、风险降低和设计边界；不复述条款。 | robot safety |
| [ISO/TS 15066:2016](https://www.iso.org/standard/62996.html) | 官方标准页 | 提醒协作机器人和人机协作条件需要专门安全要求。 | collaborative robots |
| [NIST Autonomous Systems](https://www.nist.gov/autonomous-systems) | 官方研究入口 | 作为 autonomous systems trust、safety 和 measurement 的公共机构背景。 | safety governance |
| [具身部署模式与安全案例](../embodied-ai/deployment-patterns-and-safety-cases.md) | 站内页面 | 复用 ODD、安全案例、人工接管和事故复盘框架。 | safety case |
| [VLA 闭环恢复来源台账](vla-closed-loop-recovery-source-ledger.md) | 本站台账 | 复用 failure bucket、恢复预算、安全退出和人工接管边界。 | recovery |
| [VLA 评测与数据引擎来源台账](vla-benchmark-data-engine-source-ledger.md) | 本站台账 | 复用 hard set、near-miss、safety reject、数据回流和门禁。 | benchmark |
| [Datawhale Every-Embodied](https://github.com/datawhalechina/every-embodied) | 中文开源教程 / 公众号生态 | 学习具身智能中文教程如何把模型、仿真、数据和部署路线串成学习路径。 | 中文讲法 |
| [外部精读来源台账](external-reading-ledger.md) | 本站台账 | 汇总机器之心、量子位、智源、PaperWeekly、Datawhale 等中文技术材料的讲法边界。 | 讲法参考 |

## 后续维护

新增部署、安全或机器人 foundation model 工作时，先判断它证明的是哪一层：ODD 设计、低延迟推理、控制器稳定、安全过滤、human takeover、事故复盘，还是合规评估。若来源只给 demo 或成功率，缺少 P95/P99、near-miss、safety reject、接管和日志审计，正文不能把它写成可上线证据。
