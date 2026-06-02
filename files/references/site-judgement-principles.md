# 本站判断原则

这页说明本站如何判断一项技术是否真的服务**世界模型高效训练**。它不替代 [全站证据与复现状态标准](evidence-and-reproducibility-standard.md)、[Claim Ledger](claim-ledger.md) 或 [全站效率技术覆盖矩阵](site-efficiency-coverage-matrix.md)，而是把这些页面背后的判断口径写成更短的原则。


## 五条原则

| 原则 | 写作时必须回答 | 常见误区 |
| --- | --- | --- |
| 高效是成本向量，不是形容词 | 省的是数据、token、显存、通信、rollout 延迟、采样步数、验证预算，还是部署成本 | 只写“更快/更省”，不说明成本项 |
| 表征压缩必须保留决策变量 | latent/token/geometry 是否保留动作敏感性、接触、遮挡、风险和目标状态 | token 更少就默认控制更好 |
| 系统吞吐不等于任务质量 | throughput、latency、MFU、KV memory 是否和质量回归同表报告 | kernel 快就默认模型更强 |
| Open-loop 指标不能替代 closed-loop utility | FVD、VBench、loss、静态 benchmark 是否有 action sensitivity、closed-loop success 或 failure replay 支撑 | 视频自然度直接外推为规划收益 |
| 没有第三方证据时不写独立复现 | 是否只有论文、官方代码、官方 demo、toy fixture 或本站推断 | 把 official repo 当成 independent reproduction |

## 知识讲解原则

本站的文章首先要讲清知识，而不是维护目录。参考 [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)、[colah 的 LSTM 讲解](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)、[Lil'Log](https://lilianweng.github.io/)、[Hugging Face 技术文档](https://huggingface.co/docs/transformers/main/quantization/concept_guide)、[BAIR Blog](https://bair.berkeley.edu/blog/2023/10/16/p3o/) 和 OneFlow/智源转载的 [全栈 Transformer 推理优化](https://hub.baai.ac.cn/view/33449)，本站后续写法按下面规则收敛。

| 写法 | 要做到什么 | 避免什么 |
| --- | --- | --- |
| 先给核心困惑 | 用一句话说明读者为什么会卡住，例如“attention 为什么不是查表”“Bellman 为什么能把未来拆回来” | 开头先放导航、阅读建议或大段背景 |
| 从具体例子进入 | 先用一个小场景固定变量，再抽象成公式和系统结构 | 先堆术语，再让读者自己找例子 |
| 公式解释服务推理 | 解释每个公式解决的问题、输入输出、哪一步在近似、哪里会出错 | 自动符号表、泛泛写“变量由上下文决定” |
| 图要进入论证 | 图注写清图源、原图表达什么、本站用它证明或区分什么 | 图只当插图，或把解释塞进一堆提示框 |
| 讲边界和反例 | 每个关键方法写清何时失效、指标会怎样骗你、不能外推到哪里 | 只有优点，没有失败机制 |
| 允许外部精读 | 正文需要处直接链接论文、官方博客、工程博客或高质量公众号转载 | 为了互链而互链 |

好文章的共同点不是“更长”，而是每一段都回答一个小问题：为什么需要这个概念，它改变了什么判断，读者拿它能分清哪两个容易混淆的东西。

## 证据强度

本站把证据分成不同用途。`Paper Result` 能说明论文设置下的方法有效，但不能自动跨任务；`Ablation` 能说明某个模块在该设置下有贡献，但不能把幅度照搬；`System Throughput` 能说明运行成本下降，但不证明动作质量或安全；`Closed-loop` 最接近世界模型主线，但仍受任务、平台和数据分布限制；`Official Demo` 只能展示能力形态；`Toy Fixture` 只能证明证据链和指标脚本写法；`Site Inference` 只是工程假设。

因此，每个关键 claim 至少要写清三件事：

| 字段 | 最小要求 | 例子 |
| --- | --- | --- |
| Direct Source | 论文、报告、官方仓库、项目页或本站 fixture | arXiv、official PDF、GitHub、HF model card、mini-chain |
| Evidence Type | 使用固定证据标签 | Paper Result / System Throughput / Closed-loop |
| Cannot Prove | 明确不能外推到哪里 | 不能证明真实机器人安全、跨分布泛化或闭环收益 |

## 成熟度分层

本站不会把所有新论文放在同一采用风险里。一个方向是否值得进入实验，除了看分数，也要看它处在什么成熟度层级。

| 成熟度 | 适合怎么用 | 最容易误用的地方 |
| --- | --- | --- |
| 经典已验证 | 作为 baseline、概念锚点或复现实验参照 | 忽略新任务的数据分布和硬件约束 |
| 工程常用 | 作为训练、推理、量化、算子的候选实现 | 只看 microbench 或吞吐，不看质量回归 |
| 前沿待复现 | 作为研究假设、消融对象或小规模验证路线 | 把论文主表直接当成生产收益 |
| 官方展示 | 作为能力形态、交互方式或系统界面参考 | 把 demo 当成平均成功率 |
| 本站推断 | 作为阅读路径、实验设计和 claim 审计框架 | 把综合判断写成论文已证明 |

## 世界模型主线的验收顺序

一个世界模型相关页面最好按下面顺序收口：

1. 先说明模型预测什么：pixel、latent、embedding、reward、done、risk、action 还是 video rollout。
2. 再说明动作如何进入模型：没有 action 的 masked/JEPA 表征不能直接等同于可规划世界模型。
3. 接着说明效率来源：latent 压缩、masked objective、imagination、少步蒸馏、KV/cache、低比特或系统调度。
4. 然后给出最小验收：action sensitivity、closed-loop success、cost per success、risk ECE、latency、显存或吞吐。
5. 最后写清不能证明什么：demo、open-loop 视频、系统吞吐、toy fixture 都不能替代真实闭环。

## 写作门禁

| 如果页面声称 | 必须补的门禁 | 不能只靠 |
| --- | --- | --- |
| 数据更省 | 学习曲线、样本预算、数据来源和失败桶 | 单个最终分数 |
| 训练更省 | MFU/step time、显存、通信、checkpoint 和收敛指标 | 单张吞吐图 |
| 推理更省 | TTFT/TPOT、P95/P99、KV memory、质量回归 | 平均延迟 |
| 表征更好 | downstream、action sensitivity、遮挡/接触/risk case | embedding 可视化 |
| 闭环更强 | closed-loop success、失败 replay、恢复策略和安全门禁 | open-loop loss 或视频自然度 |
| 部署更稳 | SLO、fallback、监控、回滚和成本 per success | 离线 benchmark |

读者可以把这页当成一个审稿清单：如果某个段落只剩下“方法很强、速度很快、demo 很自然”，它还没有达到本站的证据标准。

## 最佳反例

最能检验这套原则的反例，是一个视频世界模型 demo 看起来非常自然，系统吞吐也很高，但固定当前观测、替换候选动作后，未来 latent、风险分数和成功排序几乎不变。这个系统可能有优秀的视频先验和工程实现，却仍不能支撑“可用于规划”的 claim；它必须补 action sensitivity、closed-loop success、failure replay 和 cost per success。

## 和其它全站入口的关系

| 入口 | 负责什么 | 不负责什么 |
| --- | --- | --- |
| 本页 | 给出判断原则和写作门禁 | 不逐条维护论文证据 |
| [证据与复现状态标准](evidence-and-reproducibility-standard.md) | 规定 `Evidence Snapshot`、事实版本和标签写法 | 不判断具体路线优先级 |
| [全站效率矩阵](site-efficiency-coverage-matrix.md) | 按瓶颈把读者导向专题和案例 | 不替代原论文结果 |
| [Claim Ledger](claim-ledger.md) | 维护世界模型高效训练主线的关键 claim 总账 | 不覆盖全站所有基础知识 |

实际写页面时，顺序可以很简单：先用本页判断 claim 是否值得写，再用证据标准把来源和复现状态写清，最后把它接回效率矩阵或 Claim Ledger。这样能避免两种常见问题：一是把论文摘要改写成“很厉害”的短评；二是把系统工程收益、模型能力收益和闭环决策收益写成同一种证据。
