# RAG、Agent 与评估 QA

这一页回答 RAG、embedding、rerank、Agent、tool calling、LLM-as-judge、幻觉、安全和在线评估中的高频问题。问题刻意覆盖面试里容易忽略的坑：检索和生成混淆、格式正确和事实正确混淆、demo 和线上可靠性混淆。

## RAG 与检索

### Q：RAG 解决的核心问题是什么？它不能解决什么？

**面试回答。** RAG 的核心是把外部知识检索结果放进模型上下文，让模型在生成时参考可更新、可追溯的资料。它主要解决三类问题：模型参数知识过时、私有知识不在预训练语料里、长文档无法完全塞进上下文。完整链路通常是：query 改写或解析、检索、rerank、上下文拼接、生成、引用或校验。

**追问展开。** 正常面试里要强调 RAG 改变的是 \(p_\theta(y\mid x,c)\) 里的条件 \(c\)，不是改变模型参数。检索文档质量越高，模型越有机会生成有根据的答案；但模型仍可能忽略证据或错误综合。  
**易错点 / 边界。** 最容易答错的是把 RAG 说成“解决幻觉”。RAG 只能降低知识缺失导致的幻觉，不能保证检索正确、引用正确或推理正确。  
**继续读。** [RAG、Agent 与长上下文系统](../inference/rag-agents-and-long-context-systems.md) / [prompt、CoT 与 RAG 基础](../foundations/prompt-engineering-cot-and-rag-basics.md)

### Q：文档切块为什么会影响 RAG 效果？

**面试回答。** Chunk 是检索的最小证据单位。太短会丢上下文，比如只召回一个定义但丢掉适用条件；太长会引入噪声，占用窗口，并降低 embedding 表示的聚焦程度。好的切块要尽量保留标题、层级、表格、代码块、引用和语义边界。

**追问展开。** 面试可以举例：法律条文要保留条款层级，代码文档要保留函数签名和说明，论文 PDF 要保留图表 caption 和段落引用。切块之后还要给 chunk 加 metadata，例如来源、更新时间、章节、权限和实体。  
**易错点 / 边界。** 不要背一个固定 chunk size。真实系统里 chunk size、overlap 和切分策略要靠检索评测和最终答案质量调。  
**继续读。** [RAG、Agent 与长上下文系统](../inference/rag-agents-and-long-context-systems.md) / [数据与数据集基础](../foundations/data-and-dataset-basics.md)

### Q：Embedding 检索为什么可能召回错误内容？

**面试回答。** Embedding 检索用向量相似度近似语义相关性，常用 cosine 或 dot product：

\[
\cos(q,d)=\frac{q^\top d}{\|q\|\|d\|}
\]

它擅长语义近邻，但对数字、否定、时间、实体别名、版本号和精确过滤条件不一定可靠。  
**追问展开。** 例如“2024 年之后的政策”和“2024 年之前的政策”在语义上很近，但答案方向相反；“Python 3.11”这种版本号也可能被 embedding 弱化。  
**易错点 / 边界。** 不要把向量相似度当事实相关性。严肃系统通常要加 metadata filter、BM25、rerank 和答案引用校验。  
**继续读。** [RAG、Agent 与长上下文系统](../inference/rag-agents-and-long-context-systems.md) / [预训练目标与表示学习](../foundations/pretraining-objectives-and-representation-learning.md)

### Q：BM25 和向量检索为什么常做混合检索？

**面试回答。** BM25 擅长精确词匹配，尤其是实体名、编号、错误码、API 名、版本号；向量检索擅长语义改写和同义表达。混合检索把两类召回合并，再通过融合或 rerank 得到更稳的候选集合。

**追问展开。** 面试里可以说常见做法是 sparse + dense recall，然后用 RRF 或 reranker 融合。这样能覆盖“用户说法和文档说法不同”的问题，也能保留精确匹配能力。  
**易错点 / 边界。** 只做向量检索很容易漏精确约束；只做 BM25 又容易漏语义改写。融合权重必须用真实查询集调。  
**继续读。** [RAG、Agent 与长上下文系统](../inference/rag-agents-and-long-context-systems.md) / [观测与在线评估](../inference/observability-and-online-evaluation.md)

### Q：Reranker 为什么通常放在召回后？

**面试回答。** 第一阶段召回需要快，目标是覆盖尽可能多的候选；reranker 用更强但更贵的模型对少量候选做精排，提升上下文相关性。一个典型策略是先召回 top-100，再 rerank 到 top-5 或 top-10。

**追问展开。** Reranker 可以是 cross-encoder，也可以是 LLM judge。它通常能看到 query 和文档的交互，比单独 embedding 更适合判断细粒度相关性。  
**易错点 / 边界。** Rerank 分数高不等于答案正确；它只改善证据排序，生成阶段仍可能误读或拼错。  
**继续读。** [RAG、Agent 与长上下文系统](../inference/rag-agents-and-long-context-systems.md) / [成本建模与 SLO 设计](../inference/cost-modeling-and-slo-design.md)

### Q：RAG 效果差应该先查检索还是生成？

**面试回答。** 要分阶段排查。先看相关证据是否被召回：recall@k、MRR、rerank 排名、过滤条件是否正确；再看生成是否使用证据：答案是否引用正确、是否遗漏关键段落、是否把多个证据拼错。不要直接只调 prompt。

**追问展开。** 一个实用排查表是：query 是否被错误改写，chunk 是否切坏，检索是否召回，rerank 是否排前，上下文是否超长被截断，模型是否按证据回答。  
**易错点 / 边界。** 面试里最常见错误是把最终答案错误全部归因于模型。很多 RAG 失败其实发生在检索、切块、权限过滤或上下文拼接。  
**继续读。** [观测与在线评估](../inference/observability-and-online-evaluation.md) / [评测与消融方法](../training/evaluation-and-ablation-methodology.md)

### Q：RAG 评估为什么不能只看最终答案准确率？

**面试回答。** 最终答案准确率是端到端指标，但不能定位失败原因。RAG 至少要拆成检索指标、上下文质量指标和生成指标：检索看 recall@k / MRR，rerank 看排序，生成看 groundedness、faithfulness、引用正确和最终任务成功。

**追问展开。** 如果召回没有命中，再强模型也答不好；如果召回命中但模型没用证据，那是生成或 prompt 问题；如果答案正确但引用错，事实性和可追溯性仍有风险。  
**易错点 / 边界。** 不要只拿一个 LLM-as-judge 总分判断 RAG 好坏；分阶段指标才能指导优化。  
**继续读。** [评测与消融方法](../training/evaluation-and-ablation-methodology.md) / [评测与证据链来源台账](../references/evaluation-source-ledger.md)

## Agent 与工具调用

### Q：Agent 和普通聊天机器人有什么区别？

**面试回答。** 普通聊天机器人主要生成文本；Agent 会根据目标规划步骤、调用工具、观察结果、更新状态并继续行动。核心区别是闭环：模型输出会影响外部环境，环境反馈再影响下一步决策。

**追问展开。** 一个最小 Agent loop 可以写成：observe -> plan -> act -> observe。RAG 只是读知识，Agent 还会写文件、查数据库、调用 API、执行代码或控制工作流。  
**易错点 / 边界。** 多调用几个工具不等于可靠 Agent。真正难点是状态管理、权限、安全、失败恢复和可审计 trace。  
**继续读。** [RAG、Agent 与长上下文系统](../inference/rag-agents-and-long-context-systems.md) / [观测与在线评估](../inference/observability-and-online-evaluation.md)

### Q：ReAct 的直觉是什么？

**面试回答。** ReAct 把 reasoning 和 acting 交替起来：模型先分析下一步需要什么，再调用工具或观察环境，然后基于结果继续推理。它适合需要外部信息或多步操作的问题，因为模型不必一次性凭内部知识回答。

**追问展开。** 面试时可以举例：用户问一个需要最新网页信息的问题，模型先决定搜索，再读取结果，再回答。关键是 action trace 能被审计，工具结果能进入下一轮上下文。  
**易错点 / 边界。** 显式写出“思考过程”不保证推理正确；生产系统更看重可验证的工具输入输出，而不是漂亮的 chain-of-thought 文本。  
**继续读。** [prompt、CoT 与 RAG 基础](../foundations/prompt-engineering-cot-and-rag-basics.md) / [RAG、Agent 与长上下文系统](../inference/rag-agents-and-long-context-systems.md)

### Q：Function calling 和结构化输出解决什么？又不解决什么？

**面试回答。** Function calling 和结构化输出把模型输出约束成函数参数或 JSON schema，方便系统解析和执行。它解决的是“格式可解析”和“接口可接入”的问题，尤其适合工具调用、表单、数据库查询和工作流编排。

**追问展开。** 如果 schema 要求 `{city: string, date: string}`，模型输出就更容易被程序消费。严格 schema 还能减少格式错误，但参数含义是否正确仍要校验。  
**易错点 / 边界。** 格式正确不等于事实正确，也不等于工具安全。权限、参数范围、幂等、审计和回滚仍要系统层处理。  
**继续读。** [生成与解码基础](../foundations/generation-and-decoding-basics.md) / [OpenAI Structured Outputs](https://openai.com/index/introducing-structured-outputs-in-the-api/)

### Q：Agent 记忆应该存什么？

**面试回答。** Agent 记忆应该存对未来任务有稳定价值的信息，例如用户偏好、长期项目事实、任务状态、工具结果、重要决策和已验证资料。短期上下文、长期记忆、向量知识库和结构化状态要分层管理。

**追问展开。** 面试时可以说：不是所有对话都该进长期记忆。写入前最好有筛选、摘要、冲突检测和过期机制；读取时要有检索和权限控制。  
**易错点 / 边界。** 盲目存全部历史会带来隐私、噪声、过时信息和 prompt injection 风险。记忆是系统状态，不是无限上下文垃圾桶。  
**继续读。** [RAG、Agent 与长上下文系统](../inference/rag-agents-and-long-context-systems.md) / [上下文压缩、KV 驱逐与记忆层级](../inference/context-compression-kv-eviction-and-memory-hierarchies.md)

### Q：工具调用失败时系统应该怎么做？

**面试回答。** 先分类失败：参数错误、权限错误、外部服务错误、超时、幂等冲突、模型误调用。不同失败要有不同策略：参数错误可让模型修正，权限错误要拒绝或请求确认，超时可重试，副作用工具要防重复执行。

**追问展开。** 一个可靠 Agent 要把错误信息写入 trace，让模型和系统都能知道发生了什么。高风险操作需要确认、dry run、回滚或人工审批。  
**易错点 / 边界。** 不要把所有异常都交给模型“自己想办法”。模型没有天然权限边界，系统必须提供硬约束。  
**继续读。** [观测与在线评估](../inference/observability-and-online-evaluation.md) / [推理服务系统](../inference/serving-systems.md)

### Q：多 Agent 协作一定比单 Agent 好吗？

**面试回答。** 不一定。多 Agent 可以带来分工、互审、并行探索和角色专业化，但也会带来通信成本、状态不一致、责任不清和错误互相放大。是否使用多 Agent，取决于任务是否真的可分解、接口是否清楚、评估是否可自动化。

**追问展开。** 适合多 Agent 的场景是明确分工，如检索、代码、审查、测试分别独立；不适合的是状态高度耦合、目标模糊、没有验收标准的任务。  
**易错点 / 边界。** 面试里不要把多 Agent 当高级答案。很多任务单 Agent 加工具和检查点更稳定。  
**继续读。** [RAG、Agent 与长上下文系统](../inference/rag-agents-and-long-context-systems.md) / [评测与消融方法](../training/evaluation-and-ablation-methodology.md)

## 评估、幻觉与安全

### Q：幻觉的根本原因是什么？

**面试回答。** LLM 生成的是条件概率高的文本，不是从事实数据库中直接读取答案。当训练知识缺失、上下文证据不足、检索错误、问题有歧义或要求超出能力时，模型仍可能生成流畅但错误的内容。

\[
y \sim p_\theta(y\mid x,c)
\]

其中 \(c\) 可以是 prompt、检索文档或工具结果。  
**追问展开。** 降低幻觉要同时改条件和验证：RAG 提供证据，工具提供实时结果，引用和校验器约束输出，评估集发现失败模式。  
**易错点 / 边界。** 幻觉不是只靠“更好 prompt”解决；事实错误、引用错误、推理错误、工具错误和过时知识要分开处理。  
**继续读。** [RAG、Agent 与长上下文系统](../inference/rag-agents-and-long-context-systems.md) / [观测与在线评估](../inference/observability-and-online-evaluation.md)

### Q：LLM-as-judge 能替代人工评估吗？

**面试回答。** LLM-as-judge 可以快速评估大量样本，适合回归测试、初筛、解释偏好和低风险比较。它的优点是便宜、快、可扩展；缺点是 judge 本身也有偏差和不稳定。

**追问展开。** 常见偏差包括位置偏好、长度偏好、模型家族偏好、过度相信流畅文本、对细节事实不敏感。更稳的做法是用人工标注、规则指标、真实任务结果对 judge 做校准。  
**易错点 / 边界。** LLM-as-judge 是评估工具，不是地面真值。高风险评测不能完全交给另一个模型。  
**继续读。** [观测与在线评估](../inference/observability-and-online-evaluation.md) / [评测与证据链来源台账](../references/evaluation-source-ledger.md)

### Q：离线评测和在线评测的差别是什么？

**面试回答。** 离线评测用固定数据集比较模型或系统版本，速度快、可重复，适合开发迭代；在线评测观察真实用户、真实流量和真实成本，能看到满意度、错误率、延迟、安全事件和业务目标。两者互补。

**追问展开。** 一个版本离线分数高，但上线可能因为流量分布不同、延迟更高、工具失败或用户问题更复杂而表现差。线上灰度和监控是最后一道验证。  
**易错点 / 边界。** 离线通过不等于上线安全；在线指标也受产品策略和用户行为影响，不能孤立解读。  
**继续读。** [观测与在线评估](../inference/observability-and-online-evaluation.md) / [成本建模与 SLO 设计](../inference/cost-modeling-and-slo-design.md)

### Q：安全评估为什么不能只测拒答率？

**面试回答。** 拒答率只说明模型拒绝了多少请求，不说明它是否误拒正常请求、是否能抵抗 prompt injection、是否泄露隐私、是否产生危险工具动作。安全评估要按风险分类看：内容安全、隐私、权限、工具副作用、越狱和多模态风险。

**追问展开。** 面试时可以说安全有 precision/recall 取舍：拒答太少会放过危险请求，拒答太多会伤害帮助性。真实系统需要分级策略、审计日志和人工升级。  
**易错点 / 边界。** 不要把“模型会拒答”当成安全系统。工具权限、数据隔离和输出验证是系统职责。  
**继续读。** [观测与在线评估](../inference/observability-and-online-evaluation.md) / [GPT-4o System Card：Omni 安全与多模态评测](../paper-deep-dives/technical-reports/gpt-4o-system-card.md)

### Q：Prompt injection 为什么对 RAG/Agent 特别危险？

**面试回答。** RAG 和 Agent 会把外部文本、网页、文档或工具结果放进模型上下文。如果这些外部内容里包含恶意指令，模型可能把它们当成高优先级指令执行，从而泄露数据、忽略系统规则或调用危险工具。

**追问展开。** 防护不能只靠“告诉模型不要听外部指令”。更稳的做法包括上下文分区、来源标注、工具权限最小化、参数验证、输出过滤和高风险操作确认。  
**易错点 / 边界。** 最大误区是把 prompt injection 当成普通内容过滤问题；它本质是指令层级和权限边界问题。  
**继续读。** [RAG、Agent 与长上下文系统](../inference/rag-agents-and-long-context-systems.md) / [推理服务系统](../inference/serving-systems.md)

### Q：如何设计一个最小可用的 RAG/Agent 评测集？

**面试回答。** 最小评测集要覆盖真实用户问题、标准答案或证据、难负例、无答案问题、过时知识、权限样本、工具失败和安全样本。每个样本最好能标注失败阶段：检索、rerank、生成、工具、权限或最终答案。

**追问展开。** 面试回答可以给一个结构：`query`、`gold evidence`、`expected answer`、`allowed tools`、`risk tag`、`failure criteria`。这样评测不仅给分，还能定位系统该修哪里。  
**易错点 / 边界。** 不要只收集模型答得出来的问题；评测集必须包含无答案、冲突证据和失败路径。  
**继续读。** [评测与消融方法](../training/evaluation-and-ablation-methodology.md) / [观测与在线评估](../inference/observability-and-online-evaluation.md)

### Q：如何判断一个 AI 应用系统是否真的变好了？

**面试回答。** 要同时看任务成功率、事实错误率、用户满意度、安全事件、P95/P99 延迟、成本、回退率和人工介入率。模型分数、检索命中率、工具调用成功率都是中间指标，最终要回到用户任务是否被可靠完成。

**追问展开。** 一个稳的上线流程是：离线评测过线，灰度流量观察，监控核心指标，保留回滚，收集失败样本进入下一轮数据和评估。  
**易错点 / 边界。** 单项指标上升不代表系统变好。例如答案更长可能 judge 分更高，但延迟、成本和用户满意度下降。  
**继续读。** [成本建模与 SLO 设计](../inference/cost-modeling-and-slo-design.md) / [证据判断原则](../references/site-judgement-principles.md)
