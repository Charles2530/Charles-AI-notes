# Prompt、CoT 与 RAG 来源台账

## 使用方式

这份台账服务于 [Prompt、CoT 与 RAG](../foundations/prompt-engineering-cot-and-rag-basics.md)。正文不把 prompt engineering 写成技巧清单，而是写成可验证的信息流：上下文组装、权限边界、示例、schema、CoT 工作区、工具状态机、RAG 证据管线和回归评测。

第一类来源是官方和高质量教程。Prompt Engineering Guide、OpenAI、Anthropic、Google prompt engineering whitepaper 等资料用于确认提示词基础结构、示例、复杂任务拆分、清晰指令和长上下文提示的实践边界。

第二类来源是 reasoning 和 tool-use 论文。Chain-of-Thought、Self-Consistency、ReAct、Program-of-Thoughts 和 Tree-of-Thoughts 帮助本站解释 CoT 不是事实证明，而是中间工作区和 test-time search 的接口。

第三类来源是 RAG 论文和工程文档。RAG 原论文、REALM、DPR、LlamaIndex、LangChain、Haystack 和 OpenAI Cookbook 等资料用于解释检索、rerank、context assembly、citation 和 eval 的证据链。

第四类来源是安全和 prompt injection。Simon Willison、OWASP LLM Top 10、Lakera prompt injection materials 等资料用于说明外部文本、工具结果和检索文档必须被当成低权限数据，而不是新指令。

## 写作边界

正文不把“写得更长”当成更好。长 CoT、长 prompt 和长上下文都要和任务收益、延迟、成本、证据质量一起看。

正文不把 RAG 解释成把资料塞进 prompt。RAG 的关键是 retrieval recall、rerank precision、context assembly、grounded generation、citation check 和 eval。

正文不鼓励暴露完整内部草稿。面向用户的可靠输出应给出可复核的关键理由、证据、工具结果和不确定性边界。

## 来源

| 来源 | 类型 | 本站使用方式 | 关联主题 |
| --- | --- | --- | --- |
| [Prompt Engineering Guide 中文版](https://www.promptingguide.ai/zh) | 中文教程 | 参考 zero-shot、few-shot、CoT、Self-Consistency、Prompt Chaining、ReAct 和 RAG 的中文讲法。 | 中文讲法 |
| [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering) | 官方文档 | 参考清晰指令、参考文本、复杂任务拆分和外部工具的组织方式。 | 官方实践 |
| [Anthropic Prompt Engineering Overview](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview) | 官方文档 | 参考清晰直接、示例、thinking、长上下文和XML标签等提示边界。 | 官方实践 |
| [Google Prompt Engineering Whitepaper](https://www.kaggle.com/whitepaper-prompt-engineering) | 白皮书 | 参考 prompt design、examples、chain-of-thought 和 evaluation 的系统讲法。 | 提示工程 |
| [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903) | 论文 | 支撑“CoT 给复杂任务中间工作区，但不自动保证事实正确”。 | CoT |
| [Self-Consistency](https://arxiv.org/abs/2203.11171) | 论文 | 支撑多条推理路径采样和聚合的 test-time search 视角。 | 多候选推理 |
| [ReAct](https://arxiv.org/abs/2210.03629) | 论文 | 支撑 reasoning 和 acting 交替，以及工具观察更新状态的写法。 | 工具调用 |
| [Program of Thoughts](https://arxiv.org/abs/2211.12588) | 论文 | 说明把计算交给程序执行可以替代部分文本推理。 | 工具验证 |
| [Tree of Thoughts](https://arxiv.org/abs/2305.10601) | 论文 | 补充搜索多个中间状态、评估和回溯的推理模式。 | 搜索推理 |
| [Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401) | 论文 | 使用 Figure 1 和 RAG 公式解释 retriever、document latent variable 和 generator。 | RAG |
| [DPR](https://arxiv.org/abs/2004.04906) | 论文 | 参考 dense passage retrieval 如何把问题和文档编码进向量空间。 | 检索 |
| [REALM](https://arxiv.org/abs/2002.08909) | 论文 | 补充 retrieval-augmented pre-training 视角。 | 检索增强 |
| [LlamaIndex RAG docs](https://docs.llamaindex.ai/en/stable/understanding/rag/) | 官方文档 | 参考 indexing、retrieval、response synthesis 和 eval 的工程拆分。 | RAG 系统 |
| [LangChain RAG docs](https://docs.langchain.com/oss/python/langchain/rag) | 官方文档 | 参考 retrieval chain、document loading、splitting 和 generation 的系统组织。 | RAG 系统 |
| [Haystack RAG pipelines](https://docs.haystack.deepset.ai/docs/rag) | 官方文档 | 参考 pipeline 视角下的 retriever、generator 和 evaluation。 | RAG 系统 |
| [Lilian Weng: LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) | 技术博客 | 用于把 planning、memory、tool use、self-reflection 和 agent failures 连起来读。 | Agent |
| [Simon Willison: Prompt Injection](https://simonwillison.net/2022/Sep/12/prompt-injection/) | 技术博客 | 支撑外部文本不可被当成可信指令的安全边界。 | Prompt injection |
| [OWASP Top 10 for LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/) | 安全项目 | 参考 prompt injection、sensitive information disclosure、tool misuse 等应用风险。 | 安全 |
