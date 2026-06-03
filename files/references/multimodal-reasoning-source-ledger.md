# 多模态推理来源台账

## 使用方式

这份台账服务于 [多模态推理](../foundations/multimodal-cot-and-reasoning-basics.md)。正文不把多模态写成“图片问答 + 长 CoT”，而是围绕证据链组织：感知证据、模态对齐、连接器瓶颈、token 预算、推理工作区、工具/验证器和评测边界。

第一类来源是图文对齐和 VLM 架构。CLIP、Flamingo、BLIP-2、LLaVA 和 Illustrated CLIP 用来解释图文语义空间、视觉连接器、cross-attention、query/resampler 和指令微调的基本路线。

第二类来源是 reasoning 和 tool use。Chain-of-Thought、Self-Consistency、ReAct、Tree of Thoughts、Program-of-Thoughts 等来源帮助本站说明 CoT 是中间工作区，工具和验证器改变的是搜索/验证过程，而不是自动保证事实正确。

第三类来源是多模态推理评测和失败分析。MMMU、ScienceQA / Multimodal-CoT、HallusionBench、GPT-4V System Card、Gemini 技术报告等资料用来支撑“要区分看错、想错、证据不足和工具误用”的写法。

第四类来源是中文讲法和高质量可视化博客。Prompt Engineering Guide 中文版、The Illustrated CLIP、Jay Alammar 的图解风格、D2L 的 shape/token 讲法都强调先建立读者可复算的中间表示，而不是堆术语。

## 写作边界

正文不使用固定术语表式模板。公式只在需要说明机制时出现，例如 CLIP 相似度、connector 压缩、token 长度和验证器筛选。

正文不把完整 CoT 暴露当成可靠性。用户可见输出更应该包含可复核的关键观察、证据来源、工具结果和不确定性边界。

正文不把 multimodal benchmark 总分当成能力证明。必须追问模型是否看见了关键证据、是否把证据和问题对齐、是否使用了外部工具、是否区分感知错误和推理错误。

## 来源

| 来源 | 类型 | 本站使用方式 | 关联主题 |
| --- | --- | --- | --- |
| [CLIP](https://arxiv.org/abs/2103.00020) | 论文 | 使用 Figure 1 和图文对比机制说明“对齐不是推理”。 | 图文对齐 |
| [The Illustrated CLIP](https://jalammar.github.io/illustrated-clip/) | 可视化博客 | 借鉴相似度矩阵、zero-shot classifier 和图文空间的讲法。 | 可视化讲解 |
| [Flamingo](https://arxiv.org/abs/2204.14198) | 论文 | 解释 gated cross-attention 和 Perceiver Resampler 如何让语言模型读取视觉证据。 | VLM connector |
| [BLIP-2](https://arxiv.org/abs/2301.12597) | 论文 | 使用 Q-Former 图说明 connector 是信息瓶颈。 | Q-Former |
| [LLaVA](https://llava-vl.github.io/) | 项目页 / 论文 | 解释 visual instruction tuning 如何把 VLM 推向对话式使用。 | 指令微调 |
| [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903) | 论文 | 说明中间推理样例为什么能改善复杂任务。 | CoT |
| [Self-Consistency](https://arxiv.org/abs/2203.11171) | 论文 | 支撑多条 reasoning path 采样后聚合答案的 test-time search 视角。 | 多候选推理 |
| [ReAct](https://arxiv.org/abs/2210.03629) | 论文 | 解释 reasoning 和 acting 交替，工具结果作为证据进入下一步。 | 工具调用 |
| [Program of Thoughts](https://arxiv.org/abs/2211.12588) | 论文 | 说明把计算交给代码解释器或程序执行可以替代纯文本推理的一部分。 | 工具验证 |
| [Multimodal Chain-of-Thought Reasoning](https://arxiv.org/abs/2302.00923) | 论文 | 支撑多模态任务里需要先对齐视觉证据再推理的思路。 | 多模态 CoT |
| [MMMU](https://arxiv.org/abs/2311.16502) | 论文 | 作为多学科多模态理解与推理评测的代表。 | 评测 |
| [HallusionBench](https://arxiv.org/abs/2310.14566) | 论文 | 支撑视觉幻觉和反常识视觉问答需要单独诊断。 | 失败分析 |
| [GPT-4V System Card](https://openai.com/index/gpt-4v-system-card/) | 官方系统卡 | 参考多模态模型在视觉定位、OCR、风险与不确定性上的限制描述方式。 | 系统边界 |
| [Gemini 1.5 Technical Report](https://arxiv.org/abs/2403.05530) | 技术报告 | 参考长上下文多模态评测和信息检索式任务的报告方式。 | 长上下文 |
| [Prompt Engineering Guide 中文版](https://www.promptingguide.ai/zh) | 中文教程 | 参考 CoT、ReAct、RAG 和 prompt chaining 的中文解释方式。 | 中文讲法 |
