# 工具与查阅

这里按当前遇到的问题连接数学、诊断问答、可运行示例与原始资料。想了解某个技术的完整机制，先进入[技术地图](../index.md)；想核对具体方法的实验与来源，再读[论文精读](../paper-deep-dives/index.md)。

## 从错误现象查起

| 当前问题 | 问答入口 |
| --- | --- |
| shape、梯度、mask或KV行为不符合预期 | [基础计算](../knowledge-qa/foundations-and-transformer.md) |
| loss下降但目标任务未改善 | [训练与对齐](../knowledge-qa/training-and-alignment.md) |
| 吞吐、延迟、显存和量化结果矛盾 | [推理服务](../knowledge-qa/inference-serving-and-quantization.md) |
| kernel数值或性能不正确 | [算子与性能](../knowledge-qa/operators-and-performance.md) |
| 有证据或工具调用，答案仍不可靠 | [RAG与评估](../knowledge-qa/rag-agent-and-evaluation.md) |
| 视觉理解、视频生成和条件控制失败 | [多模态与生成](../knowledge-qa/multimodal-generative-and-embodied.md) |
| 下一帧准确却无法规划 | [世界模型](../knowledge-qa/world-models.md) |
| 离线动作好但机器人闭环失败 | [具身与VLA](../knowledge-qa/embodied-ai-and-vla.md) |

问答用于定位差异，正文负责展开推导。[术语表](../glossary/index.md)适合确认一个词在当前上下文指什么，[数学工具](../prerequisite-math/index.md)适合补足公式的计算对象。

## 对照公式运行一个小例子

[基础机制代码](../assets/examples/foundations/mechanisms.py)计算Linear、训练更新、attention与解码；[视频生成机制代码](../assets/examples/diffusion/video_generation_mechanisms.py)核对shape、训练目标与采样步骤；[最小DDPM](../diffusion/minimal-ddpm-implementation.md)连接噪声日程和去噪；[RL实践](../reinforcement-learning/practice-and-algorithm-selection.md)连接轨迹、价值与策略。

先比较小例子的中间数，再扩大尺寸。执行成功说明指定路径可运行，不能直接解释为模型收敛、生成质量或论文复现。每份示例的依赖与验证范围以对应正文为准。

## 如何从教程读到实现

[公式到代码的查阅方法](../roadmap/research-and-engineering-study-plan.md)说明怎样确认输入输出、最小数值、实现差异和评价条件。[技术对照](../roadmap/comparison-tables.md)区分改变结构、目标、采样与执行的方法，[具体决策例子](../roadmap/common-decision-patterns-and-tradeoffs.md)说明怎样用失败证据选择下一项修改。

[思考探索](../thinking-exploration/index.md)收录跨专题分析与待检验假设。它们与基础定理、代码执行和原始实验具有不同证据强度，阅读时可沿文内出处继续核对。
