# 知识 QA

知识 QA 是面试和速查入口：遇到一个高频问题时，先用 1-2 分钟说清对象、机制和边界，再顺着 `继续读` 进入站内长文、论文精读或来源台账。

## 按问题类型选入口

| 你正在准备的问题 | 先读哪一页 |
| --- | --- |
| Transformer、attention、tokenizer、RoPE、mask、解码策略 | [基础与 Transformer](foundations-and-transformer.md) |
| 预训练、SFT、RLHF、DPO、GRPO、分布式训练、实验治理 | [训练与对齐](training-and-alignment.md) |
| TTFT、TPOT、KV cache、vLLM、投机解码、量化上线 | [推理服务与量化](inference-serving-and-quantization.md) |
| GEMM、FlashAttention、Triton、CUDA、profiling、kernel 上线 | [算子与性能](operators-and-performance.md) |
| RAG、Agent、function calling、LLM-as-judge、prompt injection | [RAG、Agent 与评估](rag-agent-and-evaluation.md) |
| VLM、视觉连接器、扩散、视频生成、多模态评测 | [多模态与生成模型](multimodal-generative-and-embodied.md) |
| RSSM、Dreamer、JEPA、WAM/VAM、动作条件世界模型、闭环评测 | [世界模型](world-models.md) |
| VLA、动作表示、Sim2Real、机器人视觉、闭环恢复、安全部署 | [具身智能与 VLA](embodied-ai-and-vla.md) |

## 和其他板块怎么配合

| 板块 | 主要作用 | QA 里的用法 |
| --- | --- | --- |
| [前置数学知识](../prerequisite-math/index.md) | 解释公式符号和数学直觉 | 遇到 KL、value、score、ODE、四元数等符号时回去补底座 |
| [基础概念](../foundations/index.md) | 解释模型、数据、训练、推理的机制 | QA 给短答，基础页给机制细节 |
| 专题页 | 系统讲一个方向 | QA 的 `继续读` 优先指向专题页 |
| 论文精读 | 解释论文贡献、证据和边界 | QA 提供高频问法，论文精读保留完整证据链 |
