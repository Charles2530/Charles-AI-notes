# 位置编码与 Mask 来源台账

## 使用方式

这份台账服务于 [位置编码与 Mask](../foundations/positional-encoding-masks-and-context.md)。正文不把 RoPE、ALiBi、causal mask 写成术语表，而是围绕两个问题讲：token 如何知道自己在哪里，以及 token 在 attention 中能看谁。

第一类来源是位置编码论文。Attention Is All You Need、RoFormer/RoPE、ALiBi、Position Interpolation、YaRN 和后续长上下文扩展资料，帮助本站解释绝对位置、相对位置、旋转位置、线性距离偏置和 context extension 的机制边界。

第二类来源是 attention mask 与 backend 文档。Hugging Face attention interface、FlashAttention 资料和 sequence packing 博客帮助本站说明：mask 不只是数学矩阵，也是 runtime/backend 接口；causal、padding、packing、sliding window 和 block masks 必须被正确传递。

第三类来源是多模态和长上下文系统。MagiAttention、EAGLE tree mask、VLA/视频 token 资料帮助本站说明现代 mask 可能是异构可见性图，而不是简单三角矩阵。

中文讲解材料主要用于学习写法：先提出读者真正卡住的问题，再用例子、公式和工程边界逐步收束。科学空间和 Hugging Face 中文博客有稳定公开链接，可以放进正文；检索不到稳定原文的公众号内容只作为讲解风格参考，不作为事实或图片来源。

## 写作边界

正文不把“支持长上下文”归因给单一 RoPE 参数。长上下文还需要长样本训练、mask 规则、距离分桶评测、KV cache 管理和服务指标。

正文不把 mask 当成模型内部小细节。Mask 是数据契约：packed sample、未来动作标签、padding、工具结果和多模态块之间的可见性一旦错，训练结论就会被污染。

正文不把多模态 token 直接拼接当成解决方案。图像、视频、文档和机器人轨迹都需要空间、时间、相机、动作和 episode 边界。

## 来源

| 来源 | 类型 | 本站使用方式 | 关联主题 |
| --- | --- | --- | --- |
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | 论文 | 使用 sinusoidal positional encoding 和 Figure 3 说明 attention 能读远处但仍需要顺序机制。 | 位置编码 |
| [RoFormer / RoPE](https://arxiv.org/abs/2104.09864) | 论文 | 解释 RoPE 用旋转矩阵把相对位置写进 Q/K 点积。 | RoPE |
| [EleutherAI: Rotary Embeddings](https://blog.eleuther.ai/rotary-embeddings/) | 技术博客 | 借鉴 RoPE 的直觉推导和实现解释方式。 | RoPE 讲法 |
| [科学空间：旋转式位置编码](https://kexue.fm/archives/8265/comment-page-5) | 中文技术博客 | 借鉴从复数旋转、矩阵形式到相对位置性质的推导路线。 | 中文 RoPE 讲法 |
| [Hugging Face：设计位置编码](https://huggingface.co/blog/zh/designing-positional-encoding) | 中文技术博客 | 借鉴从简单位置编码逐步设计到 RoPE 的教学结构。 | 中文入门讲法 |
| [Train Short, Test Long: ALiBi](https://arxiv.org/abs/2108.12409) | 论文 | 解释 attention score 加线性距离 bias 为什么适合外推。 | ALiBi |
| [Position Interpolation](https://arxiv.org/abs/2306.15595) | 论文 | 支撑 RoPE-based LLM context window extension 的位置插值视角。 | 长上下文 |
| [YaRN](https://arxiv.org/abs/2309.00071) | 论文 | 解释 RoPE context extension 需要缩放策略和少量继续训练。 | 长上下文 |
| [Hugging Face Attention Interface](https://huggingface.co/docs/transformers/attention_interface) | 官方文档 | 支撑不同 attention backend 需要不同 mask 格式，约束可能被静默丢掉的说明。 | mask backend |
| [Hugging Face sequence packing blog](https://huggingface.co/blog/sirluk/llm-sequence-packing) | 技术博客 | 参考 packed sequence 和 masked attention 的讲法。 | packing |
| [MagiAttention](https://arxiv.org/abs/2502.13138) | 论文 | 使用 mask pattern 图说明异构 mask、packed sequence 和长上下文训练的系统问题。 | 异构 mask |
| [FlashAttention](https://arxiv.org/abs/2205.14135) | 论文 | 补充 attention mask 与 IO-aware attention kernel 的关系。 | attention kernel |
| [vLLM PagedAttention](https://arxiv.org/abs/2309.06180) | 论文 | 补充 KV cache 和可见历史的 serving 成本边界。 | KV cache |
| [EAGLE-2](https://arxiv.org/abs/2406.16858) | 论文 | 参考 speculative decoding tree mask 中“分支只能看祖先”的例子。 | tree mask |
