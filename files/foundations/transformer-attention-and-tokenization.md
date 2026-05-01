# Transformer、Attention 与 Tokenization

Transformer 是现代 LLM、VLM、DiT、世界模型和很多推理系统的核心结构。它的关键思想是：把输入变成 token，再用 attention 让 token 之间按相关性互相读取信息。

![Transformer 原论文模型结构图](../assets/images/paper-figures/transformer/attention-is-all-you-need-figure-1.png){ width="620" }

<small>图源：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)，Figure 1。原论文图意：Transformer 由 encoder 和 decoder 组成，核心模块包括 multi-head attention、feed-forward、residual connection、normalization、positional encoding 与输出 softmax。</small>

!!! note "图解：Transformer 总结构先抓三条线"
    第一次看这张图不要急着背每个框。第一条线是输入 token 先变成 embedding，并叠加 positional encoding，让模型知道顺序。第二条线是每层都交替做 attention 和 feed-forward：attention 负责跨 token 读信息，feed-forward 负责逐 token 变换表示。第三条线是 decoder 比 encoder 多了 masked self-attention 和 encoder-decoder attention，所以它既能按因果顺序生成下一个 token，也能读取 encoder 提供的源序列信息。

!!! note "初学者先抓住"
    Attention 的核心不是“模型很聪明”，而是每个 token 都能按相关性去读取其他 token。Q/K/V 可以先理解成：我想找什么、别人暴露什么、真正读走什么。

!!! example "有趣例子：开会做笔记"
    Self-Attention 像会议里每个人听同一桌其他人发言；Cross-Attention 像写报告的人去查外部资料。前者整理内部上下文，后者把图片、文本提示、工具结果等外部信息读进来。

!!! tip "学完本页你应该能"
    看到一个 LLM、VLM、DiT 或 agent 模型时，能分清 tokenization、embedding、self-attention、cross-attention 和输出 head 分别在哪一层；看到 KV cache 或长上下文问题时，能理解它为什么来自 attention 的历史读取机制。

## 1. Tokenization：模型如何“切开”输入

Token 是模型处理信息的基本单位。不同模态有不同 token：

| 模态 | token 例子 |
| --- | --- |
| 文本 | subword token |
| 图像 | patch token |
| 视频 | tubelet token 或 frame token |
| 机器人 | action token、state token |
| 世界模型 | latent token、未来状态 token |

Tokenization 的作用是把复杂输入变成一个序列：

\[
x = [x_1, x_2, \ldots, x_L]
\]

然后每个 token 被映射成 embedding：

\[
h_i = \text{Embedding}(x_i)
\]

## 2. Attention 的 Q/K/V 直觉

Attention 可以用“提问、匹配、读取”理解：

- `Q`：Query，当前位置想问什么。
- `K`：Key，每个位置暴露什么可匹配信息。
- `V`：Value，真正被读取的内容。

公式是：

\[
\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V
\]

这表示每个 token 会根据相似度，从其他 token 里加权读取信息。

![Scaled Dot-Product Attention 原论文图](../assets/images/paper-figures/transformer/attention-is-all-you-need-figure-2-left.png){ width="280" }
![Multi-Head Attention 原论文图](../assets/images/paper-figures/transformer/attention-is-all-you-need-figure-2-right.png){ width="360" }

<small>图源：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)，Figure 2。原论文图意：左图是 scaled dot-product attention 的 Q/K/V 计算流程，右图是 multi-head attention 将多组 attention 并行后拼接再投影。</small>

!!! note "难点解释：为什么要 multi-head"
    单个 attention head 只能在一套相似度空间里读取信息。Multi-head attention 等于让模型同时用多套“问题-匹配-读取”规则看同一段序列：有的 head 可能关注局部语法，有的关注长距离依赖，有的关注对齐关系。最后的线性投影再把这些视角合并起来。

## 3. Self-Attention 和 Cross-Attention

### Self-Attention

Q、K、V 都来自同一个序列。例如 LLM 中每个 token 读取上下文中的其他 token。

### Cross-Attention

Q 来自主干序列，K/V 来自外部条件。例如扩散模型中，图像 latent 的 query 去读取文本 embedding 的 key/value，从而让图像生成受 prompt 控制。

一个极简伪代码：

```text
# self-attention
Q = x @ Wq
K = x @ Wk
V = x @ Wv
y = softmax(Q @ K.T / sqrt(d)) @ V

# cross-attention
Q = image_latent @ Wq
K = text_embedding @ Wk
V = text_embedding @ Wv
y = softmax(Q @ K.T / sqrt(d)) @ V
```

## 4. 为什么 Attention 强也贵

Self-Attention 的核心矩阵是 \(L \times L\)，其中 \(L\) 是序列长度。序列越长，attention 计算和显存压力越大。

这解释了为什么长上下文推理会重点优化：

- KV cache
- FlashAttention
- sliding window attention
- context compression
- prefix cache

如果你关心为什么长上下文还会涉及位置编码、causal mask、padding mask 和 KV cache，可以继续看 [位置编码、Mask 与上下文](positional-encoding-masks-and-context.md)。

## 5. Transformer Block 的基本结构

一个常见 block 可以写成：

```text
function TransformerBlock(x):
    x = x + Attention(Norm(x))
    x = x + MLP(Norm(x))
    return x
```

这里的残差连接和归一化非常关键。没有它们，深层 Transformer 很难稳定训练。

## 6. 和后续专题的关系

- [扩散模型中的 DiT](../diffusion/training.md)：图像 patch token 进入 Transformer 去做去噪。
- [VLM 架构与训练](../vlm/architecture-and-training.md)：图像 token 和文本 token 如何连接。
- [推理系统](../inference/index.md)：KV cache 和 attention 决定长上下文成本。
- [算子与编译器](../operators/index.md)：FlashAttention 和 GEMM 是 Transformer 性能核心。
- [线性层、MLP 与 GEMM](linear-layers-mlp-and-gemm.md)：理解 QKV 投影、MLP 和矩阵乘热路径。

## 小结

Transformer 的核心不是“一个大模型名字”，而是一套 token 之间动态通信的机制。理解 token、embedding、Q/K/V、self-attention 和 cross-attention，就能读懂很多现代 AI 系统的共同骨架。
