# Charles AI notes

这里不再是一个“AI 知识地图仪表盘”。它更适合被当成一组可以独立阅读的中文技术博客：每篇文章只回答一个具体问题，先讲清直觉，再讲机制、公式、论文图和边界。

如果你第一次打开本站，不需要先读各方向总览页。直接选一篇和你当下问题最接近的文章读；读到需要背景时，再顺手打开文中的外部资料或站内补充。

首页下面只放少量建议起点，不代表全站内容被缩减。完整论文库仍在侧边栏的 `论文精读` 分组里；当前保留 63 篇独立论文页，只是它们不再被堆成一个需要先扫完的大清单。

## 推荐先读

| 你想弄清的问题 | 直接读这篇 |
| --- | --- |
| Transformer 到底如何把 token 互相“读”起来 | [Transformer 输入与注意力](foundations/transformer-attention-and-tokenization.md) |
| 一次大模型训练为什么不是“把数据喂进去” | [模型训练从零入门](training/beginner-model-training-primer.md) |
| vLLM、SGLang、TensorRT-LLM 在服务里分别解决什么 | [推理运行时](inference/serving-runtimes-vllm-sglang-and-tensorrt-llm.md) |
| GPTQ、AWQ、SmoothQuant 的差别到底在哪里 | [PTQ / GPTQ / AWQ / SmoothQuant](quantization/ptq-gptq-awq-smoothquant.md) |
| 扩散模型训练为什么要预测噪声 | [扩散训练与表示](diffusion/training.md) |
| 世界模型和普通视频生成差在哪里 | [从零理解世界模型](world-models/beginner-world-models-primer.md) |
| Dreamer 为什么能在想象轨迹里训练策略 | [RSSM、Dreamer 与规划](world-models/rssm-dreamer-and-planning.md) |
| VLA 的动作 token 为什么不是随便加几个数字 | [动作表示与控制接口](vla/action-representation-and-interfaces.md) |
| 具身智能系统为什么总卡在数据、动作和闭环 | [具身智能从零路线](embodied-ai/beginner-learning-path.md) |

## 外部经典阅读

这些外部材料讲得非常好，本站会在正文里直接引用它们，而不是强迫你只在站内跳转。

| 主题 | 值得读的外部材料 | 为什么推荐 |
| --- | --- | --- |
| Transformer | [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) | 用图把 encoder、decoder、attention 和生成过程讲得很清楚。 |
| 扩散模型 | [Lil'Log: What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) | 从 score、SDE、DDPM 到采样直觉都有扎实解释。 |
| 视觉解释 | [Distill: Circuits](https://distill.pub/2020/circuits/visualizing-weights/) | 适合学习“图不是装饰，而是解释对象”的写法。 |
| 文档结构 | [Diataxis](https://diataxis.fr/) | 区分 tutorial、how-to、explanation、reference，避免一篇文章混成大杂烩。 |
| 技术写作 | [Google developer documentation style guide](https://developers.google.com/style/headings) | 标题和段落要服务读者任务，不要服务作者目录。 |
| 量化 | [Hugging Face Quantization concept guide](https://huggingface.co/docs/transformers/quantization/concept_guide) | 把低比特格式、校准、运行时支持放在同一套概念里。 |
| 推理系统 | [vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention](https://vllm.ai/blog/2023-06-20-vllm) | PagedAttention 是理解现代 LLM serving 的好入口。 |
| 具身智能 | [DeepMind RT-2](https://deepmind.google/blog/rt-2-new-model-translates-vision-and-language-into-action/) | 展示 VLM 到动作的路线，也暴露动作 token 的限制。 |
| 世界模型 | [DeepMind Genie 2](https://deepmind.google/blog/genie-2-a-large-scale-foundation-world-model/) 和 [Meta V-JEPA 2](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/) | 一个偏生成式环境，一个偏预测表征，适合对照读。 |

## 本站现在怎么维护

新的写法有三条约束。

第一，文章必须专一。标题如果像“路线图”“总览”“矩阵”，正文就容易变成目录；标题最好能回答一个问题，例如“为什么 KV cache 会让长上下文服务变贵”。

第二，图必须进入解释链。每张论文图下面都要说明图源、原图想表达什么、本站用它解释什么。复杂图的读法要写进正文，不再塞进一堆提示框。

第三，外链是一等材料。外部博客、论文、官方技术文章如果比站内页更适合补直觉，就直接放在正文该出现的位置。
