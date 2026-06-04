# 外部精读来源台账

这页记录本站深改文章时优先学习的外部博客、官方报告、论文项目页和中文技术解读。它不是普通链接收藏，而是写作输入台账。

**维护规则。**

| 规则 | 说明 |
| --- | --- |
| 事实优先回到原始来源 | 关键 claim、数字、图和模型细节优先引用论文、官方博客、项目页、文档或报告。 |
| 博客学习讲法 | 高质量博客和中文文章主要学习叙事结构、例子、图解顺序和工程直觉。 |
| 图源必须可回查 | 论文图、官方图和项目页图可以进正文；版权不清的公众号截图只做阅读参考。 |
| 每次深改都补外部精读 | 核心文章要有真正有帮助的外部来源，放在正文相关处或文末。 |
| 中文材料要有边界 | 中文材料帮助建立直觉，但强 claim 仍回到原论文、官方博客或项目页。 |

## 写作方法参考

| 来源 | 类型 | 本站学什么 | 落到页面 |
| --- | --- | --- | --- |
| [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) | 可视化博客 | 先用数据流和图解释机制，再补公式和模型细节。 | Transformer、attention、tokenization |
| [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) | 代码讲解 | 把 attention、mask、multi-head、position 和训练循环逐段落到 PyTorch 代码，提醒本站基础页必须讲清 shape 如何流动。 | Transformer、attention、训练入门 |
| [colah: Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) | 经典技术博客 | 用一个核心直觉贯穿整篇，而不是把公式散放成符号表。 | RNN/SSM、状态记忆、世界模型 |
| [Lil'Log: What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) | 技术博客 | 从 forward process 到 reverse process、score、guidance、distillation 逐层递进。 | 扩散训练、噪声日程、采样、蒸馏 |
| [Sebastian Raschka: Build a Large Language Model From Scratch](https://github.com/rasbt/LLMs-from-scratch) | 教程/代码书 | 用可运行代码把 tokenizer、attention、pretraining、finetuning 串成一条线，提醒本站基础页少讲口号，多讲对象如何流动。 | Transformer、训练入门、tokenization |
| [Andrej Karpathy: minBPE](https://github.com/karpathy/minbpe) | 代码教程 | 用极小实现拆开 BPE 训练、merge 和 encode/decode，适合学习 tokenizer 不是黑盒预处理。 | tokenizer、BPE、训练入门 |
| [Distill: Visualizing Weights](https://distill.pub/2020/circuits/visualizing-weights/) | 可解释性博客 | 图不是装饰，而是论证对象；每张图都要告诉读者先看什么。 | 图片质量标准、视觉解释类页面 |
| [Diataxis](https://diataxis.fr/) | 文档结构方法 | tutorial、how-to、explanation、reference 要分开，避免一篇文章混成目录。 | 全站结构、首页和板块入口 |
| [Google developer documentation style guide: headings](https://developers.google.com/style/headings) | 写作规范 | 标题应服务读者任务，避免空泛名词和维护者目录式标题。 | 全站标题改写 |

## 中文技术文章与公众号讲法

中文材料的价值不在于替代论文，而在于更容易抓住读者入口：方向为什么重要、工程卡在哪里、新名词和旧路线有什么差别。事实、数字、图和强 claim 仍回到原论文、官方博客、项目页或仓库。

| 来源 | 类型 | 本站学什么 | 使用边界 |
| --- | --- | --- | --- |
| [机器之心](https://www.jiqizhixin.com/) | 中文技术媒体 / 公众号生态 | 先交代问题背景和论文意义，再用主图、方法拆解和实验结果把读者带进原文。 | 用作中文读法参考；性能数字和图必须回到论文或官方项目页。 |
| [量子位](https://www.qbitai.com/) | 中文技术媒体 / 公众号生态 | 用产业问题、读者痛点和路线对比开篇，适合学习“为什么这个问题值得看”。 | 行业判断不能直接写成技术证据；访谈类内容只标为路线观点。 |
| [智源社区](https://hub.baai.ac.cn/) | 中文社区 / 编译转载 | 把海外工程长文、报告和论文整理成中文结构，尤其适合系统工程和推理优化文章。 | 转载或编译内容要追到原文；中文页只保留为二次阅读线索。 |
| [量子位 / 智源：世界模型真正的考场在机器人执行里](https://hub.baai.ac.cn/view/54116) | 中文技术媒体 / 公众号转载 | 学习如何从“视频榜单和机器人执行错位”切入，把评测对象拉回任务成功率、闭环频率和数据基建。 | 行业材料；具体模型数字和 claim 需追原论文、官方项目页或原始发布。 |
| [PaperWeekly](https://zhuanlan.zhihu.com/paperweekly) | 中文论文解读 / 公众号生态 | 学习论文速读如何把贡献、方法图和实验表拆成读者能跟上的顺序。 | 图、表、数字和结论必须回到论文或官方项目页。 |
| [Datawhale 教程矩阵](https://github.com/datawhalechina/so-large-lm) | 中文开源教程 | 学习把概念、公式、代码和学习路径放在同一套材料里的组织方式。 | 只作为中文讲法和学习路径参考；具体 API 和实验 claim 回官方/论文。 |
| [Datawhale Every-Embodied](https://github.com/datawhalechina/every-embodied) | 中文开源教程 / 公众号生态 | 学习具身智能文章如何从硬件、仿真、数据采集、策略学习、复现和部署串成可执行学习路线。 | 只作为讲法和学习路径参考；VLA 成功率、数据规模和模型细节回论文、项目页或官方仓库。 |
| [OneFlow Official CSDN](https://blog.csdn.net/OneFlow_Official) | 中文工程博客 | 以 kernel、分布式、推理阶段、checkpoint 等工程问题为线索展开，不把系统文章写成概念词典。 | 工程细节若涉及版本或实现，优先回到官方文档、论文或仓库。 |
| [Moritz Reuss: State of VLA Research at ICLR 2026](https://mbreuss.github.io/blog_post_iclr_26_vla.html) | 研究者博客 | 学习 VLA 定义争议、benchmark 读法和趋势分类。 | 个人分析；不替代论文结果。 |
| [ICLR Blogposts: Spatial Awareness in VLMs](https://iclr-blogposts.github.io/2026/blog/2026/spatial-awareness/) | 学术博客 | 学习空间理解问题如何从位置编码、attention 和训练目标展开。 | 具体 claim 回到引用论文。 |

落到正文时，不能写成“某媒体说某模型很强”。正确做法是借它提出困惑，再回到论文图、公式和实验口径，拆成可验证机制。

## 概率、潜变量与生成模型基础

| 来源 | 类型 | 本站学什么 | 落到页面 |
| --- | --- | --- | --- |
| [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) | 论文 | VAE 的核心不是“自编码器加噪声”，而是把不可算 posterior、摊销推断、ELBO 和重参数化接到随机梯度训练。 | 概率与潜变量模型、VAE、生成模型基础 |
| [An Introduction to Variational Autoencoders](https://arxiv.org/abs/1906.02691) | 教程论文 | 用深度潜变量模型和 inference model 的框架讲 VAE，适合帮助本站把 encoder / decoder 从概率方向上说清楚。 | 概率与潜变量模型、VAE 入门 |
| [Tutorial on Variational Autoencoders](https://arxiv.org/abs/1606.05908) | 教程论文 | 从直觉、数学推导到经验行为解释 VAE，适合补充 ELBO、posterior collapse 和生成质量边界。 | VAE、潜变量、生成模型误解 |
| [Variational Inference: A Review for Statisticians](https://arxiv.org/abs/1601.00670) | 综述论文 | 把 variational inference 放回“用可优化分布近似不可算 posterior”的统计问题，而不是只背 VAE 损失。 | 概率与潜变量模型、ELBO、VI |
| [RBC Borealis: Tutorial #5 Variational Auto-Encoders](https://rbcborealis.com/research-blogs/tutorial-5-variational-auto-encoders/) | 技术博客 | 从 latent variable、posterior、ELBO、Jensen inequality、reparameterization 逐步展开，适合学习如何把公式讲成问题链。 | 概率与潜变量模型、VAE |
| [Yang Song: Generative Modeling by Estimating Gradients of the Data Distribution](https://yang-song.net/blog/2021/score/) | 研究者博客 | 用 score function、Langevin dynamics、多噪声扰动和 SDE 连接扩散模型，帮助本站解释“去噪 MSE 背后是在学分布方向”。 | 概率与潜变量模型、扩散训练、Score SDE |
| [Deep Generative Modelling: A Comparative Review](https://arxiv.org/abs/2103.04922) | 综述论文 | 把 VAE、GAN、flow、energy-based、autoregressive 模型放在同一生成建模版图里，提醒本站解释模型时要说明密度、采样和评测取舍。 | 生成模型基础、扩散与自回归对比 |
| [TensorFlow Core 中文教程：卷积变分自编码器](https://www.tensorflow.org/tutorials/generative/cvae?hl=zh-CN) | 中文官方代码教程 | 用最小 CVAE 代码看 encoder、sampling layer 和 ELBO 训练如何落地。 | 概率与潜变量模型、训练入门 |
| [Pyro 中文教程：变分自编码器](https://pyro.org.cn/examples/vae.html) | 中文教程 / 概率编程 | 用 model / guide / Trace_ELBO 解释生成模型和推断模型分工。 | 概率与潜变量模型、ELBO |
| [机器之心：深度生成模型](https://www.jiqizhixin.com/graph/technologies/77c3d598-259e-4ecc-9617-bd27ca2098fd) | 中文技术百科 / 公众号生态 | 学习中文材料如何先解释“生成模型是在表达数据概率分布”，再展开模型家族、历史和评估难点。 | 生成模型基础、评估边界 |

## 强化学习与决策基础

| 来源 | 类型 | 本站学什么 | 落到页面 |
| --- | --- | --- | --- |
| [Sutton & Barto: Reinforcement Learning, 2nd Edition](http://incompleteideas.net/book/the-book-2nd.html) | 教材 | 用 MDP、return、value、Bellman、dynamic programming 建立 RL 的共同语言。 | MDP、Bellman、价值函数 |
| [OpenAI Spinning Up: RL Intro](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) | 工程教程 | 用 trajectory、return、policy、value function 把算法家族讲成可实现的对象。 | RL 入门、PPO、Actor-Critic |
| [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) | 论文 | 用 replay、target network 和 value 曲线解释 Bellman bootstrap 的深度学习版本。 | DQN、Bellman、value 诊断 |
| [Offline Reinforcement Learning Tutorial](https://arxiv.org/abs/2005.01643) | 综述/教程 | 区分 online、off-policy、offline，并解释分布外动作为什么让 Q-value 虚高。 | Offline RL、机器人数据、VLA 约束 |

## Transformer 基础与位置机制

| 来源 | 类型 | 本站学什么 | 落到页面 |
| --- | --- | --- | --- |
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | 论文 | 原始 Transformer 的 scaled dot-product attention、multi-head、mask、position 和 encoder-decoder 图；图和公式优先回到这里。 | Transformer、attention、mask |
| [Hugging Face NLP Course: Byte-Pair Encoding](https://huggingface.co/docs/course/chapter6/5) | 官方课程 | 用语料统计、pair merge 和 subword token 解释 BPE，不把 tokenizer 当作简单切词。 | tokenizer、BPE |
| [OpenAI tiktoken](https://github.com/openai/tiktoken) | 官方工具仓库 | 把 token count 与上下文成本、API 计费和工程接口联系起来。 | tokenizer、上下文成本 |
| [RoFormer / RoPE](https://arxiv.org/abs/2104.09864) | 论文 | 旋转位置编码如何把相对距离写进 Q/K 点积，而不是简单给 token 加编号。 | 位置编码、长上下文 |
| [EleutherAI: Rotary Embeddings](https://blog.eleuther.ai/rotary-embeddings/) | 技术博客 | 用直觉和公式解释 RoPE 为什么适合外推，适合作为中文正文里的讲法参考。 | 位置编码、RoPE |
| [ALiBi](https://arxiv.org/abs/2108.12409) | 论文 | 通过 attention bias 处理长上下文外推，提醒读者位置机制也可以直接改 score。 | 位置编码、长上下文 |
| [Triton paper](https://doi.org/10.1145/3315508.3329973) | 论文 | roofline、tiling 和自定义 kernel 如何连接 GEMM 数学与硬件执行。 | GEMM、Triton、算子 |

## 生成与解码

| 来源 | 类型 | 本站学什么 | 落到页面 |
| --- | --- | --- | --- |
| [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) | 论文 | 解释 beam search 和最大似然路径为什么会让开放文本退化，nucleus sampling 为什么要按上下文自适应截断候选。 | 生成与解码基础、采样策略 |
| [Hugging Face: Generation strategies](https://huggingface.co/docs/transformers/generation_strategies) | 官方文档 | 核对 greedy、beam、sampling、contrastive search、constraints 和 cache 等现代生成配置入口。 | 生成与解码基础、推理配置 |
| [Hugging Face: How to generate text](https://huggingface.co/blog/how-to-generate) | 技术博客 | 用示例解释 greedy、beam、top-k、top-p、temperature 和 repetition penalty 的行为差异。 | 生成与解码基础 |
| [OpenAI: Introducing Structured Outputs in the API](https://openai.com/index/introducing-structured-outputs-in-the-api/) | 官方技术文章 | 结构化输出靠 schema 约束候选集合；格式正确和事实正确仍要分开。 | 结构化输出、工具调用、解码基础 |
| [SpecInfer](https://arxiv.org/abs/2305.09781) | 论文 | 用 draft tree、parallel verification 和一致性区分投机解码与 verifier sampling。 | 投机解码、生成系统 |

## 训练与推理系统

| 来源 | 类型 | 本站学什么 | 落到页面 |
| --- | --- | --- | --- |
| [PagedAttention paper](https://arxiv.org/abs/2309.06180) | 论文 | 用 logical / physical KV blocks、block table、copy-on-write 和 decoding scenarios 解释 vLLM 的内存管理机制。 | vLLM / PagedAttention、KV cache、推理系统 |
| [vLLM PagedAttention blog](https://vllm.ai/blog/2023-06-20-vllm) | 官方工程博客 | KV cache 大、动态、碎片化；PagedAttention 用 block table 管理非连续物理块。 | 推理运行时、KV cache、服务系统 |
| [vLLM documentation](https://docs.vllm.ai/en/latest/) | 官方文档 | 核对当前 serving、scheduler、KV cache、speculative decoding、quantization 和部署入口，避免只按早期论文理解现版本。 | vLLM、推理系统、部署 |
| [vLLM GitHub](https://github.com/vllm-project/vllm) | 官方仓库 | 查看实现状态、版本变化和 runtime 功能边界；性能 claim 仍回论文、博客或 benchmark。 | vLLM、推理系统 |
| [vLLM Triton Attention Backend Deep Dive](https://vllm.ai/blog/2026-03-04-vllm-triton-backend-deep-dive) | 官方工程博客 | 把 attention backend、Triton tile、autotuning、CUDA graphs 和 persistent kernels 串成 kernel 解释链。 | 算子、PagedAttention、Triton |
| [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104) | 系统论文 | 从复杂生成程序出发讲 runtime；RadixAttention 和 compressed FSM 对应 prefix cache 与结构化输出。 | SGLang、agent serving、prefix cache |
| [SGLang documentation](https://docs.sglang.io/) | 官方文档 | 核对 RadixAttention、prefix caching、并行、PD disaggregation 和 attention backend 的当前入口。 | SGLang、RadixAttention |
| [TensorRT-LLM KV Cache System](https://nvidia.github.io/TensorRT-LLM/features/kvcache.html) | 官方文档 | 把 KV cache 讲成 block pool、reuse、offload、eviction、GQA/MQA 和 attention window 组合。 | TensorRT-LLM、KV cache、生产推理 |
| [智源/OneFlow：迈向 100 倍加速，全栈 Transformer 推理优化](https://hub.baai.ac.cn/view/33449) | 中文转载/工程长文 | 用“推理受内存限制”串起硬件、FlashAttention、vLLM、MoE 和 speculative decoding。 | 推理系统、算子、FlashAttention |
| [智源：伯克利开源 LLM 推理与服务库 vLLM](https://hub.baai.ac.cn/view/27505) | 中文转载 | 用 OS 虚拟内存类比讲 PagedAttention，适合作为中文直觉入口。 | vLLM/PagedAttention 讲解 |
| [中国信通院：大模型推理优化关键技术与应用实践研究报告（2026）](https://www.caict.ac.cn/kxyj/qwfb/ztbg/202604/P020260415615308750699.pdf) | 中文产业报告 | 从产业架构看模型压缩、算子优化、PD 分离、KV 多级存储和调度。 | 推理系统、KV cache、PD 分离、服务架构 |
| [vLLM: How Speculative Decoding Boosts vLLM Performance](https://vllm.ai/blog/spec-decode) | 官方工程博客 | 用 draft、verify、acceptance rate、QPS 和 workload 说明 speculative decoding 的收益条件，而不是只写“多猜几个 token”。 | speculative decoding、服务延迟 |
| [EAGLE-2](https://arxiv.org/abs/2406.16858) | 论文 | 用 acceptance-by-position、confidence calibration 和 dynamic draft tree 说明投机解码的 token budget 也需要按上下文调度。 | EAGLE-2、speculative decoding |
| [Megatron-LM](https://arxiv.org/abs/1909.08053) | 论文 | Tensor parallelism 如何切 GEMM 和 attention/MLP。 | Megatron/DeepSpeed、分布式训练 |
| [NVIDIA Megatron Core tensor parallel docs](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/tensor_parallel.html) | 官方文档 | 把论文里的 column/row parallel、vocab parallel 和 parallel cross entropy 对应到现代训练库 API。 | Megatron-LM、张量并行 |
| [ZeRO](https://arxiv.org/abs/1910.02054) | 论文 | optimizer state、gradient、parameter 分片分别省什么显存。 | DeepSpeed、训练显存 |
| [Microsoft Research: ZeRO publication page](https://www.microsoft.com/en-us/research/publication/zero-memory-optimizations-toward-training-trillion-parameter-models/) | 官方论文页 | 核对 ZeRO 论文标题、作者、发表脉络和官方项目入口；技术细节仍回 arXiv 论文和 DeepSpeed 文档。 | ZeRO、论文精读 |
| [DeepSpeed ZeRO tutorial](https://www.deepspeed.ai/tutorials/zero/) | 官方教程 | 把 ZeRO stage、optimizer/gradient/parameter partitioning 和配置入口落到可运行 DeepSpeed 训练。 | ZeRO、DeepSpeed、训练显存 |
| [DeepSpeed ZeRO documentation](https://deepspeed.readthedocs.io/en/latest/zero3.html) | 官方文档 | 核对 ZeRO-3、offload、参数初始化、gather 和运行时 API 的当前实现边界。 | ZeRO-3、ZeRO-Offload、训练运行时 |
| [DeepSpeed Megatron tutorial](https://www.deepspeed.ai/tutorials/megatron/) | 官方教程 | 把并行维度和配置项放进可运行训练栈，而不是停在概念图。 | Megatron / DeepSpeed |
| [PyTorch FSDP docs](https://docs.pytorch.org/docs/stable/fsdp.html) | 官方文档 | 分片、all-gather、reshard 和 checkpoint 要按训练生命周期解释。 | 分布式训练与 checkpoint |
| [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/) | 工程博客 | 从小实验、过拟合小 batch、检查 loss/grad 到逐步扩容的排查顺序，提醒本站训练页必须讲“如何知道训练可信”。 | 模型训练从零入门、训练稳定性 |
| [SmolLM3: smol, multilingual, long-context reasoner](https://huggingface.co/blog/smollm3) | 官方训练复盘 | 用多阶段数据混合、long-context、reasoning、SFT/alignment 串起训练 recipe。 | 训练入门、数据混合、后训练 |
| [Smol Training Playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook) | 官方工程手册 | 从为什么训练开始讲数据、评测、规模和失败复盘。 | 模型训练从零入门、实验经济学 |
| [Datawhale: VRAM Calculation and ZeRO](https://datawhalechina.github.io/llm-algo-leetcode/01_Hardware_Math_and_Systems/06_VRAM_Calculation_and_ZeRO.html) | 中文教程 | 用手算显存账解释参数、梯度、优化器状态和 ZeRO 分片，适合作为中文读者的训练系统入口。 | 模型训练从零入门、分布式训练、显存估算 |
| [Trainy: GPU Utilization is a Misleading Metric](https://www.trainy.ai/blog/gpu-utilization-misleading) | 工程博客 | 用 GPU utilization、SM efficiency、MFU 和 Softmax 案例说明“GPU 忙”不等于训练高效。 | GPU utilization、训练性能诊断 |
| [NVIDIA Mixed Precision Training Guide](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html) | 官方文档 | 把 Tensor Core、shape、arithmetic intensity 和混合精度路径作为性能诊断的一部分。 | GPU utilization、Tensor Core |
| [Nsight Compute Roofline](https://docs.nvidia.com/nsight-compute/2023.3/NsightCompute/index.html#rooflines) | 官方文档 | 用 roofline 区分 compute-bound 与 memory-bound，避免只看单个 utilization 面板。 | GPU profiling、kernel 诊断 |
| [CUTLASS GEMM API docs](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api.html) | 官方文档 | GEMM 层级、tile、warp、threadblock 的系统读法。 | GEMM、CUDA、算子 |
| [NVIDIA CUTLASS 3.x blog](https://developer.nvidia.com/blog/cutlass-3-x-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design/) | 官方工程博客 | 用可组合抽象解释 kernel 设计，而不是只列 API。 | GEMM、Triton/CUDA |
| [SLA](https://arxiv.org/abs/2509.24006) | 论文 | 把视频 DiT attention 分成 critical / marginal / negligible，说明 sparse exact、linear compensation 和 skip 应该分工。 | SLA、DiT attention 加速 |
| [SLA2](https://arxiv.org/abs/2602.12675) | 论文 | 用 learnable router、\(\alpha\) 混合和 QAT 修正 SLA 的手工路由与低比特部署问题。 | SLA2、低比特 attention |
| [thu-ml/SLA](https://github.com/thu-ml/SLA) | 官方仓库 | 核对 sparse-linear attention、fine-tuning 和 kernel 实现入口。 | SLA、部署验证 |
| [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982) | 论文 | 把 Muon 从矩阵正交化优化器改造成可用于 LLM 预训练的系统方案，重点看 weight decay、update RMS 和 Distributed Muon。 | Muon、优化器、LLM 预训练 |
| [MoonshotAI/Moonlight](https://github.com/MoonshotAI/Moonlight) | 官方仓库 | 核对 Moonlight MoE 架构、训练 recipe、模型入口和 Muon 实现边界。 | Moonlight、Muon |
| [NVIDIA Megatron Bridge: Moonlight](https://docs.nvidia.com/nemo/megatron-bridge/0.2.0/models/llm/moonlight.html) | 官方文档 | 从训练框架角度看 Moonlight / Muon 的分布式实现接口，避免只把优化器当单行替换。 | Muon、分布式训练 |
| [MagiAttention official blog](https://sandai-org.github.io/MagiAttention/docs/main/blog/magi_attn.html) | 官方工程博客 | 用 AttnSlice、dispatch solver、GroupCast/GroupReduce 和 overlap solver 讲超长上下文 attention。 | MagiAttention、长上下文训练 |
| [MagiAttention benchmark](https://sandai-org.github.io/MagiAttention/docs/main/blog/cp_benchmark.html) | 官方 benchmark | 按硬件、forward/backward、mask pattern 和 CP size 读分布式 attention 吞吐。 | MagiAttention、context parallel |
| [SandAI-org/MagiAttention](https://github.com/SandAI-org/MagiAttention) | 官方仓库 | 核对 FFA、group collectives、benchmark 脚本和实现状态。 | MagiAttention、训练系统实现 |

## 量化与数值

| 来源 | 类型 | 本站学什么 | 落到页面 |
| --- | --- | --- | --- |
| [Hugging Face Quantization concept guide](https://huggingface.co/docs/transformers/quantization/concept_guide) | 官方文档 | 把低比特格式、校准、权重/激活/KV、运行时支持放在一张概念表里。 | PTQ/GPTQ/AWQ/SmoothQuant、FP8 |
| [Hugging Face: selecting a quantization method](https://huggingface.co/docs/transformers/quantization/selecting) | 官方文档 | 按硬件、内存、延迟和模型任务选择量化方法，提醒本站不要把 INT4/INT8 标签写成唯一决策。 | 量化选型、部署验收 |
| [General Compute: Quantization for Inference](https://www.generalcompute.com/blog/quantization-for-inference-gptq-awq-smoothquant-fp8) | 工程博客 | 用 prefill / decode 的不同瓶颈解释 weight-only、W&A、FP8 的工程分工。 | PTQ/GPTQ/AWQ/SmoothQuant、FP8 |
| [PaddleNLP 大模型量化教程](https://paddlenlp.readthedocs.io/en/stable/llm/quantization.html) | 中文官方文档 | 从校准数据、GPTQ 步数、AWQ auto clip 等参数入口理解量化不是概念名词，而是一套工具链流程。 | 量化实践、校准、部署参数 |
| [LLM.int8()](https://arxiv.org/abs/2208.07339) | 论文 | 大模型 activation outlier 与 mixed-precision decomposition 的重要早期工作，帮助解释为什么激活离群值不是普通统计噪声。 | activation outliers、INT8、混合精度 |
| [SmoothQuant](https://arxiv.org/abs/2211.10438) | 论文 | activation outlier 如何通过平滑迁移到权重侧。 | 量化校准、activation outliers |
| [GPTQ](https://arxiv.org/abs/2210.17323) | 论文 | Hessian 近似如何服务逐层权重量化补偿。 | GPTQ、PTQ |
| [AWQ](https://arxiv.org/abs/2306.00978) | 论文 | 保护少量重要权重通道，而不是均匀压所有权重。 | AWQ、serving 量化 |

## 扩散与视频生成

| 来源 | 类型 | 本站学什么 | 落到页面 |
| --- | --- | --- | --- |
| [Lil'Log: What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) | 技术博客 | 把 DDPM、score、guidance、少步蒸馏连成一条推理链。 | 扩散训练、噪声日程、蒸馏 |
| [The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion) | 技术博客/代码讲解 | 公式、PyTorch 代码和采样过程互相对照。 | 扩散训练、采样 |
| [DDPM](https://arxiv.org/abs/2006.11239) | 论文 | 前向加噪、反向去噪、\(\epsilon\)-prediction 和 progressive generation 的基础来源。 | 扩散训练、采样 |
| [Improved DDPM](https://arxiv.org/abs/2102.09672) | 论文 | cosine schedule、learned variance、VLB 分项和训练/采样改进。 | 噪声日程、扩散训练 |
| [Score-Based Generative Modeling through SDEs](https://arxiv.org/abs/2011.13456) | 论文 | 把噪声预测、score field、reverse SDE 和 probability flow ODE 统一起来。 | Score SDE、采样、训练直觉 |
| [EDM](https://arxiv.org/abs/2206.00364) | 论文 | 噪声尺度、preconditioning、solver 和 NFE 是同一个设计空间。 | 噪声日程、采样器 |
| [Latent Diffusion Models](https://arxiv.org/abs/2112.10752) | 论文 | 在 VAE latent 上训练扩散模型，说明表示空间和算力成本必须一起看。 | Latent diffusion、扩散训练 |
| [DiT](https://arxiv.org/abs/2212.09748) | 论文 | Transformer denoiser 如何接入扩散目标，适合解释“扩散目标”和“token 建模架构”的分工。 | DiT、扩散训练 |
| [Video Diffusion Models](https://arxiv.org/abs/2204.03458) | 论文 | 把图像扩散扩展到 space-time U-Net 的基础路线。 | 视频扩散 |
| [Lumiere](https://arxiv.org/abs/2401.12945) | 论文/项目 | 一次性生成完整时长视频，说明 temporal coherence 不是补帧小问题。 | 视频扩散、时序一致性 |
| [OpenAI Sora technical report](https://openai.com/index/video-generation-models-as-world-simulators/) | 官方技术报告 | 把视频生成和世界模拟器联系起来，但需要保守区分 demo、open-loop 和 closed-loop。 | 视频扩散、世界模型 |
| [Wan technical report](https://arxiv.org/abs/2503.20314) | 技术报告 | 现代视频模型由 VAE/tokenizer、Video DiT、Flow Matching、多任务接口和系统优化组成。 | 视频扩散、Wan 专题 |
| [Wan-Video/Wan2.2 README](https://github.com/Wan-Video/Wan2.2) | 官方 README | Wan2.2 没有单独论文时，用官方 README 核对 MoE、TI2V-5B、模型发布和部署要求。 | Wan 专题、视频生成系统 |
| [DMD](https://arxiv.org/abs/2311.18828) | 论文 | 一步生成需要 distribution matching，而不是只回归 teacher。 | 一步生成、蒸馏 |
| [DMD project page](https://tianweiy.github.io/dmd/) | 项目页 | 用方法图、定性样例和延迟/质量对比讲清一步蒸馏的核心取舍。 | DMD、少步蒸馏图解 |
| [DMD2](https://arxiv.org/abs/2405.14867) | 论文 | 动态 fake critic 要比 generator 更快跟踪；backward simulation 要看到学生自己的中间分布。 | DMD2、少步蒸馏 |
| [DMD2 project page](https://tianweiy.github.io/dmd2/) | 项目页 | 用 4-step SDXL 样例、指标表和 checkpoint 入口核对少步蒸馏的实际质量边界。 | DMD2、SDXL 少步蒸馏 |
| [Phased DMD](https://arxiv.org/abs/2510.27684) | 论文 | 把 few-step DMD 拆成 SNR phase，并在中间状态上改写 fake score matching 目标，避免 SGTS 把视频运动和图像多样性压窄。 | Phased DMD、少步视频蒸馏 |
| [Phased DMD project page](https://x-niper.github.io/projects/Phased-DMD/) | 项目页 | 学习如何用固定 seed 视频对比、motion metrics 和补充 ablation 展示少步蒸馏不是只看画质。 | Phased DMD、视频蒸馏图解 |
| [Latent Consistency Models](https://arxiv.org/abs/2310.04378) | 论文 | consistency distillation 是少步生成的另一条路线，适合和 distribution matching 对比。 | 一致性模型、少步生成 |
| [Consistency Models](https://arxiv.org/abs/2303.01469) | 论文 | 同一 PF-ODE 轨迹上的不同噪声点应该映射到一致结果。 | 一致性模型、LCM |
| [Diffuser](https://arxiv.org/abs/2205.09991) | 论文 | 把轨迹规划写成条件扩散生成，适合和 Diffusion Forcing 的动作-后果一致性做对照。 | Diffusion Forcing、扩散规划 |
| [Decision Diffuser](https://arxiv.org/abs/2211.15657) | 论文 | 从 offline RL 角度理解 diffusion planner、return conditioning 和 trajectory generation。 | Diffusion Forcing、扩散规划 |

## VLM、VLA 与具身智能

| 来源 | 类型 | 本站学什么 | 落到页面 |
| --- | --- | --- | --- |
| [CLIP](https://arxiv.org/abs/2103.00020) | 论文 | 静态图文语义对齐能建立 embedding 空间，但不学习动作后果。 | VLM 架构、视觉 token |
| [The Illustrated CLIP](https://jalammar.github.io/illustrated-clip/) | 可视化博客 | 用批内相似度矩阵和 zero-shot classifier 图解 CLIP，比直接写公式更适合建立第一遍直觉。 | VLM 架构、CLIP |
| [Hugging Face: Vision Language Models Explained](https://huggingface.co/blog/vlms) | 技术博客 | 用 encoder、modality projector、LLM 的流水线讲清 VLM 系统结构，适合作为 connector 解释入口。 | VLM 架构、连接器 |
| [BLIP-2](https://arxiv.org/abs/2301.12597) | 论文 | Q-Former 是视觉信息瓶颈，query 学到什么取决于目标。 | 连接器、视觉 tokenizer |
| [Flamingo](https://arxiv.org/abs/2204.14198) | 论文 | Cross-attention / resampler 让语言模型按需读视觉证据。 | VLM 连接器、长视频 |
| [LLaVA](https://llava-vl.github.io/) | 项目页/论文 | 视觉 projector + 指令微调能快速得到对话 VLM，但不等于动作模型。 | VLM 架构 |
| [DeepMind RT-2](https://deepmind.google/blog/rt-2-new-model-translates-vision-and-language-into-action/) | 官方博客 | Web VLM 知识如何通过动作 token 接到机器人控制。 | VLA 动作表示、RT-2 |
| [RT-2 paper](https://arxiv.org/abs/2307.15818) | 论文 | 核对 action tokenization、co-fine-tuning、valid action vocabulary、real-time inference、seen/unseen/emergent 评测和失败边界。 | RT-2、VLA 论文精读 |
| [RT-2 project page](https://robotics-transformer2.github.io/) | 项目页 | 用真实机器人视频和结果图理解 web knowledge transfer 进入闭环控制后的表现；系统 claim 仍回论文表格。 | RT-2、VLA 论文精读 |
| [PaLM-E](https://arxiv.org/abs/2303.03378) | 论文 | 作为 RT-2-PaLM-E 底座，帮助理解 embodied multimodal LLM 与 VLA 输出动作 token 的区别。 | RT-2、VLM/VLA 底座 |
| [RT-1](https://robotics-transformer1.github.io/) | 项目页/论文 | 大规模机器人示范、动作 token 和多任务策略训练。 | VLA 数据与策略 |
| [Open X-Embodiment / RT-X](https://robotics-transformer-x.github.io/) | 项目页/论文 | 跨机器人数据异构性、动作标准化和 embodiment 差异。 | VLA 数据、Sim2Real |
| [ALOHA / ACT](https://arxiv.org/abs/2304.13705) | 论文/项目 | action chunking 和 temporal ensembling 如何把短时连续控制做得更平滑，同时降低每步推理压力。 | VLA 动作表示、动作分块 |
| [FAST action tokenizer](https://arxiv.org/abs/2501.09747) | 论文 | 把高频连续机器人动作压成更短离散 token 序列，说明 action tokenization 本身是 VLA 训练效率和控制精度的关键研究对象。 | VLA 动作表示、action token |
| [GR-2](https://arxiv.org/abs/2410.06158) | 论文 | 用 web-scale future video prediction 学动态先验，再用机器人轨迹把未来视觉绑定到动作。 | GR-2、VLA 预训练 |
| [GR-2 project page](https://gr2-manipulation.github.io/) | 项目页 | 看真实机器人任务视频、future video 输出和 CALVIN 长链结果。 | GR-2、机器人操作 |
| [SpatialVLA](https://spatialvla.github.io/) | 项目页/论文 | Ego3D position encoding 和 adaptive action grids 把 3D 空间与动作统计变成 VLA 接口，而不是后处理细节。 | SpatialVLA、VLA 动作表示 |
| [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) | 项目页/论文 | 动作序列可以用扩散建模，多峰轨迹不必被 MSE 平均。 | VLA 策略学习、动作扩散 |
| [Octo](https://octo-models.github.io/) | 项目页/论文 | 通用机器人策略要同时处理数据混合、动作 chunk、多任务条件。 | VLA 数据、policy learning |
| [DROID](https://droid-dataset.github.io/) | 项目页/论文 | 真实机器人数据采集、长轨迹和跨场景覆盖是数据引擎问题。 | 数据引擎、长时记忆 |
| [Depth Anything 3](https://depth-anything-3.github.io/) | 项目页/技术报告 | 用 depth + ray 把任意视角 RGB 转成可融合 geometry；几何接口服务感知状态，不等于动作世界模型。 | Depth Anything 3、机器人几何感知 |
| [Depth Anything 3 GitHub](https://github.com/ByteDance-Seed/Depth-Anything-3) | 官方仓库 | 学习 model zoo、CLI/API、streaming 和 3DGS 导出这些可落地接口，而不是只读论文主图。 | Depth Anything 3、本地 demo |
| [Depth Anything 3 API docs](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/docs/API.md) | 官方文档 | 把论文里的 depth/ray/camera 输出落成 `processed_images`、`depth`、`conf`、`extrinsics`、`intrinsics` 等工程字段。 | Depth Anything 3、几何数据管线 |
| [VGGT](https://arxiv.org/abs/2503.11651) | 论文 | feed-forward 3D geometry 的统一接口；适合和 DA3 的 depth + ray 设计做对照。 | VGGT、Depth Anything 3 |
| [Video Prediction Policy](https://arxiv.org/abs/2412.14803) | 论文 | 未来预测不必完整生成视频；single-step 中间表征可以作为 robot policy 的动态状态。 | VPP、预测视觉表征 |
| [VPP project page](https://video-prediction-policy.github.io/) | 项目页 | 用真实机器人视频、one-step prediction 和 predictive representation 可视化核对方法 claim。 | VPP、具身策略 |
| [World Action Models are Zero-shot Policies](https://arxiv.org/abs/2602.15922) | 论文 | 用 joint video-action prediction、闭环真实观测刷新和系统加速理解 WAM 与 VLA 的差异。 | DreamZero、WAM |
| [DreamZero project page](https://dreamzero0.github.io/) | 项目页 | 用 zero-shot、post-training、cross-embodiment demo 核对 WAM 不是只做离线视频预测，而是接入闭环动作输出。 | DreamZero、WAM |
| [量子位：VLA 和世界模型路线访谈](https://www.qbitai.com/2026/05/424584.html) | 中文行业访谈 | 学习中文表达里的产业问题意识：空间感知、物理数据和路线融合。事实 claim 仍需回到原论文/官方报告。 | 具身路线、数据引擎 |
| [智源社区：具身与世界模型专题峰会观点集锦](https://hub.baai.ac.cn/view/41216) | 中文会议观点整理 | 学习中文长文如何把产业痛点、研究问题和真实机器人数据约束放在同一篇里；技术证据仍回到论文。 | 具身智能路线、世界模型边界 |
| [量子位智库：空间智能研究报告](https://pdf.dfcfw.com/pdf/H3_AP202505141672561095_1.pdf) | 中文行业报告 | 把 VLA、世界模型、空间感知、数据闭环和机器人产业链放进大图景，适合作为“为什么要关心空间智能”的背景材料。 | 具身智能入门、VLA 路线 |

## 世界模型与 JEPA

| 来源 | 类型 | 本站学什么 | 落到页面 |
| --- | --- | --- | --- |
| [World Models](https://worldmodels.github.io/) | 经典项目页/论文讲解 | 用 VAE、MDN-RNN 和 controller 把“在 latent dream 中学习”讲成可视化直觉，适合世界模型入门开篇。 | 世界模型入门、latent rollout |
| [PlaNet](https://arxiv.org/abs/1811.04551) | 论文 | latent dynamics、reward 和 planning 如何从像素控制中分离出来。 | 世界模型入门、PlaNet |
| [PlaNet project page](https://planetrl.github.io/) | 项目页 | 用任务视频、latent dynamics 和 planning 概览把 RSSM + CEM 讲成一条控制链，而不是只看公式。 | RSSM / Dreamer、PlaNet |
| [Dreamer](https://arxiv.org/abs/1912.01603) | 论文 | 在 imagination rollout 中训练 actor/critic。 | Dreamer、RSSM |
| [Dreamer project page](https://danijar.com/project/dreamer/) | 项目页 | 用 latent imagination、behavior learning 和可视化结果解释 Dreamer 为什么不是单纯视频预测。 | RSSM / Dreamer、imagined actor-critic |
| [Google Research Blog: Introducing Dreamer](https://research.google/blog/introducing-dreamer-scalable-reinforcement-learning-using-world-models/) | 官方博客 | 用更适合读者进入的方式解释 Dreamer 的 latent imagination、样本效率和 visual control 证据；具体方法和图仍回到论文。 | Dreamer、论文精读 |
| [DreamerV2](https://arxiv.org/abs/2010.02193) | 论文 | 离散 latent 和 Atari 控制。 | DreamerV2 |
| [DreamerV3](https://arxiv.org/abs/2301.04104) | 论文 | 更通用的世界模型训练 recipe 和 scaling 经验。 | DreamerV3 |
| [Masked World Models for Visual Control](https://arxiv.org/abs/2206.14244) | 论文 | 把 masked visual representation、latent dynamics 和 control success rate 串成实验报告证据链。 | 世界模型实验报告、MWM |
| [A Path Towards Autonomous Machine Intelligence](https://openreview.net/forum?id=BZ5a1r-kVsf) | 论文/路线图 | 世界模型可以预测抽象表示，并通过 actor/cost/planning 接入 agent，而不是只生成未来像素。 | JEPA、H-JEPA、agent 架构 |
| [I-JEPA](https://arxiv.org/abs/2301.08243) | 论文 | 预测 embedding 而不是像素，减少低层纹理负担。 | JEPA、latent prediction |
| [V-JEPA](https://arxiv.org/abs/2404.08471) | 论文 | 3D mask 和 latent video prediction 如何学时空表征。 | V-JEPA、视频状态 |
| [Meta V-JEPA blog](https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/) | 官方博客 | 官方如何解释 JEPA 与世界模型路线；本站要补动作条件和闭环边界。 | JEPA、世界模型 |
| [Meta: V-JEPA 2 launch](https://about.fb.com/news/2025/06/our-new-model-helps-ai-think-before-it-acts/) | 官方博客 | 用“understanding, prediction, planning”讲清世界模型能力形态，同时回到论文确认实验边界。 | V-JEPA 2-AC、世界模型 |
| [Meta V-JEPA 2 research page](https://ai.meta.com/research/publications/v-jepa-2-self-supervised-video-models-enable-understanding-prediction-and-planning/) | 官方研究页 | 两阶段：action-free 预训练，再接 action-conditioned planning。 | V-JEPA 2-AC |
| [V-JEPA 2](https://arxiv.org/abs/2506.09985) | 论文 | 用 action-conditioned world model 和 MPC 表格区分“预测表征”与“规划器可消费”。 | V-JEPA 2-AC、世界模型实验报告 |
| [World-in-World](https://arxiv.org/abs/2510.18135) | 论文 / Benchmark | 学习如何把世界模型评测从视觉质量拉到 closed-loop task success 和 standardized action API。 | 世界模型评测、闭环 utility |
| [WorldArena](https://arxiv.org/abs/2602.08971) | 论文 / Benchmark | 把 video perception quality、functional utility 和 human evaluation 组合起来看 embodied world model。 | 世界模型评测、功能用途 |
| [WorldArena 2.0](https://arxiv.org/abs/2605.17912) | 论文 / Benchmark | 截至 2026-06-03 的新评测扩展：vision-only -> visuotactile、offline -> interactive RL environment、simulator-only -> simulated and real-world platforms。 | 世界模型评测、跨平台验收 |
| [WoW-World-Eval](https://arxiv.org/abs/2601.04137) | 论文 / Benchmark | 用感知、规划、预测、泛化、执行和 IDM Turing Test 组织具身世界模型评测。 | 世界模型评测、失败模式 |
| [Genie](https://arxiv.org/abs/2402.15391) | 论文/项目 | 无动作标签学习可交互环境，说明视频先验和 action interface 的缺口。 | Genie、世界模型 |
| [Genie 2](https://deepmind.google/blog/genie-2-a-large-scale-foundation-world-model/) | 官方博客 | 官方世界模型能力形态，适合讨论 demo 与闭环证据边界。 | 世界模型、交互模拟器 |
| [Genie 3](https://deepmind.google/models/genie/) | 官方模型页 | 记录 2026 年交互式世界模型入口；用于能力形态和官方定位，不直接外推到机器人闭环控制。 | 世界模型、交互模拟器 |
| [Towards Video World Models](https://www.xunhuang.me/blogs/world_model.html) | 研究者长文 | 用 state-space video world model、taxonomy 和 simulation approaches 帮读者区分视频生成、状态空间模型和可交互模拟器。 | 世界模型入门、视频世界模型 |
| [Self Forcing](https://arxiv.org/abs/2506.08009) | 论文 | 训练时使用模型自己的自回归 rollout 和 KV cache，直接修 teacher forcing/free-running 分布错配。 | Self Forcing、长视频 rollout |
| [Self Forcing project page](https://self-forcing.github.io/) | 项目页 | 看 rolling KV cache、长视频定性对比和速度/质量结果如何支撑 train-test alignment。 | Self Forcing、视频世界模型 |
| [Diffusion Forcing project page](https://www.boyuan.space/diffusion-forcing/) | 项目页 | 用“时间轴 × 噪声轴”理解 per-token noise schedule，把 next-token、masking 和 full-sequence diffusion 放到同一坐标系。 | Diffusion Forcing、序列扩散 |
| [LingBot-World](https://arxiv.org/abs/2601.20540) | 论文 | 从视频底座到动作条件、因果 rollout、KV cache 和少步蒸馏的工程链路。 | LingBot-World、交互世界模拟器 |
| [LingBot-World GitHub](https://github.com/robbyant/lingbot-world) | 官方仓库 | 核对开源模型、实时 demo、README 说明和系统入口。 | LingBot-World、开源世界模型 |
| [Emu3.5](https://arxiv.org/abs/2510.26583) | 论文 | 用长视频帧、transcripts、SFT 世界任务和 DiDA 说明 native multimodal NTP 可以学习视觉语言 next-state generation。 | Emu3.5、统一多模态世界模型 |
| [BAAI Emu3.5 GitHub](https://github.com/baaivision/Emu3.5) | 官方仓库 | 核对模型发布、推理入口、DiDA 和多模态生成接口。 | Emu3.5、模型仓库 |
| [BAAI/Emu3.5 on Hugging Face](https://huggingface.co/BAAI/Emu3.5) | 模型卡 | 查看模型权重、使用说明和发布状态；能力 claim 仍回到论文。 | Emu3.5、模型卡 |
| [Beyond Language Modeling](https://beyond-llms.github.io/) | 项目页/论文 | 用 controlled exploration 拆开 visual representation、data mixture、world modeling、MoE 和 scaling，而不是把统一多模态能力写成一个笼统结论。 | 原生多模态预训练、NWM |
| [Transfusion](https://arxiv.org/abs/2408.11039) | 论文 | 文本 next-token prediction 与视觉 diffusion/flow matching 可以共存在同一 Transformer 目标里。 | 原生多模态预训练、视觉生成 |
| [Navigation World Models](https://arxiv.org/abs/2412.03572) | 论文 | action-as-text 可以作为低成本接口检验动作条件未来预测，但不等同于机器人连续控制。 | 世界模型、动作条件预测 |
| [RLVR-World](https://arxiv.org/abs/2505.13934) | 论文 | 用可验证 reward 直接优化世界模型预测，提醒 surrogate video loss 可能和下游任务错位。 | 世界模型、RLVR |
| [Diffusion Transformers with Representation Autoencoders](https://arxiv.org/abs/2510.11690) | 论文 | 语义视觉 latent + decoder 的 RAE 路线，适合解释为什么视觉表示不只是重建像素。 | RAE、多模态视觉表示 |
| [BAGEL](https://bagel-ai.org/) | 项目页/论文 | bottleneck-free shared attention、interleaved data 和 flow-based visual generation 说明统一多模态底座的可能性；world navigation 仍是 qualitative evidence。 | BAGEL、统一多模态世界模型 |
| [Unified-IO 2](https://arxiv.org/abs/2312.17172) | 论文 | 用统一 token 接口处理多模态输入输出，适合作为 BAGEL 的离散多模态路线对照。 | BAGEL、统一多模态建模 |

## 技术报告与系统卡

| 来源 | 类型 | 本站学什么 | 落到页面 |
| --- | --- | --- | --- |
| [OpenAI: GPT-4o System Card](https://openai.com/index/gpt-4o-system-card/) | 官方系统卡 | 端到端 omni model 上线时，评估单位要从文本回答扩展到语音身份、输出声音、实时拦截和跨模态迁移。 | GPT-4o System Card |
| [OpenAI: GPT-4 System Card](https://openai.com/research/gpt-4-system-card) | 官方系统卡 | 对比文本主导系统卡如何组织 capability、safety 和 deployment mitigation。 | GPT-4o System Card、系统卡读法 |
| [OpenAI: GPT-4V System Card](https://openai.com/index/gpt-4v-system-card/) | 官方系统卡 | 视觉输入安全、隐私、grounding 和多模态失败模式如何进入系统评估。 | GPT-4o System Card、视觉安全 |

## 中文材料的使用边界

中文技术文章对本站非常重要，因为它们常常把复杂工程问题讲成读者能进入的真实困惑。但本站引用它们时遵守三条边界。

中文转载和访谈只作为读法、路线和问题意识参考；事实、数字、图和性能证据回到原论文、官方博客、项目页或代码仓库。
