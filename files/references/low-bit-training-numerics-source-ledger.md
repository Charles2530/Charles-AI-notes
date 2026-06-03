# 低比特训练与数值格式来源台账

这份台账服务于 [低比特训练与数值格式](../training/low-bit-training-and-numerics.md)。正文的核心问题是：低精度误差会进入反向传播和 optimizer update，最后写回参数；因此低比特训练必须被写成一套训练状态系统，而不是一个 dtype 开关。

## 使用方式

第一类来源是数值格式与 scale。FP8 Formats、NVIDIA Transformer Engine FP8 delayed scaling、Microscaling 和 OCP MX 规范提供 E4M3/E5M2、amax history、per-tensor/per-block scale 和 MX format 的事实基础。正文使用它们时，要把公式讲成“张量如何被 scale 映射到低精度可表示范围”，不要把格式名写成结论。

第二类来源是长跑 FP8 训练。FP8-LM、COAT、Scaling FP8 Training、To FP8 and Back Again、DeepSeek-V3 和 MXFP8 recipes 支撑本站讨论 activation、optimizer state、SwiGLU outlier、scale recipe、累加精度和大规模训练稳定性。正文要特别强调：短跑 loss 重合不等于长 token 预训练稳定。

第三类来源是 FP4/MXFP/NVFP 前沿。MXFP4、Quartet、NVFP4、Four Over Six、Quartet II 等工作适合放在“更低 bit 需要更强系统耦合”下。本站不要把它们写成已经普适替代 FP8/BF16 的成熟方案；更准确的讲法是：FP4 训练需要 block scale、outlier 处理、无偏或随机梯度估计、累加精度保护和硬件 kernel 共同成立。

第四类来源是系统与 checkpoint。DeepSeek-V3、Transformer Engine、ZeRO/FSDP 和 PyTorch/NVIDIA 文档帮助本站解释：低比特训练不是孤立 quantizer，而是一张训练状态图。scale/amax history、loss scaler、optimizer state、RNG、数据游标和并行拓扑都可能影响恢复后的训练轨迹。

第五类来源是中文讲法。NVIDIA 中文技术博客、Datawhale、OneFlow、机器之心、智源社区以及高质量微信公众号长文，适合学习中文技术文章的组织方式：先建立成本账和失败案例，再讲机制，最后给实验读法。事实、公式、版本语义和性能 claim 仍以论文、官方文档或代码仓库为准；微信文章链接稳定性较弱，不作为本站链接校验对象。

## 写作边界

不要把 QLoRA 写成 full low-bit training。QLoRA 的核心是冻结量化底座并训练 adapter，它是高效微调路线；它能帮助解释显存账，但不能直接证明全模型低比特预训练稳定。

不要把 FP8 写成 dtype 开关。FP8 训练应被写成 mixed precision recipe：哪些张量 cast 到 FP8、哪些地方高精度累加、scale 如何统计、敏感模块如何保护、异常如何回退、checkpoint 保存哪些状态。

不要只报告平均 loss 或离线量化误差。低比特训练证据至少要包含 loss spike、梯度范数、activation percentile、amax/scale 饱和率、NaN/Inf、step time、peak memory、kernel profile 和恢复一致性。对于世界模型或 VLA，还要看 action ranking、event divergence、near-miss recall 和闭环收益。

不要把 scale 写成固定的 `max_abs / qmax`。这个公式只能解释最简单的整数量化；FP8/MXFP/FP4 训练里，scale 粒度、延迟更新、block size、格式离散点和 accumulator dtype 都会改变误差形态。

不要把 FP4 与 FP8 的关系写成“再少 4 bit”。FP4 元素太粗，系统收益高度依赖原生 kernel、fused dequant、scale metadata layout 和高精度保护路径。没有这些条件，理论存储节省不一定转化为端到端训练收益。

## 图片使用

本页没有使用 image2，也没有生成新图片；正文只复用了仓库里已经存在的论文图，并在图注中保留图源和读图方法。

| 正文图片 | 本地路径 | 原始来源 | 本站用途 |
| --- | --- | --- | --- |
| FP8 training loss curves | `../assets/images/paper-figures/foundations/fp8-formats-figure-1-training-loss.png` | [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433) Figure 1 | 说明 FP8 可以进入训练，但必须和格式、scale、累加、敏感算子保护一起看。 |
| DeepSeek-V3 FP8 framework | `../assets/images/paper-deep-dives/technical-reports/deepseek-v3/figure-6-fp8-framework.png` | [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) Figure 6 | 说明 FP8 mixed precision 是路径图，不是全局 dtype。 |
| DeepSeek-V3 FP8 accumulation | `../assets/images/paper-deep-dives/technical-reports/deepseek-v3/figure-7-fp8-accumulation.png` | [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) Figure 7 | 说明输入低精度和累加高精度必须同时讨论。 |

## 来源

| 来源 | 类型 | 本站使用方式 | 关联问题 |
| --- | --- | --- | --- |
| [NVIDIA Transformer Engine: FP8 Delayed Scaling](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/fp8_delayed_scaling/fp8_delayed_scaling.html) | 官方文档 | 核对 delayed scaling、amax history、FP8 recipe 和 runtime 状态。 | scale 控制面 |
| [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433) | 论文 / 图源 | 解释 E4M3/E5M2、FP8 训练曲线和格式选择。 | FP8 起点 |
| [Microscaling Data Formats for Deep Learning](https://arxiv.org/abs/2310.10537) | 论文 | 解释 block scale、MXFP/MXINT 和更低 bit 的动机。 | MX 格式 |
| [OCP Microscaling Formats MX v1.0 Spec](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) | 标准文档 | 核对 MX 格式、block scale 和规格边界。 | 标准化 |
| [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) | 技术报告 / 图源 | 使用 FP8 framework 和 accumulation 图解释大规模训练里的 mixed precision 系统。 | 系统共设计 |
| [FP8-LM: Training FP8 Large Language Models](https://arxiv.org/abs/2310.18313) | 论文 | 支撑把梯度和 optimizer state 纳入 FP8 训练讨论。 | 状态链 |
| [COAT: Compressing Optimizer states and Activations for Memory-Efficient FP8 Training](https://arxiv.org/abs/2410.19313) | 论文 | 支撑 activation 与 optimizer states 是低比特训练显存主战场。 | 激活与状态 |
| [Scaling FP8 Training to Trillion-Token LLMs](https://arxiv.org/abs/2409.12517) | 论文 | 支撑长 token 训练、SwiGLU outlier 和短跑稳定不足以证明长跑稳定。 | 长跑稳定 |
| [To FP8 and Back Again](https://arxiv.org/abs/2405.18710) | 论文 | 支撑量化训练稳定性评估与低精度误差回退讨论。 | 稳定性评估 |
| [μnit Scaling](https://arxiv.org/abs/2502.05967) | 论文 | 支撑减少动态 scaling 复杂度和参数化改造的方向。 | 简化 recipe |
| [Recipes for Pre-training LLMs with MXFP8](https://arxiv.org/abs/2506.08027) | 论文 | 支撑 MXFP8 预训练配方和 scale 粒度讨论。 | MXFP8 |
| [Training LLMs with MXFP4](https://arxiv.org/abs/2502.20586) | 论文 | 支撑 FP4 训练需要微缩放、outlier 处理和高精度保护。 | FP4 |
| [Quartet](https://arxiv.org/abs/2505.14669) | 论文 | 支撑 FP4 训练必须和 Blackwell 级原生 kernel/系统实现一起看。 | FP4 kernel |
| [Pretraining Large Language Models with NVFP4](https://arxiv.org/abs/2509.25149) | 论文 | 支撑 NVFP4、细微块和更精确 scale 的前沿讨论。 | NVFP4 |
| [Four Over Six](https://arxiv.org/abs/2512.02010) | 论文 | 支撑 FP4 scale 不是简单贴最大值，离散可表示点会改变最优缩放。 | adaptive scale |
| [Quartet II](https://arxiv.org/abs/2601.22813) | 论文 | 支撑更激进 FP4/NVFP4 训练的无偏梯度估计方向。 | 前沿 |
| [NVIDIA 中文技术博客：FP8 在大模型训练中的应用、挑战及实践](https://developer.nvidia.cn/blog/fp8-llm-app-challenges/) | 中文工程博客 | 学习中文解释节奏：先讲挑战，再讲格式、scale 和实践。 | 中文讲法 |
| [Datawhale 大模型算法教程](https://datawhalechina.github.io/llm-algo-leetcode/) | 中文教程 | 学习把显存账、ZeRO、量化和训练系统讲成可手算知识。 | 中文入口 |
| [OneFlow 官方技术博客](https://blog.csdn.net/OneFlow_Official) | 中文工程博客 | 学习把分布式、kernel、显存和端到端吞吐串起来的写法。 | 工程叙事 |

## 后续写作提醒

写低比特训练时，正文最好每次只围绕一个失效机制展开：scale 被 outlier 拖住、activation cache 伤到反向、optimizer state 误差跨 step 积累、FP4 metadata 抵消收益、checkpoint 恢复 scale 状态不一致。论文可以很多，但段落不能变成论文清单；每篇论文只承担一个解释职责。

如果需要放图，优先选择论文或官方技术报告图，并在图下写清三句话：图源是谁，原图表达什么，本站要读出什么机制。不要把图当装饰，也不要只写“来源：论文 Figure X”。
