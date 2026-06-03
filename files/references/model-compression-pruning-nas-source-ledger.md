# 模型压缩、剪枝与 NAS 来源台账

这份台账服务于 [模型压缩、剪枝与 NAS](../foundations/model-compression-pruning-and-nas-basics.md)。正文不把压缩写成“方法清单”，而是围绕一个判断问题讲：某个压缩方法到底改了哪条成本链，目标硬件能不能把它变成真实收益。

第一类来源是经典压缩和剪枝论文。Deep Compression 提供 pruning、trained quantization、Huffman coding 的三段式存储压缩框架；Lottery Ticket Hypothesis 提醒读者稀疏子网络现象和部署加速不是同一件事；DepGraph 和结构化剪枝资料支撑正文关于 dependency graph 的讲法；Comprehensive Study of Structural Pruning 用来提醒统一评测口径的重要性。

第二类来源是硬件、框架和 runtime 文档。NVIDIA structured sparsity、TensorRT sparsity、cuSPARSELt 和 PyTorch semi-structured sparsity 资料帮助本站解释：稀疏率、mask 和参数删除只有被硬件、kernel、runtime 支持时，才可能转成真实 latency。PyTorch pruning tutorial 用于核对 mask、parameter reparameterization 和 pruning API 的工程语义。

第三类来源是 NAS 与平台感知搜索。MnasNet 和 Once-for-All 帮助本站说明 search space、真实 latency、supernet、progressive shrinking 和子网选择之间的关系。正文不把 NAS 写成神秘自动机器，而是写成“在给定空间与预算下做结构选择”。

第四类来源是系统案例。Fast-FoundationStereo 是本文主要案例，因为它把 backbone distillation、cost filtering blockwise NAS、refinement structured pruning 和 TensorRT runtime 放在同一条实时视觉 pipeline 里，能展示“按瓶颈选择压缩方法”的完整逻辑。

第五类来源是高质量博客、中文技术媒体和公众号式长文。NVIDIA 技术博客同时是官方事实来源和优秀工程写法参考；智源、OneFlow、机器之心、量子位等中文材料适合学习如何从部署困惑、指标冲突和读者痛点进入问题。正文里的事实、公式、图片和性能结论仍以论文、官方文档、项目页和可稳定访问的技术博客为准；中文材料主要学习讲法，不替代强证据。

## 主要来源

| 来源 | 类型 | 本站使用方式 | 关联主题 |
| --- | --- | --- | --- |
| [Deep Compression](https://arxiv.org/abs/1510.00149) | 论文 | 支撑 pruning、trained quantization、Huffman coding 三段式压缩，以及存储/带宽视角。 | 压缩总览 |
| [The Lottery Ticket Hypothesis](https://arxiv.org/abs/1803.03635) | 论文 | 用来说明稀疏子网络现象，但强调它不能直接证明部署加速。 | 稀疏与剪枝 |
| [DepGraph: Towards Any Structural Pruning](https://arxiv.org/abs/2301.12900) | 论文 | 支撑 dependency graph 和结构化剪枝按组删除的解释。 | 结构化剪枝 |
| [A Comprehensive Study of Structural Pruning for Vision Models](https://arxiv.org/abs/2406.12315) | 论文 / benchmark | 支撑结构化剪枝需要统一 benchmark、指标和任务口径的判断。 | 剪枝评测 |
| [NVIDIA Structured Sparsity](https://developer.nvidia.com/blog/structured-sparsity-in-the-nvidia-ampere-architecture-and-applications-in-search-engines/) | 官方技术博客 | 解释 2:4 稀疏、稀疏 Tensor Core、TensorRT / cuSPARSELt 路径与训练恢复。 | 硬件稀疏 |
| [NVIDIA TensorRT sparsity](https://developer.nvidia.com/blog/accelerating-inference-with-sparsity-using-ampere-and-tensorrt/) | 官方技术博客 | 补充 TensorRT 如何消费 2:4 sparsity，以及稀疏推理需要训练恢复和 runtime 支持。 | runtime |
| [cuSPARSELt structured sparsity](https://developer.nvidia.com/blog/exploiting-ampere-structured-sparsity-with-cusparselt/) | 官方技术博客 | 支撑 2:4 sparse matrix multiply、metadata、prune/compress utilities 和 auto-tuning。 | 稀疏 GEMM |
| [PyTorch Pruning Tutorial](https://docs.pytorch.org/tutorials/intermediate/pruning_tutorial.html) | 官方教程 | 支撑 mask、reparameterization 和 pruning API 的工程落地。 | 剪枝实现 |
| [PyTorch Semi-Structured Sparsity](https://pytorch.org/blog/accelerating-neural-network-training/) | 官方工程博客 | 支撑 2:4/N:M structured sparsity 在训练和矩阵乘中的工程边界。 | 半结构稀疏 |
| [MnasNet](https://arxiv.org/abs/1807.11626) | 论文 | 支撑平台感知 NAS 和真实移动端 latency 目标。 | NAS |
| [Once-for-All](https://arxiv.org/abs/1908.09791) | 论文 | 支撑 supernet、progressive shrinking 和多设备子网选择。 | NAS / supernet |
| [Fast-FoundationStereo](https://arxiv.org/abs/2512.11130) | 论文 | 使用 Figure 3、Figure 5、Figure 8 说明分段加速、依赖图剪枝和 blockwise NAS。 | 系统案例 |
| [Fast-FoundationStereo project page](https://nvlabs.github.io/Fast-FoundationStereo/) | 项目页 | 补充实时 stereo foundation model acceleration 的项目背景和可视化材料。 | 系统案例 |
| [智源/OneFlow：全栈 Transformer 推理优化](https://hub.baai.ac.cn/view/33449) | 中文工程长文 / 转载 | 学习按硬件、算子、显存、KV cache、调度拆解系统优化的中文讲法。 | 中文讲法 |

## 写作边界

正文不把参数减少当作压缩成功。成功要看目标资源：权重体积、activation memory、KV cache、带宽、端到端 latency、P99、能耗或移动端安装大小。

正文不把稀疏率当作加速率。稀疏模式必须匹配硬件和 kernel，例如 2:4 structured sparsity；任意 unstructured sparsity 往往还需要专门 sparse kernel 才能收益。

正文不把 NAS 写成“自动发现最优网络”。NAS 只能在给定 search space、数据、硬件和预算里搜索，proxy 和真实端到端之间仍需要验证。

正文不把压缩写成单点技巧。现实系统常常混用蒸馏、量化、剪枝、NAS 和 runtime 优化，并且每一段热路径使用的方法不同。

正文不把中文技术文章、公众号或转载内容当成强事实来源。它们主要帮助本站学习“如何开篇、如何设置读者困惑、如何把术语放进部署场景”；性能数字、图和实验结论仍回到论文、官方文档、项目页或官方博客核对。

## 图片使用

正文复用了三张仓库已有的 Fast-FoundationStereo 图片，没有新增图片，也没有使用 image2 或任何图像生成：

| 图片 | 正文用途 | 边界 |
| --- | --- | --- |
| `figure-5-refinement-pruning.png` | 说明 refinement/ConvGRU 的 recurrent dependency 会约束结构化剪枝。 | 不证明所有结构化剪枝都能稳定加速。 |
| `figure-8-blockwise-search.png` | 说明 blockwise NAS 在 latency budget 下优于随机候选。 | 只说明该论文设置下的搜索案例。 |
| `figure-3-framework-padded.png` | 说明同一系统不同模块可混用 distillation、NAS、pruning 和 runtime。 | 不把 Fast-FoundationStereo 写成所有视觉模型的默认压缩流程。 |

这些图只承担机制讲解。正文不把论文图中的速度/精度关系外推到别的硬件、数据集或任务。

## 与站内其他台账的分工

[模型蒸馏基础](../foundations/model-distillation-basics.md)和相关台账负责 teacher/student、logit/feature/relation/rationale distillation。本台账只把蒸馏作为压缩工具之一，不展开完整蒸馏方法谱系。

[FP8 与混合精度推理来源台账](fp8-mixed-precision-serving-source-ledger.md)和量化页面负责 dtype、scale、KV cache、fallback 和质量桶。本台账只讨论量化在“压缩成本链”里的位置。

[CUDA 编程模型与内存层次来源台账](cuda-programming-memory-source-ledger.md)、[Reduction、Norm 与 Layout 来源台账](reduction-norm-layout-source-ledger.md)负责 kernel、layout、fusion 和小算子。本台账引用这些口径来解释为什么参数减少不一定等于端到端变快。

[推理运行时来源台账](serving-runtimes-source-ledger.md)和 [KV、缓存与投机解码来源台账](caching-routing-speculative-source-ledger.md)负责 serving lifecycle、KV cache、routing 和 speculative decoding。本台账只在压缩上线门禁里引用这些系统成本。

## 后续维护

维护这页时优先补三类来源：一是官方硬件/runtime 文档中关于 structured sparsity、TensorRT、cuSPARSELt、PyTorch sparse path 的新说明；二是大模型压缩论文中真正报告端到端 latency、P95/P99、质量桶和 fallback 的工作；三是高质量中文博客和公众号式长文中更适合中文读者的入口讲法。新增中文材料时只记录“学到的讲法”，不要直接搬二手性能数字。
