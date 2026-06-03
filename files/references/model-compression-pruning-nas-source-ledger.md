# 模型压缩、剪枝与 NAS 来源台账

这份台账服务于 [模型压缩、剪枝与 NAS](../foundations/model-compression-pruning-and-nas-basics.md)。正文不把压缩写成“方法清单”，而是围绕一个判断问题讲：某个压缩方法到底改了哪条成本链，目标硬件能不能把它变成真实收益。

## 使用方式

第一类来源是经典压缩和剪枝论文。Deep Compression 提供 pruning、trained quantization、Huffman coding 的三段式存储压缩框架；Lottery Ticket Hypothesis 提醒读者稀疏子网络和部署加速不是同一件事；DepGraph 和结构化剪枝资料支撑正文关于 dependency graph 的讲法。

第二类来源是硬件和框架资料。NVIDIA structured sparsity 博客、PyTorch pruning tutorial 和相关 runtime 文档帮助本站解释：稀疏率、mask 和参数删除只有被硬件、kernel、runtime 支持时，才可能转成真实 latency。

第三类来源是 NAS 与平台感知搜索。MnasNet 和 Once-for-All 帮助本站说明 search space、真实 latency、supernet、progressive shrinking 和子网选择之间的关系。

第四类来源是系统案例。Fast-FoundationStereo 是本文主要案例，因为它把 backbone distillation、cost filtering blockwise NAS、refinement structured pruning 和 TensorRT runtime 放在同一条实时视觉 pipeline 里，能展示“按瓶颈选择压缩方法”的完整逻辑。

中文讲解和公众号类文章主要用于学习组织方式：先给出真实部署困惑，再把指标、公式、图和工程边界串起来。正文里的事实、公式和图片以论文、官方文档、项目页和可稳定访问的技术博客为准。

## 写作边界

正文不把参数减少当作压缩成功。成功要看目标资源：权重体积、activation memory、KV cache、带宽、端到端 latency、P99、能耗或移动端安装大小。

正文不把稀疏率当作加速率。稀疏模式必须匹配硬件和 kernel，例如 2:4 structured sparsity；任意 unstructured sparsity 往往还需要专门 sparse kernel 才能收益。

正文不把 NAS 写成“自动发现最优网络”。NAS 只能在给定 search space、数据、硬件和预算里搜索，proxy 和真实端到端之间仍需要验证。

正文不把压缩写成单点技巧。现实系统常常混用蒸馏、量化、剪枝、NAS 和 runtime 优化，并且每一段热路径使用的方法不同。

## 来源

| 来源 | 类型 | 本站使用方式 | 关联主题 |
| --- | --- | --- | --- |
| [Deep Compression](https://arxiv.org/abs/1510.00149) | 论文 | 支撑 pruning、trained quantization、Huffman coding 三段式压缩，以及存储/带宽视角。 | 压缩总览 |
| [The Lottery Ticket Hypothesis](https://arxiv.org/abs/1803.03635) | 论文 | 用来说明稀疏子网络现象，但强调它不能直接证明部署加速。 | 稀疏与剪枝 |
| [DepGraph: Towards Any Structural Pruning](https://arxiv.org/abs/2301.12900) | 论文 | 支撑 dependency graph 和结构化剪枝按组删除的解释。 | 结构化剪枝 |
| [A Comprehensive Study of Structural Pruning for Vision Models](https://arxiv.org/abs/2406.12315) | 论文 / benchmark | 支撑结构化剪枝需要统一 benchmark、指标和任务口径的判断。 | 剪枝评测 |
| [NVIDIA Structured Sparsity](https://developer.nvidia.com/blog/structured-sparsity-in-the-nvidia-ampere-architecture-and-applications-in-search-engines/) | 官方技术博客 | 解释 2:4 稀疏、稀疏 Tensor Core、TensorRT / cuSPARSELt 路径与训练恢复。 | 硬件稀疏 |
| [PyTorch Pruning Tutorial](https://docs.pytorch.org/tutorials/intermediate/pruning_tutorial.html) | 官方教程 | 支撑 mask、reparameterization 和 pruning API 的工程落地。 | 剪枝实现 |
| [MnasNet](https://arxiv.org/abs/1807.11626) | 论文 | 支撑平台感知 NAS 和真实移动端 latency 目标。 | NAS |
| [Once-for-All](https://arxiv.org/abs/1908.09791) | 论文 | 支撑 supernet、progressive shrinking 和多设备子网选择。 | NAS / supernet |
| [Fast-FoundationStereo](https://arxiv.org/abs/2512.11130) | 论文 | 使用 Figure 3、Figure 5、Figure 8 说明分段加速、依赖图剪枝和 blockwise NAS。 | 系统案例 |
| [Fast-FoundationStereo project page](https://nvlabs.github.io/Fast-FoundationStereo/) | 项目页 | 补充实时 stereo foundation model acceleration 的项目背景和可视化材料。 | 系统案例 |
