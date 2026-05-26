# 图片资产治理台账

这页记录全站图片资产中“已经本地化但需要维护来源”与“当前未被正式页面引用”的资产。它不是删除清单；默认策略是保留可追溯资产，只把本轮明确替换产生的重复文件作为未来人工清理候选。

<!-- [补充] -->
| 读者 | 建议读法 | 工程落点 |
| --- | --- | --- |
| 初学者 | 先抓“图片资产治理台账”要解决的问题、关键术语和最小例子；公式只看变量方向。 | 能说清输入、输出和常见失败信号。 |
| 工程师 | 直接对照结构、指标、成本和边界条件，必要时回到前置页补概念。 | 落到数据、算力、延迟、显存、评测和回滚检查。 |


!!! note "读法定位"
    这页先回答“图片资产治理台账”在「参考与证据标准」里的位置：它不判断论文结论，只帮助维护图片来源、复用价值和清理边界。
    **前置**：先看 [图片与公式质量标准](image-and-formula-quality-standard.md)，知道图源、裁剪、复杂图解释和公式槽位的最低要求。
    **主线关系**：把审计脚本里的 `remote-image`、`unreferenced-asset` 和 `localized-remote` 输出接成可维护的资产队列。

## 当前资产治理基线

| 项目 | 当前口径 | 处理原则 |
| --- | --- | --- |
| 远程图片 | 正式页面应尽量为 0 | 论文图和公共图都优先本地化；图注保留可回查来源 |
| 本地化远程图 | 放在 `files/assets/images/localized-remote/` | 保留原网页或官方页面链接，不把公共图写成原论文图 |
| 未引用资产 | 由 `make image-quality` 报告 | 默认保留；先记录来源和用途，再决定是否复用或人工清理 |
| 删除资产 | 不作为默认动作 | 只删除本轮替换产生且确认无引用的重复/废弃文件 |

## 公共图本地化记录

| 本地文件 | 来源 / 用途 | 建议动作 |
| --- | --- | --- |
| `localized-remote/embodied-ai/cameras-depth-and-robot-vision/pinhole-camera-model-wikimedia.svg` | Wikimedia Commons；用于解释针孔投影和单目尺度歧义 | 保留，页面图注写“网页图源” |
| `localized-remote/embodied-ai/cameras-depth-and-robot-vision/stereo-camera-model-wikimedia.jpg` | Wikimedia Commons；用于解释双目 baseline、disparity 和深度 | 保留，页面图注写“网页图源” |
| `localized-remote/embodied-ai/cameras-depth-and-robot-vision/intel-realsense-d435-wikimedia.jpg` | Wikimedia Commons；用于展示 RGB-D / depth camera 外观 | 保留，页面图注写“网页图源” |
| `localized-remote/embodied-ai/cameras-depth-and-robot-vision/chessboard-calibration-setup-wikimedia.png` | Wikimedia Commons；用于解释棋盘格标定目标 | 保留，页面图注写“网页图源” |

## 工具与博客截图本地化记录

| 本地文件 | 来源 / 用途 | 建议动作 |
| --- | --- | --- |
| `tools/nsight-systems/gpu-starvation.png` | NVIDIA Technical Blog《Nsight Systems Exposes New GPU Optimization Opportunities》Figure 1；用于解释 GPU starvation 与系统级 trace | 保留，页面图注写“博客截图” |
| `tools/nsight-systems/cpu-gpu-correlation.png` | NVIDIA Technical Blog《Nsight Systems Exposes New GPU Optimization Opportunities》Figure 2；用于解释 GPU event 与 CPU-side CUDA API launch 关联 | 保留，页面图注写“博客截图” |
| `tools/nsight-systems/synchronization-calls.png` | NVIDIA Technical Blog《Nsight Systems Exposes New GPU Optimization Opportunities》Figure 3；用于解释不必要同步调用 | 保留，页面图注写“博客截图” |
| `tools/nsight-systems/nvtx-timeline.png` | NVIDIA Technical Blog《Nsight Systems Exposes New GPU Optimization Opportunities》Figure 6；用于解释 NVTX 标注如何给 timeline 加语义 | 保留，页面图注写“博客截图” |

## 未引用资产分级

| 分级 | 含义 | 默认动作 |
| --- | --- | --- |
| 历史公共概念图 | 早期基础概念图，已有来源记录，但当前正式页面不再使用 | 保留作历史素材，不主动回填页面 |
| 可复用候选 | 官方论文图、项目图或主题素材，未来扩写页面时可能有用 | 保留；复用时补图源和图解 |
| 疑似替换残留 | 与已替换图片或拆图目录高度相关，但未确认可删 | 保留；仅人工确认重复后再删 |
| 待人工归类 | 来源或用途不够明确 | 保留并追溯来源 |

## 重点未引用资产队列

| 资产组 | 文件 | 来源 / 用途 | 建议动作 |
| --- | --- | --- | --- |
| 训练公共概念图 | `training/gradient-descent.svg`、`training/neural-network.svg`、`training/precision-recall.svg`、`training/roc-curve.svg`、`training/confusion-matrix.png` | Wikimedia Commons；早期用于基础训练概念说明 | 历史公共概念图，保留来源记录，不主动回填 |
| 训练公共概念图 | `training/amdahls-law.svg`、`training/mapreduce.svg`、`training/gantt-diagram.svg`、`training/pid-loop.svg` | Wikimedia Commons；早期用于系统和工程管理概念 | 历史公共概念图，保留来源记录，不主动回填 |
| 训练公共概念图 | `training/boxplot-vs-pdf.svg`、`training/histogram-example.svg`、`training/normal-distribution.svg`、`training/pareto-chart.svg`、`training/p-control-chart.svg`、`training/variance-bias.svg` | Wikimedia Commons；早期用于统计和质量管理概念 | 历史公共概念图，保留来源记录，不主动回填 |
| DPM-Solver++ | `paper-deep-dives/diffusion/dpm-solver-plus-plus/table-1-solver-comparison.png` | `DPM-Solver++` 论文表格；可解释 solver family 对比 | 可复用候选，若补算法比较段落再引用 |
| DMD | `paper-deep-dives/diffusion/dmd/imagenet-cond.png`、`laion-more-results.png`、`teaser2.png` | DMD 论文结果图；用于更多样例或补充实验 | 可复用候选，暂不删除 |
| Wan | `paper-deep-dives/diffusion/wan/figure-scaling.png` | Wan 技术报告 scaling 相关图 | 可复用候选，若补 scaling 小节再引用 |
| Nemotron 3 | `paper-deep-dives/technical-reports/nemotron-3-super/figure-6-channel-magnitude.png`、`figure-8-zero-gradient-origin.png` | Nemotron 3 技术报告；用于解释稀疏梯度/通道现象 | 可复用候选，暂不删除 |
| Nemotron 3 | `paper-deep-dives/technical-reports/nemotron-3-super/figure-14-conversation-tool-sft.png`、`figure-15-general-tool-calling.png` | Nemotron 3 技术报告；用于工具调用和 SFT 数据说明 | 可复用候选，暂不删除 |
| 世界模型 survey 拆图 | `paper-deep-dives/world-models/world-models-embodied-ai-survey/Fig1_*.png` | survey Figure 1 拆图；用于分模块解释世界模型 taxonomy | 可复用候选，等 survey 页扩写时统一使用 |
| Masked world models / V-JEPA | `paper-deep-dives/world-models/masked-world-models/table-1-mwm-results.png`、`paper-deep-dives/world-models/v-jepa/figure-5-masking-ablation.png` | 论文结果表和消融图 | 可复用候选，适合补实验小节 |
| Contrastive learning | `paper-figures/contrastive-learning/*.png` | CLIP / BYOL / MoCo / SimCLR / SupCon 相关图 | 可复用候选，若新增对比学习基础页再引用 |
| Reinforcement learning | `paper-figures/reinforcement-learning/*.png` | Decision Transformer、RLVR、SAC 相关图 | 可复用候选，若补 RL 专题页再引用 |
| VLA / VLM | `paper-figures/vla/palme-figure-2-architecture.jpg`、`paper-figures/vlm/*.png` | PaLM-E、ChartQA、LayoutLMv3、Pix2Struct、SeeClick 图 | 可复用候选，若补 VLM/VLA 基础页再引用 |
| Operators | `paper-figures/operators/flashattention-speedup.jpg` | FlashAttention 速度对比图 | 可复用候选，若补 attention kernel 历史再引用 |

## 维护流程

新增或替换图片时按这个顺序处理：

1. 先确认来源类型：论文/官方项目、公共图、截图、自绘或重绘。
2. 本地化图片后，页面图注写清来源和用途；公共图写“网页图源”，论文图写论文和 Figure/Table 编号。
3. 运行 `make image-quality`，看 `Asset governance` 摘要里的远程图、未引用资产和目录聚合。
4. 如果新增图片后来被替换，先查 `rg` 确认无引用，再把它记到本页；不要直接删除旧资产。
5. 只有确认是本轮重复产物、来源可追溯且没有页面引用时，才允许进入人工删除候选。

!!! tip "下一站"
    - 回到 [参考与证据标准](index.md)，确认这页在全站证据链中的位置。
    - 继续看 [图片与公式质量标准](image-and-formula-quality-standard.md)，核对页面图注和公式解释是否达标。
