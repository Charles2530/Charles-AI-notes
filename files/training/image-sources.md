# 训练专题图片来源与授权

本页记录 `files/assets/images/training/` 中图片的来源链接。  
这些图片均来自 Wikimedia Commons，具体授权条款以各文件页面为准。

训练专题曾经有一批概括图；现在不再在网页中使用。训练页里的方法图统一改为论文原图、论文项目图或明确来源的公共图。

## 文件清单

1. `gradient-descent.svg`  
   来源：https://commons.wikimedia.org/wiki/File:Gradient_descent.svg
2. `neural-network.svg`  
   来源：https://commons.wikimedia.org/wiki/File:Neural_network.svg
3. `precision-recall.svg`  
   来源：https://commons.wikimedia.org/wiki/File:Precisionrecall.svg
4. `amdahls-law.svg`  
   来源：https://commons.wikimedia.org/wiki/File:AmdahlsLaw.svg
5. `boxplot-vs-pdf.svg`  
   来源：https://commons.wikimedia.org/wiki/File:Boxplot_vs_PDF.svg
6. `mapreduce.svg`  
   来源：https://commons.wikimedia.org/wiki/File:MapReduce.svg
7. `pid-loop.svg`  
   来源：https://commons.wikimedia.org/wiki/File:PID_en.svg
8. `normal-distribution.svg`  
   来源：https://commons.wikimedia.org/wiki/File:Normal_distribution_pdf.svg
9. `histogram-example.svg`  
   来源：https://commons.wikimedia.org/wiki/File:Histogram_example.svg
10. `pareto-chart.svg`  
    来源：https://commons.wikimedia.org/wiki/File:Diagrama_pareto.svg
11. `p-control-chart.svg`  
    来源：https://commons.wikimedia.org/wiki/File:P_control_chart.svg
12. `gantt-diagram.svg`  
    来源：https://commons.wikimedia.org/wiki/File:Gantt_diagramm.svg
13. `roc-curve.svg`  
    来源：https://commons.wikimedia.org/wiki/File:Roc_curve.svg
14. `variance-bias.svg`  
    来源：https://commons.wikimedia.org/wiki/File:Variance-bias.svg
15. `confusion-matrix.png`  
    来源：https://commons.wikimedia.org/wiki/File:Confusion_Matrix.png

## 训练专题论文原图

这些图片位于 `files/assets/images/paper-figures/training/`，用于替换原先的概括图。

| 文件 | 来源 | 用途 |
| --- | --- | --- |
| `chinchilla-isoflop-curves.png` | [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)，Figure 4 | 解释固定 FLOP 预算下参数量和 token 数的配平 |
| `chinchilla-tokens-vs-params.png` | [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)，Figure 15 | 解释 compute-optimal tokens / parameters scaling |
| `zero-memory-stages.png` | [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)，Figure 1 | 解释数据并行状态冗余和 ZeRO 三阶段分片 |
| `gpipe-pipeline-parallelism.png` | [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965)，Figure 2(c) | 解释 micro-batch 如何填充 pipeline bubble |
| `instructgpt-rlhf-pipeline.png` | [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)，Figure 2 | 解释 SFT、reward model 和 PPO/RLHF 的训练数据接口 |
| `instructgpt-labeler-likert.png` | [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)，Appendix Figure 19(a) | 解释单条模型输出评分和元信息标注 |
| `instructgpt-labeler-ranking.png` | [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)，Appendix Figure 19(b) | 解释同题多答排序数据如何训练 reward model |
| `instructgpt-main-preference.png` | [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)，Figure 1 | 解释 RLHF 后训练需要看人类偏好胜率 |
| `instructgpt-preference-facetted.png` | [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)，Figure 4 | 解释不同 prompt 分布和 labeler 分组下的偏好结果 |
| `ppo-algorithm-1.png` | [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)，Algorithm 1 | 解释 PPO 的 actor-critic 式采样、advantage 估计和 surrogate 优化流程 |
| `ppo-clipped-surrogate.png` | [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)，Figure 1 | 解释 PPO clipped surrogate 如何限制策略概率变化 |

训练页还复用了一些其他专题的论文图：

| 文件 | 来源 | 用途 |
| --- | --- | --- |
| `../quantization/qlora-figure-1-memory.png` | [QLoRA](https://arxiv.org/abs/2305.14314)，Figure 1 | 解释低比特微调的显存构成 |
| `../foundations/fp8-formats-figure-1-training-loss.png` | [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)，Figure 1 | 解释 FP8 训练需要通过收敛曲线验证 |
| `../foundations/sublinear-memory-figure-1-computation-graph.png` | [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174)，Figure 1 | 解释 activation checkpointing 的重计算换显存 |
| `../foundations/data-cards-typology.png` | [Data Cards](https://arxiv.org/abs/2204.01075)，typology figure | 解释数据治理和数据文档的多角色视角 |
| `../foundations/loss-landscape-figure-resnet56.png` | [Visualizing the Loss Landscape of Neural Nets](https://arxiv.org/abs/1712.09913) | 解释优化路径和稳定性排查 |
| `../inference/specinfer-workflow.png` | [SpecInfer](https://arxiv.org/abs/2305.09781)，Figure 3 | 解释投机推理和 token tree verification |

## 使用说明

1. 如果你要对外发布（网站、课程、论文附录等），建议在页面底部继续保留来源链接；
2. 若替换图片，请同时更新本页中的文件名与来源；
3. 若需严格法务审阅，请逐个打开来源页确认当前 license 字段。
