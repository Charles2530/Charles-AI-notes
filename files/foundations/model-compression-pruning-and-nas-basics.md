# 模型压缩、剪枝与 NAS 入门

模型压缩解决的是一个朴素但很硬的问题：强模型太慢、太大、太贵，能不能把它变成更适合部署的形态，同时尽量保住质量。

<!-- [补充] -->
| 读者 | 建议读法 | 工程落点 |
| --- | --- | --- |
| 初学者 | 先抓“模型压缩、剪枝与 NAS 入门”要解决的问题、关键术语和最小例子；公式只看变量方向。 | 能说清输入、输出和常见失败信号。 |
| 工程师 | 直接对照结构、指标、成本和边界条件，必要时回到前置页补概念。 | 落到数据、算力、延迟、显存、评测和回滚检查。 |


!!! note "合并提示"
    **note · 读法定位**

    这页先回答“模型压缩、剪枝与 NAS”在「基础知识」里的位置：它把 teacher/student、参数量、MACs、latency 和部署 runtime 接起来。  
    **前置**：先知道张量 shape、feature、loss、梯度和基本推理成本。必要时先看 [模型蒸馏入门](model-distillation-basics.md)、[数值、显存与运行时](numerics-memory-and-runtime-basics.md) 和 [数据划分与评测指标](data-splits-metrics-and-evaluation-basics.md)。  
    **主线关系**：读高效推理论文时，本页帮你判断一个加速方法到底是在压参数、压乘加、改结构、改 runtime，还是只在离线指标上好看。

    **note · 初学者先抓住**

    模型更小不等于一定更快。真实 latency 取决于参数量、MACs、activation、显存带宽、kernel、算子融合、shape、batch 和 runtime。剪枝和 NAS 的目标不是让表格里某个数字变小，而是让真实热路径变便宜。

<!-- [待补图：建议在此插入模型压缩 / 剪枝 / NAS 的真实来源方法图，推荐来源：论文《Deep Compression》Figure 1、《Once-for-All》Figure 1，或 Fast-FoundationStereo 论文的结构搜索 / 剪枝相关原图] -->
<!-- [建议删除：下面 Mermaid 图是旧自动补图，不满足本轮“图解只能引用论文或博客真实图片来源”的约束；保留原文并建议后续用上方真实来源图替换。] -->
```mermaid
graph LR
  A["输入或问题"] --> B["核心概念"]
  B --> C["判断信号"]
  C --> D["工程落点"]
```

## 压缩方法地图

常见压缩和加速方法可以先分成几类：

| 方法 | 改什么 | 常见收益 | 常见风险 |
| --- | --- | --- | --- |
| Distillation | 训练信号 | 让小模型继承大模型能力 | 继承 teacher 错误，student 容量不足 |
| Quantization | 数值格式 | 降低权重/activation/KV 的存储和带宽 | kernel 不支持时未必加速，精度可能掉 |
| Pruning | 删除参数、通道、head、block | 减少计算和显存 | 稀疏不一定有硬件收益，剪错会伤容量 |
| NAS / architecture search | 搜索结构 | 找到更适合 latency budget 的模块 | 搜索 proxy 可能和真实端到端表现不一致 |
| Runtime / kernel optimization | 执行路径 | 降低真实 latency 和 P99 | 依赖硬件、shape 和部署工具链 |

Fast-FoundationStereo 的有趣之处在于它没有只选一个方法：backbone 用蒸馏，cost filtering 用 blockwise NAS，refinement 用结构化剪枝，最后还接 TensorRT 路径。

## Pruning：剪掉什么

剪枝的基本想法是：模型里有些参数、通道或模块对最终质量贡献较小，可以删掉它们，再微调或重训恢复精度。

剪枝通常分两大类：

| 类型 | 剪什么 | 硬件收益 |
| --- | --- | --- |
| Unstructured pruning | 单个 weight | 稀疏率高，但普通 GPU kernel 未必变快 |
| Structured pruning | 通道、filter、head、block、layer | 更容易变成真实 latency 收益 |

为什么 structured pruning 更常用于部署？因为它会改变 dense tensor 的形状。例如把一个卷积层从 128 个输出通道剪到 80 个输出通道，后续 dense convolution kernel 可以直接跑更小 shape。单个 weight 稀疏虽然参数少，但如果 runtime 仍按 dense GEMM / convolution 跑，就不会自动变快。

!!! warning "剪枝不是删完就结束"
    剪枝通常会破坏模型的激活分布和中间接口。真正可用的剪枝流程一般包含：重要性估计、按依赖约束剪结构、短期恢复训练、独立评测和真实 runtime 验证。

## 结构化剪枝为什么要看依赖图

不是所有通道都能独立删除。残差连接、concat、attention head、recurrent hidden state 和多分支网络都会引入依赖。

例如 ConvGRU refinement 里有 hidden state：

```text
h_{k-1} -> update block -> h_k
```

如果上一轮输出的 hidden channels 被剪了，下一轮消费 \(h_{k-1}\) 的输入通道也必须同步调整。这里 \(h_{k-1}\) 表示第 \(k-1\) 轮 refinement 留给下一轮的 hidden state；否则第 \(k\) 轮和第 \(k+1\) 轮的接口对不上。

常见依赖约束包括：

| 结构 | 约束 |
| --- | --- |
| residual add | 两条分支输出 channel 必须一致 |
| concat | 后续层输入 channel 要同步重算 |
| attention head | Q/K/V 和 output projection 常要按 head 成组处理 |
| recurrent state | 输入 hidden 和输出 hidden 的 channel 需要联合剪 |
| fixed output head | 分类数、mask 通道、disparity 输出格式不能乱剪 |

Fast-FoundationStereo 的 refinement pruning 就属于这种情况：final disparity / upsampling mask 的输出格式要固定，ConvGRU hidden state 要按 recurrent dependency graph 约束剪，motion encoder 的 volume feature 接口也不能随意改。

## Taylor Importance 怎么读

剪枝需要判断“删谁伤害最小”。一种常见方法是一阶 Taylor 近似：如果把某个参数或通道移除，loss 大概会变化多少。

对单个参数 \(\theta_i\)，可以粗略写成：

\[
\Delta \mathcal{L}_i \approx \left| \frac{\partial \mathcal{L}}{\partial \theta_i} \theta_i \right|
\]

<!-- [补充：符号说明] -->
| 符号 | 含义 | 取值范围 / 常见值 |
| --- | --- | --- |
| \(\Delta\) | 公式中的变量或中间量 | 按上下文定义 |
| \(\theta_i\) | 模型或策略参数 | 实值张量，形状由模型结构决定 |


读法是：如果参数本身大，且 loss 对它的梯度也大，删掉它可能更伤；如果这个乘积小，它可能不重要。

实际工程里通常不是逐个 weight 剪，而是把一个 channel、filter 或 block 里的重要性聚合起来，再全局排序。剪完后还要 retrain，因为 Taylor 只是一阶近似，不能保证一次剪枝后模型自动恢复。

## NAS：搜索结构而不是手工猜

NAS（Neural Architecture Search）解决的是另一个问题：与其手工决定每个 block 用多宽通道、多大 kernel、几个 attention head，不如定义候选空间，让搜索过程在质量和成本之间找组合。

一个搜索问题通常包含四件事：

| 组成 | 说明 |
| --- | --- |
| Search space | 候选 layer、kernel size、channel scale、head 数、block 数 |
| Proxy / evaluation | 如何快速估计候选结构质量 |
| Cost model | latency、MACs、参数量或显存预算 |
| Selection algorithm | random search、evolution、RL、gradient、ILP 等 |

完整训练每个候选网络通常太贵，所以很多方法会用 proxy。Fast-FoundationStereo 采用的是 blockwise search：把 cost filtering 拆成多个 blocks，分别训练候选 block 去拟合 teacher block，再测每个候选替换进完整模型后的误差变化和耗时变化。

## Blockwise Search 和 ILP

假设 cost filtering 有 \(N\) 个 block。第 \(i\) 个 block 有多个候选 \(c\)。每个候选都测出两个数：

搜索就可以写成一个组合优化问题：

\[
\min \sum_{i,c} \Delta m_i^c e_i^c
\quad
\text{s.t.}
\quad
\sum_{i,c} \Delta t_i^c e_i^c \leq \Delta \tau
\]

<!-- [补充：符号说明] -->
| 符号 | 含义 | 取值范围 / 常见值 |
| --- | --- | --- |
| \(\Delta\) | 公式中的变量或中间量 | 按上下文定义 |
| \(\tau\) | 轨迹或采样路径 | 由策略与环境/数据生成过程决定 |
| \(i\) | 时间步、索引或计数 | 整数索引或归一化时间区间，见正文 |
| \(c\) | 公式中的变量或中间量 | 按上下文定义 |
| \(m_i^c\) | 公式中的变量或中间量 | 按上下文定义 |
| \(e_i^c\) | 公式中的变量或中间量 | 按上下文定义 |


读作：在满足总耗时预算的前提下，选一组候选 blocks，让误差增加尽量小。

ILP（Integer Linear Programming）适合这类选择问题，因为每个候选要么选，要么不选。它不需要端到端训练所有组合，但它也有边界：局部替换的误差变化不一定完全可加，候选之间可能有交互。

## 为什么 MACs 下降不等于 Latency 必然下降

论文表里经常同时给 params、MACs 和 runtime。三者相关，但不是同一个东西。

| 指标 | 表示什么 | 不能直接说明什么 |
| --- | --- | --- |
| Params | 权重存储规模 | 不等于每次推理都触达，也不等于 latency |
| MACs | 某个输入 shape 下的乘加规模 | 不含所有 memory movement、kernel launch、layout 转换 |
| Latency | 真实跑完一次推理的时间 | 依赖硬件、runtime、batch、shape 和实现 |

MACs 降了但 latency 不降，常见原因包括：

- kernel 太小，GPU 利用率下降；
- shape 不适合 Tensor Core 或高效 convolution；
- 算子变碎，kernel launch 和同步开销变多；
- layout 转换、dequant、reshape 或 gather/scatter 吃掉收益；
- activation / memory bandwidth 才是真瓶颈；
- runtime 无法融合新结构，fallback 到慢路径；
- batch、分辨率或 max disparity 和 benchmark 不一致。

所以读高效推理论文时，最有价值的证据不是单独的 MACs，而是 runtime decomposition：每个热路径模块是否真的变快。

## 读消融表时问什么

看剪枝、NAS 或压缩论文时，至少问这几件事：

| 问题 | 为什么重要 |
| --- | --- |
| baseline 是否同硬件同输入 shape | 防止 benchmark 口径不一致 |
| latency 是端到端还是单模块 | 防止局部快但整体不快 |
| 是否给 runtime decomposition | 判断收益来自哪里 |
| 是否 retrain / distill | 剪枝后恢复精度通常依赖训练 |
| 是否有 zero-shot / OOD 测试 | 防止只在目标域过拟合 |
| 是否报告失败区域 | 压缩常先伤长尾和难例 |

## 和 Fast-FoundationStereo 怎么接

| 论文模块 | 本页读法 |
| --- | --- |
| backbone distillation | 学 teacher 中间特征，换掉重 backbone |
| cost filtering blockwise NAS | 搜索 cost volume 处理结构，而不是直接剪小通道模块 |
| refinement structured pruning | 剪迭代 ConvGRU 的冗余通道，并保持 recurrent 依赖 |
| TensorRT runtime | 结构设计要能被部署工具链吃到 |
| runtime decomposition | 验证 feature、cost filtering、refinement 三段都进入真实热路径 |

!!! tip "下一站"
    - 蒸馏细节看 [模型蒸馏入门](model-distillation-basics.md)。
    - 速度和显存口径看 [数值、显存与运行时](numerics-memory-and-runtime-basics.md)。
    - 双目任务背景看 [双目匹配与 Cost Volume 入门](../embodied-ai/stereo-matching-and-cost-volume.md)。
    - 论文案例看 [Fast-FoundationStereo](../paper-deep-dives/inference/fast-foundation-stereo.md)。
