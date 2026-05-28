# 量化路线图

量化不是把模型文件压小这么简单。它更像给大模型重新分配数值预算：哪些张量可以用更少 bit 表示，哪些层必须保高精，哪些收益会被 kernel、KV cache、反量化和服务调度吃掉。

<!-- [补充] -->
| 读者 | 建议读法 | 工程落点 |
| --- | --- | --- |
| 初学者 | 先抓“量化路线图”要解决的问题、关键术语和最小例子；公式只看变量方向。 | 能说清输入、输出和常见失败信号。 |
| 工程师 | 直接对照结构、指标、成本和边界条件，必要时回到前置页补概念。 | 落到数据、算力、延迟、显存、评测和回滚检查。 |


!!! note "读法定位"
    这页先回答“量化路线图”在「量化」里的位置：它解决什么局部问题，依赖哪些前置，最后会影响哪类工程或研究判断。
    **前置**：先懂张量、线性层和基本推理成本；遇到 FP8、KV Cache、outlier 时回前置页补概念。 必要时先看本页路线，或回 [基础知识](../foundations/index.md) 或 [术语表](../glossary/index.md)。
    **主线关系**：把数值格式、误差来源、校准/训练方法、kernel 和服务部署连成一条效率链，而不是只比较 bit 数。

![SmoothQuant memory gap](../assets/images/paper-figures/quantization/smoothquant-figure-1-memory-gap.png){ width="740" }

<small>图源：[SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438)，Figure 1。原论文图意：模型规模增长速度明显快于单卡 GPU 显存增长速度，量化首先是在解决模型部署的显存和带宽压力。</small>

!!! note "合并提示"
    **note · 初学者先抓住**

    先不要问“INT4、FP8、GPTQ、AWQ 谁最好”。先问四件事：压的是权重、激活还是 KV cache；误差会不会伤关键任务；runtime 有没有对应 kernel；端到端延迟、吞吐和显存是否真的改善。

    **warning · 最常见误解**

    `4bit` 只说明每个量化值本身更小，不自动说明系统更快。真实服务里还会有 scale 元数据、packing、dequant、kernel fallback、KV cache、batch 调度和质量回归。

## 量化到底在做什么

<!-- [融合：算法基础] 来源：llm-algo-leetcode《20. Quantization W8A16 | 模型量化基础: INT8 绝对最大值量化与反量化 (Quantization)》-->
> **算法视角**：量化可以看成在权重或激活旁边维护 scale / zero-point，把高精度矩阵乘改写成“低比特存储 + 反量化或融合计算”的数据布局问题。
>
> 延伸阅读：[20. Quantization W8A16 | 模型量化基础: INT8 绝对最大值量化与反量化 (Quantization)](https://datawhalechina.github.io/llm-algo-leetcode/02_PyTorch_Algorithms/20_Quantization_W8A16.html)

<!-- [融合：复杂度分析] 来源：llm-algo-leetcode《20. Quantization W8A16 | 模型量化基础: INT8 绝对最大值量化与反量化 (Quantization)》-->
| 操作 | 时间复杂度 | 空间复杂度 | 说明 |
|------|-----------|-----------|------|
| 逐元素量化 / 反量化 | $O(N)$ | $O(N)$ 或按 block 存 scale | 关键收益通常来自带宽和存储减少，而不是渐进复杂度下降；参考：[对应章节](https://datawhalechina.github.io/llm-algo-leetcode/02_PyTorch_Algorithms/20_Quantization_W8A16.html) |
| 量化 GEMM | $O(MNK)$ | 权重侧约按 bit-width 缩放 | 反量化若能融合进 matmul epilogue，才更容易转成端到端收益。 |

最常见的均匀量化可以写成：

\[
q = \mathrm{clip}\left(\mathrm{round}\left(\frac{x}{s}\right)+z,\ q_{\min},\ q_{\max}\right)
\]

<!-- [补充：符号说明] -->
| 符号 | 含义 | 取值范围 / 常见值 |
| --- | --- | --- |
| \(q\) | 公式中的变量或中间量 | 按上下文定义 |
| \(s\) | 状态、序列长度或分数 | 状态空间/正整数/实数标量，依上下文而定 |
| \(z\) | 样本、状态或 latent 变量 | 张量空间，形状由数据表示决定 |
| \(q_{}\) | 公式中的变量或中间量 | 按上下文定义 |


反量化写成：

\[
\hat{x}=s(q-z)
\]

<!-- [补充：符号说明] -->
| 符号 | 含义 | 取值范围 / 常见值 |
| --- | --- | --- |
| \(\hat{x}\) | 样本、状态或 latent 变量 | 张量空间，形状由数据表示决定 |
| \(s\) | 状态、序列长度或分数 | 状态空间/正整数/实数标量，依上下文而定 |
| \(q\) | 公式中的变量或中间量 | 按上下文定义 |
| \(z\) | 样本、状态或 latent 变量 | 张量空间，形状由数据表示决定 |


**读作什么**：先用 scale 把浮点数除成“第几个格子”，四舍五入到整数，再限制到低比特能表示的范围；计算时再用 scale 把它近似还原。

**直觉例子**：如果 \(s=0.1\)，那么 \(x=1.23\) 会落到大约第 12 个格子，反量化后可能变成 \(1.2\)。量化省了存储，但丢了 \(0.03\) 的细节。

更完整的符号、shape 和 KV 显存公式放在 [量化符号与最小数学地图](symbols-and-minimal-math.md)。读后面的公式前，建议先把这页当速查表。

## 一条更顺的阅读路径

| 阶段 | 先读什么 | 读完要能回答 |
| --- | --- | --- |
| 建立直觉 | 本页、[符号与最小数学地图](symbols-and-minimal-math.md)、[论文图解导读](page-by-page-visual-guide.md) | 量化压什么、公式变量是什么、论文图该怎么看 |
| 方法入门 | [量化方法对照表](comparison-table.md)、[PTQ/GPTQ/AWQ/SmoothQuant](ptq-gptq-awq-smoothquant.md) | 每种方法在保护哪类误差 |
| 难点补课 | [激活离群值与校准策略](activation-outliers-and-calibration-strategies.md) | 为什么 activation 比权重难量化 |
| 训练与适配 | [QLoRA 与量化训练](qlora-and-quantized-training.md)、[QAT、Kernel 与 KV Cache](qat-kernels-and-kv-cache.md) | 什么时候 PTQ 不够，什么时候要训练时适应 |
| 系统落地 | [FP8 与混合精度推理](fp8-and-mixed-precision-serving.md)、[服务栈与硬件](serving-stacks-and-hardware-tradeoffs.md)、[运行时框架](quantization-runtimes-and-frameworks.md) | 量化收益能不能被硬件和 runtime 兑现 |
| 上线验收 | [多模态与 VLA 量化](quantization-for-multimodal-and-vla-models.md)、[评测与部署清单](evaluation-and-deployment-checklist.md) | 关键任务、长上下文和回退路径是否过关 |

!!! tip "怎么避免读乱"
    量化名词不在同一层。`INT4/FP8/NF4` 是数值格式，`GPTQ/AWQ/SmoothQuant` 是误差控制方法，`W4A16/W8A8` 是权重和激活的精度组合，`Marlin/TensorRT-LLM/vLLM` 才是具体执行路径。

## 先分清四层

| 层级 | 它回答的问题 | 例子 | 初学者常犯错 |
| --- | --- | --- | --- |
| 数值格式 | 数字怎么存 | INT8、INT4、FP8、NF4、MXFP | 只看 bit 数，不看动态范围和 scale |
| 量化粒度 | 一个 scale 管多大范围 | per-tensor、per-channel、group-wise、per-token | scale 越细越准，但元数据和 kernel 越贵 |
| 误差控制算法 | 怎么让输出尽量不变 | PTQ、GPTQ、AWQ、SmoothQuant、QAT、QLoRA | 把算法效果当成所有硬件都能兑现 |
| 系统执行 | 谁真正跑低比特 | vLLM、SGLang、TensorRT-LLM、ONNX Runtime、GEMM kernel | 模型文件小了，却仍然反量化回 FP16 算 |

一个量化方案的完整描述应该像这样：

```text
Qwen 7B, W4A16, group-wise INT4 weight-only, AWQ calibration,
served by vLLM on A100, FP16 KV cache, measured by TTFT/TPOT and task buckets.
```

这句话比“用了 4bit”有用得多，因为它同时说明了压缩对象、算法、runtime、硬件和验收口径。

## 为什么权重、激活和 KV cache 要分开看

| 对象 | 什么时候出现 | 好不好量化 | 主要收益 | 主要风险 |
| --- | --- | --- | --- | --- |
| 权重 \(W\) | 模型加载后常驻 | 相对容易 | 模型显存、权重带宽 | dequant 开销、敏感层掉点 |
| 激活 \(X\) | 每次前向按输入生成 | 更难 | activation 带宽、W8A8 kernel | outlier、动态分布、校准不准 |
| KV cache \(K,V\) | 自回归生成时逐 token 累积 | 任务相关 | 长上下文并发显存、decode 带宽 | 长程记忆、检索、推理一致性受损 |

**读作什么**：权重量化解决“模型放不放得下”，激活量化解决“低比特算子能不能真的跑起来”，KV cache 量化解决“长上下文和高并发能不能撑住”。

!!! example "小账：为什么 KV cache 会反客为主"
    7B 模型的 BF16 权重约 \(7\times10^9\times2\approx14GB\)。但长上下文服务里，KV cache 会随 batch、上下文长度、层数和 KV 宽度线性增长。权重已经 INT4 后，并发仍然上不去，常常就是 KV cache 成了新瓶颈。

## 主要方法放在同一张地图里

| 方法 | 主要解决什么 | 初学者理解 | 常见用法 |
| --- | --- | --- | --- |
| 基础 PTQ | 训练后直接量化 | 便宜快速的 baseline | 先看能否压缩和基本可用 |
| GPTQ | 权重量化误差补偿 | 量一列，补后面的列 | 大模型 weight-only 4bit |
| AWQ | 保护被激活放大的权重通道 | 真正常用的通道要更小心 | 低成本部署、边端和在线服务 |
| SmoothQuant | 把 activation outlier 难度迁移到 weight | 把难搬的货挪到更好处理的一边 | INT8 weight+activation |
| QLoRA | 量化底座上做低资源微调 | 底座冻结，训练小 adapter | 领域适配、对话格式适配 |
| QAT | 训练时模拟量化误差 | 让模型提前适应低比特噪声 | 高质量门槛或极低比特 |
| FP8/混合精度 | 硬件友好的低精度服务或训练 | 不全压，敏感部分保高精 | Hopper/Blackwell 等新硬件 |
| KV cache quant | 压缩在线缓存 | 长上下文的动态显存账 | 长文档、多轮、高并发服务 |

## 量化是否成功，要同表验收

| 要证明什么 | 最小指标 | 不能用什么替代 |
| --- | --- | --- |
| 质量没伤关键任务 | 任务桶、长上下文、代码/数学/工具、多模态专项回归 | 只看平均 benchmark |
| 系统真的更省 | 峰值显存、TTFT、TPOT、吞吐、P95/P99、单位 token 成本 | 只看 checkpoint 文件大小 |
| kernel 路径命中 | profiler、kernel trace、dequant 时间、fallback 次数 | 文档里写“支持 INT4” |
| 可部署可回滚 | 校准集记录、量化参数版本、runtime 版本、全精回退或混合精度保护 | 一次离线脚本成功 |

!!! failure "最佳反例"
    权重从 FP16 压到 INT4，模型文件小了 4 倍，但线上 `TPOT` 没降，因为 runtime 每层都先 dequant 到 FP16 再做 GEMM；同时长文档任务掉点，因为 KV cache 仍然高精且并发被卡住。这不是有效上线，只是有效压缩。

## 多模态和 VLA 为什么要更谨慎

文本聊天模型量化后偶尔措辞变差，用户还可能追问；文档 VLM 少看一个小数点，金额可能错一百倍；VLA 动作头偏一点，夹爪可能撞到杯沿。量化误差在多模态和控制系统里更容易被局部细节放大。

所以多模态/VLA 量化要额外保护：

| 模块 | 默认策略 | 重点验收 |
| --- | --- | --- |
| vision encoder | 可先试温和量化 | 小字、边缘、表格、UI 元素 |
| projector/connector | 更保守 | 图文对齐、grounding、OCR |
| LLM 主体 | 可用成熟 W-only 或 FP8 | 文本任务桶、工具调用 |
| action/risk head | 优先保高精 | 动作抖动、near-miss、闭环成功率 |
| KV/memory | 单独评估 | 长程一致性、检索、rollout drift |

## 本页结论

量化的核心不是“压到最低 bit”，而是在目标硬件、目标 runtime 和目标任务里，把误差放到模型最不敏感、系统最能兑现收益的位置。读任何量化论文或框架文档时，都用这四句检查：

1. 它压的是权重、激活、KV cache，还是训练状态？
2. 它用什么 scale 粒度和校准数据控制误差？
3. 它是否需要特定 kernel、硬件或 runtime 才能变快？
4. 它用哪些任务桶证明低比特版本真的可用？

这四句答清楚，再讨论 GPTQ、AWQ、SmoothQuant、QLoRA、FP8 或 QAT，路径就不会乱。

!!! tip "下一站"
    - 按导航顺序继续：[量化符号与最小数学地图](symbols-and-minimal-math.md)。
    - 概念或符号卡住时，先查 [术语表](../glossary/index.md)，再回到当前页。

## 延伸阅读

1. [量化符号与最小数学地图](symbols-and-minimal-math.md)
2. [量化论文图解导读](page-by-page-visual-guide.md)
3. [Low-bit LLM Survey：低比特大模型从格式到推理系统](../paper-deep-dives/inference/low-bit-llm-survey.md)
