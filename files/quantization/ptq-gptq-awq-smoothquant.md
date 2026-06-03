# PTQ、GPTQ、AWQ 与 SmoothQuant

量化不是“把 FP16 改成 INT4”。量化真正做的是：用更少的格子近似原来的连续数值，同时尽量让模型输出、服务延迟和长尾任务不要坏掉。一个模型文件变小，只说明存储少了；要证明量化成功，还要看误差怎么被控制，低比特 kernel 是否真的命中，真实请求的质量是否回归。

这篇只回答一个问题：**训练好的大模型为什么可以被压到 8-bit、4-bit，但有时又会突然掉质量或不变快。**

先把常见部署目标放在脑子里。70B 模型的 FP16 权重大约是 140GB，如果只用单机多卡服务，权重常驻显存和 decode 阶段的权重带宽都会很贵。把权重量化到 4-bit，模型可能更容易放进显存，也可能让 decode 读权重更轻。但如果 runtime 没有高效 INT4 GEMM，系统可能先把 INT4 解包/反量化成 FP16 再算，最后只是文件小了，端到端 TPOT 没有明显改善。

所以量化的第一问不是“INT4 还是 INT8”，而是：**瓶颈到底是权重、activation、KV cache，还是 runtime kernel。**

## 量化就是给数轴分格子

最基础的均匀量化可以写成：

\[
q=\operatorname{clip}\left(\operatorname{round}\left(\frac{x}{s}\right), q_{\min}, q_{\max}\right),
\qquad
\hat{x}=s q
\]

\(x\) 是原始浮点数，\(q\) 是低比特整数，\(\hat{x}\) 是反量化后的近似值。\(s\) 是 scale，它决定一个整数格子对应原始数轴上的多大间隔。4-bit 有限整数格子很少；INT4 对称量化常见有效范围可以近似理解为 \([-7,7]\)。如果 \(s=0.1\)，很多小数能被细分，但大值容易被截断；如果 \(s=10\)，大值能装下，普通小值会被压进很少几个格子。

非对称量化会加入 zero point：

\[
q=\operatorname{round}(x/s+z),
\qquad
\hat{x}=s(q-z)
\]

\(z\) 的作用是让浮点 0 对齐到整数格点，适合处理不以 0 为中心的分布。代价是 kernel 实现更复杂，尤其在高性能 GEMM 中，zero point、scale、解包和反量化能否融合，会直接影响性能收益。

量化误差可以粗略分成两类。rounding error 来自连续值被吸到最近格点；clipping error 来自超出可表示范围的值被截断。低比特方法的核心不是消灭误差，而是在这两类误差之间找到对任务最不伤的分配。

举一个小例子。某层 activation 大多数值在 \([-2,2]\)，但某个通道偶尔出现 30。若用 INT4 per-tensor scale，并为了装下 30 选择 \(s=30/7\approx4.29\)，那主体区域 \([-2,2]\) 几乎只剩一两个格子，很多原本不同的值会变得一样。若把 scale 调小，主体区域更清楚，但 30 会被截断。activation outlier 的危险不在于它出现次数多，而在于它决定了很多普通值还能不能被分辨。

## 量化对象不同，问题完全不同

权重、activation、KV cache 和 optimizer state 都可以低比特，但它们的难点不一样。

权重是静态的。模型加载前就知道 \(W\) 长什么样，可以离线统计、分组、搜索 scale，甚至用校准数据估计量化误差对输出的影响。因此 GPTQ、AWQ 这类 weight-only PTQ 能做到很激进的 4-bit。

Activation 是动态的。每个 prompt、每张图、每段上下文都会产生不同的中间张量。它不仅有 outlier，还和 batch、sequence length、position、数据域有关。W8A8 想让权重和 activation 都进入低比特计算路径，收益可能更大，但 scale 和 outlier 处理更难。

KV cache 是自回归推理的历史记忆。长上下文服务中，KV cache 可能比权重更快成为显存瓶颈。KV 量化省的是历史 K/V 的显存和读写带宽，但风险也很具体：attention 读历史时如果 key/value 误差改变排序，模型可能出现跨段引用错误、needle retrieval 失败、长文档后半段幻觉或代码定位错位。

Optimizer state 主要是训练问题。AdamW 的一阶/二阶动量和 master weights 常常比参数本身更占显存，但把它们低精度化会改变更新动力学。这里不展开，训练低比特可以读本站的低比特训练页。

## PTQ 的成败取决于校准

Post-Training Quantization 指模型训练完成后再量化，不重新做完整训练。PTQ 的吸引力是便宜，风险是校准集不代表真实线上输入时，scale、重要通道和敏感层都会估错。

校准不是“随便拿几条样本跑一遍”。校准集要覆盖真实服务里的长度、语言、代码、工具调用、表格、视觉输入、长上下文和高风险失败桶。若只用普通闲聊校准，一个经过量化的模型可能在闲聊和短问答上正常，在数学、代码、长文档引用、函数调用或 OCR 表格上掉得很隐蔽。

粒度也会改变误差形态：

```text
per-tensor:  一整块张量共享 scale，元数据少，outlier 影响范围大
per-channel: 每个输出/输入通道有 scale，误差更低，kernel 更复杂
per-group:   一组权重共享 scale，是 4-bit weight-only 常见折中
per-token:   每个 token 动态 scale，适合 activation，但运行时开销更高
```

同样叫 INT8 或 INT4，粒度、zero point、group size、是否对称、是否动态 scale、是否有 fused kernel 都可能完全不同。读模型卡或部署文档时，要把这些细节当成方法本身的一部分。

## GPTQ：让 4-bit 权重尽量重构层输出

GPTQ 的问题意识是：量化权重时，真正影响模型的不是单个权重的绝对误差，而是量化后这一层输出有没有偏离原模型。

对一个线性层：

\[
Y=XW
\]

若把权重 \(W\) 量化成 \(\hat{W}\)，输出误差可以写成：

\[
\|XW-X\hat{W}\|_2^2
\]

这里 \(X\) 来自校准数据。这个式子解释了 GPTQ 为什么需要校准样本：它要知道真实输入通过这一层时，哪些权重方向的误差更会伤到输出。GPTQ 使用近似 Hessian 信息，按列量化权重，并把当前列造成的误差补偿给尚未量化的列。直觉上，它不是“每个数各自 round 一下”，而是“这一步 round 坏掉的输出，能不能让后面的权重列吸收一点”。

![GPTQ procedure 原论文图](../assets/images/paper-figures/quantization/gptq-figure-2-procedure.png){ width="760" }

<small>图源：[GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)，Figure 2。原图表达 GPTQ 按列量化权重，并用近似 Hessian 信息更新尚未量化的列。本站使用这张图说明：GPTQ 的重点不是压缩率，而是让量化后的层输出尽量接近原模型。</small>

GPTQ 通常用于 weight-only 量化，例如 W4A16：权重以 4-bit 存储，计算时和更高精度 activation 结合。它很适合降低权重显存和 decode 权重带宽，但是否提速取决于目标 runtime 是否有对应 INT4 kernel、权重 layout、group size 和 batch shape 是否匹配。

GPTQ 最常见的坑有三个。第一，校准集太窄，导致 Hessian 近似偏向普通样本。第二，group size 过大，元数据少了但误差变粗。第三，部署栈只是支持“加载 GPTQ 文件”，不代表热点 GEMM 真正走了高效低比特路径。

## AWQ：保护被 activation 放大的权重通道

AWQ 的出发点和 GPTQ 不一样。它观察到 LLM 中不是所有权重同样重要：少数通道会被 activation 强烈使用，如果这些通道对应的权重被量化坏，输出会明显偏移。AWQ 因此用校准 activation 来判断哪些权重通道更值得保护。

![AWQ activation-aware 原论文图](../assets/images/paper-figures/quantization/awq-figure-1-activation-aware.png){ width="760" }

<small>图源：[AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978)，Figure 1。原图表达少量显著 activation 通道对应的权重更需要保护。本站使用这张图说明：AWQ 的关键词是 activation-aware，重要性来自运行时输入如何使用这些通道，而不只来自权重绝对值。</small>

可以把 AWQ 的直觉写成一句话：权重 \(W\) 本身是静态的，但量化误差是否伤模型，要看它被什么 activation \(X\) 乘上。一个权重通道的数值不一定最大，但如果真实输入经常在对应 activation 通道上给出大值，它的误差就会被放大。

AWQ 会搜索一组缩放策略，让重要通道在低比特表示中损伤更小。它不是把所有 outlier 都保高精，也不是只保护权重绝对值最大的通道，而是用校准 activation 识别“对输出更有影响”的通道。这个设计让 AWQ 在 4-bit weight-only 部署中很实用，尤其适合想用相对低成本校准获得较稳质量的场景。

AWQ 的边界也来自它的优点。它主要保护权重，不直接解决 activation 低比特计算、KV cache 量化和多模态 connector 的尺度漂移。若任务包含 OCR、表格、长上下文工具调用或动作输出，只看通用文本 perplexity 可能不够。

## SmoothQuant：把 activation outlier 的难度搬给权重

SmoothQuant 处理的是另一类问题：如果想做 W8A8，让 activation 也低比特，activation outlier 会严重拖累 scale。它的核心利用了线性层的缩放等价性。

对线性层：

\[
Y=XW
\]

选择一个按通道的对角缩放矩阵 \(S\)，可以改写为：

\[
Y=(XS^{-1})(SW)
\]

这不是近似，而是矩阵乘法的等价变换。\(XS^{-1}\) 把难量化的 activation 通道缩小，让 activation 更容易进入 INT8 格子；\(SW\) 把对应权重通道放大，把一部分量化难度转移到静态权重上。权重是离线固定的，更容易做校准和检查；activation 是线上动态生成的，更难救。

![SmoothQuant intuition 原论文图](../assets/images/paper-figures/quantization/smoothquant-figure-2-intuition.png){ width="720" }

<small>图源：[SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438)，Figure 2。原图表达 activation outlier 会拉大量化范围，使主体值失去分辨率；SmoothQuant 通过等价缩放把部分难度迁移到 weight。本站使用它说明：SmoothQuant 的目标不是单独压权重，而是让 activation 也更容易低比特化。</small>

![SmoothQuant main idea](../assets/images/paper-figures/quantization/smoothquant-figure-5-main-idea.png){ width="760" }

<small>图源：[SmoothQuant](https://arxiv.org/abs/2211.10438)，Figure 5。原图表达 per-channel smoothing factor 如何调整 activation 和 weight 的尺度。本站使用它说明：scale 选择是 SmoothQuant 的方法核心，\(\alpha\) 这类超参控制难度从 activation 迁移到 weight 的程度。</small>

SmoothQuant 更适合 W8A8 路线。它想让权重和 activation 都进入 INT8 GEMM，从而在 prefill、大 batch 或计算占比较高的场景获得更多收益。若部署栈没有成熟 INT8 W8A8 kernel，或者 Q/DQ 没有很好融合，SmoothQuant 的数学优势不一定能变成端到端收益。

## 三种方法其实在分配同一份误差预算

GPTQ、AWQ、SmoothQuant 不是三个互斥名词，而是三种误差控制视角。

GPTQ 关心“量化权重后，层输出重构误差能不能小”。它更像二阶/重构视角，适合 weight-only PTQ。

AWQ 关心“哪些权重通道被真实 activation 放大，必须保护”。它更像敏感通道保护视角，也常用于 4-bit weight-only 部署。

SmoothQuant 关心“activation outlier 为什么让 W8A8 难做，能不能把难度迁移到静态权重”。它更像 activation 与 weight 之间的尺度重分配。

因此，选择方法时可以按服务瓶颈倒推：

| 目标 | 更自然的起点 | 重点验证 |
| --- | --- | --- |
| 模型太大，权重显存放不下 | GPTQ / AWQ W4A16 | 模型能否加载、质量是否掉、INT4 kernel 是否命中 |
| Decode 阶段读权重太贵 | GPTQ / AWQ + 高效 serving kernel | TPOT、batch shape、dequant 开销、P95/P99 |
| Prefill 或大 batch GEMM 成本高 | SmoothQuant / W8A8 / FP8 | TTFT、INT8/FP8 GEMM、scale/QDQ 是否融合 |
| 长上下文并发显存大 | KV cache quantization / GQA/MQA / paged cache | needle retrieval、跨段引用、KV 布局和带宽 |
| 多模态或 VLA 部署 | 混合精度、模块级敏感度分析 | OCR/grounding/action head/闭环成功率 |

这个表的重点不是让读者机械选方法，而是提醒：量化收益来自模型误差、硬件 kernel 和请求分布共同闭合。一个在短 prompt 单请求上不掉分的 INT4 模型，到了长上下文 batch serving 可能被 KV cache 卡住；一个在 perplexity 上正常的 W8A8 模型，到了表格 OCR 或工具调用可能因为少数 outlier 桶掉质量。

## 上线前怎么验收

量化后的模型至少要同时验收质量、性能和可运维性。

质量上，不只看平均 perplexity。文本 LLM 要覆盖数学、代码、长上下文、工具调用、安全拒答和多语言；VLM 要覆盖 OCR、表格、坐标、图文细节；VLA 要覆盖动作误差、恢复桶和闭环成功率。校准集和评测集要刻意分开，避免把 scale 调到评测集上。

性能上，不只看模型文件大小。要看 TTFT、TPOT、tokens/s、P95/P99、显存峰值、KV cache 占用、batch 命中率、kernel trace 和 Q/DQ 开销。低比特模型如果大量时间花在解包、反量化、格式转换或 fallback kernel 上，就没有兑现系统收益。

可运维性上，要记录量化配置：方法、bit width、group size、zero point、scale 粒度、校准集 manifest、被跳过的层、runtime 版本、硬件型号、kernel 后端、评测版本。没有这些信息，线上掉分后很难判断是数据分布变了、kernel 路径变了，还是量化配置本来就不稳。

## 常见误解

**误解一：INT4 一定比 FP16 快。**  
INT4 只保证存储更小。是否更快取决于 packing layout、dequant、低比特 GEMM、batch shape 和 runtime 调度。

**误解二：校准集随便取一点就行。**  
校准集决定 scale、重要通道和敏感层。它如果不覆盖线上长尾，量化误差会在代码、数学、长上下文、多模态或动作任务里暴露。

**误解三：weight-only 量化解决所有显存问题。**  
长上下文服务里 KV cache 可能比权重更大；多模态服务里视觉 token、workspace 和 connector 也可能成为瓶颈。

**误解四：perplexity 不掉就能上线。**  
Perplexity 是有用信号，但不能替代任务分桶、工具调用、长文档引用、OCR、闭环控制和安全评测。

## 阅读结论

量化的核心不是把模型压成某个 bit 数，而是把有限的数值表示能力分给最影响任务质量的地方。PTQ 先用校准数据估计张量分布；GPTQ 用近似 Hessian 减少 weight-only 量化后的层输出重构误差；AWQ 用 activation 统计保护高影响权重通道；SmoothQuant 用等价缩放把 activation outlier 的难度迁移到 weight，让 W8A8 更可行。

真正部署时，应该反过来读这些方法：先确认服务瓶颈和硬件/runtime，再决定量化对象，再选校准集、方法和评测桶。只看“INT4/INT8/FP8”标签，最容易得到一个文件更小、性能没涨、长尾任务还掉分的模型。

## 外部精读

- [Hugging Face Quantization concept guide](https://huggingface.co/docs/transformers/quantization/concept_guide)：量化概念、对象和 Transformers 生态入口。
- [Hugging Face: selecting a quantization method](https://huggingface.co/docs/transformers/quantization/selecting)：按硬件、内存和延迟目标选择量化路线。
- [General Compute: Quantization for Inference](https://www.generalcompute.com/blog/quantization-for-inference-gptq-awq-smoothquant-fp8)：用 prefill / decode 瓶颈解释 GPTQ、AWQ、SmoothQuant 和 FP8 的工程分工。
- [PaddleNLP 大模型量化教程](https://paddlenlp.readthedocs.io/en/stable/llm/quantization.html)：中文官方文档，适合看 GPTQ、AWQ 等工具链参数如何落地。
- [GPTQ](https://arxiv.org/abs/2210.17323)：理解 Hessian-aware weight-only PTQ。
- [AWQ](https://arxiv.org/abs/2306.00978)：理解 activation-aware weight protection。
- [SmoothQuant](https://arxiv.org/abs/2211.10438)：理解 activation outlier migration。
- [LLM.int8()](https://arxiv.org/abs/2208.07339)：理解大模型 activation outlier 与 mixed-precision decomposition 的早期重要工作。
