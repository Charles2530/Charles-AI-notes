# 低精度与量化 Kernel

低精度计算和量化落到系统里时，本质上是内核与数据布局问题。很多团队第一次做量化时都会踩到同一个坑：权重文件变小了，理论带宽需求下降了，但端到端吞吐并没有变好，甚至变差。原因往往是 kernel 路径没有真正适配低精度数据格式。

低精度与量化 kernel 的重点是：
**当你已经决定使用 BF16、FP8、INT8、INT4 或权重量化之后，kernel 层到底发生了什么变化，为什么有些格式容易提速，有些则只是在节省存储。**

**初学者先抓住。**

低精度 kernel 的难点不只是位宽小，而是数据要打包、解包、读取 scale、反量化、累加、再写回。只有这些步骤和 Tensor Core、寄存器、缓存、layout 配合好，低精度才会真正变快；如果运行时频繁 dequant、layout 转换或走 fallback，节省的权重带宽会被额外开销抵消。

![Low-bit LLM quantization granularity](../assets/images/paper-deep-dives/inference/low-bit-llm-survey/figure-2-granularity.png){ width="820" }

<small>图源：[A Survey of Quantization Methods for Efficient Neural Network Inference](https://arxiv.org/abs/2103.13630) 与站内低比特综述图整理。原图表达不同量化粒度会改变 scale、group 和张量布局；读图重点是把 group size、scale 读取和 dequant fusion 放回 kernel 路径。它不能证明某个粒度一定最快，因为 runtime、tile shape 和硬件支持会改变结论。</small>

## 低精度不是单一概念

常见路径可以先按“它会改变 kernel 的哪一层”来记：

| 路径 | kernel 里真正变化的部分 |
| --- | --- |
| `FP16 / BF16` | Tensor Core 路径、累加精度、溢出/下溢风险 |
| `FP8` | scaling 元数据、fallback 层、训练/推理硬件支持 |
| `INT8` | activation / weight scale、zero-point、反量化融合 |
| `INT4 / INT3 / INT2` | 权重打包、unpack、group scale、layout 对齐 |
| Weight-only quantization | 权重带宽下降，但主计算路径仍要处理 dequant |
| Activation quantization | 动态范围随 token / batch 变化，online scaling 更重 |
| 训练态量化 | 还要考虑反向、梯度累加和优化器状态 |

## 为什么低精度会让 kernel 设计变复杂

在高精度路径里，一个 GEMM 大致就是读取输入、做乘加、写回输出。低精度路径则会把 `unpack`、scale 读取、zero-point 修正、dequant 甚至 requant 插进主路径里。

于是 kernel 真正执行的就不只是 “matmul”，而更像：

$$
\text{Output}
=
\text{PostProcess}\big(
\text{Matmul}(\text{Dequant}(W), \text{Dequant}(X))
\big).
$$

如果这些步骤不能被很好融合，量化收益就很容易被吃掉。

## BF16、FP16、FP8 的内核差异

### 3.1 FP16

**优点是** Tensor Core 支持成熟、吞吐高、存储减半。

问题是数值范围相对较窄，更容易 overflow / underflow。

### 3.2 BF16

**优点是**指数范围更宽、训练更稳，在很多现代 GPU 上也能走高吞吐路径。

问题是尾数位更少，某些精细累加场景误差会更大。

### 3.3 FP8

FP8 的吸引力在于进一步降低带宽与存储，但它更依赖 scaling 策略、per-tensor 或更细粒度的缩放元数据、原生硬件支持，以及某些层是否仍要保留更高精度。

FP8 真正难的地方并不是“位宽更小”，而是其数值管理与 kernel 协同更难。

## INT8 / INT4 的核心不是整数乘法，而是数据布局

许多初学者会以为量化提速来自“整数乘法更便宜”。这只对了一小部分。真正的收益与代价通常是成对出现的：

| 收益来源 | 额外代价 |
| --- | --- |
| 权重更小、带宽压力更低 | 需要 pack / unpack |
| cache 命中更高 | scale 与 zero-point 读取更频繁 |
| 片上存储能放下更多数据 | layout 更复杂 |
| 同样带宽能搬更多参数 | group size 要和 tile 形状匹配 |

也就是说，量化 kernel 的关键是把“压缩后的参数格式”变成“硬件喜欢的读取与计算路径”。

## Weight-only Quantization 与 Activation Quantization

### 5.1 只量化权重

这是最常见也最容易落地的路线：动态激活不需要复杂在线校准，对推理场景更友好。但 kernel 仍要处理权重 dequant、group-wise scale 和低比特权重的打包布局。

### 5.2 权重和激活都量化

收益可能更大，但复杂度也更高，因为激活是动态变化的，不同 token / batch 的范围差异更大，online scaling 更复杂，某些路径也更难保持数值稳定。

因此全量化 kernel 对系统能力要求更高。

## Group-wise、Channel-wise、Token-wise 的 kernel 含义

很多量化论文从误差角度谈 group size，但从 kernel 角度看，group size 还会影响 scale 读取次数、scale 与主数据的对齐方式、矩阵 tile 是否能整齐匹配 group 边界，以及 dequant 是否能向量化。

### 6.1 Group size 为什么不是纯精度超参

**group 太小**时，精度可能更好，但 scale 元数据更多，kernel 读取和处理 scale 的开销更高。**group 太大**时，kernel 更简单、元数据更少，但量化误差可能变大。

所以 group size 在工程上是精度与 kernel 复杂度的共同折中。

## Dequant 到底该放在哪里

量化 kernel 设计最关键的问题之一是：dequant 发生在什么位置。

### 7.1 先解量化，再用普通 GEMM

**优点**：

实现容易，便于复用现成高精度 kernel。

**缺点**：

中间张量变大，带宽和写回开销高，很可能吞掉量化收益。

### 7.2 在 GEMM 内部边算边解量化

**优点**：

可以把 dequant 融入主计算路径，少一次中间读写，更容易获得真实加速。

**缺点**：

kernel 实现复杂，对 layout、tile、Tensor Core 路径要求更高，autotuning 更难。

现实里，真正高性能的量化推理一般都在朝第二种方向走。

## 量化 GEMM 的常见融合模式

量化推理里高频出现的融合链包括 `weight unpack + dequant + GEMM`、`dequant + GEMM + bias`、`dequant + GEMM + activation`、`dequant + GEMM + residual` 和 `dequant + GEMM + requant`。

### 8.1 为什么这些链特别值得融合

因为如果拆开执行，就会出现中间高精度张量写回、再读回来做下一个操作、多个小 kernel launch 等额外成本，低比特格式带来的带宽优势很容易被抹掉。

## AWQ、GPTQ、SmoothQuant 与 kernel 的关系

这些方法常被当作“量化算法”讨论，但工程里还要追问它们对 kernel 的要求：权重格式是什么、scale 的粒度是什么、是否支持 fused dequant GEMM、是否要求特定 layout，以及更适合 prefill、decode 还是离线 batch inference。

### 9.1 为什么算法论文里的精度结论不能直接转成系统结论

**因为真实服务里最终要问的是** TTFT 是否下降、TPOT 是否下降、显存是否真的省下来、是否因为 dequant 让 kernel 变慢，以及多租户和动态 batch 下是否仍稳定。

所以量化的系统价值，必须经过 kernel 层验证。

## 训练中的低精度 kernel 与推理中的低精度 kernel 不同

| 场景 | 更关注什么 |
| --- | --- |
| 训练 | 前向、反向和优化器状态的精度组合；FP8 / BF16 混合路径；梯度累加精度；数值稳定与 loss scaling |
| 推理 | 权重压缩；KV cache 是否量化；decode 小 batch 下量化 kernel 是否仍然高效；dequant 是否能和主路径融合 |

这意味着同样叫“低精度 kernel”，训练和推理的最优设计目标未必一致。

## KV Cache 量化

随着长上下文服务越来越普遍，KV cache 量化也变得重要。它的目标通常不是大幅提升 FLOPs，而是降低显存占用、增加可服务并发、减轻带宽压力，并延缓 OOM 和 page pressure。

### 11.1 KV 量化为什么格外敏感

因为它直接影响 attention 读取路径，而且 decode 场景非常高频。
若量化误差大，模型质量会明显下降；
若 dequant 太重，又会抵消空间收益。
所以 KV 量化在工程上往往比权重量化更挑剔。

## 低精度 kernel 的常见热点症状

如果量化后没有提速，常见原因包括 dequant 与 GEMM 没融合、group size 与 tile 不匹配、scale 读取成为新瓶颈、layout 转换太多、decode 下 batch 太小，或量化后反而让 kernel 路径更碎。

### 12.1 一个很典型的误判

“模型权重从 16 bit 变成 4 bit，所以理论上应该快 4 倍。”
这在系统里几乎从不成立，因为不是所有时间都花在搬权重，还要访问激活、scale、KV，dequant 和布局处理也要花时间，某些阶段本来就不是 weight-bound。

## 数值稳定性与累加精度

低精度 kernel 一个核心设计问题是：
**输入可以低精度，累加是否也低精度？**

**很多高性能实现会采用**低精度输入、更高精度累加，再按需要写回较低精度输出。

这是因为量化和低精度最容易损伤的常常不是单次乘法，而是长链累加和归约过程。

## 一个更系统的判断框架

看一个低精度 kernel 时，可以依次问：节省的是显存、带宽还是计算；核心瓶颈是 weight movement、activation movement 还是 dequant；scale / zero-point 是否与 tile 对齐；解量化发生在主路径内还是外部；prefill 和 decode 谁受益更大；精度损失是否集中在某些层、某些头或某些长序列场景。

这比只看“多少 bit”更有价值。

## 直觉例子

可以把量化想成把大箱零件压缩装箱。压缩后运货成本变低，但如果到了工厂现场还要先把每个箱子全部拆开、重新分类、再搬到生产线，那运输节省就可能被现场拆箱成本吃掉。真正高效的量化 kernel，等于把拆箱、分拣和装配合并在一条流水线上完成，而不是先压缩、再在现场把所有收益吐回去。

## 读完以后怎么判断

低精度与量化 kernel 的核心，不是“把位宽降下来”，而是让更小的数据格式真正成为更高效的计算路径。这要求算法、数据布局、dequant 位置、tile 设计、scale 粒度和 Tensor Core 路径彼此匹配。

如果没有这层 kernel 视角，量化很容易停留在“离线模型文件更小”；只有把低精度格式与高性能内核真正接起来，它才会变成真实训练和推理系统中的吞吐、并发和成本优势。

## 工程排查

### 排查顺序

低精度优化上线前，先把问题拆成五个坐标：dequant 数据流、scale 布局、epilogue 融合、small-M 场景、精度回退。然后再按下面的顺序排查：

| 检查点 | 该看什么 |
| --- | --- |
| 真实工作负载 | 不只看论文默认 shape，要覆盖 prefill、decode、batch、长短序列混合 |
| 瓶颈层级 | 区分是 GEMM、dequant、layout conversion、runtime dispatch 还是 KV 读写 |
| microbenchmark | 只在端到端关键路径确认后使用，避免只赢单算子 |
| 热路径与尾路径 | 平均吞吐、P95/P99 延迟、小 batch、极端 group size 分开看 |
| 回退条件 | 写清哪些层、哪些 shape、哪些硬件版本要回到 BF16/FP16 |

### 最常见的失败模式

| 现象 | 常见原因 | 优先验证 |
| --- | --- | --- |
| 权重变小但速度不升 | dequant 没融合，或中间高精度张量反复写回 | kernel trace 与显存读写量 |
| INT4 比 INT8 更慢 | unpack / scale 读取压过了带宽收益 | group size、tile 对齐、scale cache |
| decode 加速不明显 | batch 太小，量化 GEMM 吞吐优势发挥不出来 | small-M kernel 与 launch 开销 |
| 精度偶发掉点 | 某些层、头或长序列更敏感 | layer-wise fallback 与全精度对照 |

### 收束口径

真正让低精度 kernel 站得住，不是一次 benchmark 跑得漂亮，而是它能否进入长期维护循环。建议把每次改动绑定四类材料：代表性 probe、端到端案例、已知黑名单 shape / 路径、驱动与编译器升级前后的对照记录。这样量化就不只是“文件更小”，而是能被复验、回退和持续演进的平台能力。

## 外部材料

- [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)。
- [Triton 文档](https://triton-lang.org/main/index.html)。
- [CUTLASS 文档](https://docs.nvidia.com/cutlass/)。
