# Reduction、Norm 与索引 Kernel

在很多大模型系统里，真正频繁拖慢整体速度的，并不总是最大的 GEMM，而是那些“单次看起来不大、但出现极其频繁”的算子：reduction、softmax、layernorm、rmsnorm、rope、gather、scatter、layout transform。它们通常更偏带宽受限，很适合做融合，对 shape、stride 和向量化极度敏感，也容易在 decode、小 batch 和长上下文场景中放大存在感。

!!! note "读法定位"
    这页先回答“Reduction、Norm 与索引 Kernel”在「算子与编译器」里的位置：它解决什么局部问题，依赖哪些前置，最后会影响哪类工程或研究判断。
    **前置**：先理解张量 shape、显存层次和 GEMM；系统指标卡住时回推理或训练专题。 必要时先回 [算子与编译器入口](index.md)、[基础知识](../foundations/index.md) 或 [术语表](../glossary/index.md)。
    **主线关系**：把模型里的矩阵乘、attention、通信和低精度路径落到 GPU 时间线上，看瓶颈如何被 kernel 与编译栈改变。

因此理解这些小中型 kernel，对于训练和服务都非常关键。

!!! note "初学者先抓住"

    小算子不是小问题。Norm、reduction、gather、scatter、layout transform 单次成本不大，但每层、每步、每个请求反复出现，尤其在 decode 和长上下文里会放大成主瓶颈。

!!! note "难点解释：为什么 layout 变换很贵"

    Layout transform 经常不做太多数学计算，却会搬大量数据、破坏连续访问、增加中间张量和 launch。很多优化的第一步不是写更复杂算法，而是减少不必要的重排。

## 为什么“小算子”不能被低估

单个 norm 或 reduction kernel 看起来没多少 FLOPs，但真实系统里它们每层都要跑、每步 decode 都要跑，前后还常夹着 layout 变换和 mask 处理，因此极易受带宽和 launch 开销影响。

如果这些算子没有被很好融合，系统就会不断读取中间结果、做一点小计算、再写回、再 launch 下一个小 kernel。这正是很多推理系统效率不高的根源之一。

## Reduction 的基本模式

Reduction 的任务，是把一组值压成更少的统计量，例如 sum、max、mean、variance、argmax，以及 top-k 前的局部统计。

### 2.1 为什么 reduction 特别考验并行组织

因为它天然会把很多 thread 的结果汇总到更少的位置，所以必须处理线程间协作、局部归约、跨 warp / block 合并，以及原子操作或多阶段归约。

### 2.2 常见层级

常见层级包括 warp 内 reduction、block 内 reduction 和多 block 全局 reduction。不同问题规模适合不同层级，不能一律套模板。

## Softmax 为什么是 reduction 的典型代表

**Softmax 看似只是**：

$$
\mathrm{softmax}(x)_i =
\frac{e^{x_i}}{\sum_j e^{x_j}},
$$

**但高性能实现时必须同时处理** max reduction、数值稳定、exp、sum reduction 和再归一化。

### 3.1 在线 softmax 的意义

长序列下，不可能总把整行都完整读入再一次性处理。  
在线 softmax 允许你边读块、边更新局部最大值与归一化项。  
这不仅是数学技巧，更是 kernel 设计基础。

## LayerNorm / RMSNorm 的 kernel 画像

Norm 类算子的核心不是“公式难”，而是每行要先算统计量、再应用归一化；数据通常是连续向量，很适合向量化加载，也很适合和 residual、scale、bias 融合。

### 4.1 为什么 RMSNorm 特别适合服务优化

因为它比完整 LayerNorm 少一些统计量和变换步骤，在很多 LLM 中出现频率极高，而且经常位于 decode 主路径。

### 4.2 Norm kernel 的常见优化点

常见优化点包括向量化 load/store、warp/block reduction、高精度累加统计量、与 residual / scale 融合，以及对齐 hidden size 和向量宽度。

## Rope 与位置编码周边 kernel

Rotary embedding 等位置编码处理在服务里常被视为“小操作”，但它们高频、紧跟在 Q/K 生成路径上、形状相对稳定，因此很适合融合。

### 5.1 为什么 rope 常和 projection 后处理一起考虑

因为 Q/K 生成后立即要施加位置编码，如果拆成单独 kernel，就会多一次中间写回；decode 下这类额外开销尤其敏感。

所以在实践里，rope 往往被当成一部分 post-projection fused path 来思考。

## Gather / Scatter / Indexing 为什么难

这类 kernel 的核心难点不在计算，而在不规则访存。

### 6.1 常见场景

常见场景包括 embedding lookup、MoE token dispatch、page table 索引、稀疏张量访问、采样和重排。

### 6.2 为什么它们常常跑不快

因为访问不连续、cache 命中难预测、coalescing 差，有时还要配合原子操作，某些操作不可避免地更像 memory-bound。

对这类 kernel，光盯 FLOPs 往往没有意义。

## Layout Transform 是被低估的大头

很多算子本身不慢，慢的是前后 layout transform，例如 transpose、permute、reshape 后的实际重排、pack/unpack 和 swizzle 相关变换。

### 7.1 为什么 layout 问题如此关键

**因为 layout 决定了**后续 kernel 是否能向量化、是否能 coalesced access、Tensor Core 是否能高效吃数据，以及是否需要额外中间张量。

很多系统瓶颈其实不是某个主算子，而是主算子前后不断发生的布局切换。

## 向量化与对齐

向量化 load/store 对这些小中型 kernel 特别重要。  
如果数据按更宽的向量边界对齐，kernel 可以：

用更少的指令完成更多数据读取，更好利用带宽，并减少地址计算和访存事务。

### 8.1 什么时候向量化难做

shape 不整齐、stride 奇怪、对齐不足、最后几个元素需要 mask，或者 layout 不是连续主序时，向量化都会变难。

这也是为什么很多 kernel 会针对常见 hidden size、head dim 做 shape-specialized 路径。

## 为什么这些 kernel 很适合融合

**典型可融合链包括** `residual + norm`、`bias + activation`、`rope + qk post-process`、`dequant + norm` 周边操作，以及 top-k 前后的局部 reduction。

### 9.1 融合收益来自哪里

融合收益主要来自少一次 HBM 往返、少一次 launch，让中间值停留在 register 或 shared memory，并减少中间张量管理。

对小算子来说，这种收益尤其明显。

## 常见失效模式

### 10.1 Norm 很快，但整体还是慢

可能原因不是 norm 本体，而是前后 layout 变换和 launch 太多。

### 10.2 Reduction 正确，但数值不稳

说明：

累加精度不够，归约顺序变化导致误差放大，或极值输入下没有做稳定处理。

### 10.3 Gather/Scatter 完全吃不满 GPU

这并不一定是实现差，而可能是这类不规则访存本来就带宽和局部性受限。优化重点应放在：

重排数据、提高局部性，并尽量降低不规则访问频率。

## 直觉例子

如果 GEMM 是主装配线，那么 reduction、norm 和索引 kernel 更像装配线上的检验、分拣、校准和搬运工位。单个工位看起来不大，但因为每批货都要经过，而且经常紧贴主线，它们一旦组织不好，整条产线都会被拖慢。现代 AI 系统里很多所谓“小 kernel 优化”，本质上就是在优化这些高频工位之间的物流和工序衔接。

## 本页结论

Reduction、norm、layout transform 和 indexing kernel 的共同主题是：  
它们常常更受内存系统而非算力系统支配，频率又非常高，因此特别适合从向量化、layout、fusion 和 shape specialization 的角度统一看待。理解这类算子，会让你对“为什么一个看似小的 kernel 能拖慢整个推理服务系统”有非常扎实的直觉。

!!! tip "下一站"
    - 回到本专题入口：[算子与编译器](index.md)，确认这页在整条路线中的位置。
    - 按导航顺序继续：[低精度与量化 Kernel](low-precision-and-quantized-kernels.md)。
    - 概念或符号卡住时，先查 [术语表](../glossary/index.md)，再回到当前页。
