# Reduction、Norm 与索引 Kernel

在很多大模型系统里，真正频繁拖慢整体速度的，并不总是最大的 GEMM，而是那些“单次看起来不大、但出现极其频繁”的算子：reduction、softmax、layernorm、rmsnorm、rope、gather、scatter、layout transform。它们共同的特点是：

1. 常常更偏带宽受限；
2. 很适合做融合；
3. 对 shape、stride 和向量化极度敏感；
4. 容易在 decode、小 batch 和长上下文场景中放大存在感。

因此理解这些小中型 kernel，对于训练和服务都非常关键。

## 1. 为什么“小算子”不能被低估

单个 norm 或 reduction kernel 看起来没多少 FLOPs，但真实系统里它们往往：

1. 每层都要跑；
2. 每步 decode 都要跑；
3. 前后还夹着 layout 变换和 mask 处理；
4. 极易受带宽和 launch 开销影响。

如果这些算子没有被很好融合，系统会不断在：

1. 读中间结果；
2. 做一点小计算；
3. 再写回；
4. 再 launch 下一个小 kernel。

这正是很多推理系统效率不高的根源之一。

## 2. Reduction 的基本模式

Reduction 的任务，是把一组值压成更少的统计量，例如：

1. sum；
2. max；
3. mean；
4. variance；
5. argmax；
6. top-k 前的局部统计。

### 2.1 为什么 reduction 特别考验并行组织

因为它天然会把很多 thread 的结果汇总到更少的位置。  
**这意味着你必须处理**：

1. 线程间协作；
2. 局部归约；
3. 跨 warp / block 合并；
4. 原子操作或多阶段归约。

### 2.2 常见层级

1. warp 内 reduction；
2. block 内 reduction；
3. 多 block 全局 reduction。

不同问题规模适合不同层级，不能一律套模板。

## 3. Softmax 为什么是 reduction 的典型代表

**Softmax 看似只是**：

$$
\mathrm{softmax}(x)_i =
\frac{e^{x_i}}{\sum_j e^{x_j}},
$$

**但高性能实现时必须处理**：

1. max reduction；
2. 数值稳定；
3. exp；
4. sum reduction；
5. 再归一化。

### 3.1 在线 softmax 的意义

长序列下，不可能总把整行都完整读入再一次性处理。  
在线 softmax 允许你边读块、边更新局部最大值与归一化项。  
这不仅是数学技巧，更是 kernel 设计基础。

## 4. LayerNorm / RMSNorm 的 kernel 画像

Norm 类算子的核心不是“公式难”，而是：

1. 每行要算统计量；
2. 每行要再应用归一化；
3. 数据通常是连续向量；
4. 很适合向量化加载；
5. 又很适合和 residual、scale、bias 融合。

### 4.1 为什么 RMSNorm 特别适合服务优化

因为它比完整 LayerNorm 少一些统计量和变换步骤，在很多 LLM 中出现频率极高，而且经常位于 decode 主路径。

### 4.2 Norm kernel 的常见优化点

1. 向量化 load/store；
2. warp/block reduction；
3. 用高精度累加统计量；
4. 和 residual / scale 融合；
5. 对齐 hidden size 和向量宽度。

## 5. Rope 与位置编码周边 kernel

Rotary embedding 等位置编码处理在服务里常被视为“小操作”，但它们：

1. 高频；
2. 紧跟在 Q/K 生成路径上；
3. 形状相对稳定；
4. 很适合融合。

### 5.1 为什么 rope 常和 projection 后处理一起考虑

因为：

1. Q/K 生成后立即要施加位置编码；
2. 如果拆成单独 kernel，就会多一次中间写回；
3. decode 下这类额外开销很敏感。

所以在实践里，rope 往往被当成一部分 post-projection fused path 来思考。

## 6. Gather / Scatter / Indexing 为什么难

这类 kernel 的核心难点不在计算，而在不规则访存。

### 6.1 常见场景

1. embedding lookup；
2. MoE token dispatch；
3. page table 索引；
4. 稀疏张量访问；
5. 采样和重排。

### 6.2 为什么它们常常跑不快

因为：

1. 访问不连续；
2. cache 命中难预测；
3. coalescing 差；
4. 常要配合原子操作；
5. 某些操作不可避免地更像 memory-bound。

对这类 kernel，光盯 FLOPs 往往没有意义。

## 7. Layout Transform 是被低估的大头

很多算子本身不慢，慢的是前后 layout transform，例如：

1. transpose；
2. permute；
3. reshape 后的实际重排；
4. pack/unpack；
5. swizzle 相关变换。

### 7.1 为什么 layout 问题如此关键

**因为 layout 决定了**：

1. 后续 kernel 是否能向量化；
2. 是否能 coalesced access；
3. Tensor Core 是否能高效吃数据；
4. 是否需要额外中间张量。

很多系统瓶颈其实不是某个主算子，而是主算子前后不断发生的布局切换。

## 8. 向量化与对齐

向量化 load/store 对这些小中型 kernel 特别重要。  
如果数据按更宽的向量边界对齐，kernel 可以：

1. 更少的指令完成更多数据读取；
2. 更好利用带宽；
3. 减少地址计算和访存事务。

### 8.1 什么时候向量化难做

1. shape 不整齐；
2. stride 奇怪；
3. 对齐不足；
4. 最后几个元素需要 mask；
5. layout 不是连续主序。

这也是为什么很多 kernel 会针对常见 hidden size、head dim 做 shape-specialized 路径。

## 9. 为什么这些 kernel 很适合融合

**典型可融合链包括**：

1. residual + norm；
2. bias + activation；
3. rope + qk post-process；
4. dequant + norm 周边操作；
5. top-k 前后的局部 reduction。

### 9.1 融合收益来自哪里

1. 少一次 HBM 往返；
2. 少一次 launch；
3. 中间值可以停留在 register 或 shared memory；
4. 更少中间张量管理。

对小算子来说，这种收益尤其明显。

## 10. 常见失效模式

### 10.1 Norm 很快，但整体还是慢

可能原因不是 norm 本体，而是前后 layout 变换和 launch 太多。

### 10.2 Reduction 正确，但数值不稳

说明：

1. 累加精度不够；
2. 归约顺序变化导致误差放大；
3. 极值输入下没有做稳定处理。

### 10.3 Gather/Scatter 完全吃不满 GPU

这并不一定是实现差，而可能是这类不规则访存本来就带宽和局部性受限。优化重点应放在：

1. 重排数据；
2. 提高局部性；
3. 降低不规则访问频率。

## 11. 一个形象比喻

如果 GEMM 是主装配线，那么 reduction、norm 和索引 kernel 更像装配线上的检验、分拣、校准和搬运工位。单个工位看起来不大，但因为每批货都要经过，而且经常紧贴主线，它们一旦组织不好，整条产线都会被拖慢。现代 AI 系统里很多所谓“小 kernel 优化”，本质上就是在优化这些高频工位之间的物流和工序衔接。

## 12. 小结

Reduction、norm、layout transform 和 indexing kernel 的共同主题是：  
它们常常更受内存系统而非算力系统支配，频率又非常高，因此特别适合从向量化、layout、fusion 和 shape specialization 的角度统一看待。理解这类算子，会让你对“为什么一个看似小的 kernel 能拖慢整个服务系统”有非常扎实的直觉。
