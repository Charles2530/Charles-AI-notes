# GEMM、Attention 与融合 Kernel

如果只允许用一句话概括现代 AI kernel 的核心，那就是：**大部分性能都围绕数据如何穿过 GEMM、Attention、Norm、Quantization 与 Memory Movement 这一串热点算子被决定。**  
其中 GEMM 是计算主引擎，Attention 是序列模型的结构性热点，融合 kernel 则是在试图减少中间读写和 launch 开销，让系统更接近硬件上限。

这一页的重点不是重复数学公式，而是回答：为什么这些算子如此重要，它们共享哪些优化模式，以及现代训练与推理系统为什么一再围绕它们重构。

## 1. 为什么 GEMM 是 AI 系统的发动机

大模型训练和推理中的绝大多数 FLOPs，都能还原为 GEMM 或 batched GEMM：

1. 线性层；
2. QKV 投影；
3. FFN 上下投影；
4. MoE expert 中的 dense matmul；
5. 一些量化后的混合精度 GEMM。

因此只要 GEMM 慢，整个系统大概率都慢。

### 1.1 GEMM 不只是矩阵乘公式

**数学上 GEMM 只是**：

$$
C = AB + D
$$

**但工程上它包含一整套问题**：

1. 输入输出是什么 layout；
2. 是否可用 Tensor Core；
3. tile 怎么切；
4. 数据怎么从 HBM 搬到 shared memory、再到 register；
5. epilogue 能否顺手融合 bias、activation、dequant；
6. 某些维度过小或不规则时，是否还有高效率。

也就是说，真正的 GEMM 优化问题，从来不是“会不会乘”，而是“怎么搬、怎么排、怎么复用、怎么写回”。

## 2. GEMM 的基本优化套路

无论是 CUDA、CUTLASS 还是 Triton，GEMM 的核心套路都高度一致：

1. 把 \(M\)、\(N\)、\(K\) 三个维度按 tile 切分；
2. 将一块 \(A\) 和一块 \(B\) 搬入更近的内存层；
3. 在块内进行大量 FMA / MMA；
4. 把部分结果保留在 register；
5. 最后再统一写回。

### 2.1 为什么 tile 是关键

tile 的作用本质上是提升数据重用。  
如果一个元素只读一次就只参与一次乘法，那必然是带宽灾难；  
如果一个 tile 被搬进来后能服务很多个输出元素，算术强度就会上升，kernel 才更可能接近算力上限。

### 2.2 Tile 不是越大越好

**tile 太小**：

1. 数据重用不足；
2. launch 和边界开销偏大；
3. Tensor Core 利用不佳。

**tile 太大**：

1. shared memory 压力高；
2. register pressure 高；
3. occupancy 下降；
4. 边界 shape 处理更麻烦。

所以 tile 选择永远是折中。

## 3. Tensor Core GEMM 的真实约束

Tensor Core 带来高吞吐，但并不意味着任何 GEMM 自动高速。想要真正吃到收益，往往要满足：

1. dtype 匹配；
2. tile 和对齐要求匹配；
3. 数据布局适合 MMA 指令；
4. epilogue 不要破坏流水；
5. `M/N/K` 维度别太离谱。

### 3.1 为什么小矩阵或细碎形状常常跑不满

因为 Tensor Core 最擅长规则、足够大的矩阵块。一旦形状太小、太细碎、边界太多，kernel 会花更多时间在：

1. mask；
2. 边界处理；
3. launch；
4. layout 转换；
5. 无法完全填满 warp 或 tile 的尾部。

这也是为什么很多推理场景中的 decode GEMM，优化逻辑和训练阶段完全不同。

## 4. Epilogue Fusion：GEMM 后面那几步常常同样重要

**很多 GEMM 后面立即跟着**：

1. bias add；
2. activation；
3. residual add；
4. scale；
5. dequant；
6. quantize。

如果把这些步骤拆成单独 kernel，就会产生额外读写。  
因此现代高性能 GEMM 常把一部分 epilogue 融合进去。

### 4.1 为什么 epilogue fusion 的收益有时比 GEMM 本体还明显

因为 GEMM 本体已经很高效了，而真正多出来的时间常花在：

1. 再读一次输出；
2. 再写一次中间结果；
3. launch 多个小 kernel；
4. 打断原本连续的数据路径。

这也是为什么 “dequant + GEMM + bias + activation” 会成为量化推理常见的融合对象。

## 5. Attention 为什么是第二大主战场

Transformer 让 Attention 成为结构性热点。即使核心数学本身不复杂：

$$
\text{Attention}(Q,K,V)
=
\text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V,
$$

但工程上它极难高效实现，因为涉及：

1. QK 乘法；
2. 大 score 矩阵；
3. softmax 归一化；
4. mask；
5. 与 V 相乘；
6. 长序列时巨大 I/O。

### 5.1 Attention 的问题不是 FLOPs 不够，而是 I/O 太重

朴素实现里最致命的问题，往往是把完整 score 矩阵 materialize 到 HBM，再读回来做 softmax 和后续乘法。  
这种实现会产生巨大的中间张量和带宽压力。

这也是为什么 FlashAttention 这类工作如此重要：它不是改变数学结果，而是重写 I/O 路径。

## 6. FlashAttention 的核心思想

FlashAttention 的关键词通常是：

1. 分块；
2. 在线 softmax；
3. 不显式写出完整 score 矩阵；
4. 尽量在片上存储内完成局部计算。

### 6.1 在线 softmax 为什么关键

softmax 看起来需要先看到整行 score 才能归一化，但实际上可以通过维护局部最大值和归一化项，在分块处理中得到精确结果。

**这允许 kernel**：

1. 一边加载 Q/K/V 块；
2. 一边更新局部 softmax 状态；
3. 一边累积输出；
4. 最终只把需要的结果写回。

### 6.2 FlashAttention 真正优化的是什么

它优化的不是数学复杂度，而是：

1. HBM 往返；
2. 中间矩阵 materialization；
3. cache / shared memory 的利用方式；
4. 序列长度增长时的 I/O 压力。

这正是 AI kernel 优化里非常典型的一类思路：**不改公式，改数据流**。

## 7. FlashAttention 之外的 Attention 家族

真实系统里的 attention 热点不止一种。

### 7.1 Prefill Attention

长上下文输入阶段，重点是高吞吐和长序列 tile。

### 7.2 Decode Attention

每步新增少量 token，重点转为：

1. 小 batch；
2. KV cache 访问；
3. launch 开销；
4. paged memory；
5. tail latency。

### 7.3 Paged Attention

当 KV cache 采用页式管理后，attention kernel 还要适应页面索引与不连续块访问。  
这让 decode attention 更像“访存系统问题”，而不是单纯矩阵乘问题。

### 7.4 Flash Decoding

长上下文 decode 场景下，需要重新组织并行方式，不能简单照搬 prefill attention 的策略。  
这也是为什么后续又出现了 Flash Decoding、PagedAttention 等专门面向 decode 场景的优化路线。

## 8. MHA、MQA、GQA 与 kernel 画像

不同注意力头组织方式也会显著改变 kernel 形态：

1. `MHA`：每个 query head 对应独立 K/V；
2. `MQA`：多个 query head 共享一组 K/V；
3. `GQA`：介于两者之间，按组共享 K/V。

这些设计不仅影响模型结构，也影响：

1. KV cache 大小；
2. 读写模式；
3. attention kernel 的数据复用；
4. decode 场景的整体带宽压力。

因此它们既是模型设计问题，也是内核与服务系统问题。

## 9. Reduce、Softmax、Norm：小算子为什么值得写

虽然 GEMM 和 Attention 是最大热点，但高频小算子同样重要，原因有三：

1. 它们调用次数极高；
2. 很多都是带宽受限；
3. 很适合融合。

### 9.1 Softmax

**关键在**：

1. 数值稳定；
2. warp / block 归约；
3. 长行分块；
4. 避免不必要全局读写。

### 9.2 LayerNorm / RMSNorm

**关键在**：

1. 向量化 load/store；
2. 统计量计算；
3. 融合 scale / bias / residual；
4. 保持累计精度。

### 9.3 Reduce

**关键在**：

1. warp-level primitive；
2. 分层归约；
3. 原子写回或 block 汇总；
4. shape 边界和对齐。

这些算子单次看起来小，累计起来却能占掉显著时间。

## 10. 融合 kernel 的几类常见模式

现代 AI 系统中常见的融合模式至少有：

1. `bias + activation`；
2. `residual + norm`；
3. `dequant + matmul + epilogue`；
4. `rope + qk projection 周边处理`；
5. `attention 内部的 scale + mask + softmax + value combine`；
6. `dropout + residual + norm`；
7. `quantize / pack` 路径上的后处理。

### 10.1 融合为什么有效

**主要收益包括**：

1. 少一次或多次 HBM 往返；
2. 少几个 kernel launch；
3. 少几个中间张量；
4. 更好的 cache / register 复用。

### 10.2 融合为什么也有代价

**代价包括**：

1. 实现复杂；
2. shape 覆盖变窄；
3. 数值错误更难定位；
4. 维护成本高；
5. 编译时间和 autotune 空间增大。

因此“能融合就融合”并不是绝对真理，应该看频率、收益和维护成本。

## 11. 量化 kernel：权重小不等于推理快

量化推理真正的难点，不在于把权重文件压缩，而在于：

1. 如何在 kernel 内高效反量化；
2. 如何让反量化与 GEMM / epilogue 融合；
3. 如何处理 group-wise、channel-wise、per-token 等不同尺度；
4. 如何保持 Tensor Core 或向量化路径仍然高效。

### 11.1 一个常见误区

很多系统在引入低比特量化后，理论显存大幅降低，却没有获得预期吞吐提升。常见原因就是：

1. 反量化开销很大；
2. 需要频繁 layout 转换；
3. 额外 scale / zero-point 访问又引入带宽瓶颈；
4. kernel 没有 fused 到主路径。

所以“量化能不能提速”最终还是 kernel 问题。

## 12. 训练与推理中的热点差异

### 12.1 训练

**更关注**：

1. 大 GEMM 吞吐；
2. attention forward/backward；
3. optimizer update kernel；
4. gradient reduce overlap；
5. 激活重计算与 memory-aware kernel。

### 12.2 推理

**更关注**：

1. prefill attention；
2. decode attention；
3. paged KV 访问；
4. quantized GEMM；
5. 高频小算子融合；
6. dynamic batching 下的 shape 不规则性。

这也是为什么很多开源项目中，训练 kernel 和服务 kernel 会逐渐走向不同实现。

## 13. 一个更完整的优化视角：算子图而不是单算子

很多性能问题不是某个单独 kernel 太慢，而是：

1. GEMM 输出立刻被 layout 转换；
2. 转换后又喂给一个 norm kernel；
3. norm 后再做一个 activation kernel；
4. 中间每一步都写回 HBM。

如果只盯单个 kernel 的 TFLOPs，你会误以为“每个 kernel 都还行”。  
**真正的问题在算子图级别**：数据在系统里来回搬得太多。

这也是为什么图编译器、fusion pass 和自定义 fused kernel 会越来越重要。

## 14. 常见失效模式

### 14.1 数学正确但访存失败

最典型的情况是实现完全对，但 layout、stride 或 tile 选择导致带宽极低。

### 14.2 融合过度

把太多逻辑塞进一个 kernel，导致：

1. register 爆；
2. occupancy 降；
3. autotune 空间失控；
4. 边界 case 很难维护。

### 14.3 只看训练热点，不看服务热点

训练里最热的可能是 attention backward；  
服务里最痛的可能是 decode attention 和 KV page 访问。  
如果优化方向没切到真实业务负载，收益会很有限。

### 14.4 只看 FLOPs，不看 I/O

这是 Attention 和量化路径里最常见的误判。

## 15. 一个面向系统设计的判断框架

看一个热点算子时，可以先问：

1. 它主要是 GEMM 型、Attention 型、Norm/Reduce 型，还是不规则访存型；
2. 它更偏算力受限还是带宽受限；
3. 是否有明显中间张量可消除；
4. 是否适合通过 Triton 快速实现；
5. 是否已经有成熟库可用；
6. 是否值得为它付出手写 CUDA 的维护成本。

这比直接问“要不要自己写 kernel”更实际。

## 16. 一个形象比喻

可以把 GEMM 想成工厂里的主装配线，Attention 想成需要按上下文临时调度零件和工序的复杂装配站，Norm 和 Reduce 像频繁出现的小质检工位，而融合 kernel 则是在试图把原本分散在多个工位上的几个连续动作合成一站完成。

主装配线再快，如果零件总要在多个工位之间来回搬运，整体产能仍然会低。现代 AI kernel 工程的本质，很大程度上就是在重构这条生产线的物流路径。

## 17. 小结

GEMM、Attention 与融合 kernel 构成了现代 AI 系统的算子主战场。GEMM 决定计算主吞吐，Attention 决定序列模型的结构瓶颈，融合 kernel 决定系统是不是在无谓地搬运中间结果。理解它们的共同模式，你就能把很多看似分散的技巧统一到同一张图里：tile、数据重用、I/O 感知、layout、epilogue、fusion、profile 和 shape specialization。

也只有在这个层面上，训练系统、推理引擎、量化加速和自定义 kernel 才真正连成了一条完整的系统优化主线。

## 18. 先把 GEMM 形状分清，再谈优化

很多人一提 GEMM 就默认想到“大方阵”。但真实 AI 系统里，决定实现选择的往往是**形状谱系**。

### 18.1 常见 GEMM 形状家族

1. **大而规则的训练主干 GEMM**：如 FFN、QKV 投影、dense expert；
2. **skinny GEMM**：某一维很小，如 `M=1` 的 decode；
3. **grouped / batched GEMM**：一批小矩阵一起做；
4. **split-K GEMM**：为大 `K` 或不规则任务做并行分解；
5. **quantized GEMM**：带反量化、scale、zero-point 或 block-wise metadata；
6. **MoE GEMM**：token 经过路由后形成不均匀 expert batch。

### 18.2 为什么这一步必须先做

因为：

1. **最优 tile 完全不同**；
2. **persistent 与否的价值不同**；
3. **是否值得 fusion 不同**；
4. **带宽与算力主导关系不同**。

## 19. Skinny GEMM 是推理系统最容易低估的坑

服务态 decode 时，很多 GEMM 其实更接近 `M=1` 或很小的 batch。数学上还是矩阵乘，但工程上已经不再像训练态大 GEMM。

### 19.1 它为什么难

1. **并行度低**，很难自然吃满 Tensor Core；
2. **launch overhead 占比上升**；
3. **反量化、layout 转换、epilogue 的固定成本变得突出**；
4. **batch 动态变化，使单点最优配置失效**。

### 19.2 常见手段

1. **grouped GEMM**，把多请求拼成一批；
2. **persistent kernel**，减少频繁 launch；
3. **权重预打包和布局特化**；
4. **尽可能把 dequant / bias / activation 合到 epilogue**。

## 20. Grouped GEMM 与普通 Batched GEMM 的区别

这两个词常被混用，但它们关注点并不完全一样。

### 20.1 Batched GEMM 更像规则批处理

通常假设一批 GEMM 的形状相同或非常接近，便于统一调度。

### 20.2 Grouped GEMM 更像不规则任务合并

它更适合：

1. **MoE expert 的不同 token 数**；
2. **多请求 decode 下的不同 shape**；
3. **服务态 mixed workload**。

### 20.3 真正难点不在数学，而在工作组织

1. **如何分组让每组足够大又不过度拖尾**；
2. **如何让 kernel 内部对不同 shape 的块做相对均衡的工作**；
3. **如何与 runtime 的 continuous batching 兼容**。

## 21. Quantized GEMM 的瓶颈常不在乘法本身

量化 GEMM 的端到端成本，常常由三部分共同决定：

1. **权重读取**；
2. **反量化与尺度恢复**；
3. **核心 MMA**。

### 21.1 为什么很多量化 GEMM 反而不快

1. **scale / zero-point 读得很散**；
2. **反量化逻辑太重**；
3. **layout 不适配 Tensor Core**；
4. **服务态 `M` 很小，dequant 固定成本被放大**。

### 21.2 更值钱的优化往往在数据流

1. **权重预打包**；
2. **块级 scale 与数据块对齐**；
3. **在寄存器或 shared memory 内完成 dequant 后立刻 MMA**；
4. **把 bias / activation / requant 继续合到 epilogue**。

## 22. Attention Kernel 先分 Prefill 和 Decode 两个世界

很多 attention 优化讨论混在一起，其实 prefill 和 decode 的最优内核常常完全不同。

### 22.1 Prefill 的特点

1. **并行度高**；
2. **更像大块矩阵运算**；
3. **FlashAttention 一类 IO-aware 内核收益大**；
4. **对长序列 tile、shared memory 和流水线设计敏感**。

### 22.2 Decode 的特点

1. **一次只生成少量 query**；
2. **大量成本落在 KV cache 读取**；
3. **更偏 memory-bound**；
4. **paged attention、MQA/GQA、cache layout 比纯算术更关键**。

### 22.3 一个常见误判

把 prefill benchmark 很快的 attention 直接推广到在线 decode，这经常会失效。原因不是论文错，而是**工作负载已经变了**。

## 23. FlashAttention 的价值不只是“更快 attention”

FlashAttention 这类方法之所以重要，是因为它展示了一个普遍原则：**对于中间张量极大、访存主导的算子，重写数据流往往比改公式更关键。**

### 23.1 它体现出的通用模式

1. **按块流式处理，避免把完整中间矩阵写回 HBM**；
2. **在线维护 softmax 所需统计量**；
3. **把 tile 复用、数值稳定和访存组织统一起来**。

### 23.2 这个模式还能迁移到哪里

1. **稀疏或局部 attention**；
2. **ring / block / hierarchical attention**；
3. **某些需要在线归一化的 reduction kernel**；
4. **带量化或页式 KV 的 attention 变体**。

## 24. MQA、GQA、MLA 对 kernel 的影响

这些结构经常从模型角度讲节省 KV cache，但从 kernel 角度看，它们在改两件事：

1. **KV 的共享粒度**；
2. **每个 query 访问 KV 的布局和重复度**。

### 24.1 为什么它们能影响性能

1. **更少的 KV 头数意味着更少显存带宽压力**；
2. **跨 query 的 cache 复用更强**；
3. **但也可能引入更复杂的 head 映射与 gather 逻辑**。

### 24.2 结构改了，kernel 不一定自动跟上

如果 runtime、layout 和 attention kernel 仍按 MHA 假设设计，那么模型层面节省下来的理论收益不会完整落到系统里。

## 25. MoE 会把 GEMM 问题变成“排序 + 打包 + GEMM + 还原”

MoE 的热点并不只是 expert 内的 GEMM，而是整个数据路径：

1. **路由分数计算**；
2. **top-k 选择**；
3. **按 expert 分桶和重排 token**；
4. **执行 grouped GEMM**；
5. **再 scatter 回原位置**。

### 25.1 这说明什么

很多 MoE 优化最终落在：

1. **索引和重排 kernel**；
2. **grouped GEMM 调度**；
3. **负载均衡与尾部 expert 的处理**；
4. **combine / reduce 路径的融合**。

也就是说，MoE GEMM 的最佳实现往往是一个**包含路由周边算子的局部子图优化**，而不是单独盯着中间那一段 matmul。

## 26. Epilogue Fusion 是最容易见效的工程杠杆

很多算子优化真正高 ROI 的地方，不在核心乘法，而在写回前的最后一段。

### 26.1 常见可融合内容

1. **bias**；
2. **scale / dequant / requant**；
3. **activation**；
4. **residual add**；
5. **dropout mask 应用**；
6. **某些简单的归一化前处理**。

### 26.2 为什么 epilogue 值钱

1. **减少一次或多次 HBM 往返**；
2. **少一次 kernel launch**；
3. **中间张量不必物化**；
4. **对服务态小 batch 尤其敏感**。

### 26.3 但不是越多越好

融合过度会导致：

1. **寄存器爆炸**；
2. **shape 特化碎片化**；
3. **可维护性下降**；
4. **一点小改动就要重写整块 kernel**。

## 27. 一个现实问题：基准快，不代表系统快

GEMM、attention 和 fused kernel 的微基准结果非常容易误导。真正上线前，至少要回答：

1. **是否纳入了真实 shape 分布**；
2. **是否算上 layout 转换和数据准备**；
3. **是否和 runtime 的 batching / routing / caching 协同**；
4. **是否测了 tail shapes 和长尾负载**；
5. **是否测了编译和首次加载成本**。

### 27.1 特别是服务态

在线服务里，最怕的是一个 kernel 对热路径平均值有帮助，但对某些尾部 shape、cache miss 或混合负载表现极差，最终拖垮 P99。

## 28. 一份更实用的算子验收框架

对 GEMM、attention 或 fused kernel，建议至少做四层验收：

1. **算子层**：固定 shape 下的 kernel 时间、带宽、Tensor Core 利用；
2. **子图层**：是否减少了中间读写和 launch；
3. **服务层**：混合长度与真实 batching 下的 TTFT/TPOT 变化；
4. **运维层**：部署后是否稳定、是否易于回滚、是否有形状黑洞。

### 28.1 为什么一定要有“子图层”

因为许多看起来强的 kernel，其真实价值来自它能不能让上游下游一起受益，而不是它单独快多少。

## 快速代码示例

```python
import torch
import torch.nn.functional as F

def fused_attn(q, k, v, is_causal=True):
    # PyTorch 2.x 会在可用时走 fused SDPA/FlashAttention 路径
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=None, dropout_p=0.0, is_causal=is_causal
    )
```

这段代码用 `scaled_dot_product_attention` 触发框架侧的融合实现，避免手写 softmax + matmul 的中间张量开销。实际 profiling 时应重点看不同序列长度和 head 维度下是否命中 Flash/高性能路径。

## 29. 小结

GEMM、Attention 与融合 kernel 的本质，不是三份孤立知识，而是一套围绕**形状、数据流、复用和写回路径**展开的共同工程学。真正的高手不会只盯着“乘法多快”或“softmax 多快”，而是会从整个子图和运行时角度判断：这条数据路径中，究竟哪一段最值得被重写。
