# Roofline、基准与案例拆解

做算子优化时，最常见的失败不是“技术不够强”，而是：

1. 没有先判断瓶颈类型；
2. benchmark 设计不可靠；
3. 看到局部提速就误以为端到端受益；
4. 没有把案例还原成通用方法。

因此这一页专门把 roofline、benchmark 设计和几个典型案例放在一起，帮助你把“性能感觉”变成“性能判断”。

## 1. Roofline 的真正用途

Roofline 不是为了精确预测每个 kernel 的最终性能，而是为了先问一个更关键的问题：

**这个 kernel 更可能受算力限制，还是受带宽限制？**

**其直觉形式为**：

$$
\text{Performance}
\le
\min(\text{Peak FLOPs}, \text{Bandwidth} \times \text{Arithmetic Intensity}).
$$

### 1.1 算术强度为什么重要

如果一个算子做了很多计算却只搬很少数据，它更可能接近算力上限；  
如果一个算子大量搬数据但每字节只做很少计算，它更可能受带宽限制。

**这能帮你决定**：

1. 该优先想 tile 和 Tensor Core；
2. 还是优先想 fusion、layout、向量化与减少写回。

## 2. Benchmark 设计的三层目标

一个有价值的 benchmark 应至少服务三个目标：

1. 看单 kernel 极限；
2. 看真实 shape 分布下的稳定收益；
3. 看端到端系统是否真的受益。

如果只满足第一个，很容易得到“实验室很快，线上不明显”的结论。

## 3. 微基准的基本规范

较稳妥的 microbenchmark 通常要做到：

1. 充分 warmup；
2. 固定 shape、dtype、layout；
3. 区分单 kernel latency 和 end-to-end latency；
4. 多轮重复；
5. 避免把 JIT 或首次编译成本算进去；
6. 报告环境。

## 4. 一个 GEMM 案例怎么看

面对 GEMM 案例时，建议先看：

1. \(M/N/K\) 多大；
2. 是否规则；
3. 是否是高频服务热点；
4. 是否已经走 Tensor Core；
5. epilogue 是否可融合。

如果是规则大矩阵，roofline 往往提示它更偏 compute-heavy；  
如果是小而碎的 GEMM，真正的问题可能反而是 launch 和 shape。

## 5. 一个 Softmax / Norm 案例怎么看

这类算子的关键问题通常不是 FLOPs，而是：

1. 数据读写次数；
2. reduction 路径；
3. 向量化；
4. 是否可融合。

**roofline 常提醒你**：  
这类算子更可能是 bandwidth-bound，因此优化方向应偏：

1. 减少中间写回；
2. 提高向量化；
3. 改善访存局部性；
4. 融合周边操作。

## 6. 一个 Attention 案例怎么看

Attention 常常是混合型案例：

1. 数学公式看起来计算很多；
2. 但真正性能又常被 I/O 主导；
3. 长上下文下 score matrix materialization 很痛；
4. decode 场景下 KV cache 读取更痛。

因此对 attention 来说，roofline 的价值在于提醒你先拆阶段：

1. prefill；
2. decode；
3. paged KV；
4. prefix reuse。

## 7. 端到端收益为什么常常小于单 kernel 收益

因为：

1. 该 kernel 在整体中占比有限；
2. 优化后新瓶颈出现；
3. 额外引入了数据准备成本；
4. 调度和通信没有同步优化。

这也是为什么好的案例拆解，不会只给“kernel 提速 2 倍”，还会解释：

1. 端到端提升多少；
2. 为什么不是同等倍数；
3. 剩余瓶颈在哪里。

## 8. 如何把案例拆解成可复用方法

面对任何优化案例，都可以问：

1. 它解决的是带宽问题、算力问题，还是 launch / layout 问题？
2. **它复用了什么通用模式**：tiling、fusion、online reduction、persistent、specialization？
3. **它依赖什么边界条件**：shape、dtype、拓扑、请求分布？
4. 它能迁移到哪些别的算子上？

这样案例才会变成“方法库”，而不是孤立技巧。

## 9. 一个形象比喻

Roofline 像是在看一台机器究竟是马力不够，还是输送带太慢；benchmark 像是在不同工况下测这台机器；案例拆解则是在弄清楚某次改造到底是换了更强的引擎，还是把物流路径改顺了。只有把这三件事放在一起，性能优化才会从“感觉更快”变成“知道为什么更快”。

## 10. 小结

Roofline、基准与案例拆解的核心意义，在于把性能优化从“经验活”变成“结构化分析”。只要你能先判断瓶颈类型，再设计可信 benchmark，再把案例抽象成可复用模式，就会比单纯堆技巧更接近真正成熟的算子工程。
