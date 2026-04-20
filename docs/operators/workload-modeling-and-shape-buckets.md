# Workload 建模与 Shape Bucket

很多 kernel 优化最终之所以有效，不是因为它在所有 shape 上都快，而是因为它精准命中了真实 workload 中最常出现的那些 shape bucket。  
因此在进入特化和 autotune 之前，先理解 workload 分布本身，往往比直接写新 kernel 更重要。

## 1. 为什么 shape bucket 重要

真实系统的输入通常不是均匀分布的：

1. 某些 batch 特别常见；
2. 某些 head dim / hidden size 固定；
3. 某些请求长度集中在少数区间；
4. 某些 LoRA 或模型配置只占很小一部分。

如果不先做 workload 建模，很容易把大量工程时间浪费在低频 shape 上。

## 2. Workload 建模在回答什么

**至少要回答**：

1. 高频 shape 是哪些；
2. 长尾 shape 有多长；
3. prefill 和 decode 各自的 shape 分布如何；
4. 量化与非量化路径比例如何；
5. 哪些 bucket 值得特化，哪些应走通用 fallback。

## 3. 为什么这会影响 kernel 设计

因为：

1. autotune 搜索空间要围绕高频 bucket 设计；
2. shape specialization 的收益取决于 bucket 覆盖率；
3. benchmark 不应只测理想 shape；
4. dispatch 逻辑需要围绕 bucket 组织。

换句话说，workload 建模是特化内核、调度系统和基准设计的共同前提。

## 4. 一个形象比喻

可以把 shape bucket 想成工厂的订单画像。不是每种尺寸的零件都值得单独开一条专用产线，只有那些真正高频、长期出现的规格，才值得配专用模具。kernel 特化也是一样：先知道订单分布，再决定哪里值得投入特化成本。

## 5. 小结

Workload 建模与 shape bucket 的价值，在于把“我能优化什么”转化成“我最应该优化什么”。没有这一步，很多高性能算子开发都会陷入局部最优；有了这一步，特化、autotune、dispatch 和回归测试才会真正围绕真实系统收益展开。
