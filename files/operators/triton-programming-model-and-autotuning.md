# Triton 编程模型与自动调优

Triton 在 AI 系统里越来越重要，不是因为它能完全替代 CUDA，而是因为它提供了一种适合张量 kernel 的中层抽象：比手写 CUDA 更接近矩阵块和张量块思维，又比纯图编译的黑盒代码生成更可控。

典型分工是：最关键、最极致的内核仍可能用 CUDA 或模板库；大量中高频、自定义、融合型算子优先尝试 Triton；上层编译器如 Inductor 也会把一部分 kernel 降到 Triton。

!!! note "初学者先抓住"
    Triton 可以先理解成“用张量块写 GPU kernel”。你不用一开始管理每个 CUDA thread，但仍然要思考 tile 大小、mask、访存连续性和自动调优。

!!! example "有趣例子：按托盘做菜"
    CUDA 像安排每个厨师手里切哪一片菜，Triton 更像安排每个托盘装多少食材、每批怎么进锅。抽象更高，但托盘大小和摆放方式仍然决定效率。

## Triton 解决的问题

在 Triton 之前，很多团队面临两难：完全依赖框架默认 kernel，难以做深度 fusion 和特定 shape 优化；凡事写 CUDA，开发门槛和维护成本又过高。

Triton 正好落在中间：抽象层次足够高，矩阵与块运算比 CUDA 更自然；抽象层次又不至于完全失控，仍可显式决定 tile、block 和加载方式；同时它天然适合 autotuning、shape-specialized kernel，并与 PyTorch 生态集成顺滑，便于快速验证热点优化。

它更适合“中等复杂度但高频”的算子：norm、softmax、elementwise fusion、专用 matmul、部分 attention、quant/dequant 和 MoE 周边搬运。

## 编程心智模型

Triton kernel 通常不是从 thread 级别出发，而是从 block / tile 级别出发。你更常思考一个 program instance 负责哪一块输出，这块输出需要加载哪几块输入，offsets 如何构造，边界是否需要 mask，块内数据如何复用。

`tl.program_id(axis=...)` 用来取得当前 program instance 的块索引，再计算输入输出偏移。`tl.load` 和 `tl.store` 负责加载与写回，mask 用于处理 shape 不是 tile 整数倍时的边界。

这不是“Python 封装版 CUDA”。Triton 有自己的张量化中间表示和代码生成路径，更像一个面向张量块 kernel 的 DSL、codegen 和 autotune 环境。

## 适合的 Kernel 类型

| 类型 | 为什么适合 Triton |
| --- | --- |
| Matmul / Batched Matmul | tile 参数清晰，容易做 shape-specialization 和 autotune |
| Softmax / Reduce | 按行或按块归约，逻辑清楚，便于融合 |
| LayerNorm / RMSNorm | 高频、带宽受限、很适合向量化和融合 |
| Elementwise fusion | bias、activation、residual、dropout、scale 可合并 |
| Quant / Dequant | scale 访问和类型转换可与周边算子融合 |
| Attention 原型 | Q/K/V 分块、mask、online softmax 便于快速验证 |
| MoE 周边 | token permutation、padding、grouping 可做专用优化 |

Triton 的优势是开发速度和可控性。对于非常复杂的跨块协作、极致 Tensor Core pipeline、新硬件特性或高度手调的生产 attention，CUDA/CUTLASS 仍可能更合适。

## Matmul、Softmax 与 Attention

Triton matmul 常见 meta-parameters 包括 `BLOCK_M`、`BLOCK_N`、`BLOCK_K`、`num_warps`、`num_stages`。它们分别控制输出块大小、K 维推进、warp 协作数量和流水深度。

Softmax 是理解 Triton 的好入口。典型实现会读取一行或子块，计算局部最大值，做 `exp(x - m)`，求和归一化并写回。长行 softmax 可以分块维护 online max/sum，这也是理解 FlashAttention 的基础。

Attention 适合用 Triton 原型化，因为它具备明确块结构、大量可融合操作和明显 I/O 优化空间。但 Triton attention 不自动等于生产最优。生产实现还要看 head dim、序列长度、causal/sliding window/paged KV、backward、目标 GPU 和 runtime 调度。

## 自动调优

不同 GPU、shape、dtype 下最优 tile 参数不同。Triton 的实用性很大程度来自 autotune。

常见搜索维度包括 `BLOCK_M/N/K`、`num_warps`、`num_stages`、向量化宽度、layout 或 swizzle 开关，以及是否使用某些融合路径。

Autotune 不能替代理解。如果访存模式本身不合理，autotune 只是在一堆差配置里找相对不差的。更合理的顺序是：先根据数据流设计候选，再用 autotune 搜索，最后配合 profiling 解释为什么某个配置更好。

也要避免 autotune 空间过大。搜索成本、编译缓存、冷启动和 shape 变化都会影响真实收益。

## 与 PyTorch 和编译器的关系

Triton 在 PyTorch 生态里常见于三种路径：手写 `@triton.jit` 自定义 kernel，作为 `torch.compile` / Inductor 的后端生成目标，或被高层库封装成特定算子的加速实现。

这带来一个工程现实：同一个算子可能同时存在框架默认实现、Inductor 生成实现、Triton 手写实现和 CUDA 库实现。上线前要明确哪条路径实际被调用，shape 变化时是否会重新编译，cache miss 和 warmup 成本是否可接受，fallback 路径是否数值一致，graph rewrite 是否破坏预期融合。

只看单次 microbenchmark 很容易误判。要把编译、缓存、动态 shape 和端到端调度算进去。

## 选型边界

当框架默认 kernel 明显慢、算子热点稳定且形状集中、需要融合多个小算子或快速验证一种数据流设计，而 CUDA 手写成本过高、库又覆盖不到时，适合优先用 Triton。反过来，如果热点还没 profile 清楚、shape 过于发散、编译和 autotune 成本过高、需要极端手调的 Tensor Core pipeline、涉及复杂跨 block 协作，或已有成熟库足够快且维护成本更低，就不适合优先上 Triton。

Triton 最有价值的位置，是把“值得优化但不值得手写 CUDA”的大片中间地带覆盖起来。

## 验收清单

写 Triton kernel 后，不要只看一个形状的速度。验收要覆盖正确性、数值稳定、性能分桶、编译成本、端到端收益、回退机制和维护成本：多 dtype、多 shape、边界 mask、随机 seed 要和 baseline 对齐；softmax、reduce、低精度累计和溢出边界要稳定；主流 shape、长尾 shape、batch 变化和动态 shape 要分桶看；首次编译、cache 命中和 autotune 时间也要计入真实收益。

Triton 的核心价值不是让所有人绕过 CUDA，而是让更多工程师能以可控成本写出足够好的张量 kernel，并把 shape-specialization、fusion 和 autotuning 纳入日常优化流程。

## 真实排查案例：Autotune 搜索出了线上冷启动抖动

**输入症状**：Triton attention 原型在离线测试中快 12%，上线灰度后首批请求 TTFT 偶发超过 10s，之后恢复正常。

**关键指标**：热态 kernel latency 符合预期；冷态请求出现编译和 autotune 峰值；编译缓存 miss 集中在少数长上下文 bucket；业务 P99 被首次命中拖高。

**Nsight / trace 观察**：Nsight Systems 里 GPU 空闲但 CPU 侧 JIT 编译占用明显；首次遇到新 shape 时没有马上进入 GPU kernel，而是在编译、加载和候选配置测试上耗时。

**判断**：问题不是 Triton kernel 热态性能，而是 autotune 和编译缓存进入了在线关键路径。离线 benchmark 漏掉了 cold start 成本。

**修复**：固定高频 bucket 的 autotune 结果并做启动预热；限制线上动态 shape 的搜索空间；低频 shape 走保守通用实现；在指标里分开记录 compile time、autotune time 和 kernel time。

**反例**：如果服务是离线批处理，冷启动成本可摊到很长任务里，动态 autotune 可能是可接受的。在线交互服务则必须把首次编译视为用户延迟的一部分。
