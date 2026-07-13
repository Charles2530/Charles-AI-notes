# 算子与性能 QA

这一页回答 GEMM、attention kernel、CUDA/Triton、FlashAttention、roofline、MFU 和 profiling 中的高频问题。面试时要把局部 kernel 加速、端到端吞吐、数值正确性和维护成本分开讲。

## GPU 与矩阵计算基础

### Q：为什么大模型里 GEMM 这么核心？

**面试回答。** Transformer 的大部分计算都可以归结为矩阵乘法：Q/K/V projection、output projection、MLP 的 up/down projection、MoE expert 里的线性层，本质都是 GEMM。GEMM 性能决定了模型能否有效利用 GPU Tensor Core、内存带宽和 cache 层级。

\[
C = AB,\quad A\in\mathbb{R}^{m\times k},\ B\in\mathbb{R}^{k\times n},\ C\in\mathbb{R}^{m\times n}
\]

**追问展开。** 面试时可以补一句：训练和 prefill 阶段常有大矩阵，适合高吞吐 GEMM；decode 阶段 batch 小、shape 动态，GEMM 可能不再是唯一瓶颈。  
**易错点 / 边界。** GEMM 快不代表模型端到端快；attention、KV cache、通信、数据加载和调度都可能成为瓶颈。  
**继续读。** [线性层、MLP 与 GEMM](../foundations/linear-layers-mlp-and-gemm.md) / [GEMM、Attention 与融合算子](../operators/gemm-attention-and-fused-kernels.md)

### Q：GPU 为什么适合大规模矩阵运算？

**面试回答。** GPU 有大量并行执行单元和高内存带宽，适合把矩阵乘法拆成 tile，并在许多线程块上并行执行。现代 GPU 的 Tensor Core 对 FP16/BF16/FP8/INT8 等矩阵乘提供专门硬件路径，所以大模型训练和推理能获得很高吞吐。

**追问展开。** 真正高效的 GEMM 不是简单并行三重循环，而是要做 tiling、共享内存复用、寄存器复用、向量化加载和合适的 warp/threadblock 布局。  
**易错点 / 边界。** GPU 忙不等于有效计算；shape 不合适、访存不连续、kernel 太小或同步过多都会让吞吐下降。  
**继续读。** [CUDA 编程模型与内存层次](../operators/cuda-programming-model-and-memory.md) / [数值显存与运行时基础](../foundations/numerics-memory-and-runtime-basics.md)

### Q：memory-bound 和 compute-bound 怎么区分？

**面试回答。** Compute-bound 表示主要受计算峰值限制，memory-bound 表示主要受数据搬运限制。Roofline 模型用算术强度 \(I\) 判断一个 kernel 更可能受哪个瓶颈限制：

\[
I=\frac{\text{FLOPs}}{\text{Bytes moved}},\quad
\text{attainable FLOPs}=\min(P_{\text{peak}}, I\cdot B_{\text{mem}})
\]

**追问展开。** 如果一个算子 FLOPs 多、数据复用高，可能 compute-bound；如果读写大量数据但计算很少，比如某些 norm、copy、KV 读取，就更可能 memory-bound。  
**易错点 / 边界。** Roofline 是一阶模型，真实性能还受 cache、bank conflict、occupancy、指令调度和同步影响。  
**继续读。** [Roofline、Benchmark 与案例](../operators/roofline-benchmarking-and-case-studies.md) / [Profiling、Debugging 与数值稳定](../operators/profiling-debugging-and-numerical-stability.md)

### Q：为什么 GPU utilization 100% 不等于训练跑满？

**面试回答。** GPU utilization 往往只表示 GPU 上有 kernel 在运行，不表示这些 kernel 使用了多少 Tensor Core、SM、带宽或有效 FLOPs。训练是否跑满更应该看 MFU、SM efficiency、Tensor Core utilization、内存吞吐、通信 overlap 和端到端 step time。

**追问展开。** 一个低效 kernel 也能让 GPU utilization 显示很高；CPU launch overhead、数据加载慢、通信等待也可能让时间线看起来“很忙但没产出”。  
**易错点 / 边界。** 监控面板只能提示方向，真正定位要看 profiler trace 和 workload 分解。  
**继续读。** [GPU Utilization：为什么 100% 不等于训练跑满](../paper-deep-dives/foundations/gpu-utilization-misleading.md) / [集群运维与实验管理](../training/cluster-operations-and-experiment-management.md)

### Q：MFU 是什么，为什么常被用来评估训练效率？

**面试回答。** MFU 是 Model FLOPs Utilization，用模型理论 FLOPs 除以实际训练时间和硬件峰值 FLOPs，衡量训练把硬件峰值计算能力用到了多少。它比单纯 GPU utilization 更贴近“有效模型计算”。

\[
\text{MFU}=\frac{\text{model FLOPs per step}}{\text{step time}\times \text{hardware peak FLOPs}}
\]

**追问展开。** 面试时要补：MFU 高说明大部分时间花在模型有效计算上，但它依赖 FLOPs 估算口径。对 MoE、activation checkpointing、sequence packing、稀疏 attention，FLOPs 估算会更复杂。  
**易错点 / 边界。** MFU 不是用户价值指标；模型质量、收敛速度和系统稳定性仍要单独看。  
**继续读。** [模型训练从零入门](../training/beginner-model-training-primer.md) / [Roofline、Benchmark 与案例](../operators/roofline-benchmarking-and-case-studies.md)

## Attention 与融合算子

### Q：FlashAttention 到底省了什么？

**面试回答。** 标准 attention 会显式构造和读写 \(n\times n\) 的 attention matrix，长序列下 HBM 读写非常贵。FlashAttention 用分块计算和 online softmax，把 Q/K/V 分块搬进 SRAM，边算边更新输出，避免把完整 attention matrix 写回 HBM。它没有改变 attention 数学，而是减少 IO。

**追问展开。** 面试回答一定要说“IO-aware”：FlashAttention 的核心不是近似，而是精确 attention 的访存优化。它用 tiled computation 保持数值稳定，并显著降低训练长序列时的显存和带宽压力。  
**易错点 / 边界。** 具体收益依赖 sequence length、head dim、mask、硬件和实现；不是所有 attention 变体都能直接套。  
**继续读。** [FlashAttention 与长上下文](../operators/long-context-and-flashattention-evolution.md) / [Transformer 输入与注意力](../foundations/transformer-attention-and-tokenization.md)

### Q：online softmax 为什么能服务 FlashAttention？

**面试回答。** Softmax 需要全局最大值和归一化分母。Online softmax 在分块读取 score 时维护当前最大值 \(m\) 和归一化项 \(l\)，每来一个新 block 就更新它们，因此不用一次保存完整 score matrix。

\[
m_{\text{new}}=\max(m_{\text{old}}, m_{\text{block}})
\]

\[
l_{\text{new}}=e^{m_{\text{old}}-m_{\text{new}}}l_{\text{old}}+
e^{m_{\text{block}}-m_{\text{new}}}l_{\text{block}}
\]

**追问展开。** 这套更新保证数值稳定，因为始终围绕当前最大值做指数缩放。输出向量也可以类似增量更新。  
**易错点 / 边界。** 低精度、mask、长序列和特殊 attention pattern 都需要仔细处理，否则容易出现误差或边界 bug。  
**继续读。** [Reduction、Norm、Layout 与索引](../operators/reduction-norm-layout-and-indexing.md) / [FlashAttention 与长上下文](../operators/long-context-and-flashattention-evolution.md)

### Q：算子融合为什么能提升性能？

**面试回答。** 算子融合把多个连续小算子合成一个 kernel，减少中间张量写回 HBM 和重复 kernel launch。比如 bias + activation、dropout + residual + norm、dequant + matmul 等，如果拆开执行会反复读写内存，融合后可以在寄存器或 shared memory 中完成更多步骤。

**追问展开。** 面试时可以说融合主要省两类成本：内存流量和 launch overhead。对 memory-bound 的小算子，融合收益特别明显。  
**易错点 / 边界。** 融合会增加寄存器压力和代码复杂度；如果 shape 多变或融合后 occupancy 下降，可能反而变慢。  
**继续读。** [GEMM、Attention 与融合算子](../operators/gemm-attention-and-fused-kernels.md) / [高级 Kernel 模式与 Shape 特化](../operators/advanced-kernel-patterns-and-shape-specialization.md)

### Q：Serving attention 和训练 attention 为什么不完全一样？

**面试回答。** 训练 attention 通常处理完整序列并需要保存中间状态用于反向传播；serving decode attention 每步只有新 token 的 query，但要读取历史 KV cache。训练更关注大块矩阵和反向显存，serving 更关注 KV layout、cache 命中、batch 调度和尾延迟。

**追问展开。** Prefill attention 更像训练前向，decode attention 更像 KV lookup + 小 batch attention。很多推理框架会分别优化 prefill kernel 和 decode kernel。  
**易错点 / 边界。** 训练 kernel 快不代表 decode 快；优化要绑定具体阶段和 workload。  
**继续读。** [Serving Attention 与 KV kernels](../operators/serving-attention-and-kv-kernels.md) / [GPU kernels、batching 与内存系统](../inference/gpu-kernels-batching-and-memory-systems.md)

### Q：低精度 kernel 最容易错在哪里？

**面试回答。** 低精度 kernel 常见错误包括 scale 计算不匹配、累加精度不足、反量化顺序错误、溢出/下溢、layout 不一致、边界 shape 漏处理和校准分布不代表真实输入。性能上，还可能因为多余 cast 或 dequant/requant 抵消低比特收益。

**追问展开。** 面试回答可以按“数值正确性 + 性能正确性”说：前者看误差、NaN、输出一致性；后者看 kernel 是否真的走低精度硬件路径，是否减少了内存流量。  
**易错点 / 边界。** bit 数更低不保证更快或更好；缺 kernel 支持时，低精度可能只是增加转换成本。  
**继续读。** [低精度与量化 kernels](../operators/low-precision-and-quantized-kernels.md) / [QAT、kernels 与 KV cache](../quantization/qat-kernels-and-kv-cache.md)

## CUDA、Triton 与编译器

### Q：Triton 相比手写 CUDA 的优势是什么？

**面试回答。** Triton 用 block-level 编程模型让开发者直接表达 tile 级计算，不必手写大量 thread/warp 细节。它适合快速写深度学习自定义 kernel，并通过 autotune 搜索 block size、num warps、pipeline stages 等配置。

**追问展开。** 面试里可以说：Triton 介于 PyTorch eager 和手写 CUDA 之间，开发效率高，适合 GEMM 变体、fused op、attention 后处理和量化 kernel 原型。  
**易错点 / 边界。** Triton 不是性能保证；复杂通信、极致硬件特性、特殊内存路径或高度定制 GEMM 仍可能需要 CUDA/CUTLASS。  
**继续读。** [Triton 编程模型与自动调优](../operators/triton-programming-model-and-autotuning.md) / [Triton paper](https://doi.org/10.1145/3315508.3329973)

### Q：Autotuning 在调什么？

**面试回答。** Autotuning 搜索 kernel 的实现参数，例如 block size、tile shape、num warps、num stages、split-k、grouping 和 layout。不同 shape 和硬件上，最优参数可能不同，所以手工固定一个配置很难覆盖所有场景。

**追问展开。** 在推理服务里，autotune 通常要配合 shape bucket；否则请求 shape 太多，搜索和缓存成本会变大。训练里则要关注常见 batch/seq shape。  
**易错点 / 边界。** Autotune 结果对硬件、驱动、输入 shape 和数据类型敏感；上线要做回归和 fallback。  
**继续读。** [Triton 编程模型与自动调优](../operators/triton-programming-model-and-autotuning.md) / [Workload 建模与 Shape Buckets](../operators/workload-modeling-and-shape-buckets.md)

### Q：CUTLASS / CuTe 适合解决什么问题？

**面试回答。** CUTLASS 提供高度可组合的 CUDA GEMM 和相关算子模板，CuTe 用 layout 和张量代数描述 tile 级数据移动。它们适合构建高性能、可维护的 NVIDIA GPU kernel，尤其是矩阵乘、卷积、低精度和融合算子。

**追问展开。** 如果面试问“为什么不用 Triton”，可以答：Triton 更快开发，CUTLASS/CuTe 更接近底层控制和极致性能，适合成熟高性能库。  
**易错点 / 边界。** 学习和维护成本高；已有 cuBLAS/cuDNN/FlashAttention 能满足需求时，不要为了自定义而自定义。  
**继续读。** [CUTLASS、CuTe 与编译器栈](../operators/cutlass-cute-and-compiler-stack.md) / [NVIDIA CUTLASS GEMM docs](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api.html)

### Q：PTX 和 SASS 有什么区别？

**面试回答。** PTX 是 NVIDIA 的中间表示，类似虚拟 ISA；SASS 是针对具体 GPU 架构生成的机器指令。检查 PTX/SASS 可以确认编译器是否用了预期指令，例如 Tensor Core 指令、向量化加载或特定 memory op。

**追问展开。** 常见使用场景是 kernel 性能和数值问题已经定位到很底层，需要确认编译结果。比如 Triton 代码看起来对，但 SASS 没走期望的 `mma` 指令，就可能解释性能差。  
**易错点 / 边界。** 不要一上来就看汇编；大多数问题先用 profiler、shape 分析和端到端 trace 定位。  
**继续读。** [PTX、SASS 与编译器检查](../operators/ptx-sass-and-compiler-inspection.md) / [Runtime Dispatch 与 Kernel 选择](../operators/runtime-dispatch-and-kernel-selection.md)

## Profiling、通信与维护

### Q：Nsight Systems 和 Nsight Compute 分别看什么？

**面试回答。** Nsight Systems 看端到端时间线：CPU launch、GPU kernel、通信、等待、数据加载和 overlap；Nsight Compute 深挖单个 kernel：occupancy、SM utilization、memory throughput、warp stall、指令和 roofline。一般先用 Systems 找时间花在哪里，再用 Compute 看单个热点 kernel 为什么慢。

**追问展开。** 面试回答可以给流程：先看 step timeline，找到 GPU 空洞、通信等待或长 kernel；再对热点 kernel 做 Compute；最后回到端到端验证优化是否有效。  
**易错点 / 边界。** 只看单 kernel 会忽略系统调度，只看系统时间线又不知道 kernel 内部瓶颈。  
**继续读。** [Profiling、Debugging 与数值稳定](../operators/profiling-debugging-and-numerical-stability.md) / [硬件感知 Debug 清单](../operators/hardware-aware-debug-checklist.md)

### Q：通信和计算重叠为什么重要？

**面试回答。** 分布式训练中 all-reduce、reduce-scatter、all-gather、all-to-all 都可能占用大量时间。如果通信完全串在计算后面，step time 就是计算时间加通信时间；如果能把通信和反向计算、梯度规约或参数预取重叠，总 step time 会下降。

**追问展开。** 例子：反向传播从后往前算梯度，较后层梯度算完后可以先通信，同时继续计算前面层梯度。FSDP/ZeRO 也会尝试参数 all-gather 和计算 overlap。  
**易错点 / 边界。** Overlap 依赖 bucket size、拓扑、stream、调度顺序和内存余量；配置不当可能更慢或更占显存。  
**继续读。** [通信 kernels 与 overlap](../operators/communication-kernels-and-overlap.md) / [分布式训练与 Checkpoint](../training/distributed-training-and-checkpointing.md)

### Q：Context parallel / RingAttention 这类长上下文并行解决什么？

**面试回答。** 长上下文训练时，单卡保存完整序列的 attention 激活和 KV 会爆显存。Context parallel 把序列维度切到多张 GPU 上，让每张卡只持有部分上下文；RingAttention 这类方法再通过环形通信让不同分片的 Q/K/V 逐段交互，避免单卡物化完整 attention。

**追问展开。** 它解决的是长序列的显存和通信组织问题，不是把 attention 数学复杂度变成免费。收益取决于序列长度、GPU 拓扑、通信带宽、overlap 和 kernel 实现。  
**易错点 / 边界。** 能训练更长上下文不代表模型一定会有效利用长上下文；还要看数据、位置编码、评测和推理 KV 成本。  
**继续读。** [FlashAttention 与长上下文](../operators/long-context-and-flashattention-evolution.md) / [RingAttention：近无限上下文训练](../paper-deep-dives/world-models/ringattention.md)

### Q：Shape bucket 对推理服务有什么用？

**面试回答。** 真实请求长度多变，完全动态 shape 会让 kernel 选择、CUDA Graph、内存规划和 autotune 变复杂。Shape bucket 把相近长度归入有限集合，用少量 padding 换取更稳定的 kernel 和更低调度开销。

**追问展开。** 比如把 prompt 长度归到 512、1024、2048 等 bucket，服务端可以预热对应 kernel 和 memory pool。这个策略常和 CUDA Graph、continuous batching、KV block 管理一起用。  
**易错点 / 边界。** bucket 太粗浪费 padding，太细增加管理复杂度；要按真实流量分布调。  
**继续读。** [Workload 建模与 Shape Buckets](../operators/workload-modeling-and-shape-buckets.md) / [推理服务系统](../inference/serving-systems.md)

### Q：CUDA Graph 为什么能降低延迟？

**面试回答。** CUDA Graph 把一段固定 GPU 操作捕获成图，重复执行时减少 CPU kernel launch 和调度开销。对 decode 这种高频小步、固定或 bucketed shape 的场景，launch overhead 可能很明显，CUDA Graph 能降低抖动和尾延迟。

**追问展开。** 它要求 shape、内存地址和执行路径相对稳定，所以常和 shape bucket、静态内存池、固定 batch slot 配合。  
**易错点 / 边界。** 动态控制流、动态 shape 和频繁内存变化会降低适用性；捕获失败或 graph 过多也会增加复杂度。  
**继续读。** [Runtime Dispatch 与 Kernel 选择](../operators/runtime-dispatch-and-kernel-selection.md) / [推理运行时](../inference/serving-runtimes-vllm-sglang-and-tensorrt-llm.md)

### Q：自定义算子上线前要测什么？

**面试回答。** 至少要测数值一致性、边界 shape、不同 dtype、不同 GPU、性能回归、反向梯度、异常输入、fallback 和端到端业务指标。Microbenchmark 只能说明局部快，不能证明训练或服务整体收益。

**追问展开。** 面试里可以给验收清单：对比 PyTorch reference，设置误差容忍；覆盖小/大/非对齐 shape；跑 profiler；接入真实模型；最后看质量和稳定性。  
**易错点 / 边界。** 自定义 kernel 会带来维护成本；如果收益只在少数 shape 上出现，要谨慎合入。  
**继续读。** [算子测试、回归与维护](../operators/testing-regression-and-maintenance.md) / [自定义 Op 开发与框架集成](../operators/custom-op-development-and-framework-integration.md)

### Q：如何判断一次 kernel 优化是否值得合入？

**面试回答。** 我会先确认它优化的是真实 workload 的主瓶颈，然后看性能提升、数值误差、代码复杂度、覆盖 shape、fallback 和维护成本。值得合入的优化应该同时有 profiler 证据和端到端收益，而不是只在孤立 microbench 上更快。

**追问展开。** 一个好的回答要包含“证据闭环”：改动前瓶颈在哪里，改动后指标提升多少，质量是否不变，哪些 shape 受益，失败时怎么回退。  
**易错点 / 边界。** 如果优化增加大量分支、破坏可读性或只对罕见 shape 有收益，可能不值得。  
**继续读。** [Kernel 成本模型与选择启发](../operators/kernel-cost-models-and-selection-heuristics.md)
