# CUDA 编程模型与内存层次来源台账

这份台账服务于 `files/operators/cuda-programming-model-and-memory.md`。正文定位是 AI kernel 入门页，不做 CUDA API 手册，也不替代 Triton、CUTLASS、FlashAttention 或 profiling 专页。它的目标是让读者先学会画数据路径：HBM、L2、shared memory、register、Tensor Core、launch 和同步怎样共同决定性能。

## 资料取舍

| 主题 | 主要来源 | 在正文中的用法 |
| --- | --- | --- |
| CUDA 编程模型 | [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) | 支撑 grid、block、thread、warp、SM、memory space 和同步语义。 |
| CUDA 优化实践 | [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) | 支撑 coalescing、shared memory、occupancy、memory throughput 和 profiling 背景。 |
| Hopper/GH100 硬件 | [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/) | 支撑 SM、L2、HBM、Tensor Core、TMA 和互连图的读法。 |
| 系统级 profiling | [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)、[Nsight Systems blog](https://developer.nvidia.com/blog/nsight-systems-exposes-gpu-optimization/) | 支撑先看系统 timeline、CPU/GPU 关联、CUDA API 和 GPU 空洞。 |
| 单 kernel profiling | [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html) | 支撑 memory throughput、warp stall、Tensor Core、occupancy 和 roofline 的下钻。 |
| GEMM kernel 组织 | [CUTLASS GEMM API documentation](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api.html) | 支撑 threadblock、warp、MMA、tile 和 epilogue 的组织方式。 |

## 写作边界

正文避免把 CUDA 优化写成“提高 occupancy”。Occupancy 只是隐藏延迟的一种条件；高性能 kernel 经常用更大的 tile、更多 register 或更多 shared memory 换数据复用，未必追求最高 occupancy。

正文也避免把 Tensor Core 写成万能答案。Tensor Core 路径依赖 dtype、shape、layout、tile、scale 和 epilogue。prefill、decode、KV cache、MoE routing 和小 batch GEMM 的瓶颈可能完全不同，只看理论 TFLOP/s 会误导。

正文没有展开具体 CUDA 代码。本站后续的 Triton、CUTLASS、DeepGEMM、FlashAttention、MoE 和 profiling 页面会处理更细的实现。这里保留在概念层：先画数据流，再判断该用库、Triton、CUTLASS 还是手写 CUDA。

## 图片使用口径

正文保留两张已有图片，没有生成新图。GH100 block diagram 用于说明 kernel 会被分配到许多 SM 和共享内存系统里，不要把 GPU 当成单一黑盒。Nsight Systems GPU starvation 图用于说明系统时间线里 GPU 空洞的诊断价值：GPU 没拿到活和 kernel 自身慢是两类问题。

这些图只作为机制解释，不用于暗示所有 GPU 都具有同样的 SM/L2/HBM 结构，也不把某篇博客截图当成完整 profiling 教程。具体 profiling 参数仍以 Nsight Systems 和 Nsight Compute 官方文档为准。

## 核对口径

正文中的 roofline 式子是教学近似，用来提醒读者先区分 compute-bound、bandwidth-bound、launch 和同步开销，不是替代 Nsight Compute 的正式 roofline 报告。式子里的 bytes moved 在真实 kernel 中还要拆到 HBM、L2、shared memory、register 和中间写回；因此正文反复强调“画数据路径”。

正文中的 AI kernel 分类是本站自己的教学组织：GEMM、attention、reduce/norm、quant/dequant、MoE routing。每类 kernel 的具体实现会随 GPU 架构、CUDA 版本、runtime、shape 和 dtype 变化；本页只给出排查主矛盾，不给固定调参表。

## 与其他页面的分工

这份台账和 `triton-programming-source-ledger.md` 的分工是：CUDA 页解释底层资源和数据流，Triton 页解释 program instance、block 级张量 DSL、autotune 和 JIT/cache/fallback。读者先懂 CUDA 页，再读 Triton 页会更容易理解为什么 Triton 不是“Python 版 CUDA”。

它和 `flashattention-long-context-source-ledger.md` 的分工是：CUDA 页解释 memory hierarchy 和 profiling 背景，FlashAttention 页解释 online softmax、IO-aware attention、长上下文和不同 GPU 架构上的演化。

它和 `linear-mlp-gemm-source-ledger.md` 的分工是：CUDA 页解释通用内存和执行层级，线性层/GEMM 页解释 \(M/N/K\)、矩阵乘数学、epilogue、MLP 和实际模型结构的关系。

## 未采用的写法

- 没有把手写 CUDA 描述成一定比 Triton 或库更快。
- 没有用单个 utilization 指标证明 kernel 好坏。
- 没有把 profiler 指标当成目标本身；指标只服务端到端延迟、吞吐和正确性。
- 没有生成新图，正文只复用仓库已有技术图和博客截图。
