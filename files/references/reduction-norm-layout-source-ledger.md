# Reduction、Norm、Layout 与 Indexing 来源台账

这份台账服务于 [Reduction、Norm、Layout 与 Indexing：小算子为什么能拖慢大模型](../operators/reduction-norm-layout-and-indexing.md)。正文目标不是列出所有小算子，而是学习高质量技术博客和公众号式长文的讲法：先用一条 hidden state 数据流建立直觉，再解释 reduction、softmax、norm、layout 和 indexing 如何共同决定端到端性能。

第一类来源是官方 GPU 与框架文档。CUDA C++ Programming Guide 和 CUDA Best Practices Guide 支撑 thread hierarchy、memory hierarchy、coalesced access、shared memory、warp/block 协作和 profiling 相关表述。PyTorch Tensor Views、Tensor Attributes、LayerNorm、RMSNorm 文档用于核对 `view`、`reshape`、`permute`、`contiguous`、stride、memory format 与归一化 API 语义。Triton Programming Guide 用于支撑 block、offset、mask、stride 和 program id 的讲法。

第二类来源是论文。FlashAttention 支撑正文中“softmax 不是单行公式，而是在线维护 max/sum 统计量并减少 HBM I/O”的表达。Layer Normalization 和 RMSNorm 论文用于核对归一化公式、均值/方差/RMS 统计量和 RMSNorm 的效率动机。正文不把 RMSNorm 写成绝对更好；它只是少掉均值中心化这条统计路径，真实收益仍受模型结构、kernel、layout 和 dtype 影响。

第三类来源是高质量博客与中文工程长文。The Illustrated Transformer、colah、Lil'Log、Distill 这类材料主要学习“先建直觉、再放公式、图进入论证”的讲解节奏。智源/OneFlow 的全栈 Transformer 推理优化长文适合学习中文系统文如何从硬件、算子、显存、KV cache 和调度分层推进。公众号、中文媒体和转载文章可以学习入口设计、例子节奏和术语表达，但性能数字、图和强 claim 必须回到论文、官方博客、项目页或官方文档。

## 主要来源

| 来源 | 类型 | 本页使用方式 | 边界 |
| --- | --- | --- | --- |
| [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html) | 官方文档 | 支撑 thread/block/warp、memory hierarchy、shared memory、warp primitive 和 GPU 执行层级。 | 不替代具体 GPU 架构 profiling。 |
| [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) | 官方文档 | 支撑 coalescing、内存访问、优化流程和 profiling 口径。 | 不给固定调参表。 |
| [PyTorch Tensor Views](https://docs.pytorch.org/docs/stable/tensor_view.html) | 官方文档 | 支撑 view/reshape/permute/contiguous、stride 与 copy 行为。 | 版本行为需按当前 PyTorch 文档核对。 |
| [PyTorch Tensor Attributes](https://docs.pytorch.org/docs/stable/tensor_attributes.html) | 官方文档 | 补充 dtype、device、layout 等张量属性概念。 | 不展开所有 memory format。 |
| [PyTorch LayerNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html) | 官方文档 | 核对 LayerNorm API 与公式语义。 | 性能实现不等同于所有 backend。 |
| [PyTorch RMSNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html) | 官方文档 | 核对 RMSNorm API 与公式语义。 | 不声称所有模型都应使用 RMSNorm。 |
| [Triton Programming Guide](https://triton-lang.org/main/programming-guide/chapter-1/introduction.html) | 官方文档 | 支撑以 block、offset、mask 和 stride 读自定义 kernel。 | Triton 示例不替代 CUDA/CUTLASS 路径。 |
| [FlashAttention](https://arxiv.org/abs/2205.14135) | 论文 | 支撑 online softmax、tile 内统计量和 I/O-aware attention。 | FlashAttention 是 attention 算法，不是所有 softmax 的唯一实现。 |
| [Layer Normalization](https://arxiv.org/abs/1607.06450) | 论文 | 支撑 LayerNorm 公式与训练稳定性背景。 | 不讨论所有现代 Transformer 变体。 |
| [RMSNorm](https://arxiv.org/abs/1910.07467) | 论文 | 支撑 RMSNorm 去均值中心化、按 RMS 重标定的公式与效率动机。 | 真实收益要看模型、kernel、dtype 和 layout。 |
| [智源/OneFlow：全栈 Transformer 推理优化](https://hub.baai.ac.cn/view/33449) | 中文工程长文 / 公众号式转载 | 学习从硬件、算子、显存、KV cache、调度逐层讲解系统瓶颈。 | 中文转述不替代论文/官方文档。 |
| [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) | 可视化博客 | 学习先用数据流建直觉，再补机制细节。 | 不作为 reduction/norm 的技术事实来源。 |
| [Distill: Visualizing Weights](https://distill.pub/2020/circuits/visualizing-weights/) | 可视化技术博客 | 学习图如何成为论证对象，而不是装饰。 | 本页未新增图片。 |

## 写作边界

正文中的“小算子常常更受内存系统支配”是工程判断，不是所有 reduction/norm/indexing kernel 的定理。某些大规模 reduction、复杂 top-k 或 fused epilogue 可能受算力、寄存器压力、同步或编译器路径影响。页面只提供排查顺序，不给固定性能预测。

正文不把 fusion 写成永远正确。融合减少 HBM 往返和 launch，但可能增加寄存器压力、降低 occupancy、缩小支持 shape 范围、破坏 graph capture 或隐藏可复用中间结果。因此正文强调按 shape bucket dispatch，并用端到端 trace、fallback 率和数值回归验收。

正文不把 `reshape`、`view`、`permute` 写成统一成本。PyTorch 视图操作是否复制数据取决于 stride、storage 和后续使用；某些操作本身只改 metadata，但后续 `contiguous()` 或 backend 要求可能触发真实搬运。页面将其作为 layout 排查原则，不替代具体框架版本行为。

正文不把 RMSNorm 写成 LayerNorm 的简单替代。RMSNorm 少掉均值中心化并常出现在现代 LLM 中，但是否更适合某模型由架构、训练稳定性、低精度路径和 kernel 实现共同决定。

正文没有新增图片，也没有使用 image2 或任何图像生成。这里刻意采用“从数据流讲起”的文字结构，学习高质量博客和公众号长文的解释节奏，而不是依赖新图支撑。

## 与站内其他台账的分工

[CUDA 编程模型与内存层次来源台账](cuda-programming-memory-source-ledger.md)负责 CUDA 基础概念、memory hierarchy、Tensor Core 和 AI kernel 分类。本台账专门服务 reduction、norm、layout、indexing 的小算子读法。

[Triton 编程模型来源台账](triton-programming-source-ledger.md)负责 Triton program id、offset、mask、autotune 和自定义 kernel 写法。本台账只引用 Triton 作为小算子表达方式，不展开完整编程教程。

[线性层、MLP 与 GEMM 来源台账](linear-mlp-gemm-source-ledger.md)负责 GEMM、epilogue、Tensor Core 和矩阵形状。本台账负责 GEMM 前后的 reduction、norm、layout 和 indexing 如何影响端到端。

[数值显存与运行时来源台账](numerics-memory-runtime-source-ledger.md)负责 dtype、显存账、roofline 和 serving runtime 的共同成本模型。本台账只把小算子放进数据流和 trace 排查中。

## 后续维护

后续维护这页时，优先补三类材料：一是官方 CUDA/PyTorch/Triton/CUTLASS 文档中关于 layout、memory format 和 reduction primitive 的新入口；二是 vLLM、TensorRT-LLM、SGLang 等 runtime 对 small kernel fusion、CUDA Graph、prefill/decode 分离的工程说明；三是高质量中文博客和公众号式长文里更适合中文读者的例子。新增中文材料时只记录“学到的讲法”，强事实仍回原始来源核对。
