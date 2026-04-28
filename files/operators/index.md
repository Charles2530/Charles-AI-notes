# 算子与编译器总览

<div class="atlas-hero">
  <div class="atlas-kicker">Operators & Compilers</div>
  <p class="atlas-lead">这一专题把模型、框架、编译器、库、kernel 和 GPU 硬件连成一条完整链路，帮助你理解性能为什么会卡在 Attention、GEMM、KV、通信和布局变换上。</p>
  <div class="atlas-chip-row">
    <span class="atlas-chip">CUDA</span>
    <span class="atlas-chip">Triton</span>
    <span class="atlas-chip">Attention Kernel</span>
    <span class="atlas-chip">Profiling</span>
  </div>
</div>

## 专题定位

<div class="atlas-meta-grid">
  <div>
    <strong>核心问题</strong>
    <p>把“模型想做什么”和“硬件实际上能高效做什么”连接起来，解释性能、显存和数值稳定性问题。</p>
  </div>
  <div>
    <strong>适合读者</strong>
    <p>适合做训练基础设施、推理 runtime、kernel 优化、量化、长上下文和系统性能分析的人。</p>
  </div>
  <div>
    <strong>阅读方式</strong>
    <p>建议先建立 CUDA/Triton 编程模型，再看热点算子、编译栈、profiling 与硬件感知排查。</p>
  </div>
</div>

## 推荐入口

<div class="atlas-card-grid">
  <a class="atlas-card" href="cuda-programming-model-and-memory/">
    <strong>CUDA 编程模型与内存层次</strong>
    <p>先弄清线程、warp、shared memory、L2、HBM 这些最基础的执行概念。</p>
  </a>
  <a class="atlas-card" href="triton-programming-model-and-autotuning/">
    <strong>Triton 编程模型与自动调优</strong>
    <p>理解更高层的块级编程和自动调优思路，建立与手写 CUDA 的互补视角。</p>
  </a>
  <a class="atlas-card" href="deepgemm-fp8-gemm-and-mega-moe.md">
    <strong>DeepGEMM 解读</strong>
    <p>从 FP8 GEMM、JIT、TMA、grouped GEMM 到 Mega MoE，读懂新一代大模型算子库在优化什么。</p>
  </a>
  <a class="atlas-card" href="deepgemm-source-and-integration.md">
    <strong>DeepGEMM 源码与接入</strong>
    <p>补充源码阅读顺序、API 地图、调用路径和真实服务接入前的检查清单。</p>
  </a>
  <a class="atlas-card" href="profiling-debugging-and-numerical-stability/">
    <strong>Profiling、调试与数值稳定性</strong>
    <p>把“会写 kernel”推进到“会验证、会定位、会判断性能问题到底卡在哪”。</p>
  </a>
</div>

大模型系统讨论如果只停留在“模型结构”“训练配方”“服务框架”三个层面，迟早会撞到一堵墙：很多关键性能问题最终都会落到算子实现、编译路径和硬件执行模型上。你也许已经知道 `FlashAttention`、`PagedAttention`、`Fused RMSNorm`、`Int4 GEMM` 这些名字，但如果不知道它们到底在优化什么、为什么会快、又为什么有些场景并不会快，那么系统层判断就很容易失真。

这一组文档的目的，不是把每个人都训练成 GPU 内核工程师，而是帮助你建立一个足够清晰的算子视角：从模型层需求，一路下钻到编译器、中间表示、kernel、内存层次和硬件执行机制，理解哪些性能瓶颈来自算法，哪些来自实现，哪些来自错误的抽象边界。

!!! tip "基础知识入口"
    算子专题默认你知道 tensor shape、dtype、显存、带宽和 kernel 的基本含义。如果不熟，先看 [张量、Shape 与计算图](../foundations/tensors-shapes-and-computation-graphs.md)、[线性层、MLP 与 GEMM](../foundations/linear-layers-mlp-and-gemm.md) 和 [数值、显存与运行时基础](../foundations/numerics-memory-and-runtime-basics.md)。

## 1. 为什么算子会成为主问题

在大模型时代，很多性能损失已经不是“模型太大”这么简单，而是：

1. 访问模式不连续，HBM 带宽被打满；
2. kernel 太碎，launch 开销和中间张量读写太多；
3. 跨卡通信和单卡 kernel 没有重叠；
4. 同样的数学公式，因为数据布局不同，性能差一个数量级；
5. 量化虽然让权重更小，但反量化和布局变换把收益吃掉；
6. 编译器没有抓住融合机会，导致系统一直在“搬数据而不是算数据”。

因此所谓“系统优化”，最后很大一部分其实是在重新安排计算与数据移动的关系。

## 2. 从模型到硬件的六层栈

一个更完整的 AI 系统栈，至少可以分成六层：

1. **模型层**：Transformer、MoE、Diffusion、World Model、VLA 等；
2. **框架层**：PyTorch、JAX、OneFlow 等提供张量和自动微分抽象；
3. 图与编译层：`torch.compile`、Inductor、XLA、TensorRT、TVM、MLIR 等做图级重写和代码生成；
4. **库与模板层**：cuBLAS、cuDNN、NCCL、CUTLASS、CuTe、FlashAttention 等提供高性能基元；
5. **Kernel 层**：CUDA、Triton 或框架自带 codegen 生成的具体 GPU 程序；
6. **硬件层**：SM、Tensor Core、register、shared memory、L2、HBM、NVLink 等。

这六层并不是完全独立的。模型决定算子模式，算子模式决定编译器是否有融合空间，编译器决定 kernel 形态，kernel 决定是否真正利用到硬件能力。

### 2.1 为什么这条链必须连起来看

如果只看最上层，你会觉得 `FlashAttention` 只是“Attention 的更快实现”；  
如果只看最底层，你又会陷入“寄存器、warp、bank conflict”的局部细节。  
真正有用的视角，是把这两端接起来：

1. 为什么模型里会出现这个算子；
2. 这个算子到底受什么硬件约束；
3. 编译器或手写 kernel 能否改变这些约束；
4. 这种优化对训练和推理的收益是否一致。

## 3. 算子并不只有 GEMM

很多人入门时会把“算子优化”等同于 GEMM 优化，这只对了一半。GEMM 确实是核心，但现代大模型系统真正常见的热点算子至少包括：

1. GEMM 与 batched GEMM；
2. Attention 及其变体；
3. Softmax、Reduce、LayerNorm、RMSNorm；
4. Embedding / gather / scatter；
5. KV cache 读写与页面管理；
6. 量化与反量化；
7. 通信相关 kernel，如 all-reduce、all-gather 的 overlap；
8. 布局变换、transpose、swizzle、packing。

其中很多算子并不是以 FLOPs 为主导，而是以带宽、布局和同步为主导。

## 4. 训练与推理的算子画像并不一样

**训练阶段更关心**：

1. 大 batch 下的高吞吐 GEMM；
2. 反向传播与 activation recompute；
3. 分布式通信与计算重叠；
4. 混合精度、loss scaling 和 optimizer kernel；
5. 长上下文训练中的 attention 和 activation 内存。

**推理阶段更关心**：

1. prefill 与 decode 的不同负载形态；
2. 小 batch 或 continuous batching 下的 kernel 粒度；
3. KV cache 管理；
4. 量化 kernel 是否真正提速；
5. tail latency 与多租户隔离。

所以“训练里很快的 kernel”不一定等价于“服务里最好用的 kernel”。

## 5. 为什么 Triton 和 CUDA 要单独成组

这两个工具之所以值得单独讲，不只是因为它们流行，而是因为它们分别代表两种常见思路：

1. `CUDA` 更接近硬件与执行细节，让你直接控制线程层级、共享内存、warp 原语、异步拷贝和 Tensor Core 使用方式；
2. `Triton` 更接近张量块级编程和 kernel 生成，让你用更高层的抽象表达 tile、mask、block pointer 和自动调优。

如果你只理解其中一种，很容易在工程判断上失衡：

1. 只懂 CUDA，容易高估手写 kernel 的必要性，低估编译器和模板库的价值；
2. 只懂 Triton，容易忽略硬件实际限制，对寄存器压力、occupancy、访存细节不够敏感。

## 6. 这组文档会回答哪些问题

**这组内容主要围绕六类问题组织**：

1. GPU 为什么适合 AI，CUDA 的执行模型是什么；
2. Triton 的 block-level 编程范式和自动调优逻辑是什么；
3. GEMM、Attention、Norm、Reduce 为什么是最核心的优化对象；
4. CUTLASS、CuTe、编译器、模板库和手写 kernel 之间如何分工；
5. 如何用 profiling、roofline、微基准和数值校验来判断优化是否真实有效；
6. 训练系统、推理引擎和内核实现之间如何连成完整链路。

## 7. 一个算子工程师真正反复做的事情

现实里，算子优化工作通常不是“凭直觉写更快代码”，而是一套反复循环的过程：

1. 先确定热点算子；
2. 判断它是算力受限还是带宽受限；
3. 找到关键数据布局与访存模式；
4. 设计 tile、分块、流水线和 fusion；
5. 验证正确性与数值稳定性；
6. 做 profile；
7. 再根据寄存器压力、occupancy、stall reason 和带宽利用率继续迭代。

这和只盯着 benchmark 排行不同，它更像一门系统化实验科学。

## 8. Roofline 思维为什么重要

很多算子优化讨论最终都会回到一个简单问题：这个 kernel 到底是算力受限，还是带宽受限？

**Roofline 的直觉表达为**：

$$
\text{Performance}
\le
\min(\text{Peak FLOPs}, \text{Bandwidth} \times \text{Arithmetic Intensity}).
$$

它并不是精确预测器，但非常适合做第一层判断：

1. 如果算子算术强度很低，就别幻想纯靠 Tensor Core 提升巨大性能；
2. 如果算子已经逼近带宽上限，就应该更多关注 fusion、layout 与访存合并；
3. 如果算子离 roofline 还很远，说明实现层面还有明显浪费。

## 9. 从参考源可以借到什么视角

你给的几个参考源，刚好覆盖了三条很互补的主线：

1. `AI System Docs` 更像全栈课程，提供从芯片、编译、框架到推理系统的完整纵深；
2. `how-to-optim-algorithm-in-cuda` 更像算子与 kernel 优化实践仓库，覆盖 GEMM、Softmax、LayerNorm、FlashAttention、Triton、Cutlass、profiling 等大量具体技巧；
3. `AwesomeWorldModels` 则提醒我们，世界模型一侧也越来越依赖生成式模拟、视频建模、长上下文和系统实现能力，最终也会回到训练与推理内核问题。

也就是说，算子专题不是一个孤立的低层补充，而是可以向上连接训练、推理、世界模型和具身系统的底盘。

## 10. 一个更实用的抽象：算子优化四问

读任何 kernel 或框架实现时，都可以先问四个问题：

1. 这个算子主要搬了什么数据；
2. 这些数据在哪一级内存里来回移动；
3. 哪些操作本来可以融合但还没融合；
4. 这个算子真正卡在什么地方，是带宽、寄存器、同步还是 launch 开销。

这四问比“它用了什么黑科技”更重要，因为它能帮助你把技巧还原成原理。

## 11. 为什么系统工程师也应该懂算子

不是每个系统工程师都要手写 CUDA，但如果完全不懂算子，很容易在架构层做出错误决策，例如：

1. 选了一个数学上看似更省的算法，却在 kernel 层面非常难高效实现；
2. 引入了过多布局变换，让 pipeline 在搬运数据上损失远大于节省的 FLOPs；
3. 误把模型服务问题当成调度问题，实际瓶颈却是 decode kernel；
4. 误把训练不稳定归因于优化器，实际问题却来自数值精度与 fused kernel 实现。

理解算子，不是为了替代上层系统设计，而是为了让上层系统设计不建立在错误假设上。

## 12. 推荐阅读顺序

**建议按下面顺序阅读**：

1. [CUDA 编程模型与内存层次](cuda-programming-model-and-memory.md)
2. [Triton 编程模型与自动调优](triton-programming-model-and-autotuning.md)
3. [GEMM、Attention 与融合 Kernel](gemm-attention-and-fused-kernels.md)
4. [CUTLASS、CuTe 与编译栈](cutlass-cute-and-compiler-stack.md)
5. [Profiling、调试与数值稳定性](profiling-debugging-and-numerical-stability.md)

前两篇建立编程模型，中间两篇建立常见热点算子的实现直觉，最后一篇帮助你把“会写”变成“会验证、会定位、会迭代”。

## 快速代码示例

```python
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)

def triton_add(x, y):
    out = torch.empty_like(x)
    n = x.numel()
    add_kernel[(triton.cdiv(n, 256),)](x, y, out, n, BLOCK=256)
    return out
```

这段代码是一个最小 Triton kernel：通过 `program_id` 映射线程块，`mask` 处理越界元素，再把结果写回输出。它适合作为模板扩展到 layernorm、激活融合等更复杂算子。

## 13. 小结

算子、编译器和内核实现，不是 AI 系统里的边角知识，而是连接“模型想做什么”和“硬件实际上能做什么”的中间层。大模型训练、推理、量化、长上下文、世界模型和具身控制之所以越来越依赖系统工程，很大程度上就是因为这个中间层不再能被简单忽略。

理解这组内容的最终目标，不是掌握所有底层细节，而是形成一种判断力：当系统变慢、变贵、变不稳时，你能不能快速判断这是模型问题、框架问题、编译器问题，还是 kernel 问题。
