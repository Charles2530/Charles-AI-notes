# DeepGEMM 解读：FP8 GEMM、JIT 与 Mega MoE

`DeepGEMM` 值得单独讲，不是因为它只是一个更快的 GEMM 库，而是因为它把现代大模型算子工程中的几个关键趋势压缩到同一个样本里：细粒度 FP8 scaling、Hopper/Blackwell 硬件特化、运行时 JIT、MoE grouped GEMM、decode masked layout，以及从单个 GEMM 走向融合通信与计算的 Mega MoE 思路。

这页定位为主线解读。源码阅读、API 名称和接入检查放到 [DeepGEMM 源码与接入附录](deepgemm-source-and-integration.md)。如果想先补底层背景，可先读 [GEMM、Attention 与融合 Kernel](gemm-attention-and-fused-kernels.md)、[低精度与量化 Kernel](low-precision-and-quantized-kernels.md) 和 [MoE 路由与稀疏 Kernel](moe-routing-and-sparse-kernels.md)。

!!! note "初学者先抓住"

    DeepGEMM 的重点不是“又一个 GEMM 更快”，而是把 FP8 scale layout、JIT、Hopper/Blackwell 数据搬运、MoE grouped GEMM 和服务态热点 shape 放在同一个接口假设里优化。它是强特化系统组件，不是通用数学库替代品。

!!! note "难点解释：为什么 scale layout 也是接口"

    FP8/FP4 低精度矩阵乘需要额外 scale 数据。scale 怎么分块、怎么对齐、怎么被 TMA 搬运，会直接影响性能和数值稳定性。因此 DeepGEMM 把 scale 排布写进 API 语义，而不是把它当成外部细节。

## 一、DeepGEMM 到底是什么

按照工程定位，`DeepGEMM` 是一组面向大模型热点工作负载的高性能 tensor core kernel 库。它覆盖 `FP8 / FP4 / BF16 GEMM`、MoE grouped GEMM、服务态注意力相关 kernel 和 Mega MoE 等路径，并采用运行时 JIT，而不是安装时预编译所有 kernel。

它的关键不是“把 dtype 改成 FP8”，而是把 workload 假设、scale layout、硬件数据搬运、kernel 形状和运行时缓存一起写进接口设计。也就是说，`DeepGEMM` 不是一个通用线性代数抽象，而是一组面向 LLM/MoE 热路径的强特化实现。

可以用一张表概括：

| 维度 | DeepGEMM 的特点 |
| --- | --- |
| 精度路线 | 细粒度 scaling 的 FP8/FP4，而不是简单低精度矩阵乘 |
| 硬件路线 | 强绑定 Hopper/Blackwell，重视 TMA、Tensor Core 和流水线组织 |
| 编译路线 | Fully JIT，按运行时 shape 和配置生成/缓存 kernel |
| 服务路线 | 关注 grouped GEMM、masked layout、decode 和 MoE 热路径 |
| 系统路线 | 从 GEMM 扩到 MQA、Mega MoE 这类更系统级 kernel |

## 二、为什么它不是普通 GEMM 库

传统 GEMM 库追求通用性：不同 dtype、layout、transpose、batch 形状、硬件和调用方式都要覆盖。`DeepGEMM` 更像把“通用性成本”换成“特定大模型热点路径的深优化”。

### 细粒度 FP8 Scaling

FP8 训练和推理最难的部分之一是 scaling。scale factor 的粒度、布局、对齐和读取方式会直接影响性能和数值稳定性。`DeepGEMM` 把 scale layout 当成接口语义的一部分，例如要求某些 scale factor 以 TMA-friendly 的方式组织。这说明低精度 kernel 的性能不只来自 tensor core，还来自“scale 数据如何被搬运和消费”。

### Fully JIT

Fully JIT 的现实吸引力在于：大模型服务中的热 shape 通常有限，但部署组合很多。如果安装时预编译全部 kernel，编译成本和二进制体积都会很高；运行时 JIT 则可以按实际 shape 编译和缓存。

代价也很明确：首次请求编译延迟、JIT cache 管理、线上可观测性和版本一致性都要处理好。JIT 是系统权衡，不是无条件收益。

### Hopper/Blackwell 特化

`DeepGEMM` 的很多价值来自对新硬件特性的直接利用。TMA、persistent thread specialization、two-level accumulation、block scheduler、SASS interleaving 等设计，都说明它把数据搬运和 tensor core 计算作为同一个流水线问题处理。它适合硬件环境较集中、工作负载较稳定、团队愿意维护特化路径的场景。

## 三、和 cuBLASLt、CUTLASS、CuTe、Triton 的关系

`DeepGEMM` 不是要替代所有现有库。更合理的理解是：它在大模型算子栈里占据“极热点特化库”的位置。

| 工具 | 更适合什么 | 主要优势 | 主要边界 |
| --- | --- | --- | --- |
| `cuBLASLt` | 通用 matmul 生产默认路径 | 稳定、覆盖广、NVIDIA 官方维护 | 对特殊 LLM/MoE 热路径不一定最激进 |
| `CUTLASS / CuTe` | 深度定制 CUDA kernel | 模板能力强、硬件控制细 | 学习和维护成本高 |
| `Triton` | 快速写中高性能自定义 kernel | 迭代快、适合实验和定制 | 极限硬件特化不如手写 CUDA |
| `DeepGEMM` | Hopper/Blackwell 上 FP8/MoE 热路径 | 假设强、路径短、JIT 友好 | 硬件和 layout 绑定强，通用性有限 |

一个成熟系统通常不是四选一，而是分层使用：

1. 默认 matmul 走 `cuBLASLt`；
2. 快速实验和非核心定制走 `Triton`；
3. 超核心长期维护 kernel 用 `CUTLASS/CuTe` 或手写 CUDA；
4. 命中 `Hopper/Blackwell + FP8 + MoE` 热点时，再考虑 `DeepGEMM`。

判断是否接入 `DeepGEMM`，关键不是“它 benchmark 快不快”，而是你的 workload 是否命中它的前提：shape 是否稳定，scale layout 是否能配合，硬件是否集中，MoE grouped GEMM 是否是瓶颈，运行时是否能管理 JIT。

## 四、在 MoE Serving 里的位置

`DeepGEMM` 在 MoE serving 中最典型的位置，是 expert dispatch 之后、combine 之前的核心 expert 计算区。简化链路如下：

```text
tokens
  -> router scores
  -> top-k expert selection
  -> dispatch / permute
  -> grouped GEMM / masked grouped GEMM
  -> activation / projection
  -> combine / unpermute
  -> residual path
```

在 prefill 阶段，token 数通常更大，grouped GEMM 更接近高吞吐路径；在 decode 阶段，token 数小、shape 动态、请求持续进出，masked layout 和 CUDA Graph 友好性更重要。

`DeepGEMM` 的价值不是“某个 GEMM 快一点”，而是把 MoE 的 expert 计算切成更适合特化的热路径块：

1. expert shape 固定但 token 数变化，适合 M 轴 grouped GEMM；
2. decode 中 shape 小而碎，masked layout 有利于稳定执行路径；
3. FP8 scale layout 与 kernel 共同设计，减少低精度额外开销；
4. Mega MoE 进一步尝试把通信、dispatch、计算和 combine 的边界向融合方向推进。

但 Mega MoE 也意味着更强系统前提。runtime、通信库、显存布局、JIT cache、CUDA Graph 和容错都要配合。它不是单个 API 替换，而是服务链路重构。

## 五、性能应该怎么读

读 `DeepGEMM` 性能时，最容易犯两个错误：把 microbenchmark 当端到端收益，把单一 shape 胜出当通用胜出。

更可靠的评估应同时看：

1. 目标 shape 是否覆盖真实流量；
2. prefill 和 decode 分开测；
3. JIT 首次编译和 cache 命中后的差异；
4. scale layout、dispatch、permute、combine 是否计入端到端；
5. 多租户、LoRA、MoE 热点专家是否破坏 shape 稳定性；
6. CUDA Graph、调度器和 KV cache 是否能与它共存；
7. 出错时是否能 fallback 到 cuBLASLt/Triton 路径。

一个常见上线顺序是：

1. 用现有 runtime 和 `cuBLASLt/Triton` 建立基线；
2. 从 profiling 中确认 MoE grouped GEMM 或 FP8 GEMM 是真实热点；
3. 在离线 replay 中替换单一路径；
4. 分 prefill/decode、分 shape bucket 测端到端收益；
5. 小流量灰度并保留 fallback；
6. 稳定后再考虑更深的 Mega MoE 融合。

## 六、边界和风险

`DeepGEMM` 的强项也是它的边界。

| 风险 | 具体表现 |
| --- | --- |
| 硬件绑定 | 非 Hopper/Blackwell 环境收益可能不明显，甚至不可用 |
| Layout 前提强 | 上游张量、scale、MoE dispatch 必须愿意配合 |
| JIT 运维复杂 | 首次编译、缓存、版本、灰度和复现都要治理 |
| Benchmark 误读 | 单 kernel 快不代表端到端快 |
| 系统耦合 | 与 runtime、CUDA Graph、通信、KV、MoE 路由强相关 |

因此 `DeepGEMM` 更适合被当成“核心热路径特化武器”，而不是默认替换所有 GEMM 的通用方案。

## 七、学习它最该抓住什么

从学习角度看，`DeepGEMM` 最值得学的不是某个 API 名字，而是三种工程思维：

1. **把 workload 假设写进接口**：dense、grouped、masked、scale layout 都不是外部杂务，而是性能路径的一部分。
2. **把硬件特性转成数据流结构**：TMA、Tensor Core、persistent scheduling、two-level accumulation 都服务于同一个目标，即减少搬运并保持计算流水线。
3. **从 kernel 走向系统热路径**：MoE serving 的瓶颈不只在 GEMM，还在 dispatch、layout、JIT、通信、combine 和 runtime 调度之间的边界。

如果只记一句话：`DeepGEMM` 展示的是现代大模型算子优化的新边界，已经从“写快一个矩阵乘”扩展到“围绕低精度、MoE、JIT、硬件和服务运行时共同设计热路径”。
