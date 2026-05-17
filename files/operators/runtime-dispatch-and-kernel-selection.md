# Runtime Dispatch 与 Kernel 选择

很多高性能系统并不是靠“一个万能 kernel”获胜，而是靠一整套运行时 dispatch 机制：根据 shape、dtype、layout、设备类型、batch 形态和任务阶段，为当前请求挑选最合适的 kernel 路径。  
也就是说，真正的性能系统不仅要有好 kernel，还要有**挑 kernel 的能力**。

!!! note "读法定位"
    这页先回答“Runtime Dispatch 与 Kernel 选择”在「算子与编译器」里的位置：它解决什么局部问题，依赖哪些前置，最后会影响哪类工程或研究判断。
    **前置**：先理解张量 shape、显存层次和 GEMM；系统指标卡住时回推理或训练专题。 必要时先回 [算子与编译器入口](index.md)、[基础知识](../foundations/index.md) 或 [术语表](../glossary/index.md)。
    **主线关系**：把模型里的矩阵乘、attention、通信和低精度路径落到 GPU 时间线上，看瓶颈如何被 kernel 与编译栈改变。

!!! note "初学者先抓住"
    Dispatch 就是运行时的“选路器”。同一个算子在不同 shape、dtype、batch 和硬件上，最优 kernel 可能不同；选错路径会让好 kernel 也发挥不出来。

!!! example "有趣例子：导航选路线"
    市区短途适合走小路，高速长途适合上快速路，货车还要避开限高。Kernel dispatch 也是根据当前请求条件选路线，而不是永远走同一条路。

## 为什么运行时选择很重要

因为现实工作负载从来不是单一 shape：长短序列混在一起，prefill 和 decode 完全不同，某些模型走 BF16、某些走 INT4，有的请求带 LoRA、有的没有，多 GPU、不同卡型和不同 batch 画像也会同时存在。

在这种情况下，单一路径几乎不可能始终最优。

更关键的是，真实系统面对的并不是“平均 shape”，而是**一整个分布**。同一套服务，白天可能以短对话为主，晚上可能混入大量长文档；训练系统在不同阶段也可能经历 warmup、长上下文 curriculum、恢复后重排等变化。若没有运行时选择逻辑，系统只能用一个静态最优去硬扛动态负载，结果通常是热路径不够快、冷路径又不够稳。

## Dispatch 真正在做什么

一个较完整的 dispatch 系统通常会根据 shape、dtype、是否 contiguous、head dim / hidden size、设备能力、训练或推理阶段，以及 prefill / decode 形态，把请求分流到不同 kernel。

一个完整的 dispatch 系统通常不只是简单 `if-else`。它背后包含可行性、性能和风险三层判断：当前输入是否满足某个 kernel 的前置条件；在可行路径里哪一条更可能快；为了一个低频 shape 走新路径，是否值得承担编译、缓存或稳定性成本。

因此，dispatch 真正做的是把“很多可能的实现”压缩成“当前这一刻最合适的一条执行路径”。

### 2.1 为什么 dispatch 是性能系统的一部分

如果 dispatch 错了，大 shape 可能走到只适合小 shape 的 kernel，decode 可能误走 prefill 路径，明明能命中更优融合实现却退回通用 fallback，编译缓存命中率也会下降。

所以 dispatch 不是简单胶水，而是决定性能边界的关键逻辑。

很多团队第一次做优化时，会把注意力几乎全部放在 kernel 本身，觉得只要把某个 kernel 提速 20%，系统自然会更快。现实往往不是这样。若最优 kernel 只命中了 10% 的请求，而其余 90% 还在走保守路径，那么整体收益可能很有限。dispatch 的职责，正是确保优化成果真的落在高频路径上。

这也是为什么成熟系统会把 dispatch 规则、命中率统计、fallback 触发原因和性能回归一起纳入 profiling。否则你只能看到“库里明明有快 kernel”，却看不到“为什么它在生产里几乎没被用到”。

### 2.2 一个常见的 dispatch 判定链

在推理和训练系统里，一个更贴近工程实际的判定链通常是：先按设备和架构决定候选集合，再按任务阶段区分 prefill、decode、训练前向和训练反向，再按长度区间、head dim、batch 画像划分 shape family，最后根据 layout 和缓存状态微调。

真正稳定的 dispatch，很少一开始就追求“最复杂的最优规则”，而是先把这条链条做清楚，再逐步把高价值分支特化出来。

!!! note "常见误区：dispatch 只是几行 if-else"
    简单 demo 里 dispatch 看起来像条件分支，但生产系统里的 dispatch 更像一份运行时合约：哪些输入能走快路径，哪些必须回退，哪些 shape 值得编译缓存，哪些异常要记录。没有命中率、回退原因和尾部 shape 统计，规则越复杂反而越难维护。

## Shape Family 的思维

成熟系统常会把工作负载划分成若干 shape family，例如小 batch decode、长 prompt prefill、常见 hidden size / head dim、量化与非量化分支。

然后为每类 family 挑选对应 kernel。  
这比试图用一个 kernel 打天下更现实。

所谓 shape family，核心不是把每个 shape 单独编号，而是找出**性能行为相近的一组输入**。例如某些 attention kernel 对 `head_dim=128` 和 `head_dim=160` 的反应完全不同，那它们就不应被看成同一类；某些 decode 负载虽然 sequence length 不同，但都受限于小 batch 和 KV 访存，那它们可以共用一类规则。

做 shape family 划分时，最常见的错误是过细或过粗。过细会导致规则爆炸、缓存膨胀和维护困难；过粗则会让“家族内部”性能差异过大，最后谁都没真正被优化。好的 family 设计，应该让同一家族既足够常见，又足够相似。

### 3.1 三类最常见的 shape family

在大模型系统里，尤其值得单独管理的通常是三类：attention family 按 prefill/decode、head dim、页大小和 KV 布局划分；GEMM family 按 `M/N/K` 区间、量化类型、转置布局和 expert 分组划分；MoE family 按每个 expert 的 token 数、top-k 策略、重排方式和 combine 路径划分。

这三类之所以重要，是因为它们经常既决定绝大部分计算量，也最容易因为输入分布变化而发生性能退化。

## Fallback 的角色

运行时 dispatch 不能只考虑“最快路径”，还必须有 fallback。边界 shape 总会出现，新卡型或新版本未必覆盖，某些输入 layout 不满足假设，编译缓存未命中时也要能继续运行。成熟系统通常让高频 shape 走特化、低频边界走通用实现，并保证正确性优先于局部性能。

很多工程师会把 fallback 理解成“没优化好时的临时方案”，其实更准确的理解是：fallback 是**执行系统的安全边界**。它保证当新路径不成立、编译缓存失效、输入不满足假设、运行时状态异常时，系统仍然有一条正确且可解释的路可以走。

在长期维护里，fallback 甚至经常比快路径更重要。因为快路径的价值来自性能，fallback 的价值来自**让系统在不确定条件下继续可用**。没有 fallback，再快的专用 kernel 也不适合真实生产环境。

### 4.1 什么样的 fallback 才算合格

一个好的 fallback 至少应满足四点：正确性稳定、语义一致、可观测、成本可接受。它可以慢，但不能静默出错，也不能慢到让系统整体失控。

只有这样，fallback 才不只是“兜底”，而是真正可进入灰度、升级和回归体系的一部分。

## 运行时选择与编译缓存

一旦系统开始大量 shape specialization，就必须考虑编译时间、缓存大小、首次命中成本和在线抖动。这意味着 dispatch 不只是选最快 kernel，还要判断当前是否已有缓存版本、为低频 shape 编译新版本是否值得，以及是否该回退到泛化但较慢的实现。

一旦系统引入 JIT、autotune 或按 shape 编译，dispatch 的职责就变得更复杂了。它不仅在选择计算路径，也在决定**是否值得为了这次请求生成一种新实现**。对在线系统来说，这个决定非常敏感，因为一次额外编译可能直接打在用户请求延迟上。

因此，很多系统会把“最快路径”改写成“当前条件下综合收益最高的路径”。若某个 shape 极少出现，哪怕理论上有更快 kernel，也可能不值得即时编译；若某个 shape 已经成为高频热点，即使首次编译有成本，也值得尽快把它纳入缓存和热路径管理。

### 5.1 冷启动、缓存命中与在线抖动

编译缓存最怕的通常不是绝对慢，而是**不可预测**。同样一类请求，在缓存命中时表现很好，缓存 miss 时却突然掉到另一档延迟，这种抖动对线上系统比“稳定但略慢”更难接受。

所以更成熟的做法通常是预热高频 shape、限制低频 shape 的在线特化、记录编译与命中事件，并把缓存策略写进 dispatch，而不是把它当成黑盒附属品。

## 直觉例子

运行时 dispatch 就像智能交通系统的分流策略。道路网络再好，如果所有车都被导向同一条路，照样会堵；反过来，即使你有几条性能很强的专用通道，如果调度系统不会根据车流类型和道路情况分流，也发挥不出优势。Kernel 系统也是一样：真正的高性能，不只是有快路，还要有会分流的交通指挥。

## 本页结论

运行时 dispatch 与 kernel 选择，是连接“多套实现”和“真实工作负载”的最后一层系统能力。没有它，再好的 shape-specialized kernel 也只是零散武器；有了它，系统才能根据真实请求分布把不同实现组织成一套稳定、高效、可演进的执行体系。

从更高一层看，dispatch 解决的其实不是“怎么挑一个 kernel”这么小的问题，而是**怎么让底层实现真正服从上层负载分布**。训练和推理系统最终都会遇到这一层约束：请求不是静止的，硬件不是统一的，shape 不是固定的，缓存也不是无限的。谁能把这几件事一起组织好，谁的算子系统才真正算成熟。

## 快速代码示例

```python
def select_gemm_kernel(m, n, k, dtype):  # 根据 shape 和 dtype 选择 GEMM 实现
    if dtype == "fp8" and n % 128 == 0 and k % 128 == 0:  # 优先匹配 FP8 Tensor Core 对齐路径
        return "gemm_fp8_tc_128"  # 返回 FP8 特化 kernel
    if m <= 128 and n <= 128:  # 小矩阵走更适合的 tile 策略
        return "gemm_small_tile"  # 返回小矩阵 kernel
    return "gemm_default"  # 其它情况回退到通用实现
```

这段代码是一个最小 dispatch 规则示例：先匹配高收益特化路径，再匹配小矩阵场景，最后回退到通用实现。核心思想是让“规则可解释、可观测、可回退”，避免上线后出现黑盒命中问题。

## 工程收束

Runtime dispatch 要让 shape、dtype、layout、cache 状态和运行时队列都可观测。dispatch 规则过多无法解释、热路径命中不到最佳 kernel、回退路径无人监控，都会让性能问题难以复盘；上线前应统计命中率，保留异常 shape 黑名单，并在灰度时观察路径分布。

## 真实排查案例：热路径没有命中最快 kernel

**输入症状**：上线一个 FP8 GEMM 特化后，离线 benchmark 预期提速 20%，线上吞吐只提升 3%，而且不同租户差异很大。

**关键指标**：高频 shape 的理论覆盖率有 65%，但实际特化 kernel 命中率只有 22%；fallback 次数集中在 LoRA 打开、padding 后的 `N/K` 不对齐和少数 batch bucket。

**Nsight / trace 观察**：trace 里多数请求走了通用 GEMM；dispatch 日志显示 dtype 是 FP8，但 layout tag 和 stride 不满足特化路径条件；LoRA 合并前后的张量布局不一致，导致规则被跳过。

**判断**：kernel 没错，dispatch 条件和真实张量状态不一致。离线 benchmark 用的是理想 contiguous shape，线上经过 LoRA、padding、packing 后已经变成另一类 workload。

**修复**：把 dispatch 日志提升为一等指标，记录命中路径、跳过原因和 fallback 代价；为 LoRA 热路径补 layout normalization，或增加一个支持该 stride 的特化版本；灰度时按租户看命中率。

**反例**：如果命中率很高但收益仍低，问题才更可能在 kernel 本身、内存带宽或上游排队。dispatch 排查的第一问永远是“我以为跑的路径，线上真的跑了吗”。

!!! tip "下一站"
    - 回到本专题入口：[算子与编译器](index.md)，确认这页在整条路线中的位置。
    - 按导航顺序继续：[Roofline 建模与性能案例](roofline-benchmarking-and-case-studies.md)。
    - 概念或符号卡住时，先查 [术语表](../glossary/index.md)，再回到当前页。
