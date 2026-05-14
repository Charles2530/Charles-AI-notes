# GPU Kernel、Batching 与显存

推理优化落到 GPU 上时，很多抽象问题都会变成具体瓶颈：prefill 是否吃满 Tensor Core，decode 是否被 KV 访存拖住，batching 是否让 shape 变得更碎，量化收益是否被 dequant 和布局转换吞掉。理解 GPU kernel、batching 和内存系统，不是为了手写所有 CUDA，而是为了判断性能问题究竟来自模型、runtime、kernel 还是缓存生命周期。

这页和 [推理服务系统](serving-systems.md)、[上下文压缩与 KV 层级](context-compression-kv-eviction-and-memory-hierarchies.md)、[推理 Attention 与 KV Kernel](../operators/serving-attention-and-kv-kernels.md) 互补。推理服务系统页讲队列和资源治理，本页讲 GPU 热路径和内存行为。

!!! note "初学者先抓住"

    推理性能不是一个 token/s 数字能讲清的。prefill 更像一次性处理大块输入，decode 更像不断读历史 KV 的小步循环；batching 决定 GPU 是否吃满，内存系统决定尾延迟是否稳定。

!!! note "难点解释：为什么训练快不代表推理也快"

    训练通常 batch 大、形状稳定，GEMM 更容易跑满 Tensor Core；在线 decode 请求随时进出、序列长度不同、KV 访问碎片化，更容易被内存带宽和 kernel launch 开销限制。所以推理优化要拆 TTFT、TPOT、P95/P99，而不是照搬训练吞吐直觉。

!!! example "有趣例子：自助餐和点菜"

    Prefill 像自助餐后厨一次做一大盘，容易批量处理；decode 像顾客一口一口点菜，每次只做一点，还要记住每桌之前点过什么。GPU 在这两种场景下的瓶颈完全不同。

## 一、推理延迟拆解：先区分 Queue、Prefill、Decode

一次请求的总时延可以粗略分成：

$$
T_{\text{total}}
=
T_{\text{queue}}
+ T_{\text{prefill}}
+ T_{\text{decode}}
+ T_{\text{post}}
+ T_{\text{network}}.
$$

其中 `TTFT` 主要受排队、prefill 和首步 decode 影响；`TPOT` 更接近后续 decode 稳态。很多优化误判都来自只看平均 token/s，而不拆 queue、prefill 和 decode。

| 阶段 | 典型瓶颈 | 关键指标 |
| --- | --- | --- |
| Queue | 调度、批处理窗口、优先级 | queue time、admission reject、P95/P99 |
| Prefill | 长上下文 attention、大 GEMM、KV 写入 | TTFT、prefill tokens/s、HBM 带宽 |
| Decode | 小 batch、KV 读取、kernel launch、碎 shape | TPOT、decode batch size、KV page miss |
| Post | detokenize、工具编排、网络 | post latency、RPC error、tool latency |

推理比训练更容易暴露带宽问题。训练时 batch 大、shape 稳定，GEMM 更容易吃满硬件；decode 阶段 batch 动态变化、每步工作量小、KV cache 访问频繁，更容易变成 memory-bound 和 launch-bound。

## 二、Prefill 与 Decode 是两种工作负载

Prefill 更像大块矩阵并行：输入上下文可以并行处理，长 prompt 会产生较大的 attention 和 GEMM 工作量。它适合大 batch、FlashAttention、prefix cache、chunked prefill 和高吞吐资源池。

Decode 更像动态小步迭代：每轮只生成新 token，活动请求集合不断变化。它的主要问题是“碎”：小 `M` 的 skinny GEMM、频繁 kernel launch、请求随时加入和退出、KV page 访问不连续，以及 LoRA、采样参数、输出长度让 batch 更难合并。

因此很多系统会把 prefill 和 decode 拆成不同队列、不同资源池或不同调度策略。长文档问答更可能 prefill 主导，聊天和 agent 更可能 decode 主导。优化前先拆阶段，否则很容易把 prefill 问题错当成 decode kernel 问题。

## 三、Batching：吞吐、尾延迟与 Shape Shaping

Batching 不只是“把请求拼在一起”。它决定 GPU 是否吃满，也决定用户等待多久。

常见方式包括静态批处理、动态批处理、continuous batching、bucketing 和 batch shaping。静态批处理适合离线推理但交互延迟差；动态批处理用短窗口聚合请求；continuous batching 允许每个 decode step 加入和离开请求；bucketing 按长度、模型、LoRA、采样参数或优先级分桶；batch shaping 则让 runtime 主动选择对 kernel 更友好的 shape family。

若当前活动请求为 $\mathcal{B}_t$，continuous batching 每步都在更新：

$$
\mathcal{B}_{t+1}
=
(\mathcal{B}_t \setminus \mathcal{F}_t) \cup \mathcal{A}_t.
$$

这里 $\mathcal{F}_t$ 是完成请求，$\mathcal{A}_t$ 是新加入请求。它能提高 GPU 利用率，但也会让内存生命周期和 shape 更动态。

### 何时吞吐会伤害体验

为了吞吐拉长 batching 窗口、把长短请求混在一起、或让 prefill 抢占 decode，可能让 token/s 更漂亮，但 TTFT 和 P99 更差。推理系统必须同时看平均吞吐、TTFT、TPOT、P95/P99、长短请求公平性，以及高价值租户是否被低价值长请求拖慢。

## 四、KV Cache 是推理内存系统的核心

自回归生成时，KV cache 保存历史 token 的 key/value，避免每步重算全部上下文。若层数为 $L$，hidden size 为 $d$，序列长度为 $T$，batch 为 $B$，每元素字节数为 $b$，KV 大小近似为：

$$
M_{\text{KV}}
\approx 2 \cdot L \cdot B \cdot T \cdot d \cdot b.
$$

这里的 \(d\) 要按 KV 实际宽度理解，而不是盲目等同于模型 hidden size。标准 MHA 通常每个 query head 都有自己的 K/V，\(d\) 接近 hidden size；GQA/MQA 会减少 KV head 数，所以更常写成 \(H_{\text{kv}}\times D_{\text{head}}\)。

例如 `32` 层、batch `4`、上下文 `32k`、BF16 KV：

| KV 结构 | 代入方式 | KV cache 数量级 |
| --- | --- | ---: |
| MHA，`32` 个 KV heads，`head_dim=128` | \(2\times32\times4\times32768\times32\times128\times2\) | 约 `64GiB` |
| GQA，`8` 个 KV heads，`head_dim=128` | \(2\times32\times4\times32768\times8\times128\times2\) | 约 `16GiB` |
| GQA + INT8 KV | 上式 bytes 从 `2` 改成 `1` | 约 `8GiB` |

这张表只是在算常驻 KV 本体。线上还要给 paged KV 的 block metadata、空闲页碎片、speculative 分支、临时 workspace 和不同请求长度的浪费留余量。

长上下文和大 batch 下，KV cache 往往比权重更先成为瓶颈。它的难点不只是“大”，还包括生命周期不同步、请求长度差异大、prefix 复用、page 回收、多租户隔离、speculative decoding 的临时分支和显存碎片。

### Paged KV 与 Prefix Cache

Paged KV 把 cache 拆成页，像虚拟内存一样管理，适合 continuous batching 下动态加入和退出请求。收益是降低连续大块显存需求、便于复用和回收；代价是地址间接、访存离散和 page size 选择问题。

Prefix cache 复用公共前缀，例如系统 prompt、热门文档块或共享工具上下文。它可以显著降低 prefill 成本，但要处理权限隔离、失效策略、租户边界和 cache locality。prefix 复用策略如果和 decode 调度脱节，也会破坏局部性。

### 内存碎片为什么线上才暴露

碎片常常不是单个请求造成的，而是长短请求、LoRA、多模型、prefix cache 和 page 回收交错运行后积累的。监控应包括 free page 分布、page reuse rate、KV 命中率、碎片率、OOM 前显存布局和分桶后的 KV 占用。

## 五、Kernel 热点：Attention、Fusion、Quantization

GPU 性能可用 roofline 直觉判断：

$$
\text{Performance}
\le
\min(\text{Peak FLOPs}, \text{Bandwidth} \times \text{Arithmetic Intensity}).
$$

如果一个算子搬很多数据但计算很少，即使 GPU 峰值 FLOPs 很高，也会被 HBM 带宽限制。推理中的 attention、RMSNorm、embedding gather、KV 读取、dequant 和 logits 后处理都可能落入这一类。

### Attention Kernel

推理里的 attention kernel 至少要区分 prefill attention、decode attention、paged attention、MQA/GQA attention、sliding window / chunked attention，以及 speculative decoding 下的验证 attention。

它们都叫 attention，但访问模式和热点不同。FlashAttention 类方法通过分块和在线 softmax 归约避免 materialize 巨大 score 矩阵；paged attention 则更关注 KV page 布局和动态请求。

### Kernel Fusion

decode 每步工作量小，小算子和 launch 开销更容易显性化。Fusion 可以减少全局内存往返，例如 RMSNorm + scale、dequant + matmul、rotary + QK 后处理、residual + norm、logits 后处理。代价是实现复杂、调试困难、硬件兼容性和 shape 覆盖更难。

### Quantization Kernel

量化推理不是把权重变小就一定更快。若 kernel 内要频繁读取 scale / zero-point、做 dequant、处理不友好的 group layout，收益会被内存 stall 和额外指令吃掉。评估量化时应看是否有 fused quantized GEMM，group size 和 layout 是否匹配 kernel，小 batch decode 是否仍然快，scale 访问是否成为新带宽瓶颈，以及多 LoRA、多模型是否破坏量化布局收益。

## 六、多模型、LoRA、MoE 与多 GPU 的额外复杂度

真实推理节点往往不是单模型、单租户、单形状服务。多 LoRA、adapter、MoE、量化版本和多模型路由会把 kernel 选择问题放大。

典型影响包括 adapter 切换破坏 batch 合并、权重局部性下降、不同请求 shape family 更杂、expert routing 造成负载不均、TP/EP 跨卡同步放大 decode 延迟，以及多租户隔离限制 prefix cache 复用。

MoE 推理尤其要看 expert 热点和 all-to-all。单个 expert 被热门请求集中命中时，即使整体 GPU 利用率看起来不低，某些 rank 仍会成为系统瓶颈。

多 GPU 推理也不只是通信问题。KV cache 如何分布、page metadata 如何同步、prefix cache 是否跨 rank 可用、fallback 是否改变拓扑，都会影响尾延迟。

## 七、诊断与优化顺序

推理系统变慢时，不建议一开始就换 GPU 或重写 kernel。更稳的排查顺序是先按 queue / prefill / decode / post 拆时延，判断主瓶颈是 TTFT、TPOT、吞吐还是 P99；若 prefill 主导，先看上下文长度、attention、prefix cache 和 chunking；若 decode 主导，先看 continuous batching、KV cache、kernel launch、fusion 和 page locality；若显存主导，先看 KV 占用、碎片、page size、量化和 memory tiering；若多租户主导，先看分池、优先级、LoRA / adapter 分桶和路由策略；最后再决定是否需要写新 kernel、换 runtime 或改模型结构。

### 最小观测集

至少应持续记录 TTFT、TPOT、P50/P95/P99，prefill / decode 分阶段 GPU 时间，实际 batch size 和 shape 分布，HBM 带宽、SM util、kernel 时间分解，KV cache 占用、page 命中、碎片率，prefix cache 命中率，queue depth、admission reject、fallback，以及 LoRA / model / expert route 分桶后的延迟和成本。

核心判断是：推理优化不是微基准游戏。只有当 kernel 性能、runtime 调度、KV 生命周期和真实流量分布一起变好，线上体验才会真正改善。

## 真实排查案例：Batch 变大后吞吐上升，交互体验下降

**输入症状**：连续批处理参数调大后，tokens/s 提升 22%，但聊天产品 TTFT P95 从 1.1s 涨到 2.0s，短请求投诉增多。

**关键指标**：实际 batch size 变大，GPU util 更高；queue time 和 prefill waiting time 上升；短请求被长文档请求一起合批；decode kernel 时间下降，但排队时间吞掉收益。

**Nsight / trace 观察**：GPU 时间线更密集，kernel 间空隙减少；系统 trace 显示请求在 scheduler 队列里等待凑 batch，长 prefill 段把短请求首 token 推迟。

**判断**：优化改善了 GPU 层吞吐，却牺牲了产品层 SLO。问题不是 kernel 慢，而是 batching 策略没有区分短交互和长文档。

**修复**：按输入长度和 SLA 分队列；短请求设置更小 max wait，长文档进入吞吐池；指标同时看 tokens/s、TTFT、TPOT 和任务级完成时间；灰度时按请求桶比较收益。

**反例**：离线批量生成或后台文档处理通常可以牺牲 TTFT 换吞吐。交互产品不能只用全局 tokens/s 判定优化成功。
