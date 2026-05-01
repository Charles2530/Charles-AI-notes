# GPU Kernel、Batching 与内存系统

推理优化落到 GPU 上时，很多抽象问题都会变成具体瓶颈：prefill 是否吃满 Tensor Core，decode 是否被 KV 访存拖住，batching 是否让 shape 变得更碎，量化收益是否被 dequant 和布局转换吞掉。理解 GPU kernel、batching 和内存系统，不是为了手写所有 CUDA，而是为了判断性能问题究竟来自模型、runtime、kernel 还是缓存生命周期。

这页和 [推理服务系统](serving-systems.md)、[上下文压缩与 KV 层级](context-compression-kv-eviction-and-memory-hierarchies.md)、[服务态 Attention 与 KV Kernel](../operators/serving-attention-and-kv-kernels.md) 互补。服务系统页讲队列和资源治理，本页讲 GPU 热路径和内存行为。

!!! note "初学者先抓住"

    推理性能不是一个 token/s 数字能讲清的。prefill 更像一次性处理大块输入，decode 更像不断读历史 KV 的小步循环；batching 决定 GPU 是否吃满，内存系统决定尾延迟是否稳定。

!!! note "难点解释：为什么训练快不代表推理也快"

    训练通常 batch 大、形状稳定，GEMM 更容易跑满 Tensor Core；在线 decode 请求随时进出、序列长度不同、KV 访问碎片化，更容易被内存带宽和 kernel launch 开销限制。所以推理优化要拆 TTFT、TPOT、P95/P99，而不是照搬训练吞吐直觉。

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

Decode 更像动态小步迭代：每轮只生成新 token，活动请求集合不断变化。它的主要问题是“碎”：

1. 小 `M` 的 skinny GEMM；
2. 频繁 kernel launch；
3. 请求随时加入和退出；
4. KV page 访问不连续；
5. LoRA、采样参数、输出长度让 batch 更难合并。

因此很多系统会把 prefill 和 decode 拆成不同队列、不同资源池或不同调度策略。长文档问答更可能 prefill 主导，聊天和 agent 更可能 decode 主导。优化前先拆阶段，否则很容易把 prefill 问题错当成 decode kernel 问题。

## 三、Batching：吞吐、尾延迟与 Shape Shaping

Batching 不只是“把请求拼在一起”。它决定 GPU 是否吃满，也决定用户等待多久。

常见方式包括：

1. **静态批处理**：适合离线推理，交互延迟差；
2. **动态批处理**：用短窗口聚合请求，平衡吞吐和等待；
3. **continuous batching**：每个 decode step 都允许请求加入和离开；
4. **bucketing**：按长度、模型、LoRA、采样参数或优先级分桶；
5. **batch shaping**：让 runtime 主动选择对 kernel 更友好的 shape family。

若当前活动请求为 $\mathcal{B}_t$，continuous batching 每步都在更新：

$$
\mathcal{B}_{t+1}
=
(\mathcal{B}_t \setminus \mathcal{F}_t) \cup \mathcal{A}_t.
$$

这里 $\mathcal{F}_t$ 是完成请求，$\mathcal{A}_t$ 是新加入请求。它能提高 GPU 利用率，但也会让内存生命周期和 shape 更动态。

### 何时吞吐会伤害体验

为了吞吐拉长 batching 窗口、把长短请求混在一起、或让 prefill 抢占 decode，可能让 token/s 更漂亮，但 TTFT 和 P99 更差。推理系统必须同时看：

1. 平均吞吐；
2. TTFT 和 TPOT；
3. P95/P99；
4. 长短请求公平性；
5. 高价值租户是否被低价值长请求拖慢。

## 四、KV Cache 是推理内存系统的核心

自回归生成时，KV cache 保存历史 token 的 key/value，避免每步重算全部上下文。若层数为 $L$，hidden size 为 $d$，序列长度为 $T$，batch 为 $B$，每元素字节数为 $b$，KV 大小近似为：

$$
M_{\text{KV}}
\approx 2 \cdot L \cdot B \cdot T \cdot d \cdot b.
$$

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

推理里的 attention kernel 至少要区分：

1. prefill attention；
2. decode attention；
3. paged attention；
4. MQA/GQA attention；
5. sliding window / chunked attention；
6. speculative decoding 下的验证 attention。

它们都叫 attention，但访问模式和热点不同。FlashAttention 类方法通过分块和在线 softmax 归约避免 materialize 巨大 score 矩阵；paged attention 则更关注 KV page 布局和动态请求。

### Kernel Fusion

decode 每步工作量小，小算子和 launch 开销更容易显性化。Fusion 可以减少全局内存往返，例如 RMSNorm + scale、dequant + matmul、rotary + QK 后处理、residual + norm、logits 后处理。代价是实现复杂、调试困难、硬件兼容性和 shape 覆盖更难。

### Quantization Kernel

量化推理不是把权重变小就一定更快。若 kernel 内要频繁读取 scale/zero-point、做 dequant、处理不友好的 group layout，收益会被内存 stall 和额外指令吃掉。评估量化时应看：

1. 是否有 fused quantized GEMM；
2. group size 和 layout 是否匹配 kernel；
3. 小 batch decode 是否仍然快；
4. scale 访问是否成为新带宽瓶颈；
5. 多 LoRA、多模型是否破坏量化布局收益。

## 六、多模型、LoRA、MoE 与多 GPU 的额外复杂度

真实推理节点往往不是单模型、单租户、单形状服务。多 LoRA、adapter、MoE、量化版本和多模型路由会把 kernel 选择问题放大。

典型影响包括：

1. adapter 切换破坏 batch 合并；
2. 权重局部性下降；
3. 不同请求 shape family 更杂；
4. expert routing 造成负载不均；
5. TP/EP 跨卡同步放大 decode 延迟；
6. 多租户隔离限制 prefix cache 复用。

MoE 推理尤其要看 expert 热点和 all-to-all。单个 expert 被热门请求集中命中时，即使整体 GPU 利用率看起来不低，某些 rank 仍会成为系统瓶颈。

多 GPU 推理也不只是通信问题。KV cache 如何分布、page metadata 如何同步、prefix cache 是否跨 rank 可用、fallback 是否改变拓扑，都会影响尾延迟。

## 七、诊断与优化顺序

推理系统变慢时，不建议一开始就换 GPU 或重写 kernel。更稳的排查顺序是：

1. 按 queue / prefill / decode / post 拆时延；
2. 判断主瓶颈是 TTFT、TPOT、吞吐还是 P99；
3. 若 prefill 主导，先看上下文长度、attention、prefix cache 和 chunking；
4. 若 decode 主导，先看 continuous batching、KV cache、kernel launch、fusion 和 page locality；
5. 若显存主导，先看 KV 占用、碎片、page size、量化和 memory tiering；
6. 若多租户主导，先看分池、优先级、LoRA/adapter 分桶和路由策略；
7. 最后再决定是否需要写新 kernel、换 runtime 或改模型结构。

### 最小观测集

至少应持续记录：

1. TTFT、TPOT、P50/P95/P99；
2. prefill/decode 分阶段 GPU 时间；
3. 实际 batch size 和 shape 分布；
4. HBM 带宽、SM util、kernel 时间分解；
5. KV cache 占用、page 命中、碎片率；
6. prefix cache 命中率；
7. queue depth、admission reject、fallback；
8. LoRA/model/expert route 分桶后的延迟和成本。

核心判断是：推理优化不是微基准游戏。只有当 kernel 性能、runtime 调度、KV 生命周期和真实流量分布一起变好，线上体验才会真正改善。
