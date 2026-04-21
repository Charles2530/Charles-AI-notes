# GPU Kernel、Batching 与内存系统

推理优化真正落到 GPU 上时，很多抽象名词都会变得具体而残酷。一个看起来“模型参数不算大”的服务，可能因为 KV cache 撑爆显存；一个理论上吞吐很高的方案，可能被 kernel launch、内存带宽或跨卡同步拖垮。理解 GPU kernel、batching 和内存系统，不是为了手写每一个 CUDA kernel，而是为了知道性能瓶颈究竟在哪里，以及该用什么代价去换什么收益。

现代推理系统之所以越来越像系统工程，而不只是“把模型 load 起来”，正是因为它要同时面对：

1. 请求到达分布；
2. prefill 与 decode 两种完全不同的负载；
3. KV cache 的空间和生命周期管理；
4. GPU kernel 粒度与 shape 不规则性；
5. 多模型、多租户、量化、LoRA、MoE 和长上下文等交错因素。

## 1. 推理延迟从哪里来

一次请求的总时延可以粗略分解为

$$
T_{\text{total}}
=
T_{\text{queue}}
+ T_{\text{prefill}}
+ T_{\text{decode}}
+ T_{\text{post}}
+ T_{\text{network}}.
$$

其中：

1. \(T_{\text{queue}}\) 是排队等待批处理的时间；
2. \(T_{\text{prefill}}\) 是把输入上下文编码进 KV cache 的阶段；
3. \(T_{\text{decode}}\) 是逐 token 生成阶段；
4. \(T_{\text{post}}\) 包含 detokenize、格式化、工具编排等；
5. \(T_{\text{network}}\) 是网络与 RPC 成本。

**大模型服务常见的误判是**：只盯着平均 token/s，却忽略排队和 prefill。在长上下文场景里，prefill 甚至比 decode 更贵。

### 1.1 系统真正关心的通常不是单一平均值

**实际服务里至少会同时关心**：

1. `TTFT`，首 token 延迟；
2. `TPOT`，每输出 token 的时间；
3. P50 延迟；
4. P95 / P99 延迟；
5. 吞吐；
6. 每请求成本；
7. 峰值负载下的退化方式。

因为一个系统即使平均很快，也可能因为 tail latency 太差而不可用。

## 2. GPU 上的两个核心约束：算力与带宽

很多算子表面上是矩阵乘，实际上瓶颈可能不是 FLOPs，而是内存带宽。可以用 roofline 思维理解：

$$
\text{Performance}
\le
\min(\text{Peak FLOPs}, \text{Bandwidth} \times \text{Arithmetic Intensity}).
$$

如果算子每搬运 1 字节数据只做很少计算，那么即便 GPU 理论算力很高，也会被带宽限制。注意力、layernorm、embedding gather、KV cache 访问等都经常落入这一类。

### 2.1 为什么推理比训练更容易暴露带宽问题

**训练时**：

1. batch 往往更大；
2. GEMM 更容易吃满硬件；
3. 许多计算可被摊薄；
4. 长时间 steady state 更稳定。

**推理时尤其是 decode 阶段**：

1. batch 小且动态变化；
2. 请求形状不规则；
3. kernel 很碎；
4. cache 读写频繁；
5. 许多操作更偏 memory-bound。

因此推理系统里，“数学上复杂度不高”的东西也可能成为主瓶颈。

## 3. Prefill 与 Decode 的系统差异

### 3.1 Prefill 更像大矩阵并行

在 prefill 阶段，输入序列长度为 \(L\)，可以并行处理所有 token，算力利用率通常较高。尤其是大 batch、长 prompt 时，prefill 更接近传统训练中的矩阵计算。

**这意味着 prefill 更像**：

1. 大 GEMM；
2. 长序列 attention；
3. 对吞吐敏感；
4. 对图融合、Tensor Core 和 FlashAttention 等优化更友好。

### 3.2 Decode 更像小批量、频繁迭代

decode 阶段每步只生成少量 token，批次中不同请求长度又不一致，所以更容易出现：

1. GPU 利用率低；
2. kernel launch 开销占比变大；
3. 内存访问离散；
4. KV cache 读写成为主瓶颈；
5. tail request 拉长整个批次。

因此，prefill 优化和 decode 优化是两套逻辑，不能一概而论。

### 3.3 为什么很多系统 prefill 很强，但 decode 很痛

因为 prefill 仍然相对像“大型张量计算”，能较好复用训练时代累积的优化；  
而 decode 更像一个带有动态调度、页面管理和细碎 kernel 的系统问题。

## 4. Batching 的几种方式

### 4.1 静态批处理

固定 batch size 收集请求后统一执行，简单但对延迟不友好。适合离线推理，不适合交互式系统。

### 4.2 动态批处理

在一个短时间窗口内收集请求并拼 batch，试图平衡吞吐与响应。若窗口过长，用户延迟上升；过短，则 GPU 吃不满。

### 4.3 Continuous batching

现代 LLM serving 常用 continuous batching，也称 iteration-level scheduling。其核心是：每个 decode step 都可以让新请求加入，已完成的请求离开。这样能更细粒度地填满 GPU。

若当前批中的活动请求集合为 \(\mathcal{B}_t\)，则每步调度都在更新

$$
\mathcal{B}_{t+1}
=
(\mathcal{B}_t \setminus \mathcal{F}_t) \cup \mathcal{A}_t,
$$

其中 \(\mathcal{F}_t\) 是本步完成的请求，\(\mathcal{A}_t\) 是新加入的请求。

### 4.4 Bucketing

按序列长度、模型、LoRA adapter、采样参数等维度分桶，让相似请求一起跑。这样能减少 padding、减少分支不一致和 cache 抖动。

### 4.5 Scheduler 不只是“怎么拼 batch”

**它还决定**：

1. 新请求什么时候能插队；
2. 长请求会不会饿死短请求；
3. prefix cache 是否能被优先复用；
4. 不同模型或不同 adapter 如何共享 GPU；
5. 某类高优先级请求是否需要保底资源。

因此调度系统本身就是一层重要优化对象。

## 5. Prefill/Decode 资源画像为什么不同

**一个很常用的判断框架是**：

1. prefill 更接近 compute-heavy；
2. decode 更接近 memory-heavy；
3. prefill 更适合大块批处理；
4. decode 更依赖 continuous batching 和 cache 管理。

### 5.1 这对系统设计意味着什么

意味着很多成熟推理系统会显式区分：

1. prefill 队列和 decode 队列；
2. prefill 专用 GPU 或资源池；
3. decode 优先的 kernel 选择；
4. 在长上下文场景中优先优化 prefill；
5. 在 agent 或聊天场景中优先优化 decode。

## 6. 注意力 kernel 为什么是焦点

Transformer 推理中最昂贵的部分之一是注意力。朴素实现需要频繁读写 Q、K、V 和中间 score 矩阵，显存与带宽压力都很大。FlashAttention 一类优化的关键，是重新安排计算顺序，避免显式 materialize 巨大中间矩阵，并尽量在 SRAM / shared memory 内完成局部归一化。

它利用 softmax 的在线归约性质：

$$
\mathrm{softmax}(x)_i
=
\frac{e^{x_i - m}}{\sum_j e^{x_j - m}},
\quad
m = \max_j x_j.
$$

通过分块处理并维护局部最大值和归一化项，可以在不存完整 score 矩阵的情况下完成同样计算。

### 6.1 推理中的 attention kernel 不止一种

**至少可以区分**：

1. prefill attention；
2. decode attention；
3. paged attention；
4. MQA/GQA 场景 attention；
5. sliding window 或 chunked attention；
6. speculative decoding 场景下的 attention 访问模式。

它们虽然都叫 attention，但热点路径和最优实现往往并不相同。

## 7. KV Cache 是什么，为什么既救命又要命

自回归生成时，如果每一步都重新计算所有历史 token 的 K、V，成本会很高。KV cache 的做法是缓存历史 token 的 key/value，只对新 token 追加。

若层数为 \(L\)，hidden size 为 \(d\)，序列长度为 \(T\)，batch 为 \(B\)，每元素字节数为 \(b\)，KV cache 大小近似为

$$
M_{\text{KV}}
\approx 2 \cdot L \cdot B \cdot T \cdot d \cdot b.
$$

这里的 2 是 K 和 V 各一份。长上下文和大 batch 下，这个量会迅速爆炸。

举例说，一个企业助手同时服务很多 `32k` 上下文请求时，真正先撑爆显存的往往不是参数，而是 KV cache。

### 7.1 KV cache 的真实挑战不只是“太大”

**还包括**：

1. 生命周期不同步；
2. 请求长度差异大；
3. 页面回收与复用；
4. 多租户隔离；
5. prefix 共享；
6. speculative decoding 引入的临时 token 分支。

因此它更像一个内存系统问题，而不是单纯的张量缓存问题。

## 8. KV Cache 管理策略

### 8.1 Paged KV Cache

把 KV cache 分成页，像虚拟内存那样管理，减少碎片化，也便于请求动态加入退出。这是很多现代 serving 框架的重要设计。

### 8.2 Prefix caching

若多个请求共享相同前缀，如系统提示词、公共文档上下文、热门模板，可以复用前缀的 prefill 结果，大幅节省时间和显存。

### 8.3 Sliding window / chunked attention

对某些模型或任务，可限制注意力只看最近窗口或分块上下文，以降低 KV cache 和注意力计算成本。但这会影响长程依赖能力。

### 8.4 KV eviction 与 memory tiering

当显存成为瓶颈时，还会出现：

1. 把部分 KV 挪到更慢内存层；
2. 做分层缓存；
3. 对低价值上下文做压缩或淘汰；
4. 针对不同优先级请求给不同 KV 预算。

这些策略都会在精度、延迟和成本之间产生新的 tradeoff。

## 9. 内存碎片与分配器

即使理论显存够用，服务仍可能因为碎片化 OOM。原因包括：

1. 请求长度变化大；
2. 动态 batch 频繁扩缩；
3. LoRA、多模型共存导致不同张量大小交错分配；
4. cache 回收不及时；
5. page size 与请求分布不匹配。

因此生产系统往往会使用页式分配、预留池化、对象复用和定期紧缩策略，而不是完全依赖通用 allocator。

### 9.1 为什么碎片问题常在负载上来后才出现

因为它通常不是单个请求造成，而是很多长短不一、生命周期不同的请求交错运行后逐渐积累的。  
这也是为什么线上系统需要持续监控：

1. free page 分布；
2. page reuse rate；
3. 平均与最坏碎片率；
4. OOM 前的显存布局变化。

## 10. Kernel fusion 为什么重要

很多推理阶段的操作如果拆成多个 kernel，会产生大量读写和 launch 开销。Kernel fusion 把多个相邻操作合并，如：

1. bias + activation；
2. RMSNorm + scale；
3. dequant + matmul；
4. rotary embedding + QK projection 后处理；
5. residual + norm；
6. logits 后处理。

融合的收益在于减少全局内存往返；风险在于实现复杂、可维护性差、跨硬件兼容难。

### 10.1 为什么 decode 场景尤其依赖 fusion

因为 decode 每步工作量本来就小，launch 开销和中间读写的相对占比更高。  
一个训练里看起来“不值得专门优化”的小算子，在 decode 场景中可能成为明显热点。

## 11. 量化推理与 kernel 协同

低比特量化不是简单把权重文件变小，而是要有匹配的 kernel 才能真正提速。否则你可能在 kernel 内先把 int4 解包回高精度，再执行普通 GEMM，吞吐未必理想。

**一个简化吞吐公式是**：

$$
T_{\text{effective}}
\approx
\frac{\text{useful compute}}
\text{compute time} + \text{dequant time} + \text{memory stall}.
$$

如果 dequant 和内存 stall 太大，量化收益就被吞掉了。

### 11.1 量化推理真正要看的几个问题

1. 是否有 fused quantized GEMM kernel；
2. group size 和量化布局是否匹配 kernel；
3. scale / zero-point 访问是否额外成为带宽负担；
4. decode 场景下低比特 kernel 是否仍能稳定快过高精度；
5. 多 LoRA 或多模型场景是否破坏量化布局收益。

## 12. 不同并行方式下的推理瓶颈

### 12.1 单卡

关注显存、KV cache、batch size 和 kernel 效率。

### 12.2 张量并行推理

适合超大模型，但 decode 阶段每步都可能有跨卡同步，容易放大通信延迟。

### 12.3 Pipeline 并行推理

较少用于低延迟在线系统，因为微批次和流水线气泡在 decode 阶段不一定划算。

### 12.4 专家并行

MoE 推理的难点在于 expert routing 和不均匀负载。如果热门 expert 被过度命中，某些 GPU 会成为热点。

### 12.5 多模型 / 多租户

**这时瓶颈还包括**：

1. 模型切换；
2. KV cache 隔离；
3. 公平性；
4. 高优先级任务抢占；
5. prefix cache 能否跨会话安全复用。

## 13. 调度视角下的吞吐与时延权衡

假设单位时间内到达请求数为 \(\lambda\)，系统服务率为 \(\mu\)，则排队延迟会随着负载接近饱和急剧上升。即使模型内核很快，只要 \(\rho = \lambda / \mu\) 接近 1，用户感知延迟仍会恶化。

因此调度策略不只要提升 token/s，还要关注：

1. P50 延迟；
2. P95 / P99 延迟；
3. 首 token 延迟（TTFT）；
4. 尾请求饿死问题；
5. 长请求与短请求的公平性；
6. 是否有优先级反转。

### 13.1 为什么“吞吐更高”有时反而让体验更差

如果系统为了吞吐把 batching 窗口拉太长、prefill 队列太深、decode 被迫等待更多请求聚合，那么：

1. GPU 可能更忙；
2. 平均 token/s 可能更好；
3. 但用户感知延迟会明显恶化。

所以服务优化必须同时看吞吐与交互体验。

## 14. 多 LoRA、Adapter 与专家路由带来的额外复杂度

现代服务系统常常不是单模型静态服务，而是：

1. 多个 LoRA adapter 热插拔；
2. 同一底座上承载多个租户；
3. MoE expert 动态路由；
4. 多模型编排。

这会带来新的 kernel 和调度问题：

1. adapter 切换是否破坏 batch 合并；
2. 不同 adapter 是否能共享 KV 前缀；
3. expert load balance 是否均衡；
4. 某些 adapter 或 expert 是否使 kernel shape 更碎。

## 15. 一个实际例子：企业长文档问答

假设系统要处理几百页合同问答。用户问题虽然短，但会拼接大量检索片段，导致 prefill 很长。此时优化重点可能不是 decode，而是：

1. 减少无效上下文；
2. 使用 prefix cache；
3. 对共享文档块做预编码；
4. 限制每次进入模型的 chunk 大小；
5. 对长文档做分段与重排。

如果只盯着每秒生成 token 数，会完全错过真正瓶颈。

## 16. 再看一个例子：实时 agent

实时屏幕 agent 的每个请求上下文较短，但需要快速多轮调用，且经常带工具结果。此时关注点可能转向：

1. 小 batch decode 利用率；
2. 高频 kernel launch 开销；
3. routing 与多模型切换；
4. 多租户下的 cache 干扰；
5. 工具调用前后上下文增量复用。

它和长文档问答是完全不同的推理画像。

## 17. 再看一个例子：超长上下文研究系统

当你需要处理 `128k` 甚至更长上下文时，痛点往往集中在：

1. prefill attention；
2. KV cache 空间；
3. page 管理；
4. 长 prompt 引发的 TTFT 爆炸；
5. 是否需要 context compression 或 chunking。

这类系统的 kernel 与内存策略，往往会和普通聊天机器人完全不同。

## 18. 诊断时该看哪些信号

**至少应监控**：

1. GPU util 与 SM util；
2. 显存占用与碎片率；
3. HBM 带宽占用；
4. prefill 与 decode 分开统计的延迟；
5. TTFT、TPOT；
6. batch 实际大小分布；
7. KV cache 命中与回收情况；
8. prefix cache 复用率；
9. scheduler queue 深度；
10. per-kernel 时间分解。

只有把这些量拆开，才能知道该优化 kernel、调度、缓存还是上下文构造。

### 18.1 系统 profile 最容易忽略的一点

很多团队会看 GPU 利用率，却不单独看：

1. prefill GPU 时间；
2. decode GPU 时间；
3. CPU 调度时间；
4. 内存分配器停顿；
5. 网络与 post-processing。

结果就会把系统瓶颈误判成“模型太大”。

## 19. 一个更实用的优化顺序

如果一个推理系统慢，通常更稳妥的排查顺序是：

1. **先按阶段拆**：queue / prefill / decode / post；
2. 再看 prefill 与 decode 哪个主导；
3. 若是 prefill，重点看 attention、长上下文和 prefix 复用；
4. 若是 decode，重点看 continuous batching、KV cache、kernel launch 和小算子 fusion；
5. 再根据 profile 决定是改 kernel、调调度、改缓存还是改上下文构造。

这比一开始就“上更快的 GPU”更有效。

## 20. 一个形象比喻

如果把大模型推理看作餐厅出餐，GPU kernel 是厨房里每道工序的效率，batching 是如何把点单拼到一起，内存系统则是冰箱、备菜台和传菜口的布局。不是厨师手速快，餐厅就一定出餐快；如果冰箱太远、备菜台太乱、同一时间进来太多不同做法的订单，厨房照样会堵。KV cache 就像提前备好的半成品，能让重复菜式很快出餐，但备得太多又会把冰箱塞满。

在这个比喻里，continuous batching 像根据后厨空位动态调整出单顺序，prefix cache 像公用预制底料，而 paged KV 更像把半成品按标准盒装管理，避免厨房台面被杂乱堆满。

## 21. 小结

推理优化不能只看模型参数量和理论 FLOPs。在线服务真正决定体验的，是 prefill 与 decode 的不同特性、KV cache 的显存占用、kernel 对带宽的利用方式，以及调度对尾延迟的控制。理解 GPU kernel、batching 与内存系统，本质上是在把“模型会不会跑”升级成“系统能不能稳定、高效地为真实请求服务”。

而一旦进入长上下文、多模型、多租户、量化和 agent 场景，这件事就会进一步从“高性能实现问题”升级成“资源分层、内存管理、调度公平与业务体验”的综合系统工程问题。

## 22. 运行时和 kernel 的边界要怎么划

很多性能问题既像 kernel 问题，又像 runtime 问题。一个实用判断是：

1. 若热点稳定、shape 稳定、瓶颈集中在单个算子，优先看 kernel；
2. 若热点随请求长度和队列结构变化，优先看 runtime 和调度；
3. 若问题只在混合负载出现，常常是 kernel 与 runtime 的交互问题。

### 22.1 为什么这个边界很重要

**因为很多团队会误把**：

1. batch 组织问题当 kernel 问题；
2. cache 碎片问题当模型问题；
3. decode 调度抖动当 GPU 算力不够。

## 23. 一个更实用的验收清单

对一条新的推理优化链路，至少应验证：

1. 固定长度下 kernel 是否确实提速；
2. 混合长度负载下 tail latency 是否仍受控；
3. prefix cache、KV cache 和调度是否相互兼容；
4. 量化后是否把收益吃在反量化和布局变换上；
5. 多租户或多 LoRA 情况下 shape 是否仍稳定。

## 24. 小结

GPU kernel、batching 与内存系统的真正价值，不在于把某个 benchmark 数字拉高，而在于让系统在真实流量、真实长度分布和真实缓存压力下仍然稳定高效。只有把 kernel 性能和 runtime 行为一起看，推理优化才不会停留在实验室里的微基准游戏。

## 25. Decode 阶段真正的瓶颈常在“碎”

很多 decode 优化迟迟见不到大收益，是因为它们只看单个 matmul，没有看到整个 decode 阶段的碎片化。

### 25.1 这些“碎”包括什么

1. **小 batch**；
2. **小 `M` 的 skinny GEMM**；
3. **频繁 kernel launch**；
4. **KV page / block 访问不连续**；
5. **请求随时加入和退出，导致 shape 持续变化**。

### 25.2 这意味着什么

在 decode 里，很多优化价值来自：

1. **减少 launch**；
2. **稳定 shape family**；
3. **改善 cache locality**；
4. **把一串小操作融合**；
5. **让 runtime 更懂 kernel 的友好工作形状**。

## 26. Paged KV 的收益和代价要一起看

paged KV 解决了连续大块 KV cache 很难管理的问题，但它不是零代价。

### 26.1 它带来的收益

1. **分配和释放更灵活**；
2. **减少大块连续显存需求**；
3. **便于多请求并存和长上下文管理**；
4. **更适合 continuous batching 下的动态生命周期**。

### 26.2 它带来的成本

1. **地址间接层级增加**；
2. **访存更容易离散**；
3. **page 大小选不好会造成内部碎片或 transaction 浪费**；
4. **与 attention kernel 的协同设计难度变高**。

### 26.3 Page Size 不是越大越好

1. **太小**：元数据和间接寻址开销高；
2. **太大**：内部碎片明显，尤其对短请求不友好；
3. **真实最优**：通常取决于 head 维度、KV layout、请求长度分布和 attention kernel 访问模式。

## 27. MQA/GQA 的系统收益必须落到 KV 访问上

从模型视角，MQA/GQA 节省的是 KV 头数；从系统视角，这应转化为：

1. **更低显存占用**；
2. **更少 KV 读取带宽**；
3. **更轻的 cache 管理压力**；
4. **更有利的 decode batching 上限**。

### 27.1 为什么系统里有时收益不明显

1. **attention kernel 仍按旧布局访问**；
2. **runtime 没有把节省下来的显存转化为更好的 batch 结构**；
3. **量化和 page metadata 把部分收益吃掉**；
4. **瓶颈其实已经转移到其他小算子或 CPU 调度**。

## 28. Batch Shaping 是 runtime 和 kernel 的共同任务

很多系统只说 continuous batching，但真正高质量的 batching 还需要**shape shaping**。

### 28.1 目标是什么

1. **尽量让同一轮 decode 中的请求形状相近**；
2. **减少极端长请求对小请求的拖累**；
3. **让 kernel 命中有限几类高效 shape**；
4. **让 cache 生命周期更可预测**。

### 28.2 常见手段

1. **按剩余长度或历史生成速率做分桶**；
2. **把超长请求迁到专门池**；
3. **限制单轮拼入的极端新请求**；
4. **对某些请求类型用不同量化版本或不同 kernel 路径**。

## 29. Cache Locality 在服务态比训练更脆弱

训练里 shape 稳定，局部性更容易维持；服务态则因为请求动态变化，局部性随时被破坏。

### 29.1 最常见的局部性破坏来源

1. **page 分配离散**；
2. **多租户混部导致同一批次内请求完全无关**；
3. **prefix 复用策略和 decode 调度脱节**；
4. **LoRA / adapter 切换导致权重局部性下降**。

### 29.2 为什么要关心它

因为很多 decode kernel 已经处于 memory-bound 边缘，局部性稍差就可能直接把 TPOT 拉高。

## 30. 量化推理里的内存系统并不轻松

量化看似省显存，但在实际服务里，它也会带来新的访存和布局问题。

### 30.1 常见额外成本

1. **scale / zero-point / codebook 的额外读取**；
2. **反量化路径引入更多寄存器与中间值**；
3. **小 `M` 场景下固定 dequant 成本被放大**；
4. **多种量化格式共存导致 runtime dispatch 变复杂**。

### 30.2 一个常见误判

只看显存下降，就以为系统一定更快。实际上若 dequant 路径和 kernel 设计不佳，量化后的 decode 仍可能被带宽和碎形状拖住。

## 31. Multi-LoRA 与多模型服务会把 kernel 选择问题放大

一旦一个服务节点同时承载多个 LoRA 或多个模型版本，runtime 和 kernel 的交互就会更复杂。

### 31.1 会发生什么

1. **权重局部性下降**；
2. **batch 内 shape 更杂**；
3. **cache 生命周期更难预测**；
4. **某些 fused path 失去适用条件**；
5. **dispatch 逻辑变得更重**。

### 31.2 应对思路

1. **对热 LoRA 做单独池化或缓存**；
2. **限制一个 batch 内可混合的 adapter 数量**；
3. **把最不规则的流量从主高吞吐池中分离出去**。

## 32. 多 GPU 推理里的内存系统问题不只是通信

很多人一说多卡推理就想到 tensor parallel 或 pipeline parallel，但真实问题还包括：

1. **KV cache 如何跨卡分布**；
2. **跨卡 attention 或 gather 带来的同步成本**；
3. **不同卡上的请求生命周期是否平衡**；
4. **跨卡 page / block 管理是否一致**。

### 32.1 为什么这会影响服务表现

因为在线服务里最怕局部热点卡拖住整条链路。哪怕平均 token/s 很高，只要少数卡的 cache 或 decode 阶段更重，P99 就会明显恶化。

## 33. 验证内存系统优化时不要只看吞吐

一个内存优化上线前，至少应验证：

1. **显存占用是否真的下降**；
2. **TPOT 是否稳定改善，而不是只在少数 shape 好看**；
3. **tail latency 是否没有被极端请求放大**；
4. **分配/释放路径是否引入新的停顿**；
5. **OOM、碎片和回退频率是否下降**。

### 33.1 为什么这特别重要

很多优化把平均吞吐做得更好，却让系统在长时间运行后更容易碎片化或出现偶发超时。在线服务最终拼的是**长期稳定性**，不是一组 30 秒 benchmark。

## 34. 一个更实用的推理内核优化顺序

1. **先拆清 prefill 与 decode**；
2. **确认瓶颈是 kernel、batch 组织、KV 访问还是 CPU/runtime**；
3. **优先治理 decode 的碎形状和 page locality**；
4. **再考虑更深的 kernel 特化和量化路径重写**；
5. **最后把优化放回真实混合流量里验收**。

### 34.1 为什么别反过来

如果一开始就把大量时间投入单个 GEMM kernel，而没有先处理 batch shaping 和 KV locality，常常只能得到漂亮的 microbenchmark，却拿不到实际 P99 改善。

## 35. 小结

推理 GPU 系统真正难的地方，不是某个 kernel 会不会写，而是**碎形状、分页缓存、量化元数据、请求动态生命周期和 runtime 调度**会不断互相作用。只有把这些因素一起建模，GPU kernel、batching 与内存系统的优化才会真正落到生产收益上。
