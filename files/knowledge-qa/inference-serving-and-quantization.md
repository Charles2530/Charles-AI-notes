# 推理服务与量化 QA

LLM 推理、服务系统、KV cache、batching、投机解码、运行时和量化的高频问题，都要把 prefill、decode、显存、吞吐、尾延迟和质量回归分开讲。

## 推理基础与服务指标

### Q：训练和推理为什么要分开看？

**面试回答。** 训练的目标是把参数学出来，核心瓶颈是数据、loss、优化器、反向传播、通信和收敛稳定性；推理的目标是把已训练模型低延迟、高吞吐、低成本地服务出去，核心瓶颈是 KV cache、batch 调度、内存带宽、尾延迟和质量回归。很多训练阶段有效的技术，到了在线推理未必划算。

**追问展开。** 面试里最好举例：activation checkpointing 训练省显存但推理没反向传播；KV cache 推理很关键但训练 teacher forcing 时不是同一个瓶颈；量化推理省显存但训练低精度要更谨慎。  
**易错点 / 边界。** 不能拿训练吞吐、离线 benchmark 或单条 demo 直接代表在线服务能力。  
**继续读。** [推理服务系统](../inference/serving-systems.md)

### Q：TTFT、TPOT、吞吐和 P99 分别看什么？

**面试回答。** TTFT 是 Time To First Token，反映用户等首字的时间，常受 prefill 和排队影响；TPOT 是 Time Per Output Token，反映 decode 速度；吞吐看单位时间处理多少 token 或请求；P99 看最慢 1% 请求的体验。一个服务要同时看平均值和尾部，否则容易牺牲少数用户体验换平均吞吐。

**追问展开。** 面试中可以把指标和阶段绑定：长 prompt 影响 TTFT，长输出影响 TPOT，总并发影响吞吐，调度和资源争用影响 P95/P99。  
**易错点 / 边界。** 指标必须带 workload：prompt 长度、输出长度、batch、模型、硬件、量化方式和并发量都要说明。  
**继续读。** [成本建模与 SLO 设计](../inference/cost-modeling-and-slo-design.md) / [观测与在线评估](../inference/observability-and-online-evaluation.md)

### Q：prefill 和 decode 的瓶颈为什么不同？

**面试回答。** Prefill 对整段 prompt 做前向计算，attention 和 MLP 都可以大块矩阵化，通常更 compute-bound；decode 每次只生成一个新 token，但要读取所有历史 KV，常常更 memory-bandwidth-bound。可以简化理解为：prefill 在“算大矩阵”，decode 在“反复读缓存”。

**追问展开。** 长 prompt 场景中 prefill 可能主导 TTFT；长回答和多轮对话中 decode 的 KV 读取和调度可能主导成本。PD 分离、chunked prefill、continuous batching 都是在围绕这两个阶段做资源匹配。  
**易错点 / 边界。** 不同模型结构和服务流量会改变瓶颈；不要只用单条请求的 profiler 推断生产系统。  
**继续读。** [GPU kernels、batching 与内存系统](../inference/gpu-kernels-batching-and-memory-systems.md) / [推理运行时](../inference/serving-runtimes-vllm-sglang-and-tensorrt-llm.md)

### Q：KV cache 为什么能加速自回归解码？

**面试回答。** 自回归生成第 \(t\) 个 token 时，前 \(t-1\) 个 token 的 K/V 已经计算过。如果不缓存，每一步都要重新跑整个前缀；有了 KV cache，只需要为新 token 计算 Q/K/V，并用新 Q 去 attend 历史 K/V。这样把重复计算变成缓存读取。

\[
\text{KV size} \propto L \times B \times S \times H_{kv} \times d_h \times 2
\]

其中 \(L\) 是层数，\(B\) 是 batch，\(S\) 是上下文长度，\(H_{kv}\) 是 KV heads。  
**追问展开。** KV cache 是推理服务的核心资产：它决定最大 batch、最大上下文、显存占用和调度策略。MHA/MQA/GQA 的区别也直接体现在 \(H_{kv}\) 上。  
**易错点 / 边界。** KV cache 加速计算但吃显存；长上下文下它可能成为主要成本。  
**继续读。** [KV、缓存与投机解码](../inference/caching-routing-and-speculative.md) / [上下文压缩、KV 驱逐与记忆层级](../inference/context-compression-kv-eviction-and-memory-hierarchies.md)

### Q：连续批处理为什么适合 LLM 服务？

**面试回答。** LLM 请求长度和生成长度差异很大，静态 batch 会被最长请求拖住，短请求结束后 GPU slot 不能及时补新请求。Continuous batching 在每个 decode step 动态加入和移除请求，让 GPU 尽量保持有活可做，提高吞吐和资源利用。

**追问展开。** 面试时可以说它把“请求级 batch”变成“token-step 级调度”。配合 KV block 管理、优先级队列和 prefix cache，可以更好处理真实流量。  
**易错点 / 边界。** 更高吞吐可能带来排队和尾延迟问题；调度策略要服务 SLO，而不是只最大化 tokens/s。  
**继续读。** [推理运行时](../inference/serving-runtimes-vllm-sglang-and-tensorrt-llm.md) / [推理服务系统](../inference/serving-systems.md)

## KV 管理与运行时

### Q：PagedAttention 的核心思想是什么？

**面试回答。** PagedAttention 把 KV cache 切成固定大小的 block，用 block table 管理逻辑 token 位置到物理 KV block 的映射，类似操作系统虚拟内存。这样可以减少连续大块分配带来的碎片，支持动态 batch、共享前缀和 copy-on-write。

**追问展开。** 没有分页时，不同请求长度差异会造成 KV 显存浪费；分页后，物理块可以非连续，调度器只需要维护映射关系。vLLM 的高吞吐核心之一就是把 KV cache 当成可管理的内存系统。  
**易错点 / 边界。** PagedAttention 解决的是 KV 内存和调度效率，不直接提升模型质量。  
**继续读。** [vLLM / PagedAttention：KV cache 分页管理](../paper-deep-dives/inference/vllm-pagedattention.md) / [vLLM 官方博客](https://vllm.ai/blog/2023-06-20-vllm)

### Q：vLLM、SGLang 和 TensorRT-LLM 怎么区分？

**面试回答。** vLLM 更强调通用 LLM serving、高吞吐调度、PagedAttention 和 OpenAI-compatible API；SGLang 更强调结构化 LLM 程序、prefix cache、RadixAttention 和复杂 agent/RAG 执行；TensorRT-LLM 更贴近 NVIDIA 生态，强调图优化、kernel、量化和生产部署性能。

**追问展开。** 面试选型可以按问题回答：如果要快速搭通用高吞吐服务，vLLM 常见；如果有大量共享前缀、结构化生成和 agent workflow，SGLang 值得看；如果深度绑定 NVIDIA 部署、追求极致优化，TensorRT-LLM 更合适。  
**易错点 / 边界。** 运行时选择不是静态答案，要看模型、硬件、并发、量化、结构化输出、团队能力和生态兼容。  
**继续读。** [推理运行时](../inference/serving-runtimes-vllm-sglang-and-tensorrt-llm.md) / [SGLang documentation](https://docs.sglang.io/)

### Q：prefix cache 和 RadixAttention 解决什么？

**面试回答。** 多轮对话、系统 prompt、RAG 模板和 agent 工具说明经常重复。如果每个请求都重新 prefill 这些公共前缀，会浪费大量计算。Prefix cache 复用共享前缀的 KV；RadixAttention 用前缀树组织不同请求共享的 token 前缀。

**追问展开。** 可以把它类比成“上下文 memoization”：相同前缀只算一次，后续请求从缓存接着 decode 或继续 prefill。对固定系统提示、few-shot 示例和多轮会话尤其有效。  
**易错点 / 边界。** 前缀复用依赖 prompt 规范化和重复率；隐私隔离、缓存失效、版本更新和动态拼接都要处理。  
**继续读。** [RAG、Agent 与长上下文系统](../inference/rag-agents-and-long-context-systems.md) / [SGLang 论文](https://arxiv.org/abs/2312.07104)

### Q：PD 分离为什么会出现？

**面试回答。** Prefill 和 decode 的资源画像不同：prefill 更像大矩阵计算，decode 更像高频小步调度和 KV 读取。PD 分离把 prefill 和 decode 放到不同资源池或服务节点，让两类阶段分别优化，并通过 KV 传输或共享机制衔接。

**追问展开。** 面试时可以说：长 prompt、高并发、低 TTFT 要求会推动 prefill 独立扩容；长输出、高 decode 并发会推动 decode 池优化。PD 分离是大规模服务架构问题，不只是模型推理技巧。  
**易错点 / 边界。** 分离会引入 KV 传输、网络延迟、调度复杂度和故障恢复成本；中小规模服务未必需要。  
**继续读。** [分离式 prefill、KV 服务与容量运维](../inference/disaggregated-prefill-kv-services-and-capacity-ops.md) / [推理服务系统](../inference/serving-systems.md)

### Q：MoE 模型推理为什么不只是“少算专家”？

**面试回答。** MoE 每个 token 只激活 top-k 专家，理论上减少单 token 计算，但推理系统要处理 router、专家并行、all-to-all 通信和专家负载均衡。真实服务中，不同请求 token 会路由到不同专家，导致负载和通信非常动态。

**追问展开。** 面试可以补：MoE 参数量大但激活参数少，适合扩容量；但 serving 时专家分布不均会造成某些 GPU 热点，batch 内 token routing 也会影响 kernel 和网络效率。  
**易错点 / 边界。** MoE 的系统收益依赖硬件拓扑、batch、token 分布和实现；不是所有场景都比 dense 模型划算。  
**继续读。** [MoE 路由与多模型服务](../inference/moe-routing-and-multi-model-serving.md) / [MoE 与大模型架构](../foundations/moe-and-large-model-architecture-basics.md)

## 投机解码与长上下文

### Q：Speculative decoding 为什么能加速？

**面试回答。** 投机解码让便宜的 draft model 先生成多个候选 token，再让昂贵的 target model 一次性验证。若 target 接受多个 token，就减少了 target model 的逐 token 调用次数。核心收益来自：draft 足够便宜、候选接受率足够高、验证能并行。

**追问展开。** 可以把速度收益粗略理解成 target 每次前向能推进多个 token，而不是一个 token。关键指标是 acceptance rate；如果 draft 经常猜错，额外 draft 成本会抵消收益。  
**易错点 / 边界。** 它优化延迟/吞吐，不提升 target model 本身质量；采样一致性和接受规则也要保证输出分布正确。  
**继续读。** [KV、缓存与投机解码](../inference/caching-routing-and-speculative.md) / [EAGLE：为什么 draft 不一定要是小模型](../paper-deep-dives/inference/eagle.md)

### Q：EAGLE 系列和普通小模型 draft 有什么不同？

**面试回答。** 普通 speculative decoding 往往用一个小语言模型预测 draft token；EAGLE 路线更强调利用 target model 的 hidden states 或更贴近 target 分布的特征来生成 draft。它不是简单“小模型猜，大模型验”，而是在提高 draft 质量和接受率。

**追问展开。** EAGLE-2 关注动态 draft tree，根据不同位置和置信度调整候选；EAGLE-3 关注训练时让 draft 见到自己的错误，减少训练-推理错配。  
**易错点 / 边界。** EAGLE 的证据主要是系统吞吐/延迟收益，不能写成模型能力提升。  
**继续读。** [EAGLE-2：动态 Draft Tree](../paper-deep-dives/inference/eagle-2.md) / [EAGLE-3：训练时见过自己犯错](../paper-deep-dives/inference/eagle-3.md)

### Q：长上下文推理为什么贵？

**面试回答。** 长上下文带来两类成本：prefill 阶段要处理更长的 prompt，attention 计算和内存读写增长；decode 阶段要保存并读取更长的 KV cache。上下文越长，TTFT、显存、KV 带宽和调度复杂度都会上升。

**追问展开。** 面试回答可以分层：位置编码解决“能否表示长距离”，attention/kernel 解决“能否高效计算”，KV 管理解决“能否服务化”，评测解决“是否真的用到远处信息”。  
**易错点 / 边界。** 长窗口不是长记忆；needle 测试通过也不代表复杂多跳推理、长期一致性或 agent 记忆可靠。  
**继续读。** [上下文压缩、KV 驱逐与记忆层级](../inference/context-compression-kv-eviction-and-memory-hierarchies.md) / [FlashAttention 与长上下文](../operators/long-context-and-flashattention-evolution.md)

### Q：KV 压缩或驱逐为什么可能伤害质量？

**面试回答。** KV cache 存的是每层历史 token 的 key/value 表示，后续 token 会通过 attention 读取它们。压缩、合并、量化或驱逐都会改变模型可访问的历史信息，因此可能伤害长程依赖、引用一致性、工具状态和安全边界。

**追问展开。** 评估时不能只看显存下降或 tokens/s 上升，还要看长文问答、代码补全、多轮对话、RAG 引用和特定任务成功率。  
**易错点 / 边界。** 系统吞吐证据只能说明成本下降，不能证明语义保真或决策可靠。  
**继续读。** [上下文压缩、KV 驱逐与记忆层级](../inference/context-compression-kv-eviction-and-memory-hierarchies.md) / [KVSlimmer：非对称 KV 合并](../paper-deep-dives/inference/kvslimmer.md)

### Q：KV cache 的压缩、驱逐和分层存储有什么区别？

**面试回答。** 压缩改变 KV 的表示精度或维度，驱逐决定哪些历史 token 不再保留，分层存储则把 KV 放到 GPU、CPU 或远端存储的不同层级。三者分别在表示大小、保留范围和驻留位置上交换成本。

**追问展开。** 量化 KV 主要降低每个 token 的字节数；token pruning 或 eviction 降低保留 token 数；offload 保留信息但增加搬运延迟。选择时要结合长文引用、多轮一致性、命中率、带宽和尾延迟分桶验证。  
**易错点 / 边界。** 显存占用下降只是资源证据。历史细节被量化、删除或换出后，质量和 TPOT 都可能出现新的瓶颈。  
**继续读。** [上下文压缩、KV 驱逐与记忆层级](../inference/context-compression-kv-eviction-and-memory-hierarchies.md) / [QAT、Kernel 与 KV Cache](../quantization/qat-kernels-and-kv-cache.md)

## 量化与部署

### Q：PTQ、QAT、GPTQ、AWQ、SmoothQuant 怎么分？

**面试回答。** PTQ 是训练后量化，不重新完整训练模型；QAT 在训练中模拟量化误差，让模型适应低精度。GPTQ 是权重量化方法，用近似二阶信息减少量化误差；AWQ 保护少量重要权重通道；SmoothQuant 把激活离群值的量化难点平滑迁移到权重侧。

**追问展开。** 可以按“量化对象”组织：weight-only 主要压权重，activation quantization 要处理激活离群值，KV quantization 则影响长上下文缓存。按“是否训练”组织：PTQ 便宜，QAT 成本高但可能质量更稳。  
**易错点 / 边界。** 方法名不是部署结论；硬件 kernel、运行时支持、校准数据和任务回归同样决定效果。  
**继续读。** [PTQ / GPTQ / AWQ / SmoothQuant](../quantization/ptq-gptq-awq-smoothquant.md) / [量化运行时与框架](../quantization/quantization-runtimes-and-frameworks.md)

### Q：为什么量化不一定显著降准确率？

**面试回答。** 大模型参数有冗余，许多权重的小幅扰动不会显著改变输出分布；好的量化方法会保护敏感通道或用校准数据估计 scale。以均匀量化为例，一个浮点值可以近似成：

\[
x_q=\operatorname{round}\left(\frac{x}{s}\right),\quad \hat{x}=s x_q
\]

其中 \(s\) 是 scale。  
**追问展开。** GPTQ、AWQ、SmoothQuant 的差异就在于如何选择补偿、保护或迁移量化误差。面试回答要说清“量化误差不是平均重要，少数通道很关键”。  
**易错点 / 边界。** 数学、代码、长上下文、少数语言和安全场景可能对量化更敏感；必须单独评估。  
**继续读。** [量化评估与部署检查清单](../quantization/evaluation-and-deployment-checklist.md) / [低比特 LLM Survey](../paper-deep-dives/inference/low-bit-llm-survey.md)

### Q：weight-only quantization 和 W&A quantization 有什么区别？

**面试回答。** Weight-only 只量化权重，主要减少权重显存和权重读取带宽，常用于 LLM 推理；W&A 同时量化权重和激活，潜在加速更大，但需要处理激活离群值、动态 scale 和高效低精度 kernel。后者更难，但如果硬件支持好，收益更完整。

**追问展开。** Decode 阶段除了权重读取，还要读 KV cache；所以 weight-only 可能不能解决所有瓶颈。Prefill 阶段大矩阵计算多，W&A 或 FP8 路径可能更重要。  
**易错点 / 边界。** 量化对象要和瓶颈匹配；只看模型大小下降不代表端到端延迟下降。  
**继续读。** [PTQ / GPTQ / AWQ / SmoothQuant](../quantization/ptq-gptq-awq-smoothquant.md) / [FP8 与混合精度推理](../quantization/fp8-and-mixed-precision-serving.md)

### Q：量化、蒸馏、剪枝和 NAS 分别改哪张成本账？

**面试回答。** 量化主要改变数值格式与存储带宽，蒸馏改变 student 的容量和行为，剪枝删除权重、通道、head 或 block，NAS 在给定搜索空间与硬件预算下选择结构。它们可能降低参数量、FLOPs、显存或延迟，但这些指标不会自动同步下降。

**追问展开。** 非结构化稀疏若没有硬件 kernel 支持，稀疏率很高也可能不加速；结构化剪枝更容易改变 dense shape，但要维护残差和通道依赖；平台感知 NAS 应把目标硬件的真实 latency 放入搜索目标。  
**易错点 / 边界。** 参数少、MACs 低和线上 P99 低是三种证据。压缩方法必须命中真实瓶颈，并在同硬件、同 runtime、同流量分布下验收。  
**继续读。** [模型压缩、剪枝与 NAS](../foundations/model-compression-pruning-and-nas-basics.md) / [量化评估与部署检查清单](../quantization/evaluation-and-deployment-checklist.md)

### Q：FP8 和 INT8 是一回事吗？

**面试回答。** 不是。INT8 是整数定点表示，通常依赖 scale 和 zero-point 把浮点范围映射到整数；FP8 是 8-bit 浮点格式，保留符号、指数和尾数，动态范围和精度分配方式不同。FP8 更贴近浮点计算路径，常用于特定硬件上的训练或推理矩阵乘。

**追问展开。** INT8 常见于推理量化，尤其权重或激活；FP8 更依赖硬件 Tensor Core、scale 管理和累加精度。面试时要强调“bit 数相同，不代表数值行为相同”。  
**易错点 / 边界。** FP8 是否更快取决于硬件、kernel、shape、累加精度和运行时支持。  
**继续读。** [FP8 与混合精度推理](../quantization/fp8-and-mixed-precision-serving.md) / [低比特训练与数值格式](../training/low-bit-training-and-numerics.md)

### Q：KV cache 也能量化吗？

**面试回答。** 可以。KV cache 量化把历史 K/V 用更低比特存储，直接降低长上下文和大 batch decode 的显存占用。但 KV 是后续每一步 attention 的输入，误差会反复影响生成，所以要比单纯权重量化更谨慎。

**追问展开。** 面试时可以说：KV 量化适合长上下文服务，但要测长文问答、引用一致性、多轮对话、代码和安全场景。还要考虑 per-token/per-channel scale、分层量化和 cache eviction 配合。  
**易错点 / 边界。** KV 显存下降不等于输出质量不变；尤其长程依赖任务可能更敏感。  
**继续读。** [QAT、kernels 与 KV cache](../quantization/qat-kernels-and-kv-cache.md) / [上下文压缩、KV 驱逐与记忆层级](../inference/context-compression-kv-eviction-and-memory-hierarchies.md)

### Q：量化上线前最少要验什么？

**面试回答。** 至少要验三类：质量、性能和稳定性。质量包括通用 benchmark、业务评测、长上下文、结构化输出和安全样本；性能包括 TTFT、TPOT、吞吐、显存和 P99；稳定性包括不同 batch、不同长度、不同硬件、fallback 和错误样本。

**追问展开。** 还要记录校准集、量化参数、runtime 版本、kernel 版本和硬件。量化是模型和系统共同变更，不能只改权重文件就上线。  
**易错点 / 边界。** 平均分数不掉不代表所有任务不掉；低频高价值场景要单独看。  
**继续读。** [量化评估与部署检查清单](../quantization/evaluation-and-deployment-checklist.md) / [观测与在线评估](../inference/observability-and-online-evaluation.md)

### Q：推理优化的最终指标应该是什么？

**面试回答。** 最终要看满足质量门槛下的成本、延迟和稳定性，而不是单点 tokens/s。比较稳的口径是 cost per successful answer、P95/P99 latency、错误率、fallback rate、GPU 利用、显存余量和质量回归。

**追问展开。** 面试里可以用一句话收束：系统优化只有在“不降低关键质量和安全”的前提下才算收益。Microbenchmark、kernel speedup、平均吞吐都是中间证据。  
**易错点 / 边界。** 如果更快导致幻觉、格式错误、安全拒答异常或尾延迟恶化，就不能说整体更好。  
**继续读。** [成本建模与 SLO 设计](../inference/cost-modeling-and-slo-design.md)

### Q：为什么推理优化不能只看 tokens/s，还要看成本模型和 SLO？

**面试回答。** tokens/s 只说明单位时间产出 token 的速度，不说明用户是否等得起、一次成功回答多少钱、尾延迟是否稳定、质量是否下降。生产推理通常要同时满足 SLO，例如 TTFT、TPOT、P95/P99 latency、错误率和成本预算。

**追问展开。** 成本模型会把 GPU 小时、模型大小、输入/输出长度、KV cache、batching、缓存命中、fallback 和多模型路由都纳入。一个优化如果提升平均吞吐，却让 P99 变差或让长文档任务质量下降，业务上可能是负收益。  
**易错点 / 边界。** Microbenchmark 的 speedup 不能直接外推到线上；真实流量有长度分布、多租户、RAG 拼接、工具等待和突发峰值。  
**继续读。** [成本建模与 SLO 设计](../inference/cost-modeling-and-slo-design.md) / [推理优化清单与容量规划](../inference/optimization-checklist-and-capacity-planning.md)
