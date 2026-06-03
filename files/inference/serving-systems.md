# 推理服务系统：一次请求怎样穿过队列、KV 和 GPU

推理服务系统不是把模型加载起来然后调用 `generate()`。它是一条持续运行的生产线：请求进入队列，调度器决定谁先上 GPU，prefill 处理输入上下文，decode 逐 token 生成，KV cache 保存历史，采样和后处理把结果流式返回，观测系统负责证明到底慢在哪里。

这页只回答一个问题：**同一份模型权重，为什么离线单请求看起来很快，上线后用户仍然觉得慢。**

## 先分清两个时间尺度

LLM 推理有两个很不一样的阶段。

```text
prefill: 读完整 prompt，一次性生成 prompt token 的 KV cache
decode: 每一步只生成新 token，并反复读取已有 KV cache
```

prefill 更像一次大块并行计算。输入越长，模型要处理的 token 越多，attention 和 MLP 都会被拉大。长文档问答、RAG、附件总结、多模态长上下文，常常慢在首 token 出来之前。

decode 更像一个服务循环。每次只新增一个 token，却要在每层读历史 KV、跑 attention 和 MLP、采样下一个 token，然后把新 KV 追加回 cache。它的瓶颈常常不是单次 GEMM，而是持续循环里的 KV 访存、batch 变化、请求进出和采样后处理。

用户感受到的也正好分成两类指标：

\[
\text{TTFT}\approx T_{\text{queue}}+T_{\text{prefill}}+T_{\text{first decode}}
\]

\[
\text{TPOT}\approx \frac{T_{\text{decode}}}{L_{\text{out}}}
\]

第一行表示 time to first token：用户看到第一个 token 前，要等排队、prefill 和第一次 decode。第二行表示 time per output token：稳定生成阶段，每个输出 token 平均间隔多少。这里 \(L_{\text{out}}\) 是输出 token 数。

TTFT 好不代表整段回答流畅，TPOT 好也不代表用户马上看到首 token。聊天产品通常先看 TTFT，长生成和代码补全更关心 TPOT，多步 agent 还要看端到端任务完成时间。

## 一次请求的生命周期

一个线上请求通常穿过这条链路：

```text
client
  -> gateway / auth / quota
  -> scheduler queue
  -> prefill
  -> KV cache allocation
  -> decode loop
  -> sampling / detokenization / safety / tool call
  -> streaming response
  -> trace and metrics
```

这条链路里，每一段都可能让“模型很快”变成“服务很慢”。

| 阶段 | 系统在做什么 | 用户会怎么感受到 |
| --- | --- | --- |
| gateway | 鉴权、限流、解析请求、选择模型 | 请求还没进模型就排队或被拒 |
| scheduler | 按 SLO、长度、优先级和 batch 策略排队 | 首 token 等很久 |
| prefill | 处理 prompt，写入初始 KV cache | 长输入特别慢 |
| KV manager | 分配、复用、释放、淘汰 KV block | 并发一高显存吃紧 |
| decode loop | 逐 token 更新 batch 并生成输出 | 流式输出卡顿 |
| post-process | detokenize、采样约束、安全过滤、工具编排 | 模型算完后仍返回慢 |
| observability | 记录分阶段耗时、质量和成本 | 没有它就只能猜瓶颈 |

所以优化推理服务时，第一步不是换 runtime，而是把请求生命周期拆开。TTFT 慢可能是 queue，也可能是 prefill；TPOT 慢可能是 decode batch、KV 带宽，也可能是采样约束；端到端慢还可能是工具调用和后处理。

## KV cache 是隐藏容量上限

KV cache 保存每层 attention 的 key 和 value。没有它，decode 每生成一个 token 都要重算全部历史。用了它，历史不用重算，但显存会随并发、上下文长度和输出长度增长。

KV cache 的数量级可以用这个公式估算：

\[
M_{\text{KV}}
\approx
2\cdot L\cdot B\cdot T\cdot H_{\text{kv}}\cdot D_{\text{head}}\cdot b
\]

这里 \(2\) 是 key 和 value 两份缓存，\(L\) 是层数，\(B\) 是活跃请求数，\(T\) 是每个请求当前已缓存 token 数，\(H_{\text{kv}}\) 是 KV head 数，\(D_{\text{head}}\) 是 head 维度，\(b\) 是每个元素的字节数。

例如 32 层、batch 4、context 32k、8 个 KV heads、head dim 128、BF16：

\[
2\times32\times4\times32768\times8\times128\times2
\approx16\text{GiB}
\]

这行式子表示：仅 KV cache 就可能占到十几 GiB。若 KV heads 从 8 变 32，显存约 4 倍；若 context 从 32k 变 128k，也再乘 4。很多“支持 128k 上下文”的模型，真正上线时会被 batch、KV 精度、GQA/MQA、prefix cache 和 SLO 重新限制。

## PagedAttention 把 KV 管成分页内存

vLLM 的经典贡献是把调度、KV cache manager、block allocator 和模型 worker 作为一个整体系统来设计。PagedAttention 负责把 KV cache 切成 block，并用 block table 把逻辑 token 位置映射到物理 block，类似操作系统分页。

![vLLM serving system overview](../assets/images/paper-figures/inference/vllm-system-overview-padded.png){ width="900" }

<small>图源：[vLLM PagedAttention blog](https://vllm.ai/blog/2023-06-20-vllm)。原图表达 scheduler、KV cache manager、CPU/GPU block allocator 和 worker 之间的关系。本文用它说明：serving runtime 的核心不是模型类，而是请求调度、KV 分配和 GPU worker 执行共同决定吞吐与尾延迟。</small>

传统做法如果按最大上下文给每个请求预留 KV，短请求会浪费显存；如果动态扩容，又容易碎片化或搬移。PagedAttention 让请求需要多长就追加多少 block，请求结束就释放 block，共享前缀时还可以复用 block。这样 continuous batching 才有空间发挥：有请求完成就从 batch 里移走，有新请求进来就加入，GPU 不必等所有序列同长。

但 PagedAttention 不会让所有瓶颈消失。长 prompt 短输出主要痛在 prefill；短 prompt 长输出更容易痛在 decode 和 KV 带宽；大量 agent 请求还会痛在共享前缀、分支和工具链。读 vLLM benchmark 时，必须看 prompt/output 长度分布，而不是只看平均 tokens/s。

## Continuous batching 提高吞吐，也可能伤害尾延迟

静态 batch 要等一批请求凑齐再跑。continuous batching 允许 decode 每一步动态加入和移除请求：

\[
\mathcal B_{t+1}=(\mathcal B_t\setminus\mathcal F_t)\cup\mathcal A_t
\]

这里 \(\mathcal B_t\) 是当前 decode batch，\(\mathcal F_t\) 是这一轮已经完成的请求，\(\mathcal A_t\) 是新加入的请求。这个公式表达的是 iteration-level scheduling：batch 在每个生成 step 都可能变化。

它的好处是 GPU 不容易空转，吞吐上升。代价是调度更复杂，batch window、优先级、输入长度和输出长度会互相影响。一个常见故障是：把 batch window 拉大后，tokens/s 提升，短问答 TTFT P95 反而变差。原因不是模型算慢了，而是短请求为了合批在 queue 里等得更久。

Orca 的 iteration-level scheduling、vLLM 的 continuous batching、TensorRT-LLM 的 in-flight batching，本质上都在处理同一个问题：**服务系统不应该把生成式请求当成一次固定 batch 的矩阵乘，而应该把它看成每一步都在变化的流。**

## Prefill 和 decode 可以分开服务

Prefill 和 decode 的资源画像不同。prefill 通常更吃算力和长序列 attention，decode 更吃 KV cache、内存带宽和调度循环。把它们放在同一池 GPU 上，容易互相干扰：长 prompt 会堵住短聊天的 TTFT，长输出会占住 KV 让新请求进不来。

DistServe 这类工作提出把 prefill 和 decode disaggregate：prefill worker 专门处理输入上下文，decode worker 专门处理逐 token 生成。这样可以针对两个阶段分别调 batch、并发和资源。

这不是免费午餐。分离后要传递 KV 或中间状态，要处理网络、显存和调度开销。它适合输入/输出长度分布差异大、SLO 清晰、规模足够大的服务；小规模单机部署可能反而被复杂度吃掉收益。

判断是否需要分离，先看请求桶：

| 请求桶 | 主瓶颈 | 常见策略 |
| --- | --- | --- |
| 短聊天 | queue、首步 decode、batch 等待 | 小等待窗口、短请求池、低 TTFT SLO |
| 长文档问答 | prefill、prefix cache、KV | 上下文裁剪、prefix cache、prefill 池 |
| 长输出生成 | decode、KV 生命周期 | decode 池、输出长度控制、投机解码 |
| agent / 工具链 | 多轮调用、共享前缀、外部工具 | prefix reuse、结构化 trace、工具缓存 |

不要用一个平均 tokens/s 代表所有用户体验。推理服务的真实单位不是 token，而是“在 SLO 内完成的有效请求”。

## 路由不是省钱小技巧，而是服务形态

生产系统通常不应该让所有请求都走同一条路。闲聊、长文档、代码补全、多模态、工具 agent、高风险任务，对质量、延迟和成本的要求不同。

路由可以抽象成：

\[
m^\star
=
\arg\min_{m\in\mathcal M}
\left(
\lambda_1\text{cost}(m)
+\lambda_2\text{latency}(m,x)
-\lambda_3\text{quality}(m,x)
\right)
\]

这里 \(m^\star\) 是选中的模型或服务链路，\(\mathcal M\) 是候选模型集合，\(\lambda\) 控制成本、延迟和质量的权重，\(x\) 是请求特征。这个公式不是要线上真的用线性打分，而是提醒：路由必须同时考虑任务类型、输入长度、输出长度、风险、缓存命中和 SLO。

路由做得好，简单请求不占贵模型，长文档不拖短聊天，agent 工具链能走可复用前缀的 runtime。路由做坏，复杂任务会被弱模型误杀，或简单任务把高端 GPU 占满。

## 投机解码只解决一部分 decode 问题

Speculative decoding 让 draft model 先猜多个 token，target model 一次验证，命中时可以减少 target decode step。它适合 target decode 是瓶颈、draft 足够便宜且接受率较高的场景。

但投机解码不是万能按钮。长 prompt 的 prefill 不会因为 draft 变快；结构化输出、低温采样、代码、工具调用和长上下文会改变接受率；batching 和 KV cache 也会影响端到端收益。

读 EAGLE、SpecInfer 或 TensorRT-LLM speculative decoding 文档时，要看四个数：draft 开销、接受率、target 验证开销、端到端 SLO。只报告 token/s speedup，不足以说明线上体验一定变好。

## 最小观测闭环

推理优化最怕“平均值看起来不错”。最小观测集应该按阶段记录。

| 类别 | 必看指标 |
| --- | --- |
| 请求形态 | 输入长度、输出长度、任务类型、模型路由 |
| 队列 | queue time、admission reject、priority、batch wait |
| prefill | prefill time、prefill tokens/s、prefix hit rate |
| decode | TPOT、decode batch、tokens/s、sampling config |
| KV | KV GiB、block usage、eviction、fragmentation、prefix reuse |
| 尾延迟 | P50/P95/P99 TTFT、TPOT、end-to-end latency |
| 质量 | 任务成功率、引用正确率、格式错误、fallback |
| 成本 | 每请求成本、每 token 成本、每成功任务成本 |

如果没有这些 trace，优化动作就会互相打架：batch window 提升吞吐但伤 TTFT；压上下文省 prefill 但伤回答质量；低比特 KV 省显存但引入长上下文退化；路由省钱但让困难任务失败。

## 阅读结论

推理服务系统的核心不是“哪个 runtime 最快”，而是一次请求在队列、prefill、KV cache、decode、采样、后处理和工具链中怎样流动。prefill 决定首 token 前的成本，decode 决定流式生成的节奏，KV cache 决定并发和长上下文上限，batching 决定吞吐和尾延迟的取舍，路由决定不同任务是否走对链路。

所以排查线上慢请求时，先不要说“模型慢”。先问：它是在排队，还是 prefill；是在 decode 读 KV，还是 sampling；是在等工具，还是走错模型；SLO 是 TTFT、TPOT、端到端，还是每成功任务成本。把这些问题拆开，推理服务才从黑箱变成可以改的系统。

## 外部精读

- [vLLM PagedAttention blog](https://vllm.ai/blog/2023-06-20-vllm)：理解 KV cache 分页、block table 和 continuous batching。
- [Orca OSDI paper](https://www.usenix.org/conference/osdi22/presentation/yu)：理解 iteration-level scheduling 为什么适合生成式模型。
- [DistServe paper](https://arxiv.org/abs/2401.09670)：理解 prefill / decode disaggregation 和 goodput-oriented serving。
- [Hugging Face Text Generation Inference](https://huggingface.co/docs/text-generation-inference/index)：看成熟 LLM serving 栈的部署入口和功能边界。
- [SGLang documentation](https://docs.sglang.ai/)：理解复杂生成程序、prefix reuse 和 RadixAttention。
- [NVIDIA TensorRT-LLM documentation](https://nvidia.github.io/TensorRT-LLM/)：看 in-flight batching、KV cache system、quantization 和 engine 构建怎样进入生产推理。
- [NVIDIA Inference Optimization blog](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)：补充 batching、quantization、KV cache 和 speculative decoding 的工程视角。
- [智源转载：全栈 Transformer 推理优化](https://hub.baai.ac.cn/view/33449)：中文长文入口，适合学习它按模型、算子、显存、调度分层讲推理优化的方式。
