# 模型推理系统入门路线图




<!-- [补充] -->
| 读者 | 建议读法 | 工程落点 |
| --- | --- | --- |
| 初学者 | 先抓“模型推理系统入门路线图”要解决的问题、关键术语和最小例子；公式只看变量方向。 | 能说清输入、输出和常见失败信号。 |
| 工程师 | 直接对照结构、指标、成本和边界条件，必要时回到前置页补概念。 | 落到数据、算力、延迟、显存、评测和回滚检查。 |

<div class="atlas-hero">
  <div class="atlas-kicker">Inference Systems</div>
  <p class="atlas-lead">这一章用“一次请求从进入系统到返回答案”的视角讲模型推理。先把 queue、prefill、decode、KV cache、batching、routing、cost 和 SLO 这些基础概念讲清楚，再连接 vLLM、SGLang、TensorRT-LLM、投机解码和长上下文系统。</p>
  <div class="atlas-chip-row">
    <span class="atlas-chip">Queue</span>
    <span class="atlas-chip">Prefill</span>
    <span class="atlas-chip">Decode</span>
    <span class="atlas-chip">KV Cache</span>
    <span class="atlas-chip">SLO</span>
  </div>
</div>

!!! note "合并提示"
    **note · 读法定位**

    这页先回答“模型推理系统入门路线图”在「推理」里的位置：它解决什么局部问题，依赖哪些前置，最后会影响哪类工程或研究判断。
    **前置**：先知道 transformer 解码、GPU kernel、显存和吞吐/延迟的区别。 必要时先看本页路线，或回 [基础知识](../foundations/index.md) 或 [术语表](../glossary/index.md)。
    **主线关系**：把请求生命周期、KV Cache、batching、runtime、SLO 和成本连接起来，看模型上线后每一毫秒花在哪里。

    **note · 初学者先抓住**

    推理不是“训练完模型后调用一次 `generate()`”。线上推理是一条服务链路：请求先排队，长输入先做 prefill，模型逐 token decode，历史 token 的 KV cache 一直占显存，调度器把许多请求合批，路由器决定走哪条模型链路，最后用成本和 SLO 判断这套系统值不值得。

    **example · 有趣例子：餐厅出餐**

    用户下单后不会直接吃到菜。前台排队、后厨备菜、厨师一道道出菜、餐具和桌位周转、外卖配送都会影响体验。推理系统也一样：模型本身只是厨师，队列、KV cache、batching、路由和监控共同决定用户看到首 token 的速度、整段回答的流畅度和系统成本。

<!-- [补充] -->
> **踩坑原因**：这类工程问题通常不是单点公式错，而是 shape、资源预算、并发路径和观测指标之间有一个假设没对齐；先把触发条件、监控信号和回滚边界拆开查。

## 先认一遍术语

| 术语 | 初学者解释 | 你应该问的问题 |
| --- | --- | --- |
| request | 一次用户调用，包含输入、配置和输出目标 | 它是短聊天、长文档、agent 还是多模态请求？ |
| queue | 请求等 GPU 或等调度的时间 | 慢是模型算得慢，还是排队太久？ |
| prefill | 模型一次性读完整个输入，并写入 KV cache | 输入上下文越长，首 token 前越贵 |
| decode | 模型根据历史逐 token 生成输出 | 输出越长，decode 步数越多 |
| KV cache | 每层 attention 为历史 token 保存的 key/value | 显存够不够，长上下文并发能不能撑住？ |
| batching | 把多个请求合在一起跑，提高 GPU 利用率 | 吞吐变好时，TTFT/P99 有没有变差？ |
| routing | 让不同请求走不同模型或服务路径 | 简单请求是否占用了昂贵模型？ |
| TTFT | time to first token，首 token 延迟 | 用户多久看到第一段输出？ |
| TPOT | time per output token，输出 token 间隔 | 流式回答是否顺滑？ |
| P95/P99 | 最慢 5%/1% 请求的体验 | 平均值好看时，尾部用户是否仍然很慢？ |
| SLO | 服务必须守住的目标 | 什么情况算可接受，什么情况必须降级或回滚？ |

## 一次请求的生命周期

最推荐的新手读法，是先固定一条请求路径：

```text
用户请求
  -> 网关和鉴权
  -> 队列与调度
  -> prefill 输入上下文
  -> 写入 KV cache
  -> decode 逐 token 输出
  -> 后处理、工具、过滤和返回
  -> trace、评测和成本复盘
```

vLLM 原论文图适合作为第一张系统图。它提醒我们：推理 runtime 不是一个裸 `forward()`，而是 scheduler、KV cache manager、block allocator 和 workers 的协作系统。

![vLLM system overview 原论文图](../assets/images/paper-figures/inference/vllm-system-overview.png){ width="760" }

<small>图源：[Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)，Figure 3。原论文图意：vLLM 使用 centralized scheduler 协调 GPU workers，并由 KV cache manager 通过 paged block 方式管理 GPU/CPU KV cache。</small>

!!! note "这张图怎么看"
    先看左侧 scheduler：它决定哪些请求一起进入 GPU。再看中间 KV cache manager：它决定历史 token 的 key/value 放在哪里。最后看右侧 workers：它们才执行模型计算。TTFT、TPOT、吞吐和 P99 往往不是某一个模块单独决定的，而是这三层一起决定。

## 最重要的时延公式

对自回归大模型，一次请求总时延可以先粗略写成：

\[
T_{\text{req}}
=
T_{\text{queue}}
+ T_{\text{prefill}}
+ T_{\text{decode}}
+ T_{\text{post}}
+ T_{\text{network}}
\]

<!-- [补充] -->
| 符号 | 含义 | 取值范围 |
| --- | --- | --- |
| \(T_{}\) | 时间步、索引或计数 | 整数索引或归一化时间区间，见正文 |


每个符号的含义：

若输出长度为 \(n\)，decode 常被近似成：

\[
T_{\text{decode}}
\approx
\sum_{i=1}^{n} T_{\text{step}}^{(i)}
\approx
n\cdot T_{\text{step}}
\]

<!-- [补充] -->
| 符号 | 含义 | 取值范围 |
| --- | --- | --- |
| \(T_{}\) | 时间步、索引或计数 | 整数索引或归一化时间区间，见正文 |
| \(i\) | 时间步、索引或计数 | 整数索引或归一化时间区间，见正文 |
| \(n\) | 时间步、索引或计数 | 整数索引或归一化时间区间，见正文 |
| \(T_{}^{(i)}\) | 时间步、索引或计数 | 整数索引或归一化时间区间，见正文 |


这里 \(T_{\text{step}}^{(i)}\) 是第 \(i\) 个输出 token 的生成时间。第二个近似成立的前提是 decode 进入相对稳定状态；真实线上系统里，batch 变化、KV page miss、投机解码失败和工具中断都会让每一步不完全一样。

!!! warning "常见误解"
    tokens/s 高，不代表用户体验一定好。一个系统可以在大 batch 下 tokens/s 很高，但新请求等待 batch 凑齐，导致 \(T_{\text{queue}}\) 和 TTFT 变差。推理性能必须拆成 queue、prefill、decode 和 post 看。

## Prefill 和 Decode 为什么要分开学

用户给模型一段输入，模型先做 prefill：把所有输入 token 过一遍 Transformer，算出每层 attention 需要的 key/value，并写进 KV cache。之后 decode 时，每生成一个新 token，就读取历史 KV，再追加这个新 token 的 KV。

| 阶段 | 更像什么 | 典型瓶颈 | 适合的优化 |
| --- | --- | --- | --- |
| prefill | 一次读完整份材料 | 长上下文 attention、大矩阵乘、KV 写入 | prefix cache、chunked prefill、长短请求分池 |
| decode | 一口一口往外说 | 小 batch、KV 读取、kernel launch、采样 | continuous batching、PagedAttention、投机解码 |

一个 80 页 PDF 问答可能输出不长，但 \(T_{\text{prefill}}\) 很大；一个短 prompt 写长文可能输入不长，但 \(T_{\text{decode}}\) 很大。新手最容易犯的错，是把两者都叫“模型慢”。

## KV Cache 是第二套显存账本

自回归 decode 不能每一步都从头重算全部历史，所以会保存历史 token 的 key/value。KV cache 大小可用数量级公式估算：

\[
M_{\text{KV}}
\approx
2 \cdot L \cdot B \cdot T \cdot H_{\text{kv}} \cdot D_{\text{head}} \cdot b
\]

<!-- [补充] -->
| 符号 | 含义 | 取值范围 |
| --- | --- | --- |
| \(M_{}\) | 公式中的变量或中间量 | 按上下文定义 |
| \(L\) | 损失或目标函数 | 实数标量，训练时按 batch 聚合 |
| \(B\) | 公式中的变量或中间量 | 按上下文定义 |
| \(T\) | 时间步、索引或计数 | 整数索引或归一化时间区间，见正文 |
| \(H_{}\) | 公式中的变量或中间量 | 按上下文定义 |
| \(D_{}\) | 公式中的变量或中间量 | 按上下文定义 |


符号逐个解释：

例如 32 层、batch 4、context 32k、GQA 下 8 个 KV heads、head dim 128、BF16 KV：

\[
2\times32\times4\times32768\times8\times128\times2
\approx 16\text{GiB}
\]

<!-- [补充] -->
| 符号 | 含义 | 取值范围 |
| --- | --- | --- |
| 公式整体 | 本公式表达的变量关系 | 按上下文定义 |


这解释了一个很反直觉的现象：权重量化以后模型也许装得下，但长上下文并发时 KV cache 仍然会先把显存吃满。

PagedAttention 图把这个问题讲得很直观：KV 不必连续摆放，而是像分页内存一样由 block table 管理。

![vLLM PagedAttention 原论文图](../assets/images/paper-figures/inference/vllm-pagedattention.png){ width="700" }

<small>图源：[Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)，Figure 4。原论文图意：PagedAttention 允许 attention 的 K/V 向量存放在非连续物理内存块中，kernel 通过 block table 找到对应 KV block 并完成注意力计算。</small>

!!! note "这张图怎么看"
    左侧是逻辑上的 token 顺序，右侧是实际显存里的物理 block。请求看到的是连续上下文，但系统内部可以把 KV 放进不连续 block，再用表查到它们。这样可以降低显存碎片和预留浪费，也让 continuous batching 更容易运行。

## Batching、Routing 和 Speculative Decoding

当许多请求同时来，系统会尝试合批。continuous batching 可以在 decode 过程中动态加入新请求：

\[
\mathcal{B}_{t+1}
=
(\mathcal{B}_t \setminus \mathcal{F}_t)\cup\mathcal{A}_t
\]

<!-- [补充] -->
| 符号 | 含义 | 取值范围 |
| --- | --- | --- |
| \(t\) | 时间步、索引或计数 | 整数索引或归一化时间区间，见正文 |


这里 \(\mathcal{B}_t\) 是第 \(t\) 步的活跃请求集合，\(\mathcal{F}_t\) 是这一轮完成的请求，\(\mathcal{A}_t\) 是新加入的请求。它能提高 GPU 利用率，但如果长短请求不分桶，也可能拉高短请求的 TTFT 和 P99。

路由则回答“这个请求该走哪条路径”。一个抽象目标可以写成：

\[
m^\star
=
\arg\min_{m\in\mathcal{M}}
\left(
\lambda_1 \text{cost}(m)
+ \lambda_2 \text{latency}(m)
- \lambda_3 \text{quality}(m,x)
\right)
\]

<!-- [补充] -->
| 符号 | 含义 | 取值范围 |
| --- | --- | --- |
| \(\lambda_1\) | 损失或目标函数 | 实数标量，训练时按 batch 聚合 |
| \(\lambda_2\) | 损失或目标函数 | 实数标量，训练时按 batch 聚合 |
| \(\lambda_3\) | 损失或目标函数 | 实数标量，训练时按 batch 聚合 |
| \(m\) | 公式中的变量或中间量 | 按上下文定义 |
| \(x\) | 样本、状态或 latent 变量 | 张量空间，形状由数据表示决定 |


其中 \(\mathcal{M}\) 是候选模型集合，\(m^\star\) 是最终选择的模型，\(x\) 是请求，\(\lambda_1,\lambda_2,\lambda_3\) 是业务给成本、延迟和质量分配的权重。这个公式不是让新手去手写路由器，而是提醒你：路由永远是在质量、成本和延迟之间做选择。

投机解码则用便宜 draft 路径先猜多个 token，再让目标模型并行验证。EAGLE 的 pipeline 图适合作为直觉入口。

![EAGLE pipeline 原论文图](../assets/images/paper-deep-dives/inference/eagle/figure-6-pipeline.png){ width="760" }

<small>图源：`EAGLE`，Figure 6。原论文图意：绿色块是 token embedding，橙色块是 feature，红框是 draft model prediction，蓝色雪花模块复用 frozen target LLM 参数。</small>

!!! note "这张图怎么看"
    先看 draft model：它用更便宜的路径提出候选 token。再看 target LLM：它不逐个慢慢生成，而是一次验证候选树。最后看 accepted prefix：只有通过验证的前缀才真正进入输出。投机解码的收益取决于接受率、验证开销、KV 分支管理和 runtime batching，不能只看 draft model 有多小。

## 成本和 SLO：什么时候算“优化成功”

推理优化不是单纯追求更快，而是让单位可接受质量的成本下降。单请求成本可先写成：

\[
C_{\text{req}}
\approx
C_{\text{queue}}
+ C_{\text{prefill}}(L_{\text{in}})
+ C_{\text{KV}}(L_{\text{ctx}}, B, q_{\text{kv}})
+ L_{\text{out}}\cdot C_{\text{decode-step}}
+ C_{\text{tool/retry}}
\]

<!-- [补充] -->
| 符号 | 含义 | 取值范围 |
| --- | --- | --- |
| \(C_{}\) | 公式中的变量或中间量 | 按上下文定义 |
| \(L_{}\) | 损失或目标函数 | 实数标量，训练时按 batch 聚合 |
| \(B\) | 公式中的变量或中间量 | 按上下文定义 |
| \(q_{}\) | 公式中的变量或中间量 | 按上下文定义 |


符号解释：

SLO 则把“体验底线”写清楚，例如 P95 TTFT < 2s、TPOT < 60ms/token、请求成功率 > 99.5%、长文档问答必须保留引用质量。没有 SLO，优化很容易变成 tokens/s 好看但用户更慢。

Low-bit survey 的 cache hierarchy 图提醒我们：低比特、cache、kernel 和硬件带宽要一起算，不能只看 checkpoint bitwidth。

![Low-bit cache hierarchy data transmission 原图](../assets/images/paper-deep-dives/inference/low-bit-llm-survey/figure-4-cache-hierarchy.png){ width="820" }

<small>图源：`A Survey of Low-bit Large Language Models`，Figure 4。原论文图意：以 NVIDIA A100 为例展示 weight 和 activation 在 host、global memory、L2、shared memory、registers 与 MatMul kernel 之间的数据传输。</small>

!!! note "这张图怎么看"
    从 host memory 到 global memory，再到 L2、shared memory、registers，数据每搬一次都有成本。量化能减少搬运量，但如果 dequant、scale 读取或 layout 转换太贵，真实 TPOT 未必下降。推理成本要看完整热路径。

## 继续阅读

| 阶段 | 先读什么 | 读完要会什么 |
| --- | --- | --- |
| 1. 建立请求生命周期 | [推理服务系统](serving-systems.md) | 能解释 queue、prefill、decode、TTFT、TPOT |
| 2. 理解执行层 | [推理运行时](serving-runtimes-vllm-sglang-and-tensorrt-llm.md)、[GPU Kernel 与 Batching](gpu-kernels-batching-and-memory-systems.md) | 知道 runtime、kernel、batching 如何影响真实延迟 |
| 3. 学缓存和加速 | [缓存、路由与投机解码](caching-routing-and-speculative.md)、[上下文压缩与 KV 内存管理](context-compression-kv-eviction-and-memory-hierarchies.md) | 能手算 KV 数量级，判断 speculative 何时失效 |
| 4. 学长上下文和复杂链路 | [解耦 Prefill 与 KV 服务](disaggregated-prefill-kv-services-and-capacity-ops.md)、[RAG、Agent 与长上下文系统](rag-agents-and-long-context-systems.md) | 能解释长文档、agent、多轮记忆为什么改变瓶颈 |
| 5. 学运营闭环 | [成本建模与 SLO 设计](cost-modeling-and-slo-design.md)、[容量规划与推理优化](optimization-checklist-and-capacity-planning.md)、[可观测性与在线评测](observability-and-online-evaluation.md) | 能按请求桶复盘成本、质量、尾延迟和回退 |

## 检查标准

读完本章最少要能回答这些问题：

1. TTFT 和 TPOT 分别受哪些阶段影响？
2. 为什么长输入主要贵在 prefill，长输出主要贵在 decode？
3. KV cache 公式里每个符号是什么意思，如何估算 GiB 数量级？
4. 为什么 batching 提升 tokens/s 时，短请求 P99 可能变差？
5. speculative decoding 为什么需要看 acceptance、verify、KV 分支和 fallback？
6. 成本模型为什么要按短问答、长文档、长输出、agent、多模态分桶？
7. 一个优化怎样才算真的成功：是平均更快，还是 SLO、质量和成本同时达标？

## 深度标杆页

| 标杆页 | 本次章节承担的角色 |
| --- | --- |
| [推理服务系统](serving-systems.md) | 最基础的请求生命周期和公式解释 |
| [GPU Kernel、Batching 与显存](gpu-kernels-batching-and-memory-systems.md) | 把 prefill/decode 落到 GPU 热路径 |
| [缓存、路由与投机解码](caching-routing-and-speculative.md) | 讲清 KV、prefix、routing 和 speculative 的耦合 |
| [上下文压缩与 KV 内存管理](context-compression-kv-eviction-and-memory-hierarchies.md) | 讲长上下文为什么是记忆管理问题 |
| [成本建模与 SLO 设计](cost-modeling-and-slo-design.md) | 把优化结果转成可验收的成本和体验目标 |

!!! tip "下一站"
    - 按导航顺序继续：[推理服务系统](serving-systems.md)。
    - 概念或符号卡住时，先查 [术语表](../glossary/index.md)，再回到当前页。
