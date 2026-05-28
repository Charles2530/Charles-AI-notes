# 推理服务系统

这页是推理章节的第一课。目标不是让你记住所有 serving 框架名字，而是让你能跟着一次请求，分清它什么时候在排队、什么时候在 prefill、什么时候在 decode、什么时候被 KV cache 和调度拖住。

<!-- [补充] -->
| 读者 | 建议读法 | 工程落点 |
| --- | --- | --- |
| 初学者 | 先抓“推理服务系统”要解决的问题、关键术语和最小例子；公式只看变量方向。 | 能说清输入、输出和常见失败信号。 |
| 工程师 | 直接对照结构、指标、成本和边界条件，必要时回到前置页补概念。 | 落到数据、算力、延迟、显存、评测和回滚检查。 |


!!! note "合并提示"
    **note · 读法定位**

    这页先回答“推理服务系统”在「推理」里的位置：它解决什么局部问题，依赖哪些前置，最后会影响哪类工程或研究判断。
    **前置**：先知道 transformer 解码、GPU kernel、显存和吞吐/延迟的区别。 必要时先回 [推理入口](index.md)、[基础知识](../foundations/index.md) 或 [术语表](../glossary/index.md)。
    **主线关系**：把请求生命周期、KV Cache、batching、runtime、SLO 和成本连接起来，看模型上线后每一毫秒花在哪里。

    **note · 初学者先抓住**

    训练决定模型“会不会”，推理服务系统决定模型“能不能稳定、便宜、低延迟地服务用户”。线上慢不一定是模型能力问题，可能是队列、长上下文、KV cache、batching、路由、工具或后处理中的任何一段出了问题。

    **example · 有趣例子：餐厅厨房**

    用户点菜后，先排队，再备菜，再一道道出菜。Prefill 像备菜：订单越复杂越慢；decode 像一道道出菜：每一步都依赖上一口。厨房如果为了吞吐把太多订单硬塞在一起，整体出菜量可能更高，但某位客人的第一道菜会更晚到。

<!-- [补充] -->
> **踩坑原因**：这类工程问题通常不是单点公式错，而是 shape、资源预算、并发路径和观测指标之间有一个假设没对齐；先把触发条件、监控信号和回滚边界拆开查。

## 一次请求到底经历什么

一个最小推理服务可以画成：

```text
client
  -> gateway / auth / quota
  -> scheduler queue
  -> prefill worker
  -> KV cache manager
  -> decode loop
  -> post-process / safety / tool
  -> response stream
  -> trace and metrics
```

每一段都可能成为瓶颈：

| 阶段 | 做什么 | 用户会怎么感受到 |
| --- | --- | --- |
| gateway | 鉴权、限流、解析请求 | 请求还没进模型就变慢或被拒 |
| queue | 等 GPU、等 batch、等优先级 | 首 token 迟迟不来 |
| prefill | 一次性处理输入上下文 | 长文档、长历史特别慢 |
| KV cache | 保存历史 token 的 K/V | 并发一高显存吃紧 |
| decode | 逐 token 生成输出 | 流式输出卡顿、TPOT 变大 |
| post | detokenize、过滤、工具编排 | 模型算完后仍然返回慢 |
| trace | 记录阶段耗时和质量指标 | 没有它就很难排障 |

## 端到端时延公式

一次请求总时延可以先写成：

\[
T_{\text{req}}
=
T_{\text{queue}}
+ T_{\text{prefill}}
+ T_{\text{decode}}
+ T_{\text{post}}
+ T_{\text{network}}
\]

<!-- [补充：符号说明] -->
| 符号 | 含义 | 取值范围 / 常见值 |
| --- | --- | --- |
| \(T_{\text{req}}\) | 请求总耗时 | end-to-end latency |
| \(T_{\text{queue}}\) | 排队耗时 | queue time、admission wait |
| \(T_{\text{prefill}}\) | 输入编码耗时 | prefill time、prefill tokens/s |
| \(T_{\text{decode}}\) | 输出生成耗时 | TPOT、decode tokens/s |
| \(T_{\text{post}}\) | 后处理耗时 | post latency |
| \(T_{\text{network}}\) | 网络耗时 | client/server transfer |

TTFT 主要由 \(T_{\text{queue}}\)、\(T_{\text{prefill}}\) 和首个 decode step 决定：

\[
\text{TTFT}
\approx
T_{\text{queue}}
+ T_{\text{prefill}}
+ T_{\text{first-step}}
\]

<!-- [补充：符号说明] -->
| 符号 | 含义 | 取值范围 / 常见值 |
| --- | --- | --- |
| \(T_{}\) | 时间步、索引或计数 | 整数索引或归一化时间区间，见正文 |


这行式子在说：用户看到首 token 之前，主要先等排队、输入上下文 prefill，以及第一次 decode step。\(T_{\text{first-step}}\) 是首个输出 token 的解码和采样开销。

TPOT 则更接近稳定 decode 阶段每个输出 token 的间隔：

\[
\text{TPOT}
\approx
\frac{T_{\text{decode}}}{L_{\text{out}}}
\]

<!-- [补充：符号说明] -->
| 符号 | 含义 | 取值范围 / 常见值 |
| --- | --- | --- |
| \(T_{}\) | 时间步、索引或计数 | 整数索引或归一化时间区间，见正文 |
| \(L_{}\) | 损失或目标函数 | 实数标量，训练时按 batch 聚合 |


这里 \(L_{\text{out}}\) 是输出 token 数。这个近似适合先估数量级；真实系统中，decode batch 大小变化、KV page 访问、投机解码验证和采样参数都会让每步不同。

!!! warning "常见误解"
    TTFT 好不代表整段回答流畅，TPOT 好也不代表用户马上看到首 token。聊天产品通常先看 TTFT，长生成和代码补全更关心 TPOT，多步 agent 还要看整条任务完成时间。

## Prefill：首 token 前最容易被忽略的成本

<!-- [融合：算法基础] 来源：llm-algo-leetcode《17. vLLM PagedAttention | 经典推理框架: 模拟 Continuous Batching 与 PagedAttention》-->
> **算法视角**：推理服务可以拆成 prefill 的并行计算和 decode 的逐 token 循环；PagedAttention、prefix cache 与 speculative decoding 分别对应分页管理、前缀复用和 draft-verify 三类基础优化。
>
> 延伸阅读：[17. vLLM PagedAttention | 经典推理框架: 模拟 Continuous Batching 与 PagedAttention](https://datawhalechina.github.io/llm-algo-leetcode/02_PyTorch_Algorithms/17_vLLM_PagedAttention.html)

<!-- [融合：复杂度分析] 来源：llm-algo-leetcode《17. vLLM PagedAttention | 经典推理框架: 模拟 Continuous Batching 与 PagedAttention》-->
| 操作 | 时间复杂度 | 空间复杂度 | 说明 |
|------|-----------|-----------|------|
| prefill attention | $O(L_in^2 d)$ | $O(L_in d)$ KV 写入 | 长输入首先推高首 token 前成本；参考：[对应章节](https://datawhalechina.github.io/llm-algo-leetcode/02_PyTorch_Algorithms/17_vLLM_PagedAttention.html) |
| decode 单步 | $O(L_ctx d)$ | $O(L_ctx d)$ KV 读取 | 输出越长，访存循环越主导 TPOT。 |

Prefill 做的是：把输入 token 一次性送进模型，计算每层 hidden states，同时为后续 decode 写好 KV cache。输入越长，prefill 越重。

粗略地，可以把 prefill 成本写成：

\[
T_{\text{prefill}}
\approx
f(L_{\text{in}}, B_{\text{prefill}}, \text{attention kernel}, \text{cache hit})
\]

<!-- [补充：符号说明] -->
| 符号 | 含义 | 取值范围 / 常见值 |
| --- | --- | --- |
| \(T_{}\) | 时间步、索引或计数 | 整数索引或归一化时间区间，见正文 |
| \(f\) | 公式中的变量或中间量 | 按上下文定义 |
| \(L_{}\) | 损失或目标函数 | 实数标量，训练时按 batch 聚合 |
| \(B_{}\) | 公式中的变量或中间量 | 按上下文定义 |


长文档问答的典型陷阱是：用户问题只有一句，但 RAG 或附件塞进了几万 token。输出也许只有 200 token，最贵却是首 token 前的 prefill。

## Decode：为什么逐 token 生成更像服务循环

Decode 每一步只新增一个或少量 token，但要反复读取历史 KV cache 和模型权重。若输出长度为 \(L_{\text{out}}\)，可以写成：

\[
T_{\text{decode}}
\approx
\sum_{i=1}^{L_{\text{out}}}
T_{\text{step}}^{(i)}
\]

<!-- [补充：符号说明] -->
| 符号 | 含义 | 取值范围 / 常见值 |
| --- | --- | --- |
| \(T_{}\) | 时间步、索引或计数 | 整数索引或归一化时间区间，见正文 |
| \(i\) | 时间步、索引或计数 | 整数索引或归一化时间区间，见正文 |
| \(L_{}\) | 损失或目标函数 | 实数标量，训练时按 batch 聚合 |
| \(T_{}^{(i)}\) | 时间步、索引或计数 | 整数索引或归一化时间区间，见正文 |


每个 \(T_{\text{step}}^{(i)}\) 会受这些因素影响：

1. 当前活跃请求数；
2. 每个请求的上下文长度；
3. KV cache 是否连续、是否 page miss；
4. batch 中请求的长度和采样参数是否相近；
5. logits 后处理、采样、结构化输出约束；
6. 是否启用 speculative decoding。

训练时 batch 大、shape 稳，GEMM 容易吃满 GPU；在线 decode 则是动态小步循环，shape 碎、请求进出频繁、KV 访存重，所以经常被内存带宽和调度限制。

## KV Cache：服务系统的隐藏容量上限

KV cache 的数量级公式是：

\[
M_{\text{KV}}
\approx
2 \cdot L \cdot B \cdot T \cdot H_{\text{kv}} \cdot D_{\text{head}} \cdot b
\]

<!-- [补充：符号说明] -->
| 符号 | 含义 | 取值范围 / 常见值 |
| --- | --- | --- |
| \(M_{}\) | 公式中的变量或中间量 | 按上下文定义 |
| \(L\) | 损失或目标函数 | 实数标量，训练时按 batch 聚合 |
| \(B\) | 公式中的变量或中间量 | 按上下文定义 |
| \(T\) | 时间步、索引或计数 | 整数索引或归一化时间区间，见正文 |
| \(H_{}\) | 公式中的变量或中间量 | 按上下文定义 |
| \(D_{}\) | 公式中的变量或中间量 | 按上下文定义 |


例子：32 层、batch 4、context 32k、8 个 KV heads、head dim 128、BF16：

\[
2\times32\times4\times32768\times8\times128\times2
\approx 16\text{GiB}
\]

<!-- [补充：符号说明] -->
| 符号 | 含义 | 取值范围 / 常见值 |
| --- | --- | --- |
| 公式整体 | 本公式表达的变量关系 | 按上下文定义 |


如果 KV heads 从 8 变成 32，显存直接变成约 4 倍。若 context 从 32k 变成 128k，也再变 4 倍。这就是为什么“支持长上下文”必须同时说明 batch、KV 精度、KV head 结构和服务 SLO。

## Continuous Batching：吞吐和尾延迟的拉扯

静态 batch 要等一批请求凑齐再跑；continuous batching 允许 decode 过程中动态加入和移除请求：

\[
\mathcal{B}_{t+1}
=
(\mathcal{B}_t \setminus \mathcal{F}_t)
\cup
\mathcal{A}_t
\]

<!-- [补充：符号说明] -->
| 符号 | 含义 | 取值范围 / 常见值 |
| --- | --- | --- |
| \(t\) | 时间步、索引或计数 | 整数索引或归一化时间区间，见正文 |


它的好处是 GPU 不容易空转；代价是调度更复杂，长短请求混在一起时，短请求可能等长请求 prefill 或大 batch 窗口，导致 TTFT 和 P99 上升。

!!! example "排查案例：吞吐上去了，用户更慢了"
    某聊天服务把 batch window 拉大后，tokens/s 提升 20%，但短问答 TTFT P95 变差。原因不是模型变慢，而是短请求为了合批在 queue 里等更久。修复通常是按输入长度和 SLO 分队列：短交互走低等待窗口，长文档走吞吐池。

## 路由：不是所有请求都该走最大模型

路由可以按请求特征 \(\phi(x)\) 选择模型或链路：

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

<!-- [补充：符号说明] -->
| 符号 | 含义 | 取值范围 / 常见值 |
| --- | --- | --- |
| \(\lambda_1\) | 损失或目标函数 | 实数标量，训练时按 batch 聚合 |
| \(\lambda_2\) | 损失或目标函数 | 实数标量，训练时按 batch 聚合 |
| \(\lambda_3\) | 损失或目标函数 | 实数标量，训练时按 batch 聚合 |
| \(m\) | 公式中的变量或中间量 | 按上下文定义 |
| \(x\) | 样本、状态或 latent 变量 | 张量空间，形状由数据表示决定 |


实际系统中，闲聊、长文档、代码、工具 agent、多模态、高风险任务不应该默认共用一条路径。路由做得好，可以省成本；路由做坏，会让复杂任务走弱模型，或者让简单任务占用昂贵资源。

## 最小观测闭环

一个新手容易低估的事实是：没有分阶段 trace，推理优化几乎只能猜。

最小观测集应包括：

| 类别 | 必看指标 |
| --- | --- |
| 请求形态 | 输入长度、输出长度、任务类型、模型路由 |
| 队列 | queue time、admission reject、priority |
| prefill | prefill time、prefill tokens/s、prefix hit |
| decode | TPOT、decode batch、tokens/s、sampling config |
| KV | KV GiB、page usage、eviction、fragmentation |
| 尾延迟 | P50/P95/P99 TTFT、TPOT、end-to-end latency |
| 质量 | 任务成功率、引用正确率、格式错误、fallback |
| 成本 | 每请求成本、每 token 成本、每成功任务成本 |

## 四个典型请求桶

| 请求桶 | 主瓶颈 | 常见策略 |
| --- | --- | --- |
| 短聊天 | queue、首步 decode、调度等待 | 小 batch 等待窗口、短请求池、低 TTFT SLO |
| 长文档问答 | prefill、KV、检索噪声 | RAG 裁剪、prefix cache、分离 prefill、上下文压缩 |
| 长输出生成 | decode、KV 生命周期 | speculative decoding、输出长度控制、低 TPOT 路径 |
| agent / 工具链 | 多轮调用、外部工具、状态膨胀 | 工具缓存、状态结构化、端到端 trace、回放评测 |

同一个模型，在这四个请求桶里的瓶颈可能完全不同。推理服务系统的第一原则，就是不要用一个平均 tokens/s 代表所有用户体验。

## 本页结论

推理服务系统的基本功，是把“模型慢”拆成具体责任层：排队慢、prefill 慢、decode 慢、KV 爆了、batching 策略伤害尾延迟、路由不合理、工具或后处理拖慢。先画请求生命周期，再看 TTFT、TPOT、KV、P95/P99 和质量指标，最后才决定是换 runtime、改 batch、压上下文、做量化，还是接入 speculative decoding。

!!! tip "下一站"
    - 回到本专题入口：[推理](index.md)，确认这页在整条路线中的位置。
    - 按导航顺序继续：[GPU Kernel 与 Batching](gpu-kernels-batching-and-memory-systems.md)。
    - 概念或符号卡住时，先查 [术语表](../glossary/index.md)，再回到当前页。
