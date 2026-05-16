# GPU Kernel、Batching 与显存

这一页把 [推理服务系统](serving-systems.md) 里的 queue、prefill、decode 和 KV cache 落到 GPU 执行层。你不需要先会写 CUDA，但要能判断：慢是因为 GPU 算不过来，还是因为 batch 太碎、KV 访存太重、量化 kernel 没命中，或者调度让用户等太久。

!!! note "初学者先抓住"
    训练通常 batch 大、shape 稳定，GPU 容易吃满；在线推理尤其是 decode，batch 动态变化、每步只生成少量 token、KV cache 反复读取，更容易被显存带宽、kernel launch 和调度限制。

!!! example "有趣例子：自助餐和点菜"
    Prefill 像一次做一大盘自助餐，适合批量处理；decode 像顾客一口一口点菜，每一步都要查上一口吃了什么。GPU 对这两种工作负载的表现完全不同。

## 1. 从公式回到 GPU 时间线

请求总时延仍然先拆成：

\[
T_{\text{req}}
=
T_{\text{queue}}
+ T_{\text{prefill}}
+ T_{\text{decode}}
+ T_{\text{post}}
+ T_{\text{network}}
\]

GPU 页面重点看中间两段：

| 阶段 | GPU 上主要发生什么 | 常见瓶颈 |
| --- | --- | --- |
| prefill | 大块 GEMM、长上下文 attention、写 KV | Tensor Core 利用率、attention kernel、HBM 带宽 |
| decode | 小步 GEMM、读历史 KV、采样和后处理 | KV 访存、batch 太小、kernel launch、shape 碎 |

TTFT 通常受 \(T_{\text{queue}} + T_{\text{prefill}}\) 影响很大；TPOT 更接近 decode 稳态。若只看全局 tokens/s，就容易把 prefill 问题错当成 decode 问题，或把调度问题错当成 kernel 问题。

## 2. Prefill 为什么更像训练

Prefill 一次处理所有输入 token。输入长度为 \(L_{\text{in}}\)，batch 为 \(B\) 时，很多矩阵乘的形状比较大，GPU 更容易发挥 Tensor Core 吞吐。

常见优化包括：

| 方法 | 直觉 | 风险 |
| --- | --- | --- |
| 更大的 prefill batch | 合并更多输入，摊薄启动和调度开销 | 新请求等待更久，TTFT 变差 |
| FlashAttention / 高效 attention | 减少 attention 中间矩阵读写 | 需要匹配 dtype、shape 和 mask |
| prefix cache | 复用系统 prompt 或共享前缀 | prompt 版本和权限隔离要做好 |
| chunked prefill | 长输入切块，避免一次阻塞 decode | 调度更复杂，可能引入额外等待 |

Prefill 最常见的错觉是“只要 GPU 利用率高就好”。如果长文档 prefill 抢占短聊天 decode，GPU 看起来很忙，用户却会觉得系统卡住。

## 3. Decode 为什么更难跑满 GPU

Decode 每一步通常只生成一个 token。第 \(i\) 步成本可以粗略理解为：

\[
T_{\text{step}}^{(i)}
\approx
T_{\text{weight-read}}
+ T_{\text{KV-read}}(T_i)
+ T_{\text{matmul}}
+ T_{\text{sampling}}
+ T_{\text{runtime-overhead}}
\]

| 符号 | 含义 |
| --- | --- |
| \(T_i\) | 第 \(i\) 步时已有上下文长度 |
| \(T_{\text{weight-read}}\) | 读模型权重的时间 |
| \(T_{\text{KV-read}}\) | 读历史 KV cache 的时间，随上下文增长 |
| \(T_{\text{matmul}}\) | 本步矩阵乘计算 |
| \(T_{\text{sampling}}\) | logits 后处理、温度、top-p、结构化约束 |
| \(T_{\text{runtime-overhead}}\) | 调度、kernel launch、page table、fallback 等开销 |

decode 的难点不是某一步公式复杂，而是它要重复很多次，并且每一步的活跃请求集合都可能变化。

## 4. Batching：吞吐和等待时间一起变

Continuous batching 的核心公式是：

\[
\mathcal{B}_{t+1}
=
(\mathcal{B}_t \setminus \mathcal{F}_t)
\cup
\mathcal{A}_t
\]

| 符号 | 含义 |
| --- | --- |
| \(\mathcal{B}_t\) | 第 \(t\) 步正在一起 decode 的请求集合 |
| \(\mathcal{F}_t\) | 本轮完成并退出的请求 |
| \(\mathcal{A}_t\) | 新加入的请求 |

batching 的收益和风险：

| 目标 | 更大 batch 可能带来什么 | 需要同时监控什么 |
| --- | --- | --- |
| 吞吐 | GPU 更少空转，tokens/s 上升 | queue time 是否上升 |
| TTFT | 有时可复用批处理，但也可能等 batch | P95/P99 TTFT |
| TPOT | decode kernel 更饱满 | 长短请求混合后的 per-bucket TPOT |
| 公平性 | 高吞吐池更便宜 | 短请求是否被长请求拖住 |

因此 batching 参数不能只按全局 tokens/s 调。在线服务通常要按长度、SLO、租户、模型、LoRA、采样参数分桶，让相似请求更容易合并。

## 5. KV Cache 是显存系统的核心

KV cache 数量级：

\[
M_{\text{KV}}
\approx
2 \cdot L \cdot B \cdot T \cdot H_{\text{kv}} \cdot D_{\text{head}} \cdot b
\]

这个公式在 GPU 页面有三个含义：

1. \(T\) 越长，每个 decode step 要读的历史越多；
2. \(B\) 越大，同时常驻的 KV 越多；
3. \(b\) 从 BF16 降到 INT8 可以省显存，但需要看 dequant 和 kernel 支持。

Paged KV 把 cache 拆成 block/page 管理，减少连续显存需求，但会引入地址间接和 page locality 问题。监控时不能只看显存总量，还要看 free page 分布、fragmentation、page reuse、eviction 和 page miss。

![vLLM logical and physical block table 原论文图](../assets/images/paper-figures/inference/vllm-logical-physical-block-table.png){ width="760" }

<small>图源：[Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)，Figure 5。原论文图意：vLLM 用 block table 把 logical KV blocks 翻译到 physical KV blocks，使请求的 KV cache 不必连续存放。</small>

!!! note "这张图怎么看"
    logical block 是请求语义上的上下文顺序，physical block 是显存里的实际位置。block table 负责翻译。这样系统可以灵活复用和回收 KV，但 decode attention kernel 每一步都要按表找到历史 KV，因此 page 布局和局部性会影响 TPOT。

## 6. Roofline 直觉：算力不等于速度

GPU 性能常用 roofline 直觉理解：

\[
\text{Performance}
\le
\min(
\text{Peak FLOPs},
\text{Bandwidth}\times\text{Arithmetic Intensity}
)
\]

| 符号 | 含义 |
| --- | --- |
| Peak FLOPs | GPU 理论峰值计算能力 |
| Bandwidth | 显存带宽，如 HBM 读写速度 |
| Arithmetic Intensity | 每搬运 1 byte 数据能做多少计算 |

如果一个操作搬很多数据但计算很少，它就更容易 memory-bound；如果计算量很大、数据复用好，它才更可能 compute-bound。推理 decode 里 KV 读取、RMSNorm、embedding gather、dequant 和 logits 后处理经常受带宽或 launch 开销限制。

## 7. 量化为什么不等于一定更快

Low-bit survey 的数据传输图很适合说明这个坑：低比特会减少某些数据搬运，但也可能增加 dequant、scale 读取、layout 转换和不友好 shape 的开销。

![Low-bit cache hierarchy data transmission 原图](../assets/images/paper-deep-dives/inference/low-bit-llm-survey/figure-4-cache-hierarchy.png){ width="860" }

<small>图源：`A Survey of Low-bit Large Language Models`，Figure 4。原论文图意：以 NVIDIA A100 为例展示 weight 和 activation 在 host、global memory、L2、shared memory、registers 与 MatMul kernel 之间的数据传输。</small>

!!! note "这张图怎么看"
    顺着箭头看数据在哪里移动：host、global memory、L2、shared memory、registers。推理慢时不一定是矩阵乘不够快，也可能是数据在层级之间搬得太多。量化只有在 kernel 能把低比特存储、dequant 和 matmul 高效融合时，才更可能转成真实加速。

KV cache 量化也一样。它能减少 KV 存储和读取，但 attention 前往往需要 dequant：

![KV cache quantization 原图](../assets/images/paper-deep-dives/inference/low-bit-llm-survey/figure-6-kv-cache.png){ width="820" }

<small>图源：`A Survey of Low-bit Large Language Models`，Figure 6。原论文图意：展示 quantized KV cache 如何减少缓存存储和数据传输，并在 attention forward 前 dequantize。</small>

!!! note "这张图怎么看"
    左边是压缩后的 KV 存储，右边是 attention 计算前的恢复路径。收益来自少存少搬，代价来自 dequant 和可能的精度损失。长上下文服务尤其要用长任务质量回归验证，不能只看短 benchmark。

## 8. Kernel 热点怎么分

| 热点 | 出现在哪里 | 看什么指标 |
| --- | --- | --- |
| Prefill attention | 长输入、RAG、PDF | prefill time、HBM bandwidth、attention kernel time |
| Decode attention | 长输出、长会话 | TPOT、KV read bandwidth、page miss |
| GEMM / quant GEMM | 权重计算 | Tensor Core util、kernel hit rate、dtype |
| Fusion | 小算子密集路径 | kernel launch count、memory round trip |
| Sampling / logits | 大 vocab、结构化输出 | logits processor time、CPU/GPU sync |
| Communication | TP/PP/EP/MoE | all-reduce/all-to-all time、rank imbalance |

新手排查顺序建议：

1. 先按 queue / prefill / decode / post 拆时延；
2. prefill 慢，先看输入长度、prefix cache、attention kernel；
3. decode 慢，先看 batch、KV cache、page locality、sampling；
4. 显存紧，先看 KV 而不是只看权重；
5. tokens/s 好但 P99 差，先看分桶和调度；
6. 只有确认热路径在 kernel 上，再考虑写新 kernel 或换 runtime。

## 9. 真实排查案例：Batch 变大后体验下降

**症状**：连续批处理参数调大后，tokens/s 提升 22%，但聊天产品 TTFT P95 从 1.1s 涨到 2.0s。

**关键指标**：实际 batch size 变大，GPU utilization 更高；queue time 和 prefill waiting time 同时上升；短请求被长文档请求拖住。

**判断**：GPU 层吞吐更好，但产品层 SLO 变差。问题不是 kernel 慢，而是 batching 策略没有区分短交互和长文档。

**修复**：按输入长度和 SLO 分队列；短请求设置更小 max wait；长文档进入吞吐池；灰度指标同时看 tokens/s、TTFT、TPOT 和 P99。

## 10. 小结

GPU 推理优化不是微基准游戏。真正要改善线上体验，需要同时看请求分桶、queue、prefill、decode、KV 生命周期、kernel 热点和 SLO。一个优化只有在真实请求桶里同时改善成本、延迟和质量，才算真的有效。
