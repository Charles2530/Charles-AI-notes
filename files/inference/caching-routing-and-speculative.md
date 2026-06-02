# 缓存、路由与投机解码：把昂贵计算花在值得的 token 上

缓存、路由和投机解码看起来是三类技术：一个管复用，一个管分流，一个管加速生成。但它们其实在回答同一个问题：**在线推理里，有限的 GPU 时间和显存应该花在哪里？**

缓存减少重复计算；路由让不同请求走不同成本路径；投机解码用便宜 draft 先猜，再让昂贵 target model 验证。三者都不是“平均延迟优化技巧”，而是请求生命周期里的资源分配策略。

## 先看一次请求经过哪些阶段

```text
request
  -> routing: 选择模型、服务池、工具链和上下文策略
  -> prefix / retrieval / tool cache: 复用可复用的前置计算或外部结果
  -> prefill: 处理输入 token，写入新的 KV
  -> decode: 逐 token 生成
  -> speculative draft: 便宜模型或 draft head 猜多个候选 token
  -> target verify: 目标模型并行验证候选
  -> fallback / finish: 回退、结束或继续下一轮
```

prefill 主要受输入长度影响，decode 主要受输出长度和 KV 读取影响。缓存通常压 prefill 或外部调用，投机解码主要压 decode 的逐步等待，路由则决定请求一开始走哪条成本曲线。

| 机制 | 主要节省 | 最容易出错的地方 |
| --- | --- | --- |
| Prefix cache | 重复系统 prompt、工具 schema、共享文档前缀的 prefill | prompt 不稳定、租户隔离、cache miss 放大 |
| KV cache | 避免每步重算历史 token | 显存增长、碎片、eviction 伤质量 |
| 检索/工具缓存 | 外部服务延迟和费用 | 过期证据、权限污染、错误结果复用 |
| Routing | 把贵模型留给真正需要的请求 | 错路由、升级重试、策略漂移 |
| Speculative decoding | 减少 target model 逐 token decode 次数 | acceptance 低时 P99 变差 |

## KV cache 为什么不是普通缓存

KV cache 是自回归推理的执行状态。每生成一个新 token，模型都需要读历史 key/value 来算注意力。如果没有 KV cache，每一步都要重算全部历史；有了 KV cache，decode 只需追加新 token 的 K/V 并复用过去状态。

KV 显存可以粗略估算为：

\[
M_{\text{KV}}
\approx
2 \cdot L \cdot B \cdot T \cdot H_{\text{kv}} \cdot D_{\text{head}} \cdot b
\]

这里 \(2\) 表示 K 和 V 两份缓存，\(L\) 是层数，\(B\) 是活跃请求数，\(T\) 是上下文长度，\(H_{\text{kv}}\) 是 KV head 数，\(D_{\text{head}}\) 是每个 head 维度，\(b\) 是每个元素的字节数。读这行公式时，重点是 \(B\) 和 \(T\) 都由在线流量动态决定：并发多、上下文长、输出长，KV 就会变成服务容量的主约束。

[vLLM PagedAttention](https://vllm.ai/blog/2023-06-20-vllm) 的核心贡献之一，就是把 KV cache 按 block/page 管理，减少连续大块分配造成的浪费，让不同请求的 KV 生命周期更像操作系统里的分页内存。KV cache 管理不好，模型权重再省也可能被长上下文并发挤爆。

## Prefix cache：收益来自“长且稳定”的公共前缀

Prefix cache 复用的是请求开头相同的 token，比如系统 prompt、工具 schema、固定模板、同一知识库前缀。粗略收益可以写成：

\[
\text{saved prefill tokens}
\approx
h \cdot L_{\text{prefix}} \cdot N_{\text{req}}
\]

其中 \(h\) 是命中率，\(L_{\text{prefix}}\) 是可复用前缀长度，\(N_{\text{req}}\) 是请求数。这个公式表示：高命中率不一定有高收益；命中的如果只是几十个 token，节省有限。真正值得缓存的是长且稳定的公共前缀，比如大型工具 schema、固定政策文档、共享 RAG 上下文。

Prefix cache 的失败通常不是算法问题，而是输入组织问题。模板里加入时间戳、随机 request id、用户私有字段，都会把公共前缀切断。多租户场景还要保证缓存 key 包含权限边界，不能因为复用 prefill 把某个用户的私有上下文暴露给另一个用户。

## 路由是在做成本、质量和风险的选择

路由可以抽象成在候选路径集合 \(\mathcal{M}\) 中选择一条路径：

\[
m^\star
=
\arg\min_{m\in\mathcal{M}}
\left(
\lambda_1 \mathrm{cost}(m)
+ \lambda_2 \mathrm{latency}(m)
- \lambda_3 \mathrm{quality}(m,x)
+ \lambda_4 \mathrm{risk}(m,x)
\right)
\]

这里 \(m^\star\) 是被选中的路径，\(\lambda_1\) 到 \(\lambda_4\) 是业务权重。公式读法是：低成本、低延迟、高质量、低风险不可能总是同时最大化，路由器必须根据请求 \(x\) 做取舍。

常见路由不是只有“小模型先答，大模型兜底”。更细的维度包括：

| 路由维度 | 例子 | 失败模式 |
| --- | --- | --- |
| 长度 | 短问答、32k 文档、128k 文档分池 | 长请求堵住短请求 |
| 任务 | 闲聊、代码、法律、表格、agent | 规则膨胀，策略难维护 |
| 风险 | 高风险请求走强模型、工具校验和审计 | 风险漏检造成质量事故 |
| 置信度 | 小模型先答，低置信升级 | 置信度估计不准，重试更贵 |
| 租户/SLA | 高价值客户优先或独立池 | 公平性、成本和容量争议 |

路由上线的验收不是“小模型占比上升”，而是每个请求桶的质量、延迟、成本和升级率一起改善。

## 投机解码：draft 猜，target 验

普通 decode 是目标模型一步一步生成。投机解码多加一条便宜路径：

1. draft model 或 draft head 先猜一段候选 token；
2. target model 并行验证这些候选；
3. 接受连续通过验证的前缀；
4. 第一个不通过的位置由 target 修正；
5. 继续下一轮。

一个粗略速度模型是：

\[
\text{Speedup}
\approx
\frac{\mathbb{E}[A]\cdot C_{\text{target-step}}}
{C_{\text{draft}} + C_{\text{verify}} + C_{\text{overhead}}}
\]

这里 \(\mathbb{E}[A]\) 是平均每轮接受 token 数，\(C_{\text{target-step}}\) 是目标模型单步 decode 成本，分母是 draft、verify 和系统额外开销。读这行公式时，最重要的是 acceptance：draft 猜得越准，一次 target verify 换来的 accepted tokens 越多；acceptance 低，投机解码会把额外 draft 和验证开销加到尾延迟上。

![SpecInfer workflow 原论文图](../assets/images/paper-figures/inference/specinfer-workflow.png){ width="780" }

<small>图源：[SpecInfer: Accelerating Generative Large Language Model Serving with Speculative Inference and Token Tree Verification](https://arxiv.org/abs/2305.09781)，Figure 1。原图表达：draft 模型生成 token tree，target LLM 用 tree verification 并行验证多个候选 token。本站读法：真正节省时间的是被 target 连续接受的前缀，不是 draft 生成得越多越好。</small>

## EAGLE 系列的启发：draft 质量比 draft 数量更重要

EAGLE 系列适合用来理解投机解码的系统取舍。

| 方法 | 它关注什么 | 工程启发 |
| --- | --- | --- |
| EAGLE | draft 可以预测 target feature，而不只是独立小模型猜 token | draft interface 会影响 acceptance 和开销 |
| EAGLE-2 | draft tree 应按 confidence 动态分配 | 不同请求、不同位置不该用同样候选预算 |
| EAGLE-3 | 训练时模拟测试时递归 draft 分布 | draft 的 train-inference mismatch 会伤多步接受率 |

![EAGLE pipeline 原论文图](../assets/images/paper-deep-dives/inference/eagle/figure-6-pipeline.png){ width="760" }

<small>图源：[EAGLE](../paper-deep-dives/inference/eagle.md)，Figure 6。原图表达：draft 路径围绕 target LLM 的 feature 做预测，并复用 frozen target LLM 的部分模块。本站读法：draft 不只是“更小的 LLM”，也可以是更贴近 target 内部表示的候选生成器。</small>

![EAGLE-2 dynamic draft tree 原论文图](../assets/images/paper-deep-dives/inference/eagle-2/figure-7-eagle2-dynamic-tree.png){ width="760" }

<small>图源：[EAGLE-2](../paper-deep-dives/inference/eagle-2.md)，Figure 7。原图表达：根据 confidence 和节点 value 动态扩展 draft tree，再构造 tree attention mask 验证候选。本站读法：投机预算要给更可能被接受的位置，而不是平均撒。</small>

![EAGLE vs EAGLE-3 acceptance rate 原论文图](../assets/images/paper-deep-dives/inference/eagle-3/figure-7-acceptance-rate.png){ width="760" }

<small>图源：[EAGLE-3](../paper-deep-dives/inference/eagle-3.md)，Figure 7。原图表达：EAGLE-3 的 draft 接受率随递归步数增加仍较稳定。本站读法：如果接受率随深度快速下降，投机收益会被回滚和验证开销吃掉。</small>

## 三者会互相影响

缓存、路由和投机解码不能分别报平均收益，因为它们会改变同一批请求的生命周期。

| 组合 | 可能收益 | 可能代价 |
| --- | --- | --- |
| routing + prefix cache | 相似请求进同池，公共前缀更稳定 | 路由太碎造成资源碎片 |
| KV cache + speculative | 长输出高 acceptance 时 TPOT 降低 | draft 分支增加 KV 生命周期和临时显存 |
| routing + speculative | 只给高 acceptance 请求开投机 | acceptance 预测错会放大 P99 |
| cache + RAG | 重复检索和共享文档前缀更便宜 | 过期证据或权限污染 |
| KV eviction + routing | 长任务放专门池，保护短任务 | 淘汰错误会让多轮任务失忆 |

一个很常见的事故是：上线投机解码后平均 TPOT 降了，但 P99 变差。排查发现普通问答 acceptance 很高，创意写作和多工具 agent acceptance 很低；高 acceptance 桶拉低平均值，低 acceptance 桶承担了 draft、verify、回滚和更长 KV 生命周期。正确修复不是关掉所有投机，而是按任务、温度、输出长度和工具状态预测 acceptance，低 acceptance 桶降级或关闭。

## 观测指标应该按请求桶拆

| 机制 | 至少记录 |
| --- | --- |
| Prefix cache | hit rate、saved prefill tokens、miss penalty、tenant boundary |
| KV cache | KV GiB、page usage、fragmentation、eviction、branch count |
| Routing | route distribution、upgrade rate、fallback rate、per-route quality |
| Speculative decoding | accepted length、rejected tokens、verify time、fallback rate |
| End-to-end | TTFT、TPOT、P95/P99、cost/request、quality regression |

只看平均延迟会掩盖问题。短输出请求、长文档请求、agent 请求、创意写作、高温采样、多模态请求，都应该分桶看，因为它们的 prefill、decode、acceptance 和 cache 生命周期完全不同。

## 外部精读

- [vLLM PagedAttention blog](https://vllm.ai/blog/2023-06-20-vllm)：理解 KV cache 为什么需要分页式管理。
- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)：理解 draft-verify 为什么可以保持目标模型分布。
- [SpecInfer](https://arxiv.org/abs/2305.09781)：理解 token tree verification。
- [EAGLE 论文页](../paper-deep-dives/inference/eagle.md)：看 feature-level draft 的设计动机。

## 本页结论

缓存、路由和投机解码的共同目标，是把昂贵 target model、KV 显存和逐 token decode 步数用在最值得的请求上。上线前不要只问“平均快了多少”，而要问：节省的是 prefill 还是 decode，哪个请求桶收益最大，低 acceptance 请求是否拖垮 P99，cache 是否越过权限边界，KV 生命周期是否仍可控，以及失败时能否干净回退。
