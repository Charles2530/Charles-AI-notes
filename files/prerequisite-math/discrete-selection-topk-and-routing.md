# 离散选择、Top-k 与稀疏路由

这篇回答一个问题：argmax、top-k、router、gating、expert load、entropy 和 load balancing 为什么会出现在 MoE、大模型解码、检索、候选排序和稀疏注意力论文里。

很多模型不是对所有候选都平均用力，而是在一组候选中选少数几个：选下一个 token、选 top-k expert、选检索文档、选候选轨迹、选需要扩展的 draft tree 节点。数学上，这些问题都可以先看成“打分、选择、归一化、再分配预算”。

## 打分：先把候选变成可比较的数字

给定候选集合 \(\{1,\ldots,n\}\)，模型通常先给每个候选一个 score：

\[
s_i=f_\theta(x,i)
\]

这里 \(x\) 是上下文，\(i\) 是候选编号。score 可以来自 logits、router MLP、retriever 相似度、value head 或 reward model。

score 的作用不是直接说明“真理”，而是提供排序依据。后面所有选择错误，都可能来自 score 本身不准，也可能来自选择规则不适合任务。

## Argmax 和 Top-k：从连续分数到离散选择

最硬的选择是 argmax：

\[
i^*=\arg\max_i s_i
\]

它只保留分数最高的一个候选。top-k 则保留前 \(k\) 个：

\[
S_k=\operatorname{TopK}(s_1,\ldots,s_n)
\]

语言解码里的 top-k、MoE router 里的 top-k experts、候选动作排序里的 top-k plans，都是这类操作。它们的共同代价是：被排除的候选不再参与后续计算，因此早期 score 错误会变成不可恢复的剪枝。

## Softmax gating：把分数变成权重

如果不想只做硬选择，可以先把 score 转成概率权重：

\[
p_i
=
\frac{\exp(s_i)}
{\sum_j\exp(s_j)}
\]

MoE router 常在 top-k 之后只对被选中的 expert 重新归一化：

\[
\tilde p_i
=
\frac{\exp(s_i)}
{\sum_{j\in S_k}\exp(s_j)},
\quad i\in S_k
\]

然后把 expert 输出加权合并：

\[
y
=
\sum_{i\in S_k}\tilde p_i E_i(x)
\]

这里 \(E_i\) 是第 \(i\) 个 expert。读这行式子时，重点是稀疏性：总共有很多 expert，但每个 token 只激活少数几个。

## 负载均衡：不能让所有 token 都挤向少数专家

如果 router 总是选择同几个 expert，MoE 会出现 hot expert：少数专家过载，其他专家闲置。一个简单负载向量可以写成：

\[
f_i
=
\frac{\text{tokens routed to expert }i}
{\text{total routed tokens}}
\]

理想状态下，\(f_i\) 不必完全相等，但不能严重塌缩。负载均衡 loss 的不同论文写法很多，核心都是约束 router 分布和实际 token 分配不要太偏。

![DeepSeek expert load](../assets/images/paper-deep-dives/technical-reports/deepseek-v3/figure-9-expert-load.png){ width="820" }

<small>图源：`DeepSeek-V3` 技术报告相关图，本站已本地化于论文专题。原图展示不同设置下 expert load 的热力图。本站在这里用它说明：稀疏路由不只是选 top-k，还必须监控专家负载是否均衡。</small>

## Entropy：选择分布有多尖

router 或采样分布的熵可以写成：

\[
H(p)
=
-\sum_i p_i\log p_i
\]

低熵表示分布很尖，模型强烈偏向少数候选；高熵表示分布更平，候选更分散。

这在两个地方很重要。第一，生成解码里，低熵上下文更容易被 draft model 猜中，高熵上下文需要更大候选预算。第二，MoE 里，router 熵过低可能导致专家塌缩；熵过高则可能说明路由没有学出清晰分工。

## Capacity 和丢弃：选择还受系统预算约束

理论上 top-k 选中了 expert，系统就该处理它。但训练和推理中，每个 expert 的容量可能有限。若某个 expert 被太多 token 选中，系统要么排队，要么溢出，要么使用 dropless routing 和更复杂通信。

所以稀疏选择有两层约束：

| 层次 | 问题 | 典型指标 |
| --- | --- | --- |
| 模型层 | 该选哪些候选 | accuracy、reward、acceptance、expert specialization |
| 系统层 | 选中的候选能否高效执行 | expert load、all-to-all、GEMM size、latency |

这解释了为什么 MoE 报告不能只看 total parameters 和 activated parameters。router 数学上选得漂亮，如果系统执行时 hot expert、all-to-all 或碎 GEMM 成为瓶颈，实际吞吐仍会受影响。

## 常见误读

| 误读 | 更稳的理解 |
| --- | --- |
| top-k 就是简单排序 | top-k 是不可逆剪枝，score 误差会直接影响后续计算 |
| router 概率就是专家质量 | router 概率只是当前上下文下的分配权重，不等于专家全局能力 |
| MoE 激活参数少就一定快 | 稀疏计算还要付 token dispatch、通信和负载均衡成本 |
| 熵越低越好 | 低熵可提高确定性，也可能导致探索不足或专家塌缩 |

## 读完以后怎么判断

看到 top-k、router、gating、expert load 或 entropy，先问：score 来自哪里，选择是硬的还是软的，被剪掉的候选是否还能恢复，负载是否均衡，稀疏选择省下的计算有没有被系统成本吃掉。这个问题链会支撑 DeepSeek-V3、Kimi-K2、Nemotron、Gemini、EAGLE-2、稀疏注意力和候选轨迹排序论文。

## 相关阅读与下一步

- 站内下一步：[MoE 与大模型架构基础](../foundations/moe-and-large-model-architecture-basics.md)。
- 站内下一步：[生成与解码基础](../foundations/generation-and-decoding-basics.md)。
- 论文精读：[DeepSeek-V3](../paper-deep-dives/technical-reports/deepseek-v3.md)。
- 论文精读：[EAGLE-2](../paper-deep-dives/inference/eagle-2.md)。
