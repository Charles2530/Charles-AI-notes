# MTP 与投机解码

`MTP`（Multi-Token Prediction）和 `speculative decoding` 经常一起出现，但它们不是同一件事。

- `MTP` 首先是**训练目标 / 模型结构设计**：让模型在一个位置上学习多个未来 token；
- `投机解码` 首先是**推理加速算法**：用便宜路径草拟，再用目标模型并行验证；
- `Self-speculative decoding` 是二者交汇的一类系统：草稿路径来自模型自身的中间层、辅助头或 MTP 模块。

这页放在训练专题里，是因为一旦决定训练 `MTP`，改变的不只是 serving 策略，还包括 loss、checkpoint 结构、评测口径和未来推理接口。推理侧实现细节可继续看 [缓存、路由与投机执行](../inference/caching-routing-and-speculative.md)。

## 概念拆分

标准自回归语言模型通常只学习 next-token prediction：

\[
\mathcal{L}_{\text{NTP}}
=
- \sum_t \log p_\theta(x_{t+1}\mid x_{\le t})
\]

`MTP` 的核心变化是让同一个位置同时监督多个未来 token：

\[
\mathcal{L}_{\text{MTP}}
=
\sum_t \sum_{k=1}^{K}
\lambda_k
\Big(
- \log p_\theta^{(k)}(x_{t+k}\mid x_{\le t})
\Big)
\]

其中 \(K\) 是预测深度，\(p_\theta^{(k)}\) 是第 \(k\) 个未来位置的预测头或预测模块，\(\lambda_k\) 是损失权重。

投机解码回答的是另一个问题：如何减少逐 token 串行解码的主模型调用次数。它通常包含三步：

1. **Draft**：便宜路径先生成若干候选 token；
2. **Verify**：目标模型一次性并行验证这些候选；
3. **Accept / Reject**：接受与目标模型一致的最长前缀，错误位置由目标模型修正。

因此两者的关系是：`MTP` 可以提供内建 draft 能力，但 speculative decoding 不一定需要 MTP；反过来，模型训练了 MTP，也不代表线上一定能提速。

## 从 NTP 到 MTP

MTP 的训练价值不只是“多预测几个 token”。它改变了 hidden state 必须承载的信息：模型不只要为 \(t+1\) 做准备，还要为更远的未来保留可解码结构。

这会带来两类潜在收益：

1. **训练收益**：更密集的监督信号、更强的前瞻性表示、更好的长程依赖建模；
2. **推理收益**：额外头或模块可以在 serving 侧成为 draft 路径，用于自投机解码。

但 MTP 也会引入额外风险：

1. 未来越远，监督越噪，过高权重可能拉坏 next-token 主质量；
2. 多头/多模块会增加 checkpoint、加载和兼容成本；
3. 训练时看到真实前缀，推理时 draft 会消费模型自己的前缀，训练-推理分布不一致会降低 acceptance；
4. 如果 runtime 不支持 draft/verify，MTP 参数只能作为训练辅助，不能自动兑现推理收益。

一个实用判断是：如果团队只能改训练或只能改 serving，而不能把两边接口一起设计，MTP 的投入产出通常会变差。

## 架构路线

外部都叫 MTP，工程实现至少有三类。

| 路线 | 做法 | 优点 | 主要风险 |
| --- | --- | --- | --- |
| 并行多头 | 共享主干，多个 head 分别预测 \(t+1,t+2,\dots,t+K\) | 结构直观，实现简单 | 远距 head 更难学，和推理递归不完全一致 |
| 串行 MTP 模块 | 每个预测深度依赖前一深度表示，形成微型因果链 | 更接近自回归过程，未来 token 之间更连贯 | 模块、KV、runtime 接口更复杂 |
| 中间层/早退草稿 | 用浅层或中间层输出作为 draft，再由完整模型验证 | 不一定增加独立小模型，KV 可能复用 | 需要训练提升中间层可解码性 |

Meta 的 `Better & Faster Large Language Models via Multi-token Prediction` 更接近“共享 trunk + 多个预测头”的并行路线。DeepSeek-V3 的 MTP 更强调串行模块：第 \(k\) 个 MTP 模块会融合前一深度表示和目标 token embedding，再通过模块产生下一深度预测，从而保留更完整的因果链。

这也是 DeepSeek 路线和简单多头路线的关键差别：它不是只在最后挂几个独立分类头，而是把多 token 预测变成单次前向里的小型自回归过程。

## 投机解码的核心机制

传统解码生成 \(N\) 个 token 通常需要主模型串行运行 \(N\) 次。投机解码利用 Transformer 可并行评分候选序列的特点，把一部分串行生成改成“草拟 + 批量验证”。

一个够用的速度模型是：

\[
\text{TPS}_{\text{spec}}
\approx
\frac{\mathbb{E}[A]}{C_{\text{draft}} + C_{\text{verify}} + C_{\text{overhead}}}
\]

\[
\text{Speedup}
\approx
\frac{\mathbb{E}[A] \cdot C_{\text{decode}}}
{C_{\text{draft}} + C_{\text{verify}} + C_{\text{overhead}}}
\]

其中 \(\mathbb{E}[A]\) 是每轮平均接受 token 数，\(C_{\text{draft}}\) 是草稿路径开销，\(C_{\text{verify}}\) 是目标模型验证开销，\(C_{\text{overhead}}\) 包括 KV 管理、调度、fallback 和 batch 组织成本。

这个公式的价值在于提醒两件事：

1. 接受 token 少，分子上不去；
2. draft、verify 或调度太贵，分母降不下来。

经典 speculative decoding 可以通过拒绝采样保持输出分布与目标模型一致，因此是无损加速算法。但工程实现里还存在近似验证、tree draft、relaxed acceptance 等变体，是否无损要看具体接受规则。

## 自投机与 Logit Lens

`Self-speculative decoding` 的目标是避免长期维护一个独立小草稿模型。草稿路径可以来自：

1. 模型自己的浅层或中间层；
2. early-exit head；
3. MTP head / MTP module；
4. 特征级 draft 模块，如 EAGLE 一类方法。

这条路线背后的一个观察来自 `Logit Lens`：很多 Transformer 中间层已经能通过最终 LM head 投射出接近最终 token 的分布。后续深层计算在不少位置并不是完全改变 token，而是在降低熵、提高置信度、修正困难位置。

因此，中间层或辅助头可以作为“便宜草稿”，完整模型作为“昂贵验证器”。这类方法的关键不是中间层是否偶尔猜对，而是：

1. 猜对的 token 是否形成足够长的可接受前缀；
2. draft 阶段的 KV / hidden state 是否能复用；
3. 早退或辅助头是否会影响主模型质量；
4. 真实 batch 和动态流量下是否仍有端到端收益。

`LayerSkip` 和 `Draft & Verify` 都属于这条思路：通过训练或解码设计，让模型自己的早期计算承担 draft 角色，再由完整模型验证。

## DeepSeek-V3 的位置

DeepSeek-V3 把 MTP 放在训练设计里，目标首先是提升训练信号密度和模型对未来 token 的建模能力。推理时有两种部署方式：

1. **丢弃 MTP 模块**：只保留 main model，像普通 next-token 模型一样部署；
2. **保留 MTP 模块**：把它作为自投机 draft 路径，用 target model 并行验证。

这解释了一个容易混淆的点：DeepSeek-V3 的 MTP 不等于线上必然启用 MTP。很多 serving 框架需要处理 dynamic batching、KV cache、prefix sharing、异构请求长度和调度开销；如果 draft/verify 路径没有和 runtime 深度集成，MTP 的理论收益可能被系统开销吃掉。

DeepSeek MTP 的工程启发是：

1. MTP 可以先作为训练目标提升数据利用率；
2. 如果推理栈成熟，再把 MTP 模块接入 self-speculative decoding；
3. 如果推理栈不支持，仍可把 MTP 模块裁掉，保留主模型部署灵活性。

这类“训练资产可选服务化”的设计，比把 MTP 直接当作单一推理加速技巧更稳。

## 评测与上线口径

MTP 和 speculative 的评测不能只看 perplexity 或平均 speedup。建议按三层闸门推进。

| 阶段 | 目标 | 必看指标 |
| --- | --- | --- |
| 离线训练 | 主质量不退化，draft 信号可用 | perplexity、任务分数、future-token accuracy、模拟 acceptance |
| 回放验证 | 真实请求分桶下收益可重复 | p50/p95/p99、tokens/s、acceptance 分布、fallback 率 |
| 小流量灰度 | 系统可运营，尾部不恶化 | 峰值流量收益、p99 波动、熔断触发、回退后等价性 |

分桶比均值更重要。至少按下面维度切：

1. 输出长度：短输出、中输出、长输出；
2. 任务类型：代码、问答、结构化生成、工具调用；
3. 解码配置：温度、top-p、重复惩罚；
4. 系统状态：batch size、队列压力、KV 命中和 prefix 复用。

上线前还要写清接口契约：

1. checkpoint 里 MTP head/module 的命名和版本；
2. draft 输出格式、verify 输入格式、回退点定义；
3. acceptance、吞吐和延迟统计口径；
4. speculative 关闭时是否严格退化为 baseline；
5. 模型版本、runtime、kernel 和配置是否绑定发布。

一个保守起步配置通常是：\(K=2\) 或 \(3\)，\(\lambda_k\) 随 \(k\) 递减，先在低温长输出任务桶验证收益，再按请求特征分层开启 speculative。

## 论文脉络

MTP 不是大模型时代才出现的新概念，它可以追溯到早期的 blockwise parallel decoding 和未来 n-gram 预测。近年的变化在于：模型足够大、推理足够贵、训练和 serving 系统足够复杂，使得“训练期未来预测 + 推理期草稿验证”重新变得有工程价值。

| 方向 | 代表工作 | 核心贡献 |
| --- | --- | --- |
| 分块并行解码 | Blockwise Parallel Decoding, NeurIPS 2018 | 训练多个辅助预测头，提出候选块并由基础模型验证 |
| 未来 n-gram 预测 | ProphetNet, ACL 2020 | Seq2Seq 预训练中同时预测未来多个 token，提高全局规划能力 |
| 经典投机解码 | Fast Inference from Transformers via Speculative Decoding, ICML 2023 | 小 draft model + 大 target model，无损拒绝采样验证 |
| 自投机解码 | LayerSkip, Draft & Verify, ACL 2024 | 用中间层/早退路径作为草稿，减少独立 draft model 依赖 |
| MTP 训练目标 | Better & Faster LLMs via Multi-token Prediction, ICML 2024 | 共享主干 + 多未来头，提升训练效率并支持自投机 |
| DeepSeek MTP | DeepSeek-V3 Technical Report, 2024 | 串行 MTP 模块，训练增强与可选 self-speculative deployment |
| 特征级草稿 | EAGLE / EAGLE-2 / EAGLE-3 | 在 feature space 预测草稿，构造动态 draft tree |
| 新型 MTP 变体 | MuToR、L-MTP | 用 register token 或 leap prediction 改善多 token 监督 |

如果把这些工作放在一起看，可以得到一个清晰判断：MTP 的长期价值不在“多几个预测头”，而在于它是否能把训练目标、模型结构、推理 runtime 和线上观测体系连成一条闭环。

## 收口结论

`MTP` 值不值得做，不取决于它是否“先进”，而取决于三件事是否同时成立：

1. 训练侧能稳定产出不伤主质量的未来预测能力；
2. 推理侧能把 acceptance 转成真实吞吐，而不是只赢理论步数；
3. 系统侧能在动态 batch、长尾请求和高峰流量下维持收益并可靠回退。

只成立其中一两项时，MTP 更像阶段性实验技巧；三者都成立时，它才是可维护的训练-推理一体化架构收益。
