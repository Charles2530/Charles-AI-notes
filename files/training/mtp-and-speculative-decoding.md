# 多token预测怎样变成可验证的生成

模型看到前缀“今天下”时，普通下一词目标预测第一个后续token。MTP可以增加更远位置的训练目标，投机解码则让便宜模型先提出一段候选，再由目标模型验证。前者改变学习信号，后者改变推理执行，两者需要明确的接口才能结合。

## 两个预测头各看到了什么

设训练序列为 `[A,B,C,D]`。普通目标在位置A预测B，在B预测C。一个额外预测头可以要求位置A预测C，但这不自动等于先采样B、再以采样出的B为条件预测C。

若第二个头只看原前缀，它学习的是更远边缘预测；若逐层条件化到前一个目标或预测，则条件信息不同。阅读MTP架构时先列每个头的输入与标签，再讨论是否能直接提出联合候选。

多头loss还会共同更新共享表示，其辅助训练收益与用于draft的推理收益是两个实验问题。

![DeepSeek-V3 MTP module](../assets/images/paper-deep-dives/technical-reports/deepseek-v3/figure-3-mtp.png){ width="820" }

<small>图源：[DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)，Figure 3。原图表达 DeepSeek-V3 的串行 MTP 模块如何融合 hidden state 与未来 token embedding；读图重点是区分“训练时的多 token 监督”和“推理时的 draft/verify 接口”。它不能证明线上一定启用 MTP，也不能替代 acceptance rate 与端到端 latency 的验证。</small>

## 投机一轮先产生候选再并行核验

设draft分布为q、目标分布为p。Draft顺序产生K个候选，目标模型通过因果前向计算各候选位置所需的条件分布。核验从第一个候选依次进行，只有前面的候选被接受，后面对应条件才仍然成立。

在标准随机投机采样中，候选x的接受概率为 $\min(1,p(x)/q(x))$。拒绝后从与 $(p-q)_+$ 成正比的残差分布采样修正token，并停止提交后续draft候选。

## 两个类别验证目标分布

令 $p=(0.6,0.4)$、$q=(0.8,0.2)$。第一类被draft提出的概率0.8，接受概率0.75，所以最终通过接受得到它的概率0.6。第二类接受概率1，通过接受贡献0.2。

剩余拒绝概率为0.2，残差只在第二类上为正，因此修正再贡献0.2。最终分布正是(0.6,0.4)。这个例子说明残差重采样的必要性，拒绝后直接随意换一个token并不保分布。

推导条件与完整算法见[Speculative Decoding](https://arxiv.org/abs/2211.17192)。确定性贪心验证、树状候选与近似策略有各自规则，不应统一套用这个接受式。

## 速度分子是实际提交的token

在简化假设下，各位置独立以a接受，K个draft候选加修正或bonus token，期望提交量为

$$
E[N]=1+a+\cdots+a^K.
$$

K=3、a=0.8时为2.952。每轮成本包含draft、目标核验、采样与状态管理；即使接受率高，draft过贵或目标核验效率差，仍可能不加速。

实际接受概率随位置和上下文变化，以上几何和只是直觉模型。测量应直接统计提交token数和整轮耗时，不能只用被接受draft数量作吞吐分子。

## 拒绝后KV状态也必须回到正确前缀

目标模型可能已经为整段候选分配或计算临时KV。后半段被拒绝后，这些状态不能当成已提交上下文继续使用。vLLM调度代码在处理投机结果时相应调整已计算token状态，具体异步路径还要考虑请求结束与取消。

MTP能否作为有效draft，最终取决于候选质量、额外计算和状态接口。训练loss下降、draft准确率提高与端到端服务更快，需分别验证。


## 继续读

继续看[缓存、路由与投机解码](../inference/caching-routing-and-speculative.md)和[推理成本模型](../inference/cost-modeling-and-slo-design.md)。

## 对照源码与延伸阅读

- [投机输出处理与computed token回退](https://github.com/vllm-project/vllm/blob/fa3b9d1377219247060d3c8b49eae199c9e7e154/vllm/v1/core/sched/scheduler.py)
- [普通下一词目标的对照实现](https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/model.py)

## 把接受率和端到端速度拆开测

一次投机迭代至少记录以下事件：draft 开始与结束、目标模型验证开始与结束、接受 token 数、拒绝位置、修正采样、KV 回滚和最终提交时间。接受率可以按位置统计：

\[
\hat a_j=\frac{\text{第 }j\text{ 个候选被接受的次数}}{\text{第 }j\text{ 个候选出现的次数}}.
\]

平均接受率高，不代表高位置候选经常被接受，也不代表每轮提交 token 数增加。服务报告应同时给出 accepted tokens/round、target forwards/request、tokens/s、TTFT、TPOT 和 p95/p99。

[vLLM Benchmark CLI](https://docs.vllm.ai/en/latest/benchmarking/cli/)和其[指标说明](https://docs.vllm.ai/en/stable/design/metrics/)区分了 TTFT、inter-token latency、request-level TPOT 与端到端时延。投机实验应沿用这些定义，避免把一次流式输出间隔误写成整请求 TPOT。

## 公平基线需要固定输出行为

比较普通解码、draft/verify 和 MTP 时，固定 tokenizer、prompt 集、最大新 token、停止条件、采样温度、top-p、并发模型和硬件。若投机配置更早遇到 EOS 或改变了采样分布，应先检查输出质量与长度，再解释速度。

建议至少做三组：

| 组别 | 目的 |
| --- | --- |
| 目标模型普通解码 | 得到质量、长度和延迟基线 |
| 相同目标模型的投机执行 | 隔离调度、KV 与验证开销 |
| 训练过的 MTP/draft | 测量候选质量带来的额外收益 |

如果第二组已经变慢，第三组即使接受率更高也未必值得部署；如果第二组变快但质量变化，先检查采样语义是否仍然等价。

## MTP训练要做可归因消融

MTP 的辅助 loss 可能改善共享 hidden state，也可能只是增加训练计算。至少比较普通 next-token、加入 MTP 但不启用推理、加入 MTP 并启用投机三种设置，固定总训练 token 或明确说明增加的 FLOPs。记录主任务 perplexity、各预测头 loss、draft 接受率和服务吞吐，不能只看最终 loss。

对训练数据按长短、代码/自然语言、重复前缀和结构化输出分桶。某个头在平均数据上表现良好，却可能在长上下文或工具调用位置失效；这些位置正是服务端最可能受益或受损的地方。

## 自测：拒绝后哪些 token 可以提交

给出一个长度为 4 的候选，其中第 3 个 token 被拒绝。画出目标模型已经计算的条件、可以保留的前缀 KV、必须丢弃的后缀 KV，以及修正采样产生的新 token。再说明为什么不能把第 4 个候选直接接在修正 token 后面。这个图比记住接受公式更能避免实现中的状态污染。
