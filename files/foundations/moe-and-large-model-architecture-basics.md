# MoE 与大模型架构表读法

读 DeepSeek、Qwen、Kimi、Gemini、Nemotron 这类技术报告时，最容易被一张模型规模表带偏：`Total Params`、`Activated Params`、`Experts`、`Top-k`、`MLA/GQA`、`MTP`、`FP8/FP4` 全部挤在一起。真正要读懂它，不是背模型名字，而是先分清：**MoE 改的是容量和计算分配，MTP 改的是预测头，Mamba/SSM 改的是序列状态路径，低精度改的是数值和硬件执行。**

<!-- [补充] -->
| 读者 | 建议读法 | 工程落点 |
| --- | --- | --- |
| 初学者 | 先抓“MoE 与大模型架构表读法”要解决的问题、关键术语和最小例子；公式只看变量方向。 | 能说清输入、输出和常见失败信号。 |
| 工程师 | 直接对照结构、指标、成本和边界条件，必要时回到前置页补概念。 | 落到数据、算力、延迟、显存、评测和回滚检查。 |


!!! note "读法定位"
    这页先回答“MoE 与大模型架构表读法”在「基础知识」里的位置：它解决什么局部问题，依赖哪些前置，最后会影响哪类工程或研究判断。
    **前置**：先看本页要补哪一个最小概念；公式或术语卡住时回到术语表，不需要一次吃完整个数学体系。 必要时先回 [基础知识入口](index.md) 或 [术语表](../glossary/index.md)。
    **主线关系**：把符号、张量、优化、评测和运行时这些前置打稳，后面的扩散、VLM/VLA、训练与系统页才不会断层。

这一页补的是“读架构表”的前置知识。它不是部署路由教程，而是帮你在技术报告里看懂稀疏专家模型到底怎样训练、怎样算账、怎样影响推理。

<!-- [补充：初学者直觉] -->
> **一句话理解**：MoE 与大模型架构表读法 可以先当成一张入门地图：它把零散术语连成“输入、变化、输出”三件事。先不用背公式，先看清哪一步在改变信息、哪一步在改变成本。

## 一句话先立住

MoE（Mixture of Experts）的核心是：**把 Transformer block 里的 dense MLP 换成多个 expert MLP，再用 router 为每个 token 只选择少数几个 expert。**

```mermaid
flowchart LR
    X["Token hidden state<br/>h"] --> A["Attention"]
    A --> R["Router / gate"]
    R --> E1["Expert 3"]
    R --> E2["Expert 17"]
    R --> E3["Shared expert<br/>(optional)"]
    E1 --> C["Weighted combine"]
    E2 --> C
    E3 --> C
    C --> Y["Next hidden state"]
```

!!! note "读图重点"
    MoE 通常替换的是 Transformer block 中的 FFN/MLP 子层，而不是 attention 本身。一个 token 不会经过所有专家，只会经过 router 选中的少数专家；这就是为什么 MoE 可以有很大的总参数量，但每个 token 的激活计算量相对小。

## 符号卡：MoE 公式怎么读

<!-- [融合：算法基础] 来源：llm-algo-leetcode《06. MoE Router | 混合专家架构: 稀疏路由与负载均衡 (MoE)》-->
> **算法视角**：MoE router 可以看成每个 token 的 Top-K 选择问题；负载均衡 loss 则是在主任务 loss 之外约束 expert 使用分布，避免少数 expert 过热、其他 expert 闲置。
>
> 延伸阅读：[06. MoE Router | 混合专家架构: 稀疏路由与负载均衡 (MoE)](https://datawhalechina.github.io/llm-algo-leetcode/02_PyTorch_Algorithms/06_MoE_Router.html)

<!-- [融合：复杂度分析] 来源：llm-algo-leetcode《06. MoE Router | 混合专家架构: 稀疏路由与负载均衡 (MoE)》-->
| 操作 | 时间复杂度 | 空间复杂度 | 说明 |
|------|-----------|-----------|------|
| router Top-K | $O(E)$ 到 $O(E\log K)$ | $O(K)$ active expert 索引 | $E$ 为 expert 数，$K$ 为激活 expert 数；参考：[对应章节](https://datawhalechina.github.io/llm-algo-leetcode/02_PyTorch_Algorithms/06_MoE_Router.html) |
| expert 计算 | 近似随 $K$ 线性增长 | 需要 token dispatch / combine buffer | MoE 的工程难点常在稀疏调度、负载均衡和通信。 |

最小读法：router 先给 expert 打分，选 top-k 个 expert 处理当前 token，再把这些 expert 的输出按权重合回来。

## Dense 与 MoE 的差别

Dense 模型里，每个 token 都经过同一套 MLP 参数；MoE 模型里，每个 token 会被分配给少数 expert。

| 对比项 | Dense MLP | MoE MLP |
| --- | --- | --- |
| 参数使用 | 每个 token 使用同一套 FFN 参数 | 每个 token 只使用 top-k expert 和可能的 shared expert |
| 容量来源 | 增大隐藏维度、层数或 FFN 宽度 | 增加 expert 数量，让不同 token 走不同参数 |
| 单 token 计算 | 接近总 FFN 规模 | 接近被激活 expert 的规模 |
| 训练难点 | 优化稳定、显存、吞吐 | router 负载均衡、expert specialization、通信和稳定性 |
| 推理难点 | attention/KV、batch、kernel | token dispatch、all-to-all、hot expert、expert batch size |

MoE 的一个常见误读是：看到 `671B`、`1T` 之类的总参数，就把它等同于每个 token 的计算量。实际要分两层看：

- `Total Params`：模型仓库里一共存了多少参数，代表容量和存储成本。
- `Activated Params`：单个 token 前向时大约用了多少参数，更接近 per-token compute。

![DeepSeek-V3 architecture](../assets/images/paper-deep-dives/technical-reports/deepseek-v3/figure-2-architecture.png){ width="920" }

<small>图源：[DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)，Figure 2。原论文图意：DeepSeek-V3 的 Transformer block 采用 MLA 和 DeepSeekMoE；MLA 用压缩 latent 降低 KV cache，DeepSeekMoE 由 shared experts 和 routed experts 共同组成。</small>

!!! note "论文图解卡：DeepSeek-V3 的 MoE 应该先看哪里"
    **先看哪里**：先看右侧 DeepSeekMoE 放大框，再回到左侧 Transformer block。  
    **图中模块**：shared experts 是所有 token 都能走的公共能力；routed experts 是 router 按 token 选择的专家；MLA 是 attention/KV 路径的压缩，不是 MoE 本身。  
    **对应公式**：\(y=E_{\text{shared}}(h)+\sum_{i\in\operatorname{TopK}(g)}\tilde g_iE_i(h)\)。  
    **容易误读**：这张图里同时有 MLA 和 MoE。MLA 主要省 KV cache，MoE 主要改变 FFN 容量和激活计算，二者不是同一件事。

## 一个 token 怎么走过 MoE

MoE 层的计算可以写成：

\[
g = \operatorname{softmax}(W_r h)
\]

<!-- [补充：符号说明] -->
| 符号 | 含义 | 取值范围 / 常见值 |
| --- | --- | --- |
| \(g\) | return、组大小或缩放系数 | 按任务定义；组大小通常为正整数 |
| \(W_r\) | 公式中的变量或中间量 | 按上下文定义 |
| \(h\) | 公式中的变量或中间量 | 按上下文定义 |


\[
\operatorname{TopK}(g) = \{i_1, i_2, \dots, i_k\}
\]

<!-- [补充：符号说明] -->
| 符号 | 含义 | 取值范围 / 常见值 |
| --- | --- | --- |
| \(g\) | return、组大小或缩放系数 | 按任务定义；组大小通常为正整数 |
| \(i_1\) | 公式中的变量或中间量 | 按上下文定义 |
| \(i_2\) | 公式中的变量或中间量 | 按上下文定义 |
| \(i_k\) | 公式中的变量或中间量 | 按上下文定义 |


\[
y = \sum_{i \in \operatorname{TopK}(g)} \tilde g_i \cdot E_i(h)
\]

<!-- [补充：符号说明] -->
| 符号 | 含义 | 取值范围 / 常见值 |
| --- | --- | --- |
| \(y\) | 公式中的变量或中间量 | 按上下文定义 |
| \(i\) | 时间步、索引或计数 | 整数索引或归一化时间区间，见正文 |
| \(g\) | return、组大小或缩放系数 | 按任务定义；组大小通常为正整数 |
| \(g_i\) | return、组大小或缩放系数 | 按任务定义；组大小通常为正整数 |
| \(E_i\) | 公式中的变量或中间量 | 按上下文定义 |
| \(h\) | 公式中的变量或中间量 | 按上下文定义 |


其中 \(h\) 是 token 的 hidden state，\(W_r\) 是 router，\(E_i\) 是第 \(i\) 个 expert，\(\tilde g_i\) 是 top-k 后重新归一化的权重。很多现代 MoE 还会加入 shared expert：

\[
y = E_{\text{shared}}(h) + \sum_{i \in \operatorname{TopK}(g)} \tilde g_i \cdot E_i(h)
\]

<!-- [补充：符号说明] -->
| 符号 | 含义 | 取值范围 / 常见值 |
| --- | --- | --- |
| \(y\) | 公式中的变量或中间量 | 按上下文定义 |
| \(E_{}\) | 公式中的变量或中间量 | 按上下文定义 |
| \(h\) | 公式中的变量或中间量 | 按上下文定义 |
| \(i\) | 时间步、索引或计数 | 整数索引或归一化时间区间，见正文 |
| \(g\) | return、组大小或缩放系数 | 按任务定义；组大小通常为正整数 |
| \(g_i\) | return、组大小或缩放系数 | 按任务定义；组大小通常为正整数 |


```mermaid
sequenceDiagram
    participant T as Token h
    participant R as Router
    participant D as Dispatch
    participant E as Selected experts
    participant C as Combine
    T->>R: compute routing scores
    R->>D: choose top-k experts
    D->>E: group tokens by expert
    E->>C: expert MLP outputs
    C->>T: weighted sum + residual path
```

这张流程图里最容易漏掉的是 `Dispatch`：训练和推理时，token 不是在原地挨个跑 expert，而是会按 expert 分组，再发到对应设备。大规模 MoE 的系统成本，很多时候就卡在 dispatch/combine 和跨卡通信。

## 架构表先看哪几列

技术报告里的模型表不要从 benchmark 分数开始读，先把下面几列翻译成成本账。

| Field in reports | 应该怎么读 | 常见误读 |
| --- | --- | --- |
| `Total Params` | 所有 dense 权重、expert 权重、embedding 等参数总和 | 误以为每个 token 都会用到全部参数 |
| `Activated Params` | 单个 token 前向实际激活的大致参数量 | 误以为它能单独决定 latency |
| `# Experts` | 总 expert 数量，通常只在 MoE FFN 层出现 | 误以为 expert 越多一定越快 |
| `Top-k` / `Activated Experts` | 每个 token 选择几个 routed experts | 忽略 top-k 增大会增加计算和通信 |
| `Shared Experts` | 所有 token 都会经过的公共 expert | 把 shared expert 当成普通 routed expert |
| `Routed Experts` | 由 router 动态选择的 expert | 忽略 router 负载不均会造成 hot expert |
| `Expert Hidden Size` | 每个 expert MLP 的中间维度 | 只看 expert 数，不看单个 expert 的宽度 |
| `Layers` | MoE 是否每层都有，还是只在部分层替换 FFN | 误以为“MoE 模型”代表所有层都是 MoE |
| `Attention Heads` / `KV Heads` | attention 路径的计算和 KV cache 成本 | 把 MoE 的节省错误归到 attention 上 |
| `Context Length` | 长上下文会放大 attention、KV 和通信压力 | 只看参数量，不看序列长度 |
| `Precision` | BF16、FP8、FP4、NVFP4 等数值和硬件路径 | 把存储格式等同于计算格式 |
| `Parallelism` | TP、PP、DP、EP、CP 如何切分模型和序列 | 忽略 EP/all-to-all 对 MoE 吞吐的影响 |

!!! tip "最小读法"
    先问四个问题：总参数有多大？单 token 激活多少？每个 token 选几个专家？专家分布在几张卡上？这四个问题答出来后，再看 benchmark 和训练细节才不会飘。

## Router 训练到底训练什么

MoE 训练不是只把很多 MLP 拼起来。Router 决定 token 走哪些 expert，因此它同时影响模型能力、梯度分配和系统负载。

```mermaid
flowchart TB
    A["Language modeling loss<br/>or multimodal loss"] --> B["Selected experts get gradients"]
    A --> C["Router scores get gradients"]
    C --> D["Expert specialization"]
    C --> E["Load balancing"]
    E --> F["Avoid hot experts<br/>avoid dropped tokens"]
    D --> G["Capacity grows without dense compute growth"]
```

| 训练细节 | 为什么重要 | 读论文时要找什么 |
| --- | --- | --- |
| Router softmax / sigmoid | 决定 token 到 expert 的概率分配 | router logits、top-k、是否重新归一化 |
| Top-k selection | 决定单 token 计算量和 expert 覆盖 | top-1、top-2、top-8 等设置 |
| Load balancing loss | 避免所有 token 挤到少数 expert | auxiliary loss、balance coefficient、token fraction |
| Auxiliary-loss-free balancing | 用动态 bias 等方式平衡负载，减少主目标干扰 | 是否报告无辅助损失路由或 bias update |
| Capacity factor | 限制每个 expert 一步能接多少 token | overflow、dropless routing、token dropping |
| Shared expert | 承担通用模式，给 routed experts 留出专业化空间 | shared/routed expert 数量和激活方式 |
| Expert parallelism | expert 分布在不同设备上 | EP group、all-to-all、communication overlap |
| Router stability | 防止训练早期 expert collapse | router z-loss、初始化、warmup、噪声策略 |

常见的 load balancing 直觉是：如果 expert \(i\) 收到的 token 比例是 \(f_i\)，router 给它的平均概率是 \(p_i\)，那么训练会惩罚“高概率但负载不均”的分布。论文公式可能不同，但目标通常类似：

\[
\mathcal{L}_{balance} \propto N \sum_i f_i p_i
\]

<!-- [补充：符号说明] -->
| 符号 | 含义 | 取值范围 / 常见值 |
| --- | --- | --- |
| \(b\) | 公式中的变量或中间量 | 按上下文定义 |
| \(N\) | 时间步、索引或计数 | 整数索引或归一化时间区间，见正文 |
| \(f_i\) | 公式中的变量或中间量 | 按上下文定义 |
| \(p_i\) | 概率、转移分布或预测量 | 概率分布或任务定义的张量 |


这里的重点不是背公式，而是理解：**MoE 需要一套机制保证专家被充分使用，否则总参数再大也只是摆设。**

## 为什么 Active Params 仍不等于延迟

`Activated Params` 更接近单 token 计算量，但它仍然不是实际延迟。推理延迟还受下面这些因素影响：

```mermaid
flowchart LR
    A["Input tokens"] --> B["Attention / KV cache"]
    B --> C["Router"]
    C --> D["All-to-all dispatch"]
    D --> E["Expert GEMM"]
    E --> F["All-to-all combine"]
    F --> G["Next layer"]

    B -.-> H["Context length"]
    D -.-> I["Network topology"]
    E -.-> J["Expert batch size"]
    F -.-> K["Load balance"]
```

| 成本项 | 对 MoE 的影响 |
| --- | --- |
| Attention / KV cache | MoE 通常不减少 attention 的 \(L^2\) prefill 或 decode KV 读写成本 |
| All-to-all communication | expert 放在不同卡上时，token dispatch/combine 会引入通信 |
| Expert batch size | 每个 expert 收到的 token 太少，GEMM 会变碎，GPU 利用率下降 |
| Hot experts | 少数 expert 过热会拖慢整个 batch |
| Decode batch | 自回归 decode 每步 token 少，MoE 通信和调度开销更明显 |
| Quantization support | expert 权重低比特不代表 router、activation、通信都低成本 |
| Kernel fusion | router、dispatch、expert MLP、combine 是否融合会影响真实吞吐 |

所以读技术报告时，不要只看“active params 比 dense 小”。还要看作者有没有报告 tokens/s、latency、MFU、通信重叠、batch setting、context length 和硬件拓扑。

## MoE 和其他结构不要混

大模型报告常把 MoE、MTP、Mamba、MLA/GQA、Muon、FP8/FP4 放在同一段里。它们属于不同层。

![Nemotron 3 Super LatentMoE](../assets/images/paper-deep-dives/technical-reports/nemotron-3-super/figure-3-latent-moe.png){ width="880" }

<small>图源：[Nemotron 3 Super Technical Report](../paper-deep-dives/technical-reports/nemotron-3-super.md)，Figure 3。原论文图意：LatentMoE 用 latent routing 组织专家选择，并把稀疏专家能力接入大模型主干。</small>

!!! note "论文图解卡：为什么再看一张 Nemotron 图"
    **先看哪里**：先找 router 或 latent routing 的位置，再看 token/hidden state 如何进入专家。  
    **图中模块**：不同报告里的 MoE 画法不完全一样，但核心总是 token 表示、router、专家计算和 combine。  
    **对应问题**：它帮助你读懂“专家怎么被选中”，而不是只看模型表里的专家数量。  
    **容易误读**：不同模型的 MoE 细节会变化，不能把某一张图里的 routing 机制直接套到所有 MoE。

| 名词 | 改的是哪一层 | 主要解决什么 | 不要误解成 |
| --- | --- | --- | --- |
| MoE | FFN/MLP 容量与稀疏激活 | 用更大总容量服务不同 token | 直接减少 attention 或 KV cache |
| MTP | 预测头和训练目标 | 训练模型预测未来多个 token，服务投机或规划信号 | 一种 MoE 路由方法 |
| Mamba / SSM | 序列建模路径 | 用 recurrent state 降低长序列状态成本 | 一种专家模型 |
| GQA / MLA | Attention 和 KV 表示 | 降低 KV cache 或 attention 表示成本 | MoE 的替代品 |
| FP8 / FP4 / NVFP4 | 数值格式与硬件执行 | 降低显存、带宽或提升 GEMM 吞吐 | 自动提升模型能力 |
| Muon / MuonClip | 优化器更新几何 | 稳定或提升大规模预训练效率 | 推理加速技术 |
| EP / all-to-all | 分布式执行 | 把 expert 分到多卡并完成 token 交换 | 模型结构本身 |

如果一篇报告说“模型是 Mamba-MoE、带 shared-weight MTP、支持 FP8/NVFP4”，可以按这条路径拆：

```mermaid
flowchart TB
    A["Backbone path<br/>Attention / Mamba / Hybrid"] --> B["Capacity path<br/>Dense FFN or MoE"]
    B --> C["Prediction path<br/>LM head / MTP head"]
    C --> D["Training path<br/>optimizer / data / precision"]
    D --> E["Serving path<br/>KV / routing / quant / parallelism"]
```

## 训练相关细节怎么补读

MoE 技术报告里和训练最相关的不是“用了多少专家”本身，而是下面这条链：

\[
\text{data mixture} \rightarrow \text{router distribution} \rightarrow \text{expert specialization} \rightarrow \text{load balance} \rightarrow \text{system throughput}
\]

<!-- [补充：符号说明] -->
| 符号 | 含义 | 取值范围 / 常见值 |
| --- | --- | --- |
| 公式整体 | 本公式表达的变量关系 | 按上下文定义 |


| 读训练部分时找什么 | 为什么会影响结论 |
| --- | --- |
| 数据 mixture 是否分阶段变化 | 不同语言、代码、数学、多模态 token 会改变 expert 专业化 |
| Router 是否有 warmup 或特殊初始化 | 训练早期路由塌缩会让少数 expert 过载 |
| Balance loss 或 bias 策略 | 影响主任务 loss 和系统负载之间的权衡 |
| 是否使用 dropless routing | dropping token 会改变训练信号，也会影响长尾样本 |
| Expert 并行是否和数据并行/张量并行组合 | 训练吞吐取决于通信和计算是否能重叠 |
| 是否报告 loss spike、NaN、回滚策略 | 大规模 MoE 常见稳定性风险不在最终 benchmark 里 |
| 是否配合 FP8/FP4 训练 | 低精度会放大 router logits、activation outlier 和 expert GEMM 的数值问题 |
| 是否做 distillation 或 post-training | MoE 的 base 能力和后训练能力要分开看 |

!!! warning "不要把训练细节读成装饰"
    MoE 的训练 recipe 往往比架构名更关键。一个专家数更多的模型，如果 router 负载不稳、expert batch 太碎、低精度保护不足，真实训练效率可能反而更差。

## 系统读法：从专家到硬件

MoE 的系统执行通常会引入 Expert Parallelism（EP）：不同 expert 放在不同 GPU 上，token 根据 router 结果被发到对应设备。

```mermaid
flowchart LR
    subgraph GPU0
        A0["tokens"] --> R0["router"]
        E0["expert 0/1"]
    end
    subgraph GPU1
        A1["tokens"] --> R1["router"]
        E1["expert 2/3"]
    end
    R0 --> X["all-to-all dispatch"]
    R1 --> X
    X --> E0
    X --> E1
    E0 --> Y["all-to-all combine"]
    E1 --> Y
```

读系统表时要特别看：

- EP group size：expert 分在多少张卡上；
- all-to-all 是否和 expert GEMM overlap；
- expert batch 是否足够大；
- router 和 dispatch 是否 fused；
- prefill 和 decode 是否分别报告；
- benchmark 的 batch size、context length、hardware topology 是否公开；
- 训练吞吐和推理吞吐是否混在一个结论里。

## 技术报告里的最短检查清单

读一篇大模型技术报告时，可以按下面顺序扫。

1. 结构表：是 dense、MoE、hybrid MoE，还是 Mamba-MoE？MoE 出现在多少层？
2. 参数表：`Total Params` 和 `Activated Params` 分别是多少？top-k 是多少？
3. 专家表：shared experts、routed experts、expert hidden size、capacity 或 dropless 设置是什么？
4. 训练表：数据 mixture、训练 token、optimizer、precision、router balance 策略是否讲清？
5. 系统表：TP/PP/DP/EP/CP 如何组合？有没有 all-to-all、MFU、tokens/s、latency？
6. 后训练表：SFT、reward、RL、distillation 是否改变能力结论？
7. 证据表：benchmark、ablation、safety、system profiling 分别支持哪类 claim？

| 报告类型 | 重点看什么 | 回到本站哪里读 |
| --- | --- | --- |
| DeepSeek-V3 / V4 | DeepSeekMoE、MLA、auxiliary-loss-free routing、FP8、MTP、并行训练 | [DeepSeek-V3 讲解](../paper-deep-dives/technical-reports/deepseek-v3.md)、[DeepSeek-V4 讲解](../paper-deep-dives/technical-reports/deepseek-v4-pro.md) |
| Qwen3 / Qwen3.5-Omni | dense/MoE 版本、thinking、omni 模态、后训练和部署边界 | [Qwen3 讲解](../paper-deep-dives/technical-reports/qwen3.md)、[Qwen3.5-Omni 讲解](../paper-deep-dives/technical-reports/qwen35-omni.md) |
| Kimi K2 | 大规模 MoE、MuonClip、agentic data、tool-use 后训练 | [Kimi K2 讲解](../paper-deep-dives/technical-reports/kimi-k2.md)、[优化与训练入门](optimization-and-training-basics.md) |
| Nemotron 3 Super | Mamba-heavy hybrid backbone、LatentMoE、shared-weight MTP、NVFP4/FP8 | [Nemotron 3 Super 讲解](../paper-deep-dives/technical-reports/nemotron-3-super.md)、[Mamba 与混合 SSM 架构](mamba-and-hybrid-ssm-transformers.md) |
| Gemini 2.5 / System Card | sparse MoE、long context、dynamic thinking、安全与产品证据 | [Gemini 2.5 讲解](../paper-deep-dives/technical-reports/gemini-2-5.md)、[数据划分与评测指标](data-splits-metrics-and-evaluation-basics.md) |

## 最容易犯的六个错

| 错误读法 | 更稳的读法 |
| --- | --- |
| 总参数越大，推理一定越慢 | 先看 activated params，再看 attention/KV、通信、batch 和硬件 |
| Activated params 小，所以 latency 一定低 | MoE 可能被 all-to-all、hot expert、碎 GEMM 卡住 |
| Expert 越多，能力一定越强 | 还要看数据、router、负载均衡、训练 token 和后训练 |
| MoE 省的是所有计算 | MoE 主要稀疏化 FFN，attention/KV 仍要单独处理 |
| Shared expert 是可有可无的细节 | shared expert 会影响通用知识和 routed expert 专业化 |
| MoE Serving 路由和模型内 MoE router 是一回事 | 前者是请求/模型级调度，后者是 token/expert 级模型结构 |

## 和已有专题的接口

MoE 相关内容在本站分散在多个层次，这一页只补“读架构表”的入口：

| 想继续读 | 去哪里 |
| --- | --- |
| 模型内 MoE 架构、参数表、训练细节 | 当前页 |
| 服务系统里的请求路由、多模型调度、SLO | [MoE 路由与多模型服务](../inference/moe-routing-and-multi-model-serving.md) |
| expert routing 的 kernel、dispatch、sparse GEMM | [MoE 路由与稀疏 Kernel](../operators/moe-routing-and-sparse-kernels.md) |
| MTP 和投机解码 | [MTP 与投机解码](../training/mtp-and-speculative-decoding.md) |
| Mamba/SSM 和混合架构 | [Mamba 与混合 SSM 架构](mamba-and-hybrid-ssm-transformers.md) |
| 低精度训练和推理 | [数值、显存与运行时](numerics-memory-and-runtime-basics.md)、[量化路线图](../quantization/index.md) |

读完这一页，再读技术报告时应该能做到：看到一张模型表，先把参数容量、激活计算、路由训练、通信成本和证据强度拆开，而不是被一个大模型规模数字带着走。

!!! tip "下一站"
    - 回到本专题入口：[基础知识](index.md)，确认这页在整条路线中的位置。
    - 按导航顺序继续：[Mamba 与混合 SSM 架构](mamba-and-hybrid-ssm-transformers.md)。
    - 概念或符号卡住时，先查 [术语表](../glossary/index.md)，再回到当前页。
