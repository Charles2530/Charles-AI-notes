---
paper_date: 2026-04
paper_title: "Nemotron 3 Super: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning"
paper_topic: 技术报告
---

# Nemotron 3 Super：Mamba-MoE 与异步多环境 RL

!!! info "论文信息"
    **论文题名：** Nemotron 3 Super: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning。<br>
    **作者：** NVIDIA。<br>
    **机构：** NVIDIA。<br>
    **时间 / 主题：** 2026-04；技术报告。<br>
    **arXiv / 官方报告：** arXiv：[2604.12374](https://arxiv.org/abs/2604.12374)。<br>
    **GitHub / 项目：** GitHub：未找到官方链接。<br>
    **元数据来源与核验口径：** 来源：[arXiv](https://arxiv.org/abs/2604.12374) / 官方发布页；Checked Date：2026-07-07；Repro Status：Paper / official materials reviewed, independent reproduction not claimed。

## 原论文故事线

### 研究背景

Nemotron 3 Super 属于 open-weight 大模型技术报告，但它的核心不是“再堆一个更大的 Transformer MoE”。它站在几条已有路线的交叉点上：MoE 用稀疏激活扩大参数容量，Mamba / state space model 用固定大小状态降低长序列生成成本，attention anchor 保留全局 token 交互，MTP 用未来 token 预测支撑 speculative decoding，RL post-training 则把模型能力推向工具调用、终端操作、代码修复和长上下文任务。

这些方向各自解决一部分问题。MoE 能增加容量，但 routed expert weights 和 all-to-all communication 会成为服务瓶颈；Mamba 的线性时间生成更适合长输出，但纯 recurrent state 难以替代全部 attention；MTP 可以加速解码，但独立 draft heads 容易出现 train-test mismatch；agentic RL 能提升真实工具任务，但 rollout 成本高、任务分布差异大。Nemotron 3 Super 的报告试图把这些系统问题放在同一个模型里解决。

### 问题（Challenge）

论文要解决的问题是：**怎样训练和部署一个 120B total、约 12B active 的开放模型，让它在 agentic reasoning 上接近同级强模型，同时在长上下文和长输出场景有更高吞吐？** 这个问题难在两个层面。模型结构层面，长上下文生成受 KV cache、attention 计算、expert 权重读取和分布式 MoE 通信共同限制，不能只优化单一模块。训练层面，agentic tasks 需要数学、代码、工具、终端、搜索和长上下文混合训练，如果只在一个环境上做 RL，很容易在其他能力上回退。

这个问题的价值很直接：agent 模型的真实成本不只来自预训练 FLOPs，也来自推理时长输出、并发 serving、工具任务 rollout 和多环境后训练。如果一个模型只有 benchmark 精度而没有吞吐和部署效率，就很难在 agent 场景中规模化使用。

### Finding

作者的 finding 是：**agentic reasoning 的效率问题不是单点 bottleneck，而是“序列状态、稀疏专家、draft decoding、低精度训练和多环境 RL”之间的系统耦合；把计算移到更便宜的表示空间，并让训练目标服务于推理路径，才能同时拿到能力和吞吐。** 这个洞见把问题从“Transformer MoE 是否足够大”改成“每个 token 的状态、通信、专家读取和后训练 rollout 是否匹配 agent 工作负载”。

这个 finding 解释了报告里的主要设计。Mamba-heavy hybrid 架构让多数层用固定大小 state 处理长序列，少量 attention 层作为 global anchors 保留全局交互；LatentMoE 不是在 hidden dimension 上做专家计算，而是先投影到 latent dimension，再路由和计算，从而降低 expert bandwidth 和 all-to-all payload；shared-weight MTP 让同一个预测头在多个 offset 上训练，减少递归 draft 时的分布偏移；异步多环境 GRPO 则把数学、代码、搜索、终端和长上下文 rollout 解耦，让训练吞吐和任务覆盖同时成立。

### 方法

模型方面，Nemotron 3 Super 120B-A12B Base 是 88 层 hybrid Mamba-Attention MoE，总参数约 120B，每次 forward 激活约 12B。多数层由 Mamba-2 和 LatentMoE 构成，少量 attention 层作为全局锚点；attention 使用 GQA，32 个 query heads、2 个 KV heads。LatentMoE 将 token 从 hidden dimension 投影到较小 latent dimension，在 latent space 中进行 routing 和 expert computation，再投影回原 hidden space；报告配置中总 experts 为 512，每 token 激活 22 个 experts。

训练方面，预训练总量为 25T text tokens，分为 20T 多样性覆盖阶段和 5T 高质量/benchmark 强化阶段，使用 Warmup-Stable-Decay 学习率日程。报告强调这是 Nemotron 3 系列中首个使用 NVFP4 预训练的 Super 模型，但并非所有层都用 FP4：最终部分网络、latent projections、MTP layers、QKV/attention projections 和 embedding 等模块保留 BF16 或更高精度，Mamba output projection 使用 MXFP8 以缓解 underflow。

后训练方面，模型先通过 SFT 扩大 agentic harness、数学、代码、工具和长上下文数据覆盖，再用异步 GRPO 做多环境 RL。异步设置把训练和推理解耦，多个环境并行产生 rollout；PivotRL 用于长时程 agentic capabilities，覆盖 programming、search、terminal、tool use 和长上下文等域。MTP 方面，模型使用 shared-weight 多 token prediction，使同一预测头在训练时覆盖多个未来 offset，推理时可递归生成 draft token 来提高 speculative decoding 的 acceptance length。

### 结论

实验结论主要有五类。第一，在 Figure 1 的报告口径下，Nemotron 3 Super 在 IFBench、HMMT、SWE-Bench、HLE、Terminal Bench、Tau Bench、RULER 1M 等能力评估上与 GPT-OSS-120B、Qwen3.5-122B 处于相近区间，同时在 8k input / 64k output 设置中达到最高 2.2x 和 7.5x 的吞吐提升。第二，LatentMoE 的目标不是单纯减少参数，而是在相似 active compute 下扩大 total experts 和 top-k active experts，同时减少 routed expert computation 与通信 payload。第三，shared-weight MTP 在 SPEED-Bench 和 throughput frontier 上提高 average acceptance length，并让 draft depth 增加时吞吐-延迟前沿向外移动。

第四，NVFP4 预训练实验显示低精度会带来更多 zero-valued gradients，尤其在 routed expert 的低范数通道上；但报告中的 MXFP8 healing 虽改善 loss trajectory，却没有带来持续下游 accuracy 收益，所以最终仍按 NVFP4 recipe 完成 25T token horizon。第五，后训练评估显示扩大 agentic SFT 数据和多环境 RL 后，模型在软件工程、终端使用、工具使用、长上下文和通用推理任务上明显强于 Nemotron 3 Nano，并在同级开放模型中保持竞争力。报告的结论不是某个模块单独有效，而是这些模块组合后形成了面向 agentic serving 的能力-吞吐折中。

### 关键术语

- **Hybrid Mamba-Attention（混合 Mamba-Attention 架构）**：用 Mamba-2 层承担大部分长序列状态更新，用少量 attention 层补充全局 token 交互，目标是在长上下文生成中降低 KV cache 和 attention 成本。
- **LatentMoE（潜空间混合专家）**：先把 hidden state 投影到较小 latent space，再做 expert routing 和 expert computation，以降低 expert 权重读取和 all-to-all communication。
- **Multi-Token Prediction, MTP（多 Token 预测）**：让模型预测多个未来 token 的训练/推理机制；Nemotron 3 Super 使用 shared-weight MTP，减少递归 draft decoding 的 train-test mismatch。
- **NVFP4（NVIDIA FP4 低精度格式）**：用于加速预训练和推理的 4-bit 浮点格式；报告采用混合精度 recipe，关键模块保留 BF16 或 MXFP8。
- **Asynchronous GRPO（异步 GRPO）**：把训练和推理 rollout 解耦的强化学习设置，用于在多个任务环境中提升 agentic 能力并降低同步等待成本。
- **Agentic Reasoning（智能体式推理）**：需要模型在工具、代码、终端、搜索、长上下文等环境中多步决策和执行的能力，不只是在静态问答 benchmark 上给出答案。

![Nemotron 3 Super benchmark and throughput](../../assets/images/paper-deep-dives/technical-reports/nemotron-3-super/figure-1-benchmark-throughput.png){ width="920" }

<small>图源：`Nemotron 3 Super Technical Report`，Figure 1。原图比较 Nemotron 3 Super、GPT-OSS-120B 和 Qwen3.5-122B 在 IFBench、HMMT、SWE-Bench、HLE、Terminal Bench、Tau Bench、RULER 1M 与 8k/64k 生成吞吐上的表现。</small>

**Figure 1 怎么读。**
这张图不是说 Super 每项都最高，而是说它在接近 GPT-OSS-120B / Qwen3.5-122B 的能力区间里，把长上下文和长输出吞吐做得更高。报告口径下，在 `8k` 输入、`64k` 输出设置中，Super 相比 GPT-OSS-120B 和 Qwen3.5-122B 分别达到最高 `2.2x` 和 `7.5x` 吞吐提升。

## 论文位置

Nemotron 3 Super 可以放在 DeepSeek-V3、Qwen3、Kimi K2 和 DeepSeek-V4 这几条技术报告线旁边看：

| Report | Core emphasis | Nemotron 3 Super 的差异 |
| --- | --- | --- |
| DeepSeek-V3 | MLA、DeepSeekMoE、FP8、DualPipe、MTP | 仍是 Transformer MoE 主干，重点是低成本训练和 KV cache 压缩 |
| Qwen3 | thinking / non-thinking、GRPO、强到弱蒸馏 | 重点是统一推理模式和模型家族蒸馏 |
| Kimi K2 | 1T sparse MoE、MuonClip、agentic data / RL | 重点是 agentic non-thinking 模型和 Muon 稳定训练 |
| Nemotron 3 Super | Hybrid Mamba-MoE、LatentMoE、NVFP4、MTP、异步多环境 RL | 把长上下文吞吐、MoE 通信/带宽、低精度训练和 agent 后训练放到同一个系统里 |

它不是只做一个更大的 MoE，而是把每 token 计算、长序列状态、expert 通信、speculative decoding、低精度路径和 RL rollout 成本一起优化。

## 模型家族

报告公开了 Super 120B-A12B 的 NVFP4、FP8、BF16、Base BF16 版本，也公开了 GenRM、pretraining specialized data 和 Super post-training data。官方集合还包括 Nano 30B-A3B、Nano Omni / 4B 等变体；本文只按报告主线解析 Super 120B-A12B。

## 总体架构

Nemotron 3 Super 是 88 层 hybrid Mamba-Attention MoE。它沿用 Nemotron 3 Nano 的混合 Mamba-Attention 思路，但把稀疏扩展从 standard MoE 换成 LatentMoE。

![Nemotron 3 Super layer pattern](../../assets/images/paper-deep-dives/technical-reports/nemotron-3-super/figure-2-layer-pattern.png){ width="920" }

<small>图源：`Nemotron 3 Super Technical Report`，Figure 2。原图展示 Nemotron 3 Super 的 Mamba-2、Attention 和 Latent MoE 周期性交错层 pattern。</small>

**layer pattern 看 Mamba 和 attention 的分工。**
图里大多数层是 Mamba-2 + LatentMoE，少量 attention 层作为全局交互锚点。读法不是数颜色块，而是问长上下文状态由谁维护、全局 token 交互由谁补足、MoE 稀疏计算插在哪些层。它支撑的是 hybrid 架构的成本分工，不是单独证明 Mamba 或 attention 任一方可以替代全部模块。

报告 Table 1 的架构配置如下，保留英文原格式：

| Configuration | Nemotron 3 Super 120B-A12B Base |
| --- | ---: |
| Total Layers | 88 |
| Model Dimension | 4096 |
| Q-Heads | 32 |
| KV-Heads | 2 |
| Head Dimension | 128 |
| Mamba State Dimension | 128 |
| Mamba Groups | 8 |
| Mamba Heads | 128 |
| Mamba Head Dimension | 64 |
| Expert Hidden Dimension | 2688 |
| Shared Expert Intermediate Size | 5376 |
| Total Experts per Layer | 512 |
| Top-k (Activated Experts) | 22 |
| MoE Latent Size | 1024 |
| MTP layers (shared weight) | 2 |

几个数字要一起看：`120.6B total parameters`，但每次 forward 约 `12.7B active parameters`；GQA 只有 `2` 个 KV heads，降低 KV cache 压力；模型省略 positional embeddings、dropout 和 linear bias，使用 RMSNorm，embedding 和 output weights 不 tied。核心差别在于：多数层用 Mamba-2 维护固定大小 recurrent state，少量 attention anchor 补全全局 token 交互。

## LatentMoE

普通 MoE 的 expert 计算在 hidden dimension `d` 上完成。LatentMoE 先把 token 从 `d` 投影到更小的 latent dimension `l`，再在 latent space 里路由和执行 expert 计算，最后投影回 `d`。

![Standard MoE vs LatentMoE](../../assets/images/paper-deep-dives/technical-reports/nemotron-3-super/figure-3-latent-moe.png){ width="860" }

<small>图源：`Nemotron 3 Super Technical Report`，Figure 3。原图比较 standard MoE 与 LatentMoE；LatentMoE 在 latent dimension 内做 expert routing / computation，以降低 routed parameter loads 和 all-to-all traffic。</small>

**LatentMoE 图看压缩发生在哪里。**
Standard MoE 在 hidden dimension \(d\) 上路由和执行 expert；LatentMoE 先降到 latent dimension \(l\)，在更小空间里路由、通信和计算，再升回 \(d\)。箭头想说明的是带宽和 all-to-all payload 被压小，而不是说 expert 数量越多永远越好；质量仍取决于路由、负载均衡和训练稳定性。

可以把它写成：

```text
Standard MoE:
  hidden d -> router -> experts in d -> combine

LatentMoE:
  hidden d -> latent down-proj l -> router / experts in l -> latent up-proj d
```

报告给出的设计逻辑很清楚：

| Bottleneck | Standard MoE 中的问题 | LatentMoE 的处理 |
| --- | --- | --- |
| Low-latency serving | 读取 expert weights 常被 memory bandwidth 支配 | 降低 expert 计算维度，从 `d` 压到 `l` |
| Throughput serving | distributed MoE 受 all-to-all routing volume 支配 | routing payload 也在 latent space 中传输 |
| Quality | 不能简单削减 FFN 的有效非线性预算 | 保持有效预算，并用节省的成本增加 total experts 和 top-k |
| Specialist diversity | expert 组合空间越大，模型可表达的专家组合越多 | 同时增加 experts 总数和每 token 激活 experts 数 |

报告的关键句可以概括为：把 routed expert computation 和 all-to-all traffic 移到 latent space 后，成本约降低 `d/l` 倍；节省的预算再用于扩大 total experts 和 top-k active experts。

在 Super 中，`d=4096`、MoE latent size `1024`，即 4:1 压缩。这让模型可以有 `512` 个 experts、每 token 激活 `22` 个 routed experts，同时保持约 12B active 的推理预算。

**LatentMoE 和 MLA 不是一回事。**
DeepSeek 系列的 MLA 压的是 attention KV 表示，主要目标是 KV cache 和 attention memory。LatentMoE 压的是 MoE expert 路由和计算维度，主要目标是 expert weight bandwidth 和 all-to-all communication。

两者都用了 latent 表示，但它们改的是不同系统瓶颈。Nemotron 3 Super 的 attention 侧靠 Mamba-heavy 混合架构和少量 GQA attention anchor；MoE 侧靠 LatentMoE。

## MTP：共享参数的递归草稿头

Nemotron 3 Super 的 MTP 不只是“多接几个未来 token head”。报告强调它针对 speculative decoding 的 autoregressive drafting 做了共享参数设计。

标准 MTP 常训练 `D` 个独立 head，各自预测固定 offset；推理时若递归复用某个 head 生成更长 draft，就会从 ground-truth hidden states 切到自生成状态，形成 train-test mismatch。Nemotron 3 Super 改为让多个 MTP heads 共享参数，训练同一个 prediction head 覆盖多个 offset；推理时它可以递归应用，acceptance rate 掉得更慢。

![MTP acceptance rate by draft index](../../assets/images/paper-deep-dives/technical-reports/nemotron-3-super/figure-4-mtp-acceptance.png){ width="520" }

<small>图源：`Nemotron 3 Super Technical Report`，Figure 4。原图展示 SPEED-Bench 上 draft length = 7 时，不同 draft token index 的 acceptance rate。</small>

报告 Table 2 的 average acceptance length 里，最能代表差异的几行如下：

| Category | DSR1 | Qwen3 Next | Nemotron3 Super |
| --- | ---: | ---: | ---: |
| Coding | 2.99 | **4.32** | 3.78 |
| Math | 2.98 | **3.89** | 3.73 |
| Multilingual | 2.83 | 3.97 | **4.05** |
| RAG | 2.79 | 3.53 | **3.78** |
| Reasoning | 2.80 | 3.47 | **3.59** |
| Average | 2.70 | 3.33 | **3.45** |

![MTP throughput frontier](../../assets/images/paper-deep-dives/technical-reports/nemotron-3-super/figure-5-mtp-throughput.png){ width="560" }

<small>图源：`Nemotron 3 Super Technical Report`，Figure 5。原图展示 NVFP4 checkpoint 在 TRT-LLM、TP=1、B300 GPU 上，MTP off、MTP D=1、MTP D=3 的 total throughput 与 median user throughput Pareto frontier。</small>

MTP 同时是训练目标和推理接口：训练时给模型更密集的未来 token 信号，推理时变成内置 draft model。Super 的重点是 shared-weight MTP，把递归 drafting 的分布偏移纳入训练设计。

## NVFP4 预训练

Nemotron 3 Super 是报告声称的 Nemotron 3 系列中首个使用 NVFP4 预训练的 Super 模型。它不是所有层都粗暴 FP4，而是混合精度：

| Layer Type | Format | Rationale |
| --- | --- | --- |
| All Linear Layers Unless Otherwise Noted | NVFP4 |  |
| Final 15% of Network | BF16 | Promote training stability at scale |
| Latent Projections | BF16 | Strategically kept in BF16 as step-time impact is negligible |
| MTP Layers | BF16 | Preserves multi-token prediction capabilities |
| QKV & Attention Projections | BF16 | Maintain fidelity of few attention layers |
| Mamba Output Projection | MXFP8 | Mitigates high incidence of underflows observed when quantizing this layer to NVFP4 at smaller scales |
| Embedding Layers | BF16 |  |

低精度细节也很具体：

| Tensor / Operation | NVFP4 Recipe |
| --- | --- |
| Weights | NVFP4 with two-dimensional block scaling |
| Gradients and activations | NVFP4 with one-dimensional blocks along GEMM reduction axis |
| wgrad inputs | Random Hadamard Transforms |
| gradients | stochastic rounding |
| element format | E2M1 |
| micro-block size | 16 elements |
| micro-block scale | E4M3 |
| second-level scale | FP32 global scale |

这部分最有价值的是报告没有只说“NVFP4 很快”，而是分析 underflow 和 zero-valued gradients：到预训练结束，zero-valued weight gradient elements 约占总参数 `7%`，并且和 routed expert 的 FC1/FC2 低范数通道相关。作者认为 NVFP4 会让 BF16/MXFP8 中仍可表达的小梯度更容易 underflow 成 true zero，但这不直接等同于训练失败。

![Zero-valued gradients](../../assets/images/paper-deep-dives/technical-reports/nemotron-3-super/figure-7-zero-gradients.png){ width="760" }

<small>图源：`Nemotron 3 Super Technical Report`，Figure 7。原图比较 Nemotron 3 Nano 的 BF16 与 NVFP4 recipe 下 zero-valued weight gradient elements 的增长。</small>

**zero gradient 图不要只看越低越好。**
横向比较的是 BF16 和 NVFP4 recipe 下的零梯度比例，重点是 NVFP4 更容易让小梯度 underflow 成真零。它提醒低精度训练要看通道、层和任务指标的联动；零梯度上升是风险信号，但报告后续也说明 MXFP8 healing 未带来持续下游收益。

工程结论：

1. NVFP4 的确会更早、更明显地产生 zero-valued gradients，尤其在 routed expert 的某些通道上；
2. 但把 19T 后模型切到 MXFP8 虽然改善 loss trajectory，却没有带来持续下游 accuracy 提升，所以最终 Super 仍按 NVFP4 recipe 跑完整 25T token horizon。

低精度训练不能只看 loss 是否更平滑，还要看最终任务指标。报告结论是：MXFP8 healing 没有兑现成下游收益。

## 预训练数据与超参

Nemotron 3 Super 预训练总量是 `25T` text tokens，分两阶段：

| Phase | Tokens | Main Goal |
| --- | ---: | --- |
| Phase 1 | 20T | diversity and broad coverage |
| Phase 2 | 5T | high-quality data and benchmark accuracy |

![Pretraining data mixtures](../../assets/images/paper-deep-dives/technical-reports/nemotron-3-super/figure-10-pretraining-mixtures.png){ width="920" }

<small>图源：`Nemotron 3 Super Technical Report`，Figure 10。原图展示 Phase 1 与 Phase 2 的 pre-training data mixture。</small>

**data mixture 图看阶段目标。**
Phase 1 更像覆盖面和多样性建设，Phase 2 更像高质量与 benchmark 相关能力强化。读这张图时要把比例和 token 阶段放在一起看，不能只说某个数据源占比高；真正影响模型性格的是数据类型、质量筛选、去污染和后续 SFT/RL 如何接上。

报告新增或强调的 synthetic data 可以概括为：

| Dataset | Construction Detail |
| --- | --- |
| Synthetic Code Concepts | 91 个 HumanEval 高层编程概念；约 14M problems；清洗后约 15M problems |
| Synthetic Unconditional Algorithmic | Qwen3-235B-A22B Base + GPT-OSS-120B 生成算法题；约 0.2B tokens；做 benchmark 去污染 |
| Synthetic Economics / Formal Logic | 覆盖经济学 MCQ、自然语言逻辑、谓词/命题逻辑、真值表等任务 |
| Synthetic Multiple Choice | 从 MMLU auxiliary training set 扩展；DeepSeek-V3 解题与 majority voting；约 3.5M samples / 1.6B tokens |

核心预训练超参如下：

| Setup | Value |
| --- | --- |
| LR schedule | Warmup-Stable-Decay；200B warmup；final 5T minus-sqrt decay |
| LR range | peak `4.5e-4` -> minimum `4.5e-6` |
| Optimizer | AdamW；weight decay 0.1；`beta1=0.9`, `beta2=0.95` |
| Sequence length | 8,192 |
| Batch | 3,072 sequences；about 25.17M tokens per batch |
| MoE routing | 512 experts, top-22；sigmoid router score + expert biasing |
| Load balancing | auxiliary-loss-free update rate `1e-3` + load balancing loss coefficient `1e-4` |
| MTP loss scale | 0.3 |
| Precision | BF16 + NVFP4 |

### Checkpoint merging

WSD stable phase 中，单个 checkpoint 的 benchmark 可能噪声较大；报告用滑动窗口对近邻 checkpoint 做 weighted averaging，作为更便宜的质量 readout。

![Checkpoint merge evaluation](../../assets/images/paper-deep-dives/technical-reports/nemotron-3-super/figure-11-checkpoint-merge.png){ width="760" }

<small>图源：`Nemotron 3 Super Technical Report`，Figure 11。原图比较 trained checkpoints 与 best offline checkpoint merge 在 25T pretraining 中的 12 benchmark average accuracy。</small>

**checkpoint merge 图看 readout 成本。**
曲线比较的是直接训练 checkpoint 与近邻 checkpoint merge 的评测表现。它支撑的不是“merge 一定提高最终模型”，而是 stable LR 阶段可以用 merge 更便宜地读出质量趋势，减少为了等单个 checkpoint 波动消失而继续烧 token 的成本。

经验是：stable LR phase 中，best merge 比对应 trained checkpoint 平均高 `2-4` 分；最后 5T LR decay 后，实际 decay checkpoint 逐渐追平。最终 base model checkpoint 是 `500B` merge；报告估计该流程可节省约 `4T` tokens 的 readout 成本，约占总 pretraining FLOP 预算的 `16%`。

### Long-context extension

长上下文阶段在预训练末尾进行：

| Long-context Detail | Value |
| --- | --- |
| Learning rate | constant `4.5e-6` |
| Global batch size | 16 |
| Hardware | GB200 GPUs |
| Context parallelism | 64-way |
| Tensor parallelism | 2-way |
| Expert parallelism | 64-way |
| Stage 1 | 1,048,576 context length for 34B tokens |
| Stage 2 | alternating 1M and 4K sequences for 17B tokens |
| Data blend | 20% long-context document QA, 80% downscaled Phase 2 data |

这解释了为什么 RULER 1M 能成为主打指标：模型不是只靠位置外推，而是在 1M context 上做 continuous pretraining，并用 1M/4K 交替训练缓解数学 benchmark 的轻微回退。

## Base Model 结果

报告 Table 4 对 Base model 和 Ling-flash-base-2.0、GLM-4.5-Air-Base 做比较。下面摘录关键行，保留英文列名：

| Task | Metric | N-3-Super 120B-A12B-Base | Ling-flash base-2.0 | GLM-4.5 Air-Base |
| --- | --- | ---: | ---: | ---: |
| MMLU | 5-shot, acc | 86.01 | 81.00 | 81.00 |
| MMLU-Pro | 5-shot, CoT EM | 75.65 | 62.10 | 58.20 |
| GPQA-Diamond | 5-shot, CoT EM | 60.00 | 36.00 | 23.20 |
| GSM8K | 8-shot, EM | 90.67 | 90.75 | 82.60 |
| MATH | 4-shot, EM | 84.84 | 63.80 | 50.36 |
| MATH Level 5 | 4-shot, EM | 70.00 | 39.80 | 26.30 |
| AIME2024 | pass@32 | 53.33 | 30.00 | 20.00 |
| HumanEval | 0-shot, pass@1 n=32 | 79.40 | 70.10 | 76.30 |
| MBPP-Sanitized | 3-shot, pass@1 n=32 | 78.38 | 77.30 | 77.50 |
| MMLU Global Lite | 5-shot, avg | 85.72 | 74.94 | 79.25 |
| MGSM | 8-shot, avg | 87.47 | 82.73 | 80.33 |
| RULER64K | 0-shot | 92.26 | 72.12 | 80.26 |
| RULER128K | 0-shot | 88.26 | 52.03 | 61.70 |
| RULER256K | 0-shot | 84.56 | - | - |
| RULER512K | 0-shot | 82.49 | - | - |
| RULER1M | 0-shot | 71.00 | - | - |

Base 阶段最能说明架构和预训练本身的价值：Super 不是只靠后训练刷 agent 分，而是在 MMLU-Pro、GPQA、MATH Level 5、AIME、RULER 等 base evaluations 上已拉开差距。

## 后大模型训练路线图

Nemotron 3 Super 的后训练分四段：SFT、RLVR、SWE-RL、RLHF，最后补一个 MTP healing。

![Post-training pipeline](../../assets/images/paper-deep-dives/technical-reports/nemotron-3-super/figure-12-posttraining-pipeline.png){ width="920" }

<small>图源：`Nemotron 3 Super Technical Report`，Figure 12。原图展示 Nemotron 3 Super 的 post-training pipeline：Base -> SFT -> RLVR round 1/2/3 -> SWE-RL -> RLHF -> MTP healing -> Nemotron 3 Super。</small>

图中关键数字：

| Stage | Report Figure Detail |
| --- | --- |
| Base | 1M Context |
| SFT | 7M samples, 88B tokens |
| RLVR Round 1 | 25 env types |
| RLVR Round 2 | 30 env types, with low effort |
| RLVR Round 3 | 26 env types, with low effort, agentic focused |
| RLVR scale | 37 environment types, up to 4,000 environment instances per batch |
| SWE-RL | 20B tokens |
| RLHF | 18B tokens |
| MTP healing | train MTP heads, freeze rest of weights |

报告正文还写到，RLVR setting 包含 `21 environments` 和 `37 RL datasets`。不要把图里的 environment types 和正文里的 environments / datasets 混成同一个数。

## SFT：两阶段损失和 agentic 数据扩容

SFT 采用两阶段设计，核心是处理长输出 reasoning 与 long-input-short-output 任务之间的 loss 权重冲突。

| SFT Stage | Loss Normalization | Setup |
| --- | --- | --- |
| Stage 1 | token-level global average | 256K sequence length packing, global batch size 64, constant LR `1e-5`, 30K warmup samples |
| Stage 2 | sample-level average | 512K sequence length packing, includes long context data up to 512K, global batch size 32, constant LR `1e-5` |

Stage 1 让长输出样本获得更高权重，适合吸收 reasoning、代码和 agent 轨迹；Stage 2 先按每个 conversation 自身输出 token 数归一化，再对 conversation 平均，避免短输出任务被超长输出淹没。MTP 在 SFT 中继续训练：两层 shared-parameter MTP，per-token auxiliary loss，scale 仍是 `0.3`。

### Agentic 与工具数据

报告把 Agentic Command Line Interface 数据管线单独画出来，说明他们不是只做普通对话 SFT，而是在模拟 Codex/OpenCode/Qwen Code CLI 这类多步工具环境。

![Agentic CLI dataset pipeline](../../assets/images/paper-deep-dives/technical-reports/nemotron-3-super/figure-13-agentic-cli-data.png){ width="920" }

<small>图源：`Nemotron 3 Super Technical Report`，Figure 13。原图展示 Agentic Command Line Interface Dataset Construction & Training Pipeline。</small>

**agentic CLI 图按数据闭环读。**
左侧是 seed、任务和工具环境，中间是多步 trajectory 生成与筛选，右侧才进入训练 blend。它想说明 agent 能力不是普通问答样本顺手学出来的，而是需要任务、环境反馈、工具 schema、失败恢复和过滤规则一起构造；图本身不能证明真实 IDE 或终端环境里的可靠性已经足够。

关键数据构造可以合并成下面几类：

| Data block | Scale / construction | Training intent |
| --- | --- | --- |
| Agentic CLI | 20K seed；15K synthesis；3K SWE；10K Node.js web dev | 训练 CLI 规划、文件操作、工具调用、失败恢复 |
| Long Context | 128K/256K/512K 样本；多条 traces + semantic majority voting | 补 long-input-short-output 和跨文档推理 |
| Financial Reasoning | 从 565 个 SecQue seed 扩展到 S&P500 公司和 2019-2024 年；最终 366,243 Q&A pairs | 强化结构化金融问答与证据定位 |
| CUDA | 100K synthetic CUDA samples，含 kernel generation、repair、optimization、Nsight Compute log | 覆盖工程型代码优化任务 |
| Search / Terminal / Tool Use | conversational tool-use、general-purpose tool-calling、Terminal-Task-Gen | 让模型学习工具 schema、环境反馈和多步轨迹 |

工具调用数据分两条管线：specialized conversational tool-use 产生 `279,116` conversations、覆盖 `838` domains；general-purpose tool-calling 基于 ToolEyes、API-Bank、UltraTools、xLAM 和自写工具构造 `144` toolsets，并生成约 `1.5M` trajectories。

SFT blend 的整体比例如下：

![SFT data blend](../../assets/images/paper-deep-dives/technical-reports/nemotron-3-super/figure-16-sft-blend.png){ width="560" .atlas-figure-compact }

<small>图源：`Nemotron 3 Super Technical Report`，Figure 16。原图展示 Nemotron 3 Super 的 SFT data blend：Agent、Reasoning、Chat、Long Context、Misc。</small>

**SFT blend 图看能力预算。**
Agent、Reasoning、Chat、Long Context 和 Misc 的占比表示后训练样本预算分配。Agent 占比最高说明 Super 把工具和 CLI 能力放在核心训练目标里，但这个饼图不能单独说明数据质量、轨迹难度或奖励模型可靠性，仍要结合后续 RL 和 benchmark。

| SFT Blend | Share |
| --- | ---: |
| Agent | 36.0% |
| Reasoning | 31.0% |
| Chat | 23.0% |
| Long Context | 8.0% |
| Misc. | 2.0% |

Agent 占比最高，说明工具与 CLI 不是后训练末尾的附加项，而是 Super 的主要能力面之一。

## 强化学习：多环境同步训练

Nemotron 3 Super 的 RL 不是逐个任务串行强化，而是 Stage 1 使用 unified RLVR 同步训练多类环境，避免 single-environment training 造成能力回退。

RLVR 环境覆盖：

| RLVR Domain | Detail |
| --- | --- |
| Math / STEM / Reasoning | competitive math, Python tool use, formal proof, harder STEM and reasoning gym |
| Code / SWE prep | competition-style code and single-step patch generation |
| Instruction / Safety | rubric rewards, over-refusal reduction, jailbreak robustness |
| Long Context / Agentic Tool Use | long-context environment, conversational tool use, terminal use |

低努力推理也在 RL 中优化：一小部分 prompt 被转为 low-effort mode，reward 同时考虑 correctness 和 generated tokens 数。初期 Math、STEM QA、competitive coding 约占所有 RL prompts 的 `2%`，后来缩到 Math 和 STEM QA 的 `1%`。这相当于把长度成本显式放进训练目标。

### 异步 GRPO

报告的 RL 算法部分非常值得单独看。它使用 asynchronous GRPO：

| Algorithm Detail | Nemotron 3 Super Setup |
| --- | --- |
| Training / inference | decoupled across separate GPU devices |
| Rollout | workers continuously fill rollout buffer; updated weights pushed once available |
| In-flight updates | one trajectory may contain tokens from different model versions; KV cache not recomputed |
| Policy lag / off-policy fix | workers at most one step behind; masked importance sampling ratio |
| RLVR batch | 256 prompts x 16 responses = 4096 samples |
| Generation | one gradient update per rollout; max generation length from 49K to 64K |

异步 rollout 会带来 policy lag 和 off-policy 问题；报告用 masked importance sampling ratio 控制训练/rollout 策略不一致，同时限制 worker 至多落后一步，避免 rollout 数据太旧。

### SWE-RL 和 PivotRL

Stage 2 单独做 SWE-RL，因为软件工程 rollout 慢、上下文长，和短 horizon 任务混训会拖慢吞吐。每个 rollout 会启动 Apptainer container，加载目标仓库，运行 OpenHands agent loop 生成 patch，再跑 ground-truth tests 给 binary reward。为了工具多样性，报告还在 OpenHands 中实现 OpenCode 和 Codex agent classes，匹配 Claude Code / Codex CLI 的工具格式。

PivotRL 用于 Agentic Programming、Search、Terminal Use、Conversational Tool Use。它复用 offline SFT expert trajectories，不对整条长轨迹做昂贵在线 RL，而是在策略对下一步 action 有不确定性的 assistant turns 上做 RL，并用 domain-appropriate reward 给相似动作 credit。这是控制 agentic RL 成本的关键设计。

### RLHF

RLHF 使用 GenRM。报告写到 GenRM 以 `Qwen3-235B-A22B-Thinking-2507` 初始化，并使用 Helpsteer 3、商业友好的 lmarena-140k 子集和近期 human preference data。Super 在 multi-environment RL stage 中也使用 GenRM，并在最后额外做 RLHF-only stage，以改善 instruction following、robustness 和交互质量。

## Post-trained 结果

报告 Table 5 覆盖 general knowledge、reasoning、agentic、instruction following、long context 和 multilingual。下面保留关键行：

| Benchmark | N-3-Super | Qwen3.5-122B-A10B | GPT-OSS-120B |
| --- | ---: | ---: | ---: |
| MMLU-Pro | 83.73 | 86.70 | 81.00 |
| AIME25 (no tools) | 90.21 | 90.36 | 92.50 |
| HMMT Feb25 (no tools) | 93.67 | 91.40 | 90.00 |
| HMMT Feb25 (with tools) | 94.73 | 89.55 | - |
| GPQA (no tools) | 79.23 | 86.60 | 80.10 |
| LiveCodeBench (v5 2024-07 -> 2024-12) | 81.19 | 78.93 | 88.00 |
| HLE (no tools) | 18.26 | 25.30 | 14.90 |
| Terminal Bench (hard subset) | 25.78 | 26.80 | 24.00 |
| SWE-Bench (OpenHands) | 60.47 | 66.40 | 41.9 |
| TauBench V2 Average | 61.15 | 74.53 | 61.0 |
| RULER1M | 91.64 | 91.33 | 22.30 |
| MMLU-ProX (avg over langs) | 79.36 | 85.06 | 76.59 |

这张表说明 Super 不是平均碾压 Qwen3.5-122B，而是把长上下文、工具、终端、SWE 和吞吐放在同一条效率线上：RULER 256K/512K/1M 基本与 Qwen3.5 持平，远超 GPT-OSS-120B；Terminal Bench hard 与 TauBench average 接近 GPT-OSS 或略优；SWE-Bench 仍低于 Qwen3.5，但明显高于 GPT-OSS；HLE no tools 不是最强项，with tools 后才拉回竞争力。

## 推理量化

报告最后一部分讲 FP8 和 NVFP4 PTQ。要区分三件事：

1. **NVFP4 pretraining**：训练阶段大量 GEMM 用 NVFP4；
2. **FP8 checkpoint**：面向 Hopper 的 W8A8 部署版本；
3. **NVFP4 checkpoint**：面向 Blackwell 的 W4A4 部署版本。

### FP8 与 NVFP4 checkpoint

FP8 PTQ 使用 `256` 个 post-training SFT samples，context length `65536`。核心选择是：Embedding、attention projection、attention BMM2、MoE latent projection、1D conv 和 output 保持 BF16；KV cache + attention BMM1、MoE GEMM、Mamba GEMM 用 FP8；Mamba SSM cache 用 FP16。

NVFP4 PTQ 面向 Blackwell W4A4。naive NVFP4 PTQ 会有 accuracy loss，所以最终 recipe 组合三件事：

| Component | Role |
| --- | --- |
| calibrated per-block weight scaling that minimizes MSE | weight quantization offline search, no runtime cost |
| dynamic per-block max-based activation scaling | activation quantization must be runtime efficient |
| selective promotion through Model-Optimizer AutoQuantize | sensitive layers promoted to FP8 or BF16 |

AutoQuantize 在 `{NVFP4, FP8, BF16}` 中搜索 per-operator assignment，约束 effective precision budget 为 `4.75 bits`。最终保留 BF16 的主要是 embedding、attention QKV、attention BMM2、Mamba 1D conv、output；保留 FP32 的是 router；可搜索 promotion 的主要是 attention output、shared experts、MoE latent projection 和 Mamba projection；routed sparse expert GEMM 主要落在 NVFP4。完整 mixed-precision PTQ 在单个 `8 x B200` node 上少于 2 小时完成，校准使用 `512` 个 SFT samples、sequence length `4096`，最终 NVFP4 模型达到 BF16 baseline 的 `99.8%` median accuracy。

Table 8 关键行如下：

| Benchmark | N-3-Super | N-3-Super FP8 | N-3-Super NVFP4 |
| --- | ---: | ---: | ---: |
| MMLU-Pro | 83.73 | 83.63 | 83.33 |
| HMMT Feb25 (with tools) | 94.73 | 94.38 | 95.36 |
| GPQA (no tools) | 79.23 | 79.36 | 79.42 |
| LiveCodeBench | 81.19 | 80.99 | 80.56 |
| Terminal Bench hard | 25.78 | 26.04 | 24.48 |
| SWE-Bench (OpenCode) | 60.47 | - | 59.90 |
| RULER1M | 91.64 | 91.43 | 91.60 |
| MMLU-ProX avg | 79.35 | 79.21 | 79.37 |

这张表支撑报告的关键部署主张：NVFP4 不是只压小模型，而是在 120B-A12B Super 上也能基本保持 BF16 精度，同时给 Blackwell 推理吞吐留出空间。

### Mamba state cache

Mamba 的 SSM cache 默认 FP32。报告尝试把 SSM cache 存成 FP16，以减少 memory-bound decoding 的 DRAM read，但发现 naive FP16 cache 会显著增加 verbosity：

| Precision | SSM cache recipe | Key observation |
| --- | --- | --- |
| W16A16 | FP32 | baseline verbosity |
| W16A16 | FP16 | livecodebench verbosity +36.95% |
| W16A16 | FP16+SR (Philox 5) | verbosity drops back near baseline |
| W8A8 | FP16 | livecodebench verbosity +40.27% |
| W8A8 | FP16+SR (Philox 5) | verbosity increase only +1.79% / +1.08% |

原因是 Mamba decoding 是 recurrent：cache quantization error 会沿 transition 累积。普通 FP16 cast 的 RTNE rounding 会产生系统性 bias；stochastic rounding 在期望上无偏，能把漂移变成零均值噪声。最终报告选择 `FP16+SR (Philox 5)`，因为它不需要额外 block scale，且 Blackwell 有 PTX 支持。

## 工程启发

这篇报告最值得复用的不是某个单点 trick，而是系统协同：

| Theme | Takeaway |
| --- | --- |
| Long context | Mamba-heavy 主干让大多数层在生成阶段只维护固定状态，少量 attention anchor 保留全局交互 |
| MoE serving | LatentMoE 瞄准 expert weight bandwidth 和 all-to-all，而不只看每 token FLOPs |
| MTP | shared-weight MTP 把递归 drafting 的分布偏移提前暴露给训练 |
| Low precision | NVFP4 要看 underflow、zero gradients 和下游指标，不只看 loss |
| Agent RL | 多环境 RLVR 避免能力跷跷板，异步 GRPO 保证 rollout/training 吞吐 |
| Quantization | Mamba SSM cache 量化会改变 verbosity，recurrent state 误差要单独处理 |

## 边界与风险

几个边界要记住：benchmark 绑定 Nemo Evaluator / harness / prompt / judge；吞吐结论绑定 B200/B300、vLLM/TRT-LLM 和具体 TP 设置；SWE-Bench 仍落后 Qwen3.5-122B；LatentMoE 与异步 RL 都依赖并行策略、kernel、runtime 和训练平台，复现门槛很高。

## 阅读结论

Nemotron 3 Super 的核心不是“120B MoE 又多强”，而是一个完整工程命题：

```text
如何让一个 120B total / 12B active 的开源模型，
在 1M context、长输出、工具调用、终端、SWE 和多环境 RL 上可用，
同时让训练和推理都落到 NVFP4 / FP8 / Blackwell / Hopper 的现实硬件路径上。
```

关键是组合拳：Mamba、LatentMoE、MTP、NVFP4、异步多环境 RL、PTQ 和 SSM cache recipe 共同服务训练与部署。

## 相关阅读与下一步

- 外部材料：[论文标题 arXiv 检索](https://arxiv.org/search/?query=Nemotron+3+Super%3A+Open%2C+Efficient+Mixture-of-Experts+Hybrid+Mamba-Transformer+Model+for+Agentic+Reasoning&searchtype=title)。
- 外部材料：[Semantic Scholar 标题检索](https://www.semanticscholar.org/search?q=Nemotron+3+Super%3A+Open%2C+Efficient+Mixture-of-Experts+Hybrid+Mamba-Transformer+Model+for+Agentic+Reasoning)。
- 外部材料：[Papers with Code 检索](https://paperswithcode.com/search?q=Nemotron+3+Super%3A+Open%2C+Efficient+Mixture-of-Experts+Hybrid+Mamba-Transformer+Model+for+Agentic+Reasoning)。
- 站内下一步：[论文精读专题](../index.md)。
- 站内下一步：[论文专题写作规范](../writing-standard.md)。
- 站内下一步：[外部精读来源台账](../../references/external-reading-ledger.md)。
