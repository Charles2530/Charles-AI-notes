# 量化服务栈与硬件选择

这一页讲量化放到服务系统里以后，怎样按硬件、延迟、吞吐、成本和质量目标做取舍。具体 runtime 兼容性放在 [量化运行时与部署框架](quantization-runtimes-and-frameworks.md)。

!!! note "初学者先抓住"
    量化收益来自三处：少占显存、少搬数据、命中更快的低精度 kernel。三者不一定同时发生。

## 1. 量化收益的三种来源

| 来源 | 例子 | 可能被什么抵消 |
| --- | --- | --- |
| 存储变小 | FP16 权重变 INT4/FP8 | scale 元数据、碎片、adapter |
| 带宽下降 | decode 读权重或 KV 更少 | dequant、unpack、非连续访存 |
| 计算更快 | FP8/INT8 tensor core | kernel 不支持、shape 不匹配、fallback |

**读作什么**：一个方案可以显存省很多，但速度不变；也可以速度提升，但质量掉点不可接受。上线必须同表看。

## 2. 一个简化成本模型

线上请求成本可以粗略拆成：

\[
\mathrm{Cost} \approx C_{\mathrm{weight}} + C_{\mathrm{kv}} + C_{\mathrm{compute}} + C_{\mathrm{runtime}} + C_{\mathrm{quality}}
\]

| 符号 | 含义 |
| --- | --- |
| \(C_{\mathrm{weight}}\) | 权重常驻显存和权重读带宽 |
| \(C_{\mathrm{kv}}\) | KV cache 显存和 decode 读写 |
| \(C_{\mathrm{compute}}\) | GEMM/attention 等计算时间 |
| \(C_{\mathrm{runtime}}\) | batching、调度、格式转换、kernel launch |
| \(C_{\mathrm{quality}}\) | 掉点、回退、人工处理和业务风险 |

**读作什么**：量化如果只降低 \(C_{\mathrm{weight}}\)，但 \(C_{\mathrm{kv}}\) 或 \(C_{\mathrm{quality}}\) 上升，整体未必划算。

## 3. 不同硬件的量化直觉

| 硬件 | 常见特点 | 量化策略直觉 |
| --- | --- | --- |
| 数据中心新 GPU | FP8/INT8/部分 INT4 路径更成熟 | 优先用硬件原生支持格式 |
| 消费级 GPU | 显存有限，低比特权重很有吸引力 | weight-only + 成熟 kernel 更现实 |
| CPU/边缘设备 | 带宽和指令集差异大 | GGUF/INT8/INT4 格式要贴设备 |
| 多卡服务 | 通信、并行和 KV 分布也重要 | 量化要和 tensor/pipeline parallel 配合 |

!!! warning "不要只按 GPU 型号拍脑袋"
    同一张卡上，不同 runtime、driver、batch shape、context length 和量化格式，结果可能差很多。硬件选择必须结合真实请求压测。

## 4. 服务目标不同，量化选择不同

| 服务目标 | 优先关注 | 推荐起点 |
| --- | --- | --- |
| 低延迟聊天 | TTFT、TPOT、P95/P99 | 稳妥 W8/FP8 或成熟 W4A16 |
| 高吞吐批处理 | tokens/s、GPU 利用率 | FP8/W8A8、较大 batch |
| 长上下文 RAG | KV 显存、decode 带宽 | 权重量化 + KV cache 评估 |
| 边缘/本地 | 模型能否装下、内存带宽 | INT4/GGUF/weight-only |
| 多模态/VLA | 细节和控制稳定性 | 主体量化，projector/action/risk 保高精 |

## 5. 权重量化和 KV 量化的分工

权重显存：

\[
\mathrm{WeightMemory}\approx N_{\mathrm{params}}\times \mathrm{bytes}
\]

KV 显存：

\[
\mathrm{KVMemory}\approx2\times L\times B\times T\times H_{\text{kv}}\times d_h\times \mathrm{bytes}
\]

| 符号 | 含义 |
| --- | --- |
| \(N_{\mathrm{params}}\) | 参数数量 |
| \(L\) | 层数 |
| \(B\) | 并发序列或 batch |
| \(T\) | 上下文长度 |
| \(H_{\text{kv}}d_h\) | KV 总宽度 |

**读作什么**：权重量化主要影响模型常驻成本；KV 量化主要影响长上下文和并发。两者要分别评估。

## 6. 低延迟和高吞吐的冲突

| 目标 | 常见做法 | 量化影响 |
| --- | --- | --- |
| 低 TTFT | 小 batch、快速 prefill | dequant 开销更容易显眼 |
| 低 TPOT | decode kernel 高效 | KV dtype 和权重带宽重要 |
| 高吞吐 | 合批、连续 batching | 低比特 kernel 更可能摊薄开销 |
| 稳 P99 | 避免 fallback 和动态转换 | 需要 shape bucket 和监控 |

**直觉**：短请求、小 batch 下，量化额外开销可能盖过收益；长请求、大 batch 或带宽受限时，量化更容易兑现。

## 7. 混合精度是生产常态

| 模块 | 更激进 | 更保守 |
| --- | --- | --- |
| LLM 主体 Linear | FP8/INT8/INT4 | BF16 |
| attention / MLP GEMM | FP8/W8A8 | BF16/FP16 |
| LayerNorm/Softmax | 通常不激进 | BF16/FP16 |
| KV cache | INT8/FP8 | BF16/FP16 |
| projector/action/risk head | 视任务谨慎 | BF16/FP16 |

生产系统经常不是“全模型 4bit”，而是把低精度放在收益最大的位置，把高精度留给风险最高的位置。

## 8. 多租户和多模型场景

量化在多租户里可能带来额外收益：

| 收益 | 说明 |
| --- | --- |
| 单卡放更多模型 | 权重显存下降 |
| 冷启动更快 | 权重加载更小 |
| KV 并发更高 | KV dtype 降低时明显 |

但也会带来复杂性：

| 风险 | 说明 |
| --- | --- |
| 格式碎片 | 不同模型不同量化格式 |
| kernel 混杂 | batch 中模型或 shape 不一致 |
| 质量分层 | 不同租户容忍度不同 |
| 回退成本 | 高价值请求可能要路由全精 |

## 9. 质量退化也是成本

量化后如果高价值任务掉点，运营成本会吞掉硬件收益。

| 掉点类型 | 业务成本 |
| --- | --- |
| RAG 引用错 | 人工复核、用户不信任 |
| 代码补全错 | 开发者返工 |
| 数字/表格错 | 财务和合规风险 |
| 工具调用错 | agent 流程中断 |
| 动作抖动 | 机器人失败或安全风险 |

因此服务栈选择要把质量回归当成本项，而不是只算 GPU 小时。

## 10. 一个企业助手例子

目标：70B 企业助手，服务长文档问答和工具调用。

推荐路径：

1. 先用 FP16/BF16 baseline 记录质量、TTFT、TPOT、峰值显存。
2. 尝试 W4A16 或 FP8 主体，确认模型常驻显存和吞吐收益。
3. 如果 32k/128k 文档并发受限，再评估 KV cache 量化。
4. 对引用、表格、代码、工具调用、高权限操作做任务桶回归。
5. 高风险请求保留全精或混合精度路由。

## 11. 一个边缘部署例子

目标：本地代码助手，显存很小，延迟可接受但必须离线运行。

推荐路径：

1. 优先选择成熟 weight-only INT4/GGUF 类路线，让模型先放得下。
2. 控制 context 和 batch，避免 KV cache 把内存吃满。
3. 对目标语言、长文件、特殊符号和补全格式做回归。
4. 接受吞吐不如服务器，但要保证稳定加载和可复现。

## 12. 设计建议

1. 先确认瓶颈：权重、KV、compute、带宽、还是质量。
2. 先选目标 runtime 和硬件，再选它们原生支持的量化格式。
3. 不要把所有层统一压低；优先混合精度保护敏感路径。
4. 短请求和长请求分开测，平均吞吐不能代表 P99。
5. 把质量掉点和回退成本写进总拥有成本。

## 13. 小结

量化服务选型不是“哪个 bit 最低”，而是“在你的硬件、runtime、请求分布和质量门槛下，哪种精度分配最划算”。权重、KV、kernel 和业务质量要放在一张表里算，才不会把压缩率误当成系统收益。
