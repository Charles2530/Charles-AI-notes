# 推理运行时：vLLM、SGLang 与 TensorRT-LLM

推理运行时不是“加载模型然后生成文本”的薄壳。它负责把很多用户请求组织成可调度的 GPU 工作：prefill 怎么排队，decode 怎么 batch，KV cache 怎么分配，prefix 能不能复用，kernel 是否适配硬件，SLO 违约时怎么降级。

## 核心问题

同一个模型，为什么换一个 serving runtime 后吞吐、延迟和显存会差很多？答案通常不在模型权重里，而在请求生命周期和 KV cache 管理里。

一个 LLM 请求大致分两段：

```text
prefill: 读完整 prompt，生成第一步所需的 KV cache
decode: 每次生成一个新 token，并持续读写历史 KV cache
```

Prefill 更像大矩阵计算，容易吃满 GPU；decode 每步只生成一个 token，常常受 KV cache 读写、batch 调度和小 kernel 开销限制。

## vLLM 的入口：PagedAttention

vLLM 最经典的贡献是 PagedAttention。它把 KV cache 管理类比成操作系统分页：逻辑上的 token 序列可以映射到不连续的物理 block，避免为每个请求预留一整块连续 KV 内存。

![vLLM system overview](../assets/images/paper-figures/inference/vllm-system-overview.png){ width="860" }

<small>图源：[vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention](https://vllm.ai/blog/2023-06-20-vllm)。原图表达 vLLM 在请求调度、KV cache block 管理和模型执行之间的关系。本站用它说明：serving runtime 的核心不是“模型类”，而是请求、内存和调度一起设计。</small>

![PagedAttention 原理图](../assets/images/paper-figures/inference/vllm-pagedattention.png){ width="860" }

<small>图源：[vLLM blog](https://vllm.ai/blog/2023-06-20-vllm)。原图表达 PagedAttention 把每个 sequence 的 KV cache 切成 block，并用 block table 映射逻辑 token 到物理 block。本站用它解释为什么 vLLM 能减少 KV 内存碎片并提升 batch 容量。</small>

这张图的读法很直接：不要把 KV cache 想成一条必须连续增长的大数组。PagedAttention 允许不同请求共享一套 block 池，sequence 需要更多 token 时再分配 block，结束时归还 block。这样可以减少浪费，也更适合动态 batch。

## SGLang 的入口：程序化请求和 RadixAttention

SGLang 更强调复杂生成程序：多轮 agent、工具调用、结构化输出、分支采样、prompt 复用。它关心的不只是单请求吞吐，还包括如何把“生成过程本身”表达成可优化的程序。

在这类 workload 里，很多请求共享长前缀：系统提示词、工具说明、few-shot 示例、检索模板都可能重复。RadixAttention 这类 prefix cache 思路会把共享前缀组织成树或 radix 结构，复用已经算过的 KV。

理解 SGLang 时先问：我的 workload 是单轮聊天，还是包含大量共享前缀、分支、约束解码、工具调用和多阶段生成？如果是后者，运行时对程序结构的理解就会明显影响延迟和吞吐。

## TensorRT-LLM 的入口：硬件和 kernel

TensorRT-LLM 更靠近 NVIDIA GPU 上的编译、kernel 和部署优化。它关注 fused kernel、quantization、FP8/INT8/INT4、GEMM、attention、in-flight batching、multi-GPU 通信和 engine 构建。

如果 vLLM 主要让你想到“KV 内存和动态 batch”，SGLang 让你想到“生成程序和 prefix 复用”，TensorRT-LLM 则让你想到“把模型图和 kernel 推到硬件友好的形态”。

这三者不是简单替代关系。真实选择取决于你最痛的是哪块：

| 痛点 | 更先看什么 | 原因 |
| --- | --- | --- |
| KV cache 浪费、batch 上不去 | vLLM | PagedAttention 和调度是入口。 |
| agent / tool / 结构化生成复杂 | SGLang | 程序化生成和前缀复用更关键。 |
| NVIDIA 生产部署、低比特、高性能 kernel | TensorRT-LLM | 编译和 kernel 优化空间更大。 |
| 研究快速试验 | vLLM 或 SGLang | 接入成本和灵活性通常更重要。 |

## 运行时共同要解决的生命周期

每个请求进入服务后，通常会经历：

```text
arrival -> queue -> prefill -> KV allocation -> decode loop -> streaming -> finish -> KV release
```

这里每一步都有指标：queue wait、prefill latency、time-to-first-token、inter-token latency、tokens/s、KV block usage、GPU memory watermark、batch size、SLO violation。只看总吞吐会掩盖问题：吞吐上去了，用户可能因为排队变慢；平均延迟好看，P99 可能被长 prompt 拖死。

## 读 benchmark 时先拆 workload

推理 benchmark 很容易误读。你至少要看：prompt 长度、输出长度、并发数、batch 策略、模型大小、量化格式、GPU 型号、是否 streaming、是否 prefix cache、是否启用 speculative decoding。

例如长 prompt + 短输出时，prefill 是主成本；短 prompt + 长输出时，decode 和 KV cache 是主成本；大量共享系统 prompt 时，prefix cache 价值很大；工具调用和结构化输出多时，runtime 对生成程序的表达能力会很重要。

## 外部精读

- [vLLM PagedAttention blog](https://vllm.ai/blog/2023-06-20-vllm)：理解 KV cache block 管理和动态 batch。
- [SGLang documentation](https://docs.sglang.ai/)：理解程序化生成、RadixAttention 和 serving 入口。
- [TensorRT-LLM documentation](https://nvidia.github.io/TensorRT-LLM/)：理解 NVIDIA 部署、kernel 和量化路径。
- [Hugging Face Text Generation Inference](https://huggingface.co/docs/text-generation-inference/index)：对照另一个成熟 serving 栈。
- [Orca](https://www.usenix.org/conference/osdi22/presentation/yu)：理解 iteration-level scheduling 的系统背景。
