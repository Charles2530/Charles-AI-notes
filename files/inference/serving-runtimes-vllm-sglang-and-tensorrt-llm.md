# 从三条共享前缀请求比较推理运行时

运行时把模型forward放进持续调度的请求系统，维护队列、已计算token、输出与KV。比较vLLM、SGLang和TensorRT-LLM时，可以先追踪相同请求如何复用状态，再检查目标版本的kernel与部署条件。

## 三条请求能共享哪些计算

设token序列分别为 `[A,B,C,D]`、`[A,B,C,E]`、`[A,B,F]`。三条共享 `[A,B]`，前两条还共享C。Prefix cache可为相同计算身份保留共享前缀KV，从分叉处继续执行。

若第三条变为 `[X,A,B,F]`，虽然中间仍出现A/B，它们的前文已不同，不能直接复用前两条A/B的因果KV。文本片段相同与前缀计算相同有明确区别。

## vLLM把调度与块管理连起来

在锁定的vLLM V1源码中，EngineCore先调用scheduler生成本轮工作，再执行模型、采样并用输出更新状态。Scheduler依据token预算和KV分配决定哪些请求推进。

PagedAttention用块表把逻辑位置映射到物理KV块，减少连续大块预留压力；continuous batching在迭代间替换已结束请求。投机与异步执行使已知、已计算、已分配长度可能不同，详见[请求生命周期](serving-systems.md)。

## SGLang的论文机制利用程序与前缀结构

[SGLang论文](https://arxiv.org/abs/2312.07104)将结构化生成程序与runtime结合，RadixAttention用前缀树组织可复用KV。上面三条请求可把共同前缀合并，分叉后的状态分别保存。

程序化多轮与并行分支让共享关系更明确，约束解码则限制可生成的token集合。这些机制并非当前某一产品永久独占的能力；其他runtime也在演进相似功能，应核对所选版本而非依据品牌推断。

## TensorRT-LLM需要同时核对图与缓存路径

[TensorRT-LLM KV文档](https://nvidia.github.io/TensorRT-LLM/features/kvcache.html)把缓存组织、复用与内存策略放进执行系统。针对NVIDIA路径的模型执行、量化kernel和并行配置，需要与具体模型、设备和版本对应。

前缀能命中、模型能加载，不代表每个shape都会选择预期kernel。输入转置、scale布局、图捕获与fallback都可能改变完整请求时间。不要把某个engine或kernel的单点结果直接当作在线服务排名。

## 同一请求集合才是公平比较起点

固定权重、tokenizer、模板、采样、最大输出、量化和adapter。先检查输出行为与任务质量，再分别测无缓存和明确缓存命中条件。若一个配置提前截短输出，延迟会变小，但完成的任务已经不同。

冷启动与稳态分别报告。开放负载下逐步提高到达率，比较相同SLO内的完成量，同时记录输入/输出长度、队列、KV占用和取消。固定并发的闭环测试可能随系统变慢自动降低到达率，应说明这一边界。

## 选型表只记录需要验证的能力

| 工作负载 | 重点验证 |
| --- | --- |
| 大量共享系统提示和分支 | cache身份、token命中、分叉和回收 |
| 长输入短输出 | prefill调度、TTFT与干扰 |
| 长输出和高并发 | KV容量、decode间隔和持续批处理 |
| 结构化工具输出 | 约束开销、正确性与重试 |
| 固定低精度硬件路径 | 实际kernel、转换、模型支持和版本 |

先用目标工作负载回答这些问题，再决定运行时。本文提供机制对照，没有在当前硬件上给出三者速度排名。

![vLLM system overview](../assets/images/paper-figures/inference/vllm-system-overview-padded.png){ width="860" }

<small>图源：[vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention](https://vllm.ai/blog/vllm)。图中把请求调度、KV block 管理和模型执行放在同一个系统里。这张图说明 runtime 的核心是请求、缓存和调度的组合。</small>

## 对照源码与延伸阅读

- [vLLM V1执行循环](https://github.com/vllm-project/vllm/blob/fa3b9d1377219247060d3c8b49eae199c9e7e154/vllm/v1/engine/core.py)
- [vLLM V1调度与KV状态](https://github.com/vllm-project/vllm/blob/fa3b9d1377219247060d3c8b49eae199c9e7e154/vllm/v1/core/sched/scheduler.py)
