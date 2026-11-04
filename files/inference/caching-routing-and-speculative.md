# 缓存与投机解码怎样复用已经完成的计算

两个请求共享4096个前缀token，可以尝试复用对应KV；生成时便宜模型提出几个候选，可以尝试一次让目标模型核验。两种优化都避免部分重复工作，但可复用状态和提交条件不同。

## 文本相同还需要计算身份相同

KV由模型参数、token序列、位置编码、attention规则以及可能的adapter或多模态输入共同决定。只比较可见文本，不足以确认两条请求的历史状态相同。跨不同模型的KV通常也不能直接互用。

Prefix cache命中后，可减少前缀对应的prefill计算。后续decode query仍可能读取这段历史KV，所以命中不会让后续attention成本消失。命中请求比例、命中token比例和实际节省时间应分别统计。

![vLLM logical and physical block table](../assets/images/paper-figures/inference/vllm-logical-physical-block-table.png){ width="760" }

<small>图源：[Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)，Figure 5。原图表达 logical KV blocks 如何映射到 physical KV blocks。这里用它说明：PagedAttention 主要解决的是动态请求长度导致的 KV 分配、碎片和共享问题，不是改变 attention 数学。</small>

## 活跃请求状态和闲置缓存生命周期不同

正在生成的请求还需要其历史KV，释放它会影响执行或迫使重算。已结束请求留下的前缀缓存可用于未来复用，回收它通常只影响未来命中与重算成本。

共享页需按引用关系管理，不能在一个请求结束时立即释放所有共享前缀。调度器的页分配、引用与回收决定状态是否仍有效，attention kernel只负责按给定映射计算。

## 投机核验产生临时状态

Draft先给出 `[a,b,c]`，目标模型核验各位置条件概率。若a接受、b拒绝，后续c所依赖的前缀就不成立，只能提交a与正确生成的修正token，不能继续提交c。

核验可能已经为整段候选分配或计算KV，拒绝后要回退未提交部分的状态。vLLM V1输出处理会调整相关computed-token计数；异步执行还要防止已取消请求的旧结果被错误提交。

精确概率规则与二类分布例子见[MTP与投机解码](../training/mtp-and-speculative-decoding.md)。不同树状或近似策略需要分别看其保证。

![SpecInfer workflow](../assets/images/paper-figures/inference/specinfer-workflow.png){ width="780" }

<small>图源：[SpecInfer](https://arxiv.org/abs/2305.09781)，Figure 1。原图展示 draft 模型生成 token tree，target LLM 用 tree verification 并行验证候选。这里用它说明：投机解码真正省时间的是被 target 连续接受的前缀，不是 draft 生成越多越好。</small>

## 路由改变由谁支付后续成本

请求可先交给小模型，失败后再升级大模型。若小模型成本为1，大模型为10，升级率30%，不共享计算时平均成本为 $1+0.3\times10=4$。

如果两次调用串行，升级请求的等待包含前一次尝试。质量估计器本身也有误判，不能只用平均模型成本选阈值。把可接受的错误率、长尾等待和重复计算一起评估。

![EAGLE-2 dynamic draft tree](../assets/images/paper-deep-dives/inference/eagle-2/figure-7-eagle2-dynamic-tree.png){ width="760" }

<small>图源：[EAGLE-2](https://arxiv.org/abs/2406.16858)，Figure 7。原图表达根据 confidence 和节点 value 动态扩展 draft tree，再用 tree attention mask 验证候选。这里用它说明：投机预算应该给更可能被接受的位置，而不是平均撒给所有分支。</small>

## 三者组合时要保持同一个请求语义

路由到另一模型可能导致prefix cache失效；投机策略可能增加临时KV需求；缓存命中会改变prefill与decode在总时间里的占比。单项加速倍数不能直接相乘。

固定请求分布，记录实际命中、每轮提交token、升级比例和端到端时间，再按长度与任务分桶。这样才能知道省下的是前缀计算、目标模型轮数还是昂贵模型调用。


## 对照源码与延伸阅读

- [前缀状态、投机输出与请求生命周期](https://github.com/vllm-project/vllm/blob/fa3b9d1377219247060d3c8b49eae199c9e7e154/vllm/v1/core/sched/scheduler.py)
