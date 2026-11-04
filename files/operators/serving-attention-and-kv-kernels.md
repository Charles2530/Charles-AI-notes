# 一次 decode 怎样找到历史 KV

生成一个新token时，attention需要当前query与历史K计算分数，再用分数加权V。历史长度不断增长，多个请求又共享同一显存池，因此服务kernel通常还要理解逻辑位置、物理页和有效长度。

## 先算一条短请求

假设请求已经计算了10个token的KV，页大小为4，它需要3个逻辑页。页表为 `[7,2,9]`，表示逻辑页0、1、2分别存放在物理页7、2、9。

逻辑token位置i对应

$$
page=\lfloor i/4\rfloor,\qquad offset=i\bmod4.
$$

位置9落在逻辑页2，页表给出物理页9，页内偏移1。Kernel再结合层、KV头和head dimension计算最终地址。这里物理页9与token位置9数值恰巧相同，没有语义关联。

最后一页只使用2个位置，读取attention时必须依据真实长度屏蔽剩余位置。分配了空间不表示其中KV已经计算完成。

<!-- systems-workflow-visual -->
```mermaid
flowchart LR
  I["逻辑token位置9"] --> L["9整除4=逻辑页2"] --> T["页表7,2,9 · 查第2项"] --> P["物理页9"]
  I --> O["9取模4=页内偏移1"]
  P --> A["按层、头、维度得到最终地址"]
  O --> A
```

逻辑位置和物理页编号是不同坐标系，页表承担转换；真实长度另行屏蔽无效槽位。

## 页管理和 attention 计算分工不同

调度器在执行前为待计算token分配空间，kernel在执行时写入新K/V并读取允许的历史位置。vLLM V1调度代码维护已计算token数、待处理token和缓存分配，返回输出后再更新请求状态。

共享前缀可让不同请求引用相同历史页。引用关系改变时要避免过早释放仍被其他请求使用的页，生成分叉写入时还需遵循相应共享策略。Kernel正确读页，不会自动保证管理层的引用计数正确。

## 单query扫描历史为什么昂贵

对于每个query头、T个key、头维d，QK与PV主运算约4Td FLOPs。若每个头各有独立KV且用BF16，读取K与V主体约4Td字节，理想算术强度约1FLOP/byte。

GQA让多个query头共享较少KV头，理论上可减少存储并提高复用。但实际读取是否被复用，取决于kernel工作分配和缓存。不能只把头数代入公式就断言实现达到理论强度。

## 历史太长时怎样增加并行

只有一条query却有很长的KV，单个程序可能工作过久或并行度不足。Split-K把历史分段，各段计算局部最大值、分母和加权输出，再进行一次合并。合并必须使用与[在线softmax](long-context-and-flashattention-evolution.md)相同的重标定，不能直接平均局部归一化输出。

更多分段增加并行，也增加临时统计量、合并kernel与同步。当历史较短时，这些新增开销可能超过收益，所以dispatch通常需要依据长度与batch选择路径。

## Prefix命中改变计算量，不消除后续历史读取

前缀KV可复用时，可以少做重复prefill。后续每个新query仍要关注允许的前缀历史，因此decode仍有KV读取。统计prefix命中率时，应区分请求命中、token命中和节省的计算量。

比较服务attention kernel时固定真实长度、页大小、物理布局、KV头数与dtype，分别测短历史和长历史。只测连续KV张量，不能覆盖分页索引与共享前缀的实际路径。

![PagedAttention KV mapping](../assets/images/paper-figures/inference/vllm-pagedattention.png){ width="820" }

<small>图源：[PagedAttention paper](https://arxiv.org/abs/2309.06180)。原图表达 logical KV blocks 到 physical KV blocks 的映射；读图重点是服务态 attention 的核心难点是动态 KV 内存、page lookup 和连续批处理，不只是 attention 数学。它不能证明 page size 或 block size 的唯一最佳值。</small>

## 对照源码与延伸阅读

- [V1调度器的KV分配与computed token状态](https://github.com/vllm-project/vllm/blob/fa3b9d1377219247060d3c8b49eae199c9e7e154/vllm/v1/core/sched/scheduler.py)
- [分块attention的稳定归约](https://github.com/triton-lang/triton/blob/cd2e01c6686d6503802e6cdfac56d5a5dfdb27a1/python/tutorials/06-fused-attention.py)
