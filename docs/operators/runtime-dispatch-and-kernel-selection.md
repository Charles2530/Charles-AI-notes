# 运行时 Dispatch 与 Kernel 选择

很多高性能系统并不是靠“一个万能 kernel”获胜，而是靠一整套运行时 dispatch 机制：根据 shape、dtype、layout、设备类型、batch 形态和任务阶段，为当前请求挑选最合适的 kernel 路径。  
也就是说，真正的性能系统不仅要有好 kernel，还要有**挑 kernel 的能力**。

## 1. 为什么运行时选择很重要

因为现实工作负载从来不是单一 shape：

1. 长短序列混在一起；
2. prefill 和 decode 完全不同；
3. 某些模型走 BF16，某些走 INT4；
4. 有的请求带 LoRA，有的没有；
5. 多 GPU、不同卡型、不同 batch 画像同时存在。

在这种情况下，单一路径几乎不可能始终最优。

## 2. Dispatch 真正在做什么

一个较完整的 dispatch 系统通常会根据：

1. shape；
2. dtype；
3. 是否 contiguous；
4. head dim / hidden size；
5. 设备能力；
6. 是否训练或推理；
7. prefill 还是 decode；

把请求分流到不同 kernel。

### 2.1 为什么 dispatch 是性能系统的一部分

**因为如果 dispatch 错了**：

1. 大 shape 可能走到只适合小 shape 的 kernel；
2. decode 可能误走 prefill 路径；
3. 明明能命中更优融合实现，却退回通用 fallback；
4. 编译缓存命中率下降。

所以 dispatch 不是简单胶水，而是决定性能边界的关键逻辑。

## 3. Shape Family 的思维

成熟系统常会把工作负载划分成若干 shape family，例如：

1. 小 batch decode；
2. 长 prompt prefill；
3. 常见 hidden size 家族；
4. 常见 head dim；
5. 量化与非量化分支。

然后为每类 family 挑选对应 kernel。  
这比试图用一个 kernel 打天下更现实。

## 4. Fallback 的角色

运行时 dispatch 不能只考虑“最快路径”，还必须有 fallback 路径，因为：

1. 边界 shape 总会出现；
2. 新卡型或新版本未必覆盖；
3. 某些输入 layout 不满足假设；
4. 编译缓存未命中时要能继续运行。

**一个成熟系统通常是**：

1. 高频 shape 走特化；
2. 低频边界走通用实现；
3. 保证正确性优先于局部性能。

## 5. 运行时选择与编译缓存

一旦系统开始大量 shape specialization，就必须考虑：

1. 编译时间；
2. 缓存大小；
3. 首次命中成本；
4. 在线抖动。

这意味着 dispatch 不只是选最快 kernel，还要考虑：

1. 当前是否已有缓存版本；
2. 为一个低频 shape 编译新版本是否值得；
3. 是否该回退到泛化但较慢的实现。

## 6. 一个形象比喻

运行时 dispatch 就像智能交通系统的分流策略。道路网络再好，如果所有车都被导向同一条路，照样会堵；反过来，即使你有几条性能很强的专用通道，如果调度系统不会根据车流类型和道路情况分流，也发挥不出优势。Kernel 系统也是一样：真正的高性能，不只是有快路，还要有会分流的交通指挥。

## 7. 小结

运行时 dispatch 与 kernel 选择，是连接“多套实现”和“真实工作负载”的最后一层系统能力。没有它，再好的 shape-specialized kernel 也只是零散武器；有了它，系统才能根据真实请求分布把不同实现组织成一套稳定、高效、可演进的执行体系。
