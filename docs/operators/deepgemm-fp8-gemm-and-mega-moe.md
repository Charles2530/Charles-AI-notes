# DeepGEMM 解读：FP8 GEMM、JIT 与 Mega MoE

`DeepGEMM` 值得单独讲，不只是因为它来自 `DeepSeek`，而是因为它把很多现代大模型算子工程里最关键的东西放进了一个非常集中的样本里：**FP8 GEMM、细粒度 scaling、Hopper/Blackwell 硬件特化、运行时 JIT、MoE grouped GEMM、decode 场景 masked layout，以及从 dense GEMM 一路扩到 Mega MoE 的统一内核思路**。

如果你想理解“为什么今天的大模型算子优化越来越像系统设计”，`DeepGEMM` 是一个非常好的切口。

## 1. 先给一个总判断：`DeepGEMM` 到底是什么

按照官方仓库的定义，`DeepGEMM` 是一个统一的高性能 tensor core kernel 库，把现代大语言模型里的一组关键计算原语放到了同一套 CUDA 代码库中，包括 `FP8 / FP4 / BF16 GEMM`、MoE 相关 grouped GEMM、`MQA` 打分内核、`Mega MoE` 等，并且采用了轻量级运行时 `JIT` 编译，而不是安装时把所有 kernel 全部预编译好。

更工程化地说，可以把它理解成：

1. **不是通用线代库**，而是面向现代 LLM / MoE 工作负载的特化库；
2. **不是纯模板库**，而是尽量压低模板复杂度、强调可读和可学的 Hopper/Blackwell 优化实现；
3. **不是只做 dense GEMM**，而是把 GEMM、grouped GEMM、部分 attention 相关 scoring kernel 与 MoE 融合路径串在一起；
4. **不是把“高性能”理解成单个 benchmark 冲榜**，而是强调一整组真实 shape 下的统一调度与实现策略。

## 2. 为什么它在算子专题里值得单独拿出来

`DeepGEMM` 的独特之处，在于它刚好横跨了几条常见但常被割裂讨论的主线：

| 主线 | `DeepGEMM` 体现了什么 |
| --- | --- |
| **低精度 GEMM** | 不是只把输入改成 `FP8`，而是把 scaling、累加、布局和硬件约束一起重写 |
| **硬件特化** | 直接围绕 `SM90/SM100`、`TMA`、tensor core、JIT 常量折叠去设计 |
| **MoE kernel** | 把 grouped GEMM、masked layout、甚至 `Mega MoE` 这种通信和计算叠加场景带入核心设计 |
| **运行时系统** | 通过 JIT、cache、环境变量和 profiling 开关，把 kernel 库做成“可部署的系统组件” |
| **可学习性** | 官方强调“少量核心 kernel 函数”和较低模板负担，便于学习 Hopper 优化技巧 |

这意味着它不是一个只能放在“论文配套代码”栏目里的仓库，而是非常适合拿来理解下面这些问题：

1. 为什么 `FP8 GEMM` 的难点不在公式而在实现；
2. 为什么 MoE 场景会逼着 kernel API 重新设计；
3. 为什么运行时 JIT 能在某些场景比大而全的预编译方案更合理；
4. 为什么今天的高性能算子库越来越需要和服务框架、通信库、CUDA Graph、对象缓存一起考虑。

## 3. 它和 `cuBLASLt`、`CUTLASS`、`CuTe`、`Triton` 是什么关系

这是理解 `DeepGEMM` 的第一层。

### 3.1 它不是在否定现有库，而是在选一个更聚焦的落点

可以把几类工具粗略放在一张图里：

| 工具 | 更像什么 | 强项 | 代价 |
| --- | --- | --- | --- |
| `cuBLAS/cuBLASLt` | 官方高性能通用库 | 稳、覆盖广、集成成熟 | 对特殊场景可控性不足 |
| `CUTLASS / CuTe` | 模板化高性能构件系统 | 表达力强，适合深度定制 | 模板复杂度高，学习曲线陡 |
| `Triton` | 块级张量编程与快速原型 | 开发效率高，原型速度快 | 极限硬件特化和某些特殊场景不一定最优 |
| `DeepGEMM` | 面向 LLM/MoE 的轻量 Hopper/Blackwell 特化库 | 针对目标负载聚焦，代码相对集中，JIT 友好 | 硬件和布局假设更强，通用性较弱 |

**所以 `DeepGEMM` 真正的站位**不是“替代所有 GEMM 库”，而是提供一条非常清晰的、面向现代大模型热点工作负载的特化实现路径。

### 3.2 它借鉴 `CUTLASS / CuTe`，但不重度依赖其模板体系

官方 README 明确提到，它借鉴了 `CUTLASS` 和 `CuTe` 的一些思想，但避免过重依赖它们的模板和代数体系。这个选择背后有一个很重要的工程态度：

1. **保留成熟优化思想**，比如 tile、pipeline、persistent thread specialization 这些共通模式；
2. **降低阅读门槛**，让读者更容易直接看到 kernel 真正在干什么；
3. **把主要复杂度集中在 JIT 配置与 kernel 设计上**，而不是埋进极深的模板展开里。

这也是为什么很多人把 `DeepGEMM` 当成“学习 Hopper FP8 GEMM 的良好样本”，而不是只把它当 benchmark 工具。

## 4. `DeepGEMM` 的第一关键词不是 FP8，而是“细粒度 scaling 的 FP8”

很多人第一次听说 `DeepGEMM`，会把关注点放在 “FP8 GEMM”。但更关键的其实是：**它服务的是带细粒度 scaling 的低精度 GEMM 路线**。

### 4.1 为什么 FP8 不是把 dtype 改一下就完了

低精度 GEMM 的真正问题，通常不在矩阵乘公式本身，而在：

1. **输入张量如何量化**；
2. **scale 因子如何组织和存放**；
3. **累加在哪里用更高精度做补偿**；
4. **输出写回前是否需要 promotion 或后处理**；
5. **整个流程如何在 tensor core 和 CUDA core 之间协调**。

官方资料和第三方说明都强调了一个关键点：`DeepGEMM` 为了缓解 `FP8` tensor core 累加不够精确的问题，引入了 **two-level accumulation / promotion** 这一类设计。也就是说，它不是盲目追求所有阶段都用最低精度，而是在性能和数值之间做局部补偿。

### 4.2 为什么 scaling layout 反而成了接口的一部分

`DeepGEMM` 的 README 里明确写到，`lhs` 的 scaling factor 需要是 **TMA-aligned 且转置后的布局**，不同架构下 scaling factor 的格式也不同：

1. `SM90` 要求 `FP32` scaling factor；
2. `SM100` 则引入了打包的 `UE8M0` 格式。

这说明一个很重要的事实：**现代低精度 GEMM 的接口，不再只有 A/B/C/D 张量本身，量化比例、对齐、布局和 pack 规则已经进入一等公民地位。**

如果系统设计还把 GEMM 当成“只喂两个矩阵就完事”，就会低估真实集成成本。

## 5. `DeepGEMM` 的第二关键词是“Fully JIT”

官方仓库和 DeepEP 页面都强调，`DeepGEMM` 采用**完全运行时 JIT** 设计，不在安装阶段编译全部 CUDA kernel。

### 5.1 为什么这个设计有现实吸引力

对这类强 shape 特化、强硬件假设的库来说，JIT 有几个明显优势：

1. **把 shape、block size、pipeline stage 等参数当作编译期常量**，便于编译器做更激进优化；
2. **减少安装时的编译和分发负担**，不用提前把巨量 shape 组合打包；
3. **让用户负载驱动 kernel 生成**，更贴近真实线上 workload；
4. **更容易做 cache 和按需编译**，避免大量用不到的 kernel 占空间。

### 5.2 但 JIT 不是没有代价

JIT 带来的代价也很直接：

| 代价 | 体现方式 |
| --- | --- |
| **首次延迟** | 某个 shape 第一次出现时要编译 |
| **cache 管理复杂** | 不同版本、不同驱动、不同参数组合都可能影响缓存命中 |
| **诊断复杂** | 线上性能波动可能来自 JIT 行为而不是 kernel 本体 |
| **可重复性成本** | 不同编译器路径可能造成细微性能差异 |

所以 `DeepGEMM` 的 Fully JIT 设计更像一种**面向特定工作负载的系统权衡**，而不是任何库都应照抄的标准答案。

### 5.3 为什么它在大模型服务里特别合理

因为大模型服务的热点 shape 往往并不是完全随机的。很多场景会出现：

1. 固定 hidden size；
2. 固定 head dim；
3. 若干常见 batch / token bucket；
4. MoE 专家维度固定、只有 token 数在变；
5. CUDA Graph 复用导致 shape 模式高度集中。

这时，JIT 的第一次成本可以被 amortize，而编译期特化带来的寄存器节省、常量折叠和调度优化，反而能在热路径里持续回本。

## 6. 它为什么强绑定 Hopper / Blackwell 这代硬件

`DeepGEMM` 当前重点支持 `SM90` 和 `SM100`。这不是简单的兼容声明，而是说明它把很多优化建立在这些架构特征上。

### 6.1 `TMA` 是理解它的第一把钥匙

DeepEP 页面把 `TMA` 放在非常显眼的位置。原因很简单：在 Hopper 及之后的架构里，**Tensor Memory Accelerator** 能显著改善异步数据搬运路径，包括 load、store、broadcast、descriptor prefetch 等。

这意味着 `DeepGEMM` 的优化逻辑并不是“多写几条 MMA 指令”，而是把**数据移动**和**tensor core 计算**一起纳入流水线设计。

### 6.2 Persistent Thread Specialization 说明它在做真正的流水线组织

DeepEP 页面也强调了 persistent thread specialization。这个词背后的核心不是花哨术语，而是：

1. 用持久化线程块减少调度抖动；
2. 让不同 warpgroup 在加载、计算、promotion 等阶段分工协作；
3. 让数据移动、tensor core MMA 和 CUDA core 上的补偿计算彼此重叠。

**这和“写一个普通 tiled GEMM”已经不是同一层问题了。** 前者是把 kernel 当小程序写，后者是把 kernel 当微型流水线系统设计。

### 6.3 它还显式利用了“不完全对齐的块大小”

DeepEP 页面给了一个很典型的例子：例如在 `M=256, N=7168` 的情况下，使用像 `BLOCK_N=112` 这类**非整齐块大小**，可能比常见的对齐块更能把更多 SM 喂起来。

这说明 `DeepGEMM` 的设计并不迷信“块大小必须规整好看”，而是更关注：

1. **SM 实际利用率**；
2. **目标 shape 的覆盖效率**；
3. **L2 复用和 block scheduler 行为**。

这类思路很重要，因为很多工程师做 GEMM 调优时，容易过度偏向“漂亮对齐”，却忽略真实业务 shape 的离散性。

## 7. `DeepGEMM` 的核心接口为什么值得看

光看 README 就能看到，`DeepGEMM` 的接口设计明显带着强工作负载假设。

## 7.1 Dense GEMM：不是万能接口，而是明确约束的高性能接口

它最基础的 dense GEMM 接口写法类似 `fp8_gemm_{nt, nn, tn, tt}`。这类接口直接把**内存布局**暴露在名字里，而不是完全藏在通用参数后面。

这背后的思想很朴素：**既然布局会强烈影响实现，那就不假装布局无关。**

这比很多“看起来通用、实际内部还在疯狂转置和 pack”的接口更诚实。

## 7.2 Grouped GEMM：为什么只按 M 轴分组

官方 README 特别说明了一点：与一些传统 grouped GEMM 不同，`DeepGEMM` 的 grouped GEMM 重点是 **M 轴分组**，而 `N`、`K` 保持固定。这很适合 `MoE` 专家同 shape、token 数不同的场景。

这其实正中现代 MoE 的典型需求：

1. 每个 expert 的权重 shape 一样；
2. 每个 expert 这一轮收到的 token 数不一样；
3. 训练 forward 或 prefill 时，往往可以把属于不同 expert 的 token 先拼接起来；
4. 然后按 expert 段做 contiguous layout 的 grouped GEMM。

**所以这不是一个“数学上更通用”的 grouped GEMM 设计，而是一个“更贴合真实 MoE 数据流”的 grouped GEMM 设计。**

## 7.3 Masked layout：为什么它特别适合 decode + CUDA Graph

README 还专门给了 masked grouped GEMM 的接口，并指出一个重要背景：**在 decode 阶段，如果启用了 CUDA Graph，CPU 未必知道每个 expert 当前收到多少 token。**

这句话很值钱，因为它直接把算子接口和服务运行时连起来了。

这意味着 `DeepGEMM` 并不是孤立在 kernel 世界里思考问题，而是考虑了：

1. 推理服务是图捕获的；
2. CPU 不一定能动态参与每一步 shape 决策；
3. expert token 数是运行时变化的；
4. 算子需要在保持高性能的同时，适应这种“形状部分未知”的服务约束。

从系统视角看，这比单纯写一个离线 benchmark 很强得多。

## 8. 为什么说它已经不只是 GEMM 库了

如果只看仓库标题，很容易把 `DeepGEMM` 理解成“FP8 GEMM 库”。但从当前 README 看，它已经明显超出了这个边界。

### 8.1 `MQA` scoring kernel 说明它在向服务态 attention 延伸

官方仓库已经列出了 `MQA logits` 相关内核，包含 paged/non-paged 两类版本。这个方向说明两件事：

1. `DeepGEMM` 已经开始覆盖**推理服务态的注意力相关热点**；
2. 它不再只关心大块 dense GEMM，而在吸收更细粒度、更贴近线上索引与检索流程的算子。

### 8.2 `Mega MoE` 说明它在向“融合通信 + 计算”的 mega-kernel 走

README 对 `Mega MoE` 的描述更值得注意：它把 `EP dispatch`、第一层线性、`SwiGLU`、第二层线性和 `EP combine` 融成单个 mega-kernel，并明确提到**重叠 NVLink 通信和 tensor core 计算**。

这意味着 `DeepGEMM` 这条线已经在进入一个更高级的算子阶段：

1. 不再只优化单个 GEMM；
2. 而是把**MoE 层完整的数据流**当成优化对象；
3. 并把**跨卡通信**视为算子内部需要协同的部分；
4. 最终目标不是单个 kernel 快，而是整个 `dispatch -> compute -> combine` 路径更快。

这正是现代大模型算子工程最值得注意的趋势之一。

## 9. `DeepGEMM` 的性能到底该怎么读

第三方页面给出了诸如“某些 shape 下 2.7x 提速”“H800 上 1358 TFLOPS 以上”这类性能数字，官方 README 里也列出了更近期的进展，例如在 2026 年 4 月新增 `Mega MoE`、`FP8xFP4 GEMM`、`FP4 Indexer`，并在 2025 年 4 月提到 H800 上可达 1550 TFLOPS 量级的结果。

### 9.1 第一原则：不要把它读成“所有 GEMM 都比所有库快”

更合理的理解是：

1. 它对**目标 shape**和**目标工作负载**做了很深的特化；
2. 它在这些场景里可以达到或超过专家调优库；
3. 这种优势往往来自**layout 假设、JIT 特化、调度器设计、硬件特性利用和数值补偿路径**的共同作用；
4. 脱离这些前提，优势不一定成立。

### 9.2 第二原则：看性能时要连同“代价模型”一起看

评估 `DeepGEMM` 是否适合接入真实系统时，除了看 TFLOPS，还要看：

| 维度 | 应该问什么 |
| --- | --- |
| **首次编译成本** | 热门 shape 的 JIT 代价能否接受 |
| **缓存命中率** | 线上热点 shape 是否足够集中 |
| **layout 改造成本** | 上游是否要额外转置、pack、scale 变换 |
| **数值行为** | 与现有训练或推理路径的一致性如何 |
| **维护成本** | CUDA / 驱动 / 架构升级后是否容易跟进 |

不把这些算进去，只盯一个峰值性能，很容易得出过度乐观结论。

## 10. 从学习角度看，`DeepGEMM` 最适合拿来学什么

如果你是做算子或推理系统的人，`DeepGEMM` 特别适合拿来学以下几件事。

### 10.1 学“怎么把 workload 假设写进 API”

这点比很多人想的重要。`DeepGEMM` 没有假装所有场景都一样，而是把：

1. 内存布局；
2. grouped 方式；
3. scale 布局；
4. TMA 对齐；
5. masked layout；
6. paged / non-paged 场景

这些真实约束直接写进接口。这能帮助你建立一个很重要的意识：**优秀算子库的接口设计，本身就是性能设计的一部分。**

### 10.2 学“怎么把硬件特性转成代码结构”

`TMA`、persistent thread specialization、两级累加、block scheduler、rasterization、SASS interleaving，这些词如果只停在 PPT 上，很容易流于口号。`DeepGEMM` 的价值是把这些东西尽量落成了相对集中、可追踪的实现逻辑。

### 10.3 学“怎么从 dense GEMM 走到 fused system kernel”

很多人能接受优化 GEMM，但一到 MoE 融合、通信重叠、对称内存、CUDA Graph 这类问题，就会觉得那是另一门学问。`DeepGEMM` 证明它们其实是连续的：

1. 先理解单个 GEMM；
2. 再理解 grouped GEMM；
3. 再理解 masked grouped GEMM；
4. 最后才能自然走向 `Mega MoE` 这种系统级 mega-kernel。

## 11. 它的边界和风险也必须说清

`DeepGEMM` 很强，但并不是一个“人人都该直接依赖”的万能答案。

### 11.1 硬件绑定很强

从当前官方说明看，它重点支持 `SM90/SM100`。这意味着：

1. 对较老 GPU 的覆盖有限；
2. 很多优化建立在 Hopper / Blackwell 特性之上；
3. 如果你的主力部署平台不是这两代硬件，集成价值要重新评估。

### 11.2 上游数据流必须愿意配合它的布局假设

例如 TMA 对齐、scale 布局、grouped M 轴约束、某些转置或 pack 操作需要上游单独处理。这些都说明：

**`DeepGEMM` 不是“扔进去任意张量就会自动变快”的黑盒。**

### 11.3 JIT 让它更灵活，也让运维更复杂

如果你在服务环境里使用它，就要认真对待：

1. 编译缓存目录；
2. 版本升级后的 cache 失效；
3. 外部 profiler 与内部 profile 的关系；
4. 不同编译器路径的性能漂移；
5. 首次热身和发布预编译策略。

这些是纯离线 benchmark 看不到，但线上一定会遇到的问题。

## 12. 如果你要把它放回整个算子专题，应该怎么理解它

可以把 `DeepGEMM` 放到这样一条链路里：

1. 它的**底层硬件逻辑**，对应 [CUDA 编程模型与内存层次](cuda-programming-model-and-memory.md)；
2. 它和 `Triton` 的对照，适合配合 [Triton 编程模型与自动调优](triton-programming-model-and-autotuning.md) 一起看；
3. 它最核心的热点仍是 GEMM / grouped GEMM / MoE 路径，对应 [GEMM、Attention 与融合 Kernel](gemm-attention-and-fused-kernels.md)；
4. 它借鉴而不重依赖 `CUTLASS / CuTe` 的做法，适合和 [CUTLASS、CuTe 与编译栈](cutlass-cute-and-compiler-stack.md) 对照阅读；
5. 它的大量性能判断最终仍要回到 [Profiling、调试与数值稳定性](profiling-debugging-and-numerical-stability.md) 中那套方法论。

换句话说，`DeepGEMM` 不是孤立案例，而是把算子专题的几条主线浓缩到同一个真实工程样本里。

## 13. `DeepGEMM vs CUTLASS vs Triton` 怎么选

这三个名字经常一起出现，但它们解决的问题并不在同一层。真正实用的选法，不是问“谁更先进”，而是问：**你现在面对的是哪类 workload、哪类硬件、哪类团队约束。**

### 13.1 一句话定位

| 方案 | 更适合的定位 | 最强优势 | 最常见代价 |
| --- | --- | --- | --- |
| `DeepGEMM` | 面向 `FP8 / MoE / Hopper / Blackwell` 的强特化生产内核 | 对目标场景打得深，JIT + 硬件特化收益高 | 硬件绑定强、布局前提多、通用性有限 |
| `CUTLASS / CuTe` | 面向高性能 CUDA kernel 的模板化基础设施 | 表达力极强，适合构建长期维护的高性能内核体系 | 模板复杂、学习成本和维护门槛高 |
| `Triton` | 面向张量算子的高效原型与中层实现 | 开发迭代快，适合快速验证和中等复杂度融合 | 对极限硬件特化和复杂服务态场景不一定最优 |

**可以把它们粗略理解为**：

1. `Triton` 更像高效率原型工坊；
2. `CUTLASS / CuTe` 更像高性能构件系统；
3. `DeepGEMM` 更像已经围绕特定大模型热点负载打磨过的成品内核库。

### 13.2 按目标来选，比按“流行度”来选更靠谱

| 你的目标 | 更优先考虑 | 原因 |
| --- | --- | --- |
| **先把一个新算子想法快速跑通** | `Triton` | 写得快，改得快，profile 闭环短 |
| **做长期维护的高性能 CUDA 基建** | `CUTLASS / CuTe` | 适合沉淀成模板化资产，覆盖面更广 |
| **在 `Hopper / Blackwell` 上把 `FP8 GEMM / MoE` 打到很深** | `DeepGEMM` | 已经围绕这类负载做了很多特化假设 |
| **做多形状、多硬件的通用库** | `CUTLASS / CuTe` 或官方库 | 比强工作负载特化库更通用 |
| **做研究验证或中小规模服务实验** | `Triton` | 试错成本低，适合快速迭代 |

### 13.3 按团队能力来选

这部分往往比技术本身更现实。

| 团队条件 | 更适合的路线 | 为什么 |
| --- | --- | --- |
| **有 CUDA 资深内核工程师，且愿意长期维护** | `CUTLASS / CuTe` 或深入接 `DeepGEMM` | 能吃下更深的模板和硬件特化复杂度 |
| **系统工程师偏多，希望快速打原型** | `Triton` | 抽象层更友好，迭代更快 |
| **部署平台高度集中在 H100/H800/B100 这类新卡** | `DeepGEMM` | 强硬件特化更容易回本 |
| **平台异构，老卡新卡混用** | `Triton` + 官方库 / `CUTLASS` | 比单一特化库更容易维持兼容性 |

### 13.4 按工作负载形态来选

| 工作负载 | 更值得优先看的方案 | 说明 |
| --- | --- | --- |
| **普通 dense GEMM、常见 fusion** | `Triton` 或官方库 | 不一定需要一上来就用最重方案 |
| **复杂 epilogue、特殊 layout、高度定制 CUDA kernel** | `CUTLASS / CuTe` | 模板系统更能承载复杂组合 |
| **MoE grouped GEMM、masked grouped GEMM、Mega MoE** | `DeepGEMM` | 这正是它最有特色的区域 |
| **服务态 decode 热点、shape 很集中** | `DeepGEMM` 或特化 CUDA 实现 | JIT 和 shape specialization 更可能回本 |
| **形状变化大、负载模式还不稳定** | `Triton` | 先把问题摸清楚比过早极限优化更重要 |

### 13.5 一个更实用的决策顺序

如果你在真实项目里做选型，建议按下面顺序判断：

1. **先问是不是热点**：不是热点，别急着上重武器；
2. **再问 shape 是否稳定**：shape 稳定才更适合强特化与 JIT；
3. **再问硬件是否集中**：硬件越集中，`DeepGEMM` 这类方案越容易回本；
4. **再问上游数据流能否配合**：如果 layout、pack、scale 前提都配合不了，再强的库也接不进去；
5. **最后问团队能维护哪一层复杂度**：选一个团队长期养不起的方案，通常不是好方案。

### 13.6 一个常见的落地策略

很多成熟团队不会在一开始就三选一，而是分层使用：

1. **先用 `Triton` 做原型与小规模验证**；
2. **当模式稳定后，用 `CUTLASS / CuTe` 或更深的 CUDA 实现做长期高性能版本**；
3. **如果 workload 恰好高度命中 `DeepGEMM` 的假设**，则直接把它当特化生产库接入。

这条路线的价值在于：**先降低试错成本，再逐步提高性能上限。**

### 13.7 几个常见误判

**误判 1：`DeepGEMM` 更快，所以一切 GEMM 都该用它。**

不对。  
它的优势建立在明显的硬件、布局和 workload 假设上，脱离这些前提，收益未必成立。

**误判 2：`Triton` 能写出来，就没必要看 `CUTLASS / CuTe`。**

也不对。  
`Triton` 很强，但在某些极限硬件特化、复杂模板复用和长期底座建设上，`CUTLASS / CuTe` 仍然很重要。

**误判 3：`CUTLASS / CuTe` 太复杂，所以不值得学。**

短期看也许成立，长期看不成立。  
如果团队要长期维护高性能 CUDA 基建，迟早要理解这层抽象。

### 13.8 一个最终判断

**如果你更关心**：

- 快速实验和中层开发效率，优先看 `Triton`；
- 长期高性能库建设，优先看 `CUTLASS / CuTe`；
- `Hopper / Blackwell + FP8 + MoE` 的真实生产热点，优先看 `DeepGEMM`。

最现实的答案通常不是“三选一”，而是：**让 `Triton`、`CUTLASS / CuTe`、`DeepGEMM` 分别落在最适合它们的层级上。**

## 14. `DeepGEMM` 和 `cuBLASLt / cublasLtMatmul` 到底是什么关系

如果前面那节是把 `DeepGEMM` 放到 `CUTLASS / CuTe / Triton` 这条开发路径里比较，那么这一节更像是在回答另一个更常见的问题：

**既然 NVIDIA 官方已经有 `cuBLASLt`，为什么还会出现像 `DeepGEMM` 这样的库？**

### 14.1 先说结论：两者不是一个抽象层

更准确地说：

1. `cuBLASLt` 是 **NVIDIA 官方的高性能 matmul 前端与算法选择框架**；
2. `cublasLtMatmul()` 是其中最核心的执行入口；
3. `DeepGEMM` 则是 **围绕特定大模型热点工作负载写死了更多前提的特化 kernel 库**。

换句话说，`cuBLASLt` 更像一套**高度参数化的通用 matmul 服务接口**，而 `DeepGEMM` 更像一组**假设更强、但路径更短的专用内核**。

### 14.2 `cuBLASLt` 强在哪里

从 NVIDIA 官方文档看，`cuBLASLt` 的核心能力主要包括：

| 能力 | 官方机制 | 工程意义 |
| --- | --- | --- |
| **算法选择** | `cublasLtMatmulAlgoGetHeuristic()` 与 heuristics cache | 可以根据 shape、layout、dtype、GPU 情况选择更合适的 kernel |
| **丰富布局描述** | matrix layout descriptor、order、ld、batch stride | 适合通用场景和复杂布局组合 |
| **低精度支持** | `scaleA / scaleB / scaleC / scaleD` 与 `FP8` matmul 描述 | 官方支持量化/反量化尺度参与计算 |
| **epilogue** | bias、ReLU、GELU、aux 输出等 | 适合大量常见 post-op 融合 |
| **缓存与重用** | heuristics cache | 减少 host 侧重复选算法的开销 |
| **同步协同** | atomics synchronization | 在某些 producer / consumer 模式下支持更细粒度协同 |

**所以 `cuBLASLt` 的长处**不是“某一个 kernel 永远最快”，而是它作为官方组件，提供了：

1. 更广的 shape 与 layout 覆盖；
2. 更完整的 descriptor 体系；
3. 更稳定的官方维护路径；
4. 更容易和通用框架、推理引擎与现有 CUDA 生态对接的接口。

### 14.3 那 `DeepGEMM` 还补了什么空

`DeepGEMM` 的价值，不在于否认 `cuBLASLt`，而在于它为某些工作负载做了更激进的特化：

1. **更强的 workload 假设**：例如 `MoE` 里只按 `M` 轴分组、masked grouped GEMM、`Mega MoE` 融合链路；
2. **更强的硬件特化**：直接围绕 `SM90/SM100`、`TMA`、persistent thread specialization 展开；
3. **更强的运行时 shape specialization**：通过 JIT 把部分参数编译期常量化；
4. **更贴近服务态约束**：例如 decode + CUDA Graph + expert token 数 CPU 不可知的 masked layout；
5. **更明确的系统整合意图**：例如把 `EP dispatch -> compute -> combine` 放进同一条优化链。

也就是说，`DeepGEMM` 不是在说“官方库不行”，而是在说：

**当 workload 足够集中、硬件足够统一、目标足够明确时，通用官方接口未必是最短路径。**

### 14.4 两者的根本区别：谁来承担“通用性”的成本

这是最值得理解的一点。

| 问题 | `cuBLASLt` 更偏向 | `DeepGEMM` 更偏向 |
| --- | --- | --- |
| **shape 覆盖** | 尽量广覆盖 | 聚焦高价值热点 shape |
| **layout 支持** | descriptor 驱动、尽量通用 | 接口层直接暴露并约束布局 |
| **算法选择** | heuristics + cache | JIT + 特化 kernel 路线 |
| **epilogue** | 官方定义的融合集合 | 面向特定链路做更深定制或外部拼装 |
| **硬件适配** | 官方统一维护 | 聚焦某几代 GPU 深挖 |
| **系统接口** | 更偏通用库 | 更偏 LLM/MoE 特定运行流 |

**更直白地说**：

- `cuBLASLt` 把更多复杂度放进**通用描述和算法选择**；
- `DeepGEMM` 则把更多复杂度放进**特定 workload 的前提假设与专用 kernel**。

### 14.5 为什么 `cublasLtMatmul` 对很多系统仍然是默认第一选择

因为在很多真实项目里，首先追求的是：

1. **稳定可用**；
2. **支持范围广**；
3. **和框架/编译器/推理引擎生态衔接顺滑**；
4. **维护成本可控**；
5. **不用为了少数 shape 大改上游数据流**。

这正是 `cublasLtMatmul` 的典型优势。  
特别是在下面这些情况下，它通常是更自然的起点：

1. shape 还不稳定；
2. 硬件平台不够统一；
3. 需要广泛支持不同 epilogue；
4. 需要快速接进成熟框架；
5. 团队不想单独维护强特化 CUDA 库。

### 14.6 什么时候 `DeepGEMM` 更可能赢

更典型的是下面这些场景：

1. **`Hopper / Blackwell` 占比很高**；
2. **热点集中在 `FP8 GEMM`、`MoE grouped GEMM`、服务态 `MoE decode`**；
3. **shape 和 layout 很稳定**；
4. **上游数据流可以配合它的 scale / pack / 对齐要求**；
5. **团队愿意为这条热路径单独维护一套特化库**。

这时，`DeepGEMM` 的强特化假设反而会变成优势，因为你不再需要为不相关的通用场景付出那么多抽象成本。

### 14.7 一个更实用的关系图

可以把两者关系理解成：

```text
通用框架 / 推理引擎
        |
        |-- 默认通用 matmul 路线 -> cuBLASLt / cublasLtMatmul
        |
        |-- 极热点特化路线 -----> DeepGEMM
```

很多成熟系统最后并不是二选一，而是：

1. **通用路径继续交给 `cuBLASLt`**；
2. **最热的少数 `FP8 / MoE` 路径交给 `DeepGEMM`**；
3. **外围 glue、layout transform、runtime dispatch 再由系统层统一管理**。

这比试图让任何一个库包办所有事情更现实。

### 14.8 一个常见误判

**误判**：`DeepGEMM` 比 `cublasLtMatmul` 更现代，所以应该全面替换。

这通常不成立。  
更合理的理解是：

1. `cuBLASLt` 是默认通用底座；
2. `DeepGEMM` 是在特定热点上“切出去”的深特化方案；
3. 谁更合适，取决于你愿不愿意为少数热点承担额外前提和维护成本。

## 15. `DeepGEMM` 在 `MoE serving` 里的调用链怎么理解

这一节把它放回真实系统链路里看。因为如果不看调用链，很容易把 `DeepGEMM` 误解成“只是把 GEMM 写快了一点”。实际上，在 `MoE serving` 里，它更像一段嵌进服务运行时的数据流节点。

### 15.1 一个最简化的 `MoE serving` 主链

先看不考虑优化细节的逻辑链：

```text
输入 hidden state
  -> router 打分
  -> 选 top-k experts
  -> 按 expert 重排 token
  -> expert 内部线性层 1
  -> 激活 / 门控
  -> expert 内部线性层 2
  -> 按原 token 顺序 combine
  -> 输出 hidden state
```

在这条链里，真正麻烦的不是“有两层线性”，而是：

1. token 要按 expert 重排；
2. 每个 expert 收到的 token 数都不同；
3. dispatch 与 combine 经常跨卡；
4. decode 阶段 batch 很小、形状抖动大；
5. CUDA Graph 又要求运行时形状管理更克制。

这正是 `DeepGEMM` 这类库开始发挥作用的地方。

### 15.2 Prefill 和 Decode 的调用链不一样

理解 `DeepGEMM` 在 `MoE serving` 里的位置，必须把 `prefill` 和 `decode` 分开看。

| 阶段 | 更常见的特点 | `DeepGEMM` 更可能用什么接口 |
| --- | --- | --- |
| **Prefill** | token 多、吞吐优先、CPU 通常知道每个 expert token 数 | contiguous grouped GEMM |
| **Decode** | token 少、时延优先、CUDA Graph 常见、CPU 未必知道 expert token 数 | masked grouped GEMM 或更深的融合路径 |

这也是为什么 README 会同时提供：

1. **contiguous layout grouped GEMM**；
2. **masked layout grouped GEMM**。

它们不是两个随便并列的 API，而是分别对应两种典型服务态负载。

### 15.3 一个更贴近真实实现的调用链

在 MoE 推理里，链路往往更像下面这样：

```text
1. router 产生 top-k expert id 和权重
2. dispatch kernel / 通信层把 token 按 expert 分发
3. 形成 contiguous 或 masked layout
4. 调用 expert 线性层 GEMM
5. 做 SwiGLU / 激活
6. 调用 expert 第二层 GEMM
7. combine kernel / 通信层把结果按 token 合回
8. 写回主干 hidden state
```

如果用普通思路做，这里面可能会拆成很多小阶段：

1. dispatch 一个 kernel；
2. 再做一次 layout transform；
3. 再调一次 GEMM；
4. 再调一个激活；
5. 再调第二次 GEMM；
6. 再做 combine；
7. 还可能穿插跨卡同步。

这样一拆，**launch 开销、HBM 往返、跨卡等待和中间 buffer 写回**都会变重。

### 一张文本时序图：decode 路径里的 `MoE serving`

如果把 `DeepGEMM` 放回一次真实 decode step，可以用下面这张简化时序图来理解它在系统中的位置：

```text
User Request
    |
    v
Serving Runtime
    |
    | 1. 取本轮 active requests，准备 decode token
    v
Router Kernel / Router Head
    |
    | 2. 计算 top-k experts 与 gate weight
    v
Dispatch / DeepEP
    |
    | 3. 按 expert 重排 token
    | 4. 如有跨卡 expert，发起 NVLink / EP 通信
    v
Layout Builder
    |
    | 5. 形成 contiguous layout 或 masked layout
    | 6. 准备 grouped GEMM 所需索引 / mask / scale
    v
DeepGEMM Kernel
    |
    | 7. expert linear 1
    | 8. SwiGLU / gating
    | 9. expert linear 2
    |    或进一步进入 Mega MoE 融合路径
    v
Combine / DeepEP
    |
    | 10. 将 expert 输出按 token 原顺序聚合
    | 11. 必要时完成跨卡回收与 combine
    v
Main Transformer Stream
    |
    | 12. 回到主干残差流
    v
Next Layer / Output Token
```

这张图里最该注意的是两点：

1. `DeepGEMM` 不在链路开头，也不在链路结尾，它卡在 **dispatch 之后、combine 之前** 的核心 expert 计算区；
2. 它真正优化的不是“孤立 GEMM”，而是 **router 之后这段最热、最碎、最容易被通信和小 shape 拖慢的计算带**。

### 进一步把图读成系统问题

如果把这张时序图翻译成工程语言，可以得到四个关键判断：

1. **router 决定 token 如何碎片化**；
2. **dispatch / combine 决定通信与重排成本**；
3. **layout builder 决定 grouped GEMM 是否真的能吃满硬件**；
4. **DeepGEMM 决定 expert 计算这段是否还能继续融合、特化和 JIT 化。**

也就是说，`MoE serving` 优化不是只有一个热点，而是一段连续热点链。  
`DeepGEMM` 的价值在于它把其中最贵、最密集的一段做成了真正可单独优化的骨架。

### 15.4 `DeepGEMM` 的价值：把链路切成更合理的“热路径块”

它的 grouped GEMM 和 `Mega MoE` 路线，本质上是在做两层优化：

1. **先把 expert GEMM 这一段从“很多零散专家调用”改写成“按 workload 组织的 grouped 路径”**；
2. **再把 dispatch / compute / combine 尽可能拉进同一个更大的融合路径里**。

这和普通 dense GEMM 最大的不同是：它优化的对象不只是矩阵乘本体，而是矩阵乘在 `MoE` 数据流中的位置。

### 15.5 为什么 grouped GEMM 对 `MoE serving` 特别关键

因为如果每个 expert 单独起一个小 GEMM，会很快遇到这些问题：

1. 每个 GEMM 太小，tensor core 吃不满；
2. kernel 数量很多，launch 开销明显；
3. token 分布不均，某些 expert 很空、某些很满；
4. 服务态延迟会被最慢那几个 expert 拖住。

grouped GEMM 的价值就在于：**把多个小而相似的 expert GEMM 组织成一条更统一的执行路径**，减少碎片化损耗。

### 15.6 为什么 masked layout 对 decode 尤其关键

decode 阶段最麻烦的点之一，就是每一步新增 token 很少，而专家分配又是动态的。此时：

1. CPU 如果每步都重新收集 expert token 统计并参与调度，开销可能不划算；
2. 开启 CUDA Graph 后，很多 shape 和运行路径又希望保持更稳定；
3. 但每个 expert 这一轮是否有 token、到底有多少 token，又确实是动态的。

这就是 masked layout 的意义：  
**用 mask 显式标出有效区域，让 kernel 在更稳定的外部形状下跳过无效部分。**

从系统角度看，这是一种把“动态 token 分布”重新编码为“静态形状 + 运行时掩码”的办法，非常适合图捕获和低时延服务。

### 15.7 `DeepEP`、dispatch/combine 和 `DeepGEMM` 的衔接

README 里提到，masked grouped GEMM 的一个使用例子，是把 `DeepEP` 的低延迟 kernel 输出作为输入。这里其实揭示了一个很重要的系统分工：

| 阶段 | 更像谁负责 |
| --- | --- |
| **低延迟 dispatch / combine / EP 通信** | `DeepEP` 这一类通信与路由组件 |
| **expert 内部核心计算** | `DeepGEMM` 这一类特化 GEMM / fused kernel |
| **调度与图捕获** | 服务 runtime 本身 |

也就是说，在完整 `MoE serving` 系统里，`DeepGEMM` 通常不是孤立工作，而是卡在：

**通信层之后、专家计算之中、combine 之前**。

### 15.8 `Mega MoE` 为什么是更进一步的路线

前面的 grouped GEMM 还是在“优化 expert 计算段”。  
`Mega MoE` 更进一步，它尝试把：

1. `EP dispatch`；
2. `linear 1`；
3. `SwiGLU`；
4. `linear 2`；
5. `EP combine`

整合成更大的 mega-kernel，并重叠 NVLink 通信与 tensor core 计算。

这条路线的意义在于：

1. 减少中间 buffer 往返；
2. 减少小 kernel 发射；
3. 把通信等待尽量埋进计算里；
4. 把 `MoE` 层真正当成一个整体来优化。

### 15.9 但 `Mega MoE` 也意味着更强的系统前提

从官方描述看，它需要多进程 launch 和 symmetric memory。这说明它的收益建立在更强的系统前提上：

1. 服务运行时要允许这种更深的多进程协同；
2. 通信、buffer 和 kernel 生命周期要协调得非常紧；
3. 一旦链路出现异常，定位难度会比普通“dispatch + 两次 GEMM + combine”高很多。

所以 `Mega MoE` 更像是：

**当普通 grouped GEMM 路线已经吃到头之后，再往前推进的系统级优化阶段。**

### 15.10 一个更实用的上线顺序

如果你真的在做 `MoE serving`，更稳妥的顺序通常不是一上来就冲 `Mega MoE`，而是：

1. **先把 dispatch / combine / expert token 统计打通**；
2. **先用 contiguous grouped GEMM 把 prefill 路线做稳**；
3. **再用 masked grouped GEMM 处理 decode + CUDA Graph 路线**；
4. **最后再评估是否值得为最热路径引入 `Mega MoE` 这类更深融合方案**。

这条顺序的好处是，问题分层更清楚：先解决数据流，再解决热点计算，最后解决跨阶段融合。

### 15.11 这节最该记住的判断

`DeepGEMM` 在 `MoE serving` 里的价值，不是“一个更快的 GEMM API”，而是：

1. **把 expert 计算写成更贴近服务态数据流的 grouped / masked 形式**；
2. **把 `MoE` 的主链从“很多小操作拼接”改写成“少数高价值热路径块”**；
3. **为更激进的 `Mega MoE` 融合路线提供中间台阶**。

只要用这个视角看，你就不会把它误认为一个孤立 kernel 库，而会把它看成 MoE 服务系统里非常核心的一段计算骨架。

## 16. 一张总表：`DeepGEMM / cuBLASLt / Triton / CUTLASS` 怎么放到一张图里

如果前面几节分别回答了局部问题，这里给一个更适合拿来做方案讨论的总表。

### 16.1 选型总表

| 维度 | `DeepGEMM` | `cuBLASLt` | `Triton` | `CUTLASS / CuTe` |
| --- | --- | --- | --- | --- |
| **抽象层** | 特化生产内核库 | 官方通用 matmul 前端 | 张量块级编程层 | 模板化高性能构件层 |
| **最擅长** | `FP8`、`MoE`、`Hopper/Blackwell` 热点路径 | 通用 GEMM、丰富 layout 与 epilogue | 快速原型、融合实验、中层实现 | 长期高性能 CUDA 基建与深度特化 |
| **shape 假设** | 强，偏热点 shape | 中等，广覆盖 | 中等，可快速适配 | 可很强，但要自己组织 |
| **硬件绑定** | 强 | 官方统一支持更广 | 相对中等 | 可深度绑定，但成本高 |
| **JIT / 特化** | 强，核心卖点之一 | 有 heuristic 与 runtime 选择，但不等同于 Fully JIT | 编译期/运行期都有一定特化空间 | 主要靠模板实例化与手工组织 |
| **MoE 适配度** | 很强，尤其 grouped / masked / Mega MoE | 有通用支持，但不是它的主要叙事核心 | 可做原型，但深度系统整合要自己补 | 能做，但工程量通常更大 |
| **生态集成** | 偏特定系统链 | 最强，官方接口友好 | 和 PyTorch 工作流很顺 | 适合底层基建，不是最轻量接入点 |
| **开发速度** | 中 | 快速接入，但深度调优受限 | 很快 | 慢 |
| **极限性能上限** | 在目标场景很高 | 很高，但偏通用最优 | 中高，视场景而定 | 很高 |
| **维护成本** | 中高 | 低到中 | 中 | 高 |

### 16.2 一句话决策版本

如果你只想记最短版本，可以直接用下面这张表：

| 如果你现在最需要 | 更优先看 |
| --- | --- |
| **稳、广、官方底座** | `cuBLASLt` |
| **快做原型、快试融合** | `Triton` |
| **做长期高性能 CUDA 底座** | `CUTLASS / CuTe` |
| **做 `Hopper / Blackwell + FP8 + MoE` 热点生产优化** | `DeepGEMM` |

### 16.3 一个常见的成熟分层方案

很多成熟系统最后不是选一个，而是分层落位：

1. **默认通用 matmul 走 `cuBLASLt`**；
2. **新算子原型和中等复杂度融合先走 `Triton`**；
3. **长期底层资产和极深特化交给 `CUTLASS / CuTe` 或手写 CUDA**；
4. **极热点的 `FP8 / MoE` 路径单独接入 `DeepGEMM`**。

这条路线的本质是：  
**让不同工具各自负责最匹配它抽象层和维护成本的那一段。**

### 16.4 为什么这个总表有价值

因为很多技术讨论一旦脱离“系统里它落在哪一层”，就会变成无效争论。  
把 `DeepGEMM / cuBLASLt / Triton / CUTLASS` 放到一张图里，最重要的不是分出绝对高下，而是避免下面这些误判：

1. 把通用官方库和热点特化库当成同一层替代关系；
2. 把快速原型工具和长期底座工具当成同一类资产；
3. 在 shape 还不稳定时就过早为极限性能付出巨大维护成本；
4. 在真正的热点路径上，又因为过度依赖通用抽象而迟迟不下沉优化。

## 17. `DeepGEMM` 源码阅读顺序

如果你准备真正进仓库看实现，建议不要一上来就钻到最底层 kernel。更有效的方式是：**先看导出层和接口边界，再看测试如何调用，再看 C++ API 层，最后再钻 JIT 和 kernel。**

### 17.1 先记住仓库里最关键的几层

根据当前官方仓库结构，最值得关注的目录可以先按下面这张图记：

| 层级 | 目录 / 文件 | 主要作用 |
| --- | --- | --- |
| **入口层** | `README.md` | 先看能力边界、支持硬件、接口分类与使用前提 |
| **Python 导出层** | `deep_gemm/__init__.py` | 看所有公开 API 名字、分组方式、runtime 配置项 |
| **Python 工具层** | `deep_gemm/utils/`、`deep_gemm/testing/` | 看 layout、cast、分布式辅助、benchmark 与数值校验工具 |
| **历史实现层** | `deep_gemm/legacy/` | 看 grouped GEMM 等旧版 Python 路线，适合理解方法演化 |
| **C++ 绑定层** | `csrc/python_api.cpp` | 看 Python API 最终如何接到底层实现 |
| **API 头文件层** | `csrc/apis/` | 按功能看 GEMM、attention、layout、mega、runtime 等接口分层 |
| **JIT / kernel 层** | `csrc/jit/`、`csrc/jit_kernels/` | 看运行时编译与真正 kernel 生成逻辑 |
| **专项测试层** | `tests/` | 看不同能力在真实 shape 和 benchmark 里怎么被调用 |

### 17.2 推荐的第一遍阅读顺序

第一遍不求全看懂底层实现，重点是建立“仓库是怎么分层的”。

**推荐顺序**：

1. `README.md`
2. `deep_gemm/__init__.py`
3. `tests/test_fp8_fp4.py`
4. `tests/test_layout.py`
5. `tests/test_mega_moe.py`
6. `csrc/python_api.cpp`
7. `csrc/apis/gemm.hpp`
8. `csrc/apis/layout.hpp`
9. `csrc/apis/attention.hpp`
10. `csrc/apis/mega.hpp`
11. `csrc/apis/runtime.hpp`
12. `csrc/jit/` 和 `csrc/jit_kernels/`

这条顺序的好处是：**先从“怎么用”倒推“怎么实现”，比从最底层代码正向往上爬更容易建立整体感。**

### 17.3 第一步：先看 `README.md`

先不要把 README 当安装说明看，而要把它当“设计边界文档”看。重点抓这些问题：

1. **支持哪些硬件**：当前重点是 `SM90 / SM100`；
2. **支持哪些 kernel 族**：`FP8 / FP4 / BF16 GEMM`、grouped GEMM、`MQA`、`Mega MoE`；
3. **接口有哪些约束**：例如 `lhs` scaling factor 的 TMA 对齐和转置布局要求；
4. **哪些事情不由库负责**：例如某些输入转置、FP8 cast、上游 pack 和 layout 处理要用户自己做。

**如果这一层不先看清**，后面读代码时很容易把很多“写死的前提”误解成奇怪实现细节。

### 17.4 第二步：看 `deep_gemm/__init__.py`

这一步的目标是建立“公开接口地图”。当前 `__init__.py` 里有几个非常关键的信息：

1. **导出了哪些 runtime 配置项**：如 `set_num_sms()`、`set_tc_util()`、`set_pdl()` 等；
2. **导出了哪些 `cuBLASLt` 路线**：如 `cublaslt_gemm_nt/nn/tn/tt`；
3. **导出了哪些 `DeepGEMM` 路线**：dense GEMM、`m_grouped_*`、`k_grouped_*`、`einsum`、`MQA logits`、`Mega MoE` 相关功能；
4. **哪些东西已经被分成“normal / grouped / attention / layout / mega”几大类。**

**这一步相当于先拿到一份函数目录。**  
后面每看一个测试文件，你都能更快对上它最终调用的是哪类 API。

### 17.5 第三步：优先读三个测试文件

如果你只读一个地方来理解仓库“怎么被使用”，最值得看的其实是测试。

#### `tests/test_fp8_fp4.py`

这几乎是理解核心 GEMM 路线的最佳入口，因为它同时包含：

1. dense GEMM 调用；
2. grouped GEMM 调用；
3. correctness 对比；
4. benchmark；
5. 和 `cuBLASLt` 的性能对照。

这让你能同时看到：

- 输入是怎么构造的；
- layout 与 alias 是怎么变的；
- quant config 是怎么影响调用的；
- 库希望用户如何组织 benchmark。

#### `tests/test_layout.py`

如果你看不懂为什么 README 里一直强调 `TMA-aligned`、`UE8M0`、transpose layout，就优先读这份。  
它能帮你把“scale layout 为什么是一等公民”这件事看明白。

#### `tests/test_mega_moe.py`

这份最值得用来理解系统层接口，因为它直接暴露了：

1. `DeepGEMM` 怎么和分布式初始化、rank、buffer 配合；
2. `Mega MoE` 为什么需要 symmetric memory；
3. `DeepEP` / legacy baseline / benchmark 是怎么接进来的。

**如果你想理解这库为什么已经不只是 GEMM 库，这份测试非常关键。**

### 17.6 第四步：回到 `deep_gemm/utils/` 和 `deep_gemm/testing/`

很多人会跳过工具层，直接冲底层 kernel。这样往往会漏掉库真正的输入组织方式。

当前最值得看的几个文件是：

| 文件 | 适合解决什么问题 |
| --- | --- |
| `deep_gemm/utils/layout.py` | 理解 layout、对齐、scale tensor 组织 |
| `deep_gemm/utils/math.py` | 看一些基础数值和计算辅助 |
| `deep_gemm/utils/dist.py` | 看分布式场景里怎么做打印、初始化、通信辅助 |
| `deep_gemm/testing/bench.py` | 看性能测试习惯与入口 |
| `deep_gemm/testing/numeric.py` | 看数值对比与误差度量 |

**这一层的价值**在于：它告诉你这个库不是只靠一个 kernel 硬冲性能，而是配了一整套输入准备、测试与校验习惯。

### 17.7 第五步：如果你想理解“从 Python 到 C++”的桥，先读 `csrc/python_api.cpp`

这一层最适合回答两个问题：

1. Python 暴露的函数名，最终是怎么绑定到 C++ 侧的；
2. 哪些参数会在这一层被整理、检查或转发。

当你已经从 `__init__.py` 和 tests 里知道有哪些公开函数后，再来看 `csrc/python_api.cpp`，你会更容易把：

- Python API；
- runtime config；
- C++ 调用入口；
- 底层功能模块

连接起来。

### 17.8 第六步：按功能读 `csrc/apis/`

这一层是最适合“按主题阅读”的部分。官方仓库当前把 API 头文件拆成：

- `gemm.hpp`
- `layout.hpp`
- `attention.hpp`
- `mega.hpp`
- `runtime.hpp`
- 以及 `einsum.hpp`、`hyperconnection.hpp`

**推荐顺序**：

1. `gemm.hpp`
2. `layout.hpp`
3. `runtime.hpp`
4. `attention.hpp`
5. `mega.hpp`

原因是：

1. **先看 GEMM 本体**，把 dense / grouped 的接口边界弄清；
2. **再看 layout**，理解这些接口为什么对输入组织这么敏感；
3. **再看 runtime**，理解 JIT 和执行时配置怎么接入；
4. **最后再看 attention / mega**，因为它们更偏系统链路上的专项扩展。

### 17.9 第七步：最后再钻 `csrc/jit/` 和 `csrc/jit_kernels/`

这是最容易一头扎进去出不来的地方，所以建议放到后面。

这一层阅读时，最好带着明确问题去看，例如：

1. **哪些参数会成为编译期常量**；
2. **JIT cache 和编译入口怎么组织**；
3. **不同 kernel family 是如何被生成的**；
4. **shape specialization 究竟落在什么层**；
5. **哪些逻辑是 runtime dispatch，哪些逻辑是 codegen。**

如果不带着这些问题，很容易淹没在大量底层细节里。

### 17.10 一个按目标分流的阅读路径

如果你不是想“全看完”，而是带着具体目的进仓库，更推荐下面这种分流读法：

| 你的目标 | 更推荐的顺序 |
| --- | --- |
| **想理解核心 GEMM API** | `README.md` -> `deep_gemm/__init__.py` -> `tests/test_fp8_fp4.py` -> `csrc/apis/gemm.hpp` |
| **想理解 layout / scale / TMA 对齐** | `README.md` -> `tests/test_layout.py` -> `deep_gemm/utils/layout.py` -> `csrc/apis/layout.hpp` |
| **想理解 `MoE serving / Mega MoE`** | `README.md` -> `tests/test_mega_moe.py` -> `deep_gemm/mega/__init__.py` -> `csrc/apis/mega.hpp` -> `csrc/jit/` |
| **想理解 Python 到 C++ 的桥接** | `deep_gemm/__init__.py` -> `csrc/python_api.cpp` -> `csrc/apis/runtime.hpp` |
| **想理解历史实现和演化** | `deep_gemm/legacy/` -> 当前 `tests/` -> 当前 `csrc/apis/` |

### 17.11 一个很实用的阅读纪律

读这种仓库时，建议强制自己保持下面这套顺序：

1. **先看接口名**；
2. **再看测试怎么调**；
3. **再看参数组织**；
4. **最后才看 kernel 内部优化。**

这样做的好处是，你会先明白“库是怎么被系统使用的”，而不是一开始就被底层优化细节拖走。

### 17.12 这节的最终建议

如果你时间有限，我会建议最小阅读集就是：

1. `README.md`
2. `deep_gemm/__init__.py`
3. `tests/test_fp8_fp4.py`
4. `tests/test_layout.py`
5. `tests/test_mega_moe.py`
6. `csrc/python_api.cpp`
7. `csrc/apis/gemm.hpp`
8. `csrc/apis/layout.hpp`
9. `csrc/apis/mega.hpp`

只要这 9 个落点读下来，你对 `DeepGEMM` 的整体架构、接口边界、热点场景和系统定位，基本就已经有比较扎实的地图了。

## 18. `DeepGEMM` 关键 API 名称对照表

如果你第一次读 `deep_gemm/__init__.py`，很容易被大量函数名淹没。更好的方式，是先按功能把它们分组记忆。

### 18.1 从 Python 导出层看，最核心的 API 可以先分五类

| 类别 | 典型 Python API | 对应 C++/头文件层 | 更适合解决什么问题 |
| --- | --- | --- | --- |
| **runtime 配置** | `set_num_sms`、`set_tc_util`、`set_pdl`、`set_ignore_compile_dims` | `csrc/apis/runtime.hpp` | 控制运行时资源使用、编译策略和执行偏好 |
| **通用官方 matmul 路线** | `cublaslt_gemm_nt/nn/tn/tt` | 通过 `_C` 导出 | 把 `DeepGEMM` 放回 `cuBLASLt` 参考系里比较 |
| **dense GEMM 路线** | `fp8_gemm_nt/nn/tn/tt`、`bf16_gemm_nt/nn/tn/tt`、`fp8_fp4_gemm_*` | `csrc/apis/gemm.hpp` | 理解 normal dense GEMM 的主入口 |
| **grouped GEMM 路线** | `m_grouped_fp8_gemm_*`、`k_grouped_fp8_gemm_*`、`m_grouped_bf16_gemm_*`、`m_grouped_fp8_fp4_gemm_*` | `csrc/apis/gemm.hpp` | 理解 MoE、分组布局和 expert 聚合计算 |
| **attention / layout / mega 路线** | `get_paged_mqa_logits_metadata`、`fp8_fp4_paged_mqa_logits`、`transform_sf_into_required_layout`、`get_symm_buffer_for_mega_moe`、`fp8_fp4_mega_moe` | `csrc/apis/attention.hpp`、`layout.hpp`、`mega.hpp` | 理解服务态 attention、scale layout 和 Mega MoE 扩展 |

### 18.2 Dense GEMM 这组名字怎么读

最基础的一组是：

- `fp8_gemm_nt`
- `fp8_gemm_nn`
- `fp8_gemm_tn`
- `fp8_gemm_tt`
- 对应的 `bf16_gemm_*`
- 以及 `fp8_fp4_gemm_*`

这组名字最重要的不是背下来，而是理解它把**内存布局**直接编码进了函数名。  
这意味着在 `DeepGEMM` 里，布局不是隐藏细节，而是显式接口语义的一部分。

### 18.3 Grouped GEMM 这组名字怎么读

最关键的 grouped 系列包括：

| Python API | 主要语义 | 更常见的场景 |
| --- | --- | --- |
| `m_grouped_fp8_gemm_nt_contiguous` | 只按 `M` 轴分组、`NT` 布局、contiguous layout | `MoE` prefill / forward |
| `m_grouped_fp8_gemm_nt_masked` | 只按 `M` 轴分组、masked layout | `MoE` decode + CUDA Graph |
| `m_grouped_bf16_gemm_nt_contiguous` | BF16 版本 grouped GEMM | 更高精度或调试路线 |
| `k_grouped_fp8_gemm_nt_contiguous` | 另一类按 `K` 组织的 grouped 路线 | 更专项的 grouped shape 组织 |
| `m_grouped_fp8_fp4_gemm_nt_contiguous` | FP8xFP4 版本 grouped GEMM | 更激进低精度路线 |

如果你只想抓最重要的判断，可以记成：

1. **`contiguous` 更像 prefill / forward**；
2. **`masked` 更像 decode / graph capture**；
3. **`m_grouped` 是 `DeepGEMM` 理解 MoE 的关键 API 族。**

### 18.4 Layout API 为什么必须单独记

`transform_sf_into_required_layout` 这一类函数特别值得记，因为它提醒你：

**scale factor 的 layout 在 `DeepGEMM` 里不是杂务，而是性能主路径的一部分。**

配套还包括：

- `get_tma_aligned_size`
- `get_mn_major_tma_aligned_tensor`
- `get_mn_major_tma_aligned_packed_ue8m0_tensor`
- `get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor`

这组 API 对应的不是“数学公式”，而是**输入准备、TMA 对齐与量化 layout 组织**。

### 18.5 Attention / MQA 这组 API 怎么看

这里最关键的不是普通 attention，而是服务态 MQA scoring 相关入口：

| Python API | 更像解决什么问题 | 对应头文件 |
| --- | --- | --- |
| `fp8_gemm_nt_skip_head_mid` | 一类 attention scoring / 中间阶段特殊 GEMM 变体 | `csrc/apis/attention.hpp` |
| `get_paged_mqa_logits_metadata` | paged MQA 所需 metadata 准备 | `csrc/apis/attention.hpp` |
| `fp8_fp4_paged_mqa_logits` | paged MQA logits 核心计算 | `csrc/apis/attention.hpp` |
| `fp8_paged_mqa_logits` | legacy attention scoring 路线 | `csrc/apis/attention.hpp` |

这组 API 的存在说明：  
`DeepGEMM` 已经在向服务态 attention 热点扩展，而不是只停留在“通用 GEMM 库”。  

### 18.6 Mega MoE 这组 API 为什么单独成段

当前最关键的 `mega` 相关入口包括：

- `get_symm_buffer_for_mega_moe`
- `transform_weights_for_mega_moe`
- `fp8_fp4_mega_moe`

从 `csrc/apis/mega.hpp` 可以看出，这组 API 已经不是普通“算一个 GEMM”，而是在组织：

1. 对称内存 buffer；
2. 权重变换；
3. 完整 Mega MoE 热路径。

如果你看到这里，说明你已经从“看单 kernel”进入“看系统级融合路径”了。

### 18.7 一个最值得记的 API 地图

如果你只想保留一张最简地图，可以直接记成下面这样：

```text
runtime      -> set_num_sms / set_tc_util / set_pdl
baseline     -> cublaslt_gemm_*
dense gemm   -> fp8_gemm_* / bf16_gemm_* / fp8_fp4_gemm_*
grouped gemm -> m_grouped_* / k_grouped_*
layout       -> transform_sf_into_required_layout / get_*_aligned_*
attention    -> get_paged_mqa_logits_metadata / fp8_fp4_paged_mqa_logits
mega moe     -> get_symm_buffer_for_mega_moe / fp8_fp4_mega_moe
```

这张图的价值在于：当你再进 `tests/` 或 `csrc/apis/` 时，不会再把函数名看成散点，而会知道它们落在哪个模块上。

## 19. `tests -> Python API -> C++ API -> JIT/kernel` 调用路径映射

这部分就是“源码导图”的最后一块：把测试、Python 导出层、C++ API 层和底层 kernel 之间的关系连起来。

### 19.1 先记住一条总路径

在 `DeepGEMM` 里，最常见的主路径可以粗略写成：

```text
tests/*.py
  -> deep_gemm/__init__.py 暴露的 Python API
  -> deep_gemm._C (由 csrc/python_api.cpp 绑定)
  -> csrc/apis/*.hpp 中的功能入口
  -> csrc/jit/ 与 csrc/jit_kernels/ 中的 JIT / kernel 实现
  -> SM90 / SM100 特化 kernel
```

这条路径的价值在于：  
**你看到测试里的一个函数名时，知道它不是“直接神秘跑到 GPU 上”，而是经过了明确的导出、绑定、分层 API 和 JIT/kernel 生成链。**

### 19.2 三条最值得看的具体映射链

#### 路线一：普通 `FP8` dense GEMM

| 层级 | 典型落点 |
| --- | --- |
| **测试入口** | `tests/test_fp8_fp4.py` |
| **Python API** | `deep_gemm.fp8_fp4_gemm_nt()` 或 `deep_gemm.fp8_gemm_nt()` |
| **绑定层** | `deep_gemm._C`，由 `csrc/python_api.cpp` 统一注册 |
| **C++ API 层** | `csrc/apis/gemm.hpp` |
| **底层实现** | `csrc/jit_kernels/impls/sm90_fp8_gemm_1d1d.hpp`、`sm90_fp8_gemm_1d2d.hpp` 等 |

这一条最适合用来理解：

1. normal GEMM 的输入组织；
2. layout 选择如何影响 kernel 路线；
3. dense GEMM 是如何一步步落到架构特化实现上的。

#### 路线二：layout / scale 变换

| 层级 | 典型落点 |
| --- | --- |
| **测试入口** | `tests/test_layout.py` |
| **Python API** | `transform_sf_into_required_layout()`、`get_mn_major_tma_aligned_*()` |
| **绑定层** | `deep_gemm._C` |
| **C++ API 层** | `csrc/apis/layout.hpp` |
| **底层作用** | 为后续 GEMM / grouped GEMM 准备对齐与 packed scale layout |

这一条最适合用来理解：

1. 为什么 README 一直强调 TMA alignment；
2. 为什么 scale layout 是主路径的一部分；
3. 为什么 `DeepGEMM` 的接口不像普通 GEMM 那样“只接两个矩阵”。  

#### 路线三：Mega MoE / 系统级融合路径

| 层级 | 典型落点 |
| --- | --- |
| **测试入口** | `tests/test_mega_moe.py` |
| **Python API** | `get_symm_buffer_for_mega_moe()`、`fp8_fp4_mega_moe()` |
| **Python 辅助层** | `deep_gemm/mega/__init__.py` |
| **绑定层** | `deep_gemm._C` / `csrc/python_api.cpp` |
| **C++ API 层** | `csrc/apis/mega.hpp` |
| **底层实现** | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp` 等 |

这一条最适合用来理解：

1. `DeepGEMM` 为什么已经进入系统级 `MoE` 优化；
2. symmetric memory 和 buffer 准备为什么会进入核心 API；
3. 为什么这已经不是单纯的“写个更快 GEMM”。  

### 19.3 再加一条：attention / paged MQA 路线

这一条虽然不一定是所有人第一优先看，但很能说明 `DeepGEMM` 向服务态热点扩展的方向。

| 层级 | 典型落点 |
| --- | --- |
| **测试 / 使用入口** | attention 相关调用或后续专项测试 |
| **Python API** | `get_paged_mqa_logits_metadata()`、`fp8_fp4_paged_mqa_logits()` |
| **绑定层** | `deep_gemm._C` |
| **C++ API 层** | `csrc/apis/attention.hpp` |
| **底层实现** | `csrc/jit_kernels/impls/smxx_fp8_fp4_paged_mqa_logits.hpp` 等 |

这条路径最值得看的点在于：

1. metadata 准备本身就是接口一部分；
2. paged / non-paged attention 场景已经进入库的主叙事；
3. `DeepGEMM` 在逐步覆盖更多真实服务热点。  

### 19.4 为什么 `python_api.cpp` 很关键

从我们刚才看的仓库结构里，`csrc/python_api.cpp` 的角色非常明确：它统一注册了 `apis/` 目录下各个模块的导出，例如 `mega::register_apis(m)` 这种调用。

这说明一个非常好的阅读习惯：

1. 在 Python 侧先认函数名；
2. 再到 `python_api.cpp` 看它们属于哪个 C++ 模块；
3. 再进入相应的 `csrc/apis/*.hpp`；
4. 最后才追到更底层 `jit_kernels/impls/*`。

这样你看到任何一个 API，都能快速知道它挂在哪个功能族下面，而不是在仓库里盲搜。

### 19.5 一个更适合手边对照的调用图

```text
tests/test_fp8_fp4.py
  -> deep_gemm.fp8_fp4_gemm_nt
  -> deep_gemm._C.fp8_fp4_gemm_nt
  -> csrc/apis/gemm.hpp
  -> sm90_fp8_gemm_* / sm100_* kernel impl

tests/test_layout.py
  -> deep_gemm.transform_sf_into_required_layout
  -> deep_gemm._C.transform_sf_into_required_layout
  -> csrc/apis/layout.hpp
  -> aligned / packed scale layout helpers

tests/test_mega_moe.py
  -> deep_gemm.get_symm_buffer_for_mega_moe
  -> deep_gemm.fp8_fp4_mega_moe
  -> csrc/python_api.cpp
  -> csrc/apis/mega.hpp
  -> sm100_fp8_fp4_mega_moe kernel impl
```

### 19.6 这节最该记住的事

如果你读 `DeepGEMM` 时总觉得“函数很多、文件很多、入口很散”，通常是因为脑子里还没有这张映射表。  
一旦把：

**tests -> Python API -> C++ API -> JIT/kernel**

这条链建立起来，仓库会立刻清晰很多。你会知道：

1. 测试在验证什么；
2. Python 层只是导出与组织接口；
3. 真正的功能边界主要在 `csrc/apis/`；
4. JIT 和 kernel 生成是最后一层，而不是第一层阅读入口。

## 20. 把 `DeepGEMM` 接进真实系统前，最该检查什么

如果你已经接受了前面的判断，下一步通常不是“它看起来很强，所以接进系统”，而是先回答一组很现实的问题。  
`DeepGEMM` 这类库的真正门槛，不是 API 会不会调，而是**你的 workload、layout、运行时和运维习惯是否真的匹配它的假设**。

### 20.1 接入前的五项硬检查

| 检查项 | 应该问什么 | 不满足时会怎样 |
| --- | --- | --- |
| **硬件统一性** | 是否主要运行在 `SM90 / SM100` 这类目标 GPU 上 | 强特化收益会被硬件分裂吃掉 |
| **shape 稳定性** | 热点 batch、token bucket、expert shape 是否足够集中 | JIT 回本慢，cache 命中率低 |
| **layout 可控性** | 上游是否能稳定提供所需 pack、transpose、scale layout | 前后处理可能比 GEMM 本体还贵 |
| **服务运行时配合度** | 是否允许热身、JIT cache、图捕获约束和专项 dispatch | 首次延迟、发布复杂度会上升 |
| **团队维护能力** | 是否有人能长期追驱动、CUDA、架构升级与回归 | 线上收益可能被后续维护成本反噬 |

如果这五项里有三项都答不上来，通常更稳妥的起点还是：

1. 先用 `cuBLASLt` 或现有通用路径跑稳定；
2. 再把真实热点 shape 抽出来做对照；
3. 确认回报显著后，再局部切入 `DeepGEMM`。

### 20.2 一个更实用的接入清单

真正准备进系统时，建议至少补齐下面这些验证：

1. **功能正确性**：和现有 `cuBLASLt` 或参考实现逐 shape 对齐；
2. **数值一致性**：重点看 `FP8`、`FP8xFP4` 和 grouped / masked 路线的误差边界；
3. **冷热启动差异**：区分首次 JIT 与 cache 命中后的延迟；
4. **shape 覆盖率**：统计线上前 90% 请求是否真的落在目标热区；
5. **fallback 策略**：遇到不支持或收益不明显的 shape，是否自动切回通用库；
6. **回归成本**：CUDA 升级、驱动升级、GPU 代际变更后，是否有自动化回归集。

这份清单的重点是：  
**不要只证明它在理想 benchmark 上更快，要证明它在真实系统里更值得。**

### 20.3 最常见的 benchmark 误读

`DeepGEMM` 这一类高性能库最容易被误读的地方，不是实现，而是结果。

**误读 1：峰值 TFLOPS 高，就说明线上一定赚。**

不对。  
线上可能输在：

1. 首次 JIT；
2. layout transform；
3. dispatch / combine；
4. 小 shape 长尾；
5. cache 不命中。

**误读 2：某个热点 shape 2x 提升，就说明整条链也接近 2x。**

也不对。  
整条链的瓶颈可能根本不在 expert GEMM 本体，而在：

1. router；
2. expert token 重排；
3. 跨卡通信；
4. graph capture 约束；
5. 中间 buffer 管理。

**误读 3：对 `cuBLASLt` 快，就说明任何系统都该替换。**

这也不成立。  
更合理的说法应该是：

**它在某些高度匹配的 `FP8 / MoE / Hopper` 热点路径上，可能比默认官方通用路径更值得单独下沉。**

### 20.4 一个更稳妥的灰度顺序

如果你准备在线上或大规模离线系统里逐步接入，建议按这个顺序推进：

1. **先做离线单算子对照**：证明正确性和局部性能收益；
2. **再做链路内对照**：把前后处理、dispatch、combine 一起算进去；
3. **再做热 shape 灰度**：只切最稳定、最常见的一批请求；
4. **最后再扩大覆盖**：逐步把更多 grouped / masked 路线纳入。

这种顺序的价值在于，能把风险压在最可控的范围里，而不是一上来把整个 GEMM 主路径换掉。

## 21. 建议的阅读方式

如果你想系统读 `DeepGEMM`，建议顺序如下：

1. **先读 README 的接口和 requirements**：先看它假设了什么硬件与布局前提；
2. **再看 dense GEMM 与 grouped GEMM**：理解它为什么这么设计 API；
3. **再看 JIT 相关配置和缓存逻辑**：理解它为什么不走纯预编译路线；
4. **再看 `MQA` 和 `Mega MoE`**：理解它为什么已经不是单纯 GEMM 库；
5. **最后做 profile 和 shape 对照**：把 benchmark 结果和真实 workload 形状放在一起看。

## 快速代码示例

```python
def bucket_moe_tokens(expert_ids):
    buckets = {}
    for idx, eid in enumerate(expert_ids):
        buckets.setdefault(int(eid), []).append(idx)
    return buckets

def choose_gemm_path(m_size, use_masked=True):
    if use_masked and m_size <= 128:
        return "grouped_masked_gemm"
    return "dense_gemm"
```

这段代码抽象了 `DeepGEMM` 在 MoE 场景常见的两个关键动作：先按 expert 把 token 分桶，再按 shape/布局选择 dense 或 grouped+masked 路径。真实工程中这一步通常与 JIT cache 命中率一起决定端到端收益。

## 22. 小结

`DeepGEMM` 最值得学的，不只是“怎么把 FP8 GEMM 写快”，而是**怎么把低精度、硬件特性、JIT、服务约束、MoE 数据流和系统接口统一进一个算子库设计里**。它提醒我们，现代大模型算子优化的真正边界早就不止于单个矩阵乘公式，而是在向一整条“布局变换 -> 数据搬运 -> 计算 -> 融合 -> 通信 -> 运行时缓存”的系统链路扩展。

如果你把这页看明白，再回去看 `FlashAttention`、`PagedAttention`、量化 GEMM、MoE fusion 或者服务态 decode kernel，会更容易发现它们背后其实共享同一套工程逻辑。
