# 从源码假设追到编译结果

假设把GEMM输出tile扩大后，逻辑上的数据复用增加，实际速度却下降。下一步要验证是否增加了寄存器占用、spill或其他成本。编译检查应服务这个具体问题，不能只凭看到某条矩阵指令就宣布优化成功。

## PTX与SASS分别处于哪一层

PTX是NVIDIA虚拟指令集表示，仍需进一步编译为目标GPU的机器指令。SASS反映具体架构上的指令。相同PTX在不同架构或工具链下可能产生不同资源使用与调度，不能把两者当作同一层证据。

高层IR还能帮助确认布局转换与融合是否已经发生。先从接近源码的层看语义变化，再向下追具体指令，通常比直接读完整汇编更容易定位。

## 把比较范围缩到一个shape

固定输入M/N/K、dtype、stride、编译器和GPU，只改变tile参数。保存两版编译输出与资源信息，再比较实际耗时。否则一次编译器升级、dtype改变与tile改变同时发生，很难知道哪项导致差异。

更大的输出tile需要更多累加器。如果寄存器需求超出可用资源，编译器可能产生local memory访问。看到spill统计或相关访存是支持证据，但local memory也可能来自明确的局部数组，仍应追源码用途。

<!-- systems-workflow-visual -->
```mermaid
flowchart LR
  S["更大tile的源码假设"] --> I["IR与PTX"] --> A["目标GPU SASS"] --> R["寄存器与spill证据"] --> P["同shape profile"] --> C["验证性能原因"]
```

编译结果与运行测量要绑定同一设备、shape和配置，避免把不同变体的证据拼在一起。

## 源码表达dot不保证目标指令和效率

矩阵尺寸、dtype、对齐、布局和编译配置都影响降级路径。检查是否生成预期矩阵运算，再看周围的转换、搬运与同步。即使矩阵指令存在，数据供给不足或小shape也可能限制整体效率。

同理，向量化load存在不代表全局访问一定连续；需要把线程地址映射与实际内存事务联系起来。编译静态信息解释“程序会怎样执行”，profile解释“这次执行为何耗时”。

## 利用库提供的可观测入口

DeepGEMM固定版本README列出 `DG_JIT_DEBUG`、`DG_JIT_PTXAS_VERBOSE`、`DG_JIT_DUMP_PTX`、`DG_JIT_DUMP_SASS`以及检查spill/local memory的选项。这些是该版本查证过的接口，其他库和版本应查询各自文档。

开启检查后先确认输出属于正在基准的那一个变体。JIT缓存、不同shape和fallback可能让源码与实际加载kernel不一致。记录编译配置和实际命中路径，才能把汇编证据与测量对应起来。

## 什么时候可以停止下钻

若完整请求时间主要在排队或CPU前处理，继续优化一个只占极少时间的kernel汇编，通常缺少收益来源。回到关键路径决定工作优先级。底层检查的目标是验证一个可行动假设，而非把所有指令都解释一遍。

![Nsight Systems synchronization calls](../assets/images/tools/nsight-systems/synchronization-calls.png){ width="820" }

<small>图源：[NVIDIA Nsight Systems 文档](https://docs.nvidia.com/nsight-systems/)。原图表达同步调用和 kernel 时间线如何暴露源码意图之外的执行路径；读图重点是提醒编译检查要和 profile 结合：PTX/SASS 说明生成了什么，timeline 说明它何时真正阻塞。它不能替代逐条 SASS 审查。</small>

## 对照源码与延伸阅读

- [该版本JIT调试、汇编与资源检查选项](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/README.md)
- [tile与架构条件下的不同实现](https://github.com/triton-lang/triton/blob/cd2e01c6686d6503802e6cdfac56d5a5dfdb27a1/python/tutorials/09-persistent-matmul.py)
