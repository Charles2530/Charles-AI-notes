# 四张卡的梯度怎样汇合，通信又何时能隐藏

数据并行让各rank在不同样本上计算梯度，再同步形成同一个更新。同步需要交换真实张量，而且只有已经计算完的梯度才能发送。通信重叠因此是一个依赖调度问题。

## 用四个向量看collective

设四个rank的局部梯度分别为 `[1,2,3,4]`、`[10,20,30,40]`、`[100,200,300,400]`、`[1000,2000,3000,4000]`。求和结果为 `[1111,2222,3333,4444]`。

All-reduce让所有rank都拿到完整和。Reduce-scatter先求和，再把对应分片分别交给rank0到3，此例各得到一个数。随后all-gather这些分片，可以让所有rank重新得到完整向量。

“平均”需要再除以rank数4，不能把NCCL求和原语与框架梯度平均约定混为一谈。若各rank监督token数不同，局部loss还必须使用一致的全局分母，见[分布式训练](../training/distributed-training-and-checkpointing.md)。

## 分片状态为何需要不同collective

普通DDP每卡有完整参数，平均梯度后可以各自执行相同更新。优化器状态分片后，每rank只更新自己负责的片段，reduce-scatter的结果恰好落在对应持有者。参数分片时，计算某层前可能需要all-gather重建所需参数，反向后再把梯度归约回分片。

张量并行则在一层内切分计算。若各卡算出输出的部分和，需要归约；若各卡持有不同输出通道，需要按后续算子要求保留分片或拼接。Collective由张量语义决定，不能单看并行策略名字。

## Bucket怎样制造重叠窗口

假设反向共有三段，每段耗时4个单位，每段产生一个梯度bucket，通信各耗时3。全部反向结束后再依次通信，总时间为12+9=21。若每个bucket一就绪就发送，通信分别可在4到7、8到11、12到15运行，最后完成时间为15。

这个教学时间线假设计算与通信能够独立重叠，没有争抢资源。实际NCCL可能使用SM、内存带宽或网络资源，与计算同时执行反而降低双方速度，需查看完整时间线。

小bucket更早就绪，但增加collective启动次数；大bucket提高传输效率，却推迟发送。最后一个bucket之后通常没有足够反向计算可隐藏它，暴露通信还与层顺序有关。

## Ring的字节模型能说明什么

对P个rank、每卡S字节数据，理想ring all-reduce每卡发送量约 $2(P-1)S/P$，由reduce-scatter与all-gather两段组成。简单时间模型还需加入每轮启动延迟，不能只用总字节除带宽解释小消息。

实际库可选tree、分层或其他算法，并受拓扑、消息尺寸与协议影响。这个公式用于建立量级，不表示某次运行一定用了ring。

## 观察重叠要看最后一项依赖

Trace里通信和计算的色块相互覆盖，只说明时间上并发。真正收益取决于下一步需要的所有状态何时就绪。比较step结束时间、各rank拖尾、bucket就绪与完成时刻，才能判断通信是否被隐藏，还是只是换了一种等待位置。

![DeepSeek-V3 DualPipe schedule](../assets/images/paper-deep-dives/technical-reports/deepseek-v3/figure-5-dualpipe.png){ width="900" }

<small>图源：[DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)，Figure 5。原图表达 DualPipe 如何把双向 pipeline micro-batches 与通信/计算重叠起来；读图重点是通信优化的核心不是“少一次 collective”这么简单，而是把通信放到关键路径之外。它不能证明 DualPipe 适合所有并行度和拓扑。</small>

## 对照源码与延伸阅读

- [nanoGPT DDP反向同步与梯度累积位置](https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/train.py)
- [MegaMoE对通信与计算联合调度的具体边界](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/README.md)
