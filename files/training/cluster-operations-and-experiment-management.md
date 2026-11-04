# 一个慢rank怎样拖住整个训练作业

同步训练中，各rank可能使用不同数据、经历不同读取时间，但梯度归约需要它们按同一协议参与。只观察平均GPU利用率，很容易错过最早落后的那一张卡。

## 把八个rank放到同一时间线

教学设定中，七个rank在100毫秒完成反向，一个rank因数据读取迟到30毫秒。其他rank可能在collective等待到130毫秒，随后才能完成更新。等待出现在通信段，根因却可能早在loader。

因此同时记录数据就绪、计算结束、collective进入与完成时刻。最早发生的偏离比最后报出的超时更能缩小范围。若每个rank进入时间相近，才进一步调查链路与通信路径。

## 保存间隔决定故障损失

若每隔I秒保存一次，故障在区间内近似均匀发生，平均丢失I/2秒工作。单次保存耗时c，平均故障间隔M，忽略恢复与重叠时，损失比例近似为

$$
f(I)=c/I+I/(2M).
$$

求导得到 $I_*\approx\sqrt{2cM}$。例如c=75秒、M=36000秒，近似最优间隔2324秒，约38.7分钟。这是简化模型，真实故障并非独立均匀、异步保存也会与计算争抢资源，需用实际作业记录修正。

## 写完文件以后还差一次提交

分布式checkpoint由多个分片组成。部分文件写完但其他rank失败，不应被标成可恢复版本。异步保存还要确保读取的是冻结快照，避免优化器更新同时改动待写数据。

保留明确的完整性标记与版本信息，再验证模型、优化器、日程和数据进度能一起恢复。不要仅凭目录存在或文件大小非零判断保存成功。

![Kimi K2 RL checkpoint engine 原论文图](../assets/images/paper-deep-dives/technical-reports/kimi-k2/figure-10-rl-checkpoint-engine.png){ width="760" }

<small>图源：[Kimi K2: Open Agentic Intelligence](https://arxiv.org/abs/2507.20534)，Figure 10。原论文图意：co-located distributed checkpoint engine 在 training engine 与 inference engine 之间管理参数状态，避免通过普通网络文件系统重新切分和广播超大模型参数。</small>

## 恢复要回到哪个状态

在step400保存、step420失败，恢复从400开始就会重做后面一段。日志可能已记录到419，训练状态却没有。为恢复分支记录父运行、checkpoint与起始更新，避免曲线连续造成“没有丢工作”的错觉。

短程故障演练应比较连续运行与恢复路径的输入及更新。若只保存权重用于推理，恢复承诺与完整续训不同，需要明确用途。

## 实验记录围绕一个可复现更新组织

固定代码、环境、模型、数据、tokenizer与并行配置，记录实际token曝光、step时间、显存和失败位置。版本发生变化时先做最小作业，再扩到长跑。

费用可按GPU小时统计，但效率还取决于有效监督token和可用checkpoint进度。设备一直运行却重复失败重做，与稳定推进训练具有不同成本。


## 对照源码与延伸阅读

- [训练更新、日志与checkpoint的执行边界](https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/train.py)
