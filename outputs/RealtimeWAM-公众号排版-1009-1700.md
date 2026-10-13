# 近乎无损，25 倍加速！RealtimeWAM 让 WAM 迈入单步实时推理

LightX2V 团队联合南洋理工大学、北京航空航天大学等研究团队推出的 RealtimeWAM，面向 Mixture-of-Transformers（MoT）架构的 World Action Model（WAM）实现极致加速。

在将动作生成从 10 步压缩到 1 步的同时，RealtimeWAM 保持了接近原模型的任务成功率：在 RoboTwin 2.0 上，基于 Fast-WAM 和 Faster-WAM 的两个版本分别达到 **90.84% 和 92.64%**，相比各自的 10 步模型仅下降 **0.67 和 0.29 个百分点**。

在速度上，基于 Fast-WAM 的 RealtimeWAM 在单张 NVIDIA H100 上将单次推理从约 300 ms 降至 **12.2 ms**，实现**约 25 倍加速**。

现有 MoT 架构 WAM 的推理主要面临两个瓶颈：一是动作专家反复去噪带来的大量重复计算，二是视频与动作专家串行执行带来的等待开销。

为此，RealtimeWAM 提出 **TACD 蒸馏**，将多步动作生成压缩为高精度单步生成；同时提出 **CEWP 异步流水线**，通过逐块共享 Video KV，让视频与动作专家并行执行，进一步降低推理延迟。

推理代码、测评代码及模型权重均已开源。

<img src="https://raw.githubusercontent.com/Charles2530/Charles-AI-notes/dev/v4/files/assets/images/realtimewam-image1.png" alt="论文首页｜RealtimeWAM: One-Step Asynchronous World Action Models" width="600" style="display: block; width: 100%; max-width: 600px; height: auto; margin: 24px auto 8px;">

<p align="center" style="margin: 0 0 24px; color: #888888; font-size: 12px; line-height: 1.6; text-align: center;">论文首页｜RealtimeWAM: One-Step Asynchronous World Action Models</p>

---

## 01｜两个瓶颈：反复去噪，以及专家之间的等待

WAM 将未来视觉预测作为学习目标，让机器人借助视频模型的表示理解动作与环境变化之间的联系。在 Fast-WAM、Faster-WAM 等 MoT 架构中，视频专家（Video Expert）与动作专家（Action Expert）分别负责视觉表示和动作生成，视频专家只需计算一次，其 Key/Value（KV）缓存便可供动作专家重复使用。

### 瓶颈一：动作专家内部仍在反复计算

视频表示能够复用，动作去噪却仍要运行多步。每一次去噪都需要经过动作网络，多轮计算累积出可观的延迟。论文的耗时分析显示，Fast-WAM 的动作专家在多种 GPU 上占据**接近 90%** 的推理时间。减少动作去噪步数，因此成为加速最直接的切入点。

### 瓶颈二：两个专家仍按整网顺序执行

即使动作生成已经压缩为一步，串行调度仍会要求视频专家先完成整个前向过程，再启动动作专家。实际上，动作网络的某一层并不需要等视频网络的所有层结束。把整张网络设为同步边界，会让已经具备计算条件的操作继续等待。

<img src="https://raw.githubusercontent.com/Charles2530/Charles-AI-notes/dev/v4/files/assets/images/realtimewam-image2.png" alt="图 1｜Fast-WAM（左）与 Faster-WAM（右）的推理延迟及原模型各组件耗时占比" width="600" style="display: block; width: 100%; max-width: 600px; height: auto; margin: 24px auto 8px;">

<p align="center" style="margin: 0 0 24px; color: #888888; font-size: 12px; line-height: 1.6; text-align: center;">图 1｜Fast-WAM（左）与 Faster-WAM（右）的推理延迟及原模型各组件耗时占比</p>

RealtimeWAM 分别处理这两种开销：TACD 减少动作专家的重复去噪，CEWP 让剩余的视频与动作计算尽可能重叠。

---

## 02｜TACD：让学生既保持一致，也对准教师终点

少跑几步，并不自然等于高质量的单步生成。以 RoboTwin 2.0 为例，直接把 Fast-WAM 的动作去噪从 10 步改为 1 步，整体成功率会从 **91.51% 降至 69.27%**。真正的难点，是在压缩计算的同时保留动作精度。

常规一致性蒸馏约束相邻噪声状态下的预测结果保持一致，但这种局部一致性并不自动保证最终动作准确。学生和 EMA 目标模型即使彼此接近，也可能共同偏离正确的动作终点。论文将这一问题称为**局部—全局误差差距**（local-global error gap）。对抓取、放置等操作而言，细小的动作偏差也可能影响任务是否成功。

<img src="https://raw.githubusercontent.com/Charles2530/Charles-AI-notes/dev/v4/files/assets/images/realtimewam-image3.png" alt="图 2｜TACD 在局部一致性监督之外，引入冻结教师多步 rollout 终点的约束" width="600" style="display: block; width: 100%; max-width: 600px; height: auto; margin: 24px auto 8px;">

<p align="center" style="margin: 0 0 24px; color: #888888; font-size: 12px; line-height: 1.6; text-align: center;">图 2｜TACD 在局部一致性监督之外，引入冻结教师多步 rollout 终点的约束</p>

为此，**教师锚定一致性蒸馏**（Teacher-Anchored Consistency Distillation，TACD）保留局部一致性目标，同时增加一个明确的全局参照：从同一个带噪动作出发，让冻结教师完成多步去噪，得到干净动作终点，再用这一终点引导学生的一步预测。具体实现通过对齐学生速度与指向教师终点的区间平均速度，约束最终动作误差。

训练时，视频专家保持冻结，Teacher、Student 和 EMA 模型共享同一份 Video KV。EMA 是学生参数的指数滑动平均，用于提供一致性训练目标。论文默认使用**教师 10 步 rollout**，并只训练动作专家的**LoRA** 参数，使蒸馏集中在动作生成能力上。

<img src="https://raw.githubusercontent.com/Charles2530/Charles-AI-notes/dev/v4/files/assets/images/realtimewam-image4.png" alt="图 3｜两种方法均能降低一致性损失（左）；TACD 进一步减小学生速度与教师平均速度的偏差（右）" width="600" style="display: block; width: 100%; max-width: 600px; height: auto; margin: 24px auto 8px;">

<p align="center" style="margin: 0 0 24px; color: #888888; font-size: 12px; line-height: 1.6; text-align: center;">图 3｜两种方法均能降低一致性损失（左）；TACD 进一步减小学生速度与教师平均速度的偏差（右）</p>

这一设计也得到消融实验支持：保持视频专家冻结时，在局部一致性损失之外加入教师锚定约束，RoboTwin 2.0 整体成功率从 **89.85% 提升到 90.84%**（论文 Table 3）。局部一致性与终点约束，承担了不同的训练职责。

---

## 03｜CEWP：KV 就绪一层，动作就向前推进一层

将动作生成压缩为单步后，视频与动作专家之间的串行等待成为进一步提速的瓶颈。**跨专家波前流水线**（Cross-Expert Wavefront Pipelining，CEWP）利用两者的块级数据依赖，让两个专家沿模型深度交错推进：**动作块只需等待对应的视频 KV，无须等整个视频专家完成前向计算**。视频 KV 完成投影及必要预处理后即可逐块共享，动作专家也可提前执行自身的 Q/K/V 投影，从而让两侧计算尽可能重叠。

<img src="https://raw.githubusercontent.com/Charles2530/Charles-AI-notes/dev/v4/files/assets/images/realtimewam-image5.png" alt="图 4｜专家级串行执行（上）与 CEWP（下）。动作流仅在注意力使用对应视频 KV 前同步" width="600" style="display: block; width: 100%; max-width: 600px; height: auto; margin: 24px auto 8px;">

<p align="center" style="margin: 0 0 24px; color: #888888; font-size: 12px; line-height: 1.6; text-align: center;">图 4｜专家级串行执行（上）与 CEWP（下）。动作流仅在注意力使用对应视频 KV 前同步</p>

具体实现中，CEWP 将视频与动作专家放在不同的 CUDA stream 上：视频流逐块发布 KV 并记录就绪事件，动作流先执行不依赖视频 KV 的计算，仅在注意力读取对应 KV 前进行必要同步。这样，原本整网级的等待被拆解为块级同步，进一步降低推理延迟。该调度保留原有参数、算子和数据依赖，无需额外训练；论文附录也给出了与串行计算的等价性分析。

TACD 负责减少重复去噪，CEWP 负责重叠剩余计算，二者共同实现高精度、低延迟的动作生成。

---

## 04｜精度实验：从 10 步到 1 步，成功率变化有多大

论文在 RoboTwin 2.0、LIBERO 和 LIBERO-Plus 上以任务成功率进行评估。其中，RealtimeWAM\* 基于 Fast-WAM，RealtimeWAM† 基于 Faster-WAM；两个版本均使用单步动作生成，对应原模型使用 10 步动作去噪。

<img src="https://raw.githubusercontent.com/Charles2530/Charles-AI-notes/dev/v4/files/assets/images/realtimewam-image6.png" alt="表 1｜RoboTwin 2.0 与 LIBERO 的完整对比；NFE 表示去噪步数；*、† 分别对应 Fast-WAM、Faster-WAM" width="600" style="display: block; width: 100%; max-width: 600px; height: auto; margin: 24px auto 8px;">

<p align="center" style="margin: 0 0 24px; color: #888888; font-size: 12px; line-height: 1.6; text-align: center;">表 1｜RoboTwin 2.0 与 LIBERO 的完整对比<br>NFE 表示去噪步数；*、† 分别对应 Fast-WAM、Faster-WAM</p>

### RoboTwin 2.0：两个版本均保持接近原模型的成功率

RealtimeWAM\* 的整体成功率为 90.84%，对应 10 步 Fast-WAM 的 91.51%，仅下降 0.67 个百分点；RealtimeWAM† 为 92.64%，对应 Faster-WAM 的 92.93%，仅下降 0.29 个百分点。这里的 Overall 是 Clean 与 Random 两种条件下成功率的平均值。

在同一张结果表中，普通一致性蒸馏和 DMD 的整体成功率分别为 89.63% 和 88.22%。RealtimeWAM\* 达到 90.84%，说明针对动作终点设计的约束，有助于保留单步生成的任务表现。

### LIBERO 与 LIBERO-Plus：兼顾标准任务和扰动场景

在 LIBERO 上，RealtimeWAM\* 保持 **97.0%** 的整体成功率，与 10 步 Fast-WAM 持平；RealtimeWAM† 达到 **99.0%**，对应 Faster-WAM 的 98.9%。单步生成在这些评测中保留了与多步模型相当的整体表现。

<img src="https://raw.githubusercontent.com/Charles2530/Charles-AI-notes/dev/v4/files/assets/images/realtimewam-image7.png" alt="表 2｜LIBERO-Plus 七类扰动测试；RealtimeWAM† 与 Faster-WAM 的整体成功率分别为 73.0% 和 73.6%" width="600" style="display: block; width: 100%; max-width: 600px; height: auto; margin: 24px auto 8px;">

<p align="center" style="margin: 0 0 24px; color: #888888; font-size: 12px; line-height: 1.6; text-align: center;">表 2｜LIBERO-Plus 七类扰动测试<br>RealtimeWAM† 与 Faster-WAM 的整体成功率分别为 73.0% 和 73.6%</p>

LIBERO-Plus 进一步引入视角、语言、光照、背景和传感噪声等七类扰动。RealtimeWAM† 达到 73.0% 的整体成功率，较 Faster-WAM 的 73.6% 仅下降 0.6 个百分点，并在相机、噪声和布局扰动子集上略有提升。这些结果表明，蒸馏后模型也保留了较强的分布外鲁棒性。

---

## 05｜速度实验：极致加速是怎样实现的

RealtimeWAM 将算法与系统优化组合到同一条推理链路中。论文在单张 H100 上逐项加入 TACD、CUDA Graph、CEWP 和高效算子，展示了每一项设计带来的延迟变化。

<img src="https://raw.githubusercontent.com/Charles2530/Charles-AI-notes/dev/v4/files/assets/images/realtimewam-image8.png" alt="图 5｜H100 上的逐项加速效果：Fast-WAM（左），Faster-WAM（右）" width="600" style="display: block; width: 100%; max-width: 600px; height: auto; margin: 24px auto 8px;">

<p align="center" style="margin: 0 0 24px; color: #888888; font-size: 12px; line-height: 1.6; text-align: center;">图 5｜H100 上的逐项加速效果：Fast-WAM（左），Faster-WAM（右）</p>

对 Fast-WAM，TACD 首先将延迟从 299.7 ms 降到 65.8 ms；加入 CUDA Graph 后降到 23.3 ms；CEWP 进一步降到 17.4 ms；结合 LightX2V 高效算子后，最终达到 12.2 ms，总体加速**约 24.6 倍**。

对 Faster-WAM，同样的优化流程将延迟从 218.9 ms 降到 16.1 ms，总体加速**约 13.6 倍**。

CUDA Graph 减少 CPU 提交开销，算子融合与投影打包减少 kernel 启动和中间内存访问；这些优化与单步蒸馏、异步流水线相互补充。

### 从 H100 到消费级 GPU，收益同样可见

在 RTX 5090 和 RTX 4090D 上，基于 Fast-WAM 的版本分别达到 17.8 ms 和 27.2 ms，对应 14.1 倍和 17.7 倍加速；基于 Faster-WAM 的版本分别达到 30.9 ms 和 44.7 ms，对应 6.9 倍和 8.1 倍加速。

加速幅度随模型和硬件变化，最高约 25 倍对应 H100 上的 Fast-WAM 版本。

<img src="https://raw.githubusercontent.com/Charles2530/Charles-AI-notes/dev/v4/files/assets/images/realtimewam-image9.png" alt="表 F.3｜RTX 5090 与 RTX 4090D 上的延迟及累计加速；加速比均相对于同一模型、同一 GPU 的未优化基线" width="600" style="display: block; width: 100%; max-width: 600px; height: auto; margin: 24px auto 8px;">

<p align="center" style="margin: 0 0 24px; color: #888888; font-size: 12px; line-height: 1.6; text-align: center;">表 F.3｜RTX 5090 与 RTX 4090D 上的延迟及累计加速<br>加速比均相对于同一模型、同一 GPU 的未优化基线</p>

> **测速口径**
>
> 论文统计的推理延迟包含 VAE 编码，不包含文本编码，因为语言指令在每个任务回合只需编码一次。
>
> 该指标是模型单次推理耗时，不包含机器人完成整段动作所需的执行时间。

---

## 06｜真实机器人：连续完成长时序折衣

除仿真基准外，论文还在由两台 AgileX PiPER 机械臂组成的双臂平台上，使用 RTX 4090 部署 RealtimeWAM\*，进行**真实 T 恤折叠的定性验证**。任务包含连续抓取和多次折叠，每一步都会改变衣物形态及下一步的可抓取位置，对动作精度与连续决策提出要求。

<img src="https://raw.githubusercontent.com/Charles2530/Charles-AI-notes/dev/v4/files/assets/images/realtimewam-image10.png" alt="图 F.1｜真实双臂折衣过程的四个关键帧" width="600" style="display: block; width: 100%; max-width: 600px; height: auto; margin: 24px auto 8px;">

<p align="center" style="margin: 0 0 24px; color: #888888; font-size: 12px; line-height: 1.6; text-align: center;">图 F.1｜真实双臂折衣过程的四个关键帧</p>

论文展示的执行序列中，机器人能够连续完成折叠，动作之间没有明显停顿。这一展示补充了仿真中的成功率与延迟结果，也说明单步动作生成与专家间的异步流水线可以用于真实长时序操作。

---

## 07｜代码与模型已开源

RealtimeWAM 的思路很明确：用教师终点约束保留单步动作质量，再通过块级数据依赖把视频与动作计算重叠起来。结合 LightX2V 的推理优化，WAM 的单次推理从数百毫秒缩短到十余毫秒，为更及时的机器人决策提供了计算基础。

推理、测评代码及模型权重均已开放，欢迎阅读论文、复现实验、试用模型，也欢迎交流与 Star！

- **论文**：[RealtimeWAM: One-Step Asynchronous World Action Models](https://arxiv.org/abs/2610.06617)

- **GitHub**：[LightX2V / RealtimeWAM](https://github.com/ModelTC/LightX2V/tree/main/examples/realtimewam)

- **Hugging Face**：[RealtimeWAM 模型权重](https://huggingface.co/lightx2v/RealtimeWAM)
