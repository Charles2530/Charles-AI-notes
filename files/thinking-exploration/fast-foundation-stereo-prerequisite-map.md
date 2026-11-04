# 怎样读懂双目模型的几何与加速

双目网络输入左右图，输出视差；部署时又希望它足够快。理解Fast-FoundationStereo需要把两件事连起来，视差误差怎样改变深度，以及特征、代价体和迭代细化各消耗什么计算。本文沿同一组相机参数解释这些基础，再对照论文方法图。

## 先看任务：双目到底在估什么

双目相机有左右两个镜头，中间距离叫 baseline。同一个 3D 点会落在左右图的不同水平位置，这个横向位移叫 disparity。知道焦距 \(f\)、baseline \(B\) 和 disparity \(d\) 后，可以用下面这个公式理解深度：

\[
z = \frac{fB}{d}
\]

disparity 越大，通常说明物体越近；disparity 越小，通常说明物体越远。这个公式很短，但它解释了 stereo 为什么适合机器人和 AR：它不是只给一张“看起来有深度感”的图，而是在每个像素上给出可转成几何距离的信号。

![Stereo camera model 网页图](../assets/images/localized-remote/embodied-ai/cameras-depth-and-robot-vision/stereo-camera-model-wikimedia.jpg){ width="760" }

<small>网页图源：[Wikimedia Commons: Stereo-camera-model.jpg](https://commons.wikimedia.org/wiki/File:Stereo-camera-model.jpg)。原图表达：双目深度来自 baseline 和左右图上的 disparity；同一个点在左右图中位置差越大，通常离相机越近。读图重点：这张图只解释几何前提，真正读 Fast-FoundationStereo 时还要继续检查 rectification、匹配代价和深度误差传播。</small>

这里有一个容易被初学者跳过的输入假设：stereo matching 通常默认左右图已经 rectified，也就是同一个 3D 点在左右图里位于同一条水平扫描线。这样模型只需要沿横向搜索 \(d\)，而不是在整张右图里找匹配点。如果相机没标定好、左右不同步、畸变没去掉，再强的模型也会被输入质量拖垮。

所以读 Fast-FoundationStereo 时，先把任务写成：

```text
输入：rectified left / right stereo images
输出：dense disparity map，再转成 depth
关键风险：低纹理、反光、透明、遮挡、远距离和标定漂移
```

## 远处为何对同样的视差误差更敏感

设校正后的双目焦距f=700 pixel，基线B=0.12 m，则fB=84 pixel·m。视差d=28 pixel时，深度z=84/28=3 m；d=7时，深度为12 m。两种距离都发生1 pixel误差，结果却不同。

| 真实视差 | 真实深度 | 视差少1 pixel的深度 | 深度偏差 |
| --- | ---: | ---: | ---: |
| 28 pixel | 3 m | 84/27≈3.111 m | 约0.111 m |
| 7 pixel | 12 m | 84/6=14 m | 2 m |

局部导数 `dz/dd=−fB/d²` 说明小视差更敏感，有限误差还呈非线性。正负1 pixel造成的米制误差并不对称。BP2统计超出2 pixel阈值的比例，不能独自表示不同距离的米制风险。

```python
fB = 700*0.12
for d in [28.,7.]:
    z = fB/d
    shifted = fB/(d-1)
    print(z, round(shifted-z,6))
```

## 从 Disparity 到 Cost Volume

双目网络不是凭空吐出深度。它先提取左右图的 feature，再为每个像素搜索一组可能的 disparity，把每个候选的匹配信息堆成 cost volume。cost volume 可以理解成一张四维评分表：

\[
\mathbf{V} \in \mathbb{R}^{C \times D \times H \times W}
\]

这里 \(H,W\) 是图像空间位置，\(D\) 是 disparity 候选数，\(C\) 是匹配特征通道。它贵就贵在多了一维 \(D\)：每个像素不只要算一次，还要沿许多候选位移反复比较。

```mermaid
flowchart LR
    L["Left image"] --> FL["Left features"]
    R["Right image"] --> FR["Right features"]
    FL --> CV["Cost volume<br/>D x H x W"]
    FR --> CV
    CV --> CF["Cost filtering<br/>3D conv / transformer"]
    CF --> ID["Initial disparity"]
    ID --> RF["Refinement<br/>ConvGRU / upsampling"]
    RF --> OUT["Dense disparity / depth"]
```

cost volume 只是原始证据，还很嘈杂。低纹理墙面、玻璃、反光金属、重复花纹和遮挡区域都会制造错误匹配。因此模型还需要 cost filtering：用 3D convolution、hourglass、axial/planar convolution 或 Transformer 在空间和 disparity 维度上传播上下文，判断哪些匹配更可信。

这些名字可以先不用当成新算法背。它们在这里都服务同一件事：**让一个像素的匹配判断不要只看自己，而是参考周围像素、相邻 disparity 和更大范围的几何上下文**。

| 模块 | 在 cost filtering 里做什么 | 可以怎么直觉理解 |
| --- | --- | --- |
| 3D convolution | 同时沿 \(D,H,W\) 三个维度看邻居，让相邻 disparity 和相邻像素互相校正 | 不只问“这个像素像不像”，还问“附近像素和附近深度候选是否也支持这个判断” |
| 3D hourglass | 先下采样聚合大范围上下文，再上采样回到细分辨率 | 先远看整块墙、桌面和物体轮廓，再回来修每个像素的 disparity |
| axial / planar convolution | 把大范围 3D 处理拆成沿轴或平面传播，减少直接 3D 卷积的成本 | 不一次看完整大方块，而是分方向、分平面传递信息，省一些计算 |
| Transformer | 用 attention 让远处但相关的位置互相参考，建模长程匹配关系 | 重复纹理或遮挡时，允许模型参考更远的上下文，而不是困在局部窗口里 |

所以 cost filtering 的输出不是“重新拍脑袋预测深度”，而是在清理 cost volume：压低不一致的候选 disparity，抬高被上下文支持的候选 disparity，最后形成更可靠的 initial disparity。

最后还需要 refinement。初始 disparity 往往在边界、薄结构和遮挡区域不够细，迭代 refinement 会用图像特征和 hidden state 一轮轮修正它。Fast-FoundationStereo 里提到的 ConvGRU refinement，就属于这条路径。

## 多一个视差轴，内存增加多少

假设特征高度128、宽度256，搜索64个视差，匹配特征有8通道，单个cost volume有 `8×64×128×256=16,777,216` 个元素，FP16约32 MiB。高度和宽度各扩大两倍变成128 MiB；若为了覆盖同一实际范围又把视差候选加倍，则变成256 MiB。

这只是一份代价体，未包含反向激活和临时workspace。它解释了为什么搜索分辨率、视差范围和通道数都影响资源，不能仅看backbone参数量。具体实现可能用相关性、多尺度或按需查询，必须按实际张量计算。

## 特征、代价过滤与细化各为什么耗时

FoundationStereo将视觉特征、代价过滤与迭代细化接入同一前向。具体zero-shot效果要看原文任务与数据集；从计算结构上可以先确认，三部分都处于推理路径，需要分别计时。

![Fast-FoundationStereo framework 原图](../assets/images/paper-deep-dives/inference/fast-foundation-stereo/figure-3-framework.png){ width="920" }

<small>图源：`Fast-FoundationStereo`，Figure 3。原论文图意：FoundationStereo pipeline 包含 feature extraction、cost filtering 和 disparity refinement；Fast-FoundationStereo 分别用 hybrid prior distillation、cost filtering blockwise search 和 refinement pruning 加速。</small>

这张图可以从左到右读成三段：

| 阶段 | 它在做什么 | 为什么贵 |
| --- | --- | --- |
| Feature extraction | 从左右图提取可匹配的视觉特征 | DepthAnything V2 + side-tuning CNN 这类强 backbone 很重 |
| Cost filtering | 在 cost volume 上判断哪些 disparity 合理 | cost volume 多了 \(D\) 维，3D hourglass / Transformer 都不便宜 |
| Disparity refinement | 迭代修正初始 disparity | ConvGRU 多轮更新，重复调用会累积 latency |

读到这里，Fast-FoundationStereo 的主张就清楚了：它不是把整个 teacher 粗暴换成一个小模型，而是先问“热路径里每一段为什么慢”，再给每一段配不同工具。

## 三个瓶颈，对应三种加速手段

这篇论文最像工程方法论的一点，是它承认不同模块的冗余形态不同。

| 瓶颈 | 加速手段 | 为什么这样选 |
| --- | --- | --- |
| Backbone 太重 | Feature distillation | teacher 的中间 feature 可以直接监督 student，适合把强先验换成轻执行形态 |
| Cost filtering 太重 | Blockwise NAS | cost volume 通道本来就小，直接剪枝收益有限；搜索替代 block 更合适 |
| Refinement 有迭代冗余 | Structured pruning | ConvGRU 重复更新，通道冗余更容易变成真实 runtime 收益 |

蒸馏的关键是 teacher/student：teacher 能力强但慢，student 更小更快。Fast-FoundationStereo 的 backbone 蒸馏不是只学最终 disparity，而是让 student 学 teacher 的 multi-level feature pyramid。这样 student 不只是背答案，而是在学习 teacher 的中间视觉表示。

NAS 的关键是不要手工猜结构。论文把 cost filtering 拆成多个 blocks，训练候选 block 贴近 teacher block，再记录每个候选带来的误差变化和耗时变化，最后用 ILP 在 latency budget 下选组合。初学者可以先把它理解成：不是试遍所有完整网络，而是把大搜索拆成一组局部替换题。

剪枝的关键是 structured pruning。删单个 weight 不一定让 GPU 更快；删通道、filter、head 或 block，才更容易变成更小的 dense tensor shape。Refinement ConvGRU 还有 recurrent hidden state，所以剪枝必须遵守依赖图，不能随便删通道。

## 为什么不能只看 MACs

高效推理论文里最容易误读的是“参数少、MACs 少，所以一定快”。真实系统没有这么乖。latency 还受 kernel、memory bandwidth、layout 转换、算子融合、TensorRT 支持、输入 shape 和 batch 影响。

![Runtime decomposition 原图](../assets/images/paper-deep-dives/inference/fast-foundation-stereo/figure-10-runtime-detail.png){ width="760" }

<small>图源：`Fast-FoundationStereo`，Figure 10。原论文图意：在 NVIDIA 3090 上，FoundationStereo 的 feature extraction、cost filtering 和 refinement 都比 Fast-FoundationStereo 更耗时，三段共同构成超过 10x 的总加速。</small>

这张 runtime decomposition 比单个 FPS 数字更有用，因为它告诉你收益是不是真的进入了热路径。如果只看到总 runtime 从 496ms 到 49ms，你还不知道是哪个阶段快了；拆开后才能判断 feature、cost filtering、refinement 三段是否都被压下来。

读 Table 5 这类效率表时，可以把三种数字分开：

| 数字 | 说明什么 | 不能说明什么 |
| --- | --- | --- |
| Params | 模型存了多少权重 | 不等于每次推理耗时 |
| MACs | 某个输入 shape 下有多少乘加 | 不包含所有 memory movement 和 runtime 开销 |
| Runtime | 真实硬件上跑完一次的时间 | 依赖实现、硬件、TensorRT 和输入设置 |

Fast-FoundationStereo 的 TensorRT 版本进一步变快，也提醒我们：模型结构和部署工具链是一起设计的。一个理论上 MACs 少的结构，如果难以导出、难以融合或 fallback 到慢 kernel，线上收益会打折。

## 伪标签为什么也是高效推理的一部分

很多人会把高效推理理解成“只改结构”。Fast-FoundationStereo 反过来提醒：结构变小以后，数据质量更重要。小模型容量更有限，如果只在窄分布或合成数据上训练，很容易变成“快但不稳”的传统实时 stereo 网络。

![Pseudo-labeling pipeline 原图](../assets/images/paper-deep-dives/inference/fast-foundation-stereo/figure-6-pseudo-labeling.png){ width="720" }

<small>图源：`Fast-FoundationStereo`，Figure 6。原论文图意：对 in-the-wild internet stereo data，先用 stereo teacher 预测 disparity，再用 monocular depth estimator 和 normal consistency 做质量检查，最后生成可用于训练的 pseudo labels。</small>

论文从 Stereo4D 取真实双目图，用 FoundationStereo teacher 预测 disparity，再用 monocular depth estimator 提供另一条几何证据。两边都反投影成 3D 后，用 normal consistency 检查局部表面方向是否一致。sky region 还会单独过滤，因为天空没有稳定可匹配的真实深度。

为什么不直接比较 depth？因为真实场景的深度范围差异很大，尺度和远近会让直接数值比较不稳。normal map 更关注局部表面方向，适合做质量门禁。对 student 来说，这一步的意义是：别把 teacher 的明显错误也一起学进去。

所以这篇论文里的“高效”包含两层：

1. 推理时模型更轻，latency 更低；
2. 训练时用更好的真实分布伪标签，让轻模型不要因为容量小而丢掉 zero-shot 泛化。

## 把加速结果与几何误差一起检查

先保持同一相机、输入分辨率、视差范围与任务集，分别比较特征替换、结构搜索和剪枝。延迟表要包含硬件与runtime，质量表要包含距离、遮挡和薄结构等分组。若更快的版本主要在远处出错，总体BP指标可能不足以解释后续机器人定位误差。

论文的组合方案收益不能全部归给其中一种压缩。蒸馏数据和伪标签过滤也变化时，应查对应消融，再判断哪个机制支持哪个结论。

## 建议阅读路线

最稳的站内路线是：

1. [相机、深度与机器人视觉](../embodied-ai/cameras-depth-and-robot-vision.md)：先知道双目、baseline、disparity、标定和深度公式。
2. [双目匹配与 Cost Volume 入门](../embodied-ai/stereo-matching-and-cost-volume.md)：理解 cost volume、cost filtering、refinement 和 BP/D1 指标。
3. [模型蒸馏入门](../foundations/model-distillation-basics.md)：理解 teacher/student、feature distillation 和中间监督。
4. [模型压缩、剪枝与 NAS 入门](../foundations/model-compression-pruning-and-nas-basics.md)：理解 structured pruning、blockwise NAS、ILP 和 latency 验证。
5. [Depth Anything V2](../paper-deep-dives/embodied-ai/depth-anything-v2.md)：理解单目深度 foundation model、pseudo-label 和真实分布补齐。
6. [Fast-FoundationStereo](../paper-deep-dives/inference/fast-foundation-stereo.md)：回到论文，看它如何把这些知识合成一个实时 zero-shot stereo model。
