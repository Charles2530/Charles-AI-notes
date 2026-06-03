# 扩散模型阅读入口：从加噪训练到少步生成

扩散模型最容易被讲成一堆公式：DDPM、score、SDE、ODE、guidance、distillation。更实用的读法是先把它看成一条数据和噪声之间的路径：训练阶段学会在路径上识别方向，采样阶段沿着方向把噪声带回数据，工程阶段再想办法减少步数、接入条件并控制失败模式。

这一章的写法会吸收 Lilian Weng、Distill 式图解、国内高质量博客和公众号的优点：从直觉进入，再把数学和系统代价补齐。具体事实仍回到 DDPM、Score SDE、DPM-Solver、DMD、Rectified Flow 等论文与官方材料。

## 按问题进入

| 你想弄清的问题 | 建议先读 |
| --- | --- |
| 扩散训练为什么常预测噪声，而不是直接生成图像 | [扩散训练与表示](training.md) |
| beta、sigma、v-pred、EDM 这些参数化到底改了什么 | [噪声日程与参数化](noise-schedules-and-parameterization.md) |
| score matching、SDE、probability flow ODE 如何接起来 | [Score Matching、SDE 与 Probability Flow](score-matching-sde-and-probability-flow.md) |
| prompt、ControlNet、classifier-free guidance 为什么会影响构图和过饱和 | [条件控制与 Guidance](guidance-and-conditioning.md) |

## 按工程目标进入

| 目标 | 建议先读 |
| --- | --- |
| 降低采样步数和线上成本 | [扩散推理与采样加速](inference.md) |
| 在 DDPM、DDIM、Solver、Flow、Consistency 之间做选择 | [扩散方法对比表](comparison-table.md) |
| 理解一步/少步生成的训练代价 | [扩散蒸馏](distillation.md) |
| 对比 Consistency 与 Rectified Flow 的路径设计 | [一致性模型与 Rectified Flow](consistency-models-and-rectified-flow.md) |
| 处理视频、多模态和长时一致性 | [视频与多模态扩散](video-and-multimodal-diffusion.md) |

## 读论文时抓什么

扩散论文的关键不只在“生成效果更好”。建议每篇都追问四件事：它改变的是训练目标、噪声路径、采样器、条件接口还是数据配方；它把成本转移到了训练、蒸馏、推理还是人工筛选；它的评测有没有拆开画质、语义、身份一致性和时序一致性；它在少步、长视频或强 guidance 下最容易坏在哪里。
