# 扩散模型专题：从加噪训练到少步生成

扩散模型最容易被讲成一堆公式：DDPM、score、SDE、ODE、guidance、distillation。更实用的读法是先把它看成一条数据和噪声之间的路径：训练阶段学会在路径上识别方向，采样阶段沿着方向把噪声带回数据，工程阶段再想办法减少步数、接入条件并控制失败模式。

这一章先从直觉进入，再把数学和系统代价补齐。直觉负责让概念能被看见，公式负责说明这些概念为什么成立，工程部分则回答它们在训练、采样和部署里会付出什么成本。

## 按问题进入

| 你想弄清的问题 | 建议先读 |
| --- | --- |
| 扩散训练为什么常预测噪声，而不是直接生成图像 | [扩散训练与表示](training.md) |
| beta、sigma、v-pred、EDM 这些参数化到底改了什么 | [噪声日程与参数化](noise-schedules-and-parameterization.md) |
| score matching、SDE、probability flow ODE 如何接起来 | [Score Matching、SDE 与 Probability Flow](score-matching-sde-and-probability-flow.md) |
| prompt、ControlNet、classifier-free guidance 为什么会影响构图和过饱和 | [条件控制与 Guidance](guidance-and-conditioning.md) |

## 一条更顺的理解路径

扩散模型不必一开始就从变分推断、SDE 或 solver 名词表进入。更省力的读法，是先抓住“数据和噪声之间的路径”。

第一步是 DDPM 的“拆楼 / 建楼”：前向过程不是模型学出来的，而是我们人为规定的拆解过程。它把一张真实样本 \(x_0\) 一步步拆成接近高斯噪声的 \(x_T\)。训练要学的不是凭空造图，而是在任意中间层 \(x_t\) 上判断该往哪一步修回去。

第二步是几何图：如果把 \(x_0\) 和噪声 \(\epsilon\) 近似画成正交轴，那么 DDPM / DDIM 的样本可以理解成沿圆弧改变数据系数和噪声系数；Flow Matching 则常用从数据到噪声的直线插值来组织速度学习。这个图解很适合建立直觉，但不要把它理解成“Flow Matching 只是 DDIM 换了条线”：它们的训练目标和采样接口不同。

第三步是随机性：DDIM 的随机项不是随手加噪，而是在保持下一时间步总噪声幅度的前提下，用一部分新噪声替换一部分旧噪声。把这个原则推广到 Flow Matching 时，也要检查数据系数和噪声系数是否仍符合目标时间步，否则随机性会变成额外污染。

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
