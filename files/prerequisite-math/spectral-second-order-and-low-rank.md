# 谱分解、二阶近似与低秩结构

这篇回答一个问题：为什么论文精读里会频繁出现特征值、SVD、谱能量、Hessian、Taylor 近似和低秩结构。

很多论文不是只看一个矩阵有多大，而是看它的能量集中在哪些方向；不是只看 loss 变大变小，而是看小扰动会不会被二阶项放大。谱分解和二阶近似，就是把这些“方向”和“敏感度”讲清楚。

## 特征值：一个方向被放大多少

如果一个方阵 \(A\) 满足：

\[
Av=\lambda v
\]

那么 \(v\) 是特征向量，\(\lambda\) 是特征值。

这行公式的意思是：矩阵 \(A\) 作用在方向 \(v\) 上时，没有把它转到别的方向，只是把它缩放了 \(\lambda\) 倍。如果 \(|\lambda|>1\)，这个方向会被放大；如果 \(|\lambda|<1\)，这个方向会被压缩；如果 \(\lambda<0\)，方向还会翻转。

后面看动态系统、RNN、世界模型 rollout 或 optimizer 稳定性时，特征值经常在背后起作用：某些方向反复被放大，误差就会越来越大。

## SVD：把任意矩阵拆成方向、强度和方向

奇异值分解写成：

\[
W=U\Sigma V^\top
\]

逐项读：

| 符号 | 含义 |
| --- | --- |
| \(V^\top\) | 先把输入投到一组右奇异方向上 |
| \(\Sigma\) | 每个方向乘以一个奇异值，表示这个方向有多重要 |
| \(U\) | 再把结果转到输出空间的一组左奇异方向上 |

如果少数奇异值很大，很多奇异值很小，说明矩阵主要依赖少数方向。这就是低秩近似、LoRA、谱分析、KV 压缩和某些优化器分析的共同底座。

![Muon SVD entropy](../assets/images/localized-remote/paper-deep-dives/foundations/muon-is-scalable/muon-svd-entropy-原图.png){ width="760" }

<small>图源：[Muon is Scalable](https://arxiv.org/abs/2502.16982) 相关图，本站已本地化于 Muon 论文精读。原图用 SVD entropy 比较矩阵谱能量分布。本站在这里用它说明：看矩阵时不只看大小，也要看能量是否集中在少数谱方向。</small>

## 谱能量：信息是不是集中在少数方向

如果奇异值是 \(\sigma_1,\sigma_2,\dots,\sigma_r\)，可以把每个方向的相对能量看成：

\[
p_i=\frac{\sigma_i^2}{\sum_j\sigma_j^2}
\]

这里 \(p_i\) 像一个概率分布：第 \(i\) 个谱方向占总能量多少。若 \(p_1,p_2\) 很大，说明能量集中；若很多 \(p_i\) 都差不多，说明信息分散。

KVSlimmer、低秩适配、剪枝和量化论文经常会问类似问题：Key、Value、权重矩阵或 activation 的信息，是不是集中在少数方向？如果是，压缩更可能安全；如果不是，粗暴压缩会丢掉细节。

## Newton-Schulz 迭代：用矩阵乘逼近正交化

Muon 这类优化器论文里会出现 Newton-Schulz iteration。它常用于近似矩阵的 polar factor 或 inverse square root，让一个矩阵更接近“列方向正交”的形状。

一个常见的 Newton-Schulz 风格更新可以写成：

\[
X_{k+1}
=
\frac{1}{2}X_k(3I-X_k^\top X_k)
\]

在合适归一化和谱条件下，如果 \(X_k^\top X_k\) 接近 \(I\)，这个迭代会把它进一步推向 \(I\)。直觉上，它不是在求某个标量最小值，而是在用一串矩阵乘法修正方向之间的相关性。

真正实现里，论文可能使用不同阶数、多项式系数或归一化策略。读 Muon 时不必先背每个系数，先抓住一点：Newton-Schulz 是一种便宜的矩阵正交化近似，适合放进 GPU 友好的优化器步骤里。

## Taylor 展开：用局部近似看扰动影响

一元函数在 \(x\) 附近的二阶 Taylor 展开是：

\[
f(x+\Delta x)
\approx
f(x)+f'(x)\Delta x+\frac{1}{2}f''(x)(\Delta x)^2
\]

这行式子说：扰动后的函数值，可以用当前值、一阶变化和二阶弯曲近似。

多元 loss 对参数扰动 \(\Delta\theta\) 的近似是：

\[
\mathcal{L}(\theta+\Delta\theta)
\approx
\mathcal{L}(\theta)
+g^\top\Delta\theta
+\frac{1}{2}\Delta\theta^\top H\Delta\theta
\]

其中 \(g=\nabla_\theta\mathcal{L}\) 是梯度，\(H=\nabla_\theta^2\mathcal{L}\) 是 Hessian。第一阶项告诉你往哪个方向会让 loss 变；第二阶项告诉你这个方向有多弯、多敏感。

## Hessian：二阶敏感度矩阵

Hessian 的元素是：

\[
H_{ij}
=
\frac{\partial^2\mathcal{L}}
{\partial\theta_i\partial\theta_j}
\]

如果 \(H_{ij}\) 很大，说明参数 \(\theta_i\) 和 \(\theta_j\) 的联合扰动会明显影响 loss。剪枝、量化、KV 合并和二阶优化里常用 Hessian 或近似 Hessian 判断“改这里会不会伤质量”。

后续例子：KVSlimmer 这类 KV cache 压缩论文会用 Hessian 近似来衡量合并 Key/Value 后输出误差的二阶影响。真正工程难点是：完整 Hessian 太贵，所以论文会尝试把二阶信息化简成 forward-pass 中可计算的量。

## Fisher 信息：概率模型里的二阶近似

Fisher 信息常写成：

\[
F
=
\mathbb{E}
\left[
\nabla_\theta \log p_\theta(x)
\nabla_\theta \log p_\theta(x)^\top
\right]
\]

它衡量模型分布对参数变化有多敏感。直觉上，如果稍微改参数，模型给数据的概率分布变化很大，这个方向的 Fisher 信息就大。

自然梯度、某些二阶优化、模型压缩和不确定性分析会借用 Fisher 或它的近似。读这类公式时要注意：期望来自哪个数据分布，log probability 是哪个模型输出，近似是否只在当前 batch 上成立。

## 低秩和二阶近似在论文里怎么用

| 论文主题 | 用到的通用数学 |
| --- | --- |
| Muon / 优化器 | 矩阵谱、Newton-Schulz 正交化、奇异值分布 |
| KVSlimmer / KV 压缩 | 谱同质性、Hessian 二阶误差 |
| LoRA / 低秩适配 | 用少数谱方向表达参数更新 |
| 剪枝 / NAS | Taylor 近似估计删除某结构的 loss 变化 |
| 量化 | 误差在敏感方向上会被放大 |

## 常见误读

| 误读 | 更稳的理解 |
| --- | --- |
| 奇异值小的方向一定没用 | 小方向可能承载长尾知识或安全边界 |
| Hessian 精确才有用 | 很多论文只需要可用的局部敏感度近似 |
| 低秩一定省且不伤质量 | 低秩省参数和带宽，但会改变表达空间 |
| 二阶方法一定比一阶方法好 | 二阶信息贵，近似质量和系统开销都要算 |

## 读完以后怎么判断

看到谱图、SVD、Hessian 或 Taylor 近似时，先问：它在分析哪个对象的方向结构，扰动是什么，误差衡量在哪里，近似是否便宜到能放进真实系统。这样读 Muon、KVSlimmer、剪枝、量化和低秩论文会稳得多。

## 相关阅读与下一步

- 站内下一步：[线性代数与几何直觉](linear-algebra-and-geometry.md)。
- 站内下一步：[优化、梯度与约束](optimization-and-gradient-methods.md)。
- 论文精读：[Muon：LLM 预训练优化器](../paper-deep-dives/foundations/muon-is-scalable.md)。
- 论文精读：[KVSlimmer：非对称 KV 合并](../paper-deep-dives/inference/kvslimmer.md)。
