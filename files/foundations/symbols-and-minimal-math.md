# 符号与最小数学地图

这页是基础知识模块的随手查表。第一次读 AI 教程时，真正卡人的往往不是公式本身，而是符号没有说明：\(x\) 是输入还是图像，\(W\) 是权重还是宽度，\(H\) 是图像高度还是预测 horizon。

!!! note "初学者先抓住"
    看到公式先不要急着推导。按四步读：谁是输入，谁是输出，条件是什么，模型在优化什么。只要这四件事能说清，大多数公式都会从“看不懂的数学”变成“数据流的压缩写法”。

## 1. Shape 字母：先知道张量有多大

Shape 是模型模块之间的接口契约。本站会尽量沿用下面的写法。

| 符号 | 读法 | 常见含义 | 小例子 |
| --- | --- | --- | --- |
| \(B\) | batch size | 一次处理多少样本 | `B=8` 表示 8 条样本一起算 |
| \(L\) | sequence length | token 数或序列长度 | 文本长度、视觉 token 数 |
| \(D\) | hidden dimension | 每个 token 的向量维度 | `D=4096` |
| \(C\) | channel | 图像通道或特征通道 | RGB 图像里 `C=3` |
| \(H_{\text{img}}\) | image height | 图像高度 | 224 像素 |
| \(W_{\text{img}}\) | image width | 图像宽度 | 224 像素 |
| \(P\) | patch size | 图像 patch 边长 | ViT 常见 `P=16` |
| \(N\) | token 或样本数量 | 具体含义看上下文 | \(N\) 个 patch、\(N\) 条数据 |
| \(T\) | time length | 时间步或总序列长度 | 16 帧视频、64 步轨迹 |

一个文本 batch 常写成：

\[
x \in \mathbb{R}^{B \times L \times D}
\]

读作：\(x\) 是一个三维张量，有 \(B\) 条样本，每条样本有 \(L\) 个 token，每个 token 是 \(D\) 维向量。

!!! warning "同一个字母在不同页面可能换意思"
    \(H\) 在图像里常是 height，在世界模型里常是 horizon，在 attention 里也可能是 head 数。教程会尽量写成 \(H_{\text{img}}\)、\(H_{\text{hor}}\)、\(H_{\text{heads}}\) 来避免混淆。

## 2. 模型和参数：谁在被训练

| 符号 | 读法 | 含义 |
| --- | --- | --- |
| \(f_\theta\) | f theta | 参数为 \(\theta\) 的模型函数 |
| \(\theta\) | theta | 模型参数的集合 |
| \(W\) | weight matrix | 权重矩阵 |
| \(b\) | bias | 偏置 |
| \(h\) | hidden state | 中间隐藏表示 |
| \(z\) | latent state | 模型内部压缩状态 |
| \(\hat y\) | y hat | 模型预测值 |
| \(y\) | y | 真实标签或目标输出 |

最常见的线性层公式是：

\[
y = xW + b
\]

逐项读：

| 部分 | 含义 |
| --- | --- |
| \(x\) | 输入向量或输入张量 |
| \(W\) | 可训练权重，决定怎么重新组合输入特征 |
| \(b\) | 偏置，让输出可以整体平移 |
| \(y\) | 输出表示 |

如果写成 \(\hat y=f_\theta(x)\)，意思就是“模型用参数 \(\theta\) 处理输入 \(x\)，得到预测 \(\hat y\)”。

## 3. 概率符号：模型在给可能性打分

| 符号 | 读法 | 含义 |
| --- | --- | --- |
| \(p(x)\) | probability of x | \(x\) 出现的概率或密度 |
| \(p_\theta(x)\) | model distribution | 参数为 \(\theta\) 的模型分布 |
| \(p(y\mid x)\) | y given x | 给定 \(x\) 时 \(y\) 的条件概率 |
| \(q_\phi(z\mid x)\) | q phi | 参数为 \(\phi\) 的近似后验或推断模型 |
| \(\log p(x)\) | log probability | 对数概率，训练里更容易相加 |
| \(\mathcal{N}(\mu,\Sigma)\) | Gaussian | 均值为 \(\mu\)、协方差为 \(\Sigma\) 的高斯分布 |

条件概率里的竖线 \(\mid\) 读作“given”。例如：

\[
p_\theta(y \mid x)
\]

读作：给定输入 \(x\)，模型输出 \(y\) 的概率。它不保证 \(y\) 一定正确，只表示模型认为哪个输出更可能。

在 VLA 或世界模型里常见：

\[
p_\theta(z_{t+1}\mid z_t,a_t)
\]

读作：给定当前 latent state \(z_t\) 和动作 \(a_t\)，模型预测下一步 latent state \(z_{t+1}\) 的分布。

## 4. Loss、梯度和优化：模型怎么变好

| 符号 | 读法 | 含义 |
| --- | --- | --- |
| \(\mathcal{L}\) | loss | 损失函数，衡量预测和目标的差距 |
| \(\ell(\hat y,y)\) | per-example loss | 单个样本的损失 |
| \(\nabla_\theta \mathcal{L}\) | gradient w.r.t. theta | loss 对参数 \(\theta\) 的梯度 |
| \(\eta\) | learning rate | 学习率，每次更新走多大一步 |
| \(\leftarrow\) | update to | 更新赋值 |

最小训练更新可以写成：

\[
\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}
\]

逐项读：

| 部分 | 含义 |
| --- | --- |
| \(\theta\) | 当前模型参数 |
| \(\nabla_\theta \mathcal{L}\) | 哪个方向会让 loss 变大 |
| \(-\nabla_\theta \mathcal{L}\) | 往相反方向走，通常能让 loss 变小 |
| \(\eta\) | 步长，太大可能发散，太小会很慢 |

代码里它大致对应：

```text
loss.backward()
optimizer.step()
```

## 5. 求和、期望和范数：把很多样本合起来

| 符号 | 读法 | 含义 |
| --- | --- | --- |
| \(\sum_i\) | sum over i | 对很多项求和 |
| \(\prod_i\) | product over i | 对很多项连乘 |
| \(\mathbb{E}[\cdot]\) | expectation | 对随机样本取平均意义上的值 |
| \(\|x\|_2\) | L2 norm | 向量长度 |
| \(\|x-y\|_2^2\) | squared L2 distance | MSE 常见形式 |
| \(\operatorname{KL}(q\|p)\) | KL divergence | 两个分布差多少 |

例如行为克隆损失：

\[
\mathcal{L}_{\text{BC}}
=-\sum_{(o,l,a)\in\mathcal{D}}\log \pi_\theta(a\mid o,l)
\]

可以拆成一句话：训练集 \(\mathcal{D}\) 里每条样本都有观测 \(o\)、语言 \(l\)、专家动作 \(a\)；我们让策略 \(\pi_\theta\) 更倾向于在同样输入下输出专家动作。

## 6. 时间下标：模型在第几步看见什么

| 符号 | 含义 |
| --- | --- |
| \(t\) | 当前时间步 |
| \(t+1\) | 下一步 |
| \(o_t\) | 第 \(t\) 步观测 |
| \(a_t\) | 第 \(t\) 步动作 |
| \(o_{\le t}\) | 从过去到当前的所有观测 |
| \(a_{t:t+H-1}\) | 从 \(t\) 开始的一段动作序列 |
| \(H_{\text{hor}}\) | 预测 horizon，往未来看多少步 |

例如：

\[
\pi_\theta(a_t\mid o_{\le t}, l)
\]

读作：策略根据到目前为止的观测 \(o_{\le t}\) 和语言指令 \(l\)，输出当前动作 \(a_t\) 的概率或动作值。

## 7. 复杂度符号：为什么有的模型跑不动

| 写法 | 直觉 |
| --- | --- |
| \(O(L)\) | 序列长度翻倍，成本大约翻倍 |
| \(O(L^2)\) | 序列长度翻倍，成本大约变四倍 |
| \(O(BLD)\) | batch、长度、维度都会线性增加成本 |
| \(B\times L\times D\times bytes\) | 一个 hidden tensor 的大致显存 |

Attention 最常见的成本提醒是：

\[
\text{score matrix size} \propto L^2
\]

也就是说，token 数 \(L\) 一多，注意力分数矩阵会平方放大。多相机视频、长上下文和多轮对话都容易在这里变贵。

## 8. 读公式的固定四问

以后在基础页或论文页看到公式，可以按下面四问走：

1. 输入是什么：\(x\)、\(o_t\)、\(z_t\)、\(a_t\) 分别来自哪里？
2. 输出是什么：预测 token、动作、latent、reward，还是概率？
3. 条件是什么：竖线 \(\mid\) 后面的变量是不是都在模型可见范围内？
4. 优化什么：loss 是让输出更像标签，还是让分布、风险、奖励或成本更好？

如果这四问答不清，先回到本页查符号，再继续读具体模型。
