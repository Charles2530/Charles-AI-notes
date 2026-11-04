# 卷积与多尺度视觉特征

一张图片中的相邻像素往往有关联，同一种局部形状又可能出现在不同位置。卷积把这两件事写进计算方式：只连接一个小窗口，并在全图重复使用同一组权重。

先看一个单通道 `3×3` 输入与 `2×2` kernel。这里按深度学习框架常用的互相关约定，不翻转 kernel。

\[
X=\begin{bmatrix}1&2&0\\0&1&3\\2&1&0\end{bmatrix},\qquad
K=\begin{bmatrix}1&0\\0&-1\end{bmatrix}.
\]

## 一个窗口产生一个数

左上窗口输出 \(1\times1+2\times0+0\times0+1\times(-1)=0\)。窗口右移一步，输出 \(2-3=-1\)；下一行两个输出分别为 \(0-1=-1\) 和 \(1-0=1\)。所以结果是

\[
Y=\begin{bmatrix}0&-1\\-1&1\end{bmatrix}.
\]

![卷积窗口与输出位置逐个对应](../assets/images/foundations/convolution-window-example.svg)

四个位置共享四个 kernel 参数。梯度反传时，这四次使用对同一个参数的贡献要相加；共享权重同时是减少参数和组合训练信号的机制。

```python
x = [[1,2,0], [0,1,3], [2,1,0]]
k = [[1,0], [0,-1]]
y = [[sum(x[i+u][j+v]*k[u][v] for u in range(2) for v in range(2))
      for j in range(2)] for i in range(2)]
print(y)
```

输出 `[[0,-1],[-1,1]]`。代码故意保留窗口循环，真实库会用适合硬件的 kernel 或矩阵化方式执行，不一定逐点照此循环。

## 增加通道以后，求和在哪里发生

RGB 图像有三个输入通道。一个输出通道会为每个输入通道准备一张 kernel，把三张局部加权结果相加，再加一次 bias。多个输出通道使用不同的权重，形成不同特征响应。

对 stride 为 1、无 padding 的普通卷积，

\[
y_{o,i,j}=b_o+\sum_c\sum_u\sum_vW_{o,c,u,v}x_{c,i+u,j+v}.
\]

o 是输出通道，c 是输入通道，i/j 是输出位置，u/v 是窗口内部位置。权重尺寸为 `[C_out,C_in,K_h,K_w]`，不含 bias 的参数量为 \(C_{out}C_{in}K_hK_w\)。例如 3 输入、64 输出、3×3 kernel，共 1728 个权重，无需随图片尺寸增长。

输出的 64 个通道不是 64 种颜色，也不能保证逐个对应“边缘”“杯柄”等人类概念。它们是训练目标塑造的特征方向。1×1 卷积不混合邻近位置，只在每个位置做通道线性变换；depthwise 卷积则把各输入通道的空间处理分开，通常再用 pointwise 层混合通道。

## 输出尺寸决定保留多少位置

在一个空间方向上，输入长度 H、kernel K、padding P、stride S、dilation D 对应输出长度

\[
H_{out}=\left\lfloor\frac{H+2P-D(K-1)-1}{S}\right\rfloor+1.
\]

本例 H=3、K=2、P=0、S=D=1，得到 2。若 stride 变成 2，只留下一个窗口输出，计算减少，但也不再保留每个位置的响应。Padding 通过补值保留边界输出，却引入人为边界条件；平移等变性会受到边界、步幅和采样网格影响。

对 `640×480` 图像，累计 stride 16 后约为 `40×30`，stride 32 后约为 `20×15`。更粗的网格可能仍足够分类，却难保留小字、细孔或接触边缘。不能只用参数量判断视觉表示对定位任务是否足够。

## 多层怎样看到更大范围

令 \(r_l\) 为第 l 层理论感受野，\(j_l\) 为相邻输出中心在原图上的间隔，初始均为 1。递推为

\[
r_l=r_{l-1}+(k_l-1)d_lj_{l-1},\qquad j_l=s_lj_{l-1}.
\]

连续两个 3×3、stride1 卷积，感受野从 1 变为 3 再变为 5。若第一层 stride2，第一层仍看 3 个像素，但相邻中心间隔变为 2；第二个 3×3 层就看到 7 个像素的范围。

理论感受野只表示存在计算路径。训练后哪些输入真正强烈影响输出，要看权重、激活和梯度，通常并不均匀。更大感受野也不能恢复早期采样已经丢掉的信息。

## U-Net 为什么把高分辨率特征接回来

分割、重建和去噪需要为多个空间位置输出结果。编码器降采样获得大范围语义，解码器升采样恢复输出网格。但单靠放大低分辨率特征，无法保证重建细边界。

U-Net 因此把编码器同尺度特征接到解码器。例如解码分支升采样得到 `[B,64,32,32]`，同尺度跳连也是 `[B,64,32,32]`，沿通道拼接后成为 `[B,128,32,32]`，再用卷积融合。拼接保留两组特征，与 residual 相加要求同形状并直接求和不同。

这种路径是信息复用，不保证所有细节都正确。跳连本身也受早期卷积和降采样影响。DiT 则常把潜空间 patch 组织成 token，用 Transformer 处理；不能把所有扩散网络都画成同一种 U-Net 下采样结构。

![U-Net 的下采样、上采样与跳连](../assets/images/paper-figures/foundations/unet-figure-1-architecture.png){ width="820" }

<small>图源：[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)，Figure 1。</small>

先看竖向的分辨率变化，再看横向 copy-and-crop 箭头。横向连接带回较细的空间信息，原始 U-Net 还需裁剪以配合有效卷积；现代 padding 版本可能采用不同尺寸规则。拼接会增加通道数，下一层卷积需接收这个新尺寸。

## 从这个例子读实现

先对齐输入布局、kernel轴和互相关约定，再检查 stride、padding、dilation、groups。一个 shape 相同的输出也可能采用不同边界规则，因此应同时用手算值比较。图像任务越依赖精确位置，预处理坐标和输出映射越不能省略。

- [D2L 卷积神经网络](https://github.com/d2l-ai/d2l-zh/tree/e6b18ccea71451a55fcd861d7b96fddf2587b09a/chapter_convolutional-neural-networks)从互相关窗口逐步展开多通道与池化。
- [U-Net](https://arxiv.org/abs/1505.04597)给出编码器与解码器之间的高分辨率特征连接。
- [卷积、频域与多尺度](../prerequisite-math/convolution-frequency-and-multiscale.md)进一步解释滤波、相位与降采样混叠。
