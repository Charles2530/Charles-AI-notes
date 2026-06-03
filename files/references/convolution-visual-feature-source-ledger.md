# 卷积与视觉特征来源台账

这页记录卷积、CNN、感受野、多尺度视觉特征、U-Net 和 dense prediction 相关来源。它服务的写作原则是：不要把卷积写成“滑动窗口”就停住，而要说明局部连接、权重共享、通道混合、shape 变化和多尺度融合分别解决什么问题。

## 使用方式

写卷积文章时，先把来源分成四类。

第一类是基础机制来源。CS231n、PyTorch Conv2d、Dumoulin/Visin convolution arithmetic 和动手学深度学习帮助本站把 kernel、stride、padding、dilation、channel、groups 和输出 shape 讲清楚。正文里公式要落到变量槽位和任务后果：不是只写 \(H_{out}\)，还要说明 stride 如何减少 token、显存和定位粒度。

第二类是特征解释来源。Zeiler & Fergus、Distill feature visualization、CNN Explainer 和相关可视化材料帮助本站解释浅层/中层/深层特征不是口号。正文应说明 channel 是检测器集合或表征方向，不是颜色数量；同时也要提醒有效感受野和可解释性有边界。

第三类是多尺度和 dense prediction 来源。U-Net、FPN、DeepLab、SegNet 和扩散 U-Net 相关资料帮助本站解释为什么分类和 dense prediction 对空间信息的要求不同。U-Net 的 skip connection 不应只被写成梯度通路，它在分割、去噪、深度和控制里更重要的是恢复定位细节。

第四类是机器人和多模态来源。VLA、深度估计、stereo cost volume、VGGT、Depth Anything 和世界模型材料帮助本站说明：视觉模型不能只保语义 token。抓取、插入、UI 操作、OCR 和闭环控制需要边缘、接触线、深度跳变和小物体；早期强下采样会把这些信息丢掉。

## 写作边界

卷积不是 Transformer 的反面。很多现代模型使用 ViT 主干，但仍在 patch embedding、局部窗口、depthwise convolution、多尺度 decoder、U-Net、FPN、cost volume 和低层视觉接口里使用卷积思想。本站写作时不做“CNN 已过时 / ViT 全面替代”这种二分，而是说明任务需要哪些局部与多尺度先验。

中文材料和交互教程适合学习讲法，尤其是 shape、窗口滑动和特征图可视化；涉及论文 claim、实验指标和图源时，正文仍回原论文、官方文档或可复查项目页核对。

## 来源

| 来源 | 类型 | 本站学什么 | 落到页面 |
| --- | --- | --- | --- |
| [CS231n: Convolutional Neural Networks](https://cs231n.github.io/convolutional-networks/) | 课程讲义 | local connectivity、parameter sharing、activation volume 和 stride/padding 的经典讲法。 | 卷积基础 |
| [PyTorch Conv2d](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) | 官方文档 | Conv2d 的 cross-correlation 形式、groups、dilation 和输出 shape。 | 公式与 API |
| [A guide to convolution arithmetic for deep learning](https://arxiv.org/abs/1603.07285) | 技术报告 | 用统一图示解释 padding、stride、dilation、transposed convolution。 | shape 规则 |
| [Deep Learning book, Chapter 9](https://www.deeplearningbook.org/contents/convnets.html) | 教材 | 卷积网络的 sparse interactions、parameter sharing 和 equivariance。 | 结构先验 |
| [Visualizing and Understanding Convolutional Networks](https://arxiv.org/abs/1311.2901) | 论文 | 通过 deconv/可视化理解不同层级特征。 | 特征层级 |
| [Distill: Feature Visualization](https://distill.pub/2017/feature-visualization/) | 可视化博客 | feature/channel/neuron 可视化如何帮助理解深层表示及其边界。 | 特征解释 |
| [CNN Explainer](https://poloclub.github.io/cnn-explainer/) | 交互教程 | 用可交互方式展示卷积、ReLU、pooling 和分类路径。 | 入门图解 |
| [U-Net](https://arxiv.org/abs/1505.04597) | 论文 | contracting path、expanding path 和 skip connection 如何服务 dense segmentation。 | U-Net、多尺度 |
| [Feature Pyramid Networks](https://arxiv.org/abs/1612.03144) | 论文 | top-down pathway 与 lateral connection 如何融合语义和定位。 | 多尺度视觉 |
| [DeepLabv3+](https://arxiv.org/abs/1802.02611) | 论文 | atrous convolution、encoder-decoder 和多尺度上下文。 | dilation、分割 |
| [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) | 论文 | 扩散模型里的 U-Net denoiser 如何把卷积多尺度结构用于生成。 | 扩散 U-Net |
| [动手学深度学习：卷积神经网络](https://zh.d2l.ai/chapter_convolutional-neural-networks/index.html) | 中文教程 | 用中文、代码和张量 shape 讲清卷积、padding、stride、channel。 | 中文讲法 |
| [OneFlow 官方博客](https://blog.csdn.net/OneFlow_Official) | 中文工程博客 | 学习从 kernel、算子、显存和工程实现角度解释视觉基础模块。 | 中文工程讲法 |
