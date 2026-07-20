# 条件控制与 Guidance

扩散模型的条件控制，常被一句“调 prompt 和 CFG scale”带过。真正做系统时，这句话太粗：prompt、边缘图、姿态、mask、参考图和首帧，首先要被模型在每一步去噪时读到；而 guidance 只是在采样时，把“有条件”和“基准方向”的差额放大。前者是条件怎么进入模型，后者是采样轨迹怎么被拉扯。

把这两件事分开以后，很多现象会变得清楚：为什么 CFG scale 太大容易过饱和，为什么 negative prompt 不是万能禁止词，为什么 ControlNet 不能被理解成“更强 prompt”，也为什么视频生成里单帧更贴词并不等于整段更稳定。

## 先分清：给模型看条件，和放大条件方向

无条件扩散可以把去噪网络写成：

\[
\epsilon_\theta(x_t,t)
\]

这里 \(x_t\) 是当前噪声水平下的带噪样本，\(t\) 告诉模型现在噪声大概有多重。模型只靠这两件事预测应当去掉的噪声。条件扩散则多看一个 \(c\)：

\[
\epsilon_\theta(x_t,t,c)
\]

\(c\) 可以是文本、类别、深度图、边缘图、姿态图、mask、参考图、视频首帧或动作历史。这个写法的重点不是“最后按条件筛图”，而是每一个去噪步骤都在问：在当前噪声水平下，如果我要满足这个条件，哪些变化更像信号，哪些变化更像噪声？

所以 conditioning 是证据入口。文本常通过 text encoder 和 cross-attention 进入 U-Net / DiT；类别可以变成 embedding 或归一化里的调制信号；深度、边缘、姿态这类空间条件需要保住每个位置的结构信息；参考图通常要经过 image encoder 或额外 adapter，把身份、风格或局部外观传进模型。条件入口弱，guidance 再高也只是放大一个不可靠方向；条件之间冲突，guidance 会把冲突推得更明显。

## Classifier Guidance：外部分类器给采样轨迹加坡度

早期 classifier guidance 的想法很直接：训练一个能看带噪图的分类器 \(p_\phi(c\mid x_t)\)，采样时让当前样本朝“更像类别 \(c\)”的方向移动：

\[
\nabla_{x_t}\log p_\phi(c\mid x_t)
\]

这不是让分类器生成图片，而是借它提供一个梯度。当前样本稍微往哪个方向挪，分类器会更确信它属于目标类别，采样器就把这个方向加进反向扩散过程。[Diffusion Models Beat GANs](https://arxiv.org/abs/2105.05233) 展示了这种做法能显著提升类别条件图像质量，也让“guidance 改变采样方向场”这个概念变得很具体。

它的问题同样具体：需要额外训练噪声鲁棒分类器，而且分类器必须在每个噪声水平下都靠谱。文本、边缘图、参考图、mask 这类条件很难都配一个外部分类器，因此后来更常用的是 classifier-free guidance。

## CFG：把有条件和无条件预测之间的差额放大

[Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) 的关键训练技巧是 condition dropout：训练时有时给模型条件 \(c\)，有时把条件丢掉，让同一个网络同时学会“看条件去噪”和“不看条件去噪”。推理时，模型在同一步会给出两种预测：

\[
\epsilon_\theta(x_t,t,c)
\quad\text{和}\quad
\epsilon_\theta(x_t,t,\varnothing)
\]

第一项是看条件后的方向，第二项是空条件下的自然生成方向。CFG 采样把它们组合成：

\[
\hat{\epsilon}_{\text{cfg}}
=
\epsilon_\theta(x_t,t,\varnothing)
+
s\left[
\epsilon_\theta(x_t,t,c)
-
\epsilon_\theta(x_t,t,\varnothing)
\right]
\]

这行公式可以直接按采样动作读：先站在空条件预测给出的自然方向上，再看条件预测相对空条件多出了哪一段方向差，最后把这段差乘以 \(s\) 加回去。\(s=1\) 接近普通条件预测；\(s>1\) 是额外强调条件；\(s\) 越大，采样器看到的方向场越偏向条件差额。

![Classifier-Free Guidance guidance scale](../assets/images/paper-figures/diffusion/cfg-figure-1-guidance-scale.png){ width="920" }

<small>图源：[Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)，Figure 1。原图展示 ImageNet 类别条件下 guidance weight 增大时，样本更贴近目标类别。读图重点：guidance 的确会强化条件方向，但这张图不能推出“scale 越大越好”；它同时提醒我们，多样性、自然度和稳定性也在被采样轨迹重新分配。</small>

## 为什么 CFG scale 不是质量旋钮

很多界面把 CFG scale 做成滑块，于是用户很容易把它当成“质量”或“听话程度”的单轴控制。更准确地说，它是在 prompt adherence、分布自然度和多样性之间重新分配概率质量。

以 `rainy cyberpunk convenience store at night` 为例，scale 太低时，模型可能只生成普通夜景，因为文本条件带来的方向差没有被充分强调；scale 适中时，霓虹、雨夜反光、便利店主体和赛博朋克材质会更明确；scale 太高时，颜色可能过饱和，边缘变硬，对象关系开始僵化，甚至出现局部纹理崩坏。这里坏掉的不是 prompt，而是采样器每一步都沿着被放大的条件差额走，轨迹变陡以后更容易离开模型训练中最自然的区域。

少步采样会放大这个问题。50 步采样时，错误方向还有很多次微调机会；4 到 10 步采样时，每一步跨得更远，高 guidance 更容易过冲。Imagen 论文里讨论了高 guidance 下的阈值处理，DPM-Solver++ 也专门分析了 guided sampling 下高阶 solver 的稳定性。工程上不能只报告“用了什么 sampler”，还要同时报告步数、参数化、CFG scale、是否做 thresholding，以及 prompt 难度。

## Negative prompt 改的是基准方向，不是事实过滤器

在很多 Stable Diffusion 风格的界面里，negative prompt 会被放到 CFG 的基准分支里。没有 negative prompt 时，基准通常接近空条件；有 negative prompt 时，基准会更像“朝负面条件生成”的方向。可以把它写成：

\[
\hat{\epsilon}_{\text{neg}}
=
\epsilon_\theta(x_t,t,c_{\text{neg}})
+
s\left[
\epsilon_\theta(x_t,t,c_{\text{pos}})
-
\epsilon_\theta(x_t,t,c_{\text{neg}})
\right]
\]

这行公式的读法是：先用负面条件给出一个基准方向，再放大正面条件相对负面条件的差。于是 negative prompt 更像“从这个方向拉开”，不是一个可以绝对禁止内容的规则系统。写 `low quality, blurry, extra fingers` 可能会减少一些常见坏模式，但如果负面词太宽、太强，或者和正面 prompt 共享语义，它也会把本来需要的纹理、风格和对象细节一起拉走。

这也解释了为什么同一组 prompt 换 sampler、scale 或 seed 后，negative prompt 的效果会变。它参与的是每一步采样方向，不是在成图后做语义审核。

## ControlNet、Adapter 和参考图：结构条件不是更长的文本

文本条件擅长描述语义和风格，但它很难稳定表达“这个边缘必须在这里”“这只手的骨架必须按这个姿态”“相机深度关系不能变”。[ControlNet](https://arxiv.org/abs/2302.05543) 的价值在于给预训练扩散模型加一个可训练控制分支，把边缘、深度、姿态、分割等空间条件送进中间层，让每个位置都能看到结构约束。[T2I-Adapter](https://arxiv.org/abs/2302.08453) 走的是更轻量的 adapter 思路，同样是在承认结构条件和文本条件不是同一种信号。

参考图又是另一类条件。[IP-Adapter](https://arxiv.org/abs/2308.06721) 把图像 prompt 接到文本到图像模型中，让模型可以利用参考图里的身份、风格或外观信息。它和 ControlNet 的目标不同：ControlNet 更像固定几何和布局，IP-Adapter 更像提供视觉身份和风格证据。把它们都叫“conditioning”是可以的，但不能把它们都调成同一个 prompt 权重问题。

多条件生成真正难的是优先级和作用范围。文本可能想改变风格，深度图要求保几何，参考图要求保身份，mask 只允许局部区域变化。强 guidance 会放大最终的条件差额，却不会自动解决“谁覆盖谁”的冲突。好的系统要明确每个条件进哪一层、影响哪一块区域、和 CFG scale 或 control strength 怎样一起扫参。

## 视频和世界模型里，过强 guidance 会伤到时间

单张图里，高 CFG 造成的坏处通常表现为过饱和、边缘发硬、构图僵硬或多样性下降。视频里，同样的过冲会沿时间轴积累：前几帧看起来更贴 prompt，后面却可能出现身份漂移、纹理闪烁、动作忽快忽慢，或者首帧条件逐渐丢失。

世界模型和具身场景更敏感。动作条件下的未来预测，目标不是“生成看起来像真实视频的一段画面”，而是让动作、状态变化和物理后果保持对应。如果 guidance 只强化视觉语义，却削弱 action sensitivity，模型会生成漂亮但不能用于规划的未来。这类系统评测要同时看条件遵循、时间一致性、动作改变后的反应、长时 rollout 和服务延迟，而不能只看单帧审美或 text alignment。

## 读论文和调系统时该看什么

读 guidance 相关论文，可以先问三个问题。第一，条件是怎么进入模型的：文本 cross-attention、类别 embedding、ControlNet 分支、adapter、参考图 encoder、mask，还是视频历史？第二，采样时有没有改变方向场：classifier guidance、CFG、negative prompt、guidance schedule，还是把 guidance 蒸馏进学生模型？第三，评测有没有拆开条件强度、scale、sampler、步数和失败模式，而不是只给一组最好看的样例图。

真实调参也应该这样拆。先固定 sampler 和步数扫 CFG scale，看 prompt adherence、自然度和多样性的拐点；再固定 scale 换 sampler，看少步下是否过冲；最后按条件类型分桶评估，比如纯文本、ControlNet、参考图、局部编辑、视频首帧和动作条件。只有这样，guidance 才不会被误用成一个玄学旋钮。

## 外部材料

- [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233)：理解 classifier guidance 如何用外部分类器梯度改变采样方向。
- [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)：CFG 原论文，重点看 condition dropout、公式和 guidance weight 对样本的影响。
- [GLIDE](https://arxiv.org/abs/2112.10741)：理解文本到图像系统中 guidance、文本条件和编辑能力如何结合。
- [Imagen](https://arxiv.org/abs/2205.11487)：看大规模文生图系统如何处理文本条件、高 guidance 和 thresholding。
- [ControlNet](https://arxiv.org/abs/2302.05543)：理解结构条件为什么需要额外控制分支，而不是只靠 prompt。
- [T2I-Adapter](https://arxiv.org/abs/2302.08453) 与 [IP-Adapter](https://arxiv.org/abs/2308.06721)：分别理解轻量结构控制和参考图条件接口。
- [DPM-Solver++](https://arxiv.org/abs/2211.01095)：理解 guided diffusion 下少步 ODE solver 为什么需要专门处理稳定性。
- [Lil'Log: What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)：补 DDPM、score、classifier guidance 和 CFG 的连续直觉。

- [Lil'Log：What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)。
- [DDPM 论文](https://arxiv.org/abs/2006.11239)。
- [Score SDE 论文](https://arxiv.org/abs/2011.13456)。
