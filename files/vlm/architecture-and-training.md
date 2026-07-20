# VLM 架构：视觉证据怎样进入语言模型

图片、视频或多相机观测要先变成语言模型能使用的证据；到了 VLA 和世界模型里，这条链还要保留空间、时间和动作信息。

VLM 不是“给 LLM 加一张图”这么简单。真正的链路是：像素先被视觉 encoder 变成视觉特征，连接器把这些特征压成语言模型可消费的 token 或 memory，LLM 再用这些视觉证据生成文本、工具调用、动作 token 或中间推理。每一段都会丢掉一些信息；丢掉的如果只是纹理细节，问答可能没事；如果是位置、接触点、遮挡前状态或时间变化，机器人和世界模型就会出问题。

## 先分清 VLM 的三种任务

同一张桌面图片，三类问题需要的视觉证据不同：

| 任务 | 需要什么视觉证据 | 典型模型接口 |
| --- | --- | --- |
| 图文检索 / zero-shot 分类 | 物体类别、场景语义、文本描述相似度 | dual encoder，例如 CLIP |
| 图像问答 / 多模态对话 | 可被语言解释的对象、关系、文字、细节 | vision encoder + connector + LLM |
| VLA / 世界模型 | 可执行位置、接触、遮挡、动作后果、时间状态 | visual state + policy / dynamics |

问“这是什么杯子”时，语义类别和外观足够重要；问“机械臂该从哪里抓”时，杯沿、把手、桌边、深度、夹爪距离和可达性才是关键。VLM 架构的难点不是有没有视觉输入，而是哪些证据能穿过 encoder 和 connector，真正到达后续模型。

## 视觉 token 不是免费上下文

一张图像可以写成：

\[
I\in\mathbb{R}^{H_{\text{img}}\times W_{\text{img}}\times C}
\]

这里 \(H_{\text{img}}\) 和 \(W_{\text{img}}\) 是图像高宽，\(C\) 是通道数。若用 \(P\times P\) patch 切图，patch 数是：

\[
N=\frac{H_{\text{img}}}{P}\cdot\frac{W_{\text{img}}}{P}
\]

\(224\times224\) 图像配 \(16\times16\) patch 会得到 \(14\times14=196\) 个视觉 token。这个数字看起来不大，但视频和多相机会立刻放大：4 路相机、16 帧历史就是 \(4\times16\times196=12{,}544\) 个 patch token，还没算语言、动作、状态和 memory。

视觉 encoder 会把 patch token 变成视觉特征：

\[
h_v=f_v(e_1,\ldots,e_N)
\]

这里 \(e_i\) 是第 \(i\) 个 patch embedding，\(f_v\) 是视觉 encoder，\(h_v\) 是视觉特征序列。读 VLM 架构时，先问两个问题：视觉 token 数是多少，视觉特征保留的是语义、局部几何、文字 OCR、时间变化，还是动作相关状态。Token 多只说明输入细，不保证任务关键证据还在。

## CLIP 建立静态图文语义空间

CLIP 是理解 VLM 的好入口，因为它把问题化成图像和文本的对齐。

![CLIP pre-training and zero-shot transfer](../assets/images/paper-figures/contrastive-learning/clip-figure-1.png){ width="860" }

<small>图源：[Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)，Figure 1。原图表达图像 encoder 和文本 encoder 通过图文对比学习进入同一表示空间，并用文本 prompt 做 zero-shot 分类。这张图说明：CLIP 解决的是静态图文语义对齐，不等于动作、接触或时间后果建模。</small>

CLIP 对图像向量 \(v\) 和文本向量 \(t\) 常用归一化相似度：

\[
s(v,t)=\frac{\langle v,t\rangle}{\|v\|\|t\|}
\]

这里 \(\langle v,t\rangle\) 是点积，分母把向量长度归一化，所以比较的是方向是否接近。训练时，一个 batch 里的正确图文对要被拉近，错误配对要被推远。这个目标非常适合建立开放词表语义：图片像不像“a red mug”、像不像“a dog”、像不像“a stop sign”。

但 CLIP 的边界也正由这个目标决定。它不要求模型知道杯口边缘是否适合抓取，不要求知道左推会把杯子推下桌，也不要求在遮挡后持续记住物体位置。CLIP 是强语义底座，不是完整视觉行动系统。

## 连接器决定哪些视觉证据能进 LLM

对话型 VLM 需要把视觉特征接到语言模型。最直接的方法是 projector：把视觉特征映射到 LLM embedding 维度，再当作一段特殊 token 放进上下文。更强的方法会用 query、resampler 或 cross-attention 选择视觉证据。

BLIP-2 的 Q-Former 是典型连接器：冻结图像 encoder 和 LLM，中间训练少量 query token 去读视觉特征。

![BLIP-2 Q-Former objectives](../assets/images/paper-figures/vlm/blip2-figure-2-qformer-objectives.jpg){ width="920" }

<small>图源：[BLIP-2](https://arxiv.org/abs/2301.12597)，Figure 2。原图表达 Q-Former 用 learnable query 从冻结视觉 encoder 输出中抽取信息，并服务 image-text contrastive learning、image-grounded generation 和 image-text matching。这张图说明：连接器是信息瓶颈，query 学到什么取决于训练目标。</small>

可以把连接器抽象写成：

\[
c=g_\theta(h_v,q)
\]

这里 \(h_v\) 是视觉特征，\(q\) 是 learnable query、文本 token 或策略 token，\(g_\theta\) 是连接器，\(c\) 是交给语言模型的视觉 token。这个公式的关键不是 \(g_\theta\) 长什么样，而是它一定在做选择：有限数量的 \(c\) 不可能完整保留所有像素、局部几何和时间证据。

常见连接方式的取舍如下：

| 连接方式 | 它怎么读视觉信息 | 适合什么 | 容易丢什么 |
| --- | --- | --- | --- |
| Linear / MLP projector | 把视觉 patch 映射到 LLM embedding | 简单 VLM、LLaVA 式指令微调 | 细粒度选择能力弱 |
| Q-Former / resampler | 少量 query 主动抽取视觉证据 | 控制 token 数、冻结大模型对齐 | query 数不足会压掉局部证据 |
| Cross-attention | LLM token 按需读视觉 memory | 多图、多轮、few-shot 图文任务 | 成本高，缓存和长视频更复杂 |
| 统一多模态 token | 视觉、文本、动作放进同一序列 | 原生多模态/世界模型 | 上下文长度和训练成本高 |

评估连接器时，不能只看 VQA。要看它是否保留文字、位置、计数、相对关系、小物体、遮挡、接触和动作敏感信息。一个连接器可能让模型会说“红杯在桌上”，却已经丢掉“杯口离桌边只有 2 cm”。

## Flamingo 和 LLaVA 代表两种常见接法

Flamingo 的思路是把强视觉 encoder 和强语言模型桥接起来：Perceiver Resampler 把可变数量的图像/视频特征压成固定数量视觉 token，gated cross-attention 层让语言模型按需读取这些 token。这条路线强调 few-shot 和 interleaved image-text context，适合多图多轮提示。

LLaVA 代表更简单、工程上影响很大的路线：用 CLIP 视觉 encoder，加一个 projector 接到 LLM，再用图文指令数据做 visual instruction tuning。它证明了简单连接器配合高质量指令数据，可以快速得到好用的对话 VLM。但这条路线也容易被误读：对话能力强不等于视觉状态适合机器人控制。

两条路线的差别可以这样看：

| 路线 | 主要强项 | 主要风险 |
| --- | --- | --- |
| Flamingo-style cross-attention | LLM 可以在层内按需读取视觉 memory | 系统复杂、视觉 cache 和长视频成本高 |
| LLaVA-style projector | 实现简单，指令微调效果直接 | 视觉证据一次性压入 token，动作/几何证据可能不足 |

Hugging Face 的 VLM 教程也把 VLM 拆成 vision encoder、modality projector 和 text decoder / LLM 三段。这个拆法很实用：debug 时可以分别问视觉 encoder 看到了什么，projector 丢了什么，LLM 是否在按训练格式消费这些视觉 token。

## 训练目标决定模型学会哪种“看见”

VLM 不是只靠架构，训练目标同样决定信息瓶颈。

| 训练目标 | 模型学到什么 | 不保证什么 |
| --- | --- | --- |
| Contrastive alignment | 图像和文本语义相似度 | 细粒度定位、动作后果 |
| Image-text generation | 根据图像生成描述或回答 | 闭环行动、物理预测 |
| Image-text matching | 判断图文是否匹配 | 多步推理和可执行动作 |
| Visual instruction tuning | 按指令对话、解释、OCR、问答 | 机器人坐标和控制稳定性 |
| Video latent prediction | 对象持续性、运动和上下文 | 动作条件、reward、risk |

所以看到“用了 CLIP / BLIP-2 / LLaVA / Flamingo”时，不要只背模型名。要看训练数据和 loss：是图文配对、caption、问答、OCR、定位、视频预测，还是机器人动作数据。训练目标没有要求模型保留的证据，部署时很可能就不存在。

## 视频记忆把 VLM 推向状态建模

静态 VLM 主要回答“这张图里有什么”。VLA 和世界模型还要回答“刚才发生了什么”和“动作后会怎样”。这需要状态：

\[
z_t=f_\theta(o_{\le t})
\]

这里 \(o_{\le t}\) 是到当前为止的观测历史，\(z_t\) 是模型压出的当前状态。这个公式看起来简单，但关键是 \(z_t\) 是否真的保留对象持久性、运动、遮挡、接触和可操作关系。

V-JEPA 是理解视频状态的好例子：它不重建像素，而是在 latent space 里预测被 mask 的视频表示。

![V-JEPA training procedure](../assets/images/paper-deep-dives/world-models/v-jepa/figure-3-training-procedure.png){ width="920" }

<small>图源：[V-JEPA](https://arxiv.org/abs/2404.08471)，Figure 3。原图表达 context encoder 只处理 masked video 的可见 token，predictor 用 context output 和 mask tokens 预测 target encoder 对完整视频产生的 masked token representations。这张图说明：视频 VLM/世界模型需要学习时空状态，不只是静态图文语义。</small>

V-JEPA 的启发是：如果目标是运动理解和状态表示，不必把训练压力全放在像素重建上。预测 latent representation 可以让模型更关注对象、运动和上下文。但是它本身没有动作条件、奖励、风险和 planner 接口，因此更像世界模型的视觉状态 encoder 候选，不是完整 action-conditioned world model。

## 从 VLM 到 VLA，缺的是动作接口和后果模型

VLM 可以描述场景，VLA 要输出动作，世界模型还要预测动作后果。三者接口可以这样区分：

\[
\text{VLM:}\quad p_\theta(y\mid I,x)
\]

\[
\text{VLA:}\quad \pi_\theta(a_t\mid z_t,l)
\]

\[
\text{World model:}\quad p_\theta(z_{t+1}\mid z_t,a_t,l)
\]

第一行表示给定图像 \(I\) 和文本输入 \(x\)，生成回答 \(y\)。第二行表示策略根据视觉状态 \(z_t\) 和语言指令 \(l\) 输出动作 \(a_t\)。第三行表示世界模型预测执行动作后的下一状态。

这里最容易误读的是把 VLM 能力直接外推成 VLA 能力。能说“杯子在桌边”不等于知道从侧面推会掉下桌；能描述“抽屉半开”不等于知道拉手的 3D 位置和力控方向。VLA 还需要动作表示、坐标系、限幅、控制频率、失败恢复和闭环评测。

## 读 VLM 架构时先画四张账本

第一张是 token 账本：输入分辨率、patch size、帧数、相机数、resampler/query 数、最终进入 LLM 的视觉 token 数。没有这张账，就无法判断训练和推理成本。

第二张是证据账本：模型需要保留语义、文字、空间位置、局部几何、时间状态、动作后果中的哪些证据。不同任务不能共用同一个“视觉好坏”指标。

第三张是接口账本：视觉 encoder 是否冻结，connector 是 projector、Q-Former、resampler 还是 cross-attention，LLM 是否全参训练，视觉 token 放在 prompt 哪个位置，mask 是否允许正确读取。

第四张是评测账本：VQA、OCR、caption、定位、视频问答、动作成功率、world model rollout、闭环任务分别证明什么。不要用 VQA 分数证明机器人控制能力，也不要用 demo 图证明状态表示可靠。

VLM 架构的核心不是“图片接进 LLM”，而是视觉证据如何被选择、压缩和消费。CLIP 证明图文语义能对齐；BLIP-2 说明连接器是信息瓶颈；Flamingo 和 LLaVA 展示了 cross-attention 与 projector 两条常见路线；V-JEPA 提醒静态图文语义不够，视频状态还要学对象持续性和运动。

如果目标只是图文检索，静态语义空间可能足够；如果目标是多模态对话，connector 和指令数据很关键；如果目标是 VLA 或世界模型，就必须继续检查几何、接触、遮挡、时间记忆、动作条件和闭环评测。VLM 的成败，常常发生在“视觉证据穿过瓶颈时丢了什么”。

继续读相邻内容时，可以接 [视觉 Tokenizer 与连接器](visual-tokenizers-and-connectors.md)、[VLM/VLA 与世界模型高效训练接口](vlm-vla-for-efficient-world-model-training.md)、[VLA 动作表示与控制接口](../vla/action-representation-and-interfaces.md) 和 [Masked / JEPA 与潜变量预测](../world-models/masked-jepa-and-latent-prediction.md)。

## 外部材料

- [CLIP](https://arxiv.org/abs/2103.00020)：理解图文对比学习和 zero-shot 分类。
- [The Illustrated CLIP](https://jalammar.github.io/illustrated-clip/)：用图建立 CLIP 的相似度矩阵和 prompt classifier 直觉。
- [Hugging Face: Vision Language Models Explained](https://huggingface.co/blog/vlms)：用 vision encoder、projector、LLM 的三段结构理解 VLM。
- [BLIP-2](https://arxiv.org/abs/2301.12597)：理解 Q-Former 为什么是连接器瓶颈。
- [Flamingo](https://arxiv.org/abs/2204.14198)：理解 Perceiver Resampler 和 gated cross-attention。
- [LLaVA](https://llava-vl.github.io/)：理解 projector + visual instruction tuning 的简单强路线。
- [V-JEPA](https://arxiv.org/abs/2404.08471)：理解 latent video prediction 如何补静态 VLM 的时间状态短板。
- [OpenBMB BLIP-2 论文速读](https://hub.baai.ac.cn/view/25807)：中文公众号式读法示例，适合对照“问题、方法、消融、局限”结构复盘 BLIP-2。
- [AI Insight 多模态统一架构综述](https://www.ai-insight.org/reports/custom_multimodal-unified-architecture-2026)：中文报告式梳理，适合查 ViT-Bridge-LLM 范式、InternVL/Qwen-VL 等后续路线。
