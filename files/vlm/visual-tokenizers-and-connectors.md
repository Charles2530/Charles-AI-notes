# 视觉 Tokenizer 与连接器：视觉证据怎样被压成可用 token

VLM/VLA 的第一道瓶颈是视觉证据怎样进入后续模型。图片、视频、多相机、深度、点云和机器人 proprioception 都不能原样无限塞进上下文；它们必须先被切成 token、压成 latent，或者通过 query / cross-attention 被抽取成少量证据。

视觉 tokenizer 和 connector 会压缩视觉输入，也会丢掉部分信息；关键是判断它们压缩什么、丢掉什么，以及这些丢失为什么会影响问答、动作和世界模型 rollout。

## 视觉 token 先来自空间切块

ViT 把图像当成一串 patch token。假设输入图像大小是 \(H\times W\)，patch size 是 \(P\)，一帧图像会变成：

\[
N_{\text{patch}}=\frac{H}{P}\times\frac{W}{P}
\]

这里 \(N_{\text{patch}}\) 表示视觉 encoder 看到的 patch 数。若 \(224\times224\) 图像用 \(16\times16\) patch，就得到 \(14\times14=196\) 个 patch token；若分辨率升到 \(448\times448\)，同样 patch size 下就变成 \(28\times28=784\) 个 token。

这就是视觉 tokenizer 的第一层含义：它不是“把图像变成一个向量”，而是把图像变成一串有空间位置的局部证据。patch 越细，小字、边缘、按钮、夹爪和接触点越可能保留；token 越多，后续 attention、KV cache、激活和通信成本越高。

文本 token 和视觉 token 的差别在这里很关键。一个句子少几个 token，语义可能仍然靠上下文补回来；视觉少几个局部 token，可能就是杯沿、按钮边界或遮挡前位置不见了。

## 多相机视频会把 token 数迅速放大

一帧还好，多帧和多相机会把视觉长度乘起来。若 \(V\) 路相机、\(T\) 帧历史，每帧 \(N\) 个 patch token，原始视觉长度是：

\[
L_{\text{vision}}=V\times T\times N
\]

这行式子表示：相机数、历史帧数和每帧 patch 数相乘，才是进入连接器前的视觉 token 量。一个桌面机器人若使用 4 路相机、16 帧历史，每帧 \(14\times14=196\) 个 patch，就有：

\[
4\times16\times196=12544
\]

也就是说，仅视觉侧就有 12,544 个 token。还没算语言指令、动作 token、proprioception、memory token 和未来预测 token。

如果 connector 把每个相机每帧压成 \(K\) 个任务 token，压缩后长度变成：

\[
L_{\text{compressed}}=V\times T\times K
\]

当 \(K=32\) 时：

\[
4\times16\times32=2048
\]

视觉长度约下降 \(12544/2048\approx6.1\) 倍。如果后续模块对视觉 token 做全局 self-attention，attention score 的理论上界会从 \(12544^2\approx157M\) 个元素降到 \(2048^2\approx4.2M\)，大约 37 倍。FlashAttention 不会把 score 全量写回内存，但计算压力和上下文预算仍然受 \(L\) 或 \(L^2\) 影响。

所以 connector 不是装饰层。它决定一个多相机长视频系统是能训练，还是只停留在 demo。

## 连接器不是格式转换，而是选择证据

现代 VLM 通常有三段：

\[
I \xrightarrow{\text{vision encoder}} H_v
\xrightarrow{\text{connector}} Z_v
\xrightarrow{\text{LLM / policy / world model}} y,a,\hat s
\]

这里 \(I\) 是图像或视频，\(H_v\) 是视觉 encoder 输出的大量 patch / frame features，\(Z_v\) 是 connector 交给后续模型的视觉证据，\(y\) 是语言回答，\(a\) 是动作，\(\hat s\) 是预测状态。公式想表达的不是某个固定架构，而是一条信息链：后续模型从来没有直接读原图，它读的是已经被筛选后的 \(Z_v\)。

最小 connector 可以写成：

\[
Z_v=g_\theta(H_v,Q)
\]

这里 \(g_\theta\) 是 connector，\(Q\) 可以是可学习 query、语言 token、policy token 或 resampler latent。若 \(Q\) 很少，token 成本低，但信息选择压力大；若 \(Q\) 很多，细节更容易保留，但长上下文成本会上升。

这就是为什么同一个视觉 encoder，换不同 connector 后表现会变。模型不是“看见了整张图再思考”，而是“看见 connector 允许它看见的证据再思考”。

## BLIP-2：Q-Former 把视觉证据压成 query token

BLIP-2 是连接器瓶颈的好例子。它冻结强视觉 encoder 和强语言模型，中间训练 Q-Former，让少量 learnable query 从视觉特征里读信息。

![BLIP-2 Q-Former objectives](../assets/images/paper-figures/vlm/blip2-figure-2-qformer-objectives.jpg){ width="900" }

<small>图源：[BLIP-2](https://arxiv.org/abs/2301.12597)，Figure 2。原图表达 Q-Former 用 learnable query 从冻结视觉 encoder 输出中抽取信息，并服务 image-text contrastive learning、image-grounded generation 和 image-text matching。这里的重点是：连接器是信息瓶颈，query 学到什么取决于训练目标。</small>

读这张图时先看中间的 query tokens。它们不是普通文本 token，而是一组可训练的“视觉问题”：从海量 patch features 里问出哪些信息值得交给语言模型。训练目标如果是图文对比和 caption，query 会倾向保留类别、属性、关系和可描述细节；训练目标如果包含定位、OCR、动作或未来预测，query 才更可能保留局部几何、文字区域、接触状态和时间变化。

这解释了一个常见现象：VQA 很好的 connector，不一定适合 VLA。它可能准确知道“桌上有红色杯子”，却没有保留杯口边缘、夹爪距离、桌边风险和遮挡前位置。

## 三条常见接法，各自丢不同证据

| 接法 | 机制 | 优点 | 主要风险 |
| --- | --- | --- | --- |
| Linear / MLP projector | 把视觉 patch 映射到 LLM embedding | 简单、稳定、LLaVA 式路线容易落地 | token 多，局部选择能力弱，依赖指令数据弥补 |
| Q-Former / resampler | 少量 query 主动读视觉特征 | 控制 token 数，适合冻结大模型桥接 | query 数或训练目标不够时会压掉小结构 |
| Cross-attention / visual memory | 语言或策略 token 按需读视觉 memory | 细节可以在层内被访问，适合多图/视频上下文 | 系统复杂，视觉 cache、调度和显存压力高 |

Flamingo 代表 cross-attention / resampler 路线：Perceiver Resampler 把可变数量的图像或视频特征压成固定数量视觉 token，再让语言模型通过 gated cross-attention 读取。LLaVA 代表简单 projector 路线：CLIP vision encoder 加 projector，再用视觉指令数据把视觉 token 接到 LLM。两条路线都能做 VLM，但失败模式不同：前者工程复杂，后者更容易把视觉证据一次性压死。

Perceiver 和 TokenLearner 这类工作提供了更一般的直觉：面对高维视觉输入，模型可以用少量 latent / learned tokens 主动抽取信息，而不是把所有 token 逐层保留。压缩不等于平均池化，好的压缩应该让 token 对任务有选择性。

## 为什么“省 token”不等于“保留状态”

把 12,544 个视觉 token 压成 2,048 个，成本显著下降。但问题是：剩下的 token 到底是什么？

| 压缩策略 | token 省在哪里 | 容易保留 | 容易丢失 |
| --- | --- | --- | --- |
| 全图均匀 resampler | 每帧固定少量 latent | 场景类别、主要物体、粗位置 | 小物体、文字、接触边界 |
| 目标/夹爪优先 token | 给目标物和 wrist crop 更多预算 | 抓取点、局部边缘、夹爪关系 | 背景布局、远处约束 |
| 当前窗口 + memory token | 最近帧保细，历史压进 memory | 短期动作、长任务上下文 | memory 写错后长期污染 |
| 3D / 深度增强 token | 把几何位置显式注入视觉 token | 距离、相对位置、可达性 | 深度噪声和标定误差会变成模型输入 |

如果任务是静态图片问答，第一种可能足够。如果任务是机器人抓取，第二种或第四种通常更关键。如果任务是世界模型 rollout，第三种的 memory 写入质量会直接影响长期状态。

视觉 token 压缩最危险的地方，是它在离线指标里不一定立刻暴露。caption 仍然正确，VQA 分数也许不掉，但动作坐标会飘，遮挡后物体会丢，未来预测会把不同动作滚成相似结果。

## 世界模型和 VLA 要求动作敏感

VLM 的典型问题是：

\[
p_\theta(y\mid I,x)
\]

给定图像 \(I\) 和文本 \(x\)，输出回答 \(y\)。这类任务要求视觉 token 支持语言解释。

VLA 或 world model 更接近：

\[
\pi_\theta(a_t\mid Z_{\le t},l),
\qquad
p_\theta(s_{t+1}\mid s_t,a_t,l)
\]

这里 \(Z_{\le t}\) 是历史视觉证据，\(l\) 是语言指令，\(a_t\) 是动作，\(s_t\) 是当前状态。第一行表示策略要从视觉证据中选动作；第二行表示世界模型要预测动作后的下一状态。

这两行式子说明：动作系统不仅要“看懂画面”，还要让视觉表示对候选动作敏感。同一历史下，左推、右抓、上提、noop 应该产生不同的 latent future。若 connector 把所有动作相关局部证据都压没了，后续模型会把不同动作想成差不多的未来。

## 怎样判断 connector 是否压坏了信息

不要只看 token compression ratio，也不要只看 VQA。更可靠的读法是把证据分成四类。

| 证据 | 证明什么 | 不证明什么 |
| --- | --- | --- |
| Token / latency / memory 统计 | 压缩是否真的省成本 | 不证明状态还能控制 |
| VQA / caption / OCR | 语义和文字是否保留 | 不证明动作坐标和接触 |
| Grounding / coordinate error | 位置和局部证据是否保留 | 不证明多步未来 |
| Action sensitivity / closed-loop success | 视觉证据是否能服务动作和 rollout | 不证明开放世界泛化 |

一个真实排查案例通常长这样：换成更小 resampler 后，VQA 指标几乎不变，训练 step time 变好，但抓小物体偏 2-4 cm，遮挡后更容易抓空。回放 attention 或 saliency 后发现，模型保留了目标大轮廓，却压掉了 wrist crop、接触点和遮挡前位置。这个 connector 对问答是成功的，对 VLA 是失败的。

## 多模态报告里的 video token 数要认真读

很多技术报告会给一个看似小的数字：每帧多少视觉 token。这个数字其实非常关键。比如 Gemini 2.5 技术报告提到视频 tokenizer 更高效，每帧 token 数相比前代显著下降，从而让 1M context 能覆盖更长视频。这里的能力提升不只是“模型更大”，而是 video tokenizer、采样率、上下文预算和推理成本一起改变了可处理的时间长度。

同理，Qwen2-VL / Qwen2.5-VL 一类模型强调动态分辨率和多模态位置编码，本质上也是在处理同一个矛盾：不同图片、文档、视频和界面截图需要不同 token 预算；如果强行固定分辨率，简单场景浪费 token，复杂场景又丢细节。

读这类报告时不要只抄 benchmark。先找三个数字：视觉 token 数、最大上下文、视频采样策略。它们决定模型到底是在“看几张图”，还是能持续跟踪一段时间里的视觉状态。

## 读新论文时先问五个问题

第一，视觉 token 从哪里来：ViT patch、CNN feature、VAE latent、VQ token、object token、3D token，还是多种混合？

第二，压缩发生在哪里：encoder 内部、connector、cross-attention memory、KV cache，还是 world model latent？

第三，训练目标要求保留什么：caption、contrastive alignment、OCR、grounding、动作、reward、future prediction，还是 closed-loop success？

第四，后续模型怎样消费：LLM 一次性读视觉 token，还是每层 cross-attention 按需读 memory；policy 是直接读 token，还是先进入 state estimator？

第五，失败证据有没有公开：小物体、遮挡、文字、坐标、接触、动作分叉、长视频记忆分别有没有 ablation？

视觉 tokenizer 决定原始视觉证据怎样被切开，connector 决定哪些证据能穿过瓶颈进入 LLM、policy 或 world model。ViT 说明视觉 token 从 patch 来，Perceiver / TokenLearner 说明少量 latent 可以主动抽取信息，BLIP-2 说明 Q-Former 是连接器瓶颈，Flamingo 和 LLaVA 展示了 cross-attention 与 projector 两条常见路线。

真正要记住的是：**token 少只是成本收益，证据没丢才是能力收益。** 对 VQA 来说，语义类别和文字可能足够；对 VLA 和世界模型来说，局部几何、接触、遮挡记忆、动作敏感性和多相机一致性才是关键。连接器一旦把这些压掉，后面的模型再大也只是基于不完整证据做推理。

继续读相邻内容时，可以接 [VLM 架构：视觉证据怎样进入语言模型](architecture-and-training.md)、[VLM/VLA 与世界模型高效训练接口](vlm-vla-for-efficient-world-model-training.md)、[VLA 数据与策略学习](../vla/data-and-policy-learning.md) 和 [世界模型评测与失效模式](../world-models/evaluation-and-failure-modes.md)。

## 外部材料

- [Vision Transformer](https://arxiv.org/abs/2010.11929)：理解图片怎样从 patch 变成 token 序列。
- [Perceiver](https://arxiv.org/abs/2103.03206)：理解用少量 latent array 读取高维输入的通用思想。
- [TokenLearner](https://arxiv.org/abs/2106.11297)：理解图像/视频里学习少量 informative tokens 的路线。
- [BLIP-2](https://arxiv.org/abs/2301.12597)：看 Q-Former 怎样从冻结视觉特征中抽取少量 query token。
- [Flamingo](https://arxiv.org/abs/2204.14198)：理解 Perceiver Resampler 和 gated cross-attention。
- [LLaVA](https://llava-vl.github.io/)：对照 projector + visual instruction tuning 的简单强路线。
- [Hugging Face VLMs](https://huggingface.co/blog/vlms)：用 vision encoder、connector、language model 三段式复盘现代 VLM。
- [Octo](https://octo-models.github.io/)：对照 VLA 场景里视觉 token、动作 chunk 和多机器人数据如何一起验收。
