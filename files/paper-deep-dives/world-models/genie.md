---
paper_date: 2024-02
paper_title: "Genie: Generative Interactive Environments"
paper_topic: 世界模型
---

# Genie：无标注视频到可交互环境

!!! info "论文信息"
    - 论文：`Genie: Generative Interactive Environments`
    - 链接：[arXiv:2402.15391](https://arxiv.org/abs/2402.15391)
    - 项目页：[Genie: Generative Interactive Environments](https://sites.google.com/view/genie-2024/)
    - 关键词：生成式交互环境、视频世界模型、latent action、ST-transformer、MaskGIT、无动作标签训练


Genie 的关键点不是“视频生成质量更好”，而是它把一个更难的问题拆出来：**只有互联网视频，没有动作标签，也要学出可以被用户逐帧控制的潜在动作接口。**

这使它在世界模型谱系里处在一个很特殊的位置。Dreamer 这类方法从环境交互轨迹中学习 \(p(z_{t+1}\mid z_t,a_t)\)，动作是天然给定的；普通视频模型学习 \(p(x_{t+1:T}\mid x_{\le t}, c)\)，通常没有逐帧控制。Genie 试图从纯视频里同时学出“世界如何动”和“什么动作造成这种变化”。

## 论文位置

Genie 把自己定义成 `generative interactive environments`。这不是普通 video continuation，也不是传统 model-based RL 世界模型。它的训练数据只有视频，推理时却要求有 frame-level controllability。

论文 Table 1 的定位非常直接：

| Model Class | Training Data | Controllability |
| --- | --- | --- |
| World Models | Video + Actions | Frame-level |
| Video Models | Video + Text | Video-level |
| Genie | Video | Frame-level |

<small>表源：`Genie: Generative Interactive Environments`，Table 1。原论文表格要点：传统 world models 依赖视频和动作，video models 通常只做到 video-level 控制；Genie 强调只用 video data，却学习 frame-level latent action 控制。</small>

!!! note "这张图怎么读"
    **输入输出**：输入是无标注视频 token，输出是 latent action 和可交互生成环境。  
    **效率机制**：用视频数据学习可操作动态，减少对显式动作标注和手写模拟器的依赖。  
    **对主线意义**：它展示外部世界模拟器路线如何从视频中提取交互结构。  
    **不能证明什么**：latent action 可交互不等于真实机器人动作因果或安全规划可用。

这张表里的 `Genie = Video + Frame-level` 是整篇论文的核心声明。它不是说 Genie 已经解决了所有世界模型问题，而是说：如果动作标签是构建可交互世界模型的主要瓶颈，那么可以先从视频中无监督地抽出一套离散潜在动作，再用这套潜在动作驱动未来帧生成。

![Genie model training 原论文图](https://arxiv.org/html/2402.15391v1/figures/genie_architecture.png){ width="920" }

<small>图源：`Genie: Generative Interactive Environments`，arXiv HTML Figure 2。原论文图意：输入视频帧先经过 video tokenizer 得到离散视频 token，同时由 latent action model 推断相邻帧之间的 latent actions；二者一起输入 dynamics model，迭代预测下一帧。</small>

!!! note "这张总图怎么读"
    图里有三条线。蓝色线是把视频帧压成离散 token，黄色线是从相邻帧中反推出潜在动作，橙色模块是基于历史 token 和 latent action 预测未来 token 的 dynamics model。

    最关键的是黄色线。Genie 没有真实键盘、手柄或机器人动作标签，因此它必须从“画面变化”里推断一组可复用的 action codes。推理时，用户不再输入真实动作，而是选择这些 learned latent actions。只要这些 action codes 在不同 prompt 上具有一致语义，模型就获得了类似“新手柄按键”的控制接口。

## 和 LingBot-World 的关系

LingBot-World 和 Genie 都属于视频生成走向交互世界模型的路线，但二者的切入点不同。

| 维度 | Genie | LingBot-World |
| --- | --- | --- |
| 训练起点 | 从视频-only 数据学习 latent action 和 dynamics | 从强视频生成底座继续训练成交互模拟器 |
| 动作来源 | 无监督 latent action，训练不需要真实动作标签 | 更强调动作条件、相机控制、交互轨迹和系统接口 |
| 生成主干 | 离散 token + autoregressive MaskGIT dynamics | 大视频生成模型、动作适配、因果化和少步蒸馏 |
| 关键问题 | 没有动作标签时如何仍然获得 frame-level control | 如何把高质量视频生成器工程化成实时可交互 world simulator |
| 局限焦点 | 16 帧记忆、约 1 FPS、控制语义仍是潜在动作 | 长时一致性、实时延迟、动作 grounding 和部署成本 |

所以读 Genie 时不要把它当成 LingBot-World 的低配版。Genie 解决的是更早一层的数据问题：**如果互联网视频没有动作标签，能不能先学出一套可迁移的 latent action interface？** LingBot-World 解决的是更靠后的系统问题：**有了强视频模型和交互数据后，怎样把它因果化、实时化、长时化？**

## 核心问题

普通视频模型只需要回答：

$$
p(x_{t+1:T}\mid x_{\le t}, c)
$$

它可以沿着视频惯性生成一个合理未来。世界模型还要回答：

$$
p(x_{t+1:T}\mid x_{\le t}, a_{t:T-1})
$$

也就是未来必须随着动作改变。Genie 的难点是训练数据里没有 \(a_t\)，于是它把问题改成：

1. 从视频中学习离散 latent action \(\tilde a_t\)；
2. 用 \(\tilde a_t\) 作为 dynamics model 的控制输入；
3. 推理时允许用户直接选择 \(\tilde a_t\)，逐帧驱动生成环境。

这一步很像给一个没有标签的视频库反推出“控制器按键”。如果 action code 只是记住某个数据集里的短期视觉差异，那它不能泛化；如果 action code 在不同场景中都稳定对应左、右、跳、停止或机械臂方向，那么它就能成为可交互世界模型的动作接口。

## ST-transformer：为什么整篇都用同一类结构

Genie 的 latent action model、video tokenizer 和 dynamics model 都采用 ST-transformer。原因很直接：视频 token 数量很大，全局 self-attention 对 \(T\times H\times W\) token 做二次复杂度不现实。

![ST-transformer architecture 原论文图](https://arxiv.org/html/2402.15391v1/figures/sttransformer.png){ width="860" }

<small>图源：`Genie: Generative Interactive Environments`，arXiv HTML Figure 3。原论文图意：每个 spatiotemporal block 由 spatial attention、temporal attention 和 feed-forward layer 组成；spatial layer 在同一帧内做注意力，temporal layer 在相同空间位置跨时间做注意力。</small>

!!! note "为什么它适合视频世界模型"
    spatial attention 处理同一帧内的 \(H\times W\) token，temporal attention 处理同一空间位置跨 \(T\) 帧的 token。这样主计算成本随帧数近似线性增长，而不是对所有时空 token 做全局二次 attention。

    这对世界模型很重要。模型既要看懂单帧里角色、平台、障碍物和机器人手臂的位置，也要保留跨帧运动方向。Genie 还让 temporal layer 使用 causal mask，使每个离散编码只汇聚过去信息，更接近自回归交互生成时的依赖结构。

## 三个模型组件

### 1. Video Tokenizer：先把视频压成离散 token

Genie 不是直接在像素上预测未来帧，而是先训练一个视频 tokenizer，将视频帧压成离散 token：

$$
x_{1:T}\rightarrow z_{1:T}
$$

它使用 VQ-VAE 风格目标，并在 encoder 和 decoder 中加入 ST-transformer。论文强调这和只做 spatial compression 的 tokenizer 不同：ST-ViViT 的编码 \(z_t\) 含有过去帧 \(x_{1:t}\) 的时序信息，因此更适合后续 dynamics model 做视频预测。

![Video tokenizer 原论文图](https://arxiv.org/html/2402.15391v1/figures/tokenizer_architecture.png){ width="820" }

<small>图源：`Genie: Generative Interactive Environments`，arXiv HTML Figure 5。原论文图意：video tokenizer 是带 ST-transformer 的 VQ-VAE，用于把视频帧压缩到离散 video tokens，再由 decoder 还原回图像空间。</small>

训练细节里最值得注意的是：tokenizer 先单独训练，之后 dynamics model 使用冻结后的 token 表示。这样可以把“视频压缩质量”和“动作条件动力学”拆开，避免在大模型训练初期同时解决像素重建和长期预测两个难题。

### 2. Latent Action Model：从相邻帧反推出动作

Latent Action Model（LAM）是 Genie 的核心。它做的不是 policy，也不是最终推理时会保留的完整模块，而是训练阶段用于学习 action codebook 的辅助模型。

![Latent action model 原论文图](https://arxiv.org/html/2402.15391v1/figures/LAM_architecture.png){ width="860" }

<small>图源：`Genie: Generative Interactive Environments`，arXiv HTML Figure 4。原论文图意：LAM encoder 读取历史帧和下一帧，输出连续 latent actions；VQ codebook 将其离散化；decoder 只拿历史帧和 latent action 来预测下一帧，从而迫使 latent action 编码关键变化。</small>

LAM 的训练逻辑可以拆成三步：

1. encoder 看到 \(x_{1:t}\) 和 \(x_{t+1}\)，推断连续 latent action \(\tilde a_t\)；
2. VQ codebook 把 \(\tilde a_t\) 离散化，论文主实验使用 8 个 latent actions；
3. decoder 只根据历史帧和 latent action 重建下一帧 \(\hat x_{t+1}\)。

这个设计有一个很强的约束：decoder 不能直接看到真实下一帧，所以 latent action 必须携带“从过去到未来发生了什么变化”的信息。如果角色向右移动、机器人手臂下压、物体被推开，这些变化都必须被压进少量离散 action codes 里。

!!! note "为什么 codebook 不能随便变大"
    latent actions 越多，重建下一帧可能越容易，因为模型可以用更多 code 表示细粒度变化。但交互性会变差：人或 agent 很难探索一个过大的隐式手柄。论文也明确提到，增加 codes 有收益，但会降低 human 和 AI agents 的可玩性。因此 Genie 的 8-action codebook 是重建能力和可控性的折中。

推理时，LAM 的 encoder 和 decoder 都不再使用，只有 VQ codebook 保留下来。用户选择整数动作 \(a_t\in[0,|\mathcal A|)\)，模型查表得到对应 latent action embedding，再交给 dynamics model 生成下一帧。

### 3. Dynamics Model：用 MaskGIT 自回归预测下一帧 token

Genie 的 dynamics model 是 decoder-only MaskGIT transformer。它接收历史 video tokens 和 stop-gradient latent actions，预测未来 frame tokens。

![Dynamics model 原论文图](https://arxiv.org/html/2402.15391v1/figures/dynamics_architecture.png){ width="820" }

<small>图源：`Genie: Generative Interactive Environments`，arXiv HTML Figure 6。原论文图意：dynamics model 接收 video tokens 与 action embeddings，预测 masked future video tokens。</small>

训练目标是 token-level cross-entropy：

$$
\mathcal L_{\text{dyn}}=
\operatorname{CE}\left(\hat z_{2:T}, z_{2:T}\right)
$$

其中 dynamics model 使用 \(z_{1:T-1}\) 和 \(\tilde a_{1:T-1}\) 预测 \(z_{2:T}\)。训练时会随机 mask 输入 token，masking rate 从 \(0.5\) 到 \(1.0\) 均匀采样。这个设置让模型学会在不同缺失程度下补全未来 token，也贴近 MaskGIT 多步解码。

另一个容易忽略的设计是 action 注入方式。许多 transformer world model 会把 action 拼接到对应 frame token 上；Genie 发现把 latent action 当作 additive embeddings，用在 LAM 和 dynamics model 中，能提升生成可控性。直觉上，这让动作更像“调制每一步动态”的条件，而不是被塞在序列里等模型自己发现。

## 推理：像按一个新手柄一样控制生成环境

![Genie inference 原论文图](https://arxiv.org/html/2402.15391v1/figures/genie_inference.png){ width="920" }

<small>图源：`Genie: Generative Interactive Environments`，arXiv HTML Figure 7。原论文图意：prompt frame 先被 tokenizer encoder 编码，用户输入 latent action，dynamics model 迭代预测下一帧 token，再由 tokenizer decoder 解码回图像空间。</small>

推理流程是：

1. 用户给一个初始图像 \(x_1\)，可以是生成图、草图、照片或真实视频帧；
2. tokenizer encoder 将 \(x_1\) 变成 \(z_1\)；
3. 用户选择一个离散 latent action；
4. dynamics model 预测下一帧 token；
5. tokenizer decoder 解码成下一帧；
6. 新帧继续进入下一轮，形成自回归 rollout。

这里的交互体验不是“输入一句 prompt 生成一个视频”，而是每一步都能选动作。论文把 latent action 的理解比作学习一个新控制器：一开始不知道 0 到 7 分别对应什么，但在多个环境里试几次后，某些动作会稳定表现为左、右、跳、停止、机械臂上移或下移等语义。

## 数据管线

Genie 的数据管线很值得展开，因为它解释了为什么这篇论文能在没有动作标签时仍然得到可玩控制。

主数据集是 2D Platformer 互联网视频：

| Stage | Detail |
| --- | --- |
| Initial filter | public Internet videos whose title contains 2D platformer keywords |
| Extra keyword constraint | title or description contains an action word such as `speedrun` or `playthrough` |
| Negative filter | title must not contain words such as `movie` or `unboxing` |
| Clip format | 16s clips at 10 FPS |
| Raw scale | 55M clips, around 244k hours |
| Curated scale | 6.8M clips, over 30k hours |
| Resolution | 160x90 in the main Platformers setup |

论文没有简单相信“数据越多越好”。它发现低质量视频会明显影响模型表现，于是做了一个 learned filtering pipeline：

1. 团队手工标注 10k videos，用大约 10 小时人力，评分从 1 到 5；
2. 删除 2 到 4 分的中间样本，只把 5 视作 good、1 视作 bad；
3. 训练 11M 参数 ResNet18 做二分类；
4. 按模型预测和置信度筛选最终视频。

Table 4 的结果说明，筛选后数据只有原始集合的约十分之一，但 FVD 更好：

|  | #Params | FVD (↓) |
| --- | --- | --- |
| Original dataset (55M videos) | 580M | 61.4 |
| Curated dataset (6.8M videos) | 580M | 54.8 |

<small>表源：`Genie: Generative Interactive Environments`，Table 4。原论文表格要点：高质量 curated dataset 虽然规模小很多，但 FVD 优于原始 55M videos 数据集。</small>

这对世界模型训练很重要。可交互模型不只是拟合像素分布，还要从画面变化中推断动作。如果视频里有菜单、主播脸、剪辑、静态画面或非玩法片段，latent action 会学到混乱的变化来源，后续控制接口自然不稳定。

## 训练细节

Genie 的训练不是一个端到端单阶段。更合理的理解是：

```text
Video tokenizer pretraining
  -> freeze / use tokenizer tokens
  -> LAM + dynamics model co-training
  -> dynamics scaling
  -> latent-action-controlled inference
```

### 主模型训练设定

| Component | Key setting |
| --- | --- |
| Video tokenizer | 200M parameters |
| Tokenizer patch size | 4 |
| Tokenizer codebook | embedding size 32, 1024 unique codes |
| Latent action model | 300M parameters |
| LAM patch size | 16 |
| LAM codebook | embedding size 32, 8 unique codes |
| Sequence length | 16 frames |
| FPS | 10 |
| Dynamics stabilization | bfloat16, QK norm |
| Inference sampling | 25 MaskGIT steps per frame |
| Sampling temperature | 2 |

<small>表源：论文主文 `Training Details` 段落。保留英文配置名，中文说明仅用于解释。</small>

这里有几个工程判断点。

第一，tokenizer codebook 1024 负责视觉重建与视频 tokenization，LAM codebook 8 负责控制接口。二者不能混同。前者要足够表达视觉细节，后者要足够小以便用户或 agent 可探索。

第二，16 frames at 10 FPS 意味着主模型上下文大约 1.6 秒。这能展示短期交互，但远不是长时世界记忆。论文后面也把 16 帧记忆列为局限。

第三，25 MaskGIT steps per frame 和约 1 FPS 的推理速度说明 Genie 更像概念验证和研究原型，还不是 LingBot-World / Genie 3 那类实时交互系统。

### Latent Action Model hyperparameters

| Component | Parameter | Value |
| --- | --- | --- |
| Encoder | num layers | 20 |
| Encoder | d model | 1024 |
| Encoder | num heads | 16 |
| Decoder | num layers | 20 |
| Decoder | d model | 1024 |
| Decoder | num heads | 16 |
| Codebook | num codes | 8 |
| Codebook | patch size | 16 |
| Codebook | latent dim | 32 |

<small>表源：`Genie: Generative Interactive Environments`，Table 5。原论文补充说明：model inputs normalized between 0 and 1，decoder final outputs go through sigmoid。</small>

LAM 的 encoder/decoder 都不小，但它们不是最终推理主干。它们承担的是“动作发现”的训练任务：把连续视觉变化压成少量离散 code，并让这些 code 对下一帧重建有用。

### Video tokenizer hyperparameters

| Component | Parameter | Value |
| --- | --- | --- |
| Encoder | num layers | 12 |
| Encoder | d model | 512 |
| Encoder | num heads | 8 |
| Encoder | k/q size | 64 |
| Decoder | num layers | 20 |
| Decoder | d model | 1024 |
| Decoder | num heads | 16 |
| Decoder | k/q size | 64 |
| Codebook | num codes | 1024 |
| Codebook | patch size | 4 |
| Codebook | latent dim | 32 |

<small>表源：`Genie: Generative Interactive Environments`，Table 7。原论文还说明 tokenizer 训练 300k steps，AdamW optimizer，cosine decay。</small>

论文还报告了 tokenizer batch size scaling：

| batch size | training hardware | FLOPs | PSNR |
| --- | --- | --- | --- |
| 64 | 64 TPUv2 | 4.22 × 10^20 | 35.7 |
| 384 | 64 TPUv3 | 2.57 × 10^21 | 36.5 |

<small>表源：`Genie: Generative Interactive Environments`，Table 6。</small>

这组结果说明 tokenizer decoder scale 和 batch size 对重建质量有影响，但收益不是无限大。对后续 dynamics 来说，tokenizer 的职责是给出足够稳定、足够可预测的离散状态，而不是单独追求最好看的重建。

### Optimizer hyperparameters

| Video tokenizer optimizer | Value |
| --- | --- |
| max lr | 3e-4 |
| min lr | 3e-4 |
| β1 | 0.9 |
| β2 | 0.9 |
| weight decay | 1e-4 |
| warmup steps | 10k |

<small>表源：`Genie: Generative Interactive Environments`，Table 8。</small>

| Dynamics model optimizer | Value |
| --- | --- |
| max lr | 3e-5 |
| min lr | 3e-6 |
| β1 | 0.9 |
| β2 | 0.9 |
| weight decay | 1e-4 |
| warmup steps | 5k |

<small>表源：`Genie: Generative Interactive Environments`，Table 9。</small>

dynamics model 的学习率比 tokenizer 低一个数量级，且使用 decay。结合 bfloat16 和 QK norm，可以看出论文在大模型稳定性上采取的是保守路线：先稳定 tokenization，再让大 dynamics transformer 学预测和控制。

### Scaling experiments

![Scaling results 原论文图](https://arxiv.org/html/2402.15391v1/x2.png){ width="920" }

<small>图源：`Genie: Generative Interactive Environments`，arXiv HTML Figure 8。原论文图意：左图展示不同 dynamics model size 的 training curves，中图展示最后 300 updates 平均 training loss，右图展示 2.3B 模型在不同 batch size 下的 final training loss。</small>

论文做了两个 scaling 方向：model size 和 batch size。model size 实验固定 batch size 256，每个模型训练 200k steps，总计 750B training tokens。

| Parameters | num layers | num heads | d model | k/q size | training hardware | training time | FLOPs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 41M | 18 | 8 | 512 | 64 | 64 TPUv2 | 3 days | 2.05 × 10^20 |
| 96M | 16 | 16 | 768 | 64 | 64 TPUv2 | 6 days | 3.58 × 10^20 |
| 192M | 20 | 18 | 1024 | 64 | 64 TPUv2 | 9 days | 6.4 × 10^20 |
| 404M | 21 | 12 | 1536 | 128 | 64 TPUv2 | 18 days | 1.2 × 10^21 |
| 811M | 20 | 20 | 2048 | 128 | 128 TPUv3 | 7 days | 2.2 × 10^21 |
| 1.6B | 28 | 22 | 2560 | 128 | 128 TPUv3 | 12 days | 4.04 × 10^21 |
| 2.7B | 36 | 22 | 3072 | 128 | 256 TPUv3 | 16 days | 6.91 × 10^21 |

<small>表源：`Genie: Generative Interactive Environments`，Table 10。原论文说明：所有模型训练 200k steps，batch size 256，约 750B tokens；使用 batch parallelism、stage-3 ZeRO sharding，大模型还使用 tensor parallelism。</small>

最终 Genie dynamics model 配置如下：

| Parameters | num layers | num heads | d model | k/q size | FLOPs |
| --- | --- | --- | --- | --- | --- |
| 10.1B | 48 | 36 | 5120 | 128 | 6.6 × 10^22 |

<small>表源：`Genie: Generative Interactive Environments`，Table 12。论文主文说明最终 dynamics model batch size 512，125k steps，256 TPUv5p；加上 tokenizer 和 action model 后总参数约 10.7B，训练 942B tokens。</small>

从世界模型角度看，Figure 8 和这些表格支持的是一个有限但重要的结论：在固定 tokenizer 和 LAM 设计下，dynamics model 的 loss 随模型规模和 batch size 增大而下降。这说明“纯视频 + latent action + token dynamics”的路线是可 scale 的，但论文并没有证明它天然解决长期一致性、真实动作 grounding 或复杂 3D 世界控制。

## 评测：质量和可控性要分开看

Genie 报告两个维度：

1. `FVD`：视频质量或视频分布距离，越低越好；
2. `∆tPSNR`：动作可控性指标，越高越好。

论文定义：

$$
\Delta_t \mathrm{PSNR}
= \mathrm{PSNR}(x_t,\hat x_t)
- \mathrm{PSNR}(x_t,\hat x'_t)
$$

其中 \(\hat x_t\) 是使用从 ground-truth 视频推断出来的 latent actions 生成的帧，\(\hat x'_t\) 是使用随机 latent actions 生成的帧。若随机动作生成结果明显偏离 ground truth，而 inferred actions 生成结果更接近 ground truth，则 \(\Delta_t\mathrm{PSNR}\) 更大，说明动作对未来有更强影响。论文报告 \(t=4\)。

!!! note "为什么这个指标有用但不充分"
    这个指标检查的是“换动作后视频是否变化”，比只看 FVD 更接近可控性。但它不能完全证明动作因果正确。一个模型可能对 latent action 很敏感，却把动作解释成错误语义；也可能短期可控，长时 rollout 仍然漂移。因此 Genie 的结果要结合 qualitative trajectories、robotics latent actions 和 BC transfer 一起看。

### LAM 输入消融

|  | Dataset | #Params | FVD (↓) | ∆tPSNR(↑) |
| --- | --- | --- | --- | --- |
| Token-input | Platformers | 2.3B | 38.8 | 1.33 |
| Pixel-input (Genie) | Platformers | 2.5B | 40.1 | 1.91 |
| Token-input | Robotics | 1B | 257.8 | 1.65 |
| Pixel-input (Genie) | Robotics | 1B | 136.4 | 2.07 |

<small>表源：`Genie: Generative Interactive Environments`，Table 2。原论文表格要点：LAM 直接使用 pixels 虽然在 Platformers 上 FVD 略高于 token-input，但 ∆tPSNR 更高；Robotics 上 pixel-input 同时显著降低 FVD 并提高可控性。</small>

这个消融很重要。它说明如果 LAM 只看 tokenizer 后的 tokens，某些运动细节可能已经在压缩时丢失，导致 latent action 不够可控。也就是说，动作发现最好直接从原始视频变化中学习，而不是完全依赖已经为重建优化过的视觉 token。

### Tokenizer architecture 消融

|  | #Params | Memory | FVD (↓) | ∆tPSNR(↑) |
| --- | --- | --- | --- | --- |
| ViT | 230M | 0.3GB | 114.5 | 1.39 |
| [C-ViViT / Phenaki (Villegas et al., 2022)](https://arxiv.org/abs/2210.02399) | 225M | 1.6GB | 272.7 | 1.37 |
| ST-ViViT (ours) | 205M | 0.9GB | 81.4 | 1.66 |

<small>表源：`Genie: Generative Interactive Environments`，Table 3。原论文表格要点：ST-ViViT 在相近参数下同时取得更好 FVD 与 ∆tPSNR，并在 memory 上低于 C-ViViT。</small>

这支持了 ST-transformer 的核心设计：视频 tokenizer 不能只看单帧空间压缩，也不能用过重的全时空 attention 让计算不可扩展。Genie 选择 spatial/temporal factorization，是在效果、内存和长视频可扩展性之间的折中。

## Qualitative 结果怎么读

![Playing from image prompts 原论文图](https://arxiv.org/html/2402.15391v1/x3.png){ width="920" }

<small>图源：`Genie: Generative Interactive Environments`，arXiv HTML Figure 9。原论文图意：Genie 可以用 text-to-image 生成图、手绘草图或真实照片作为 prompt，并在连续执行同一 latent action 后产生可见角色移动。</small>

Figure 9 最重要的不是画质，而是 OOD prompt 下的可玩性。训练数据来自 2D platformer 视频，但 prompt 可以是生成图、草图和真实照片。模型仍然能把这些图片解释成可交互平台环境，并让角色发生运动。这说明 dynamics model 学到了一种“平台游戏式世界演化先验”。

![Robotics latent actions 原论文图](https://arxiv.org/html/2402.15391v1/figures/action_grid_robotics.png){ width="900" }

<small>图源：`Genie: Generative Interactive Environments`，arXiv HTML Figure 12。原论文图意：Robotics 数据集上，同一 latent action 在不同起始帧中产生一致语义，例如 down、up 和 left，且训练时没有使用动作标签。</small>

Robotics 结果更接近世界模型的泛化论点。论文把 RT1、仿真数据和 209k real robot episodes 当作视频使用，不使用其中动作标签。模型仍然能学出一致的机械臂动作方向，并能生成物体交互和形变。

![Deformable object 原论文图](https://arxiv.org/html/2402.15391v1/figures/chips.png){ width="860" }

<small>图源：`Genie: Generative Interactive Environments`，arXiv HTML Figure 10。原论文图意：模型在连续 rollout 中模拟机械臂与薯片袋等可变形物体的交互。</small>

这类结果说明 Genie 不只是移动相机或平移贴图。它在一定程度上学到了对象状态变化和接触后果。但需要保持边界意识：论文展示的是短序列视觉预测，不等于可作为真实机器人控制的高保真物理仿真器。

## 用 latent action 训练 agent

Genie 还做了一个很关键的 proof of concept：用冻结 LAM 给 unseen expert videos 打 latent action 标签，然后训练一个 policy 去预测这些 latent actions。执行到真实环境时，再用少量 action-labeled expert samples 把 latent actions 映射到 real actions。

流程可以写成：

```text
expert video without actions
  -> frozen Genie LAM labels latent actions
  -> train policy pi(latent_action | observation)
  -> small real-action dataset builds latent-to-real mapping
  -> execute in CoinRun
```

![Playing from RL environments 原论文图](https://arxiv.org/html/2402.15391v1/x4.png){ width="820" }

<small>图源：`Genie: Generative Interactive Environments`，arXiv HTML Figure 13。原论文图意：Genie 可以在未见过的 RL 环境初始图像上生成多样轨迹。</small>

这个实验的意义是：latent action 不只是给人玩 demo 的按钮，也可能成为 imitation-from-observation 的中间表示。如果一个视频里没有真实动作，LAM 仍然可以给每个相邻帧打上“潜在动作标签”；policy 学的是这些标签，再通过少量真实动作样本校准到环境动作空间。

论文报告 LAM-based policy 在 CoinRun hard/easy setting 中，只需少量 expert labels 做适配，就能接近 oracle behavioral cloning。这个结论不能外推到所有环境，但它证明了 latent action interface 对 agent training 有实际用途。

## 可复现实验：小规模 CoinRun recipe

论文附录给了一个能在单张中端 TPU/GPU 下跑通的小规模 case study。这个部分对工程复现很有价值。

数据收集：

| Item | Value |
| --- | --- |
| Environment | Procgen CoinRun hard mode |
| Policy | random policy with no action repeats |
| Level seeds | 0 to 10,000 |
| Timesteps per level | 1,000 |
| Total transitions | 10M |

Tokenizer 小模型：

| Component | Parameter | Value |
| --- | --- | --- |
| Encoder | num layers | 8 |
| Encoder | d model | 512 |
| Encoder | num heads | 8 |
| Decoder | num layers | 8 |
| Decoder | d model | 512 |
| Decoder | num heads | 8 |
| Codebook | num codes | 1024 |
| Codebook | patch size | 4 |
| Codebook | latent dim | 32 |

训练设置是 batch size 48 sequences、sequence length 16，总计 768 images per batch；16G 单 TPU 可容纳，训练 3 天完成 300k steps。

Dynamics + LAM 小模型：

| Component | Parameter | Value |
| --- | --- | --- |
| Action model Encoder | num layers | 8 |
| Action model Encoder | d model | 512 |
| Action model Encoder | num heads | 8 |
| Action model Decoder | num layers | 8 |
| Action model Decoder | d model | 512 |
| Action model Decoder | num heads | 8 |
| Action model Codebook | num codes | 6 |
| Action model Codebook | latent dim | 32 |
| Dynamics Architecture | num layers | 12 |
| Dynamics Architecture | d model | 512 |
| Dynamics Architecture | num heads | 8 |
| Dynamics Sampling | temperature | 1.0 |
| Dynamics Sampling | maskgit steps | 25 |

<small>表源：`Genie: Generative Interactive Environments`，Tables 15-17。原论文说明：LAM 和 dynamics model 并行训练 200k steps，batch size 36 sequences，每段 16 frames，总计 576 images，使用 Table 9 的 optimizer hyperparameters。</small>

这套 recipe 的价值在于，它把 Genie 从 10B 级系统压成一个可验证的最小路线：

1. 收集无动作视频；
2. 先训练 video tokenizer；
3. 再并行训练 LAM 和 dynamics；
4. 检查 latent actions 是否有一致语义；
5. 用 MaskGIT steps 做可交互 rollout。

如果要在自己的项目里复现思想，不应该一开始就追 11B。更稳的做法是先在 CoinRun、MiniGrid、简化机器人视频或合成 2D 环境中验证 latent action 是否真的可控，再考虑扩大数据和模型。

## 局限

论文自己指出了三个很关键的边界。

第一，Genie 仍会 hallucinate unrealistic futures。自回归视频模型一旦滚动生成，短期错误可能被后续帧继承并放大。

第二，模型只有 16 frames memory。以 10 FPS 计算，这大约是 1.6 秒上下文，对长时探索、回到旧位置、保持地图结构和长期任务状态远远不够。

第三，推理速度约 1 FPS。即使它可以逐帧控制，也还达不到真实游戏或机器人闭环所需的交互帧率。

此外还有两个工程边界需要补充。Genie 的 latent actions 是模型自己发现的，不保证和真实动作空间一一对应；在平台游戏里这可能能被人工解释成左、右、跳，但在复杂 3D、机器人多关节或自动驾驶场景里，latent-to-real mapping 会更难。另一个边界是评测仍偏 open-loop 和短期 qualitative，距离“用世界模型提升真实策略收益”还有距离。

## 对世界模型训练的启发

Genie 给后续世界模型训练留下了几个可复用经验。

第一，动作标签不是唯一入口。对于大量无动作视频，可以先训练 inverse-style latent action model，把变化压成离散 action code，再让 dynamics model 以这些 code 为条件学习未来。

第二，动作空间大小是建模和交互之间的折中。更大的 codebook 可能提升重建，但更小的 codebook 更容易形成可探索、可解释、可迁移的控制接口。

第三，video tokenizer 的时序性会影响 dynamics。只做空间压缩可能丢掉运动线索；过重的全时空 tokenizer 又难以 scale。ST-ViViT 说明 tokenizer 本身也是世界模型训练的一部分。

第四，数据质量比原始规模更重要。菜单、剪辑、主播脸和非玩法片段会污染 latent action 学习。对交互世界模型来说，过滤规则必须围绕“清晰可控的状态变化”设计，而不仅是视频美观。

第五，评测必须拆成 fidelity 和 controllability。FVD 低不代表动作有效；\(\Delta_t\mathrm{PSNR}\) 高也不代表长期世界一致。真正扎实的评测应同时报告短期动作敏感性、长时漂移、闭环任务收益和 latent-to-real action transfer。

## 读完该怎么放回世界模型路线

Genie 不是 Dreamer 那种直接服务 policy optimization 的 latent dynamics world model，也不是 LingBot-World 那种强调实时系统落地的视频世界模拟器。它最值得记住的是一个训练范式：

```text
unlabelled Internet videos
  -> discover discrete latent actions
  -> train action-conditioned video token dynamics
  -> expose latent action interface for humans / agents
```

如果你在做机器人、游戏或视频世界模型，而手头大量视频没有动作标签，Genie 是必须读的一篇。它提醒我们：动作不一定一开始就来自传感器或控制器，也可以先作为“解释相邻状态变化的离散潜变量”被学习出来。真正落到工程系统时，再把这套 latent action interface 接到真实动作、策略、规划器或交互 UI。
