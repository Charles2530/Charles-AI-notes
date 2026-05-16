# 视觉 Tokenizer、连接器与信息瓶颈

VLM/VLA 的第一个工程问题不是“模型够不够大”，而是视觉信息如何进入语言模型或动作模型。图像 patch、视频帧、3D 点云、深度图和相机外参都不能原样无限塞进上下文。必须先把视觉压成 token、latent 或查询结果，再交给后续模型消费。

!!! note "初学者先抓住"
    视觉连接器的本质是信息瓶颈：保留任务需要的信息，丢掉暂时不重要的细节。VLM 关心回答问题，VLA 关心动作可执行，世界模型关心未来可预测性，这三者需要的视觉信息并不完全相同。

!!! example "贯穿例子：红杯到托盘"
    如果任务是“把红杯从桌边放到托盘上”，VLM 问答只需要知道“红杯在哪里”。VLA 还要知道杯沿、夹爪、托盘边界和障碍物。世界模型更进一步，要保留“从侧面推会不会滑落、从上方夹会不会更稳”这类动作后果信号。

## 三层接口

| 层级 | 典型形式 | 主要问题 | 失败表现 |
| --- | --- | --- | --- |
| 视觉 encoder | ViT、CNN、视频 encoder、3D encoder | 如何得到稳定视觉特征 | 小物体漏检、运动模糊、坐标漂移 |
| 连接器 | projector、Q-Former、resampler、cross-attention | 如何控制 token 数和语义密度 | 回答泛泛、定位不准、细节丢失 |
| 消费端 | LLM、policy head、world model latent | 如何服务语言、动作或未来预测 | 会描述但不会行动，或会行动但不能解释失败 |

连接器可以先用一个简化公式理解：

\[
c = g_\theta(h_v, q)
\]

| 符号 | 含义 |
| --- | --- |
| \(h_v\) | 视觉 encoder 输出的大量视觉特征 |
| \(q\) | learnable query、文本查询或任务查询 |
| \(g_\theta\) | 连接器，例如 projector、Q-Former、resampler、cross-attention |
| \(c\) | 压缩后交给 LLM、VLA policy 或 world model 的视觉 token |

**读作什么**：连接器从大量视觉特征里抽取少量后续模型能消费的信息。

**这个公式不说明什么**：它没有自动告诉我们该保留什么。VLM、VLA、world model 的验收指标不同，连接器的训练目标也要跟着不同。

## 论文依据：BLIP-2 把连接器做成信息瓶颈

![BLIP-2 Q-Former objectives](../assets/images/paper-figures/vlm/blip2-figure-2-qformer-objectives.jpg){ width="760" }

<small>图源：[BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/abs/2301.12597)，Figure 2。原论文图意：冻结图像编码器，用 Q-Former 通过少量 query token 从视觉特征中抽取信息，并分别服务 image-text contrastive learning、image-grounded text generation 和 image-text matching。</small>

!!! note "读图：连接器不是简单投影层"
    **先看哪里**：看中间的 query token，它们从冻结视觉 encoder 的输出中读取信息，而不是把所有 patch 原样交给 LLM。  
    **它证明什么**：视觉信息可以通过少量 query token 变成语言模型可用的证据。  
    **它不证明什么**：这类证据不一定保留机器人动作需要的接触、几何和遮挡信息。  
    **和红杯例子的关系**：如果 query 只保留“红杯”这个语义，丢掉杯口边缘和夹爪距离，问答仍可能正确，抓取却会偏。

## 为什么连接器是高效训练关键

视觉 token 太多，会直接放大训练成本：

1. 注意力成本随序列长度增加。
2. 多帧视频会把 token 数乘上时间维。
3. VLA 还要加入动作 token、状态 token 和语言指令。
4. 世界模型训练还要保留未来预测所需的时序信息。

因此高效训练常常不是先换更大模型，而是先问：视觉 token 是否过密，连接器是否能按任务抽取信息，视频帧是否需要全部进入上下文，哪些状态可以放进 latent memory。

### 先算清楚 token，再谈模型大小

若一帧图像被切成 \(N\) 个 patch token，\(V\) 路相机、\(T\) 帧历史的原始视觉 token 量可以粗写成：

\[
L_{\text{vision}} = V \times T \times N
\]

若连接器把每帧每路相机压成 \(K\) 个任务 token，则：

\[
L_{\text{compressed}} = V \times T \times K
\]

| 符号 | 含义 |
| --- | --- |
| \(V\) | 相机路数 |
| \(T\) | 历史帧数 |
| \(N\) | 每帧原始 patch token 数 |
| \(K\) | 每帧压缩后的任务 token 数 |
| \(L_{\text{vision}}\) | 原始视觉序列长度 |
| \(L_{\text{compressed}}\) | 压缩后视觉序列长度 |

**读作什么**：压缩收益来自 \(K \ll N\)，但前提是被保留下来的 token 仍然对任务有用。

**红杯例子**：若 \(K\) 太小，模型可能保留“红杯在桌上”，却丢掉“杯子已经靠近桌边、夹爪应从上方抓”的细节。

**这个公式不说明什么**：它只说明序列长度，不说明压缩后的状态是否还支持动作敏感预测。

## 三种常见路线

### 1. Projector：简单直接

把视觉 encoder 输出线性映射到语言模型 embedding 空间。优点是实现简单、训练稳定；缺点是 token 数和信息选择能力较弱，容易把大量视觉 patch 原样交给 LLM。

适合做基础 VLM 对齐，但在长视频、机器人多相机和世界模型训练中，通常还需要更强压缩。

### 2. Query / Resampler：主动抽取

用少量 learnable query 或 resampler token 从视觉特征中抽取信息。它更像一个可训练的摘要器：不是把所有 patch 都传下去，而是选择后续任务最需要的视觉状态。

这条路线对 VLA 很重要，因为动作决策通常不需要像素级完整重建，却需要保留物体、位姿、可达性、接触关系和目标位置。

### 3. Cross-attention / Memory：按需读取

让语言或策略 token 通过 cross-attention 从视觉 memory 中读取信息。优点是细节仍可按需访问，缺点是系统复杂度、显存和缓存管理上升。

这更接近长视频和世界模型场景：当前动作只需要一部分视觉证据，但失败恢复和反事实预测可能需要回看更早状态。

## 一个 token 成本小账

假设桌面机器人使用 4 路相机、16 帧历史，每帧视觉 encoder 输出 \(14\times14=196\) 个 patch token：

\[
4 \times 16 \times 196 = 12544
\]

这还没有算语言、动作、proprioception、memory 和未来预测 token。若连接器把每个相机帧压到 32 个任务 token，则视觉侧变成：

\[
4 \times 16 \times 32 = 2048
\]

序列长度下降约 \(12,544 / 2,048 \approx 6.1\) 倍。这个比例会线性进入 hidden activation、KV cache 和很多通信量；如果后续做全局 self-attention，attention score 的上界还会按平方下降，约 \((12,544 / 2,048)^2 \approx 37.5\) 倍。但这个收益有前提：压缩后仍要保留目标物、夹爪、接触边界、遮挡前位置和多相机几何关系。否则 token 省下来了，world model 可能失去最关键的动作后果信号。

### 压缩前后对比：不要只看 token 数

同一条“从桌面拿起红色量杯”的训练样本，可以把视觉侧压成几种不同形态。下面的数字不追求代表所有系统，而是给一个能复算的口径：4 路相机、16 帧历史、每个 patch token hidden size 为 1024，进入世界模型前用 BF16 保存。表里的“视觉激活约占”只算一个 `[L, D]` buffer，即 \(L\times1024\times2\) bytes；真实训练还会叠加 Q/K/V、MLP activation、梯度保存或 checkpoint 重算。

| 方案 | 视觉 token 数 | 视觉激活约占 | 省下了什么 | 最容易丢什么 |
| --- | ---: | ---: | --- | --- |
| 原始 patch 全进上下文 | \(4\times16\times196=12,544\) | \(12,544\times1024\times2 \approx 25.7MB\) | 几乎不省 | 计算太贵，长 horizon 难训 |
| 每帧 resampler 到 32 token | \(4\times16\times32=2,048\) | 约 4.2MB | token 约降 6.1 倍 | 小物体、接触边界、遮挡前位置 |
| 目标/夹爪优先 token | \(16\times(4\times8+12+12)=896\) | 约 1.8MB | 保留局部控制证据 | 背景布局和远处约束 |
| 当前窗口 + memory token | 最近 4 帧 512 token + 128 memory token | 约 1.3MB | 长任务上下文压力最低 | memory 写入错误会长期污染 |

如果后续模块对视觉 token 做全局 self-attention，注意力 score 的上界从 \(12,544^2\approx157M\) 个元素降到 \(2,048^2\approx4.2M\)，是约 37 倍差异。若按 16 个 heads、BF16 显式 materialize score 来粗看，前者会是数 GiB 级别的单层中间量；FlashAttention 等实现通常不会这样完整写回，但 \(L^2\) 的计算压力仍然存在。真实系统可能用 cross-attention、window attention 或稀疏读取，收益不会完全按平方兑现，但这个量级足以解释：视觉 tokenizer 不是锦上添花，而是能决定长视频世界模型是否训得动。

压缩是否成功，要看四个对照指标：

| 对照项 | 合格信号 | 失败信号 |
| --- | --- | --- |
| 语义问答 | 目标物、颜色、数量不变 | VQA 分数基本不掉 |
| 几何/坐标 | 抓取点、左右、距离稳定 | 坐标误差从 1cm 变 4cm |
| 动作敏感性 | 左推/右抓预测未来分叉 | 不同动作生成相似未来 |
| 记忆保真 | 遮挡后仍记得物体去向 | 物体回到常见初始位置 |

这也是为什么“压缩前后图像看起来差不多”不够。世界模型真正关心的是：压缩后的 latent 是否还能解释动作后果。

## 硬证据模块：视觉压缩必须给出 token 账和失败桶

视觉 tokenizer 页最容易退回“少 token 就更高效”。更可靠的证据模块要同时给压缩收益和被压坏的内容。

| 证据项 | 最小可复算例子 | 失败案例 | 验收指标 |
| --- | --- | --- | --- |
| 本页解决哪项成本 | 原始 patch token、resampler token、memory token 三列对比 | token 降了，但训练 step time 被 gather/scatter 吃掉 | token compression ratio、step time、显存 |
| 动作敏感性是否保留 | 同一历史下 left/right/grasp/noop 四个动作的 latent 差异 | 不同动作未来都像“继续当前画面” | action sensitivity、candidate ranking |
| 接触与小物体是否保留 | wrist crop / object token / full-frame 三路对照 | 抓取点、杯沿、按钮边界消失 | contact event F1、坐标误差 |
| 遮挡记忆是否保留 | 目标被遮挡后 2-6 秒再出现 | 物体回到常见初始位置 | object ID consistency、latent drift |
| 证据等级 | 只给示意图、离线 token 统计、ablation、闭环结果要分开写 | demo 图好看但策略无收益 | 证据来源明确标成 demo / ablation / closed-loop |

一个最小日志应像 `world-model-mini-chain` 那样可复算：原始 patch token 为 `82368`，压缩后 state token 为 `12292`，压缩约 `6.70x`；但同时必须报告 `top1_safe_success_agreement=2/4` 这类失败指标。否则读者只知道“视觉变省了”，不知道世界模型是否还能用它做动作判断。

## 面向 VLA 的检查点

判断一个视觉连接器是否适合 VLA，不要只看 VQA 分数，还要看：

1. 小物体、按钮、抓取点和接触边界是否保留。
2. 坐标、方向、左右、相对位置是否稳定。
3. 多相机视角是否能对齐到同一动作坐标系。
4. 跨帧状态是否能支持速度、遮挡和物体持久性。
5. 视觉 token 数是否允许长任务训练。

## 面向世界模型的检查点

世界模型不只需要“当前图像语义”，还需要“未来可预测状态”。所以连接器要额外回答：

| 问题 | 为什么重要 |
| --- | --- |
| latent 是否保留运动线索 | 没有速度和接触线索，rollout 很快漂移 |
| latent 是否对动作敏感 | 动作不同但未来一样，说明状态压缩错了 |
| token 数是否支持长 horizon | 视觉 token 过多会让长序列训练成本爆炸 |
| 是否能接入 3D / 深度 / 位姿 | 机器人和自动驾驶常需要几何一致性 |

## 真实排查案例：视觉 token 省了，抓取开始飘

| 环节 | 观察 |
| --- | --- |
| 输入症状 | 换成更小 resampler 后，VQA 指标几乎不变，但机器人抓小物体时经常偏 2-4 cm，遮挡后更容易抓空 |
| 指标 | token 数下降、step time 变好；grasp success、contact event accuracy、object permanence 明显下降 |
| Trace / 回放观察 | attention 主要集中在物体大轮廓和语言目标，夹爪附近 crop、接触点和遮挡前位置 token 被压掉 |
| 判断 | 连接器按语义问答压缩是成功的，但按动作和未来预测压缩是失败的 |
| 修复 | 增加 gripper-centric crop token；给接触、目标物和遮挡区域更高采样权重；在评测中加入抓取点、接触事件和 action sensitivity |
| 反例 | 如果任务只是图片问答或静态检索，这种压缩可能完全合理。问题不在 resampler 本身，而在把 VQA 合格误当成 VLA / world model 合格 |

## 一个实践建议

先建立一个小型连接器评测集：同一批图像/短视频，覆盖问答、定位、动作选择、未来预测四类任务。一个连接器如果只在问答上好，却在动作点、接触关系和跨帧状态上不稳，就不应该直接作为 VLA 或世界模型底座。

读下一页时，可以接 [视频表征与长时记忆](video-representation-and-memory.md)，看静态视觉 token 如何进一步变成时间状态。
