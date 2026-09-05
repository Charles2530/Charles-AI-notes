# VLM 专题：视觉证据怎样进入大模型

VLM 把像素先交给视觉 encoder，切成 token 或 latent；connector 再把其中一部分证据交给语言模型，最后模型生成回答、工具调用、动作 token 或世界模型状态。读 VLM 时先追问：视觉证据在哪一层被压缩，哪些信息穿过了瓶颈，哪些信息被丢掉了。

如果只是问答，类别、文字和关系可能已经够用；如果要做 VLA 或世界模型，局部几何、遮挡记忆、动作敏感性、接触边界和时间状态就会变成主角。

VLM 的难点不在于把图像接到语言模型前面，而在于决定哪些视觉证据要被保留、压缩和再次读取。相同的语言模型，换一个视觉 token 数量、connector 或训练目标，可能得到完全不同的 OCR、定位、视频记忆和动作表现。

## 先读这几篇

| 你想弄清楚 | 建议入口 | 读完应能回答 |
| --- | --- | --- |
| VLM 的整体结构 | [VLM 架构：视觉证据怎样进入语言模型](architecture-and-training.md) | vision encoder、connector、LLM 分别在做什么 |
| 视觉 token 为什么贵 | [视觉 Tokenizer 与连接器](visual-tokenizers-and-connectors.md) | 分辨率、帧数、多相机、query 数怎样影响成本和证据 |
| 视频状态怎样保留 | [视频表征与长时记忆](video-representation-and-memory.md) | 长视频和多轮视觉记忆为什么不能只靠静态图文语义 |
| 多模态系统如何评测 | [多模态评测与失败模式](multimodal-evaluation-and-failure-modes.md) | VQA、OCR、grounding、长视频和动作指标分别证明什么 |
| VLM 怎样服务 VLA / 世界模型 | [VLM/VLA 与世界模型高效训练接口](vlm-vla-for-efficient-world-model-training.md) | 视觉压缩何时伤到动作、rollout 和闭环评测 |

## 读新模型时先问四件事

第一，视觉 token 从哪里来：ViT patch、CNN feature、VAE latent、VQ token、object token、3D token，还是多种混合？

第二，connector 在哪里压缩：projector、Q-Former、resampler、cross-attention memory、KV cache，还是 world model latent？

第三，训练目标要求保留什么：caption、contrastive alignment、OCR、grounding、动作、future prediction，还是 closed-loop success？

第四，评测证据是否匹配 claim：VQA 不能证明 VLA，视频自然度不能证明规划，demo 不能替代失败桶、动作敏感性和闭环任务。

## 从一张图追到一次决策

读论文或系统图时，可以沿着下面这条链追踪同一条视觉证据：图像如何切块，encoder 输出多少 token，connector 丢掉了哪些细节，语言模型在什么层读取视觉信息，输出最终被谁消费。这样能把“模型理解了图像”拆成可检查的接口，而不是停留在 demo 印象。

| 证据环节 | 需要记录的量 | 常见失败 |
| --- | --- | --- |
| 输入 | 分辨率、帧数、相机数、裁剪与时间采样 | 小目标消失、跨帧错位 |
| 表征 | patch/token 数、空间网格、预训练数据 | 细节不足、领域偏移 |
| 连接 | projector、resampler、cross-attention 的压缩比例 | OCR、定位和动作相关信息被抹平 |
| 消费 | 生成答案、工具调用、动作或 latent rollout | 语言指标好，但真实任务失败 |

## 按任务进入

| 任务 | 优先关注 |
| --- | --- |
| OCR、图表和文档 | 分辨率、局部裁剪、文字 token 与 grounding 评测 |
| 长视频理解 | 时间采样、记忆更新、跨片段检索和状态保持 |
| VLA | action-conditioned 表征、延迟、动作空间和闭环恢复 |
| 世界模型 | 时间一致性、动作敏感性、不确定性与 rollout 误差 |
