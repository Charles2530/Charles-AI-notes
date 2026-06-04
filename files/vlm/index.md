# VLM 专题：视觉证据怎样进入大模型

VLM 不是“LLM 加一张图”。更准确的读法是：像素先被视觉 encoder 切成 token 或 latent，connector 再把其中一部分证据交给语言模型，最后模型才生成回答、工具调用、动作 token 或世界模型状态。真正值得追问的是：视觉证据在哪一层被压缩，哪些信息穿过了瓶颈，哪些信息被丢掉了。

如果只是问答，类别、文字和关系可能已经够用；如果要做 VLA 或世界模型，局部几何、遮挡记忆、动作敏感性、接触边界和时间状态就会变成主角。所以本站把 VLM 入口拆成几条具体问题线，而不是做模型清单。

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

VLM 这组文章会参考高质量博客和中文公众号的讲法，比如用图和小账本先建立直觉；事实、图源、模型细节和强 claim 仍回到论文、官方项目页和技术报告。
