# 论文精读旧入口

这个页面保留给旧链接和搜索结果使用；它不再作为必读总览页。现在站点优先让读者直接进入一篇能独立讲清楚问题的文章。

## 推荐直接进入

- [ZeRO：零冗余优化器](foundations/zero.md)
- [Dreamer：想象中训练控制](world-models/dreamer.md)
- [RT-2：VLM 到机器人动作](embodied-ai/rt2.md)
- [Megatron-LM：Tensor Model Parallel](foundations/megatron-lm.md)
- [EAGLE-3：投机推理加速](inference/eagle-3.md)

## 统一多模态世界模型专题

这组页面按 LingBot-World 的证据写法读：每篇都保留关键原论文图，英文表格尽量按原格式重绘，并把训练数据、token、loss、阶段、推理成本和评测边界写清楚。Lance 已有独立专题页，本轮不重复新增，只作为同主题参照。

| 页面 | 一句话定位 | 重点看 |
| --- | --- | --- |
| [BAGEL](world-models/bagel.md) | 大规模交错多模态预训练里的 MoT 底座与世界建模涌现 | MoT、双视觉编码、video/web interleaved data、generalized causal attention |
| [Emu3.5](world-models/emu3-5.md) | 用视觉语言 next-state prediction 做原生多模态世界学习器 | 13T+ pretraining、video frames + transcripts、SFT 世界任务、DiDA |
| [Beyond Language Modeling](world-models/beyond-language-modeling.md) | 从零控制实验验证统一多模态预训练的表示、数据、MoE 和 scaling 规律 | RAE、Transfusion、action-conditioned video、MoE specialization、vision data hunger |
| [Lance](world-models/lance.md) | 已有标杆页，用 shared interleaved context 和 dual expert pathways 统一图像/视频理解、生成和编辑 | MaPE、PT/CT/SFT/RL、LM head + flow head、cross-task data synergy |

## 候选延伸阅读

| 论文 | 适合补到哪里 | 暂不展开原因 |
| --- | --- | --- |
| InternVL-U | 统一理解、推理、生成和编辑的轻量 UMM | 与 Lance/BAGEL 主题重合，先作为效率对照 |
| UniVideo | 视频理解、生成和编辑统一框架 | 很适合后续视频专题，但本轮优先写世界模型底座 |
| Uni-ViGU | 以 video generator 为基础反向统一理解 | 可接 LingBot-World 和 video generator 路线 |
| Omni-Video | MLLM 产生连续 visual clues，再接 diffusion decoder | 属于 adapter / decoder 路线，和 BAGEL 的 bottleneck-free 路线对照 |
| Janus-Pro | 解耦视觉编码的统一理解与生成模型 | 图像域经典 baseline，本轮作为前置背景 |
| Transfusion | text NTP + image diffusion 的统一训练 recipe | Beyond Language Modeling 已把它作为实验底座展开 |

## 新的阅读方式

如果你是第一次来，不需要按目录从头读。请选择一个具体问题：想理解概念，就读基础文章；想排查工程瓶颈，就读训练、推理、量化或算子文章；想看论文，就直接读论文精读页。

本站后续会继续把长总览拆成专一文章，并在正文里加入外部高质量博客、论文和官方技术文档链接。
