# 术语表

这一页把当前知识库中反复出现的术语做一个统一整理。  
目标不是给出百科定义，而是给出“在这套文档语境里，它主要指什么、与哪些概念最容易混淆”。

## A

### `Activation`

神经网络中间层输出。  
在量化场景里，activation 往往比权重更难压低比特，因为动态范围更不稳定，也更容易出现 outlier。

### `AWQ`

Activation-aware Weight Quantization。  
**核心思想是**：量化误差不能只看权重本身，还要看激活如何放大误差。

## B

### `BYOL`

Bootstrap Your Own Latent。  
一种不显式依赖负样本的自监督表征学习方法，依靠在线网络与目标网络的不对称结构避免塌缩。

## C

### `CFG`

Classifier-Free Guidance。  
扩散模型中的条件引导技巧，常写为无条件分支和条件分支的线性组合，用于提升条件对齐强度。

### `CLIP`

Contrastive Language-Image Pre-training。  
现代图文对齐和多模态检索的代表方法之一，本质上是跨模态对比学习。

### `Continuous Batching`

推理服务中的连续批处理。  
与静态批处理不同，允许请求在 decode 过程中动态加入与退出，以提高 GPU 利用率并降低排队时间。

## D

### `DDIM`

Denoising Diffusion Implicit Models。  
在不重训模型的前提下实现更少步采样的经典方法，可被理解为扩散从离散链走向 ODE 视角的重要桥梁。

### `DDPM`

Denoising Diffusion Probabilistic Models。  
现代扩散模型主流路线的起点，核心是逐步加噪和逐步去噪。

### `DMD`

Distribution Matching Distillation。  
一步扩散蒸馏代表路线之一，强调从“模仿轨迹”转向“对齐分布”。

### `DMD2`

对 DMD 的系统改进版本。  
重点解决一步蒸馏中的模糊、训练不稳和细节不足问题。

### `DPM-Solver`

针对 diffusion ODE 设计的专用高阶求解器。  
目标是在极少步条件下仍维持较高采样质量。

### `DPO`

Direct Preference Optimization。  
一种常用偏好对齐方法，直接利用偏好对优化策略分布，而不显式进行在线强化学习。

## E

### `ELBO`

Evidence Lower Bound。  
变分推断中的经典目标，在扩散模型、世界模型等地方都常出现。

### `Embedding`

向量表示。  
在检索和对比学习中，embedding 空间的几何结构往往比单步分类准确率更重要。

## G

### `GPTQ`

一种典型后训练权重量化方法。  
通过二阶近似减少量化引入的输出误差。

## H

### `Heun`

一种二阶数值求解器。  
在扩散采样中，常可理解为带校正的一阶预测方法。

### `Hard Negative`

困难负样本。  
在对比学习和检索中，指那些“看起来很像，但关键语义不同”的负样本。

## I

### `InfoNCE`

对比学习的经典目标函数。  
它要求模型在一组候选中，把正确匹配项排到最前。

## K

### `KV Cache`

Transformer 推理时为历史 token 缓存的 key/value 张量。  
在长上下文服务里，它往往是显存和吞吐瓶颈的核心来源之一。

## L

### `LCM`

Latent Consistency Models。  
少步扩散采样路线之一，强调在 latent 空间中学习更少步的一致映射。

### `LoRA`

Low-Rank Adaptation。  
通过低秩增量 \(BA\) 替代全量参数更新的参数高效微调方法。

## M

### `MPC`

Model Predictive Control。  
一种经典控制与规划方法，在具身智能中常作为高层策略与低层执行之间的重要桥梁。

### `MoCo`

Momentum Contrast。  
通过动量编码器和队列维护大量负样本的对比学习方法。

## P

### `Phased DMD`

将 DMD 的分布匹配思想做阶段化拆分的方法。  
目标是在极少步或一步生成时更稳地保住结构与细节。

### `Prefix Cache`

把重复出现的 prompt 前缀缓存起来，避免反复 prefill 的系统技巧。  
在模板化 agent 或企业问答系统里价值很高。

### `PTQ`

Post-Training Quantization。  
模型训练完后再进行量化，不重新做大规模训练。

## Q

### `QAT`

Quantization-Aware Training。  
在训练时显式模拟量化误差，以提升低比特部署时的稳定性。

### `QLoRA`

在量化底座上训练 LoRA 增量的一种高效微调方法。  
重点是用更低显存完成大模型适配。

## R

### `Rectified Diffusion`

少步扩散与整流路线中的代表性概念。  
强调让生成路径更适合粗粒度离散，而不一定只追求几何上的“绝对直”。

### `Rectified Flow`

一类希望学习更直生成轨迹的方法家族。  
目标是让一步或极少步生成更自然。

### `Recall@K`

检索系统常用指标。  
表示正确结果是否出现在前 \(K\) 个候选中。

## R-S

### `RSSM`

Recurrent State-Space Model。  
世界模型中的经典结构，结合确定性记忆与随机潜变量。

### `Sim2Real`

Simulation-to-Reality。  
指策略从仿真迁移到真实世界时遇到的分布偏差问题，以及相应的缓解方法。

### `SimCLR`

对比学习代表方法之一。  
强调大 batch、强增强和 projection head。

### `SmoothQuant`

一种缓解激活离群值问题的量化方法。  
核心是把部分激活尺度转移到权重中。

### `Speculative Decoding`

投机解码。  
由一个较便宜的草稿模型先提出候选 token，再由较强模型验证，以加速 decode。

### `SFT`

Supervised Fine-Tuning。  
监督微调。  
在大模型训练链路里，常负责把底座能力塑形成更符合任务要求的输出行为。

## V

### `VLA`

Vision-Language-Action。  
给定视觉观察和语言指令，模型直接输出动作。

### `VLM`

Vision-Language Model。  
给定视觉和文本输入，模型输出回答、标签、定位、工具调用等结果。

## W

### `World Model`

学习环境潜在动态并支持内部模拟的模型。  
在智能体、机器人和自动驾驶中常用于前瞻预测和规划。
