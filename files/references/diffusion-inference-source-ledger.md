# 扩散采样与推理加速来源台账

这份台账服务于 [扩散采样与推理加速](../diffusion/inference.md)。正文不把 sampler 写成菜单名词表，而是围绕一个问题讲：模型给出方向后，采样器怎样在有限 NFE 和端到端延迟预算下，把方向积分成稳定样本。

## 使用方式

第一类来源是基础采样链。DDPM 支撑随机反向链、多步纠错和 progressive generation 图；DDIM 支撑非马尔可夫路径、确定性采样和不重训跳步。

第二类来源是连续时间和 ODE 视角。Score SDE 支撑 reverse SDE、probability flow ODE 和“采样是数值求解”的统一解释；DPM-Solver 支撑专门利用 diffusion ODE 半线性结构的少步求解器。

第三类来源是 guidance 与工程稳定性。DPM-Solver++ 支撑 strong guidance 下普通高阶 solver 可能不稳、data prediction solver 更适合少步 guided sampling 的讨论；Guidance 正文和台账负责解释 CFG 为什么会改变方向场。

第四类来源是采样设计空间。EDM 支撑把噪声尺度、preconditioning、step schedule、solver、随机性和 NFE 作为同一个工程设计空间来写；Diffusers scheduler 文档用于核对实际接口和命名。

第五类来源是讲法参考。Lil'Log 用一条线连接 DDPM、score、SDE、sampling 和 guidance；正文借鉴它的递进顺序，但公式、图源和边界以论文与官方文档为准。

第六类来源是随机性与系数保持。DDIM 论文支撑确定/随机采样公式；中文“零推导”随机性文章用几何图解释新噪声项 \(\sigma_t z\) 为什么要替换旧预测噪声的一部分，而不是叠加到总噪声之外；CPS 论文支撑“采样时数据系数和总噪声水平必须与 scheduler 对齐”的正式边界，尤其用于解释 Flow Matching 中直接改 SDE 注入随机性可能带来的噪声伪影。

图像使用三张：DDPM Figure 14 用来讲多步逐渐成形，DDIM Figure 2 用来讲跳过时间子序列，DPM-Solver++ Figure 1 用来讲 high guidance 下少步 solver 的稳定性差异。算法截图不放入基础页，避免把读者拖进实现细节。

## 写作边界

不要把采样器写成“提质魔法”。采样器只能在已有模型方向场上减少离散化误差和推理成本，不能补训练数据、模型容量或条件接口的缺陷。

不要把少步采样写成简单跳步。每一步跨越更大噪声区间，会放大模型预测误差、方向场误差和 guidance 过冲。

不要只比较 sampler 名字。必须同时报告 prediction type、噪声日程、timestep schedule、NFE、CFG scale、是否 thresholding、是否随机、分辨率和硬件延迟。

不要把 NFE 等同于端到端延迟。文本编码器、VAE、ControlNet / adapter、CFG 分支、batching、显存峰值和后处理都会改变真实成本。

不要把视频采样只按单帧质量判断。视频还要看 temporal consistency、motion bucket、首帧保真、动作敏感性和长时 rollout。

## 来源

| 来源 | 类型 | 本站使用方式 | 关联问题 |
| --- | --- | --- | --- |
| [DDPM](https://arxiv.org/abs/2006.11239) | 论文 / 图源 | 使用 Figure 14 解释随机反向链和 progressive generation。 | 多步反向链 |
| [DDIM](https://arxiv.org/abs/2010.02502) | 论文 / 图源 | 使用 Figure 2 解释 accelerated generation、时间子序列和确定性路径。 | 跳步采样 |
| [Score SDE](https://arxiv.org/abs/2011.13456) | 论文 | 支撑 reverse SDE、probability flow ODE 和数值求解器视角。 | 连续时间 |
| [DPM-Solver](https://arxiv.org/abs/2206.00927) | 论文 | 支撑利用 diffusion ODE 结构的高阶少步 solver。 | ODE solver |
| [DPM-Solver++](https://arxiv.org/abs/2211.01095) | 论文 / 图源 | 使用 Figure 1 解释 strong guidance 下少步稳定性和 data prediction solver。 | guided sampling |
| [EDM](https://arxiv.org/abs/2206.00364) | 论文 | 支撑噪声尺度、preconditioning、solver、stochasticity 和 NFE 的设计空间。 | 采样设计 |
| [Diffusers schedulers overview](https://huggingface.co/docs/diffusers/en/api/schedulers/overview) | 官方文档 | 核对 scheduler / sampler 在工程库中的接口和命名边界。 | 工程接口 |
| [Lil'Log: What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) | 技术博客 | 借鉴从 DDPM 到 score / SDE / guidance / sampling 的递进讲法。 | 讲法参考 |
| [零推导理解 Diffusion 和 Flow Matching（二）：随机性](https://zhuanlan.zhihu.com/p/1947597090311090928) | 中文技术博客 | 用 DDIM 完整公式和几何图说明随机项替换旧噪声份额、保持下一步总噪声幅度。 | 随机采样 |
| [系数保持采样：Flow Matching 随机性注入的正确打开方式](https://zhuanlan.zhihu.com/p/1948388095151026330) | 中文技术博客 | 作为 Flow Matching 随机性注入与系数保持问题的中文阅读入口。 | Flow 随机性 |
| [Coefficients-Preserving Sampling for Reinforcement Learning with Flow Matching](https://arxiv.org/abs/2509.05952) | 论文 | 正式支撑系数保持采样定义、Flow-SDE 随机性伪影和 CPS 方案。 | 系数保持 / RL 采样 |

## 后续写作提醒

写 sampler 时，优先围绕失败机制组织：高噪声阶段大结构漂、中噪声阶段几何不稳、低噪声阶段细节软、强 CFG 过冲、ControlNet 条件变弱、视频闪烁或动作不敏感、NFE 下降但端到端延迟没有线性下降。每个失败都要能映射到模型输出坐标、时间表、更新规则、条件分支或部署成本之一。

图片优先来自论文、官方项目页和高质量技术博客。图注必须说明：原图表达什么，本站正文要读出什么机制，它不能证明什么边界。
