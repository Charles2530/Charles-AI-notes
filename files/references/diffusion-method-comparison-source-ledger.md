# 扩散路线选择来源台账

这份台账服务于 `files/diffusion/comparison-table.md`。正文不是做论文榜单，而是把扩散方法按工程改动层级拆开：推理侧采样器、少步学生、分布匹配蒸馏、路径与速度场重写。

## 资料取舍

| 主题 | 主要来源 | 在正文中的用法 |
| --- | --- | --- |
| DDPM 基线 | [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) | 作为多步去噪教师和原始采样成本的基线。 |
| 不重训采样路径 | [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) | 支撑“DDIM 是换采样路径，不是提升模型能力”的边界。 |
| Score / SDE / ODE 视角 | [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456) | 支撑把采样器理解为 SDE/ODE 离散化的写法。 |
| 高阶采样器 | [DPM-Solver](https://arxiv.org/abs/2206.00927)、[DPM-Solver++](https://arxiv.org/abs/2211.01095) | 支撑 DPM-Solver 系列属于推理侧高阶求解器，并强调 guided sampling。 |
| 少步蒸馏 | [Progressive Distillation for Fast Sampling of Diffusion Models](https://arxiv.org/abs/2202.00512) | 支撑“逐级减半教师轨迹”的描述。 |
| 一致性路线 | [Consistency Models](https://arxiv.org/abs/2303.01469)、[Latent Consistency Models](https://arxiv.org/abs/2310.04378) | 支撑一到几步生成适合交互预览，但需要训练或蒸馏。 |
| 分布匹配蒸馏 | [Distribution Matching Distillation](https://arxiv.org/abs/2311.18828)、[DMD2](https://arxiv.org/abs/2405.14867)、[Phased DMD](https://arxiv.org/abs/2510.27684) | 支撑一步/少步生成不只是 solver 问题，而是分布匹配和训练稳定性问题。 |
| Flow 与 Rectified 路线 | [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)、[Rectified Flow](https://arxiv.org/abs/2209.03003)、[Rectified Diffusion](https://arxiv.org/abs/2410.07303) | 支撑“路径与速度场重写”这一层级。 |
| 条件引导 | [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) | 支撑正文把 CFG、强条件控制和采样稳定性纳入验收。 |

## 写作边界

正文避免把这些方法按“先进程度”排序。DDIM、Euler、Heun、DPM-Solver++ 属于推理侧替换；Progressive Distillation、Consistency、LCM 属于学生模型；DMD 系列属于分布匹配蒸馏；Flow Matching 和 Rectified 路线属于路径或速度场重写。

正文也不把“采样步数”当成唯一指标。真实评估需要同时看图像质量、prompt 对齐、强 CFG、控制条件、局部裁剪、多样性、P50/P95 延迟、显存峰值和失败样本尾部。视频或多模态系统还需要单独检查时序一致性、条件编码器、解码器和 token 数带来的系统成本。

## 核对口径

正文中的方法归类按“需要改动哪里”来核对，而不是按论文发布日期核对。DDIM 的主要边界是复用已有训练模型并改变采样过程；DPM-Solver 系列的主要边界是把扩散采样写成更适合高阶近似的 ODE 求解问题；Consistency / LCM 的主要边界是通过训练得到少步映射；DMD 系列的主要边界是让少步学生输出分布贴近教师或数据分布；Flow / Rectified 路线的主要边界是重设路径或速度场。

文中涉及工程建议时，只保留能从上述边界推出的判断。例如“不能重训时先看采样器”来自 DDIM 和 DPM-Solver 一类方法的低切换成本；“一步生成要看多样性和局部裁剪”来自分布匹配蒸馏可能把长轨迹教师的逐步纠错压缩到单次前向；“视频系统不能只看步数”来自视频 latent、条件编码器、解码器和时空注意力共同决定端到端延迟。

没有把具体模型排行榜、社区采样器偏好或某个 UI 工具的默认参数写进正文。那些信息会随版本快速变化，适合放在单独的工程配置页，而不适合放在这篇路线选择页。

## 未采用的写法

- 没有把所有方法压进一张超长排行榜，因为不同方法改动层级不同。
- 没有把 DMD 系列描述成“DDIM 的一步版本”，因为它的核心是分布匹配蒸馏。
- 没有把 Rectified / Flow 路线描述成旧模型的直接替代品，因为它们通常牵涉训练目标、路径、时间采样和条件接口。
- 没有只引用二手博客。正文的事实边界以论文和项目原文为准，外部长文只作为表达风格参考。
