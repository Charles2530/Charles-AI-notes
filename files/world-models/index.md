# 世界模型

世界模型学习环境怎样随时间和行动变化。它可以把历史压缩成内部状态，预测动作后果，估计奖励和终止，或生成供人和 agent 观察的未来视频。不同用途需要不同输出，判断模型是否有用，要看预测是否满足具体使用目的。用于控制时，需要进一步验证它是否改善动作选择。

## 一个状态模型要接上哪些接口

把相机图像、机器人关节与动作历史记为 $h_t$，内部状态为 $z_t$。最小计算链是

\[
z_t\sim q_\phi(z_t\mid h_t),\qquad
\hat z_{t+1}\sim p_\theta(z_{t+1}\mid z_t,a_t).
\]

第一步回答“根据已经看见的证据，现在可能处于什么状态”，第二步回答“如果执行这个动作，下一步会怎样”。观察 decoder 可以生成图像，reward/continuation head 可以服务价值估计，risk head 可以为候选动作打分。内部状态不必等于物理真值，但必须保留下游所需的信息。

例如杯子被手遮住后，单帧图像里看不到杯子，历史状态仍应保留它大致在哪里、是否已经抓稳。若压缩把这些信息丢掉，生成背景再清晰也无法补回抓取判断。若模型只学到视频中常见的下一帧，却不区分松手和夹紧，就无法比较这两个动作的后果。

## 预测对象与使用方式

| 路线 | 训练时预测什么 | 怎样被使用 | 需要额外验证什么 |
| --- | --- | --- | --- |
| Latent dynamics | 隐状态、奖励、继续信号 | MPC 或想象中的 actor-critic | 多步误差与任务回报 |
| Predictive representation | 被遮挡或未来片段的目标表示 | 理解、检索、动作条件预测 | 表示是否保留控制因素 |
| Generative simulation | 未来像素、视频 latent 或交互环境 | 可视化、合成数据、agent 环境 | 动作响应、记忆和物理事件 |

三者可以组合。视频 autoencoder 可以定义压缩空间，动作条件 Transformer 预测其演化，额外输出奖励与风险。也可以只预测目标表示，完全不解码像素。是否生成视频不是唯一分类依据，输入和输出的数学关系更重要。

## 按技术问题查阅

| 问题 | 正文 |
| --- | --- |
| 观测、隐藏状态、belief 与动力学有什么区别 | [世界模型基础](beginner-world-models-primer.md) |
| RSSM 的 prior/posterior 怎样支持规划与策略学习 | [RSSM、Dreamer 与规划](rssm-dreamer-and-planning.md) |
| 预测像素、masked feature 与 JEPA embedding 有何差别 | [Masked / JEPA 与潜变量预测](masked-jepa-and-latent-prediction.md) |
| 视频模型怎样接动作并持续交互 | [生成式模拟与视频](generative-simulation-and-video.md) |
| WM、VLA、WAM、VAM 的接口如何区分 | [动作条件建模](wm-wam-vam-and-action-conditioned-modeling.md) |
| 奖励怎样变成轨迹后验，latent action 怎样落到控制器 | [规划即推理与潜在动作](planning-as-inference-and-latent-actions.md) |
| 随机未来和模型未知怎样影响选择 | [不确定性与风险规划](uncertainty-and-risk-aware-planning.md) |
| 合成未来能用来训练什么 | [模拟器、反事实与合成 Rollout](simulators-counterfactuals-and-synthetic-rollouts.md) |
| 如何测动作排序、误差传播与闭环收益 | [评测与失败模式](evaluation-and-failure-modes.md) |
| 机器人与驾驶怎样接入世界模型 | [应用接口](applications-in-robotics-and-autonomous-driving.md) |
| VLA 候选动作怎样经过模型评分与执行 | [VLA 全链路案例](vla-world-model-full-chain-case.md) |
| 多相机历史、动作、损失与 rollout 怎样组织 | [从轨迹构造训练目标](action-conditioned-video-world-model-training-case.md) |
| 失败样本怎样影响下一轮数据采集 | [数据引擎](data-engines-and-self-improvement.md) |
| 数据、几何、仿真和模型应怎样联调 | [开发路线](development-roadmap-data-simulation-and-evaluation.md) |
| 真实交互、token、显存和延迟怎样分别计算 | [高效训练路线图](efficient-training-roadmap.md) |
| 哪种效率技术针对哪个瓶颈 | [效率技术覆盖矩阵](efficiency-technique-coverage-matrix.md) |
| KV、量化、kernel 与 rollout 服务怎样协作 | [Serving 案例](rollout-serving-quantization-kernel-case.md) |
| 怎样复算并解释一份模型评测 | [手工评测示例逐项复算](world-model-efficient-training-experiment-report.md) |

## 一个数值预测为何还不是一项能力

设两个动作真实回报为 $[2,1]$，模型预测为 $[1.1,1.2]$。预测的平均绝对误差只有 $0.55$，却把两个动作的顺序反了。若 planner 取预测最大值，它会选第二个动作，真实回报损失为 $1$。用于规划时，排序和选择后悔值比单独的平均预测误差更接近消费目标。

同样，视觉压缩减少 token 不自动缩短等比例的总时间；模型内想象减少真实交互，也会增加计算并继承模型偏差。本站的[小型评测数据](../assets/examples/world-model-mini-chain/episodes.jsonl)与[计算脚本](../assets/examples/world-model-mini-chain/eval_mini_chain.py)专门保留了排序失败。它们用于解释指标，没有训练真实模型，也不构成机器人实验结果。
