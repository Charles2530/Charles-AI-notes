# 视觉语言动作模型

让机器人把红杯拿起来，模型输出一个动作数组还不够。系统要确认它指的是哪只杯子，动作是在相机坐标还是底座坐标中表达，数值对应位置、速度还是增量，控制器执行后杯子是否真的离开了桌面。VLA 主要负责根据观测提出动作，时间、单位、约束和反馈则由部署链路一起承担。

## 先问策略交出来的是什么

统一写法为

\[
A_t=(a_t,\ldots,a_{t+H-1})\sim\pi_\theta(\cdot\mid o_{\le t},q_{\le t},l).
\]

这里动作块恰有 \(H\) 步，\(q\) 表示本体状态，\(l\) 表示任务。输出可以是离散 action token，也可以是连续关节目标、末端位姿或技能。动作表示必须同时指定单位、坐标、控制周期和解码方式。

例如同一个数 0.01，如果表示每 100 毫秒移动 0.01 米，理想平均速度为 0.1 米每秒；如果表示速度，则是 0.01 米每秒。模型参数没有变化，接口解释已经相差十倍。比较架构以前，先把这种接口差异解释清楚。很多看起来像模型能力的差别，最后落在单位、坐标或控制周期上。

## 训练数据和真实执行不是同一条轨迹

```mermaid
%% vla-training-execution
flowchart LR
  Data[示范 episode] --> Align[时间与动作接口对齐]
  Align --> Loss[条件动作分布训练]
  Observe[最新观测] --> Policy[策略推理]
  Policy --> Adapter[解码与坐标变换]
  Adapter --> Control[约束检查与控制]
  Control --> Environment[机器人和环境]
  Environment --> Observe
```

训练数据通常来自专家或已有策略，部署时观测来自模型自己的动作后果。两者的闭环状态分布不同，解释了为什么离线动作误差很低仍可能中途失败。重观测、动作块取消和恢复数据都属于这个问题的一部分。

| 需要理解的机制 | 正文 |
| --- | --- |
| 数组如何变成有单位的命令 | [动作表示与控制接口](action-representation-and-interfaces.md) |
| 示范如何形成概率训练目标 | [数据与策略学习](data-and-policy-learning.md) |
| 一段动作如何与实时反馈结合 | [动作块、层级策略与技能](action-chunking-hierarchical-policies-and-latent-skills.md) |
| 偏离后怎样取消旧计划并恢复 | [闭环恢复](closed-loop-recovery-and-failure-analysis.md) |
| 成功率与训练数据如何互相约束 | [评测与数据引擎](benchmarks-and-data-engines.md) |
| 延迟、命令有效期与约束如何执行 | [部署与安全](deployment-and-safety.md) |

假设第一段动作让夹爪比示范偏右 1 厘米，下一帧里杯子相对夹爪的位置已经改变。闭环策略读取新观察后应重新决定动作，不能仅继续比较原示范时间索引下的标签。若数据中没有偏移后的状态，模型可能不知道怎样修正，这就是离线动作误差与闭环成功之间的缺口。

动作表示回答输出的含义，chunk 决定预测和执行多长，恢复机制处理状态已经偏离后的重新开始。显式世界模型还可以预测候选动作的后果，但建立基本反馈闭环并不以它为前提。

## 与世界模型组合时，新增的证据是什么

策略学习 \(\pi_\theta(a\mid o,l)\)，世界模型学习 \(p(z_{t+1}\mid z_t,a_t)\)。前者提出动作，后者估计后果。若世界模型用于选动作，需要证明它能正确区分候选，而不只是生成流畅视频。只记录实际执行动作的轨迹，也不能提供所有未执行候选的真实未来标签。

系统分工见[VLA、WAM 与世界模型](../embodied-ai/vla-wam-world-model-system-map.md)。推理优化可进一步看[投机推理与闭环验证](../inference/vla-speculative-decoding-0928-1144.md)，但加速结果应与任务成功、取消延迟和观测年龄一起解释。

[RT-2](https://arxiv.org/abs/2307.15818)、[OpenVLA](https://arxiv.org/abs/2406.09246)、[ACT](https://arxiv.org/abs/2304.13705)和 [Diffusion Policy](https://arxiv.org/abs/2303.04137)分别提供不同接口与训练方式的原始材料，不代表一种统一的动作协议。

## 来源与改编说明

- **原始材料**：[RT-2](https://arxiv.org/abs/2307.15818)、[OpenVLA](https://arxiv.org/abs/2406.09246)、[ACT](https://arxiv.org/abs/2304.13705)和[Diffusion Policy](https://arxiv.org/abs/2303.04137)展示不同的输入、动作和训练接口，不能合并为一个统一协议。
- **实践入口**：[LeRobot 文档](https://huggingface.co/docs/lerobot/en/index)和[SmolVLA](https://huggingface.co/blog/smolvla)适合做数据、训练和部署案例；版本和 benchmark 条件必须保留。
- **自测**：把“输出一个动作块”拆成坐标系、单位、控制周期、取消时机和反馈更新五项，说明离线动作误差下降后还缺哪一项闭环证据。再选一个偏移后的状态，写出策略会读取哪条新观测、丢弃哪段旧动作。
