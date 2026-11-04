# 跨专题问题与技术判断

基础页解释一个机制，专题页连接一组方法。这里沿一个具体问题跨越两者，例如同样是加速，减少模型计算、减少旧观测排队和改变动作块长度，会对结果产生不同影响。

## 从一个可观察现象进入

| 你正在想的问题 | 建议先读 |
| --- | --- |
| 具身模型推理更快，为什么动作仍可能跟不上环境变化 | [具身推理加速，先看动作执行时观测有多旧](embodied-inference-observation-age-1009-2059.md) |
| “世界模型”究竟该怎样定义，视频生成为什么不必然等于世界模型 | [世界模型是什么：从预测画面到承受行动后果](what-is-a-world-model.md) |
| 训练与推理优化为什么常常卡在数据搬运和状态生命周期 | [高效 AI 的状态账本：少搬状态，比少算一次更重要](efficient-ai-is-state-movement.md) |
| 扩散模型减少采样步数后，误差和瓶颈去了哪里 | [扩散模型加速：每删一步，都要重新安排误差](diffusion-speedup-redistributes-error.md) |
| 世界模型究竟应该预测像素、latent、value 还是别的状态 | [世界模型该预测什么：答案取决于谁消费未来](world-model-future-consumers.md) |
| 世界模型的 latent 怎样进入动作生成与规划 | [从预测未来到利用未来：Latent World Model 怎样进入规划闭环](latent-world-models-for-planning.md) |
| 具身智能怎样把场景、任务、动作、执行和恢复接成可靠闭环 | [具身智能：一条可纠错的状态链](embodied-ai-correctable-state-chain.md) |
| 为什么前沿模型的真实能力已经超出单个 checkpoint | [技术报告读后：前沿模型正在变成受约束的闭环](model-capability-is-a-constrained-loop.md) |
| 端侧 AI、XR 芯片和量化部署有什么关系 | [XR2 Gen 2：端侧 AI 与量化部署](qualcomm-xr2-gen2.md) |
| 自动驾驶里的世界模型如何从生成视频走向规划闭环 | [自动驾驶世界模型：从渲染、仿真到规划闭环](autonomous-driving-world-models.md) |
| 自动驾驶世界模型为什么需要重建作为可审计状态接口 | [从生成视频到维护世界：重建为什么是自动驾驶世界模型的状态接口](reconstruction-as-world-model-state-interface-0724-1059.md) |
| 世界模型如何从模型论文变成可复用工程栈 | [世界模型 Infra：从模型谱系到物理 AI 工程栈](world-model-infrastructure.md) |
| 世界模型从视频模拟走向工程闭环卡在哪里 | [世界模型现状：从视频模拟到工程闭环](world-models-status-and-engineering-loop.md) |
| WAM 为什么会和 3D 视觉合流 | [WAM 与 3D 视觉：世界模型从视频想象走向物理闭环](wam-3dv-world-models.md) |
| World Model 的能力边界如何推出机器人数据飞轮 | [世界模型驱动的机器人数据飞轮：能持续生产经验的系统，才是真正的基础设施](robot-data-flywheel-world-models.md) |
| 具身智能为什么总绕不开 VLA、数据工厂和真实闭环 | [具身智能现状：VLA、数据工厂与真实闭环](embodied-ai-status-and-real-world-loop.md) |
| 为什么“更多机器人数据”不等于更强具身智能 | [具身智能的数据鸿沟：先对齐，再扩规模](embodied-ai-data-gap-alignment-and-scaling-0731-1219.md) |
| 读 Fast-FoundationStereo 前要补哪些视觉和推理基础 | [读懂 Fast-FoundationStereo](fast-foundation-stereo-prerequisite-map.md) |


## 读图时先追踪输入与消费者

预测得到的未来可能交给渲染器、奖励函数、规划器或人类查看。消费者不同，需要保留的细节也不同。读每张方法图时，把输入、内部状态、输出和后续使用位置连起来，再去判断论文的指标是否覆盖了这条路径。

```mermaid
flowchart LR
    I["实际输入与条件"] --> M["所讨论的机制"]
    M --> S["输出或保存的状态"]
    S --> C["后续消费者"]
    C --> R["任务结果与资源测量"]
```

例如视频更清晰可以支持生成质量判断，但若消费者是机器人规划器，还要检查动作条件、坐标、时间与闭环任务。[世界模型](../world-models/index.md)解释这些接口，[具身系统](../embodied-ai/index.md)继续追踪实际执行。

## 事实、算例与判断怎样区分

文章中的原图和来源数字需要结合原文设置阅读；手算例子负责解释公式与接口，不充当模型性能；提出的技术判断则需要能被实验削弱或推翻。对某个结论有疑问时，先看它属于哪一种，再返回来源或构造最小对照。

系统细节可以沿[训练](../training/index.md)、[推理](../inference/index.md)、[量化](../quantization/index.md)和[算子](../operators/index.md)继续查。这里不规定阅读周次，也不把论文选题作为入门目标。
