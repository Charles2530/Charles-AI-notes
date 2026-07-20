# 生成式模拟与视频世界模型：从好看视频到可交互未来

视频世界模型很容易被误读成“更强的视频生成”。在世界模型语境里，它真正要回答的是：给定历史、动作、目标和场景约束，未来是否会以可验证、可交互、可用于决策的方式演化。

视频生成模型要补上动作因果、交互约束和评测接口，才能变成服务规划和数据回流的世界模型。

## 普通视频生成缺少动作因果

普通视频生成常建模：

$$
p_\theta(v_{t+1:t+H}\mid v_{\le t}, c)
$$

这里 \(v_{\le t}\) 是历史视频帧或视频 latent，\(c\) 是文本、首帧、风格或场景条件，模型输出未来 \(H\) 步视频。这能学到外观、光照、构图和一般运动规律，但还不能证明它理解动作后果。

视频世界模型至少要把动作放进条件里：

$$
p_\theta(v_{t+1:t+H}, u_{t+1:t+H}
\mid v_{\le t}, a_{t:t+H-1}, c)
$$

这里 \(a_{t:t+H-1}\) 是未来动作，\(u\) 可以是风险、不确定性、可达性、碰撞或成功概率。公式里的关键变化是：固定同一历史时，换动作应该得到不同未来。如果动作只写在 prompt 里但不改变模型内部生成路径，模型很可能仍然靠视频惯性生成平均未来。

## 为什么要继承视频生成底座

从零训练一个世界模型很贵。大规模视频生成底座已经学到物体外观、短中程运动、场景布局、相机变化和视觉连续性。很多视频世界模型因此选择先继承视频底座，再继续训练动作条件、长时一致性、因果 rollout 和低延迟生成。

![LingBot-World training pipeline](../assets/images/paper-deep-dives/world-models/lingbot-world/figure-4-training-pipeline.png){ width="920" }

<small>图源：[Advancing Open-source World Models](https://arxiv.org/abs/2601.20540)，Figure 4。原图展示 LingBot-World 从 foundation video generator 出发，经过 pre-training、middle-training 和 post-training，逐步加入通用视频先验、交互数据、长时动态和因果低延迟能力。读图重点：继承视频底座只是起点，动作数据、因果结构和评测门禁才是世界模型分水岭。</small>

这类路线可以理解成三阶段。第一阶段学“世界看起来怎样”，主要依靠大规模视频和图文数据。第二阶段学“动作如何改变未来”，引入游戏、机器人、相机控制或交互轨迹。第三阶段学“怎样实时向前滚”，包括因果 attention、KV cache、少步生成和流式解码。

## 动作必须进入模型内部

动作可以是键盘输入、相机轨迹、机器人关节动作、末端位姿、离散技能、语言子目标或工具调用。关键不是动作名字，而是它是否影响去噪路径、latent dynamics 或预测头。

![LingBot-World system pipeline](../assets/images/paper-deep-dives/world-models/lingbot-world/figure-5-system-pipeline.png){ width="920" }

<small>图源：[Advancing Open-source World Models](https://arxiv.org/abs/2601.20540)，Figure 5。原图展示图像/视频、噪声 latent、文本和用户动作共同进入生成系统，并在 DiT block 中通过 action embedding、adaptive normalization、self-attention 和 text cross-attention 影响未来视频。读图重点：动作条件要成为模型内部变量，而不是生成后贴上的解释标签。</small>

最小反事实测试很简单：

```text
固定同一历史画面
  -> 给三组不同动作
  -> 检查未来视频、latent、风险和成功排序是否合理分叉
```

如果三组未来几乎一样，模型可能只是普通视频续写器。如果未来分叉合理，但 planner 不能用它选动作，它还只是动作条件预测器。只有接入 planner 或 policy 后成功率、恢复率、风险控制或数据回流变好，才进入世界模型证据层。

## 交互 rollout 需要因果化

普通视频生成可以整段视频一起生成；交互式世界模型必须一步步向前，持续接收用户或 agent 的动作，并保持环境记忆。这会带来 train-test gap：训练时模型可能看完整片段，推理时只能看过去和当前 chunk。

![LingBot-World causal adaptation](../assets/images/paper-deep-dives/world-models/lingbot-world/figure-6-causal-adaptation.png){ width="790" }

<small>图源：[Advancing Open-source World Models](https://arxiv.org/abs/2601.20540)，Figure 6。原图展示 causal generator adaptation 和 discriminator architecture。读图重点：block causal attention、KV cache、长时 discriminator 和少步生成共同解决“只能看过去还要持续交互”的问题。</small>

在线系统还要分清频率。高频控制通常需要传统控制器或轻量 policy，不能等待昂贵视频采样；中频规划可以用 latent rollout、低分辨率未来或候选风险排序；低频分析和人审才适合高清视频反事实回放。

这就是为什么视频解码不一定在控制主路径上。主路径更需要的是哪个动作更安全、更可达、更可能完成任务。

## 生成式模拟的三种用法

第一种是在线规划。机器人可以对多条候选轨迹做短 horizon rollout，筛掉会碰撞、滑落或遮挡关键物体的动作；自动驾驶可以比较刹车、让行、变道的未来风险。

第二种是反事实与失败回放。真实危险事件稀有、采集成本高、复现困难。视频世界模型可以回答：如果当时慢一点、换一条路径、先移动另一个物体，会不会避免失败。

第三种是数据引擎。模型可以生成相邻反事实、扩增长尾场景、辅助人工标注和错误归因。但合成数据不能直接信任，必须经过真实数据校准、failure replay 和 closed-loop evaluation。

## 评测不要只看视频质量

视频世界模型至少要过六层评测。视觉层看清晰度、连续性和对象身份；动力学层看接触、遮挡、碰撞、速度和空间结构；动作层看固定历史换动作是否分叉；任务层看接入 planner 或 policy 后成功率是否提升；风险层看 near-miss、碰撞和不确定性是否被低估；系统层看 latency、显存、吞吐和 cost per success。

如果方法声称是 VAM/WAM/视频世界模型，还应报告 action sensitivity、candidate ranking agreement、temporal consistency、risk calibration 和 cost per success。只报告 FVD、视频 demo 或少数交互样例，不足以证明它能服务规划。

## 常见失效模式

| 失败 | 表现 | 为什么危险 |
| --- | --- | --- |
| 视觉上对，动力学上错 | 画面自然但接触、摩擦、碰撞不对 | planner 会选错动作 |
| 动作被历史惯性淹没 | 换动作后未来差异很小 | 反事实和规划失效 |
| 长时重造世界 | 地标、房间结构、物体身份漂移 | 数据回流会污染策略 |
| 平均化未来 | 稀有危险分支被抹掉 | 风险被低估 |
| 采样太慢 | 每个候选动作都要多步扩散 | 无法在线规划 |
| 自生成偏差回灌 | 合成数据越来越像模型自己 | 策略适应模拟器而非现实 |

最直接的判断是：如果一个视频世界模型不能让系统更好地选择动作、发现风险或构造有价值的反事实，那它仍然只是视频生成模型；只有当它改变规划、训练和数据回流决策时，才真正进入世界模型工程范畴。

## 外部材料

- [LingBot-World / Advancing Open-source World Models](https://arxiv.org/abs/2601.20540)：适合看视频底座如何继续训练成交互式世界模型。
- [CausVid: From Slow Bidirectional to Fast Autoregressive Video Diffusion Models](https://arxiv.org/abs/2412.07772)：理解为什么因果化和少步生成对交互 rollout 很关键。
- [Genie 3](https://deepmind.google/models/genie/)：适合看实时、可交互世界模型的前沿展示。
- [NVIDIA Cosmos](https://www.nvidia.com/en-us/ai/cosmos/)：适合看 physical AI 数据生成和世界基础模型平台。
- [Towards Video World Models](https://www.xunhuang.me/blogs/world_model.html)：适合作为视频世界模型路线综述的外部参考。

- [World Models 论文](https://arxiv.org/abs/1803.10122)。
- [DeepMind Genie 2](https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/)。
- [Meta V-JEPA 2](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/)。
