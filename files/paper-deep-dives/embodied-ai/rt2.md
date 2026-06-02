# RT-2：把视觉语言模型接到机器人动作

Checked date: 2026-06-02  
Repro status: 论文级理解；本站未复现实验，只引用原论文图、DeepMind 博客和公开结果。

RT-2 的核心不是“机器人也用了大模型”这么简单。它把机器人动作离散化成 token，让 VLM 可以像生成文本一样生成动作，从而把 web-scale 视觉语言知识迁移到真实机器人控制。

## 论文解决的问题

机器人数据少、贵、分布窄，而互联网图文数据巨大。RT-2 想回答：能不能用 VLM 学到的语义知识，帮助机器人完成没有大量机器人示范的新任务？

它的关键接口是 action tokenization：把连续机器人动作转成离散 token，并混入模型的文本输出空间。

## 方法结构

![RT-2 overall 原论文图](../../assets/images/paper-deep-dives/embodied-ai/rt2/figure-1-overview.png){ width="920" }

<small>图源：[RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control](https://arxiv.org/abs/2307.15818)，Figure 1。原图表达 web-scale VLM 数据和机器人 trajectory 数据共同训练，模型最终输出 action token。本站用它说明：RT-2 的桥梁是把动作变成可由 VLM 解码的 token，而不是只在视觉端加机器人图片。</small>

读这张图时，先看数据混合：一边是 web-scale vision-language data，一边是 robot trajectory data。再看输出：不是自然语言答案，而是机器人动作。RT-2 的创新点就在这两端被同一个模型接口连接。

## 动作 token 的意义和代价

动作 token 让 VLM 可以复用语言建模框架，但它也把控制问题变成离散化问题。连续动作的精度、频率、坐标系、夹爪开合和执行延迟，都要被编码进 token 设计和 controller。

这意味着 RT-2 的成功不能被理解为“文本模型直接控制机器人”。更准确地说，是视觉语言表示、机器人数据、动作离散化和低层控制器共同组成了控制系统。

一个容易忽略的细节是，动作 token 进入语言模型以后，训练目标看起来像 next-token prediction，但物理后果并不像文本那样可撤销。文本 token 生成错了可以继续解释或改写，机器人动作生成错了可能已经碰到桌沿、推倒物体或让任务状态不可逆。因此 RT-2 的动作 token 设计必须和安全限幅、低层控制器、episode reset、人工干预和失败标注一起看。

这也是为什么读 RT-2 时不能只看“模型规模”和“VLM 底座”。更关键的问题是：动作被离散成多少 bin，控制频率是多少，训练数据里的动作来自什么机器人，部署时相机和坐标系是否一致，模型输出动作后是否还有 controller 做平滑、限幅和碰撞保护。

## 实验结果怎么读

![RT-2 generalization 原论文图](../../assets/images/paper-deep-dives/embodied-ai/rt2/figure-4-generalization.png){ width="860" }

<small>图源：[RT-2](https://arxiv.org/abs/2307.15818)，Figure 4。原图展示 RT-2 在 seen、unseen 和 web knowledge 相关任务上的泛化表现。本站用它说明：RT-2 的主张是 web knowledge transfer，而不是单纯提升已见任务成功率。</small>

这张图要看任务类别。若只在 seen tasks 上提升，说明机器人数据训练更好；若在需要语义知识的 unseen tasks 上提升，才更接近论文核心 claim。读 RT-2 时应把“机器人技能泛化”和“web knowledge 泛化”分开。

![RT-2 emergent examples](../../assets/images/paper-deep-dives/embodied-ai/rt2/figure-2-emergent-examples.png){ width="860" }

<small>图源：[RT-2](https://arxiv.org/abs/2307.15818)，Figure 2。原图展示模型利用语义知识执行新指令的例子。本站用它提醒：这些例子适合建立直觉，但仍要回到系统性评测和失败分析。</small>

## RT-2 的工程启发

| 启发 | 工程含义 |
| --- | --- |
| 大规模图文知识能帮机器人 | 但必须通过动作接口和真实执行验证。 |
| 动作 token 是关键桥梁 | 需要清楚离散化、频率、坐标系和 controller。 |
| 数据混合决定迁移能力 | web data 和 robot data 的比例、格式和目标都重要。 |
| 泛化要分桶看 | seen、unseen、semantic reasoning、physical skill 不能混成一个分数。 |

## 边界

RT-2 仍然依赖真实机器人数据和固定控制接口。它没有解决所有低层控制、接触稳定、长时任务规划和安全恢复问题。动作 token 也不天然适合所有机器人和所有精细控制任务。

## 外部精读

- [DeepMind RT-2 blog](https://deepmind.google/blog/rt-2-new-model-translates-vision-and-language-into-action/)
- [RT-2 paper](https://arxiv.org/abs/2307.15818)
- [RT-1 paper](https://arxiv.org/abs/2212.06817)
- [Open X-Embodiment](https://arxiv.org/abs/2310.08864)
- [Octo](https://arxiv.org/abs/2405.12213)
