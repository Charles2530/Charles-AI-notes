# 论文专题写作与验收规范

这页只维护论文专题的写作标准。专题导航负责分组；这里负责回答“怎样才算一篇 deep dive 写扎实了”。

## 推荐结构

每篇专题讲解尽量保持固定结构，方便不同主题之间横向比较。

| 模块 | 需要回答的问题 |
| --- | --- |
| 论文信息 | 标题、链接、代码、关键词和适合读者 |
| 论文位置 | 它在方向谱系里解决哪一段问题 |
| 核心问题 | 旧方法为什么不够，论文要突破什么瓶颈 |
| 方法结构 | 数据、模型、训练目标、推理路径如何组织 |
| 实验结论 | 哪些结果真正支撑了论文主张 |
| 消融诊断 | 哪个模块、数据、scale 或系统条件被拿掉后结果怎样变 |
| 局限风险 | 哪些结论不能过度外推 |
| 项目启发 | 如果落到工程系统里，哪些设计最值得借鉴 |
| 阅读结论 | 成熟度、可复用机制、不能外推项和下一步证据 |

专题页不追求逐字复述论文，而是优先解释“为什么这样设计”。数据构造、训练阶段、推理约束和评测口径这类会改变工程判断的部分，应写得比普通摘要更具体。

## 硬约束

新增或重写论文页时必须满足三条底线。

1. 关键 claim 必须能接回原论文图表、实验 setting 或官方材料，不能只写摘要复述。
2. 复杂图必须在正文里讲清“图源、原图表达什么、支撑哪条 claim、不能证明什么”；标题要围绕具体图意命名，避免保留模板化占位词。
3. 项目启发只能吸收证据能支撑的部分；demo、benchmark、system throughput、closed-loop 和本站推断不能混写。

最终页面不要保留“二轮补强：方法和实验抓手”这类临时补丁块。相关内容应融入方法、实验、诊断或局限段落。

## 方法和实验深度

为了避免专题页写成摘要库，方法部分和实验部分单独按下面标准验收。

| 部分 | 必须讲清 | 合格形态 | 不合格形态 |
| --- | --- | --- | --- |
| 方法接口 | 输入是什么、输出是什么、哪些模块改变信息流 | 用一段文字或小流程图把 `data -> model -> loss -> inference` 串起来 | 只列模块名 |
| 公式/目标 | 变量、参数、条件、优化对象分别是什么 | 至少解释关键 loss、采样公式或动作/状态接口里的每个槽位 | 只贴公式名或把公式当装饰 |
| 训练/推理链路 | 哪些阶段训练，哪些阶段冻结，部署时走哪条路径 | 写出 pretrain / finetune / rollout / verification 的顺序和成本点 | 只写“经过训练后效果更好” |
| 实验 setting | 数据、硬件、任务、指标、baseline、是否 closed-loop | 每个主结果附近交代口径，复杂表格配 `读数边界` | 只说“在多个 benchmark 上表现最好” |
| 主结果 | 最突出的数字或趋势支撑了哪个 claim | 图表后立刻说明“这能证明什么、不能证明什么” | 堆表但不解释 |
| 消融/诊断 | 哪个模块被拿掉，结果怎样变，说明什么 | 至少保留一个关键 ablation、scaling、acceptance、failure 或 qualitative diagnostic | 完全跳过消融 |

一个读完合格专题页的读者，应该能复述原论文约 90% 的重要信息：问题是什么、核心机制为什么这样设计、训练和推理怎么跑、实验在什么条件下成立、最强证据和最大短板在哪里。页面可以有趣，但有趣必须服务脉络，不能用轻快语气掩盖证据空缺。

## 阅读结论收口

每篇 deep dive 末尾用一个简短的“阅读结论”正文段落收束即可，不要再加提示框。这个收口不重复摘要，只回答四件事。

| 收口问题 | 最小写法 | 不要写成 |
| --- | --- | --- |
| 成熟度 | 经典已验证 / 工程常用 / 前沿待复现 / 官方展示 / 本站推断 | “很有潜力” |
| 可复用机制 | 明确对应数据、训练、推理、显存、验证或部署哪项成本 | “效果很好” |
| 不可外推项 | 说明哪些 benchmark、硬件、数据或 closed-loop 条件没有覆盖 | “全面证明” |
| 下一步证据 | 需要补 ablation、closed-loop、system trace、质量回归或独立复现 | “值得关注” |

## 图表读法

论文图要回答：模块做什么、支撑哪条 claim、不能证明什么。图表不是装饰，而是 claim 的证据节点。图下可以写短 caption，真正的读法要进正文；如果一张图复杂，就用一个有信息量的小标题，例如“树形验证要先分清 draft 和 target”，而不是保留通用模板词。

![V-JEPA training procedure 原论文图](../assets/images/paper-deep-dives/world-models/v-jepa/figure-3-training-procedure.png){ width="920" }

<small>图源：[V-JEPA: Latent Video Prediction for Visual Representation Learning](https://arxiv.org/abs/2404.08471)，Figure 3。原论文图意：context encoder 只处理 masked video 中未遮挡 token；predictor 接收 context output 和 mask tokens，预测 target encoder 对完整视频产生的 masked token representations。</small>

**先分清两条 encoder 路径。**
左侧蓝色 token 是可见上下文，经 context encoder 得到 \(N\times d\) 表示；predictor 把 context output 和红色 mask tokens 拼接，预测被遮挡位置的 representation。右侧 target encoder 看完整视频，但 loss 只作用在 masked target representations 上，并通过 stop-gradient 阻止 teacher 分支被 predictor 反向更新。这张图支撑的是 latent prediction，不是像素重建，也不能单独证明闭环规划能力。

![DeepSeek-V3 architecture 原论文图](../assets/images/paper-deep-dives/technical-reports/deepseek-v3/figure-2-architecture.png){ width="920" }

<small>图源：[DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)，Figure 2。原论文图意：展示 MLA、DeepSeekMoE、shared/routed experts 与 Multi-Token Prediction 在整体架构中的位置。</small>

**架构图要按成本链路读。**
先看 token 在主干里的路径，再看哪些模块改变显存、通信和训练目标。MLA 影响 KV cache 与长上下文成本，MoE 决定每个 token 激活哪些专家，MTP 是额外训练信号而不是推理时必须多吐 token。读技术报告时不要把“结构存在”直接等同于“端到端更快”，还要回到吞吐、负载均衡、FP8、并行策略和 serving 约束。

![EAGLE-3 inference pipeline 原论文图](../assets/images/paper-deep-dives/inference/eagle-3/figure-5-inference-pipeline.png){ .atlas-figure-page width="560" }

<small>图源：[EAGLE-3](https://arxiv.org/abs/2503.01840)，Figure 5。原论文图意：展示 EAGLE-3 在推理时用 draft model 生成候选 token tree，再由 target LLM 验证和接受部分 token 的 speculative decoding 流程。</small>

**推理 pipeline 要抓住草稿和验证。**
这类图的核心不是“多了一个小模型”，而是把昂贵 target LLM 的逐 token 解码改成“先草拟一批候选，再一次验证一段”。draft model 决定候选质量和树形展开，target model 决定哪些 token 能被接受。它支撑的是吞吐/延迟优化 claim，不能证明模型回答质量提升；如果 acceptance rate 下降，额外 draft 成本可能抵消收益。

## 复现和证据口径

前沿论文和技术报告必须区分五类证据：论文自述、官方代码/模型、官方 demo、第三方复现、本站推断。闭源技术报告尤其不能把内部 benchmark 当成可复现实验。

| 状态 | 可写成 | 不要写成 |
| --- | --- | --- |
| Paper Only | 报告披露了方法和评测，但缺权重/完整配方 | 可复现 recipe |
| Author Code / Official Repo | 官方代码或模型可用，仍需检查是否覆盖论文主结果 | 独立复现 |
| Official Demo | 能说明功能形态或产品边界 | benchmark 证据 |
| Independent Reproduced | 有第三方复现实验或可运行复现记录 | 只因为 repo 能跑就标独立复现 |
| Site Inference | 本站根据证据做的工程判断 | 作者结论 |

## 外部精读

论文专题可以直接链接高质量外部材料，不要求都转成站内互链。优先级通常是：原论文/技术报告、官方博客或项目页、官方代码、可复现 benchmark、权威技术博客。外部博客和微信公众号文章可以学习讲法、结构和例子，但关键事实、图表和数值仍要回到论文或官方材料核对。

## 选题和维护

不是所有论文都适合写成专题。更适合进入这个模块的论文，通常满足至少一个条件：代表方向转折点；数据、训练、推理和评测强耦合；能为项目提供可复用模板；局限同样有启发价值；能连接站内多个主题。

新增专题页时遵守三个维护约定：标题使用“论文或系统名 + 一句话定位”；保留原论文链接和代码链接；结尾必须写阅读结论、局限、风险或不可外推结论。

专题页不要重复主题总览页已经讲清的基础概念。需要背景时链接回对应主题页，这样能保持“单篇论文拆解”的定位，也能减少同一概念在多个页面里不一致。

## 深度标杆页

| 标杆页 | 对齐标准 |
| --- | --- |
| [MagiAttention](foundations/magiattention.md) | 要能把长序列训练、mask、并行和系统收益讲成一条线 |
| [LingBot-World：世界模拟器](world-models/lingbot-world.md) | 要明确世界模型训练配方、数据路径、rollout 与效率贡献 |
| [DreamerV3：跨域通用世界模型](world-models/dreamerv3.md) | 要把 latent dynamics、imagined rollout 和样本效率讲清楚 |
| [π0.5：开放世界 VLA](embodied-ai/pi05.md) | 要解释 VLA 数据混合、动作专家和开放世界泛化 |
| [CausVid：流式自回归视频扩散](diffusion/causvid.md) | 要说明视频扩散如何被 causalize，并接到实时世界模型推理 |
