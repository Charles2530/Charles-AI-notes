# 论文专题讲解

这一模块用于放单篇论文的深度讲解。专题讲解页负责把一篇论文拆成问题背景、方法结构、训练流程、实验结论、局限和可复用经验。

## 模块定位

专题讲解适合三类论文：

1. 能代表一个方向转折点的论文；
2. 方法、数据、系统和评测高度耦合，不能只靠摘要理解的论文；
3. 适合作为项目设计参考，需要抽出工程经验的论文。

每篇讲解尽量固定回答六个问题：

1. 论文想解决的核心问题是什么；
2. 它和前一代方法的差异在哪里；
3. 数据、模型、训练和推理链路如何组织；
4. 实验结果到底支持了什么结论；
5. 这篇论文最值得复用的工程经验是什么；
6. 它的边界、风险和后续问题是什么。

## 和主题正文的关系

主题正文负责建立方向地图：它说明一个方向里有哪些核心问题、方法边界和工程接口。论文专题讲解则负责拆单篇论文：它默认读者已经选中一篇论文，希望真正理解系统结构、训练细节、实验依据和工程启发。

推荐顺序是：

1. 先用主题总览页确定问题位置；
2. 再用专题讲解页精读关键论文；
3. 最后回到正文主题页，把论文结论放进更大的方法和工程框架里。

这种分工可以避免两个常见问题：主题正文越写越像论文列表，专题页越写越像孤立摘要。好的论文讲解应该既能单篇读懂，也能回到整个知识库的主题结构里。

## 推荐写法

每篇专题讲解尽量保持固定结构，方便不同主题之间横向比较。

| 模块 | 需要回答的问题 |
| --- | --- |
| 论文信息 | 标题、链接、代码、关键词和适合读者 |
| 论文位置 | 它在方向谱系里解决哪一段问题 |
| 核心问题 | 旧方法为什么不够，论文要突破什么瓶颈 |
| 方法结构 | 数据、模型、训练目标、推理路径如何组织 |
| 实验结论 | 哪些结果真正支撑了论文主张 |
| 局限风险 | 哪些结论不能过度外推 |
| 项目启发 | 如果落到工程系统里，哪些设计最值得借鉴 |

专题页不追求把论文所有细节逐字复述，而是优先解释“为什么这样设计”。如果某个公式、模块或实验无法改变对论文的理解，就不必展开太长。相反，数据构造、训练阶段、推理约束和评测口径这类容易影响工程判断的部分，应写得更具体。

## 当前专题

各主题下的论文条目按论文年月从早到晚展示。新增专题页时只需要在页面 front matter 中填写 `paper_date: YYYY-MM`，导航和索引展示会按该字段排序。

!!! tip "技术报告怎么读"
    技术报告不要只看 benchmark 排名。更稳的读法是先抓“训练目标和系统约束”：模型结构为什么这样选，数据和阶段如何安排，后训练的 reward / SFT / distillation 如何分工，推理或部署是否改变了架构选择。读完一篇报告后，最好能把它接回 [训练](../training/index.md)、[强化学习](../reinforcement-learning/index.md)、[推理](../inference/index.md)、[量化](../quantization/index.md) 或 [算子与编译器](../operators/index.md) 中的某条工程主线。

### 基础知识

<table data-paper-sort="asc">
  <thead>
    <tr>
      <th>时间</th>
      <th>主题</th>
      <th>论文讲解</th>
      <th>阅读重点</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2025-02</td>
      <td>基础知识</td>
      <td><a href="foundations/muon-is-scalable/">Muon：面向 LLM 预训练的可扩展优化器</a></td>
      <td>如何把 Muon 从矩阵正交化优化器扩展到 5.7T token LLM 预训练，并系统对比 AdamW 的更新几何、状态内存、分布式通信和 SFT 边界</td>
    </tr>
    <tr>
      <td>2025-09</td>
      <td>基础知识</td>
      <td><a href="foundations/sla-sparse-linear-attention/">SLA / SLA2：DiT 里的 Sparse-Linear Attention</a></td>
      <td>如何把视频 DiT 的 attention 分解成 sparse exact、linear compensation 和 skip，并用 learnable routing、比例混合与 QAT 推进到 SLA2</td>
    </tr>
  </tbody>
</table>

### 扩散模型

<table data-paper-sort="asc">
  <thead>
    <tr>
      <th>时间</th>
      <th>主题</th>
      <th>论文讲解</th>
      <th>阅读重点</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2023-11</td>
      <td>扩散模型</td>
      <td><a href="diffusion/dmd/">DMD：One-step Diffusion with Distribution Matching Distillation</a></td>
      <td>如何用 real/fake score 差分和 regression loss，把多步扩散 teacher 蒸馏成一步图像生成器</td>
    </tr>
    <tr>
      <td>2024-05</td>
      <td>扩散模型</td>
      <td><a href="diffusion/dmd2/">DMD2：Improved Distribution Matching Distillation for Fast Image Synthesis</a></td>
      <td>如何用 TTUR 稳住 fake diffusion critic，用 GAN 真实数据监督和 backward simulation 把 DMD 推向更强的一步/少步生成</td>
    </tr>
    <tr>
      <td>2024-07</td>
      <td>扩散模型</td>
      <td><a href="diffusion/diffusion-forcing/">Diffusion Forcing：Next-token Prediction Meets Full-Sequence Diffusion</a></td>
      <td>如何用独立 per-token 噪声水平，把因果 next-token 生成、full-sequence diffusion guidance、长视频 rollout 和扩散规划放进同一个训练范式</td>
    </tr>
    <tr>
      <td>2024-12</td>
      <td>扩散模型</td>
      <td><a href="diffusion/causvid/">CausVid：From Slow Bidirectional to Fast Autoregressive Video Diffusion Models</a></td>
      <td>如何把双向视频 DiT 通过 ODE 初始化、非对称 DMD 和 KV cache 改造成少步流式自回归视频扩散生成器</td>
    </tr>
    <tr>
      <td>2025-03</td>
      <td>扩散模型</td>
      <td><a href="diffusion/wan/">Wan2.1：Open and Advanced Large-Scale Video Generative Models</a></td>
      <td>如何用视频 VAE、Video DiT、Flow Matching、分阶段图文视频训练和系统并行构建开源视频 foundation model，并补充 Wan2.2 README 的 MoE 与 TI2V 路线</td>
    </tr>
    <tr>
      <td>2025-10</td>
      <td>扩散模型</td>
      <td><a href="diffusion/phased-dmd/">Phased DMD：Few-step Distribution Matching Distillation via Score Matching within Subintervals</a></td>
      <td>如何把少步 DMD 拆成 SNR 子区间专家训练，并用子区间 score matching 保住图像多样性与视频运动动态</td>
    </tr>
  </tbody>
</table>

### 具身智能

<table data-paper-sort="asc">
  <thead>
    <tr>
      <th>时间</th>
      <th>主题</th>
      <th>论文讲解</th>
      <th>阅读重点</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2024-10</td>
      <td>具身智能</td>
      <td><a href="embodied-ai/gr2/">GR-2：用 web-scale 视频知识迁移到机器人操作</a></td>
      <td>如何把 38M 视频片段和 50B token 的未来视频预训练迁移到真实机器人轨迹预测，并通过 WBC 落到 Kinova Gen3</td>
    </tr>
    <tr>
      <td>2024-12</td>
      <td>具身智能</td>
      <td><a href="embodied-ai/video-prediction-policy/">Video Prediction Policy：用预测视觉表征训练通用机器人策略</a></td>
      <td>如何把 text-guided video prediction 的中间预测表征当作视觉 encoder，用单步 forward 支撑 CALVIN、MetaWorld 和真实机器人策略</td>
    </tr>
    <tr>
      <td>2025-01</td>
      <td>具身智能</td>
      <td><a href="embodied-ai/spatialvla/">SpatialVLA：把 3D 空间表示接进 VLA</a></td>
      <td>如何用 Ego3D position encoding 和 adaptive action grids 把 3D 空间与跨机器人动作 token 接进 VLA</td>
    </tr>
    <tr>
      <td>2025-04</td>
      <td>具身智能</td>
      <td><a href="embodied-ai/pi05/">π0.5：面向开放世界泛化的 VLA</a></td>
      <td>如何用 heterogeneous co-training、FAST 离散动作、flow matching action expert 和 high-level subtask prediction 支撑新家庭长任务泛化</td>
    </tr>
    <tr>
      <td>2025-07</td>
      <td>具身智能</td>
      <td><a href="embodied-ai/gr3/">GR-3：任务状态、少样本人类轨迹与长时程 VLA</a></td>
      <td>如何用 Qwen2.5-VL + flow matching Action DiT、任务状态监督、VR 人类轨迹和 ByteMini 系统推进长时程 VLA</td>
    </tr>
    <tr>
      <td>2026-02</td>
      <td>具身智能</td>
      <td><a href="embodied-ai/dreamzero/">DreamZero：World Action Models are Zero-shot Policies</a></td>
      <td>joint video-action prediction 如何让 WAM 直接成为机器人 policy，并把世界模型的动作后果预测放进具身闭环</td>
    </tr>
  </tbody>
</table>

### 世界模型

<table data-paper-sort="asc">
  <thead>
    <tr>
      <th>时间</th>
      <th>主题</th>
      <th>论文讲解</th>
      <th>阅读重点</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2018-11</td>
      <td>世界模型</td>
      <td><a href="world-models/planet/">PlaNet：Learning Latent Dynamics for Planning from Pixels</a></td>
      <td>RSSM latent dynamics 如何从像素交互轨迹中学习，并通过 CEM 在 latent space 里做在线规划</td>
    </tr>
    <tr>
      <td>2019-12</td>
      <td>世界模型</td>
      <td><a href="world-models/dreamer/">Dreamer：Dream to Control</a></td>
      <td>如何把 RSSM world model 从在线规划接口扩展为 latent imagination actor-critic 训练接口</td>
    </tr>
    <tr>
      <td>2020-10</td>
      <td>世界模型</td>
      <td><a href="world-models/dreamerv2/">DreamerV2：Mastering Atari with Discrete World Models</a></td>
      <td>categorical latent、KL balancing 和 image gradients 如何让 Dreamer 路线打到 Atari 强基准</td>
    </tr>
    <tr>
      <td>2022-06</td>
      <td>世界模型</td>
      <td><a href="world-models/jepa/">JEPA：A Path Towards Autonomous Machine Intelligence</a></td>
      <td>representation-space prediction、non-contrastive training 和 H-JEPA 如何定义非生成式世界模型路线</td>
    </tr>
    <tr>
      <td>2023-01</td>
      <td>世界模型</td>
      <td><a href="world-models/dreamerv3/">DreamerV3：Mastering Diverse Domains through World Models</a></td>
      <td>RSSM world model 如何通过 latent dynamics、reward prediction 和 imagined rollout 支撑 actor-critic</td>
    </tr>
    <tr>
      <td>2023-06</td>
      <td>世界模型</td>
      <td><a href="world-models/h-jepa/">H-JEPA：Latent Variable EBMs and Hierarchical JEPA</a></td>
      <td>latent-variable EBM、energy collapse 和 regularized JEPA training 如何支撑层级表征世界模型</td>
    </tr>
    <tr>
      <td>2023-10</td>
      <td>世界模型</td>
      <td><a href="world-models/ringattention/">RingAttention：Ring Attention with Blockwise Transformers for Near-Infinite Context</a></td>
      <td>如何用 blockwise attention、KV 环形通信和 sequence parallelism 支撑百万 token 视频/语言/轨迹世界模型训练</td>
    </tr>
    <tr>
      <td>2024-02</td>
      <td>世界模型</td>
      <td><a href="world-models/v-jepa/">V-JEPA：Latent Video Prediction for Visual Representation Learning</a></td>
      <td>如何用 latent video prediction、3D Multi-Block Masking 和 EMA target encoder 学习视频世界表征</td>
    </tr>
    <tr>
      <td>2024-02</td>
      <td>世界模型</td>
      <td><a href="world-models/genie/">Genie：Generative Interactive Environments</a></td>
      <td>如何从无标注互联网视频中学习 latent actions，并把视频 token dynamics 变成可逐帧控制的生成式交互环境</td>
    </tr>
    <tr>
      <td>2024-02</td>
      <td>世界模型</td>
      <td><a href="world-models/lwm/">LWM：百万 token 视频语言世界模型</a></td>
      <td>如何用 RingAttention、VQGAN 离散视觉 token、masked sequence packing 和渐进式 context extension 训练 1M token 多模态自回归世界模型</td>
    </tr>
    <tr>
      <td>2025-06</td>
      <td>世界模型</td>
      <td><a href="world-models/self-forcing/">Self Forcing：Bridging the Train-Test Gap in Autoregressive Video Diffusion</a></td>
      <td>如何通过训练时自回归 self-rollout、视频级 distribution matching 和 rolling KV cache 缓解流式视频世界模型的 exposure bias 与误差累积</td>
    </tr>
    <tr>
      <td>2025-07</td>
      <td>世界模型</td>
      <td><a href="world-models/towards-video-world-models/">Towards Video World Models：视频生成走向世界模型的五个门槛</a></td>
      <td>用 causal、interactive、persistent、real-time 和 physical accuracy 五个约束梳理视频生成到世界模型的训练路线</td>
    </tr>
    <tr>
      <td>2025-10</td>
      <td>世界模型</td>
      <td><a href="world-models/world-models-embodied-ai-survey/">A Comprehensive Survey on World Models for Embodied AI：具身世界模型综述</a></td>
      <td>用 functionality、temporal modeling 和 spatial representation 三轴 taxonomy 梳理具身世界模型、训练数据、评测指标和代表性能表格</td>
    </tr>
    <tr>
      <td>2026-01</td>
      <td>世界模型</td>
      <td><a href="world-models/lingbot-world/">LingBot-World：从视频生成到开源世界模拟器</a></td>
      <td>视频基础模型如何继续训练成可交互的世界模型</td>
    </tr>
  </tbody>
</table>

### 技术报告

<table data-paper-sort="asc">
  <thead>
    <tr>
      <th>时间</th>
      <th>主题</th>
      <th>报告讲解</th>
      <th>阅读重点</th>
      <th>站内延伸</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2024-08</td>
      <td>技术报告</td>
      <td><a href="technical-reports/gpt-4o-system-card/">GPT-4o System Card：端到端 Omni 模型、语音安全与多模态部署评测</a></td>
      <td>如何把端到端 text/audio/image/video omni model 的训练边界、语音输出安全、red teaming、Preparedness Framework 和第三方 agentic evaluation 串成上线前评测体系</td>
      <td><a href="../vlm/">VLM/VLA</a> / <a href="../inference/serving-systems/">推理服务系统</a></td>
    </tr>
    <tr>
      <td>2024-12</td>
      <td>技术报告</td>
      <td><a href="technical-reports/deepseek-v3/">DeepSeek-V3 Technical Report：671B MoE、MLA、FP8 训练与低成本预训练</a></td>
      <td>如何用 MLA、DeepSeekMoE、auxiliary-loss-free routing、MTP、FP8 mixed precision、DualPipe、SFT/RL 和 R1 reasoning distillation 训练 671B MoE</td>
      <td><a href="../training/low-bit-training-and-numerics/">低比特训练</a> / <a href="../operators/communication-kernels-and-overlap/">通信重叠</a></td>
    </tr>
    <tr>
      <td>2025-01</td>
      <td>技术报告</td>
      <td><a href="technical-reports/deepseek-r1/">DeepSeek-R1：用强化学习激发大模型推理能力</a></td>
      <td>如何用 GRPO、rule-based reward、长 CoT、rejection sampling、两阶段 RL 和蒸馏把可验证推理任务转成大模型自我探索能力</td>
      <td><a href="../reinforcement-learning/policy-gradient-actor-critic-ppo-grpo/">PPO / GRPO</a> / <a href="../training/preference-data-and-alignment-pitfalls/">对齐失效</a></td>
    </tr>
    <tr>
      <td>2025-05</td>
      <td>技术报告</td>
      <td><a href="technical-reports/qwen3/">Qwen3 Technical Report：统一 Thinking / Non-Thinking、36T 预训练与强到弱蒸馏</a></td>
      <td>如何用 36T token 三阶段预训练、四阶段后训练、GRPO、thinking budget 和 strong-to-weak distillation 构建双模式开源模型家族</td>
      <td><a href="../reinforcement-learning/">强化学习</a> / <a href="../training/post-training-data-engines-and-judge-models/">后训练数据引擎</a></td>
    </tr>
    <tr>
      <td>2025-07</td>
      <td>技术报告</td>
      <td><a href="technical-reports/gemini-2-5/">Gemini 2.5 Technical Report：动态 Thinking、多模态长上下文与 Agentic 能力</a></td>
      <td>如何用 native multimodal sparse MoE、teacher distillation、TPUv5p/Pathways、SFT/RM/RL、dynamic thinking 和工具/多模态数据构建 1M context Gemini 2.5</td>
      <td><a href="../training/">训练</a> / <a href="../inference/rag-agents-and-long-context-systems/">Agent / 长上下文</a></td>
    </tr>
    <tr>
      <td>2025-07</td>
      <td>技术报告</td>
      <td><a href="technical-reports/kimi-k2/">Kimi K2：Open Agentic Intelligence、MuonClip 与万亿 MoE Agent 模型</a></td>
      <td>如何用 MuonClip 跑稳 1T MoE 预训练，并通过 agentic data synthesis、tool-use SFT、RLVR/self-critique reward 和 checkpoint engine 训练开放 agent 模型</td>
      <td><a href="../training/distributed-training-and-checkpointing/">分布式训练</a> / <a href="../inference/rag-agents-and-long-context-systems/">Agent 系统</a></td>
    </tr>
    <tr>
      <td>2026-04</td>
      <td>技术报告</td>
      <td><a href="technical-reports/nemotron-3-super/">Nemotron 3 Super：Hybrid Mamba-MoE、LatentMoE、MTP 与异步多环境 RL</a></td>
      <td>如何用 Mamba-heavy hybrid backbone、LatentMoE、shared-weight MTP、NVFP4 预训练、多环境异步 GRPO 和 FP8/NVFP4 PTQ 构建高吞吐 agentic MoE</td>
      <td><a href="../inference/caching-routing-and-speculative/">推测解码</a> / <a href="../quantization/fp8-and-mixed-precision-serving/">低精度服务</a></td>
    </tr>
    <tr>
      <td>2026-04</td>
      <td>技术报告</td>
      <td><a href="technical-reports/qwen35-omni/">Qwen3.5-Omni Technical Report：Thinker-Talker、ARIA 与 256K 全模态 Agent</a></td>
      <td>如何用 Thinker-Talker、AuT、Hybrid MoE、ARIA、三阶段预训练和分模块后训练构建可流式语音交互的全模态 agent</td>
      <td><a href="../vlm/">VLM/VLA</a> / <a href="../inference/serving-systems/">推理服务系统</a></td>
    </tr>
    <tr>
      <td>2026-05</td>
      <td>技术报告</td>
      <td><a href="technical-reports/deepseek-v4-pro/">DeepSeek-V4 Technical Report：百万上下文、混合注意力与 FP4 后训练</a></td>
      <td>如何用 CSA/HCA、mHC、Muon、FP4 QAT、异构 KV cache 和 OPD 基础设施把 MoE 大模型推向 1M context 与长流程 agent 场景</td>
      <td><a href="../inference/context-compression-kv-eviction-and-memory-hierarchies/">KV / 长上下文</a> / <a href="../quantization/fp8-and-mixed-precision-serving/">低精度服务</a></td>
    </tr>
  </tbody>
</table>

## 选题原则

不是所有论文都适合写成专题讲解。更适合进入这个模块的论文，通常满足至少一个条件：

1. 它代表一个方向从概念走向系统的关键节点；
2. 它的数据、训练、推理和评测强耦合，只看摘要容易误读；
3. 它能为后续项目提供可复用的设计模板；
4. 它的局限同样有启发价值，能帮助判断某条路线目前还做不到什么；
5. 它和站内多个主题有关，适合做跨页面连接。

例如 `LingBot-World` 适合放在这里，不是因为它已经解决了所有世界模型问题，而是因为它把“视频生成模型如何继续训练成交互式世界模拟器”这条路线讲得比较完整，能同时连接世界模型、视频生成、动作条件建模和推理实时化。

## 阅读建议

读专题讲解时，建议先看“论文位置”和“训练路线”，再看实验与局限。很多论文的真正价值不在某个单点指标，而在它把数据、模型、训练目标和系统接口组织成一条可执行路径。

如果只是想快速建立方向地图，先看对应主题总览页和 [参考文献总表](../references/index.md)。如果已经选定某篇论文准备精读、复现或用于项目设计，再看这里的专题讲解。

## 使用方式

更推荐把专题讲解当成“读论文后的结构化笔记”，而不是读论文前的替代品。先快速扫一遍论文摘要、图表和实验，再回来看专题页，会更容易判断哪些内容是作者的核心主张，哪些是本文对工程落地的二次归纳。

如果要用专题页支持项目讨论，可以按下面顺序使用：

1. 先看“论文位置”，确认它解决的问题是否和当前项目一致；
2. 再看“方法结构”，确认数据、模型和系统假设是否能复用；
3. 然后看“实验与局限”，确认论文证据能否支撑项目里的风险判断；
4. 最后看“项目启发”，把可执行动作拆成数据、训练、评测和部署任务。

这个顺序比直接抄论文方法更稳，因为很多论文的关键收益依赖特定数据、算力、评测口径或系统接口。专题讲解的目标就是把这些隐含条件提前暴露出来。

## 维护约定

新增专题页时尽量遵守三个约定：

1. 标题使用“论文或系统名 + 一句话定位”，避免只写英文标题；
2. 页面中保留原论文链接和代码链接，方便回查来源；
3. 结尾必须写“局限、风险或不可外推结论”，避免把论文总结写成单向宣传。

如果后续同一主题下论文变多，可以再增加主题索引页。例如世界模型下可以继续分成视频世界模型、RSSM/Dreamer、规划与反事实、数据引擎四组。当前先保持轻量结构，避免导航层级过深。

另一个维护原则是避免重复正文主题页。专题页可以更细地讲一篇论文，但不要把主题页已经讲清的基础概念再完整复制一遍；需要背景时用链接回到对应主题页即可。这样能让专题页保持“单篇论文拆解”的定位，也能减少后续维护时同一概念在多个页面里不一致的问题。

## 后续扩展

后续可以继续按主题增加专题页，例如：

1. 训练系统中的代表性工程论文；
2. FP8、量化和低精度训练的关键论文；
3. VLM/VLA 中真正改变数据或评测方式的论文；
4. 推理系统中影响服务架构的论文；
5. 世界模型、仿真和数据引擎方向的系统论文。

新增专题时优先保持“少而深”。这个模块不是论文收藏夹，而是为那些值得反复回看、适合支撑工程判断的论文建立稳定讲解入口。
