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

!!! tip "论文页的读法"
    不要从第一个公式一路刷到最后。先看它解决的瓶颈，再看方法结构，随后只追那些会改变工程判断的细节：数据怎么来、训练分几段、推理路径是否变化、实验支持了什么、边界在哪里。

<div class="atlas-meta-grid">
  <div>
    <strong>读前先定位</strong>
    <p>这篇论文属于训练、推理、量化、VLA、世界模型还是安全评测。</p>
  </div>
  <div>
    <strong>读中抓证据</strong>
    <p>区分作者公开事实、实验结果、系统约束和本文给出的工程判断。</p>
  </div>
  <div>
    <strong>读完要回填</strong>
    <p>把论文结论写回主题路线图，而不是只留下一个模型名字和 benchmark 印象。</p>
  </div>
</div>

!!! tip "快速定位"
    **一句话结论**：论文专题不是摘要库，而是把论文 claim 接回全站成本、证据和成熟度判断。
    **适合读者**：想快速判断一篇论文能否进入实验方案、工程 baseline 或只是前沿假设的人。
    **先看哪张表**：先看“最小验收指标”和“本站判断收口”，再进入具体论文列表。

!!! tip "10 分钟读法 | 深入读法 | 实验/工程读法"
    **10 分钟读法**：先读论文解决的问题、核心图表和本站判断。
    **深入读法**：深读时按 Figure/Table/Setting 拆 claim，区分论文结果、系统吞吐、官方 demo 和站内推断。
    **实验/工程读法**：写专题时补图解读法、实验口径和不能外推的结论。

!!! info "写作与验收规范"
    新增或重写专题页时，先看 [论文专题写作与验收规范](writing-standard.md)。入口页只做导航，规范页负责方法深度、实验口径、图表读法、复现状态和本站判断收口。

## 当前专题

各主题下的论文条目按论文年月从早到晚展示。新增专题页时只需要在页面 front matter 中填写 `paper_date: YYYY-MM`，导航和索引展示会按该字段排序。

!!! tip "技术报告怎么读"
    技术报告不要只看 benchmark 排名。更稳的读法是先抓“训练目标和系统约束”：模型结构为什么这样选，数据和阶段如何安排，后训练的 reward / SFT / distillation 如何分工，推理或部署是否改变了架构选择。如果报告里出现 MoE、activated params、shared/routed experts、EP/all-to-all，先回到 [MoE 与大模型架构表](../foundations/moe-and-large-model-architecture-basics.md) 拆清容量、计算、路由训练和系统成本。读完一篇报告后，最好能把它接回 [训练](../training/index.md)、[强化学习](../reinforcement-learning/index.md)、[推理](../inference/index.md)、[量化](../quantization/index.md) 或 [算子与编译器](../operators/index.md) 中的某条工程主线。

### 高效训练

!!! note "这组文章怎么连起来读"
    高效训练不是单一技巧，而是把同一笔训练预算花得更有效。建议按层级读：先看 Megatron-LM，理解 tensor model parallel、data parallel group、mixed precision 和 activation checkpointing 如何组成大模型训练底座；再看 ZeRO，理解 optimizer states、gradients、parameters 和 activations 的零冗余显存账；然后看 Muon 理解 optimizer 如何提升 token/FLOP efficiency；接着看 GPU Utilization，理解高利用率、MFU、SM efficiency 和 Tensor Core 利用率之间的指标口径；再看 MagiAttention 理解超长上下文训练的 CP、mask 和通信瓶颈；再看 SLA / SLA2 理解视频 DiT 里如何用 sparse-linear decomposition 减少 attention 计算；最后看 Attn-QAT 理解 FP4 attention 为什么必须同时改 forward、backward 和 inference kernel。七篇合起来，对应“模型并行 -> 状态分片 -> 优化器 -> 指标诊断 -> 分布式 attention -> 近似 attention -> 低比特 attention”这条工程链。

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
      <td>2019-09</td>
      <td>高效训练</td>
      <td><a href="foundations/megatron-lm/">Megatron-LM：Tensor Model Parallel</a></td>
      <td>如何把 Transformer 的 MLP、attention 和词表 loss 切成 tensor parallel，并结合 mixed precision、activation checkpointing、数据去重和 8-way MP + 64-way DP 训练 8.3B GPT-2 / 3.9B BERT</td>
    </tr>
    <tr>
      <td>2019-10</td>
      <td>高效训练</td>
      <td><a href="foundations/zero/">ZeRO：大模型训练的零冗余状态优化</a></td>
      <td>如何把 data parallel 中重复保存的 optimizer states、gradients、parameters 逐步分片，并用 activation partition、constant buffers 和 defragmentation 支撑 100B/170B 级 GPT-like 训练</td>
    </tr>
    <tr>
      <td>2025-02</td>
      <td>高效训练</td>
      <td><a href="foundations/muon-is-scalable/">Muon：LLM 预训练优化器</a></td>
      <td>如何把 Muon 从矩阵正交化优化器扩展到 5.7T token LLM 预训练，并系统对比 AdamW 的更新几何、状态内存、分布式通信和 SFT 边界</td>
    </tr>
    <tr>
      <td>2025-02</td>
      <td>高效训练</td>
      <td><a href="foundations/gpu-utilization-misleading/">GPU Utilization：为什么 100% 不等于训练跑满</a></td>
      <td>如何区分 GPU utilization、MFU、SM efficiency、Tensor Core utilization、memory bandwidth 和 profiler timeline，并用 fused kernels 与合适并行度解决高利用率低效率训练</td>
    </tr>
    <tr>
      <td>2025-04</td>
      <td>高效训练</td>
      <td><a href="foundations/magiattention/">MagiAttention：超长上下文分布式 Attention</a></td>
      <td>如何把超长上下文和 varlen/block-causal 异构 mask 训练拆成 AttnSlice、负载均衡 dispatch、GroupCast/GroupReduce 和多阶段 overlap，并在 H100/B200 上验证分布式 attention 可扩展性</td>
    </tr>
    <tr>
      <td>2025-09</td>
      <td>高效训练</td>
      <td><a href="foundations/sla-sparse-linear-attention/">SLA / SLA2：DiT 稀疏线性 Attention</a></td>
      <td>如何把视频 DiT 的 attention 分解成 sparse exact、linear compensation 和 skip，并用 learnable routing、比例混合与 QAT 推进到 SLA2</td>
    </tr>
    <tr>
      <td>2026-03</td>
      <td>高效训练</td>
      <td><a href="foundations/attn-qat/">Attn-QAT：4-bit Attention 量化感知训练</a></td>
      <td>为什么 4-bit attention 不能直接套普通 QAT，Attn-QAT 如何修正 FlashAttention backward 的精度假设，并在 Wan、Qwen 与 Llama 实验中恢复 FP4 attention 质量</td>
    </tr>
  </tbody>
</table>

### 高效推理

!!! note "这组文章怎么连起来读"
    高效推理的核心不是只追 tokens/s，而是把上下文长度、KV cache、decode latency、batch、视觉感知延迟和质量损失放在同一张成本表里。先读 EAGLE，理解 feature-level draft 和 speculative verification；再读 EAGLE-2，理解 dynamic draft tree；接着读 Low-bit LLM Survey，建立量化格式、系统数据流、低比特训练和 PTQ/QAT 的全景；然后读 EAGLE-3，理解 draft model 训练、投机验证和 runtime 吞吐如何耦合；再读 Fast-FoundationStereo，把视角扩展到机器人/AR 场景里的实时视觉 foundation model 加速；最后读 KVSlimmer，进入 KV cache compression 的具体机制。

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
      <td>2024-01</td>
      <td>高效推理</td>
      <td><a href="inference/eagle/">EAGLE：Feature-level Draft 投机推理</a></td>
      <td>如何用 target LLM 的 second-to-top-layer feature、shifted-token 输入和轻量 Autoregression Head 构造 lossless speculative sampling，并通过训练损失、tree attention 与 batch/throughput 实验解释速度边界</td>
    </tr>
    <tr>
      <td>2024-06</td>
      <td>高效推理</td>
      <td><a href="inference/eagle-2/">EAGLE-2：动态 Draft Tree 投机推理</a></td>
      <td>如何用 draft model confidence 近似 acceptance rate，按路径 value 动态扩展和 rerank draft tree，在不额外训练、不改原模型、不放宽接受条件的前提下提升 average acceptance length 和 speedup</td>
    </tr>
    <tr>
      <td>2024-09</td>
      <td>高效推理</td>
      <td><a href="inference/low-bit-llm-survey/">Low-bit LLM Survey：低比特大模型量化</a></td>
      <td>如何把低比特 LLM 拆成 number format、granularity、dynamic/static quantization、system support、low-bit training、PEFT、QAT 和 PTQ，并理解为什么量化必须落到 runtime、kernel 与 KV cache 数据流上</td>
    </tr>
    <tr>
      <td>2025-03</td>
      <td>高效推理</td>
      <td><a href="inference/eagle-3/">EAGLE-3：Training-time Test 的投机推理加速</a></td>
      <td>如何把 speculative decoding 的 draft model 从 feature prediction 改成 direct token prediction，并用 training-time test、多层 feature fusion 和真实 runtime 表验证多步 draft 的接受率与吞吐收益</td>
    </tr>
    <tr>
      <td>2025-12</td>
      <td>高效推理</td>
      <td><a href="inference/fast-foundation-stereo/">Fast-FoundationStereo：实时 Zero-Shot 双目匹配</a></td>
      <td>如何把 FoundationStereo 的 feature backbone、cost filtering 和 refinement 三个瓶颈分别用蒸馏、分块 NAS、结构化剪枝和 in-the-wild pseudo labels 压到实时感知推理</td>
    </tr>
    <tr>
      <td>2026-03</td>
      <td>高效推理</td>
      <td><a href="inference/kvslimmer/">KVSlimmer：非对称 KV 合并</a></td>
      <td>如何用 Q/K/V 谱分布解释 KV 非对称，再把 Key merge 的 exact Hessian coupling 化简成只依赖 forward-pass 变量的闭式权重，从而降低长上下文推理的 KV memory 和 decoder latency</td>
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
      <td>2020-11</td>
      <td>扩散模型</td>
      <td><a href="diffusion/score-sde/">Score SDE：连续时间扩散框架</a></td>
      <td>如何把 SMLD 和 DDPM 统一成前向 SDE、反向 SDE 与 probability flow ODE，并用 NCSN++ / DDPM++ 训练配方、PC sampler 和 continuous score matching 推动生成质量与 likelihood</td>
    </tr>
    <tr>
      <td>2022-11</td>
      <td>扩散模型</td>
      <td><a href="diffusion/dpm-solver-plus-plus/">DPM-Solver++：Guided Sampling 求解器</a></td>
      <td>如何把扩散 ODE 从 noise prediction 改写到 data prediction，用 \(x_\theta\)、thresholding 和 multistep 更新在不训练的前提下稳定加速 CFG / guided sampling</td>
    </tr>
    <tr>
      <td>2023-11</td>
      <td>扩散模型</td>
      <td><a href="diffusion/dmd/">DMD：一步扩散蒸馏</a></td>
      <td>如何用 real/fake score 差分和 regression loss，把多步扩散 teacher 蒸馏成一步图像生成器</td>
    </tr>
    <tr>
      <td>2024-05</td>
      <td>扩散模型</td>
      <td><a href="diffusion/dmd2/">DMD2：更稳的少步分布匹配蒸馏</a></td>
      <td>如何用 TTUR 稳住 fake diffusion critic，用 GAN 真实数据监督和 backward simulation 把 DMD 推向更强的一步/少步生成</td>
    </tr>
    <tr>
      <td>2024-07</td>
      <td>扩散模型</td>
      <td><a href="diffusion/diffusion-forcing/">Diffusion Forcing：next-token 与全序列扩散</a></td>
      <td>如何用独立 per-token 噪声水平，把因果 next-token 生成、full-sequence diffusion guidance、长视频 rollout 和扩散规划放进同一个训练范式</td>
    </tr>
    <tr>
      <td>2024-12</td>
      <td>扩散模型</td>
      <td><a href="diffusion/causvid/">CausVid：流式自回归视频扩散</a></td>
      <td>如何把双向视频 DiT 通过 ODE 初始化、非对称 DMD 和 KV cache 改造成少步流式自回归视频扩散生成器</td>
    </tr>
    <tr>
      <td>2025-03</td>
      <td>扩散模型</td>
      <td><a href="diffusion/wan/">Wan2.1：开源视频生成系统路线</a></td>
      <td>如何用视频 VAE、Video DiT、Flow Matching、分阶段图文视频训练和系统并行构建开源视频 foundation model，并补充 Wan2.2 README 的 MoE 与 TI2V 路线</td>
    </tr>
    <tr>
      <td>2025-10</td>
      <td>扩散模型</td>
      <td><a href="diffusion/phased-dmd/">Phased DMD：分阶段少步蒸馏</a></td>
      <td>如何把少步 DMD 拆成 SNR 子区间专家训练，并用子区间 score matching 保住图像多样性与视频运动动态</td>
    </tr>
    <tr>
      <td>2026-05</td>
      <td>扩散模型</td>
      <td><a href="diffusion/anyflow/">AnyFlow：任意步视频扩散蒸馏</a></td>
      <td>如何把 endpoint consistency 蒸馏改成 flow map transition 学习，并用 on-policy flow map distillation 让同一个视频学生模型支持 4 到 32 step 的测试时质量伸缩</td>
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
      <td>2023-07</td>
      <td>具身智能</td>
      <td><a href="embodied-ai/rt2/">RT-2：把 VLM 变成机器人动作模型</a></td>
      <td>如何把 6-DoF 机器人动作离散成语言 token，并用 PaLI-X / PaLM-E 与 web VQA、caption 和机器人轨迹做 co-fine-tuning</td>
    </tr>
    <tr>
      <td>2024-01</td>
      <td>具身智能</td>
      <td><a href="embodied-ai/depth-anything/">Depth Anything：大规模无标注单目深度</a></td>
      <td>如何用 1.5M labeled images、62M unlabeled images、strong perturbation、CutMix 和 DINOv2 feature alignment 训练开放域相对深度基础模型</td>
    </tr>
    <tr>
      <td>2024-06</td>
      <td>具身智能</td>
      <td><a href="embodied-ai/depth-anything-v2/">Depth Anything V2：单目深度的数据配方</a></td>
      <td>如何用 synthetic teacher、62M pseudo-labeled real images、DINOv2 + DPT 和 gradient matching loss 构建更细、更稳、更高效的单目深度底座</td>
    </tr>
    <tr>
      <td>2024-10</td>
      <td>具身智能</td>
      <td><a href="embodied-ai/gr2/">GR-2：Web 视频知识迁移到机器人</a></td>
      <td>如何把 38M 视频片段和 50B token 的未来视频预训练迁移到真实机器人轨迹预测，并通过 WBC 落到 Kinova Gen3</td>
    </tr>
    <tr>
      <td>2024-12</td>
      <td>具身智能</td>
      <td><a href="embodied-ai/video-prediction-policy/">Video Prediction Policy：预测视觉表征训练机器人策略</a></td>
      <td>如何把 text-guided video prediction 的中间预测表征当作视觉 encoder，用单步 forward 支撑 CALVIN、MetaWorld 和真实机器人策略</td>
    </tr>
    <tr>
      <td>2025-01</td>
      <td>具身智能</td>
      <td><a href="embodied-ai/spatialvla/">SpatialVLA：3D 空间表征接入 VLA</a></td>
      <td>如何用 Ego3D position encoding 和 adaptive action grids 把 3D 空间与跨机器人动作 token 接进 VLA</td>
    </tr>
    <tr>
      <td>2025-03</td>
      <td>具身智能</td>
      <td><a href="embodied-ai/vggt/">VGGT：feed-forward 3D 几何</a></td>
      <td>如何用 DINOv2 tokenization、Alternating-Attention、多任务 loss 和 3D annotated datasets 一次前向预测 camera、depth、point map 与 point tracks</td>
    </tr>
    <tr>
      <td>2025-04</td>
      <td>具身智能</td>
      <td><a href="embodied-ai/pi05/">π0.5：开放世界 VLA</a></td>
      <td>如何用 heterogeneous co-training、FAST 离散动作、flow matching action expert 和 high-level subtask prediction 支撑新家庭长任务泛化</td>
    </tr>
    <tr>
      <td>2025-07</td>
      <td>具身智能</td>
      <td><a href="embodied-ai/gr3/">GR-3：少样本长时程 VLA</a></td>
      <td>如何用 Qwen2.5-VL + flow matching Action DiT、任务状态监督、VR 人类轨迹和 ByteMini 系统推进长时程 VLA</td>
    </tr>
    <tr>
      <td>2025-09</td>
      <td>具身智能</td>
      <td><a href="embodied-ai/mapanything/">MapAnything：统一前向 Metric 3D 重建骨干</a></td>
      <td>如何用 factored scene representation、geometric input augmentation、DINOv2 + Alternating-Attention 和 two-stage view curriculum 统一 SfM、MVS、depth completion 与 metric 3D reconstruction</td>
    </tr>
    <tr>
      <td>2025-10</td>
      <td>具身智能</td>
      <td><a href="embodied-ai/vodp/">VO-DP：纯视觉语义-几何自适应扩散策略</a></td>
      <td>如何把 VGGT 的 DINOv2 语义特征与 Alternating-Attention 几何特征融合，接入 DDPM policy head，在只用单视角 RGB 的情况下逼近或超过点云 DP3</td>
    </tr>
    <tr>
      <td>2025-11</td>
      <td>具身智能</td>
      <td><a href="embodied-ai/depth-anything-3/">Depth Anything 3：任意视角 3D 几何底座</a></td>
      <td>如何用 depth-ray representation、cross-view attention 和 teacher-student 训练把单目深度扩展成可估姿、可融合、可渲染的任意视角几何</td>
    </tr>
    <tr>
      <td>2026-02</td>
      <td>具身智能</td>
      <td><a href="embodied-ai/dreamzero/">DreamZero：WAM 零样本策略</a></td>
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
      <td><a href="world-models/planet/">PlaNet：像素规划的潜变量动力学</a></td>
      <td>RSSM latent dynamics 如何从像素交互轨迹中学习，并通过 CEM 在 latent space 里做在线规划</td>
    </tr>
    <tr>
      <td>2019-12</td>
      <td>世界模型</td>
      <td><a href="world-models/dreamer/">Dreamer：Latent Imagination 控制</a></td>
      <td>如何把 RSSM world model 从在线规划接口扩展为 latent imagination actor-critic 训练接口</td>
    </tr>
    <tr>
      <td>2020-10</td>
      <td>世界模型</td>
      <td><a href="world-models/dreamerv2/">DreamerV2：离散世界模型玩 Atari</a></td>
      <td>categorical latent、KL balancing 和 image gradients 如何让 Dreamer 路线打到 Atari 强基准</td>
    </tr>
    <tr>
      <td>2022-06</td>
      <td>世界模型</td>
      <td><a href="world-models/jepa/">JEPA：预测式表征学习路线</a></td>
      <td>representation-space prediction、non-contrastive training 和 H-JEPA 如何定义非生成式世界模型路线</td>
    </tr>
    <tr>
      <td>2023-01</td>
      <td>世界模型</td>
      <td><a href="world-models/dreamerv3/">DreamerV3：跨域通用世界模型</a></td>
      <td>RSSM world model 如何通过 latent dynamics、reward prediction 和 imagined rollout 支撑 actor-critic</td>
    </tr>
    <tr>
      <td>2023-06</td>
      <td>世界模型</td>
      <td><a href="world-models/h-jepa/">H-JEPA：层级 JEPA 与潜变量 EBM</a></td>
      <td>latent-variable EBM、energy collapse 和 regularized JEPA training 如何支撑层级表征世界模型</td>
    </tr>
    <tr>
      <td>2023-10</td>
      <td>世界模型</td>
      <td><a href="world-models/ringattention/">RingAttention：近无限上下文训练</a></td>
      <td>如何用 blockwise attention、KV 环形通信和 sequence parallelism 支撑百万 token 视频/语言/轨迹世界模型训练</td>
    </tr>
    <tr>
      <td>2024-02</td>
      <td>世界模型</td>
      <td><a href="world-models/v-jepa/">V-JEPA：视频潜变量预测表征</a></td>
      <td>如何用 latent video prediction、3D Multi-Block Masking 和 EMA target encoder 学习视频世界表征</td>
    </tr>
    <tr>
      <td>2024-02</td>
      <td>世界模型</td>
      <td><a href="world-models/genie/">Genie：无标注视频到可交互环境</a></td>
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
      <td><a href="world-models/self-forcing/">Self Forcing：对齐自回归视频训练与推理</a></td>
      <td>如何通过训练时自回归 self-rollout、视频级 distribution matching 和 rolling KV cache 缓解流式视频世界模型的 exposure bias 与误差累积</td>
    </tr>
    <tr>
      <td>2025-07</td>
      <td>世界模型</td>
      <td><a href="world-models/towards-video-world-models/">Towards Video World Models：视频世界模型五个门槛</a></td>
      <td>用 causal、interactive、persistent、real-time 和 physical accuracy 五个约束梳理视频生成到世界模型的训练路线</td>
    </tr>
    <tr>
      <td>2025-10</td>
      <td>世界模型</td>
      <td><a href="world-models/world-models-embodied-ai-survey/">Embodied World Model Survey：具身世界模型综述</a></td>
      <td>用 functionality、temporal modeling 和 spatial representation 三轴 taxonomy 梳理具身世界模型、训练数据、评测指标和代表性能表格</td>
    </tr>
    <tr>
      <td>2026-01</td>
      <td>世界模型</td>
      <td><a href="world-models/lingbot-world/">LingBot-World：视频基础模型到世界模拟器</a></td>
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
      <td><a href="technical-reports/gpt-4o-system-card/">GPT-4o System Card：Omni 安全与多模态评测</a></td>
      <td>如何把端到端 text/audio/image/video omni model 的训练边界、语音输出安全、red teaming、Preparedness Framework 和第三方 agentic evaluation 串成上线前评测体系</td>
      <td><a href="../vlm/">VLM/VLA</a> / <a href="../inference/serving-systems/">推理服务系统</a></td>
    </tr>
    <tr>
      <td>2024-12</td>
      <td>技术报告</td>
      <td><a href="technical-reports/deepseek-v3/">DeepSeek-V3：671B MoE、MLA 与 FP8 训练</a></td>
      <td>如何用 MLA、DeepSeekMoE、auxiliary-loss-free routing、MTP、FP8 mixed precision、DualPipe、SFT/RL 和 R1 reasoning distillation 训练 671B MoE</td>
      <td><a href="../training/low-bit-training-and-numerics/">低比特训练</a> / <a href="../operators/communication-kernels-and-overlap/">通信重叠</a></td>
    </tr>
    <tr>
      <td>2025-01</td>
      <td>技术报告</td>
      <td><a href="technical-reports/deepseek-r1/">DeepSeek-R1：RL 激发推理能力</a></td>
      <td>如何用 GRPO、rule-based reward、长 CoT、rejection sampling、两阶段 RL 和蒸馏把可验证推理任务转成大模型自我探索能力</td>
      <td><a href="../reinforcement-learning/policy-gradient-actor-critic-ppo-grpo/">PPO / GRPO</a> / <a href="../training/preference-data-and-alignment-pitfalls/">对齐失效</a></td>
    </tr>
    <tr>
      <td>2025-05</td>
      <td>技术报告</td>
      <td><a href="technical-reports/qwen3/">Qwen3：Thinking 模式、36T 预训练与蒸馏</a></td>
      <td>如何用 36T token 三阶段预训练、四阶段后训练、GRPO、thinking budget 和 strong-to-weak distillation 构建双模式开源模型家族</td>
      <td><a href="../reinforcement-learning/">强化学习</a> / <a href="../training/post-training-data-engines-and-judge-models/">后训练数据引擎</a></td>
    </tr>
    <tr>
      <td>2025-07</td>
      <td>技术报告</td>
      <td><a href="technical-reports/gemini-2-5/">Gemini 2.5：动态 Thinking 与长上下文 Agent</a></td>
      <td>如何用 native multimodal sparse MoE、teacher distillation、TPUv5p/Pathways、SFT/RM/RL、dynamic thinking 和工具/多模态数据构建 1M context Gemini 2.5</td>
      <td><a href="../training/">训练</a> / <a href="../inference/rag-agents-and-long-context-systems/">Agent / 长上下文</a></td>
    </tr>
    <tr>
      <td>2025-07</td>
      <td>技术报告</td>
      <td><a href="technical-reports/kimi-k2/">Kimi K2：MuonClip、万亿 MoE 与 Agent 数据</a></td>
      <td>如何用 MuonClip 跑稳 1T MoE 预训练，并通过 agentic data synthesis、tool-use SFT、RLVR/self-critique reward 和 checkpoint engine 训练开放 agent 模型</td>
      <td><a href="../training/distributed-training-and-checkpointing/">分布式训练</a> / <a href="../inference/rag-agents-and-long-context-systems/">Agent 系统</a></td>
    </tr>
    <tr>
      <td>2026-03</td>
      <td>技术报告</td>
      <td><a href="technical-reports/incoder-32b/">InCoder-32B：工业代码基础模型与执行验证训练</a></td>
      <td>如何用 Code-Flow 三阶段训练、工业代码召回/清洗、8K -> 32K -> 128K mid-training、execution-grounded SFT 和仿真/编译/性能验证，把 32B dense code model 推向 Verilog、CUDA/Triton、嵌入式、CAD 和代码优化场景</td>
      <td><a href="../operators/triton-programming-model-and-autotuning/">Triton / GPU kernel</a> / <a href="../training/post-training-data-engines-and-judge-models/">后训练数据引擎</a></td>
    </tr>
    <tr>
      <td>2026-04</td>
      <td>技术报告</td>
      <td><a href="technical-reports/nemotron-3-super/">Nemotron 3 Super：Mamba-MoE 与异步多环境 RL</a></td>
      <td>如何用 Mamba-heavy hybrid backbone、LatentMoE、shared-weight MTP、NVFP4 预训练、多环境异步 GRPO 和 FP8/NVFP4 PTQ 构建高吞吐 agentic MoE</td>
      <td><a href="../inference/caching-routing-and-speculative/">推测解码</a> / <a href="../quantization/fp8-and-mixed-precision-serving/">低精度服务</a></td>
    </tr>
    <tr>
      <td>2026-04</td>
      <td>技术报告</td>
      <td><a href="technical-reports/qwen35-omni/">Qwen3.5-Omni：Thinker-Talker 与全模态 Agent</a></td>
      <td>如何用 Thinker-Talker、AuT、Hybrid MoE、ARIA、三阶段预训练和分模块后训练构建可流式语音交互的全模态 agent</td>
      <td><a href="../vlm/">VLM/VLA</a> / <a href="../inference/serving-systems/">推理服务系统</a></td>
    </tr>
    <tr>
      <td>2026-05</td>
      <td>技术报告</td>
      <td><a href="technical-reports/deepseek-v4-pro/">DeepSeek-V4：百万上下文与 FP4 后训练</a></td>
      <td>如何用 CSA/HCA、mHC、Muon、FP4 QAT、异构 KV cache 和 OPD 基础设施把 MoE 大模型推向 1M context 与长流程 agent 场景</td>
      <td><a href="../inference/context-compression-kv-eviction-and-memory-hierarchies/">KV / 长上下文</a> / <a href="../quantization/fp8-and-mixed-precision-serving/">低精度服务</a></td>
    </tr>
  </tbody>
</table>
