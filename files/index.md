<div class="atlas-hero">
  <div class="atlas-kicker">Charles AI notes</div>
  <h1>世界模型高效训练技术知识库</h1>

  <p class="atlas-lead">以世界模型的高效训练技术为主线，串起 VLM/VLA、具身智能、训练系统、推理、量化、扩散模型、强化学习和底层算子栈。这里不是只保留单一方向，而是把基础知识和关键旁支放到同一张工程地图里：哪些内容支撑状态表示，哪些内容降低训练成本，哪些内容决定 rollout 和闭环评测是否可用。</p>

  <div class="atlas-chip-row">
    <span class="atlas-chip">世界模型</span>
    <span class="atlas-chip">高效训练</span>
    <span class="atlas-chip">系统约束</span>
    <span class="atlas-chip">闭环验证</span>
  </div>
</div>

!!! note "先选一个入口"
    如果你是围绕主线阅读，先看 [世界模型高效训练技术路线图](world-models/efficient-training-roadmap.md)。如果概念卡住，再回到 [基础知识](foundations/index.md) 和 [术语表](glossary/index.md)；工程卡住看训练、推理、量化或算子；论文卡住看 [论文专题讲解](paper-deep-dives/index.md) 和 [参考文献总表](references/index.md)。主线会帮你决定旁支内容该读到什么深度。

## 主线与支撑层

本站的主线是“如何用更少真实交互、更少视频 token、更低显存/通信/rollout 成本，训练出对动作敏感、能长时推演、能改善决策的世界模型”。但高效训练不是孤立专题，它需要一组支撑层共同工作。

| 层级 | 回答的问题 | 推荐入口 |
| --- | --- | --- |
| 主线枢纽 | 世界模型高效训练到底省什么成本，技术栈如何组织 | [世界模型高效训练技术路线图](world-models/efficient-training-roadmap.md) |
| 状态与动作接口 | 视觉状态、视频 latent、动作 token、VLA 数据如何进入世界模型 | [VLM/VLA](vlm/index.md)、[VLA 专题](vla/index.md) |
| 建模与数据闭环 | RSSM/Dreamer、WAM/VAM、反事实、near-miss 和失败回流如何工作 | [世界模型](world-models/index.md)、[世界模型数据引擎](world-models/data-engines-and-self-improvement.md) |
| 训练系统支撑 | 长序列、低比特、分布式、checkpoint 和数据管线如何降低训练成本 | [训练](training/index.md)、[算子与编译器](operators/index.md) |
| 推理与部署支撑 | KV cache、batch、量化、蒸馏和 SLO 如何决定 rollout 是否可用 | [推理](inference/index.md)、[量化](quantization/index.md) |
| 必要背景 | 张量、Attention、概率、优化、强化学习、扩散模型等公共语言 | [基础知识](foundations/index.md)、[扩散模型](diffusion/index.md)、[强化学习](reinforcement-learning/index.md) |

## 怎么使用这个知识库

<div class="atlas-meta-grid">
  <div>
    <strong>10 分钟抓主线</strong>
    <p>先读高效训练路线图，再用端到端案例落地，只回答一个问题：这项技术到底降低了数据、token、显存、通信、rollout 还是评测成本。</p>
  </div>
  <div>
    <strong>1 小时补概念</strong>
    <p>遇到硬词不要硬读，先跳到基础知识或术语表，把它放回表示、目标、执行、评测这几层。</p>
  </div>
  <div>
    <strong>半天查工程问题</strong>
    <p>带着具体问题进入训练、推理、量化或算子页面，优先看瓶颈、取舍、失败模式和检查清单。</p>
  </div>
  <div>
    <strong>精读一篇论文</strong>
    <p>先用专题页定位它解决的系统问题，再读论文讲解，最后回到主题页更新自己的判断框架。</p>
  </div>
</div>

!!! tip "读完首页先做一个选择"
    如果你不知道从哪里开始，默认先走 [世界模型高效训练技术路线图](world-models/efficient-training-roadmap.md)。如果目标很明确，再选身份：想补基础，就走 [基础知识](foundations/index.md)；想做工程，就走 [训练](training/index.md)、[推理](inference/index.md)、[量化](quantization/index.md) 或 [算子](operators/index.md)；想理解机器人闭环，就走 [具身智能](embodied-ai/index.md)；想跟论文，就走 [论文专题讲解](paper-deep-dives/index.md)。

## 读者路径

<div class="section-grid">
  <div>
    <strong>查概念</strong>
    <p>从张量、Transformer、概率生成、优化、数值和运行时建立公共语言，避免读论文时每个词都要重新补课。</p>
  </div>
  <div>
    <strong>做工程</strong>
    <p>围绕显存、吞吐、低精度、并行、服务稳定性和线上评测，把训练到部署的链路串起来。</p>
  </div>
  <div>
    <strong>读论文</strong>
    <p>先用专题页确定论文解决的问题，再看数据、模型、训练、实验和工程边界，不把摘要当理解。</p>
  </div>
  <div>
    <strong>看闭环</strong>
    <p>把 VLM/VLA、强化学习、世界模型和具身智能放进感知、规划、动作、反馈和失败恢复的同一条线上。</p>
  </div>
</div>

## 主题入口

<div class="atlas-card-grid">
  <a class="atlas-card" href="world-models/efficient-training-roadmap/">
    <strong>主线：世界模型高效训练</strong>
    <p>从数据效率、视觉 token、动作条件、长序列训练、低精度、rollout 到闭环评测建立总地图。</p>
  </a>
  <a class="atlas-card" href="world-models/action-conditioned-video-world-model-training-case/">
    <strong>端到端训练案例</strong>
    <p>用一个动作条件视频世界模型案例串起数据、动作、训练、推理、量化和闭环评测。</p>
  </a>
  <a class="atlas-card" href="foundations/">
    <strong>基础知识</strong>
    <p>张量、Shape、Transformer、概率模型、优化、数值和运行时。</p>
  </a>
  <a class="atlas-card" href="training/">
    <strong>训练</strong>
    <p>预训练、SFT、对齐、数据管线、分布式训练、稳定性和实验治理。</p>
  </a>
  <a class="atlas-card" href="inference/">
    <strong>推理</strong>
    <p>服务系统、运行时、KV cache、解耦 prefill、容量规划和在线评测。</p>
  </a>
  <a class="atlas-card" href="operators/">
    <strong>算子与编译器</strong>
    <p>CUDA、Triton、CUTLASS、Attention Kernel、通信重叠和 profiling。</p>
  </a>
  <a class="atlas-card" href="quantization/">
    <strong>量化</strong>
    <p>PTQ、QAT、QLoRA、FP8、KV cache 量化、低精度 Kernel 与部署取舍。</p>
  </a>
  <a class="atlas-card" href="diffusion/">
    <strong>扩散模型</strong>
    <p>DDPM、Score SDE、采样器、少步蒸馏、Rectified Flow 和视频扩散。</p>
  </a>
  <a class="atlas-card" href="vlm/">
    <strong>VLM/VLA</strong>
    <p>图文对齐、视觉连接器、视频时序、动作 token、策略学习和闭环恢复。</p>
  </a>
  <a class="atlas-card" href="world-models/">
    <strong>世界模型</strong>
    <p>RSSM、Dreamer、WAM/VAM、视频世界模型、反事实数据和风险规划。</p>
  </a>
  <a class="atlas-card" href="embodied-ai/">
    <strong>具身智能</strong>
    <p>任务谱系、相机深度、仿真数据、规划控制、安全部署和现实失败模式。</p>
  </a>
  <a class="atlas-card" href="reinforcement-learning/">
    <strong>强化学习</strong>
    <p>MDP、Bellman、policy gradient、PPO/GRPO、RLVR 与 verl 训练流程。</p>
  </a>
  <a class="atlas-card" href="paper-deep-dives/">
    <strong>论文专题讲解</strong>
    <p>按方向拆解关键论文和技术报告，重点看方法结构、训练细节和边界。</p>
  </a>
  <a class="atlas-card" href="thinking-exploration/">
    <strong>思考探索</strong>
    <p>芯片平台、软件栈、产品约束、开源工具链和工程路线判断。</p>
  </a>
</div>

## 近期补强线索

<div class="atlas-route-grid">
  <div class="atlas-route">
    <span>具身几何</span>
    <strong><a href="paper-deep-dives/embodied-ai/depth-anything-3/">Depth Anything 3</a></strong>
    <p>任意视角 depth / ray / camera / 3D geometry 如何成为具身系统的几何状态层。</p>
    <small>接到 <a href="embodied-ai/cameras-depth-and-robot-vision/">相机、深度与机器人视觉</a>、<a href="world-models/development-roadmap-data-simulation-and-evaluation/">世界模型开发路线</a></small>
  </div>
  <div class="atlas-route">
    <span>高效训练</span>
    <strong><a href="paper-deep-dives/foundations/magiattention/">MagiAttention</a> / <a href="paper-deep-dives/foundations/attn-qat/">Attn-QAT</a></strong>
    <p>超长上下文分布式 attention 和 4-bit attention 训练要一起看 kernel、通信和 backward。</p>
    <small>接到 <a href="training/">训练</a>、<a href="operators/">算子与编译器</a></small>
  </div>
  <div class="atlas-route">
    <span>技术报告</span>
    <strong><a href="paper-deep-dives/technical-reports/deepseek-v3/">DeepSeek-V3</a> / <a href="paper-deep-dives/technical-reports/kimi-k2/">Kimi K2</a></strong>
    <p>MoE、FP8、MuonClip、agentic data synthesis 和训练系统如何进入大模型报告。</p>
    <small>接到 <a href="training/">训练</a>、<a href="inference/">推理</a></small>
  </div>
  <div class="atlas-route">
    <span>多模态实时系统</span>
    <strong><a href="paper-deep-dives/technical-reports/qwen35-omni/">Qwen3.5-Omni</a></strong>
    <p>Thinker-Talker、流式语音、多模态 agent 和低延迟后训练如何组合。</p>
    <small>接到 <a href="vlm/">VLM/VLA</a>、<a href="inference/">推理</a></small>
  </div>
  <div class="atlas-route">
    <span>端侧部署</span>
    <strong><a href="thinking-exploration/qualcomm-xr2-gen2/">XR2 Gen 2 / QCS8450</a></strong>
    <p>端侧 AI 平台、QNN / LiteRT / ONNX Runtime 和 proxy profile 如何进入部署判断。</p>
    <small>接到 <a href="quantization/">量化</a>、<a href="inference/">推理</a></small>
  </div>
</div>

## 按问题选入口

<div class="atlas-route-list">
  <a class="atlas-route atlas-route--compact" href="foundations/">
    <span>概念卡住</span>
    <strong>先读基础知识</strong>
    <p>再接术语表，先把张量、Transformer、优化和运行时语言补齐。</p>
  </a>
  <a class="atlas-route atlas-route--compact" href="training/">
    <span>想理解训练</span>
    <strong>先读训练主线</strong>
    <p>再接低比特训练、分布式训练、数据系统和实验治理。</p>
  </a>
  <a class="atlas-route atlas-route--compact" href="inference/">
    <span>服务慢贵不稳</span>
    <strong>先读推理系统</strong>
    <p>再接 serving、容量规划、KV cache、路由和在线评测。</p>
  </a>
  <a class="atlas-route atlas-route--compact" href="quantization/">
    <span>显存吞吐不够</span>
    <strong>先读量化路线图</strong>
    <p>再接低精度 kernel、量化 runtime、评测和回退策略。</p>
  </a>
  <a class="atlas-route atlas-route--compact" href="world-models/">
    <span>多模态到闭环</span>
    <strong>先读世界模型</strong>
    <p>再接 VLM/VLA、具身智能、WAM/VAM 和仿真数据引擎。</p>
  </a>
  <a class="atlas-route atlas-route--compact" href="embodied-ai/">
    <span>机器人 / VLA</span>
    <strong>先读具身智能</strong>
    <p>再接 VLA 数据与策略、闭环恢复和世界模型开发路线。</p>
  </a>
  <a class="atlas-route atlas-route--compact" href="paper-deep-dives/">
    <span>系统读论文</span>
    <strong>先读论文专题讲解</strong>
    <p>再接参考文献总表，用专题页定位方法、训练细节和边界。</p>
  </a>
</div>

## 维护约定

- 总览页负责讲地图、边界和路线；专题页负责讲机制、工程实现、评测口径和失效模式。
- 论文专题优先解释“为什么这样设计”，训练阶段、数据构造、推理约束和实验口径要比普通摘要更具体。
- 表格只保留能支撑判断的内容；长表放到可折叠区或引用总表里，正文优先讲结论。
- 图片优先使用论文原图、官方图或明确来源的截图；信息量少、长宽比极端的图片用较小展示宽度。
- 新增内容要同步接入导航、索引、术语表或参考文献，避免页面成为孤岛。

## 本地运行

```bash
make serve
```

`make serve` 会启用热更新。若只改单页正文并希望更快，可用 `make serve-fast`；如果页面状态异常，停掉旧进程后重新运行 `make serve`。
