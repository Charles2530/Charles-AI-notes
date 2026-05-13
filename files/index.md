<div class="atlas-hero">
  <div class="atlas-kicker">Charles AI notes</div>
  <h1>AI 系统知识库</h1>

  <p class="atlas-lead">面向本地阅读、持续扩写与工程落地的中文 AI 知识库。内容围绕训练、推理、量化、多模态、世界模型、具身智能与底层算子栈展开，重点不是堆论文名，而是把方法、系统约束和工程判断接起来。</p>

  <div class="atlas-chip-row">
    <span class="atlas-chip">先建地图</span>
    <span class="atlas-chip">再读论文</span>
    <span class="atlas-chip">回到工程</span>
    <span class="atlas-chip">持续复盘</span>
  </div>
</div>

!!! note "先选一个入口"
    第一次读不必顺着左侧目录从头刷。概念卡住先看 [基础知识](foundations/index.md) 和 [术语表](glossary/index.md)；工程卡住先看训练、推理、量化或算子；论文卡住先看 [论文专题讲解](paper-deep-dives/index.md) 和 [参考文献总表](references/index.md)。

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
