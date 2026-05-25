<div class="atlas-hero atlas-dashboard-hero">
  <div class="atlas-dashboard-hero__main">
    <div class="atlas-kicker">Charles AI notes</div>
    <h1>AI 系统与世界模型阅读仪表盘</h1>
    <p class="atlas-lead">围绕世界模型高效训练，把 VLM/VLA、具身智能、训练、推理、量化、扩散、强化学习和算子栈放到同一张工程地图里。先选问题，再进入对应专题；长文负责细节，首页只负责帮你快速找到下一站。</p>
  </div>
  <div class="atlas-hero-actions" aria-label="常用入口">
    <a href="world-models/efficient-training-roadmap/">主线速读</a>
    <a href="training/">训练系统</a>
    <a href="paper-deep-dives/">论文专题</a>
    <a href="glossary/">术语表</a>
  </div>
</div>

<div class="atlas-dashboard-strip" aria-label="最短路径">
  <a href="world-models/efficient-training-roadmap/">
    <span>只读一页</span>
    <strong>世界模型高效训练路线图</strong>
  </a>
  <a href="world-models/world-model-efficient-training-experiment-report/">
    <span>要做实验</span>
    <strong>完整实验报告样例</strong>
  </a>
  <a href="references/site-judgement-principles/">
    <span>要查证据</span>
    <strong>本站判断原则</strong>
  </a>
  <a href="references/claim-ledger/">
    <span>要核结论</span>
    <strong>Claim Ledger</strong>
  </a>
</div>

## 读者入口

<div class="atlas-reader-paths">
  <a class="atlas-reader-path atlas-reader-path--primary" href="world-models/efficient-training-roadmap/">
    <span>主线速读</span>
    <strong>先抓住成本、证据和闭环</strong>
    <p>适合第一次进站，先看世界模型高效训练如何把数据、token、显存、通信、rollout 和评测串起来。</p>
  </a>
  <a class="atlas-reader-path" href="inference/">
    <span>工程排障</span>
    <strong>带着瓶颈进入系统专题</strong>
    <p>慢、贵、不稳、显存不够、吞吐不够时，从训练、推理、量化和算子页面拆成本账。</p>
  </a>
  <a class="atlas-reader-path" href="paper-deep-dives/">
    <span>论文精读</span>
    <strong>把单篇论文接回工程判断</strong>
    <p>先看论文解决的问题、方法结构、训练细节和边界，再回填到主题路线图。</p>
  </a>
  <a class="atlas-reader-path" href="foundations/">
    <span>概念补课</span>
    <strong>遇到硬词就回基础层</strong>
    <p>张量、Transformer、概率、优化、数值和运行时是全站公共语言，不需要一次读完。</p>
  </a>
</div>

## 专题矩阵

<div class="atlas-topic-matrix">
  <a class="atlas-topic-card atlas-topic-card--primary" href="world-models/">
    <span>主线</span>
    <strong>世界模型</strong>
    <p>状态、动作、latent rollout、RSSM/Dreamer、JEPA、WAM/VAM、视频世界模型和闭环评测。</p>
    <small>先读：从零理解世界模型、高效训练路线图</small>
  </a>
  <a class="atlas-topic-card" href="vlm/">
    <span>接口</span>
    <strong>VLM/VLA</strong>
    <p>视觉表征、连接器、视频记忆、动作 token、策略学习和世界模型接口。</p>
    <small>适合：多模态到动作链路</small>
  </a>
  <a class="atlas-topic-card" href="embodied-ai/">
    <span>闭环</span>
    <strong>具身智能</strong>
    <p>任务谱系、相机深度、仿真数据、规划控制、安全部署和现实失败模式。</p>
    <small>适合：机器人/VLA 系统判断</small>
  </a>
  <a class="atlas-topic-card" href="training/">
    <span>系统</span>
    <strong>训练</strong>
    <p>目标函数、数据管线、分布式训练、稳定性、checkpoint、实验治理和训练经济学。</p>
    <small>适合：训练成本和质量回归</small>
  </a>
  <a class="atlas-topic-card" href="inference/">
    <span>服务</span>
    <strong>推理</strong>
    <p>请求生命周期、prefill/decode、KV cache、batching、routing、SLO 和容量规划。</p>
    <small>适合：线上延迟、吞吐和成本</small>
  </a>
  <a class="atlas-topic-card" href="quantization/">
    <span>低比特</span>
    <strong>量化</strong>
    <p>PTQ、GPTQ、AWQ、SmoothQuant、QLoRA、QAT、FP8、KV cache 和部署取舍。</p>
    <small>适合：显存、吞吐和精度边界</small>
  </a>
  <a class="atlas-topic-card" href="operators/">
    <span>底层</span>
    <strong>算子与编译器</strong>
    <p>CUDA、Triton、CUTLASS、Attention/GEMM kernel、通信重叠和 profiling。</p>
    <small>适合：性能瓶颈和硬件账</small>
  </a>
  <a class="atlas-topic-card" href="diffusion/">
    <span>生成</span>
    <strong>扩散模型</strong>
    <p>DDPM、Score SDE、采样器、少步蒸馏、Rectified Flow、视频与多模态扩散。</p>
    <small>适合：生成式模拟和视频模型</small>
  </a>
  <a class="atlas-topic-card" href="reinforcement-learning/">
    <span>决策</span>
    <strong>强化学习</strong>
    <p>MDP、Bellman、policy gradient、PPO/GRPO、RLVR 和 verl 训练流程。</p>
    <small>适合：reward、policy 和闭环优化</small>
  </a>
  <a class="atlas-topic-card" href="foundations/">
    <span>基础</span>
    <strong>基础知识</strong>
    <p>张量、shape、Transformer、概率、优化、生成、数值、显存和运行时。</p>
    <small>适合：读论文前补公共语言</small>
  </a>
  <a class="atlas-topic-card" href="paper-deep-dives/">
    <span>论文</span>
    <strong>论文专题讲解</strong>
    <p>按方向拆解关键论文和技术报告，重点看方法结构、实验口径和不能外推项。</p>
    <small>适合：从论文变成工程判断</small>
  </a>
  <a class="atlas-topic-card" href="thinking-exploration/">
    <span>探索</span>
    <strong>思考探索</strong>
    <p>芯片平台、软件栈、产品约束、开源工具链和工程路线判断。</p>
    <small>适合：具体平台和路线选择</small>
  </a>
</div>

## 近期补强线索

<div class="atlas-signal-grid">
  <a class="atlas-signal-card" href="paper-deep-dives/embodied-ai/depth-anything-3/">
    <span>具身几何</span>
    <strong>Depth Anything 3</strong>
    <p>把 depth、ray、camera 和 3D geometry 放进具身系统的几何状态层。</p>
  </a>
  <a class="atlas-signal-card" href="paper-deep-dives/foundations/magiattention/">
    <span>高效训练</span>
    <strong>MagiAttention / Attn-QAT</strong>
    <p>超长上下文分布式 attention 与低比特 attention 要同时看 kernel、通信和 backward。</p>
  </a>
  <a class="atlas-signal-card" href="paper-deep-dives/technical-reports/deepseek-v3/">
    <span>技术报告</span>
    <strong>DeepSeek-V3 / Kimi K2</strong>
    <p>MoE、FP8、MuonClip、agentic data synthesis 和训练系统如何进入大模型报告。</p>
  </a>
  <a class="atlas-signal-card" href="paper-deep-dives/technical-reports/qwen35-omni/">
    <span>实时系统</span>
    <strong>Qwen3.5-Omni</strong>
    <p>Thinker-Talker、流式语音、多模态 agent 和低延迟后训练如何组合。</p>
  </a>
  <a class="atlas-signal-card" href="thinking-exploration/qualcomm-xr2-gen2/">
    <span>端侧部署</span>
    <strong>XR2 Gen 2 / QCS8450</strong>
    <p>端侧 AI 平台、QNN、LiteRT、ONNX Runtime 和 proxy profile 如何进入部署判断。</p>
  </a>
</div>

## 按问题选入口

<div class="atlas-question-grid">
  <a href="foundations/">
    <span>概念卡住</span>
    <strong>基础知识 + 术语表</strong>
    <p>先补张量、Transformer、优化、概率和运行时，不要硬刷长文。</p>
  </a>
  <a href="training/">
    <span>训练不稳</span>
    <strong>训练主线</strong>
    <p>检查数据版本、目标函数、分布式状态、低精度路径、checkpoint 和评测回流。</p>
  </a>
  <a href="inference/">
    <span>服务慢贵</span>
    <strong>推理系统</strong>
    <p>把 queue、prefill、decode、KV cache、batching、routing 和 SLO 分开看。</p>
  </a>
  <a href="quantization/">
    <span>显存不够</span>
    <strong>量化路线</strong>
    <p>区分权重、激活、KV cache、kernel 和质量回归，不把低比特当单点开关。</p>
  </a>
  <a href="world-models/">
    <span>多模态闭环</span>
    <strong>世界模型</strong>
    <p>检查状态、动作、rollout、消费方和 closed-loop gain 是否真的连通。</p>
  </a>
  <a href="paper-deep-dives/">
    <span>论文太多</span>
    <strong>论文专题</strong>
    <p>先看它省什么成本、证据强到哪里、哪些结论不能外推。</p>
  </a>
</div>

!!! quote "本站核心 thesis"
    世界模型效率不是单一训练技巧，而是表示、交互、系统与验证共同决定的闭环问题。全站所有专题最终都要回到同一组问题：省什么成本、牺牲什么、证据能证明什么、什么时候不能外推。

??? note "维护约定和新增页面 Checklist"
    总览页负责讲地图、边界和路线；专题页负责讲机制、工程实现、评测口径和失效模式。新增内容要同步接入导航、索引、术语表或参考文献，避免页面成为孤岛。

    | 检查项 | 通过标准 |
    | --- | --- |
    | 接回主线 | 写清它服务世界模型高效训练的哪项成本 |
    | 成本项 | 至少说明数据、token、显存、通信、rollout 或验证中的一项 |
    | 证据来源 | 论文、日志、fixture、benchmark、ablation、system throughput 或 closed-loop 分开写 |
    | 反例和边界 | 说明何时收益会消失，或何时会伤动作、风险、长时一致性 |
    | 下一站 | 指向相邻主题页或论文专题 |
    | 术语内链 | 首次出现的核心术语尽量能跳到基础页或术语表 |
    | 图文合适 | 图片有图源、原图意和本站读法；表格保留原字段时注明表源 |
    | 可搜索 | 中英文关键词、缩写和别名要在正文自然出现 |

    论文专题页还要区分论文证据、本站推断和工程外推；关键主线页尽量提供可复算数字账、失败 replay 和验收指标。

## 本地运行

```bash
make serve
```

`make serve` 会启用热更新。若只改单页正文并希望更快，可用 `make serve-fast`；如果页面状态异常，停掉旧进程后重新运行 `make serve`。
