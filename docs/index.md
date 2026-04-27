<div class="atlas-hero">
  <div class="atlas-kicker">Charles AI notes</div>
  <h1>Charles AI notes</h1>

<p class="atlas-lead">一个面向本地阅读、持续扩写与工程落地的中文 AI 系统知识库，围绕训练、推理、量化、多模态、世界模型、具身智能与底层算子栈持续沉淀工程笔记。</p>

  <div class="atlas-chip-row">
    <span class="atlas-chip">方法地图</span>
    <span class="atlas-chip">系统工程</span>
    <span class="atlas-chip">部署与评测</span>
    <span class="atlas-chip">可持续扩写</span>
  </div>
</div>

## 站点定位

<div class="atlas-meta-grid">
  <div>
    <strong>研究地图</strong>
    <p>用“概念、方法、系统、风险、选型”五层结构组织主题，不把文档写成论文摘要堆砌。</p>
  </div>
  <div>
    <strong>工程工作台</strong>
    <p>把训练、推理、量化、评测、数据回流和部署问题放到同一张系统图里，适合做项目复盘与方案设计。</p>
  </div>
  <div>
    <strong>持续底稿</strong>
    <p>适合本地长期扩写，既能读方法，也能沉淀自己的实验记录、系统决策和失败案例。</p>
  </div>
</div>

## 内容地图

这套站点按“公共底座、方法族、系统工程、评测治理、论文索引”组织，而不是按论文发布时间简单堆叠。阅读时可以把每个主题放到同一张图里：

| 层级 | 负责回答的问题 | 对应入口 |
| --- | --- | --- |
| 公共底座 | 张量、模型结构、训练、数值和运行时这些共通概念是什么 | [基础知识](foundations/index.md)、[术语表](glossary/index.md) |
| 方法族 | 某类模型或技术为什么出现、核心假设是什么 | [扩散模型](diffusion/index.md)、[VLM](vlm/index.md)、[VLA](vla/index.md)、[世界模型](world-models/index.md)、[对比学习](contrastive-learning/index.md) |
| 系统工程 | 训练、推理、量化、算子和部署如何落地 | [训练](training/index.md)、[推理](inference/index.md)、[量化](quantization/index.md)、[算子与编译器](operators/index.md) |
| 评测治理 | 怎么判断方法真的有效，线上是否稳定、可控、可回滚 | 各主题评测页、[在线评测](inference/observability-and-online-evaluation.md)、[论文复现](paper-guides/reproducibility-and-replication-guide.md) |
| 论文索引 | 想深入读原文时从哪里开始 | [论文导读](paper-guides/index.md)、[参考文献总表](references/index.md) |

## 章节风格约定

为了让各页面读起来像同一套文档，后续扩写尽量遵循同一粒度：

- 每页优先控制在 5-8 个二级章节，右侧目录只承担主线导航，不把每个小点都升成 `##`。
- 总览页负责讲地图、边界和路线；专题页负责讲机制、工程实现、评测与失效模式。
- 单节内容保持适中：先讲问题和判断标准，再放公式、例子、系统约束或论文脉络。
- 避免在每页末尾重复模板化“小结”；如果需要收口，直接给出可执行清单或决策判断。
- 跨主题内容优先互链，例如低比特训练、FP8 Kernel、量化服务和推理成本应互相指向，而不是各写一套孤立解释。

## 主题入口

<div class="atlas-card-grid">
  <a class="atlas-card" href="foundations/index.md">
    <strong>基础知识</strong>
    <p>先打通张量、卷积、Transformer、概率生成、训练优化、数值精度和运行时这些公共底座。</p>
  </a>
  <a class="atlas-card" href="training/index.md">
    <strong>训练</strong>
    <p>从预训练、SFT、对齐到 Megatron-LM、DeepSpeed、输入管线、稳定性与 checkpoint。</p>
  </a>
  <a class="atlas-card" href="inference/index.md">
    <strong>推理</strong>
    <p>覆盖服务系统、运行时、KV cache、解耦式 prefill、在线评测和容量治理。</p>
  </a>
  <a class="atlas-card" href="operators/index.md">
    <strong>算子与编译器</strong>
    <p>从 CUDA、Triton、Attention Kernel、通信重叠到 profiling、PTX 和硬件感知排障。</p>
  </a>
  <a class="atlas-card" href="world-models/index.md">
    <strong>世界模型</strong>
    <p>从 RSSM、Dreamer 到 WM/WAM/VAM、视频世界模型、风险规划与反事实数据引擎。</p>
  </a>
  <a class="atlas-card" href="quantization/index.md">
    <strong>量化</strong>
    <p>覆盖 PTQ、QAT、QLoRA、FP8、KV cache 量化与部署取舍。</p>
  </a>
  <a class="atlas-card" href="diffusion/index.md">
    <strong>扩散模型</strong>
    <p>从 DDPM、Score SDE、采样器到少步蒸馏、Rectified Flow、视频与多模态扩散。</p>
  </a>
  <a class="atlas-card" href="vlm/index.md">
    <strong>VLM</strong>
    <p>覆盖架构训练、OCR、图表推理、Grounding、屏幕代理、检索和长尾评测。</p>
  </a>
  <a class="atlas-card" href="vla/index.md">
    <strong>VLA</strong>
    <p>从视觉语言动作模型、动作表示、遥操作数据到 Sim2Real、Benchmark 和部署安全。</p>
  </a>
  <a class="atlas-card" href="embodied-ai/index.md">
    <strong>具身智能</strong>
    <p>关注任务谱系、规划控制、安全部署、人机协作、数据引擎和现实失败模式。</p>
  </a>
  <a class="atlas-card" href="contrastive-learning/index.md">
    <strong>对比学习</strong>
    <p>从 InfoNCE、负样本、增广、自蒸馏到多模态检索和表示评测。</p>
  </a>
  <a class="atlas-card" href="references/index.md">
    <strong>参考文献总表</strong>
    <p>按扩散、VLM/VLA、世界模型、量化、训练系统、推理和算子整理核心论文与项目。</p>
  </a>
</div>

## 按问题选入口

| 你现在的问题 | 先读哪里 | 再接哪里 |
| --- | --- | --- |
| 基础概念不稳，读论文容易卡住 | [基础知识](foundations/index.md) | [术语表](glossary/index.md) |
| 想理解大模型怎么训出来 | [训练总览](training/index.md) | [低比特训练](training/low-bit-training-and-numerics.md)、[分布式训练](training/distributed-training-and-checkpointing.md) |
| 线上服务慢、贵、不稳定 | [推理总览](inference/index.md) | [推理服务系统](inference/serving-systems.md)、[vLLM/SGLang/TensorRT-LLM](inference/serving-runtimes-vllm-sglang-and-tensorrt-llm.md) |
| 模型太大，显存或吞吐不够 | [量化总览](quantization/index.md) | [量化运行时](quantization/quantization-runtimes-and-frameworks.md)、[低精度 Kernel](operators/low-precision-and-quantized-kernels.md) |
| 想做图文、多模态、文档理解 | [VLM 总览](vlm/index.md) | [文档理解与 OCR](vlm/document-understanding-and-ocr.md)、[图表推理](vlm/charts-tables-and-structured-reasoning.md) |
| 想做机器人、VLA 或具身系统 | [VLA 总览](vla/index.md) | [具身智能](embodied-ai/index.md)、[世界模型](world-models/index.md) |
| 想系统读论文和找参考 | [论文导读](paper-guides/index.md) | [参考文献总表](references/index.md) |

## 推荐阅读路径

<div class="atlas-card-grid">
  <a class="atlas-card" href="diffusion/index.md">
    <strong>路线 A：生成与多模态</strong>
    <p>适合从扩散模型出发，继续读到 VLM、VLA、具身智能和多模态系统。</p>
  </a>
  <a class="atlas-card" href="training/index.md">
    <strong>路线 B：训练到部署</strong>
    <p>适合关心数据系统、并行训练、量化、推理服务和线上优化的读者。</p>
  </a>
  <a class="atlas-card" href="world-models/index.md">
    <strong>路线 C：表征到智能体</strong>
    <p>适合把对比学习、世界模型、规划、机器人和具身系统连起来理解。</p>
  </a>
</div>

## 这套站点适合怎么用

### 1. 作为技术地图

**适合在进入一个新主题时先建立**：

- 这个主题在解决什么问题
- 代表方法之间是什么关系
- 训练、推理和部署分别卡在哪

### 2. 作为论文导读手册

当你知道关键词，但不知道该先读哪些论文时，可以直接从：

- [论文导读总览](paper-guides/index.md)
- [训练与系统导读](paper-guides/training-systems-reading-guide.md)
- [世界模型导读](paper-guides/world-models-reading-guide.md)
- [量化与高效推导读](paper-guides/quantization-efficient-serving-reading-guide.md)

进入。

## 使用建议

<div class="section-grid">
  <div>
    <strong>先建结构，再看细节</strong>
    <p>进入新主题时先读基础知识和总览页，再进入方法页与工程页，避免只记模型名字。</p>
  </div>
  <div>
    <strong>围绕问题跳读</strong>
    <p>真实问题往往跨主题，例如慢、贵、不稳、掉点，通常要同时看训练、推理、评测和系统章节。</p>
  </div>
</div>

## 本地运行

```bash
make serve
```

`make serve` 会启用热更新。若只改单页正文并希望更快，可用 `make serve-fast`；如果页面状态异常，停掉旧进程后重新运行 `make serve`。

## 扩写原则

- 每个主题至少覆盖“总览、方法、工程、评测、失败模式”几层。
- 先解释方法关系，再罗列代表工作。
- 重点写适用边界、系统约束和真实取舍，而不是只写定义。
- 能沉淀为流程的内容，优先落到导读页和专题深度页。
- 对长期维护有价值的内容，优先补充真实案例、回归清单和决策复盘。
