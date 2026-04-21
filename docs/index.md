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

## 主题入口

<div class="atlas-card-grid">
  <a class="atlas-card" href="roadmap/index.md">
    <strong>路线图</strong>
    <p>先看整体结构、推荐阅读路径和跨主题关联，再决定从哪条主线切入。</p>
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
</div>

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
    <p>进入新主题时先读总览页和路线图，再进入方法页与工程页，避免只记模型名字。</p>
  </div>
  <div>
    <strong>围绕问题跳读</strong>
    <p>真实问题往往跨主题，例如慢、贵、不稳、掉点，通常要同时看训练、推理、评测和系统章节。</p>
  </div>
</div>

## 本地运行

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdocs serve
```

## 扩写原则

- 每个主题至少覆盖“总览、方法、工程、评测、失败模式”几层。
- 先解释方法关系，再罗列代表工作。
- 重点写适用边界、系统约束和真实取舍，而不是只写定义。
- 能沉淀为流程的内容，优先落到导读页和专题深度页。
- 对长期维护有价值的内容，优先补充真实案例、回归清单和决策复盘。
