# Charles AI notes

一个以 **中文 AI 系统工程笔记** 为核心的文档仓库，站点名为 **Charles AI notes**。  
内容重点不放在零散模型介绍，而是围绕 **训练、推理、量化、算子、世界模型、多模态与具身智能** 这些真正会在工程里反复遇到的问题，持续沉淀成可检索、可扩展的知识库。

项目使用 **MkDocs + Material for MkDocs** 构建，适合本地阅读、持续扩写，以及通过 GitHub Pages 部署成静态站点。

## 内容范围

当前主要专题包括：

- **扩散模型**：训练目标、参数化、采样器、蒸馏、一致性模型、视频扩散
- **VLM**：架构、评测、OCR、图表理解、grounding、屏幕代理、检索系统
- **VLA / 具身智能**：动作表示、示范数据、闭环恢复、Sim2Real、部署安全
- **量化**：PTQ、AWQ、GPTQ、QLoRA、FP8、量化 kernel 与部署权衡
- **训练系统**：Megatron-LM、DeepSpeed、FSDP、checkpoint、输入管线、稳定性排障
- **推理系统**：vLLM、SGLang、TensorRT-LLM、KV cache、长上下文、容量与 SLO
- **算子与编译器**：CUDA、Triton、CUTLASS、FlashAttention、DeepGEMM、profiling
- **对比学习与世界模型**：WM / WAM / VAM、Dreamer、视频世界模型、规划与评测
- **论文导读与研究地图**：阅读路径、研究脉络、方法关系与系统全景

## 仓库结构

```text
.
├── docs/                  # MkDocs 文档正文
├── docs/assets/           # 站点样式与脚本
├── .github/workflows/     # GitHub Pages 部署工作流
├── mkdocs.yml             # 站点导航与配置
├── requirements.txt       # Python 依赖
├── environment.yml        # Conda 环境定义
└── diffusion-model-evolution/
```

其中：

- `docs/` 是主要维护区域；
- `mkdocs.yml` 定义导航结构和站点行为；
- `.github/workflows/deploy-docs.yml` 已配置好 Pages 部署流程。

## 本地预览

### 方式一：使用 venv

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdocs serve
```

默认访问：

```text
http://127.0.0.1:8000
```

### 方式二：使用 Conda

```bash
conda env create -f environment.yml -p ./conda-env
conda activate ./conda-env
mkdocs serve
```

如果你更习惯直接用已有前缀环境，也可以：

```bash
conda activate /absolute/path/to/conda-env
mkdocs serve
```

## 构建静态站点

```bash
mkdocs build
```

构建产物输出到：

```text
site/
```

## GitHub Pages 部署

仓库已包含 GitHub Actions 工作流：

- [deploy-docs.yml](./.github/workflows/deploy-docs.yml)

推送到 `main` 分支后会自动执行：

1. 安装依赖
2. 运行 `mkdocs build --strict`
3. 上传站点产物
4. 部署到 GitHub Pages

在 GitHub 仓库设置中，将 **Pages** 的来源切换为 **GitHub Actions** 即可。

## 维护方式

这个项目更适合按“专题知识树”持续扩写，而不是只记录短笔记。  
目前的写法重点是：

- **从系统视角组织内容**
- **强调工程判断与取舍**
- **把方法、实现、评测和部署放到同一条链上**
- **尽量让单篇文章既能独立阅读，也能纳入整体导航**

如果你要继续扩充，推荐优先在现有专题下增加：

- 更细的小标题
- 方法对照表
- 失效模式与排障经验
- 系统实现路径
- 论文与代码仓库之间的映射

## 适合谁

这个仓库尤其适合：

- 做大模型训练与推理基础设施的人
- 做 VLM / VLA / 世界模型系统落地的人
- 想系统梳理 CUDA / Triton / kernel / 编译栈的人
- 想把 AI 论文阅读沉淀为工程知识库的人

## 备注

当前文档以中文为主，内容会持续增补与重构。  
如果你只想快速进入某个方向，建议直接从站点首页或相应专题的 `index.md` 开始。
