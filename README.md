# Charles AI notes

Charles AI notes 是一份可自由查阅的中文 AI 技术入门手册。内容连接数学、模型计算、训练、推理、生成模型、强化学习及多模态与具身系统，通过数值例子、实现和论文精读帮助读者理解机制及其适用条件。
站点使用 MkDocs 与 Material for MkDocs 构建，支持本地阅读、全文检索、静态站点部署和可选的本地 AI 阅读助手。

## 当前内容

站点从[技术地图](./files/index.md)进入，按技术问题组织八个主要方向，并提供工具与查阅入口。

| 方向 | 内容 |
| --- | --- |
| [基础知识](./files/foundations/index.md) | 张量、梯度、数据与目标；[数学工具](./files/prerequisite-math/index.md)按需查阅 |
| [模型结构](./files/model-architectures/index.md) | 线性层、MLP、卷积、Transformer、MoE 与 SSM |
| [训练](./files/training/index.md) | 参数更新、预训练、微调、数据系统和分布式训练 |
| [推理与效率](./files/inference/index.md) | 解码、缓存、调度、量化、算子与编译器 |
| [生成模型](./files/diffusion/index.md) | 扩散目标、网络、采样、Flow Matching 与蒸馏 |
| [强化学习](./files/reinforcement-learning/index.md) | 交互、价值、策略优化、离线数据与应用 |
| [多模态与具身](./files/world-models-and-embodied-ai/index.md) | VLM、世界模型、VLA 与机器人系统 |
| [论文精读](./files/paper-deep-dives/index.md) | 连接基础机制、方法实现和原始实验证据 |
| [工具与查阅](./files/technical-reference/index.md) | 术语、知识 QA、探索文章与可运行示例 |

[基础机制示例](./files/assets/examples/foundations/mechanisms.py)仅使用 Python 标准库，可以直接运行，对照线性层、梯度、因果注意力与解码的数值结果。

## 站点功能

- 中文全文检索，并提供搜索建议、高亮和兜底搜索
- 术语提示与术语表掌握状态标记
- 页面阅读进度、侧边栏阅读状态和论文专题折叠记忆
- 字号切换、正文宽屏、沉浸阅读、左右栏折叠、页首/页尾跳转与链接复制
- 当前页面打印或导出 PDF
- 图片灯箱、Mermaid 图表、MathJax 公式、代码复制和表格复制/折叠
- 论文日期排序、同专题页面推荐、图片懒加载和静态资源性能优化
- 仅在本地预览中启用的 AI 阅读助手，可围绕整篇文章或划线内容连续提问

阅读偏好、阅读状态和术语掌握状态保存在浏览器本地，不会写回仓库。

## 仓库结构

```text
.
├── files/                         # MkDocs 内容根目录
│   ├── prerequisite-math/         # 前置数学知识
│   ├── foundations/               # 基础概念
│   ├── training-and-inference-systems/
│   ├── reinforcement-learning/
│   ├── world-models-and-embodied-ai/
│   ├── diffusion/
│   ├── thinking-exploration/
│   ├── knowledge-qa/
│   ├── paper-deep-dives/
│   └── assets/                    # 样式、脚本、图片与本地前端依赖
├── hooks/                         # MkDocs 构建钩子
├── scripts/                       # 本地服务、AI 代理与质量检查脚本
├── tests/                         # 脚本、构建钩子与站点功能测试
├── .github/workflows/             # GitHub Pages 部署工作流
├── mkdocs.yml                     # 导航、主题、扩展与资源配置
├── Makefile                       # 常用开发命令
├── requirements.txt               # Python 依赖
└── environment.yml                # Conda 环境定义
```

`files/roadmap/` 用于维护规划，不参与站点构建。`site/` 是本地构建产物。

## 安装环境

### 使用 venv

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
make serve MKDOCS=mkdocs
```

### 使用 Conda

项目默认会优先使用 `./docs/bin/mkdocs`，因此可以把 Conda 环境创建到 `docs/`：

```bash
conda env create -f environment.yml -p ./docs
conda activate ./docs
make serve
```

也可以激活已有环境并显式指定 MkDocs：

```bash
conda activate /absolute/path/to/conda-env
make serve MKDOCS=mkdocs
```

## 本地预览

```bash
make serve
```

该命令会启动 MkDocs livereload，自动打开 `http://127.0.0.1:8000`，同时尝试启动本地 AI 阅读助手代理。没有配置 AI 密钥时，文档站仍会正常启动。

只修改单页正文时，可以使用增量构建缩短刷新时间：

```bash
make serve-fast
```

需要更换监听地址或端口时：

```bash
make serve DEV_ADDR=127.0.0.1:8010
```

如果页面状态异常，停止旧进程后重新运行 `make serve`，以完整重建导航、搜索索引和静态资源。

## 本地 AI 阅读助手

AI 阅读助手是可选的本地功能，只在通过 `localhost` 或 `127.0.0.1` 访问站点时加载，不会在 GitHub Pages 等静态部署中启用。

先复制配置模板：

```bash
cp .env.local.example .env.local
```

然后在 `.env.local` 中填写 `OPENAI_API_KEY`。如需切换兼容 Responses API 的服务、模型或推理强度，可以调整：

```dotenv
OPENAI_API_KEY=
ATLAS_AI_BASE_URL=https://your-api-endpoint.example
ATLAS_AI_MODEL=your-model
ATLAS_AI_REASONING_EFFORT=high
```

配置完成后运行 `make serve`。阅读页面时可以直接询问整篇文章，也可以先选中文字，再点击“问 AI”进行聚焦提问。浏览器只连接本机 `127.0.0.1:8765` 代理，密钥由本地 Python 进程读取，不会写入前端资源。请勿提交 `.env.local`。

## 构建静态站点

```bash
make build
```

该命令执行严格构建，产物输出到 `site/`。

## 质量检查

常用命令如下：

| 命令 | 用途 |
| --- | --- |
| `make quality` | 检查内容结构、本地链接、图源说明和精读索引完整性 |
| `make quality-hints` | 在常规检查基础上显示低优先级改进提示 |
| `make quality-pages` | 显示提示及对应页面，便于逐篇处理 |
| `make paper-quality` | 检查论文元数据、来源链接、索引与本地资源 |
| `make image-quality` | 检查图片引用、清晰度、裁剪风险和未引用资源，并生成临时 contact sheet |
| `make math-quality` | 检查数学公式的 Markdown 与渲染风险 |
| `make blog-quality` | 评估文章作为独立博客发布时的完整度 |
| `make build` | 执行 `mkdocs build --strict`，验证导航、链接和静态构建 |

常规提交前建议至少运行：

```bash
make quality
make build
```

自动检查验证结构与资源完整性，标题、篇幅和关键词仅作为人工复核提示，不据此证明知识深度或论文结论正确。涉及图片、公式或论文精读时，再运行对应的专项检查。图片检查生成的 contact sheet 位于 `/private/tmp/charles-ai-notes-image-quality/`，只用于人工复核。

## GitHub Pages 部署

仓库已包含 [deploy-docs.yml](./.github/workflows/deploy-docs.yml)。推送到 `main` 分支后，工作流会安装依赖、执行严格构建、上传站点产物并部署到 GitHub Pages。

首次启用时，需要在 GitHub 仓库设置中把 Pages 的来源切换为 GitHub Actions。

## 内容维护约定

这个仓库按技术问题维护，新增内容应连接已有概念、实现或证据来源。编写或更新文章时，建议同时检查：

- 概念定义、适用边界和必要前置知识是否完整
- 方法、实现、评测、部署与失效模式是否形成闭环
- 公式、图表和代码是否有足够的上下文解释
- 与基础章节、知识 QA、论文精读之间是否可以建立互链
- 导航、站内链接和严格构建是否仍然通过

## 适合谁

- 正在补齐 AI 数学、模型与系统基础的学习者
- 从事大模型训练、推理、量化、算子或编译器工作的工程师
- 研究 VLM、VLA、强化学习、世界模型与具身智能的研究者
- 希望把论文阅读沉淀为结构化工程知识的人
