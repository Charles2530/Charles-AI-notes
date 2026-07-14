# 扩散模型章节课程化重构设计

## 结论

把现有扩散模型专题重构为“基础主线 + 进阶分支”的双层课程。目标不是继续增加术语和论文摘要，而是先让第一次系统学习扩散模型的深度学习开发者建立完整闭环：

```text
扩散模型解决什么生成问题
-> 训练时怎样构造带噪监督
-> DDPM 的前向、反向和损失怎样连接
-> U-Net / DiT 与 Stable Diffusion 组件怎样协作
-> 怎样实现并验证一个最小扩散模型
-> 怎样继续进入 Score/SDE、Flow、Guidance、Solver、蒸馏与视频扩散
```

采用以下范围：新增 4 篇文章，重写专题首页和 `training.md`，调整现有高级文章的职责与导航，不批量删除或重命名旧文件。

## 目标读者与学习终点

目标读者具备以下基础：

- 理解深度学习训练循环、反向传播、损失函数和优化器。
- 能阅读基础 PyTorch 代码，理解 tensor shape、batch 和 device。
- 学过高斯分布、条件概率和期望，但不要求掌握随机微分方程或变分推断。

完成基础主线后，读者应能：

1. 用自己的话解释扩散模型是什么，并区分训练链和生成链。
2. 解释 DDPM 为什么主动给真实样本加噪、模型预测什么，以及噪声 MSE 如何服务反向采样。
3. 看懂前向闭式采样、反向高斯过程和简化训练目标中的核心公式。
4. 画出 Stable Diffusion 的文本编码、latent 去噪、scheduler 更新和 VAE 解码数据流。
5. 运行一个最小 PyTorch DDPM，实现加噪、训练和反向采样。
6. 根据问题继续阅读 DDIM、Score SDE、Flow Matching、DiT、Guidance、蒸馏和视频扩散文章。

## 当前结构审计

当前专题已经有较多高质量内容，但存在以下结构问题：

1. `files/diffusion/index.md` 没有进入 `mkdocs.yml` 导航，读者会直接从“扩散模型发展脉络”开始。
2. 首页主要承担链接分发，没有在首屏回答“扩散模型是什么”，也没有完整展示训练和生成两条链。
3. `training.md` 已解释加噪、噪声预测、条件和表示空间，但同时提前进入 DiT、latent diffusion、视频、世界模型、蒸馏和少步采样，基础数学闭环反而不够连续。
4. 现有文章缺少独立的网络架构与 Stable Diffusion 系统页。U-Net、DiT、VAE、文本编码器、cross-attention 和 scheduler 分散在多篇文章中。
5. 缺少从公式逐行映射到 PyTorch 的可运行最小示例，读者无法验证自己是否真正理解。
6. `consistency-models-and-rectified-flow.md` 同时承担 Consistency、LCM、Flow Matching 和 Rectified Flow，两个训练接口容易混淆。
7. `practical-training-recipes-and-failure-analysis.md` 已存在，但未进入扩散模型导航。
8. 多篇文章分别重新解释数据—噪声路径、预测目标和采样，存在重复；同时缺少统一的前置知识、学习目标和自测契约。

问题的核心不是文章数量不足，而是基础知识分散、高级概念过早出现、导航没有表达依赖关系。

## 方案选择

采用“基础主线 + 进阶分支”。

没有选择只增加一篇导论，因为它只能修复入口，无法补齐架构、代码和跨页边界。没有选择全量教材式重写，因为现有高级文章和论文精读已有可复用内容，大范围合并或重命名会增加链接迁移和事实复核成本。

本方案的原则是：

- 基础层必须能够连续读完并形成完整闭环。
- 进阶层按理论、控制与采样、少步部署、应用扩展分叉。
- 一个概念只在一个页面完整主讲，其他页面用最短回顾和链接连接。
- 保留现有 URL，降低站内外链接失效风险。

## 信息架构

### 专题入口

重写 `files/diffusion/index.md`，并把它放到扩散模型导航首位。首页只承担四项职责：

1. 用一句准确的话定义扩散模型。
2. 用一张图并排展示训练链和生成链。
3. 给出四篇基础主线的线性阅读路径。
4. 给出进阶分支地图和按任务进入的快捷入口。

首页不展开 SDE、Solver、DMD 或完整公式推导。

### 第一层：基础主线

1. **从零理解扩散模型**，新增 `files/diffusion/beginner-primer.md`。
2. **DDPM 数学与训练**，重写现有 `files/diffusion/training.md`。
3. **网络架构与 Stable Diffusion**，新增 `files/diffusion/architecture-and-stable-diffusion.md`。
4. **从零实现最小 DDPM**，新增 `files/diffusion/minimal-ddpm-from-scratch.md`，代码放在 `files/assets/examples/diffusion/`。

这四篇按顺序阅读。每篇可以独立回顾必要概念，但不得重复完整推导。

### 第二层：进阶分支

**理论坐标系**：

```text
发展脉络
-> 噪声日程与参数化
-> Score Matching、SDE 与 Probability Flow ODE
-> Flow Matching 与 Rectified Flow
```

**条件与采样**：

```text
条件控制与 Guidance
-> DDPM、DDIM、Euler、Heun 与 DPM-Solver
-> 扩散方法选择与对照
```

**少步生成与训练工程**：

```text
扩散蒸馏
-> Consistency Models 与 LCM
-> 训练配方与失败分析
```

**应用扩展**：

```text
视频与多模态扩散
-> Score SDE、DPM-Solver++、DMD、Wan 等站内论文精读
```

### 建议导航顺序

`mkdocs.yml` 中的“扩散与生成模型”按以下顺序组织：

```text
扩散模型专题
基础主线
  从零理解扩散模型
  DDPM 数学与训练
  网络架构与 Stable Diffusion
  从零实现最小 DDPM
进阶理论
  扩散模型发展脉络
  噪声日程与参数化
  Score Matching 到 SDE
  Flow Matching 与 Rectified Flow
条件与采样
  条件控制与 Guidance
  采样与推理加速
少步与工程
  扩散路线选择与方法对照
  一步生成、蒸馏与整流
  Consistency Models 与 LCM
  扩散训练配方与排障
应用扩展
  视频与多模态扩散
```

导航标题保持短名称，页面 H1 可以保留完整教学语义。

## 核心文章设计

### `beginner-primer.md`：建立心智模型

本页回答“扩散模型究竟是什么”，不负责完整数学推导。

建议结构：

1. 生成模型要解决什么问题：从数据集学习分布并生成新样本。
2. 一句话定义扩散模型：人为构造从数据到简单噪声分布的路径，再学习反向生成过程。
3. 训练链：真实样本、随机时间步、噪声、带噪样本、denoiser、监督目标。
4. 生成链：随机噪声、重复预测和 scheduler 更新、最终样本。
5. 为什么不是把训练图片“背下来”：训练目标是跨样本和噪声水平学习分布方向；同时明确记忆与数据风险仍需单独评测。
6. 扩散模型与 VAE、GAN、自回归模型的最小边界，只比较生成方式、训练信号和采样成本。
7. 图像、视频、音频、动作轨迹和科学数据中的应用。
8. 优势、代价和常见误解。
9. 四篇基础主线地图与自测题。

本页不出现 ELBO 展开、反向 SDE、DPM-Solver、DMD 或复杂参数化。

### `training.md`：完成 DDPM 数学闭环

重写后只主讲 DDPM 基础数学和训练，不再同时承担系统架构和现代路线总览。

建议结构：

1. 统一记号表：`x_0`、`x_t`、`epsilon`、`beta_t`、`alpha_t`、`bar_alpha_t`、`q`、`p_theta`。
2. 逐步前向链 `q(x_t | x_{t-1})`。
3. 任意时刻闭式采样 `q(x_t | x_0)`，解释为什么训练不必逐步加噪。
4. 已知 `x_0` 时的后验 `q(x_{t-1} | x_t, x_0)`，说明它为何可计算。
5. 学习反向过程 `p_theta(x_{t-1} | x_t)`，解释模型需要估计什么。
6. 从变分下界到简化噪声预测 MSE 的最短推导链。正文保留理解训练目标所需的步骤，完整代数可放折叠框。
7. `epsilon`、`x_0` 和 `v` 参数化只给出转换关系和后文入口，不在本页深入比较。
8. DDPM 训练算法伪代码。
9. DDPM 反向采样算法伪代码。
10. 每个公式的输入、输出、shape、采样来源和实际用途。
11. 常见错误与自测题。

现有关于 Stable Diffusion 组件、DiT、视频世界模型、少步蒸馏和复杂日志诊断的内容迁移到对应专页或保留一句桥接。

### `architecture-and-stable-diffusion.md`：看懂真实系统

本页围绕统一 denoiser 接口组织：

```text
(noisy sample x_t, timestep t, optional condition c) -> prediction
```

建议结构：

1. 扩散算法和神经网络主干的分工。
2. 时间步 / 噪声水平嵌入为什么必须进入网络。
3. U-Net 的下采样、瓶颈、上采样、残差块、skip connection 和 attention。
4. DiT 如何把 latent 切成 token，并用 Transformer 替换 U-Net 主干。
5. 像素空间扩散与 latent diffusion 的成本和信息损失边界。
6. Stable Diffusion 的 tokenizer、text encoder、VAE encoder/decoder、conditional denoiser、cross-attention 和 scheduler。
7. 训练数据流：哪些模块参与前向，哪些模块冻结，loss 作用于哪里。
8. 文生图推理数据流：哪些模块只运行一次，denoiser 和 scheduler 为什么循环运行。
9. CFG 在系统中的位置，只解释接口并链接 Guidance 专页。
10. 从 Stable Diffusion 旧式 U-Net 到现代 latent DiT / Flow 模型的对应关系。
11. 组件故障到视觉症状的最小映射和自测题。

### `minimal-ddpm-from-scratch.md`：把公式变成代码

实践使用 PyTorch 和 MNIST 或 FashionMNIST，默认低分辨率、小 U-Net。正文优先保证透明度和可运行性，不追求生成质量或训练吞吐。

代码目录：

```text
files/assets/examples/diffusion/
├── README.md
├── requirements.txt
├── minimal_ddpm.py
└── test_minimal_ddpm.py
```

`minimal_ddpm.py` 至少提供：

- 噪声日程和预计算系数。
- `q_sample(x0, t, noise)`。
- timestep embedding。
- 教学用小型 denoiser。
- 单 batch loss。
- 反向一步更新。
- 完整采样循环。
- 固定 seed 的训练与采样入口。

正文按照 schedule、加噪、网络、loss、反向一步和采样循环的顺序解释，并逐段链接实际脚本。正文代码片段从可运行实现抽取或保持逐行一致，避免两份实现漂移。

短跑 smoke test 只验证链路完整，不把高画质作为测试门槛。

### `flow-matching-and-rectified-flow.md`：接上现代路线

本页假定读者已经理解 DDPM 和基础 ODE 直觉。

建议结构：

1. 从数据分布到噪声分布的条件概率路径。
2. 条件 velocity target 如何构造，为什么能用监督回归训练。
3. 模型怎样从 conditional target 学到 marginal velocity field。
4. ODE 采样怎样使用速度场。
5. Flow Matching 与 DDPM 噪声预测的共同点和差异。
6. 明确区分 DDPM `v`-prediction、Flow velocity 和视频中的物体运动速度。
7. Rectified Flow 的路径变直直觉、reflow 和少步积分收益。
8. 路径曲率、数值误差、条件保持和少步质量的评测边界。
9. 与 Consistency/LCM 的终点映射目标进行接口级比较。
10. 现代 latent DiT / 视频模型中的位置和后续论文入口。

## 现有文章调整

| 文件 | 重构后职责 | 主要调整 |
| --- | --- | --- |
| `files/diffusion/index.md` | 专题门户 | 加入定义、训练/生成闭环、基础主线和进阶地图，并进入导航首位 |
| `files/diffusion/evolution.md` | 历史地图 | 后移到基础主线之后；不再承担第一次概念解释 |
| `files/diffusion/training.md` | DDPM 数学与训练 | 补齐反向后验、ELBO 到噪声 MSE、训练与采样算法；迁出现代系统内容 |
| `noise-schedules-and-parameterization.md` | 高级日程与参数化 | 假定读者已懂 DDPM；删除重复基础推导，补充与采样器的接口表 |
| `score-matching-sde-and-probability-flow.md` | Score 与连续时间理论 | 增加与 DDPM 基础页和 Flow 页的接口说明，不重复入门定义 |
| `guidance-and-conditioning.md` | 条件与控制 | 串起条件训练、condition dropout、CFG 推理和失效模式 |
| `inference.md` | 采样算法与推理成本 | 以同一预训练模型下的路径、随机性、求解器和 NFE 差异组织 |
| `comparison-table.md` | 方法选择 | 作为读完多个分支后的决策页，不作为新手入口 |
| `distillation.md` | 少步蒸馏 | 保留 teacher-student、DMD 和分布匹配主线，减少 DDPM 基础回顾 |
| `consistency-models-and-rectified-flow.md` | Consistency/LCM 与 Flow 边界 | 详细 Flow 内容迁到新页；保留 Consistency/LCM 和两条路线的简短边界比较 |
| `practical-training-recipes-and-failure-analysis.md` | 训练排障 | 纳入导航，连接最小实现中的可观察症状和真实系统指标 |
| `video-and-multimodal-diffusion.md` | 应用扩展 | 假定读者已经掌握 latent、DiT、Flow 和条件接口，减少重复定义 |

`consistency-models-and-rectified-flow.md` 保留现有路径，页面标题改为能够同时表达主体和边界的名称，不重命名文件。

## 单篇写作契约

基础主线和新增 Flow 页面按以下契约组织：

1. 开头写“本章回答什么”、前置知识和读完后的能力。
2. 先给一个完整问题或数据流，再引入公式和术语。
3. 新符号第一次出现时同时给数学定义、白话解释和代码变量名。
4. 关键公式后紧跟输入、输出、shape、用途和成立条件。
5. 每篇只保留一条主叙事；扩展推导使用折叠框或链接下沉。
6. 图片说明写清图实际支持的结论和不能证明的内容。
7. 结尾提供 3–5 个自测问题、下一章入口和分层外部阅读。

高级旧文不强制完全改成同一模板，但必须补齐前置知识、职责说明和明确的下一步链接。

## 图解设计

新增六组本站原创图解：

1. **训练链与生成链总闭环**：并排展示两个运行过程，标出训练时已知的 `x_0` 和噪声，以及生成时只有 `x_T`。
2. **二维分布演化**：用 toy distribution 展示数据分布逐渐接近高斯，再沿学到的方向返回。
3. **DDPM 系数图**：展示 `x_0`、`epsilon`、`x_t` 与 signal/noise coefficient 的关系。
4. **U-Net / DiT 统一接口图**：展示 noisy sample、timestep、condition 和 prediction。
5. **Stable Diffusion 训练/推理双数据流**：标出 tokenizer、text encoder、VAE、denoiser、scheduler，以及重复调用边界。
6. **扩散方法坐标系**：比较 DDPM、DDIM、Score/SDE、Flow 和 Consistency 分别改变训练目标、生成路径、求解器还是终点映射。

基础直觉优先使用本站原创 SVG 或 Mermaid。论文原图只在忠实解释原方法或实验结果时使用，并保留来源、原图含义和本站解读边界。

## 实践代码验收

最小代码必须满足：

- `q_sample(x0, t, noise)` 输出 shape、dtype 和 device 与输入兼容。
- 低噪声时间步保留更多输入信号，高噪声时间步包含更多噪声；测试检查系数和统计趋势，不依赖单个像素相等。
- 单个 batch 可以完成前向、loss、反向传播和一次参数更新。
- 反向一步和完整采样循环在固定 seed 下可复现。
- 时间步 batch、广播系数、随机噪声和 terminal step 有单元测试。
- smoke test 在小 batch 和少量时间步下快速执行，只验证数值有限、shape 正确和链路完整。
- README 说明完整训练的预期成本，避免把 smoke test 输出误认为模型质量结论。

## 外部资料与证据边界

在线资料调研于 2026-07-14 完成。它们用于比较教学顺序和内容覆盖，技术定义与公式仍回到原论文或官方材料核对。

- [Hugging Face Diffusion Models Course：Unit 1](https://huggingface.co/learn/diffusion-course/en/unit1/1)：先用迭代去噪解释生成，再通过 Diffusers 和 from-scratch notebook 建立实践闭环。本设计借鉴“先完整看见过程，再动手实现”。
- [The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion)：把 forward process、U-Net、训练和采样代码放在一条解释链中。本设计借鉴公式—代码映射，但拆成多篇以控制单页认知负荷。
- [Lil'Log: What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)：覆盖前向/反向过程、训练目标、guidance、加速、latent diffusion 和模型架构。本设计借鉴统一地图，但把高级主题下沉到分支。
- [MIT 6.S183: A Practical Introduction to Diffusion Models](https://www.practical-diffusion.org/)：课程目标同时要求理解组件协作、控制预训练模型和训练自定义模型。本设计借鉴“理解组件 + 可运行实践”的学习终点。
- [Cambridge Diffusion Models and Generative AI Workshop](https://docs.science.ai.cam.ac.uk/diffusion-models/)：把 VAE、U-Net、CLIP 等 building blocks 与 DDPM 算法连接。本设计据此把真实系统组件单独成篇，但不展开过长物理类比。
- [Step-by-Step Diffusion: An Elementary Tutorial](https://www.nowpublishers.com/article/Details/CGV-114)：采用 Fundamentals、DDPM、DDIM、Flow Matching、Practice 的递进顺序。本设计借鉴数学依赖顺序，同时保留本站的系统工程与视频方向。
- [DDPM](https://arxiv.org/abs/2006.11239)：前向过程、反向建模和训练目标的主要事实来源。
- [Latent Diffusion Models](https://arxiv.org/abs/2112.10752)：latent diffusion、cross-attention 和计算折中的主要事实来源。

博客或课程不作为具体性能数字、论文贡献或算法等价关系的唯一来源。现代模型示例必须核对对应论文或官方技术报告。

## 验证与验收标准

### 读者验收

- 首页首屏能够在不使用高级缩写的情况下回答“扩散模型是什么”。
- 读完 primer 后，读者能区分训练时的随机加噪任务和生成时的迭代采样。
- 读完 `training.md` 后，读者能解释闭式加噪为何允许随机抽时间步训练，并能把噪声预测转换成 `x_0` 估计。
- 读完架构页后，读者能标出 Stable Diffusion 每个组件的输入、输出和调用频率。
- 读完实践页后，读者能运行 smoke test，并能沿代码追踪 `x_0 -> x_t -> loss` 与 `x_T -> x_0`。

### 内容验收

- 每个核心术语和符号有唯一主讲页。
- 基础主线不存在必须先读未列出的高级文章才能理解的跳跃。
- `training.md`、参数化页、Score/SDE 页和 Flow 页之间没有复制整段推导。
- 所有公式变量在首次使用前定义；训练与采样的时间方向保持一致。
- 原创图与论文原图的来源边界清楚，图片附近有解释。
- 所有新增和调整页面都有前置知识、下一章入口和自测。

### 仓库验收

按风险由低到高运行：

```bash
python -m pytest files/assets/examples/diffusion/test_minimal_ddpm.py
make quality
make image-quality
make build
```

其中 `make build` 执行 MkDocs strict build。最后人工检查：

- 桌面和移动视口中的专题首页、嵌套导航和宽表格。
- 公式密集页是否溢出或符号未渲染。
- SVG、Mermaid、代码块和图片说明是否可读。
- 新增页面是否全部进入导航，旧站内链接是否仍可访问。
- 最小代码文档中的命令是否能从仓库根目录直接复制执行。

## 实施顺序

### P0：先建立入口和边界

- 重写专题首页。
- 调整 `mkdocs.yml` 导航。
- 为全部目标页面写清职责、前置知识和明确的下一步链接。

### P1：让基础定义和数学闭环成立

- 完成 `beginner-primer.md`。
- 重写 `training.md`。
- 完成训练/生成总图、二维分布图和 DDPM 系数图。

### P2：打通算法、架构与代码

- 完成 `architecture-and-stable-diffusion.md`。
- 完成 `minimal-ddpm-from-scratch.md` 和示例代码测试。
- 完成 U-Net / DiT 接口图与 Stable Diffusion 双数据流图。

### P3：整理进阶分支

- 新增 `flow-matching-and-rectified-flow.md`。
- 收敛 Consistency/LCM 页面职责。
- 清理参数化、Score/SDE、Guidance、Inference、蒸馏和视频页的重复基础内容。
- 把训练排障页纳入导航并补齐交叉链接。

### P4：全量验证

- 运行示例测试、内容质量、图片质量和 strict build。
- 人工检查移动端导航、公式、图片和代码块。
- 修复断链、孤立页、重复定义和表述不一致。

每个阶段结束时运行与该阶段相关的最小检查，不把链接、公式和代码问题集中到最后处理。

## 风险与控制

1. **基础页与旧文重复**：先定义唯一职责，再迁移内容；旧文只保留必要回顾和链接。
2. **数学推导过重**：正文保留理解训练目标所需的最短链，额外代数放折叠框或前置数学页。
3. **最小代码过度工程化**：只实现教学所需接口，不引入训练框架、配置系统或分布式能力。
4. **短跑结果被误解为质量承诺**：smoke test 只验证链路，在 README 明确完整训练成本和预期。
5. **导航改动影响用户现有工作**：实施时只修改扩散模型对应的 `mkdocs.yml` 区块，并先检查当前未提交差异。
6. **旧链接失效**：不重命名现有扩散文章；新增文件只增加链接，不替换旧 URL。
7. **外部资料时效变化**：结构可以参考教程，算法事实以论文和官方材料为准；不依赖易变的 API 参数讲核心概念。

## 实施边界

本轮包含扩散模型专题首页、基础主线、Flow 专页、必要旧文调整、原创图、最小代码、导航和验证。不包含：

- 训练可用于生产的图像或视频基础模型。
- 新建通用扩散训练框架、实验追踪系统或配置系统。
- 批量重写 `paper-deep-dives/diffusion/` 下的论文精读。
- 为每一种 sampler、蒸馏方法或视频模型增加独立百科页。
- 修改与扩散模型专题无关的现有工作区改动。
