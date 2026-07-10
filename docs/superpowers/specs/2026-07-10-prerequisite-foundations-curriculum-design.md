# 前置数学与基础概念课程重构设计

## 目标

把现有知识库重构成一条可连续阅读、也可从下游反查的学习链：

```text
数学原理 -> 基础机制 -> 专题正文 -> 论文精读
```

本次不追求把前置数学写成完整数学教材，而是确保后续训练、推理、量化、扩散、强化学习、世界模型、VLM、VLA 和具身智能文章中反复出现的通用数学概念，都能在前置层找到足够清楚的定义、公式拆解、具体例子和继续阅读入口。

## 当前问题

1. 前置数学已有 18 篇正文加 1 篇索引，基础概念已有 22 篇正文加 1 篇索引，内容总量足够，但两层之间的显式依赖很弱。基础概念页绝大多数没有指出需要先会哪些数学概念。
2. 数学首页列出 Jacobian，但现有正文只讲标量导数、梯度和链式法则，没有系统解释向量到向量映射、局部线性化、JVP 和 VJP。
3. 重参数化、Straight-Through Estimator、stop-gradient、EMA teacher 和 importance ratio 分散在 VAE、DreamerV2、QAT、JEPA、蒸馏和 RL 内容中，缺少统一的训练信号视角。
4. 协方差、重要性采样、表示坍缩、EBM、MPC/CEM 等概念虽然在下游出现，但前置解释过短或只存在于单篇论文精读中。
5. 位置编码、卷积、优化、潜变量和分布偏移同时出现在数学层与基础层，两层没有稳定声明各自的职责，容易形成重复阅读。

## 范围与约束

- 新增页面最多两篇：一篇属于前置数学，一篇属于基础概念。
- 保留现有文章 URL，不重命名现有文件，避免破坏站内外链接。
- 调整 `mkdocs.yml` 中前置数学和基础概念的分组与顺序，但不改其他顶层专题的导航结构。
- 为全部基础概念页添加简短的数学先修提示，但只深改与本轮缺口直接相关的正文。
- 不批量重写论文精读和专题正文；仅在确有必要时补充反向链接。
- 图示使用本站自绘 SVG、已有且来源明确的论文图，或有明确来源说明的外部博客图；不使用图片生成。
- 继续沿用现有 Markdown、公式、图注、常见误读和相关阅读风格。

## 信息架构

### 前置数学知识

前置数学从平铺列表调整为五组：

1. **数学语言与变化**
   - 线性代数与几何直觉
   - 谱分解、二阶近似与低秩结构
   - 微积分与牛顿-莱布尼茨公式
   - 微分、Jacobian 与梯度传播（新增）
2. **概率、统计与信息**
   - 概率论、分布与期望
   - 采样、估计与校准
   - 信息论、熵与分布距离
   - 随机过程与马尔可夫性
3. **优化与学习目标**
   - 优化、梯度与约束
   - 潜变量、似然与生成模型目标
   - 离散选择、Top-k 与稀疏路由
   - 价值函数、策略梯度与强化学习目标
4. **时序、控制与空间**
   - 序列、位置编码与注意力掩码
   - 动态系统、控制与状态空间
   - 卷积、频域与多尺度表示
   - 几何、视觉与三维空间
5. **研究判断与工程**
   - 因果、反事实与分布偏移
   - 缩放律、幂律与实验曲线
   - 数值计算、复杂度与估算

分组只改变导航层次，不改变现有数学页的 URL。

### 基础概念

基础概念保留现有领域分组，第一组扩展为“计算、梯度与运行时”：

1. **计算、梯度与运行时**
   - 读懂公式的最小数学
   - 张量、Shape 与计算图
   - 线性层、MLP 与 GEMM
   - 自动微分与激活显存
   - 梯度估计、离散变量与训练信号（新增）
   - 数值、显存与运行时
2. **模型结构**
3. **训练目标与数据**
4. **生成、多模态与检索**
5. **压缩与架构搜索**

## 内容职责

| 层级 | 主要职责 | 不负责的内容 |
| --- | --- | --- |
| 前置数学 | 定义数学对象，拆公式，解释假设、方向、尺度和近似 | 不完整复述具体模型 pipeline |
| 基础概念 | 解释数学怎样组成模型结构、训练信号和工程行为 | 不重复推导所有数学定义 |
| 专题正文 | 解释多个机制怎样组成技术路线和系统 | 不为单个基础符号重新开课 |
| 论文精读 | 解释论文采用的具体定义、算法、实验与证据边界 | 不把论文特有写法冒充通用定义 |

当数学页与基础页涉及同一主题时，基础页在第一次使用公式前链接数学页，然后只展开模型机制。例如数学卷积页负责卷积定理、频率和 SNR，基础卷积页负责 feature map、channel、receptive field 和 U-Net 数据流。

## 新页面一：微分、Jacobian 与梯度传播

**文件：** `files/prerequisite-math/differential-jacobian-and-gradient-flow.md`

### 叙事主线

从“一个输入变一点，输出怎样变”开始，逐步走到“整层网络的扰动怎样向后传”和“loss 的梯度怎样向前一层传”。页面不从符号表开始，而从一个两维机器人状态映射到三维控制量的例子切入，说明标量导数不足以描述向量映射。

### 内容顺序

1. **从导数到微分**：用
   \[
   \Delta y \approx f'(x)\Delta x
   \]
   解释导数首先是一种局部线性近似。
2. **Jacobian**：对 \(f:\mathbb{R}^n\to\mathbb{R}^m\) 定义
   \[
   J_f(x)_{ij}=\frac{\partial f_i}{\partial x_j},
   \qquad
   \Delta y\approx J_f(x)\Delta x.
   \]
   逐项解释行、列和 shape。
3. **梯度与 Jacobian 的关系**：标量输出时 Jacobian 退化为梯度的转置；明确行向量/列向量约定可能不同。
4. **向量链式法则**：
   \[
   J_{g\circ f}(x)=J_g(f(x))J_f(x).
   \]
   用 shape 检查解释矩阵相乘顺序。
5. **JVP 与 VJP**：分别解释 \(Jv\) 和 \(v^\top J\)，连接 forward-mode 与 reverse-mode autodiff，并说明深度学习反向传播为什么常用 VJP。
6. **Softmax Jacobian 示例**：
   \[
   \frac{\partial p_i}{\partial z_j}=p_i(\delta_{ij}-p_j).
   \]
   用三个类别的数值例子解释一个 logit 变化为什么会影响所有概率，以及概率和为 1 带来的耦合。
7. **误差传播示例**：
   \[
   \delta f\approx J\delta x,
   \qquad
   \|\delta f\|\le \|J\|\|\delta x\|.
   \]
   连接量化误差、KV 合并误差、机器人状态扰动和网络鲁棒性。
8. **与 Hessian / Fisher 的边界**：Jacobian 看一阶输入输出敏感度，Hessian 看标量目标的二阶曲率，Fisher 看模型分布的局部变化。
9. **常见误读与下游入口**：链接自动微分、Attn-QAT、量化比较、KVSlimmer、动态系统和控制内容。

### 图示

- 新增本站自绘 `files/assets/images/prerequisite-math/jacobian-local-linear-map.svg`，展示同一个小输入扰动经过不同 Jacobian 方向后被旋转、拉伸或压缩。
- 复用已有计算图或矩阵乘图时保留原来源图注；若现有素材不能直接支撑 JVP/VJP，则新增一张本站自绘数据流 SVG。

## 新页面二：梯度估计、离散变量与训练信号

**文件：** `files/foundations/gradient-estimators-and-training-signals.md`

### 叙事主线

从一个统一问题开始：模型前向过程包含采样、argmax、量化、teacher 或旧策略时，普通链式法则为什么断掉，训练又怎样获得可用信号。页面围绕“梯度来自哪里、估计是否有偏、方差多大、前向和反向是否使用同一对象”组织，而不是把技巧写成互不相关的名词表。

### 内容顺序

1. **为什么普通反向传播会断**：离散采样、argmax、round、外部 teacher 和 off-policy 数据分别在哪一步切断直接梯度。
2. **路径导数 / 重参数化**：
   \[
   z=\mu_\theta(x)+\sigma_\theta(x)\odot\epsilon,
   \qquad \epsilon\sim\mathcal{N}(0,I).
   \]
   用 VAE/RSSM 例子说明随机性被移到与参数无关的 ε 后，梯度可以沿确定性路径传播。
3. **Score-function estimator**：
   \[
   \nabla_\theta \mathbb{E}_{z\sim p_\theta}[f(z)]
   =
   \mathbb{E}[f(z)\nabla_\theta\log p_\theta(z)].
   \]
   连接 REINFORCE，并解释 baseline/advantage 为什么降低方差；同时区分这里对参数求导的 score 与扩散模型中对数据变量求导的 score。
4. **Straight-Through Estimator**：前向使用 hard/quantized value，反向借用 identity 或 soft surrogate 的梯度；连接 DreamerV2 categorical latent 和 Attn-QAT。
5. **Gumbel-Softmax 与连续松弛**：解释 temperature 控制 hard choice 的近似程度，并明确它不是“真正把 argmax 变得可导”。
6. **stop-gradient 与 EMA teacher**：解释哪些分支不接收梯度、teacher 参数怎样通过
   \[
   \bar\theta_t=\tau\bar\theta_{t-1}+(1-\tau)\theta_t
   \]
   更新，连接 BYOL、JEPA、target network 和蒸馏。
7. **重要性权重与策略比率**：
   \[
   \mathbb{E}_{p}[f(x)]
   =
   \mathbb{E}_{q}\left[\frac{p(x)}{q(x)}f(x)\right].
   \]
   解释 PPO ratio、off-policy correction 和权重爆炸。
8. **偏差与方差比较表**：比较 pathwise、score-function、STE、continuous relaxation 和 importance weighting 的适用对象、偏差、方差与主要风险。
9. **贯穿示例**：依次解释 VAE、DreamerV2、Attn-QAT、JEPA 和 PPO 中梯度从哪里来。
10. **术语重载提醒**：区分 VAE 的随机变量重参数化、扩散 scheduler 的时间/噪声参数化，以及工程 API 中的 parameter reparameterization。

### 图示

- 新增本站自绘 `files/assets/images/foundations/gradient-estimator-decision-map.svg`，按“连续可重参数 / 离散采样 / 不可导算子 / teacher 或旧分布”选择训练信号路线。
- 复用已有 V-JEPA、DDPM、CLIP 等图时，只在它确实说明训练信号路径时使用，避免一张架构图承担不支持的结论。

## 现有数学页的定点补强

### `probability-distributions-and-expectation.md`

- 在方差后增加协方差、相关系数和协方差矩阵。
- 用二维机器人状态误差说明正相关、负相关和“相关不等于因果”。
- 解释多元高斯中的 Σ 如何描述不同方向的不确定性，并连接 RSSM、DPM-Solver++ 和风险规划。

### `sampling-estimation-and-calibration.md`

- 增加重要性采样及权重退化，和新增基础页形成“数学定义 / 训练机制”的分工。
- 在成功率置信区间后补 Wilson 区间适用条件，避免小样本时把正态近似当作精确答案。
- 增加 bootstrap 的最小流程，用于复杂指标或没有简单解析标准误时的近似区间。

### `latent-likelihood-and-generative-objectives.md`

- 在 ELBO 推导中明确 Jensen 不等式作用的位置。
- 用一段简例解释重参数化为什么让 latent sampling 可训练，并链接新增基础页。
- 增加 EBM 的最小统一视角：低能量表示兼容、规范化概率需要 partition function、推理可通过优化或采样寻找低能量状态。
- 明确 EBM、显式 likelihood 模型和 score-based 模型的边界，不展开 H-JEPA 特有结构。

### `dynamic-systems-and-control.md`

- 增加一个短 horizon MPC 例子：枚举/采样动作序列、用模型 rollout、累计 cost、只执行第一步、重新观测。
- 增加 CEM 迭代：从候选分布采样、保留 elite、更新均值方差，再次采样。
- 连接 PlaNet、V-JEPA 2-AC、具身规划与安全页。

### `calculus-and-newton-leibniz.md`

- 保留标量导数、梯度、链式法则和 ODE/SDE 主线。
- 在链式法则末尾增加明确的“向量映射继续阅读”入口，不在本页重复 Jacobian 细节。

## 现有基础页的定点补强

### `pretraining-objectives-and-representation-learning.md`

- 在 contrastive / masked prediction 后增加表示坍缩的统一定义。
- 比较 negative samples、stop-gradient、EMA teacher、variance/covariance regularization 和 bottleneck/capacity constraint 的作用。
- 用 JEPA/VICReg 风格例子说明“prediction loss 低”不等于表示保留信息。
- 链接协方差数学和新增训练信号页。

### `probability-latent-variables-and-generative-models.md`

- 增加“显式密度、隐式生成、能量模型”比较表。
- 把重参数化的数学细节交给新增训练信号页，本页只解释它在 VAE/RSSM pipeline 中的位置。

### `autograd-activation-checkpointing-and-memory.md`

- 增加 Jacobian/VJP 先修链接。
- 用一小段解释 backward 实际计算 VJP，而不是显式物化整个 Jacobian。

### `model-compression-pruning-and-nas-basics.md`

- 增加 STE 与量化感知训练的桥接入口。
- 保持本页主体聚焦结构依赖和成本账，不扩写完整 QAT 算法。

## 基础页数学先修卡

每篇基础页在导语后加入统一但简短的提示：

```markdown
!!! tip "数学先修"
    本文会用到……。如果这些公式还不熟，先读……；已经熟悉可直接继续。
```

映射如下：

| 基础页 | 数学先修 |
| --- | --- |
| 读懂公式的最小数学 | 前置数学导读 |
| 张量、Shape 与计算图 | 线性代数；数值与复杂度 |
| 线性层、MLP 与 GEMM | 线性代数；数值与复杂度 |
| 自动微分与激活显存 | 微积分；Jacobian 与梯度传播 |
| 梯度估计、离散变量与训练信号 | 微积分；概率；采样估计 |
| 数值、显存与运行时 | 数值与复杂度；线性代数 |
| Transformer 输入与注意力 | 线性代数；序列与掩码；概率 |
| 位置编码与 Mask | 序列与掩码；线性代数 |
| 归一化、残差与激活 | 线性代数；优化；Jacobian |
| 卷积与视觉特征 | 卷积频域；线性代数 |
| MoE 与大模型架构 | 离散选择；概率；数值与复杂度 |
| Mamba 与混合 SSM | 动态系统；序列；微积分 |
| 优化与训练入门 | 优化；微积分；采样估计 |
| 预训练目标与表征学习 | 信息论；潜变量；线性代数 |
| 概率与潜变量模型 | 概率；信息论；潜变量目标 |
| 模型蒸馏入门 | 信息论；概率；训练信号 |
| 数据与数据集基础 | 概率；因果与分布偏移 |
| 数据划分与评测指标 | 采样估计；实验曲线 |
| 泛化、正则化与分布偏移 | 因果与分布偏移；采样估计；优化 |
| 生成与解码基础 | 概率；离散选择；序列与掩码 |
| 多模态推理基础 | 线性代数；信息论；三维几何 |
| Prompt、CoT 与 RAG | 序列与掩码；概率；线性代数 |
| 模型压缩、剪枝与 NAS | 谱与二阶近似；数值与复杂度；训练信号 |

先修卡不是考试门槛。文字必须告诉读者“本文具体会用到什么”，避免只列三个标题。

## 首页与导航改造

### `files/prerequisite-math/index.md`

- 将现有单表改成五组学习入口。
- 在每组前写一句该组解决的问题。
- 更新 Mermaid 依赖图，加入 Jacobian 与梯度传播，并减少交叉边数量，保证移动端可读。
- 增加按下游方向反查表：训练/量化、扩散、RL/世界模型、VLM/VLA/3D、推理与算子。

### `files/foundations/index.md`

- 覆盖导航中的全部基础页，不再只展示部分代表文章。
- 新增“数学先修 -> 基础机制 -> 专题入口 -> 论文例子”四列表。
- 保留从下游倒读方式，但合并重复链接，避免首页成为链接清单。

### `mkdocs.yml`

- 前置数学按五组嵌套导航。
- 基础概念第一组改名为“计算、梯度与运行时”，加入新增基础页。
- 保留现有顶层顺序：前置数学知识位于基础概念之前。
- 不触碰用户当前新增的论文导航项。

## 图文与来源规则

1. 两篇新增页面至少各有一张真正解释核心关系的图。
2. 自绘 SVG 使用稳定尺寸、清晰箭头和中英文必要标签，不使用装饰性图形。
3. 外部图必须在图注中给出来源页面、论文或博客链接，并说明原图表达什么、本站借它说明什么。
4. 不从博客截取只有文字或装饰作用的图片；公式和关系能更清楚地自绘时优先自绘。
5. 不使用图片生成，不引入版权来源不明的素材。
6. 新增外部来源若尚未登记，则同步补入 `files/references/external-reading-ledger.md`，正文仍优先链接原论文或官方材料。

## 验收标准

### 内容

- 新增两篇页面均包含故事化开场、公式逐项解释、至少一个数值或下游实例、常见误读、读完判断和相关阅读。
- Jacobian、JVP/VJP、重参数化、score-function estimator、STE、stop-gradient、EMA teacher、importance weighting 均能在前置层找到完整定义。
- 协方差、重要性采样、bootstrap、Jensen、EBM、MPC/CEM 和表示坍缩得到与下游用途相称的解释。
- Fisher、Hessian、Jacobian 三者边界一致，不互相冒充。

### 架构

- 前置数学导航按五组显示，两篇新增页面进入正确分组。
- 新增后 23 篇基础正文全部有准确的数学先修卡。
- 两个首页能分别支持顺序阅读和从下游反查。
- 现有页面 URL 不变，DM0.5 等用户当前改动不被覆盖。

### 视觉与构建

- 新增 SVG 能通过 XML 检查并被 MkDocs 复制到构建目录。
- Markdown 内部链接和图片路径无误。
- `python -m pytest tests/test_content_quality_report.py` 通过。
- `mkdocs build --strict` 或仓库等价严格构建命令通过。
- 构建后的两篇新增页面、两个首页和至少三篇带先修卡的基础页进行渲染抽查。

## 非目标

- 不把数学专题扩写成证明型大学教材。
- 不新增第三篇页面。
- 不全面重写训练、推理、扩散、世界模型或论文精读正文。
- 不重新设计 MkDocs 主题、顶部视觉或 CSS。
- 不因为重构导航而删除现有文章或更改其 URL。
