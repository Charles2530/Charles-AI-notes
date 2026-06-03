# 全站证据与复现状态标准

这页是全站统一证据规范。它不替代正文、论文专题或 [Claim Ledger](claim-ledger.md)，而是规定每个关键结论应该怎样写清楚：证据来自哪里、强到什么程度、能支持什么、不能外推到哪里。

## Evidence Snapshot 模板

高风险页面、前沿论文页、技术报告页和系统效率页优先使用这个固定表头：

| Claim | Direct Source | Figure/Table/Setting | Evidence Type | Repro Status | Can Support | Cannot Prove |
| --- | --- | --- | --- | --- | --- | --- |
| 需要被读者复用的关键结论 | 论文、报告、官方仓库、专题页或站内 fixture | Figure、Table、benchmark、demo、日志或实验设置 | 固定证据标签之一 | 固定复现标签之一 | 这条证据能支撑的最强说法 | 不能从这条证据外推的说法 |

正文可以继续用中文说明，但表头保持英文，方便全站检索：

```text
Claim | Direct Source | Figure/Table/Setting | Evidence Type | Repro Status | Can Support | Cannot Prove
```

## 事实版本块

`Evidence Snapshot` 附近应补一个短事实版本块，尤其是前沿论文、官方技术报告、系统吞吐页和具身/世界模型闭环页：

| Checked Date | Official Source | Repro Status | Notes |
| --- | --- | --- | --- |
| `2026-05-16` | arXiv / official PDF / official repo / official project page | 固定复现标签之一 | 写清是否只有论文、是否有作者代码、是否只是官方 demo，未见第三方证据时不要写 `Independent Reproduced` |

推荐写法：

```markdown
**事实版本。**

| Checked Date | Official Source | Repro Status | Notes |
| --- | --- | --- | --- |
| 2026-05-16 | arXiv / official PDF / official repo | Paper Only | 未见本站记录的第三方独立复现；按论文/官方材料保守引用。 |
```

### 资源类型补充

`Repro Status` 不是“链接数量”的同义词。页面如果写 `Author Code / Official Repo`，正文或事实版本附近应能看到 GitHub、Hugging Face、模型卡、权重、benchmark script 或官方仓库等可检查资源；如果只有项目页、视频或 demo，应优先写 `Official Demo` 或 `Paper Only`，并说明它不能支撑平均成功率。

| 资源类型 | 可以支持的 Repro Status | 仍然不能证明 |
| --- | --- | --- |
| arXiv / official PDF | `Paper Only` | 代码可运行、结果已独立复现 |
| GitHub / official repo | `Author Code / Official Repo` | 所有论文 claim 都已复现 |
| Hugging Face model card / weights | `Author Code / Official Repo` | benchmark 数字和闭环任务自动成立 |
| official project page / video / interactive demo | `Official Demo` | 平均成功率、鲁棒性、跨硬件泛化 |
| 本站脚本或 mini fixture | `Toy Fixture` | 真实模型、真实机器人或论文方法有效 |

## 固定证据标签

| Evidence Type | 何时使用 | 典型能支持什么 | 不能证明什么 |
| --- | --- | --- | --- |
| `Paper Result` | 论文或技术报告给出主实验、主表、主图 | 在该论文设置下方法有效或优于基线 | 跨任务、跨硬件、跨数据分布稳定成立 |
| `Ablation` | 模块、数据、目标函数、超参或系统组件消融 | 某个因素在该设置下有贡献 | 消融幅度能照搬到其他系统 |
| `System Throughput` | 报告延迟、吞吐、显存、通信、kernel 或服务指标 | 系统路径能降低某类运行成本 | 任务质量、动作质量或安全性同步提升 |
| `Closed-loop` | 仿真或真实环境中执行任务并反馈成败 | 方法能在该任务分布和评测规则下影响决策 | 开放世界、长时部署或未见硬件安全 |
| `Official Demo` | 官方页面、视频、交互 demo 或少量 qualitative case | 系统展示了某种能力或交互形态 | 平均成功率、鲁棒性、可复现收益 |
| `Toy Fixture` | 本站小样例、脚本、schema、mini-chain | 说明证据链、字段契约或指标计算方式 | 真实模型或论文方法有效 |
| `Site Inference` | 本站把多篇论文和工程经验合成判断 | 形成路线假设、实验设计或工程优先级 | 直接当作论文结论或复现实验 |

## 固定复现标签

| Repro Status | 含义 | 写作边界 |
| --- | --- | --- |
| `Independent Reproduced` | 有独立第三方复现实验或多方可比报告 | 只能在确有第三方证据时使用 |
| `Author Code / Official Repo` | 作者发布代码、权重、模型、benchmark 或运行脚本 | 说明可检查接口，不等于所有 claim 已复现 |
| `Paper Only` | 只有论文或技术报告中的结果 | 不能写成已复现，前沿 claim 要保留边界 |
| `Official Demo` | 主要证据来自官方 demo、项目页或视频 | 只能支持展示能力，不支持平均收益 |
| `Site Inference` | 主要是本站归纳和工程判断 | 必须说明需要单独实验验证 |
| `Toy Fixture` | 主要是本站小型可复算样例 | 只能证明证据链写法和指标脚本可运行 |

## 成熟度分层

全站读法里还要区分“技术成熟度”。同样是 `Paper Result`，经典基础方法、工程常用系统、前沿模型报告和本站推断的采用风险完全不同。

| 成熟度 | 典型形态 | 采用建议 | 必须补的边界 |
| --- | --- | --- | --- |
| 经典已验证 | PlaNet、Dreamer、ZeRO、Megatron-LM、SmoothQuant 等被广泛讨论的基础路线 | 可作为 baseline 或解释框架 | 仍要核对任务、硬件和数据分布 |
| 工程常用 | vLLM/KV、FSDP/ZeRO、FP8 路径、profiling 与 SLO 工具链 | 可进入工程方案比较 | 端到端 trace 和质量回归必须同表 |
| 前沿待复现 | 2025-2026 新模型、技术报告、闭源系统或新视频世界模型 | 适合作为实验假设 | 默认不写独立复现，不能外推到生产 |
| 官方展示 | 官方 demo、项目视频、少量 qualitative case | 适合说明能力形态 | 不能支持平均成功率或稳定收益 |
| 本站推断 | 跨论文合成、路线判断、toy fixture | 适合设计实验和阅读路径 | 不能替代论文结果或真实闭环 |

## 图与表的解释规则

复杂图必须紧跟正文解释或图注读法。只写“图源”不够，读者还需要知道这张图如何进入判断链；新版正文不再要求使用提示框。

推荐格式：

```markdown
<small>图源：论文名，Figure X。原图表达：...。本站读法：先看输入输出，再看效率机制，最后看它不能证明什么。</small>

这张图在正文里的作用是说明 ...。它能支撑 ...，但不能单独证明 ...。
```

结果表还要额外写 `读数边界`：

```markdown
**读数边界。** 写清 benchmark 条件、是否 closed-loop、是否真实机器人、是否系统吞吐、是否独立复现。
```

## 常见误写法

| 容易误导的写法 | 更稳的写法 |
| --- | --- |
| “该 demo 证明系统可用于真实机器人” | “官方 demo 展示交互形态；真实机器人闭环收益仍需 closed-loop eval” |
| “kernel 快 2 倍，所以模型更好” | “System Throughput 支持成本下降；质量和安全要单独验收” |
| “论文 SOTA，说明路线通用” | “Paper Result 支持该 benchmark 设置；跨分布和部署需要复现” |
| “本站 mini-chain 证明方法有效” | “Toy Fixture 只证明证据链字段和指标计算方式” |

## 和 Claim Ledger 的关系

[Claim Ledger](claim-ledger.md) 是世界模型高效训练主线的证据总账；本页是全站写作标准。正文页面最好两者都接入：局部 `Evidence Snapshot` 让读者在当前页读懂证据边界，Claim Ledger 则用于跨页面核对前沿 claim、复现状态和不能外推项。
