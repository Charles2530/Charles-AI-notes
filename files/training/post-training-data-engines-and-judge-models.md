# 后训练数据引擎与 Judge 模型

基础预训练结束后，真正决定模型“像不像产品”的，往往是后训练：指令跟随、偏好对齐、工具使用、拒答边界、风格控制和领域适配。后训练的核心竞争力越来越不是单个算法名词，而是能否持续找到高价值样本、生成有效候选、用 judge 扩大评价吞吐、再把结果稳定回流到 SFT、偏好训练、reward model 和回归集。


这页建议和 [偏好数据与对齐失效](preference-data-and-alignment-pitfalls.md)、[训练评测与消融设计](evaluation-and-ablation-methodology.md)、[可观测性与在线评测](../inference/observability-and-online-evaluation.md) 一起读。它们分别覆盖偏好数据风险、实验方法学和线上回流。

> **踩坑原因**：这类工程问题通常不是单点公式错，而是 shape、资源预算、并发路径和观测指标之间有一个假设没对齐；先把触发条件、监控信号和回滚边界拆开查。

## 一、后训练为什么是数据引擎

<!-- [融合：算法基础] 来源：llm-algo-leetcode《12. RLHF PPO Memory | RLHF 对齐：PPO 算法的核心 Loss 与显存流转 (RLHF PPO)》-->
> **算法视角**：PPO / DPO 都是在候选回答之间定义偏好信号；PPO 通过策略更新和 KL 约束控制模型漂移，DPO 则把 chosen / rejected 对直接转成 pairwise loss。
>
> 延伸阅读：[12. RLHF PPO Memory | RLHF 对齐：PPO 算法的核心 Loss 与显存流转 (RLHF PPO)](https://datawhalechina.github.io/llm-algo-leetcode/02_PyTorch_Algorithms/12_RLHF_PPO_Memory.html)

<!-- [融合：复杂度分析] 来源：llm-algo-leetcode《12. RLHF PPO Memory | RLHF 对齐：PPO 算法的核心 Loss 与显存流转 (RLHF PPO)》-->
| 操作 | 时间复杂度 | 空间复杂度 | 说明 |
|------|-----------|-----------|------|
| DPO pair loss | $O(BL)$ logprob 评估 | 需要 policy / reference 的中间结果 | $B$ 为样本对数，$L$ 为序列长度；参考：[对应章节](https://datawhalechina.github.io/llm-algo-leetcode/02_PyTorch_Algorithms/12_RLHF_PPO_Memory.html) |
| PPO rollout + update | 多次模型前向 / 反向 | 常同时保留 policy、reference、reward 路径 | 成本主要来自多模型调用、采样和 KL 监控。 |

预训练更像建立通用表示底座，后训练更像持续雕刻行为边界。它依赖的不是一次性数据包，而是一条长期运行的闭环：收集线上/离线 prompt、上下文和失败日志，聚类高频问题和长尾异常，生成多个候选回答或工具轨迹，用 judge、规则和程序信号粗筛，对高价值样本人审，再导出到 SFT、DPO/RLAIF、reward model、judge 训练和回归集。新模型上线后，还要继续观察同类问题是否下降。

可以抽象成：

$$
\mathcal{D}_{t+1}
=
\mathcal{D}_t \cup \mathcal{F}(\text{traffic}, \text{candidates}, \text{judge}, \text{human}, \text{online signals}).
$$


这里的关键是“持续”。如果没有稳定的数据生产线，后训练就会变成几次孤立对齐实验。

## 二、高价值样本：先定义什么值得标

后训练数据不是平均采集最优。最有价值的通常是边界样本和高代价失败：容易幻觉但用户高度关心的问题，工具选择困难、参数容易错、轨迹空转的任务，多轮对话里的上下文误解或状态遗忘，格式/引用/数字一致性容易出错的请求，过度拒答或过度冒进的风险边界，以及高价值客户、投诉、人工接管、线上事故和多模型分歧样本。

随机采样会得到大量普通问题，它们能证明模型在常规样本上还可以，却很难推动行为边界。后训练数据引擎应先回答：模型在哪类问题上最常错，错了之后业务代价多大，这种错误能否通过监督或偏好信号被修正。

## 三、Candidate 生成与数据导出

Candidate 生成不是越多越好，而是要产生有信息量的分歧。常见来源包括当前线上模型、新候选模型、更强 teacher 模型、不同 prompt 或 decoding 策略、工具路径与非工具路径对照，以及人工修订答案。

如果所有候选都很像，judge 再强也学不到区分边界。真正值钱的是同一问题下更短但更准、工具使用更合理、引用更充分、风险处理更稳的差异。

### 不同目标的数据形态

| 目标 | 更适合的数据 |
| --- | --- |
| SFT | 明确好示范、格式/schema、工具参数、引用结构 |
| DPO/偏好训练 | 多个候选都可行但优劣明显的对比样本 |
| RM/Judge 训练 | 大规模排序、自动 triage、模型间比较样本 |
| Agent/工具后训练 | 中间轨迹、工具返回、失败恢复、过程监督 |
| 回归集 | 历史事故、已修复失败、judge 容易偏置的样本 |

如果不分导出目标，数据很容易“收集很多、用起来混乱”。

## 四、Judge 模型：可扩展评价器，不是万能裁判

Judge 的作用是把部分昂贵人审转成可规模化评价。给定输入 $x$ 和候选 $y_a,y_b$，judge 可以输出相对偏好：

$$
J(x, y_a, y_b) \in \{a,b\}
$$


也可以输出多维分数 $s(x,y)$。但 judge 本质上是偏好近似器，不是世界真理。

更稳的 judge 体系不应只有一个总分，而应拆成 helpfulness、factuality、format/schema、tool-use correctness、safety/risk、execution success、citation support 等维度。

一个单一总分很容易奖励冗长、奖励格式、奖励保守或过拟合固定模板。复杂系统里，judge 还应看到检索证据、工具轨迹、schema、风险标签和任务类型；只看最终回答，无法发现“工具本来不该调却调了”“证据被忽略”“参数错但最终没暴露”等过程错误。

## 五、Judge 校准、人审与反馈分层

Judge 一旦偏，后训练飞轮会放大偏差：错误样本被当成高质量，主动采样聚焦错方向，DPO/RM 学到错误偏好，离线分数变漂亮但线上体验不一定更好。

Judge 校准至少要覆盖与人工偏好的一致性，对长度、格式、语言和风格的偏差，不同任务桶、风险桶、业务价值桶上的一致性，事实、代码执行、工具调用和安全边界的错误类型，以及 judge-human disagreement 池和 judge 专属 holdout。

人审资源应被当作高精度标签预算，而不是兜底劳动力。更高 ROI 的使用方式是集中在：judge 分歧大、高价值失败桶、模型高分但业务失败、新任务新功能、高风险政策边界。常见做法是两级人审：普通标注员粗筛，专家或产品 owner 复核关键样本。

Judge 体系也建议分层：筛选 judge 负责快速低成本粗筛，发布 judge 负责关键发布集和高风险桶，程序验证器则用单元测试、schema 校验、静态检查和工具返回码提供确定性信号。

## 六、线上信号与长尾回流

后训练引擎不应只消费离线标注，还应显式消费线上 proxy：用户是否继续追问、是否点击引用、是否复制输出、是否触发人工接管或二次求助、是否触发安全拦截、工具是否成功闭环。

这些 proxy 不是最终标签，但能帮助决定哪些样本值得优先抽样。线上回流必须保留事件上下文，而不只是 prompt 和最终回答：多轮上下文、检索证据、工具调用轨迹、候选回答、judge 分数、人审批注、线上结果都应进入数据引擎。

长尾错误回流最好先做 taxonomy，例如 factuality、style、tool use、memory、safety、refusal、formatting、business policy。taxonomy 的价值在于把错误送到不同修复路径，而不是统统归为“模型质量不够好”。

```mermaid
flowchart LR
    A["线上失败 / 人工接管"] --> B["错误 taxonomy"]
    B --> C["candidate 生成"]
    C --> D["judge 粗筛"]
    D --> E["人审复核"]
    E --> F["导出: SFT / DPO / RM / 回归集"]
    F --> G["新模型"]
    G --> H["shadow / 灰度"]
    H --> A
```

这张图说明后训练数据引擎不是“收集一些 prompt 再微调”。每个失败样本都要经过归因、候选生成、judge、人审和导出目标选择。否则数据越堆越多，真正修复的问题却不一定变多。

### 导出前检查

| 检查项 | 目的 |
| --- | --- |
| 错误类型是否明确 | 避免把不同问题混成一个训练集 |
| candidate 是否有真实分歧 | 避免 judge 只学格式偏好 |
| judge-human disagreement 是否记录 | 用于校准 judge |
| 是否进入回归集 | 防止同类问题复发 |
| 是否有线上指标闭环 | 验证训练是否真的改善产品 |

### 真实场景：Judge 分数涨了，线上人工接管没降

一个后训练飞轮常见日志如下：

```text
judge_total_score: 7.8 -> 8.6
format_score:      8.1 -> 9.3
tool_success:      71% -> 73%
human_handoff:     12.4% -> 12.1%
repeat_question:   18.0% -> 21.6%
```

如果只看 judge 总分，会误判新数据引擎有效。把样本分桶后可能发现，candidate 生成主要制造了格式差异，judge 又偏爱结构化长答案；真正导致人工接管的工具参数错误、引用缺失和多轮状态遗忘没有被抽中。

排查链应该是：

```text
症状：离线 judge 分数上涨，线上接管率和重复追问不降
观察：candidate disagreement 来源、judge-human disagreement、任务桶接管率、工具返回码
判断：数据引擎优化了 judge 容易看的格式，而非真实失败模式
修复：把线上接管样本设为高优先 bucket，引入执行验证器，限制纯格式类样本占比
反例：如果产品目标正是合规格式，format score 提升本身有价值
边界：judge 适合放大评价吞吐，不适合单独定义“产品变好”
```

## 七、飞轮验收清单

一个后训练数据引擎是否成熟，可以看七件事：是否明确定义高价值样本，candidate 生成是否提供真实分歧，judge 是否多维、校准并有 holdout，人审是否集中在高 ROI 样本，SFT、偏好、RM/Judge、回归集导出是否分清，线上信号是否能在可控周期内回流，新版本上线后是否能归因“哪类错误真的下降”。

后训练飞轮常见失效不是算法不够新，而是采样没对准高价值问题、judge 偏差没校准、人审花在平均样本、训练导出没有针对性、回归集缺失。最终判断很直接：谁能更稳定地把真实失败样本变成下一轮有效监督信号，谁就更可能在后训练阶段形成持续优势。
