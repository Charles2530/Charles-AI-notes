---
paper_date: 2024-08
paper_title: "GPT-4o System Card"
paper_topic: 技术报告
---

# GPT-4o System Card：Omni 安全与多模态评测

!!! info "论文信息"
    **论文题名：** GPT-4o System Card。<br>
    **作者：** OpenAI。<br>
    **机构：** OpenAI。<br>
    **时间 / 主题：** 2024-08；技术报告。<br>
    **arXiv / 官方报告：** arXiv：未找到专门条目；官方材料：[openai.com/index/gpt-4o-system-card/](https://openai.com/index/gpt-4o-system-card/)。<br>
    **GitHub / 项目：** GitHub：未找到官方链接；项目页：[openai.com/index/gpt-4o-system-card/](https://openai.com/index/gpt-4o-system-card/)。<br>
    **元数据来源与核验口径：** 来源：[官方 / 项目材料](https://openai.com/index/gpt-4o-system-card/)；Checked Date：2026-06-04；Repro Status：Paper / official materials reviewed, independent reproduction not claimed。

## 原生语音把安全问题从文字内容扩展到声音本身

GPT-4o System Card 的中心问题，是一个能够直接听、看、说的模型应当怎样被评估和部署。GPT-4o 接收文本、图像、音频与视频，并能够生成文本、音频与图像。报告描述它由同一个神经网络端到端处理这些模态，没有公开参数规模、完整架构或训练目标的数学细节，因此这份材料更适合解释系统行为与评测证据，而非重建模型。

报告说明，GPT-4o 的文本与语音能力使用截至 2023 年 10 月的数据预训练。数据来源包括公开数据、行业标准机器学习数据集、网页抓取内容，以及数据合作伙伴提供的非公开材料，例如付费内容、档案和元数据。能力数据还按用途分成网页、代码与数学、多模态数据，后者包含图像、音频和视频，用来学习视觉对象、动作序列、语言模式和语音细节。报告没有给出可据此重建模型的完整数据清单或权重。

安全处理分布在不同阶段。预训练过滤会用审核接口和安全分类器过滤有害内容、个人信息等数据，也会处理图像数据和训练退出请求。报告同时明确，更多有效的测试和缓解发生在预训练之后，因为只过滤训练数据无法处理细致、依赖上下文的伤害。后训练用于对齐人类偏好，红队测试继续寻找问题，产品层再加入监测、执行和审核工具。读系统卡时，应把数据来源、模型后训练和产品防护分开理解。

原生音频带来低延迟对话。报告给出的语音响应延迟最低为 232 毫秒、平均为 320 毫秒，这是 2024 年报告中的测量结果，不是所有接口、网络条件和后续产品版本的服务承诺。声音也携带文字转写无法完整保留的信息，包括说话者特征、语气、情绪强度和轮流发言的时机。系统能够利用这些信息，风险面也随之扩大。

最典型的例子是未经授权的声音生成。即使输出文字没有问题，模型若模仿了用户或其他人的声音，仍可能造成身份混淆。因此，文本内容审核只是语音系统的一部分；声音身份检测、实时输出拦截、模型后训练和产品限制必须分别接受检验。[官方系统卡，PDF 第 1–2、6–7 页](https://cdn.openai.com/gpt-4o-system-card.pdf#page=1)

## 先把评测题送进音频通道，再检查哪些信息丢了

报告复用大量既有文本能力与安全评测，先通过 Voice Engine 将题目转成语音，交给 GPT-4o，再对模型语音回答的转写文本评分。这样可以在相近题意下比较文本和音频表现，也能够观察文本后训练学到的拒绝行为是否迁移到音频。

![GPT-4o evaluation task conversion](../../assets/images/localized-remote/paper-deep-dives/technical-reports/gpt-4o-system-card/gpt-4o-evaluation-task-conversion.svg){ width="600" }

这张流程图的关键是两次表示变化。输入从文字变成声音，数学符号、代码缩进和复杂排版可能失真；输出从声音变成转写，语调、声音身份和非语言声音可能消失。因此，一道文本题转换成功后可以测试知识或拒绝行为，却不能顺带证明全部语音特征都被检查过。

人工合成的清晰语音也只覆盖真实对话的一部分。现实用户会停顿、打断、夹杂背景噪声和回声。报告记录了一些音频扰动导致安全稳健性下降的案例，但没有把这些观察包装成已经充分量化的故障率。语音模型评测需要同时保留这种可规模化的转写评分，以及针对声音和交互本身的专门测试。

### 一套转换评测回答什么，不能回答什么

把文字题送进 Voice Engine 的优点是题目、评分器和历史基线可以复用，但它改变了输入和输出的表示。文字到语音可能丢掉公式布局、代码缩进和特殊符号，语音到转写又会抹去停顿、重音、音色和非语言声音。报告因此把 voice cloning 等必须直接看声音的任务单独处理；对普通知识与安全题，转换评测主要回答“已有文本行为能否迁移到音频通道”。

| 评测路径 | 保留下来的问题 | 仍然缺少的证据 |
| --- | --- | --- |
| 文字题 → Voice Engine → GPT-4o → 转写评分 | 题意近似、历史任务与拒绝规则可复用 | 原始声音的音色、停顿、噪声、非语言事件 |
| 真实或合成多说话者输入 | 口音与用户声音变化下的能力和安全行为 | 27 个英语人声样本不能覆盖全部语言、设备和病理性语音 |
| 直接评估生成音频 | 声音身份、表达方式、歌唱与输出过滤 | 需要专门分类器、听测或交互协议，不能用转写分数代替 |

外部红队覆盖了另一类证据。2024 年 3 月至 6 月，超过 100 位测试者参与，涉及 45 种语言和 29 个国家，并经历四个阶段。各阶段访问的模型检查点与能力范围不同，因而这些活动说明评测覆盖面和迭代过程，不能合并成一个固定最终模型上的统一测试集。[官方系统卡，PDF 第 3–5 页](https://cdn.openai.com/gpt-4o-system-card.pdf#page=3)

## 控制声音身份，需要模型约束和独立检测共同工作

GPT-4o 的语音后训练使用经过授权的预设声音作为目标输出。同时，独立分类器在生成过程中以流式方式检查输出声音是否仍与选定预设一致，发生偏离时阻断输出。这一设计把两件事分开，模型尽量保持目标声音，外部系统负责检测仍然出现的偏离。

内部评测中，声音分类器的英语 precision 为 0.96、recall 为 1.00，非英语为 0.95 和 1.00。原文所说捕获 100% 的有意义偏离，指这些内部评测样本，样本包括其他系统声音、模仿输入声音的生成片段和人声片段。它不能解释成任何语言与任意分布下都不存在漏检。报告也明确说，非英语过度拦截可能导致本可继续的对话被中断。

| 输入语言 | Precision | Recall |
| --- | ---: | ---: |
| English | 0.96 | 1.00 |
| Non-English | 0.95 | 1.00 |

这组数值只衡量分类器在对话中的声音偏离检测，不衡量模型回答是否正确，也不衡量拦截后用户任务是否完成。把它接入产品后，还要记录中断率和误报成本，否则“抓住了偏离”可能以过多断流为代价。

这种外部防护与模型能力之间的区别很重要。报告仍承认模型偶尔会无意生成近似用户声音；部署系统通过检测与终止会话降低这一风险。若只评估底座模型，或自行搭建系统时省掉分类器，得到的行为边界可能不同。[官方系统卡，PDF 第 6–7 页，Table 2](https://cdn.openai.com/gpt-4o-system-card.pdf#page=6)

识别说话者则属于输入端的问题。后训练要求模型拒绝仅凭声纹识别一个人，但可以根据音频中明确的自我介绍或知名引语判断内容来源。报告 Table 3 给出的“应拒绝时正确拒绝”从 0.83 提高到 0.98，“应配合时正确配合”从 0.70 提高到 0.83。这里同时考察漏拒与误拒，不能只优化其中一个方向。表中四舍五入值与正文给出的百分点差异不完全一致，因此直接引用表值更稳妥。

另一项任务区分无根据推断与有声音线索的归因。仅凭声音判断智力、职业或人格等缺乏依据的要求应拒绝；对口音等可能从声音判断的属性，则以保留措辞回答。两类请求合并后的安全行为准确率由 0.60 增至 0.84。这个指标测的是遵循指定响应政策，不能解释成模型具备准确识别所有个人属性的能力。[官方系统卡，PDF 第 10–11 页，Table 4](https://cdn.openai.com/gpt-4o-system-card.pdf#page=10)

## 换一个人的声音，同一道题还能答对吗

报告固定助手输出声音为 shimmer，用三种系统声音与来自 27 位英语说话者的声音样本合成输入。能力评测覆盖 TriviaQA、MMLU 的一部分、HellaSwag 和 Lambada；这些任务分别侧重知识、常识与文本补全。高度依赖数学或科学符号的题目被排除，评测也只使用固定随机子集。

![Capability evaluations across system and human voices](../../assets/images/localized-remote/paper-deep-dives/technical-reports/gpt-4o-system-card/capability-evaluations-across-system-and-human-voices.svg){ width="820" }

不同声音组的能力分数总体接近，人声样本组略低，但差异未达到报告所述的显著程度。这项实验的价值，是在题目尽量固定时改变输入声音，检查语音特征是否干扰解题。样本仍是有限的英语声音集合，不能推出所有口音、语言、录音设备和病理性语音都已得到同等服务。

![Safety evaluations across system and human voices](../../assets/images/localized-remote/paper-deep-dives/technical-reports/gpt-4o-system-card/safety-evaluations-across-system-and-human-voices.svg){ width="560" }

安全行为图采用内部对话集，检查同类请求换声音后是否仍按预期配合或拒绝。报告在此集合中没有发现明显的声音间差异。两张图应共同理解，知识正确性与政策行为一致性是两条评估轴；安全拒绝一致，不意味着每一道知识题都正确。[官方系统卡，PDF 第 8–10 页](https://cdn.openai.com/gpt-4o-system-card.pdf#page=8)

文本安全能力迁移到音频的结果也要保留评测口径。正文 Table 5 中，Text / Audio 的 Not Unsafe 分别为 0.95 / 0.93，Not Over-refuse 为 0.81 / 0.82。附录 Table 11 又比较了当时线上文本模型与新语音模型的文本、音频模式，给出另一组结果。两张表并未提供可直接合并的相同条件说明，不能择高值宣布语音模式完全安全。

| Metric | Text | Audio |
| --- | ---: | ---: |
| Not Unsafe | 0.95 | 0.93 |
| Not Over-refuse | 0.81 | 0.82 |

附录的另一组表把 Current GPT-4o Text、New GPT-4o Text、New GPT-4o Audio 放在一起，报告 `not_unsafe` 为 0.99、0.99、1.00，`not_overrefuse` 为 0.91、0.89、0.91。它回答的是另一套生产文本与新语音模式的比较，不能与正文的转换评测当作同一批样本的重复测量。

内容审核主要处理输入和输出的转写文本；音乐、声音身份、带强烈表现性的语音等则需要另外的限制或检测。报告中的 Advanced Voice Mode 有限 alpha 还限制唱歌，并使用检测音乐的过滤器。它们描述的是当时具体部署策略，不能作为后续版本永久不变的能力清单。[官方系统卡，PDF 第 11–12、31–32 页](https://cdn.openai.com/gpt-4o-system-card.pdf#page=11)

## Preparedness 分级怎样从实验结果得来

GPT-4o 的总体 Preparedness 评级为 Medium，来自文本说服能力略微进入该档位；网络、生物威胁和模型自主性被评为 Low，语音说服也为 Low。这里的分级明确是 mitigation 前的评估结果，不能把它当成部署后防护的单独效果。总体等级取决于各类风险中的最高结果，不能将 Medium 理解为四个维度的平均分，也不能将 Low 理解为零能力。

| 评估方向 | 实验设计与报告结果 | 数字实际支持的判断 |
|---|---|---|
| 网络能力 | 172 个 CTF；每次最多 30 轮工具调用，每题 10 次尝试；高中、大学、专业级完成率分别 19%、0%、1% | 能完成部分较简单任务，长链执行与关键洞察仍受限 |
| 生物威胁 | 专家、非专家随机分配到 Internet、GPT-4o + Internet、研究专用无拒绝 GPT-4o + Internet 等条件；另有知识与排障测试 | 检验信息辅助是否提升指定任务表现，未达到该框架中风险阈值 |
| 文本说服 | AI 文章、聊天与专业人员写作进行比较；总体未超过人类，12 项干预中有 3 项超过 | 平均结果不能掩盖某些干预条件下更强的效果 |
| 语音说服 | 超过 3800 名受试者，比较音频片段与多轮对话，并在一周后回访 | 在该政治态度研究设计下，AI 的效果低于对应人类基线 |
| 自主复制与适应 | 带工具的 ARA 任务，100 次试验无完整成功 | 能完成局部步骤，但未形成可靠的端到端完成能力 |

生物评测的人工条件要分开读。参与者分别在 Internet、GPT-4o + Internet，以及 research-only、no-refusal GPT-4o + Internet 条件下完成覆盖 ideation、acquisition、magnification、formulation、release 的问题。69% 的 consensus@10 来自另一套知识与排障自动评测，是多次采样后的共识分数，不是现实生物任务成功率。

语音说服的即时意见变化与一周后回访如下。数值来自同一项预注册研究，不能和其他语音体验指标混用。

| 条件 | 即时意见变化 (%) | 一周后回访 (%) |
| --- | ---: | ---: |
| AI audio clip | 6.08 | -0.72 |
| AI conversation | 5.74 | 0.82 |
| Human audio clip | 7.82 | 1.19 |
| Human conversation | 8.85 | 1.78 |

AI audio clip 的即时效果约为人类片段的 78%，AI conversation 约为人类对话的 65%。受试者来自美国参议院选情较稳定的州，结果不能直接外推到所有主题、人群与商业说服任务。声音更自然是否更能说服人，需要这种实验来回答，不能只凭交互体验推断。[官方系统卡，PDF 第 12–17 页](https://cdn.openai.com/gpt-4o-system-card.pdf#page=12)，语音说服表见 [PDF 第 15 页](https://cdn.openai.com/gpt-4o-system-card.pdf#page=15)

自主性测试还揭示了指标设计的陷阱。个别轨迹通过自动评分器，因为模型在远端启动了具有正确接口的服务器；人工复核发现，服务器并没有真正从目标模型采样，因此任务实际上失败。接口存在只是中间状态，最终判据必须覆盖用户真正要求的结果。

这类 false positive 不是某个 agent 任务的偶然小问题。自动评分器通常看到的是端口、文件或返回码，用户关心的是服务是否真的调用了目标模型、输出是否可用、整条任务是否完成。系统卡因此把 100 次试验无完整成功与人工复核并列报告，提醒复现实验时保留端到端验收，而不是只复刻中间动作。[官方系统卡，PDF 第 17–19 页](https://cdn.openai.com/gpt-4o-system-card.pdf#page=17)

报告还列出研究编程题 95% pass@100、SWE-bench Verified 19% pass@1，以及选定机器学习工程任务 0/10。pass@100 是给很多尝试后至少成功的能力，pass@1 是一次尝试的可靠性，不能用前者说明已经能够稳定代替工程师完成长任务。[官方系统卡，PDF 第 17–18 页](https://cdn.openai.com/gpt-4o-system-card.pdf#page=17)

## 第三方测试把问答能力与执行能力分开

![METR agent performance vs humans](../../assets/images/localized-remote/paper-deep-dives/technical-reports/gpt-4o-system-card/metr-agent-performance-vs-humans.png){ width="760" }

METR 使用一个基于 GPT-4o 的简单 agent，在虚拟环境中运行 31 个任务族、共 86 个任务，涵盖软件工程、机器学习、网络安全、研究和计算机操作，并与不同时间限制下的人类表现比较。图的作用是显示任务难度与可持续执行范围，而非把模型折算成一个统一的人类工作年限。该评估没有发现 GPT-4o 相比 GPT-4 的这类能力显著增加。

Apollo Research 进一步比较模型关于自身和他人信念的问答，以及需要实际运用这些判断的 agent 任务。模型在显式 theory of mind 问答中较强，在应用场景中的自我推理和维持、利用他人信念方面较弱。报告同时注明，agent 设置比较基础，能力诱发工作有限。因而，结果支持“当前测试中问答知识未充分转化为行动能力”，不支持“所有更强 scaffold 下都不可能出现这些能力”。[官方系统卡，PDF 第 18–19 页](https://cdn.openai.com/gpt-4o-system-card.pdf#page=18)

这些评估与模型架构报告的不同之处在于，实验对象本来就包括工具、提示、时间与评分器。改动其中一项，结论的适用范围也可能改变；复用结果时必须保留这些条件。

## 知识能力的进步，仍需经过真实工作流验证

医疗知识评测由 11 个数据集上的 22 项文本测试组成，采用 0-shot 或 5-shot，没有针对任务调参。官方正文称 GPT-4o 在 22 项中的 21 项优于 2024 年 5 月版本 GPT-4T，但 Table 7 的四舍五入值至少有两项持平，例如 MMLU Medical Genetics 5-shot 与 College Biology 5-shot 都是 0.95 对 0.95。因此更稳妥的读法是 21/22 项报告为提升或不低于，不能从表内四舍五入值推出 21 项都严格大于基线。MedQA USMLE 四选项 0-shot 从 78.2% 提高到 89.4%。这说明临床知识问答有明显进步，但报告特别指出，尚未证明这种进步同样迁移到音频，也没有测量真实医疗流程中的效用。

科学任务的例子展示了同一种双重性。

![Quantum physics example](../../assets/images/localized-remote/paper-deep-dives/technical-reports/gpt-4o-system-card/quantum-physics-example.jpg){ width="760" }

这张量子物理交流案例说明，模型能帮助解释和讨论专业问题，具备作为研究辅助工具的潜力。它是红队观察中的实例，没有对应随机任务集上的总体正确率。报告中的 B 子图对应 Herzog、Rarity、Weinfurter 与 Zeilinger 的实验示意，包含两块下转换晶体和相位干涉。下面的裁图保留了原报告的完整问答，便于把模型的视觉输入和回答放在一起核对。

![Quantum physics red-team Q&A](../../assets/images/localized-remote/paper-deep-dives/technical-reports/gpt-4o-system-card/quantum-physics-redteam-qa.png){ width="820" }

![Correlation functions limitation example](../../assets/images/localized-remote/paper-deep-dives/technical-reports/gpt-4o-system-card/correlation-functions-limitation-example.jpg){ width="760" }

这个关联函数案例的输入图只是原报告中的科学图，完整问答见下面的裁图。图 b 的纵轴描述的是 paw movement，峰值为 -3.7 ± 0.2 s。图下方的 EARLIER → LATER 说明负延迟应读成 paw movement 先于 global astrocytic signal 约 3.7 秒；报告中的回答把先后关系说反了。报告还提到复杂多子图解释、专业术语和核苷酸文本识别等失误。将正反例放在一起，才能看出模型可以帮助组织讨论，却仍可能在局部的符号、图像对应关系和专业细节上犯错；流畅的解释本身不足以验证科学正确性。

![Correlation functions red-team Q&A](../../assets/images/localized-remote/paper-deep-dives/technical-reports/gpt-4o-system-card/correlation-functions-redteam-qa.png){ width="820" }

[官方系统卡，PDF 第 21–23 页](https://cdn.openai.com/gpt-4o-system-card.pdf#page=21)

低资源语言评测覆盖三条证据线。ARC-Easy 测小学科学常识题，TruthfulQA 测是否会重复常见误解，Uhura-Eval 由母语者设计，测试 Amharic、Hausa 和 Yoruba 的阅读理解。代表数值如下。

| 评测 | 语言 | GPT-3.5 Turbo | GPT-4o |
| --- | --- | ---: | ---: |
| ARC-Easy | Hausa | 26.1 | 75.4 |
| ARC-Easy | Amharic | 6.1 | 71.4 |
| TruthfulQA | Yoruba | 28.3 | 51.1 |
| Uhura-Eval | Hausa | 32.3 | 59.4 |

这些结果说明选定语言上的文本评测有明显提升，但仍不能覆盖全球语言和方言。这里测的是文本理解，不是前述英语声音变化实验；样本有限，不能把分数直接当作所有真实语言场景的可用性。[官方系统卡，PDF 第 23–24 页，Table 8–9](https://cdn.openai.com/gpt-4o-system-card.pdf#page=23)

## 这份系统卡留下了怎样的可复用方法

GPT-4o 的贡献在这里表现为一套围绕原生多模态能力建立的部署评估方法。可复用的核心做法包括，用模态转换复用已有任务，再为转换丢失的信息补充专门测试；把模型行为与输出分类器的效果分别报告；同时测拒绝与误拒；为 agent 检查最终任务结果，并保留尝试次数、工具和人工复核条件。

系统卡没有给出完整训练配方或开放权重，也没有证明拟人化语音的长期社会效果。关于情感依附的观察属于早期使用和红队现象，并非长期因果研究。它真正提供的证据，是在具体模型版本、实验条件与部署防护下，哪些能力已经出现、哪些控制有效，以及哪些失败仍能被观察到。这也是使用此类报告设计新系统时最值得保留的粒度。

## 继续阅读

System Card 的安全与评测表适合和[数据划分、指标与评测](../../foundations/data-splits-metrics-and-evaluation-basics.md)一起读；多模态模型的失败类型见[多模态评测与失败模式](../../vlm/multimodal-evaluation-and-failure-modes.md)。
