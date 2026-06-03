# 数据与数据集基础来源台账

## 使用方式

这份台账服务于 [数据与数据集基础](../foundations/data-and-dataset-basics.md)。正文不把数据写成“文件夹/表格/样本数”，而是把数据集解释成样本契约、标签来源、processor、split、去重、metadata、版本和使用边界的组合。

第一类来源是数据文档框架。Datasheets for Datasets 和 Data Cards 让正文把 dataset documentation 讲成可追问、可审计、可复现的接口，而不是文末礼貌性说明。

第二类来源是大规模预训练语料报告。The Pile、Dolma、FineWeb、DataComp 和 C4 这类资料帮助本站解释来源 mixture、过滤、去重、contamination、质量打分和数据版本为什么会改变模型能力。

第三类来源是数据污染和 benchmark 可靠性研究。Data contamination 相关论文支撑“近重复、题库改写、网页镜像、代码 fork、相邻轨迹片段都可能破坏评测”的写法。

第四类来源是具身智能数据。Open X-Embodiment、RT-X、BridgeData、DROID 等资料用来说明机器人数据的样本边界是轨迹而不是图片，metadata 还必须包含 embodiment、相机、控制频率、动作接口和成功信号。

## 写作边界

正文不把“数据越多越好”当成默认结论。更多数据只有在增加有效覆盖、降低污染、保持标签质量和保留关键长尾时才有意义。

正文不把 processor 当成预处理小节。Tokenizer、image processor、video sampler、document chunker 和 robot data loader 都会改变模型实际看到的分布。

正文不把 random split 当作默认可靠。实体、时间、来源、轨迹和隐藏 holdout 的切分方式要随任务而变。

## 来源

| 来源 | 类型 | 本站使用方式 | 关联主题 |
| --- | --- | --- | --- |
| [Datasheets for Datasets](https://arxiv.org/abs/1803.09010) | 论文 | 用来解释数据集文档应覆盖来源、用途、采集、处理、分发、维护和风险。 | 数据文档 |
| [Data Cards](https://arxiv.org/abs/2204.01075) | 论文 | 使用 typology 图说明 producers、agents、users 对数据集文档有不同问题。 | 数据卡 |
| [The Pile](https://arxiv.org/abs/2101.00027) | 论文 | 参考大规模语言数据 mixture、来源说明、去重和 benchmark contamination 检查。 | 预训练语料 |
| [Dolma](https://arxiv.org/abs/2402.00159) | 论文 | 参考开放预训练语料的来源、过滤、去重、版本和公开数据管线写法。 | 数据管线 |
| [FineWeb](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1) | 技术博客 | 参考网页语料处理、去重、过滤和 benchmark 表现之间的讲法。 | 网页数据 |
| [DataComp](https://arxiv.org/abs/2304.14108) | 论文 | 说明数据过滤策略本身可以被系统 benchmark，而不只是模型结构附属。 | 数据过滤 |
| [C4](https://www.jmlr.org/papers/v21/20-074.html) | 论文 | 参考 cleaned Common Crawl 作为文本预训练数据管线的早期代表。 | 清洗语料 |
| [Measuring Data Contamination in Modern Benchmarks](https://arxiv.org/abs/2310.17623) | 论文 | 支撑 benchmark contamination 和近重复风险的讨论。 | 数据污染 |
| [Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499) | 论文 | 支撑去重能降低记忆和提升训练效率，但也会改变分布的讨论。 | 去重 |
| [Dataset Cartography](https://arxiv.org/abs/2009.10795) | 论文 | 用训练动态分析 easy/ambiguous/hard 样本，支持按样本难度追踪数据质量。 | 数据分桶 |
| [Open X-Embodiment](https://arxiv.org/abs/2310.08864) | 论文 | 使用 Figure 1 说明跨机器人、场景、物体、技能和轨迹的异质性。 | 机器人数据 |
| [RT-X project](https://robotics-transformer-x.github.io/) | 项目页 | 参考开放机器人数据集和 RT-X 模型的公开说明。 | VLA 数据 |
| [BridgeData](https://rail-berkeley.github.io/bridgedata/) | 项目页 / 论文 | 参考真实机器人轨迹数据的任务、视角、动作和语言标注。 | 轨迹数据 |
| [DROID](https://droid-dataset.github.io/) | 项目页 / 论文 | 参考大规模真实机器人操作数据采集和多场景覆盖。 | 机器人数据 |
| [Hugging Face Datasets documentation](https://huggingface.co/docs/datasets/index) | 官方文档 | 参考 dataset split、features、loading、streaming 和 dataset card 组织方式。 | 数据工具 |
