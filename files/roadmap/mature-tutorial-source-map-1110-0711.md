# 成熟教程来源地图

这份清单只回答一个编辑问题：某个主题应该先去哪里核对概念、接口和实验口径。它不是待翻译文章目录。本站保留事实、公式和接口的出处，再用自己的例子、代码和失败记录重写页面。网页的许可证、图片和代码许可要按具体页面单独核对，不能因为一个项目有公开仓库就默认整站内容可以搬运。

## 目前优先使用的来源

| 来源 | 适合核对的主题 | 在本站的使用方式 | 当前访问核验 |
| --- | --- | --- | --- |
| [Stanford CS336 Spring 2025](https://cs336.stanford.edu/spring2025/) | tokenizer、Transformer、训练系统、数据和评测 | 借用课程的问题顺序，自己重写推导和实现；不复制讲义或作业 | 2026-11-10 返回 200 |
| [Dive into Deep Learning](https://d2l.ai/) | 从数学到代码的基础模型教程 | 用于检查符号、推导和可运行例子；公式按本站符号重写 | 2026-11-10 返回 200 |
| [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/) | policy gradient、actor-critic、RL 实验口径 | 用于核对算法定义和伪代码，再补 Gymnasium 接口与失败诊断 | 2026-11-10 返回 200 |
| [Hugging Face TRL](https://huggingface.co/docs/trl/) | SFT、DPO、PPO、GRPO 训练器和日志接口 | 只核对当前 trainer 参数、mask 和日志字段，偏好数据算例与评测协议由本站重做 | 2026-11-10 入口已记录；具体 trainer 版本需随环境固定 |
| [Gymnasium Tutorials](https://gymnasium.farama.org/tutorials/) | environment API、terminated、truncated、wrapper | 直接核对当前 API 语义，页面中只保留必要的短代码和链接 | 2026-11-10 返回 200 |
| [PyTorch Tutorials](https://docs.pytorch.org/tutorials/) | autograd、模块参数、分布式和性能 | 用于核对参数布局、梯度和 API 行为；本地例子单独注明是否实际运行 | 2026-11-10 返回 200 |
| [Transformers quantization](https://huggingface.co/docs/transformers/en/quantization/bitsandbytes) | bitsandbytes、4-bit/8-bit 加载、计算 dtype 和模块跳过 | 只核对当前配置字段与加载边界，页面中的显存数字和训练回归由本站重算 | 2026-11-10 搜索抓取成功；字段需随 Transformers 版本复核 |
| [PEFT quantization guide](https://huggingface.co/docs/peft/en/developer_guides/quantization) | LoRA/QLoRA 注入、adapter 保存与合并 | 借鉴 adapter 生命周期和可训练参数检查，不复制教程代码块 | 2026-11-10 入口已记录；安装版本与底座架构仍需本地验证 |
| [vLLM Documentation](https://docs.vllm.ai/en/latest/) | serving、KV cache、调度、指标和版本接口 | 绑定文档版本或源码 commit，不能用当前文档替代旧版本行为 | 2026-11-10 返回 200；metrics design 已抓取并用于 serving trace |
| [Hugging Face Learn](https://huggingface.co/learn) | Transformer、扩散、RL 和课程化实践 | 只作路线和术语入口，具体接口回到对应项目仓库和版本文档 | 本次请求被远端重置，待再次核验 |
| [LeRobot Documentation](https://huggingface.co/docs/lerobot/) | 机器人数据、策略训练、评测和硬件接口 | 作为机器人实践入口，必须记录版本、硬件和数据格式 | 2026-11-10 搜索抓取成功；LeRobotDataset v3 页面已核对字段、timestamp、episode metadata 与 `delta_timestamps` |
| [Robotics Book](https://www.roboticsbook.org/intro.html) | 机器人系统、坐标、运动学与控制入门 | 借用问题拆分，公式和示例按本站坐标约定重写 | 页面入口已记录，图和代码许可仍按页核对 |
| [Modern Robotics](https://modernrobotics.northwestern.edu/nu-gm-book-resource/introduction/) | 螺旋理论、FK/IK、Jacobian 和轨迹 | 用于和 DH、指数积、body/space frame 做参数化对照 | 课程页面与代码仓库分别核对许可 |
| [MIT Manipulation](https://manipulation.mit.edu/) | 感知、规划、控制和操作闭环 | 参考系统任务顺序，本站只重绘接口图并绑定本地检查 | 课程材料中的图片和代码需单独核验 |

## 一页内容怎样引用这些来源

先选一个主要来源，回答它解决的具体问题，再补一份能检查边界的来源。例如 `terminated` 与 `truncated` 以 Gymnasium 教程为主，价值目标再回到 Sutton 与 Barto；vLLM 的调度页面以官方文档和固定 commit 为主，延迟指标定义再对照服务系统页面。

一页正文至少要留下四件能被复查的东西：

1. 原始链接、访问日期和版本或 commit。
2. 哪些句子是事实转述，哪些公式和例子由本站重新推导。
3. 本地代码是否真正运行过，运行环境和输出是什么。
4. 该来源没有回答的限制，例如数据分布、硬件、评测分母或接口自动 reset。

如果一个博客的解释很清楚但许可证不明，只提取问题、术语和阅读入口，正文改用自己的例子。若必须保留原文结构或近似翻译，先核对明确许可，再把作者、原题名、链接、许可证和改动范围写在页面中。

## 不能由来源直接推出的结论

- 教程里的一次 benchmark 不能直接变成本地机器的速度承诺。
- 论文中的图和表不能因为正文有引用就自动复制，优先自己重绘示意图。
- 一个仓库的许可证不一定覆盖其中的论文图、第三方数据和网页文字。
- “没有查到长句重复”只能降低逐句复制的疑虑，不能替代结构、近似翻译和素材许可审计。

这份清单会随版本变化更新。动态文档优先在页面正文写明核验日期，旧版本的结论保留在审计日志中。
