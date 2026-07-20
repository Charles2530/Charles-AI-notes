# Sim2Real 与具身数据引擎：真实失败怎样变成下一轮能力

Sim2Real 要让仿真里的任务、资产、传感器和物理参数接近真实部署。真实机器人失败时，问题往往来自夹爪摩擦不对、透明包装反光、相机外参漂移、桌面遮挡、控制器延迟、物体比仿真更软，或者人突然把目标物挪走。

仿真、真实日志和评测 hard set 需要形成闭环，让每轮真实失败都变成下一轮训练资产。

结论先说清楚：Sim2Real 的核心是一套数据引擎。仿真负责低成本覆盖变量，system identification 负责把仿真中心调准，真实部署日志负责暴露长尾，数据筛选和标注负责把失败变成训练样本，hard set 负责验证新模型是否真的补上短板。只做其中一环，系统很容易在演示里漂亮、上线后脆弱。

## Sim2Real gap 是三种分布差

可以把仿真和真实的差异写成：

\[
p_{\mathrm{sim}}(o_{t+1}, s_{t+1}, y_t\mid o_t,s_t,a_t,c)
\neq
p_{\mathrm{real}}(o_{t+1}, s_{t+1}, y_t\mid o_t,s_t,a_t,c)
\]

这里 \(o_t\) 是观测，\(s_t\) 是状态，\(a_t\) 是动作，\(c\) 是任务或环境上下文，\(y_t\) 是成功、失败、接触、碰撞等结果标签。这个式子的意思是：同样的观测、状态和动作，在仿真和真实世界里可能产生不同的下一步观测、下一步状态和任务结果。

差异通常来自三层。

| 层 | 典型差异 | 会造成什么失败 |
| --- | --- | --- |
| 观测差 | 光照、纹理、反光、运动模糊、相机外参、深度噪声 | 看错物体、误判距离、找不到边界 |
| 动力学差 | 质量、摩擦、刚度、接触、控制延迟、夹爪形变 | 抓空、打滑、推偏、插不进去 |
| 任务差 | 家庭杂乱、物体摆放、人的干扰、语言粒度、成功标准 | 任务中断、错误回收、无法恢复 |

很多 Sim2Real 讨论只盯第一层：画面像不像。机器人部署却经常死在第二层和第三层：视觉识别对了，但接触错了；抓到物体了，但长任务中判断错了下一步。

## 仿真首先是任务数据工厂

高质量仿真不是一张漂亮 3D 图，而是一套可复现的数据工厂：资产、场景、机器人、相机、动作接口、专家轨迹、扰动、成功判定、失败日志和导出 schema 都要在一起。

![RoboTwin 2.0 benchmark overview 官方图](../assets/images/localized-remote/embodied-ai/isaacsim-and-robotwin/robotwin-2-0-benchmark-overview-官方图.png){ width="900" }

<small>图源：[RoboTwin 2.0 官方文档](https://robotwin-platform.github.io/doc/)。原图展示双臂任务、资产、domain randomization、专家数据、benchmark 和 Sim2Real 组织方式。读图重点：仿真数据能否进入 VLA/世界模型训练，不取决于渲染是否漂亮，而取决于任务脚本、success checker、轨迹导出和失败回放是否形成系统。</small>

这张图应该按流水线读。一个仿真任务至少要回答：物体资产从哪里来，碰撞网格和抓取点是否标好，任务如何重置，专家轨迹如何生成，成功条件如何判定，失败时保存哪些中间状态，导出的数据能否和真实机器人日志放进同一套 schema。

如果只导入一个物体、跑一段看起来顺的轨迹，这还不是数据引擎。数据引擎要求每条轨迹失败时能追溯：是资产尺度错、碰撞网格错、摩擦错、相机错、planner 错、controller 错，还是 success checker 被钻空子。

## Domain randomization 让策略别太脆

Domain randomization 的想法是承认仿真不可能完全真实，于是主动随机化环境参数：

\[
\xi\sim p(\xi),\qquad
\max_\pi \mathbb{E}_{\xi\sim p(\xi)}[R(\pi;\xi)]
\]

这里 \(\xi\) 是仿真参数，可以包含光照、纹理、相机、质量、摩擦、接触、背景、控制延迟；\(p(\xi)\) 是参数分布；\(R(\pi;\xi)\) 是策略 \(\pi\) 在这一组参数下的回报。这个目标的意思是：策略不要只适应一个完美实验室，而要在一批有差异的世界里都能完成任务。

随机化不只是视觉增强。视觉随机化能防止模型记住固定灯光和纹理；动力学随机化能防止模型假设物体永远一样重、一样滑、接触永远理想；相机和控制随机化能防止模型过拟合精确标定和零延迟控制。

Domain randomization 最容易犯两个错误。第一，随机化范围太窄，真实世界仍然落在分布外。第二，随机化太宽，任务变成噪声海，策略学不到可执行规律。实用做法通常是先用真实数据估计中心范围，再围绕中心做有任务意义的随机化。

## System ID 把仿真中心调准

另一条路线是 system identification：用真实轨迹反推仿真参数，让仿真更接近真实系统。

\[
\hat{\xi}
=\arg\min_\xi
\mathcal{D}\big(\tau_{\mathrm{real}}, \tau_{\mathrm{sim}}(\xi)\big)
\]

这里 \(\tau_{\mathrm{real}}\) 是真实机器人轨迹，\(\tau_{\mathrm{sim}}(\xi)\) 是参数为 \(\xi\) 的仿真轨迹，\(\mathcal{D}\) 是两条轨迹的差异度量。这个式子想表达的是：调质量、摩擦、相机、控制延迟等参数，让仿真行为更接近真实回放。

System ID 和 domain randomization 不是互斥关系。System ID 负责把中心调准，randomization 负责承认中心附近还有不确定性。机械装配、插拔、推拉、软体和夹持任务里，单靠视觉随机化通常不够，接触动力学和控制延迟的校准会更重要。

## 数据引擎把失败变成资产

现代机器人系统不是一次性训练完成，而是一个循环：

\[
\mathcal{D}_{k+1}
=\mathcal{D}_k
\cup
\mathrm{Gate}\big(\mathrm{Label}(\mathrm{Select}(\mathcal{L}_k))\big)
\]

这里 \(\mathcal{D}_k\) 是第 \(k\) 轮训练数据，\(\mathcal{L}_k\) 是部署日志，`Select` 负责筛出有价值片段，`Label` 负责标注失败类型、阶段和恢复动作，`Gate` 负责决定哪些样本进训练集、哪些进 hard set、哪些只做诊断。这个公式的重点是：线上日志不能直接倒进训练集，它必须经过筛选、标注和门禁。

值得回流的样本通常不是普通成功片段，而是：

| 样本 | 为什么值钱 | 回流后补什么 |
| --- | --- | --- |
| 明确失败 | 抓空、掉落、碰撞、卡住、误按 | 修正模型当前短板 |
| near-miss | 差一点撞、差一点掉、低置信但成功 | 提前补安全边界 |
| 人工接管 | 人类在关键状态接手并恢复 | 学习纠偏策略 |
| OOD 观测 | 反光、透明、遮挡、杂乱、新背景 | 补视觉长尾 |
| 任务阶段混乱 | 该放下却继续抓、该重看却继续执行 | 补高层状态和语言阶段 |

这也是为什么失败样本比普通成功样本更值钱。继续收一千条“干净桌面抓红杯成功”，对透明包装抓空没有帮助。真正有用的是把透明、反光、塌陷、遮挡、抓偏后恢复这些样本单独分桶，并在下一轮 hard set 上验证。

## 跨机器人数据先解决接口异构

Open X-Embodiment 这类数据集说明：具身数据不是图片数据。不同机器人本体、相机、动作空间、控制频率、任务语言和成功定义混在一起，数据量越大，接口噪声也越可能被放大。

![Open X-Embodiment dataset overview 原论文图](../assets/images/paper-figures/embodied-ai/open-x-embodiment-figure-1-dataset.png){ width="780" }

<small>图源：[Open X-Embodiment / RT-X](https://robotics-transformer-x.github.io/)，Figure 1。原图统计不同 robot embodiment、scene、trajectory、skill 和 object category 的分布。读图重点：跨机器人数据的价值是扩大经验覆盖；风险是动作、相机和任务语义不对齐时，模型学到的不是泛化，而是混乱接口。</small>

跨机器人数据至少要对齐五件事：观测口径、动作口径、时间戳、任务语言和结果标签。比如同样是 gripper close，不同夹爪的力度、速度和闭合定义可能不同；同样是相对位移，不同机器人坐标系和控制频率会让动作语义改变。数据引擎如果不记录这些 metadata，后续模型很难知道哪个差异来自任务，哪个差异来自机器人本体。

## 数据源贡献要做消融

π0.5 的数据消融图适合说明一个关键点：数据 recipe 决定模型补上哪种能力。Web 数据补语义，跨机器人数据补动作和本体覆盖，多环境真实数据补场景泛化，移动操作数据补长任务和开放家庭分布。

![π0.5 data ablation 原论文图](../assets/images/paper-deep-dives/embodied-ai/pi05/figure-10-data-ablation.png){ width="900" }

<small>图源：[π0.5](https://arxiv.org/abs/2504.16054)，Figure 10。原图比较去掉 web data、cross-embodiment data、multi-environment data 等来源后的表现变化。读图重点：数据引擎不能只报告总样本量，要说明每类数据源补的是语义、动作、环境、长任务还是恢复能力。</small>

读这类图时不要只问“哪种数据最重要”。更好的问题是：当前系统失败在哪个桶？如果失败是语义理解错，web/VLM 数据可能有帮助；如果失败是新厨房布局，multi-environment 数据更关键；如果失败是动作不稳，跨机器人数据未必够，还要看动作接口、controller 和失败恢复轨迹。

## Hard set 是数据引擎的验收器

数据回流是否有效，不能只看下一轮平均成功率。平均分很容易被简单场景淹没。更可靠的是维护一组 hard set：每个 hard bucket 对应一种系统短板。

| hard bucket | 例子 | 验收指标 |
| --- | --- | --- |
| 透明/反光物体 | 透明塑料盒、金属杯、玻璃门 | 抓取成功率、误检率 |
| 接触动力学 | 滑动物体、软袋、卡扣、插拔 | 打滑率、卡住率、恢复率 |
| 相机/标定扰动 | 外参轻微偏、深度噪声、遮挡 | 定位误差、重观察成功率 |
| 长任务阶段 | 收纳、整理、开合柜门后继续操作 | 阶段完成率、错误恢复 |
| 人类干扰 | 物体被移动、临时口头修正 | 接管率、安全停止率 |

每次训练后，数据引擎都要回答：新增数据改善了哪个 bucket？有没有污染其他能力？失败样本进训练后，模型是否在原失败场景上真正恢复，还是只在平均任务上看起来更好？

## 读 Sim2Real 论文时问四个问题

第一，仿真覆盖的是视觉差、动力学差、任务差中的哪一类。只展示漂亮渲染，不足以证明真实机器人收益。

第二，真实数据如何进入闭环。是只用少量 real validation，还是有部署日志、失败标注、人工接管和回放验证？

第三，数据源贡献是否可分解。Web data、robot data、sim data、failure data、human correction 各自补什么能力？

第四，评测是否有 hard bucket。没有长尾桶，平均成功率很容易掩盖最贵的失败。

## 读完以后怎么判断

Sim2Real 可以按四句话记：

1. 仿真不是照片生成器，而是任务、资产、轨迹、扰动、判定和日志 schema 的数据工厂。
2. Domain randomization 让策略别太脆，system ID 让仿真中心别太假。
3. 数据引擎的核心不是多收日志，而是筛选、标注、门禁和 hard set 验收。
4. 真实失败最有价值，因为它告诉你下一轮随机化、数据采集和评测该补哪里。

继续读相邻内容时，可以接 [VLA 数据与策略学习](../vla/data-and-policy-learning.md)、[Isaac Sim 与 RoboTwin 仿真数据链](isaacsim-and-robotwin.md)、[Rerun 机器人数据可视化](rerun-robotics-data-visualization.md) 和 [世界模型数据引擎](../world-models/data-engines-and-self-improvement.md)。

## 外部材料

- [Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World](https://arxiv.org/abs/1703.06907)：理解“随机到足够宽，让真实世界落进训练分布”的经典思路。
- [Learning Dexterous In-Hand Manipulation](https://arxiv.org/abs/1808.00177)：看大规模随机化、自动域随机化和真实灵巧手迁移的工程案例。
- [Open X-Embodiment / RT-X](https://robotics-transformer-x.github.io/)：理解跨机器人数据为什么首先是接口对齐问题。
- [DROID](https://droid-dataset.github.io/)：看真实世界机器人数据如何采集多场景、多任务和长轨迹。
- [RoboCat](https://deepmind.google/discover/blog/robocat-a-self-improving-robotic-agent/)：理解“收集新任务数据，再训练，再扩展能力”的机器人自改进闭环。
- [RoboTwin 2.0 docs](https://robotwin-platform.github.io/doc/)：理解仿真任务、资产、expert trajectory 和 success checker 如何组织成数据链。
