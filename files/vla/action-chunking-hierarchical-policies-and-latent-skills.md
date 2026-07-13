# 动作分块、层级策略与潜在技能

想象一个“打开冰箱、拿酸奶、放到桌上”的机器人任务。若策略以 20Hz 直接输出末端位姿增量，一个 40 秒任务就是 800 个低层动作；中间任意一次抓偏、门没完全打开、酸奶被遮挡，都会改变后续观测。让 VLA 从一句话一路直接回归 800 个微动作，等于把任务理解、阶段切换、接触控制和失败恢复都塞进同一个输出头。

动作分块、层级策略和潜在技能解决的是同一件事：给动作序列加上时间结构。模型不再把每个控制周期都当成全新决策，而是学会在合适的时间尺度上输出短时动作、调用技能、检查是否完成，再回到闭环观察。

## 先分清三种时间尺度

机器人策略里的“动作”不是一个层级。

| 层级 | 典型对象 | 时间尺度 | 主要问题 |
| --- | --- | --- | --- |
| 低层动作 | 关节速度、末端位姿增量、gripper 开合 | 20ms 到 200ms | 控制频率、限幅、坐标系、接触稳定性 |
| 动作 chunk | 未来 \(H\) 步动作序列 | 0.5s 到数秒 | 短时连贯性、抖动、重观察频率 |
| 技能 / option | 抓取、对齐、拉开、放置、导航到目标 | 数秒到数十秒 | 技能边界、终止条件、失败恢复 |
| 任务计划 | 清理桌面、做咖啡、整理冰箱 | 数十秒到数分钟 | 目标分解、记忆、约束、安全 |

逐步动作预测只覆盖第一层。VLA 真正难的是把语言目标落到第二、第三层，再由低层控制器稳定执行。

## 逐步动作为什么会抖

最朴素的策略可以写成：

\[
\pi_\theta(a_t\mid o_{\le t}, l)
\]

这里 \(o_{\le t}\) 是历史观测，\(l\) 是语言任务，\(a_t\) 是当前动作。这个式子看起来干净，但它把所有决策压力都压在单步动作上：模型既要判断“现在应该继续接近把手，还是已经该夹紧”，又要输出几毫米级的位姿变化。

如果训练目标是逐帧 MSE，模型会倾向于预测演示动作的平均值。接触任务里平均值经常不是好动作：同一位置可以从左侧绕，也可以从右侧绕；平均到中间就可能撞到物体。即使动作方向没错，单步预测的微小噪声也会在高频控制里变成抖动。

动作分块的第一层价值就在这里：把“当前一帧该动哪里”改成“接下来一小段该怎样连续移动”。

## 动作 chunk 是短时控制接口

动作分块让模型一次输出未来 \(H\) 步动作：

\[
\hat A_{t:t+H-1}=\pi_\theta(o_{\le t}, l)
\]

\(\hat A_{t:t+H-1}\) 是动作序列，不是任务计划。部署时通常只执行前 \(h\) 步，再重新观察并预测新 chunk：

\[
\text{predict } \hat A_{t:t+H-1},\quad
\text{execute } \hat A_{t:t+h-1},\quad h \le H
\]

这就是 receding horizon。它让策略拥有短时连贯性，但仍然保持闭环。

![ALOHA ACT algorithm](../assets/images/paper-figures/vla/aloha-act-algorithm-2x.png){ width="760" }

<small>图源：[Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware / ACT](https://arxiv.org/abs/2304.13705)，方法图。原图展示 ACT 用 CVAE/Transformer 从多视角图像和关节状态预测 action chunk，并用 temporal ensembling 平滑执行。本站读法：chunk 的核心不是“输出更多动作”，而是把精细操作改成短时序生成，再用重叠预测降低抖动。</small>

ACT 里的 temporal ensembling 可以理解为：多个历史时刻都预测过当前时刻的动作，执行时把这些预测融合起来：

\[
\tilde a_t=
\frac{\sum_i w_i \hat a_t^{(t-i)}}{\sum_i w_i}
\]

其中 \(\hat a_t^{(t-i)}\) 表示从较早观测生成的 chunk 里覆盖到当前时刻的动作，\(w_i\) 是权重。直观上，它不是盲目相信最新一帧，而是让连续几次预测共同决定当前动作，从而减少跳变。

## Chunk size 是控制参数

\(H\) 不是越大越好。它决定策略“闭眼”往前看的距离，也决定错误多久才会被新观测纠正。

| \(H\) 的选择 | 适合场景 | 主要风险 |
| --- | --- | --- |
| 很短 | 接触瞬间、窄缝插入、需要高频纠偏 | 和逐步动作差别小，容易抖 |
| 中等 | 抓取、对齐、放置、短距离推动 | 需要调好重观察频率和 ensemble |
| 很长 | 导航接近、粗略移动、稳定重复动作 | 环境变化后还按旧 chunk 执行，漂移明显 |

所以 chunk size 应和任务阶段绑定：接近阶段可以长一些，接触阶段要短一些；视觉延迟大时要短一些，控制器稳定且环境变化慢时可以长一些。把 \(H\) 写死并不是原则错误，但上线前必须按失败桶验收。

## 层级策略把技能做成可终止子策略

层级策略处理的是更长时间尺度。经典 options 框架把一个技能写成：

\[
o=(\mathcal I_o,\pi_o,\beta_o)
\]

这里 \(\mathcal I_o\) 是技能可启动的状态集合，\(\pi_o(a\mid s)\) 是技能内部策略，\(\beta_o(s)\) 是终止概率。高层策略选择 option：

\[
o_t\sim \mu(o\mid s_t,l)
\]

低层 option 再执行动作，直到终止条件触发。这个写法很重要，因为技能不是一个名字，而是一段带启动条件、执行策略和结束判断的闭环行为。

例如“拉开抽屉”这个技能至少要知道：什么时候可以开始拉，拉到什么程度算完成，拉不动时如何退回，抽屉半开时是否要重新抓把手。没有终止条件和失败处理，所谓技能只是更大的开环动作块。

![SayCan language skill affordance](../assets/images/paper-figures/embodied-ai/saycan-figure-3-llm-affordance.png){ width="900" }

<small>图源：[Do As I Can, Not As I Say / SayCan](https://say-can.github.io/)，方法图。原图展示语言模型给候选技能打任务相关分，价值函数给技能打可执行性分，二者结合选择下一步技能。本站读法：高层技能选择不能只看语言合理性，还要被当前场景 affordance 约束。</small>

SayCan 不是现代 VLA 的唯一形态，但它把层级策略的核心讲得很清楚：语言模型可以提出“找苹果、拿苹果、放苹果”等候选技能，机器人系统还要判断这些技能在当前状态是否可执行。VLA 里如果跳过 affordance 和终止条件，语言规划会显得聪明，真实执行却容易在第一步就卡住。

## 潜在技能不是自动可解释

很多数据集没有人工分好的技能标签。潜在技能的想法是从轨迹中学一个压缩变量 \(z\)，再用 \(z\) 生成一段动作：

\[
z_t \sim p_\theta(z\mid o_{\le t},l), \qquad
A_{t:t+H-1}\sim p_\phi(A\mid o_{\le t},z_t)
\]

\(z\) 可以是连续 latent、离散 code、VQ token，或高层子任务表示。Play-LMP、Behavior Transformer / VQ-BeT、FAST action tokenizer、Octo 和 π0.5 这些工作虽然目标不同，但都在处理同一个张力：低层动作太长、太细、太多峰，需要一个更适合 Transformer 或策略搜索的中间表示。

潜在技能的风险是“看起来压缩了，实际不可用”。如果 \(z\) 太大，它只是把原始动作换了个名字；如果 \(z\) 太小，它会丢掉接触时机、夹爪开合和关键朝向；如果 \(z\) 没有终止条件，高层很难知道何时切换技能；如果 \(z\) 与语言不对齐，用户说“先打开抽屉”时，系统可能找不到对应的稳定行为单元。

所以潜在技能要同时回答四个问题：能不能重构动作，能不能跨任务复用，能不能被语言或子目标选择，能不能在失败时安全退出。

## 语言对齐要落到技能边界

对 VLA 来说，语言不是直接控制每个关节的遥控器。更自然的结构是：

\[
l \rightarrow \text{subtask / skill} \rightarrow \text{action chunk} \rightarrow \text{controller}
\]

用户说“把酸奶放到桌上”，高层需要拆成“打开冰箱、定位酸奶、抓取、关门、移动到桌边、放下”。每一段都需要自己的观测证据和完成条件。π0.5 这类 open-world VLA 工作强调 high-level subtask、heterogeneous co-training 和 low-level action expert，就是因为真实家庭任务很难只靠一条平铺动作序列完成。

语言对齐的好处不是让技能名字更漂亮，而是让错误更容易定位。失败时可以问：是任务分解错了、技能选择错了、chunk 执行错了、还是低层控制器不稳？如果所有东西都混在一个动作头里，调试只能看成功率，几乎不知道该补数据、改模型还是换控制接口。

## 与世界模型的接口

世界模型适合在比原始动作更粗的空间里预测后果。若每个 rollout step 都是 20Hz 末端动作，长任务规划成本很高，模型误差也容易累积。若 rollout step 是“打开抽屉”“拿起杯子”这类技能，预测更接近任务结构。

但技能级世界模型也有边界。它可能知道“拉开抽屉后可以看到里面物体”，却不知道夹爪会不会打滑、把手是否反光、抽屉轨道是否卡住。一个可靠系统通常需要两层预测：高层预测技能后果，低层用真实观测和短 chunk 修正接触细节。

## 按失败模式选择结构

| 失败模式 | 优先看什么 | 为什么 |
| --- | --- | --- |
| 高频抖动 | action chunk、temporal ensembling、控制频率 | 让短时动作更连贯，减少单帧噪声 |
| 长任务忘阶段 | 层级策略、子任务状态、历史记忆 | 把几百步动作拆成可跟踪阶段 |
| 多条合理轨迹被平均 | diffusion / mixture / VQ skill | 保留多峰动作模式 |
| 执行到一半偏离 | receding horizon、失败检测、恢复轨迹 | 及时重新观察和纠偏 |
| 语言计划不可执行 | affordance、value function、技能启动条件 | 让高层候选被当前场景约束 |
| 跨机器人迁移差 | action tokenizer、坐标系统一、技能接口 | 避免低层动作语义混乱 |

这个表比“选 ACT 还是选 diffusion”更重要。方法名只是工具，真正的选择依据是失败在哪个时间尺度发生。

## 评测要看切换、恢复和抖动

动作分块和技能抽象不能只用总成功率验收。至少要分四类证据：

| 证据 | 看什么 |
| --- | --- |
| Chunk 内部 | 动作平滑度、末端速度突变、gripper 时机、接触成功率 |
| Chunk 之间 | 重叠预测是否一致、replan 后是否跳变、延迟是否可控 |
| 技能层 | 技能启动是否合理、终止是否过早或过晚、切换是否卡住 |
| 失败恢复 | 抓偏、遮挡、碰撞边缘、任务中断后能否重新进入正确阶段 |

实机复盘时，不要只写“成功 / 失败”。更有用的是标出失败发生在哪一层：语言目标没拆对、技能选错、chunk 漂移、低层控制不稳、感知缺证据，还是安全层阻止执行。这样下一轮才知道该补哪种数据。

动作分块、层级策略与潜在技能可以按一句话收束：它们不是给 VLA 加概念装饰，而是在动作序列里放入可调试的时间边界。Chunk 解决短时连贯性，option/skill 解决阶段和终止，latent skill 解决数据中重复行为的压缩与复用；三者都必须回到闭环观察、失败恢复和真实机器人评测里验收。

继续读相邻内容时，可以接 [VLA 动作表示与控制接口](action-representation-and-interfaces.md)、[VLA 数据与策略学习](data-and-policy-learning.md)、[WM / WAM / VAM：动作到底怎样进入世界模型](../world-models/wm-wam-vam-and-action-conditioned-modeling.md) 和 [世界模型评测与失效模式](../world-models/evaluation-and-failure-modes.md)。

## 外部精读

- [ALOHA / ACT](https://arxiv.org/abs/2304.13705)：理解 action chunk、CVAE 和 temporal ensembling 如何服务精细操作。
- [Options 框架](http://incompleteideas.net/papers/SPS-aij.pdf)：理解技能为什么需要启动集合、内部策略和终止函数。
- [SayCan](https://say-can.github.io/)：理解语言技能选择为什么必须和 affordance / value function 结合。
- [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/)：理解动作序列作为生成分布时怎样处理多峰连续控制。
- [Octo](https://octo-models.github.io/)：看通用机器人策略如何把多数据集、多任务条件、action chunk 和 diffusion decoder 放在同一系统里。
- [FAST](https://arxiv.org/abs/2501.09747)：理解高频连续动作怎样被压缩成更短的 action token 序列。
- [Datawhale Every-Embodied](https://github.com/datawhalechina/every-embodied)：学习具身智能中文材料如何把硬件、仿真、数据、模型和部署路线组织成学习路径；事实 claim 仍回到论文和项目页。
