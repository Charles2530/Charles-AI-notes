# 规划、反馈控制与安全约束

一个目标位姿只说明末端应该在哪里，没有规定怎样绕过障碍、沿路径多快移动、接触后用多大力。路径、轨迹与控制输入是不同对象。路径规定几何顺序，轨迹加上时间，控制器根据实时误差产生执行命令。

![SayCan LLM and affordance scores 原论文图](../assets/images/paper-figures/embodied-ai/saycan-figure-3-llm-affordance.png){ width="920" }

<small>图源：[Do As I Can, Not As I Say](https://research.google/pubs/do-as-i-can-not-as-i-say-grounding-language-in-robotic-affordances/)，Figure 3。原图展示语言模型分数和 affordance 分数如何组合。读图重点：高层语义判断“该做什么”，低层 affordance / value 判断“现在能不能做”。</small>

## 用一个可手算的 MPC 看清优化与反馈

考虑教学系统 \(x_{k+1}=x_k+u_k\)，当前位置 \(x_0=0\)，目标为 1，预测两步。代价为

\[
J=(x_2-1)^2+\rho(u_0^2+u_1^2),\qquad \rho=1.
\]

因为 \(x_2=u_0+u_1\)，对两个动作求导得到 \(2(u_0+u_1-1)+2u_i=0\)，对称解为 \(u_0=u_1=1/3\)。若增加动作约束 \(|u_i|\le0.2\)，最优解变为两步都取 0.2，终点只有 0.4。有限窗口与动作限制下，不一定能立刻到达目标。

MPC 执行第一个动作后重测状态。若实际只到 0.15，而不是计划的 0.2，下一轮应以 0.15 重新求解。直接执行原来整段计划属于开环回放，不具备这项误差修正。

一般问题是

\[
\min_{u_{0:H-1}}\sum_{k=0}^{H-1}\ell(x_k,u_k)+\ell_f(x_H)
\quad\text{s.t. }x_{k+1}=f(x_k,u_k),\ x_k\in\mathcal X,\ u_k\in\mathcal U.
\]

动力学模型、代价和约束共同决定结果。世界模型可以提供近似 \(f\)，但模型误差会改变可行性；预测到达不等于真实到达。

## 轨迹给出参考，LQR把偏差拉回来

MPC 每一轮都会重新求解，但很多系统仍会在两次求解之间沿一条局部轨迹运行。只把规划器输出的 \(u_0,u_1,\ldots\) 依次发送出去，仍然是开环执行。传感器读数和名义状态有一点偏差，后面的动作就不再对应真实位置。

在名义轨迹 \(x_k^0,u_k^0\) 附近，把状态和动作写成

$$
\delta x_k=x_k-x_k^0,\qquad
\delta u_k=u_k-u_k^0.
$$

对非线性动力学 \(x_{k+1}=f(x_k,u_k)\) 做一阶展开，可以得到

$$
\delta x_{k+1}\approx A_k\delta x_k+B_k\delta u_k,
\qquad
A_k=\left.\frac{\partial f}{\partial x}\right|_{(x_k^0,u_k^0)},
\quad
B_k=\left.\frac{\partial f}{\partial u}\right|_{(x_k^0,u_k^0)}.
$$

于是控制器不再追一个固定的终点，而是根据当前偏差修正名义动作

$$
u_k=u_k^0-K_k(x_k-x_k^0).
$$

若轨迹是定常点，\(K_k\) 可以由 LQR 的 Riccati 方程得到；若轨迹在运动，线性化矩阵和增益随时间变化，得到的是 time-varying LQR。MIT Underactuated 的 [LQR 讲义](https://underactuated.mit.edu/lqr.html)把这条关系从线性系统推到非线性系统的局部轨迹稳定，并给出 Drake 的 `FiniteHorizonLinearQuadraticRegulator` 作为实现入口。

### 一个二阶系统的数量级

连续时间双积分器取

$$
A=\begin{bmatrix}0&1\\0&0\end{bmatrix},
\qquad B=\begin{bmatrix}0\\1\end{bmatrix},
\qquad Q=I,
\qquad R=1.
$$

无限时域 LQR 给出 \(K=[1,\sqrt3]\)。如果当前位置误差为 \(0.2\)，速度误差为 \(-0.1\)，反馈修正为

$$
\delta u=-K\begin{bmatrix}0.2\\-0.1\end{bmatrix}
\approx-0.0268.
$$

这里的数值只说明反馈项怎样由状态误差产生，不能直接当成某台机械臂的控制增益。真实系统还要考虑采样时间、执行器限幅、状态估计延迟和线性化所在的工作区间。离开这个局部区域后，LQR 可能仍然给出一个数值，却没有原来的稳定性依据。

## 规划器不只输出一条路，还要说明这条路能承受多大误差

一个名义轨迹可以在无噪声仿真中穿过障碍之间的窄缝，实际执行时却因为初始位置、摩擦或感知误差偏离。MIT 的 [Feedback Motion Planning](https://underactuated.mit.edu/feedback_motion_planning.html) 讨论了一个更适合组合的接口：把局部反馈控制器和它能吸收的状态集合一起当成一个 skill。

对第 \(i\) 个 skill，设 \(V_i(x)\) 是误差的 Lyapunov 函数，\(V_i(x)\le\rho_i\) 是允许进入的区域，控制器运行一段时间后把状态带进更小的 \(V_i(x)\le\bar\rho_i\)。高层规划器只有在

$$
V_i(x)\le\bar\rho_i
\quad\Longrightarrow\quad
V_j(x)\le\rho_j
$$

成立时，才应把 skill \(i\) 接到 skill \(j\)。这比把“抓取前的最后一个位姿”当作唯一前置条件更接近真实系统，因为下一个控制器接手时，状态通常落在一个区域里，而不是落在一个精确点上。

工程上可以把一个 skill 记录成四部分：启动条件、名义目标、反馈律和退出条件。启动条件写成可检查的状态范围；退出条件说明何时进入下一个 skill，何时因碰撞风险、跟踪误差或超时进入恢复分支。这样高层任务规划器不必知道每个关节的控制细节，却能判断当前状态是否适合切换。

这里的“可承受误差”不是一句口号。若只在仿真中画出一条轨迹，没有计算或测量它的吸引域，就不能声称轨迹对扰动鲁棒。funnel、可达集合或多次扰动回放都可以提供证据，但它们回答的范围不同，不能互相替代。

## 不确定性进入约束后，先决定你要哪一种保证

设安全条件写成 \(g(x)\le0\)。有三种常被混在一起的说法。

| 约束方式 | 写法 | 需要知道什么 | 典型代价 |
| --- | --- | --- | --- |
| 确定性约束 | 对当前估计状态要求 \(g(x)\le0\) | 当前状态和模型 | 没有显式处理估计误差 |
| 机会约束 | \(P(g(x)>0)\le\alpha\) | 误差分布或可用上界 | 分布错了，概率保证会失效 |
| 最坏情况约束 | 对所有 \(w\in\mathcal W\) 要求 \(g(x,w)\le0\) | 扰动集合 \(\mathcal W\) | 集合过大时会很保守 |

MIT 的 [Robust Control 讲义](https://underactuated.mit.edu/robust.html)把机会约束和最坏情况约束分开讨论。机会约束适合“碰撞概率低于某个阈值”这类问题，但需要可信的分布假设；最坏情况方法只需要描述扰动集合，却可能为了覆盖极少见的情况而牺牲可行空间。

一个简单的高斯近似能说明这项取舍。若障碍距离估计为 \(d\sim\mathcal N(\mu,\sigma^2)\)，希望 \(P(d\ge0)\ge0.997\)，在单侧正态近似下可要求 \(\mu\ge3\sigma\)。当 \(\sigma=0.02\) 米时，平均距离至少留出约 \(0.06\) 米；若传感器换了、误差不再近似高斯，这个数字就不能继续当安全余量。

### Tube MPC 保护的是一束状态，而不是一条中心线

线性系统带有有界扰动时，可以写成

$$
x_{k+1}=Ax_k+Bu_k+w_k,
\qquad w_k\in\mathcal W.
$$

若用集合 \(\mathcal X_k\) 表示第 \(k\) 步可能出现的状态，最基本的传播关系是

$$
\mathcal X_{k+1}=A\mathcal X_k+Bu_k\oplus\mathcal W.
$$

Tube MPC 会用反馈把误差限制在一条随名义轨迹移动的 tube 里，再把障碍和终点约束收紧给中心轨迹。它的工程意义是把“实际状态可能偏离多少”提前放进优化问题，代价是集合传播和约束收紧会让可行域变小。MIT 讲义同时提醒，集合的表示规模可能随时间增长，因此长预测窗不一定免费。

如果任务更关心平均性能，机会约束可能更合适；如果安全事故代价极高，宁愿保守一些，最坏情况或 reachability 分析更容易解释。选择哪一种保证，应该写进实验协议，而不是在结果出来以后用“鲁棒”一词概括。

## 用 Safe Control Gym 做一轮可复现实验

[Safe Control Gym](https://github.com/utiasDSL/safe-control-gym) 的官方仓库提供 CartPole 和 quadrotor 环境，包含符号动力学、安全约束以及输入、参数和动力学扰动。README 列出了 PID、LQR、iLQR、线性 MPC、GP-MPC、CBF 和 MPSC 等控制器或安全过滤器，适合用来比较“跟踪得好”和“违反约束少”是否同时成立。

官方 README 给出的 CartPole LQR 入口是

```bash
git clone https://github.com/utiasDSL/safe-control-gym.git
cd safe-control-gym
python -m pip install -e .
cd examples
python3 lqr/lqr_experiment.py \
  --algo lqr \
  --task cartpole \
  --overrides \
    ./lqr/config_overrides/cartpole/cartpole_stab.yaml \
    ./lqr/config_overrides/cartpole/lqr_cartpole_stab.yaml
```

这段命令来自仓库 README，本站没有在本机声称执行成功。真正做比较时，至少固定任务、仿真步长、控制频率、扰动种子和终止条件，再记录以下几列。

| 记录项 | 作用 | 常见误读 |
| --- | --- | --- |
| 任务成功率 | 是否达到目标并满足终止条件 | 不能单独说明过程是否安全 |
| 最大约束违例 | 是否越过位置、输入或碰撞边界 | 只记录平均值会掩盖极端回合 |
| 累积代价 | 路径长度、控制能量或任务损失 | 不同代价定义不能直接横比 |
| 恢复次数 | 进入保护、重规划或急停的次数 | 恢复成功不代表没有风险 |
| 推理与控制周期 | 能否在规定时间内给出下一步 | 离线轨迹质量不能替代实时性 |

可以先用 LQR 作为低方差基线，再加入参数扰动和输入限幅；随后把 MPC 或 CBF 作为独立变量加入。每次只改变一个条件，才能知道收益来自控制器、约束处理还是更宽松的终止规则。若要比较学习控制器，还应把相同的安全过滤器、观测权限和训练预算写出来。

## 反馈增益为什么不能随便增大

对一阶离散系统 \(x_{k+1}=x_k+\Delta t\,u_k\)，固定目标 \(r\)，使用比例反馈 \(u_k=-K(x_k-r)\)。误差满足

\[
e_{k+1}=(1-K\Delta t)e_k.
\]

收敛需要 \(|1-K\Delta t|<1\)，即 \(0<K\Delta t<2\)。取 \(\Delta t=0.1,K=5\)，每步误差缩为一半；取 K=25，系数为 −1.5，误差会交替放大。高增益并不自动更精确，延迟、噪声、饱和和真实高阶动力学还会进一步限制选择。

接触控制通常也不能只追求位置误差归零。阻抗控制用期望刚度与阻尼规定误差和力的关系，简化为 \(F=K_p(x_{\mathrm{ref}}-x)+D(v_{\mathrm{ref}}-v)\)。大刚度会使小位置误差对应大接触力；是否可用取决于硬件、接触对象和控制稳定性。

## 安全集合怎样约束下一步

设一维允许位置为 \(x\ge0\)，当前 \(x=0.1\)，时间步长 0.1 秒。若动作是速度，为保持下一步非负，需要 \(x+\Delta t u\ge0\)，即 \(u\ge-1\)。模型提出 −2 时，仅从这条离散约束出发应将其改为不小于 −1。

连续控制障碍函数用 \(\mathcal C=\{x:h(x)\ge0\}\) 描述安全集合，在控制仿射动力学 \(\dot x=f(x)+g(x)u\) 下，常用约束为

\[
\nabla h(x)^\top(f(x)+g(x)u)+\alpha(h(x))\ge0.
\]

它约束安全裕量的变化速度。在模型与可行性等假设成立时，可用于论证不离开安全集合。采样实现、状态误差、未知障碍和制动限制都会影响实际保证，不能只写出 \(h(x)\ge0\) 就宣称系统安全。

安全机制改变动作后，预测链也要使用改变后的量。设世界模型按原请求速度 −2 预测下一位置 −.1，安全投影最终执行 −1，理想下一位置为 0。若日志仍把 −2 当作 executed action 来训练转移模型，它就会从同一动作学到错误位移，下一轮规划也会重复提出被拒绝的候选。

应分别保存策略请求、限幅后命令和实测响应。预测规划时可显式模拟已知安全处理，或将实际执行量反馈给状态估计。若保护机制不满足规划假设，就需要重新求解或进入可解释的退化行为。闭环修正依赖这条实际发生的动作链，而非只依赖最早的策略输出。

## 接口顺序与否决

策略给出候选，adapter 解释单位和坐标，planner 或控制器形成可执行参考，安全机制检查并保留否决通道。具体系统可能把约束嵌入规划或控制，而非只有一个末端过滤模块。任何需要停止的事件都应能取消旧队列。

规划器找到无碰撞离散节点也未必保证节点之间无碰撞，需要连续检查或足够可靠的分段验证。跟踪器出现饱和后，实际轨迹又可能偏离规划轨迹，因此要记录候选、最终命令与实测状态。

## 用一个小回放检查接口顺序

固定一段带障碍的短轨迹，保存策略候选、坐标转换后的目标、planner 轨迹、限幅后的命令和实际状态。先验证没有安全投影时的参考实现，再插入速度边界、碰撞约束和一次人工停止。每次只改变一个环节，比较下一状态和终止原因，才能知道是坐标、约束、跟踪还是取消协议导致差异。

这类回放不能证明真实机器人安全，却能抓住最容易被演示掩盖的接口错误：planner 以为执行了 −2，控制器实际只执行 −1；安全层拒绝了命令，但状态机仍把任务记成进行中；旧队列在人工停止后又被重新取出。把三种动作和状态切换分开记录，后续才有可解释的修复入口。

继续看[运动学](robot-kinematics-and-dh-parameters.md)、[部署时间与命令](../vla/deployment-and-safety.md)、[安全案例](deployment-patterns-and-safety-cases.md)和[状态、动作与机器人数据集](state-action-robot-datasets.md)。理论依据见 [Control Barrier Functions](https://arxiv.org/abs/1903.11199)，控制与运动生成工具可查 [cuRobo](https://curobo.org/)。

**来源与改编说明**：MPC、反馈增益、LQR 数值和安全集合的教学算例为本站构造；LQR、反馈运动规划和 Tube MPC 的概念参考 MIT Underactuated 的 [LQR](https://underactuated.mit.edu/lqr.html)、[Feedback Motion Planning](https://underactuated.mit.edu/feedback_motion_planning.html) 与 [Robust Control](https://underactuated.mit.edu/robust.html) 讲义；Safe Control Gym 的环境、控制器和命令参考其 [官方 README](https://github.com/utiasDSL/safe-control-gym)。这些材料用于组织学习路径，不代表本站复现了其 benchmark。页面中的命令尚未在本站环境执行，也不能替代具体硬件、载荷、传感器和适用标准下的安全验证。
