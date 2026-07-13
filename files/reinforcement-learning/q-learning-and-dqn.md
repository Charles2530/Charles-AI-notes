# Q-learning 与 DQN

## 本章回答什么

怎样从 TD prediction 走到离散动作控制，又怎样把 Q-table 换成神经网络而不让 bootstrap 目标不断追逐自己？本章先划清 SARSA 与 Q-learning 的 on-policy / off-policy 边界，再解释 DQN 的函数逼近、experience replay 与 frozen target network。

这里的可执行实验只证明 CartPole 上发生了一次有限的真实优化更新。它不是收敛基准，也不证明这组超参数能解决 CartPole。

## 先修知识

先读 [动态规划、Monte Carlo 与 TD](dynamic-programming-monte-carlo-td.md)，尤其是 TD error、terminal mask 与 truncation。最小 GridWorld 的 tabular Q-learning 实现在 [`gridworld.py`](../assets/examples/reinforcement-learning/gridworld.py)；深度版本的环境安装与命令见 [示例 README](../assets/examples/reinforcement-learning/README.md)。

## 最小任务

先运行表格版本，确认 terminal-masked Q-learning update；再运行深度版本。CartPole-v1 的 observation 是 4 维连续向量，action 只有向左、向右两个离散选项。DQN 用小型 MLP 输出两个 \(Q(s,a)\)，再用 epsilon-greedy 选择动作。

隔离环境安装完成后，从仓库根目录运行 [`gridworld.py`](../assets/examples/reinforcement-learning/gridworld.py) 与 [`deep_rl_smoke.py`](../assets/examples/reinforcement-learning/deep_rl_smoke.py)：

```bash
python files/assets/examples/reinforcement-learning/gridworld.py --seed 7
python files/assets/examples/reinforcement-learning/deep_rl_smoke.py dqn --seed 7 --steps 256
```

输出包含 interaction steps、optimizer updates、末次 loss、episode return 均值和参数变化量。有限且非零的更新说明 replay sample、target、loss 与反向传播接通；短回报曲线不能证明学习稳定或已经收敛。

## 核心机制

### SARSA 与 Q-learning 的数据策略边界

SARSA 的一步 target 是 \(r_t+\gamma Q(s_{t+1},a_{t+1})\)，其中 \(a_{t+1}\) 来自当前实际行为策略，所以它是 on-policy。Q-learning 改用下一状态的 greedy action：

\[
Q(s_t,a_t)\leftarrow Q(s_t,a_t)+\alpha
\left[r_t+\gamma(1-d_t)\max_{a'}Q(s_{t+1},a')-Q(s_t,a_t)\right]
\]

这里 \(\alpha\) 是步长，\(\gamma\) 是 discount，且 \(d_t=1\) 只表示 `terminated`。行为仍可用 epsilon-greedy 探索，但 target 假设下一步 greedy，因此 Q-learning 是 off-policy。区别不在“有没有探索”，而在 target 对哪个策略做 backup。

### 从 Q-table 到函数逼近

状态连续时不能为每个 \((s,a)\) 建表。DQN 用 \(Q_\theta(s,a)\) 共享参数，并最小化预测值与 TD target 的误差。神经网络能在相似状态间泛化，也会让一个样本的更新同时改变许多状态估计；bootstrap target、采样分布和参数因而一起移动，训练比表格法更容易不稳定。

### 当前 Q 与 frozen target network

DQN 的 target 写成：

\[
y_t=r_t+\gamma(1-d_t)\max_{a'}Q_{\bar\theta}(s_{t+1},a')
\]

当前网络 \(Q_\theta\) 拟合 \(y_t\)，frozen target network \(Q_{\bar\theta}\) 在若干更新间保持不变，再周期性复制当前参数。`terminated` 令 bootstrap 为零；`truncated` 若仍有有效 successor observation，则保留 bootstrap。把二者合成 `done` 会系统性低估时间上限附近的价值。

### Replay、探索与损失

replay buffer 保存 `(state, action, reward, next_state, terminated)`，随机 minibatch 打散相邻样本并让旧数据被重复利用。这使 DQN 具备明显的 off-policy 数据复用能力，但旧数据过多也会和当前访问分布脱节。

smoke 实现的探索日程是 `epsilon=max(0.1, 1-step/steps)`：在 `step=0` 时从 `epsilon=1.0` 开始，分母 `steps` 就是命令行请求的 `--steps` interaction horizon；之后每步线性减小 \(1/\text{steps}\)，直到被 floor 0.1 截住。以 `--steps 256` 为例，第一个使用 floor 的索引是 `step=231`，因为 `step` 取 `0` 到 `255`，而不是额外训练 256 步后才开始衰减。

epsilon-greedy 在训练时给随机动作概率 \(\epsilon\)；评估时应单独关闭或固定很小的探索，否则评测把策略质量和探索噪声混在一起。Huber loss 在 TD error 小时近似平方误差、大时近似绝对误差，比纯 MSE 更不容易被少量巨大 target 支配。

[2013 年早期 DQN](https://arxiv.org/abs/1312.5602)展示了深度网络结合 experience replay 的 Atari 结果；[2015 Nature DQN](https://doi.org/10.1038/nature14236)给出包含 frozen target network 的完整方法与更广泛结果。引用两者时不能把后者的完整稳定机制倒写成前者已全部使用。

## 训练数据流

```text
epsilon-greedy policy -> CartPole transition -> replay buffer
replay minibatch -> frozen target Q -> terminal-masked target
current Q(s,a) + target -> Huber loss -> optimizer update
periodic copy: current Q -> target Q
```

采样和优化不是同一步：environment step 先产生数据，optimizer 再从 replay 中抽旧数据更新。报告实验时至少分别记录 environment steps、gradient updates、buffer size、training epsilon 和 evaluation return。

## 常见失败

| 失败 | 为什么误导 | 检查方式 |
| --- | --- | --- |
| 当前网络同时生成并追逐 target | target 每步随参数移动 | 保留独立 target network，并记录同步频率 |
| 将 truncation 当 terminal | 丢失有效 bootstrap | 只用 `terminated` 构造 \(d_t\) |
| 训练和评估共用 epsilon | return 下降可能只是随机动作 | 用独立评估 episode 与明确 exploration 设置 |
| 只看 max Q 上升 | 过估计也会抬高 value | 同时看评估 return、TD error、Q 尺度与多 seed |
| replay 未热身就大批更新 | 高度相关的少量样本被反复拟合 | 达到最小 buffer 后再采 minibatch |
| 把单次 reward 峰值当学会 | 偶然长 episode 不是策略改善 | 看滑动统计、独立评估和种子分布 |

## 自检与练习

1. 对 `reward=1, next_q=10, gamma=0.9`，分别算 `terminated=True/False` 的 target。
2. 写出同一 epsilon-greedy trajectory 下 SARSA 和 Q-learning 对下一状态使用的 action，并说明哪个 target 与行为策略一致。
3. 将 target 同步频率从 32 次更新改为每次更新，观察 loss 与 Q 尺度；不要据单个 seed 下结论。
4. 解释为什么 evaluation return 和 replay batch 中的 reward 均值不是同一个指标。

## 下一步

Q-learning 通过 \(\max_a Q\) 间接得到策略；下一章改为直接优化动作概率，见 [Policy Gradient 与 REINFORCE](policy-gradient-and-reinforce.md)。课程位置与尚未创建的主题只在 [完整学习路线](index.md#完整学习路线) 中列出。
