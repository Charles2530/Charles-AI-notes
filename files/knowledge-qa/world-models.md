# 世界模型问答

“世界模型”在不同工作中范围很宽，可以指预测观测、潜状态或环境演化的模型。若要把它用于规划和控制，就需要进一步证明它能区分动作后果，并在所需时间尺度上支持决策。

## 下一帧预测很准，为什么规划仍然失败

背景像素可能占大多数，即使遗漏接触点或速度，平均图像误差仍很小。规划需要的却可能是“这个动作是否碰到障碍”这一个差别。

固定当前观测，改变候选动作，检查预测是否在与任务相关的位置发生合理变化。再比较模型选中的动作在真实环境或指定仿真中的结果。视频自然度只评价其中一部分，不能代替规划效用。见[世界模型评测](../world-models/evaluation-and-failure-modes.md)。

## 状态、观测和belief为什么不能互换

状态是足以描述演化的信息，观测是传感器给出的数据，belief是给定历史后对隐藏状态的分布。单帧中同位置的物体可能速度不同，所以图像不一定是充分状态。

把相邻观测压进latent可以帮助估计速度或遮挡，但不能自动保证马尔可夫性。检查后续预测是否仍需要遗漏的历史，才能评价状态表示。见[随机过程与马尔可夫性](../prerequisite-math/stochastic-processes-and-markov.md)。

## RSSM为什么区分posterior和prior

训练时当前真实观测可用，posterior可以据此修正隐藏状态；想象未来时真实未来观测尚不存在，只能从先前状态与动作产生prior。若训练loss一直依赖真实观测纠正，长rollout时的误差就可能比单步重建大。

因此要分别测观测条件下的重建和不看未来观测的多步预测。KL等项用于联系两条路径，但系数和梯度方向会改变训练行为。见[RSSM与Dreamer](../world-models/rssm-dreamer-and-planning.md)。

## JEPA不重建像素，就一定保留更有用的状态吗

潜空间目标避免逐像素拟合某些细节，但目标编码器保留什么由其训练与结构决定。接触、方向或细小障碍也可能被压掉。表示更抽象不自动等于更适合控制。

可以固定编码器，在明确下游设置中比较预测、检索或动作效果；冻结线性探测与端到端微调的结论不同。还要检查停止梯度、EMA和防坍缩组件的完整组合。见[Masked与JEPA](../world-models/masked-jepa-and-latent-prediction.md)。

## 模型不确定性高，直接多采样就够了吗

多采样能表示同一模型给出的随机未来，不能自动覆盖模型自身的系统性错误。Aleatoric uncertainty指过程或观测固有随机性，epistemic uncertainty涉及知识不足与模型不确定性，两者需要不同证据。

模型ensemble分歧也只是诊断，多个模型可能共享偏差。规划时应同时看校准、分布外状态、长时误差和风险代价；不能把采样方差小当成安全证明。见[不确定性与风险规划](../world-models/uncertainty-and-risk-aware-planning.md)。

## MPC只执行第一步，会不会浪费后面的规划

后续步骤用于评价当前动作的长期后果。执行一步后重新观察，能修正预测偏差；如果一次执行整段，就失去了这次反馈机会。

例如从位置2计划(-1,-1)，预测到0；第一步实际到1.2，下一轮从1.2重新计算。频繁重规划要付推理与搜索成本，控制延迟也必须计入。见[动态系统与控制](../prerequisite-math/dynamic-systems-and-control.md)。

## 一个smoke demo能证明什么

能证明数据格式、前向、损失或报告路径按预期连接，前提是确实运行并检查输出。随机初始化预测、固定fixture和手工评分，不能证明模型训练有效、动作排序可信或闭环成功。

报告应写清哪些数来自解析构造、哪些来自代码执行、哪些来自训练后独立评测。将同一fixture重复通过十次，也不等于十个独立控制实验。见[世界模型入门](../world-models/beginner-world-models-primer.md)。

## 一个动作序列怎样区分三种评测

设模型从状态 `s0` 开始，动作序列为 `[left, left, stop]`。同一条序列至少可以生成三种不同的测试：

| 测试 | 第一步输入 | 后续输入 | 能回答什么 |
| --- | --- | --- | --- |
| teacher-forced | 真实 `s0` | 每步都换成真实后继 | 局部转移是否拟合训练分布 |
| open-loop | 真实 `s0` | 使用模型上一步预测 | 误差怎样随 horizon 累积 |
| closed-loop | 真实 `s0` | 执行动作后读取新观察 | 预测是否改善真实决策 |

最小的记录接口可以写成：

```python
def rollout(model, initial_state, actions, true_states=None, refresh=False):
    state = initial_state
    rows = []
    for t, action in enumerate(actions):
        prediction = model.predict_next(state, action)
        row = {"t": t, "predicted": prediction}
        if true_states is not None:
            row["true"] = true_states[t + 1]
        rows.append(row)
        if refresh:
            state = true_states[t + 1]
        else:
            state = prediction
    return rows
```

这里 `refresh=True` 只是表示评测接口每一步读取了真实后继，不能把它的误差曲线当成开放循环结果。真实机器人或仿真环境还要记录动作实际执行值、reset 原因和事件标签；只有位置误差下降，却没有测碰撞或任务成功，仍不足以证明规划变好了。

## 快速自测

给一个候选动作写出当前观测、belief、预测状态和真实验证结果。再说明视频自然度、一步误差和决策收益分别由什么数据支持。完整推导可回到[世界模型入门](../world-models/beginner-world-models-primer.md)和[世界模型评测](../world-models/evaluation-and-failure-modes.md)；本页用于定位问题，不能替代多步 rollout 或闭环实验。

**来源与改编说明**：潜变量规划对照 [PlaNet](https://arxiv.org/abs/1811.04551)，想象轨迹对照 [DreamerV3](https://arxiv.org/abs/2301.04104)，表示预测可查 [I-JEPA](https://arxiv.org/abs/2301.08243)；MPC 的控制背景可看 [动态规划教材](http://www.athenasc.com/dpbook.html)。问答中的数值是解释机制的构造例子，不代表真实模型 benchmark。
