# 一对偏好回答怎样改变相对概率

对同一道题给出chosen和rejected，模型收到的是两条完整回答之间的比较。若chosen同时更正确、更长、更礼貌，偏好标签并没有自动告诉模型究竟应学习哪种差异。先看目标函数，再看数据怎样控制这些混杂。

![InstructGPT RLHF pipeline 原论文图](../assets/images/paper-figures/training/instructgpt-rlhf-pipeline.png){ width="920" }

<small>图源：[Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)，Figure 2。原论文图意：展示 InstructGPT 的三步训练流程：收集 demonstration data 做 SFT，收集模型输出排序训练 reward model，再用 PPO 按 reward model 优化 policy。</small>

## 序列概率由每个token共同决定

回答y的log概率是条件token log概率之和。长回答包含更多项，因此序列长度会影响比较。某种偏好损失是否做长度归一化，是目标定义的一部分，不能在评测时随意替换。

同一prompt的chosen与rejected应共享问题条件。若两条回答使用不同检索资料或工具结果，标签可能同时比较了信息可见性，而非回答策略本身。

## DPO增加的是相对参考模型的间隔

设当前策略为 $\pi_\theta$、固定参考为 $\pi_{ref}$，chosen为 $y_w$、rejected为 $y_l$，定义

$$
z=\beta\left[\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}-
\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right],
\qquad L=-\log\sigma(z).
$$

$\partial L/\partial z=\sigma(z)-1$，所以梯度下降推动z增大。在z=0时loss为ln2，导数为-0.5。

假设参考chosen/rejected概率为0.4/0.2，当前为0.3/0.1，概率比从2变成3，偏好间隔改善，但chosen绝对概率从0.4下降到0.3。因此不能把DPO说成保证每条chosen绝对概率都上升。

原始推导见[DPO论文](https://arxiv.org/abs/2305.18290)。这个例子只解释目标，没有模拟一次完整网络更新。

![InstructGPT labeler Likert interface 原论文图](../assets/images/paper-figures/training/instructgpt-labeler-likert.png){ width="860" }

<small>图源：[Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)，Appendix Figure 19(a)。原论文图意：标注者先对单个模型输出做 Likert 质量评分，并标注输出是否有帮助、真实、无害、是否遵循指令等元信息。</small>

![InstructGPT labeler ranking interface 原论文图](../assets/images/paper-figures/training/instructgpt-labeler-ranking.png){ width="860" }

<small>图源：[Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)，Appendix Figure 19(b)。原论文图意：标注者在同一个 prompt 下比较多个模型输出，并把它们从最好到最差排序；这种排序数据用于训练 reward model。</small>

![InstructGPT facetted preference result 原论文图](../assets/images/paper-figures/training/instructgpt-preference-facetted.png){ width="860" }

<small>图源：[Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)，Figure 4。原论文图意：按 prompt 来源和 labeler 分组比较不同模型相对 175B SFT baseline 的偏好胜率，用来检查偏好提升是否跨分布稳定。</small>

## 标签噪声可以系统性偏向某种表面特征

若标注者倾向长答案，正确但简短的回答可能反复被拒绝。若候选位置固定，judge还可能产生位置偏差。将候选顺序交换、构造长度匹配对、把正确性与表达分别评分，可以检查这些因素。

一对回答如果两个都错，只能提供“更偏好哪一个”的相对信息，未必教会正确解法。可执行任务尽量保留执行结果和错误原因；对于难以验证的偏好，记录分歧和不确定性，不强行把模糊判断包装成硬标签。

![InstructGPT human preference result 原论文图](../assets/images/paper-figures/training/instructgpt-main-preference.png){ width="760" }

<small>图源：[Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)，Figure 1。原论文图意：在 API prompt 分布上比较不同模型输出相对 175B SFT baseline 的人类偏好胜率，展示 InstructGPT/RLHF 后训练相对 GPT-3 和 SFT baseline 的偏好提升。</small>

## 偏好数据怎样进入训练与评测

训练集用于改变策略，独立评测集检查正确性、偏好胜率、长度、拒答和特定能力。若训练与评测共享judge，模型可能优化judge偏好而不改善目标任务，应辅以其他验证。

采集策略也影响数据覆盖。只比较当前模型容易生成的两种回答，可能遗漏严重失败或少见正确路线。记录候选生成模型、温度、上下文和过滤规则，才能解释偏好数据实际覆盖了什么。

偏好训练可以传递事实、任务与风格信号，效果取决于数据。不要预先把它限定为“只对齐风格”，也不要用一项胜率替代全部能力评测。


## 对照源码与延伸阅读

- [序列token logit与目标概率的计算基础](https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/model.py)
