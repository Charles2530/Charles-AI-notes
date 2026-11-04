# 公式、源码与实验怎样对照

看到论文公式，最常见的障碍是每个符号似乎都认识，却不知道它在程序里出现在哪里。可以先选一条完整计算，把输入、前向、损失和更新连起来，再扩展到大模型。这个方法不要求先通读所有基础，也不以完成多少章节衡量理解。

## 从一个能算完的目标开始

设模型输出 \(\hat y=wx\)，训练目标为 \(\ell=\tfrac12(wx-y)^2\)。取 \(x=2,y=3,w=1\)，前向预测为2，误差为−1，损失为0.5。链式法则给出

\[
\frac{\partial\ell}{\partial w}=(wx-y)x=-2.
\]

学习率0.1时，新权重为1.2，新预测2.4，新损失0.18。每个数字都对应一个程序对象。这里没有 batch、正则化、动量和混合精度，方便先检查核心机制。

```python
x, y, w, lr = 2.0, 3.0, 1.0, 0.1
loss = lambda v: 0.5 * (v*x-y)**2
grad = (w*x-y)*x
eps = 1e-5
finite_difference = (loss(w+eps)-loss(w-eps))/(2*eps)
w_new = w-lr*grad
print(grad, round(finite_difference,6), w_new, round(loss(w_new),6))
assert abs(grad-finite_difference) < 1e-6
```

输出 `-2.0 -2.0 1.2 0.18`。有限差分检验的是这一个点附近的导数实现，并没有证明训练在任意数据上都收敛。

## 公式中的约定会改变代码

如果把半平方误差改成平方误差，梯度会变为−4；如果对 batch 求和改成平均，梯度会除以样本数。它们都可能是合理目标，代码必须与实际公式一致。分布式训练的有效 token 分母、蒸馏的温度平方、扩散的时间权重，也属于同类问题。

先给符号一张具体映射表，再读网络代码。

| 公式对象 | 要在源码中找到什么 | 容易遗漏的条件 |
| --- | --- | --- |
| 输入 x | dataset、collate、forward 参数 | 归一化、轴顺序、padding、单位 |
| 参数 θ | 可训练 tensor 与 optimizer 参数组 | 冻结部分、共享参数、weight decay |
| 预测 fθ(x) | 生成目标前的输出 | logits 还是概率，噪声还是速度 |
| 目标 y | label、教师输出或环境反馈 | 是否右移、停止梯度、泄漏未来 |
| 损失 L | reduction、mask、权重 | 按 token、序列还是设备平均 |
| 更新 | backward、step、scheduler | 累积次数、精度、跳步与状态恢复 |

## 沿源码的数据流阅读

[micrograd 的 Value 实现](https://github.com/karpathy/micrograd/blob/7bc720e951fe422b8f8814aa5aa1b64121d26b4c/micrograd/engine.py)把标量运算、局部反向函数与拓扑遍历放在较短代码中。可先沿乘法和加法找到上例的前向值，再沿 `backward()` 看梯度如何累加。共享变量被使用多次时，反向需要累加贡献，这一点很容易被手写实现遗漏。

[PyTorch 优化教程](https://github.com/pytorch/tutorials/blob/5672cd98a7671eda4d9122b10f8eec580f74d03e/beginner_source/basics/optimization_tutorial.py)再把过程扩成 tensor、batch 和优化循环。`zero_grad()`、`backward()`、`step()`分别清理历史梯度、计算当前梯度和更新参数。移动调用顺序可能改变算法，不能只确认这三个名字都出现了。

```mermaid
flowchart LR
    A["公式约定"] --> B["手算小输入"]
    B --> C["参考代码与局部导数"]
    C --> D["边界输入对照"]
    D --> E["实际任务与资源测量"]
    E -. "发现不一致后回查" .-> A
```

小输入对照能检验公式到代码的映射；实际任务测量才进一步检验泛化和成本。两种证据需要同时保留，不能用一段 smoke test 宣称复现了论文结果。

## 怎样增加复杂度而不丢掉依据

将标量换成二维输入时，先检查形状与梯度；换成 batch 时，检查归约分母；加入 mask 时，构造只有一个有效目标的样本；加入缓存时，对照无缓存结果；加入低精度时，对照误差和真实算子路径。每次增加一个机制，就给它一个能辨别对错的输入。

例如文档打包后的张量尺寸可能完全正确，但 B 文档读取了 A 文档，训练目标已经改变。[位置与 Mask](../foundations/positional-encoding-masks-and-context.md)中的4×4例子专门暴露这种错误。测试形状不能替代测试信息能否流动。

## 论文中的结论要和实验设置一起带回

阅读新方法时，先定位它改动的那一段计算，再看实验如何隔离这一改动。如果新方案同时扩大数据、增加训练步数和更换硬件，结果支持的是组合方案，不能把全部提升归给某个单独模块。

记录最关键的可复查信息即可，包括来源 commit、数据划分、模型配置、目标定义、随机性、评价协议和运行设备。若本地只验证了机制，应写“机制算例”；若确实训练了模型，再给训练日志与评测。原论文的性能数字和自己的测量应明确区分。

[数据划分与评测](../foundations/data-splits-metrics-and-evaluation-basics.md)解释比较协议，[性能分析](../operators/profiling-debugging-and-numerical-stability.md)解释算子计时。遇到不一致时，保留产生它的最小输入，通常比继续扩大实验更容易找到原因。
