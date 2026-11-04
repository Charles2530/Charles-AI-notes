# 一行 hidden state 怎样归一化、重排与写回

给定一行 `[1,2,3,4]`，LayerNorm需要先算整行统计，再产生四个输出。它的乘加很少，却需要多个元素共同决定一个结果。把这一行追完，可以同时理解reduction、mask、stride和梯度累积。

## LayerNorm先把四个数变成两个统计量

均值为2.5，平方偏差平均为1.25。忽略epsilon、令gamma=1和beta=0，输出为

$$
\frac{[1,2,3,4]-2.5}{\sqrt{1.25}}
\approx[-1.34164,-0.44721,0.44721,1.34164].
$$

前向至少需要得到均值和方差，再应用逐元素变换。实际算法可能分多遍读取，也可能使用Welford等合并统计的办法；选择受行宽、资源和数值稳定性影响。

RMSNorm不减均值，使用 $\sqrt{\operatorname{mean}(x^2)+\epsilon}$。本例平方均值为7.5，输出约 `[0.36515,0.73030,1.09545,1.46059]`。它与LayerNorm得到的函数不同，不能只因为省一次统计就无条件替换。

## Padding不能偷偷进入分母

假设真实行长为3，kernel为了方便加载4个位置。无效位置可填0帮助求和，但均值分母仍应为3。算方差时，还要屏蔽无效位置的平方偏差，否则填入的0经过减均值后会产生非零贡献。

Softmax的无效logit通常填负无穷，让指数权重为0。若填0，而有效logit都为负，无效位置反而会抢占概率。Mask既保护内存，也定义计算对象，这两个作用必须同时正确。

## 反向传播多了一层跨行归约

对每行的LayerNorm，输入梯度可以用该行统计量计算。共享参数gamma和beta却会收到所有行的贡献，$d\beta$是上游梯度按行求和，$d\gamma$是上游梯度乘归一化输入再按行求和。

Triton官方LayerNorm教程先按行计算dx及部分dw/db，再用另一段归约合并参数梯度。这解释了为何一个前向看起来很短的norm算子，训练实现仍需考虑临时缓冲、写入竞争与额外归约。

## Shape相同，地址可以不同

行优先2×3矩阵的stride为(3,1)，元素(1,2)在偏移5。转置视图为3×2，stride变成(1,3)，元素(2,1)仍在偏移5。转置可只改元数据，原存储没有移动。

若后续kernel要求行连续，转置视图就可能需要真正复制。`contiguous()`的成本来自读写全部元素。优化完整层时要把这次复制算进去，不能只看归一化kernel。

## Gather与scatter改变数据对应关系

若索引为 `[2,0,2]`，gather输出 `[x2,x0,x2]`。反向时x2要收到两处梯度之和，不能简单用后一次写入覆盖前一次。这是scatter-add或等价归约要解决的问题。

在MoE里，同一token可能被送给多个专家，合并时也需要恢复原位置并按路由权重累加。[MoE算子](moe-routing-and-sparse-kernels.md)会把这个索引例子扩展成完整专家输入。

## 融合的收益要落到少了哪次搬运

若残差加法后立即做norm，可以尝试在同一kernel读取两路输入、相加、归约并输出，避免把中间和写回后再读。代价是更多同时存活的值及可能更复杂的反向。先用上述小例子确认数学与mask，再在真实行宽和stride上测量。

![Nsight Systems CPU/GPU correlation](../assets/images/tools/nsight-systems/cpu-gpu-correlation.png){ width="820" }

<small>图源：[NVIDIA Nsight Systems 文档](https://docs.nvidia.com/nsight-systems/)。原图表达短 kernel、CPU 调度和 GPU 时间线的关联；读图重点是 reduction、norm、layout 和 indexing 这类小算子往往以“很多短条”拖慢端到端路径。它不能证明具体哪个小算子是瓶颈，仍需按 trace 下钻。</small>

## 对照源码与延伸阅读

- [LayerNorm前向、dx与dw/db分段归约](https://github.com/triton-lang/triton/blob/cd2e01c6686d6503802e6cdfac56d5a5dfdb27a1/python/tutorials/05-layer-norm.py)
- [Softmax的负无穷mask与片上归约](https://github.com/triton-lang/triton/blob/cd2e01c6686d6503802e6cdfac56d5a5dfdb27a1/python/tutorials/02-fused-softmax.py)
