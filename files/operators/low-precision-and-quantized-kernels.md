# 低比特权重从打包字节到矩阵乘

一个 W4A16 checkpoint 只有原来四分之一左右的权重字节，并不自动意味着服务更快。权重以 4-bit 表示、激活走 16-bit 路径时，kernel 还要找到 packed 码值，恢复对应权重，再与激活相乘。若每次前向都先在显存里展开一份完整 BF16 矩阵，压缩省下的带宽可能被额外读写吃掉。

## 解开一个字节

用教学布局把两个无符号4-bit码值3和12存为195。低四位是3，高四位是12。若scale=0.2、zero point=5，则还原为-0.4和1.4。

```python
byte = 195
q0, q1 = byte & 15, (byte >> 4) & 15
w0, w1 = 0.2 * (q0 - 5), 0.2 * (q1 - 5)
x0, x1 = 2.0, 3.0
print(x0 * w0 + x1 * w1)  # 3.4
```

真实packing常按矩阵指令和访存粒度交错多行。Kernel必须知道位顺序、group映射与scale轴。两个文件同为4-bit，并不能直接共享解包代码。

## 先展开整矩阵还是在tile里恢复

设权重有N个元素。忽略元数据，4-bit码值约N/2字节，BF16约2N字节。如果每次先反量化写出BF16，再由GEMM读取，额外写入与读入合计4N字节，可能抵消压缩读取的收益。

融合路径读取当前tile的packed权重，局部解包并反量化，直接进入乘加。它省下全矩阵中间写回，但增加整数位操作、scale读取和寄存器占用。若weight-only kernel最后使用浮点乘法，其收益主要可能来自权重带宽，而非INT4运算峰值。

## Group size也决定地址和元数据流

每128个权重一个FP16 scale，码值64字节、scale2字节，共66字节。每32个一组时，同样128个权重需要8字节scale，变成72字节。更细分组可能减少量化误差，同时增加scale加载与索引。

若分组沿K方向，输出是各组部分乘积恢复尺度后的和。不同组scale不能在K归约结束后任取一个乘回。设计kernel时要把scale的生命周期和部分累加器一起考虑。

## 一个 tile 内部到底先做什么

把 $Ainmathbb R^{M\times K}$ 与打包后的 $Winmathbb R^{K\times N}$ 相乘时，低比特 kernel 通常不会先把整个 $W$ 展开。一个输出 tile 可以沿下面的顺序推进：

1. 根据 tile 的 $k$ 起点读取 packed bytes、scale 和可选的 zero point。
2. 用位运算取出当前 $K$ 片段的码值，并用 mask 屏蔽尾部越界位置。
3. 在寄存器或共享内存中把码值转换成计算 dtype，直接乘上对应 scale。
4. 读取激活 tile，做矩阵乘加，累加器通常使用 FP32 或硬件指定的更高精度。
5. 继续下一个 $K$ 片段，直到完成归约，最后写回输出 tile。

这里的顺序解释了两个常见取舍。把 scale 提前展开到完整矩阵会增加写回流量；把 scale 留到每个 tile 里读取则会增加地址计算和寄存器压力。tile 越小，边界 mask 和元数据开销越明显；tile 越大，寄存器占用和 occupancy 可能下降。Triton 的矩阵乘教程把同样的 M、N、K 分块、stride 和尾部 mask 写成了可运行的 FP16 示例，低比特版本只是把 B tile 的加载和转换换成了 packed 路径。

![Low-bit LLM quantization granularity](../assets/images/paper-deep-dives/inference/low-bit-llm-survey/figure-2-granularity.png){ width="820" }

<small>图源：[A Survey of Low-bit Large Language Models: Basics, Systems, and Algorithms](https://arxiv.org/abs/2409.16694)，Figure 2。原图表达不同量化粒度会改变 scale、group 和张量布局；读图重点是把 group size、scale 读取和 dequant fusion 放回 kernel 路径。它不能证明某个粒度一定最快，因为 runtime、tile shape 和硬件支持会改变结论。</small>

## 激活和KV带来动态转换

权重通常可离线准备，激活要随每批输入量化，新增范围归约与转换。KV则每步写入新位置、读取历史页，scale既要和页或通道对应，也要在释放与复用时保持一致。

K误差改变attention分数，V误差改变加权值。低精度attention的数值检查需要覆盖两条路径，不能只检查最终平均误差。长上下文还会改变读取量和误差传播。

## 按执行边界验证

先用小张量比较浮点参考、模拟量化与真实packed kernel，确认位序、zero point和scale一致。再测已打包权重的纯kernel、包含转换的完整层及完整请求。

若低batch decode受权重读取限制，压缩可能有明显收益；若大batch矩阵乘主要受计算限制或转换占比高，结果会不同。每个速度结论都应绑定shape、dtype、layout、设备和计时范围。

可以把误差定位拆成四个版本。F 是浮点参考，Q 是模拟量化，K 是真实 packed kernel，T 是完整任务或服务。若 F→Q 已经偏离，先查校准和 group；若 Q 接近而 K 偏离，优先查位序、layout 和累加；若 K 接近而 T 退化，再查长上下文、解码和请求调度。四个版本都要用同一输入和同一 tokenizer，不能用不同样本互相解释。

形状也要按桶验证。至少取一个所有维度都整除 tile 的形状、一个 $M$ 或 $N$ 只差一列的尾部形状、一个 $K$ 不是 group size 整数倍的形状，以及一个真实 decode 的小 batch。尾部形状最容易暴露 mask 与 scale 索引错位；decode 小 batch 则常常暴露 kernel 启动和解包开销。每个桶都先与 reference 做数值比较，再决定是否进入性能表。

CUDA 计时不能只包住 Python 调用。若前一条 kernel 还没有完成，CPU 记录的结束时间会偏早；第一次调用还可能包含 JIT 或 workspace 分配。一个最小的测量框架应明确 warmup、同步点与重复次数：

```python
for _ in range(warmup):
    packed_linear(x)
torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(iters):
    packed_linear(x)
end.record()
end.synchronize()
ms = start.elapsed_time(end) / iters
```

这段代码只测已经准备好的输入和权重。若要报告完整层或服务时间，应另外把 dequant、权重加载、tokenizer、队列和输出传输放进独立的时间边界，不能把不同边界的数字放在同一列。


## 代码入口和延伸阅读

- [AWQ模拟量化与实际量化层构造](https://github.com/mit-han-lab/llm-awq/blob/d6e797a42b9ef7778de8ee2352116e0f48a78d61/awq/quantize/quantizer.py)
- [低精度矩阵乘要求的scale布局](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/README.md)

## 低比特 kernel 的三层验证

第一层验证 packed code、位序、scale 和 zero point；第二层验证 tile 内解包、乘加和累加 dtype；第三层验证完整层和服务的显存、延迟、质量与长上下文行为。只在浮点张量上做 `dequant`，不能证明真实 packed kernel 的地址和布局正确。

**来源与改编说明**：[AWQ quantizer](https://github.com/mit-han-lab/llm-awq/blob/d6e797a42b9ef7778de8ee2352116e0f48a78d61/awq/quantize/quantizer.py) 用来对照模拟与实际量化边界，[DeepGEMM README](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/README.md) 用来核对低精度 scale layout。自测时说明为什么 group size 变小可能降低误差，却增加 metadata 和索引开销。

低层实现可继续阅读 [PyTorch INT4 GQA 优化说明](https://pytorch.org/blog/int4-decoding/) 和 [Triton 矩阵乘教程](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)。前者展示低精度 KV 读取、融合反量化与带宽瓶颈之间的关系，后者提供 block pointer、stride、mask 和 autotune 的基础示例。两份材料的硬件与 benchmark 条件都不能直接当作本站结果。

所以低比特方案至少有两个不同的选择问题。一个是精度问题，group size、scale 粒度和累加格式是否让任务结果可接受；另一个是执行问题，解包和反量化是否落在可隐藏的路径上。两者都通过以后，才有理由比较 checkpoint 大小、显存和端到端延迟。

继续阅读可接[量化 runtime 与框架](../quantization/quantization-runtimes-and-frameworks.md)，看 WQLinear、loader 和真实打包的接口，再看[Serving attention 与 KV kernel](serving-attention-and-kv-kernels.md)，比较权重解包和 KV 读取各自占据哪一段关键路径。
