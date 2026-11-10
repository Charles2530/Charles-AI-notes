# 量化

模型文件从 16 GB 变成 5 GB，很容易让人以为部署问题已经解决了。真正启动服务以后，KV cache、scale、解包、workspace 和长尾请求还会一起占显存。量化页不先给方法排一个名次，而是沿着一个数进入矩阵乘、进入任务输出的路径，把这笔账算出来。

量化做的事情很简单，拿有限的码值去近似原来的数。难点在于误差后来会落在哪里。一个普通激活也许不受影响，一个动作头的 0.02 扰动却可能把离散决策翻过去。

## 先从一个数走到模型

[量化的格子、误差与线性层](symbols-and-minimal-math.md)从舍入、截断与zero point开始，手算同一权重误差怎样在两条输入上放大或抵消。接着读[GPTQ、AWQ与SmoothQuant](ptq-gptq-awq-smoothquant.md)，看三种方法分别怎样使用输入统计、通道缩放和补偿。

当少量大值使其他数都舍入成0时，[激活离群值与校准](activation-outliers-and-calibration-strategies.md)解释如何在截断、粒度和输入分布之间选择。

## 训练和部署各自改了什么

[QLoRA](qlora-and-quantized-training.md)推导冻结4-bit底座时adapter与输入的梯度，说明低比特存储怎样减少微调预算。[模拟量化与真实执行](qat-kernels-and-kv-cache.md)则从STE的一次更新走到packing，并区分权重和不断增长的KV。

若已经有checkpoint，[量化运行时](quantization-runtimes-and-frameworks.md)沿两个4-bit码值追到loader和kernel。[FP8服务](fp8-and-mixed-precision-serving.md)专门解释数据、scale与累加的接口；[显存与硬件取舍](serving-stacks-and-hardware-tradeoffs.md)把权重、KV和workspace放进同一预算。

## 碰到问题时从哪一页开始

| 当前问题 | 入口 |
| --- | --- |
| 方法名称和格式分不清 | [方法与格式对照](comparison-table.md) |
| 数值看起来接近，任务仍掉分 | [多模态与VLA误差传播](quantization-for-multimodal-and-vla-models.md) |
| 模型文件小了，运行却没有更快 | [对照实验与部署检查](evaluation-and-deployment-checklist.md) |
| 想看解包如何融进矩阵乘 | [低精度Kernel](../operators/low-precision-and-quantized-kernels.md) |

各页的手算数字用于验证机制。具体模型能压到多少bit、在哪种硬件上更快，需要在明确配置下测量。

## 对照源码与延伸阅读

- [一个实际量化器的完整定义](https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/quant.py)

## 这组页面怎么读

先拿一个标量和一个小矩阵，把 scale、zero point、舍入、截断和反量化算一遍。接着把 PTQ、QAT、权重量化、激活量化和 KV cache 分开，最后再看 GPTQ、AWQ、SmoothQuant、FP8 以及具体 runtime。方法名称描述的是误差怎么被处理，bit 数本身不会告诉你服务到底多快。

每个量化实验至少保存四种结果：权重文件大小、端到端延迟、峰值显存和任务质量。若只测模型加载时间，无法判断 kernel 是否真的更快；若只看困惑度，无法判断长上下文、视觉细节或工具调用是否退化。

## 量化故障的排查顺序

1. 先检查量化前后输入、输出和累加 dtype。
2. 再检查 scale 的粒度、校准数据和离群值覆盖。
3. 再检查 dequant 是否融入目标 kernel，是否引入额外搬运。
4. 最后在固定任务、长度、batch 和硬件上比较速度与质量。

如果文件变小但速度不变，常见原因是运行时仍解包到高精度、矩阵形状不适合 kernel，或瓶颈在 KV、通信和调度。若质量只在某类输入下降，应保留失败样本和对应激活统计，不能只报告平均分。

## 最小量化实验要同时保留三条路径

对一个固定模型和输入集，至少保存浮点基线、模拟量化和真实量化 kernel 三条结果。浮点基线回答任务本身的上限，模拟量化回答码值和舍入带来的误差，真实 kernel 还会暴露打包、解包、累加 dtype、布局和 fallback 的影响。

| 路径 | 主要检查 | 常见误判 |
| --- | --- | --- |
| FP16/BF16 baseline | 质量、显存、端到端时间 | 把高精度路径当作理想零成本基线 |
| fake quant / dequant | scale、截断、误差传播 | 认为模拟接近就等于 kernel 接近 |
| real low-bit kernel | 解包、累加、workspace、shape | 只测整齐矩阵，忽略真实请求分布 |

三条路径都使用同一 tokenizer、输入切片、解码预算和停止规则。若模拟量化已经掉分，先修校准和粒度；若模拟正常、真实 kernel 掉分，优先检查布局、累加和 dequant；若两者都正常但端到端变慢，瓶颈可能在 KV、通信或调度。这个排错顺序比继续更换量化算法更快缩小范围。

## 来源与改编说明

- **方法直觉**参考 [Quantization](https://www.maartengrootendorst.com/blog/quantization/)，用于表示范围、裁剪和校准的连续解释。
- **工程细节**参考 [PyTorch INT4 decoding](https://pytorch.org/blog/int4-decoding/) 与 [KV-cache quantization](https://huggingface.co/blog/kv-cache-quantization)，版本、硬件和 benchmark 条件必须保留。
- **自测**：给出一个权重张量和一个激活张量，分别说明为什么权重量化可能只影响加载/乘法，而 KV 量化会随上下文长度增长；再写出一个“文件更小但端到端更慢”的可测原因。
