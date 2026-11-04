# 量化

量化用有限的码值近似模型里的数。若scale为0.25，浮点0.9可以存为整数4，读取时恢复成1.0。模型因此引入0.1的误差，同时有机会减少存储和数据搬运。这个误差进入矩阵乘以后是否重要，要看输入、后续计算以及任务输出。

## 从一个数走到模型

[量化的格子、误差与线性层](symbols-and-minimal-math.md)从舍入、截断与zero point开始，手算同一权重误差怎样在两条输入上放大或抵消。接着读[GPTQ、AWQ与SmoothQuant](ptq-gptq-awq-smoothquant.md)，看三种方法分别怎样使用输入统计、通道缩放和补偿。

当少量大值使其他数都舍入成0时，[激活离群值与校准](activation-outliers-and-calibration-strategies.md)解释如何在截断、粒度和输入分布之间选择。

## 训练和部署改变不同的状态

[QLoRA](qlora-and-quantized-training.md)推导冻结4-bit底座时adapter与输入的梯度，说明低比特存储怎样减少微调预算。[模拟量化与真实执行](qat-kernels-and-kv-cache.md)则从STE的一次更新走到packing，并区分权重和不断增长的KV。

若已经有checkpoint，[量化运行时](quantization-runtimes-and-frameworks.md)沿两个4-bit码值追到loader和kernel。[FP8服务](fp8-and-mixed-precision-serving.md)专门解释数据、scale与累加的接口；[显存与硬件取舍](serving-stacks-and-hardware-tradeoffs.md)把权重、KV和workspace放进同一预算。

## 查表和验证

| 当前问题 | 入口 |
| --- | --- |
| 方法名称和格式分不清 | [方法与格式对照](comparison-table.md) |
| 数值看起来接近，任务仍掉分 | [多模态与VLA误差传播](quantization-for-multimodal-and-vla-models.md) |
| 模型文件小了，运行却没有更快 | [对照实验与部署检查](evaluation-and-deployment-checklist.md) |
| 想看解包如何融进矩阵乘 | [低精度Kernel](../operators/low-precision-and-quantized-kernels.md) |

各页的手算数字用于验证机制。具体模型能压到多少bit、在哪种硬件上更快，需要在明确配置下测量。

## 对照源码与延伸阅读

- [一个实际量化器的完整定义](https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/quant.py)
