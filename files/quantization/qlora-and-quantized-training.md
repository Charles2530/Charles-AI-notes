# QLoRA 中，冻结的 4-bit 权重怎样参与训练

一个线性层有4096个输入和4096个输出，共约1678万个权重。仅用 BF16 保存，就需要32 MiB；若每个权重只存4 bit，码值占8 MiB，还要另加尺度等元数据。QLoRA 利用这个差别保存冻结底座，再训练一个较小的低秩增量。

## 冻结底座，训练哪两块矩阵

设底座权重 $W_0\in\mathbb R^{d_{out}\times d_{in}}$，量化码与元数据记为 Q，反量化结果为 $\hat W_0=D(Q)$。LoRA 使用 $A\in\mathbb R^{r\times d_{in}}$、$B\in\mathbb R^{d_{out}\times r}$，对列向量输入 x 计算

$$
y=\hat W_0x+cBAx,\qquad c=\alpha/r.
$$

Q 在训练中冻结，A、B 可以更新。底座低比特存储、矩阵乘的计算 dtype、累加 dtype 是不同选择。QLoRA 的官方脚本通过 `BitsAndBytesConfig` 分别设置4-bit加载、量化类型、double quant和计算dtype，再用 PEFT 加入 LoRA 层。

对于4096×4096、r=8的例子，A与B合计 $8(4096+4096)=65536$ 个参数，只占原矩阵参数量的约0.39%。这部分还需要梯度和优化器状态，激活也仍占显存，所以整次训练不会简单缩到原来的四分之一。

![QLoRA memory comparison](../assets/images/paper-figures/quantization/qlora-figure-1-memory.png){ width="760" }

<small>图源：[QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)，Figure 1。原图表达：对比 32-bit/16-bit 全量微调、LoRA 与 QLoRA 的显存组织；QLoRA 将冻结底座量化存储，只训练 adapter，并用 paged optimizer 缓解峰值。读图重点：看哪些参数“会训练”，哪些只是“低比特常驻”。</small>

## 冻结权重仍然要传递输入梯度

令上游梯度 $g=\partial L/\partial y$。链式法则给出

$$
\frac{\partial L}{\partial B}=cg(Ax)^\top,\qquad
\frac{\partial L}{\partial A}=c(B^\top g)x^\top,
$$

$$
\frac{\partial L}{\partial x}=(\hat W_0+cBA)^\top g.
$$

冻结底座表示不更新 $W_0$，并不表示梯度在这一层被截断。输入梯度还要穿过底座的线性映射，才能训练更早层的 adapter。这里的导数使用反量化后的近似权重，不需要对冻结的离散码值求梯度。

取 r=1、$A=[1,0]$、B初始全零、$x=[2,3]^\top$，c=1。此时 Ax=2，$\partial L/\partial B=2g$，但 $\partial L/\partial A=0$。第一步先让B离开零，后续A才能从这条路径获得非零梯度。这解释了常见“一侧随机、另一侧置零”初始化为何初始输出不变且仍能开始学习；若A和B都置零，则两边梯度都会卡住。

<!-- systems-workflow-visual -->
```mermaid
flowchart LR
  X["输入x"] --> W["冻结量化底座反量化后乘x"]
  X --> A["可训练A"] --> B["可训练B与比例c"]
  W --> Y["两路相加得到y"]
  B --> Y
  Y --> G["上游梯度g"] --> U["更新A/B · 底座码值不更新"]
  G --> DX["输入梯度仍通过底座传回前层"]
```

冻结参数与截断输入梯度是不同操作。QLoRA保留经过底座的输入梯度。

## NF4、double quant 与分页各减少哪一项

NF4 使用适配近似正态权重分布的非均匀码本。4-bit 数字表示码本索引，反量化需要查表并乘上块尺度，不能套用一个均匀整数格距解释全部码值。

Double quantization 进一步压缩量化尺度等统计量。它影响元数据存储，不能被理解为把模型连续做两次任意量化。Paged optimizer 借助分页内存机制处理优化器状态的内存压力；数据发生迁移时仍有开销，也不意味着激活显存无需考虑。

官方脚本还把部分 norm 模块转为 FP32，并按配置处理 adapter dtype。这说明“4-bit训练”不足以描述完整计算图。实际实验应记录量化类型、计算dtype、LoRA目标模块、rank、序列长度与checkpoint设置。

### 最小配置骨架应该检查哪几处

下面的代码只展示当前 Transformers 与 PEFT 常见的接口关系，不能替代所安装版本的 API 文档，也没有在本机 GPU 上运行。

```python
import torch
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)

model = load_base_model(quantization_config=bnb_config)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

这段骨架有四个验收点。`load_in_4bit` 只决定底座的存储路径，`bnb_4bit_compute_dtype` 决定部分计算的输入与累加配置；`target_modules` 决定 LoRA 插入哪些线性层，不能因为模型都叫 Transformer 就照抄同一组名字；`print_trainable_parameters()` 应确认底座没有意外变成可训练。具体字段以 [Transformers bitsandbytes 文档](https://huggingface.co/docs/transformers/en/quantization/bitsandbytes) 和 [PEFT 量化指南](https://huggingface.co/docs/peft/en/developer_guides/quantization) 的当前版本为准。

训练结束只保存 adapter 时，还要把底座的模型 ID、revision、tokenizer、量化配置、LoRA target modules 和训练数据切分一起写入 manifest。重新加载的顺序应是“同一底座与 tokenizer → 同一量化配置 → adapter”，不能先把 adapter 合并到另一个量化版本再比较输出。

| 现象 | 第一处应检查什么 |
| --- | --- |
| trainable 参数接近全部参数 | 是否在量化加载后错误地解冻底座，`print_trainable_parameters()` 的百分比是否合理 |
| loss 很快出现 NaN | 计算 dtype、norm 的精度、梯度裁剪和输入归一化；不要只把学习率调小 |
| 序列变长后突然 OOM | 激活、梯度累积和 checkpoint，而不是只看 4-bit 权重大小 |
| adapter 能加载但输出明显变化 | 底座 revision、tokenizer、LoRA 目标层和 merge/re-quantize 路径是否一致 |

这里的“最小回归”仍应包含未训练底座、独立 adapter、合并但未重新量化和重新量化四种输出。每种结果都固定同一批输入并保存 logits 或生成文本，才能定位差异是在 adapter 注入、合并，还是再次量化时产生的。

## 保存 adapter 后，推理要装回哪个底座

A、B是在固定的 $\hat W_0$ 上学习到的。换成另一个底座版本、不同量化格式或不同tokenizer，即使adapter文件能加载，也可能改变函数。保存时应记录底座身份与量化配置。

如果部署选择把增量合并为 $\hat W_0+cBA$，再重新量化，得到的是一个新近似。原先分开执行的结果与重新打包后的结果未必一致，需要重新比较输出和任务质量。若保持独立adapter，则runtime要支持对应低比特底座与adapter路径。

读[QAT](qat-kernels-and-kv-cache.md)时可用这个区别定位目标。QLoRA优化的是冻结近似底座上的增量；QAT会在训练前向模拟离散误差，让可训练参数适应目标量化路径。

## 一个最小 QLoRA 回归应固定什么

先固定底座 commit、tokenizer、量化类型、计算 dtype、LoRA 目标模块、rank、序列长度和数据切分。用同一小批样本比较四条结果：未训练的量化底座、LoRA 训练后的独立 adapter、合并后未重新量化的权重、重新量化后的权重。若差异从第二条开始出现，重点看 adapter 注入和训练状态；若只在第四条出现，重点看合并与再次量化。

报告显存时把冻结权重、adapter、梯度、优化器、激活和临时 dequant 缓冲分别列出。这样才能解释为什么参数量少了，峰值显存却仍被长序列激活主导。这个小回归也能防止把“4-bit checkpoint 能加载”误写成“4-bit 训练已经完整复现”。


## 继续读

继续看[QAT、kernel 与 KV cache](qat-kernels-and-kv-cache.md)和[低 bit 训练与数值稳定性](../training/low-bit-training-and-numerics.md)。

## 对照源码与延伸阅读

- [QLoRA 官方加载、dtype 与 LoRA 注入实现](https://github.com/artidoro/qlora/blob/7f4e95a68dc076bea9b3a413d2b512eca6d004e5/qlora.py)

**来源与改编说明**：量化底座、NF4、double quant 和 paged optimizer 的机制参考 [QLoRA 论文](https://arxiv.org/abs/2305.14314)，配置与注入边界对照固定版本的 [QLoRA 官方脚本](https://github.com/artidoro/qlora/blob/7f4e95a68dc076bea9b3a413d2b512eca6d004e5/qlora.py)。页面中的矩阵尺寸和梯度例子是教学计算，不是训练 benchmark。
