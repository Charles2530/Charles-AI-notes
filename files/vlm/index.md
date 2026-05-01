# VLM 总览

视觉语言模型（VLM）研究的是如何联合处理视觉输入 \(x_v\) 与文本输入 \(x_t\)，并输出文本、标签、框、检索结果或工具调用。

!!! tip "基础知识入口"
    VLM 的公共底座是 `Tokenization`、`Embedding`、`Self-Attention`、`Cross-Attention`、位置编码和视觉特征提取。第一次读可以先看 [Transformer 与 Attention](../foundations/transformer-attention-and-tokenization.md)、[位置编码、Mask 与上下文](../foundations/positional-encoding-masks-and-context.md) 和 [卷积与特征提取](../foundations/convolution-and-feature-extraction.md)。

**一个统一抽象是**：

\[
p_\theta(y \mid x_v, x_t)
\]

其中 \(y\) 可以是：

- 一段回答
- 一个类别标签
- 一组框坐标
- 一次工具调用
- 一个多步代理动作

![CLIP pre-training and zero-shot transfer](../assets/images/paper-figures/contrastive-learning/clip-figure-1.png){ width="920" }

<small>图源：[Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)，Figure 1。原论文图意：用图像编码器和文本编码器把图文对拉到同一表示空间，并把类别文本 prompt 当作 zero-shot 分类器。</small>

!!! note "图解：为什么 VLM 首先要学“图文对齐”"
    CLIP 这张图说明了 VLM 的一个基础入口：先不急着让模型长篇回答，而是先让“图片”和“文字描述”在同一个向量空间里靠近。这样做的好处是检索、分类、召回都很高效；缺点是它主要学到全局匹配，未必能精确回答“左下角第二行金额是多少”或“点哪个按钮”。所以后续 VLM 会在这个对齐底座上继续加入 cross-attention、LLM 连接器、高分辨率阅读和 grounding 能力。

## 1. 为什么 VLM 会成为基础设施

因为很多真实任务本来就是“看图 + 理解语言 + 做决策”：

- 读图表并解释趋势
- 看发票并抽取字段
- 看屏幕并决定点击哪个按钮
- 看机器人相机图像并理解指令
- 看商品图并完成检索或问答

可以说，VLM 是把“感知”和“语言接口”真正接起来的一层。

## 2. 从输入到输出，VLM 在做什么

**最常见的内部流程可以概括成**：

\[
x_v \xrightarrow{f_v} h_v,\qquad x_t \xrightarrow{f_t} h_t,\qquad (h_v,h_t)\xrightarrow{P} z \xrightarrow{\text{decoder}} y
\]

这里：

- \(f_v\) 是视觉编码器
- \(f_t\) 是文本编码器或 LLM
- \(P\) 是跨模态连接器或交互模块

直觉上，VLM 至少要解决三件事：

1. 看清局部视觉证据
2. 把视觉实体和文本概念对上
3. 按任务形式输出结果

## 3. 一个直观例子：咖啡店收银台

输入一张咖啡店收银台照片和问题“哪种甜点卖完了？”  
纯视觉模型只能识别图片区域；纯语言模型看不到价牌。VLM 则要把两者合在一起，读懂图上的手写标记 `Sold out`，并把视觉证据组织成答案。

**这个过程背后其实包含**：

- OCR
- 文本条件理解
- 目标定位
- 结果组织

## 4. VLM 的四个主要能力层

### 感知层

**是否看清**：

- 文本
- 物体
- 布局
- 颜色
- 关系

### 对齐层

是否知道问题里说的“那个按钮”“左边那张表”“蓝色柱子”对应图里的什么。

### 推理层

**是否能基于视觉证据做**：

- 比较
- 统计
- 多跳推理
- 因果解释

### 执行层

若接了工具或代理动作，是否真能：

- 选对工具
- 调对参数
- 走对步骤

## 5. 为什么 VLM 不是简单“给 LLM 插个眼睛”

很多初学者会把 VLM 理解成“把图像特征喂给 LLM”。  
这只说对了一部分。真正的难点在于：

- 图像是高维空间信号
- 文本是离散符号序列
- 任务可能要求定位、检索、问答、代理等完全不同输出

因此一个好 VLM 不仅要语言强，还要具备：

- 稳定视觉表征
- 可解释对齐
- 可落地的系统设计

## 6. VLM 常见应用面

### 文档理解

例如：

- 发票抽取
- 合同审阅
- 表格问答

### 屏幕代理

例如：

- 理解界面
- 点击控件
- 多步操作业务系统

### 多模态问答

例如：

- 图表分析
- 医疗影像问答
- 商品问答

### 多模态检索

例如：

- 图搜文
- 文搜图
- 商品向量召回

## 7. 三类最常见的工程路线

### 双塔路线

更偏检索和召回。

### 视觉编码器 + LLM 路线

更偏问答和代理。

### 高分辨率文档路线

更偏 OCR、布局、表格和长文档场景。

## 8. 一个足够实用的理解方式

如果把传统 CV 看成“模型会看”，LLM 看成“模型会说”，那么 VLM 真正追求的是：

**让模型既看得见，又说得对，还能在必要时做得动。**

## 9. 推荐阅读顺序

若你是第一次系统看 VLM，建议顺序是：

1. [架构与训练](architecture-and-training.md)
2. [评测与工具使用](evaluation-and-tool-use.md)
3. [方法对照表](comparison-table.md)

先理解模型内部结构与训练目标，再看它如何在真实任务链路里被评测和部署。

## 快速代码示例

```python
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForVision2Seq.from_pretrained(model_id, device_map="auto")

image = Image.open("invoice.png")
prompt = "提取发票中的金额、日期、开票方。"
inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=128)
print(processor.decode(out[0], skip_special_tokens=True))
```

这段代码展示了 VLM 的标准推理入口：`processor` 负责把图像与文本统一编码，`model.generate` 负责多模态解码输出。用票据类 prompt 可以快速验证 OCR 与结构化信息抽取能力。


## 学习路径与阶段检查

VLM 专题建议从“看清 -> 对齐 -> 推理 -> 执行”四层推进。不要只看模型参数量，也不要只看通用问答分数。

| 阶段 | 先读 | 读完要能回答 |
| --- | --- | --- |
| 1. 架构和数据 | [架构与训练](architecture-and-training.md)、[多模态数据治理与合成增强](multimodal-data-curation-and-synthetic-generation.md) | 视觉编码器、连接器、LLM、指令数据和合成数据分别影响哪类能力 |
| 2. 视觉证据 | [文档理解与 OCR](document-understanding-and-ocr.md)、[图表、表格与结构化推理](charts-tables-and-structured-reasoning.md) | 模型是否真的读到了文字、布局、表格和图形关系，而不是靠语言先验猜 |
| 3. 任务闭环 | [评测与工具使用](evaluation-and-tool-use.md)、[屏幕代理与 Grounding](screen-agents-and-grounding.md) | VLM 输出是答案、坐标、工具调用还是多步动作，不同输出如何验收 |
| 4. 系统消费 | [检索、重排与向量系统](retrieval-reranking-and-vector-systems.md)、[鲁棒性与长尾评测](robustness-and-long-tail-evaluation.md) | 表征、检索、长尾评测和线上回放是否支撑同一套业务目标 |

读完后可以继续接 [VLA 总览](../vla/index.md) 或 [具身智能总览](../embodied-ai/index.md)。判断是否可以进入下一阶段的标准很简单：你能否把一次错误拆成“没看清、没对齐、推理错、工具接口错、评测没覆盖”中的哪一种。
