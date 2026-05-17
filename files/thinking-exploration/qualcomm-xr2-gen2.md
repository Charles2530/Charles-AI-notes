# XR2 Gen 2：端侧 AI 与量化部署

读法定位：端侧部署案例；区分公开平台事实、AI Hub proxy 口径和本站判断。回 [量化](../quantization/index.md) 或 [术语表](../glossary/index.md)。
下一站：回 [思考探索](index.md)。

!!! tip "一句话总结"
    **XR2 Gen 2 的价值不是单点 AI 跑分，而是把显示、相机、渲染、无线和端侧 AI 放在同一个电池供电头显里同时跑起来；QCS8450 / XR2 Gen 2 Proxy 则是 AI Hub 里帮助开发者提前做模型适配和性能判断的工程入口。**

先想一个具体画面：戴上 Quest 3，打开彩色透视，低头看桌面，抬手点虚拟按钮。几百毫秒里，头显不只是“跑了一个 AI 模型”，还要接收多路相机、估计头手位置、做 MR 透视、渲染双眼画面、处理无线连接，并让语音、场景理解或应用侧视觉模型并发运行。

这就是 XR2 Gen 2 的工作现场：**不是一次性 benchmark，而是持续运行的感知-渲染-交互闭环。**所以讨论 Qualcomm XR 平台时，只问“AI TOPS 有多少”不够；更关键的是模型如何进入软件栈，并在真实头显里稳定、低延迟、低功耗地跑。

这篇文章就沿着这个场景回答三个问题：

1. `XR2 Gen 2` 是面向 MR/VR 头显的产品平台，不只是一个 AI 加速器。
2. `QCS8450 / XR2 Gen 2 (Proxy)` 是 AI Hub 里的目标设备口径，适合早筛模型；后文第一次解释 Proxy 时会把边界讲清楚。
3. `AI Hub + QNN + AIMET + QIDK` 共同组成从模型到端侧部署的工具链，量化是否更快要看 HTP 代际、runtime 和模型图。

## 先看一台头显里发生了什么

把 Quest 3 这类一体式 MR 头显拆开看，XR2 Gen 2 要同时支撑两条主线：一条是用户能不能看得稳、跟得上；另一条是模型能不能在不拖慢系统的前提下持续工作。前者对应显示、GPU、相机、ISP、tracking 和无线；后者对应 Hexagon / HTP / NPU、QNN、LiteRT、ONNX Runtime、AIMET 和 AI Hub。

如果只记一张图，可以把 Qualcomm 这套东西理解成下面这条链：

```text
训练好的 PyTorch / ONNX 模型
  -> AI Hub Workbench 做 compile / quantize / profile / inference
  -> 生成 LiteRT、ONNX Runtime 或 QNN 部署资产
  -> 跑到 Snapdragon XR2 Gen 2 / QCS8450 这类 XR 设备的 NPU、GPU、CPU 上
  -> 服务头显里的渲染、感知、语音、手眼追踪、深度和应用 AI
```

这条链路里有两个关节点。第一个关节点是 **Workbench**：它把一个训练好的模型变成某个目标设备能跑的部署资产，并返回延迟、内存、layer placement 和数值结果。第二个关节点是 **真实设备验证**：XR 头显不是离线 benchmark 机器，模型推理会和显示、摄像头、SLAM、无线、音频和应用渲染争同一个功耗与散热空间。

这也是后面所有判断的主线：公开文档能让我们知道平台能力边界，AI Hub 能让我们提前筛选模型和 runtime，QIDK / QNN 能帮助工程集成；但最终产品是否成立，仍然要看真实头显里的连续运行、热稳定、帧时间和交互延迟。

## 再把名字捋顺

有了上面的场景，再看名字会更容易。产品页讲的是 Snapdragon XR2 Gen 2，模型库里出现的是 QCS8450 / XR2 Gen 2 Proxy，SDK 文档又在讲 QNN、LiteRT、ONNX Runtime、AIMET、QAIRT。它们不是同一个东西的不同叫法，而是同一条部署链路上的不同角色：芯片平台、云端目标设备、模型编译服务、本地 runtime 和量化工具。

几个最容易混淆的名字可以先这样分开：

| Name | What it means | How to read it |
| --- | --- | --- |
| Snapdragon XR2 Gen 2 Platform | Qualcomm 官方对外的 XR 产品平台名称 | 面向 standalone MR / VR headset 的 SoC 平台 |
| QCS8450 XR Platform | AI Hub Models README 里列出的 XR chipset 口径 | 更偏开发、scorecard、兼容性测试里的芯片目标名 |
| QCS8450 (Proxy) | AI Hub Workbench / AI Hub Models 中的 proxy device | 替代设备口径，用来早期判断 NPU 兼容性和生成部署资产 |
| XR2 Gen 2 (Proxy) | AI Hub Models 页面上可见的 XR proxy device | 面向 XR scorecard 的 proxy 名称，和 QCS8450 (Proxy) 属于同一类工程入口 |

最重要的是最后两行。[AI Hub Devices 文档](https://workbench.aihub.qualcomm.com/docs/hub/devices.html)把 `(Proxy)` 定义为：当 Workbench 没有托管真实设备时，用替代设备服务于目标设备。它可以帮助判断模型兼容性和生成资产，但官方也提醒，真实设备的性能和准确率会因为操作系统、固件、频率、内存、散热封装等因素而变化。因此，**Proxy 数字适合做早期选型，不适合作为最终产品 KPI。**这个结论后文会直接引用，不再反复展开。

换句话说，`QCS8450 (Proxy)` 可以回答“这个模型大概率能不能走 Qualcomm NPU 路径”“部署资产大概怎么生成”“哪些 runtime 组合值得试”。至于连续运行、热稳定、相机透视和渲染并发，那是下一阶段的整机验证问题。

## QCS8450 在芯片代际里的位置

接下来要再澄清一个关键点：AI Hub 里的 QCS8450 不是“名字里带 8450，所以一定等价于某个更新一代的 8 Gen 2 / 8 Gen 3 平台”。在 `qualcomm/ai-hub-models` 的 [`devices_and_chipsets.yaml`](https://github.com/qualcomm/ai-hub-models/blob/main/src/qai_hub_models/devices_and_chipsets.yaml) 里，`qualcomm-qcs8450-proxy` 被标成 XR world，`htp_version: 69`，`soc_model: 36`；同一个文件里，`qualcomm-snapdragon-8gen1` / `sm8450` 也是 `htp_version: 69`、`soc_model: 36`。也就是说，从 AI Hub scorecard 的硬件后端口径看，QCS8450 Proxy 更接近 SM8450 / Snapdragon 8 Gen 1 这一代，而不是 SM8550 / Snapdragon 8 Gen 2 之后的 HTP v73+ 平台。

这个定位很重要，因为端侧 AI 的很多限制不是由“产品名”决定，而是由 SoC 后端、HTP 版本、runtime 和模型图共同决定。某个量化格式、算子融合或 QNN 编译选项在 v73 / v75 上表现正常，不代表可以直接迁移到 v69；反过来，v69 上能跑通的路径，也不一定代表它是更新芯片上的最佳路径。

| Chipset / proxy target | Typical alias / product line | World | HTP version in AI Hub Models | SoC model | How to read it |
| --- | --- | --- | ---: | ---: | --- |
| Snapdragon 888 | SM8350 | Mobile | 68 | 30 | 前一代高端移动平台，可作为 v69 之前的参照 |
| Snapdragon 8 Gen 1 | SM8450 | Mobile | 69 | 36 | 和 QCS8450 Proxy 在 AI Hub 元数据里处在同一 HTP / SoC model 口径 |
| QCS8450 (Proxy) | XR2 Gen 2 / QCS8450 target | XR | 69 | 36 | 本文重点，适合做 XR 模型早筛，但不是 v73+ 后端 |
| QCS8550 (Proxy) | QCS8550 target | IoT | 73 | 43 | 进入 HTP v73 口径，很多较新的 QNN / 量化路径会更值得优先验证 |
| Snapdragon 8 Gen 2 | SM8550 | Mobile | 73 | 43 | 和 QCS8550 Proxy 在 HTP / SoC model 口径上对齐 |
| Snapdragon 8 Gen 3 | SM8650 | Mobile | 75 | 57 | 后续移动旗舰平台，NPU / HTP 能力继续演进 |
| Snapdragon 8 Elite | SM8750 | Mobile | 79 | 69 | 更晚一代旗舰平台，不能拿它的 profile 倒推 QCS8450 |
| Snapdragon 8 Elite Gen 5 | SM8850 | Mobile | 81 | 87 | 当前更新一代移动旗舰口径，和 QCS8450 已经不是同一代后端 |

<small>表源：`qualcomm/ai-hub-models` 的 [`devices_and_chipsets.yaml`](https://github.com/qualcomm/ai-hub-models/blob/main/src/qai_hub_models/devices_and_chipsets.yaml)。这里的 `htp_version` 和 `soc_model` 是 AI Hub 模型库用于 scorecard / device target 的工程元数据，不等同于 Qualcomm 对外产品命名。</small>

## XR2 Gen 2 到底是什么芯片

把命名和代际放好之后，再回到产品本身。XR2 Gen 2 最核心的定位不是“手机芯片换个名字”，而是专门给一体式头显做的空间计算平台。

Snapdragon XR2 Gen 2 是 Qualcomm 在 2023 年发布、2024 年 [Product Brief](https://docs.qualcomm.com/doc/87-73689-1/) 中继续整理规格的 XR 平台。它不是手机 Snapdragon 8 系列的简单改名，而是围绕 MR/VR 头显做的专用平台：显示、渲染、视频透视、相机并发、头手眼追踪、深度感知、空间音频、低功耗 CV 和端侧 AI 都被放进同一个平台叙事里。

Qualcomm 的[官方发布稿](https://www.qualcomm.com/news/releases/2023/09/qualcomm-launches-its-next-generation-xr-and-ar-platforms--enabl)把它定位为 next-generation MR / VR devices 的 spatial computing platform，并说明首批商业化设备来自 Meta：Meta Quest 3 使用 Snapdragon XR2 Gen 2，Ray-Ban Meta smart glasses 使用 Snapdragon AR1。后续 Quest 3S 也使用 XR2 Gen 2。这个定位说明 Qualcomm 卖的不是裸芯片爱好者板卡，而是给头显 OEM 和平台厂商做整机方案的核心计算平台。

看手机芯片时，我们习惯问 CPU 大核、小核、GPU、AI TOPS；看 XR 芯片时，还必须问显示 pipeline、摄像头输入、低延迟透视、sensor fusion、time-warp、空间音频和无线连接。XR 里的 SoC 更像“空间计算中枢”：一边负责用户看到的虚拟世界，一边持续理解真实世界。

这也解释了为什么 Product Brief 很少把重点放在单个 CPU benchmark 上。XR 设备最怕的不是某个单点任务慢一点，而是整个感知-渲染闭环断掉：用户转头之后，传感器要读数，相机要输入，SLAM 和追踪要更新，渲染要补偿，显示要刷新，中间还可能有手势、语音、场景理解和网络同步。只要链路里某一段持续抖动，用户感受到的就是延迟、眩晕或交互不跟手。

![Snapdragon XR2 Gen 2 Product Brief page 1](../assets/images/thinking-exploration/qualcomm-xr2-gen2/xr2-gen2-product-brief-page-1.png){ width="720" .atlas-figure-page }

<small>图源：`Snapdragon XR2 Gen 2 Platform Product Brief (87-73689-1 Rev A)`，page 1。该页给出平台定位：next-generation MR and VR、on-device AI、3K-by-3K display、2.5x GPU、8x AI、10 concurrent cameras、12ms video see-through latency、Wi-Fi 7 / Wi-Fi 6E 等。</small>

!!! info "这张图真正想表达什么"
    这页 Product Brief 不是在说“XR2 Gen 2 只有 AI 很强”，而是在讲一颗头显 SoC 必须同时处理几条链路。`3K-by-3K display` 和 `2.5x GPU` 指向双眼高分辨率渲染；`10 concurrent cameras` 和 `12ms video see-through latency` 指向 MR 透视、追踪和空间定位；`Wi-Fi 7 / Wi-Fi 6E` 则服务 PC 串流、多人同步和高带宽内容。`8x AI` 可以当作方向性信号带过，真正做模型时仍要回到 runtime、layer placement、内存和持续功耗。换成一句通俗的话：XR2 Gen 2 卖的不是单个跑分，而是把显示、相机、渲染、AI 和无线都塞进一台电池供电头显里，并尽量让它们同时稳定运行。

## 放进头显后，它到底在忙什么

XR2 Gen 2 的核心场景是**一体式 MR/VR 头显**，特别是无需外接电池包、无需外接 PC 的 standalone headset。为了把规格和真实体验连起来，可以继续用开头那个画面：用户戴着头显打开彩色透视，低头看桌面，抬手点虚拟按钮，系统同时识别手、桌面、墙面和障碍物。这个场景看起来只是一个普通交互，但底层已经把相机、ISP、CV、NPU、GPU、显示和无线都拉进同一个闭环。

拆成工作负载，大致是这样：

| Workload | Why it matters in XR |
| --- | --- |
| Stereo rendering | 两只眼睛都要高帧率、低抖动显示，GPU 负载持续存在 |
| Time-warp / space-warp | 降低头动到显示的感知延迟，缓解眩晕 |
| Video see-through | 彩色透视把现实世界送进头显，是 MR 的入口 |
| 6DoF tracking | 头部、控制器、手势和空间定位都依赖低功耗感知 |
| Eye / hand / face tracking | 支撑交互、注视点渲染、表情和社交 avatar |
| Depth / 3D reconstruction | 支撑现实世界几何理解、遮挡和空间放置 |
| On-device AI | 让追踪、语音、视觉理解和应用模型尽量不依赖云端 |

这和手机 AI 最大的差别是：XR 的 AI 不是偶尔跑一次分类模型，而是和显示、相机、SLAM、渲染一起持续运行。硬件瓶颈不是单个 TOPS 数字，而是多路传感器、NPU/GPU/CPU 调度、内存带宽、散热和电池之间的并发平衡。

可以把一个 MR 头显的在线循环想成下面这样：

```text
传感器 / 相机输入
  -> ISP / CV / SLAM / tracking
  -> 应用逻辑和 AI 模型
  -> GPU 渲染
  -> time-warp / display scanout
  -> 用户移动和交互反馈
```

这个循环每秒要发生很多次。首先，相机和 ISP 持续处理外界画面：`2x IFE for 12 MP @ 90 FPS Bayer for video see-through` 对应彩色透视，`8x IFE-Lite for 720P @ 120 FPS mono for perception` 更接近 tracking、perception camera 和低功耗感知输入。它们不是偶尔拍一张照片，而是连续流式输入。

接着，CV / Sensor Hub / Hexagon 这类模块会参与姿态估计、手势、深度、空间定位或轻量 AI 模型。GPU 同时负责双目渲染、time-warp 和 UI 合成。无线模块可能还在处理串流、同步或联网内容。任何一个环节抖动，用户感受到的都不是“模型慢了一点”，而是画面跟手性变差、透视延迟增加、手势识别卡顿，或者整机开始降频。

所以 XR2 Gen 2 不能只按“AI 加速器”阅读。传统 AI 部署只要关心“模型一次推理多少毫秒”，XR 部署还要关心“这次推理是否抢了 GPU、是否挤占相机处理、是否造成下一帧 display miss”。模型部署的价值，是在这套并发系统里找到稳定、可控、能长期运行的位置。

## 硬件规格该怎么读

下面这张图是 Product Brief 的完整规格页，适合直接查官方参数。

![Snapdragon XR2 Gen 2 specifications and features](../assets/images/thinking-exploration/qualcomm-xr2-gen2/xr2-gen2-product-brief-page-2.png){ width="720" .atlas-figure-page }

<small>图源：`Snapdragon XR2 Gen 2 Platform Product Brief (87-73689-1 Rev A)`，page 2。完整官方字段保留在原图里；正文只提炼模型部署最相关的部分。</small>

这张规格页最值得看的是显示和相机：10 路并发相机、2x IFE 视频透视、8x IFE-Lite 感知、每眼 3.1K x 3.1K @ 90 FPS、time-warp 2.8K x 3K @ 90 FPS。它说明 XR2 Gen 2 是多传感器低延迟闭环平台，不是单纯“AI 加速器”。

从模型部署视角，要把每个模型放回系统分工里读：

| Hardware block | Role in an XR product | Practical question |
| --- | --- | --- |
| Adreno GPU | stereo rendering、time-warp、空间重投影和部分图形后处理 | 应用画质提高时，是否还能保住 90 FPS 和稳定帧时间 |
| Hexagon / NPU / DSP | 分类、检测、语音、tracking 辅助、轻量感知模型 | 模型是否能被 QNN / LiteRT / ONNX Runtime 放到 NPU 上 |
| Spectra ISP + IFE | video see-through、perception camera 输入和图像质量处理 | 多路相机同时开时，延迟、带宽和功耗是否可控 |
| Sensor Hub / CV hardware | 低功耗姿态、手眼头追踪和视觉分析 | 哪些任务可以脱离大 CPU/GPU 持续运行 |
| FastConnect 7800 | PC-to-HMD、多人同步、Wi-Fi 7 / Wi-Fi 6E 连接 | 无线链路是否影响云串流、多人互动和边缘协同 |
| LP-DDR5 / LLC | 多路视频、AI tensor、渲染资源和系统服务共享内存 | 峰值内存和带宽是否成为模型部署瓶颈 |

!!! note "术语框：Hexagon / HTP / NPU"
    Qualcomm 文档里经常同时出现 `Hexagon`、`HTP` 和 `NPU`。可以粗略理解为：`Hexagon` 是 Qualcomm SoC 内承载 DSP / AI 计算的一组处理器与加速单元名称，`HTP` 更偏 Hexagon Tensor Processor 这类张量加速后端，`NPU` 则常作为 AI Hub profile 里“神经网络加速路径”的计算单元口径。工程上不要把 `NPU` 想成一块独立显卡，而应看模型的哪些 layer 最终被 runtime 放到了 Qualcomm 的 AI 加速后端上。

真正做模型时，应该追问三件事：输入来自哪条传感器或应用管线，输出喂给哪个闭环任务，推理和 GPU、ISP、CPU、内存带宽、热预算怎么共享。只有这些说清楚，AI Hub 上的延迟数字才有工程意义。

## 价格和功耗别直接猜

公开资料里没有 Qualcomm 给出的裸 SoC 价格，也没有芯片级 TDP。这类平台通常按 OEM / ODM 商务协议、整机方案、SDK 支持和量产规模报价，不能像消费级显卡一样查公开零售价。

| Question | Public answer | Engineering reading |
| --- | --- | --- |
| Chip price | Not publicly listed by Qualcomm | 需要按 OEM 项目询价，不能从公开网页得到准确裸片价格 |
| End-device price | Meta [Quest 3 launch article](https://about.fb.com/news/2023/09/meet-meta-quest-3-mixed-reality-headset/) 中，Quest 3 发布时 128GB 版本 starting at `$499.99`，512GB 版本为 `$649.99` | 说明 XR2 Gen 2 覆盖的是几百美元级消费头显，不是低价 MCU 或纯开发板市场 |
| Power | Qualcomm [Product Brief](https://docs.qualcomm.com/doc/87-73689-1/) 未公开芯片 TDP；官方只写 built for premium standalone headsets，并给出 similar visuals with up to 50% power savings 的产品口径 | 项目里不要写“XR2 Gen 2 = x W”。更稳妥的说法是电池供电 standalone headset 级功耗预算，最终取决于显示亮度、相机、渲染、AI、无线和散热 |

工程上更可执行的做法是建立整机功耗账本：空场景渲染、video see-through、手势追踪、AI 模型、Wi-Fi、录屏或串流分别测电流、温升和帧时间。模型工程师不只看平均 latency，还要看模型是否拉高功耗档位、长期占用 GPU/NPU 或触发降频。

## 谁会买这类芯片

主要客户不是个人开发者，而是 XR 头显 OEM / ODM、平台型客户，以及培训、工业巡检、医疗教育、远程协作等垂直行业设备商。

个人开发者一般不是直接买 XR2 Gen 2 裸片，而是通过具体头显、开发套件、AI Hub Workbench、AI Hub Models、QIDK 和 Qualcomm AI Runtime 进入生态。

## 软件栈先看全景

Qualcomm 端侧 AI 软件栈可以按云端工具链和设备端运行时拆开：

![AI Hub Workbench how it works](../assets/images/thinking-exploration/qualcomm-xr2-gen2/ai-hub-how-it-works.png){ width="920" }

<small>图源：Qualcomm AI Hub Workbench `How-It-Works.png`，展示 upload -> optimize -> validate -> deploy 的 device-in-the-loop 流程。</small>

| Layer | Qualcomm software | Main function |
| --- | --- | --- |
| Model source | PyTorch, ONNX, TensorFlow via ONNX, AIMET quantized models | 输入训练好的模型，通常需要固定 shape 和清晰的 input / output spec |
| Cloud workflow | Qualcomm AI Hub Workbench | 编译、量化、profile、on-device inference、生成性能报告和部署资产 |
| Model catalog | Qualcomm AI Hub Models | 开源模型适配、导出脚本、性能/数值 scorecard、示例 demo |
| Quantization | AIMET / AIMET-ONNX / AI Hub Quantize Job | PTQ、校准、生成 QDQ / encodings 等量化资产 |
| Runtime | Qualcomm AI Engine Direct / QNN | Qualcomm NPU/HTP 的核心运行时，可生成 QNN DLC、context binary、precompiled QNN ONNX |
| Portable runtime | LiteRT / TensorFlow Lite | Android / Linux 上常用，适合移动端 App 集成 |
| Portable runtime | ONNX Runtime + QNN Execution Provider | Android / Linux / Windows 上通过 ONNX Runtime 调用 QNN EP |
| Local examples | QIDK | SNPE/QNN/AIMET/Android 示例、模型转换、量化、部署和端上 demo |

!!! info "术语框：QNN / Qualcomm AI Engine Direct"
    `QNN` 是 Qualcomm 面向端侧 AI 加速器的底层运行时和图执行接口。常见资产里，`QNN DLC` 更偏硬件无关，适合后续 link / finalize；`QNN context binary` 面向特定 SoC，设备特异性更强，也更接近最终部署形态。

!!! tip "术语框：LiteRT、ONNX Runtime 和 QNN EP"
    `LiteRT` 适合 Android 移动端快速集成。`ONNX Runtime + QNN Execution Provider` 更适合跨平台项目，通过 QNN SDK / QAIRT 把 ONNX 图交给 Qualcomm CPU / GPU / HTP 后端执行。

Workbench 里常见 job 可以对应到研发流程：

| Job / step | Input | Output | Why it matters |
| --- | --- | --- | --- |
| Compile job | PyTorch / ONNX / AIMET quantized model | LiteRT、ONNX、QNN DLC、context binary 等部署资产 | 把训练框架里的图转换成设备 runtime 能执行的图 |
| Quantize job | optimized ONNX + calibration data | quantized ONNX / QDQ-style asset | 用代表性数据确定 scale / zero point，降低模型大小和提升 NPU 友好度 |
| Profile job | compiled model + target device | latency、load time、compute unit utilization、memory 等 | 判断模型是否满足帧预算和是否落到 NPU |
| Inference job | compiled model + sample inputs | on-device outputs | 和本地 PyTorch / ONNX 输出对齐，检查数值差异 |
| Download artifacts | finished jobs | model files + metadata | 进入 App / SDK 集成阶段 |

注意边界：Quantize job 不等于精度合格，Profile job 不等于整机稳定，Compile job 成功也不等于 App pipeline 已对齐。

## 一个模型怎样走到头显里

把工具链串起来，一个务实的 XR AI 流程是：

| Step | What to do | Exit criterion |
| --- | --- | --- |
| Model cleanup | 明确 input / output、shape、layout、预处理、后处理和时间窗口 | 本地 PyTorch / ONNX 输出可复现 |
| Cloud compile/profile | 选择 `QCS8450 (Proxy)` / `XR2 Gen 2 (Proxy)`，跑 compile 和 profile | 看到 NPU placement、latency、峰值内存和失败算子 |
| Quantization | 准备代表性 calibration data，跑 PTQ/QDQ，再重新 compile/profile | 同时通过 numerics 与性能检查 |
| Local integration | 接入 Android / native library / camera buffer / runtime init / thread scheduling | 真机能在 App pipeline 里持续推理 |
| System validation | 覆盖冷启动、连续运行、相机+渲染并发、无线高负载、升温降频、不同光照和用户动作 | 闭环体验稳定，而不只是单个 job passed |

AI Hub 提速前两三步；QIDK 和 QNN sample 帮助后两步落成本地代码。

## Proxy 数字该怎么用

AI Hub [Release Notes](https://workbench.aihub.qualcomm.com/docs/hub/release_notes.html) 在 2024 年 4 月 22 日写到：Added QCS8450 Proxy devices。这里直接引用前面的结论：`QCS8450 (Proxy)` 是早筛口径，不是产品 KPI。Proxy 的价值在于让开发者在拿到真实头显之前，先判断模型兼容性、生成部署资产、跑早期 profile。

| Proxy item | Source | Meaning |
| --- | --- | --- |
| QCS8450 Proxy devices | [AI Hub Release Notes, 2024-04-22](https://workbench.aihub.qualcomm.com/docs/hub/release_notes.html) | Workbench 添加了 QCS8450 的 proxy target |
| QCS8450 XR Platform | [AI Hub Models README](https://github.com/qualcomm/ai-hub-models) | XR chipsets 列表中的平台名 |
| XR2 Gen 2 (Proxy) / QCS8450 (Proxy) | [VIT model page](https://aihub.qualcomm.com/models/vit) / [`perf.yaml`](https://github.com/qualcomm/ai-hub-models/blob/main/src/qai_hub_models/models/vit/perf.yaml) | 模型页面和 profile 数据里的 XR proxy device |
| `qualcomm-qcs8450-proxy` -> `htp_version: 69` | [`devices_and_chipsets.yaml`](https://github.com/qualcomm/ai-hub-models/blob/main/src/qai_hub_models/devices_and_chipsets.yaml) | 和 `sm8450` / Snapdragon 8 Gen 1 同为 HTP v69、SoC model 36 |

实际工程里可以这样用：早期模型选型看 `QCS8450 (Proxy)` / `XR2 Gen 2 (Proxy)` 是否支持；性能预算阶段看 proxy profile；产品定版前回到真实头显做完整 profile、热稳定和准确率回归。

后面看 ViT 量化时要带着这个前提：本文讨论的是 QCS8450 / SM8450 这一代 v69 后端上的工程取舍，不把 Qualcomm 最新 NPU 能力直接投射回 XR2 Gen 2。

## 用 ViT 例子看量化部署

这里用 AI Hub Models 里的 VIT 做观察窗口。它不是 MobileNet 这类天然移动友好的 CNN，而是 86.6M 参数的 Vision Transformer，包含 patch embedding、attention、LayerNorm、Softmax、MLP 和大量 MatMul。它不等同于 LLM，但能暴露端侧 Transformer workload 的共同问题：模型包大小、矩阵乘法效率、激活范围、NPU placement、前后处理一致性和 runtime 差异。

VIT 页面和开源仓库给出的技术信息如下，主要来自 Qualcomm AI Hub 的 [VIT model page](https://aihub.qualcomm.com/models/vit) 和模型仓库中的 [`code-gen.yaml`](https://github.com/qualcomm/ai-hub-models/blob/main/src/qai_hub_models/models/vit/code-gen.yaml)：

| Field | Value |
| --- | --- |
| Model checkpoint | Imagenet |
| Input resolution | 224x224 |
| Number of parameters | 86.6M |
| Model size (float) | 330 MB |
| Model size (w8a16) | 86.2 MB |
| Model size (w8a8) | 83.2 MB |
| Numerics benchmark | Top-1 Accuracy, 81.1%, ImageNet |
| Supported form factors | Phone, Tablet, IoT, XR |
| Supported precisions in code-gen.yaml | float, w8a8_mixed_int16, w8a16, w8a8 |

AI Hub Workbench 的 [Quantization 文档](https://workbench.aihub.qualcomm.com/docs/hub/quantize_examples.html) 给出的 PTQ 路径可以简化成：

!!! note "术语框：AIMET、PTQ 和 QDQ"
    `AIMET` 是 Qualcomm 的模型效率工具，常用于量化、压缩和量化精度分析。`PTQ` 是 post-training quantization。`QDQ` 是 ONNX 常见量化表达，用 `QuantizeLinear / DequantizeLinear` 标出量化边界；AI Hub quantize job 输出的 quantized ONNX 属于这种 fake quantization / QDQ 风格资产。

```text
PyTorch model
  -> trace / export
  -> compile to optimized ONNX
  -> collect representative calibration data
  -> submit_quantize_job()
  -> get quantized ONNX in QDQ / fake quantization representation
  -> compile to TFLite, QNN, or ONNX
  -> profile / inference on target device
```

关键点是先把图变成更接近部署形态的 ONNX，再做校准和量化。部署图会经历算子融合、常量折叠、shape 固化、layout 调整和 runtime 适配；如果直接在原始训练图上插 Q/DQ，后续编译可能改变图结构，让校准假设和真实执行路径错位。校准数据也要贴近线上输入，随机样本最多用于快速估计 latency，不适合判断 accuracy。

AI Hub [Quantization 文档](https://workbench.aihub.qualcomm.com/docs/hub/quantize_examples.html) 的量化精度支持表如下，保留英文原格式：

|  | Weights | Activations | Mixed Precision* |
| --- | --- | --- | --- |
| TFLite | int8 | int8 | Not supported |
| QNN | int8 | int8, int16 | Not supported |
| ONNX | int8 | int8, int16 | Not supported |

<small>表源：Qualcomm AI Hub Workbench `Quantization` 文档。文档同时说明 long-term 目标是支持 mixed precision，以及 int4 / int8 / int16 weights and activations。</small>

对 ViT / QCS8450，量化决策可以压成四个检查点：

| Decision | How to read it |
| --- | --- |
| Calibration | 用真实图像分布、真实 resize/crop/normalize/layout；随机输入不适合做最终精度判断 |
| Precision | `w8a8` 压缩最明确，但 attention / MLP 激活范围更敏感 |
| HTP version | QCS8450 / SM8450 在 AI Hub Models 元数据里是 HTP v69；本文把 `w8a8` 作为可靠量化路径 |
| Runtime | TFLite、QNN DLC、ONNX Runtime QNN EP 的 layer placement、峰值内存和 latency 可能不同 |

!!! warning "QCS8450 上的 w8a16 口径"
    要区分“模型库声明支持某种 precision”和“某个目标 SoC / runtime path 真正适合用它”。ViT 配置里列出 `w8a16`，但 QCS8450 Proxy / SM8450 在 AI Hub Models 设备元数据里是 HTP v69；本文按实验口径处理为：QCS8450 上优先比较 `float` 和 `w8a8`，不要把 v73+ 路径直接套回 XR2 Gen 2。

开源仓库中的 VIT [`perf.yaml`](https://github.com/qualcomm/ai-hub-models/blob/main/src/qai_hub_models/models/vit/perf.yaml) 给出 QCS8450 Proxy 上的部分 profile 数字：

| Precision | Runtime | Device | Job status | Inference time milliseconds | Estimated peak memory range MB | Primary compute unit | Layer counts |
| --- | --- | --- | --- | ---: | --- | --- | --- |
| float | tflite | QCS8450 (Proxy) | Passed | 12.966 | 0-290 | NPU | total 1399, npu 1399 |
| float | qnn_dlc | QCS8450 (Proxy) | Passed | 13.207 | 0-229 | NPU | total 888, npu 888 |
| w8a8 | tflite | QCS8450 (Proxy) | Passed | 20.594 | 0-447 | NPU | total 1451, npu 1451 |
| w8a8 | qnn_dlc | QCS8450 (Proxy) | Passed | 13.179 | 0-310 | NPU | total 889, npu 889 |

<small>数据源：`qualcomm/ai-hub-models` 中 VIT 的 [`perf.yaml`](https://github.com/qualcomm/ai-hub-models/blob/main/src/qai_hub_models/models/vit/perf.yaml)。当前仓库文件的 QCS8450 Proxy 片段来自 profile job 记录，适合做方向判断；但它仍然是 proxy 数字，而且同一文件里不同设备、runtime、tool version 的内存区间有明显波动，所以我会把它当成“需要复现实验的参考值”，不是产品验收值。</small>

这组数字说明两件事：量化不必然让所有 runtime 都更快，`w8a8 tflite` 在这个 profile 里慢于 float；但 QNN DLC 的 `w8a8` 和 float 延迟接近，同时模型尺寸从 330 MB 降到 83.2 MB，部署包和权重带宽压力明显改善。XR 项目里要把 latency、memory、model size、accuracy、runtime 集成成本放在一起看，不能只看“是否 INT8”。

`w8a8` 变慢通常先查这几类原因：

| Suspect | What to inspect |
| --- | --- |
| Q/DQ overhead | 量化、反量化、requantization、scale / zero point 是否占用了图执行时间 |
| Shape and packing | ViT 里的小 MatMul、reshape、transpose、softmax、LayerNorm 是否让 INT8 kernel 利用率下降 |
| Fusion boundary | Q/DQ 是否切断原本可融合的 pattern；`w8a8 tflite` 的 layer count 从 `1399` 增到 `1451` 是一个信号 |
| Runtime path | 同样是 `w8a8`，TFLite 为 `20.594 ms`，QNN DLC 为 `13.179 ms`，说明 backend 和 kernel 选择很关键 |

激活函数也值得查。ViT MLP 常见 `Linear -> GELU -> Linear`，ONNX 里 `GELU` 可能展开成 `Erf / Tanh / Add / Mul / Div` 子图。若业务精度允许，可在训练或微调阶段尝试 ReLU / ReLU-like 激活，再导出 ONNX 对比 profile。AI Hub Models 的 AIMET 默认配置中，[v69](https://github.com/qualcomm/ai-hub-models/blob/main/src/qai_hub_models/utils/aimet/default_per_tensor_config_v69.json) / [v73](https://github.com/qualcomm/ai-hub-models/blob/main/src/qai_hub_models/utils/aimet/default_per_tensor_config_v73.json) per-tensor config 都把 `["Gemm", "Relu"]` 放在 `supergroups` 中，但没有等价的 `Gemm + Gelu` supergroup。这个配置不是完整 QNN 编译规则，却足以提醒我们：`Gemm -> Relu` 更接近后端常见融合模式。

!!! tip "用 Netron 看什么"
    打开导出的 ONNX，先看是否有 `MatMul/Gemm -> Erf/Tanh/Mul/Add` 这类 GELU 展开链路，再看 Q/DQ 是否切在 MatMul、LayerNorm、Softmax 前后。真正上线前要用 accuracy 对照确认替换激活不会伤害下游任务，再用 layer timing 确认加速来自激活和融合。

最后，把 ViT 量化落实成一张上线检查表：

| Checkpoint | What to verify | Why it matters |
| --- | --- | --- |
| Input pipeline | resize、crop、normalization、layout 和训练 / 评测一致 | 前处理不一致会把量化误差和数据偏移混在一起 |
| Calibration data | 样本覆盖实际图像分布，不用纯随机输入做最终校准 | scale / zero point 依赖激活范围，数据越偏，精度越不稳 |
| Numerics | float vs quantized 输出差异、Top-1 / Top-5 或任务指标 | 模型变小不代表业务可用 |
| Runtime placement | 是否全层 NPU，是否有 CPU fallback | fallback 会带来延迟抖动和功耗问题 |
| Operator pattern | `GELU` 是否展开成慢子图，`MatMul/Gemm + ReLU` 是否能形成融合 | Transformer MLP block 里激活和融合会直接影响 v69 HTP 上的延迟 |
| HTP version gate | QCS8450 / SM8450 是 v69，v73+ 才稳定启用的路径不要直接套用 | precision 和 runtime 能力常常跟 HTP 代际绑定 |
| Memory | 模型大小、峰值内存、buffer 生命周期 | XR 里相机、渲染和模型共享内存资源 |
| Thermal loop | 连续运行后的 latency 和帧时间 | 单次 profile 不能代表佩戴场景下的长期表现 |

这张表的意义在于提醒自己：量化不是一个孤立算法步骤，而是部署链路中的一次系统级折中。对 XR 来说，精度、延迟、内存、热稳定和用户体验是一组联动指标。

## QIDK 在栈里的位置

QIDK 更像本地开发和样例仓库，而不是 AI Hub 的替代品。README 里写得很清楚，它提供的是展示 AI 硬件加速器和软件 AI stack 能力的 sample applications，内容覆盖：

| QIDK component | What it is used for |
| --- | --- |
| Qualcomm® Neural Processing SDK for AI / SNPE | 老一些或已有项目里的 DLC 转换、Android 端推理和 Java / C++ 示例 |
| Qualcomm® AI Engine Direct SDK / QNN | 更底层、更直接面向 Qualcomm NPU / HTP 的运行时和模型构建路径 |
| AIMET | 量化、精度分析和模型效率工具 |
| AIMET Model Zoo | 已适配或可作为参考的模型 zoo |
| Android sample apps | 把模型、前后处理、相机输入、UI 和设备运行时接起来 |
| Model Enablement | 不支持算子替换、混合精度、量化精度损失调试 |

如果 AI Hub Workbench 是“云端把模型编译和测试出来”，QIDK 就是“本地怎么理解 SDK、怎么在 Android App 里接运行时、怎么处理模型转换和量化问题”。对于 XR 项目来说，两者最好配合使用：AI Hub 做快速 scorecard，QIDK / QAIRT / QNN 做本地集成和端上验证。

一个更贴近工程的分工是：AI Hub 适合回答“模型资产能不能生成、理论上跑在哪、延迟大概多少”；QIDK 适合回答“我怎么把这个资产接到 Android / native App、怎么准备输入输出、怎么拉起 runtime、怎么处理 SDK 环境和设备推理”。前者让模型工程师快，后者让应用工程师和系统工程师能真正落地。

如果团队里有模型、App、系统三类角色，分工可以很清楚：模型同学负责导出、校准、数值回归和模型替换；App 同学负责把 runtime 资产接到相机、UI 和业务流程；系统同学负责 profiling、功耗、线程调度、热稳定和设备差异。Qualcomm 的工具链把这些角色连接起来，但不会自动替团队做这些边界判断。

## 真正落地时怎么选 runtime

不同 runtime 的选择不是高低之分，而是工程约束不同：

| Runtime asset | Best fit | Tradeoff |
| --- | --- | --- |
| LiteRT / TFLite | Android App 快速集成，移动端团队熟悉 | 可移植性好，但复杂模型的 NPU 映射和调优空间有限 |
| ONNX Runtime | Windows / Linux / Android 统一接口，便于跨平台 | 需要关注 QNN Execution Provider、预编译模型和部署包结构 |
| QNN DLC | 更贴近 Qualcomm AI Engine Direct，适合 NPU 优化 | 集成门槛更高，需要理解 QNN 工具链 |
| QNN context binary / precompiled QNN ONNX | 面向特定 SoC 的 AOT 编译部署 | 设备特异性强，尤其要按前文 Proxy 口径看待 profile |

XR 场景通常更在意稳定延迟和热稳定，而不是一次性 benchmark 的最低值。因为头显里同时有渲染、sensor fusion、video see-through 和后台系统服务，模型推理不能抢走过多内存带宽或导致持续降频。

更稳的判断是：demo 阶段优先选最容易集成的路径，产品阶段优先选可观测、可复现、可控的路径。比如 LiteRT 可能让 Android 团队上手最快；ONNX Runtime 适合跨 OS 统一接口；QNN DLC / context binary 更贴近 NPU 性能和 SoC 特化，但也要求团队能处理更复杂的部署、版本和调试问题。没有一个 runtime 永远正确，只有和项目阶段匹配的 runtime。

因此，runtime 选择最好不要在项目开始时一次性定死。早期可以用 AI Hub 和熟悉的 runtime 快速跑通；中期比较 LiteRT、ONNX Runtime QNN EP 和 QNN DLC 的延迟、内存、集成成本；后期再根据真实头显 profile 选择最稳定的路径。这里的“稳定”不只是平均延迟低，还包括版本可控、日志可读、失败可复现、资产更新流程清楚。

## 项目启发

把 XR2 Gen 2 / QCS8450 用到项目里，建议按下面顺序推进：

1. **先明确 workload**：分类、检测、分割、手势、语音、SLAM 辅助、生成式模型，对 runtime 和功耗的要求完全不同。
2. **先用 AI Hub Models 找相似模型**：如果 VIT、MobileNet、YOLO、SAM、Whisper、LLM 等已有模型能跑，就先参考它的 `export.py`、`perf.yaml` 和 `numerics.yaml`。
3. **再用 Workbench 做 proxy profile**：看目标设备是否支持、哪些层落在 NPU、延迟和峰值内存是否可接受。
4. **量化前先看 HTP 版本**：QCS8450 / SM8450 是 HTP v69；本文的 ViT 实验按 `w8a16` 需要 HTP v73+ 的口径处理，所以 QCS8450 上先把 `w8a8` 跑扎实，再把 `w8a16` 留给 QCS8550、8 Gen 2 或更新平台验证。
5. **用图可视化找融合机会**：Netron 里重点看 `GELU` 是否被展开、`MatMul/Gemm + ReLU` 是否能形成更干净的 pattern，再用 QNN / runtime profile 确认收益。
6. **最后做整机闭环**：在目标头显上跑摄像头、渲染、网络、热稳定和用户交互，把早期 profile 变成产品级判断。

收口来看，我会强调三个判断。第一，XR2 Gen 2 是 XR 产品平台，不是单纯的 AI 加速芯片；讨论它时要把显示、相机、感知、渲染和无线一起看。第二，QCS8450 / XR2 Gen 2 Proxy 是很有用的开发入口，主要解决提前筛选和生成资产的问题。第三，Qualcomm 的软件栈越来越像一个从模型库到云端编译、再到设备 runtime 的完整工作流；真正的难点不再只是把模型转成某个文件，而是把模型放进 XR 整机的持续并发预算里。

## 参考资料

1. [Qualcomm AI Hub Workbench Documentation](https://workbench.aihub.qualcomm.com/docs/)
2. [Qualcomm AI Hub](https://aihub.qualcomm.com/)
3. [AI Hub Release Notes](https://workbench.aihub.qualcomm.com/docs/hub/release_notes.html)
4. [Snapdragon XR2 Gen 2 Platform Product Brief](https://docs.qualcomm.com/doc/87-73689-1/)
5. [VIT - Qualcomm AI Hub](https://aihub.qualcomm.com/models/vit)
6. [Qualcomm AI Hub Models GitHub](https://github.com/qualcomm/ai-hub-models)
7. [AI Hub Models `devices_and_chipsets.yaml`](https://github.com/qualcomm/ai-hub-models/blob/main/src/qai_hub_models/devices_and_chipsets.yaml)
8. [AI Hub Models VIT `perf.yaml`](https://github.com/qualcomm/ai-hub-models/blob/main/src/qai_hub_models/models/vit/perf.yaml)
9. [AI Hub Models AIMET v69 config](https://github.com/qualcomm/ai-hub-models/blob/main/src/qai_hub_models/utils/aimet/default_per_tensor_config_v69.json)
10. [QIDK GitHub](https://github.com/quic/qidk)
11. [Netron](https://netron.app/)
12. [Qualcomm launch release: Snapdragon XR2 Gen 2 and AR1 Gen 1](https://www.qualcomm.com/news/releases/2023/09/qualcomm-launches-its-next-generation-xr-and-ar-platforms--enabl)
13. [Meta Quest 3 launch article](https://about.fb.com/news/2023/09/meet-meta-quest-3-mixed-reality-headset/)
