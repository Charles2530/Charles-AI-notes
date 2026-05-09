# 从 QCS8450 到 AI Hub：通俗拆解 Qualcomm XR2 Gen 2 与端侧量化部署

如果你只是在产品页上看过 `Snapdragon XR2 Gen 2`，再打开 Qualcomm AI Hub，很容易被 `QCS8450`、`XR2 Gen 2 (Proxy)`、`QNN`、`AIMET`、`LiteRT` 这些名字绕住。它们不是同一个东西的不同叫法，而是同一条链路上的不同角色：芯片平台、云端目标设备、模型编译服务、本地 runtime 和量化工具。

这篇文章想回答一个更工程化的问题：**如果一个视觉模型要跑进 XR 头显，它会怎样从 PyTorch / ONNX 模型，变成 Qualcomm SoC 上的 NPU / HTP 推理任务？**我们会先把名字讲清楚，再看 XR2 Gen 2 为什么面向头显设计，最后用 ViT 量化作为例子，讨论 QCS8450 上 `w8a8`、`w8a16`、GELU/ReLU 融合和 proxy profile 这些更贴近落地的问题。

读完这篇，至少应该能分清三件事：

1. `XR2 Gen 2` 是面向 MR/VR 头显的产品平台，不只是一个 AI 加速器。
2. `QCS8450 / XR2 Gen 2 (Proxy)` 是 AI Hub 里的目标设备口径，适合早筛模型，但不能代替真机验收。
3. `AI Hub + QNN + AIMET + QIDK` 共同组成从模型到端侧部署的工具链，量化是否更快要看 HTP 代际、runtime 和模型图。

## 先给一个总览

先用一条主线把全文串起来：Qualcomm 在 XR 里卖的是一套“头显计算平台”，AI Hub 则是开发者把模型塞进这套平台的入口。前者回答“芯片要支撑什么体验”，后者回答“模型怎样变成可部署资产”。

如果只记一张图，可以把 Qualcomm 这套东西理解成下面这条链：

```text
训练好的 PyTorch / ONNX 模型
  -> AI Hub Workbench 做 compile / quantize / profile / inference
  -> 生成 LiteRT、ONNX Runtime 或 QNN 部署资产
  -> 跑到 Snapdragon XR2 Gen 2 / QCS8450 这类 XR 设备的 NPU、GPU、CPU 上
  -> 服务头显里的渲染、感知、语音、手眼追踪、深度和应用 AI
```

这条链路里有两个关节点。第一个关节点是 **Workbench**：它把一个训练好的模型变成某个目标设备能跑的部署资产，并返回延迟、内存、layer placement 和数值结果。第二个关节点是 **真实设备验证**：XR 头显不是离线 benchmark 机器，模型推理会和显示、摄像头、SLAM、无线、音频和应用渲染争同一个功耗与散热空间。因此，云端 profile 能帮你早筛，但不能替代整机闭环。

这也是后面所有判断的主线：公开文档能让我们知道平台能力边界，AI Hub 能让我们提前筛选模型和 runtime，QIDK / QNN 能帮助工程集成；但最终产品是否成立，仍然要看真实头显里的连续运行、热稳定、帧时间和交互延迟。

## 先把名字捋顺

读这类资料时，最容易卡住的不是技术本身，而是名字。产品页讲的是 Snapdragon XR2 Gen 2，模型库里出现的是 QCS8450 / XR2 Gen 2 Proxy，SDK 文档又在讲 QNN、LiteRT、ONNX Runtime、AIMET、QAIRT。可以先把它们放进一张表：

几个容易混淆的名字需要先分开：

| Name | What it means | How to read it |
| --- | --- | --- |
| Snapdragon XR2 Gen 2 Platform | Qualcomm 官方对外的 XR 产品平台名称 | 面向 standalone MR / VR headset 的 SoC 平台 |
| QCS8450 XR Platform | AI Hub Models README 里列出的 XR chipset 口径 | 更偏开发、scorecard、兼容性测试里的芯片目标名 |
| QCS8450 (Proxy) | AI Hub Workbench / AI Hub Models 中的 proxy device | 不是云端真实托管的 QCS8450 设备，而是替代设备估计 NPU 特性 |
| XR2 Gen 2 (Proxy) | AI Hub Models 页面上可见的 XR proxy device | 用来筛选模型兼容性和生成部署资产，实机表现仍要在真实头显验证 |

最重要的是最后两行。AI Hub 文档把 `(Proxy)` 定义为：当 Workbench 没有托管真实设备时，用替代设备服务于目标设备。它可以帮助判断模型兼容性和生成资产，但官方也提醒，真实设备的性能和准确率会因为操作系统、固件、频率、内存、散热封装等因素而变化。因此，**Proxy 数字适合做早期选型，不适合作为最终产品 KPI。**

换句话说，`QCS8450 (Proxy)` 可以回答“这个模型大概率能不能走 Qualcomm NPU 路径”“部署资产大概怎么生成”“哪些 runtime 组合值得试”。但它不能回答“我的头显戴在用户头上连续运行 30 分钟是否降频”“和相机透视、手势追踪同时跑是否还满足帧预算”。这些问题必须回到真实硬件和应用场景里验证。

## QCS8450 在芯片代际里的位置

接下来要再澄清一个关键点：AI Hub 里的 QCS8450 不是“名字里带 8450，所以一定等价于某个更新一代的 8 Gen 2 / 8 Gen 3 平台”。在 `qualcomm/ai-hub-models` 的设备元数据里，`qualcomm-qcs8450-proxy` 被标成 XR world，`htp_version: 69`，`soc_model: 36`；同一个文件里，`qualcomm-snapdragon-8gen1` / `sm8450` 也是 `htp_version: 69`、`soc_model: 36`。也就是说，从 AI Hub scorecard 的硬件后端口径看，QCS8450 Proxy 更接近 SM8450 / Snapdragon 8 Gen 1 这一代，而不是 SM8550 / Snapdragon 8 Gen 2 之后的 HTP v73+ 平台。

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

<small>表源：`qualcomm/ai-hub-models` 的 `devices_and_chipsets.yaml`。这里的 `htp_version` 和 `soc_model` 是 AI Hub 模型库用于 scorecard / device target 的工程元数据，不等同于 Qualcomm 对外产品命名。</small>

## XR2 Gen 2 到底是什么芯片

把命名和代际放好之后，再回到产品本身。XR2 Gen 2 最核心的定位不是“手机芯片换个名字”，而是专门给一体式头显做的空间计算平台。

Snapdragon XR2 Gen 2 是 Qualcomm 在 2023 年发布、2024 年 Product Brief 中继续整理规格的 XR 平台。它不是手机 Snapdragon 8 系列的简单改名，而是围绕 MR/VR 头显做的专用平台：显示、渲染、视频透视、相机并发、头手眼追踪、深度感知、空间音频、低功耗 CV 和端侧 AI 都被放进同一个平台叙事里。

官方发布稿把它定位为 next-generation MR / VR devices 的 spatial computing platform，并说明首批商业化设备来自 Meta：Meta Quest 3 使用 Snapdragon XR2 Gen 2，Ray-Ban Meta smart glasses 使用 Snapdragon AR1。后续 Quest 3S 也使用 XR2 Gen 2。这个定位说明 Qualcomm 卖的不是裸芯片爱好者板卡，而是给头显 OEM 和平台厂商做整机方案的核心计算平台。

看手机芯片时，我们习惯问 CPU 大核、小核、GPU、AI TOPS；看 XR 芯片时，还必须问显示 pipeline、摄像头输入、低延迟透视、sensor fusion、time-warp、空间音频和无线连接。XR 里的 SoC 更像“空间计算中枢”：一边负责用户看到的虚拟世界，一边持续理解真实世界。

这也解释了为什么 Product Brief 很少把重点放在单个 CPU benchmark 上。XR 设备最怕的不是某个单点任务慢一点，而是整个感知-渲染闭环断掉：用户转头之后，传感器要读数，相机要输入，SLAM 和追踪要更新，渲染要补偿，显示要刷新，中间还可能有手势、语音、场景理解和网络同步。只要链路里某一段持续抖动，用户感受到的就是延迟、眩晕或交互不跟手。

![Snapdragon XR2 Gen 2 Product Brief page 1](../assets/images/thinking-exploration/qualcomm-xr2-gen2/xr2-gen2-product-brief-page-1.png){ width="920" }

<small>图源：`Snapdragon XR2 Gen 2 Platform Product Brief (87-73689-1 Rev A)`，page 1。该页给出平台定位：next-generation MR and VR、on-device AI、3K-by-3K display、2.5x GPU、8x AI、10 concurrent cameras、12ms video see-through latency、Wi-Fi 7 / Wi-Fi 6E 等。</small>

!!! info "这张图真正想表达什么"
    这页 Product Brief 不是在说“XR2 Gen 2 只有 AI 很强”，而是在讲一颗头显 SoC 必须同时处理几条链路。`3K-by-3K display` 和 `2.5x GPU` 指向双眼高分辨率渲染；`8x AI` 指向端侧感知和应用模型；`10 concurrent cameras` 和 `12ms video see-through latency` 指向 MR 透视、追踪和空间定位；`Wi-Fi 7 / Wi-Fi 6E` 则服务 PC 串流、多人同步和高带宽内容。换成一句通俗的话：XR2 Gen 2 卖的不是单个跑分，而是把显示、相机、渲染、AI 和无线都塞进一台电池供电头显里，并尽量让它们同时稳定运行。

## 它主要面向哪些场景

XR2 Gen 2 的核心场景是**一体式 MR/VR 头显**，特别是无需外接电池包、无需外接 PC 的 standalone headset。它要同时支撑几类任务：

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

这个循环每秒要发生很多次。传统 AI 部署只要关心“模型一次推理多少毫秒”，XR 部署还要关心“这次推理是否抢了 GPU、是否挤占相机处理、是否造成下一帧 display miss”。所以 XR2 Gen 2 的技术重点是并发：多路 camera、低延迟 video see-through、视觉分析硬件、NPU/Hexagon、Adreno GPU、FastConnect 和安全/内存子系统都要一起看。

## 放进头显后，它每天在忙什么

为了把这些规格和真实体验连起来，可以想象一个很普通的 MR 场景：用户戴着一台 standalone 头显，在房间里打开彩色透视，手势选择一个虚拟面板，同时系统还要识别桌面、墙面和障碍物。这个场景听起来像一个交互 demo，但底层已经把 XR2 Gen 2 的大部分模块都调动起来了。

首先，相机和 ISP 要持续处理外界画面。`2x IFE for 12 MP @ 90 FPS Bayer for video see-through` 对应彩色透视；`8x IFE-Lite for 720P @ 120 FPS mono for perception` 更接近 tracking、perception camera 和低功耗感知输入。它们不是偶尔拍一张照片，而是连续流式输入。

接着，CV / Sensor Hub / Hexagon 这类模块会参与姿态估计、手势、深度、空间定位或轻量 AI 模型。GPU 同时负责双目渲染、time-warp 和 UI 合成。无线模块可能还在处理串流、同步或联网内容。任何一个环节抖动，用户感受到的都不是“模型慢了一点”，而是画面跟手性变差、透视延迟增加、手势识别卡顿，或者整机开始降频。

所以 XR2 Gen 2 不能只按“AI 加速器”阅读。它的价值在于把持续显示、持续感知和持续 AI 放进电池供电设备；模型部署的价值，也是在这套并发系统里找到稳定、可控、能长期运行的位置。

## 硬件规格该怎么读

下面这张图是 Product Brief 的完整规格页，适合直接查官方参数。

![Snapdragon XR2 Gen 2 specifications and features](../assets/images/thinking-exploration/qualcomm-xr2-gen2/xr2-gen2-product-brief-page-2.png){ width="920" }

<small>图源：`Snapdragon XR2 Gen 2 Platform Product Brief (87-73689-1 Rev A)`，page 2。下方表格按原英文项目重绘，便于检索。</small>

| Category | Specification |
| --- | --- |
| Artificial Intelligence | Qualcomm® Adreno™ GPU; Qualcomm® Kryo™ CPU; Qualcomm® Hexagon™ Processor; Fused AI Accelerator Architecture; Qualcomm® Hexagon™ Vector eXtensions; Qualcomm® Hexagon™ Scalar Accelerator; Qualcomm® Hexagon™ Matrix eXtensions |
| Display | On-device display support; 3.1K x 3.1K @ 90 FPS per eye; 4x DSI, 2x eDP, 1x DP1.4 over USB3 |
| CPU | Qualcomm Kryo CPU; 4 + 2 performance cores, up to 2.4/2.0 GHz |
| Camera | Qualcomm Spectra™ Image Signal Processor; Support for 10 concurrent cameras; 2x IFE for 12 MP @ 90 FPS Bayer for video see-through; 8x IFE-Lite for 720P @ 120 FPS mono for perception |
| Computer Vision | Engine for Visual Analytics with dedicated computer vision hardware |
| Video | 8K60 decode, 8K30 encode, low-latency slice-based decoding; AVC, HEVC, VP9, and AV1 decode |
| Audio | Qualcomm® Heaxagon™ DSP, embedded neural processing units, Qualcomm® Sensor Hub, Voice UI, Spatial Audio / Recording |
| Visual Subsystem | Qualcomm Adreno GPU; Render: 2.2K x 2.4K @ 90 FPS; Time-warp: 2.8K x 3K @ 90 FPS |
| Wi-Fi and Bluetooth® | Qualcomm® FastConnect™ 7800 Mobile Connectivity System; Wi-Fi Generation: Wi-Fi 7; Peak speed: 5.8 Gbps; Wi-Fi Security: WPA3-Enterprise, WPA3-Enhanced Open, WPA3 Easy Connect, WPA3-Personal; High-band simultaneous Multi-Link; Integrated Bluetooth; Bluetooth 5.3; Dual Bluetooth |
| Security | Secure Boot, Secure Debug, Trust Zone, Hardware Root of Trust, Full DDR Encryption; Secure Processor, Trusted VM |
| Memory | 4x16 LP-DDR5 memory, up to 3.2 GHz; 8 MB system cache (LLC) |
| General Specifications | Storage: UFS 3.1, SD 3.0 / SDExpress 7.0; Peripherals: 1x USB3.1, 3x PCIe |

这张表里最值得注意的不是 CPU 频率，而是相机和显示规格：10 路并发相机、2x IFE 视频透视、8x IFE-Lite 感知、每眼 3.1K x 3.1K @ 90 FPS、time-warp 2.8K x 3K @ 90 FPS。这些都说明 XR2 Gen 2 是围绕“多传感器低延迟闭环”设计的，而不是单纯堆大 CPU。

如果从模型部署视角读这张表，最该盯住的是三件事。第一，模型输入来自哪里：RGB 相机、mono tracking camera、IMU、麦克风还是应用渲染结果。第二，模型输出喂给哪里：UI、tracking、手势、空间理解、语音交互还是云边协同。第三，模型推理和哪些系统任务共享资源：NPU、GPU、CPU、内存带宽、缓存、摄像头 pipeline 和热预算。只有把这三件事说清楚，AI Hub 上的延迟数字才有工程意义。

从系统分工看，可以这样读这张表：

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

这样读会更接近真实项目。比如“8x AI”这个宣传口径只有方向意义；真正做模型时，你更应该追问：模型走哪个 runtime、哪些层在 NPU、是否出现 CPU fallback、是否需要 INT8 / INT16、峰值内存和持续温升是否可接受。

## 价格和功耗别直接猜

公开资料里没有 Qualcomm 给出的裸 SoC 价格，也没有芯片级 TDP。原因很现实：这类平台通常按 OEM / ODM 商务协议、整机方案、SDK 支持和量产规模报价，不像消费级显卡那样有公开零售价。

可以引用的公开口径有三类：

| Question | Public answer | Engineering reading |
| --- | --- | --- |
| Chip price | Not publicly listed by Qualcomm | 需要按 OEM 项目询价，不能从公开网页得到准确裸片价格 |
| End-device price | Meta 官方发布文章中，Quest 3 发布时 128GB 版本 starting at `$499.99`，512GB 版本为 `$649.99` | 说明 XR2 Gen 2 覆盖的是几百美元级消费头显，不是低价 MCU 或纯开发板市场 |
| Power | Qualcomm Product Brief 未公开芯片 TDP；官方只写 built for premium standalone headsets，并给出 similar visuals with up to 50% power savings 的产品口径 | 项目里不要写“XR2 Gen 2 = x W”。更稳妥的说法是电池供电 standalone headset 级功耗预算，最终取决于显示亮度、相机、渲染、AI、无线和散热 |

所以如果要做方案估算，建议把功耗拆成系统预算：显示和背光、SoC/GPU/NPU、相机和传感器、无线、音频、存储、散热余量。XR2 Gen 2 的价值不是给出一个固定瓦数，而是在这个预算内提供足够强的 GPU、NPU、ISP 和 CV 并发。

公开资料适合回答“平台面向哪一档产品”和“官方强调哪些功耗方向”，不适合直接回答“单颗芯片热设计功耗是多少”。Quest 3 的首发价格只能说明 XR2 Gen 2 已进入消费级头显；Product Brief 里的 power savings 口径只能说明 Qualcomm 把同等视觉效果下的功耗作为平台卖点之一，不能反推出固定瓦数。

工程上更可执行的做法是建立整机功耗账本：空场景渲染、video see-through、手势追踪、AI 模型、Wi-Fi、录屏或串流，每种模式分别测电流、温升和帧时间。模型工程师不只看平均 latency，还要看模型是否拉高功耗档位、长期占用 GPU/NPU 或触发降频。

## 谁会买这类芯片

主要客户不是个人开发者，而是三类：

1. **XR 头显 OEM / ODM**：做一体式 MR/VR 设备，需要芯片、参考设计、摄像头/显示/无线/音频接口和低层 SDK。
2. **平台型客户**：例如 Meta 这类有系统软件、内容生态和大规模出货能力的厂商，需要和 Qualcomm 一起定义平台指标。
3. **企业和垂直行业设备商**：培训、工业巡检、医疗教育、远程协作、仿真和空间计算设备，需要长期供货、软件栈和模型部署支持。

个人开发者一般不是直接买 XR2 Gen 2 裸片，而是通过具体头显、开发套件、AI Hub Workbench、AI Hub Models、QIDK 和 Qualcomm AI Runtime 进入生态。

这也解释了 Qualcomm 为什么要把 AI Hub、模型库和 SDK 做得越来越完整。真正推动平台扩散的，不只是芯片被哪家 OEM 采用，还包括开发者能不能把自己的模型顺利转成可部署资产，能不能在 Android / Linux / Windows 侧接入 runtime，能不能对量化精度、NPU placement 和性能报告做基本判断。

## 软件栈先看全景

Qualcomm 端侧 AI 软件栈可以按“云端工具链”和“设备端运行时”拆开：

![AI Hub Workbench how it works](../assets/images/thinking-exploration/qualcomm-xr2-gen2/ai-hub-how-it-works.png){ width="920" }

<small>图源：Qualcomm AI Hub Workbench 文档 `How-It-Works.png`。图中流程是 Upload model -> Auto optimize for target device -> Validate performance and numerics -> Optimized model + performance report -> Deploy，并强调 device in the loop。</small>

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
    `QNN` 可以把它理解成 Qualcomm 面向端侧 AI 加速器的底层运行时和图执行接口。它不是 PyTorch / ONNX 这种训练或交换格式，而是把已经导出的模型图变成 Qualcomm 后端可执行图的部署层。AI Hub compile 文档里常见两种 QNN 资产：`QNN DLC` 更偏硬件无关，适合后续再 link / finalize；`QNN context binary` 面向特定 SoC，设备特异性更强，但通常更接近最终部署形态。

!!! tip "术语框：LiteRT、ONNX Runtime 和 QNN EP"
    `LiteRT` 是 TensorFlow Lite 的新名称，更适合 Android 移动端团队快速集成。`ONNX Runtime` 是跨平台推理 runtime；其中 `QNN Execution Provider` 会调用 Qualcomm QNN SDK / QAIRT，把 ONNX 图交给 Qualcomm CPU / GPU / HTP 等后端执行。选择哪条路径，主要取决于 App 平台、团队熟悉度、是否需要跨 OS，以及是否愿意承担更底层的 QNN 集成成本。

AI Hub Workbench 的文档明确说，它可以把 PyTorch / ONNX 等模型转换到目标设备运行时，做硬件感知优化，并在云端设备上 profile 和 inference。AI Hub Models README 则把这套过程固化成每个模型的 `export.py`、`demo.py`、`evaluate.py` 等脚本。

更具体地说，Workbench 里常见的 job 可以对应到研发流程：

| Job / step | Input | Output | Why it matters |
| --- | --- | --- | --- |
| Compile job | PyTorch / ONNX / AIMET quantized model | LiteRT、ONNX、QNN DLC、context binary 等部署资产 | 把训练框架里的图转换成设备 runtime 能执行的图 |
| Quantize job | optimized ONNX + calibration data | quantized ONNX / QDQ-style asset | 用代表性数据确定 scale / zero point，降低模型大小和提升 NPU 友好度 |
| Profile job | compiled model + target device | latency、load time、compute unit utilization、memory 等 | 判断模型是否满足帧预算和是否落到 NPU |
| Inference job | compiled model + sample inputs | on-device outputs | 和本地 PyTorch / ONNX 输出对齐，检查数值差异 |
| Download artifacts | finished jobs | model files + metadata | 进入 App / SDK 集成阶段 |

这套流程的好处是把“端侧部署”从本地反复配 SDK、找设备、跑命令，抽象成云端 job；代价是开发者必须理解这些 job 的边界。比如 Quantize job 给的是量化资产，不代表精度一定合格；Profile job 给的是目标设备或 proxy 设备上的测量，不代表你的整机应用一定满足热稳定；Compile job 成功也不代表前后处理、输入输出格式和 App pipeline 已经处理好。

## 一个模型怎样走到头显里

把上面的工具链串起来，一个比较务实的 XR AI 流程可以这样走。

第一步是模型整理。无论模型来自 PyTorch、ONNX 还是 AIMET 量化资产，都要先明确输入输出、shape、预处理和后处理。XR 项目里相机图像可能有旋转、裁剪、畸变校正、色彩空间转换，手势或空间理解模型还可能依赖时间窗口；这些前处理不和训练时对齐，后面即使 compile 成功，数值结果也未必可信。

第二步是云端编译和 profile。开发者可以先选一个接近目标设备的 AI Hub target，例如 `QCS8450 (Proxy)` / `XR2 Gen 2 (Proxy)`，提交 compile job 和 profile job，看模型能否通过 runtime、哪些层落在 NPU、延迟和峰值内存大致在哪个范围。这个阶段的目标不是证明产品一定能上线，而是快速排除明显不合适的模型形态。

第三步是量化。端侧模型通常需要在模型大小、内存、延迟和精度之间做交换。对于视觉模型，PTQ 往往是第一步：准备代表性 calibration data，提交 quantize job，得到 QDQ / fake quantization 形式的量化 ONNX，再重新 compile 和 profile。这个阶段要同时看 numerics 和性能，不能只看模型文件变小。

第四步是本地集成。AI Hub 下载的部署资产需要接入 Android App、native library、相机输入、UI、线程调度、runtime 初始化和输入输出 buffer。这里 QIDK、QNN sample、ONNX Runtime QNN EP 示例比云端模型卡更有帮助，因为它们回答的是“代码怎么写、so 怎么带、设备上怎么跑”。

第五步是整机验证。问题会从“模型能不能跑”变成“能不能在真实头显里持续跑”。验证项包括冷启动、连续运行、场景切换、摄像头和渲染同时开启、无线高负载、低电量、升温后降频、不同光照和不同用户动作。XR 项目最终考验的是闭环体验，而不是单个模型 job 的成功状态。

这五步看起来比“上传模型，下载结果”麻烦，但它更接近真实开发节奏。AI Hub 的价值是把前两三步大幅提速；QIDK 和 QNN 的价值是把后两步落到本地工程；Product Brief 的价值则是告诉你这套硬件系统本来打算服务什么样的 XR 工作负载。

## Proxy 数字该怎么用

AI Hub Release Notes 在 2024 年 4 月 22 日写到：Added QCS8450 Proxy devices，并且同一条 release 还加入了 QNN path 的 `w4a16 quantization` 支持。到 2026 年 3 月 2 日，release notes 又补充了 proxy device 的警告：proxy 设备用于模拟指定设备的 NPU 特征，但 CPU/GPU 性能、真实设备表现会受 OS、firmware、clock speed、memory、thermal packaging 等影响。

这说明 `QCS8450 (Proxy)` 更像一个云端工具链中的目标设备抽象，而不是“我手里有一台真实 QCS8450 头显可以直接跑”。在 AI Hub Models 开源仓库里，`cs_xr_8450` 的 `reference_device_name` 是 `QCS8450 (Proxy)`；VIT 的 `perf.yaml` 里也把 `QCS8450 (Proxy)` 列为 supported device，并给出了 NPU profile 数字。

为什么 Qualcomm 要提供这种 proxy 口径？原因是 AI Hub 要服务很多设备、芯片和行业形态，不可能每个目标设备都长期托管一台真实机。Proxy 的价值在于把“非托管设备”的模型兼容性和资产生成提前暴露出来，让开发者能先跑通大部分工具链。对 XR 来说，这很有用，因为真实头显通常不是开放云设备，开发者拿到真机之前就可以先判断模型是否值得投入。

但 proxy 的风险也要写在前面。它模拟的是目标设备的部分 NPU 特征，不等于完整模拟整机。真实头显的 OS build、firmware、runtime 版本、SoC binning、内存配置、散热结构、相机工作模式、屏幕刷新率和后台服务都可能影响结果。尤其是 XR 设备，CPU/GPU/NPU 的调度不是孤立发生的；一个模型在 proxy 上全层 NPU 通过，只能说明它有很好的候选价值，不能直接说明它已经满足产品 SLA。

| Proxy item | Source | Meaning |
| --- | --- | --- |
| QCS8450 Proxy devices | AI Hub Release Notes, 2024-04-22 | Workbench 添加了 QCS8450 的 proxy target |
| QCS8450 XR Platform | AI Hub Models README | XR chipsets 列表中的平台名 |
| `cs_xr_8450` -> `QCS8450 (Proxy)` | AI Hub Models scorecard code | 开源模型 scorecard 的 XR 目标设备名 |
| XR2 Gen 2 (Proxy) | AI Hub Models VIT page supported devices | 模型页面可见的 XR proxy device |
| `qualcomm-qcs8450-proxy` -> `htp_version: 69` | AI Hub Models `devices_and_chipsets.yaml` | 和 `sm8450` / Snapdragon 8 Gen 1 同为 HTP v69、SoC model 36 |

实际工程里可以这样用：早期模型选型看 `QCS8450 (Proxy)` / `XR2 Gen 2 (Proxy)` 是否支持；性能预算阶段看 proxy profile；产品定版前必须在真实头显上跑完整 profile、热稳定和准确率回归。

因此，后面看 ViT 量化时要带着这个前提：本文讨论的是 QCS8450 / SM8450 这一代 v69 后端上的工程取舍，不是把 Qualcomm 最新 NPU 能力一概投射回 XR2 Gen 2。尤其是 `w8a16`、mixed precision、context binary 和算子融合这类能力，是否可用、是否值得用，最终都要看具体 HTP 版本、QNN/QAIRT 版本、模型图和 runtime path。

## 用 ViT 例子看量化部署

前面讲的是平台和工具链，这里换成一个具体模型来看。ViT 的好处是它不是最容易优化的 CNN，而是带有大量 MatMul、LayerNorm、Softmax 和 MLP block 的 Transformer-style 视觉模型，更能暴露端侧 NPU 部署时的真实问题。

AI Hub Models 里的 VIT 是一个很好的例子，因为它不是 MobileNet 这种天然移动友好的 CNN，而是 86.6M 参数的 Vision Transformer。它能跑通，说明 Qualcomm 的工具链已经把常见 Transformer 视觉分类模型纳入了端侧部署路径。

选择 ViT 来看量化也有代表性。卷积网络的计算模式比较规则，移动端生态里已经优化很多年；ViT 则包含 patch embedding、multi-head attention、MLP、LayerNorm 和大量矩阵乘法，激活分布和算子形态更接近现代多模态模型。它不等同于 LLM，但它能帮助我们观察 Transformer-style workload 在端侧 NPU / runtime 上会遇到的典型问题。

把 ViT 放到 XR 语境里，它未必就是最终产品里的主模型，但很适合作为“端侧 Transformer 可部署性”的观察窗口。MR 头显未来会越来越多地使用视觉编码器、场景理解模型、多模态检索模型和小型 agent 模块；这些模型不一定都和 ImageNet 分类一样简单，但它们会共享很多工程问题：模型尺寸、矩阵乘法效率、LayerNorm / Softmax 支持、量化后精度、NPU placement 和前后处理成本。

VIT 页面和开源仓库给出的技术信息如下：

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

AI Hub Workbench 文档给出的通用量化流程是：

!!! note "术语框：AIMET、PTQ 和 QDQ"
    `AIMET` 是 Qualcomm 开源的模型效率工具，常用于量化、压缩和量化精度分析。`PTQ` 指 post-training quantization，也就是不重新训练或少量校准地把 float 模型转成定点模型。`QDQ` 是 ONNX 里常见的量化表达方式，用 `QuantizeLinear / DequantizeLinear` 节点标出量化瓶颈；AI Hub 的 quantize job 输出的 quantized ONNX 就是这种 fake quantization / QDQ 风格资产，再继续 compile 到 TFLite、QNN 或 ONNX。

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

为什么要先 compile 到 ONNX 再量化？文档里的原因很实用：即使源模型已经是 ONNX，也建议先通过 compile job 生成优化后的 ONNX，因为编译器会先做图优化，把一些未优化 pattern 处理掉，降低后续量化失败或精度异常的概率。

这一步背后有一个常见经验：量化不是在“原始训练图”上机械地把 float 换成 int。部署图会经过算子融合、常量折叠、shape 固化、layout 调整和 runtime 适配；如果直接对一个未优化图做量化，后面再编译时可能改变图结构，使得校准假设、Q/DQ 节点位置和真实执行路径不一致。先得到更接近部署形态的 ONNX，再做校准和量化，通常更符合端侧工程逻辑。

这里的“校准数据”也不能随便糊弄。Workbench 文档里提到可以用随机样本快速估计 latency，但也明确这种量化模型的 accuracy 会很差。对真正要部署的 ViT，校准集应尽量贴近实际输入分布：图像尺寸、归一化方式、颜色空间、预处理和 batch 形态都要和线上一致。否则量化参数虽然生成了，attention 和 MLP 中间激活的范围却可能偏离真实数据，最终表现为 top-1 accuracy 掉点或某些类别异常。

AI Hub 文档的量化精度支持表如下，保留英文原格式：

|  | Weights | Activations | Mixed Precision* |
| --- | --- | --- | --- |
| TFLite | int8 | int8 | Not supported |
| QNN | int8 | int8, int16 | Not supported |
| ONNX | int8 | int8, int16 | Not supported |

<small>表源：Qualcomm AI Hub Workbench `Quantization` 文档。文档同时说明 long-term 目标是支持 mixed precision，以及 int4 / int8 / int16 weights and activations。</small>

对 ViT 来说，量化的关键点有四个：

1. **校准数据要代表真实输入分布**：分类模型用 ImageNet / Imagenette 一类图像做 calibration，比随机输入更能保住 accuracy。
2. **权重和激活一起看**：`w8a8` 压得更狠，模型更小，但 attention / MLP 层的激活范围更容易损失精度。
3. **先确认 HTP 版本再选精度**：VIT 的 `code-gen.yaml` 虽然列出 `w8a8_mixed_int16`、`w8a16`、`w8a8`，但 QCS8450 / SM8450 这一代在 AI Hub 元数据里是 HTP v69；在本文的 ViT / QCS8450 实验口径里，`w8a16` 要求设备 SoC 配备的 Hexagon DSP / HTP 架构版本达到 v73 或更高，因此 SM8450 / QCS8450 只能把 `w8a8` 当成可靠量化路径。
4. **profile 要看 runtime 和 placement**：同一个模型走 TFLite、QNN DLC、ONNX Runtime，最终 layer placement、峰值内存和延迟都可能不同。

!!! warning "QCS8450 上的 w8a16 口径"
    这里要区分“模型库声明支持某种 precision”和“某个目标 SoC / runtime path 真正适合用它”。ViT 的模型配置里确实列出了 `w8a16`，但 QCS8450 Proxy 和 SM8450 在 AI Hub Models 设备元数据里都是 HTP v69；本文按实验口径处理为：`w8a16` 只在 Hexagon DSP / HTP v73+ 设备上作为可选路径，SM8450 / QCS8450 v69 不把它列入 ViT 的可靠量化方案。因此 QCS8450 上优先比较 `float` 和 `w8a8`，不要只看到模型页面有 `w8a16` 字样，就默认 XR2 Gen 2 这颗目标芯片也能稳定、快速地吃下它。

这也是为什么我更愿意把 QCS8450 上的 ViT 量化写成“实验路径选择”，而不是抽象地说“INT8 一定更快”。如果目标是 XR2 Gen 2 / QCS8450，第一步应先锁定设备代际：HTP v69、SoC model 36；第二步再锁定 runtime：TFLite、QNN DLC、ONNX Runtime QNN EP 或 context binary；第三步才比较 `float`、`w8a8`、可能的 mixed int16 / `w8a16`。顺序反过来，很容易把更新芯片上的经验误用到 v69。

开源仓库中的 VIT `perf.yaml` 给出 QCS8450 Proxy 上的部分 profile 数字：

| Precision | Runtime | Device | Job status | Inference time milliseconds | Estimated peak memory range MB | Primary compute unit | Layer counts |
| --- | --- | --- | --- | ---: | --- | --- | --- |
| float | tflite | QCS8450 (Proxy) | Passed | 12.966 | 0-290 | NPU | total 1399, npu 1399 |
| float | qnn_dlc | QCS8450 (Proxy) | Passed | 13.207 | 0-229 | NPU | total 888, npu 888 |
| w8a8 | tflite | QCS8450 (Proxy) | Passed | 20.594 | 0-447 | NPU | total 1451, npu 1451 |
| w8a8 | qnn_dlc | QCS8450 (Proxy) | Passed | 13.179 | 0-310 | NPU | total 889, npu 889 |

<small>数据源：`qualcomm/ai-hub-models` 中 VIT 的 [`perf.yaml`](https://github.com/qualcomm/ai-hub-models/blob/main/src/qai_hub_models/models/vit/perf.yaml)。当前仓库文件的 QCS8450 Proxy 片段来自 profile job 记录，适合做方向判断；但它仍然是 proxy 数字，而且同一文件里不同设备、runtime、tool version 的内存区间有明显波动，所以我会把它当成“需要复现实验的参考值”，不是产品验收值。</small>

这组数字有两个启发。第一，量化不必然让所有 runtime 都更快，`w8a8 tflite` 在这个 proxy profile 里反而慢于 float；第二，QNN DLC 的 `w8a8` 和 float 延迟接近，但模型尺寸从 330 MB 降到 83.2 MB，部署包大小和内存压力仍然明显改善。真正做产品时，应该同时比较 latency、memory、model size、accuracy 和 runtime 集成成本，而不是只看“是否 INT8”。

这也是端侧量化最容易被误解的地方。量化的收益可能体现在多个维度：模型包变小、权重带宽降低、NPU 支持更好、峰值内存下降、功耗降低、冷启动更快。但某个 runtime 上的单次 latency 未必一定下降，因为图转换、Q/DQ 插入、算子融合、内存布局、fallback 和硬件调度都会影响最终结果。尤其在 XR 设备上，`13 ms` 和 `20 ms` 的差异不能孤立看，要放到一帧的完整预算里：相机、tracking、渲染和 display 都还要分时间。

### 为什么 w8a8 的 MatMul 可能比 float 慢

在 ViT 这种模型里，“量化后 MatMul 更慢”不是完全反常。这里的 `float` 也不一定意味着桌面意义上的 FP32；在 Qualcomm NPU / HTP 路径上，float profile 常常会走 FP16-friendly 的 NPU kernel。对于矩阵尺寸固定、batch 较小、图结构已经被编译器优化过的 ViT block，FP16 kernel 可能已经非常接近硬件甜点区。

`w8a8` 则多了几个潜在成本：

1. **Q/DQ 和 requantization 开销**：ONNX / TFLite 的量化图通常会显式或隐式插入量化、反量化、scale / zero point 处理。即使 profile 里显示全层 NPU，这些转换也会占用图执行时间。
2. **layout 和 packing 成本**：INT8 MatMul 往往需要特定权重打包、activation layout 和对齐方式。ViT 的 MatMul 不全是大 batch GEMM，attention 里的小矩阵、reshape、transpose、softmax、LayerNorm 会让 INT8 kernel 难以一直保持高利用率。
3. **融合机会可能变少**：如果量化后图里出现额外的 Q/DQ 边界，某些原本可被编译器合并的 pattern 可能被切断。VIT `perf.yaml` 中 `w8a8 tflite` 的 layer count 从 float 的 `1399` 增到 `1451`，峰值内存范围也从 `0-290 MB` 增到 `0-447 MB`，这至少说明图执行形态并没有因为 INT8 自动变简单。
4. **runtime path 差异**：同样是 `w8a8`，TFLite 是 `20.594 ms`，QNN DLC 是 `13.179 ms`。这说明瓶颈不只是“量化格式”，还和具体 runtime、delegate / backend、图优化和 kernel 选择有关。

所以实验里看到 `w8a8 MatMul > FP MatMul`，我会先怀疑三件事：MatMul 形状是否太小或太碎、Q/DQ 是否切断了融合、以及该 runtime 在 v69 HTP 上的 INT8 kernel 是否比 FP16 path 更成熟。下一步应该用 QNN profile / layer timing 把单个 MatMul、Gemm、Reshape、Transpose、Softmax、LayerNorm 的时间拆出来，而不是只看整网平均 latency。

### GELU、ReLU 与算子融合

另一个很实际的优化点是激活函数。ViT 的 MLP block 常见写法是 `Linear -> GELU -> Linear`。在 ONNX 图里，`GELU` 可能不是一个芯片后端喜欢的单一高效 kernel，而是展开成 `Erf / Tanh / Add / Mul / Div` 一类子图；即使某些 runtime 支持 `Gelu` op，实际在 HTP v69 上也可能比简单激活慢。

如果业务精度允许，把 `GELU` 改成 `ReLU` 有两个收益。第一，`ReLU` 本身比 `GELU` 简单，近似成本低；第二，它更容易和前面的 `MatMul` / `Gemm` 形成融合 pattern。AI Hub Models 仓库的 AIMET 默认配置里，v69 / v73 per-tensor config 都把 `["Gemm", "Relu"]` 放在 `supergroups` 中，而没有等价的 `Gemm + Gelu` supergroup。这个配置不是最终 QNN 编译器的完整规则，但它能说明 Qualcomm 量化配置中确实把 `Gemm -> Relu` 当作一类可整体处理的常见模式。

!!! tip "用 Netron 看什么"
    可以用 Netron 打开导出的 ONNX：如果看到 `MatMul/Gemm -> Erf/Tanh/Mul/Add` 这类长链路，通常就是 GELU 被展开后的形态；如果替换为 ReLU，图上会更接近 `MatMul/Gemm -> Relu`。后者在 Qualcomm AIMET 配置里有明确的 `Gemm + Relu` supergroup，后续交给 QNN / QAIRT 时也更容易触发融合。真正上线前要做两件事：一是比较替换前后的 accuracy，二是用 layer timing 确认加速来自激活和融合，而不是其它图优化偶然变化。

这个替换不能机械套用。ImageNet 分类里的 ViT 可能对 `GELU -> ReLU` 有可接受的精度损失，也可能掉点明显；如果是视觉编码器、检索模型或多模态 backbone，embedding 分布变化还会影响下游任务。更稳妥的做法是先在训练 / 微调阶段就使用 ReLU 或 ReLU-like 激活，至少做一个小规模 fine-tuning，再导出 ONNX、用 Netron 检查图、跑 `float` / `w8a8` 的 numerics 和 profile 对照。

如果把 ViT 量化落实成一张检查表，我会按下面顺序看：

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
| QNN context binary / precompiled QNN ONNX | 面向特定 SoC 的 AOT 编译部署 | 设备特异性强，proxy 结果不能替代实机验证 |

XR 场景通常更在意稳定延迟和热稳定，而不是一次性 benchmark 的最低值。因为头显里同时有渲染、sensor fusion、video see-through 和后台系统服务，模型推理不能抢走过多内存带宽或导致持续降频。

我的理解是：如果只是做 demo，优先选最容易集成的路径；如果是做产品，优先选可观测、可复现、可控的路径。比如 LiteRT 可能让 Android 团队上手最快；ONNX Runtime 适合跨 OS 统一接口；QNN DLC / context binary 更贴近 NPU 性能和 SoC 特化，但也要求团队能处理更复杂的部署、版本和调试问题。没有一个 runtime 永远正确，只有和项目阶段匹配的 runtime。

因此，runtime 选择最好不要在项目开始时一次性定死。早期可以用 AI Hub 和熟悉的 runtime 快速跑通；中期比较 LiteRT、ONNX Runtime QNN EP 和 QNN DLC 的延迟、内存、集成成本；后期再根据真实头显 profile 选择最稳定的路径。这里的“稳定”不只是平均延迟低，还包括版本可控、日志可读、失败可复现、资产更新流程清楚。

## 项目启发

把 XR2 Gen 2 / QCS8450 用到项目里，建议按下面顺序推进：

1. **先明确 workload**：分类、检测、分割、手势、语音、SLAM 辅助、生成式模型，对 runtime 和功耗的要求完全不同。
2. **先用 AI Hub Models 找相似模型**：如果 VIT、MobileNet、YOLO、SAM、Whisper、LLM 等已有模型能跑，就先参考它的 `export.py`、`perf.yaml` 和 `numerics.yaml`。
3. **再用 Workbench 做 proxy profile**：看目标设备是否支持、哪些层落在 NPU、延迟和峰值内存是否可接受。
4. **量化前先看 HTP 版本**：QCS8450 / SM8450 是 HTP v69；本文的 ViT 实验按 `w8a16` 需要 HTP v73+ 的口径处理，所以 QCS8450 上先把 `w8a8` 跑扎实，再把 `w8a16` 留给 QCS8550、8 Gen 2 或更新平台验证。
5. **用图可视化找融合机会**：Netron 里重点看 `GELU` 是否被展开、`MatMul/Gemm + ReLU` 是否能形成更干净的 pattern，再用 QNN / runtime profile 确认收益。
6. **最后实机闭环**：在真实头显上跑摄像头、渲染、网络、热稳定和用户交互，proxy 只代表早期工程判断。

一句话总结：**XR2 Gen 2 的价值不是单点 AI 跑分，而是把显示、感知、渲染、无线和端侧 AI 放在同一个电池供电头显里同时跑起来。QCS8450 / XR2 Gen 2 Proxy 则是 Qualcomm AI Hub 里帮助开发者提前做模型适配和性能判断的工程入口。**

如果把这篇当成一次分享的收口，我会强调三个判断。第一，XR2 Gen 2 是 XR 产品平台，不是单纯的 AI 加速芯片；讨论它时要把显示、相机、感知、渲染和无线一起看。第二，QCS8450 / XR2 Gen 2 Proxy 是很有用的开发入口，但它的意义是“提前筛选和生成资产”，不是“替代实机验收”。第三，Qualcomm 的软件栈越来越像一个从模型库到云端编译、再到设备 runtime 的完整工作流；真正的难点不再只是把模型转成某个文件，而是把模型放进 XR 整机的持续并发预算里。

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
