(function () {
  const TERM_GROUPS = [
    {
      aliases: ["LLM", "LLM/VLM"],
      tip: "LLM：Large Language Model，大语言模型。通常指以 Transformer 为主体、通过海量文本训练出的生成式语言模型。"
    },
    {
      aliases: ["VLM", "VLM/VLA"],
      tip: "VLM：Vision-Language Model，视觉语言模型。输入通常包含图像和文本，输出回答、定位、结构化结果或工具调用。"
    },
    {
      aliases: ["VLA"],
      tip: "VLA：Vision-Language-Action，视觉-语言-动作模型。它把视觉观察和语言指令进一步接到动作输出。"
    },
    {
      aliases: ["RAG"],
      tip: "RAG：Retrieval-Augmented Generation，检索增强生成。先检索外部资料，再把资料交给模型生成答案。"
    },
    {
      aliases: ["MoE"],
      tip: "MoE：Mixture of Experts，专家混合模型。每个 token 通常只路由到部分 expert，以较低计算量扩大模型容量。"
    },
    {
      aliases: ["KV Cache", "KV cache", "KV"],
      tip: "KV Cache：Transformer 推理时缓存历史 token 的 Key/Value 张量。长上下文和高并发场景里，它经常是显存瓶颈。"
    },
    {
      aliases: ["LoRA"],
      tip: "LoRA：Low-Rank Adaptation，低秩适配。冻结主模型，只训练低秩增量参数来降低微调成本。"
    },
    {
      aliases: ["QLoRA"],
      tip: "QLoRA：在量化底座上训练 LoRA 增量的微调方法，常用于低显存大模型适配。"
    },
    {
      aliases: ["SFT"],
      tip: "SFT：Supervised Fine-Tuning，监督微调。用高质量指令或任务样本把底座模型塑造成更符合任务要求的行为。"
    },
    {
      aliases: ["RLHF"],
      tip: "RLHF：Reinforcement Learning from Human Feedback，基于人类反馈的强化学习对齐。"
    },
    {
      aliases: ["DPO"],
      tip: "DPO：Direct Preference Optimization，直接偏好优化。用偏好对直接优化模型策略，常用于后训练对齐。"
    },
    {
      aliases: ["PPO"],
      tip: "PPO：Proximal Policy Optimization，近端策略优化。RLHF 中常见的强化学习算法。"
    },
    {
      aliases: ["GRPO"],
      tip: "GRPO：Group Relative Policy Optimization，组相对策略优化。常用于推理模型后训练中的偏好或奖励优化。"
    },
    {
      aliases: ["MTP"],
      tip: "MTP：Multi-Token Prediction，多 token 预测。训练时让模型同时预测后续多个 token，可用于提升表示和推理效率。"
    },
    {
      aliases: ["DDPM"],
      tip: "DDPM：Denoising Diffusion Probabilistic Models，现代扩散模型的经典起点，核心是逐步加噪和逐步去噪。"
    },
    {
      aliases: ["DDIM"],
      tip: "DDIM：Denoising Diffusion Implicit Models。不重训模型，通过改变采样路径实现更少步的确定性或半确定性采样。"
    },
    {
      aliases: ["DPM-Solver++", "DPM-Solver", "DPM"],
      tip: "DPM-Solver：面向 diffusion ODE 的专用高阶求解器，目标是在较少采样步下保持生成质量。"
    },
    {
      aliases: ["Euler"],
      tip: "Euler：最简单的一阶数值积分方法。扩散采样里常作为快速、直接、好调试的低成本 solver。"
    },
    {
      aliases: ["Heun"],
      tip: "Heun：二阶预测-校正数值方法。扩散采样里常用额外一次模型调用换更平滑、更稳定的更新。"
    },
    {
      aliases: ["CFG"],
      tip: "CFG：Classifier-Free Guidance，无分类器引导。把条件和无条件预测组合起来，提高扩散生成对提示词的贴合度。"
    },
    {
      aliases: ["Classifier-Free Guidance"],
      tip: "Classifier-Free Guidance：无分类器引导。扩散模型中同时使用条件和无条件预测，通过线性外推增强提示词或条件控制。"
    },
    {
      aliases: ["ODE", "ODE/SDE", "SDE/ODE"],
      tip: "ODE：Ordinary Differential Equation，常微分方程。扩散里常指确定性的 probability flow 轨迹。"
    },
    {
      aliases: ["SDE"],
      tip: "SDE：Stochastic Differential Equation，随机微分方程。扩散里常指带随机噪声项的连续时间过程。"
    },
    {
      aliases: ["SNR"],
      tip: "SNR：Signal-to-Noise Ratio，信噪比。扩散训练里常用来描述信号与噪声在不同时间步的相对强度。"
    },
    {
      aliases: ["ELBO"],
      tip: "ELBO：Evidence Lower Bound，证据下界。变分推断常用目标，在扩散和潜变量模型里经常出现。"
    },
    {
      aliases: ["LCM"],
      tip: "LCM：Latent Consistency Models，潜空间一致性模型。目标是把扩散采样压到更少步。"
    },
    {
      aliases: ["DMD2", "DMD"],
      tip: "DMD：Distribution Matching Distillation，分布匹配蒸馏。DMD2 是其改进版本，常用于一步或少步扩散生成。"
    },
    {
      aliases: ["Distribution Matching Distillation"],
      tip: "Distribution Matching Distillation：分布匹配蒸馏。重点不是逐步模仿 teacher 轨迹，而是让学生模型的最终生成分布贴近目标分布。"
    },
    {
      aliases: ["Rectified Diffusion", "Rectified Flow", "rDM"],
      tip: "Rectified Diffusion：扩散快速生成中的路径整流路线。核心不是盲目追求几何上最直，而是让生成轨迹更适合少步或一步离散采样。rDM 在不同文献中可能有歧义，本文默认指 Rectified Diffusion/Flow 语境。"
    },
    {
      aliases: ["Score Matching"],
      tip: "Score Matching：学习数据分布 score 的训练思想。扩散模型里可理解为学习“从噪声样本往高概率数据区域走”的方向场。"
    },
    {
      aliases: ["Score SDE"],
      tip: "Score SDE：把扩散生成放进随机微分方程框架，用 score 网络描述连续时间的反向生成过程。"
    },
    {
      aliases: ["Probability Flow ODE", "Probability Flow"],
      tip: "Probability Flow ODE：与某个扩散 SDE 共享边缘分布的确定性 ODE 路径，是 DDIM 和许多高阶采样器的重要理论入口。"
    },
    {
      aliases: ["Consistency Models", "Consistency Model", "Consistency"],
      tip: "Consistency Models：一致性模型。目标是让同一生成轨迹上不同噪声时刻能直接映射到一致的干净结果，从而支持少步或一步生成。"
    },
    {
      aliases: ["Flow Matching"],
      tip: "Flow Matching：直接拟合从噪声分布到数据分布的连续速度场，常和 Rectified Flow、ODE 生成路线一起讨论。"
    },
    {
      aliases: ["Progressive Distillation"],
      tip: "Progressive Distillation：渐进式蒸馏。逐轮把 teacher 的多步采样压缩成更少步，让学生逐步学会快速生成。"
    },
    {
      aliases: ["Phased DMD"],
      tip: "Phased DMD：阶段化的 DMD 路线，把一步或少步生成拆成更稳的训练阶段，降低直接蒸馏的难度。"
    },
    {
      aliases: ["InstaFlow"],
      tip: "InstaFlow：基于 Rectified Flow 的一步文生图路线，代表了把扩散生成压缩到极少步的一类尝试。"
    },
    {
      aliases: ["EDM"],
      tip: "EDM：Elucidated Diffusion Models，系统梳理扩散噪声日程、参数化和采样设计空间的一类方法。"
    },
    {
      aliases: ["Latent Diffusion", "Stable Diffusion"],
      tip: "Latent Diffusion / Stable Diffusion：在压缩后的 latent 空间中做扩散生成，显著降低高分辨率图像生成成本。"
    },
    {
      aliases: ["ControlNet"],
      tip: "ControlNet：给扩散模型增加可控条件分支的方法，常用于边缘、深度、姿态、线稿等强结构控制。"
    },
    {
      aliases: ["UNet", "U-Net", "DiT"],
      tip: "扩散主干结构：UNet/U-Net 是经典卷积式去噪网络，DiT 是 Diffusion Transformer，用 Transformer 作为扩散去噪主干。"
    },
    {
      aliases: ["Transformer", "Attention", "Self-Attention", "Cross-Attention", "cross-attention"],
      tip: "Transformer / Attention：现代大模型核心结构。Self-Attention 建模同一序列内部关系，Cross-Attention 用于把文本、图像或其他条件注入主干。"
    },
    {
      aliases: ["QKV", "Q/K/V", "Query", "Key", "Value"],
      tip: "Q/K/V：Attention 的三组投影。Query 表示当前位置要找什么，Key 用于匹配，Value 是最终被加权读取的信息。"
    },
    {
      aliases: ["Embedding", "Tokenizer", "Latent"],
      tip: "基础表示术语：Tokenizer 把输入切成 token，Embedding 把离散 token 映射成向量，Latent 指模型内部更压缩的潜表示。"
    },
    {
      aliases: ["Tokenization", "Token", "Patch Token"],
      tip: "Tokenization：把文本、图像 patch、视频片段或动作切成模型能处理的 token 序列，是 Transformer 输入的第一步。"
    },
    {
      aliases: ["RoPE", "ALiBi", "Causal Mask", "Padding Mask", "Attention Mask"],
      tip: "位置与可见性术语：RoPE/ALiBi 用于注入位置信息，Causal/Padding/Attention Mask 决定哪些 token 能互相看见。"
    },
    {
      aliases: ["Conv", "Convolution", "CNN", "Receptive Field"],
      tip: "卷积/CNN 基础：Conv 用共享滑窗提取局部特征，Receptive Field 表示某个输出位置能看到的输入范围。"
    },
    {
      aliases: ["Linear", "Linear Layer"],
      tip: "Linear Layer：线性层，通常做 xW+b 投影，是 QKV、输出投影和 MLP 中最常见的基础模块。"
    },
    {
      aliases: ["LayerNorm", "BatchNorm", "RMSNorm", "Residual", "SwiGLU", "GELU"],
      tip: "深层网络稳定组件：Norm 控制数值尺度，Residual 保留梯度通路，GELU/SwiGLU 等激活函数提供非线性表达。"
    },
    {
      aliases: ["Backprop", "Backpropagation", "Optimizer", "AdamW", "Gradient Clipping"],
      tip: "训练基础：Backprop 用链式法则计算梯度，Optimizer 根据梯度更新参数，Gradient Clipping 用于限制异常梯度。"
    },
    {
      aliases: ["Activation Checkpointing", "Checkpointing", "Autograd"],
      tip: "训练显存术语：Autograd 自动记录计算图并回传梯度；Activation Checkpointing 用反向重算换取更低激活显存。"
    },
    {
      aliases: ["PTQ"],
      tip: "PTQ：Post-Training Quantization，后训练量化。模型训练完成后再做量化，不重新大规模训练。"
    },
    {
      aliases: ["QAT"],
      tip: "QAT：Quantization-Aware Training，量化感知训练。训练时显式模拟量化误差，让模型提前适应低精度。"
    },
    {
      aliases: ["GPTQ"],
      tip: "GPTQ：一种后训练权重量化方法，用二阶近似和误差补偿降低低比特量化损失。"
    },
    {
      aliases: ["AWQ", "AutoAWQ"],
      tip: "AWQ：Activation-aware Weight Quantization。按激活敏感性保护关键权重或通道，常用于 LLM 低比特部署。"
    },
    {
      aliases: ["SmoothQuant"],
      tip: "SmoothQuant：通过重参数化把部分激活离群值迁移到权重侧，让激活量化更稳定。"
    },
    {
      aliases: ["FP32", "FP16", "BF16", "FP8", "FP4"],
      tip: "FP：Floating Point，浮点格式。FP32/FP16/BF16/FP8/FP4 表示不同位宽和数值范围的浮点精度。"
    },
    {
      aliases: ["MXFP8", "MXFP4", "MXFP", "MXINT", "Microscaling"],
      tip: "MXFP / Microscaling：微缩放低精度格式。通常让一小块元素共享 scale，再用 FP8/FP4 等低比特元素表示，兼顾局部分布和硬件效率。"
    },
    {
      aliases: ["NVFP4", "NVFP"],
      tip: "NVFP4：NVIDIA 低比特浮点格式路线，常结合更小微块、FP8 scale 和分层缩放来提升 FP4 训练或推理精度。"
    },
    {
      aliases: ["HiFloat8", "HIFP8", "HiF8"],
      tip: "HiFloat8 / HIFP8：面向深度学习的 8-bit 浮点格式路线，通过调整指数和尾数分配改善动态范围与精度折中。"
    },
    {
      aliases: ["INT8", "INT4", "W4A16", "W8A8", "W4A8", "NF4"],
      tip: "低比特量化格式：INT8/INT4 是整数低精度，W4A16 等描述权重和激活精度，NF4 常用于 QLoRA。"
    },
    {
      aliases: ["QDQ"],
      tip: "QDQ：QuantizeLinear / DequantizeLinear 表示。ONNX 图里常用它显式标注量化和反量化位置。"
    },
    {
      aliases: ["QOperator", "MatMulNBits"],
      tip: "ONNX 量化算子表示：QOperator 直接使用量化算子，MatMulNBits 常用于低比特权重矩阵乘路径。"
    },
    {
      aliases: ["ONNX Runtime", "ONNX", "ORT"],
      tip: "ONNX Runtime，常简称 ORT。它执行 ONNX 图，并通过不同 Execution Provider 对接 CPU、GPU 和厂商后端。"
    },
    {
      aliases: ["Execution Provider", "EP", "OpenVINO", "DirectML", "QNN"],
      tip: "ONNX Runtime 后端术语：Execution Provider 负责把 ONNX 图交给具体硬件或库执行，例如 OpenVINO、DirectML、QNN、TensorRT EP。"
    },
    {
      aliases: ["TensorRT-LLM", "TensorRT"],
      tip: "TensorRT-LLM：NVIDIA 面向大模型推理的高性能优化栈，常与 engine 构建、FP8、GPTQ/AWQ 和专用 kernel 绑定。"
    },
    {
      aliases: ["vLLM"],
      tip: "vLLM：常用 LLM serving runtime。它通过 PagedAttention、batching、KV cache 管理和量化格式支持提升在线吞吐。"
    },
    {
      aliases: ["SGLang"],
      tip: "SGLang：面向 LLM/VLM serving、结构化输出和 agent workload 的运行框架，也支持多种量化路径。"
    },
    {
      aliases: ["LightLLM"],
      tip: "LightLLM：偏轻量、高性能和易扩展的 LLM serving framework，常和特定 backend、kernel 与 ModelTC 生态工具结合。"
    },
    {
      aliases: ["GGUF"],
      tip: "GGUF：常见大模型本地推理权重格式，尤其常见于 llama.cpp 生态。"
    },
    {
      aliases: ["HQQ", "RTN"],
      tip: "低比特权重量化算法。RTN 通常指 round-to-nearest，HQQ 是一种半二次量化路线。"
    },
    {
      aliases: ["TorchAO"],
      tip: "TorchAO：PyTorch 生态中的量化和低精度工具集合，可用于实验和部分服务路径。"
    },
    {
      aliases: ["bitsandbytes"],
      tip: "bitsandbytes：常用于 Transformers 生态的 8bit/4bit 加载和低显存微调工具，研究和 QLoRA 场景很常见。"
    },
    {
      aliases: ["LLM Compressor", "LightCompress", "LLMC"],
      tip: "LLM 压缩工具链：常用于离线量化、剪枝、格式转换和导出到 vLLM、SGLang、LightLLM 等 runtime。"
    },
    {
      aliases: ["CPU", "GPU", "TPU", "NPU", "DSP"],
      tip: "硬件缩写：CPU 是通用处理器，GPU/TPU/NPU/DSP 常用于并行计算、AI 加速或边缘推理。"
    },
    {
      aliases: ["CUDA"],
      tip: "CUDA：NVIDIA GPU 编程平台和运行时，是很多深度学习 kernel、通信和推理优化的底层基础。"
    },
    {
      aliases: ["Triton"],
      tip: "Triton：面向 GPU 的高层 kernel 编程语言，常用于快速实现和自动调优深度学习算子。"
    },
    {
      aliases: ["GEMM"],
      tip: "GEMM：General Matrix-Matrix Multiplication，通用矩阵乘。Transformer 训练和推理中最核心的计算模式之一。"
    },
    {
      aliases: ["Kernel", "Fused Kernel", "KV Kernel"],
      tip: "Kernel：在 GPU/加速器上执行的底层计算程序。Fused Kernel 会把多个操作融合，减少中间读写和 launch 开销。"
    },
    {
      aliases: ["CUTLASS"],
      tip: "CUTLASS：NVIDIA 的 CUDA 矩阵乘和张量计算模板库，常用于构建高性能 GEMM kernel。"
    },
    {
      aliases: ["cuBLAS", "cuBLASLt", "cuDNN"],
      tip: "NVIDIA 高性能库：cuBLAS/cuBLASLt 主要提供矩阵乘能力，cuDNN 主要服务深度学习常用算子。"
    },
    {
      aliases: ["CuTe"],
      tip: "CuTe：CUTLASS 生态里的张量布局和算法组合抽象，常用于写更复杂的 GPU kernel。"
    },
    {
      aliases: ["FlashAttention"],
      tip: "FlashAttention：通过重排 attention 计算和内存访问，减少 HBM 读写，从而提升长序列注意力效率。"
    },
    {
      aliases: ["PagedAttention"],
      tip: "PagedAttention：把 KV cache 像分页内存一样管理，降低碎片并提升 LLM serving 的并发能力。"
    },
    {
      aliases: ["HBM", "L2", "SM", "Tensor Core", "Tensor Cores"],
      tip: "GPU 硬件术语：HBM 是高带宽显存，L2 是片上缓存，SM 是流式多处理器，Tensor Core 是矩阵计算单元。"
    },
    {
      aliases: ["NCCL", "NVLink"],
      tip: "NVIDIA 多卡通信相关术语。NCCL 是通信库，NVLink 是 GPU 间高速互联。"
    },
    {
      aliases: ["NVSwitch", "Hopper", "Blackwell", "SM90", "SM100"],
      tip: "NVIDIA 硬件与互联术语：Hopper/Blackwell 是 GPU 架构，SM90/SM100 是对应代际 SM，NVSwitch 用于多 GPU 高带宽互联。"
    },
    {
      aliases: ["PTX", "SASS"],
      tip: "GPU 编译结果层级：PTX 类似中间汇编，SASS 是更接近 NVIDIA GPU 机器指令的最终汇编。"
    },
    {
      aliases: ["TMA", "MMA", "WGMMA", "SIMT"],
      tip: "GPU kernel 术语：TMA 管张量内存搬运，MMA/WGMMA 管矩阵乘指令，SIMT 描述 GPU 的线程执行模型。"
    },
    {
      aliases: ["JIT"],
      tip: "JIT：Just-In-Time，即时编译。运行时根据实际形状或硬件生成优化后的代码。"
    },
    {
      aliases: ["CUDA Graph", "Inductor", "XLA", "TVM", "MLIR"],
      tip: "编译与执行优化术语：CUDA Graph 减少 launch 开销，Inductor/XLA/TVM/MLIR 属于图编译或中间表示生态。"
    },
    {
      aliases: ["Roofline", "Profiling", "Nsight Compute"],
      tip: "性能分析术语：Roofline 用计算强度和硬件上限定位瓶颈，Profiling/Nsight Compute 用于观察 kernel 的真实执行表现。"
    },
    {
      aliases: ["RMSNorm", "LayerNorm", "Softmax", "MLP", "Reduction", "Reduce"],
      tip: "常见神经网络算子：Norm 稳定激活尺度，Softmax 做概率归一化，MLP 是前馈层，Reduction/Reduce 做聚合计算。"
    },
    {
      aliases: ["MQA", "GQA", "MQA/GQA"],
      tip: "Attention 变体：MQA/GQA 通过共享或分组 Key/Value 头减少 KV cache 和带宽压力，常用于高效推理。"
    },
    {
      aliases: ["FLOPs"],
      tip: "FLOPs：浮点运算次数。常用于估算模型计算量，但实际速度还受内存带宽、kernel 和调度影响。"
    },
    {
      aliases: ["TTFT"],
      tip: "TTFT：Time To First Token，首 token 延迟。在线生成系统里衡量用户等待起始响应的关键指标。"
    },
    {
      aliases: ["TPOT"],
      tip: "TPOT：Time Per Output Token，每个输出 token 的平均生成时间，常用于衡量 decode 阶段速度。"
    },
    {
      aliases: ["QPS"],
      tip: "QPS：Queries Per Second，每秒请求数。衡量服务吞吐，但需要和延迟、质量一起看。"
    },
    {
      aliases: ["SLO", "SLA", "SLI"],
      tip: "服务可靠性术语：SLI 是指标，SLO 是目标，SLA 是对外承诺。推理服务常用它们约束延迟和可用性。"
    },
    {
      aliases: ["P50", "P95", "P99", "P95/P99"],
      tip: "延迟分位数。P99 表示 99% 请求不超过该延迟，常用来观察尾延迟。"
    },
    {
      aliases: ["OOM"],
      tip: "OOM：Out Of Memory，内存或显存不足。大模型训练和推理中常见。"
    },
    {
      aliases: ["Prefill", "Decode", "Continuous Batching"],
      tip: "LLM 推理阶段术语：Prefill 处理输入上下文，Decode 逐 token 生成输出，Continuous Batching 允许请求动态进出批次。"
    },
    {
      aliases: ["Prefix Cache", "Speculative Decoding"],
      tip: "推理加速术语：Prefix Cache 复用重复前缀的 prefill 结果，Speculative Decoding 用草稿模型提案、主模型验证来提速。"
    },
    {
      aliases: ["EAGLE", "EAGLE-2", "EAGLE-3"],
      tip: "EAGLE：一类投机解码加速方法。核心是用更便宜的特征级或草稿路径提出候选 token，再由目标模型验证。"
    },
    {
      aliases: ["QSpec"],
      tip: "QSpec：把量化和投机解码结合的路线，常用低精度路径生成 draft，再用更高精度路径验证。"
    },
    {
      aliases: ["LayerSkip", "Logit Lens", "Early Exit"],
      tip: "LayerSkip / Logit Lens / Early Exit：利用中间层已经具备预测信息这一现象，让浅层或中间层提前输出或充当自投机 draft。"
    },
    {
      aliases: ["Agent", "ReAct", "Toolformer", "Function Calling"],
      tip: "Agent 与工具调用术语：ReAct 交替推理和行动，Toolformer 学习工具使用，Function Calling 让模型输出结构化工具调用。"
    },
    {
      aliases: ["DP", "TP", "PP", "CP", "SP"],
      tip: "并行策略缩写：DP 数据并行，TP 张量并行，PP 流水并行，CP 上下文并行，SP 序列并行。"
    },
    {
      aliases: ["Sequence Parallelism", "sequence parallelism"],
      tip: "Sequence Parallelism：沿序列维度把 token 或时间步切到多张卡上，适合长序列，但 attention 和状态同步会带来跨卡通信压力。"
    },
    {
      aliases: ["BPTT", "Truncated BPTT", "truncated BPTT"],
      tip: "BPTT：Backpropagation Through Time，时序反向传播。Truncated BPTT 会按时间段截断梯度，用更低显存换取近似的长序列训练。"
    },
    {
      aliases: ["Mixed Precision", "mixed precision"],
      tip: "Mixed Precision：混合精度训练或推理。常把不同张量放在 FP32、BF16、FP16、FP8 等不同精度中，以平衡速度、显存和稳定性。"
    },
    {
      aliases: ["FSDP", "ZeRO"],
      tip: "分布式训练显存优化方法。FSDP 和 ZeRO 都会切分参数、梯度或优化器状态来降低单卡压力。"
    },
    {
      aliases: ["Megatron-LM", "Megatron Core", "Megatron", "DeepSpeed"],
      tip: "大模型训练系统：Megatron-LM/Core 常用于张量/流水/上下文并行，DeepSpeed 以 ZeRO、offload 和 pipeline runtime 闻名。"
    },
    {
      aliases: ["Checkpoint", "DataLoader", "Packing"],
      tip: "训练工程术语：Checkpoint 保存可恢复状态，DataLoader 负责输入管线，Packing 把短样本拼接以提升 token 利用率。"
    },
    {
      aliases: ["Gradient Checkpointing", "Offload", "Pipeline"],
      tip: "训练系统术语：Gradient Checkpointing 用重算换显存，Offload 把状态搬到 CPU/NVMe，Pipeline 把模型阶段化并行执行。"
    },
    {
      aliases: ["Scaling Law", "Curriculum Learning", "Judge Model"],
      tip: "训练方法术语：Scaling Law 描述规模与性能关系，Curriculum Learning 控制训练难度节奏，Judge Model 用于自动评估或偏好打分。"
    },
    {
      aliases: ["Adam", "AdamW", "EMA", "RNG", "LR"],
      tip: "训练优化术语：Adam/AdamW 是常用优化器，EMA 是指数滑动平均，RNG 是随机数状态，LR 是学习率。"
    },
    {
      aliases: ["CLIP"],
      tip: "CLIP：Contrastive Language-Image Pre-training，图文对比预训练模型，是多模态对齐和检索的经典基础。"
    },
    {
      aliases: ["OCR"],
      tip: "OCR：Optical Character Recognition，光学字符识别。用于从图片、截图或文档中读取文字。"
    },
    {
      aliases: ["VQA"],
      tip: "VQA：Visual Question Answering，视觉问答。模型根据图像内容回答问题。"
    },
    {
      aliases: ["Grounding", "Referring", "Screen Agent", "Document Understanding"],
      tip: "VLM 任务术语：Grounding/Referring 关注语言到图像区域的定位，Screen Agent 面向屏幕操作，Document Understanding 处理文档结构与文字。"
    },
    {
      aliases: ["Retrieval", "Reranking", "Reranker", "Vector Database"],
      tip: "检索系统术语：Retrieval 召回候选，Reranking 重排候选，Vector Database 存储和查询向量表示。"
    },
    {
      aliases: ["InfoNCE"],
      tip: "InfoNCE：对比学习常用目标函数，要求模型在候选集合中把正确匹配项排到更前。"
    },
    {
      aliases: ["BYOL", "MoCo", "SimCLR"],
      tip: "自监督表征学习方法。BYOL 不显式依赖负样本，MoCo 使用动量编码器和队列，SimCLR 强调大 batch 和增强。"
    },
    {
      aliases: ["BM25", "ANN", "MRR", "F1", "FID"],
      tip: "评测或检索术语：BM25 是稀疏检索方法，ANN 是近似最近邻，MRR/F1/FID 是常见指标。"
    },
    {
      aliases: ["FVD", "VBench", "PSNR", "SSIM", "LPIPS", "next-frame prediction loss", "next-frame loss"],
      tip: "视频与世界模型评测术语：这些指标常衡量重建、下一帧预测或视觉质量，但不直接等价于规划、控制或闭环任务收益。"
    },
    {
      aliases: ["Benchmark", "Ablation", "Ablation Study"],
      tip: "实验评测术语：Benchmark 是基准测试，Ablation/Ablation Study 是消融实验，用来判断某个组件或设计是否真的带来收益。"
    },
    {
      aliases: ["Calibration", "Calibration Data", "Hessian", "Outlier", "Activation"],
      tip: "量化与训练分析术语：Calibration 用代表性数据估计量化参数，Hessian 描述二阶敏感性，Outlier 是异常大值，Activation 是中间层输出。"
    },
    {
      aliases: ["Dispatch", "Shape Bucket", "Workload"],
      tip: "服务与 kernel 调度术语：Dispatch 决定请求或算子走哪条实现路径，Shape Bucket 把相近形状归桶，Workload 指实际负载分布。"
    },
    {
      aliases: ["RSSM"],
      tip: "RSSM：Recurrent State-Space Model，循环状态空间模型。Dreamer 等世界模型常用的潜动态结构。"
    },
    {
      aliases: ["World Model", "World Models", "Dreamer", "Imagined Rollout", "Rollout"],
      tip: "世界模型术语：World Model 学习环境动态，Dreamer 在潜空间 imagined rollout 中学习策略，Rollout 指向未来展开轨迹。"
    },
    {
      aliases: ["WM", "WAM", "VAM"],
      tip: "世界模型谱系缩写：WM 是 World Model，WAM 常指 World-Action Model，VAM 常指 Vision-Action Model。"
    },
    {
      aliases: ["MPC"],
      tip: "MPC：Model Predictive Control，模型预测控制。不断用模型预测未来，再滚动选择当前动作。"
    },
    {
      aliases: ["Planning", "Affordance", "Teleoperation"],
      tip: "具身智能术语：Planning 负责规划动作序列，Affordance 表示环境中可执行动作机会，Teleoperation 是人工遥操作采集示范。"
    },
    {
      aliases: ["PID"],
      tip: "PID：比例、积分、微分控制器。经典低层控制方法，常作为机器人执行层基础。"
    },
    {
      aliases: ["Sim2Real"],
      tip: "Sim2Real：Simulation-to-Reality，仿真到现实迁移。关注策略从模拟环境落到真实世界时的分布差异。"
    },
    {
      aliases: ["RL", "BC", "IL", "MDP", "POMDP"],
      tip: "控制与强化学习术语：RL 是强化学习，BC 是行为克隆，IL 是模仿学习，MDP/POMDP 是决策过程建模。"
    },
    {
      aliases: ["OOD", "ODD"],
      tip: "风险边界术语：OOD 指分布外样本，ODD 指 Operational Design Domain，即系统被设计允许工作的运行域。"
    },
    {
      aliases: ["SLAM", "ROS", "ROS2"],
      tip: "机器人系统术语：SLAM 是同步定位与建图，ROS/ROS2 是常用机器人软件框架。"
    },
    {
      aliases: ["API", "JSON", "DOM"],
      tip: "工程接口术语：API 是程序接口，JSON 是常用数据交换格式，DOM 是浏览器页面结构树。"
    }
  ];

  const SKIP_SELECTOR = [
    "a",
    "abbr",
    "code",
    "pre",
    "kbd",
    "samp",
    "script",
    "style",
    "textarea",
    ".arithmatex",
    ".MathJax",
    ".highlight",
    ".term-tip",
    ".md-nav",
    ".md-search"
  ].join(",");

  const termTips = new Map();
  let tooltipElement = null;
  let activeTerm = null;
  let floatingTooltipsReady = false;

  for (const group of TERM_GROUPS) {
    for (const alias of group.aliases) {
      termTips.set(alias, group.tip);
    }
  }

  const escapeRegex = (value) => value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const aliases = Array.from(termTips.keys()).sort((a, b) => b.length - a.length);
  const pattern = new RegExp("(^|[^A-Za-z0-9_])(" + aliases.map(escapeRegex).join("|") + ")(?![A-Za-z0-9_])", "g");

  function shouldSkipTextNode(node) {
    const parent = node.parentElement;
    return !parent || parent.closest(SKIP_SELECTOR);
  }

  function makeTermElement(text, tip) {
    const span = document.createElement("span");
    span.className = "term-tip term-auto";
    span.dataset.tip = tip;
    span.textContent = text;
    span.tabIndex = 0;
    span.setAttribute("role", "term");
    span.setAttribute("aria-label", text + "：" + tip);
    return span;
  }

  function getTooltipElement() {
    if (tooltipElement) {
      return tooltipElement;
    }

    tooltipElement = document.createElement("div");
    tooltipElement.className = "term-tooltip-popover";
    tooltipElement.setAttribute("role", "tooltip");
    tooltipElement.setAttribute("aria-hidden", "true");
    document.body.appendChild(tooltipElement);
    return tooltipElement;
  }

  function positionTooltip(term, tooltip) {
    const termRect = term.getBoundingClientRect();
    const tooltipRect = tooltip.getBoundingClientRect();
    const margin = 10;
    const gap = 10;

    let left = termRect.left + termRect.width / 2 - tooltipRect.width / 2;
    left = Math.max(margin, Math.min(left, window.innerWidth - tooltipRect.width - margin));

    let top = termRect.top - tooltipRect.height - gap;

    if (top < margin) {
      top = termRect.bottom + gap;
    }

    top = Math.max(margin, Math.min(top, window.innerHeight - tooltipRect.height - margin));

    tooltip.style.left = left + "px";
    tooltip.style.top = top + "px";
  }

  function showTooltip(term) {
    const tip = term.dataset.tip;

    if (!tip) {
      return;
    }

    const tooltip = getTooltipElement();
    activeTerm = term;
    tooltip.textContent = tip;
    tooltip.dataset.visible = "true";
    tooltip.setAttribute("aria-hidden", "false");
    positionTooltip(term, tooltip);
  }

  function hideTooltip(term) {
    if (term && activeTerm && term !== activeTerm) {
      return;
    }

    const tooltip = getTooltipElement();
    activeTerm = null;
    tooltip.dataset.visible = "false";
    tooltip.setAttribute("aria-hidden", "true");
  }

  function setupFloatingTooltips() {
    if (floatingTooltipsReady) {
      return;
    }

    floatingTooltipsReady = true;
    getTooltipElement();

    document.addEventListener("mouseover", (event) => {
      const term = event.target.closest(".term-tip[data-tip]");

      if (term) {
        showTooltip(term);
      }
    });

    document.addEventListener("mouseout", (event) => {
      const term = event.target.closest(".term-tip[data-tip]");

      if (term && !term.contains(event.relatedTarget)) {
        hideTooltip(term);
      }
    });

    document.addEventListener("focusin", (event) => {
      const term = event.target.closest(".term-tip[data-tip]");

      if (term) {
        showTooltip(term);
      }
    });

    document.addEventListener("focusout", (event) => {
      const term = event.target.closest(".term-tip[data-tip]");

      if (term) {
        hideTooltip(term);
      }
    });

    window.addEventListener("scroll", () => {
      if (activeTerm) {
        positionTooltip(activeTerm, getTooltipElement());
      }
    }, true);

    window.addEventListener("resize", () => {
      if (activeTerm) {
        positionTooltip(activeTerm, getTooltipElement());
      }
    });

    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        hideTooltip();
      }
    });
  }

  function decorateTextNode(node) {
    const text = node.nodeValue;
    const fragment = document.createDocumentFragment();
    let lastIndex = 0;
    let match;

    pattern.lastIndex = 0;

    while ((match = pattern.exec(text)) !== null) {
      const prefix = match[1] || "";
      const term = match[2];
      const termStart = match.index + prefix.length;
      const termEnd = termStart + term.length;
      const tip = termTips.get(term);

      if (!tip) {
        continue;
      }

      if (termStart > lastIndex) {
        fragment.appendChild(document.createTextNode(text.slice(lastIndex, termStart)));
      }

      fragment.appendChild(makeTermElement(term, tip));
      lastIndex = termEnd;

      if (pattern.lastIndex === match.index) {
        pattern.lastIndex += 1;
      }
    }

    if (lastIndex === 0) {
      return;
    }

    if (lastIndex < text.length) {
      fragment.appendChild(document.createTextNode(text.slice(lastIndex)));
    }

    node.parentNode.replaceChild(fragment, node);
  }

  function applyTermTooltips() {
    setupFloatingTooltips();

    const roots = document.querySelectorAll(".md-content .md-typeset");

    for (const root of roots) {
      const textNodes = [];
      const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
        acceptNode(node) {
          if (!node.nodeValue || !node.nodeValue.trim() || shouldSkipTextNode(node)) {
            return NodeFilter.FILTER_REJECT;
          }

          pattern.lastIndex = 0;
          return pattern.test(node.nodeValue) ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT;
        }
      });

      while (walker.nextNode()) {
        textNodes.push(walker.currentNode);
      }

      for (const node of textNodes) {
        if (!shouldSkipTextNode(node)) {
          decorateTextNode(node);
        }
      }
    }
  }

  if (window.document$) {
    window.document$.subscribe(applyTermTooltips);
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", applyTermTooltips);
  } else {
    applyTermTooltips();
  }
})();
