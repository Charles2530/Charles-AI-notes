# FP8 矩阵乘里的数据、尺度和累加

FP8用8个bit保存浮点数，但不同格式的指数位和尾数位分配不同，范围与精度也不同。服务系统要同时决定数据格式、scale粒度、累加方式和硬件路径。一个tensor标记为FP8，还不足以说明它会怎样参与计算。

## 从一块输入到一次矩阵乘

先约定 $A\approx s_A A_q$、$B\approx s_B B_q$，即s表示反量化尺度。若两块各用一个scale，则

$$
AB\approx s_As_B(A_qB_q).
$$

量化时先除以s，再转换到FP8可表示值；矩阵乘通常把低精度操作数送入适用的矩阵指令，在选定精度下累加，最后恢复尺度。某些库保存的是这里s的倒数，读API时必须核对，不能只看变量名scale。

用十进制教学数说明尺度关系，设 $A_q=[2,4]$、$B_q=[3,1]^\top$，$s_A=0.5,s_B=0.25$。整数样式的点积为10，恢复结果为1.25。实际FP8操作数可以带小数，这个例子只演示scale乘回的位置。

## 分块尺度会改变归约结构

若K方向分成两块，每块有不同尺度，结果为

$$
C=\sum_g s_{A,g}s_{B,g}(A_{q,g}B_{q,g}).
$$

不能先把所有未缩放部分和加起来，最后随便乘一个scale。Kernel需要把scale读取、部分累加和布局一起组织。因此细粒度scale既是数值选择，也是执行接口。

DeepGEMM锁定版本的README明确区分SM90与SM100的scale布局。前者使用FP32尺度，后者相应路径要求打包UE8M0尺度；左操作数scale还需符合其转置和TMA对齐要求。支持某种FP8数据格式，并不自动意味着现有scale张量就可直接传入。

![DeepSeek-V3 FP8 mixed precision framework](../assets/images/paper-deep-dives/technical-reports/deepseek-v3/figure-6-fp8-framework.png){ width="920" }

<small>图源：[DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)，Figure 6。原图表达大模型训练中的 FP8 mixed precision framework。这里用它说明：FP8 是低精度计算、高精度保护、scale 与通信策略的组合，不是单一开关。</small>

![DeepSeek-V3 FP8 accumulation](../assets/images/paper-deep-dives/technical-reports/deepseek-v3/figure-7-fp8-accumulation.png){ width="860" }

<small>图源：[DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)，Figure 7。原图表达 FP8 quantization、scale、accumulation 和高精度处理的关系。这里用它说明：FP8 工程难点在于计算路径和 scale 路径一起设计。</small>

## 静态范围和动态范围各花什么成本

静态scale可在校准时决定，减少请求期统计，但输入范围变化可能增加截断或下溢。动态scale依据当前张量统计范围，能适应当前输入，却增加归约、转换与元数据操作。小batch的短kernel里，这些开销可能占较大比例。

例如一个教学流水线原先GEMM花10个时间单位，其余操作花10。改FP8后GEMM变成6，量化与布局多花3，整条路径从20降到19。算子内部节省40%，端到端只节省5%。这组数字用于拆账，不是任何GPU的实测。

![FP8 training loss curves](../assets/images/paper-figures/foundations/fp8-formats-figure-1-training-loss.png){ width="630" }

<small>图源：[FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)，Figure 1。原图用多类模型训练曲线说明 FP8 路线可以接近高精度基线。这里用它说明：格式定义本身不能证明可用，必须回到任务曲线和质量回归。</small>

## 混合精度如何由误差定位得到

先固定输入，对照每层浮点与FP8输出。若某层误差显著放大，再尝试只恢复该层或某段计算的高精度，并观察任务质量是否恢复。归一化、softmax或输出头常值得单独检查，但不应预先宣布它们在所有模型里最敏感。

同样，保留高精度模块可能引入转换与阻断融合。最终方案要比较完整计算路径，同时保留质量与延迟结果。FP8权重、FP8激活、FP8 KV是三个独立配置，修改其中一个时不要把另外两个默认算入收益。

## 从源码确认收益会落在哪里

DeepGEMM README明确要求输入转置和FP8转换由调用方负责，辅助PyTorch函数可能较慢。因此计时至少分为“已准备好的矩阵乘”和“包含转换的完整层”两种边界。框架加载成功只能证明某个路径能执行，trace中实际kernel与输入布局才说明执行了什么。


## 对照源码与延伸阅读

- [DeepGEMM 固定版本的格式、布局和调用方职责](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/README.md)
