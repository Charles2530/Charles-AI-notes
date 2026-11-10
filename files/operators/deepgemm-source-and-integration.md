# 沿一次 DeepGEMM 调用阅读源码

读 DeepGEMM 时，最容易迷失在 kernel、配置和专家路由之间。可以先拿一次普通 FP8 GEMM 调用做锚点：输入从哪里来，scale 怎样排布，Python 入口选了哪条路径，最后哪个 kernel 被计时。普通路径的 shape 和输出对上以后，再进入 grouped 或 MegaMoE，错误才不会一下叠成一团。

本文锁定 commit `5081f851`。下面说到的导出名和 README 前提都只对这个版本负责，其他版本要重新核对。

## Python 入口先确认哪些名字真的存在

`deep_gemm/__init__.py`从 `_C`导出普通、grouped、attention及布局接口，并从 `.mega`导出MegaMoE相关入口。模块末尾调用 `_C.init(...)`初始化扩展，还注册关闭回调。仅阅读Python导出能确认入口存在，不能推断每个C++函数内部的执行细节。

普通矩阵乘可先看 `fp8_gemm_nt`及更一般的FP8/FP4组合接口；专家前向看M-grouped contiguous或masked；专家权重反向则涉及K-grouped路径。选入口时以当前版本文档与真实签名为准。

## 调用前把四份元数据写在纸上

分别记录A、B逻辑shape与存储stride，数据格式，scale的shape/格式，以及输出dtype与累加约定。对nt路径，A为M×K、B为N×K。Scale不是随便附加的同shape张量，需要遵守架构对应布局。

先用一个小输入计算高精度参考，再用相同量化数据反量化得到参考值。比较低精度kernel与后者，有助于把量化误差和执行误差分开。不要在一开始就加入多机路由，把错误来源叠在一起。

## 把准备成本和 kernel 成本分开计时

第一次调用可能涉及JIT与加载。稳态基准先预热，再测已经准备好的输入；完整层基准则将每次实际发生的cast、scale变换和转置纳入。若权重只在启动时转换，其成本应放在加载时间，不必重复算进每个请求。

README提供 `DG_PRINT_CONFIGS` 观察选择配置，`DG_JIT_DEBUG`与相关选项帮助查看编译命令、PTXAS输出或汇编。调试开关可能影响输出和性能，正式计时前应固定设置。

## 从普通 GEMM 扩到专家批量

先让两个专家有相同N/K、不同M，核对每段token顺序、段对齐和输出回填。再加入真实路由的置换与加权合并。Masked模式还要测试有效计数为0、很小和接近容量的情况。

MegaMoE另有对称内存、多进程和权重布局前提。该版本README给出缓冲建立、权重变换、填入x/scale/topk信息、执行的顺序；进一步接入应运行其对应测试，而不能将这一流程示意当作已验证部署脚本。

## 先把一次调用冻结成 fixture

在进入真实路由前，可以为一个普通 GEMM 保存一份最小 fixture：

```yaml
commit: 5081f851
path: fp8_gemm_nt
A: {shape: [M, K], dtype: fp8, stride: [...], scale_shape: [...]}
B: {shape: [N, K], dtype: fp8, stride: [...], scale_shape: [...]}
output: {dtype: bf16, accumulator: fp32}
reference: dequantized_matmul
checks: [max_abs_error, max_rel_error, finite, warmup, p50, p95]
```

fixture 的作用是让“接口变了”“scale 变了”和“kernel 变慢了”分开。升级 commit 或切换 SM90/SM100 路径时，先重跑同一份 fixture，再决定是否需要重新解释结果。它不是 DeepGEMM 官方配置，也不代表某张卡的最佳参数。

## 接入失败时先缩小接口边界

若小矩阵结果不对，先检查方向、scale与packing；若小矩阵正确而专家输出错误，检查段边界和反置换；若完整结果正确但慢，再比较准备开销、padding、JIT与实际kernel。

保留固定输入、版本和输出误差，问题才会回到可重复调用。本文没有运行 DeepGEMM GPU 测试，说明的是已查证的入口与阅读路径，不能把它当成某台机器上的性能结论。

![DeepSeek-V3 FP8 framework](../assets/images/paper-deep-dives/technical-reports/deepseek-v3/figure-6-fp8-framework.png)

<small>图源：[DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)。原图表达 FP8 训练由 scale、累加、通信、存储和高精保护共同组成；读图重点是：读 DeepGEMM 源码时先追 scale layout、JIT dispatch、grouped GEMM 和端到端调用边界。它不能证明 DeepGEMM 接入你的 serving 栈一定变快，只说明源码阅读要围绕低精度系统契约展开。</small>

## 代码入口和延伸阅读

- [Python导出与初始化代码](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/__init__.py)
- [接口前提、调试选项与调用示意](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/README.md)

## 源码接入的最小验证顺序

1. 读取 Python 导出，确认入口真实存在。
2. 写出逻辑 shape、stride、scale layout 和输出 dtype。
3. 用小输入对照高精度参考，区分方向、packing 和数值误差。
4. 分开测 JIT、准备、纯 kernel 与完整专家层。
5. 只有结果和边界稳定后，才加入真实路由、padding、combine 和多进程。

**来源与改编说明**：[DeepGEMM Python export](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/__init__.py) 与固定版本 README 共同决定本页的接口边界。自测时记录一次成功调用的所有张量 shape，并标出哪些由调用方准备。等这些字段和误差都稳定以后，再去比较真实路由或端到端收益。

下一步可看[FP8 训练与 optimizer kernel](fp8-training-and-optimizer-kernels.md)理解 scale 与累加状态，再看[DeepGEMM FP8 GEMM 与 MegaMoE](deepgemm-fp8-gemm-and-mega-moe.md)对照 grouped 路径的负载和通信边界。若要从源码继续向下追，先保持 commit、shape 和 scale layout 不变。
