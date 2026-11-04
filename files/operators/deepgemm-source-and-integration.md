# 沿一次 DeepGEMM 调用阅读源码

先选一个普通低精度GEMM，确认逻辑shape和输出，再进入grouped或MegaMoE。这样能够把数据准备、API选择、代码生成与kernel计时分别定位。本文锁定commit `5081f851`，不假定其他版本具有相同导出。

## Python入口告诉我们哪些名字真实存在

`deep_gemm/__init__.py`从 `_C`导出普通、grouped、attention及布局接口，并从 `.mega`导出MegaMoE相关入口。模块末尾调用 `_C.init(...)`初始化扩展，还注册关闭回调。仅阅读Python导出能确认入口存在，不能推断每个C++函数内部的执行细节。

普通矩阵乘可先看 `fp8_gemm_nt`及更一般的FP8/FP4组合接口；专家前向看M-grouped contiguous或masked；专家权重反向则涉及K-grouped路径。选入口时以当前版本文档与真实签名为准。

## 在调用前写清四份元数据

分别记录A、B逻辑shape与存储stride，数据格式，scale的shape/格式，以及输出dtype与累加约定。对nt路径，A为M×K、B为N×K。Scale不是随便附加的同shape张量，需要遵守架构对应布局。

先用一个小输入计算高精度参考，再用相同量化数据反量化得到参考值。比较低精度kernel与后者，有助于把量化误差和执行误差分开。不要在一开始就加入多机路由，把错误来源叠在一起。

## 将准备成本和kernel成本分别计时

第一次调用可能涉及JIT与加载。稳态基准先预热，再测已经准备好的输入；完整层基准则将每次实际发生的cast、scale变换和转置纳入。若权重只在启动时转换，其成本应放在加载时间，不必重复算进每个请求。

README提供 `DG_PRINT_CONFIGS` 观察选择配置，`DG_JIT_DEBUG`与相关选项帮助查看编译命令、PTXAS输出或汇编。调试开关可能影响输出和性能，正式计时前应固定设置。

## 从普通GEMM扩到专家批量

先让两个专家有相同N/K、不同M，核对每段token顺序、段对齐和输出回填。再加入真实路由的置换与加权合并。Masked模式还要测试有效计数为0、很小和接近容量的情况。

MegaMoE另有对称内存、多进程和权重布局前提。该版本README给出缓冲建立、权重变换、填入x/scale/topk信息、执行的顺序；进一步接入应运行其对应测试，而不能将这一流程示意当作已验证部署脚本。

## 接入失败时缩小到接口边界

若小矩阵结果不对，先检查方向、scale与packing；若小矩阵正确而专家输出错误，检查段边界和反置换；若完整结果正确但慢，再比较准备开销、padding、JIT与实际kernel。

保留固定输入、版本与输出误差，能让问题回到可重复调用。本文没有运行DeepGEMM GPU测试，说明的是已查证的入口与阅读路径。

![DeepSeek-V3 FP8 framework](../assets/images/paper-deep-dives/technical-reports/deepseek-v3/figure-6-fp8-framework.png)

<small>图源：[DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)。原图表达 FP8 训练由 scale、累加、通信、存储和高精保护共同组成；读图重点是：读 DeepGEMM 源码时先追 scale layout、JIT dispatch、grouped GEMM 和端到端调用边界。它不能证明 DeepGEMM 接入你的 serving 栈一定变快，只说明源码阅读要围绕低精度系统契约展开。</small>

## 对照源码与延伸阅读

- [Python导出与初始化代码](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/deep_gemm/__init__.py)
- [接口前提、调试选项与调用示意](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/README.md)
