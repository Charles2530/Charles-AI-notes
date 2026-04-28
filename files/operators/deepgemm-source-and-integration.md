# DeepGEMM 源码与接入附录

这页是 [DeepGEMM 解读](deepgemm-fp8-gemm-and-mega-moe.md) 的工程附录，重点放在源码阅读、API 地图、调用路径和接入检查。主页面负责建立系统判断，这页负责帮助你真的读仓库和评估能不能接进自己的服务链路。

## 一、源码阅读顺序

第一次读 `DeepGEMM` 不建议从最底层 JIT kernel 开始。更稳的顺序是从“外部接口和测试”向下钻：

1. `README.md`：先看官方定义、支持架构、核心能力和 benchmark 口径；
2. `deep_gemm/__init__.py`：确认 Python 层导出了哪些能力；
3. `tests/test_fp8_fp4.py`：看普通低精度 dense GEMM 怎么调用；
4. `tests/test_layout.py`：看 scale、layout、transpose 相关假设；
5. `tests/test_mega_moe.py`：看 MoE 和系统级路径如何组织；
6. `deep_gemm/utils/`、`deep_gemm/testing/`：看 benchmark、数据生成和校验方式；
7. `csrc/python_api.cpp`：看 Python 到 C++ 的桥；
8. `csrc/apis/`：看不同功能族的 C++ API；
9. `csrc/jit/`、`csrc/jit_kernels/`：最后再看 JIT 与 kernel 生成细节。

这个顺序的好处是：先知道“外部用户以为什么方式使用它”，再理解“内部为什么必须这样设计”。

## 二、关键 API 地图

从功能上看，可以先把 API 分成五类：

| API 类别 | 关注点 | 适合先看什么 |
| --- | --- | --- |
| Dense GEMM | 普通 FP8/FP4/BF16 GEMM | shape 约束、scale layout、transpose |
| Grouped GEMM | MoE expert 计算 | M 轴分组、expert token 分布 |
| Layout / Scale | 低精度数据准备 | scale factor 排布、TMA-friendly layout |
| Attention / MQA | 服务态注意力热点 | decode 访问模式、paged/masked 路线 |
| Mega MoE | 系统级融合路径 | dispatch、combine、通信和计算边界 |

读 API 时不要只看参数类型。更重要的是问：

1. 哪些 layout 被显式暴露；
2. 哪些 shape 被假设稳定；
3. scale factor 是否在接口里有特殊格式；
4. API 是否服务 dense、grouped、masked 还是 fused path；
5. 这个接口在 prefill 还是 decode 中更可能是热路径。

## 三、调用路径怎么追

一个常见主路径可以写成：

```text
Python test / user call
  -> deep_gemm Python wrapper
  -> C++ binding
  -> API dispatch
  -> JIT config / cache lookup
  -> generated CUDA kernel
  -> runtime launch
```

建议追三条链：

1. **普通 FP8 dense GEMM**：理解 scale layout、JIT 和 kernel launch 的基本链路；
2. **layout / scale 变换**：理解为什么低精度不是 dtype 改一下；
3. **Mega MoE / grouped GEMM**：理解 token 分组、expert shape、masked layout 和系统融合。

`python_api.cpp` 很关键，因为它通常暴露了 Python 世界和 C++/CUDA 世界之间的真实边界。很多“接口为什么长这样”的答案，都能在绑定层和测试用例里找到。

## 四、接入真实系统前检查什么

接入前先做五项硬检查：

1. **硬件检查**：目标环境是否主要是 H100/H800/B100 等支持路径；
2. **Shape 检查**：真实请求的 M/N/K、expert token 分布是否集中；
3. **Layout 检查**：上游能否提供 DeepGEMM 需要的 scale 和数据布局；
4. **Runtime 检查**：JIT cache、CUDA Graph、fallback、灰度和监控是否准备好；
5. **端到端检查**：dispatch、permute、combine、dequant、同步是否全部计入收益。

如果这些前提不满足，直接替换 kernel 很可能只会得到漂亮的 microbenchmark，而不是线上收益。

## 五、最常见 Benchmark 误读

`DeepGEMM` 这类库最容易被误读的地方不是实现，而是结果解释。

| 误读 | 更稳的解释 |
| --- | --- |
| 单 shape 快，所以所有 GEMM 都该换 | 只说明该 shape 命中假设 |
| GEMM 快，所以 MoE serving 一定快 | dispatch/combine、路由和通信也可能主导 |
| JIT 快，所以没有运维成本 | 首次编译、缓存失效和版本复现都要治理 |
| FP8 快，所以端到端成本下降 | scale、layout、数值补偿和 fallback 都要算 |

上线前应做 replay，而不是只做 synthetic benchmark。真实流量里的长尾 shape、租户隔离、LoRA、expert 热点和 fallback 才会决定收益是否稳定。

## 六、灰度上线顺序

一个更稳妥的上线顺序是：

1. 保留原有 `cuBLASLt/Triton` 路径作为 fallback；
2. 在离线 replay 中按 shape bucket 对比端到端 latency；
3. 单独测 prefill 和 decode；
4. 记录 JIT compile latency 和 cache hit rate；
5. 先替换最稳定的 dense/grouped GEMM 热点；
6. 再小流量灰度 masked decode 或 MoE path；
7. 最后才考虑 Mega MoE 这类更深融合。

灰度指标至少包括：P50/P95/P99、TTFT/TPOT、JIT cache miss、fallback rate、错误率、显存峰值、expert token 分布和端到端单位请求成本。

## 七、最小伪代码

```python
def moe_expert_compute(tokens, expert_ids, runtime):
    groups = group_by_expert(tokens, expert_ids)
    shape_key = make_shape_key(groups)

    if runtime.deepgemm_enabled and runtime.jit_cache.ready(shape_key):
        return runtime.deepgemm.grouped_gemm(groups)

    return runtime.fallback.grouped_gemm(groups)
```

这段伪代码强调两个判断：先按 expert 组织 token，再根据 shape、JIT cache 和 fallback 状态选择路径。真实工程里，端到端收益往往取决于这个选择逻辑，而不只是底层 kernel 的峰值 TFLOPS。
