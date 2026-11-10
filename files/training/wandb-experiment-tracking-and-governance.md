# 用 W&B 记录一次可解释的参数更新

实验平台能保存配置、曲线和产物关联，训练循环仍需自己定义每个数是什么意思。先用两条数据做五次更新，再把本地日志接到W&B，便能看清step、loss分母与恢复状态。

## 先生成不依赖平台的记录

设输入 `[1,2]`、目标 `[2,4]`，预测为wx，初始w=1，loss为两条平方误差的一半再平均。下面仅需Python标准库。

```python
xs, ys = [1.0, 2.0], [2.0, 4.0]
w, lr = 1.0, 0.1
records = []
for update in range(5):
    errors = [w * x - y for x, y in zip(xs, ys)]
    loss = sum(e * e for e in errors) / (2 * len(xs))
    grad = sum(e * x for e, x in zip(errors, xs)) / len(xs)
    records.append({'update': update, 'loss/before': loss,
                    'parameter/before': w, 'gradient': grad})
    w -= lr * grad
print(records[0], w)
```

第一条应为loss=1.25、w=1、gradient=-2.5；第一次更新后w=1.25，五次后约1.762695。`before`说明loss和参数都来自更新前状态，避免把两个时刻拼成一条日志。

## 再把同一记录送进run

已有W&B SDK时，可以用offline模式记录这组数据，不需要本例上传。以下是接在前一段之后的示例接口，需按已安装SDK验证。

```python
import wandb
with wandb.init(project='training-example', mode='offline',
                config={'lr': lr, 'updates': 5, 'data_version': 'two_points_v1'}) as run:
    for row in records:
        run.log({k: v for k, v in row.items() if k != 'update'}, step=row['update'])
    run.summary['parameter/final'] = w
```

配置记录实验条件，逐step日志记录过程，summary记录最终摘要。平台不负责替这段代码定义loss，也不会自动保存optimizer或sampler状态。官方入口见[W&B文档](https://docs.wandb.ai/)。

![W&B quickstart 官方效果图](../assets/images/tools/wandb/wandb-quickstart-runs.png){ width="860" }

<small>图源：[W&B Quickstart](https://docs.wandb.ai/models/quickstart)。官方示例记录 simulated accuracy/loss，进入 W&B 后可以看到每个 run 的曲线和 runs 列表。</small>

## 多卡曲线必须先统一分子和分母

若rank0的loss总和2、有效token2，rank1总和18、token6，全局token平均为20/8=2.5。平均两个局部均值会得到2，表示另一个量。

所有rank参与必要统计规约，通常只由rank0写主run。Collective不能只由rank0单独调用，否则可能等待其他rank。梯度累积多个microbatch才更新一次时，主横轴可用optimizer update，另外记录消费token与样本量。

## Artifact关联不等于训练快照完整

可以把checkpoint与数据版本作为输入输出记录，但完整续训内容仍由训练框架产生。仅记录一个可变文件路径，无法保证未来读到相同字节；应保存固定版本或校验标识。

恢复日志与恢复训练是不同协议。模型从step400恢复，日志却已到419时，需要按SDK规则处理重叠step，或建立记录父运行与checkpoint的新run。仪表盘看起来连续，不能证明输入与优化器状态连续。

## 给每一次比较留下可追溯的身份

一次实验至少需要四个身份：代码版本、数据版本、配置版本和运行状态。学习率或模型结构变化时，不要只改 run 名称；把改变的字段写进配置，并把评测输入和 checkpoint 关联起来。这样回看一条曲线时，能知道它来自哪一份代码和哪一批数据。

离线记录适合先验证日志字段和训练循环，在线记录适合团队查看和比较。两者都要保存相同的字段名与 step 定义，不能因为切换记录模式就改变 loss 分母或横轴。W&B 文档中的 run、config、summary 和 artifact 入口可以组织这些对象，实际续训仍要由训练框架保存优化器、随机数和 sampler 状态。

## 把一次实验连接到 server_monitor 和 PaperPilot

如果训练由 `server_monitor` 提交，W&B run 只是其中一个观察入口。建议给每次实验生成一份稳定的实验身份：

```yaml
experiment_id: exp-2026-0110-017
server_job_id: sm-88421
wandb_run_id: abc123
paperpilot_note_id: paperpilot://notes/...
code_commit: 3adf61e1
data_version: corpus-v4
checkpoint_uri: s3://.../step-40000
```

`server_job_id` 负责追踪队列、主机、GPU 和容器日志，`wandb_run_id` 负责曲线与 artifact，PaperPilot 记录假设、论文依据和结论。三者都指向同一个 `experiment_id`，回看时才能从一条异常曲线找到实际 stderr，再找到为什么改变配置的论文或实验记录。

不要把服务器路径或 dashboard URL 当作唯一身份。路径可能因迁移改变，run 名称也可能被重用。固定 ID、代码 commit、数据版本和 checkpoint manifest 才能抵抗环境变化。

## 指标命名要先写数据字典

`loss` 这个名字不够。训练语言模型时至少区分 `loss/sum`、`loss/valid_token_mean`、`tokens/consumed` 和 `optimizer_step`；多卡时还要说明它们是 rank 局部、global reduce 还是只由 rank0 记录。

一个最小数据字典可以写成

| 名称 | 分子/分母 | 更新频率 | 聚合方式 |
| --- | --- | --- | --- |
| `train/loss` | 有效 token 的 loss / 有效 token 数 | optimizer step | 全局 token 加权平均 |
| `train/tokens` | 本次消费 token | optimizer step | rank 求和 |
| `system/step_ms` | 完成一次更新的墙钟时间 | optimizer step | p50/p95 或均值 |
| `system/gpu_mem` | 峰值显存 | microbatch | rank max |
| `eval/accuracy` | 正确样本 / 评测样本 | eval run | 固定数据集 |

数据字典会阻止不同 run 用同一个名字表达不同分母。仪表盘美观不能弥补定义不一致。

## 告警应该对应可执行动作

告警不应只是“loss 很高”。一个可用规则要包含指标、窗口、阈值、责任人和动作：

```yaml
- name: loss_nan
  condition: train/loss is not finite for 3 updates
  action: stop_job_and_preserve_checkpoint
- name: dataloader_stall
  condition: batch_ready_p95 > 2 * step_ms for 5 minutes
  action: capture_loader_trace
- name: checkpoint_uncommitted
  condition: checkpoint_tmp_age > 10 minutes
  action: mark_run_degraded_and_check_storage
```

`server_monitor` 可以负责资源和进程级事件，训练脚本负责 loss、overflow、数据和 checkpoint 事件。两边的时间戳要使用同一时区或明确使用 UTC，否则告警发生顺序无法回放。

告警触发后保留最小现场：最近若干 step 的配置、rank 状态、GPU 利用率、数据 ID、随机状态和未提交的 checkpoint 文件。只发一条消息而不保留现场，无法支持之后的诊断。

## Artifact 要有 manifest 和不可变版本

一个 checkpoint 目录存在，并不表示它能恢复。为每个可交付 artifact 保存 manifest：

```json
{
  "artifact": "checkpoint",
  "global_step": 40000,
  "code_commit": "3adf61e1",
  "data_version": "corpus-v4",
  "files": {
    "model_rank0": {"sha256": "...", "bytes": 123},
    "optimizer_rank0": {"sha256": "...", "bytes": 456}
  },
  "world_size": 8,
  "committed": true
}
```

只有 manifest 完整、所有分片校验通过且数据/优化器状态一致时，才把版本标记为 `committed`。上传中的临时目录和校验失败的版本必须能被 dashboard 区分，不能让最新路径自动指向半成品。

Artifact 还要有保留策略。只保留最后一个 checkpoint 可能失去出现异常前的回滚点；保留所有 checkpoint 会增加存储成本。可以按最近若干步、最佳验证结果、里程碑和父运行各保留一份，并把删除决策写入日志。

## 恢复 run 时不要伪造连续曲线

从 step 400 恢复后重新执行 step 401，W&B 中可能会出现与原 run 重叠的 step。最清楚的方式是新建 child run，并记录 `parent_run_id`、恢复 checkpoint 和重跑范围；若平台支持明确的 resume 语义，也要确认其 step、summary 和 artifact 行为。

曲线连续只是展示效果，不能证明训练连续。报告中应区分：原 run 已完成的更新、恢复后重做的更新、被丢弃的日志和最终采用的 checkpoint。重复 step 还可能导致自动 early stopping 或比较脚本产生误判。

## 用记录回答一个具体比较

比较学习率时固定数据、初始权重和预算；比较吞吐时记录有效token、设备与等待时间；比较质量时保留同一评测输入的逐样本结果。工具提供的是可检索的证据载体，结论仍来自这些条件是否可比。

一个实用的回归做法是保留一条小数据 smoke run。它不追求模型效果，只检查配置是否写入、step 是否连续、恢复后是否出现重复日志、artifact 是否指向固定版本。smoke run 通过以后，再把同一套字段带到多卡和长训练中，减少“仪表盘正常但实验无法复现”的情况。


## 继续读

继续看[集群运行与实验管理](cluster-operations-and-experiment-management.md)和[研究闭环](research-loop-from-hypothesis-to-evidence-1123-2351.md)。

## 对照源码与延伸阅读

- [实际W&B记录位置与训练状态边界](https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/train.py)

**来源与改编说明**：记录对象和 dashboard 用法参考 [W&B 文档](https://docs.wandb.ai/)及其 Quickstart；两点数据的梯度和数值为本站教学算例，不能当成真实训练性能结果。
