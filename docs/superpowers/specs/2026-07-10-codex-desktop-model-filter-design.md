# Codex Desktop 模型过滤持久修复设计

## 文档状态

- 设计日期：2026-07-10
- 状态：用户已确认
- 适用版本：`/Applications/ChatGPT.app` 26.707.31428
- 适用平台：macOS

## 背景与根因

Codex CLI 与 Desktop 内置 app-server 都能列出 `gpt-5.6-sol`、`gpt-5.6-terra` 和 `gpt-5.6-luna`。三个模型在 `model/list` 中均为 `hidden: false`，其中 Sol 为默认模型；用户的 `~/.codex/config.toml` 也已经设置 `model = "gpt-5.6-sol"`。

Desktop 前端还会读取 Statsig 动态配置 `107580212`。远端配置当前返回旧 `available_models` 白名单并设置 `use_hidden_models = true`，随后前端执行以下筛选：

```js
useHiddenModels
  ? availableModels.has(model.model)
  : !model.hidden
```

LevelDB 缓存补丁能暂时改变 Statsig 值，但 App 启动后的 `values_updated` 网络事件会直接写入新响应。`receivedAt` 和内层 `time` 不构成防刷新锁，因此未来时间戳不能提供持久性。

## 目标

1. provider/app-server 返回的非隐藏模型始终保留在 Desktop 模型列表中。
2. Statsig 白名单中的隐藏模型继续可见，避免缩小现有模型集合。
3. 修复在普通退出、重启及 Statsig 重新拉取后继续有效。
4. 不读取、修改或输出 `~/.codex/auth.json`、API key 或 token。
5. 不清空 Codex App 数据目录，不修改用户的 `config.toml`。
6. 修改前提供完整、可验证的 App 备份和回滚路径。

## 非目标

1. 不伪造模型 slug，不把 5.6 映射成旧模型名。
2. 不拦截或屏蔽全部 Statsig 网络请求。
3. 不硬编码只允许三个 5.6 模型。
4. 不关闭 Codex 的自动更新机制；App 更新后允许脚本重新检测并重新应用兼容补丁。
5. 不保证 OpenAI Developer ID、notarization 或受 Team ID 约束的系统权限在本地重签名后保持原状。

## 方案比较

### 方案 A：`config.toml` 与 `model_catalog_json`

这些配置能控制默认模型、provider 和 app-server 模型目录，但前端的 Statsig 筛选发生在 `model/list` 之后。当前三个 5.6 已经是 `hidden: false`，继续调整 catalog 不能绕过旧白名单。

### 方案 B：缓存补丁并阻断 Statsig

保留现有 App 签名，但需要长期拦截 Statsig 网络请求，会同时冻结其他功能开关和实验配置，且端点变化后容易失效。

### 方案 C：修改前端模型过滤条件

直接修复错误的客户端交集逻辑。远端配置仍可正常刷新，provider 返回的可见模型不会再被过期白名单删除。

采用方案 C。

## 补丁语义

当前压缩代码中的目标表达式为：

```js
l?t.has(n.model):!n.hidden
```

其中 `l` 表示启用 Statsig 隐藏模型白名单，`t` 是 `available_models` 集合，`n.hidden` 来自 app-server。

使用等长替换改为：

```js
!n.hidden||t.has(n.model) 
```

新条件的语义是“app-server 标记为可见，或模型位于 Statsig 白名单”。末尾空格用于保持字节长度不变，避免改变 ASAR 文件偏移。

脚本必须要求旧表达式在整个 ASAR 中恰好出现一次；出现零次或多次时停止，不尝试模糊修改。已经处于新表达式状态时报告为已应用并保持幂等。

## 脚本接口

新增独立脚本：

```text
scripts/patch_codex_desktop_model_filter_macos.sh
```

支持以下模式：

```text
--dry-run                 默认；只补丁临时副本并验证
--apply                   修改真实 ChatGPT.app
--yes                     跳过 apply 确认
--restore <backup-dir>    从指定完整备份恢复 App
--no-restart              成功后不重启 App
--self-test               运行纯本地单元测试
```

脚本不使用 `sudo`。当前 App 及目标文件归当前用户所有；若未来权限不满足，脚本应明确失败，不自行提权。

## Apply 数据流

1. 检查 App 路径、版本、依赖、文件归属和可写权限。
2. 确认 ASAR 内旧表达式恰好出现一次，并记录原始哈希。
3. 使用 `ditto` 把完整 `ChatGPT.app` 备份到：

   ```text
   ~/Library/Application Support/Codex/codex-app-backups/<version>-<timestamp>/ChatGPT.app
   ```

4. 验证备份的结构和 OpenAI 原始签名，然后请求 `com.openai.codex` 正常退出并等待相关进程结束；只处理该 App 的进程。
5. 在临时副本中进行等长表达式替换。
6. 重新计算目标 chunk 的 SHA256，并更新 ASAR 头中该文件的 `hash` 与单块 `blocks[0]`。
7. 对修改后的 ASAR 头部 JSON 计算 SHA256，写入 `Info.plist` 的 `ElectronAsarIntegrity["Resources/app.asar"].hash`。
8. 提取并执行目标 chunk 的 JavaScript 语法检查。
9. 通过各自目录中的临时文件分别替换真实 `app.asar` 和 `Info.plist`；App 在替换期间保持关闭。
10. 对 App 做本地 ad-hoc 重签名；保留必要的标识和主程序 entitlements，不使用 `--deep` 重签未修改的嵌套组件。
11. 运行 ASAR、plist 和 `codesign` 验证。任一步失败时立即从完整备份恢复。
12. 成功后按原路径重新打开 App。

## Dry-run 数据流

`--dry-run` 只复制 `app.asar` 和 `Info.plist` 到 `/private/tmp`，执行相同的表达式、chunk 哈希、ASAR 头部哈希及语法验证。它不退出 App、不修改签名、不修改真实 App 或用户数据。

## 回滚

`--restore <backup-dir>` 必须：

1. 验证备份目录中存在完整 `ChatGPT.app`，且版本和基础结构可读。
2. 正常退出当前 Codex App。
3. 把当前已补丁 App 移到同目录的失败保留位置，避免直接删除唯一副本。
4. 使用 `ditto` 恢复完整备份，保留原始 OpenAI 签名、notarization 资源和嵌套组件。
5. 运行 `codesign --verify --deep --strict`，成功后重新打开 App。

## 错误处理

- 目标表达式数量不是 1：停止并提示 App 版本可能变化。
- ASAR 原始 chunk 哈希与头部不一致：停止，不写真实文件。
- 备份失败或磁盘空间不足：停止，不退出或修改 App。
- App 无法退出：停止，不强杀当前 Codex 任务；提示用户从系统 Terminal 重试。
- 签名或启动前验证失败：自动恢复完整备份并报告日志路径。
- App 更新覆盖补丁：重新运行 dry-run；只有新版本仍满足唯一表达式和完整性检查时才允许 apply。

## 测试与验收

### 自动测试

1. 旧表达式存在一次时执行等长替换。
2. 零次、多次匹配时拒绝修改。
3. 已补丁状态可重复运行且不二次修改。
4. 修改后目标 chunk SHA256 与 ASAR 头一致。
5. 修改后 ASAR 头 SHA256 与待写入 plist 的值一致。
6. 提取后的 chunk 通过 `node --check`。
7. dry-run 不改变真实 App 的哈希和修改时间。
8. 完整 App 副本重签名后通过 `codesign --verify --deep --strict`。
9. 回滚后 App 哈希和 OpenAI 签名验证恢复到备份状态。

### 手工验收

1. 用户从系统 Terminal 执行 `--apply --yes`。
2. App 启动后模型下拉显示 Sol、Terra 和 Luna。
3. 等待 Statsig 写回旧 `107580212` 配置后，下拉仍显示三个模型。
4. 完整退出并重启 App 后模型仍显示。
5. 运行一个新任务，确认选择的 5.6 模型能实际发送请求。
6. 执行回滚后，官方签名恢复且 App 可正常启动。

## 风险

ad-hoc 重签名会使运行中的副本不再具有 OpenAI Developer ID 与 notarization 状态。依赖 OpenAI Team ID 的 Keychain access group、application group 或系统授权可能不可用；用户当前基于 API key/custom provider 的 Codex 调用不依赖这些 ChatGPT 登录能力，但其他 Desktop 功能仍需在手工验收中观察。

完整 App 备份约占 1.4 GB。自动更新会恢复官方 App 文件并移除补丁，这是可预期行为，不应阻止更新；脚本需要按版本重新执行 dry-run 后再决定是否应用。
