# 论文阅读器第一版交付记录

## 结果

入口位于“论文精读 → 论文阅读器”，页面路径为 `paper-library/`。沿用 MkDocs Material、原生 JavaScript 和 IndexedDB；没有引入 React、Node 服务或前端构建步骤。Easy Reviewer 和 Wren 仅作为静态交互与论文条目结构的设计参考，没有复制其应用代码。

本文记录最终实现及验证结果，补充先前 1010-1405 设计与计划，保留原版本供对照。

## 已完成

- [x] 固定 PDF.js 5.4.624，随站点部署 module、worker、CMaps、标准字体、WASM 解码器及许可证。
- [x] 本地文件选择、拖拽和 HTTP/HTTPS URL 导入；先解析再保存，损坏文件不留下无效条目。URL 跨域限制、HTTP 错误有提示。
- [x] IndexedDB 保存 PDF、标题、作者、年份、标签、笔记、批注和上次页码；支持标题、作者、标签筛选。
- [x] 连续页面布局、可见页按需渲染、页码跳转、缩放、搜索并跳到首次匹配页、PDF 自带目录。
- [x] 滚动更新阅读页码，重新打开时恢复；页码输入取整数并限制在有效范围内。
- [x] 选中文字后保存彩色高亮或引用批注；扫描页仍可添加页码笔记；支持删除批注。
- [x] 点击批注跳回原页，独立“插入笔记”按钮插入引用及可回到本地论文的页码链接。
- [x] Markdown 原文编辑与自动保存，Markdown / JSON 导出；存储写满时仍可导出编辑器里的内容。
- [x] 同一事务删除 PDF、条目、笔记和批注；同步抛错时主动中止事务，不遗留部分删除；失败不会显示成功。
- [x] IndexedDB 不可用时切换临时内存模式，并持续显示刷新丢失数据的提示。
- [x] 桌面分栏与窄屏布局，普通文章页不加载论文库脚本或 PDF.js。

## 文件范围

| 文件 | 职责 |
| --- | --- |
| `files/paper-library/index.md` | 页面结构、URL 导入表单、本地数据与导出范围说明 |
| `files/assets/javascripts/paper-library.js` | 存储、导入、PDF 阅读、高亮、笔记和导出 |
| `files/assets/stylesheets/extra.css` | atlas-paper 样式、文字层规则、页面宽度和移动端布局 |
| `files/assets/javascripts/vendor/pdfjs/` | 固定第三方资源及 NOTICE / LICENSE |
| `mkdocs.yml` | 阅读器导航及样式缓存版本 |
| `tests/test_paper_library.py` | 页面、导航与资源契约检查 |
| `tests/browser/paper_library_test.cjs` | 真实 Chromium、真实 PDF 和 IndexedDB 行为测试 |

工作区原有其他文章和索引的未提交修改保持不动。本次没有提交、推送或发布。

## 验证

- MkDocs `build --strict` 通过；构建产物中的 JS、CSS、PDF.js module 和 worker 与源码一致。
- 阅读器、站点性能和 AI 侧栏的相关 Python 测试共 38 项通过。
- Chromium 浏览器测试共 12 项通过，无未处理浏览器错误。覆盖损坏文件、URL 正常/404/跨域失败、笔记恢复、滚动页码、高亮几何、缩放、搜索、Markdown / JSON 下载、导出页码链接重开、删除成功/失败回滚、存储满时导出、临时阅读、扫描页、390px 窄屏、普通页面隔离、60 页 PDF 按需渲染、快速重复缩放及小数/越界页码。
- JavaScript 语法检查与 `git diff --check` 通过。
- 完整 Python 测试结果为 164 通过、9 跳过、27 失败。失败分布为扩散课程 1、强化学习课程 18、训练推理课程 1、世界模型课程 5、Zotero 摘要 2。它们涉及其他课程内容契约与摘要预期，未修改相关文件，不能将全站测试描述为全部通过。
- 只读代码审查指出的存储满时不能导出、删除失败仍提示成功问题均已修复，并添加运行时回归测试。

### 复验命令

```sh
mkdocs build --strict
python -m pytest tests/test_paper_library.py tests/test_site_performance.py tests/test_ai_reading_assistant_site.py -q
node --check files/assets/javascripts/paper-library.js
git diff --check
```

浏览器测试依赖只安装在临时目录，不进入站点依赖。首次在其他机器复验时可执行以下命令，下载范围仅为临时测试依赖与 Playwright 浏览器缓存；可删除该临时目录回收测试依赖。

```sh
npm install --prefix /tmp/atlas-paper-reader-tests --no-audit --no-fund playwright@1.57.0
/tmp/atlas-paper-reader-tests/node_modules/.bin/playwright install chromium
PLAYWRIGHT_MODULE=/tmp/atlas-paper-reader-tests/node_modules/playwright node --test tests/browser/paper_library_test.cjs
```

测试使用独立浏览器上下文和随机本地端口，不访问用户真实论文库；测试 PDF 由代码生成。

## 与初始设计的具体差异

- 笔记在输入事件发生时立即提交 IndexedDB，避免原计划 500ms 防抖造成刷新前未保存的空窗。
- 页面骨架连续排列，canvas / 文字层按可见区域加载，避免首次打开就绘制全部页面。
- 增加用户最初方案中的 URL 导入，不使用代理或服务端下载。
- 明确将 JSON 称为笔记与批注导出，不宣称它是完整 PDF 备份或可回导备份。
- 原有 PDF.js 6.4.299 在测试 Chromium 中调用不可用的 Map.getOrInsertComputed，造成实际渲染失败；换用已验证的固定版本 5.4.624，没有声明它是最新版本。

## 使用边界

数据属于同一浏览器的同一站点来源，协议、域名或端口变化时不会共享；清除站点数据会丢失本地库。请另外保存原 PDF。

Markdown / JSON 包含笔记、条目元数据和批注，不含 PDF 文件；当前没有 JSON 回导、跨设备同步、OCR、AI 问答或批注写回原 PDF。页码链接需要同一浏览器及原站点中的论文仍然存在。

URL 导入需要来源网站允许 CORS；下载会请求来源网站，PDF 内容和笔记不会被此阅读器上传到服务器。

本次运行时验证使用 Chromium，移动端为视口模拟；未宣称已覆盖 Safari、Firefox、真实移动设备或所有特殊字体/加密 PDF。长文档按需绘制，但已访问页的 canvas 会保留到重新缩放或关闭论文，极长文档仍可能占用较多内存。
