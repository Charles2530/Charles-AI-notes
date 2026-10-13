# 论文 PDF 阅读器设计

**日期：** 2026-10-10

## 目标

在现有 MkDocs 知识库中增加一个本地优先的论文阅读区，允许用户导入自己的 PDF，按论文管理，阅读时恢复上次页码，并对文字做高亮、添加页码关联笔记，最后把笔记导出为 Markdown 或 JSON。

第一版以 Mozilla PDF.js 负责 PDF 解析和渲染，参考 Easy Reviewer 的静态站点和本地存储方式，借鉴 Wren 的论文条目、集合和笔记关系。实现保持为静态前端，不引入 React、Node 服务或登录系统。

## 范围

### 第一版包含

- 论文库入口与独立页面，不改变现有知识库文章导航。
- 通过文件选择器或拖拽导入本地 PDF。
- 自动生成论文条目：标题默认使用文件名，用户可以编辑标题、作者和标签。
- 浏览器 IndexedDB 持久化 PDF 文件、论文元数据、阅读位置、批注和 Markdown 笔记。
- PDF.js 阅读器：
  - 连续滚动显示页面
  - 上一页 / 下一页和页码输入
  - 缩放
  - 文本搜索
  - 文档目录（PDF 内置 outline 存在时）
  - 最近阅读位置恢复
- 文字选区高亮：
  - 保存页码、选区矩形、原文摘录、颜色和时间
  - 高亮项可在侧栏中查看并跳回对应页
- 论文侧栏笔记：
  - Markdown 原文编辑
  - 按钮把当前高亮引用插入笔记，并带页码链接
  - 自动保存
  - 导出 Markdown 和 JSON
- 论文删除、笔记和批注删除。
- 无 PDF 或批注数据上传到服务器；页面只在本地浏览器存储。

### 明确不包含

- Zotero Connector、DOI 元数据抓取、跨设备同步。
- 扫描版 PDF OCR。
- 内嵌 PDF 文件写回和 XFA/表单编辑。
- 多用户权限、云端数据库和 AI 问答。
- 直接复制 GPL/AGPL 项目的代码；只复用公开的交互和数据模型思路。

## 页面与文件边界

- files/paper-library/index.md
  - 论文库入口页和语义化容器。
  - 页面文本说明本地存储方式、导入按钮和基础使用提示。
- files/assets/javascripts/paper-library.js
  - 唯一的页面编排脚本：初始化 IndexedDB、绑定导入和阅读器 UI、调用 PDF.js、协调批注与笔记状态。
  - 只在论文库页面加载，不能影响普通文章页。
- files/assets/stylesheets/extra.css
  - 增加以 .atlas-paper- 为前缀的论文库和阅读器样式，沿用现有颜色变量、响应式断点和打印规则。
- files/assets/javascripts/vendor/pdfjs/
  - 固定版本的 PDF.js browser build、worker 和对应许可证说明。
  - 不从 CDN 运行时加载脚本。
- mkdocs.yml
  - 将论文库页面放在“论文精读”附近，加载新脚本。
- tests/test_paper_library.py
  - 检查页面导航、脚本边界、IndexedDB 数据模型、批注和导出逻辑的静态契约。

## 数据模型

IndexedDB 数据库名称为 atlas-paper-library，版本为 1，包含以下 object stores：

- papers
  - id、title、authors、year、tags[]、filename、mime、size、createdAt、updatedAt、lastPage、totalPages。
- files
  - paperId、blob、updatedAt，以 paperId 为 key。
- annotations
  - id、paperId、type（highlight 或 note）、page、rects[]、quote、color、comment、createdAt、updatedAt。
- notes
  - paperId、markdown、updatedAt，以 paperId 为 key。

所有写入通过一个小型 repository 层完成，UI 不直接操作事务。批注矩形使用 PDF.js viewport-independent 坐标，恢复时根据当前 viewport 重算像素位置。

## 交互流程

1. 用户进入“论文库”，看到本地论文列表和“导入 PDF”。
2. 选择 PDF 后，生成 paperId，写入 papers 与 files，默认打开阅读器。
3. 阅读器左侧渲染 PDF，顶部显示标题、页码、缩放、搜索和返回论文库。
4. 用户选中文本，浮动工具条提供颜色高亮和“添加笔记”。
5. 保存批注后，右侧侧栏出现引用、页码和备注输入框；用户可以将引用插入 Markdown 笔记。
6. 页面离开或阅读位置变化时，节流保存 lastPage；笔记输入采用 500ms 防抖自动保存。
7. 用户下次打开论文时，从 IndexedDB 读取 PDF、笔记、批注和上次页码。
8. 导出 Markdown 时生成标题、元数据、正文笔记和按页码排序的批注；导出 JSON 时包含完整论文元数据、笔记和批注，便于备份。

## 错误与边界

- 非 PDF 文件或读取失败：显示中文错误提示，不写入半成品记录。
- 浏览器不支持 IndexedDB：显示“当前浏览器无法保存论文”的提示，仍允许当前会话临时阅读。
- PDF 无文本层：允许阅读和页码笔记，但禁用文字高亮并说明原因。
- 文件过大：第一版不设置人为大小上限，但捕获 QuotaExceededError 并提示清理存储空间。
- 导入同名 PDF：按新论文处理，不覆盖原条目。
- 页面切换时存在未保存笔记：先 flush 当前 debounce，再切换。
- 论文删除：在同一事务中删除 PDF、元数据、批注和笔记。

## 验证

- Python 静态测试验证：
  - 论文库页面存在且加入导航
  - 脚本只在目标页面初始化
  - IndexedDB store 名称和关键字段存在
  - 导出包含页码与批注引用
  - 普通页面不会加载 PDF.js 或创建论文库 DOM
- make build 验证 MkDocs 严格构建和本地资源路径。
- 手工浏览器检查：
  - 导入一篇文本型论文
  - 刷新页面后论文仍在库中
  - 高亮、笔记、页码恢复和导出
  - 移动端窄屏布局
  - 删除论文后所有相关数据消失

