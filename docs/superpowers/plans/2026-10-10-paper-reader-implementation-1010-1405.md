# 论文 PDF 阅读器 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有 MkDocs 静态站点中增加可本地导入、阅读、批注和记录 Markdown 笔记的论文 PDF 阅读器。

**Architecture:** 使用固定版本 PDF.js 的 browser build 渲染 PDF。论文元数据、PDF Blob、阅读位置、批注和笔记通过 IndexedDB repository 保存；页面脚本只在论文库页面挂载。单页采用论文列表、PDF 阅读区、批注/笔记侧栏的分栏布局。

**Tech Stack:** MkDocs Material、原生 JavaScript、IndexedDB、PDF.js browser build、现有 CSS 变量和 Python unittest/pytest。

**Spec:** docs/superpowers/specs/2026-10-10-paper-reader-design-1010-1405.md

## Global Constraints

- PDF.js 资源固定版本并随站点本地部署，不运行时从 CDN 加载。
- 页面前缀统一使用 atlas-paper-，不能影响普通文章页。
- 用户 PDF、笔记和批注只写入浏览器 IndexedDB，不发送到服务器。
- 保持用户已有未提交改动，不修改与论文阅读器无关的文件。
- 新生成的 Markdown 文件使用 MMDD-HHMM 时间戳命名。

## Review Focus

- 普通文章页加载时不应创建论文库 DOM 或初始化 IndexedDB。
- 非 PDF 文件、读取失败和 IndexedDB 配额错误必须显示可理解的中文错误。
- 刷新页面后论文、笔记、批注和上次页码必须恢复。
- PDF 无文本层时仍可阅读和添加页码笔记，但不能伪造文字高亮。
- 删除论文必须一并删除 PDF、元数据、批注和笔记。

---

### Task 1: Vendor PDF.js and create repository contracts

**Files:**
- Create: files/assets/javascripts/vendor/pdfjs/pdf.min.mjs
- Create: files/assets/javascripts/vendor/pdfjs/pdf.worker.min.mjs
- Create: files/assets/javascripts/vendor/pdfjs/LICENSE
- Modify: files/assets/javascripts/paper-library.js

**Interfaces:**
- Produces IndexedDB helpers openPaperDatabase(), listPapers(), getPaper(), createPaper(), updatePaper(), deletePaper(), listAnnotations(), saveAnnotation(), deleteAnnotation(), getNote(), saveNote().
- Produces exportMarkdown(paper, note, annotations) and exportPaperJson(paper, note, annotations).

- [ ] Step 1: Fetch a fixed pdfjs-dist browser build and license into the vendor directory.
- [ ] Step 2: Add a repository layer in paper-library.js with database atlas-paper-library version 1 and stores papers, files, annotations, notes.
- [ ] Step 3: Implement atomic paper deletion and quota/read failure errors.
- [ ] Step 4: Implement deterministic Markdown and JSON export functions.
- [ ] Step 5: Add static tests for store names, key fields, export contents, and deletion coverage.
- [ ] Step 6: Run the focused Python tests and confirm they fail only on missing page wiring.

### Task 2: Add the paper library page and site wiring

**Files:**
- Create: files/paper-library/index.md
- Modify: mkdocs.yml
- Modify: files/assets/javascripts/paper-library.js
- Modify: files/assets/stylesheets/extra.css

**Interfaces:**
- Consumes repository functions from Task 1.
- Produces a page shell with library view, import dropzone, paper list, reader workspace, toolbar, annotation list, note editor, and status region.

- [ ] Step 1: Create semantic page containers and Chinese instructions for local storage.
- [ ] Step 2: Add the page under the existing 论文精读 navigation.
- [ ] Step 3: Load paper-library.js only through the global extra_javascript entry, with a page guard inside the script.
- [ ] Step 4: Add responsive .atlas-paper- styles for desktop, narrow screens, dark/print-safe states, and hidden panels.
- [ ] Step 5: Add tests that ordinary pages do not mount the reader and the nav/page assets exist.
- [ ] Step 6: Run mkdocs build and focused tests.

### Task 3: Implement the PDF reader, text layer, and annotations

**Files:**
- Modify: files/assets/javascripts/paper-library.js
- Modify: files/assets/stylesheets/extra.css
- Modify: files/paper-library/index.md

**Interfaces:**
- Consumes PDF.js module and repository functions.
- Produces reader state with currentPage, totalPages, scale, search query, selection, and active paper.
- Produces annotations containing page, quote, normalized rects, color, and optional comment.

- [ ] Step 1: Implement file input and drag/drop import with title metadata defaults.
- [ ] Step 2: Render PDF pages with PDF.js canvas and text layer, including worker configuration.
- [ ] Step 3: Implement page navigation, zoom, search, outline, and last-page persistence.
- [ ] Step 4: Implement selection toolbar, text highlight persistence, normalized rectangles, and page jumps from annotation cards.
- [ ] Step 5: Implement page note creation for documents without a text layer and delete controls.
- [ ] Step 6: Implement paper title/author/year/tag editing and list filtering.
- [ ] Step 7: Run browser-independent static tests and build; manually exercise import, reload, highlight, page note, and delete.

### Task 4: Implement Markdown notes, backup/export, and final verification

**Files:**
- Modify: files/assets/javascripts/paper-library.js
- Modify: files/assets/stylesheets/extra.css
- Modify: tests/test_paper_library.py

**Interfaces:**
- Consumes note and annotation repository functions.
- Produces debounced note autosave, quote insertion with page references, Markdown download, JSON backup download, and user-visible status/error messages.

- [ ] Step 1: Add 500ms debounced note autosave with flush before switching papers.
- [ ] Step 2: Add quote insertion from selected annotation into the Markdown editor.
- [ ] Step 3: Add Markdown and JSON download buttons with stable filenames.
- [ ] Step 4: Add focused tests for autosave contracts, export names, and ordinary-page isolation.
- [ ] Step 5: Run pytest, make build, and inspect git diff to ensure unrelated user changes remain untouched.
- [ ] Step 6: Record any browser-only limitations in the page help text.

