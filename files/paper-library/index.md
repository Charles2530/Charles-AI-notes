---
title: 论文阅读器
hide:
  - navigation
  - toc
---

<div class="atlas-paper-library" data-atlas-paper-library>
  <header class="atlas-paper-hero">
    <div>
      <span class="atlas-paper-kicker">LOCAL RESEARCH DESK</span>
      <h1>论文阅读器</h1>
      <p>把正在读的论文放在一个可回到原文的位置。PDF、阅读进度、高亮和笔记都保存在当前浏览器里。</p>
    </div>
    <div class="atlas-paper-hero-actions">
      <button class="atlas-paper-button atlas-paper-button--primary" type="button" data-paper-import>导入 PDF</button>
      <input class="atlas-paper-visually-hidden" type="file" accept="application/pdf,.pdf" data-paper-file-input>
      <button class="atlas-paper-button" type="button" data-paper-back hidden>返回论文库</button>
    </div>
  </header>

  <div class="atlas-paper-status" role="status" aria-live="polite" data-paper-status>正在准备本地论文库…</div>

  <p class="atlas-paper-storage-warning" data-paper-storage-warning role="alert" hidden>当前浏览器无法保存论文库，已切换为临时阅读模式。刷新或关闭页面会丢失 PDF、笔记和批注，请及时导出。</p>

  <section class="atlas-paper-library-view" data-paper-library-view>
    <div class="atlas-paper-library-toolbar">
      <div>
        <h2>我的论文</h2>
        <p>导入本地 PDF 后，它会出现在这里，并记住你上次读到的页码。</p>
      </div>
      <label class="atlas-paper-search">
        <span class="atlas-paper-visually-hidden">筛选论文</span>
        <input type="search" placeholder="按标题、作者或标签筛选" data-paper-filter>
      </label>
    </div>
    <form class="atlas-paper-url-form" data-paper-url-form>
      <label for="paper-url">PDF 地址</label>
      <input id="paper-url" type="url" placeholder="https://example.org/paper.pdf" data-paper-url-input required>
      <button class="atlas-paper-button" type="submit" data-paper-url-import>从 URL 导入</button>
      <small>地址需要允许跨域读取；无法导入时，请先下载 PDF。文件将保存在本机，下载请求会发送到来源网站。</small>
    </form>
    <div class="atlas-paper-dropzone" tabindex="0" data-paper-dropzone>
      <strong>拖入 PDF，或点击选择文件</strong>
      <span>当前版本只在本机浏览器保存文件，不会上传到服务器。</span>
    </div>
    <div class="atlas-paper-grid" data-paper-list></div>
    <div class="atlas-paper-empty" data-paper-empty hidden>
      <div class="atlas-paper-empty-icon">⌁</div>
      <h3>还没有论文</h3>
      <p>先导入一篇你想慢慢读的 PDF。</p>
    </div>
  </section>

  <section class="atlas-paper-reader-view" data-paper-reader-view hidden>
    <div class="atlas-paper-reader-header">
      <div class="atlas-paper-reader-title">
        <button class="atlas-paper-icon-button" type="button" aria-label="返回论文库" data-paper-reader-back>←</button>
        <div>
          <span class="atlas-paper-reader-eyebrow">READING SESSION</span>
          <h2 data-paper-reader-title>论文</h2>
        </div>
      </div>
      <div class="atlas-paper-reader-tools">
        <label class="atlas-paper-page-control">页码
          <input type="number" min="1" value="1" data-paper-page-input>
          <span>/ <b data-paper-page-total>—</b></span>
        </label>
        <button class="atlas-paper-icon-button" type="button" title="上一页" data-paper-prev>‹</button>
        <button class="atlas-paper-icon-button" type="button" title="下一页" data-paper-next>›</button>
        <button class="atlas-paper-icon-button" type="button" title="缩小" data-paper-zoom-out>−</button>
        <span class="atlas-paper-zoom-label" data-paper-zoom-label>100%</span>
        <button class="atlas-paper-icon-button" type="button" title="放大" data-paper-zoom-in>＋</button>
        <label class="atlas-paper-reader-search">
          <span class="atlas-paper-visually-hidden">搜索 PDF</span>
          <input type="search" placeholder="搜索原文" data-paper-pdf-search>
        </label>
        <button class="atlas-paper-button atlas-paper-button--quiet" type="button" data-paper-export-markdown>导出笔记</button>
        <button class="atlas-paper-button atlas-paper-button--quiet" type="button" data-paper-export-json>导出 JSON</button>
      </div>
    </div>

    <div class="atlas-paper-reader-layout">
      <aside class="atlas-paper-meta-panel">
        <div class="atlas-paper-panel-heading">
          <span>论文条目</span>
          <button class="atlas-paper-text-button" type="button" data-paper-delete>删除</button>
        </div>
        <label>标题<input type="text" data-paper-title-input></label>
        <label>作者<input type="text" placeholder="可选" data-paper-authors-input></label>
        <div class="atlas-paper-meta-row">
          <label>年份<input type="number" placeholder="—" data-paper-year-input></label>
          <label>标签<input type="text" placeholder="逗号分隔" data-paper-tags-input></label>
        </div>
        <div class="atlas-paper-outline" data-paper-outline hidden>
          <div class="atlas-paper-panel-heading"><span>目录</span></div>
          <div data-paper-outline-list></div>
        </div>
        <div class="atlas-paper-annotation-panel">
          <div class="atlas-paper-panel-heading"><span>批注 <b data-paper-annotation-count>0</b></span></div>
          <div class="atlas-paper-annotation-list" data-paper-annotation-list></div>
          <button class="atlas-paper-button atlas-paper-button--full" type="button" data-paper-page-note>在当前页写笔记</button>
        </div>
      </aside>

      <main class="atlas-paper-document-shell">
        <div class="atlas-paper-document" data-paper-document aria-label="PDF 阅读区"></div>
        <div class="atlas-paper-reader-placeholder" data-paper-reader-placeholder>
          <span class="atlas-paper-spinner"></span>
          <p>正在加载 PDF…</p>
        </div>
        <div class="atlas-paper-selection-toolbar" data-paper-selection-toolbar hidden>
          <span>标记</span>
          <button type="button" data-paper-highlight-color="#f4c95d" aria-label="黄色高亮" class="atlas-paper-color-dot atlas-paper-color-dot--yellow"></button>
          <button type="button" data-paper-highlight-color="#8fd5c5" aria-label="绿色高亮" class="atlas-paper-color-dot atlas-paper-color-dot--green"></button>
          <button type="button" data-paper-highlight-color="#a8c7ed" aria-label="蓝色高亮" class="atlas-paper-color-dot atlas-paper-color-dot--blue"></button>
          <button type="button" data-paper-selection-note>加批注</button>
        </div>
      </main>

      <aside class="atlas-paper-notes-panel">
        <div class="atlas-paper-panel-heading">
          <span>阅读笔记</span>
          <span class="atlas-paper-save-state" data-paper-note-state>自动保存</span>
        </div>
        <textarea data-paper-note-editor placeholder="写下你对这篇论文的理解…&#10;&#10;选中文本后，可以把引用插入这里。"></textarea>
        <div class="atlas-paper-notes-help">
          <strong>快捷提示</strong>
          <span>选中文本 → 点击颜色，创建页码关联高亮。</span>
          <span>点击批注跳回原页；点击“插入笔记”添加引用和页码链接。</span>
        </div>
      </aside>
    </div>
  </section>
</div>

!!! info "本地数据说明"
    论文文件和笔记只保存在当前浏览器的 IndexedDB 中。清除站点数据会删除本地论文库；导出的 Markdown / JSON 包含笔记和批注，不含 PDF 文件，暂不支持 JSON 回导。页码链接依赖当前浏览器中的原论文；请另行保留 PDF。第一版暂不提供跨设备同步和扫描版 PDF OCR。

<script src="../assets/javascripts/paper-library.js?v=20261010-reader-v1"></script>

