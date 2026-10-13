/*!
 * Atlas paper reader
 * PDF.js 5.4.624 is vendored under vendor/pdfjs (Apache-2.0).
 */
(function () {
  "use strict";

  var root = document.querySelector("[data-atlas-paper-library]");
  if (!root) return;
  if (root.dataset.ready === "true") return;
  root.dataset.ready = "true";

  var DB_NAME = "atlas-paper-library";
  var DB_VERSION = 1;
  // IndexedDB schema version(1) is intentionally stable for local backups.
  var STORE_NAMES = ["papers", "files", "annotations", "notes"];
  // Resolve relative to this script, so subpath deployments and index.html both work.
  var ASSET_BASE = new URL("vendor/pdfjs/", document.currentScript.src);
  var PDF_MODULE_PATH = new URL("pdf.min.mjs", ASSET_BASE).href;
  var PDF_WORKER_PATH = new URL("pdf.worker.min.mjs", ASSET_BASE).href;
  var state = {
    db: null,
    pdfjs: null,
    papers: [],
    annotations: [],
    paper: null,
    pdf: null,
    scale: 1,
    currentPage: 1,
    pendingSelection: null,
    textLayerReady: false,
    busy: false,
    rendering: false,
    pageObserver: null,
    scrollTimer: null,
    memory: null
  };

  function qs(selector) {
    return root.querySelector(selector);
  }

  function now() {
    return new Date().toISOString();
  }

  function uid(prefix) {
    return prefix + "-" + Date.now().toString(36) + "-" + Math.random().toString(36).slice(2, 8);
  }

  function text(value) {
    return value == null ? "" : String(value);
  }

  function escapeHtml(value) {
    return text(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function setStatus(message, tone) {
    var node = qs("[data-paper-status]");
    if (!node) return;
    node.textContent = message;
    node.dataset.tone = tone || "normal";
  }

  function reportError(error) {
    var message = error.name === "QuotaExceededError"
      ? "浏览器存储空间不足，请先导出笔记并删除不需要的论文后重试。"
      : error.message || "操作失败，请重试。";
    setStatus(message, "error");
  }

  async function runOperation(action) {
    if (state.busy) return;
    state.busy = true;
    var controls = Array.from(root.querySelectorAll("button, input"));
    controls.forEach(function (node) { node.disabled = true; });
    try { await action(); } catch (error) { reportError(error); }
    finally {
      state.busy = false;
      controls.forEach(function (node) { node.disabled = false; });
    }
  }

  function requestToPromise(request) {
    return new Promise(function (resolve, reject) {
      request.onsuccess = function () { resolve(request.result); };
      request.onerror = function () { reject(request.error || new Error("IndexedDB 请求失败")); };
    });
  }

  function transactionPromise(transaction) {
    return new Promise(function (resolve, reject) {
      transaction.oncomplete = function () { resolve(); };
      transaction.onerror = function () { reject(transaction.error || new Error("IndexedDB 事务失败")); };
      transaction.onabort = function () { reject(transaction.error || new Error("IndexedDB 事务已取消")); };
    });
  }

  function openPaperDatabase() {
    if (!window.indexedDB) {
      return Promise.reject(new Error("当前浏览器不支持 IndexedDB"));
    }
    return new Promise(function (resolve, reject) {
      var request = window.indexedDB.open(DB_NAME, DB_VERSION);
      request.onupgradeneeded = function () {
        var db = request.result;
        if (!db.objectStoreNames.contains("papers")) {
          var papers = db.createObjectStore("papers", { keyPath: "id" });
          papers.createIndex("updatedAt", "updatedAt", { unique: false });
        }
        if (!db.objectStoreNames.contains("files")) {
          db.createObjectStore("files", { keyPath: "paperId" });
        }
        if (!db.objectStoreNames.contains("annotations")) {
          var annotations = db.createObjectStore("annotations", { keyPath: "id" });
          annotations.createIndex("paperId", "paperId", { unique: false });
          annotations.createIndex("page", "page", { unique: false });
        }
        if (!db.objectStoreNames.contains("notes")) {
          db.createObjectStore("notes", { keyPath: "paperId" });
        }
      };
      request.onsuccess = function () {
        request.result.onversionchange = function () { request.result.close(); };
        resolve(request.result);
      };
      request.onblocked = function () { setStatus("请关闭其他打开论文库的标签页，然后刷新。", "error"); };
      request.onerror = function () { reject(request.error || new Error("无法打开本地论文库")); };
    });
  }

  function listPapers() {
    if (state.memory) return Promise.resolve(Array.from(state.memory.papers.values()).map(function (item) { return structuredClone(item); }));
    return requestToPromise(state.db.transaction("papers", "readonly").objectStore("papers").getAll())
      .then(function (items) {
        return items.sort(function (a, b) {
          return String(b.updatedAt || "").localeCompare(String(a.updatedAt || ""));
        });
      });
  }

  function getPaper(paperId) {
    if (state.memory) return Promise.resolve(structuredClone(state.memory.papers.get(paperId)));
    return requestToPromise(state.db.transaction("papers", "readonly").objectStore("papers").get(paperId));
  }

  function getFile(paperId) {
    if (state.memory) return Promise.resolve(state.memory.files.get(paperId));
    return requestToPromise(state.db.transaction("files", "readonly").objectStore("files").get(paperId));
  }

  function createPaper(paper, file) {
    if (state.memory) { state.memory.papers.set(paper.id, structuredClone(paper)); state.memory.files.set(paper.id, {paperId: paper.id, blob: file}); return Promise.resolve(); }
    var tx = state.db.transaction(["papers", "files"], "readwrite");
    tx.objectStore("papers").put(paper);
    tx.objectStore("files").put({ paperId: paper.id, blob: file, updatedAt: paper.updatedAt });
    return transactionPromise(tx);
  }

  function updatePaper(paper) {
    if (state.memory) { state.memory.papers.set(paper.id, structuredClone(paper)); return Promise.resolve(); }
    return transactionPromise((function () {
      var tx = state.db.transaction("papers", "readwrite");
      tx.objectStore("papers").put(paper);
      return tx;
    })());
  }

  async function deletePaper(paperId) {
    if (state.memory) {
      ["papers", "files", "notes"].forEach(function (store) { state.memory[store].delete(paperId); });
      state.memory.annotations.forEach(function (item, id) { if (item.paperId === paperId) state.memory.annotations.delete(id); });
      return;
    }
    var tx = state.db.transaction(["files", "papers", "annotations", "notes"], "readwrite");
    var completion = transactionPromise(tx);
    try {
      tx.objectStore("files").delete(paperId);
      tx.objectStore("papers").delete(paperId);
      tx.objectStore("notes").delete(paperId);
      var annotationStore = tx.objectStore("annotations");
      var index = annotationStore.index("paperId");
      index.openCursor(IDBKeyRange.only(paperId)).onsuccess = function (event) {
        var cursor = event.target.result;
        if (cursor) {
          cursor.delete();
          cursor.continue();
        }
      };
    } catch (error) {
      tx.abort();
      await completion.catch(function () {});
      throw error;
    }
    await completion;
  }

  async function listAnnotations(paperId) {
    if (state.memory) return Array.from(state.memory.annotations.values()).filter(function (item) { return item.paperId === paperId; }).map(function (item) { return structuredClone(item); });
    var store = state.db.transaction("annotations", "readonly").objectStore("annotations");
    return requestToPromise(store.index("paperId").getAll(IDBKeyRange.only(paperId))).then(function (items) {
      return items.sort(function (a, b) {
        return (a.page - b.page) || String(a.createdAt).localeCompare(String(b.createdAt));
      });
    });
  }

  function saveAnnotation(annotation) {
    if (state.memory) { state.memory.annotations.set(annotation.id, structuredClone(annotation)); return Promise.resolve(); }
    var tx = state.db.transaction("annotations", "readwrite");
    tx.objectStore("annotations").put(annotation);
    return transactionPromise(tx);
  }

  function deleteAnnotation(annotationId) {
    if (state.memory) { state.memory.annotations.delete(annotationId); return Promise.resolve(); }
    var tx = state.db.transaction("annotations", "readwrite");
    tx.objectStore("annotations").delete(annotationId);
    return transactionPromise(tx);
  }

  function getNote(paperId) {
    if (state.memory) return Promise.resolve(structuredClone(state.memory.notes.get(paperId) || { paperId: paperId, markdown: "" }));
    return requestToPromise(state.db.transaction("notes", "readonly").objectStore("notes").get(paperId))
      .then(function (note) { return note || { paperId: paperId, markdown: "", updatedAt: now() }; });
  }

  function saveNote(note) {
    if (state.memory) { state.memory.notes.set(note.paperId, structuredClone(note)); return Promise.resolve(); }
    var tx = state.db.transaction("notes", "readwrite");
    tx.objectStore("notes").put(note);
    return transactionPromise(tx);
  }

  function exportMarkdown(paper, note, annotations) {
    // Markdown page reference format: 第${annotation.page}页
    var lines = [
      "# " + (paper.title || "未命名论文"),
      "",
      "- 作者：" + (paper.authors || "未填写"),
      "- 年份：" + (paper.year || "未填写"),
      "- 文件：" + (paper.filename || ""),
      "",
      "## 我的笔记",
      "",
      note && note.markdown ? note.markdown.trim() : "（暂无正文笔记）",
      "",
      "## 批注索引",
      ""
    ];
    if (!annotations.length) {
      lines.push("（暂无批注）");
    } else {
      annotations.slice().sort(function (a, b) { return a.page - b.page; }).forEach(function (annotation) {
        var label = annotation.type === "highlight" ? "高亮" : "页码笔记";
        var quote = annotation.quote ? "「" + annotation.quote.trim() + "」" : "";
        var comment = annotation.comment ? " — " + annotation.comment.trim() : "";
        lines.push("- " + pageReference(paper, annotation.page) + " · " + label + " " + quote + comment);
      });
    }
    return lines.join("\n") + "\n";
  }

  function exportPaperJson(paper, note, annotations) {
    return JSON.stringify({
      version: 1,
      exportedAt: now(),
      paper: paper,
      note: note,
      annotations: annotations
    }, null, 2);
  }

  function pageReference(paper, page) {
    var url = new URL(window.location.href);
    url.hash = "paper=" + encodeURIComponent(paper.id) + "&page=" + page;
    // An exported note can navigate back to the local library on this browser.
    return "[第" + page + "页](" + url.href + ")";
  }

  function exportTimestamp() {
    var parts = new Intl.DateTimeFormat("en-GB", {
      timeZone: "Asia/Shanghai", month: "2-digit", day: "2-digit",
      hour: "2-digit", minute: "2-digit", hourCycle: "h23"
    }).formatToParts(new Date());
    var values = {};
    parts.forEach(function (part) { values[part.type] = part.value; });
    return values.month + values.day + "-" + values.hour + values.minute;
  }

  function downloadText(filename, content, mime) {
    var blob = new Blob([content], { type: mime });
    var url = URL.createObjectURL(blob);
    var link = document.createElement("a");
    link.href = url;
    link.download = filename;
    link.click();
    window.setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
  }

  function safeFilename(value) {
    return text(value || "paper").replace(/[\\/:*?"<>|]+/g, "-").trim().slice(0, 80) || "paper";
  }

  function showView(reader) {
    qs("[data-paper-library-view]").hidden = reader;
    qs("[data-paper-reader-view]").hidden = !reader;
    qs("[data-paper-back]").hidden = !reader;
  }

  function setPaperMetadata(paper) {
    qs("[data-paper-reader-title]").textContent = paper.title || paper.filename || "未命名论文";
    qs("[data-paper-title-input]").value = paper.title || "";
    qs("[data-paper-authors-input]").value = paper.authors || "";
    qs("[data-paper-year-input]").value = paper.year || "";
    qs("[data-paper-tags-input]").value = (paper.tags || []).join(", ");
  }

  function renderPaperList() {
    var list = qs("[data-paper-list]");
    var empty = qs("[data-paper-empty]");
    var filter = text(qs("[data-paper-filter]").value).trim().toLowerCase();
    var items = state.papers.filter(function (paper) {
      return !filter || [paper.title, paper.authors, (paper.tags || []).join(" ")].join(" ").toLowerCase().indexOf(filter) >= 0;
    });
    list.textContent = "";
    empty.hidden = state.papers.length > 0;
    empty.querySelector("h3").textContent = "还没有论文";
    empty.querySelector("p").textContent = "先导入一篇你想慢慢读的 PDF。";
    items.forEach(function (paper) {
      var card = document.createElement("article");
      card.className = "atlas-paper-card";
      card.tabIndex = 0;
      var progress = paper.totalPages ? Math.round((paper.lastPage || 1) / paper.totalPages * 100) : 0;
      card.innerHTML = [
        '<div class="atlas-paper-card-top"><span class="atlas-paper-file-icon">PDF</span><span class="atlas-paper-card-date">',
        escapeHtml(paper.year || "本地文件"),
        '</span></div>',
        '<h3>', escapeHtml(paper.title || paper.filename), '</h3>',
        '<p>', escapeHtml(paper.authors || paper.filename || "未填写作者"), '</p>',
        '<div class="atlas-paper-card-footer"><span>读到第 ', String(paper.lastPage || 1), ' 页</span><span>', String(progress), '%</span></div>',
        '<div class="atlas-paper-progress"><i style="width:', String(progress), '%"></i></div>'
      ].join("");
      card.addEventListener("click", function () { runOperation(function () { return openPaper(paper.id); }); });
      card.addEventListener("keydown", function (event) {
        if (event.key === "Enter" || event.key === " ") { event.preventDefault(); runOperation(function () { return openPaper(paper.id); }); }
      });
      list.appendChild(card);
    });
    if (state.papers.length && !items.length) {
      empty.hidden = false;
      empty.querySelector("h3").textContent = "没有匹配论文";
      empty.querySelector("p").textContent = "换一个标题、作者或标签试试。";
    }
  }

  async function refreshPaperList() {
    state.papers = await listPapers();
    renderPaperList();
  }

  function clearDocument() {
    if (state.pageObserver) state.pageObserver.disconnect();
    if (state.scrollTimer) window.clearTimeout(state.scrollTimer);
    state.pendingSelection = null;
    qs("[data-paper-selection-toolbar]").hidden = true;
    qs("[data-paper-document]").textContent = "";
    qs("[data-paper-outline-list]").textContent = "";
    qs("[data-paper-outline]").hidden = true;
    qs("[data-paper-reader-placeholder]").hidden = false;
    state.textLayerReady = false;
  }

  function renderOutline(outline) {
    var box = qs("[data-paper-outline]");
    var list = qs("[data-paper-outline-list]");
    list.textContent = "";
    if (!outline || !outline.length) {
      box.hidden = true;
      return;
    }
    box.hidden = false;
    outline.forEach(function addItem(item) {
      var button = document.createElement("button");
      button.type = "button";
      button.className = "atlas-paper-outline-item";
      button.textContent = item.title || "未命名章节";
      button.addEventListener("click", async function () {
        if (!state.pdf || !item.dest) return;
        try {
          var dest = typeof item.dest === "string" ? await state.pdf.getDestination(item.dest) : item.dest;
          if (!dest) return;
          var ref = dest[0];
          var pageIndex = Number.isInteger(ref) ? ref : await state.pdf.getPageIndex(ref);
          goToPage(pageIndex + 1);
        } catch (error) {
          setStatus("目录项暂时无法跳转。", "error");
        }
      });
      list.appendChild(button);
      (item.items || []).forEach(addItem);
    });
  }

  function normalizeRects(rects, bounds) {
    return Array.prototype.map.call(rects, function (rect) {
      return {
        x: Math.max(0, (rect.left - bounds.left) / bounds.width),
        y: Math.max(0, (rect.top - bounds.top) / bounds.height),
        width: Math.max(0, (Math.min(bounds.right, rect.right) - Math.max(bounds.left, rect.left)) / bounds.width),
        height: Math.max(0, (Math.min(bounds.bottom, rect.bottom) - Math.max(bounds.top, rect.top)) / bounds.height)
      };
    }).filter(function (rect) { return rect.width > 0.002 && rect.height > 0.002; });
  }

  function annotationRectStyle(rect) {
    return {
      left: (rect.x * 100) + "%",
      top: (rect.y * 100) + "%",
      width: (rect.width * 100) + "%",
      height: (rect.height * 100) + "%"
    };
  }

  function renderPageAnnotations(pageNumber, layer) {
    Array.prototype.forEach.call(layer.querySelectorAll(".atlas-paper-highlight"), function (node) { node.remove(); });
    state.annotations.filter(function (annotation) {
      return annotation.page === pageNumber && annotation.rects && annotation.rects.length;
    }).forEach(function (annotation) {
      annotation.rects.forEach(function (rect) {
        var mark = document.createElement("button");
        mark.type = "button";
        mark.className = "atlas-paper-highlight atlas-paper-highlight--" + (annotation.color || "#f4c95d").replace("#", "");
        var style = annotationRectStyle(rect);
        Object.keys(style).forEach(function (key) { mark.style[key] = style[key]; });
        mark.style.backgroundColor = annotation.color || "#f4c95d";
        mark.title = annotation.quote || "查看批注";
        mark.addEventListener("click", function () { focusAnnotation(annotation); });
        layer.appendChild(mark);
      });
    });
  }

  async function renderPage(pageNumber, pageShell, pdf, scale) {
    if (pageShell.dataset.rendered || pdf !== state.pdf) return;
    pageShell.dataset.rendered = "loading";
    var page = await pdf.getPage(pageNumber);
    if (pdf !== state.pdf || !pageShell.isConnected) return;
    var viewport = page.getViewport({ scale: scale });
    var surface = pageShell.querySelector(".atlas-paper-page-surface");
    var canvas = document.createElement("canvas");
    canvas.className = "atlas-paper-canvas";
    var outputScale = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.floor(viewport.width * outputScale);
    canvas.height = Math.floor(viewport.height * outputScale);
    canvas.style.width = viewport.width + "px";
    canvas.style.height = viewport.height + "px";
    var textLayer = document.createElement("div");
    textLayer.className = "atlas-paper-text-layer";
    textLayer.style.setProperty("--total-scale-factor", String(viewport.scale));
    textLayer.style.setProperty("--scale-factor", String(viewport.scale));
    var overlay = document.createElement("div");
    overlay.className = "atlas-paper-highlight-layer";
    surface.replaceChildren(canvas, textLayer, overlay);
    await page.render({ canvasContext: canvas.getContext("2d", { alpha: false }), viewport: viewport,
      transform: outputScale !== 1 ? [outputScale, 0, 0, outputScale, 0, 0] : null }).promise;
    if (pdf !== state.pdf || !pageShell.isConnected) return;
    var layer = new state.pdfjs.TextLayer({ textContentSource: page.streamTextContent(), container: textLayer, viewport: viewport });
    await layer.render();
    state.textLayerReady = state.textLayerReady || layer.textDivs.length > 0;
    if (!layer.textDivs.length) pageShell.querySelector(".atlas-paper-page-label").textContent += " · 无文字层，可添加页码笔记";
    renderPageAnnotations(pageNumber, overlay);
    textLayer.addEventListener("mouseup", function () { captureSelection(pageNumber, textLayer); });
    textLayer.addEventListener("touchend", function () { window.setTimeout(function () { captureSelection(pageNumber, textLayer); }, 100); });
    pageShell.dataset.rendered = "true";
  }

  async function renderDocument() {
    clearDocument();
    state.rendering = true;
    var pdf = state.pdf;
    var scale = state.scale;
    var documentNode = qs("[data-paper-document]");
    try {
      state.currentPage = Math.min(Math.max(Math.trunc(Number(state.paper.lastPage)) || 1, 1), pdf.numPages);
      qs("[data-paper-page-total]").textContent = String(pdf.numPages);
      qs("[data-paper-page-input]").max = String(pdf.numPages);
      qs("[data-paper-page-input]").value = String(state.currentPage);
      state.paper.totalPages = pdf.numPages;
      await updatePaper(state.paper);
      renderOutline(await pdf.getOutline().catch(function () { return null; }));
      for (var n = 1; n <= pdf.numPages; n += 1) {
        var page = await pdf.getPage(n);
        var viewport = page.getViewport({ scale: scale });
        var shell = document.createElement("section");
        shell.className = "atlas-paper-page";
        shell.dataset.page = String(n);
        shell.id = "paper-page-" + n;
        shell.innerHTML = '<div class="atlas-paper-page-label">第 ' + n + ' 页</div><div class="atlas-paper-page-surface"></div>';
        var surface = shell.lastElementChild;
        surface.style.width = viewport.width + "px";
        surface.style.height = viewport.height + "px";
        documentNode.appendChild(shell);
      }
      var queue = Promise.resolve();
      state.pageObserver = new IntersectionObserver(function (entries) {
        entries.forEach(function (entry) {
          if (!entry.isIntersecting) return;
          var shell = entry.target;
          queue = queue.then(function () { return renderPage(Number(shell.dataset.page), shell, pdf, scale); }).catch(function (error) {
            if (pdf === state.pdf && shell.isConnected) reportError(error);
          });
        });
      }, { root: documentNode, rootMargin: "600px 0px" });
      await renderPage(state.currentPage, documentNode.children[state.currentPage - 1], pdf, scale);
      qs("[data-paper-reader-placeholder]").hidden = true;
      scrollToPage(state.currentPage);
      Array.from(documentNode.children).forEach(function (shell) { state.pageObserver.observe(shell); });
      updateZoomLabel();
      setStatus(state.memory ? "临时阅读模式：刷新会丢失数据，请及时导出笔记。" : "论文已加载，本地自动保存已开启。", state.memory ? "info" : "success");
    } finally { state.rendering = false; }
  }

  async function loadPdf(blob) {
    await ensurePdfjs();
    var task = state.pdfjs.getDocument({ data: await blob.arrayBuffer(),
      cMapUrl: new URL("cmaps/", ASSET_BASE).href, cMapPacked: true,
      standardFontDataUrl: new URL("standard_fonts/", ASSET_BASE).href,
      wasmUrl: new URL("wasm/", ASSET_BASE).href,
      isEvalSupported: false });
    task.onPassword = function (updatePassword) {
      var password = window.prompt("此 PDF 需要密码（密码不会保存）：");
      if (password === null) task.destroy();
      else updatePassword(password);
    };
    try { return await task.promise; }
    catch (error) { await task.destroy(); throw error; }
  }

  function updateZoomLabel() {
    qs("[data-paper-zoom-label]").textContent = Math.round(state.scale * 100) + "%";
  }

  function scrollToPage(pageNumber) {
    var target = document.getElementById("paper-page-" + String(pageNumber));
    if (target) {
      var container = qs("[data-paper-document]");
      container.scrollTo({ top: container.scrollTop + target.getBoundingClientRect().top - container.getBoundingClientRect().top - 16, behavior: "instant" });
    }
    state.currentPage = pageNumber;
    qs("[data-paper-page-input]").value = String(pageNumber);
    saveLastPage();
  }

  function goToPage(pageNumber) {
    if (!state.pdf) return;
    var page = Math.min(Math.max(Math.trunc(Number(pageNumber)) || 1, 1), state.pdf.numPages);
    scrollToPage(page);
  }

  function saveLastPage() {
    if (!state.paper) return;
    state.paper.lastPage = state.currentPage;
    state.paper.updatedAt = now();
    updatePaper(state.paper).catch(reportError);
  }

  function updateRenderedScale() {
    if (!state.pdf || state.scale < 0.6 || state.scale > 2.2) return;
    return renderDocument();
  }

  function captureSelection(pageNumber, textLayer) {
    var selection = window.getSelection();
    var quote = text(selection && selection.toString()).trim();
    if (!quote || !selection.rangeCount) return;
    var range = selection.getRangeAt(0);
    if (!textLayer.contains(range.commonAncestorContainer)) return;
    var bounds = textLayer.getBoundingClientRect();
    var rects = normalizeRects(range.getClientRects(), bounds);
    if (!rects.length) return;
    state.pendingSelection = { page: pageNumber, quote: quote.slice(0, 2000), rects: rects };
    var toolbar = qs("[data-paper-selection-toolbar]");
    toolbar.hidden = false;
    var selectionRect = range.getBoundingClientRect();
    toolbar.style.left = Math.min(Math.max(selectionRect.left, 12), window.innerWidth - toolbar.offsetWidth - 12) + "px";
    toolbar.style.top = Math.max(selectionRect.top - 48, 12) + "px";
  }

  async function saveSelection(color, comment) {
    if (!state.pendingSelection || !state.paper) return;
    var annotation = {
      id: uid("annotation"),
      paperId: state.paper.id,
      type: comment ? "note" : "highlight",
      page: state.pendingSelection.page,
      rects: state.pendingSelection.rects,
      quote: state.pendingSelection.quote,
      color: color || "#f4c95d",
      comment: comment || "",
      createdAt: now(),
      updatedAt: now()
    };
    await saveAnnotation(annotation);
    state.annotations.push(annotation);
    qs("[data-paper-selection-toolbar]").hidden = true;
    window.getSelection().removeAllRanges();
    state.pendingSelection = null;
    renderAnnotationList();
    var layer = document.querySelector("#paper-page-" + String(annotation.page) + " .atlas-paper-highlight-layer");
    if (layer) renderPageAnnotations(annotation.page, layer);
    setStatus("批注已保存到本地。", "success");
  }

  function focusAnnotation(annotation) {
    goToPage(annotation.page);
  }

  function insertQuote(annotation) {
    var note = qs("[data-paper-note-editor]");
    if (annotation.quote) {
      note.focus();
      if (note.value && !note.value.endsWith("\n")) note.value += "\n";
      note.value += "> " + annotation.quote.replace(/\n/g, " ") + " " + pageReference(state.paper, annotation.page) + "\n\n";
      autosaveNote();
    }
  }

  function renderAnnotationList() {
    var list = qs("[data-paper-annotation-list]");
    list.textContent = "";
    qs("[data-paper-annotation-count]").textContent = String(state.annotations.length);
    if (!state.annotations.length) {
      var empty = document.createElement("p");
      empty.className = "atlas-paper-muted";
      empty.textContent = "选中文本或在当前页添加一条笔记。";
      list.appendChild(empty);
      return;
    }
    state.annotations.forEach(function (annotation) {
      var card = document.createElement("article");
      card.className = "atlas-paper-annotation-card";
      var quote = annotation.quote ? "「" + annotation.quote + "」" : "页码笔记";
      card.innerHTML = '<button type="button" class="atlas-paper-annotation-main"><span class="atlas-paper-annotation-page">第' + String(annotation.page) + '页</span><span class="atlas-paper-annotation-quote">' + escapeHtml(quote) + '</span>' + (annotation.comment ? '<span class="atlas-paper-annotation-comment">' + escapeHtml(annotation.comment) + '</span>' : "") + '</button><button type="button" class="atlas-paper-annotation-delete" aria-label="删除批注">×</button>';
      card.querySelector(".atlas-paper-annotation-main").addEventListener("click", function () { focusAnnotation(annotation); });
      card.querySelector(".atlas-paper-annotation-delete").addEventListener("click", async function () {
        try {
          await deleteAnnotation(annotation.id);
          state.annotations = state.annotations.filter(function (item) { return item.id !== annotation.id; });
          renderAnnotationList();
          var layer = qs("#paper-page-" + String(annotation.page) + " .atlas-paper-highlight-layer");
          if (layer) renderPageAnnotations(annotation.page, layer);
        } catch (error) { reportError(error); }
      });
      if (annotation.quote) {
        var insert = document.createElement("button");
        insert.type = "button";
        insert.className = "atlas-paper-text-button";
        insert.dataset.paperInsertQuote = "";
        insert.textContent = "插入笔记";
        insert.addEventListener("click", function () { insertQuote(annotation); });
        card.appendChild(insert);
      }
      list.appendChild(card);
    });
  }

  function autosaveNote() {
    qs("[data-paper-note-state]").textContent = "正在保存…";
    flushNote().catch(function (error) {
      qs("[data-paper-note-state]").textContent = "保存失败，请导出备份";
      reportError(error);
    });
  }

  async function flushNote() {
    if (!state.paper) return;
    await saveNote({ paperId: state.paper.id, markdown: qs("[data-paper-note-editor]").value, updatedAt: now() });
    qs("[data-paper-note-state]").textContent = "已自动保存";
  }

  async function openPaper(paperId, requestedPage) {
    await flushNote();
    if (state.pdf) { clearDocument(); await state.pdf.destroy(); state.pdf = null; }
    state.paper = null;
    try {
      var paper = await getPaper(paperId);
      if (!paper) throw new Error("论文条目不存在，请在原浏览器中打开此页码链接。");
      var file = await getFile(paperId);
      if (!file || !file.blob) throw new Error("找不到论文文件");
      var note = await getNote(paperId);
      state.annotations = await listAnnotations(paperId);
      qs("[data-paper-note-editor]").value = note.markdown || "";
      state.paper = paper;
      if (requestedPage) paper.lastPage = requestedPage;
      setPaperMetadata(paper);
      renderAnnotationList();
      showView(true);
      state.pdf = await loadPdf(file.blob);
      await renderDocument();
    } catch (error) {
      if (state.pdf) await state.pdf.destroy();
      state.pdf = null;
      state.paper = null;
      clearDocument();
      showView(false);
      throw error;
    }
  }

  async function closeReader() {
    await flushNote();
    clearDocument();
    if (state.pdf) await state.pdf.destroy();
    state.pdf = null;
    state.paper = null;
    state.annotations = [];
    showView(false);
    await refreshPaperList();
    setStatus(state.memory ? "临时论文库：刷新会丢失数据。" : "已返回论文库。", "normal");
  }

  async function importFile(file, sourceUrl) {
    if (!file || (file.type !== "application/pdf" && !/\.pdf$/i.test(file.name))) {
      throw new Error("请选择 PDF 文件。");
    }
    setStatus("正在检查 PDF…");
    // Parse before committing anything, so malformed files never leave orphan entries.
    var pdf = await loadPdf(file);
    var totalPages = pdf.numPages;
    await pdf.destroy();
    var timestamp = now();
    var paper = { id: uid("paper"), title: file.name.replace(/\.pdf$/i, ""),
      authors: "", year: "", tags: [], filename: file.name,
      mime: "application/pdf", size: file.size, sourceUrl: sourceUrl || "",
      createdAt: timestamp, updatedAt: timestamp, lastPage: 1, totalPages: totalPages };
    await createPaper(paper, file);
    await refreshPaperList();
    await openPaper(paper.id);
  }

  async function importUrl() {
    var value = qs("[data-paper-url-input]").value.trim();
    var url;
    try { url = new URL(value); } catch (_) { throw new Error("请输入完整的 HTTP 或 HTTPS PDF 地址。"); }
    if (!/^https?:$/.test(url.protocol) || url.username || url.password) throw new Error("只支持不含账号密码的 HTTP 或 HTTPS 地址。");
    setStatus("正在下载 PDF…");
    var response;
    try { response = await fetch(url.href, { credentials: "omit", signal: AbortSignal.timeout(60000) }); }
    catch (_) { throw new Error("无法下载 PDF，可能是网络或跨域（CORS）限制。请下载到本机后导入。"); }
    if (!response.ok) throw new Error("下载失败（HTTP " + response.status + "），请检查地址或下载后导入。");
    var blob = await response.blob();
    var filename = decodeURIComponent(url.pathname.split("/").pop() || "paper.pdf");
    if (!/\.pdf$/i.test(filename)) filename += ".pdf";
    await importFile(new File([blob], filename, { type: "application/pdf" }), url.href);
  }

  async function ensurePdfjs() {
    if (state.pdfjs) return state.pdfjs;
    var url = new URL(PDF_MODULE_PATH, window.location.href).href;
    state.pdfjs = await import(url);
    state.pdfjs.GlobalWorkerOptions.workerSrc = PDF_WORKER_PATH;
    return state.pdfjs;
  }

  async function searchPdf(query) {
    query = text(query).trim().toLowerCase();
    if (!query || !state.pdf) return;
    setStatus("正在搜索…", "normal");
    for (var page = 1; page <= state.pdf.numPages; page += 1) {
      var content = await state.pdf.getPage(page).then(function (item) { return item.getTextContent(); });
      var fullText = content.items.map(function (item) { return item.str; }).join(" ");
      if (fullText.toLowerCase().indexOf(query) >= 0) {
        goToPage(page);
        setStatus("在第" + String(page) + "页找到匹配文本。", "success");
        return;
      }
    }
    setStatus("没有找到匹配文本。", "info");
  }

  function bindMetadataInputs() {
    [
      ["[data-paper-title-input]", "title"],
      ["[data-paper-authors-input]", "authors"],
      ["[data-paper-year-input]", "year"]
    ].forEach(function (pair) {
      qs(pair[0]).addEventListener("change", function (event) {
        if (!state.paper) return;
        state.paper[pair[1]] = event.target.value.trim();
        state.paper.updatedAt = now();
        updatePaper(state.paper).then(refreshPaperList).catch(reportError);
        if (pair[1] === "title") qs("[data-paper-reader-title]").textContent = state.paper.title || state.paper.filename;
      });
    });
    qs("[data-paper-tags-input]").addEventListener("change", function (event) {
      if (!state.paper) return;
      state.paper.tags = event.target.value.split(",").map(function (item) { return item.trim(); }).filter(Boolean);
      state.paper.updatedAt = now();
      updatePaper(state.paper).then(refreshPaperList).catch(reportError);
    });
  }

  function bindEvents() {
    var input = qs("[data-paper-file-input]");
    qs("[data-paper-url-form]").addEventListener("submit", function (event) {
      event.preventDefault();
      runOperation(importUrl);
    });
    qs("[data-paper-document]").addEventListener("scroll", function () {
      qs("[data-paper-selection-toolbar]").hidden = true;
      if (state.scrollTimer) window.clearTimeout(state.scrollTimer);
      state.scrollTimer = window.setTimeout(function () {
        if (!state.pdf || state.rendering) return;
        var box = qs("[data-paper-document]").getBoundingClientRect();
        var pages = Array.from(qs("[data-paper-document]").children);
        var best = pages.find(function (page) { return page.getBoundingClientRect().bottom > box.top + 60; });
        if (!best || state.currentPage === Number(best.dataset.page)) return;
        state.currentPage = Number(best.dataset.page);
        qs("[data-paper-page-input]").value = String(state.currentPage);
        saveLastPage();
      }, 120);
    }, { passive: true });
    window.addEventListener("pagehide", function () { flushNote().catch(reportError); });
    window.addEventListener("hashchange", function () { runOperation(openHashLink); });
    qs("[data-paper-import]").addEventListener("click", function () { input.click(); });
    input.addEventListener("change", function (event) {
      var file = event.target.files && event.target.files[0];
      if (file) runOperation(function () { return importFile(file); });
      event.target.value = "";
    });
    var dropzone = qs("[data-paper-dropzone]");
    ["dragenter", "dragover"].forEach(function (eventName) {
      dropzone.addEventListener(eventName, function (event) {
        event.preventDefault();
        dropzone.classList.add("is-dragging");
      });
    });
    ["dragleave", "drop"].forEach(function (eventName) {
      dropzone.addEventListener(eventName, function (event) {
        event.preventDefault();
        dropzone.classList.remove("is-dragging");
      });
    });
    dropzone.addEventListener("drop", function (event) {
      var file = event.dataTransfer && event.dataTransfer.files && event.dataTransfer.files[0];
      if (file) runOperation(function () { return importFile(file); });
    });
    dropzone.addEventListener("click", function () { input.click(); });
    dropzone.addEventListener("keydown", function (event) {
      if (event.key === "Enter" || event.key === " ") input.click();
    });
    qs("[data-paper-filter]").addEventListener("input", renderPaperList);
    qs("[data-paper-reader-back]").addEventListener("click", function () { runOperation(closeReader); });
    qs("[data-paper-back]").addEventListener("click", function () { runOperation(closeReader); });
    qs("[data-paper-prev]").addEventListener("click", function () { goToPage(state.currentPage - 1); });
    qs("[data-paper-next]").addEventListener("click", function () { goToPage(state.currentPage + 1); });
    qs("[data-paper-page-input]").addEventListener("change", function (event) { goToPage(event.target.value); });
    qs("[data-paper-zoom-out]").addEventListener("click", function () {
      runOperation(function () { state.scale = Math.max(0.6, state.scale - 0.1); return updateRenderedScale(); });
    });
    qs("[data-paper-zoom-in]").addEventListener("click", function () {
      runOperation(function () { state.scale = Math.min(2.2, state.scale + 0.1); return updateRenderedScale(); });
    });
    qs("[data-paper-pdf-search]").addEventListener("change", function (event) { runOperation(function () { return searchPdf(event.target.value); }); });
    qs("[data-paper-note-editor]").addEventListener("input", autosaveNote);
    qs("[data-paper-page-note]").addEventListener("click", async function () {
      if (!state.paper) return;
      var comment = window.prompt("写下这一页的笔记：");
      if (!comment || !comment.trim()) return;
      var annotation = {
        id: uid("annotation"),
        paperId: state.paper.id,
        type: "note",
        page: state.currentPage,
        rects: [],
        quote: "",
        color: "#f1a46a",
        comment: comment.trim(),
        createdAt: now(),
        updatedAt: now()
      };
      try {
        await saveAnnotation(annotation);
        state.annotations.push(annotation);
        renderAnnotationList();
        setStatus("页码笔记已保存。", "success");
      } catch (error) { reportError(error); }
    });
    qs("[data-paper-delete]").addEventListener("click", async function () {
      if (!state.paper || !window.confirm("删除这篇论文及其本地笔记、批注？")) return;
      await runOperation(async function () {
        await deletePaper(state.paper.id);
        state.paper = null; // Do not flush the deleted note back into the database.
        await closeReader();
        setStatus("论文及其本地数据已删除。", "success");
      });
    });
    qs("[data-paper-export-markdown]").addEventListener("click", async function () {
      if (!state.paper) return;
      downloadText(safeFilename(state.paper.title) + "-notes-" + exportTimestamp() + ".md", exportMarkdown(state.paper, { markdown: qs("[data-paper-note-editor]").value }, state.annotations), "text/markdown;charset=utf-8");
    });
    qs("[data-paper-export-json]").addEventListener("click", async function () {
      if (!state.paper) return;
      downloadText(safeFilename(state.paper.title) + "-notes-" + exportTimestamp() + ".json", exportPaperJson(state.paper, { markdown: qs("[data-paper-note-editor]").value }, state.annotations), "application/json;charset=utf-8");
    });
    Array.prototype.forEach.call(root.querySelectorAll("[data-paper-highlight-color]"), function (button) {
      button.addEventListener("click", function () { saveSelection(button.dataset.paperHighlightColor, "").catch(reportError); });
    });
    qs("[data-paper-selection-note]").addEventListener("click", function () {
      var comment = window.prompt("给这段引用加一句批注：");
      if (comment && comment.trim()) saveSelection("#f1a46a", comment.trim()).catch(reportError);
    });
    bindMetadataInputs();
  }

  async function openHashLink() {
    var params = new URLSearchParams(window.location.hash.slice(1));
    if (params.has("paper")) await openPaper(params.get("paper"), Number(params.get("page")) || 1);
  }

  async function boot() {
    try { state.db = await openPaperDatabase(); }
    catch (_) {
      state.memory = {};
      STORE_NAMES.forEach(function (name) { state.memory[name] = new Map(); });
      qs("[data-paper-storage-warning]").hidden = false;
    }
    await refreshPaperList();
    bindEvents();
    setStatus(state.memory ? "临时论文库已准备好，刷新会丢失数据。" : "论文库已准备好。", state.memory ? "info" : "success");
    await runOperation(openHashLink);
  }

  window.AtlasPaperLibrary = {
    openPaperDatabase: openPaperDatabase,
    listPapers: listPapers,
    getPaper: getPaper,
    createPaper: createPaper,
    updatePaper: updatePaper,
    deletePaper: deletePaper,
    listAnnotations: listAnnotations,
    saveAnnotation: saveAnnotation,
    deleteAnnotation: deleteAnnotation,
    getNote: getNote,
    saveNote: saveNote,
    exportMarkdown: exportMarkdown,
    exportPaperJson: exportPaperJson
  };

  if (window.document$ && typeof window.document$.subscribe === "function") {
    window.document$.subscribe(function () {
      var pageRoot = document.querySelector("[data-atlas-paper-library]");
      if (pageRoot && pageRoot === root && !root.dataset.booted) {
        root.dataset.booted = "true";
        boot().catch(reportError);
      }
    });
  } else {
    boot().catch(reportError);
  }
})();

