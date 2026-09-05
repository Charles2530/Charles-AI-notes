/*! atlas-ai-build: 20260827-0001 */
(function () {
  "use strict";

  const LOCAL_HOSTS = new Set(["127.0.0.1", "localhost"]);
  if (window.location.protocol !== "http:" || !LOCAL_HOSTS.has(window.location.hostname)) return;

  const MAX_HISTORY = 12;
  const MAX_HISTORY_ITEM_CHARS = 8_000;
  const MAX_HISTORY_CHARS = 48_000;
  const MAX_CONTEXT_CHARS = 100_000;
  const MAX_REQUEST_BYTES = 128 * 1024;
  const GENERIC_ERROR = "AI 服务暂时不可用，请稍后重试。";
  const BLOCK_SELECTOR = "p, li, td, th, pre, blockquote";
  const RENDERER_ASSET_VERSION = "20260825-1943";
  const scriptBaseUrl = new URL(".", document.currentScript ? document.currentScript.src : window.location.href);
  const versionedAsset = (path) => new URL(`${path}?v=${RENDERER_ASSET_VERSION}`, scriptBaseUrl).href;
  const MARKED_URL = versionedAsset("vendor/marked/marked.umd.js");
  const DOMPURIFY_URL = versionedAsset("vendor/dompurify/purify.min.js");
  const RENDERER_URL = versionedAsset("ai-answer-renderer.min.js");
  const proxyHost = window.location.hostname === "localhost" ? "localhost" : "127.0.0.1";
  const API_ORIGIN = `http://${proxyHost}:8765`;
  let markdownRuntimePromise = null;
  let markdownConfigured = false;

  const state = {
    path: "",
    messages: [],
    reference: null,
    selectionCandidate: null,
    busy: false,
    controller: null,
    elements: null,
    selectionBound: false,
    fullscreen: false
  };

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function button(className, label, title) {
    const node = element("button", className, label);
    node.type = "button";
    node.title = title;
    node.setAttribute("aria-label", title);
    return node;
  }

  function mountShell() {
    const existing = document.querySelector(".atlas-ai-root");
    if (existing && state.elements) return state.elements;

    const root = element("div", "atlas-ai-root");
    const selectionAction = button("atlas-ai-selection-action", "✦ 问 AI", "引用选中文字并询问 AI");
    selectionAction.hidden = true;

    const launcher = button("atlas-ai-launcher", "✦", "打开 AI 阅读助手");
    launcher.setAttribute("aria-expanded", "false");

    const drawer = element("section", "atlas-ai-drawer");
    drawer.hidden = true;
    drawer.setAttribute("aria-label", "AI 阅读助手");
    drawer.setAttribute("aria-hidden", "true");

    const header = element("header", "atlas-ai-header");
    const mark = element("span", "atlas-ai-mark", "✦");
    mark.setAttribute("aria-hidden", "true");
    const headingGroup = element("div", "atlas-ai-heading-group");
    const heading = element("strong", "atlas-ai-heading", "AI 阅读助手");
    const status = element("span", "atlas-ai-status", "正在连接本地服务…");
    status.setAttribute("role", "status");
    headingGroup.append(heading, status);
    const windowControls = element("div", "atlas-ai-window-controls");
    const fullscreen = button("atlas-ai-fullscreen", "", "全屏显示");
    fullscreen.setAttribute("aria-pressed", "false");
    windowControls.append(fullscreen);
    const clear = button("atlas-ai-icon-button", "↺", "清空当前页面对话");
    const close = button("atlas-ai-icon-button", "×", "关闭 AI 阅读助手");
    header.append(mark, headingGroup, windowControls, clear, close);

    const quote = element("aside", "atlas-ai-quote");
    quote.hidden = true;
    const quoteHeader = element("div", "atlas-ai-quote__header");
    const quoteLabel = element("span", "", "当前引用");
    const quoteRemove = button("atlas-ai-quote__remove", "×", "移除当前引用");
    const quoteText = element("p", "atlas-ai-quote__text");
    quoteHeader.append(quoteLabel, quoteRemove);
    quote.append(quoteHeader, quoteText);

    const empty = element("div", "atlas-ai-empty");
    empty.append(
      element("strong", "", "可以直接询问整篇文章"),
      element("p", "", "直接提问会读取当前文章；也可以先划线，让回答更聚焦。")
    );

    const messages = element("div", "atlas-ai-messages");
    messages.setAttribute("aria-live", "polite");
    messages.appendChild(empty);

    const error = element("div", "atlas-ai-error", GENERIC_ERROR);
    error.hidden = true;
    error.setAttribute("role", "alert");

    const form = element("form", "atlas-ai-composer");
    const textarea = element("textarea", "atlas-ai-input");
    textarea.rows = 2;
    textarea.maxLength = 4000;
    textarea.placeholder = "询问整篇文章…";
    textarea.setAttribute("aria-label", "向 AI 提问");
    const send = button("atlas-ai-send", "↑", "发送问题");
    send.type = "submit";
    const hint = element("small", "atlas-ai-hint", "Enter 发送 · Shift + Enter 换行");
    form.append(textarea, send, hint);

    drawer.append(header, quote, messages, error, form);
    root.append(selectionAction, launcher, drawer);
    document.body.appendChild(root);

    state.elements = {
      root,
      selectionAction,
      launcher,
      drawer,
      status,
      fullscreen,
      clear,
      close,
      quote,
      quoteRemove,
      quoteText,
      messages,
      empty,
      error,
      form,
      textarea,
      send
    };

    selectionAction.addEventListener("pointerdown", (event) => event.preventDefault());
    selectionAction.addEventListener("click", openFromSelection);
    launcher.addEventListener("click", toggleDrawer);
    fullscreen.addEventListener("click", toggleFullscreen);
    close.addEventListener("click", closeDrawer);
    clear.addEventListener("click", clearConversation);
    quoteRemove.addEventListener("click", clearReference);
    form.addEventListener("submit", submitQuestion);
    textarea.addEventListener("keydown", handleComposerKeydown);
    send.addEventListener("click", (event) => {
      if (!state.busy) return;
      event.preventDefault();
      stopGeneration();
    });
    document.addEventListener("keydown", handleGlobalKeydown);

    return state.elements;
  }

  function openDrawer() {
    const ui = mountShell();
    ui.drawer.hidden = false;
    ui.drawer.setAttribute("aria-hidden", "false");
    ui.launcher.setAttribute("aria-expanded", "true");
    document.body.classList.add("atlas-ai-open");
  }

  function closeDrawer() {
    if (!state.elements) return;
    state.elements.drawer.hidden = true;
    state.elements.drawer.setAttribute("aria-hidden", "true");
    state.elements.launcher.setAttribute("aria-expanded", "false");
    document.body.classList.remove("atlas-ai-open");
  }

  function setFullscreen(enabled) {
    const ui = mountShell();
    state.fullscreen = Boolean(enabled);
    ui.drawer.classList.toggle("atlas-ai-drawer--fullscreen", state.fullscreen);
    ui.fullscreen.title = state.fullscreen ? "退出全屏" : "全屏显示";
    ui.fullscreen.setAttribute("aria-label", ui.fullscreen.title);
    ui.fullscreen.setAttribute("aria-pressed", state.fullscreen ? "true" : "false");
  }

  function toggleFullscreen() {
    setFullscreen(!state.fullscreen);
  }

  function toggleDrawer() {
    const ui = mountShell();
    if (ui.drawer.hidden) {
      hideSelectionAction();
      const selection = window.getSelection();
      if (selection) selection.removeAllRanges();
      openDrawer();
    } else {
      closeDrawer();
    }
  }

  function openFromSelection() {
    if (!state.selectionCandidate) return;
    state.reference = state.selectionCandidate;
    hideSelectionAction();
    renderReference();
    openDrawer();
    state.elements.textarea.focus();
  }

  function clearReference() {
    state.reference = null;
    renderReference();
  }

  function clearConversation() {
    stopGeneration();
    state.messages = [];
    state.reference = null;
    state.elements.error.hidden = true;
    renderReference();
    renderMessages();
  }

  function normalizeArticleText(value) {
    return String(value || "")
      .split(/\r?\n/)
      .map((line) => line.replace(/\s+/g, " ").trim())
      .filter(Boolean)
      .join("\n");
  }

  function sliceCodePoints(value, maximum) {
    return Array.from(String(value || "")).slice(0, maximum).join("");
  }

  function currentArticleReference() {
    const content = document.querySelector(".md-content__inner");
    if (!content) return null;
    const contextCharacters = Array.from(
      normalizeArticleText(content.innerText || content.textContent)
    );
    const context = contextCharacters.slice(0, MAX_CONTEXT_CHARS).join("");
    if (!context) return null;
    const title = content.querySelector("h1") || document.querySelector("h1");
    return {
      selection: "",
      context,
      title: sliceCodePoints(
        title ? title.textContent.replace(/\s+/g, " ").trim() : document.title,
        300
      ),
      path: window.location.pathname
    };
  }

  function referenceForRequest() {
    return state.reference || currentArticleReference();
  }

  function handleComposerKeydown(event) {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      state.elements.form.requestSubmit();
    }
  }

  function handleGlobalKeydown(event) {
    if (event.key !== "Escape") return;
    if (state.elements && !state.elements.selectionAction.hidden) {
      hideSelectionAction();
      return;
    }
    closeDrawer();
  }

  function getTextNodeElement(node) {
    return node && (node.nodeType === Node.ELEMENT_NODE ? node : node.parentElement);
  }

  function selectedBlocks(content, range) {
    const blocks = Array.from(content.querySelectorAll(BLOCK_SELECTOR));
    const matching = blocks.filter((node) => {
      try {
        return range.intersectsNode(node);
      } catch (_) {
        return false;
      }
    });
    if (matching.length) return matching;
    const start = getTextNodeElement(range.startContainer);
    const closest = start && start.closest(BLOCK_SELECTOR);
    return closest && content.contains(closest) ? [closest] : [];
  }

  function captureSelection() {
    const content = document.querySelector(".md-content__inner");
    const selection = window.getSelection();
    if (!content || !selection || selection.rangeCount === 0 || selection.isCollapsed) {
      hideSelectionAction();
      return;
    }

    const anchor = getTextNodeElement(selection.anchorNode);
    const focus = getTextNodeElement(selection.focusNode);
    if (!anchor || !focus || !content.contains(anchor) || !content.contains(focus)) {
      hideSelectionAction();
      return;
    }
    if (anchor.closest(".atlas-ai-root") || focus.closest(".atlas-ai-root")) {
      hideSelectionAction();
      return;
    }

    const selectionText = selection.toString().replace(/\s+/g, " ").trim();
    if (!selectionText) {
      hideSelectionAction();
      return;
    }

    const range = selection.getRangeAt(0);
    const blocks = selectedBlocks(content, range);
    const context = blocks
      .map((node) => node.textContent.replace(/\s+/g, " ").trim())
      .filter(Boolean)
      .join("\n");
    const title = (content.querySelector("h1") || document.querySelector("h1"));
    state.selectionCandidate = {
      selection: sliceCodePoints(selectionText, 4000),
      context: sliceCodePoints(context || selectionText, 10000),
      title: sliceCodePoints(
        title ? title.textContent.replace(/\s+/g, " ").trim() : document.title,
        300
      ),
      path: window.location.pathname
    };

    const rect = range.getBoundingClientRect();
    showSelectionAction(rect);
  }

  function showSelectionAction(rect) {
    const action = mountShell().selectionAction;
    action.hidden = false;
    const width = action.offsetWidth || 76;
    const left = Math.min(
      window.innerWidth - width - 8,
      Math.max(8, rect.left + rect.width / 2 - width / 2)
    );
    const top = Math.max(8, rect.top - 42);
    action.style.left = `${left}px`;
    action.style.top = `${top}px`;
  }

  function hideSelectionAction() {
    state.selectionCandidate = null;
    if (state.elements) state.elements.selectionAction.hidden = true;
  }

  function renderReference() {
    const ui = mountShell();
    ui.quote.hidden = !state.reference;
    ui.quoteText.textContent = state.reference ? state.reference.selection : "";
    ui.textarea.placeholder = state.reference ? "继续追问当前内容…" : "询问整篇文章…";
  }

  function appendInlineCode(target, text) {
    text.split(/(`[^`\n]+`)/g).filter(Boolean).forEach((part) => {
      if (part.startsWith("`") && part.endsWith("`")) {
        target.appendChild(element("code", "", part.slice(1, -1)));
      } else {
        target.appendChild(document.createTextNode(part));
      }
    });
  }

  function loadRuntimeScript(url, globalName, runtimeName) {
    if (window[globalName]) return Promise.resolve();
    return new Promise((resolve, reject) => {
      const existing = document.querySelector(`script[data-atlas-runtime="${runtimeName}"]`);
      const script = existing || document.createElement("script");
      const finish = () => {
        if (window[globalName]) {
          resolve();
        } else {
          reject(new Error(`${runtimeName} unavailable`));
        }
      };
      script.addEventListener("load", finish, { once: true });
      script.addEventListener("error", () => reject(new Error(`${runtimeName} unavailable`)), { once: true });
      if (!existing) {
        script.src = url;
        script.async = true;
        script.dataset.atlasRuntime = runtimeName;
        document.head.appendChild(script);
      }
    });
  }

  function ensureMarkdownRuntime() {
    if (window.marked && window.DOMPurify && window.AtlasAIAnswerRenderer) {
      markdownConfigured = true;
      return Promise.resolve();
    }
    if (!markdownRuntimePromise) {
      markdownRuntimePromise = Promise.all([
        loadRuntimeScript(MARKED_URL, "marked", "marked"),
        loadRuntimeScript(DOMPURIFY_URL, "DOMPurify", "dompurify"),
        loadRuntimeScript(RENDERER_URL, "AtlasAIAnswerRenderer", "ai-answer-renderer")
      ]).then(() => {
        markdownConfigured = true;
      }).catch((error) => {
        console.warn(`AI Markdown renderer unavailable category=${error.name}`);
        throw error;
      });
    }
    return markdownRuntimePromise;
  }

  function renderSafeMarkdown(target, source) {
    target.replaceChildren();
    String(source || "")
      .split(/\n{2,}/)
      .map((block) => block.replace(/\s+/g, " ").trim())
      .filter(Boolean)
      .forEach((block) => {
        const node = element("p");
        appendInlineCode(node, block);
        target.appendChild(node);
      });
  }

  function renderAssistantMarkdown(body, message) {
    const source = message.content || (message.pending ? "正在思考…" : "");
    if (markdownConfigured && window.AtlasAIAnswerRenderer) {
      try {
        window.AtlasAIAnswerRenderer.render(body, source);
      } catch (error) {
        console.warn(`AI Markdown render failed category=${error.name}`);
        renderSafeMarkdown(body, source);
      }
    } else {
      renderSafeMarkdown(body, source);
    }
    if (
      !state.busy &&
      !message.pending &&
      body.querySelector(".arithmatex") &&
      typeof window.atlasTypesetMath === "function"
    ) {
      Promise.resolve(window.atlasTypesetMath(body)).catch((error) => {
        console.warn(`AI math render failed category=${error.name}`);
      });
    }
  }

  function renderMessages() {
    const ui = mountShell();
    ui.messages.replaceChildren();
    if (!state.messages.length) {
      ui.messages.appendChild(ui.empty);
      return;
    }

    state.messages.forEach((message) => {
      const article = element("article", `atlas-ai-message atlas-ai-message--${message.role}`);
      if (message.role === "assistant") {
        const body = element("div", "atlas-ai-answer");
        renderAssistantMarkdown(body, message);
        article.appendChild(body);
        if (message.content && !message.pending) {
          const copy = button("atlas-ai-copy", "复制", "复制这条回答");
          copy.addEventListener("click", async () => {
            try {
              await navigator.clipboard.writeText(message.content);
              copy.textContent = "已复制";
            } catch (_) {
              copy.textContent = "复制失败";
            }
          });
          article.appendChild(copy);
        }
      } else {
        article.textContent = message.content;
      }
      ui.messages.appendChild(article);
    });
    ui.messages.scrollTop = ui.messages.scrollHeight;
  }

  function setBusy(busy) {
    state.busy = busy;
    state.elements.textarea.disabled = busy;
    state.elements.send.textContent = busy ? "■" : "↑";
    state.elements.send.title = busy ? "停止生成" : "发送问题";
    state.elements.send.setAttribute("aria-label", state.elements.send.title);
  }

  function stopGeneration() {
    if (state.controller) state.controller.abort();
    state.controller = null;
    setBusy(false);
    for (let index = state.messages.length - 1; index >= 0; index -= 1) {
      if (!state.messages[index].pending) continue;
      state.messages[index].pending = false;
      break;
    }
    renderMessages();
  }

  function setModelStatus(model) {
    const name = String(model || "").trim();
    if (!name) return;
    const ui = mountShell();
    ui.status.textContent = name;
    ui.status.classList.add("atlas-ai-status--online");
    ui.launcher.classList.remove("atlas-ai-launcher--offline");
  }

  function setServiceUnavailableStatus() {
    const ui = mountShell();
    ui.status.textContent = "AI 服务暂时不可用";
    ui.status.classList.remove("atlas-ai-status--online");
    ui.launcher.classList.add("atlas-ai-launcher--offline");
  }

  function historyForRequest() {
    const candidates = state.messages
      .filter((message) => !message.pending && !message.failed && message.content)
      .slice(-MAX_HISTORY);
    const history = [];
    let totalCharacters = 0;
    for (let index = candidates.length - 1; index >= 0; index -= 1) {
      const item = candidates[index];
      const available = MAX_HISTORY_CHARS - totalCharacters;
      if (available <= 0) break;
      const content = sliceCodePoints(
        item.content,
        Math.min(MAX_HISTORY_ITEM_CHARS, available)
      );
      if (!content) continue;
      history.unshift({ role: item.role, content });
      totalCharacters += Array.from(content).length;
    }
    return history;
  }

  function fitPayloadToRequestLimit(payload) {
    while (payload.history.length && encodedPayloadSize(payload) > MAX_REQUEST_BYTES) {
      payload.history.shift();
    }
    if (encodedPayloadSize(payload) <= MAX_REQUEST_BYTES) return payload;

    const contextCharacters = Array.from(payload.context);
    let low = 0;
    let high = contextCharacters.length;
    while (low < high) {
      const middle = Math.ceil((low + high) / 2);
      payload.context = contextCharacters.slice(0, middle).join("");
      if (encodedPayloadSize(payload) <= MAX_REQUEST_BYTES) {
        low = middle;
      } else {
        high = middle - 1;
      }
    }
    payload.context = contextCharacters.slice(0, low).join("").trim();
    return payload;
  }

  function encodedPayloadSize(payload) {
    return new TextEncoder().encode(JSON.stringify(payload)).length;
  }

  async function submitQuestion(event) {
    event.preventDefault();
    if (state.busy) return;
    const question = state.elements.textarea.value.trim();
    if (!question) return;
    const reference = referenceForRequest();
    if (!reference) {
      state.elements.error.hidden = false;
      console.warn("AI context unavailable category=missing_article");
      return;
    }

    const history = historyForRequest();
    void ensureMarkdownRuntime().then(() => {
      if (state.elements) renderMessages();
    }).catch(() => {});
    state.elements.error.hidden = true;
    state.messages.push({ role: "user", content: question });
    const answer = { role: "assistant", content: "", pending: true, failed: false };
    state.messages.push(answer);
    state.elements.textarea.value = "";
    renderMessages();
    setBusy(true);

    const controller = new AbortController();
    state.controller = controller;
    const payload = fitPayloadToRequestLimit({
      title: reference.title,
      path: reference.path,
      selection: reference.selection,
      context: reference.context,
      question,
      history
    });

    try {
      const response = await fetch(`${API_ORIGIN}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: controller.signal
      });
      if (!response.ok || !response.body) throw new Error("service unavailable");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let complete = false;
      while (!complete) {
        const chunk = await reader.read();
        if (chunk.done) break;
        buffer += decoder.decode(chunk.value, { stream: true });
        const parts = buffer.split("\n\n");
        buffer = parts.pop() || "";
        parts.forEach((part) => {
          const line = part.split("\n").find((item) => item.startsWith("data: "));
          if (!line) return;
          const streamEvent = JSON.parse(line.slice(6));
          if (streamEvent.type === "delta") {
            answer.content += streamEvent.delta || "";
            renderMessages();
          } else if (streamEvent.type === "model") {
            setModelStatus(streamEvent.model);
          } else if (streamEvent.type === "done") {
            complete = true;
          } else if (streamEvent.type === "error") {
            throw new Error("stream unavailable");
          }
        });
      }
      if (!complete) throw new Error("stream unavailable");
      answer.pending = false;
    } catch (error) {
      if (error.name !== "AbortError") {
        answer.pending = false;
        answer.failed = true;
        if (!answer.content) answer.content = GENERIC_ERROR;
        state.elements.error.hidden = false;
      }
    } finally {
      if (state.controller === controller) state.controller = null;
      setBusy(false);
      renderMessages();
      state.elements.textarea.focus();
    }
  }

  async function checkHealth() {
    mountShell();
    try {
      const response = await fetch(`${API_ORIGIN}/health`, { cache: "no-store" });
      const health = response.ok ? await response.json() : { ready: false };
      if (health.ready) {
        setModelStatus(health.model);
      } else {
        setServiceUnavailableStatus();
      }
    } catch (_) {
      setServiceUnavailableStatus();
    }
  }

  function bindSelectionEvents() {
    if (state.selectionBound) return;
    state.selectionBound = true;
    document.addEventListener("pointerup", () => window.setTimeout(captureSelection, 0));
    document.addEventListener("keyup", (event) => {
      if (event.key.startsWith("Arrow") || event.key === "Shift") {
        window.setTimeout(captureSelection, 0);
      }
    });
    document.addEventListener("scroll", hideSelectionAction, true);
  }

  function initPage() {
    if (!document.body) return;
    mountShell();
    bindSelectionEvents();
    const path = window.location.pathname;
    if (state.path && state.path !== path) {
      stopGeneration();
      state.messages = [];
      state.reference = null;
      setFullscreen(false);
      renderReference();
      renderMessages();
      closeDrawer();
    }
    state.path = path;
    hideSelectionAction();
    checkHealth();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initPage);
  } else {
    initPage();
  }

  if (window.document$ && typeof window.document$.subscribe === "function") {
    window.document$.subscribe(initPage);
  }
})();
