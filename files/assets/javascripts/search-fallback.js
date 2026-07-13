(function () {
  const FALLBACK_ATTR = "data-atlas-search-fallback";
  const MIN_QUERY_LENGTH = 1;
  const MAX_RESULTS = 12;
  const SEARCH_DELAY_MS = 520;

  let indexPromise = null;
  let renderTimer = 0;
  let bindTimer = 0;

  function readConfig() {
    const node = document.getElementById("__config");
    if (!node) return {};
    try {
      return JSON.parse(node.textContent || "{}");
    } catch (_) {
      return {};
    }
  }

  function siteRoot() {
    const config = readConfig();
    const base = config.base || ".";
    return new URL(base.endsWith("/") ? base : `${base}/`, window.location.href);
  }

  function indexUrl() {
    return new URL("search/search_index_full.json", siteRoot()).href;
  }

  function loadIndex() {
    if (!indexPromise) {
      indexPromise = fetch(indexUrl(), { credentials: "same-origin" })
        .then((response) => {
          if (!response.ok) throw new Error(`search index ${response.status}`);
          return response.json();
        })
        .then((payload) => Array.isArray(payload.docs) ? payload.docs : []);
    }
    return indexPromise;
  }

  function escapeHtml(value) {
    return String(value || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function stripHtml(value) {
    const text = String(value || "")
      .replace(/<[^>]+>/g, " ")
      .replace(/&nbsp;/g, " ")
      .replace(/&amp;/g, "&")
      .replace(/&lt;/g, "<")
      .replace(/&gt;/g, ">")
      .replace(/\s+/g, " ")
      .trim();
    return text;
  }

  function normalize(value) {
    return stripHtml(value).toLowerCase();
  }

  function queryTerms(query) {
    const normalized = normalize(query);
    const spaced = normalized
      .replace(/([a-z0-9])([\u4e00-\u9fff])/g, "$1 $2")
      .replace(/([\u4e00-\u9fff])([a-z0-9])/g, "$1 $2");
    const parts = spaced.split(/[\s,，。；;:：/|]+/).filter(Boolean);
    return Array.from(new Set(parts));
  }

  function scoreDoc(doc, terms, rawQuery) {
    const title = normalize(doc.title);
    const text = normalize(doc.text);
    const haystack = `${title} ${text}`;
    const fullQuery = normalize(rawQuery);
    let score = 0;

    if (title.includes(fullQuery)) score += 80;
    if (haystack.includes(fullQuery)) score += 25;

    for (const term of terms) {
      if (!haystack.includes(term)) return 0;
      if (title.includes(term)) score += 40;
      const matches = haystack.split(term).length - 1;
      score += Math.min(matches, 12) * 3;
    }

    if (doc.location && !String(doc.location).includes("#")) score += 8;
    return score;
  }

  function snippet(doc, terms, rawQuery) {
    const text = stripHtml(doc.text);
    const lower = text.toLowerCase();
    const candidates = [normalize(rawQuery), ...terms].filter(Boolean);
    let index = -1;

    for (const term of candidates) {
      index = lower.indexOf(term);
      if (index >= 0) break;
    }

    if (index < 0) index = 0;
    const start = Math.max(0, index - 56);
    const end = Math.min(text.length, index + 156);
    const prefix = start > 0 ? "..." : "";
    const suffix = end < text.length ? "..." : "";
    return `${prefix}${text.slice(start, end)}${suffix}`;
  }

  function resultHref(location) {
    if (!location) return siteRoot().href;
    return new URL(location, siteRoot()).href;
  }

  function clearFallback(list) {
    list.querySelectorAll(`[${FALLBACK_ATTR}]`).forEach((node) => node.remove());
  }

  function hasNativeResults(list) {
    return Array.from(list.children).some((node) => !node.hasAttribute(FALLBACK_ATTR));
  }

  function renderResults(query, docs) {
    const input = document.querySelector("[data-md-component='search-query']");
    const list = document.querySelector(".md-search-result__list");
    const meta = document.querySelector(".md-search-result__meta");
    if (!list || !meta) return;
    if (input && input.value !== query) return;

    clearFallback(list);
    if (hasNativeResults(list)) return;

    const terms = queryTerms(query);
    if (terms.length === 0 || normalize(query).length < MIN_QUERY_LENGTH) {
      meta.textContent = "键入以开始搜索";
      return;
    }

    const results = docs
      .map((doc) => ({ doc, score: scoreDoc(doc, terms, query) }))
      .filter((item) => item.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, MAX_RESULTS);

    meta.textContent = results.length
      ? `找到 ${results.length} 个符合条件的结果`
      : "没有找到符合条件的结果";

    const fragment = document.createDocumentFragment();
    for (const { doc } of results) {
      const item = document.createElement("li");
      item.className = "md-search-result__item atlas-search-fallback__item";
      item.setAttribute(FALLBACK_ATTR, "true");
      item.innerHTML = `
        <a class="md-search-result__link" href="${escapeHtml(resultHref(doc.location))}">
          <article class="md-search-result__article md-typeset">
            <div class="md-search-result__title">${escapeHtml(stripHtml(doc.title || "未命名页面"))}</div>
            <p class="md-search-result__teaser">${escapeHtml(snippet(doc, terms, query))}</p>
          </article>
        </a>
      `;
      fragment.appendChild(item);
    }
    list.appendChild(fragment);
  }

  function scheduleFallback(input) {
    window.clearTimeout(renderTimer);
    const query = input.value || "";
    renderTimer = window.setTimeout(() => {
      const list = document.querySelector(".md-search-result__list");
      if (list && hasNativeResults(list)) return;

      loadIndex()
        .then((docs) => renderResults(query, docs))
        .catch(() => {
          const meta = document.querySelector(".md-search-result__meta");
          if (meta) meta.textContent = "搜索索引暂时不可用";
        });
    }, SEARCH_DELAY_MS);
  }

  function bindSearchFallback() {
    const input = document.querySelector("[data-md-component='search-query']");
    const list = document.querySelector(".md-search-result__list");
    if (!input || !list) {
      window.clearTimeout(bindTimer);
      bindTimer = window.setTimeout(bindSearchFallback, 350);
      return;
    }

    if (!list.dataset.atlasFallbackObserver) {
      const observer = new MutationObserver(() => {
        if (hasNativeResults(list)) clearFallback(list);
      });
      observer.observe(list, { childList: true });
      list.dataset.atlasFallbackObserver = "true";
    }

    if (input.dataset.atlasFallbackBound) return;

    input.dataset.atlasFallbackBound = "true";
    input.addEventListener("input", () => scheduleFallback(input));
    input.addEventListener("focus", () => {
      if (input.value) scheduleFallback(input);
    });

    const form = input.closest("form");
    if (form) {
      form.addEventListener("reset", () => {
        window.setTimeout(() => {
          clearFallback(list);
          const meta = document.querySelector(".md-search-result__meta");
          if (meta) meta.textContent = "键入以开始搜索";
        }, 0);
      });
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bindSearchFallback);
  } else {
    bindSearchFallback();
  }

  if (window.document$ && typeof window.document$.subscribe === "function") {
    window.document$.subscribe(bindSearchFallback);
  }
})();
