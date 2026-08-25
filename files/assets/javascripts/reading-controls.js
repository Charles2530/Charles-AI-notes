(function () {
  const STORAGE_KEY = "atlas-reading-preferences";
  const SIDEBAR_NAV_STORAGE_KEY = "atlas-sidebar-nav-open-topics";
  const ATLAS_READING_STATUS_STORAGE_KEY = "atlas-reading-status";
  const ATLAS_GLOSSARY_MASTERY_STORAGE_KEY = "atlas-glossary-mastery";
  const READ_MARKER_CLASS = "atlas-nav-read-status--read";
  const UNREAD_MARKER_CLASS = "atlas-nav-read-status--unread";

  function loadPrefs() {
    try {
      return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
    } catch (_) {
      return {};
    }
  }

  function savePrefs(prefs) {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(prefs));
    } catch (_) {
      // Ignore storage failures in private browsing or locked environments.
    }
  }

  function applyPrefs(prefs) {
    const body = document.body;
    body.classList.toggle("atlas-wide", !!prefs.wide);
    body.classList.toggle("atlas-focus", !!prefs.focus);
    body.classList.toggle("atlas-hide-left", !!prefs.hideLeft);
    body.classList.toggle("atlas-hide-right", !!prefs.hideRight);
    body.classList.remove("atlas-zoom-0", "atlas-zoom-1", "atlas-zoom-2");
    body.classList.add(`atlas-zoom-${prefs.zoom || 0}`);
    window.requestAnimationFrame(() => {
      window.dispatchEvent(new CustomEvent("atlas:layoutchange"));
    });
  }

  function markPageContext() {
    const path = window.location.pathname.replace(/\/index\.html$/, "/");
    document.body.classList.toggle("atlas-paper-deep-dive", path.includes("/paper-deep-dives/"));
    document.body.classList.toggle("atlas-home", !!document.querySelector(".md-content__inner .atlas-hero"));
  }

  function directChild(parent, selector) {
    return Array.from(parent ? parent.children : []).find((child) => child.matches(selector));
  }

  function loadSidebarNavState() {
    try {
      return JSON.parse(localStorage.getItem(SIDEBAR_NAV_STORAGE_KEY) || "{}");
    } catch (_) {
      return {};
    }
  }

  function saveSidebarNavState(state) {
    try {
      localStorage.setItem(SIDEBAR_NAV_STORAGE_KEY, JSON.stringify(state));
    } catch (_) {
      // Ignore storage failures in private browsing or locked environments.
    }
  }

  function loadReadingStatus() {
    try {
      return JSON.parse(localStorage.getItem(ATLAS_READING_STATUS_STORAGE_KEY) || "{}");
    } catch (_) {
      return {};
    }
  }

  function saveReadingStatus(status) {
    try {
      localStorage.setItem(ATLAS_READING_STATUS_STORAGE_KEY, JSON.stringify(status));
    } catch (_) {
      // Ignore storage failures in private browsing or locked environments.
    }
  }

  function loadGlossaryMastery() {
    try {
      return JSON.parse(localStorage.getItem(ATLAS_GLOSSARY_MASTERY_STORAGE_KEY) || "{}");
    } catch (_) {
      return {};
    }
  }

  function saveGlossaryMastery(status) {
    try {
      localStorage.setItem(ATLAS_GLOSSARY_MASTERY_STORAGE_KEY, JSON.stringify(status));
    } catch (_) {
      // Ignore storage failures in private browsing or locked environments.
    }
  }

  function sidebarTopicLabel(item) {
    const link = directChild(item, "label.md-nav__link");
    if (link) return link.textContent.replace(/\s+/g, " ").trim();

    const container = directChild(item, ".md-nav__container");
    const anchor = directChild(container, "a.md-nav__link");
    return anchor ? anchor.textContent.replace(/\s+/g, " ").trim() : "";
  }

  function sidebarTopicToggle(item) {
    const link = directChild(item, "label.md-nav__link");
    if (link) return link;

    const container = directChild(item, ".md-nav__container");
    return directChild(container, "label.md-nav__link");
  }

  function sidebarTopicHeader(item) {
    return directChild(item, "label.md-nav__link") || directChild(item, ".md-nav__container");
  }

  function sidebarTopicKey(item) {
    const primary = document.querySelector(".md-sidebar--primary");
    const labels = [];
    let current = item;

    while (current && current !== primary) {
      if (current.matches && current.matches(".md-nav__item--nested")) {
        const label = sidebarTopicLabel(current);
        if (label) labels.unshift(label);
      }
      current = current.parentElement ? current.parentElement.closest(".md-nav__item--nested") : null;
    }

    return labels.join(" / ");
  }

  function setSidebarTopicExpanded(item, expanded, persist) {
    const input = directChild(item, "input.md-nav__toggle");
    const nav = directChild(item, "nav.md-nav");
    const link = sidebarTopicHeader(item);
    const toggle = sidebarTopicToggle(item);
    const label = sidebarTopicLabel(item);
    const key = sidebarTopicKey(item);

    item.classList.toggle("atlas-sidebar-nav-expanded", expanded);
    item.classList.toggle("atlas-sidebar-nav-collapsed", !expanded);
    item.classList.toggle("atlas-paper-nav-expanded", expanded);
    item.classList.toggle("atlas-paper-nav-collapsed", !expanded);
    item.dataset.atlasSidebarNavExpanded = String(expanded);

    if (input) input.checked = expanded;
    if (nav) nav.setAttribute("aria-expanded", String(expanded));
    if (link) {
      link.setAttribute("aria-expanded", String(expanded));
      link.title = expanded ? `折叠 ${label}` : `展开 ${label}`;
    }
    if (toggle && toggle !== link) {
      toggle.setAttribute("aria-expanded", String(expanded));
      toggle.title = expanded ? `折叠 ${label}` : `展开 ${label}`;
    }

    if (persist && key) {
      const state = loadSidebarNavState();
      state[key] = expanded;
      saveSidebarNavState(state);
    }
  }

  function bindSidebarTopicToggle(item) {
    const input = directChild(item, "input.md-nav__toggle");
    const link = sidebarTopicToggle(item);
    if (!input || !link || link.dataset.atlasSidebarNavBound) return;

    const nav = directChild(item, "nav.md-nav");
    if (nav && !nav.id) {
      nav.id = `${link.getAttribute("for") || "atlas-sidebar-nav"}_panel`;
    }

    link.dataset.atlasSidebarNavBound = "true";
    link.setAttribute("role", "button");
    if (nav) link.setAttribute("aria-controls", nav.id);

    input.addEventListener("change", () => {
      setSidebarTopicExpanded(item, input.checked, true);
    });

    link.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        setSidebarTopicExpanded(item, item.dataset.atlasSidebarNavExpanded !== "true", true);
      }
    });
  }

  function markSidebarTopicNav() {
    document.querySelectorAll(".atlas-sidebar-nav-topic, .atlas-paper-nav-root, .atlas-paper-nav-topic").forEach((node) => {
      node.classList.remove(
        "atlas-sidebar-nav-topic",
        "atlas-sidebar-nav-expanded",
        "atlas-sidebar-nav-collapsed",
        "atlas-paper-nav-root",
        "atlas-paper-nav-topic",
        "atlas-paper-nav-expanded",
        "atlas-paper-nav-collapsed"
      );
    });

    const primary = document.querySelector(".md-sidebar--primary");
    if (!primary) return;

    const state = loadSidebarNavState();
    primary.querySelectorAll(".md-nav__item--nested").forEach((item) => {
      const input = directChild(item, "input.md-nav__toggle");
      const link = sidebarTopicToggle(item);
      const nav = directChild(item, "nav.md-nav");
      const list = directChild(nav, "ul.md-nav__list");
      const key = sidebarTopicKey(item);
      const level = Number(nav ? nav.getAttribute("data-md-level") : 0);

      if (!input || !link || !nav || !list || !key || level < 2) return;

      item.classList.add("atlas-sidebar-nav-topic");
      if (key === "论文专题讲解") item.classList.add("atlas-paper-nav-root");
      if (key.startsWith("论文专题讲解 /")) item.classList.add("atlas-paper-nav-topic");

      const hasActivePage = !!item.querySelector(".md-nav__link--active, .md-nav__item--active");
      const expanded = hasActivePage
        ? true
        : Object.prototype.hasOwnProperty.call(state, key)
          ? !!state[key]
          : false;

      setSidebarTopicExpanded(item, expanded, false);
      bindSidebarTopicToggle(item);
    });
  }

  function applyDockState(dock, prefs, toggle) {
    const collapsed = !!prefs.controlsCollapsed;
    dock.classList.toggle("atlas-reader-controls--collapsed", collapsed);
    dock.setAttribute("aria-expanded", String(!collapsed));
    if (toggle) {
      toggle.textContent = collapsed ? "工具" : "收起";
      toggle.title = collapsed ? "展开阅读工具" : "收起阅读工具";
      toggle.setAttribute("aria-label", toggle.title);
    }
  }

  function makeButton(label, title, onClick) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "atlas-reader-button";
    button.textContent = label;
    button.title = title;
    button.addEventListener("click", onClick);
    return button;
  }

  function pageTitle() {
    const heading = document.querySelector(".md-content__inner h1");
    const title = heading ? heading.textContent.replace(/\s+/g, " ").trim() : document.title;
    return title || "AI Notes Page";
  }

  function printCurrentPage() {
    const title = pageTitle();
    const previousTitle = document.title;
    document.title = `${title} - Charles AI notes`;
    document.body.classList.add("atlas-printing");

    const cleanup = () => {
      document.body.classList.remove("atlas-printing");
      document.title = previousTitle;
      window.removeEventListener("afterprint", cleanup);
    };

    window.addEventListener("afterprint", cleanup);
    window.setTimeout(() => {
      window.print();
      window.setTimeout(cleanup, 60000);
    }, 80);
  }

  function updateReadingProgress() {
    const bar = document.querySelector(".atlas-scroll-progress__bar");
    const shell = document.querySelector(".atlas-scroll-progress");
    if (!bar || !shell) return;

    const doc = document.documentElement;
    const maxScroll = Math.max(1, doc.scrollHeight - window.innerHeight);
    const scrolled = Math.min(1, Math.max(0, (window.scrollY || doc.scrollTop || 0) / maxScroll));
    bar.style.transform = "none";
    bar.style.width = `${Math.round(scrolled * 1000) / 10}%`;
    shell.hidden = maxScroll < 80;
  }

  function mountReadingProgress() {
    if (!document.querySelector(".atlas-scroll-progress")) {
      const shell = document.createElement("div");
      shell.className = "atlas-scroll-progress";
      shell.setAttribute("aria-hidden", "true");

      const bar = document.createElement("div");
      bar.className = "atlas-scroll-progress__bar";
      shell.appendChild(bar);
      document.body.appendChild(shell);
    }

    if (!document.body.dataset.atlasProgressBound) {
      window.addEventListener("scroll", updateReadingProgress, { passive: true });
      window.addEventListener("resize", updateReadingProgress);
      window.addEventListener("atlas:layoutchange", updateReadingProgress);
      document.body.dataset.atlasProgressBound = "true";
    }

    updateReadingProgress();
    window.requestAnimationFrame(updateReadingProgress);
    window.setTimeout(updateReadingProgress, 120);
  }

  function mountReaderControls() {
    const storedPrefs = loadPrefs();
    const prefs = Object.assign({ zoom: 0, controlsCollapsed: true }, storedPrefs);
    applyPrefs(prefs);

    if (document.querySelector(".atlas-reader-controls")) return;

    const dock = document.createElement("div");
    dock.className = "atlas-reader-controls";
    dock.setAttribute("aria-label", "阅读控制");

    const toggle = makeButton("收起", "收起阅读工具", () => {
      prefs.controlsCollapsed = !prefs.controlsCollapsed;
      applyDockState(dock, prefs, toggle);
      savePrefs(prefs);
    });
    toggle.classList.add("atlas-reader-toggle");

    const zoom = makeButton("A+", "切换正文放大", () => {
      prefs.zoom = ((prefs.zoom || 0) + 1) % 3;
      applyPrefs(prefs);
      savePrefs(prefs);
    });

    const wide = makeButton("宽屏", "切换正文宽屏", () => {
      prefs.wide = !prefs.wide;
      applyPrefs(prefs);
      savePrefs(prefs);
    });

    const focus = makeButton("全屏", "切换沉浸阅读；支持浏览器全屏时会尝试进入全屏", async () => {
      prefs.focus = !prefs.focus;
      applyPrefs(prefs);
      savePrefs(prefs);
      try {
        if (prefs.focus && !document.fullscreenElement && document.documentElement.requestFullscreen) {
          await document.documentElement.requestFullscreen();
        } else if (!prefs.focus && document.fullscreenElement && document.exitFullscreen) {
          await document.exitFullscreen();
        }
      } catch (_) {
        // Browser fullscreen can be blocked; CSS focus mode still works.
      }
    });

    const left = makeButton("左栏", "折叠/展开左侧目录", () => {
      prefs.hideLeft = !prefs.hideLeft;
      applyPrefs(prefs);
      savePrefs(prefs);
    });

    const right = makeButton("右栏", "折叠/展开右侧小章节", () => {
      prefs.hideRight = !prefs.hideRight;
      applyPrefs(prefs);
      savePrefs(prefs);
    });

    const top = makeButton("顶部", "跳到当前页开头", () => {
      window.scrollTo({ top: 0, behavior: "smooth" });
    });

    const bottom = makeButton("末尾", "跳到当前页末尾", () => {
      const height = Math.max(
        document.body.scrollHeight,
        document.documentElement.scrollHeight,
        document.body.offsetHeight,
        document.documentElement.offsetHeight
      );
      window.scrollTo({ top: height, behavior: "smooth" });
    });

    const link = makeButton("链接", "复制当前页链接", async () => {
      try {
        await copyText(window.location.href);
        link.textContent = "已复制";
        window.setTimeout(() => {
          link.textContent = "链接";
        }, 1300);
      } catch (_) {
        link.textContent = "失败";
        window.setTimeout(() => {
          link.textContent = "链接";
        }, 1600);
      }
    });

    const pdf = makeButton("PDF", "导出当前页内容为 PDF；在打印窗口选择“保存为 PDF”", printCurrentPage);

    dock.append(toggle, zoom, wide, focus, left, right, top, bottom, link, pdf);
    applyDockState(dock, prefs, toggle);
    document.body.appendChild(dock);

    let idleTimer = 0;
    const markActive = () => {
      dock.classList.remove("atlas-reader-controls--idle");
      window.clearTimeout(idleTimer);
      idleTimer = window.setTimeout(() => {
        dock.classList.add("atlas-reader-controls--idle");
      }, 2600);
    };

    dock.addEventListener("pointerenter", markActive);
    dock.addEventListener("focusin", markActive);
    dock.addEventListener("pointerleave", markActive);
    dock.addEventListener("focusout", markActive);
    window.addEventListener("scroll", markActive, { passive: true });
    markActive();
  }

  function tableToText(table) {
    return Array.from(table.rows)
      .map((row) =>
        Array.from(row.cells)
          .map((cell) => cell.innerText.replace(/\s+/g, " ").trim())
          .join("\t")
      )
      .join("\n");
  }

  async function copyText(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      await navigator.clipboard.writeText(text);
      return;
    }
    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.setAttribute("readonly", "readonly");
    textarea.style.position = "fixed";
    textarea.style.left = "-9999px";
    document.body.appendChild(textarea);
    textarea.select();
    const ok = document.execCommand("copy");
    textarea.remove();
    if (!ok) throw new Error("copy failed");
  }

  function setGlossaryMasteryButtonState(button, mastered) {
    button.classList.toggle("atlas-glossary-mastery-button--mastered", mastered);
    button.textContent = "";
    button.title = mastered ? "已掌握，点击标记为未掌握" : "未掌握，点击标记为已掌握";
    button.setAttribute("aria-pressed", String(mastered));
    button.setAttribute("aria-label", button.title);
  }

  function mountGlossaryMasteryControls() {
    const path = window.location.pathname.replace(/\/index\.html$/, "/");
    if (!path.endsWith("/glossary/")) return;

    const mastery = loadGlossaryMastery();
    document.querySelectorAll(".md-content__inner table").forEach((table) => {
      const headerRow = table.tHead && table.tHead.rows[0];
      const headerCells = headerRow ? headerRow.cells : [];
      if (headerCells.length < 2) return;
      const existingStatusHeader = Array.from(headerCells).find((cell) => cell.classList.contains("atlas-glossary-status-column"));
      if (headerCells[0].textContent.trim() !== "术语" || (!existingStatusHeader && headerCells[1].textContent.trim() !== "含义")) return;

      table.dataset.atlasGlossary = "true";
      let statusHeader = existingStatusHeader;
      if (!statusHeader) {
        statusHeader = document.createElement("th");
        statusHeader.className = "atlas-glossary-status-column";
        statusHeader.scope = "col";
        headerRow.insertBefore(statusHeader, headerCells[1]);
      }
      statusHeader.textContent = "";
      statusHeader.setAttribute("aria-label", "掌握状态");

      Array.from(table.tBodies).forEach((body) => {
        Array.from(body.rows).forEach((row) => {
          const termCell = row.cells[0];
          if (!termCell) return;

          const term = termCell.textContent.replace(/\s+/g, " ").trim();
          if (!term) return;

          let statusCell = Array.from(row.cells).find((cell) => cell.classList.contains("atlas-glossary-status-column"));
          if (!statusCell) {
            statusCell = document.createElement("td");
            statusCell.className = "atlas-glossary-status-column";
            row.insertBefore(statusCell, row.cells[1]);
          }

          let button = statusCell.querySelector(".atlas-glossary-mastery-button");
          if (!button) {
            button = document.createElement("button");
            button.type = "button";
            button.className = "atlas-glossary-mastery-button";
            button.addEventListener("click", () => {
              const current = loadGlossaryMastery();
              const mastered = !Object.prototype.hasOwnProperty.call(current, term);

              if (mastered) {
                current[term] = { masteredAt: new Date().toISOString() };
              } else {
                delete current[term];
              }

              saveGlossaryMastery(current);
              setGlossaryMasteryButtonState(button, mastered);
            });
            statusCell.appendChild(button);
          }

          setGlossaryMasteryButtonState(button, Object.prototype.hasOwnProperty.call(mastery, term));
        });
      });
    });
  }

  function mountTableCopyButtons() {
    document.querySelectorAll(".md-typeset table").forEach((table) => {
      if (table.closest(".atlas-table-wrap")) return;

      const wrapper = document.createElement("div");
      wrapper.className = "atlas-table-wrap";
      table.parentNode.insertBefore(wrapper, table);

      const toolbar = document.createElement("div");
      toolbar.className = "atlas-table-tools";

      const hint = document.createElement("span");
      hint.className = "atlas-table-hint";
      hint.textContent = "可横向滚动";

      const button = document.createElement("button");
      button.type = "button";
      button.className = "atlas-table-copy";
      button.textContent = "复制";
      button.title = "复制当前表格内容，不包含术语浮窗说明";
      button.addEventListener("click", async () => {
        const text = tableToText(table);
        try {
          await copyText(text);
          button.textContent = "已复制";
          window.setTimeout(() => {
            button.textContent = "复制";
          }, 1300);
        } catch (_) {
          button.textContent = "复制失败";
          window.setTimeout(() => {
            button.textContent = "复制";
          }, 1600);
        }
      });

      const expand = document.createElement("button");
      expand.type = "button";
      expand.className = "atlas-table-copy atlas-table-expand";
      expand.textContent = "展开";
      expand.title = "展开或收起长表格";
      expand.addEventListener("click", () => {
        const collapsed = wrapper.classList.toggle("atlas-table-wrap--collapsed");
        expand.textContent = collapsed ? "展开" : "收起";
        window.dispatchEvent(new CustomEvent("atlas:layoutchange"));
      });

      toolbar.appendChild(hint);
      toolbar.appendChild(expand);
      toolbar.appendChild(button);
      wrapper.appendChild(toolbar);
      wrapper.appendChild(table);
      window.requestAnimationFrame(() => {
        if ((table.rows[0]?.cells.length || 0) > 3 || table.scrollWidth > wrapper.clientWidth + 8) {
          wrapper.classList.add("atlas-table-wrap--wide");
        }
        if (table.rows.length > 14) {
          wrapper.classList.add("atlas-table-wrap--long", "atlas-table-wrap--collapsed");
        } else {
          expand.hidden = true;
        }
      });
    });
  }

  function softenNumberedHeadings() {
    document.querySelectorAll(".md-content .md-typeset h2").forEach((heading) => {
      if (heading.dataset.atlasNumberedHeading) return;

      const firstText = Array.from(heading.childNodes).find((node) => node.nodeType === Node.TEXT_NODE && node.nodeValue.trim());
      if (!firstText) return;

      const match = firstText.nodeValue.match(/^(\s*)(\d+(?:\.\d+)*[.、])(?:\s+(.+)|\s*)$/);
      if (!match) return;

      const titleText = match[3] || "";
      const trailingNodes = [];
      let node = firstText.nextSibling;
      while (node) {
        const next = node.nextSibling;
        trailingNodes.push(node);
        node = next;
      }

      const headerlink = trailingNodes.find((child) => child.nodeType === Node.ELEMENT_NODE && child.classList.contains("headerlink"));
      const hasTrailingTitle = trailingNodes.some((child) => child !== headerlink && child.textContent.replace(/\s+/g, "").length);
      if (!titleText.trim() && !hasTrailingTitle) return;

      const number = document.createElement("span");
      number.className = "atlas-heading-number";
      number.textContent = match[2].replace(/[.、]$/, "");

      const title = document.createElement("span");
      title.className = "atlas-heading-title";
      if (titleText) {
        title.appendChild(document.createTextNode(titleText));
      }

      trailingNodes.forEach((child) => {
        if (child !== headerlink) {
          title.appendChild(child);
        }
      });
      if (headerlink) {
        title.appendChild(headerlink);
      }

      firstText.nodeValue = match[1];
      heading.insertBefore(number, firstText.nextSibling);
      heading.insertBefore(title, number.nextSibling);
      heading.classList.add("atlas-numbered-heading");
      heading.dataset.atlasNumberedHeading = "true";
    });
  }

  function parsePaperDate(value) {
    const match = String(value || "").match(/(\d{4})(?:-(\d{1,2}))?/);
    if (!match) return Number.MAX_SAFE_INTEGER;
    const year = Number(match[1]);
    const month = match[2] ? Number(match[2]) : 99;
    return year * 100 + month;
  }

  function sortPaperTable(table) {
    const body = table.tBodies && table.tBodies[0];
    if (!body) return;

    const headers = Array.from(table.tHead ? table.tHead.rows[0].cells : table.rows[0]?.cells || []);
    const dateIndex = Math.max(
      0,
      headers.findIndex((cell) => /^(时间|日期|Date)$/i.test(cell.innerText.replace(/\s+/g, " ").trim()))
    );
    const rows = Array.from(body.rows);
    if (rows.length < 2) return;

    rows
      .sort((a, b) => {
        const aDate = parsePaperDate(a.cells[dateIndex]?.innerText);
        const bDate = parsePaperDate(b.cells[dateIndex]?.innerText);
        if (aDate !== bDate) return aDate - bDate;
        return a.innerText.localeCompare(b.innerText, "zh-Hans-CN");
      })
      .forEach((row) => body.appendChild(row));
  }

  function mountPaperDeepDiveSorting() {
    document.querySelectorAll("table[data-paper-sort], [data-paper-sort] table, table.paper-deep-dive-table").forEach(sortPaperTable);

    document.querySelectorAll("[data-paper-sort]").forEach((container) => {
      const items = Array.from(container.children).filter((child) => child.hasAttribute("data-paper-date"));
      if (items.length < 2) return;

      items
        .sort((a, b) => {
          const aDate = parsePaperDate(a.getAttribute("data-paper-date"));
          const bDate = parsePaperDate(b.getAttribute("data-paper-date"));
          if (aDate !== bDate) return aDate - bDate;
          return a.innerText.localeCompare(b.innerText, "zh-Hans-CN");
        })
        .forEach((item) => container.appendChild(item));
    });
  }

  function linkInfo(anchor) {
    if (!anchor) return null;
    const title = anchor.querySelector(".md-footer__title");
    const label = title ? title.textContent.replace(/\s+/g, " ").trim() : anchor.textContent.replace(/\s+/g, " ").trim();
    return {
      href: anchor.getAttribute("href"),
      label: label || anchor.getAttribute("href")
    };
  }

  function normalizeUrl(url) {
    const parsed = new URL(url, window.location.href);
    parsed.hash = "";
    parsed.search = "";
    let path = parsed.pathname.replace(/\/index\.html$/, "/");
    if (!path.endsWith("/") && !PathLikeFile(path)) path += "/";
    return `${parsed.origin}${path}`;
  }

  function PathLikeFile(path) {
    return /\.[a-z0-9]+$/i.test(path.split("/").pop() || "");
  }

  function navLinkInfo(anchor) {
    const label = anchor.textContent.replace(/\s+/g, " ").trim();
    const href = anchor.getAttribute("href");
    if (!href || href.startsWith("#")) return null;
    return { href, label: label || href, key: normalizeUrl(anchor.href) };
  }

  function markCurrentPageRead() {
    const current = normalizeUrl(window.location.href);
    const status = loadReadingStatus();

    status[current] = Object.assign({}, status[current], {
      readAt: new Date().toISOString()
    });
    saveReadingStatus(status);
  }

  function clearSidebarReadingMarker(anchor) {
    Array.from(anchor.children)
      .filter((child) => child.classList.contains("atlas-nav-read-status"))
      .forEach((child) => child.remove());
    anchor.classList.remove("atlas-nav-link--read", "atlas-nav-link--unread");
    delete anchor.dataset.atlasReadingState;
  }

  function markSidebarReadingState() {
    const primary = document.querySelector(".md-sidebar--primary");
    if (!primary) return;

    const status = loadReadingStatus();
    const current = normalizeUrl(window.location.href);

    primary.querySelectorAll("a.md-nav__link[href]").forEach((anchor) => {
      clearSidebarReadingMarker(anchor);

      const info = navLinkInfo(anchor);
      if (!info) return;

      const url = new URL(anchor.href, window.location.href);
      if (url.origin !== window.location.origin) return;

      const read = info.key === current || Object.prototype.hasOwnProperty.call(status, info.key);
      const marker = document.createElement("span");
      marker.className = `atlas-nav-read-status ${read ? READ_MARKER_CLASS : UNREAD_MARKER_CLASS}`;
      marker.setAttribute("aria-hidden", "true");
      marker.title = read ? "已读" : "未读";

      anchor.classList.add(read ? "atlas-nav-link--read" : "atlas-nav-link--unread");
      anchor.dataset.atlasReadingState = read ? "read" : "unread";
      anchor.insertBefore(marker, anchor.firstChild);
    });
  }

  function adjacentFromPrimaryNav() {
    const links = Array.from(document.querySelectorAll(".md-sidebar--primary a.md-nav__link[href]"));
    const seen = new Set();
    const pages = [];

    links.forEach((anchor) => {
      const info = navLinkInfo(anchor);
      if (!info) return;
      const url = new URL(anchor.href, window.location.href);
      if (url.origin !== window.location.origin) return;
      if (seen.has(info.key)) return;
      seen.add(info.key);
      pages.push(info);
    });

    const current = normalizeUrl(window.location.href);
    const index = pages.findIndex((page) => page.key === current);
    if (index < 0) return { prev: null, next: null };

    return {
      prev: index > 0 ? pages[index - 1] : null,
      next: index < pages.length - 1 ? pages[index + 1] : null
    };
  }

  function mountPageNav() {
    const content = document.querySelector(".md-content__inner");
    if (!content || content.querySelector(".atlas-page-nav")) return;

    const fallback = adjacentFromPrimaryNav();
    const prev = linkInfo(document.querySelector(".md-footer__link--prev")) || fallback.prev;
    const next = linkInfo(document.querySelector(".md-footer__link--next")) || fallback.next;
    if (!prev && !next) return;

    const nav = document.createElement("nav");
    nav.className = "atlas-page-nav";
    nav.setAttribute("aria-label", "上一页和下一页");

    if (prev) {
      const prevLink = document.createElement("a");
      prevLink.className = "atlas-page-nav__link atlas-page-nav__link--prev";
      prevLink.href = prev.href;
      prevLink.innerHTML = `<span>上一页</span><strong>${prev.label}</strong>`;
      nav.appendChild(prevLink);
    }

    if (next) {
      const nextLink = document.createElement("a");
      nextLink.className = "atlas-page-nav__link atlas-page-nav__link--next";
      nextLink.href = next.href;
      nextLink.innerHTML = `<span>下一页</span><strong>${next.label}</strong>`;
      nav.appendChild(nextLink);
    }

    content.appendChild(nav);
  }

  function currentNavItem() {
    const current = normalizeUrl(window.location.href);
    const links = Array.from(document.querySelectorAll(".md-sidebar--primary a.md-nav__link[href]"));
    return links
      .map((anchor) => ({ anchor, info: navLinkInfo(anchor) }))
      .find((item) => item.info && item.info.key === current)?.anchor.closest(".md-nav__item") || null;
  }

  function navItemLinkInfo(item) {
    if (!item) return null;
    const anchor = directChild(item, "a.md-nav__link");
    if (!anchor) return null;
    return navLinkInfo(anchor);
  }

  function siblingTopicLinks(item) {
    const parentList = item && item.parentElement;
    if (!parentList) return [];

    return Array.from(parentList.children)
      .filter((sibling) => sibling !== item && sibling.matches(".md-nav__item"))
      .map(navItemLinkInfo)
      .filter(Boolean)
      .slice(0, 8);
  }

  function childTopicLinks(item) {
    const nav = directChild(item, "nav.md-nav");
    const list = directChild(nav, "ul.md-nav__list");
    if (!list) return [];

    return Array.from(list.children)
      .filter((child) => child.matches(".md-nav__item"))
      .map(navItemLinkInfo)
      .filter(Boolean)
      .slice(0, 8);
  }

  function mountTopicRelatedLinks() {
    const content = document.querySelector(".md-content__inner");
    if (!content || content.querySelector(".atlas-topic-links")) return;

    const item = currentNavItem();
    if (!item) return;

    const links = childTopicLinks(item).length ? childTopicLinks(item) : siblingTopicLinks(item);
    if (links.length < 2) return;

    const section = document.createElement("section");
    section.className = "atlas-topic-links";
    section.setAttribute("aria-label", "本专题相关页面");

    const title = document.createElement("h2");
    title.textContent = "本专题相关";

    const intro = document.createElement("p");
    intro.textContent = "读到这里时，不必退回左侧目录重新找路。下面这些页面和当前页面处在同一组主题里，适合作为下一步横向补读。";

    const list = document.createElement("div");
    list.className = "atlas-topic-links__grid";

    links.forEach((link) => {
      const anchor = document.createElement("a");
      anchor.href = link.href;
      anchor.className = "atlas-topic-links__item";
      anchor.textContent = link.label;
      list.appendChild(anchor);
    });

    section.append(title, intro, list);

    const pageNav = content.querySelector(".atlas-page-nav");
    if (pageNav) {
      content.insertBefore(section, pageNav);
    } else {
      content.appendChild(section);
    }
  }

  function init() {
    if (!document.body) return;
    markPageContext();
    markSidebarTopicNav();
    markCurrentPageRead();
    markSidebarReadingState();
    softenNumberedHeadings();
    mountReadingProgress();
    mountReaderControls();
    mountPaperDeepDiveSorting();
    mountGlossaryMasteryControls();
    mountTableCopyButtons();
    mountTopicRelatedLinks();
    mountPageNav();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }

  if (window.document$ && typeof window.document$.subscribe === "function") {
    window.document$.subscribe(init);
  }
})();
