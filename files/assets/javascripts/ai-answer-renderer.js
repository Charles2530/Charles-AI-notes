/*! atlas-ai-renderer-build: 20260825-1929 */
(function () {
  "use strict";

  let configured = false;

  function escapeMathHtml(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  const blockMath = {
    name: "blockMath",
    level: "block",
    start(source) {
      const starts = [source.indexOf("$$"), source.indexOf("\\[")].filter((index) => index >= 0);
      return starts.length ? Math.min(...starts) : undefined;
    },
    tokenizer(source) {
      const dollar = /^\$\$[ \t]*\n?([\s\S]+?)\n?\$\$(?:\n|$)/.exec(source);
      if (dollar) return { type: "blockMath", raw: dollar[0], text: dollar[1].trim() };
      const bracket = /^\\\[([\s\S]+?)\\\](?:\n|$)/.exec(source);
      if (bracket) return { type: "blockMath", raw: bracket[0], text: bracket[1].trim() };
      return undefined;
    },
    renderer(token) {
      return `<div class="arithmatex">\\[${escapeMathHtml(token.text)}\\]</div>\n`;
    }
  };

  const inlineMath = {
    name: "inlineMath",
    level: "inline",
    start(source) {
      const starts = [source.indexOf("$"), source.indexOf("\\(")].filter((index) => index >= 0);
      return starts.length ? Math.min(...starts) : undefined;
    },
    tokenizer(source) {
      const dollar = /^\$(?!\$)((?:\\.|[^$\\\n])+?)\$(?!\$)/.exec(source);
      if (dollar && dollar[1].trim() === dollar[1]) {
        return { type: "inlineMath", raw: dollar[0], text: dollar[1] };
      }
      const paren = /^\\\(((?:\\.|[^\\\n])+?)\\\)/.exec(source);
      if (paren) return { type: "inlineMath", raw: paren[0], text: paren[1] };
      return undefined;
    },
    renderer(token) {
      return `<span class="arithmatex">\\(${escapeMathHtml(token.text)}\\)</span>`;
    }
  };

  function configure() {
    if (configured) return;
    if (!window.marked || !window.DOMPurify) throw new Error("markdown runtime unavailable");
    window.marked.use({
      gfm: true,
      breaks: true,
      extensions: [blockMath, inlineMath]
    });
    configured = true;
  }

  function element(tag, className) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    return node;
  }

  function secureRichContent(fragment) {
    fragment.querySelectorAll("a").forEach((link) => {
      const href = link.getAttribute("href");
      if (!href) return;
      try {
        const target = new URL(href, window.location.href);
        if (!(["http:", "https:"].includes(target.protocol))) {
          link.removeAttribute("href");
          return;
        }
        if (target.origin !== window.location.origin) {
          link.target = "_blank";
          link.rel = "noopener noreferrer";
        }
      } catch (_) {
        link.removeAttribute("href");
      }
    });
    fragment.querySelectorAll('input:not([type="checkbox"])').forEach((input) => {
      input.remove();
    });
    fragment.querySelectorAll('input[type="checkbox"]').forEach((input) => {
      input.disabled = true;
      const item = input.closest("li");
      if (item) item.classList.add("task-list-item");
    });
    fragment.querySelectorAll("table").forEach((table) => {
      const wrapper = element("div", "atlas-ai-table-scroll");
      table.replaceWith(wrapper);
      wrapper.appendChild(table);
    });
    return fragment;
  }

  function render(target, source) {
    configure();
    const parsed = window.marked.parse(String(source || ""), { gfm: true, breaks: true });
    const fragment = window.DOMPurify.sanitize(parsed, {
      ALLOWED_TAGS: [
        "a", "blockquote", "br", "code", "del", "div", "em", "h1", "h2", "h3",
        "h4", "h5", "h6", "hr", "input", "li", "ol", "p", "pre", "span", "strong",
        "table", "tbody", "td", "th", "thead", "tr", "ul"
      ],
      ALLOWED_ATTR: ["checked", "class", "disabled", "href", "title", "type"],
      FORBID_TAGS: ["img", "svg", "iframe", "form"],
      FORBID_ATTR: ["style"],
      RETURN_DOM_FRAGMENT: true
    });
    target.replaceChildren(secureRichContent(fragment));
  }

  window.AtlasAIAnswerRenderer = { render };
})();
