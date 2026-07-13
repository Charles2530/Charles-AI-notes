(function () {
  const runtimeUrl = new URL("vendor/mermaid.min.js", document.currentScript.src).href;
  let runtimePromise = null;

  const config = {
    startOnLoad: false,
    securityLevel: "loose",
    theme: "base",
    flowchart: {
      curve: "basis",
      htmlLabels: true,
      padding: 12
    },
    sequence: {
      actorMargin: 70,
      mirrorActors: false
    },
    themeVariables: {
      background: "transparent",
      primaryColor: "#fffdf8",
      primaryTextColor: "#14292b",
      primaryBorderColor: "#0f5558",
      lineColor: "#0f5558",
      secondaryColor: "#fff8ef",
      tertiaryColor: "#f6f1e6",
      edgeLabelBackground: "#fffdf8",
      fontFamily: '"Source Han Sans SC", "Noto Sans SC", "PingFang SC", sans-serif',
      fontSize: "15px",
      nodeBorder: "#0f5558",
      mainBkg: "#fffdf8",
      nodeTextColor: "#14292b",
      textColor: "#14292b",
      titleColor: "#14292b",
      noteBkgColor: "#fff8ef",
      noteTextColor: "#14292b",
      noteBorderColor: "#cb5d31",
      clusterBkg: "rgba(238, 246, 244, 0.72)",
      clusterBorder: "rgba(18, 95, 99, 0.18)"
    }
  };

  function loadRuntime() {
    if (window.mermaid) return Promise.resolve(window.mermaid);
    if (runtimePromise) return runtimePromise;

    runtimePromise = new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = runtimeUrl;
      script.async = true;
      script.dataset.atlasRuntime = "mermaid";
      script.addEventListener("load", () => resolve(window.mermaid), { once: true });
      script.addEventListener("error", () => reject(new Error("Unable to load Mermaid runtime")), { once: true });
      document.head.appendChild(script);
    });
    return runtimePromise;
  }

  function markPending() {
    document.querySelectorAll(".mermaid").forEach((node) => {
      if (!node.dataset.atlasMermaidSource) {
        node.dataset.atlasMermaidSource = node.textContent.trim();
      }
    });
  }

  async function renderMermaid() {
    if (!document.querySelector(".mermaid")) return;

    markPending();

    document.querySelectorAll(".mermaid[data-processed]").forEach((node) => {
      node.removeAttribute("data-processed");
      node.textContent = node.dataset.atlasMermaidSource || node.textContent;
    });

    try {
      const mermaid = await loadRuntime();
      mermaid.initialize(config);
      await mermaid.run({ querySelector: ".mermaid" });
    } catch (error) {
      document.querySelectorAll(".mermaid").forEach((node) => {
        if (!node.querySelector("svg")) {
          node.classList.add("atlas-mermaid-error");
          node.textContent = node.dataset.atlasMermaidSource || node.textContent;
        }
      });
      console.warn("Mermaid rendering failed", error);
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", renderMermaid);
  } else {
    renderMermaid();
  }

  if (window.document$ && typeof window.document$.subscribe === "function") {
    window.document$.subscribe(renderMermaid);
  }
})();
