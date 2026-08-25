(function () {
  const runtimeUrl = new URL("vendor/mathjax/tex-svg.js", document.currentScript.src).href;
  let runtimePromise = null;

  function configureMathJax() {
    if (window.MathJax && typeof window.MathJax.typesetPromise === "function") return;
    window.MathJax = {
      tex: {
        inlineMath: [["\\(", "\\)"]],
        displayMath: [["\\[", "\\]"]],
        processEscapes: true,
        processEnvironments: true
      },
      options: {
        skipHtmlTags: ["script", "noscript", "style", "textarea", "pre"]
      }
    };
  }

  function ensureMathJaxRuntime() {
    if (window.MathJax && typeof window.MathJax.typesetPromise === "function") {
      return Promise.resolve(window.MathJax);
    }
    if (runtimePromise) return runtimePromise;
    configureMathJax();
    runtimePromise = new Promise((resolve, reject) => {
      const existing = document.querySelector('script[data-atlas-runtime="mathjax"]');
      const script = existing || document.createElement("script");
      script.addEventListener("load", () => {
        const startup = window.MathJax && window.MathJax.startup && window.MathJax.startup.promise;
        Promise.resolve(startup).then(() => resolve(window.MathJax), reject);
      }, { once: true });
      script.addEventListener("error", () => reject(new Error("mathjax unavailable")), { once: true });
      if (!existing) {
        script.src = runtimeUrl;
        script.async = true;
        script.dataset.atlasRuntime = "mathjax";
        document.head.appendChild(script);
      }
    });
    return runtimePromise;
  }

  window.atlasTypesetMath = async function atlasTypesetMath(container) {
    if (!container || !container.querySelector(".arithmatex")) return;
    const mathJax = await ensureMathJaxRuntime();
    if (!mathJax || typeof mathJax.typesetPromise !== "function") {
      throw new Error("mathjax unavailable");
    }
    if (typeof mathJax.typesetClear === "function") mathJax.typesetClear([container]);
    await mathJax.typesetPromise([container]);
  };

  if (document.querySelector(".arithmatex")) {
    window.atlasTypesetMath(document).catch(() => {
      console.warn("Unable to load MathJax runtime");
    });
  }
})();
