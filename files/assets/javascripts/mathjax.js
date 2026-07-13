(function () {
  if (!document.querySelector(".arithmatex")) return;

  const runtimeUrl = new URL("vendor/mathjax/tex-svg.js", document.currentScript.src).href;
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

  const script = document.createElement("script");
  script.src = runtimeUrl;
  script.async = true;
  script.dataset.atlasRuntime = "mathjax";
  script.addEventListener("error", () => {
    console.warn("Unable to load MathJax runtime");
  }, { once: true });
  document.head.appendChild(script);
})();
