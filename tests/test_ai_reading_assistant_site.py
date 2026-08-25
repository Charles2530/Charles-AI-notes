from pathlib import Path
import re
import subprocess
from unittest import TestCase


ROOT = Path(__file__).resolve().parents[1]


class AIReadingAssistantSiteTests(TestCase):
    def test_private_ai_configuration_is_ignored_and_example_has_no_key(self):
        gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
        example = (ROOT / ".env.local.example").read_text(encoding="utf-8")

        self.assertIn(".env.local", gitignore.splitlines())
        self.assertIn("OPENAI_API_KEY=", example)
        self.assertIn("ATLAS_AI_BASE_URL=https://47.95.249.76", example)
        self.assertIn("ATLAS_AI_MODEL=gpt-5.6-sol", example)
        self.assertIn("ATLAS_AI_REASONING_EFFORT=high", example)
        self.assertNotIn("ATLAS_AI_ADDR", example)
        self.assertNotRegex(example, r"OPENAI_API_KEY=sk-[A-Za-z0-9]")

    def test_make_serve_starts_the_local_ai_coordinator(self):
        makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

        self.assertIn("python3 scripts/serve_local.py", makefile)
        self.assertIn('--mkdocs "$(MKDOCS)"', makefile)
        self.assertIn('--dev-addr "$(DEV_ADDR)"', makefile)
        self.assertIn("--dirty", makefile)

    def test_site_loads_a_safe_page_scoped_ai_reader(self):
        config = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
        script = (
            ROOT / "files" / "assets" / "javascripts" / "ai-reading-assistant.js"
        ).read_text(encoding="utf-8")

        self.assertIn("assets/javascripts/ai-reading-assistant.min.js", config)
        self.assertIn('document.querySelector(".md-content__inner")', script)
        self.assertIn("window.getSelection()", script)
        self.assertIn("range.intersectsNode", script)
        self.assertIn("const MAX_HISTORY = 12", script)
        self.assertIn("renderSafeMarkdown", script)
        self.assertIn("textContent", script)
        self.assertIn("AI 服务暂时不可用，请稍后重试。", script)
        self.assertIn("window.document$.subscribe", script)
        self.assertIn('window.location.protocol !== "http:"', script)
        self.assertIn('LOCAL_HOSTS.has(window.location.hostname)', script)
        self.assertIn("if (!complete) throw new Error", script)
        self.assertIn("const MAX_HISTORY_ITEM_CHARS = 8_000", script)
        self.assertIn("const MAX_HISTORY_CHARS = 48_000", script)
        self.assertIn("const MAX_REQUEST_BYTES = 128 * 1024", script)
        self.assertIn("new TextEncoder().encode(JSON.stringify(payload)).length", script)
        self.assertNotIn("message.innerHTML", script)

    def test_ai_reader_falls_back_to_the_current_article_without_selection(self):
        script = (
            ROOT / "files" / "assets" / "javascripts" / "ai-reading-assistant.js"
        ).read_text(encoding="utf-8")

        self.assertIn("function currentArticleReference()", script)
        self.assertIn('document.querySelector(".md-content__inner")', script)
        self.assertIn("content.innerText || content.textContent", script)
        self.assertIn("function referenceForRequest()", script)
        self.assertIn("return state.reference || currentArticleReference()", script)
        self.assertIn("const reference = referenceForRequest()", script)
        self.assertIn("if (!question) return", script)
        self.assertIn("if (!reference) {", script)
        self.assertIn("category=missing_article", script)
        self.assertNotIn("if (!question || !state.reference) return", script)
        self.assertIn('"询问整篇文章…"', script)
        self.assertIn('"可以直接询问整篇文章"', script)
        self.assertIn("selectionCandidate", script)
        self.assertIn("state.reference = state.selectionCandidate", script)

    def test_ai_reader_uses_code_point_safe_text_limits(self):
        script = (
            ROOT / "files" / "assets" / "javascripts" / "ai-reading-assistant.js"
        ).read_text(encoding="utf-8")

        self.assertIn("function sliceCodePoints", script)
        self.assertRegex(script, r"sliceCodePoints\(\s*item\.content")
        self.assertIn("sliceCodePoints(selectionText", script)
        self.assertIn("sliceCodePoints(context || selectionText", script)

    def test_ai_reader_trims_whole_article_context_to_the_utf8_request_budget(self):
        script = (
            ROOT / "files" / "assets" / "javascripts" / "ai-reading-assistant.js"
        ).read_text(encoding="utf-8")

        self.assertIn("Array.from(payload.context)", script)
        self.assertIn("payload.context =", script)
        self.assertIn("encodedPayloadSize(payload)", script)
        self.assertIn("while (payload.history.length", script)

    def test_loaded_minified_script_matches_source_build_marker(self):
        config = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
        source = (
            ROOT / "files" / "assets" / "javascripts" / "ai-reading-assistant.js"
        ).read_text(encoding="utf-8")
        production = (
            ROOT / "files" / "assets" / "javascripts" / "ai-reading-assistant.min.js"
        ).read_text(encoding="utf-8")
        marker = re.compile(r"/\*! atlas-ai-build: ([0-9-]+) \*/")

        self.assertIsNotNone(source_marker := marker.search(source))
        self.assertIsNotNone(production_marker := marker.search(production))
        self.assertEqual(source_marker.group(1), production_marker.group(1))
        self.assertIn(
            f"assets/javascripts/ai-reading-assistant.min.js?v={source_marker.group(1)}",
            config,
        )

        renderer_source = (
            ROOT / "files" / "assets" / "javascripts" / "ai-answer-renderer.js"
        ).read_text(encoding="utf-8")
        renderer_production = (
            ROOT / "files" / "assets" / "javascripts" / "ai-answer-renderer.min.js"
        ).read_text(encoding="utf-8")
        renderer_marker = re.compile(
            r"/\*! atlas-ai-renderer-build: ([0-9-]+) \*/"
        )
        self.assertIsNotNone(source_match := renderer_marker.search(renderer_source))
        self.assertIsNotNone(
            production_match := renderer_marker.search(renderer_production)
        )
        self.assertEqual(source_match.group(1), production_match.group(1))

    def test_public_pages_do_not_mount_or_probe_the_local_ai_service(self):
        for filename in (
            "ai-reading-assistant.js",
            "ai-reading-assistant.min.js",
        ):
            script = (
                ROOT / "files" / "assets" / "javascripts" / filename
            ).read_text(encoding="utf-8")
            result = subprocess.run(
                [
                    "node",
                    "-e",
                    (
                        "const vm=require('node:vm');"
                        "const fs=require('node:fs');"
                        "vm.runInNewContext(fs.readFileSync(0,'utf8'),"
                        "{window:{location:{protocol:'https:',hostname:'example.com'}}});"
                    ),
                ],
                input=script,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_ai_reader_has_desktop_mobile_print_and_reduced_motion_styles(self):
        stylesheet = (
            ROOT / "files" / "assets" / "stylesheets" / "extra.css"
        ).read_text(encoding="utf-8")

        self.assertIn(".atlas-ai-selection-action", stylesheet)
        self.assertIn(".atlas-ai-drawer", stylesheet)
        self.assertIn(".atlas-ai-drawer[hidden]", stylesheet)
        self.assertIn("display: none !important", stylesheet)
        self.assertIn(".atlas-ai-message--assistant", stylesheet)
        self.assertIn("@media screen and (max-width: 44.9375em)", stylesheet)
        self.assertIn("prefers-reduced-motion: reduce", stylesheet)
        print_styles = stylesheet[stylesheet.index("@media print"):]
        self.assertIn(".atlas-ai-root", print_styles)

    def test_gfm_and_sanitizer_runtimes_are_local_and_traceable(self):
        marked_dir = (
            ROOT / "files" / "assets" / "javascripts" / "vendor" / "marked"
        )
        purifier_dir = (
            ROOT / "files" / "assets" / "javascripts" / "vendor" / "dompurify"
        )

        self.assertTrue((marked_dir / "marked.umd.js").is_file())
        self.assertTrue((marked_dir / "LICENSE").is_file())
        self.assertIn("marked 18.0.11", (marked_dir / "README.md").read_text())
        self.assertTrue((purifier_dir / "purify.min.js").is_file())
        self.assertTrue((purifier_dir / "LICENSE").is_file())
        self.assertTrue((purifier_dir / "LICENSE-MPL").is_file())
        self.assertIn("DOMPurify 3.4.14", (purifier_dir / "README.md").read_text())

        node_check = subprocess.run(
            [
                "node",
                "-e",
                (
                    "const marked=require(process.argv[1]);"
                    "const html=marked.parse('| A | B |\\n|---|---|\\n| 1 | 2 |');"
                    "if(!html.includes('<table>')||!html.includes('<td>1</td>'))process.exit(1);"
                ),
                str(marked_dir / "marked.umd.js"),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(node_check.returncode, 0, node_check.stderr)

    def test_ai_reader_uses_lazy_safe_gfm_and_math_rendering(self):
        loader = (
            ROOT / "files" / "assets" / "javascripts" / "ai-reading-assistant.js"
        ).read_text(encoding="utf-8")
        renderer = (
            ROOT / "files" / "assets" / "javascripts" / "ai-answer-renderer.js"
        ).read_text(encoding="utf-8")

        self.assertIn("vendor/marked/marked.umd.js", loader)
        self.assertIn("vendor/dompurify/purify.min.js", loader)
        self.assertIn("ai-answer-renderer.min.js", loader)
        self.assertIn("ensureMarkdownRuntime", loader)
        self.assertIn("window.AtlasAIAnswerRenderer.render", loader)
        self.assertIn("window.atlasTypesetMath(body)", loader)
        self.assertIn("renderSafeMarkdown", loader)
        self.assertIn("window.marked.parse", renderer)
        self.assertIn("window.DOMPurify.sanitize", renderer)
        self.assertIn("RETURN_DOM_FRAGMENT", renderer)
        self.assertIn(
            'FORBID_TAGS: ["img", "svg", "iframe", "form"]', renderer
        )
        self.assertIn("blockMath", renderer)
        self.assertIn("inlineMath", renderer)
        self.assertIn("atlas-ai-table-scroll", renderer)
        self.assertNotIn("https://cdn", loader + renderer)
        self.assertNotIn("unpkg.com", loader + renderer)

    def test_mathjax_loader_supports_late_ai_answers(self):
        script = (
            ROOT / "files" / "assets" / "javascripts" / "mathjax.js"
        ).read_text(encoding="utf-8")

        self.assertIn("window.atlasTypesetMath", script)
        self.assertIn("typesetPromise([container])", script)
        self.assertIn('script.dataset.atlasRuntime = "mathjax"', script)
        self.assertNotIn('if (!document.querySelector(".arithmatex")) return;', script)

    def test_rich_answers_keep_wide_content_inside_the_drawer(self):
        stylesheet = (
            ROOT / "files" / "assets" / "stylesheets" / "extra.css"
        ).read_text(encoding="utf-8")

        self.assertIn(".atlas-ai-table-scroll", stylesheet)
        self.assertIn(".atlas-ai-answer table", stylesheet)
        self.assertIn(".atlas-ai-answer .arithmatex", stylesheet)
        self.assertIn(".atlas-ai-answer .task-list-item", stylesheet)
        self.assertIn("overflow-x: auto", stylesheet)
        inline_math = stylesheet[
            stylesheet.index(".atlas-ai-answer span.arithmatex"):
            stylesheet.index(".atlas-ai-answer .arithmatex mjx-container")
        ]
        self.assertIn("overflow-x: auto", inline_math)

    def test_renderer_removes_non_task_inputs_and_versions_lazy_assets(self):
        loader = (
            ROOT / "files" / "assets" / "javascripts" / "ai-reading-assistant.js"
        ).read_text(encoding="utf-8")
        renderer = (
            ROOT / "files" / "assets" / "javascripts" / "ai-answer-renderer.js"
        ).read_text(encoding="utf-8")

        self.assertIn('input:not([type="checkbox"])', renderer)
        self.assertIn("RENDERER_ASSET_VERSION", loader)
        self.assertIn("?v=${RENDERER_ASSET_VERSION}", loader)

    def test_production_renderer_has_a_browser_behavior_harness(self):
        harness = ROOT / "tests" / "browser" / "ai_answer_renderer_test.html"

        self.assertTrue(harness.is_file())
        source = harness.read_text(encoding="utf-8")
        self.assertIn("ai-answer-renderer.min.js", source)
        self.assertIn("marked.umd.js", source)
        self.assertIn("purify.min.js", source)
        for case in (
            "table wrapper",
            "task checkbox",
            "four math delimiters",
            "external link safety",
            "dangerous markup removed",
            "inline math overflow",
        ):
            self.assertIn(case, source)

    def test_article_context_has_a_production_browser_behavior_harness(self):
        harness = ROOT / "tests" / "browser" / "ai_article_context_test.html"

        self.assertTrue(harness.is_file())
        source = harness.read_text(encoding="utf-8")
        self.assertIn("ai-reading-assistant.min.js", source)
        for case in (
            "whole article fallback",
            "utf8 request budget",
            "unconfirmed selection stays article",
            "selection takes priority",
            "unicode safe selection and history",
            "remove selection returns to article",
        ):
            self.assertIn(case, source)
