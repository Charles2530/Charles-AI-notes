import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from urllib.parse import urlsplit

from mkdocs.config import load_config


ROOT = Path(__file__).resolve().parents[1]


class SitePerformanceTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(str(ROOT / "mkdocs.yml"))

    def test_theme_does_not_block_on_remote_fonts(self):
        self.assertIs(self.config["theme"].get("font"), False)

    def test_navigation_prunes_inactive_branches(self):
        self.assertIn("navigation.prune", self.config["theme"]["features"])

    def test_sidebar_typeset_titles_inherit_navigation_typography(self):
        stylesheet = (ROOT / "files" / "assets" / "stylesheets" / "extra.css").read_text(
            encoding="utf-8"
        )

        self.assertIn(".md-sidebar--primary .md-nav__link .md-typeset", stylesheet)
        self.assertIn("font-size: inherit", stylesheet)
        self.assertIn("line-height: inherit", stylesheet)
        self.assertIn("font-weight: inherit", stylesheet)

    def test_search_compaction_hook_omits_full_article_bodies(self):
        try:
            from hooks.site_performance import on_post_build
        except ModuleNotFoundError:
            self.fail("hooks.site_performance is not implemented")

        with TemporaryDirectory() as tmpdir:
            search_dir = Path(tmpdir) / "search"
            search_dir.mkdir()
            index_path = search_dir / "search_index.json"
            index_path.write_text(
                json.dumps(
                    {
                        "config": {"fields": {"title": {}, "text": {}, "tags": {}}},
                        "docs": [
                            {"location": "guide/#part", "title": "Part", "text": "large article body"}
                        ],
                    }
                ),
                encoding="utf-8",
            )

            on_post_build({"site_dir": tmpdir})
            compacted = json.loads(index_path.read_text(encoding="utf-8"))

        self.assertNotIn("text", compacted["config"]["fields"])
        self.assertEqual(compacted["docs"], [{"location": "guide/#part", "title": "Part"}])

    def test_search_compaction_preserves_an_on_demand_full_text_index(self):
        from hooks.site_performance import on_post_build

        with TemporaryDirectory() as tmpdir:
            search_dir = Path(tmpdir) / "search"
            search_dir.mkdir()
            index_path = search_dir / "search_index.json"
            payload = {
                "config": {"fields": {"title": {}, "text": {}}},
                "docs": [{"location": "guide/", "title": "Guide", "text": "body-only phrase"}],
            }
            index_path.write_text(json.dumps(payload), encoding="utf-8")

            on_post_build({"site_dir": tmpdir})
            full_index = json.loads(
                (search_dir / "search_index_full.json").read_text(encoding="utf-8")
            )

        self.assertIn("text", full_index["config"]["fields"])
        self.assertEqual(full_index["docs"][0]["text"], "body-only phrase")

    def test_search_fallback_uses_the_on_demand_full_text_index(self):
        fallback = (ROOT / "files" / "assets" / "javascripts" / "search-fallback.js").read_text(
            encoding="utf-8"
        )

        self.assertIn('"search/search_index_full.json"', fallback)

    def test_global_custom_javascript_is_local_and_under_128_kib(self):
        scripts = [str(script) for script in self.config["extra_javascript"]]
        remote_scripts = [script for script in scripts if "://" in script]
        local_bytes = sum(
            (ROOT / "files" / urlsplit(script).path.lstrip("/")).stat().st_size
            for script in scripts
            if "://" not in script
        )

        self.assertEqual(remote_scripts, [])
        self.assertLessEqual(local_bytes, 128 * 1024)

    def test_runtime_loaders_have_cache_versions(self):
        scripts = [str(script) for script in self.config["extra_javascript"]]
        runtime_loaders = [
            script
            for script in scripts
            if urlsplit(script).path.endswith(
                ("/search-fallback.js", "/diagram-renderer.js", "/mathjax.js")
            )
        ]

        self.assertEqual(len(runtime_loaders), 3)
        for script in runtime_loaders:
            self.assertTrue(urlsplit(script).query, script)

    def test_vendored_mathjax_has_license_and_provenance(self):
        vendor_dir = ROOT / "files" / "assets" / "javascripts" / "vendor" / "mathjax"
        license_text = (vendor_dir / "LICENSE").read_text(encoding="utf-8")
        provenance = (vendor_dir / "README.md").read_text(encoding="utf-8")

        self.assertIn("Apache License", license_text)
        self.assertIn("Version 2.0", license_text)
        self.assertIn("MathJax 3.2.2", provenance)
        self.assertIn("d4295dc33744836935c1399feece5159577b34c5c8ffb9f1c6324cd82e03a882", provenance)
