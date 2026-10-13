from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]


class PaperLibraryTests(unittest.TestCase):
    def test_page_is_in_navigation_and_has_local_reader_script(self):
        config = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
        page = ROOT / "files" / "paper-library" / "index.md"
        self.assertTrue(page.is_file())
        self.assertIn("paper-library/index.md", config)
        script = (ROOT / "files" / "assets" / "javascripts" / "paper-library.js").read_text(
            encoding="utf-8"
        )
        self.assertIn("atlas-paper-library", script)
        self.assertIn("atlas-paper-library", page.read_text(encoding="utf-8"))
        self.assertIn("pdfjs", script)
        self.assertIn("IndexedDB", script)

    def test_repository_contract_names_and_export_fields_exist(self):
        script = (ROOT / "files" / "assets" / "javascripts" / "paper-library.js").read_text(
            encoding="utf-8"
        )
        for name in (
            "openPaperDatabase",
            "listPapers",
            "getPaper",
            "createPaper",
            "updatePaper",
            "deletePaper",
            "listAnnotations",
            "saveAnnotation",
            "deleteAnnotation",
            "getNote",
            "saveNote",
            "exportMarkdown",
            "exportPaperJson",
        ):
            self.assertRegex(script, rf"function {name}\b")
        for store in ("papers", "files", "annotations", "notes"):
            self.assertIn(f'"{store}"', script)
        self.assertIn("atlas-paper-library", script)
        self.assertIn("version(1)", script)
        self.assertIn("page", script)
        self.assertIn("quote", script)
        self.assertIn("rects", script)

    def test_exports_include_page_references_and_stable_download_names(self):
        script = (ROOT / "files" / "assets" / "javascripts" / "paper-library.js").read_text(
            encoding="utf-8"
        )
        self.assertIn("第${annotation.page}页", script)
        self.assertIn("application/json", script)
        self.assertIn(".md", script)
        self.assertIn(".json", script)

    def test_script_is_page_scoped_and_assets_are_local(self):
        config = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
        script = (ROOT / "files" / "assets" / "javascripts" / "paper-library.js").read_text(
            encoding="utf-8"
        )
        self.assertIn("paper-library", config)
        self.assertIn('document.querySelector("[data-atlas-paper-library]")', script)
        self.assertIn('if (!root) return', script)
        self.assertIn("vendor/pdfjs", script)
        self.assertNotIn("https://cdn", script)
        css = (ROOT / "files" / "assets" / "stylesheets" / "extra.css").read_text(
            encoding="utf-8"
        )
        self.assertIn(".atlas-paper-", css)

    def test_deletion_covers_all_related_stores(self):
        script = (ROOT / "files" / "assets" / "javascripts" / "paper-library.js").read_text(
            encoding="utf-8"
        )
        delete_body = script[script.index("async function deletePaper"):script.index("async function listAnnotations")]
        for store in ("files", "papers", "annotations", "notes"):
            self.assertIn(store, delete_body)

    def test_pdfjs_vendor_has_license_and_worker(self):
        vendor = ROOT / "files" / "assets" / "javascripts" / "vendor" / "pdfjs"
        self.assertTrue((vendor / "LICENSE").is_file())
        self.assertTrue((vendor / "pdf.min.mjs").is_file())
        self.assertTrue((vendor / "pdf.worker.min.mjs").is_file())


if __name__ == "__main__":
    unittest.main()

