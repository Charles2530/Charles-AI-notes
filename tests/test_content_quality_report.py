from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch
from contextlib import redirect_stdout
from io import StringIO

import scripts.content_quality_report as report


class ContentQualityReportTests(TestCase):
    def test_vendor_docs_are_excluded_but_teaching_examples_are_scanned(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "files"
            vendor = root / "assets/javascripts/vendor/library/README.md"
            example = root / "assets/examples/foundations/README.md"
            other_vendor = root / "assets/examples/vendor/README.md"
            reference_guide = root / "roadmap/comparison-tables.md"
            for page in (vendor, example, other_vendor, reference_guide):
                page.parent.mkdir(parents=True)
                page.write_text("# Readme\n", encoding="utf-8")
            with patch.object(report, "DOCS_ROOT", root):
                self.assertEqual(report.iter_markdown_pages(), [example, other_vendor, reference_guide])

    def test_length_and_heading_counts_are_optional_review_hints(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            page = root / "detailed.md"
            page.write_text("# Detailed\n" + "\n## A question\nA detailed explanation." * 2000, encoding="utf-8")
            with patch.object(report, "DOCS_ROOT", root):
                self.assertEqual(report.analyze(page, include_hints=False).flags, [])
                hints = report.analyze(page, include_hints=True).flags
                self.assertTrue(hints)
                self.assertEqual(report.hard_flags(hints), [])

    def test_broken_example_link_is_structural_failure_by_default(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            page = root / "example.md"
            page.write_text("# Example\n[Run it](missing.py)\n", encoding="utf-8")
            with patch.object(report, "DOCS_ROOT", root), patch("sys.argv", ["content_quality_report"]), redirect_stdout(StringIO()):
                self.assertEqual(report.main(), 1)

    def test_tilde_code_fences_do_not_create_broken_link_findings(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            page = root / "index.md"
            page.write_text("# Example\n\n~~~markdown\n[Example](not-a-file.md)\n~~~\n", encoding="utf-8")
            with patch.object(report, "DOCS_ROOT", root):
                self.assertEqual(report.analyze(page).flags, [])

    def test_longer_code_fence_is_not_closed_by_shorter_example_fence(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            page = root / "index.md"
            page.write_text("# Example\n\n````markdown\n```python\n```\n[Example](not-a-file.md)\n````\n", encoding="utf-8")
            with patch.object(report, "DOCS_ROOT", root):
                self.assertEqual(report.analyze(page).flags, [])

    def test_existing_example_and_shared_image_links_are_accepted(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "example.py").write_text("print('example')\n", encoding="utf-8")
            (root / "figure.svg").write_text("<svg></svg>\n", encoding="utf-8")
            page = root / "topic.md"
            page.write_text('# Topic\n[Example](example.py)\n![Figure](figure.svg)\n<img src="../figure.svg">\n', encoding="utf-8")
            with patch.object(report, "DOCS_ROOT", root):
                self.assertEqual(report.analyze(page).flags, [])

    def test_optional_hints_do_not_fail_legacy_fail_on_warn_mode(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "short.md").write_text("# Brief topic\n", encoding="utf-8")
            with patch.object(report, "DOCS_ROOT", root), patch("sys.argv", ["content_quality_report", "--include-hints", "--fail-on-warn"]), redirect_stdout(StringIO()):
                self.assertEqual(report.main(), 0)

    def test_robot_learning_survey_is_not_flagged_long(self):
        old_root = report.DOCS_ROOT
        try:
            with TemporaryDirectory() as tmpdir:
                root = Path(tmpdir) / "files"
                page = root / "paper-deep-dives" / "world-models" / "world-model-for-robot-learning-survey.md"
                page.parent.mkdir(parents=True)
                page.write_text("# Survey\n\n" + ("这是一段标杆长文内容。\n" * 6000), encoding="utf-8")

                report.DOCS_ROOT = root
                result = report.analyze(page, include_hints=False)

            self.assertNotIn("long", result.flags)
        finally:
            report.DOCS_ROOT = old_root
