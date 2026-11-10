from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import scripts.blog_readiness_report as report


class BlogReadinessReportTests(TestCase):
    def test_mermaid_diagram_counts_as_page_visual(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "files"
            page = root / "topic.md"
            page.parent.mkdir(parents=True)
            page.write_text(
                "# Topic\n\n为什么会失败？\n\n"
                "```mermaid\nflowchart LR\n A --> B\n```\n\n"
                "[来源](https://example.com)\n[下一页](next.md)\n",
                encoding="utf-8",
            )
            with patch.object(report, "DOCS_ROOT", root):
                audit = report.audit_page(page, set())

            self.assertEqual(audit.images, 0)
            self.assertEqual(audit.diagrams, 1)
            self.assertNotIn("no-image", audit.flags)
