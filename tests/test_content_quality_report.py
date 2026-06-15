from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import scripts.content_quality_report as report


class ContentQualityReportTests(TestCase):
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
