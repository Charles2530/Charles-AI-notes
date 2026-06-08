from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from hooks.paper_deep_dives_sort import _paper_date_from_file


class PaperDeepDivesSortTests(TestCase):
    def test_paper_date_falls_back_to_info_block_date(self):
        with TemporaryDirectory() as tmpdir:
            page = Path(tmpdir) / "legacy-paper.md"
            page.write_text(
                """# Legacy Paper

!!! info "论文信息"
    **时间 / 主题：** 2023-09；高效推理。<br>
""",
                encoding="utf-8",
            )

            self.assertEqual(_paper_date_from_file(str(page)), (2023, 9))
