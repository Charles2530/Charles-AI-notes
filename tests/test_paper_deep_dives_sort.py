from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase

from hooks.paper_deep_dives_sort import _paper_date_from_file, _sync_page_titles


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

    def test_foundations_pages_use_the_h1_prefix_for_sidebar_navigation(self):
        with TemporaryDirectory() as tmpdir:
            page = Path(tmpdir) / "foundations" / "index.md"
            page.parent.mkdir()
            page.write_text(
                "# 基础概念专题：把数学工具组装成模型机制\n",
                encoding="utf-8",
            )
            item = SimpleNamespace(
                file=SimpleNamespace(src_path="foundations/index.md"),
                title="基础概念专题",
                typeset={"title": "基础概念专题：把数学工具组装成模型机制"},
            )

            _sync_page_titles([item], Path(tmpdir))

            self.assertEqual(item.title, "基础概念专题：把数学工具组装成模型机制")
            self.assertEqual(item.typeset["title"], "基础概念专题")

    def test_live_rebuild_reads_changed_article_title(self):
        from hooks.paper_deep_dives_sort import on_nav
        with TemporaryDirectory() as tmpdir:
            page = Path(tmpdir) / 'video.md'
            page.write_text('# 原标题\n', encoding='utf-8')
            item = SimpleNamespace(file=SimpleNamespace(src_path='video.md'), title='配置标题')
            nav = SimpleNamespace(items=[item])
            on_nav(nav, {'docs_dir': tmpdir}, None)
            self.assertEqual(item.title, '原标题')
            page.write_text('# 新的正文标题\n', encoding='utf-8')
            on_nav(nav, {'docs_dir': tmpdir}, None)
            self.assertEqual(item.title, '新的正文标题')

    def test_heading_anchor_is_not_part_of_navigation_title(self):
        with TemporaryDirectory() as tmpdir:
            page = Path(tmpdir) / 'video.md'
            page.write_text('# 视频生成如何工作 { #video-generation }\n', encoding='utf-8')
            item = SimpleNamespace(file=SimpleNamespace(src_path='video.md'), title='配置标题')
            _sync_page_titles([item], Path(tmpdir))
            self.assertEqual(item.title, '视频生成如何工作')
