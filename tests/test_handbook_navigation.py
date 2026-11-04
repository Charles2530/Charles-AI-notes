from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]


def page_paths(items):
    for item in items:
        for value in item.values():
            if isinstance(value, list):
                yield from page_paths(value)
            else:
                yield value


class HandbookNavigationTests(unittest.TestCase):
    def setUp(self):
        self.nav = yaml.load((ROOT / "mkdocs.yml").read_text(), Loader=yaml.BaseLoader)["nav"]

    def test_root_home_is_the_default_entry(self):
        self.assertEqual(next(page_paths(self.nav)), "index.md")
        self.assertTrue((ROOT / "files/index.md").is_file())

    def test_papers_open_existing_analyses_without_pdf_import(self):
        papers = next(section["论文精读"] for section in self.nav if "论文精读" in section)
        self.assertEqual(next(page_paths(papers)), "paper-deep-dives/index.md")
        self.assertNotIn("paper-library/index.md", list(page_paths(papers)))
        self.assertNotIn("paper-library/index.md", list(page_paths(self.nav)))

    def test_all_navigation_targets_exist_and_have_one_primary_home(self):
        paths = list(page_paths(self.nav))
        self.assertEqual(len(paths), len(set(paths)))
        for path in paths:
            with self.subTest(path=path):
                self.assertTrue((ROOT / "files" / path).is_file())

    def test_topic_tabs_start_at_their_overviews(self):
        expected = {
            "基础知识": "foundations/index.md",
            "模型结构": "model-architectures/index.md",
            "训练": "training/index.md",
            "推理与效率": "inference/index.md",
            "生成模型": "diffusion/index.md",
            "强化学习": "reinforcement-learning/index.md",
            "多模态与具身": "world-models-and-embodied-ai/index.md",
            "工具与查阅": "technical-reference/index.md",
        }
        sections = {key: value for item in self.nav for key, value in item.items()}
        for name, overview in expected.items():
            with self.subTest(topic=name):
                self.assertEqual(next(page_paths(sections[name])), overview)

    def test_all_nonpaper_reference_pages_remain_discoverable(self):
        paths = set(page_paths(self.nav))
        for page in (ROOT / "files").rglob("*.md"):
            relative = page.relative_to(ROOT / "files")
            if relative.parts[0] in {"roadmap", "assets", "paper-deep-dives"}:
                continue
            with self.subTest(page=str(relative)):
                self.assertIn(relative.as_posix(), paths)
