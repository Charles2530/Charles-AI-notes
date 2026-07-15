import re
from pathlib import Path
from unittest import TestCase


ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = ROOT / "files"


class TrainingInferenceCurriculumTests(TestCase):
    CORE_HEADINGS = (
        "## 本文解决什么问题",
        "## 前置知识",
        "## 核心机制",
        "## 常见误解与失效边界",
        "## 检查理解",
        "## 相关阅读与下一步",
    )

    def assert_ordered_subsequence(self, required, actual):
        remaining = iter(actual)
        for item in required:
            self.assertIn(item, remaining)

    def assert_local_links_resolve(self, page, minimum=2):
        text = page.read_text(encoding="utf-8")
        targets = re.findall(r"!?\[[^]]*\]\(([^)]+)\)", text)
        local_targets = set()
        for target in targets:
            target = target.split("#", 1)[0].split("?", 1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            path = (page.parent / target).resolve()
            local_targets.add(path)
            self.assertTrue(path.exists(), f"{page}: {target}")
        self.assertGreaterEqual(len(local_targets), minimum, str(page))

    def test_unified_index_connects_training_artifacts_to_online_requests(self):
        page = DOCS_ROOT / "training-and-inference-systems" / "index.md"
        self.assertTrue(page.is_file())
        text = page.read_text(encoding="utf-8")
        headings = [line for line in text.splitlines() if line.startswith("## ")]
        self.assertEqual(
            headings,
            [
                "## 训练与推理系统的统一接口",
                "## 一条从数据到请求再回到数据的闭环",
                "## 训练系统脉络",
                "## 推理系统脉络",
                "## 量化、算子与编译器怎样贯穿两端",
                "## 按问题进入",
            ],
        )
        for term in (
            "%% training-serving-feedback-loop",
            "training step",
            "model artifact",
            "prefill",
            "decode",
            "SLO",
            "offline-to-online gap",
        ):
            self.assertIn(term, text)
        self.assert_local_links_resolve(page, minimum=8)

    def test_topic_indexes_define_complete_system_reading_paths(self):
        expected = {
            "training/index.md": (
                "## 训练系统的统一接口",
                "## 一次训练怎样运行",
                "## 数据与训练目标",
                "## 规模化执行与稳定性",
                "## 评测、治理与后训练",
                "## 按问题进入",
            ),
            "inference/index.md": (
                "## 推理系统的统一接口",
                "## 一条请求的生命周期",
                "## 执行引擎与显存",
                "## 调度、缓存与扩展",
                "## SLO、评测与反馈",
                "## 按问题进入",
            ),
        }
        for relative, headings in expected.items():
            page = DOCS_ROOT / relative
            actual = [
                line
                for line in page.read_text(encoding="utf-8").splitlines()
                if line.startswith("## ")
            ]
            with self.subTest(page=relative):
                self.assertEqual(actual, list(headings))
                self.assert_local_links_resolve(page, minimum=8)

    def test_training_core_page_follows_shared_contract_without_reader_labels(self):
        page = DOCS_ROOT / "training" / "beginner-model-training-primer.md"
        text = page.read_text(encoding="utf-8")
        headings = [line for line in text.splitlines() if line.startswith("## ")]
        self.assert_ordered_subsequence(self.CORE_HEADINGS, headings)
        for phrase in ("初学者", "新手", "零基础", "小白", "从零"):
            self.assertNotIn(phrase, text)
        for term in (
            "effective tokens",
            "global batch",
            "optimizer step",
            "checkpoint contract",
            "data parallel",
            "tensor parallel",
            "pipeline parallel",
        ):
            self.assertIn(term, text)
        self.assert_local_links_resolve(page, minimum=6)

    def test_navigation_exposes_every_system_page_in_grouped_order(self):
        nav = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
        start = nav.index("  - 训练与推理系统:")
        end = nav.index("  - 强化学习:", start)
        section = nav[start:end]
        paths = re.findall(
            r"((?:training-and-inference-systems|training|inference|quantization|operators)/[^\s]+\.md)",
            section,
        )
        self.assertEqual(
            paths[:3],
            [
                "training-and-inference-systems/index.md",
                "training/index.md",
                "training/beginner-model-training-primer.md",
            ],
        )
        expected = {"training-and-inference-systems/index.md"}
        for directory in ("training", "inference", "quantization", "operators"):
            expected.update(
                f"{directory}/{page.name}"
                for page in (DOCS_ROOT / directory).glob("*.md")
            )
        self.assertEqual(set(paths), expected)
