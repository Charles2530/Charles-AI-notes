from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from scripts.paper_deep_dive_audit import audit_page, audit_root


def _valid_page(title: str, body: str = "消融 Figure 1 实验 setting。") -> str:
    return f"""---
paper_date: 2026-01
paper_title: "{title}"
paper_topic: 世界模型
---

# {title}

!!! info "论文信息"
    **论文题名：** {title}。<br>
    **时间 / 主题：** 2026-01；世界模型。<br>
    **元数据来源与核验口径：** Checked Date：2026-06-15；Repro Status：Paper reviewed。

## 原论文故事线

### 研究背景

{title} 的研究背景。

### 问题（Challenge）

{title} 的挑战。

### Finding

{title} 的关键发现。

### 方法

{title} 的方法。

### 结论

{title} 的实验结论。

### 关键术语

- **Term（术语）**：解释。

## 方法

{body}

## 阅读结论

{title} 的主要内容已经闭合。
"""


class PaperDeepDiveAuditTests(TestCase):
    def test_robot_learning_survey_is_allowed_as_longform_marker_page(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            page = root / "world-models" / "world-model-for-robot-learning-survey.md"
            page.parent.mkdir(parents=True)
            page.write_text(_valid_page("World Model for Robot Learning") + ("\n实验 消融 Figure 1。" * 5000), encoding="utf-8")

            findings = audit_page(page, include_hints=True, root=root)

            self.assertFalse(any("exceeds" in finding.reason for finding in findings))

    def test_root_audit_reports_pages_missing_from_index(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paper_dir = root / "world-models"
            paper_dir.mkdir(parents=True)
            (paper_dir / "covered.md").write_text(_valid_page("Covered"), encoding="utf-8")
            (paper_dir / "missing.md").write_text(_valid_page("Missing"), encoding="utf-8")
            (root / "index.md").write_text("- [Covered](world-models/covered.md)\n", encoding="utf-8")

            findings = audit_root(root, include_hints=False)

            self.assertTrue(any("index.md missing links" in finding.reason and "world-models/missing.md" in finding.reason for finding in findings))

    def test_short_page_is_not_flagged_by_character_count(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            page = root / "world-models" / "short-but-complete.md"
            page.parent.mkdir(parents=True)
            page.write_text(_valid_page("Short But Complete"), encoding="utf-8")

            findings = audit_page(page, include_hints=True, root=root)

            self.assertFalse(any("short deep dive" in finding.reason for finding in findings))

    def test_missing_reading_conclusion_is_reported(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            page = root / "world-models" / "missing-conclusion.md"
            page.parent.mkdir(parents=True)
            page.write_text(_valid_page("Missing Conclusion").replace("## 阅读结论\n\nMissing Conclusion 的主要内容已经闭合。\n", ""), encoding="utf-8")

            findings = audit_page(page, include_hints=False, root=root)

            self.assertTrue(any("missing 阅读结论 section" in finding.reason for finding in findings))

    def test_missing_frontmatter_field_is_reported(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            page = root / "world-models" / "missing-topic.md"
            page.parent.mkdir(parents=True)
            page.write_text(_valid_page("Missing Topic").replace("paper_topic: 世界模型\n", ""), encoding="utf-8")

            findings = audit_page(page, include_hints=False, root=root)

            self.assertTrue(any("missing frontmatter fields: paper_topic" in finding.reason for finding in findings))

    def test_missing_story_section_is_reported(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            page = root / "world-models" / "missing-story-section.md"
            page.parent.mkdir(parents=True)
            page.write_text(_valid_page("Missing Story Section").replace("### Finding\n\nMissing Story Section 的关键发现。\n\n", ""), encoding="utf-8")

            findings = audit_page(page, include_hints=False, root=root)

            self.assertTrue(any("missing required story sections: ### Finding" in finding.reason for finding in findings))
