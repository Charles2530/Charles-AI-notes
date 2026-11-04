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
    **原始来源：** [Paper](https://arxiv.org/abs/2401.00001)。<br>
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

    def test_reading_conclusion_heading_is_not_required(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            page = root / "world-models" / "missing-conclusion.md"
            page.parent.mkdir(parents=True)
            page.write_text(_valid_page("Missing Conclusion").replace("## 阅读结论\n\nMissing Conclusion 的主要内容已经闭合。\n", ""), encoding="utf-8")

            findings = audit_page(page, include_hints=False, root=root)

            self.assertEqual(findings, [])

    def test_missing_frontmatter_field_is_reported(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            page = root / "world-models" / "missing-topic.md"
            page.parent.mkdir(parents=True)
            page.write_text(_valid_page("Missing Topic").replace("paper_topic: 世界模型\n", ""), encoding="utf-8")

            findings = audit_page(page, include_hints=False, root=root)

            self.assertTrue(any("missing frontmatter fields: paper_topic" in finding.reason for finding in findings))

    def test_alternative_story_headings_are_accepted(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            page = root / "world-models" / "missing-story-section.md"
            page.parent.mkdir(parents=True)
            page.write_text(_valid_page("Missing Story Section").replace("### Finding\n\nMissing Story Section 的关键发现。\n\n", ""), encoding="utf-8")

            findings = audit_page(page, include_hints=False, root=root)

            self.assertEqual(findings, [])

    def test_official_technical_material_header_is_accepted(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            page = root / "report.md"
            page.write_text(_valid_page("Official report").replace('info "论文信息"', 'info "资料信息"'), encoding="utf-8")
            self.assertEqual(audit_page(page, include_hints=False, root=root), [])

    def test_any_long_article_is_a_review_hint_not_an_error(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            page = root / "long-method.md"
            page.write_text(_valid_page("Detailed method") + "\nTechnical discussion." * 1800, encoding="utf-8")
            self.assertEqual(audit_page(page, include_hints=False, root=root), [])
            findings = audit_page(page, include_hints=True, root=root)
            self.assertTrue(any(item.severity == "hint" and "length" in item.reason for item in findings))
            self.assertFalse(any(item.severity == "error" for item in findings))

    def test_absent_diagnostic_keywords_do_not_claim_missing_evidence(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            page = root / "method.md"
            page.write_text(_valid_page("Method", body="控制其他变量后比较两种设置。"), encoding="utf-8")
            self.assertEqual(audit_page(page, include_hints=False, root=root), [])

    def test_source_link_must_exist_in_information_block(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            page = root / "unsourced.md"
            text = _valid_page("Unsourced").replace('    **原始来源：** [Paper](https://arxiv.org/abs/2401.00001)。<br>\n', '')
            text += "\n[Unrelated further reading](https://example.org/another-paper)\n"
            page.write_text(text, encoding="utf-8")
            findings = audit_page(page, include_hints=False, root=root)
            self.assertTrue(any(item.severity == "error" and "source URL" in item.reason for item in findings))

    def test_empty_metadata_value_is_missing(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            page = root / "empty-title.md"
            for value in ('', '""', "'  '", 'null', '~', '# not supplied', '"" # not supplied'):
                with self.subTest(value=value):
                    page.write_text(_valid_page("Title").replace('paper_title: "Title"', f'paper_title: {value}'), encoding="utf-8")
                    findings = audit_page(page, include_hints=False, root=root)
                    self.assertTrue(any("paper_title" in item.reason and item.severity == "error" for item in findings))

    def test_missing_or_duplicated_information_block_is_an_error(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            page = root / "method.md"
            original = _valid_page("Method")
            for text in (original.replace('!!! info "论文信息"', '!!! note "背景"'), original + '\n!!! info "资料信息"\n    https://example.org/paper\n'):
                with self.subTest(text=text):
                    page.write_text(text, encoding="utf-8")
                    findings = audit_page(page, include_hints=False, root=root)
                    self.assertTrue(any(item.severity == "error" and "exactly one" in item.reason for item in findings))

    def test_source_url_without_a_host_is_an_error(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            page = root / "method.md"
            page.write_text(_valid_page("Method").replace('https://arxiv.org/abs/2401.00001', 'https:///missing-host'), encoding="utf-8")
            findings = audit_page(page, include_hints=False, root=root)
            self.assertTrue(any(item.severity == "error" and "source URL" in item.reason for item in findings))

    def test_missing_markdown_and_html_resources_are_errors(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "papers"
            root.mkdir()
            page = root / "method.md"
            page.write_text(_valid_page("Method") + '\n[Missing](absent.md)\n![Figure](absent.png)\n<img src="../missing.svg">\n', encoding="utf-8")
            findings = audit_page(page, include_hints=False, root=root)
            reasons = "\n".join(item.reason for item in findings if item.severity == "error")
            for name in ("absent.md", "absent.png", "../missing.svg"):
                self.assertIn(name, reasons)

    def test_existing_shared_assets_and_fragment_links_are_accepted(self):
        with TemporaryDirectory() as tmpdir:
            docs = Path(tmpdir)
            root = docs / "papers"
            root.mkdir()
            (docs / "figure.svg").write_text("<svg></svg>", encoding="utf-8")
            (root / "related.md").write_text("# Related", encoding="utf-8")
            page = root / "method.md"
            page.write_text(_valid_page("Method") + '\n[Related](related.md#topic)\n![Figure](../figure.svg)\n<img src="../../figure.svg">\n[Same page](#topic)\n', encoding="utf-8")
            self.assertEqual(audit_page(page, include_hints=False, root=root), [])

    def test_links_in_code_examples_are_ignored(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            page = root / "method.md"
            page.write_text(_valid_page("Method") + '\n~~~markdown\n[Template](missing.md)\n~~~\n```markdown\n![Template](missing.png)\n```\n', encoding="utf-8")
            self.assertEqual(audit_page(page, include_hints=False, root=root), [])

    def test_index_reports_stale_links_as_well_as_coverage(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "index.md").write_text("[Deleted article](gone.md)\n", encoding="utf-8")
            findings = audit_root(root, include_hints=False)
            self.assertTrue(any(item.path == Path("index.md") and "gone.md" in item.reason for item in findings))

    def test_index_code_sample_does_not_count_as_article_coverage(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "method.md").write_text(_valid_page("Method"), encoding="utf-8")
            (root / "index.md").write_text("~~~markdown\n[Method](method.md)\n~~~\n", encoding="utf-8")
            findings = audit_root(root, include_hints=False)
            self.assertTrue(any("index.md missing links for: method.md" in item.reason for item in findings))
