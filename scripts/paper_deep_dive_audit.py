#!/usr/bin/env python3
"""Audit paper deep-dive pages for structure and evidence coverage."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path("files/paper-deep-dives")
MAX_CHARS = 30000
NEAR_LONG_CHARS = 28500
LONGFORM_EXEMPT_PAGES = {
    Path("world-models/world-model-for-robot-learning-survey.md"),
}
REQUIRED_FRONTMATTER_FIELDS = ("paper_date", "paper_title", "paper_topic")
REQUIRED_STORY_SECTIONS = (
    "### 研究背景",
    "### 问题（Challenge）",
    "### Finding",
    "### 方法",
    "### 结论",
    "### 关键术语",
)

SITE_JUDGEMENT_RE = re.compile(
    r'(^!!!\s+info\s+"本站判断"|^\s+\*\*info · 本站判断\*\*|本站判断)',
    re.MULTILINE,
)
PAPER_INFO_RE = re.compile(r'^!!!\s+info\s+"论文信息"', re.MULTILINE)
SECOND_PASS_RE = re.compile(r'二轮补强：方法和实验抓手')
IMAGE_RE = re.compile(r"!\[[^\]]*\]\(")
LOCAL_MD_LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\((?P<target>[^)]+\.md)(?:#[^)]+)?\)")
FIGURE_READING_RE = re.compile(
    r"这张图怎么读|图解|怎么读|表格怎么读|读图|读数边界|Figure\s+\d+|Table\s+\d+",
    re.IGNORECASE,
)
DIAGNOSTIC_RE = re.compile(
    r"消融|Ablation|ablation|diagnostic|诊断|scaling|Scaling|failure|失败|acceptance|读数边界",
    re.IGNORECASE,
)
METHOD_RE = re.compile(r"方法|结构|模型|训练|推理|objective|loss|采样|pipeline|architecture", re.IGNORECASE)
EXPERIMENT_RE = re.compile(r"实验|结果|评测|benchmark|Benchmark|Table|Figure|指标|setting", re.IGNORECASE)
REPRO_RE = re.compile(r"Checked Date|Repro Status|Author Code|Official Repo|Paper Only|Official Demo")
READING_CONCLUSION_RE = re.compile(r"^##\s+阅读结论\b", re.MULTILINE)
FRONTMATTER_RE = re.compile(r"\A---\n(?P<body>.*?)\n---\n", re.DOTALL)


@dataclass
class Finding:
    path: Path
    severity: str
    reason: str


def iter_pages(root: Path) -> list[Path]:
    skipped = {"index.md", "writing-standard.md"}
    return sorted(path for path in root.rglob("*.md") if path.name not in skipped)


def _relative_to_root(path: Path, root: Path) -> Path:
    try:
        return path.relative_to(root)
    except ValueError:
        return path.relative_to(Path.cwd()) if path.is_absolute() else path


def _index_links(root: Path) -> set[Path]:
    index = root / "index.md"
    if not index.exists():
        return set()

    text = index.read_text(encoding="utf-8", errors="ignore")
    links: set[Path] = set()
    for match in LOCAL_MD_LINK_RE.finditer(text):
        target = match.group("target").strip().strip("<>")
        if "://" in target or target.startswith(("#", "/")):
            continue
        links.add(Path(target.split("?", 1)[0]))
    return links


def _frontmatter(text: str) -> str:
    match = FRONTMATTER_RE.match(text)
    return match.group("body") if match else ""


def _missing_frontmatter_fields(text: str) -> list[str]:
    frontmatter = _frontmatter(text)
    return [
        field
        for field in REQUIRED_FRONTMATTER_FIELDS
        if not re.search(rf"^{re.escape(field)}\s*:\s*\S", frontmatter, re.MULTILINE)
    ]


def _missing_story_sections(text: str) -> list[str]:
    return [
        section
        for section in REQUIRED_STORY_SECTIONS
        if not re.search(rf"^{re.escape(section)}\s*$", text, re.MULTILINE)
    ]


def audit_index_coverage(root: Path) -> list[Finding]:
    pages = {_relative_to_root(path, root) for path in iter_pages(root)}
    links = _index_links(root)
    missing = sorted(page for page in pages if page not in links)
    if not missing:
        return []

    formatted = ", ".join(path.as_posix() for path in missing)
    return [Finding(Path("index.md"), "error", f"index.md missing links for: {formatted}")]


def audit_page(path: Path, include_hints: bool, root: Path = ROOT) -> list[Finding]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    findings: list[Finding] = []
    rel = _relative_to_root(path, root)
    is_longform_exempt = rel in LONGFORM_EXEMPT_PAGES

    missing_frontmatter = _missing_frontmatter_fields(text)
    if missing_frontmatter:
        missing = ", ".join(missing_frontmatter)
        findings.append(Finding(rel, "error", f"missing frontmatter fields: {missing}"))

    paper_info = PAPER_INFO_RE.findall(text)
    if len(paper_info) != 1:
        findings.append(Finding(rel, "error", f"expected exactly one 论文信息 admonition, found {len(paper_info)}"))

    missing_story_sections = _missing_story_sections(text)
    if missing_story_sections:
        missing = ", ".join(missing_story_sections)
        findings.append(Finding(rel, "error", f"missing required story sections: {missing}"))

    if SITE_JUDGEMENT_RE.search(text):
        findings.append(Finding(rel, "error", "remove 本站判断 wording from paper deep-dive pages"))

    if SECOND_PASS_RE.search(text):
        findings.append(Finding(rel, "error", "visible second-pass helper block remains"))

    if not READING_CONCLUSION_RE.search(text):
        findings.append(Finding(rel, "error", "missing 阅读结论 section"))

    if len(text) > MAX_CHARS and not is_longform_exempt:
        findings.append(Finding(rel, "error", f"page exceeds {MAX_CHARS} chars: {len(text)}"))

    if not DIAGNOSTIC_RE.search(text):
        findings.append(Finding(rel, "error", "missing ablation/diagnostic/scaling/failure signal"))

    image_count = len(IMAGE_RE.findall(text))
    if image_count >= 3 and not FIGURE_READING_RE.search(text):
        findings.append(Finding(rel, "error", f"{image_count} images but no figure/table reading note"))

    if include_hints:
        if len(text) > NEAR_LONG_CHARS and not is_longform_exempt:
            findings.append(Finding(rel, "hint", f"near long threshold; prefer compression before adding: {len(text)} chars"))
        if not REPRO_RE.search(text):
            findings.append(Finding(rel, "hint", "no explicit Checked Date/Repro Status style signal"))
        if not METHOD_RE.search(text):
            findings.append(Finding(rel, "hint", "method vocabulary is weak; manually confirm method interface is clear"))
        if not EXPERIMENT_RE.search(text):
            findings.append(Finding(rel, "hint", "experiment vocabulary is weak; manually confirm experiment setting is clear"))

    return findings


def audit_root(root: Path, include_hints: bool) -> list[Finding]:
    findings: list[Finding] = []
    for page in iter_pages(root):
        findings.extend(audit_page(page, include_hints=include_hints, root=root))
    findings.extend(audit_index_coverage(root))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-hints", action="store_true", help="include non-blocking improvement hints")
    parser.add_argument("--root", type=Path, default=ROOT, help="paper deep-dive root")
    args = parser.parse_args()

    pages = iter_pages(args.root)
    findings = audit_root(args.root, include_hints=args.include_hints)

    errors = [item for item in findings if item.severity == "error"]
    hints = [item for item in findings if item.severity == "hint"]

    if findings:
        print(f"Scanned {len(pages)} paper deep-dive pages under {args.root}/")
        print("Paper deep-dive audit findings:")
        for item in findings:
            print(f"- [{item.severity}] {item.path}: {item.reason}")
    else:
        print(f"Paper deep-dive audit passed. Scanned {len(pages)} pages under {args.root}/")

    if hints and not errors:
        print(f"Hints only: {len(hints)}")

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
