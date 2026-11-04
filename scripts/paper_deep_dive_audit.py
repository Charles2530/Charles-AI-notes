#!/usr/bin/env python3
"""Audit paper metadata, source presence, index coverage, and local resources."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

if __package__:
    from .content_quality_report import (
        LOCAL_LINK_RE, broken_local_targets, clean_target, is_external_target, strip_fenced_code,
    )
else:
    from content_quality_report import (
        LOCAL_LINK_RE, broken_local_targets, clean_target, is_external_target, strip_fenced_code,
    )


ROOT = Path("files/paper-deep-dives")
NEAR_LONG_CHARS = 28500
REQUIRED_FRONTMATTER_FIELDS = ("paper_date", "paper_title", "paper_topic")

SITE_JUDGEMENT_RE = re.compile(
    r'(^!!!\s+info\s+"本站判断"|^\s+\*\*info · 本站判断\*\*|本站判断)',
    re.MULTILINE,
)
PAPER_INFO_RE = re.compile(
    r'''^!!![ \t]+info[ \t]+(?P<quote>["'])(?:论文信息|资料信息)(?P=quote)[ \t]*\n'''
    r"(?P<body>(?:(?: {4}|\t)[^\n]*(?:\n|$)|[ \t]*\n)*)",
    re.MULTILINE,
)
SOURCE_URL_RE = re.compile(r'''https?://[^\s<>\[\]()"']+''', re.IGNORECASE)
SECOND_PASS_RE = re.compile(r'二轮补强：方法和实验抓手')
IMAGE_RE = re.compile(r"!\[[^\]]*\]\(")
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
REPRO_RE = re.compile(r"Checked Date|Repro Status|Author Code|Official Repo|Paper Only|Official Demo|核验日期|复现状态")
FRONTMATTER_RE = re.compile(r"\A---\n(?P<body>.*?)\n---\n", re.DOTALL)


@dataclass
class Finding:
    path: Path
    severity: str
    reason: str


def iter_pages(root: Path) -> list[Path]:
    skipped = {"index.md"}
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

    text = strip_fenced_code(index.read_text(encoding="utf-8", errors="ignore"))
    links: set[Path] = set()
    for raw_target in LOCAL_LINK_RE.findall(text):
        target = clean_target(raw_target)
        if is_external_target(target) or target.startswith("/"):
            continue
        target_path = target.split("#", 1)[0].split("?", 1)[0]
        if Path(target_path).suffix.lower() != ".md":
            continue
        try:
            links.add((root / target_path).resolve().relative_to(root.resolve()))
        except ValueError:
            continue
    return links


def _frontmatter(text: str) -> str:
    match = FRONTMATTER_RE.match(text)
    return match.group("body") if match else ""


def _missing_frontmatter_fields(text: str) -> list[str]:
    frontmatter = _frontmatter(text)
    missing: list[str] = []
    for field in REQUIRED_FRONTMATTER_FIELDS:
        match = re.search(rf"^{re.escape(field)}[ \t]*:[ \t]*([^\n]*)", frontmatter, re.MULTILINE)
        value = match.group(1).strip() if match else ""
        if re.match(r'''^(?:"\s*"|'\s*'|null|Null|NULL|~)?\s*(?:#.*)?$''', value):
            missing.append(field)
    return missing


def _has_source_url(block: str) -> bool:
    for url in SOURCE_URL_RE.findall(block):
        try:
            if urlparse(url).hostname:
                return True
        except ValueError:
            continue
    return False


def audit_index_coverage(root: Path) -> list[Finding]:
    pages = {_relative_to_root(path, root) for path in iter_pages(root)}
    links = _index_links(root)
    missing = sorted(page for page in pages if page not in links)
    findings: list[Finding] = []
    if missing:
        formatted = ", ".join(path.as_posix() for path in missing)
        findings.append(Finding(Path("index.md"), "error", f"index.md missing links for: {formatted}"))
    index = root / "index.md"
    if index.exists():
        for target in broken_local_targets(index, index.read_text(encoding="utf-8"), root.parent):
            findings.append(Finding(Path("index.md"), "error", f"broken local resource: {target}"))
    return findings


def audit_page(path: Path, include_hints: bool, root: Path = ROOT) -> list[Finding]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    searchable_text = strip_fenced_code(text)
    findings: list[Finding] = []
    rel = _relative_to_root(path, root)

    missing_frontmatter = _missing_frontmatter_fields(text)
    if missing_frontmatter:
        missing = ", ".join(missing_frontmatter)
        findings.append(Finding(rel, "error", f"missing frontmatter fields: {missing}"))

    paper_info = list(PAPER_INFO_RE.finditer(searchable_text))
    if len(paper_info) != 1:
        findings.append(Finding(rel, "error", f"expected exactly one 论文信息 or 资料信息 admonition, found {len(paper_info)}"))
    elif not _has_source_url(paper_info[0].group("body")):
        findings.append(Finding(rel, "error", "missing source URL in 论文信息 or 资料信息 admonition"))

    for target in broken_local_targets(path, searchable_text, root.parent):
        findings.append(Finding(rel, "error", f"broken local resource: {target}"))

    if include_hints:
        if len(text) > NEAR_LONG_CHARS:
            findings.append(Finding(rel, "hint", f"length {len(text)} chars; manually review scope and navigation, with no hard limit"))
        if SITE_JUDGEMENT_RE.search(searchable_text):
            findings.append(Finding(rel, "hint", "review 本站判断 wording for unsupported editorial claims"))
        if SECOND_PASS_RE.search(searchable_text):
            findings.append(Finding(rel, "hint", "review visible second-pass helper block"))
        if not DIAGNOSTIC_RE.search(searchable_text):
            findings.append(Finding(rel, "hint", "no diagnostic keywords; manually review evidence where relevant to this article type"))
        image_count = len(IMAGE_RE.findall(searchable_text))
        if image_count >= 3 and not FIGURE_READING_RE.search(searchable_text):
            findings.append(Finding(rel, "hint", f"{image_count} images; manually review explanations of figures and tables"))
        if not REPRO_RE.search(searchable_text):
            findings.append(Finding(rel, "hint", "no explicit Checked Date/Repro Status style signal"))
        if not METHOD_RE.search(searchable_text):
            findings.append(Finding(rel, "hint", "method vocabulary is weak; manually confirm method interface is clear"))
        if not EXPERIMENT_RE.search(searchable_text):
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
    print("Structural checks do not verify source credibility, technical claims, formula correctness, or evidence quality.")

    if findings:
        print(f"Scanned {len(pages)} paper deep-dive pages under {args.root}/")
        print("Paper deep-dive structural audit findings:")
        for item in findings:
            print(f"- [{item.severity}] {item.path}: {item.reason}")
    else:
        print(f"Paper deep-dive structural audit passed. Scanned {len(pages)} pages under {args.root}/")

    if hints and not errors:
        print(f"Hints only: {len(hints)}")

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
