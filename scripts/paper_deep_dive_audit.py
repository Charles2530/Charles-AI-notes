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
SHORT_HINT_CHARS = 12000

JUDGEMENT_RE = re.compile(
    r'(^!!!\s+info\s+"本站判断"|^\s+\*\*info · 本站判断\*\*)',
    re.MULTILINE,
)
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
REPRO_RE = re.compile(r"Checked Date|Repro Status|Author Code|Official Repo|Paper Only|Official Demo")


@dataclass
class Finding:
    path: Path
    severity: str
    reason: str


def iter_pages(root: Path) -> list[Path]:
    skipped = {"index.md", "writing-standard.md"}
    return sorted(path for path in root.rglob("*.md") if path.name not in skipped)


def audit_page(path: Path, include_hints: bool) -> list[Finding]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    findings: list[Finding] = []
    rel = path.relative_to(Path.cwd()) if path.is_absolute() else path

    judgements = JUDGEMENT_RE.findall(text)
    if len(judgements) != 1:
        findings.append(Finding(rel, "error", f"expected exactly one 本站判断 admonition, found {len(judgements)}"))

    if "\n## 本站判断" in text:
        findings.append(Finding(rel, "error", "remove duplicate ## 本站判断 heading; use only the admonition"))

    if SECOND_PASS_RE.search(text):
        findings.append(Finding(rel, "error", "visible second-pass helper block remains"))

    if len(text) > MAX_CHARS:
        findings.append(Finding(rel, "error", f"page exceeds {MAX_CHARS} chars: {len(text)}"))

    if not DIAGNOSTIC_RE.search(text):
        findings.append(Finding(rel, "error", "missing ablation/diagnostic/scaling/failure signal"))

    image_count = len(IMAGE_RE.findall(text))
    if image_count >= 3 and not FIGURE_READING_RE.search(text):
        findings.append(Finding(rel, "error", f"{image_count} images but no figure/table reading note"))

    if include_hints:
        if len(text) < SHORT_HINT_CHARS:
            findings.append(Finding(rel, "hint", f"short deep dive; consider method/experiment expansion: {len(text)} chars"))
        if len(text) > NEAR_LONG_CHARS:
            findings.append(Finding(rel, "hint", f"near long threshold; prefer compression before adding: {len(text)} chars"))
        if not REPRO_RE.search(text):
            findings.append(Finding(rel, "hint", "no explicit Checked Date/Repro Status style signal"))
        if not METHOD_RE.search(text):
            findings.append(Finding(rel, "hint", "method vocabulary is weak; manually confirm method interface is clear"))
        if not EXPERIMENT_RE.search(text):
            findings.append(Finding(rel, "hint", "experiment vocabulary is weak; manually confirm experiment setting is clear"))

    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-hints", action="store_true", help="include non-blocking improvement hints")
    parser.add_argument("--root", type=Path, default=ROOT, help="paper deep-dive root")
    args = parser.parse_args()

    pages = iter_pages(args.root)
    findings: list[Finding] = []
    for page in pages:
        findings.extend(audit_page(page, include_hints=args.include_hints))

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
