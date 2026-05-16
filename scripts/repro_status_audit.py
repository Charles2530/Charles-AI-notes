#!/usr/bin/env python3
"""Audit evidence reproducibility labels in markdown pages."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_ROOTS = (
    Path("files/paper-deep-dives"),
    Path("files/world-models"),
    Path("files/references"),
)

SKIP_PAGES = {
    Path("files/references/evidence-and-reproducibility-standard.md"),
}

VALID_REPRO_STATUS = {
    "Independent Reproduced",
    "Author Code / Official Repo",
    "Paper Only",
    "Official Demo",
    "Site Inference",
    "Toy Fixture",
}

OFFICIAL_CODE_RE = re.compile(
    r"代码与模型|代码与数据|官方代码|官方仓库|GitHub[:：]|GitHub repo|github\.com|"
    r"huggingface\.co|Hugging Face|model card|模型卡|模型权重[:：]|model weights|checkpoint[:：]|weights[:：]|"
    r"benchmark script|运行脚本|official repo|Official Repo",
    re.IGNORECASE,
)

OFFICIAL_DEMO_RE = re.compile(
    r"Official Demo|官方 demo|官方演示|demo|Demo|project page|项目页|video|视频",
    re.IGNORECASE,
)

OFFICIAL_SOURCE_LINK_RE = re.compile(
    r"https?://|arxiv\.org|github\.com|huggingface\.co|official|官方|PDF|项目页|仓库",
    re.IGNORECASE,
)


@dataclass
class Finding:
    path: Path
    reason: str


def iter_pages(roots: tuple[Path, ...]) -> list[Path]:
    pages: list[Path] = []
    for root in roots:
        if root.is_file():
            pages.append(root)
        elif root.exists():
            pages.extend(root.rglob("*.md"))
    return sorted(set(pages))


def fact_row(text: str) -> list[str] | None:
    for line in text.splitlines():
        if "| 2026-05-16 |" not in line:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) >= 4:
            return cells
    return None


def fact_status(text: str) -> str | None:
    cells = fact_row(text)
    if cells and len(cells) >= 3:
        return cells[2]
    return None


def table_repro_statuses(text: str) -> set[str]:
    statuses: set[str] = set()
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        for status in VALID_REPRO_STATUS:
            if f"| {status} |" in line:
                statuses.add(status)
    return statuses


def has_independent_reproduced_data_row(text: str) -> bool:
    for line in text.splitlines():
        if not line.startswith("|") or "Independent Reproduced" not in line:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if "Independent Reproduced" in cells:
            return True
    return False


def audit_page(path: Path) -> list[Finding]:
    if path in SKIP_PAGES:
        return []

    text = path.read_text(encoding="utf-8", errors="ignore")
    findings: list[Finding] = []

    if "Repro Status" not in text:
        return findings

    status = fact_status(text)
    row = fact_row(text)
    has_official_code = bool(OFFICIAL_CODE_RE.search(text))
    has_official_demo = bool(OFFICIAL_DEMO_RE.search(text))

    if status and status not in VALID_REPRO_STATUS:
        findings.append(Finding(path, f"invalid fact Repro Status: {status}"))

    if row and len(row) >= 2 and not OFFICIAL_SOURCE_LINK_RE.search(row[1]):
        findings.append(
            Finding(path, "fact block Official Source lacks an obvious external/source marker")
        )

    if status == "Paper Only" and has_official_code:
        findings.append(
            Finding(path, "fact block is Paper Only but page mentions official code/repo/model assets")
        )

    if status == "Author Code / Official Repo" and not has_official_code:
        findings.append(
            Finding(path, "fact block is Author Code / Official Repo but no official code/repo marker was found")
        )

    if status == "Official Demo" and not has_official_demo:
        findings.append(
            Finding(path, "fact block is Official Demo but no demo/project/video marker was found")
        )

    if has_independent_reproduced_data_row(text):
        findings.append(
            Finding(path, "Independent Reproduced is present; require explicit third-party evidence before using it")
        )

    unknown_statuses = table_repro_statuses(text) - VALID_REPRO_STATUS
    for item in sorted(unknown_statuses):
        findings.append(Finding(path, f"unknown table Repro Status: {item}"))

    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="exit non-zero when findings are present")
    parser.add_argument("paths", nargs="*", type=Path, help="optional files or directories to audit")
    args = parser.parse_args()

    roots = tuple(args.paths) if args.paths else DEFAULT_ROOTS
    findings: list[Finding] = []
    for page in iter_pages(roots):
        findings.extend(audit_page(page))

    if findings:
        print("Repro status audit findings:")
        for finding in findings:
            print(f"- {finding.path}: {finding.reason}")
        return 1 if args.strict else 0

    print("Repro status audit passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
