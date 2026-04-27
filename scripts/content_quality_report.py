#!/usr/bin/env python3
"""Report content pages that are likely too fragmented, too long, or templated."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


DOCS_ROOT = Path("docs")
EXCLUDED_PARTS = {
    "bin",
    "conda-meta",
    "etc",
    "include",
    "lib",
    "man",
    "roadmap",
    "share",
    "ssl",
}

MIN_CHARS = 2500
MAX_CHARS = 25000
MAX_H2 = 18
MAX_H3 = 60

FULLY_EXEMPT_PAGES = {
    Path("docs/references/index.md"),
    Path("docs/training/page-by-page-visual-guide.md"),
}

SHORT_EXEMPT_PREFIXES = (
    "docs/foundations/",
    "docs/paper-guides/",
)

SHORT_EXEMPT_PAGES = {
    Path("docs/operators/hardware-aware-debug-checklist.md"),
    Path("docs/operators/kernel-cost-models-and-selection-heuristics.md"),
}

TEMPLATE_PATTERNS = [
    "如果要把 **",
    "什么算这页真正写扎实了",
    "读完这页后最好能明确回答",
    "比继续堆内容更重要",
    "项目推进提醒",
    "阶段性小结",
    "最终小结",
    "收口小结",
    "终局小结",
]


@dataclass
class PageReport:
    path: Path
    chars: int
    h2: int
    h3: int
    flags: list[str]


def iter_markdown_pages() -> list[Path]:
    pages: list[Path] = []
    for path in DOCS_ROOT.rglob("*.md"):
        if path.name.startswith("._"):
            continue
        if any(part in EXCLUDED_PARTS for part in path.parts):
            continue
        pages.append(path)
    return sorted(pages)


def analyze(path: Path, include_hints: bool = False) -> PageReport:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    h2_lines = [line for line in lines if line.startswith("## ")]
    h3_lines = [line for line in lines if line.startswith("### ")]
    flags: list[str] = []

    if path in FULLY_EXEMPT_PAGES:
        return PageReport(path=path, chars=len(text), h2=len(h2_lines), h3=len(h3_lines), flags=flags)

    short_exempt = path in SHORT_EXEMPT_PAGES or any(
        path.as_posix().startswith(prefix) for prefix in SHORT_EXEMPT_PREFIXES
    )

    if len(text) < MIN_CHARS and not short_exempt:
        flags.append("short")
    if len(text) > MAX_CHARS:
        flags.append("long")
    if len(h2_lines) > MAX_H2:
        flags.append("too-many-h2")
    if len(h3_lines) > MAX_H3:
        flags.append("too-many-h3")

    numbered_h2 = sum(1 for line in h2_lines if re.match(r"##\s+\d+[.、]", line))
    if len(h2_lines) > MAX_H2 and numbered_h2 >= 10:
        flags.append("numbered-h2-run")

    repeated_templates = [
        pattern for pattern in TEMPLATE_PATTERNS if text.count(pattern) >= 2
    ]
    if repeated_templates:
        flags.append("repeated-summary")

    if include_hints and path.name != "index.md" and len(text) > MIN_CHARS:
        local_links = re.findall(r"\]\((?!https?://|#)([^)]+)\)", text)
        if len(local_links) == 0:
            flags.append("hint:no-local-cross-link")

    if include_hints and numbered_h2 >= 10 and len(h2_lines) <= MAX_H2:
        flags.append("hint:numbered-h2-style")

    return PageReport(path=path, chars=len(text), h2=len(h2_lines), h3=len(h3_lines), flags=flags)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fail-on-warn",
        action="store_true",
        help="return a non-zero exit code when any page is flagged",
    )
    parser.add_argument(
        "--include-hints",
        action="store_true",
        help="also report lower-priority style hints such as missing local links",
    )
    args = parser.parse_args()

    reports = [analyze(path, include_hints=args.include_hints) for path in iter_markdown_pages()]
    flagged = [report for report in reports if report.flags]
    flagged.sort(key=lambda r: (len(r.flags), r.chars, r.h2), reverse=True)

    print(f"Scanned {len(reports)} markdown pages under {DOCS_ROOT}/")
    print(
        "Thresholds: "
        f"short<{MIN_CHARS} chars, long>{MAX_CHARS} chars, "
        f"h2>{MAX_H2}, h3>{MAX_H3}"
    )

    if not flagged:
        print("No content quality warnings.")
        return 0

    print("\nFlagged pages:")
    for report in flagged:
        rel = report.path.as_posix()
        flag_text = ", ".join(report.flags)
        print(
            f"- {rel}: chars={report.chars}, h2={report.h2}, "
            f"h3={report.h3}, flags={flag_text}"
        )

    return 1 if args.fail_on_warn else 0


if __name__ == "__main__":
    raise SystemExit(main())
