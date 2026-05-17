#!/usr/bin/env python3
"""Report content pages that are likely too fragmented, too long, or templated."""

from __future__ import annotations

import argparse
from collections import Counter
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse


DOCS_ROOT = Path("files")
EXCLUDED_PARTS = {
    "roadmap",
}

MIN_CHARS = 2500
MAX_CHARS = 25000
MAX_H2 = 18
MAX_H3 = 60

FULLY_EXEMPT_PAGES = {
    Path("files/references/index.md"),
}

NUMBERED_SEQUENCE_PAGES = {
    Path("files/quantization/page-by-page-visual-guide.md"),
    Path("files/training/page-by-page-visual-guide.md"),
}

SHORT_EXEMPT_PREFIXES = (
    "files/foundations/",
    "files/paper-guides/",
)

SHORT_EXEMPT_PAGES = {
    Path("files/operators/hardware-aware-debug-checklist.md"),
    Path("files/operators/kernel-cost-models-and-selection-heuristics.md"),
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

LOCAL_LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
HTML_LINK_RE = re.compile(r"<a\s+[^>]*href=[\"']([^\"']+)[\"']", re.IGNORECASE)
HTML_IMAGE_RE = re.compile(r"<img\s+[^>]*src=[\"']([^\"']+)[\"']", re.IGNORECASE)
SOURCE_CAPTION_RE = re.compile(r"(图源|图片来源|表源|Source|source|原论文图|官方截图)")
FENCED_CODE_RE = re.compile(r"(^|\n)```.*?(\n```|$)", re.DOTALL)


@dataclass
class PageReport:
    path: Path
    chars: int
    h2: int
    h3: int
    flags: list[str]


def page_group(path: Path) -> str:
    parts = path.relative_to(DOCS_ROOT).parts
    if len(parts) >= 3 and parts[0] == "paper-deep-dives":
        return "/".join(parts[:2])
    return parts[0] if parts else path.name


def page_type(path: Path) -> str:
    rel = path.relative_to(DOCS_ROOT)
    if rel.name == "index.md":
        return "index"
    if rel.parts and rel.parts[0] == "paper-deep-dives":
        return "paper-deep-dive"
    if "page-by-page-visual-guide" in path.stem:
        return "visual-guide"
    return "topic"


def clean_target(raw: str) -> str:
    target = raw.strip().strip("<>")
    if not target:
        return ""

    # Markdown links may include an optional title after whitespace:
    # (path/to/file.png "caption"). Paths in this repo are currently space-free.
    if " " in target and not target.startswith(("http://", "https://")):
        target = target.split(" ", 1)[0]
    return unquote(target.strip())


def is_external_target(target: str) -> bool:
    parsed = urlparse(target)
    if parsed.scheme in {"http", "https", "mailto", "tel", "javascript", "data"}:
        return True
    if target.startswith("#"):
        return True
    return False


def target_exists(base_dir: Path, target: str) -> bool:
    target = clean_target(target)
    if not target or is_external_target(target):
        return True

    link_path = target.split("#", 1)[0].split("?", 1)[0]
    if not link_path:
        return True

    candidate = (base_dir / link_path).resolve()
    root = Path.cwd().resolve()

    try:
        candidate.relative_to(root)
    except ValueError:
        return False

    if candidate.exists():
        return True

    suffix = candidate.suffix.lower()
    if suffix:
        return False

    candidates = [
        Path(str(candidate).rstrip("/") + ".md"),
        candidate / "index.md",
    ]

    if str(link_path).endswith("/"):
        candidates.append(Path(str(candidate).rstrip("/") + ".md"))

    return any(item.exists() for item in candidates)


def resolve_markdown_target(path: Path, target: str) -> bool:
    return target_exists(path.parent, target)


def rendered_page_dir(path: Path) -> Path:
    rel = path.relative_to(DOCS_ROOT)
    if rel.name == "index.md":
        return DOCS_ROOT / rel.parent
    return DOCS_ROOT / rel.parent / rel.stem


def resolve_html_target(path: Path, target: str) -> bool:
    return target_exists(rendered_page_dir(path), target)


def strip_fenced_code(text: str) -> str:
    return FENCED_CODE_RE.sub("\n", text)


def image_has_nearby_source(lines: list[str], line_index: int) -> bool:
    window = "\n".join(lines[line_index : min(len(lines), line_index + 24)])
    return bool(SOURCE_CAPTION_RE.search(window))


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
    searchable_text = strip_fenced_code(text)
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
    intentional_sequence = path in NUMBERED_SEQUENCE_PAGES
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

    if include_hints and intentional_sequence and numbered_h2 >= 3:
        flags.append("info:intentional-numbered-sequence")
    elif include_hints and numbered_h2 >= 10 and len(h2_lines) <= MAX_H2:
        flags.append("hint:numbered-h2-style")

    markdown_targets = [
        *LOCAL_LINK_RE.findall(searchable_text),
        *IMAGE_RE.findall(searchable_text),
    ]
    html_targets = [
        *HTML_LINK_RE.findall(searchable_text),
        *HTML_IMAGE_RE.findall(searchable_text),
    ]
    broken_targets = [
        *[
            target for target in markdown_targets if not resolve_markdown_target(path, target)
        ],
        *[
            target for target in html_targets if not resolve_html_target(path, target)
        ],
    ]
    if broken_targets:
        flags.append(f"broken-local-link:{len(broken_targets)}")

    if include_hints:
        image_lines = [
            index
            for index, line in enumerate(lines)
            if IMAGE_RE.search(line) or HTML_IMAGE_RE.search(line)
        ]
        unsourced_images = [
            index for index in image_lines if not image_has_nearby_source(lines, index)
        ]
        if unsourced_images:
            flags.append(f"hint:image-without-nearby-source:{len(unsourced_images)}")

    return PageReport(path=path, chars=len(text), h2=len(h2_lines), h3=len(h3_lines), flags=flags)


def hard_flags(flags: list[str]) -> list[str]:
    return [flag for flag in flags if not flag.startswith("info:")]


def print_summary(reports: list[PageReport], flagged: list[PageReport]) -> None:
    if not flagged:
        return

    flag_counts = Counter(flag for report in flagged for flag in report.flags)
    group_counts = Counter(page_group(report.path) for report in flagged)
    type_counts = Counter(page_type(report.path) for report in flagged)

    print("\nSummary:")
    print("  by flag:")
    for flag, count in flag_counts.most_common():
        print(f"    {flag}: {count}")

    print("  by directory:")
    for group, count in group_counts.most_common(12):
        print(f"    {group}: {count}")

    print("  by page type:")
    for item, count in type_counts.most_common():
        print(f"    {item}: {count}")


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
    failing = [report for report in reports if hard_flags(report.flags)]
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

    print_summary(reports, flagged)

    print("\nFlagged pages:")
    for report in flagged:
        rel = report.path.as_posix()
        flag_text = ", ".join(report.flags)
        print(
            f"- {rel}: chars={report.chars}, h2={report.h2}, "
            f"h3={report.h3}, flags={flag_text}"
        )

    return 1 if args.fail_on_warn and failing else 0


if __name__ == "__main__":
    raise SystemExit(main())
