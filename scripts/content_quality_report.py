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
MAX_CHARS = 30000
MAX_H2 = 18
MAX_H3 = 60

FULLY_EXEMPT_PAGES = {
    Path("files/references/index.md"),
}

LONG_EXEMPT_REL_PAGES = {
    Path("paper-deep-dives/world-models/world-model-for-robot-learning-survey.md"),
}

NUMBERED_SEQUENCE_PAGES = {
    Path("files/quantization/page-by-page-visual-guide.md"),
    Path("files/training/page-by-page-visual-guide.md"),
}

LEGACY_ENTRY_MARKER = "保留给旧链接和搜索结果使用；它不再作为"

SHORT_EXEMPT_PREFIXES = (
    "files/foundations/",
    "files/paper-guides/",
)

SHORT_EXEMPT_PAGES = {
    Path("files/operators/hardware-aware-debug-checklist.md"),
    Path("files/operators/kernel-cost-models-and-selection-heuristics.md"),
}

TEMPLATE_PATTERNS = [
    "合并提示",
    "读法定位",
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
INLINE_MATH_RE = re.compile(r"\\\(.+?\\\)")
FORMULA_EXPLANATION_RE = re.compile(
    r"(符号含义|公式片段|白话读法|读作什么|读作|这行式子在说|这两行式子在说|逐项读法|槽位|"
    r"其中|这里|表示|含义|变量|参数|条件|优化对象|拆开读|"
    r"定义为|写成|可以写成|估算|计算|满足|等于|约为|近似|是)"
)
GENERIC_SYMBOL_EXPLANATION_RE = re.compile(
    r"公式中的变量或中间量|按上下文定义|return、组大小或缩放系数|value、速度场或向量量"
)
ADMONITION_RE = re.compile(r"^(!!!|\?\?\?)\s+", re.MULTILINE)
MERGED_PROMPT_BOX_RE = re.compile(r"!!!\s+\w+\s+[\"']合并提示[\"']|合并提示")
READING_POSITIONING_RE = re.compile(r"读法定位")
AUTO_GENERATED_HINT_RE = re.compile(
    r"> \*\*(踩坑原因|算法视角)\*\*|<!-- \[融合：(算法基础|复杂度分析)\]"
)


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


def has_formula_explanation_nearby(lines: list[str], start: int, end: int) -> bool:
    window_start = max(0, start - 4)
    window_end = min(len(lines), end + 10)
    window = "\n".join(lines[window_start:window_end])
    return bool(FORMULA_EXPLANATION_RE.search(window))


def iter_display_formula_blocks(lines: list[str]) -> list[tuple[int, int]]:
    blocks: list[tuple[int, int]] = []
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        delimiter = ""
        if stripped == r"\[":
            delimiter = r"\]"
        elif stripped == "$$":
            delimiter = "$$"

        if not delimiter:
            index += 1
            continue

        start = index
        index += 1
        while index < len(lines) and lines[index].strip() != delimiter:
            index += 1
        end = index if index < len(lines) else len(lines) - 1
        blocks.append((start, end))
        index += 1

    return blocks


def inline_formula_context_hint_count(lines: list[str]) -> int:
    missing_context = 0
    for index, line in enumerate(lines):
        if line.lstrip().startswith("|"):
            continue

        inline_math_count = len(INLINE_MATH_RE.findall(line))
        if inline_math_count < 3:
            continue

        if not has_formula_explanation_nearby(lines, index, index + 1):
            missing_context += 1

    return missing_context


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
    rel = path.relative_to(DOCS_ROOT)
    lines = text.splitlines()
    h2_lines = [line for line in lines if line.startswith("## ")]
    h3_lines = [line for line in lines if line.startswith("### ")]
    flags: list[str] = []

    if path in FULLY_EXEMPT_PAGES:
        return PageReport(path=path, chars=len(text), h2=len(h2_lines), h3=len(h3_lines), flags=flags)

    short_exempt = (
        rel.name == "index.md"
        or path in SHORT_EXEMPT_PAGES
        or LEGACY_ENTRY_MARKER in text
        or any(
            path.as_posix().startswith(prefix) for prefix in SHORT_EXEMPT_PREFIXES
        )
    )

    if len(text) < MIN_CHARS and not short_exempt:
        flags.append("short")
    if len(text) > MAX_CHARS and rel not in LONG_EXEMPT_REL_PAGES:
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

    if MERGED_PROMPT_BOX_RE.search(searchable_text):
        flags.append("template:merged-prompt-box")

    if AUTO_GENERATED_HINT_RE.search(searchable_text):
        flags.append("template:auto-generated-hint-block")

    first_chunk = "\n".join(searchable_text.splitlines()[:80])
    if READING_POSITIONING_RE.search(first_chunk):
        flags.append("template:reading-positioning-start")

    # A page does not need forced internal links. The current editorial
    # standard favors standalone explanations plus a few genuinely valuable
    # external sources when they help the reader more than site navigation.

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

        searchable_lines = searchable_text.splitlines()
        unexplained_display_formulae = [
            (start, end)
            for start, end in iter_display_formula_blocks(searchable_lines)
            if not has_formula_explanation_nearby(searchable_lines, start, end)
        ]
        if unexplained_display_formulae:
            flags.append(f"hint:formula-context:{len(unexplained_display_formulae)}")

        inline_hint_count = inline_formula_context_hint_count(searchable_lines)
        if inline_hint_count:
            flags.append(f"hint:inline-formula-context:{inline_hint_count}")

        generic_symbol_count = len(GENERIC_SYMBOL_EXPLANATION_RE.findall(searchable_text))
        if generic_symbol_count:
            flags.append("hint:generic-symbol-explanation")

        admonition_count = len(ADMONITION_RE.findall(searchable_text))
        ordinary_article = page_type(path) == "topic"
        if ordinary_article and admonition_count > 2:
            flags.append(f"hint:too-many-admonitions:{admonition_count}")

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
    parser.add_argument(
        "--list-pages",
        action="store_true",
        help="print every markdown page included in the scan",
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
    if EXCLUDED_PARTS:
        excluded = ", ".join(sorted(EXCLUDED_PARTS))
        print(f"Excluded directories: {excluded}")

    if args.list_pages:
        print("\nScanned pages:")
        for report in sorted(reports, key=lambda item: item.path.as_posix()):
            rel = report.path.as_posix()
            print(f"- {rel}: chars={report.chars}, h2={report.h2}, h3={report.h3}")

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
