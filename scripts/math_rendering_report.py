#!/usr/bin/env python3
"""Find Markdown patterns that usually render LaTeX as plain code/text."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import re


DOCS_ROOT = Path("files")
EXCLUDED_PARTS = {"assets"}

LATEX_COMMAND_RE = re.compile(
    r"\\(?:"
    r"alpha|beta|gamma|delta|epsilon|eta|lambda|mu|pi|sigma|theta|"
    r"approx|cdot|frac|ge|infty|le|mathbb|mathcal|mathrm|mid|"
    r"operatorname|partial|prod|sqrt|sum|text|times|top|"
    r"lVert|rVert|lceil|rceil"
    r")\b|\\(?:[\\[\\]])"
)
CODE_SPAN_RE = re.compile(r"`([^`\n]+)`")
INLINE_MATH_RE = re.compile(r"\\\((?P<paren>[^\n]*?)\\\)|\$(?P<dollar>[^$\n]+)\$")


@dataclass(frozen=True)
class Issue:
    path: Path
    line_no: int
    code: str
    detail: str


def iter_markdown_pages() -> list[Path]:
    pages: list[Path] = []
    for path in DOCS_ROOT.rglob("*.md"):
        rel_parts = set(path.relative_to(DOCS_ROOT).parts)
        if path.name.startswith("._"):
            continue
        if rel_parts.intersection(EXCLUDED_PARTS):
            continue
        pages.append(path)
    return sorted(pages)


def strip_inline_math_and_code(line: str) -> str:
    line = CODE_SPAN_RE.sub("", line)
    return INLINE_MATH_RE.sub("", line)


def math_brace_error(segment: str) -> str | None:
    depth = 0
    escaped = False
    for offset, char in enumerate(segment):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth < 0:
                return f"extra close brace at offset {offset}"
    if depth:
        return f"missing close brace depth {depth}"
    return None


def check_math_segment(path: Path, line_no: int, segment: str) -> Issue | None:
    error = math_brace_error(segment)
    if not error:
        return None
    detail = f"{error}: {segment.strip()[:160]}"
    return Issue(path, line_no, "unbalanced-math-braces", detail)


def inline_math_issues(path: Path, line_no: int, line: str) -> list[Issue]:
    issues: list[Issue] = []
    scan_line = CODE_SPAN_RE.sub("", line)
    for match in INLINE_MATH_RE.finditer(scan_line):
        segment = match.group("paren") if match.group("paren") is not None else match.group("dollar")
        issue = check_math_segment(path, line_no, segment)
        if issue:
            issues.append(issue)
    return issues


def audit_page(path: Path) -> list[Issue]:
    issues: list[Issue] = []
    in_fence = False
    in_display_math = False
    fence_marker = ""
    display_math_start = 0
    display_math_lines: list[str] = []

    for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith(("```", "~~~")):
            marker = stripped[:3]
            if in_fence and marker == fence_marker:
                in_fence = False
                fence_marker = ""
            elif not in_fence:
                in_fence = True
                fence_marker = marker
            continue
        if in_fence:
            if LATEX_COMMAND_RE.search(line):
                issues.append(Issue(path, line_no, "latex-in-code-fence", stripped))
            continue

        for match in CODE_SPAN_RE.finditer(line):
            content = match.group(1)
            if LATEX_COMMAND_RE.search(content):
                issues.append(Issue(path, line_no, "latex-in-code-span", content))

        issues.extend(inline_math_issues(path, line_no, line))

        scan_line = CODE_SPAN_RE.sub("", line)
        if "\\[" in scan_line:
            before, _, rest = scan_line.partition("\\[")
            if LATEX_COMMAND_RE.search(strip_inline_math_and_code(before)):
                issues.append(Issue(path, line_no, "naked-latex-command", stripped))
            if "\\]" in rest:
                segment, _, after = rest.partition("\\]")
                issue = check_math_segment(path, line_no, segment)
                if issue:
                    issues.append(issue)
                scan_line = after
                in_display_math = False
            else:
                in_display_math = True
                display_math_start = line_no
                display_math_lines = [rest]
                continue

        if in_display_math:
            if "\\]" in scan_line:
                segment, _, after = scan_line.partition("\\]")
                display_math_lines.append(segment)
                issue = check_math_segment(path, display_math_start, "\n".join(display_math_lines))
                if issue:
                    issues.append(issue)
                scan_line = after
                in_display_math = False
            else:
                display_math_lines.append(scan_line)
                continue

        if stripped == "$$":
            in_display_math = not in_display_math
            if in_display_math:
                display_math_start = line_no
                display_math_lines = []
            else:
                issue = check_math_segment(path, display_math_start, "\n".join(display_math_lines))
                if issue:
                    issues.append(issue)
            continue

        if LATEX_COMMAND_RE.search(strip_inline_math_and_code(scan_line)):
            issues.append(Issue(path, line_no, "naked-latex-command", stripped))

    return issues


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fail-on-issues", action="store_true")
    parser.add_argument("--list", action="store_true", help="Print all detected issues.")
    args = parser.parse_args()

    issues = [issue for page in iter_markdown_pages() for issue in audit_page(page)]
    counts = Counter(issue.code for issue in issues)

    print(f"Scanned {len(iter_markdown_pages())} markdown pages under {DOCS_ROOT}/")
    print("Math rendering flags:")
    for code, count in counts.most_common():
        print(f"  {code}: {count}")

    if args.list and issues:
        print("\nFlagged locations:")
        for issue in issues:
            rel = issue.path.as_posix()
            print(f"- {rel}:{issue.line_no}: {issue.code}: {issue.detail}")

    if args.fail_on_issues and issues:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
