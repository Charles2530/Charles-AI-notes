#!/usr/bin/env python3
"""Remove old prompt-style scaffolding from markdown articles.

The site used to contain generated reading-position boxes and generic symbol
tables. They make pages feel like maintenance notes rather than standalone
articles, so this script removes those mechanical blocks while leaving normal
warnings, examples, figures, and prose intact.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re


DOCS_ROOT = Path("files")

COMMENT_RE = re.compile(r"^\s*<!-- \[补充(?:[:：][^\]]+)?\] -->\s*$")
ADMONITION_START_RE = re.compile(r"^(\s*)(!!!|\?\?\?)\s+\w+(?:\s+['\"]([^'\"]+)['\"])?")
PROMPT_TITLES = ("合并提示", "读法定位", "下一站")
GENERIC_SYMBOL_PHRASES = (
    "公式中的变量或中间量",
    "按上下文定义",
    "return、组大小或缩放系数",
    "value、速度场或向量量",
)


def is_table_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|")


def table_block_end(lines: list[str], start: int) -> int:
    end = start
    while end < len(lines) and (is_table_line(lines[end]) or not lines[end].strip()):
        if not lines[end].strip():
            lookahead = end + 1
            if lookahead >= len(lines) or not is_table_line(lines[lookahead]):
                break
        end += 1
    return end


def is_reader_table(block: list[str]) -> bool:
    joined = "\n".join(block[:3])
    return "| 读者 |" in joined and "建议读法" in joined and "工程落点" in joined


def is_generic_symbol_table(block: list[str]) -> bool:
    joined = "\n".join(block)
    if not any(phrase in joined for phrase in GENERIC_SYMBOL_PHRASES):
        return False
    return "| 符号 |" in joined or "| 公式整体 |" in joined or "取值范围" in joined


def admonition_block_end(lines: list[str], start: int) -> int:
    base_indent = len(lines[start]) - len(lines[start].lstrip(" "))
    end = start + 1
    while end < len(lines):
        line = lines[end]
        stripped = line.strip()
        if not stripped:
            end += 1
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent <= base_indent:
            break
        end += 1
    return end


def should_drop_admonition(line: str) -> bool:
    match = ADMONITION_START_RE.match(line)
    if not match:
        return False
    title = match.group(3) or ""
    return any(token in line or token in title for token in PROMPT_TITLES)


def cleanup_text(text: str) -> tuple[str, dict[str, int]]:
    lines = text.splitlines()
    output: list[str] = []
    stats = {
        "comments": 0,
        "reader_tables": 0,
        "generic_symbol_tables": 0,
        "prompt_admonitions": 0,
        "positioning_lines": 0,
    }

    i = 0
    while i < len(lines):
        line = lines[i]

        if COMMENT_RE.match(line):
            stats["comments"] += 1
            if "符号说明" in line:
                scan = i + 1
                while scan < len(lines) and not lines[scan].strip():
                    scan += 1
                if scan < len(lines) and is_table_line(lines[scan]):
                    end = table_block_end(lines, scan)
                    stats["generic_symbol_tables"] += 1
                    i = end
                    continue
            i += 1
            continue

        if is_table_line(line):
            end = table_block_end(lines, i)
            block = lines[i:end]
            if is_reader_table(block):
                stats["reader_tables"] += 1
                i = end
                continue
            if is_generic_symbol_table(block):
                stats["generic_symbol_tables"] += 1
                i = end
                continue

        if should_drop_admonition(line):
            end = admonition_block_end(lines, i)
            stats["prompt_admonitions"] += 1
            i = end
            continue

        stripped = line.strip()
        if stripped.startswith(("读法定位：", "读法定位:", "下一站：", "下一站:")):
            stats["positioning_lines"] += 1
            i += 1
            continue

        output.append(line)
        i += 1

    cleaned = "\n".join(output).rstrip() + "\n"
    cleaned = re.sub(r"\n{4,}", "\n\n\n", cleaned)
    return cleaned, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Only report changed files.")
    parser.add_argument("--root", default=str(DOCS_ROOT), help="Markdown root to scan.")
    args = parser.parse_args()

    root = Path(args.root)
    changed: list[tuple[Path, dict[str, int]]] = []
    totals = {
        "comments": 0,
        "reader_tables": 0,
        "generic_symbol_tables": 0,
        "prompt_admonitions": 0,
        "positioning_lines": 0,
    }

    for path in sorted(root.rglob("*.md")):
        before = path.read_text(encoding="utf-8")
        after, stats = cleanup_text(before)
        if before == after:
            continue
        changed.append((path, stats))
        for key, value in stats.items():
            totals[key] += value
        if not args.dry_run:
            path.write_text(after, encoding="utf-8")

    mode = "would change" if args.dry_run else "changed"
    print(f"{mode} {len(changed)} markdown files")
    for key, value in totals.items():
        print(f"{key}: {value}")

    if args.dry_run:
        for path, stats in changed[:50]:
            nonzero = ", ".join(f"{k}={v}" for k, v in stats.items() if v)
            print(f"- {path}: {nonzero}")
        if len(changed) > 50:
            print(f"... {len(changed) - 50} more")


if __name__ == "__main__":
    main()
