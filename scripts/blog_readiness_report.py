#!/usr/bin/env python3
"""Audit whether pages read like standalone knowledge-blog articles."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import re
from urllib.parse import urlparse

import yaml


DOCS_ROOT = Path("files")
EXCLUDED_PARTS = {"assets", "references", "roadmap"}
UTILITY_PAGES = {
    "training/image-sources.md",
}

URL_RE = re.compile(r"https?://[^\s)<>\"]+")
LOCAL_LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\(([^)#?]+)(?:#[^)]+)?\)")
IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)|<img\b", re.IGNORECASE)
SOURCE_RE = re.compile(
    r"(主要来源|相关阅读|参考|图源|图片来源|论文|官方文档|官方博客|项目页|"
    r"arXiv|OpenReview|GitHub|Source|source)"
)
QUESTION_RE = re.compile(
    r"(这篇|本文|本页|这一页|为什么|如何|怎么|是什么|解决什么|"
    r"要回答|读完|核心问题|一句话)"
)
class MkdocsLoader(yaml.SafeLoader):
    """YAML loader that tolerates MkDocs python/name tags."""


MkdocsLoader.add_multi_constructor(
    "tag:yaml.org,2002:python/name:",
    lambda loader, suffix, node: f"!!python/name:{suffix}",
)


@dataclass(frozen=True)
class PageAudit:
    path: Path
    kind: str
    urls: int
    images: int
    local_links: int
    flags: tuple[str, ...]


def iter_pages() -> list[Path]:
    pages: list[Path] = []
    for path in DOCS_ROOT.rglob("*.md"):
        rel_parts = set(path.relative_to(DOCS_ROOT).parts)
        if path.name.startswith("._"):
            continue
        if rel_parts.intersection(EXCLUDED_PARTS):
            continue
        pages.append(path)
    return sorted(pages)


def nav_targets() -> set[str]:
    config_path = Path("mkdocs.yml")
    if not config_path.exists():
        return set()
    config = yaml.load(config_path.read_text(encoding="utf-8"), Loader=MkdocsLoader)
    targets: set[str] = set()

    def walk(item: object) -> None:
        if isinstance(item, list):
            for child in item:
                walk(child)
        elif isinstance(item, dict):
            for child in item.values():
                walk(child)
        elif isinstance(item, str):
            targets.add(item)

    walk(config.get("nav", []))
    return targets


def clean_link(raw: str) -> str:
    return raw.strip().strip("<>").split(" ", 1)[0]


def local_link_count(text: str) -> int:
    count = 0
    for match in LOCAL_LINK_RE.finditer(text):
        target = clean_link(match.group(1))
        if not target or target.startswith("#"):
            continue
        parsed = urlparse(target)
        if parsed.scheme:
            continue
        count += 1
    return count


def audit_page(path: Path, nav: set[str]) -> PageAudit:
    text = path.read_text(encoding="utf-8", errors="ignore")
    rel = path.relative_to(DOCS_ROOT).as_posix()
    first_block = "\n".join(text.splitlines()[:28])
    flags: list[str] = []
    if rel in UTILITY_PAGES:
        kind = "utility"
    elif path.name == "index.md":
        kind = "entry"
    else:
        kind = "topic"
    urls = len(set(URL_RE.findall(text)))
    images = len(IMAGE_RE.findall(text))
    links = local_link_count(text)
    source_signal = bool(SOURCE_RE.search(text))

    if kind != "entry":
        if not QUESTION_RE.search(first_block):
            flags.append("weak-opening-question")
        if not source_signal:
            flags.append("missing-source-signal")
        if links < 2:
            flags.append("missing-next-reading-links")

    if urls == 0 and kind == "topic":
        flags.append("no-external-url")
    if images == 0 and kind == "topic":
        flags.append("no-image")

    return PageAudit(path=path, kind=kind, urls=urls, images=images, local_links=links, flags=tuple(flags))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fail-on-high", action="store_true", help="Return non-zero for missing source signals.")
    parser.add_argument("--list-pages", action="store_true", help="Print every flagged page.")
    args = parser.parse_args()

    nav = nav_targets()
    audits = [audit_page(path, nav) for path in iter_pages()]
    flag_counts: Counter[str] = Counter(flag for audit in audits for flag in audit.flags)
    by_dir: dict[str, Counter[str]] = defaultdict(Counter)
    for audit in audits:
        top = audit.path.relative_to(DOCS_ROOT).parts[0]
        for flag in audit.flags:
            by_dir[top][flag] += 1

    print(f"Scanned {len(audits)} standalone content pages under {DOCS_ROOT}/")
    print(f"Navigation targets: {len(nav)}")
    print("Blog-readiness flags:")
    for flag, count in flag_counts.most_common():
        print(f"  {flag}: {count}")
    print("\nFlags by section:")
    for section in sorted(by_dir):
        summary = ", ".join(f"{flag}={count}" for flag, count in by_dir[section].most_common())
        print(f"  {section}: {summary}")

    if args.list_pages:
        print("\nFlagged pages:")
        for audit in audits:
            if audit.flags:
                rel = audit.path.as_posix()
                print(
                    f"- {rel}: kind={audit.kind}, urls={audit.urls}, images={audit.images}, "
                    f"local_links={audit.local_links}, flags={','.join(audit.flags)}"
                )

    if args.fail_on_high and flag_counts["missing-source-signal"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
