#!/usr/bin/env python3
"""Find long exact text overlaps with the cached external-source corpus.

The report is intentionally conservative. It only reports repeated long n-grams
found in the local Firecrawl cache, so a clean result is not proof of originality
and a hit still needs a human license and context review.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path


DEFAULT_PAGES = Path("files")
DEFAULT_SOURCES = Path(".firecrawl/blog-source-research-1109")
SKIP_PARTS = {"assets", "references", "roadmap"}
URL_RE = re.compile(r"https?://\S+")
CJK_RE = re.compile(r"[\u3400-\u9fff]")
WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_+-]{2,}")
COMMON_WORDS = {
    "a", "an", "and", "are", "as", "at", "by", "can", "for", "from",
    "in", "is", "it", "model", "of", "on", "or", "that", "the", "to",
    "use", "we", "with",
}


def clean(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = URL_RE.sub(" ", text)
    text = re.sub(r"!\[[^]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"\[[^]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def cjk_ngrams(text: str, size: int = 10) -> set[str]:
    chars = [char for char in clean(text) if CJK_RE.match(char)]
    return {"".join(chars[i : i + size]) for i in range(len(chars) - size + 1)}


def word_ngrams(text: str, size: int = 8) -> set[str]:
    words = [word.lower() for word in WORD_RE.findall(clean(text))]
    return {
        " ".join(window)
        for i in range(len(words) - size + 1)
        if COMMON_WORDS.intersection(window := words[i : i + size])
    }


def source_index(source_root: Path) -> dict[str, set[str]]:
    index: dict[str, set[str]] = defaultdict(set)
    for path in sorted(source_root.glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        # Cached scrapes usually start with title, author, navigation, and
        # license boilerplate. Those names are expected to appear in paper
        # metadata and are not useful evidence of copied prose.
        text = text[1200:]
        for gram in cjk_ngrams(text) | word_ngrams(text):
            index[gram].add(path.name)
    return index


def page_paths(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*.md")
        if not (set(path.relative_to(root).parts) & SKIP_PARTS)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=Path, default=DEFAULT_PAGES)
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--min-matches", type=int, default=3)
    parser.add_argument("--limit", type=int, default=80)
    args = parser.parse_args()

    if not args.sources.exists():
        raise SystemExit(f"source cache not found: {args.sources}")

    index = source_index(args.sources)
    source_files = {name for names in index.values() for name in names}
    findings: list[tuple[int, Path, list[tuple[str, list[str]]]]] = []
    for page in page_paths(args.pages):
        text = page.read_text(encoding="utf-8", errors="ignore")
        grams = cjk_ngrams(text) | word_ngrams(text)
        matches: dict[str, list[str]] = defaultdict(list)
        for gram in grams:
            # Ignore fragments common to several unrelated sources.
            names = index.get(gram, set())
            if len(names) == 1:
                matches[next(iter(names))].append(gram)
        total = sum(len(items) for items in matches.values())
        if total >= args.min_matches:
            ranked = sorted(matches.items(), key=lambda item: -len(item[1]))
            findings.append((total, page, [(name, grams[:5]) for name, grams in ranked[:3]]))

    findings.sort(key=lambda item: (-item[0], item[1].as_posix()))
    print(f"Indexed {len(source_files)} cached source text files under {args.sources}/")
    print(f"Scanned {len(page_paths(args.pages))} markdown pages under {args.pages}/")
    print("Exact-overlap candidates (manual review required):")
    if not findings:
        print("  none")
    for total, page, sources in findings[: max(0, args.limit)]:
        print(f"- {page}: {total} unique long n-gram matches")
        for name, grams in sources:
            print(f"  {name}: {' / '.join(grams)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
