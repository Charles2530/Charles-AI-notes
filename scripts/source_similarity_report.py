#!/usr/bin/env python3
"""Find approximate prose overlap with the cached source corpus.

This is a manual-review triage tool. It uses normalized CJK character shingles
and English word shingles, so it can catch light edits in the same language.
It does not detect translation reliably and it is not a plagiarism verdict.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
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
    text = re.sub(r"^\s*#{1,6}\s+", "", text, flags=re.MULTILINE)
    return re.sub(r"\s+", " ", text).strip()


def shingles(text: str, cjk_size: int = 8, word_size: int = 6) -> set[str]:
    text = clean(text)
    cjk = [char for char in text if CJK_RE.match(char)]
    cjk_grams = {"c:" + "".join(cjk[i : i + cjk_size])
                 for i in range(len(cjk) - cjk_size + 1)}
    words = [word.lower() for word in WORD_RE.findall(text)]
    word_grams = {
        "w:" + " ".join(window)
        for i in range(len(words) - word_size + 1)
        if COMMON_WORDS.intersection(window := words[i : i + word_size])
    }
    return cjk_grams | word_grams


def page_paths(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*.md")
        if not (set(path.relative_to(root).parts) & SKIP_PARTS)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=Path, default=DEFAULT_PAGES)
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--min-shared", type=int, default=12)
    parser.add_argument("--min-page-coverage", type=float, default=0.08)
    parser.add_argument("--limit", type=int, default=80)
    args = parser.parse_args()

    source_sets: dict[str, set[str]] = {}
    for path in sorted(args.sources.glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        source_sets[path.name] = shingles(text[1200:])
    if not source_sets:
        raise SystemExit(f"source cache not found: {args.sources}")

    inverse: dict[str, set[str]] = defaultdict(set)
    for name, values in source_sets.items():
        for value in values:
            inverse[value].add(name)

    findings: list[tuple[float, int, float, Path, str]] = []
    for page in page_paths(args.pages):
        page_values = shingles(page.read_text(encoding="utf-8", errors="ignore"))
        if not page_values:
            continue
        candidates = Counter(
            source for value in page_values for source in inverse.get(value, ())
        )
        for source, shared in candidates.items():
            if shared < args.min_shared:
                continue
            source_values = source_sets[source]
            union = len(page_values | source_values)
            jaccard = shared / union if union else 0.0
            coverage = shared / len(page_values)
            if coverage >= args.min_page_coverage:
                findings.append((coverage, shared, jaccard, page, source))

    findings.sort(key=lambda item: (-item[0], -item[1], item[3].as_posix()))
    print(f"Indexed {len(source_sets)} cached source text files under {args.sources}/")
    print(f"Scanned {len(page_paths(args.pages))} markdown pages under {args.pages}/")
    print("Approximate-overlap candidates (manual review required):")
    if not findings:
        print("  none")
    for coverage, shared, jaccard, page, source in findings[: max(0, args.limit)]:
        print(
            f"- {page}: page_coverage={coverage:.3f}, shared={shared}, "
            f"jaccard={jaccard:.3f}, source={source}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
