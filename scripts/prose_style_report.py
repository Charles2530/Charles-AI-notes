#!/usr/bin/env python3
"""Report likely template-heavy Chinese prose for manual review.

This is a triage report, not a plagiarism detector and not a grammar checker.
It deliberately reports candidates instead of rewriting pages automatically.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


DEFAULT_ROOT = Path("files")
SKIP_PARTS = {"assets", "references", "roadmap"}

# These are common in technical writing as well. A hit is only a review signal.
PATTERNS: dict[str, re.Pattern[str]] = {
    "preview": re.compile(r"本文将|本页将|这篇文章将|下面将|接下来将"),
    "roadmap": re.compile(r"首先|其次|最后|第一步|第二步|第三步|第一层|第二层|第三层"),
    "insight_marker": re.compile(r"值得注意的是|需要指出的是|关键在于|这意味着|从这个角度"),
    "generic_transition": re.compile(r"在实际应用中|从而|进一步地|总的来说|综上所述|不难发现"),
    "abstract_opening": re.compile(r"随着.*发展|在.*背景下|近年来|随着人工智能|本质上|归根结底"),
}

QUESTION_OR_EVENT = re.compile(
    r"\d|[0-9]|为什么|如何|怎样|假设|设 |例如|报错|失败|trace|kernel|loss|shape|\[|`"
)
HEADING_RE = re.compile(r"^#{2,4}\s+(.+)$", re.MULTILINE)


@dataclass(frozen=True)
class Finding:
    path: Path
    score: int
    counts: dict[str, int]
    reason: str


def iter_pages(root: Path) -> list[Path]:
    pages: list[Path] = []
    for path in root.rglob("*.md"):
        if set(path.relative_to(root).parts) & SKIP_PARTS:
            continue
        pages.append(path)
    return sorted(pages)


def opening(text: str) -> str:
    # Skip frontmatter, headings, and HTML blocks when assessing the first prose.
    body = re.sub(r"\A---\n.*?\n---\n", "", text, flags=re.DOTALL)
    lines: list[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "!", "<", "```", "|")):
            continue
        lines.append(stripped)
        if len("".join(lines)) >= 500:
            break
    return "\n".join(lines)


def audit(path: Path) -> Finding | None:
    text = path.read_text(encoding="utf-8", errors="ignore")
    # Count a bounded prose window. Long paper notes naturally contain more
    # numbered steps; density in the first part is a better triage signal.
    prose = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    prose = "\n".join(
        line for line in prose.splitlines() if not line.lstrip().startswith("|")
    )
    prose = prose[:12000]
    counts = {name: len(pattern.findall(prose)) for name, pattern in PATTERNS.items()}
    first = opening(text)
    opening_score = sum(1 for name, pattern in PATTERNS.items() if pattern.search(first))
    score = sum(counts.values()) + opening_score

    # A page with concrete opening material can tolerate more connective language.
    if QUESTION_OR_EVENT.search(first):
        score -= 1
    if score < 4:
        return None

    reasons: list[str] = []
    if opening_score >= 1 and not QUESTION_OR_EVENT.search(first):
        reasons.append("开头抽象或像提纲")
    if counts["roadmap"] >= 3:
        reasons.append("步骤路标较密")
    if counts["insight_marker"] >= 2:
        reasons.append("洞察路标重复")
    if counts["preview"] >= 1:
        reasons.append("存在文章预告句")
    if not reasons:
        reasons.append("连接句密度偏高")
    return Finding(path, score, counts, "、".join(reasons))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--limit", type=int, default=80)
    args = parser.parse_args()

    findings = [finding for path in iter_pages(args.root) if (finding := audit(path))]
    findings.sort(key=lambda item: (-item.score, item.path.as_posix()))
    totals: Counter[str] = Counter()
    for finding in findings:
        totals.update({name: count for name, count in finding.counts.items() if count})

    print(f"Scanned {len(iter_pages(args.root))} markdown pages under {args.root}/")
    print(f"Manual-review candidates: {len(findings)}")
    print("Candidate pattern totals:")
    for name, count in totals.most_common():
        print(f"  {name}: {count}")
    print("\nTop candidates:")
    for finding in findings[: max(0, args.limit)]:
        active = ", ".join(
            f"{name}={count}" for name, count in finding.counts.items() if count
        )
        print(f"- {finding.path}: score={finding.score}; {finding.reason}; {active}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
