"""Sort paper deep-dive navigation entries by paper date."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any


FRONT_MATTER_RE = re.compile(r"\A---\s*\n(?P<body>.*?)\n---\s*\n", re.DOTALL)
PAPER_DATE_RE = re.compile(r"^paper_date:\s*['\"]?(?P<date>\d{4}(?:-\d{2})?)['\"]?\s*$", re.MULTILINE)


def _src_path(item: Any) -> str:
    file = getattr(item, "file", None)
    return getattr(file, "src_path", "") or ""


def _is_paper_deep_dive_page(item: Any) -> bool:
    src_path = _src_path(item)
    return src_path.startswith("paper-deep-dives/") and src_path != "paper-deep-dives/index.md"


@lru_cache(maxsize=None)
def _paper_date_from_file(path: str) -> tuple[int, int]:
    try:
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return (9999, 99)

    front = FRONT_MATTER_RE.match(text)
    if not front:
        return (9999, 99)

    match = PAPER_DATE_RE.search(front.group("body"))
    if not match:
        return (9999, 99)

    date = match.group("date")
    if "-" not in date:
        return (int(date), 99)

    year, month = date.split("-", 1)
    return (int(year), int(month))


def _paper_sort_key(item: Any, docs_dir: Path) -> tuple[int, int, str]:
    src_path = _src_path(item)
    year, month = _paper_date_from_file(str(docs_dir / src_path))
    title = getattr(item, "title", "") or src_path
    return (year, month, title)


def _sort_sections(items: list[Any], docs_dir: Path) -> None:
    for item in items:
        children = getattr(item, "children", None)
        if not children:
            continue

        child_list = list(children)
        _sort_sections(child_list, docs_dir)

        paper_children = [child for child in child_list if _is_paper_deep_dive_page(child)]
        if len(paper_children) < 2:
            item.children = child_list
            continue

        sorted_papers = iter(sorted(paper_children, key=lambda child: _paper_sort_key(child, docs_dir)))
        paper_ids = {id(child) for child in paper_children}
        item.children = [next(sorted_papers) if id(child) in paper_ids else child for child in child_list]


def on_nav(nav: Any, config: Any, files: Any) -> Any:
    docs_dir = Path(config["docs_dir"])
    _sort_sections(list(nav.items), docs_dir)
    return nav
