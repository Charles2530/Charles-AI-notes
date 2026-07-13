"""Compact generated site assets that otherwise delay every page load."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def compact_search_index(payload: dict[str, Any]) -> dict[str, Any]:
    fields = payload.get("config", {}).get("fields", {})
    if isinstance(fields, dict):
        fields.pop("text", None)

    for document in payload.get("docs", []):
        if isinstance(document, dict):
            document.pop("text", None)
    return payload


def on_post_build(config: Any) -> None:
    index_path = Path(config["site_dir"]) / "search" / "search_index.json"
    if not index_path.is_file():
        return

    payload = json.loads(index_path.read_text(encoding="utf-8"))
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    index_path.with_name("search_index_full.json").write_text(serialized, encoding="utf-8")

    compacted = compact_search_index(payload)
    index_path.write_text(
        json.dumps(compacted, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
