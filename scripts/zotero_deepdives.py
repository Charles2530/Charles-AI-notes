#!/usr/bin/env python3
"""Audit and populate Zotero for paper-deep-dives arXiv coverage.

The script is deliberately conservative:
- `audit` only reads project files and a temporary copy of the Zotero DB.
- `fetch` downloads arXiv metadata/PDFs into the repo tmp directory.
- `apply` writes to Zotero and should be run only after Zotero is closed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
import os
import random
import re
import shutil
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT_COLLECTION_KEY = "J4R9J4Y8"
UNREAD_TAG = "/unread"
READ_STATE_TAGS = {"√已读", "-正在阅读"}
WORK_DIR = Path("tmp/zotero-deepdives")
PDF_DIR = WORK_DIR / "pdfs"
META_PATH = WORK_DIR / "arxiv-meta.json"
AUDIT_JSON = WORK_DIR / "audit.json"
AUDIT_MD = WORK_DIR / "audit.md"

ZOTERO_DB = Path.home() / "Zotero" / "zotero.sqlite"
ZOTERO_STORAGE = Path.home() / "Zotero" / "storage"
ZOTERO_ALLOWED_KEY_CHARS = "23456789ABCDEFGHIJKLMNPQRSTUVWXYZ"
ZOTERO_OBJECT_KEY_RE = re.compile(rf"^[{ZOTERO_ALLOWED_KEY_CHARS}]{{8}}$")
SUMMARY_NOTE_MARKER = "paper summary"
SUMMARY_NOTE_TITLE = "paper summary"
SUMMARY_NOTE_LEGACY_MARKERS = ("Codex paper summary",)

ITEM_TYPE = {
    "annotation": 1,
    "attachment": 3,
    "conferencePaper": 11,
    "journalArticle": 22,
    "note": 28,
    "preprint": 31,
    "webpage": 40,
}

FIELD = {
    "title": 1,
    "abstractNote": 2,
    "date": 6,
    "DOI": 8,
    "citationKey": 9,
    "url": 10,
    "accessDate": 11,
    "archive": 12,
    "archiveLocation": 13,
    "shortTitle": 14,
    "libraryCatalog": 16,
    "extra": 19,
    "repository": 69,
    "archiveID": 107,
}

COLORS = {
    "claim": "#ffd400",
    "position": "#2ea8e5",
    "mechanism": "#5fb236",
    "evidence": "#f19837",
    "boundary": "#ff6666",
    "context": "#a28ae5",
    "note": "#e56eee",
    "gray": "#aaaaaa",
}


@dataclass
class PaperRef:
    arxiv_id: str
    pages: set[str] = field(default_factory=set)
    leaves: set[tuple[str, ...]] = field(default_factory=set)


@dataclass
class ArxivMeta:
    arxiv_id: str
    title: str
    abstract: str
    authors: list[str]
    published: str
    updated: str
    categories: str
    pdf_url: str
    abs_url: str


def now_sql() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_work_dirs() -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)


def strip_version(arxiv_id: str) -> str:
    return re.sub(r"v\d+$", "", arxiv_id)


def parse_paper_deepdives(root: Path) -> dict[str, PaperRef]:
    refs: dict[str, PaperRef] = {}
    base = root / "files" / "paper-deep-dives"
    patterns = [
        re.compile(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})(?:v\d+)?", re.I),
        re.compile(r"arXiv[：:]\s*\[?([0-9]{4}\.[0-9]{4,5})(?:v\d+)?", re.I),
    ]
    for path in sorted(base.rglob("*.md")):
        if path.name == "index.md":
            continue
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(base).as_posix()
        ids: set[str] = set()
        for pattern in patterns:
            ids.update(strip_version(x) for x in pattern.findall(text))
        if not ids:
            continue
        leaves = target_leaves(rel)
        for arxiv_id in ids:
            ref = refs.setdefault(arxiv_id, PaperRef(arxiv_id))
            ref.pages.add(rel)
            ref.leaves.update(leaves)
    return refs


def target_leaves(rel: str) -> set[tuple[str, ...]]:
    name = Path(rel).stem
    if rel.startswith("diffusion/"):
        if name in {"score-sde", "dpm-solver-plus-plus"}:
            return {("扩散模型", "连续时间与 ODE 采样")}
        if name in {"dmd", "dmd2", "phased-dmd"}:
            return {("扩散模型", "分布匹配与少步蒸馏")}
        return {("扩散模型", "视频生成")}

    if rel.startswith("foundations/"):
        if name in {"zero", "megatron-lm", "magiattention", "gpu-utilization-misleading"}:
            return {("高效训练", "分布式训练与系统效率")}
        return {("高效训练", "优化器与低比特训练")}

    if rel.startswith("inference/"):
        if name in {"eagle", "eagle-2", "eagle-3"}:
            return {("高效推理", "投机解码")}
        if name in {"vllm-pagedattention", "kvslimmer"}:
            return {("高效推理", "服务系统与 KV Cache")}
        if name == "low-bit-llm-survey":
            return {("高效推理", "低比特推理与系统综述")}
        return {("高效推理", "实时视觉推理")}

    if rel.startswith("technical-reports/"):
        if name in {"qwen35-omni", "kimi-k2", "incoder-32b"}:
            return {("技术报告", "全模态与 Agent 技术报告")}
        return {("技术报告", "开放大模型技术报告")}

    if rel.startswith("world-models/"):
        if name == "ringattention":
            return {("高效训练", "分布式训练与系统效率")}
        if name in {"planet", "dreamer", "dreamerv2", "dreamerv3"}:
            return {("WM", "RSSM 与 Dreamer")}
        if name in {"jepa", "h-jepa", "v-jepa"}:
            return {("WM", "JEPA 与潜变量预测")}
        if name in {"world-models-embodied-ai-survey", "world-model-for-robot-learning-survey"}:
            return {("WM", "具身世界模型综述")}
        if name in {"bagel", "emu3-5", "beyond-language-modeling"}:
            return {("WM", "多模态世界模型")}
        return {("WM", "生成式视频世界模型")}

    if rel.startswith("embodied-ai/"):
        if name.startswith("vggt") or name in {"d4rt", "depth-anything", "depth-anything-v2", "depth-anything-3", "mapanything"}:
            return {("具身智能", "3D 视觉与几何基础")}
        if name in {"video-prediction-policy", "fast-wam", "dreamzero", "motus"}:
            return {("VLA", "世界模型辅助策略")}
        if name == "vodp":
            return {("VLA", "扩散策略与闭环控制")}
        return {("VLA", "Generative VLA 与开放世界策略")}

    return {("具身智能", "待整理")}


def copy_db_snapshot() -> Path:
    ensure_work_dirs()
    snap = WORK_DIR / "zotero-snapshot.sqlite"
    shutil.copy2(ZOTERO_DB, snap)
    return snap


def connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def db_collections(conn: sqlite3.Connection) -> dict[int, dict[str, Any]]:
    rows = conn.execute(
        "select collectionID, key, collectionName, parentCollectionID from collections"
    ).fetchall()
    return {r["collectionID"]: dict(r) for r in rows}


def descendants(collections: dict[int, dict[str, Any]], root_id: int) -> set[int]:
    children: dict[int | None, list[int]] = {}
    for cid, row in collections.items():
        children.setdefault(row["parentCollectionID"], []).append(cid)
    out: set[int] = set()
    stack = [root_id]
    while stack:
        cid = stack.pop()
        out.add(cid)
        stack.extend(children.get(cid, []))
    return out


def collection_path(collections: dict[int, dict[str, Any]], cid: int) -> tuple[str, ...]:
    parts: list[str] = []
    cur = cid
    while cur:
        row = collections[cur]
        parts.append(row["collectionName"])
        cur = row["parentCollectionID"]
    return tuple(reversed(parts))


def children_map(collections: dict[int, dict[str, Any]]) -> dict[int, list[int]]:
    children: dict[int, list[int]] = {}
    for cid, row in collections.items():
        parent = row["parentCollectionID"]
        if parent is not None:
            children.setdefault(parent, []).append(cid)
    return children


def item_field_values(conn: sqlite3.Connection) -> dict[int, list[str]]:
    rows = conn.execute(
        """
        select d.itemID, v.value
        from itemData d join itemDataValues v on d.valueID = v.valueID
        """
    ).fetchall()
    values: dict[int, list[str]] = {}
    for row in rows:
        values.setdefault(row["itemID"], []).append(row["value"])
    return values


def arxiv_index(conn: sqlite3.Connection) -> dict[str, list[int]]:
    index: dict[str, list[int]] = {}
    child_items = set()
    for row in conn.execute("select itemID from itemAttachments where parentItemID is not null"):
        child_items.add(row["itemID"])
    for row in conn.execute("select itemID from itemNotes where parentItemID is not null"):
        child_items.add(row["itemID"])
    for item_id, values in item_field_values(conn).items():
        if item_id in child_items:
            continue
        joined = "\n".join(values)
        for arxiv_id in sorted(set(re.findall(r"(?i)(?:arxiv:|arxiv\.org/(?:abs|pdf)/|10\.48550/arxiv\.)([0-9]{4}\.[0-9]{4,5})", joined))):
            index.setdefault(arxiv_id, []).append(item_id)
    return index


def item_titles(conn: sqlite3.Connection) -> dict[int, str]:
    rows = conn.execute(
        """
        select d.itemID, v.value
        from itemData d join itemDataValues v on d.valueID = v.valueID
        where d.fieldID = ?
        """,
        (FIELD["title"],),
    ).fetchall()
    return {r["itemID"]: r["value"] for r in rows}


def item_collections(conn: sqlite3.Connection) -> dict[int, set[int]]:
    out: dict[int, set[int]] = {}
    for row in conn.execute("select itemID, collectionID from collectionItems"):
        out.setdefault(row["itemID"], set()).add(row["collectionID"])
    return out


def item_tags(conn: sqlite3.Connection) -> dict[int, set[str]]:
    out: dict[int, set[str]] = {}
    rows = conn.execute(
        """
        select it.itemID, t.name
        from itemTags it join tags t on it.tagID = t.tagID
        """
    ).fetchall()
    for row in rows:
        out.setdefault(row["itemID"], set()).add(row["name"])
    return out


def pdf_children(conn: sqlite3.Connection) -> dict[int, list[int]]:
    rows = conn.execute(
        """
        select parentItemID, itemID
        from itemAttachments
        where parentItemID is not null
          and contentType = 'application/pdf'
          and path like 'storage:%'
        """
    ).fetchall()
    out: dict[int, list[int]] = {}
    for row in rows:
        out.setdefault(row["parentItemID"], []).append(row["itemID"])
    return out


def annotation_counts(conn: sqlite3.Connection, *, codex_only: bool = False) -> dict[int, int]:
    where = "where a.comment like '%Codex deep-dive%'" if codex_only else ""
    rows = conn.execute(
        f"""
        select a.parentItemID as attachmentID, count(*) as n
        from itemAnnotations a
        {where}
        group by a.parentItemID
        """
    ).fetchall()
    return {r["attachmentID"]: r["n"] for r in rows}


def canonical_item(
    arxiv_id: str,
    items: list[int],
    pdfs: dict[int, list[int]],
    collections_by_item: dict[int, set[int]],
    root_desc: set[int],
) -> int | None:
    if not items:
        return None
    return sorted(
        items,
        key=lambda item_id: (
            -int(bool(collections_by_item.get(item_id, set()) & root_desc)),
            -int(bool(pdfs.get(item_id))),
            item_id,
        ),
    )[0]


def audit(root: Path) -> dict[str, Any]:
    refs = parse_paper_deepdives(root)
    snap = copy_db_snapshot()
    conn = connect(snap)
    collections = db_collections(conn)
    root_row = next((r for r in collections.values() if r["key"] == ROOT_COLLECTION_KEY), None)
    if not root_row:
        raise SystemExit(f"Root collection key not found: {ROOT_COLLECTION_KEY}")
    root_id = root_row["collectionID"]
    root_desc = descendants(collections, root_id)
    child_map = children_map(collections)
    leaf_paths = {
        collection_path(collections, cid)
        for cid in root_desc
        if cid != root_id and not child_map.get(cid)
    }
    arxiv_to_items = arxiv_index(conn)
    titles = item_titles(conn)
    collections_by_item = item_collections(conn)
    tags_by_item = item_tags(conn)
    pdfs = pdf_children(conn)
    ann_counts = annotation_counts(conn)
    codex_ann_counts = annotation_counts(conn, codex_only=True)

    records: list[dict[str, Any]] = []
    for arxiv_id in sorted(refs):
        ref = refs[arxiv_id]
        items = arxiv_to_items.get(arxiv_id, [])
        item_id = canonical_item(arxiv_id, items, pdfs, collections_by_item, root_desc)
        pdf_ids = pdfs.get(item_id, []) if item_id else []
        anns = sum(ann_counts.get(pid, 0) for pid in pdf_ids)
        codex_anns = sum(codex_ann_counts.get(pid, 0) for pid in pdf_ids)
        tags = sorted(tags_by_item.get(item_id, set())) if item_id else []
        current_paths = [
            "/".join(collection_path(collections, cid))
            for cid in sorted(collections_by_item.get(item_id, set()) & root_desc)
        ] if item_id else []
        wanted_paths = sorted("具身智能实验室/" + "/".join(path) for path in ref.leaves)
        records.append(
            {
                "arxiv_id": arxiv_id,
                "pages": sorted(ref.pages),
                "wanted_leaf_paths": wanted_paths,
                "item_id": item_id,
                "title": titles.get(item_id, "") if item_id else "",
                "duplicate_item_ids": items,
                "has_item": item_id is not None,
                "has_pdf": bool(pdf_ids),
                "annotation_count": anns,
                "codex_annotation_count": codex_anns,
                "tags": tags,
                "has_unread_or_read_state": bool(set(tags) & (READ_STATE_TAGS | {UNREAD_TAG})),
                "current_root_paths": sorted(current_paths),
                "missing_leaf_paths": [p for p in wanted_paths if p not in current_paths],
            }
        )

    non_leaf_with_items = []
    for cid in sorted(root_desc):
        if cid == root_id or not child_map.get(cid):
            continue
        count = conn.execute("select count(*) from collectionItems where collectionID = ?", (cid,)).fetchone()[0]
        if count:
            non_leaf_with_items.append(
                {"collection": "/".join(collection_path(collections, cid)), "item_count": count}
            )

    summary = {
        "paper_pages_with_arxiv": len({p for ref in refs.values() for p in ref.pages}),
        "unique_arxiv_ids": len(refs),
        "missing_items": sum(1 for r in records if not r["has_item"]),
        "missing_pdfs": sum(1 for r in records if not r["has_pdf"]),
        "needs_annotations": sum(1 for r in records if r["codex_annotation_count"] < 3),
        "needs_read_state_tag": sum(1 for r in records if not r["has_unread_or_read_state"]),
        "needs_collection_membership": sum(1 for r in records if r["missing_leaf_paths"]),
        "non_leaf_collections_with_items": non_leaf_with_items,
        "target_leaf_paths_missing": sorted(
            "具身智能实验室/" + "/".join(path)
            for path in {leaf for ref in refs.values() for leaf in ref.leaves}
            if ("具身智能实验室", *path) not in leaf_paths
        ),
    }
    out = {"summary": summary, "records": records}
    ensure_work_dirs()
    AUDIT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    write_audit_md(out)
    return out


def write_audit_md(out: dict[str, Any]) -> None:
    summary = out["summary"]
    lines = ["# Zotero Deep-Dives Audit", ""]
    for key, value in summary.items():
        if key == "non_leaf_collections_with_items":
            lines.append(f"- {key}: {len(value)}")
        elif key == "target_leaf_paths_missing":
            lines.append(f"- {key}: {len(value)}")
        else:
            lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## Missing Or Incomplete")
    lines.append("")
    lines.append("| arXiv | item | pdf | annotations | read tag | missing collections | pages |")
    lines.append("| --- | --- | --- | ---: | --- | --- | --- |")
    for r in out["records"]:
        if r["has_item"] and r["has_pdf"] and r["codex_annotation_count"] >= 3 and r["has_unread_or_read_state"] and not r["missing_leaf_paths"]:
            continue
        lines.append(
            "| {arxiv} | {item} | {pdf} | {ann} | {tag} | {cols} | {pages} |".format(
                arxiv=r["arxiv_id"],
                item="yes" if r["has_item"] else "no",
                pdf="yes" if r["has_pdf"] else "no",
                ann=f"{r['codex_annotation_count']}/{r['annotation_count']}",
                tag="yes" if r["has_unread_or_read_state"] else "no",
                cols="<br>".join(r["missing_leaf_paths"]) or "",
                pages="<br>".join(r["pages"]),
            )
        )
    lines.append("")
    if summary["target_leaf_paths_missing"]:
        lines.append("## Target Leaf Collections To Create")
        lines.append("")
        for p in summary["target_leaf_paths_missing"]:
            lines.append(f"- {p}")
        lines.append("")
    if summary["non_leaf_collections_with_items"]:
        lines.append("## Non-Leaf Collections With Items")
        lines.append("")
        for row in summary["non_leaf_collections_with_items"]:
            lines.append(f"- {row['collection']}: {row['item_count']}")
    AUDIT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_meta() -> dict[str, ArxivMeta]:
    if not META_PATH.exists():
        return {}
    raw = json.loads(META_PATH.read_text(encoding="utf-8"))
    return {k: ArxivMeta(**v) for k, v in raw.items()}


def save_meta(meta: dict[str, ArxivMeta]) -> None:
    ensure_work_dirs()
    META_PATH.write_text(
        json.dumps({k: v.__dict__ for k, v in sorted(meta.items())}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def fetch_url(url: str, *, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Codex Zotero deep-dives audit"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def download_pdf(url: str, dest: Path, *, timeout: int = 45) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "Codex Zotero deep-dives audit"})
    part = dest.with_suffix(dest.suffix + ".part")
    if part.exists():
        part.unlink()
    with urllib.request.urlopen(req, timeout=timeout) as resp, part.open("wb") as fh:
        first = resp.read(8192)
        if not first.startswith(b"%PDF"):
            raise RuntimeError("download did not return a PDF")
        fh.write(first)
        while True:
            chunk = resp.read(1024 * 256)
            if not chunk:
                break
            fh.write(chunk)
    part.replace(dest)


def fetch_metadata(ids: list[str]) -> dict[str, ArxivMeta]:
    meta = load_meta()
    missing = [x for x in ids if x not in meta]
    ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
    for i in range(0, len(missing), 80):
        batch = missing[i : i + 80]
        if not batch:
            continue
        url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
            {"id_list": ",".join(batch), "max_results": str(len(batch))}
        )
        data = fetch_url(url)
        root = ET.fromstring(data)
        for entry in root.findall("atom:entry", ns):
            raw_id = entry.findtext("atom:id", "", ns).rstrip("/").split("/")[-1]
            arxiv_id = strip_version(raw_id)
            title = " ".join(entry.findtext("atom:title", "", ns).split())
            abstract = " ".join(entry.findtext("atom:summary", "", ns).split())
            authors = [
                " ".join(a.findtext("atom:name", "", ns).split())
                for a in entry.findall("atom:author", ns)
                if a.findtext("atom:name", "", ns)
            ]
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            for link in entry.findall("atom:link", ns):
                if link.attrib.get("title") == "pdf":
                    pdf_url = link.attrib.get("href", pdf_url)
            cats = " ".join(c.attrib.get("term", "") for c in entry.findall("atom:category", ns)).strip()
            meta[arxiv_id] = ArxivMeta(
                arxiv_id=arxiv_id,
                title=title,
                abstract=abstract,
                authors=authors,
                published=entry.findtext("atom:published", "", ns),
                updated=entry.findtext("atom:updated", "", ns),
                categories=cats,
                pdf_url=pdf_url,
                abs_url=f"https://arxiv.org/abs/{arxiv_id}",
            )
        time.sleep(3)
    save_meta(meta)
    return meta


def safe_filename(text: str, limit: int = 90) -> str:
    text = re.sub(r"[\\/:*?\"<>|]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit].rstrip() or "paper"


def pdf_path(arxiv_id: str, meta: dict[str, ArxivMeta]) -> Path:
    title = safe_filename(meta.get(arxiv_id, ArxivMeta(arxiv_id, arxiv_id, "", [], "", "", "", "", "")).title)
    return PDF_DIR / f"{arxiv_id} - {title}.pdf"


def fetch(root: Path) -> dict[str, Any]:
    refs = parse_paper_deepdives(root)
    ids = sorted(refs)
    meta = fetch_metadata(ids)
    downloaded = []
    failed = []
    for arxiv_id in ids:
        path = pdf_path(arxiv_id, meta)
        if path.exists() and path.stat().st_size > 100_000:
            continue
        try:
            print(f"download {arxiv_id}", flush=True)
            download_pdf(meta[arxiv_id].pdf_url, path)
            downloaded.append(str(path))
            time.sleep(1)
        except Exception as exc:
            failed.append({"arxiv_id": arxiv_id, "error": str(exc)})
    result = {"metadata_count": len(meta), "downloaded": downloaded, "failed": failed}
    (WORK_DIR / "fetch.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def random_object_key(existing: set[str]) -> str:
    while True:
        key = "".join(random.choice(ZOTERO_ALLOWED_KEY_CHARS) for _ in range(8))
        if key not in existing:
            existing.add(key)
            return key


def rand_key(conn: sqlite3.Connection, table: str = "items") -> str:
    existing = {row[0] for row in conn.execute(f"select key from {table}")}
    return random_object_key(existing)


def is_valid_object_key(key: str | None) -> bool:
    return bool(key and ZOTERO_OBJECT_KEY_RE.fullmatch(key))


def get_or_create_collection(
    conn: sqlite3.Connection,
    collections: dict[int, dict[str, Any]],
    root_id: int,
    path: tuple[str, ...],
) -> int:
    parent = root_id
    for part in path:
        found = None
        for cid, row in collections.items():
            if row["parentCollectionID"] == parent and row["collectionName"] == part:
                found = cid
                break
        if found is None:
            key = rand_key(conn, "collections")
            now = now_sql()
            cur = conn.execute(
                """
                insert into collections
                (collectionName, parentCollectionID, clientDateModified, libraryID, key, version, synced)
                values (?, ?, ?, 1, ?, 0, 0)
                """,
                (part, parent, now, key),
            )
            found = cur.lastrowid
            collections[found] = {
                "collectionID": found,
                "key": key,
                "collectionName": part,
                "parentCollectionID": parent,
            }
        parent = found
    return parent


def value_id(conn: sqlite3.Connection, value: str) -> int:
    row = conn.execute("select valueID from itemDataValues where value = ?", (value,)).fetchone()
    if row:
        return row["valueID"]
    cur = conn.execute("insert into itemDataValues (value) values (?)", (value,))
    return cur.lastrowid


def set_field(conn: sqlite3.Connection, item_id: int, field_id: int, value: str) -> None:
    if not value:
        return
    vid = value_id(conn, value)
    conn.execute(
        "insert or replace into itemData (itemID, fieldID, valueID) values (?, ?, ?)",
        (item_id, field_id, vid),
    )


def split_author(name: str) -> tuple[str, str, int]:
    parts = name.strip().split()
    if len(parts) <= 1:
        return "", name.strip(), 1
    return " ".join(parts[:-1]), parts[-1], 0


def creator_id(conn: sqlite3.Connection, name: str) -> int:
    first, last, mode = split_author(name)
    row = conn.execute(
        "select creatorID from creators where firstName is ? and lastName = ? and fieldMode = ?",
        (first or None, last, mode),
    ).fetchone()
    if row:
        return row["creatorID"]
    cur = conn.execute(
        "insert into creators (firstName, lastName, fieldMode) values (?, ?, ?)",
        (first or None, last, mode),
    )
    return cur.lastrowid


def create_preprint(conn: sqlite3.Connection, meta: ArxivMeta) -> int:
    key = rand_key(conn)
    now = now_sql()
    cur = conn.execute(
        """
        insert into items
        (itemTypeID, dateAdded, dateModified, clientDateModified, libraryID, key, version, synced)
        values (?, ?, ?, ?, 1, ?, 0, 0)
        """,
        (ITEM_TYPE["preprint"], now, now, now, key),
    )
    item_id = cur.lastrowid
    year = meta.published[:4] if meta.published else ""
    set_field(conn, item_id, FIELD["title"], meta.title)
    set_field(conn, item_id, FIELD["abstractNote"], meta.abstract)
    set_field(conn, item_id, FIELD["date"], meta.published[:10] if meta.published else year)
    set_field(conn, item_id, FIELD["DOI"], f"10.48550/arXiv.{meta.arxiv_id}")
    set_field(conn, item_id, FIELD["url"], meta.abs_url.replace("https://", "http://"))
    set_field(conn, item_id, FIELD["accessDate"], now)
    set_field(conn, item_id, FIELD["archiveID"], f"arXiv:{meta.arxiv_id}")
    set_field(conn, item_id, FIELD["libraryCatalog"], "arXiv.org")
    set_field(conn, item_id, FIELD["repository"], "arXiv")
    set_field(conn, item_id, FIELD["extra"], f"arXiv:{meta.arxiv_id} [{meta.categories}]".strip())
    short = meta.title.split(":")[0]
    set_field(conn, item_id, FIELD["shortTitle"], short)
    for idx, name in enumerate(meta.authors[:30]):
        cid = creator_id(conn, name)
        conn.execute(
            """
            insert or ignore into itemCreators
            (itemID, creatorID, creatorTypeID, orderIndex)
            values (?, ?, 10, ?)
            """,
            (item_id, cid, idx),
        )
    return item_id


def storage_filename(meta: ArxivMeta) -> str:
    first = "arXiv"
    if meta.authors:
        first = split_author(meta.authors[0])[1]
    year = meta.published[:4] if meta.published else ""
    title = safe_filename(meta.title, 72)
    return f"{first} et al - {year} - {title}.pdf"


def attach_pdf(conn: sqlite3.Connection, item_id: int, meta: ArxivMeta, source_pdf: Path) -> int:
    key = rand_key(conn)
    now = now_sql()
    cur = conn.execute(
        """
        insert into items
        (itemTypeID, dateAdded, dateModified, clientDateModified, libraryID, key, version, synced)
        values (?, ?, ?, ?, 1, ?, 0, 0)
        """,
        (ITEM_TYPE["attachment"], now, now, now, key),
    )
    attachment_id = cur.lastrowid
    folder = ZOTERO_STORAGE / key
    folder.mkdir(parents=True, exist_ok=True)
    filename = storage_filename(meta)
    dest = folder / filename
    shutil.copy2(source_pdf, dest)
    conn.execute(
        """
        insert into itemAttachments
        (itemID, parentItemID, linkMode, contentType, charsetID, path, syncState)
        values (?, ?, 0, 'application/pdf', null, ?, 0)
        """,
        (attachment_id, item_id, f"storage:{filename}"),
    )
    set_field(conn, attachment_id, FIELD["title"], filename)
    return attachment_id


def tag_id(conn: sqlite3.Connection, name: str) -> int:
    row = conn.execute("select tagID from tags where name = ?", (name,)).fetchone()
    if row:
        return row["tagID"]
    cur = conn.execute("insert into tags (name) values (?)", (name,))
    return cur.lastrowid


def add_unread_if_needed(conn: sqlite3.Connection, item_id: int) -> None:
    rows = conn.execute(
        """
        select t.name from itemTags it join tags t on it.tagID = t.tagID
        where it.itemID = ?
        """,
        (item_id,),
    ).fetchall()
    tags = {r["name"] for r in rows}
    if tags & (READ_STATE_TAGS | {UNREAD_TAG}):
        return
    conn.execute(
        "insert or ignore into itemTags (itemID, tagID, type) values (?, ?, 1)",
        (item_id, tag_id(conn, UNREAD_TAG)),
    )


def clamp_text(text: str, limit: int = 320) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def first_matching_sentence(sentences: list[str], pattern: str, fallback: str) -> str:
    match = next((s for s in sentences if re.search(pattern, s, re.I)), "")
    return clamp_text(match or fallback or "摘要中未明确说明。")


def strip_markdown(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"^---.*?^---", " ", text, flags=re.S | re.M)
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)(?:\{[^}]*\})?", " ", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1（\2）", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"[*_#>]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def read_blog_sources(root: Path | None, pages: list[str]) -> list[tuple[str, Path | None, str]]:
    if root is None:
        return []
    base = root / "files" / "paper-deep-dives"
    sources = []
    for rel in pages:
        path = base / rel
        if not path.exists():
            continue
        sources.append((rel, path, path.read_text(encoding="utf-8")))
    return sources


def frontmatter_value(text: str, key: str) -> str:
    match = re.match(r"^---\s*\n(.*?)\n---", text, re.S)
    if not match:
        return ""
    for line in match.group(1).splitlines():
        if not line.startswith(f"{key}:"):
            continue
        value = line.split(":", 1)[1].strip()
        return value.strip("'\"")
    return ""


def arxiv_ids_in_text(text: str) -> list[str]:
    patterns = [
        re.compile(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})(?:v\d+)?", re.I),
        re.compile(r"arXiv[：:]\s*\[?([0-9]{4}\.[0-9]{4,5})(?:v\d+)?", re.I),
    ]
    ids: list[str] = []
    for pattern in patterns:
        for arxiv_id in pattern.findall(text):
            clean = strip_version(arxiv_id)
            if clean not in ids:
                ids.append(clean)
    return ids


def primary_blog_sources(
    sources: list[tuple[str, Path | None, str]],
    meta: ArxivMeta,
) -> list[tuple[str, Path | None, str]]:
    if len(sources) <= 1:
        return sources
    meta_title = normalized(meta.title)
    scored: list[tuple[int, int, tuple[str, Path | None, str]]] = []
    for index, source in enumerate(sources):
        rel, _, text = source
        score = 0
        page_title = normalized(frontmatter_value(text, "paper_title"))
        if page_title and (page_title == meta_title or page_title in meta_title or meta_title in page_title):
            score += 100
        ids = arxiv_ids_in_text(text)
        if ids and ids[0] == meta.arxiv_id:
            score += 50
        if Path(rel).stem.replace("-", " ") in meta_title:
            score += 5
        scored.append((score, -index, source))
    best = max(score for score, _, _ in scored)
    if best <= 0:
        return sources
    return [source for score, _, source in scored if score == best]


def compact_markdown_table(lines: list[str]) -> str:
    rows: list[list[str]] = []
    for line in lines:
        if not line.strip().startswith("|"):
            return ""
        cells = [strip_markdown(cell.strip()) for cell in line.strip().strip("|").split("|")]
        if not cells or all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
            continue
        rows.append(cells)
    if len(rows) < 2:
        return ""
    header, data = rows[0], rows[1:]
    if len(header) == 2 and re.search(r"item|setting|hyperparameter|配置|字段", header[0], re.I):
        return "；".join(f"{row[0]}：{row[1]}" for row in data if len(row) >= 2 and row[0] and row[1])
    if len(header) == 2 and re.search(r"detail|value|说明|值", header[1], re.I):
        return "；".join(f"{row[0]}：{row[1]}" for row in data if len(row) >= 2 and row[0] and row[1])
    return "；".join(" / ".join(cell for cell in row if cell) for row in rows if row)


def markdown_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    raw_blocks = re.split(r"\n\s*\n", text)
    for raw in raw_blocks:
        lines = []
        for line in raw.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("!["):
                continue
            if stripped.startswith("|"):
                lines.append(stripped)
                continue
            if re.match(r"^\d+\.\s+", stripped) or stripped.startswith("- "):
                lines.append(stripped)
                continue
            if stripped.startswith(("---", "paper_", "!!!")):
                continue
            lines.append(stripped)
        table = compact_markdown_table(lines)
        block = table or strip_markdown(" ".join(lines))
        if 20 <= len(block) <= 900:
            blocks.append(block)
    return blocks


def extract_numbered_items(text: str) -> list[str]:
    lines = text.splitlines()
    groups: list[list[str]] = []
    anchor_re = re.compile(r"核心贡献|核心思想|主要内容|阅读结论|总结|可以概括为三点|落成两个损失|核心方法")
    reject_re = re.compile(r"局限|风险|不能证明|限制|短板|缺点|失败")
    for i, line in enumerate(lines):
        if not anchor_re.search(line) or reject_re.search(line):
            continue
        window = lines[i + 1 : i + 18]
        items: list[str] = []
        for candidate in window:
            stripped = candidate.strip()
            if not stripped:
                continue
            if stripped.startswith("## "):
                break
            match = re.match(r"^([123])\.\s+(.+)", stripped)
            if not match:
                if items:
                    break
                continue
            item = strip_markdown(match.group(2))
            if item:
                items.append(item)
            if len(items) == 3:
                break
        if items:
            groups.append(items)
            if len(items) == 3:
                return items
    if groups:
        return max(groups, key=len)
    items = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        match = re.match(r"^([123])\.\s+(.+)", stripped)
        if not match:
            continue
        nearby = "\n".join(lines[max(0, i - 8) : i + 1])
        if reject_re.search(nearby):
            continue
        if not re.search(r"核心贡献|核心思想|主要内容|阅读结论|总结|贡献|可以概括为三点|三点", nearby):
            continue
        item = strip_markdown(match.group(2))
        if item:
            items.append(item)
        if len(items) == 3:
            return items
    return items


def best_blocks(
    blocks: list[str],
    pattern: str,
    *,
    fallback: list[str] | None = None,
    require_number: bool = False,
    reject_pattern: str | None = None,
    limit: int = 2,
) -> list[str]:
    scored: list[tuple[int, int, str]] = []
    for index, block in enumerate(blocks):
        if reject_pattern and re.search(reject_pattern, block, re.I):
            continue
        if require_number and not re.search(r"\d", block):
            continue
        matches = len(re.findall(pattern, block, re.I))
        if matches == 0:
            continue
        score = matches * 10
        score += min(len(re.findall(r"\d+(?:\.\d+)?%?|\d+[KMB]?|A100|H100|TPU|GPU|FID|FVD", block, re.I)), 6)
        if re.search(r"表源|Table|Figure|图源", block, re.I):
            score += 2
        scored.append((score, -index, block))
    if not scored:
        return fallback or []
    scored.sort(reverse=True)
    picked: list[str] = []
    seen: set[str] = set()
    for _, _, block in scored:
        compact = clamp_text(block, 360)
        key = compact[:80]
        if key in seen:
            continue
        seen.add(key)
        picked.append(compact)
        if len(picked) == limit:
            break
    return picked


def first_good_block(
    blocks: list[str],
    pattern: str,
    *,
    fallback: str,
    reject_pattern: str | None = None,
    limit: int = 360,
) -> str:
    for block in blocks:
        if reject_pattern and re.search(reject_pattern, block, re.I):
            continue
        if re.search(pattern, block, re.I):
            return clamp_text(block, limit)
    return clamp_text(fallback, limit)


def abstract_ideas(meta: ArxivMeta) -> list[str]:
    sentences = split_sentences(meta.abstract)
    proposal = first_matching_sentence(
        sentences,
        r"\bwe (propose|present|introduce|develop|build)|\bour (approach|model|method|framework|architecture)|\bthis paper (proposes|presents|introduces|develops)",
        sentences[0] if sentences else meta.abstract,
    )
    value = first_matching_sentence(
        sentences,
        r"enable|allow|efficient|scalable|robust|unified|general|reduce|faster|improve|benefit|novel|new",
        sentences[1] if len(sentences) > 1 else proposal,
    )
    experiment = first_matching_sentence(
        sentences,
        r"experiment|result|benchmark|outperform|achieve|improve|state-of-the-art|evaluation|demonstrate|show",
        "摘要中未给出清晰的第三点贡献。",
    )
    return [proposal, value, experiment]


def build_page_links(root: Path | None, pages: list[str]) -> list[str]:
    links = []
    base = root / "files" / "paper-deep-dives" if root else None
    for rel in pages:
        if base:
            path = base / rel
            href = path.resolve().as_uri()
        else:
            href = rel
        links.append(href)
    return links


def li(items: list[str]) -> str:
    return "".join(f"<li>{html.escape(item)}</li>" for item in items)


def linked_li(urls: list[str]) -> str:
    return "".join(
        f'<li><a href="{html.escape(url, quote=True)}">{html.escape(url)}</a></li>'
        for url in urls
    )


def build_summary_note_html(meta: ArxivMeta, *, pages: list[str], root: Path | None = None) -> str:
    all_sources = read_blog_sources(root, pages)
    sources = primary_blog_sources(all_sources, meta)
    source_text = "\n\n".join(text for _, _, text in sources)
    blocks = [block for _, _, text in sources for block in markdown_blocks(text)]
    sentences = split_sentences(meta.abstract)
    first = sentences[0] if sentences else meta.abstract
    motivation = first_good_block(
        blocks,
        r"动机|提出的问题|核心思想|问题很直接|困难|挑战|瓶颈|为什么|样本效率|需要|目标|如果|能否|不要强迫",
        fallback=[
            first_matching_sentence(
                sentences,
                r"challenge|problem|difficult|limitation|need|because|however|lack|cost|expensive|slow|bottleneck",
                first,
            )
        ][0],
        reject_pattern=r"^\d+\.|\||论文题名|局限|风险|不能证明|限制|短板|缺点|失败",
    )
    ideas = extract_numbered_items(source_text)
    if len(ideas) < 3:
        for candidate in best_blocks(
            blocks,
            r"提出|核心思想|关键|方法|框架|模型|训练|规划|rollout|latent|score|distillation|attention",
            reject_pattern=r"^\d+\.|\||^(Dimension|Route|Object|维度) /|局限|风险|不能证明|限制|短板|缺点|失败",
            limit=3,
        ):
            if candidate not in ideas:
                ideas.append(candidate)
            if len(ideas) == 3:
                break
    if len(ideas) < 3:
        for candidate in abstract_ideas(meta):
            if candidate not in ideas:
                ideas.append(candidate)
            if len(ideas) == 3:
                break
    ideas = [clamp_text(item, 260) for item in ideas[:3]]
    value = best_blocks(
        blocks,
        r"价值|意义|说明|证明|优势|效率|降低|提升|可复用|对主线意义|最值得|重要",
        fallback=[
            first_matching_sentence(
                sentences,
                r"enable|allow|efficient|scalable|robust|unified|general|reduce|faster|improve|benefit|novel|new",
                ideas[0] if ideas else first,
            )
        ],
        limit=1,
    )[0]
    experiment_items = best_blocks(
        blocks,
        r"实验|结果|benchmark|评估|FID|FVD|CLIP|success|成功率|分数|accuracy|latency|throughput|超过|优于|达到|提升|降低|消融",
        require_number=True,
        reject_pattern=r"训练使用|主实验使用|GPU|A100|H100|TPU|hours|teacher outputs|预计算",
        limit=2,
    )
    if not experiment_items:
        experiment_items = ["博客正文和摘要未给出可核验的实验数字；需要回到论文实验表格确认。"]
    training_items = best_blocks(
        blocks,
        r"训练|数据|数据集|dataset|episodes|tokens|samples|clips|hours|小时|GPU|TPU|A100|H100|batch|iteration|step|teacher outputs|预计算|显卡|训练多久",
        require_number=True,
        reject_pattern=r"^\d+\.|^(Dimension|Route|Object|维度) /|^维度：",
        limit=2,
    )
    if not training_items:
        training_items = ["博客正文和摘要未给出训练数据集、数据量、硬件或训练时长等可核验细节。"]
    links = [meta.abs_url, meta.pdf_url, *build_page_links(root, pages)]
    title = html.escape(meta.title)
    arxiv = html.escape(meta.arxiv_id)
    return (
        '<div class="zotero-note znv1">'
        f'<p><b>{SUMMARY_NOTE_MARKER}</b></p>'
        f"<p><b>论文：</b>{title}（arXiv:{arxiv}）</p>"
        f"<p><b>动机：</b>{html.escape(motivation)}</p>"
        "<p><b>主要 idea：</b></p>"
        f"<ol>{li(ideas)}</ol>"
        f"<p><b>核心价值：</b>{html.escape(value)}</p>"
        "<p><b>实验效果：</b></p>"
        f"<ul>{li(experiment_items)}</ul>"
        "<p><b>训练/数据细节：</b></p>"
        f"<ul>{li(training_items)}</ul>"
        "<p><b>关联链接：</b></p>"
        f"<ul>{linked_li(links)}</ul>"
        "<p><small>自动生成的可编辑速览；用于 Zotero 初筛，不替代论文正文和博客精读。</small></p>"
        "</div>"
    )


def has_summary_note(conn: sqlite3.Connection, parent_item_id: int) -> bool:
    return bool(
        summary_note_row(conn, parent_item_id)
    )


def summary_note_row(conn: sqlite3.Connection, parent_item_id: int) -> sqlite3.Row | None:
    note_patterns = [f"%{SUMMARY_NOTE_MARKER}%"] + [f"%{marker}%" for marker in SUMMARY_NOTE_LEGACY_MARKERS]
    title_values = [SUMMARY_NOTE_TITLE, *SUMMARY_NOTE_LEGACY_MARKERS]
    title_placeholders = ", ".join("?" for _ in title_values)
    note_placeholders = " or ".join("note like ?" for _ in note_patterns)
    return conn.execute(
        f"""
        select itemID, note
        from itemNotes
        where parentItemID = ?
          and (title in ({title_placeholders}) or {note_placeholders})
        order by itemID
        limit 1
        """,
        (parent_item_id, *title_values, *note_patterns),
    ).fetchone()


def create_summary_note(conn: sqlite3.Connection, parent_item_id: int, note_html: str) -> int:
    parent = conn.execute("select libraryID from items where itemID = ?", (parent_item_id,)).fetchone()
    if not parent:
        raise ValueError(f"Parent item not found: {parent_item_id}")
    key = rand_key(conn)
    now = now_sql()
    cur = conn.execute(
        """
        insert into items
        (itemTypeID, dateAdded, dateModified, clientDateModified, libraryID, key, version, synced)
        values (?, ?, ?, ?, ?, ?, 0, 0)
        """,
        (ITEM_TYPE["note"], now, now, now, parent["libraryID"], key),
    )
    note_item_id = cur.lastrowid
    conn.execute(
        "insert into itemNotes (itemID, parentItemID, note, title) values (?, ?, ?, ?)",
        (note_item_id, parent_item_id, note_html, SUMMARY_NOTE_TITLE),
    )
    return note_item_id


def update_summary_note(conn: sqlite3.Connection, note_item_id: int, note_html: str) -> None:
    now = now_sql()
    conn.execute(
        "update itemNotes set note = ?, title = ? where itemID = ?",
        (note_html, SUMMARY_NOTE_TITLE, note_item_id),
    )
    conn.execute(
        "update items set dateModified = ?, clientDateModified = ?, synced = 0 where itemID = ?",
        (now, now, note_item_id),
    )


def summary_note_status(root: Path, conn: sqlite3.Connection) -> dict[str, Any]:
    refs = parse_paper_deepdives(root)
    meta = load_meta()
    collections = db_collections(conn)
    root_row = next((r for r in collections.values() if r["key"] == ROOT_COLLECTION_KEY), None)
    if not root_row:
        raise SystemExit(f"Root collection key not found: {ROOT_COLLECTION_KEY}")
    root_desc = descendants(collections, root_row["collectionID"])
    arxiv_to_items = arxiv_index(conn)
    pdfs = pdf_children(conn)
    collections_by_item = item_collections(conn)
    records = []
    for arxiv_id in sorted(refs):
        item_id = canonical_item(
            arxiv_id,
            arxiv_to_items.get(arxiv_id, []),
            pdfs,
            collections_by_item,
            root_desc,
        )
        has_note = bool(item_id and has_summary_note(conn, item_id))
        records.append(
            {
                "arxiv_id": arxiv_id,
                "item_id": item_id,
                "has_item": item_id is not None,
                "has_metadata": arxiv_id in meta,
                "has_summary_note": has_note,
                "pages": sorted(refs[arxiv_id].pages),
            }
        )
    return {
        "summary": {
            "target_papers": len(records),
            "missing_items": sum(1 for r in records if not r["has_item"]),
            "missing_metadata": sum(1 for r in records if not r["has_metadata"]),
            "missing_summary_notes": sum(1 for r in records if r["has_item"] and r["has_metadata"] and not r["has_summary_note"]),
            "has_summary_notes": sum(1 for r in records if r["has_summary_note"]),
        },
        "records": records,
    }


def apply_summary_notes(root: Path) -> dict[str, Any]:
    ensure_work_dirs()
    backup = ZOTERO_DB.with_name(f"zotero.sqlite.codex-summary-notes-{dt.datetime.now():%Y%m%d-%H%M%S}.bak")
    shutil.copy2(ZOTERO_DB, backup)
    conn = connect(ZOTERO_DB)
    status = summary_note_status(root, conn)
    refs = parse_paper_deepdives(root)
    meta = load_meta()
    stats = {
        "backup": str(backup),
        "target_papers": status["summary"]["target_papers"],
        "existing_notes": status["summary"]["has_summary_notes"],
        "created_notes": 0,
        "updated_notes": 0,
        "missing_items": [],
        "missing_metadata": [],
        "created_arxiv_ids": [],
        "updated_arxiv_ids": [],
    }
    with conn:
        for record in status["records"]:
            arxiv_id = record["arxiv_id"]
            if not record["has_item"]:
                stats["missing_items"].append(arxiv_id)
                continue
            if not record["has_metadata"]:
                stats["missing_metadata"].append(arxiv_id)
                continue
            note_html = build_summary_note_html(meta[arxiv_id], pages=sorted(refs[arxiv_id].pages), root=root)
            if record["has_summary_note"]:
                note_row = summary_note_row(conn, record["item_id"])
                if note_row and note_row["note"] != note_html:
                    update_summary_note(conn, note_row["itemID"], note_html)
                    stats["updated_notes"] += 1
                    stats["updated_arxiv_ids"].append(arxiv_id)
                continue
            create_summary_note(conn, record["item_id"], note_html)
            stats["created_notes"] += 1
            stats["created_arxiv_ids"].append(arxiv_id)
    after = summary_note_status(root, conn)
    stats["final_summary"] = after["summary"]
    (WORK_DIR / "summary-notes.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def verify_summary_notes(root: Path) -> dict[str, Any]:
    conn = connect(ZOTERO_DB)
    status = summary_note_status(root, conn)
    status["complete"] = (
        status["summary"]["missing_items"] == 0
        and status["summary"]["missing_metadata"] == 0
        and status["summary"]["missing_summary_notes"] == 0
    )
    (WORK_DIR / "summary-notes-verify.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    return status


def add_to_collection(conn: sqlite3.Connection, item_id: int, collection_id: int) -> None:
    conn.execute(
        "insert or ignore into collectionItems (collectionID, itemID, orderIndex) values (?, ?, 0)",
        (collection_id, item_id),
    )


def remove_non_leaf_target_memberships(
    conn: sqlite3.Connection,
    item_id: int,
    root_desc: set[int],
    child_map: dict[int, list[int]],
) -> None:
    non_leaf = {cid for cid in root_desc if child_map.get(cid)}
    for cid in non_leaf:
        conn.execute("delete from collectionItems where collectionID = ? and itemID = ?", (cid, item_id))


def leaf_for_existing_title(title: str, parent_name: str) -> tuple[str, ...]:
    t = normalized(title)
    if "world action models" in t or "fast-wam" in t:
        return ("VLA", "世界模型辅助策略")
    if "vo-dp" in t or "diffusion policy" in t:
        return ("VLA", "扩散策略与闭环控制")
    if "vggt" in t or "depth" in t or "dino" in t:
        return ("具身智能", "3D 视觉与几何基础")
    if "survey on vision-language-action" in t or "vision-language-action models for robotics" in t:
        return ("VLA", "综述与路线图")
    if "openvla" in t or "multi-camera" in t:
        return ("VLA", "Generative VLA 与开放世界策略")
    return (parent_name, "待整理")


def move_non_leaf_items_to_leaves(
    conn: sqlite3.Connection,
    collections: dict[int, dict[str, Any]],
    root_id: int,
) -> int:
    child_map = children_map(collections)
    root_desc = descendants(collections, root_id)
    titles = item_titles(conn)
    moved = 0
    for cid in sorted(root_desc):
        if cid == root_id or not child_map.get(cid):
            continue
        parent_name = collections[cid]["collectionName"]
        rows = conn.execute("select itemID from collectionItems where collectionID = ?", (cid,)).fetchall()
        for row in rows:
            item_id = row["itemID"]
            title = titles.get(item_id, "")
            leaf = leaf_for_existing_title(title, parent_name)
            leaf_cid = get_or_create_collection(conn, collections, root_id, leaf)
            add_to_collection(conn, item_id, leaf_cid)
            conn.execute("delete from collectionItems where collectionID = ? and itemID = ?", (cid, item_id))
            moved += 1
    return moved


def candidate_annotation_snippets(meta: ArxivMeta) -> list[tuple[str, str, str]]:
    sentences = split_sentences(meta.abstract)
    out: list[tuple[str, str, str]] = []
    if sentences:
        out.append((sentences[0], COLORS["position"], "Codex deep-dive｜问题定位：这句用来确定论文真正要解决的任务对象、输入输出和适用边界。读这篇 deep dive 时先把它和博客里的“核心问题”段落对齐，避免后面只记住方法名而忽略问题设定。"))
    if len(sentences) > 1:
        out.append((sentences[1], COLORS["mechanism"], "Codex deep-dive｜核心机制：这句通常对应论文最重要的结构、训练目标或数据路径改动。阅读时要追问它改变了哪条信息流，以及博客专题中是否把这个机制拆成了可复用的工程模块。"))
    evidence = next((s for s in sentences if re.search(r"outperform|state-of-the-art|improve|achieve|results|benchmark|experiments", s, re.I)), None)
    if evidence:
        out.append((evidence, COLORS["evidence"], "Codex deep-dive｜证据读法：这里是论文 claim 和实验结果相连的位置。需要回到正文表格/图，核对数据集、baseline、指标和是否 closed-loop，不能把一个 benchmark 的提升直接外推成所有任务都成立。"))
    boundary = next((s for s in sentences if re.search(r"however|limitation|challenge|without|only|exclusive|not|cannot", s, re.I)), None)
    if boundary:
        out.append((boundary, COLORS["boundary"], "Codex deep-dive｜边界提醒：这类表述通常暗示数据、模型、评测或部署条件的限制。把它和博客里的“局限/不能外推项”对照，重点记录哪些结论还不能迁移到真实机器人、长时任务或不同硬件场景。"))
    if len(out) < 4 and meta.title:
        out.insert(0, (meta.title, COLORS["context"], "Codex deep-dive｜标题定位：标题先帮助把论文放回 paper-deep-dives 的主题谱系。后续阅读时要判断它是方向综述、机制论文、系统报告，还是某个模块的工程化实现。"))
    return out[:4]


def fallback_annotation_targets(pdf_path_: Path) -> list[tuple[int, list[list[float]], str, str, str]]:
    try:
        import pdfplumber
    except Exception:
        return []
    colors = [COLORS["claim"], COLORS["position"], COLORS["mechanism"], COLORS["evidence"], COLORS["gray"]]
    comments = [
        "Codex deep-dive｜核心 claim：这里是进入论文的第一层判断，先确认作者到底声称解决了什么问题。继续读时要把这句话映射到博客专题的主线：问题、机制、证据和不能外推的部分。",
        "Codex deep-dive｜问题定位：这处文字适合连接博客里的“核心问题”段落。重点看旧方法为什么不够，以及论文把难点归因于数据、模型结构、训练目标、推理成本还是评测协议。",
        "Codex deep-dive｜方法机制：这处文字通常能引出模型结构、训练流程或控制接口。读正文时沿着 data -> model -> loss -> inference 的链路追踪，判断真正可复用的是哪一段机制。",
        "Codex deep-dive｜证据入口：后续要回到实验表格、图和消融，核对 benchmark、baseline、数据规模和任务口径。这个标记不是结论本身，而是提醒把 claim 绑定到可验证证据。",
        "Codex deep-dive｜读数边界：这里用来提醒不要过度外推。尤其是 VLA、世界模型和高效系统论文，要区分论文自述、官方 demo、第三方复现、closed-loop 真实机器人和本站推断。",
    ]
    targets: list[tuple[int, list[list[float]], str, str, str]] = []
    secondary: list[tuple[int, list[list[float]], str, str, str]] = []
    keyword = re.compile(
        r"\b(we|present|propose|introduce|demonstrate|show|model|method|training|"
        r"experiments?|results?|benchmark|outperform|achieve|learn|policy|world)\b",
        re.I,
    )
    with pdfplumber.open(str(pdf_path_)) as pdf:
        for page_index, page in enumerate(pdf.pages[:2]):
            words = page.extract_words(x_tolerance=2, y_tolerance=3, keep_blank_chars=False)
            candidates: list[list[dict[str, Any]]] = []
            for word in words:
                top = float(word.get("top", 0))
                bottom = float(word.get("bottom", 0))
                text = str(word.get("text", ""))
                if top < 35 or bottom > float(page.height) - 35:
                    continue
                if re.fullmatch(r"\d+", text):
                    continue
                candidates.append([word])
            flat_candidates = [w[0] for w in candidates]
            for segment_words in line_word_segments(flat_candidates):
                text = " ".join(str(w["text"]) for w in segment_words)
                clean = normalized(text)
                if len(clean) < 38:
                    continue
                if any(skip in clean for skip in ["arxiv:", "preprint", "proceedings", "published as"]):
                    continue
                top = min(float(w["top"]) for w in segment_words)
                has_keyword = bool(keyword.search(text))
                # Skip title-block author/affiliation lines unless they carry
                # method/evidence language. Exact title matching is handled
                # separately by candidate_annotation_snippets().
                if page_index == 0 and top < 220 and not has_keyword:
                    continue
                rects = words_to_rects(segment_words[: min(len(segment_words), 18)])
                if not rects:
                    continue
                idx = len(targets) % len(colors)
                row = (page_index, rects, text, colors[idx], comments[idx])
                if has_keyword:
                    targets.append(row)
                else:
                    secondary.append(row)
                if len(targets) >= 5:
                    return targets
    for row in secondary:
        if len(targets) >= 5:
            break
        targets.append(row)
    return targets


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [p.strip() for p in parts if len(p.strip()) > 40]


def normalized(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def find_text_rects(pdf_path_: Path, needle: str) -> tuple[int, list[list[float]], str] | None:
    try:
        import pdfplumber
    except Exception:
        return None
    target = normalized(needle)
    if not target:
        return None
    words_target = target.split()
    if len(words_target) > 28:
        words_target = words_target[:28]
        target = " ".join(words_target)
    with pdfplumber.open(str(pdf_path_)) as pdf:
        for page_index, page in enumerate(pdf.pages[:4]):
            words = page.extract_words(x_tolerance=2, y_tolerance=3, keep_blank_chars=False)
            tokens = [normalized(w.get("text", "")) for w in words]
            for i in range(0, max(0, len(tokens) - len(words_target) + 1)):
                if tokens[i : i + len(words_target)] == words_target:
                    rects = words_to_rects(words[i : i + len(words_target)])
                    text = " ".join(w["text"] for w in words[i : i + len(words_target)])
                    return page_index, rects, text
            # Fuzzy prefix match for arXiv extraction differences.
            for i in range(len(tokens)):
                joined = " ".join(tokens[i : i + min(20, len(words_target))])
                if joined and target.startswith(joined) and len(joined) > 45:
                    rects = words_to_rects(words[i : i + min(20, len(words_target))])
                    text = " ".join(w["text"] for w in words[i : i + min(20, len(words_target))])
                    return page_index, rects, text
    return None


def line_word_segments(words: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group PDF words into contiguous visual line segments.

    Two-column papers often have words with nearly identical y coordinates in
    both columns. Grouping only by y creates a rectangle that spans the gutter.
    This function first groups by baseline and then splits large x gaps.
    """
    if not words:
        return []
    sorted_words = sorted(words, key=lambda w: ((float(w["top"]) + float(w["bottom"])) / 2, float(w["x0"])))
    lines: list[list[dict[str, Any]]] = []
    for word in sorted_words:
        mid = (float(word["top"]) + float(word["bottom"])) / 2
        height = max(1.0, float(word["bottom"]) - float(word["top"]))
        for line in lines:
            ref = line[0]
            ref_mid = (float(ref["top"]) + float(ref["bottom"])) / 2
            ref_height = max(1.0, float(ref["bottom"]) - float(ref["top"]))
            if abs(mid - ref_mid) <= max(2.0, min(height, ref_height) * 0.45):
                line.append(word)
                break
        else:
            lines.append([word])

    segments: list[list[dict[str, Any]]] = []
    for line in lines:
        line = sorted(line, key=lambda w: float(w["x0"]))
        current: list[dict[str, Any]] = []
        for word in line:
            if not current:
                current.append(word)
                continue
            prev = current[-1]
            gap = float(word["x0"]) - float(prev["x1"])
            prev_width = max(1.0, float(prev["x1"]) - float(prev["x0"]))
            prev_chars = max(1, len(str(prev.get("text", ""))))
            char_width = prev_width / prev_chars
            # Normal word spacing is small; a column gutter or unrelated table
            # cell is much larger. Split there so each highlight hugs text.
            if gap > max(8.0, char_width * 3.0):
                segments.append(current)
                current = [word]
            else:
                current.append(word)
        if current:
            segments.append(current)
    return segments


def words_to_rects(words: list[dict[str, Any]]) -> list[list[float]]:
    rects: list[list[float]] = []
    for line_words in line_word_segments(words):
        x0 = min(float(w["x0"]) for w in line_words)
        x1 = max(float(w["x1"]) for w in line_words)
        top = min(float(w["top"]) for w in line_words)
        bottom = max(float(w["bottom"]) for w in line_words)
        rects.append([round(x0, 3), round(top, 3), round(x1, 3), round(bottom, 3)])
    return rects


def pdf_file_for_attachment(conn: sqlite3.Connection, attachment_id: int) -> Path | None:
    row = conn.execute("select key from items where itemID = ?", (attachment_id,)).fetchone()
    att = conn.execute("select path from itemAttachments where itemID = ?", (attachment_id,)).fetchone()
    if not row or not att or not att["path"] or not att["path"].startswith("storage:"):
        return None
    return ZOTERO_STORAGE / row["key"] / att["path"].split("storage:", 1)[1]


def existing_annotation_count(conn: sqlite3.Connection, attachment_id: int) -> int:
    return conn.execute(
        "select count(*) from itemAnnotations where parentItemID = ?",
        (attachment_id,),
    ).fetchone()[0]


def existing_codex_annotation_count(conn: sqlite3.Connection, attachment_id: int) -> int:
    return conn.execute(
        """
        select count(*) from itemAnnotations
        where parentItemID = ? and comment like '%Codex deep-dive%'
        """,
        (attachment_id,),
    ).fetchone()[0]


def next_sort_index(conn: sqlite3.Connection, attachment_id: int, page_index: int, seq: int) -> str:
    return f"{page_index:05d}|{seq:06d}|00000"


def add_annotations(
    conn: sqlite3.Connection,
    attachment_id: int,
    meta: ArxivMeta,
    *,
    codex_only: bool = False,
) -> int:
    pdf_file = pdf_file_for_attachment(conn, attachment_id)
    if not pdf_file or not pdf_file.exists():
        return 0
    existing = existing_codex_annotation_count(conn, attachment_id) if codex_only else existing_annotation_count(conn, attachment_id)
    if existing >= 3:
        return 0
    added = 0

    def insert_annotation(page_index: int, rects: list[list[float]], text: str, color: str, comment: str, seq: int) -> None:
        key = rand_key(conn)
        now = now_sql()
        cur = conn.execute(
            """
            insert into items
            (itemTypeID, dateAdded, dateModified, clientDateModified, libraryID, key, version, synced)
            values (?, ?, ?, ?, 1, ?, 0, 0)
            """,
            (ITEM_TYPE["annotation"], now, now, now, key),
        )
        ann_item_id = cur.lastrowid
        position = json.dumps({"pageIndex": page_index, "rects": rects}, separators=(",", ":"))
        comment_html = (
            "<p><b>"
            + html.escape(comment.split("：", 1)[0])
            + "：</b>"
            + html.escape(comment.split("：", 1)[1] if "：" in comment else comment)
            + "</p>"
        )
        conn.execute(
            """
            insert into itemAnnotations
            (itemID, parentItemID, type, authorName, text, comment, color, pageLabel, sortIndex, position, isExternal)
            values (?, ?, 1, null, ?, ?, ?, ?, ?, ?, 0)
            """,
            (
                ann_item_id,
                attachment_id,
                text,
                comment_html,
                color,
                str(page_index + 1),
                next_sort_index(conn, attachment_id, page_index, seq),
                position,
            ),
        )

    seq = existing + 1
    seen_text: set[str] = set()
    for snippet, color, comment in candidate_annotation_snippets(meta):
        if existing + added >= 3:
            break
        found = find_text_rects(pdf_file, snippet)
        if not found:
            continue
        page_index, rects, text = found
        clean = normalized(text)
        if not rects or clean in seen_text:
            continue
        insert_annotation(page_index, rects, text, color, comment, seq)
        seen_text.add(clean)
        added += 1
        seq += 1
    for page_index, rects, text, color, comment in fallback_annotation_targets(pdf_file):
        if existing + added >= 3:
            break
        clean = normalized(text)
        if clean in seen_text:
            continue
        insert_annotation(page_index, rects, text, color, comment, seq)
        seen_text.add(clean)
        added += 1
        seq += 1
    return added


def delete_codex_annotations(conn: sqlite3.Connection) -> int:
    rows = conn.execute(
        "select itemID from itemAnnotations where comment like '%Codex deep-dive%'"
    ).fetchall()
    item_ids = [row["itemID"] for row in rows]
    for item_id in item_ids:
        conn.execute("delete from items where itemID = ?", (item_id,))
    return len(item_ids)


def reannotate(root: Path) -> dict[str, Any]:
    refs = parse_paper_deepdives(root)
    meta = load_meta()
    backup = ZOTERO_DB.with_name(f"zotero.sqlite.codex-reannotate-{dt.datetime.now():%Y%m%d-%H%M%S}.bak")
    shutil.copy2(ZOTERO_DB, backup)
    conn = connect(ZOTERO_DB)
    collections = db_collections(conn)
    root_row = next((r for r in collections.values() if r["key"] == ROOT_COLLECTION_KEY), None)
    if not root_row:
        raise SystemExit(f"Root collection key not found: {ROOT_COLLECTION_KEY}")
    root_id = root_row["collectionID"]
    root_desc = descendants(collections, root_id)
    stats = {"backup": str(backup), "deleted_codex_annotations": 0, "added_annotations": 0}
    with conn:
        stats["deleted_codex_annotations"] = delete_codex_annotations(conn)
        arxiv_to_items = arxiv_index(conn)
        pdfs = pdf_children(conn)
        collections_by_item = item_collections(conn)
        for arxiv_id in sorted(refs):
            if arxiv_id not in meta:
                continue
            item_id = canonical_item(
                arxiv_id,
                arxiv_to_items.get(arxiv_id, []),
                pdfs,
                collections_by_item,
                root_desc,
            )
            if item_id is None:
                continue
            pdf_ids = pdfs.get(item_id, [])
            if not pdf_ids:
                continue
            stats["added_annotations"] += add_annotations(
                conn,
                pdf_ids[0],
                meta[arxiv_id],
                codex_only=True,
            )
    (WORK_DIR / "reannotate.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def apply(root: Path, *, skip_annotations: bool = False) -> dict[str, Any]:
    refs = parse_paper_deepdives(root)
    meta = load_meta()
    missing_meta = sorted(set(refs) - set(meta))
    if missing_meta:
        raise SystemExit(f"Run fetch first; missing metadata for {len(missing_meta)} IDs")
    missing_pdf = [x for x in refs if not pdf_path(x, meta).exists()]
    if missing_pdf:
        raise SystemExit(f"Run fetch first; missing PDF files for {len(missing_pdf)} IDs")

    backup = ZOTERO_DB.with_name(f"zotero.sqlite.codex-deepdives-{dt.datetime.now():%Y%m%d-%H%M%S}.bak")
    shutil.copy2(ZOTERO_DB, backup)
    conn = connect(ZOTERO_DB)
    collections = db_collections(conn)
    root_row = next((r for r in collections.values() if r["key"] == ROOT_COLLECTION_KEY), None)
    if not root_row:
        raise SystemExit(f"Root collection key not found: {ROOT_COLLECTION_KEY}")
    root_id = root_row["collectionID"]
    stats = {
        "backup": str(backup),
        "created_collections": 0,
        "created_items": 0,
        "attached_pdfs": 0,
        "added_collection_memberships": 0,
        "added_unread_tags": 0,
        "added_annotations": 0,
        "moved_non_leaf_items": 0,
    }
    with conn:
        # Ensure all target leaves exist first.
        for path in sorted({leaf for ref in refs.values() for leaf in ref.leaves}):
            before = len(collections)
            get_or_create_collection(conn, collections, root_id, path)
            stats["created_collections"] += len(collections) - before

        child_map = children_map(collections)
        root_desc = descendants(collections, root_id)
        arxiv_to_items = arxiv_index(conn)
        pdfs = pdf_children(conn)
        collections_by_item = item_collections(conn)

        for arxiv_id in sorted(refs):
            ref = refs[arxiv_id]
            item_id = canonical_item(
                arxiv_id,
                arxiv_to_items.get(arxiv_id, []),
                pdfs,
                collections_by_item,
                root_desc,
            )
            if item_id is None:
                item_id = create_preprint(conn, meta[arxiv_id])
                arxiv_to_items.setdefault(arxiv_id, []).append(item_id)
                stats["created_items"] += 1

            pdf_ids = pdfs.get(item_id, [])
            if not pdf_ids:
                attachment_id = attach_pdf(conn, item_id, meta[arxiv_id], pdf_path(arxiv_id, meta))
                pdfs.setdefault(item_id, []).append(attachment_id)
                pdf_ids = [attachment_id]
                stats["attached_pdfs"] += 1

            before_tags = set(item_tags(conn).get(item_id, set()))
            add_unread_if_needed(conn, item_id)
            after_tags = set(item_tags(conn).get(item_id, set()))
            if UNREAD_TAG in after_tags and UNREAD_TAG not in before_tags:
                stats["added_unread_tags"] += 1

            for leaf in sorted(ref.leaves):
                cid = get_or_create_collection(conn, collections, root_id, leaf)
                before = conn.total_changes
                add_to_collection(conn, item_id, cid)
                if conn.total_changes > before:
                    stats["added_collection_memberships"] += 1
            remove_non_leaf_target_memberships(conn, item_id, root_desc, child_map)

            if not skip_annotations:
                for attachment_id in pdf_ids[:1]:
                    stats["added_annotations"] += add_annotations(conn, attachment_id, meta[arxiv_id])

        stats["moved_non_leaf_items"] = move_non_leaf_items_to_leaves(conn, collections, root_id)

    (WORK_DIR / "apply.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def verify(root: Path, *, skip_annotations: bool = False) -> dict[str, Any]:
    out = audit(root)
    summary = out["summary"]
    complete = (
        summary["missing_items"] == 0
        and summary["missing_pdfs"] == 0
        and (skip_annotations or summary["needs_annotations"] == 0)
        and summary["needs_collection_membership"] == 0
        and summary["needs_read_state_tag"] == 0
    )
    result = {"complete": complete, "summary": summary}
    (WORK_DIR / "verify.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def repair_invalid_keys() -> dict[str, Any]:
    ensure_work_dirs()
    backup = ZOTERO_DB.with_name(f"zotero.sqlite.codex-repair-keys-{dt.datetime.now():%Y%m%d-%H%M%S}.bak")
    shutil.copy2(ZOTERO_DB, backup)

    conn = connect(ZOTERO_DB)
    invalid_collections = [
        dict(row)
        for row in conn.execute("select collectionID, key, synced from collections order by collectionID")
        if not is_valid_object_key(row["key"])
    ]
    invalid_items = [
        dict(row)
        for row in conn.execute("select itemID, key, synced from items order by itemID")
        if not is_valid_object_key(row["key"])
    ]

    synced_invalid = [
        {"table": "collections", "id": row["collectionID"], "key": row["key"]}
        for row in invalid_collections
        if row.get("synced")
    ] + [
        {"table": "items", "id": row["itemID"], "key": row["key"]}
        for row in invalid_items
        if row.get("synced")
    ]
    if synced_invalid:
        raise SystemExit(f"Refusing to rewrite synced invalid Zotero keys: {synced_invalid[:10]}")

    collection_keys = {row[0] for row in conn.execute("select key from collections")}
    item_keys = {row[0] for row in conn.execute("select key from items")}
    storage_dirs = {path.name for path in ZOTERO_STORAGE.iterdir() if path.is_dir()} if ZOTERO_STORAGE.exists() else set()

    collection_mapping = {
        row["collectionID"]: {"old": row["key"], "new": random_object_key(collection_keys)}
        for row in invalid_collections
    }
    item_mapping: dict[int, dict[str, str]] = {}
    for row in invalid_items:
        while True:
            new_key = random_object_key(item_keys)
            if new_key not in storage_dirs:
                break
        item_mapping[row["itemID"]] = {"old": row["key"], "new": new_key}
        storage_dirs.add(new_key)

    attachment_rows = conn.execute(
        """
        select i.itemID, i.key, a.path
        from items i join itemAttachments a on a.itemID = i.itemID
        where a.path like 'storage:%'
        """
    ).fetchall()
    storage_moves: list[tuple[Path, Path]] = []
    missing_storage_dirs = []
    for row in attachment_rows:
        mapping = item_mapping.get(row["itemID"])
        if not mapping:
            continue
        old_dir = ZOTERO_STORAGE / mapping["old"]
        new_dir = ZOTERO_STORAGE / mapping["new"]
        if old_dir.exists():
            if new_dir.exists():
                raise SystemExit(f"Refusing to overwrite existing Zotero storage dir: {new_dir}")
            storage_moves.append((old_dir, new_dir))
        else:
            missing_storage_dirs.append(str(old_dir))

    moved: list[tuple[Path, Path]] = []
    try:
        for old_dir, new_dir in storage_moves:
            old_dir.rename(new_dir)
            moved.append((old_dir, new_dir))
        with conn:
            for collection_id, mapping in collection_mapping.items():
                conn.execute("update collections set key = ?, synced = 0 where collectionID = ?", (mapping["new"], collection_id))
            for item_id, mapping in item_mapping.items():
                conn.execute("update items set key = ?, synced = 0 where itemID = ?", (mapping["new"], item_id))
    except Exception:
        for old_dir, new_dir in reversed(moved):
            if new_dir.exists() and not old_dir.exists():
                new_dir.rename(old_dir)
        raise

    result = {
        "backup": str(backup),
        "repaired_items": len(item_mapping),
        "repaired_collections": len(collection_mapping),
        "renamed_storage_dirs": len(storage_moves),
        "missing_storage_dirs": missing_storage_dirs,
        "item_key_mapping": item_mapping,
        "collection_key_mapping": collection_mapping,
    }
    (WORK_DIR / "repair-keys.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["audit", "fetch", "apply", "reannotate", "verify", "repair-keys", "apply-notes", "verify-notes"])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--skip-annotations",
        action="store_true",
        help="skip creating or requiring Codex deep-dive PDF annotations",
    )
    args = parser.parse_args()
    ensure_work_dirs()
    if args.command == "audit":
        result = audit(args.root)
    elif args.command == "fetch":
        result = fetch(args.root)
    elif args.command == "apply":
        result = apply(args.root, skip_annotations=args.skip_annotations)
    elif args.command == "reannotate":
        result = reannotate(args.root)
    elif args.command == "repair-keys":
        result = repair_invalid_keys()
    elif args.command == "apply-notes":
        result = apply_summary_notes(args.root)
    elif args.command == "verify-notes":
        result = verify_summary_notes(args.root)
    else:
        result = verify(args.root, skip_annotations=args.skip_annotations)
    print(json.dumps(result["summary"] if "summary" in result else result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
