#!/usr/bin/env python3
"""Audit image references and image assets for the MkDocs site."""

from __future__ import annotations

import argparse
import html
import math
import re
import sys
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote, urlparse

try:
    from PIL import Image, ImageChops, ImageDraw, ImageOps
except ImportError as exc:  # pragma: no cover - exercised only in missing envs.
    raise SystemExit(
        "Pillow is required for image-quality checks. Install requirements.txt first."
    ) from exc

warnings.filterwarnings("ignore", category=UserWarning, module="PIL.Image")


DOCS_ROOT = Path("files")
ASSETS_ROOT = DOCS_ROOT / "assets" / "images"
EXCLUDED_DOC_PARTS = {"roadmap"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg"}

MARKDOWN_IMAGE_RE = re.compile(
    r"!\[(?P<alt>[^\]]*)\]\((?P<target><[^>]+>|[^)\s]+)(?:\s+\"[^\"]*\")?\)"
    r"(?P<attrs>\{[^}]*\})?"
)
HTML_IMAGE_RE = re.compile(r"<img\b(?P<attrs>[^>]*)>", re.IGNORECASE)
HTML_ATTR_RE = re.compile(
    r"(?P<name>[A-Za-z_:][-A-Za-z0-9_:.]*)\s*=\s*"
    r"(?P<quote>[\"'])(?P<value>.*?)(?P=quote)",
    re.DOTALL,
)
WIDTH_ATTR_RE = re.compile(r"\bwidth\s*=\s*[\"']?(?P<width>\d+)")
CLASS_ATTR_RE = re.compile(r"\.([A-Za-z0-9_-]+)")
SOURCE_RE = re.compile(r"(图源|图片来源|表源|Source|source|原论文图|官方截图|官方仓库|arXiv|OpenReview)")
NOTE_RE = re.compile(
    r"!!!\s+(note|info|tip|abstract)\s+[\"']?"
    r"[^\"'\n]*(图解|怎么读|读图|支撑什么判断|这张算法图|解释框|初学者先抓住|符号解释|公式解释|为什么|关键|差别|怎么看|工程点|重点)"
)
PROSE_NOTE_RE = re.compile(
    r"(原图在说什么|初学者怎么看|容易误读|本教程读法|这张图有两个读法|这张图分|读这张图|"
    r"这张图支撑|这张图就是|图里的重点|图中几个数字|可以拆成三个结论)"
)
REMOTE_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")
SVG_SIZE_RE = re.compile(
    r"<svg\b[^>]*(?:width=[\"'](?P<width>[\d.]+)(?:px)?[\"'][^>]*)?"
    r"(?:height=[\"'](?P<height>[\d.]+)(?:px)?[\"'][^>]*)?",
    re.IGNORECASE | re.DOTALL,
)
SVG_VIEWBOX_RE = re.compile(
    r"viewBox=[\"'][\d.\-]+\s+[\d.\-]+\s+(?P<width>[\d.]+)\s+(?P<height>[\d.]+)[\"']",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ImageRef:
    page: Path
    line: int
    raw: str
    alt: str
    target: str
    kind: str
    display_width: int | None
    classes: tuple[str, ...]
    resolved: Path | None
    is_remote: bool


@dataclass
class ImageMeta:
    path: Path
    width: int | None = None
    height: int | None = None
    mode: str | None = None
    file_size: int = 0
    unreadable: str | None = None
    edge_touch_score: float | None = None
    whitespace_fraction: float | None = None
    bottom_text_band_score: float | None = None


@dataclass
class Issue:
    severity: str
    code: str
    subject: str
    detail: str
    page: Path | None = None
    line: int | None = None
    image: Path | None = None

    @property
    def sort_key(self) -> tuple[int, str, str]:
        order = {"high": 0, "medium": 1, "low": 2, "info": 3}
        return (
            order.get(self.severity, 9),
            self.page.as_posix() if self.page else "",
            self.subject,
        )


@dataclass
class Audit:
    refs: list[ImageRef] = field(default_factory=list)
    metas: dict[Path, ImageMeta] = field(default_factory=dict)
    issues: list[Issue] = field(default_factory=list)
    contact_sheet: Path | None = None


def iter_markdown_pages() -> list[Path]:
    pages: list[Path] = []
    for path in DOCS_ROOT.rglob("*.md"):
        if path.name.startswith("._"):
            continue
        if any(part in EXCLUDED_DOC_PARTS for part in path.relative_to(DOCS_ROOT).parts):
            continue
        pages.append(path)
    return sorted(pages)


def iter_image_assets() -> list[Path]:
    return sorted(
        path
        for path in ASSETS_ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and not path.name.startswith("._")
    )


def clean_target(target: str) -> str:
    target = target.strip().strip("<>")
    target = target.split("#", 1)[0].split("?", 1)[0]
    return unquote(target)


def is_remote_target(target: str) -> bool:
    if target.startswith("#"):
        return False
    parsed = urlparse(target)
    return bool(parsed.scheme) or bool(REMOTE_SCHEME_RE.match(target))


def rendered_page_dir(page: Path) -> Path:
    rel = page.relative_to(DOCS_ROOT)
    if rel.name == "index.md":
        return DOCS_ROOT / rel.parent
    return DOCS_ROOT / rel.parent / rel.stem


def safe_resolve(base: Path, target: str) -> Path | None:
    cleaned = clean_target(target)
    if not cleaned or is_remote_target(cleaned):
        return None
    candidate = (base / cleaned).resolve()
    root = Path.cwd().resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


def attrs_to_width(attrs: str | None) -> int | None:
    if not attrs:
        return None
    match = WIDTH_ATTR_RE.search(attrs)
    if not match:
        return None
    try:
        return int(match.group("width"))
    except ValueError:
        return None


def attrs_to_classes(attrs: str | None) -> tuple[str, ...]:
    if not attrs:
        return ()
    classes = CLASS_ATTR_RE.findall(attrs)
    html_class = re.search(r"\bclass\s*=\s*[\"']([^\"']+)[\"']", attrs)
    if html_class:
        classes.extend(html_class.group(1).split())
    return tuple(sorted(set(classes)))


def parse_html_attrs(raw_attrs: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for match in HTML_ATTR_RE.finditer(raw_attrs):
        attrs[match.group("name").lower()] = html.unescape(match.group("value"))
    return attrs


def extract_refs(page: Path) -> list[ImageRef]:
    text = page.read_text(encoding="utf-8", errors="ignore")
    refs: list[ImageRef] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        for match in MARKDOWN_IMAGE_RE.finditer(line):
            target = match.group("target")
            cleaned = clean_target(target)
            is_remote = is_remote_target(cleaned)
            refs.append(
                ImageRef(
                    page=page,
                    line=line_no,
                    raw=match.group(0),
                    alt=match.group("alt"),
                    target=cleaned,
                    kind="markdown",
                    display_width=attrs_to_width(match.group("attrs")),
                    classes=attrs_to_classes(match.group("attrs")),
                    resolved=None if is_remote else safe_resolve(page.parent, cleaned),
                    is_remote=is_remote,
                )
            )
        for match in HTML_IMAGE_RE.finditer(line):
            raw_attrs = match.group("attrs")
            attrs = parse_html_attrs(raw_attrs)
            target = attrs.get("src", "")
            cleaned = clean_target(target)
            is_remote = is_remote_target(cleaned)
            width = None
            if "width" in attrs:
                try:
                    width = int(re.sub(r"\D.*$", "", attrs["width"]))
                except ValueError:
                    width = None
            refs.append(
                ImageRef(
                    page=page,
                    line=line_no,
                    raw=match.group(0),
                    alt=attrs.get("alt", ""),
                    target=cleaned,
                    kind="html",
                    display_width=width,
                    classes=tuple(sorted(attrs.get("class", "").split())),
                    resolved=None if is_remote else safe_resolve(rendered_page_dir(page), cleaned),
                    is_remote=is_remote,
                )
            )
    return refs


def svg_dimensions(path: Path) -> tuple[int | None, int | None]:
    text = path.read_text(encoding="utf-8", errors="ignore")[:12000]
    viewbox = SVG_VIEWBOX_RE.search(text)
    if viewbox:
        return int(float(viewbox.group("width"))), int(float(viewbox.group("height")))
    size = SVG_SIZE_RE.search(text)
    if size and size.group("width") and size.group("height"):
        return int(float(size.group("width"))), int(float(size.group("height")))
    return None, None


def load_image_meta(path: Path) -> ImageMeta:
    meta = ImageMeta(path=path, file_size=path.stat().st_size if path.exists() else 0)
    if path.suffix.lower() == ".svg":
        try:
            meta.width, meta.height = svg_dimensions(path)
            meta.mode = "svg"
        except OSError as exc:
            meta.unreadable = str(exc)
        return meta

    try:
        with Image.open(path) as image:
            meta.width, meta.height = image.size
            meta.mode = image.mode
            if image.n_frames if getattr(image, "is_animated", False) else 1:
                image.seek(0)
            rgb = image.convert("RGB")
            meta.edge_touch_score = edge_touch_score(rgb)
            meta.whitespace_fraction = whitespace_fraction(rgb)
            meta.bottom_text_band_score = bottom_text_band_score(rgb)
    except Exception as exc:  # noqa: BLE001 - report corrupt or unsupported files.
        meta.unreadable = str(exc)
    return meta


def downsample_gray(image: Image.Image, max_side: int = 900) -> Image.Image:
    image = image.convert("L")
    width, height = image.size
    scale = min(1.0, max_side / max(width, height))
    if scale < 1.0:
        image = image.resize((max(1, int(width * scale)), max(1, int(height * scale))))
    return image


def content_bbox(image: Image.Image, tolerance: int = 16) -> tuple[int, int, int, int] | None:
    gray = downsample_gray(image)
    bg = Image.new("L", gray.size, 255)
    diff = ImageChops.difference(gray, bg)
    mask = diff.point(lambda value: 255 if value > tolerance else 0)
    return mask.getbbox()


def edge_touch_score(image: Image.Image) -> float:
    gray = downsample_gray(image)
    bg = Image.new("L", gray.size, 255)
    mask = ImageChops.difference(gray, bg).point(lambda value: 1 if value > 18 else 0)
    width, height = gray.size
    band = max(2, int(min(width, height) * 0.018))
    edge_pixels = 0
    edge_total = 0
    samples = [
        (0, 0, width, band),
        (0, height - band, width, height),
        (0, 0, band, height),
        (width - band, 0, width, height),
    ]
    for box in samples:
        crop = mask.crop(box)
        edge_pixels += sum(crop.getdata())
        edge_total += crop.size[0] * crop.size[1]
    return edge_pixels / edge_total if edge_total else 0.0


def whitespace_fraction(image: Image.Image) -> float:
    gray = downsample_gray(image)
    mask = ImageOps.invert(gray).point(lambda value: 1 if value > 18 else 0)
    total = mask.size[0] * mask.size[1]
    content = sum(mask.getdata())
    return 1.0 - (content / total if total else 0.0)


def bottom_text_band_score(image: Image.Image) -> float:
    gray = downsample_gray(image)
    width, height = gray.size
    if height < 80 or width < 240:
        return 0.0
    band_h = max(24, int(height * 0.08))
    band = gray.crop((0, height - band_h, width, height))
    top = gray.crop((0, 0, width, band_h))
    band_edges = ImageOps.autocontrast(band.filter(ImageFilterCompat.FIND_EDGES)).point(
        lambda value: 1 if value > 42 else 0
    )
    top_edges = ImageOps.autocontrast(top.filter(ImageFilterCompat.FIND_EDGES)).point(
        lambda value: 1 if value > 42 else 0
    )
    band_density = sum(band_edges.getdata()) / (width * band_h)
    top_density = sum(top_edges.getdata()) / (width * band_h)
    return max(0.0, band_density - top_density)


class ImageFilterCompat:
    try:
        from PIL import ImageFilter

        FIND_EDGES = ImageFilter.FIND_EDGES
    except Exception:  # pragma: no cover
        FIND_EDGES = None


def image_complexity(ref: ImageRef, meta: ImageMeta | None) -> str:
    name = " ".join([ref.alt, ref.target, " ".join(ref.classes)]).lower()
    text_signals = [
        "architecture",
        "framework",
        "pipeline",
        "overview",
        "algorithm",
        "attention",
        "transformer",
        "system",
        "dataflow",
        "figure-1",
        "figure-2",
        "figure-3",
        "table",
        "survey",
        "skeleton",
        "mask",
        "graph",
        "model",
        "流程",
        "结构",
        "框架",
        "原论文图",
    ]
    if any(signal in name for signal in text_signals):
        return "complex"
    if meta and meta.width and meta.height:
        area = meta.width * meta.height
        aspect = meta.width / max(1, meta.height)
        if area > 1_600_000 or aspect > 2.4 or aspect < 0.58:
            return "complex"
    return "simple"


def has_nearby_source(lines: list[str], index: int) -> bool:
    window = "\n".join(lines[index : min(len(lines), index + 32)])
    return bool(SOURCE_RE.search(window))


def has_nearby_note(lines: list[str], index: int) -> bool:
    after = "\n".join(lines[index : min(len(lines), index + 32)])
    before = "\n".join(lines[max(0, index - 8) : index + 1])
    return bool(NOTE_RE.search(after) or NOTE_RE.search(before) or PROSE_NOTE_RE.search(after))


def analyze_refs(audit: Audit) -> None:
    page_lines_cache: dict[Path, list[str]] = {}
    for ref in audit.refs:
        if ref.page not in page_lines_cache:
            page_lines_cache[ref.page] = ref.page.read_text(encoding="utf-8", errors="ignore").splitlines()
        lines = page_lines_cache[ref.page]
        line_index = max(0, ref.line - 1)

        if ref.is_remote:
            audit.issues.append(
                Issue(
                    "info",
                    "remote-image",
                    ref.target,
                    "remote image cannot be checked for crop or intrinsic resolution locally",
                    page=ref.page,
                    line=ref.line,
                )
            )
            continue

        if ref.resolved is None or not ref.resolved.exists():
            audit.issues.append(
                Issue(
                    "high",
                    "missing-file",
                    ref.target,
                    "local image reference does not resolve",
                    page=ref.page,
                    line=ref.line,
                )
            )
            continue

        meta = audit.metas.get(ref.resolved)
        if meta and meta.unreadable:
            audit.issues.append(
                Issue(
                    "high",
                    "unreadable-image",
                    ref.resolved.as_posix(),
                    meta.unreadable,
                    page=ref.page,
                    line=ref.line,
                    image=ref.resolved,
                )
            )
            continue

        if not has_nearby_source(lines, line_index):
            audit.issues.append(
                Issue(
                    "medium",
                    "missing-source-caption",
                    ref.resolved.as_posix(),
                    "image has no nearby source caption",
                    page=ref.page,
                    line=ref.line,
                    image=ref.resolved,
                )
            )

        complexity = image_complexity(ref, meta)
        if complexity == "complex" and not has_nearby_note(lines, line_index):
            audit.issues.append(
                Issue(
                    "medium",
                    "complex-without-note",
                    ref.resolved.as_posix(),
                    "complex figure should have a nearby 图解/怎么读 explanation box",
                    page=ref.page,
                    line=ref.line,
                    image=ref.resolved,
                )
            )

        if meta and meta.width and meta.height:
            declared = ref.display_width
            effective_width = declared or min(meta.width, 920)
            min_intrinsic = 1.2 if complexity == "complex" else 1.1
            if declared and meta.width < declared:
                audit.issues.append(
                    Issue(
                        "high",
                        "upscaled-for-display",
                        ref.resolved.as_posix(),
                        f"{meta.width}x{meta.height}px displayed at {declared}px",
                        page=ref.page,
                        line=ref.line,
                        image=ref.resolved,
                    )
                )
            elif declared and meta.width < declared * min_intrinsic and meta.width < 1400:
                audit.issues.append(
                    Issue(
                        "medium",
                        "low-resolution-for-display",
                        ref.resolved.as_posix(),
                        f"{meta.width}x{meta.height}px displayed at {declared}px; acceptable but not high-DPI",
                        page=ref.page,
                        line=ref.line,
                        image=ref.resolved,
                    )
                )
            compact_classes = {"atlas-figure-tall", "atlas-figure-compact"}
            if complexity == "complex" and effective_width < 620 and not compact_classes.intersection(ref.classes):
                audit.issues.append(
                    Issue(
                        "medium",
                        "display-too-small",
                        ref.resolved.as_posix(),
                        f"complex figure displayed at {effective_width}px; target 760-920px unless intentionally tall",
                        page=ref.page,
                        line=ref.line,
                        image=ref.resolved,
                    )
                )
            if meta.width < 520 or meta.height < 220:
                audit.issues.append(
                    Issue(
                        "medium",
                        "small-intrinsic-size",
                        ref.resolved.as_posix(),
                        f"intrinsic size is only {meta.width}x{meta.height}px",
                        page=ref.page,
                        line=ref.line,
                        image=ref.resolved,
                    )
                )
            if meta.edge_touch_score is not None and meta.edge_touch_score > 0.24:
                audit.issues.append(
                    Issue(
                        "medium",
                        "content-touches-edge",
                        ref.resolved.as_posix(),
                        f"edge content density {meta.edge_touch_score:.2f}; check for over-tight crop",
                        page=ref.page,
                        line=ref.line,
                        image=ref.resolved,
                    )
                )
            if (
                meta.bottom_text_band_score is not None
                and meta.bottom_text_band_score > 0.055
                and meta.height > 360
                and complexity == "complex"
            ):
                audit.issues.append(
                    Issue(
                        "low",
                        "possible-caption-residue",
                        ref.resolved.as_posix(),
                        f"bottom text-like band score {meta.bottom_text_band_score:.2f}; inspect crop",
                        page=ref.page,
                        line=ref.line,
                        image=ref.resolved,
                    )
                )
            if meta.whitespace_fraction is not None and meta.whitespace_fraction > 0.82 and complexity == "complex":
                audit.issues.append(
                    Issue(
                        "low",
                        "excessive-whitespace",
                        ref.resolved.as_posix(),
                        f"estimated whitespace {meta.whitespace_fraction:.0%}; inspect crop and scaling",
                        page=ref.page,
                        line=ref.line,
                        image=ref.resolved,
                    )
                )


def analyze_unreferenced_assets(audit: Audit) -> None:
    referenced = {ref.resolved for ref in audit.refs if ref.resolved is not None}
    for path in iter_image_assets():
        resolved = path.resolve()
        if resolved not in referenced:
            audit.issues.append(
                Issue(
                    "info",
                    "unreferenced-asset",
                    path.as_posix(),
                    "image asset is not referenced by a non-roadmap markdown page",
                    image=path,
                )
            )


def issue_line(issue: Issue) -> str:
    loc = ""
    if issue.page:
        loc = f"{issue.page.as_posix()}:{issue.line or 1}: "
    return f"[{issue.severity}] {issue.code}: {loc}{issue.subject} — {issue.detail}"


def representative_issues(issues: Iterable[Issue], limit: int) -> list[Issue]:
    selected: list[Issue] = []
    seen_images: set[Path] = set()
    for issue in sorted(issues, key=lambda item: item.sort_key):
        if issue.image and issue.image in seen_images:
            continue
        if issue.image:
            seen_images.add(issue.image)
        selected.append(issue)
        if len(selected) >= limit:
            break
    return selected


def top_issue_codes(issues: Iterable[Issue], *, limit: int) -> list[tuple[str, int]]:
    counts = Counter(issue.code for issue in issues)
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]


def top_issue_pages(issues: Iterable[Issue], *, limit: int) -> list[tuple[Path, int, Counter[str]]]:
    page_counts: Counter[Path] = Counter()
    code_by_page: dict[Path, Counter[str]] = defaultdict(Counter)
    for issue in issues:
        if not issue.page:
            continue
        page_counts[issue.page] += 1
        code_by_page[issue.page][issue.code] += 1
    return [
        (page, count, code_by_page[page])
        for page, count in page_counts.most_common(limit)
    ]


def top_remote_domains(refs: Iterable[ImageRef], *, limit: int) -> list[tuple[str, int]]:
    domains = Counter()
    for ref in refs:
        if not ref.is_remote:
            continue
        parsed = urlparse(ref.target)
        domains[parsed.netloc or parsed.scheme or ref.target] += 1
    return sorted(domains.items(), key=lambda item: (-item[1], item[0]))[:limit]


def is_localized_remote_ref(ref: ImageRef) -> bool:
    if ref.resolved is None:
        return False
    try:
        rel = ref.resolved.relative_to(Path.cwd().resolve())
    except ValueError:
        rel = ref.resolved
    return "localized-remote" in rel.parts


def unreferenced_asset_issues(issues: Iterable[Issue]) -> list[Issue]:
    return [issue for issue in issues if issue.code == "unreferenced-asset" and issue.image]


def classify_unreferenced_asset(path: Path) -> str:
    try:
        rel = path.relative_to(ASSETS_ROOT)
    except ValueError:
        rel = path
    rel_text = rel.as_posix()
    if rel_text.startswith("training/"):
        return "历史公共概念图"
    if rel_text.startswith("localized-remote/"):
        return "疑似替换残留"
    if rel_text.startswith("paper-deep-dives/") or rel_text.startswith("paper-figures/"):
        return "可复用候选"
    return "待人工归类"


def asset_bucket(path: Path) -> str:
    try:
        rel = path.relative_to(ASSETS_ROOT)
    except ValueError:
        rel = path
    parts = rel.parts
    if not parts:
        return "."
    if parts[0] in {"paper-deep-dives", "localized-remote"} and len(parts) >= 3:
        return "/".join(parts[:3])
    if parts[0] == "paper-figures" and len(parts) >= 2:
        return "/".join(parts[:2])
    return parts[0]


def print_asset_governance_summary(audit: Audit) -> None:
    remote_refs = [ref for ref in audit.refs if ref.is_remote]
    localized_refs = [ref for ref in audit.refs if is_localized_remote_ref(ref)]
    unreferenced = unreferenced_asset_issues(audit.issues)
    class_counts = Counter(
        classify_unreferenced_asset(issue.image)
        for issue in unreferenced
        if issue.image is not None
    )
    bucket_counts = Counter(
        asset_bucket(issue.image)
        for issue in unreferenced
        if issue.image is not None
    )

    print("Asset governance:")
    print(f"  remote-image refs: {len(remote_refs)}")
    print(f"  localized-remote refs: {len(localized_refs)}")
    print(f"  unreferenced assets: {len(unreferenced)}")
    if class_counts:
        print("  unreferenced classes:")
        for label, count in sorted(class_counts.items(), key=lambda item: (-item[1], item[0])):
            print(f"    {label}: {count}")
    if bucket_counts:
        print("  top unreferenced directories:")
        for bucket, count in sorted(bucket_counts.items(), key=lambda item: (-item[1], item[0]))[:10]:
            print(f"    {bucket}: {count}")


def draw_contact_sheet(issues: list[Issue], output_dir: Path, limit: int = 80) -> Path | None:
    selected = [
        issue
        for issue in representative_issues(
            (item for item in issues if item.image and item.image.suffix.lower() != ".svg"),
            limit=limit,
        )
        if issue.image and issue.image.exists()
    ]
    if not selected:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    thumb_w, thumb_h = 260, 190
    label_h = 62
    gap = 14
    cols = 3
    rows = math.ceil(len(selected) / cols)
    sheet = Image.new(
        "RGB",
        (cols * thumb_w + (cols + 1) * gap, rows * (thumb_h + label_h) + (rows + 1) * gap),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for index, issue in enumerate(selected):
        assert issue.image is not None
        col = index % cols
        row = index // cols
        x = gap + col * (thumb_w + gap)
        y = gap + row * (thumb_h + label_h)
        try:
            with Image.open(issue.image) as image:
                image.thumbnail((thumb_w, thumb_h))
                tile = Image.new("RGB", (thumb_w, thumb_h), (246, 246, 246))
                pos = ((thumb_w - image.width) // 2, (thumb_h - image.height) // 2)
                tile.paste(image.convert("RGB"), pos)
                sheet.paste(tile, (x, y))
        except Exception:
            draw.rectangle((x, y, x + thumb_w, y + thumb_h), outline=(180, 40, 40), width=2)
        text = f"{issue.severity}:{issue.code}\n{issue.image.name}\n{issue.page.name if issue.page else ''}:{issue.line or ''}"
        draw.multiline_text((x, y + thumb_h + 6), text[:150], fill=(30, 45, 48), spacing=2)

    output = output_dir / "image-quality-contact-sheet.jpg"
    sheet.save(output, quality=88)
    return output


def audit_site(args: argparse.Namespace) -> Audit:
    audit = Audit()
    for page in iter_markdown_pages():
        audit.refs.extend(extract_refs(page))

    local_paths = sorted({ref.resolved for ref in audit.refs if ref.resolved is not None and ref.resolved.exists()})
    for path in local_paths:
        audit.metas[path] = load_image_meta(path)

    analyze_refs(audit)
    analyze_unreferenced_assets(audit)
    if args.contact_sheet:
        audit.contact_sheet = draw_contact_sheet(
            audit.issues,
            Path(args.contact_sheet_dir),
            limit=args.contact_sheet_limit,
        )
    return audit


def print_report(audit: Audit, *, limit: int | None) -> None:
    counts: dict[str, int] = {}
    code_counts: dict[str, int] = {}
    for issue in audit.issues:
        counts[issue.severity] = counts.get(issue.severity, 0) + 1
        code_counts[issue.code] = code_counts.get(issue.code, 0) + 1

    local_refs = [ref for ref in audit.refs if not ref.is_remote]
    remote_refs = [ref for ref in audit.refs if ref.is_remote]
    print(f"Scanned {len(audit.refs)} image refs ({len(local_refs)} local, {len(remote_refs)} remote).")
    print(f"Loaded metadata for {len(audit.metas)} referenced local image assets.")
    print(
        "Issues by severity: "
        + ", ".join(f"{severity}={counts.get(severity, 0)}" for severity in ["high", "medium", "low", "info"])
    )
    print("Top issue codes:")
    for code, count in top_issue_codes(audit.issues, limit=16):
        print(f"  {code}: {count}")
    print("Top medium/high pages:")
    medium_or_high = [issue for issue in audit.issues if issue.severity in {"high", "medium"}]
    for page, count, page_codes in top_issue_pages(medium_or_high, limit=16):
        code_summary = ", ".join(
            f"{code}:{code_count}"
            for code, code_count in sorted(page_codes.items(), key=lambda item: (-item[1], item[0]))[:5]
        )
        print(f"  {count}: {page.as_posix()} ({code_summary})")
    print("Top remote domains:")
    for domain, count in top_remote_domains(audit.refs, limit=10):
        print(f"  {domain}: {count}")
    print_asset_governance_summary(audit)
    if audit.contact_sheet:
        print(f"Contact sheet: {audit.contact_sheet}")
    print()

    sorted_issues = sorted(audit.issues, key=lambda item: item.sort_key)
    if limit is not None:
        sorted_issues = sorted_issues[:limit]
    for issue in sorted_issues:
        print(issue_line(issue))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=220, help="maximum issue lines to print; use 0 for all")
    parser.add_argument("--fail-on-high", action="store_true", help="return non-zero if high severity issues remain")
    parser.add_argument(
        "--contact-sheet",
        action="store_true",
        help="write a representative issue contact sheet outside the repository",
    )
    parser.add_argument(
        "--contact-sheet-dir",
        default="/private/tmp/charles-ai-notes-image-quality",
        help="directory for contact-sheet output",
    )
    parser.add_argument("--contact-sheet-limit", type=int, default=80)
    args = parser.parse_args(argv)

    audit = audit_site(args)
    print_report(audit, limit=None if args.limit == 0 else args.limit)
    if args.fail_on_high and any(issue.severity == "high" for issue in audit.issues):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
