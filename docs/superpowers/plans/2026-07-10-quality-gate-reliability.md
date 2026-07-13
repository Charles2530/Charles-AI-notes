# Quality Gate Reliability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove current false failures, blind spots, and hot-reload stale state, then make pull requests and `main` use one reproducible `make check` gate.

**Architecture:** Keep each audit script focused. Extract small pure parsers and classifiers, verify them with temporary fixtures, and let the Makefile compose only checks whose failure semantics are trustworthy. The image report remains a separate strict diagnostic until real high-severity raster findings are resolved.

**Tech Stack:** Python 3.11, pytest/unittest, Pillow, MkDocs, GNU Make, GitHub Actions.

---

## Source of Truth and File Ownership

This plan implements the quality portion of `docs/superpowers/specs/2026-07-10-personal-ai-tutorial-design.md`.

It owns:

```text
tests/test_zotero_deepdives.py
scripts/repro_status_audit.py
tests/test_repro_status_audit.py
scripts/image_quality_report.py
tests/test_image_quality_report.py
hooks/paper_deep_dives_sort.py
tests/test_paper_deep_dives_sort.py
files/paper-deep-dives/world-models/bagel.md
files/paper-deep-dives/world-models/emu3-5.md
Makefile
requirements.txt
.github/workflows/deploy-docs.yml
README.md
```

The information-architecture plan owns `scripts/site_structure_audit.py` and its tests. This plan only wires that script into `make check` after it exists. It must not edit `mkdocs.yml`, the homepage, topic indexes, prerequisite math, foundations, DM0.5, or thinking-exploration articles.

## Shared Working-Tree Rule

The checkout contains user-owned concurrent edits. Never reset, stash, discard, or broadly stage them. Reread any owned file before editing. Use exact-path staging only, and do not commit unless the user separately authorizes it. Before an authorized commit, run:

```bash
git diff --cached --name-only
```

The staged output must match the task boundary exactly.

### Task 1: Capture the failure baseline

**Files:**
- Read: `tests/test_zotero_deepdives.py`
- Read: `scripts/repro_status_audit.py`
- Read: `scripts/image_quality_report.py`
- Read: `hooks/paper_deep_dives_sort.py`
- Read: `Makefile`
- Read: `.github/workflows/deploy-docs.yml`

- [ ] **Step 1: Run the complete current test suite**

Run:

```bash
pytest -q
```

Expected baseline:

```text
2 failed, 21 passed
```

The two failures are the PlaNet wording assertions in `tests/test_zotero_deepdives.py`.

- [ ] **Step 2: Confirm the reproduction audit's false green**

Run:

```bash
python scripts/repro_status_audit.py --strict
```

Expected before Task 3: `Repro status audit passed.` even though `fact_row()` hard-codes `2026-05-16` and unknown table statuses cannot reach the unknown-status branch.

- [ ] **Step 3: Confirm the hook caches file content**

Run:

```bash
rg -n '@lru_cache|def on_nav|def _sync_page_titles' hooks/paper_deep_dives_sort.py
```

Expected: both file readers are unbounded caches, `on_nav()` does not clear them, and title synchronization is not limited to paper pages.

- [ ] **Step 4: Record dependency and CI versions**

Run:

```bash
cat requirements.txt
cat environment.yml
rg -n 'python-version|mkdocs build|make check|pull_request' .github/workflows/deploy-docs.yml
```

Expected: environment Python is 3.11, CI is 3.13, pytest is undeclared, and CI only runs a strict MkDocs build on `main`.

### Task 2: Decouple Zotero extraction tests from one PlaNet sentence

**Files:**
- Modify: `tests/test_zotero_deepdives.py`

Do not change `scripts/zotero_deepdives.py`; current behavior selects the correct source paragraph. Only the assertion is brittle.

- [ ] **Step 1: Import HTML decoding**

Add:

```python
import html
```

- [ ] **Step 2: Replace the single-page literal assertion**

In `test_actual_planet_page_uses_opening_problem_as_motivation`, replace the assertion for `提出的问题很直接` with semantic source assertions:

```python
motivation = html.unescape(
    note.split("<p><b>动机：</b>", 1)[1].split("</p>", 1)[0]
)
planet_text = zd.strip_markdown(page.read_text(encoding="utf-8"))

self.assertIn(motivation, planet_text)
self.assertNotEqual(motivation, meta.abstract)
self.assertNotIn("| 维度 |", motivation)
```

This proves the note used a real PlaNet prose paragraph, not the supplied arXiv abstract or a comparison table, without freezing article wording.

- [ ] **Step 3: Replace the multi-page literal assertion**

In `test_summary_note_uses_primary_page_when_arxiv_is_related_elsewhere`, use:

```python
motivation = html.unescape(
    note.split("<p><b>动机：</b>", 1)[1].split("</p>", 1)[0]
)
planet_text = zd.strip_markdown(planet.read_text(encoding="utf-8"))
dreamer_text = zd.strip_markdown(dreamer.read_text(encoding="utf-8"))

self.assertIn(motivation, planet_text)
self.assertNotIn(motivation, dreamer_text)
self.assertIn("file://", note)
```

- [ ] **Step 4: Run the two tests**

Run:

```bash
pytest -q tests/test_zotero_deepdives.py::ZoteroDeepdivesTests::test_actual_planet_page_uses_opening_problem_as_motivation tests/test_zotero_deepdives.py::ZoteroDeepdivesTests::test_summary_note_uses_primary_page_when_arxiv_is_related_elsewhere
```

Expected: `2 passed`.

- [ ] **Step 5: Run the full existing suite**

Run:

```bash
pytest -q
```

Expected at this point: `23 passed`.

**Authorized commit boundary:**

```bash
git add tests/test_zotero_deepdives.py
git diff --cached --name-only
git commit -m "test: decouple Zotero checks from PlaNet wording"
```

### Task 3: Parse structured reproduction statuses by header

**Files:**
- Create: `tests/test_repro_status_audit.py`
- Modify: `scripts/repro_status_audit.py`
- Modify: `files/paper-deep-dives/world-models/bagel.md`
- Modify: `files/paper-deep-dives/world-models/emu3-5.md`

Controlled statuses apply to Markdown table cells under a `Repro Status` header. Free-form prose such as `Repro Status：Paper / official materials reviewed...` remains an explanatory evidence sentence and is not treated as a controlled cell.

- [ ] **Step 1: Write four parser tests**

Create temporary pages and test:

```text
test_fact_status_accepts_current_checked_date
test_fact_status_uses_header_position_in_three_column_table
test_official_source_uses_header_position
test_unknown_table_status_is_reported
```

Use fixtures equivalent to:

```python
def test_fact_status_accepts_current_checked_date(self):
    text = """
| Checked Date | Official Source | Repro Status | Notes |
| --- | --- | --- | --- |
| 2026-07-10 | https://arxiv.org/abs/2601.00001 | Paper Only | reviewed |
"""
    self.assertEqual(fact_status(text), "Paper Only")
    self.assertEqual(self._reasons(text), [])


def test_fact_status_uses_header_position_in_three_column_table(self):
    text = """
| Checked Date | Repro Status | 本页证据口径 |
| --- | --- | --- |
| 2026-06-02 | Author Code / Official Repo | GitHub repo reviewed |
"""
    self.assertEqual(fact_status(text), "Author Code / Official Repo")
    self.assertEqual(self._reasons(text), [])


def test_unknown_table_status_is_reported(self):
    text = """
| Claim | Repro Status | Can Support |
| --- | --- | --- |
| result | Community Reproduced | fixture |
"""
    self.assertEqual(table_repro_statuses(text), {"Community Reproduced"})
    self.assertIn(
        "unknown table Repro Status: Community Reproduced",
        self._reasons(text),
    )
```

The official-source test places `Official Source` in a non-default column and verifies a missing external marker is still reported.

- [ ] **Step 2: Run the new tests red**

Run:

```bash
pytest -q tests/test_repro_status_audit.py
```

Expected: failures expose the fixed date, fixed column positions, and unreachable unknown-status detection.

- [ ] **Step 3: Implement generic Markdown-table parsing**

Add:

```python
def table_cells(line: str) -> list[str]:
    return [cell.strip().strip("`") for cell in line.strip().strip("|").split("|")]


def is_separator_row(cells: list[str]) -> bool:
    return bool(cells) and all(
        re.fullmatch(r":?-{3,}:?", cell) for cell in cells
    )


def repro_status_rows(text: str) -> list[dict[str, str]]:
    lines = text.splitlines()
    rows: list[dict[str, str]] = []
    for index, line in enumerate(lines[:-1]):
        if not line.lstrip().startswith("|"):
            continue
        headers = table_cells(line)
        if "Repro Status" not in headers:
            continue
        separator = table_cells(lines[index + 1])
        if not is_separator_row(separator):
            continue
        for data_line in lines[index + 2:]:
            if not data_line.lstrip().startswith("|"):
                break
            cells = table_cells(data_line)
            cells.extend([""] * (len(headers) - len(cells)))
            rows.append(dict(zip(headers, cells)))
    return rows
```

- [ ] **Step 4: Replace position-based status functions**

Use:

```python
def fact_row(text: str) -> dict[str, str] | None:
    return next(
        (row for row in repro_status_rows(text) if "Checked Date" in row),
        None,
    )


def fact_status(text: str) -> str | None:
    row = fact_row(text)
    return row.get("Repro Status") if row else None


def table_repro_statuses(text: str) -> set[str]:
    return {
        row["Repro Status"]
        for row in repro_status_rows(text)
        if row.get("Repro Status")
    }


def has_independent_reproduced_data_row(text: str) -> bool:
    return "Independent Reproduced" in table_repro_statuses(text)
```

In `audit_page()`, retrieve `Official Source` by header name rather than `row[1]`. Only run that marker check when the fact table actually has an `Official Source` column.

- [ ] **Step 5: Run the parser tests green**

Run:

```bash
pytest -q tests/test_repro_status_audit.py
```

Expected: four tests pass.

- [ ] **Step 6: Expose real repository findings**

Run:

```bash
python scripts/repro_status_audit.py --strict
```

Expected before normalization: findings for `bagel.md` status `Author code / checkpoints reported` and `emu3-5.md` status `Author code / model release reported`.

- [ ] **Step 7: Normalize only the verified table cells**

Both pages link an official GitHub repository or official model material. Replace those two table-cell values with:

```text
Author Code / Official Repo
```

Do not change the prose evidence sentence or claim independent reproduction.

- [ ] **Step 8: Run the strict audit**

Run:

```bash
python scripts/repro_status_audit.py --strict
```

Expected: `Repro status audit passed.`

**Authorized commit boundary:**

```bash
git add scripts/repro_status_audit.py tests/test_repro_status_audit.py files/paper-deep-dives/world-models/bagel.md files/paper-deep-dives/world-models/emu3-5.md
git diff --cached --name-only
git commit -m "fix: parse reproduction tables by header"
```

### Task 4: Separate vector graphics from raster-resolution findings

**Files:**
- Create: `tests/test_image_quality_report.py`
- Modify: `scripts/image_quality_report.py`

The intended policy is:

- SVG viewBox dimensions are coordinate systems, not pixels; never emit intrinsic-resolution or upscaling findings for SVG.
- A raster displayed at no more than 5% above intrinsic width receives a medium high-DPI warning, not a high-severity upscaling error.
- Raster enlargement above 5% remains high severity.
- `display-too-small` still applies to complex SVG because it evaluates rendered width, not intrinsic pixel density.

- [ ] **Step 1: Add three temporary-asset tests**

Construct `ImageRef` values directly and use `load_image_meta()` plus `analyze_refs()`:

```text
test_svg_viewbox_is_not_treated_as_raster_resolution
test_minor_raster_upscale_is_not_high_severity
test_material_raster_upscale_remains_high_severity
```

Use these cases:

```text
SVG viewBox 400x200 displayed at 620
PNG 915x400 displayed at 920
PNG 800x400 displayed at 920
```

For the SVG, reject `upscaled-for-display`, `low-resolution-for-display`, and `small-intrinsic-size`. For 915→920, require medium `low-resolution-for-display` and reject high upscaling. For 800→920, require high `upscaled-for-display`.

- [ ] **Step 2: Run tests red**

Run:

```bash
pytest -q tests/test_image_quality_report.py
```

Expected: the SVG and minor-raster cases fail under current logic.

- [ ] **Step 3: Add an explicit tolerance**

At module scope add:

```python
UPSCALE_TOLERANCE = 1.05
```

- [ ] **Step 4: Gate intrinsic checks on raster images**

Inside `analyze_refs()` use:

```python
is_raster = meta.mode != "svg"

if declared and is_raster and declared > meta.width * UPSCALE_TOLERANCE:
    # existing high upscaled-for-display issue
elif (
    declared
    and is_raster
    and meta.width < declared * min_intrinsic
    and meta.width < 1400
):
    # existing medium low-resolution-for-display issue

if is_raster and (meta.width < 520 or meta.height < 220):
    # existing medium small-intrinsic-size issue
```

Keep `display-too-small`, source captions, explanatory notes, and crop heuristics unchanged.

- [ ] **Step 5: Run tests green**

Run:

```bash
pytest -q tests/test_image_quality_report.py
```

Expected: three tests pass.

- [ ] **Step 6: Review the real high findings without gating CI**

Run:

```bash
python scripts/image_quality_report.py --fail-on-high
```

Expected: SVG and ≤5% cases disappear from high severity. Real raster enlargement such as 793→920 remains high, so this command may exit 1. Record real findings; do not edit concurrent untracked assets or add image strictness to `make check` in this plan.

**Authorized commit boundary:**

```bash
git add scripts/image_quality_report.py tests/test_image_quality_report.py
git diff --cached --name-only
git commit -m "fix: classify vector and raster image resolution"
```

### Task 5: Make paper navigation safe for live reload and custom labels

**Files:**
- Modify: `tests/test_paper_deep_dives_sort.py`
- Modify: `hooks/paper_deep_dives_sort.py`

This task owns the hook completely so the information-architecture plan does not edit it.

- [ ] **Step 1: Extend imports for fixtures**

Import `SimpleNamespace` and these functions:

```python
from hooks.paper_deep_dives_sort import (
    _h1_title_from_file,
    _paper_date_from_file,
    on_nav,
)
```

- [ ] **Step 2: Test cache invalidation in one process**

Create a temporary `paper.md` with `paper_date: 2023-09` and `# Old title`. Read both cached values, overwrite it with `paper_date: 2024-10` and `# New title`, call `on_nav()` with an empty navigation object, then require both readers to return the new values.

The test name is:

```text
test_on_nav_invalidates_file_caches_for_livereload
```

- [ ] **Step 3: Test that non-paper custom labels survive**

Create two nav leaf fixtures:

```text
index.md                         title="首页"
paper-deep-dives/world/paper.md title="导航中的论文标题"
```

Give each source a different H1. After `on_nav()`, require the homepage title to remain `首页` and the paper title to match its H1. Use the test name:

```text
test_on_nav_syncs_only_paper_page_titles
```

- [ ] **Step 4: Run tests red**

Run:

```bash
pytest -q tests/test_paper_deep_dives_sort.py
```

Expected: the cache and non-paper-label tests fail.

- [ ] **Step 5: Clear both caches at every navigation rebuild**

At the start of `on_nav()` add:

```python
_paper_date_from_file.cache_clear()
_h1_title_from_file.cache_clear()
```

This is more reliable than mtime-based invalidation during rapid saves.

- [ ] **Step 6: Restrict H1 synchronization to paper pages**

In `_sync_page_titles()`, after obtaining `src_path`, skip every leaf for which `_is_paper_deep_dive_page(item)` is false. Keep recursive traversal and paper sorting unchanged. `paper-deep-dives/index.md` retains its configured label because it is intentionally excluded by `_is_paper_deep_dive_page()`.

- [ ] **Step 7: Run tests green**

Run:

```bash
pytest -q tests/test_paper_deep_dives_sort.py
```

Expected: the existing date-fallback test and both new tests pass.

**Authorized commit boundary:**

```bash
git add hooks/paper_deep_dives_sort.py tests/test_paper_deep_dives_sort.py
git diff --cached --name-only
git commit -m "fix: refresh paper navigation safely"
```

### Task 6: Establish one local Makefile gate and align README claims

**Dependency:** `scripts/site_structure_audit.py` from the information-architecture plan must exist before expecting `make check` to pass.

**Files:**
- Modify: `Makefile`
- Modify: `README.md`

- [ ] **Step 1: Add configurable Python and MkDocs commands**

At the top of `Makefile`, add:

```make
PYTHON ?= python
```

Replace hard-coded `python` invocations in existing targets with `$(PYTHON)`. Change the existing `MKDOCS` preference from `./docs/bin/mkdocs` to `./.venv/bin/mkdocs`:

```make
MKDOCS ?= $(shell [ -x ./.venv/bin/mkdocs ] && echo ./.venv/bin/mkdocs || echo mkdocs)
```

The repository's `docs/` directory stores design and implementation documents and must not double as a Python environment prefix.

- [ ] **Step 2: Add `test`, `repro-quality`, and `check` targets**

Extend `.PHONY` and implement:

```make
test:
	$(PYTHON) -m pytest -q

repro-quality:
	$(PYTHON) scripts/repro_status_audit.py --strict

quality:
	$(PYTHON) scripts/content_quality_report.py
	$(PYTHON) scripts/paper_deep_dive_audit.py
	$(PYTHON) scripts/math_rendering_report.py --fail-on-issues
	$(PYTHON) scripts/repro_status_audit.py --strict
	$(PYTHON) scripts/site_structure_audit.py

check: test quality build
```

Keep `content_quality_report.py` report-only because its historical style warnings are not yet reliable blockers. `paper_deep_dive_audit.py`, math rendering, reproduction status, site structure, tests, and strict build are hard blockers through their exit codes.

Do not add `image-quality` to `check`: Task 4 removes false highs but leaves genuine raster enlargement to resolve separately.

- [ ] **Step 3: Preserve diagnostic targets**

Keep `quality-hints`, `quality-pages`, `image-quality`, `blog-quality`, and topic-specific targets. Update them to use `$(PYTHON)` but do not silently weaken `image-quality --fail-on-high`.

- [ ] **Step 4: Dry-run the composed gate**

Run:

```bash
make -n check
```

Expected order: full pytest, five quality/audit commands, then `mkdocs build --strict` through `make build`.

- [ ] **Step 5: Align the README positioning**

Because this plan is the sole README owner, update the opening to describe a personal layered tutorial that also preserves independent technical articles. Do not claim OCR, chart, grounding, GUI agents, RL intermediate layers, or Agent runtime as complete; label them as coverage gaps where mentioned.

Keep the repository layout accurate: describe `docs/superpowers/` as design and implementation planning material, not a Conda prefix. State Python 3.11 as the supported baseline. Replace `conda env create -f environment.yml -p ./docs` and `conda activate ./docs` with the named-environment workflow:

```bash
conda env create -f environment.yml
conda activate docs
```

Keep the `.venv` workflow for users who prefer venv. Do not instruct users to place an environment inside `docs/`.

- [ ] **Step 6: Document the gate**

Replace the current three-command pre-submit recipe with:

```bash
make check
```

Explain which checks are blocking. Document `make quality-hints` as advisory detail and `make image-quality` as a stricter image diagnostic that can expose unresolved real assets.

- [ ] **Step 7: Run the real gate after structure work is present**

Run:

```bash
make check
```

Expected: exit 0 only after the structure plan registers DM0.5 and all public pages. Before that dependency lands, a structure-audit failure is expected and must not be bypassed.

**Authorized commit boundary:**

```bash
git add Makefile README.md
git diff --cached --name-only
git commit -m "build: add one local quality gate"
```

### Task 7: Align dependencies and GitHub Actions

**Files:**
- Modify: `requirements.txt`
- Modify: `.github/workflows/deploy-docs.yml`

- [ ] **Step 1: Declare the test dependency**

Append:

```text
pytest>=8,<9
```

Do not add a `pyproject.toml` solely for a documentation repository. `environment.yml` already defines Python 3.11 and installs `requirements.txt`.

- [ ] **Step 2: Expand workflow triggers**

Use:

```yaml
on:
  pull_request:
  push:
    branches:
      - main
  workflow_dispatch:
```

- [ ] **Step 3: Restrict default permissions and isolate concurrency**

Set global permissions to `contents: read` and use:

```yaml
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true
```

- [ ] **Step 4: Replace `build` with a Python 3.11 `check` job**

The job must:

1. check out with `actions/checkout@v4`;
2. set up Python 3.11 with `actions/setup-python@v5` and pip caching;
3. run `python -m pip install -r requirements.txt`;
4. run `make check`;
5. upload `site/` with `actions/upload-pages-artifact@v3` only when `github.ref == 'refs/heads/main'`.

- [ ] **Step 5: Gate deployment on the check job**

Keep the Pages environment and deploy action, but configure:

```yaml
deploy:
  if: github.ref == 'refs/heads/main'
  needs: check
  permissions:
    pages: write
    id-token: write
```

On a pull request, `check` runs and `deploy` is skipped. On `main`, deployment can begin only after the complete gate and artifact upload succeed.

- [ ] **Step 6: Verify dependency and workflow text**

Run:

```bash
python -m pytest --version
rg -n 'pull_request|python-version: "3.11"|run: make check|if: github.ref|needs: check' .github/workflows/deploy-docs.yml
git diff --check
```

Expected: pytest reports 8.x, every workflow contract is found, and the diff check has no output.

**Authorized commit boundary:**

```bash
git add requirements.txt .github/workflows/deploy-docs.yml
git diff --cached --name-only
git commit -m "ci: gate pull requests and main"
```

### Task 8: Run the complete acceptance suite

**Files:**
- Verify: all files owned by this plan

- [ ] **Step 1: Run all tests**

Run:

```bash
pytest -q
```

Expected: all existing and newly added tests pass.

- [ ] **Step 2: Run the strict evidence audit**

Run:

```bash
python scripts/repro_status_audit.py --strict
```

Expected: `Repro status audit passed.`

- [ ] **Step 3: Run image classifier regression tests and report**

Run:

```bash
pytest -q tests/test_image_quality_report.py
python scripts/image_quality_report.py
```

Expected: classifier tests pass. The report may list genuine raster enlargement or advisory findings but exits 0 without `--fail-on-high`.

- [ ] **Step 4: Run the unified gate**

Run:

```bash
make check
```

Expected: tests, hard audits, structure audit, and strict build all exit 0.

- [ ] **Step 5: Verify patch integrity and ownership**

Run:

```bash
git diff --check
git status --short
```

Expected: no whitespace errors and no unrelated concurrent file is staged or overwritten.

## Deferred Hard Gates

- Do not add `image_quality_report.py --fail-on-high` to `make check` until every genuine raster high is resolved, including assets owned by concurrent work.
- Do not add `content_quality_report.py --fail-on-warn` while historical long-page and style heuristics remain noisy; first separate hard correctness errors from editorial hints.
- Do not delete unreferenced assets, contact sheets, `.tmp` evidence, or Git history through this plan.
- Do not use broad status replacements. A canonical `Author Code / Official Repo` table cell requires an actual official repository or release marker on that page.

## Completion Boundary

This plan is complete when all tests pass, reproduction tables are parsed by headers, SVG and minor raster upscaling no longer create false high findings, live reload sees current paper metadata, custom non-paper navigation labels survive, `make check` is the local source of truth, and PR/main CI run that same gate under Python 3.11.
