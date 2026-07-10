# Sidebar Short Title Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep complete article, browser, and search titles while showing only the text before the first Chinese colon in the left sidebar.

**Architecture:** Continue using `hooks/paper_deep_dives_sort.py` as the single source of navigation-title synchronization. Preserve the complete Markdown H1 in `item.title`, derive an optional short title with a small pure helper, and store that value in MkDocs Material's navigation-only `item.typeset.title` field.

**Tech Stack:** Python 3.13, MkDocs 1.6, MkDocs Material 9.7, `unittest`/pytest, Markdown/YAML configuration

---

## File map

- `hooks/paper_deep_dives_sort.py`: derive a sidebar-only title and attach it while recursively synchronizing page titles.
- `tests/test_paper_deep_dives_sort.py`: test the split rules and the navigation-item state produced by the hook.
- `.tmp/sidebar-title-site/`: disposable build output used only for verification; do not commit it.

### Task 1: Define the sidebar-title rule

**Files:**
- Modify: `tests/test_paper_deep_dives_sort.py`
- Modify: `hooks/paper_deep_dives_sort.py`

- [ ] **Step 1: Write failing tests for the title conversion rule**

Replace the current hook import with:

```python
from hooks.paper_deep_dives_sort import _paper_date_from_file, _sidebar_title
```

Add these methods to `PaperDeepDivesSortTests`:

```python
    def test_sidebar_title_uses_text_before_first_chinese_colon(self):
        self.assertEqual(
            _sidebar_title("  标题  ：副标题：补充说明"),
            "标题",
        )

    def test_sidebar_title_leaves_other_titles_unchanged(self):
        cases = (
            "没有副标题",
            "English: subtitle",
            "：缺少主标题",
        )

        for title in cases:
            with self.subTest(title=title):
                self.assertIsNone(_sidebar_title(title))
```

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
pytest tests/test_paper_deep_dives_sort.py::PaperDeepDivesSortTests::test_sidebar_title_uses_text_before_first_chinese_colon tests/test_paper_deep_dives_sort.py::PaperDeepDivesSortTests::test_sidebar_title_leaves_other_titles_unchanged -v
```

Expected: collection fails because `_sidebar_title` is not defined or cannot be imported.

- [ ] **Step 3: Implement the minimal pure helper**

Add this function after `_is_paper_deep_dive_page` in `hooks/paper_deep_dives_sort.py`:

```python
def _sidebar_title(title: str) -> str | None:
    prefix, separator, _ = title.partition("：")
    if not separator:
        return None
    return prefix.strip() or None
```

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run:

```bash
pytest tests/test_paper_deep_dives_sort.py::PaperDeepDivesSortTests::test_sidebar_title_uses_text_before_first_chinese_colon tests/test_paper_deep_dives_sort.py::PaperDeepDivesSortTests::test_sidebar_title_leaves_other_titles_unchanged -v
```

Expected: both tests pass.

- [ ] **Step 5: Commit the rule and its tests**

```bash
git add hooks/paper_deep_dives_sort.py tests/test_paper_deep_dives_sort.py
git commit -m "test: define sidebar short-title rule"
```

### Task 2: Attach the short title to navigation pages

**Files:**
- Modify: `tests/test_paper_deep_dives_sort.py`
- Modify: `hooks/paper_deep_dives_sort.py`

- [ ] **Step 1: Write failing recursive navigation tests**

Add this import at the top of `tests/test_paper_deep_dives_sort.py`:

```python
from types import SimpleNamespace
```

Extend the hook import to include `_sync_page_titles`:

```python
from hooks.paper_deep_dives_sort import (
    _paper_date_from_file,
    _sidebar_title,
    _sync_page_titles,
)
```

Add these methods to `PaperDeepDivesSortTests`:

```python
    def test_sync_keeps_full_page_title_and_sets_nested_sidebar_title(self):
        with TemporaryDirectory() as tmpdir:
            docs_dir = Path(tmpdir)
            page_path = docs_dir / "foundations" / "tensors.md"
            page_path.parent.mkdir(parents=True)
            full_title = "张量、Shape 与计算图：为什么很多模型问题先是接口问题"
            page_path.write_text(f"# {full_title}\n", encoding="utf-8")
            page = SimpleNamespace(
                file=SimpleNamespace(src_path="foundations/tensors.md"),
                children=None,
                title="旧导航标题",
            )
            section = SimpleNamespace(children=[page])

            _sync_page_titles([section], docs_dir)

            self.assertEqual(page.title, full_title)
            self.assertEqual(page.typeset, {"title": "张量、Shape 与计算图"})

    def test_sync_does_not_add_typeset_to_title_without_chinese_colon(self):
        with TemporaryDirectory() as tmpdir:
            docs_dir = Path(tmpdir)
            page_path = docs_dir / "foundations" / "signals.md"
            page_path.parent.mkdir(parents=True)
            full_title = "梯度估计、离散变量与训练信号"
            page_path.write_text(f"# {full_title}\n", encoding="utf-8")
            page = SimpleNamespace(
                file=SimpleNamespace(src_path="foundations/signals.md"),
                children=None,
                title="旧导航标题",
            )

            _sync_page_titles([page], docs_dir)

            self.assertEqual(page.title, full_title)
            self.assertFalse(hasattr(page, "typeset"))
```

- [ ] **Step 2: Run the navigation tests and verify RED**

Run:

```bash
pytest tests/test_paper_deep_dives_sort.py::PaperDeepDivesSortTests::test_sync_keeps_full_page_title_and_sets_nested_sidebar_title tests/test_paper_deep_dives_sort.py::PaperDeepDivesSortTests::test_sync_does_not_add_typeset_to_title_without_chinese_colon -v
```

Expected: the first test fails because `page.typeset` is not set; the second test already passes and protects the unchanged case.

- [ ] **Step 3: Set Material's navigation-only title during synchronization**

Change the final block of `_sync_page_titles` to:

```python
        title = _h1_title_from_file(str(docs_dir / src_path))
        if title:
            item.title = title
            sidebar_title = _sidebar_title(title)
            if sidebar_title:
                item.typeset = {"title": sidebar_title}
```

- [ ] **Step 4: Run the complete hook test file and verify GREEN**

Run:

```bash
pytest tests/test_paper_deep_dives_sort.py -v
```

Expected: all tests in the file pass.

- [ ] **Step 5: Commit the hook behavior**

```bash
git add hooks/paper_deep_dives_sort.py tests/test_paper_deep_dives_sort.py
git commit -m "feat: shorten titles only in the sidebar"
```

### Task 3: Verify rendered sidebar, article, browser, and search titles

**Files:**
- Verify: `files/foundations/tensors-shapes-and-computation-graphs.md`
- Verify: `.tmp/sidebar-title-site/foundations/tensors-shapes-and-computation-graphs/index.html`
- Verify: `.tmp/sidebar-title-site/search/search_index.json`

- [ ] **Step 1: Run the full test suite**

Run:

```bash
pytest -q
```

Expected: the suite passes with no failures.

- [ ] **Step 2: Build the site into disposable output**

Run:

```bash
rm -rf .tmp/sidebar-title-site
mkdocs build --strict --site-dir .tmp/sidebar-title-site
```

Expected: MkDocs completes successfully with `Documentation built` and no strict-mode warnings.

- [ ] **Step 3: Assert the four rendered title contexts**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path

site = Path(".tmp/sidebar-title-site")
full = "张量、Shape 与计算图：为什么很多模型问题先是接口问题"
short = "张量、Shape 与计算图"
unchanged = "梯度估计、离散变量与训练信号"
html = (site / "foundations/tensors-shapes-and-computation-graphs/index.html").read_text(encoding="utf-8")
primary_sidebar = html.split('<div class="md-sidebar md-sidebar--primary"', 1)[1].split(
    '<div class="md-sidebar md-sidebar--secondary"', 1
)[0]
search = json.loads((site / "search/search_index.json").read_text(encoding="utf-8"))
page_record = next(
    doc
    for doc in search["docs"]
    if doc["location"] == "foundations/tensors-shapes-and-computation-graphs/"
)

assert short in primary_sidebar
assert full not in primary_sidebar
assert unchanged in primary_sidebar
assert f"<title>{full} - Charles AI notes</title>" in html
assert f">{full}<a class=\"headerlink\"" in html
assert page_record["title"] == full
print("sidebar, article, browser, and search titles verified")
PY
```

Expected output:

```text
sidebar, article, browser, and search titles verified
```

- [ ] **Step 4: Inspect the final diff and repository state**

Run:

```bash
git diff --check
git status --short
```

Expected: `git diff --check` prints nothing. `git status --short` still shows the user's pre-existing unrelated changes, while the hook and test files are clean after their focused commits; `.tmp/sidebar-title-site/` is ignored or remains untracked disposable output and must not be staged.
