# Tutorial Information Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the current collection into a navigable personal tutorial without sacrificing direct access to independent technical articles.

**Architecture:** Keep article URLs stable. Add a thin tutorial layer made of a dual-entry homepage, three learning paths, a coverage matrix, and complete topic indexes. Keep `mkdocs.yml` curated rather than exhaustive, and enforce discoverability with a structure audit that distinguishes navigation, owning-index registration, and true inbound links.

**Tech Stack:** MkDocs Material, Markdown, Python 3.11, pytest, MkDocs configuration APIs, existing JavaScript term tooltips.

---

## Source of Truth and File Ownership

This plan implements the structure portion of `docs/superpowers/specs/2026-07-10-personal-ai-tutorial-design.md` and must coexist with:

- `docs/superpowers/plans/2026-07-10-prerequisite-foundations-curriculum.md`
- `docs/superpowers/plans/2026-07-10-quality-gate-reliability.md`

Ownership is strict:

- The prerequisite/foundations plan owns `files/prerequisite-math/**`, `files/foundations/**`, their assets, and those two navigation subtrees.
- This plan owns the homepage, learning-path pages, ten topic indexes, references index, DM0.5 paper-index registration, VLA tooltip link, structure audit, and final top-level navigation convergence.
- The quality plan owns `hooks/paper_deep_dives_sort.py`, `Makefile`, `requirements.txt`, `.github/workflows/deploy-docs.yml`, `README.md`, and quality-audit fixes.
- RL, VLM-application, and Agent/RAG articles need focused content designs after the coverage matrix records the gaps. This plan must not create thin placeholder articles.

## Shared Working-Tree Rule

The checkout contains user-owned concurrent edits. Never reset, stash, discard, or broadly stage them. Preserve current changes in `prerequisite-math/**`, `foundations/**`, `files/paper-deep-dives/embodied-ai/dm05.md`, `files/thinking-exploration/**`, `files/references/external-reading-ledger.md`, and `mkdocs.yml`.

Before every overlapping edit, reread the live file. Use `git add` only with explicit paths. Do not commit unless the user separately authorizes commits. When authorization exists, always run:

```bash
git diff --cached --name-only
```

The staged set must contain only the files named by that task.

## Structural Contract

The target top-level navigation order is exactly:

```text
首页
学习路径
数学基础
深度学习基础
训练与系统
生成与多模态
决策与物理智能
论文精读
资料库
```

The ten governed topic directories are:

```text
training
inference
quantization
operators
diffusion
vlm
reinforcement-learning
world-models
vla
embodied-ai
```

Every topic index must contain these H2 sections:

```text
专题边界
前置知识
学习目标
推荐阅读顺序
完整文章目录
覆盖矩阵
学完后的检查问题
```

The only coverage statuses are `核心已覆盖`, `需要补强`, `缺失待写`, and `参考案例`.

### Task 1: Capture the live baseline and freeze conflict boundaries

**Files:**
- Read: `mkdocs.yml`
- Read: `files/index.md`
- Read: `files/paper-deep-dives/index.md`
- Read: `files/assets/javascripts/term-tooltips.js`
- Read: all ten governed topic indexes
- Read: `files/references/index.md`

- [ ] **Step 1: Record the dirty tree**

Run:

```bash
git status --short
```

Expected: concurrent prerequisite/foundations and research-observation work is visible; the command changes nothing.

- [ ] **Step 2: Inspect the live navigation diff**

Run:

```bash
git diff -- mkdocs.yml
```

Expected: current prerequisite/foundations additions, DM0.5, and thinking-exploration entries are identifiable and must survive Task 10.

- [ ] **Step 3: Confirm the current paper-index defect**

Run:

```bash
python scripts/paper_deep_dive_audit.py
```

Expected before Task 9: the only index-coverage error is `paper-deep-dives/embodied-ai/dm05.md`. If another error appears, record it and do not hide it in a whitelist.

- [ ] **Step 4: Confirm the current tooltip defect**

Run:

```bash
rg -n 'aliases: \["VLA"\]|href: siteUrl\("vlm/"\)|href: siteUrl\("vla/"\)' files/assets/javascripts/term-tooltips.js
```

Expected before Task 9: the standalone `VLA` alias resolves to `vlm/`.

### Task 2: Build the site-structure audit test-first

**Files:**
- Create: `tests/test_site_structure_audit.py`
- Create: `scripts/site_structure_audit.py`

Expose this implementation surface:

```python
@dataclass(frozen=True)
class Finding:
    code: str
    path: str
    detail: str


@dataclass(frozen=True)
class SiteModel:
    docs_dir: Path
    public_pages: frozenset[str]
    nav_targets: tuple[str, ...]
    inbound_links: dict[str, frozenset[str]]
    index_registrations: dict[str, tuple[str, ...]]


def load_site_model(config_path: Path = Path("mkdocs.yml")) -> SiteModel: ...
def extract_h2_section(text: str, heading: str) -> str: ...
def extract_local_markdown_links(page: Path, docs_dir: Path) -> set[str]: ...
def audit_site(
    model: SiteModel,
    scopes: tuple[str, ...] | None = None,
    allowlist: frozenset[str] | None = None,
) -> list[Finding]: ...
```

Define:

```python
TOPIC_DIRECTORIES = (
    "training", "inference", "quantization", "operators", "diffusion",
    "vlm", "reinforcement-learning", "world-models", "vla", "embodied-ai",
)
AUDIT_SCOPES = TOPIC_DIRECTORIES + (
    "references", "learning-paths", "paper-deep-dives",
)
INTERNAL_PAGE_ALLOWLIST: frozenset[str] = frozenset()
```

The whitelist starts empty; current orphans must receive real index links. When `allowlist` is `None`, `audit_site()` uses `INTERNAL_PAGE_ALLOWLIST`; tests pass an explicit fixture allowlist without mutating module state.

- [ ] **Step 1: Write a temporary MkDocs fixture helper**

The helper writes `mkdocs.yml`, `docs/index.md`, `docs/<topic>/index.md`, and article pages. Every test passes its temporary config path to `load_site_model` and never reads the real repository.

- [ ] **Step 2: Add eight isolated tests**

Use these test names:

```text
test_duplicate_and_missing_nav_targets_are_reported
test_page_linked_from_owning_complete_directory_is_registered
test_unrelated_inbound_link_does_not_replace_topic_registration
test_page_without_nav_or_inbound_link_is_orphaned
test_explicit_allowlist_exempts_internal_page
test_exclude_docs_pages_are_not_public
test_broken_link_in_complete_directory_is_reported
test_duplicate_link_in_complete_directory_is_reported
```

Assert stable codes: `duplicate-nav-target`, `missing-nav-target`, `unregistered-page`, `orphan-page`, `broken-index-link`, `duplicate-index-link`, and `missing-from-complete-directory`.

- [ ] **Step 3: Run the tests red**

Run:

```bash
pytest -q tests/test_site_structure_audit.py
```

Expected: collection fails because `scripts.site_structure_audit` does not exist.

- [ ] **Step 4: Load public files through MkDocs**

Implement `load_site_model()` with `mkdocs.config.load_config(config_file=str(config_path))` and `mkdocs.structure.files.get_files(config)` so `docs_dir` and `exclude_docs` match the build. Public pages are Markdown source paths only; ignore excluded files and AppleDouble entries.

- [ ] **Step 5: Flatten and validate navigation**

Recursively traverse `config["nav"]`. Report repeated targets as `duplicate-nav-target` and targets absent from `public_pages` as `missing-nav-target`.

- [ ] **Step 6: Resolve local Markdown links**

For each public page, remove query strings and fragments, ignore external schemes and fragment-only links, resolve relative paths against the source page, normalize real directory/extensionless links, and reject paths escaping `docs_dir`. Build `inbound_links[target] = {source pages}`.

- [ ] **Step 7: Parse owning-index registrations**

For the ten topics, inspect only the text between `## 完整文章目录` and the next H2. Preserve link order long enough to report a repeated target as `duplicate-index-link`; a link elsewhere does not satisfy topic registration. Also register:

- `learning-paths/coverage.md` from `learning-paths/index.md`;
- every `references/*.md` page under `## 本地来源台账目录` in `references/index.md`;
- paper pages from `paper-deep-dives/index.md`, while retaining the existing paper-specific audit.

- [ ] **Step 8: Implement registration and orphan rules**

A public page is registered by navigation, its owning index, or the explicit whitelist. Report all others as `unregistered-page`. Independently report `orphan-page` when a page has no navigation entry, no public inbound link, and no whitelist entry. An unrelated inbound link may suppress the orphan finding but never the unregistered finding.

- [ ] **Step 9: Add focused CLI modes**

Support:

```bash
python scripts/site_structure_audit.py
python scripts/site_structure_audit.py --topic training
python scripts/site_structure_audit.py --topic references
```

Set `argparse` choices from `AUDIT_SCOPES`, including `references`. Reject unknown scopes. Findings exit 1. A clean run prints `Site structure audit passed.` and exits 0.

- [ ] **Step 10: Run unit tests green, then expose the repository baseline**

Run:

```bash
pytest -q tests/test_site_structure_audit.py
python scripts/site_structure_audit.py
```

Expected: eight unit tests pass; the real audit exits 1 until the index tasks repair current omissions.

**Authorized commit boundary:**

```bash
git add scripts/site_structure_audit.py tests/test_site_structure_audit.py
git diff --cached --name-only
git commit -m "test: enforce public page registration"
```

### Task 3: Encode the information-architecture contract

**Files:**
- Create: `tests/test_site_information_architecture.py`

- [ ] **Step 1: Add reusable heading and link helpers**

Read source Markdown directly; generated `site/` output must not be a test dependency.

- [ ] **Step 2: Test the homepage**

Require exact H2 headings `从头学习`, `按目标学习`, `按问题查阅`, and `当前补全重点`. Require direct links to the math, foundations, learning-path, knowledge-QA, and paper indexes. Reject `\d+\s*篇` anywhere on the homepage.

- [ ] **Step 3: Test the learning paths and coverage vocabulary**

Require `files/learning-paths/index.md` headings `训练与系统`, `生成与多模态`, and `决策与物理智能`, plus a direct `coverage.md` link. Parse the coverage status column, require all four approved statuses, and reject every other nonempty value.

- [ ] **Step 4: Test all topic indexes**

For each `TOPIC_DIRECTORIES` entry, require all seven H2 sections in the structural contract.

- [ ] **Step 5: Test navigation and VLA mapping**

Load `mkdocs.yml`, extract top-level labels, and compare them to the exact nine-label list. In `term-tooltips.js`, require the standalone `aliases: ["VLA"]` object to use `href: siteUrl("vla/")` and reject `vlm/` in that same object.

- [ ] **Step 6: Run the contract tests red**

Run:

```bash
pytest -q tests/test_site_information_architecture.py
```

Expected: failures identify the old homepage, missing learning-path files, incomplete topic indexes, old navigation, and VLA wrong link.

### Task 4: Rewrite the homepage and add tutorial routes

**Files:**
- Modify: `files/index.md`
- Create: `files/learning-paths/index.md`
- Create: `files/learning-paths/coverage.md`

- [ ] **Step 1: Replace the homepage positioning**

State that this is a personal layered AI tutorial which also preserves independent technical articles. Explain the four levels in one paragraph: zero-basis math, foundations assuming that math, topics assuming foundations, and independent paper deep dives with minimal background.

Remove the old “independent blog collection only”, “first visit need not read an overview”, and hard-coded paper-count claims.

- [ ] **Step 2: Write `## 从头学习`**

Present and link:

```text
数学基础 -> 深度学习基础 -> 选择一条专题主线 -> 用论文精读验证理解
```

- [ ] **Step 3: Write `## 按目标学习` and `## 按问题查阅`**

Use three routes: training/systems, generation/multimodality, and decision/physical intelligence. The problem lookup section links all ten topic indexes, knowledge QA, paper deep dives, and site search, and explains the difference between selected paths and complete directories.

- [ ] **Step 4: Write `## 当前补全重点`**

List only stable gaps: RL intermediate concepts; VLM document/OCR/chart/grounding/GUI applications; Agent/RAG runtime state, tool boundaries, security, and evaluation. Do not include page counts or delivery dates.

- [ ] **Step 5: Write the learning-path page**

Start with a “先选起点” table. Each of the three H2 routes states assumed knowledge, ordered topic indexes, a stop condition, and one paper deep dive as a reference case rather than a prerequisite.

- [ ] **Step 6: Write the coverage matrix**

Define the four statuses verbatim and add rows for math, foundations, training/systems, diffusion, VLM, RL, Agent/RAG, world-model/VLA/embodied, and paper deep dives. Use the approved initial states:

- math and foundations: `需要补强`;
- training/systems: `需要补强` until all indexes and gates are green;
- diffusion: `核心已覆盖`;
- VLM and world-model/VLA/embodied: `需要补强`;
- RL and Agent/RAG: `缺失待写`;
- paper deep dives: `参考案例`.

State that a status changes only after a real article or verified rewrite, not a navigation-only edit.

- [ ] **Step 7: Run focused tests**

Run:

```bash
pytest -q tests/test_site_information_architecture.py -k 'homepage or learning_path or coverage'
```

Expected: homepage, learning-path, and coverage tests pass; topic-index and navigation tests remain red.

**Authorized commit boundary:**

```bash
git add files/index.md files/learning-paths/index.md files/learning-paths/coverage.md tests/test_site_information_architecture.py
git diff --cached --name-only
git commit -m "docs: add layered learning paths"
```

### Task 5: Rebuild the training-and-systems indexes

**Files:**
- Modify: `files/training/index.md`
- Modify: `files/inference/index.md`
- Modify: `files/quantization/index.md`
- Modify: `files/operators/index.md`

For every index, preserve useful existing explanation and recommended tables. Reorganize them under the seven required H2 sections. `推荐阅读顺序` remains selective; `完整文章目录` links every public Markdown file in that directory except `index.md`.

- [ ] **Step 1: Rebuild `training/index.md`**

Register these currently omitted pages:

```text
data-systems-and-optimization.md
evaluation-and-ablation-methodology.md
image-sources.md
mtp-and-speculative-decoding.md
```

Place `image-sources.md` under a `配套资料` subsection inside the full directory; do not present it as a core tutorial step.

- [ ] **Step 2: Rebuild `inference/index.md`**

Separate model-level inference, serving runtimes, KV/cache policy, speculative decoding, scheduling/capacity, and failure diagnosis. Keep every current article in the full directory.

- [ ] **Step 3: Rebuild `quantization/index.md`**

Separate numeric representation, PTQ calibration, weight-only methods, QAT/low-bit training, kernels, and deployment validation. List each article exactly once in the full directory.

- [ ] **Step 4: Rebuild `operators/index.md`**

Register these currently omitted pages:

```text
advanced-kernel-patterns-and-shape-specialization.md
communication-kernels-and-overlap.md
deepgemm-fp8-gemm-and-mega-moe.md
deepgemm-source-and-integration.md
hardware-aware-debug-checklist.md
kernel-cost-models-and-selection-heuristics.md
moe-routing-and-sparse-kernels.md
performance-anti-patterns-and-failure-modes.md
ptx-sass-and-compiler-inspection.md
serving-attention-and-kv-kernels.md
```

- [ ] **Step 5: Validate the batch**

Run:

```bash
python scripts/site_structure_audit.py --topic training
python scripts/site_structure_audit.py --topic inference
python scripts/site_structure_audit.py --topic quantization
python scripts/site_structure_audit.py --topic operators
```

Expected: every focused audit exits 0. The all-topic contract can remain red until Tasks 6 and 7 finish.

**Authorized commit boundary:**

```bash
git add files/training/index.md files/inference/index.md files/quantization/index.md files/operators/index.md
git diff --cached --name-only
git commit -m "docs: complete training and systems indexes"
```

### Task 6: Rebuild the generation-and-multimodal indexes

**Files:**
- Modify: `files/diffusion/index.md`
- Modify: `files/vlm/index.md`

- [ ] **Step 1: Rebuild `diffusion/index.md`**

Register `evolution.md` and `practical-training-recipes-and-failure-analysis.md`. Organize coverage around noising/training objectives, parameterizations, samplers, guidance/conditioning, distillation/consistency, video generation, and evaluation.

- [ ] **Step 2: Rebuild `vlm/index.md`**

Register `vlm-vla-symbols-and-minimal-math.md`. Mark architecture, connectors, visual tokenization, reasoning, and evaluation according to real pages. Mark document/OCR, chart understanding, grounding, and GUI/screen agents `缺失待写`; a broad survey mention does not count as a tutorial page.

- [ ] **Step 3: Validate the batch**

Run:

```bash
python scripts/site_structure_audit.py --topic diffusion
python scripts/site_structure_audit.py --topic vlm
```

Expected: both commands exit 0.

**Authorized commit boundary:**

```bash
git add files/diffusion/index.md files/vlm/index.md
git diff --cached --name-only
git commit -m "docs: complete generation and multimodal indexes"
```

### Task 7: Rebuild the decision-and-physical-intelligence indexes

**Files:**
- Modify: `files/reinforcement-learning/index.md`
- Modify: `files/world-models/index.md`
- Modify: `files/vla/index.md`
- Modify: `files/embodied-ai/index.md`

- [ ] **Step 1: Rebuild `reinforcement-learning/index.md`**

Keep every existing article in the full directory. Mark value learning/Q-learning, offline RL, reward/verifier modeling, and credit assignment `缺失待写`; do not treat the existing policy-gradient pages as complete RL coverage.

- [ ] **Step 2: Rebuild `world-models/index.md`**

Register:

```text
action-conditioned-video-world-model-training-case.md
development-roadmap-data-simulation-and-evaluation.md
efficiency-technique-coverage-matrix.md
planning-as-inference-and-latent-actions.md
world-model-efficient-training-experiment-report.md
```

Distinguish foundations, latent dynamics, video world models, action-conditioned models, planning/control, efficiency, evaluation, and engineering roadmaps. State the unique responsibility of each roadmap or matrix page so overlapping pages do not claim the same role.

- [ ] **Step 3: Rebuild `vla/index.md`**

Keep all current articles in the full directory. Separate action representation, action chunking, data/policy learning, evaluation, closed-loop recovery, and deployment safety.

- [ ] **Step 4: Rebuild `embodied-ai/index.md`**

Register:

```text
human-robot-interaction-and-evaluation.md
vla-data-model-and-evaluation-roadmap.md
vla-wam-world-model-system-map.md
```

Keep DM0.5 as a paper case in `paper-deep-dives`, not as an embodied tutorial article.

- [ ] **Step 5: Validate all ten topic contracts**

Run:

```bash
python scripts/site_structure_audit.py --topic reinforcement-learning
python scripts/site_structure_audit.py --topic world-models
python scripts/site_structure_audit.py --topic vla
python scripts/site_structure_audit.py --topic embodied-ai
pytest -q tests/test_site_information_architecture.py -k topic_index
```

Expected: every command exits 0.

**Authorized commit boundary:**

```bash
git add files/reinforcement-learning/index.md files/world-models/index.md files/vla/index.md files/embodied-ai/index.md
git diff --cached --name-only
git commit -m "docs: complete decision and embodied indexes"
```

### Task 8: Turn the references index into the owning directory

**Files:**
- Modify: `files/references/index.md`

- [ ] **Step 1: Preserve reader-facing governance entries**

Keep current evidence, reproducibility, image-governance, external-reading, and writing-framework entries. Do not edit the body of `external-reading-ledger.md`.

- [ ] **Step 2: Add `## 本地来源台账目录`**

Classify every other public `files/references/*.md` page exactly once under:

```text
全站标准与治理
数学、基础与训练
推理、量化与算子
扩散与生成
多模态、世界模型与具身智能
```

The directory includes all source-ledger pages, `knowledge-blog-writing-framework.md`, and `external-reading-ledger.md`; it must not link to itself.

- [ ] **Step 3: Verify exhaustive, non-duplicated registration**

Run:

```bash
python scripts/site_structure_audit.py --topic references
```

Expected: exit 0 with no missing, broken, or duplicate references registrations.

**Authorized commit boundary:**

```bash
git add files/references/index.md
git diff --cached --name-only
git commit -m "docs: index local evidence ledgers"
```

### Task 9: Repair DM0.5 registration and the standalone VLA link

**Files:**
- Modify: `files/paper-deep-dives/index.md`
- Modify: `files/assets/javascripts/term-tooltips.js`

- [ ] **Step 1: Register DM0.5 once**

Under the embodied-intelligence paper group, add `embodied-ai/dm05.md`. Do not edit the paper body or duplicate an existing live navigation entry.

- [ ] **Step 2: Correct only the standalone alias**

In the object whose aliases are exactly `aliases: ["VLA"]`, change `href: siteUrl("vlm/")` to `href: siteUrl("vla/")`. Leave the combined `VLM/VLA` alias unchanged.

- [ ] **Step 3: Verify both repairs**

Run:

```bash
python scripts/paper_deep_dive_audit.py
pytest -q tests/test_site_information_architecture.py -k vla
```

Expected: the paper audit exits 0 and the VLA test passes.

**Authorized commit boundary:**

```bash
git add files/paper-deep-dives/index.md files/assets/javascripts/term-tooltips.js
git diff --cached --name-only
git commit -m "docs: repair paper and VLA index links"
```

### Task 10: Converge navigation to nine top-level entries

**Dependency:** The prerequisite/foundations plan's navigation task must finish first. Work from the live `mkdocs.yml`, not the version seen while writing this plan.

**Files:**
- Modify: `mkdocs.yml`

- [ ] **Step 1: Reread the overlapping file**

Run:

```bash
git diff -- mkdocs.yml
rg -n '^nav:|^  - ' mkdocs.yml
```

Expected: newest curriculum pages, DM0.5, and thinking-exploration pages are visible.

- [ ] **Step 2: Preserve curriculum subtrees**

Retain every live prerequisite-math entry and rename only its top-level label to `数学基础`. Retain every live foundations entry and rename only its top-level label to `深度学习基础`.

- [ ] **Step 3: Add the two direct entry leaves**

Use:

```yaml
- 首页: index.md
- 学习路径: learning-paths/index.md
```

The quality plan restricts H1 title synchronization to paper pages so these custom labels remain visible.

- [ ] **Step 4: Build the three topic groups**

- `训练与系统`: indexes for training, inference, quantization, and operators first, followed by curated articles.
- `生成与多模态`: diffusion and VLM indexes first, followed by curated articles.
- `决策与物理智能`: RL, world-model, VLA, and embodied indexes first, followed by curated articles.

Do not duplicate a Markdown target across groups.

- [ ] **Step 5: Preserve the paper collection**

Under `论文精读`, make `paper-deep-dives/index.md` first and retain all live paper groups and DM0.5.

- [ ] **Step 6: Build `资料库`**

Include knowledge-QA and thinking-exploration indexes, `glossary.md`, `references/index.md`, and `paper-deep-dives/writing-standard.md`. Preserve every currently published knowledge-QA and thinking-exploration article beneath its corresponding index so none becomes unregistered. Remove flat source-ledger entries from the sidebar without deleting their files.

- [ ] **Step 7: Verify exact labels and registration**

Run:

```bash
pytest -q tests/test_site_information_architecture.py -k navigation
python scripts/site_structure_audit.py
```

Expected: both commands exit 0 after all prior index tasks are complete.

**Authorized commit boundary:**

```bash
git add mkdocs.yml
git diff --cached --name-only
git commit -m "docs: organize the site as a layered tutorial"
```

### Task 11: Run the structural acceptance suite

**Files:**
- Verify: every file changed by this plan

- [ ] **Step 1: Run focused tests**

Run:

```bash
pytest -q tests/test_site_structure_audit.py
pytest -q tests/test_site_information_architecture.py
```

Expected: both test files pass.

- [ ] **Step 2: Run repository audits**

Run:

```bash
python scripts/site_structure_audit.py
python scripts/paper_deep_dive_audit.py
```

Expected: `Site structure audit passed.` and no paper-audit errors.

- [ ] **Step 3: Check patch integrity**

Run:

```bash
git diff --check
git status --short
```

Expected: no whitespace errors. Concurrent user work may remain, but no unrelated file is staged or overwritten.

- [ ] **Step 4: Build strictly**

Run:

```bash
mkdocs build --strict
```

Expected: exit 0; homepage, learning paths, topic indexes, and links render.

- [ ] **Step 5: Run the unified gate after the quality plan lands**

Run:

```bash
make check
```

Expected: all tests, hard audits, structure audit, and strict build pass.

## Completion Boundary

This plan is complete when tutorial routes and exhaustive indexes are real, nine-group navigation is live, all public pages are registered without a whitelist, DM0.5 and VLA resolve correctly, and the structure audit passes. It records RL, VLM-application, and Agent/RAG gaps instead of filling them with shallow articles; those additions begin with focused article designs and follow the approved writing contract.
