# Diffusion Model Curriculum Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the diffusion-model topic into a dependency-ordered course that teaches the full training and sampling loop, DDPM mathematics, Stable Diffusion architecture, a runnable minimal implementation, and clear routes into modern score/flow/sampling/distillation topics.

**Architecture:** Keep every existing diffusion URL stable, make `files/diffusion/index.md` the explicit entry point, and add four focused pages rather than rewriting the topic as one long article. Enforce the curriculum contract with repository tests, place runnable teaching code under `files/assets/examples/diffusion/`, and use existing advanced pages as branches whose responsibilities are narrowed and cross-linked.

**Tech Stack:** MkDocs Material, Markdown, MathJax/Arithmatex, Mermaid and SVG, Python 3.11+, PyTorch 2.2+, torchvision, unittest/pytest.

---

## Working-Tree Rule

Implementation must run in a dedicated worktree created with the `using-git-worktrees` skill, on a branch such as `codex/diffusion-curriculum`. The current checkout contains user-owned changes in `mkdocs.yml`, paper indexes, and unrelated pages; do not reset, stash, overwrite, or broadly stage them. Before eventual integration, compare the worktree's `mkdocs.yml` diffusion hunk with the user's current uncommitted navigation changes and merge only the `扩散与生成模型` block.

Do not add `.superpowers/brainstorm/` to commits. It contains visual-design session artifacts, not product files.

## File Map

### Entry and foundation pages

- Rewrite: `files/diffusion/index.md` — definition, dual-loop overview, required path, advanced map.
- Create: `files/diffusion/beginner-primer.md` — first-principles mental model and generative-model boundary.
- Rewrite: `files/diffusion/training.md` — DDPM forward, posterior, reverse model, ELBO bridge, training and sampling algorithms.
- Create: `files/diffusion/architecture-and-stable-diffusion.md` — U-Net/DiT and Stable Diffusion component data flow.
- Create: `files/diffusion/minimal-ddpm-from-scratch.md` — formula-to-code walkthrough.
- Create: `files/diffusion/flow-matching-and-rectified-flow.md` — modern velocity-field and rectification route.

### Existing advanced pages

- Refocus: `files/diffusion/evolution.md`.
- Refocus: `files/diffusion/noise-schedules-and-parameterization.md`.
- Refocus: `files/diffusion/score-matching-sde-and-probability-flow.md`.
- Refocus: `files/diffusion/guidance-and-conditioning.md`.
- Refocus: `files/diffusion/inference.md`.
- Refocus: `files/diffusion/comparison-table.md`.
- Refocus: `files/diffusion/distillation.md`.
- Refocus: `files/diffusion/consistency-models-and-rectified-flow.md` without renaming its file.
- Refocus: `files/diffusion/practical-training-recipes-and-failure-analysis.md` and add it to navigation.
- Refocus: `files/diffusion/video-and-multimodal-diffusion.md`.

### Examples, tests, and visuals

- Create: `files/assets/examples/diffusion/README.md`.
- Create: `files/assets/examples/diffusion/requirements.txt`.
- Create: `files/assets/examples/diffusion/minimal_ddpm.py`.
- Create: `files/assets/examples/diffusion/test_minimal_ddpm.py`.
- Create: `tests/test_diffusion_curriculum.py`.
- Create: `files/assets/images/diffusion/diffusion-distribution-path.svg`.
- Reuse and, only if labels need correction, modify: `files/assets/images/paper-figures/diffusion/ddpm-orthogonal-coefficients.svg`.
- Modify: `mkdocs.yml` only inside the `扩散与生成模型` navigation block.

## Shared Page Contract

The four required foundation chapters and the Flow chapter must contain these exact H2 headings in this order, with topic-specific H2 sections allowed between them:

```markdown
## 本章回答什么
## 先修知识
## 核心机制
## 常见误解与失败
## 自检
## 下一步
```

Every page must define symbols before use, link to at least one prerequisite or earlier page, and link to a later page or the topic index. The index is exempt from the shared headings and instead must contain:

```markdown
## 什么是扩散模型
## 训练和生成是两条不同的链
## 基础主线
## 进阶地图
## 按目标进入
```

## Task 1: Establish the clean baseline and curriculum contract

**Files:**
- Create: `tests/test_diffusion_curriculum.py`
- Read: `mkdocs.yml:183-194`
- Read: `files/diffusion/*.md`
- Read: `Makefile`

- [ ] **Step 1: Create the isolated worktree and record the baseline**

From the original checkout, invoke `using-git-worktrees`, then in the new worktree run:

```bash
git status --short
python -m pytest tests/test_content_quality_report.py -q
make build
```

Expected: the worktree is clean; existing content-quality tests pass; `mkdocs build --strict` exits 0. If a baseline command fails, record the exact error before changing files and do not attribute it to this curriculum.

- [ ] **Step 2: Write the failing curriculum structure test**

Create `tests/test_diffusion_curriculum.py` with:

```python
import re
from pathlib import Path
from unittest import TestCase


ROOT = Path(__file__).resolve().parents[1]
DIFFUSION_ROOT = ROOT / "files" / "diffusion"


class DiffusionCurriculumTests(TestCase):
    FOUNDATION_PAGES = (
        "beginner-primer.md",
        "training.md",
        "architecture-and-stable-diffusion.md",
        "minimal-ddpm-from-scratch.md",
        "flow-matching-and-rectified-flow.md",
    )
    REQUIRED_HEADINGS = (
        "## 本章回答什么",
        "## 先修知识",
        "## 核心机制",
        "## 常见误解与失败",
        "## 自检",
        "## 下一步",
    )

    def assert_ordered_subsequence(self, required, actual):
        remaining = iter(actual)
        for item in required:
            self.assertIn(item, remaining)

    def test_index_defines_the_topic_and_learning_routes(self):
        text = (DIFFUSION_ROOT / "index.md").read_text(encoding="utf-8")
        headings = [line for line in text.splitlines() if line.startswith("## ")]
        self.assertEqual(
            headings,
            [
                "## 什么是扩散模型",
                "## 训练和生成是两条不同的链",
                "## 基础主线",
                "## 进阶地图",
                "## 按目标进入",
            ],
        )

    def test_navigation_starts_with_the_topic_index(self):
        nav = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
        start = nav.index("  - 扩散与生成模型:")
        end = nav.index("  - 思考探索:", start)
        section = nav[start:end]
        first_page = next(line for line in section.splitlines() if "diffusion/" in line)
        self.assertIn("diffusion/index.md", first_page)

```

- [ ] **Step 3: Run the contract test and verify the expected failures**

Run:

```bash
python -m pytest tests/test_diffusion_curriculum.py -q
```

Expected: two failures because the index headings are different and navigation does not start with `diffusion/index.md`.

- [ ] **Step 4: Keep the red test for the entry implementation**

Do not commit the failing test by itself. Continue directly to Task 2, make both tests green, then commit the contract and implementation together.

## Task 2: Build the topic entry and staged navigation

**Files:**
- Rewrite: `files/diffusion/index.md`
- Modify: `mkdocs.yml:183-194`
- Test: `tests/test_diffusion_curriculum.py`

- [ ] **Step 1: Rewrite the index with the exact five-section contract**

Use these H2 sections, in this order and with no additional H2 headings:

```markdown
## 什么是扩散模型
## 训练和生成是两条不同的链
## 基础主线
## 进阶地图
## 按目标进入
```

The opening definition must state that a diffusion model defines a known corruption path from data toward a simple noise distribution and learns information needed to generate in the reverse direction. It must not say that training literally runs the full forward chain for every example.

At this staged point, list not-yet-created foundation pages as plain text rather than broken Markdown links. Replace each label with a link in the task that creates that page; Task 10 verifies the final linked route.

- [ ] **Step 2: Add the dual-loop overview to the index**

Insert this Mermaid structure under `## 训练和生成是两条不同的链` and explain every arrow in prose:

```mermaid
flowchart LR
  subgraph Train[训练]
    X0[真实样本 x_0] --> Mix[随机时间步 t 与噪声 epsilon]
    Mix --> XT[带噪样本 x_t]
    XT --> Net[denoiser(x_t, t, c)]
    Net --> Loss[与已知目标计算 loss]
  end
  subgraph Sample[生成]
    Noise[随机噪声 x_T] --> Predict[denoiser 预测]
    Predict --> Update[scheduler / sampler 更新]
    Update -->|重复多个时间步| Predict
    Update --> Output[生成样本 x_0]
  end
```

State explicitly that the forward corruption is known and not learned, while the reverse-generation information is learned.

- [ ] **Step 3: Add only existing pages to the staged navigation**

Because strict MkDocs builds cannot reference missing files, first make the block:

```yaml
  - 扩散与生成模型:
      - 扩散模型专题: diffusion/index.md
      - 扩散模型发展脉络: diffusion/evolution.md
      - DDPM 数学与训练: diffusion/training.md
      - 噪声日程与参数化: diffusion/noise-schedules-and-parameterization.md
      - Score Matching 到 SDE: diffusion/score-matching-sde-and-probability-flow.md
      - 条件控制与 Guidance: diffusion/guidance-and-conditioning.md
      - 采样与推理加速: diffusion/inference.md
      - 扩散路线选择与方法对照: diffusion/comparison-table.md
      - 一步生成、蒸馏与整流: diffusion/distillation.md
      - Consistency Models 与 Flow 边界: diffusion/consistency-models-and-rectified-flow.md
      - 扩散训练配方与排障: diffusion/practical-training-recipes-and-failure-analysis.md
      - 视频与多模态扩散: diffusion/video-and-multimodal-diffusion.md
```

Modify only this YAML block.

- [ ] **Step 4: Run the index and navigation tests**

```bash
python -m pytest \
  tests/test_diffusion_curriculum.py::DiffusionCurriculumTests::test_index_defines_the_topic_and_learning_routes \
  tests/test_diffusion_curriculum.py::DiffusionCurriculumTests::test_navigation_starts_with_the_topic_index -q
make build
```

Expected: both focused tests pass; strict build exits 0. The full curriculum test still fails because the new pages are not yet present.

- [ ] **Step 5: Commit the entry batch**

```bash
git add files/diffusion/index.md mkdocs.yml tests/test_diffusion_curriculum.py
git commit -m "docs: add diffusion curriculum entry"
```

## Task 3: Write the beginner primer and distribution visual

**Files:**
- Create: `files/diffusion/beginner-primer.md`
- Create: `files/assets/images/diffusion/diffusion-distribution-path.svg`
- Modify: `tests/test_diffusion_curriculum.py`

- [ ] **Step 1: Add a primer-specific failing test**

Append this method to `DiffusionCurriculumTests`:

```python
    def test_primer_builds_a_complete_beginner_mental_model(self):
        text = (DIFFUSION_ROOT / "beginner-primer.md").read_text(encoding="utf-8")
        for required in (
            "生成模型",
            "数据分布",
            "前向加噪",
            "反向生成",
            "训练时已知",
            "采样时未知",
            "VAE",
            "GAN",
            "自回归",
            "diffusion-distribution-path.svg",
        ):
            self.assertIn(required, text)

    def test_primer_does_not_require_advanced_methods(self):
        text = (DIFFUSION_ROOT / "beginner-primer.md").read_text(encoding="utf-8")
        for advanced_term in ("DPM-Solver", "DMD2", "反向 SDE", "ELBO 展开"):
            self.assertNotIn(advanced_term, text)
```

- [ ] **Step 2: Run the new test and verify it fails**

```bash
python -m pytest tests/test_diffusion_curriculum.py::DiffusionCurriculumTests::test_primer_builds_a_complete_beginner_mental_model -q
```

Expected: FAIL because `beginner-primer.md` does not exist.

- [ ] **Step 3: Create the primer with this section sequence**

Use the shared contract plus these topic sections:

```markdown
# 从零理解扩散模型：训练是在造题，生成是在逐步解题

## 本章回答什么
## 先修知识
## 核心机制
### 生成模型到底在学什么
### 为什么先把真实样本变坏
### 训练时模型看见什么
### 生成时为什么能从随机噪声开始
### 一次训练样本和一次完整生成
## 扩散模型和 VAE、GAN、自回归模型的边界
## 能做什么，以及代价是什么
## 常见误解与失败
## 自检
## 下一步
```

Include a compact comparison table with columns `方法 / 学习或生成接口 / 采样方式 / 主要代价`. Avoid claims that diffusion never memorizes training data or that iterative sampling guarantees correctness.

- [ ] **Step 4: Create the SVG distribution path**

Create a 1200×420 accessible SVG with three labeled panels: `真实数据分布 p_data`, `中间带噪分布 p_t`, and `近似高斯 p_T`. Use the same point identities fading and spreading across panels, add arrow labels `已知加噪过程`, and include `<title>` and `<desc>`. Do not imply that individual points have a unique deterministic reverse path.

- [ ] **Step 5: Link the primer forward and verify**

Link `beginner-primer.md` to `training.md`, the prerequisite probability page, and `index.md`. Then run:

```bash
python -m pytest tests/test_diffusion_curriculum.py::DiffusionCurriculumTests::test_primer_builds_a_complete_beginner_mental_model -q
make image-quality
make build
```

Expected: the primer test passes; image quality and strict build exit 0.

- [ ] **Step 6: Commit the primer batch**

```bash
git add \
  files/diffusion/beginner-primer.md \
  files/assets/images/diffusion/diffusion-distribution-path.svg \
  tests/test_diffusion_curriculum.py
git commit -m "docs: add diffusion beginner primer"
```

## Task 4: Refocus `training.md` into the DDPM mathematical loop

**Files:**
- Rewrite: `files/diffusion/training.md`
- Modify: `tests/test_diffusion_curriculum.py`

- [ ] **Step 1: Add a failing DDPM coverage test**

Append:

```python
    def test_training_page_closes_the_ddpm_math_loop(self):
        text = (DIFFUSION_ROOT / "training.md").read_text(encoding="utf-8")
        compact = re.sub(r"\s+", "", text)
        for required in (
            r"q(x_t\midx_{t-1})",
            r"q(x_t\midx_0)",
            r"q(x_{t-1}\midx_t,x_0)",
            r"p_\theta(x_{t-1}\midx_t)",
            r"\tilde{\beta}_t",
            r"\epsilon_\theta(x_t,t)",
        ):
            self.assertIn(required, compact)
        for heading in ("## DDPM 训练算法", "## DDPM 采样算法"):
            self.assertIn(heading, text)
```

Use `\mid` consistently in the page so the normalized string test is stable.

- [ ] **Step 2: Run the test and verify the missing posterior/algorithm failure**

```bash
python -m pytest tests/test_diffusion_curriculum.py::DiffusionCurriculumTests::test_training_page_closes_the_ddpm_math_loop -q
```

Expected: FAIL because the existing page does not contain the full posterior and exact algorithm headings.

- [ ] **Step 3: Rewrite the forward and posterior sections**

Use this mathematical sequence and define every symbol before use:

```math
q(x_t\mid x_{t-1})=\mathcal N(\sqrt{\alpha_t}x_{t-1},\beta_t I),
\qquad \alpha_t=1-\beta_t
```

```math
q(x_t\mid x_0)=\mathcal N(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I)
```

```math
q(x_{t-1}\mid x_t,x_0)=\mathcal N(\tilde\mu_t(x_t,x_0),\tilde\beta_t I)
```

```math
\tilde\beta_t=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t
```

Explain that the posterior is available during derivation because training starts from known `x_0`; generation does not know `x_0`, so a network supplies the needed estimate.

- [ ] **Step 4: Add the learned reverse mean and loss bridge**

Include:

```math
p_\theta(x_{t-1}\mid x_t)=\mathcal N(\mu_\theta(x_t,t),\Sigma_t)
```

```math
\mu_\theta(x_t,t)=\frac{1}{\sqrt{\alpha_t}}
\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\right)
```

```math
\mathcal L_{\text{simple}}
=\mathbb E_{x_0,t,\epsilon}\left[\lVert\epsilon-\epsilon_\theta(x_t,t)\rVert_2^2\right]
```

Describe the ELBO-to-weighted-denoising relationship without claiming the simplified unweighted MSE is numerically identical to every VLB term.

- [ ] **Step 5: Add explicit training and sampling algorithms**

Under `## DDPM 训练算法`, provide pseudocode that samples `x_0`, `t`, and `epsilon`, constructs `x_t` in one closed-form step, predicts noise, and updates parameters. Under `## DDPM 采样算法`, start from `x_T ~ N(0,I)`, iterate `t=T-1...0`, use fresh noise only when `t>0`, and state which variance convention is used.

- [ ] **Step 6: Remove or relocate out-of-scope branches**

Replace detailed Stable Diffusion, DiT, video/world-model, training-log, and distillation passages with one-paragraph bridges to the architecture, practical training, and advanced pages. Preserve valid source links that still support DDPM claims.

- [ ] **Step 7: Verify and commit**

```bash
python -m pytest tests/test_diffusion_curriculum.py::DiffusionCurriculumTests::test_training_page_closes_the_ddpm_math_loop -q
make math-quality
make build
git add files/diffusion/training.md tests/test_diffusion_curriculum.py
git commit -m "docs: complete the DDPM mathematical loop"
```

Expected: focused test, math-quality check, and strict build all pass before commit.

## Task 5: Implement and test the forward diffusion API

**Files:**
- Create: `files/assets/examples/diffusion/minimal_ddpm.py`
- Create: `files/assets/examples/diffusion/test_minimal_ddpm.py`
- Create: `files/assets/examples/diffusion/requirements.txt`

- [ ] **Step 1: Write failing tests for schedule extraction and `q_sample`**

Create `test_minimal_ddpm.py` with an import helper and these tests:

```python
import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

MODULE_PATH = Path(__file__).with_name("minimal_ddpm.py")
SPEC = importlib.util.spec_from_file_location("minimal_ddpm", MODULE_PATH)
minimal_ddpm = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = minimal_ddpm
assert SPEC.loader is not None
SPEC.loader.exec_module(minimal_ddpm)


def test_linear_schedule_has_valid_monotone_coefficients():
    schedule = minimal_ddpm.make_linear_schedule(8)
    assert schedule.betas.shape == (8,)
    assert torch.all((schedule.betas > 0) & (schedule.betas < 1))
    assert torch.all(schedule.alpha_bars[1:] < schedule.alpha_bars[:-1])
    assert torch.all(schedule.posterior_variance >= 0)


def test_q_sample_matches_the_closed_form_without_mutation():
    schedule = minimal_ddpm.make_linear_schedule(8)
    x0 = torch.tensor([[[[1.0, -1.0]]], [[[0.5, 0.25]]]])
    noise = torch.tensor([[[[0.2, -0.4]]], [[[1.0, -1.0]]]])
    t = torch.tensor([0, 7], dtype=torch.long)
    snapshots = (x0.clone(), noise.clone())
    actual = minimal_ddpm.q_sample(x0, t, schedule, noise)
    expected = (
        schedule.sqrt_alpha_bars[t].view(2, 1, 1, 1) * x0
        + schedule.sqrt_one_minus_alpha_bars[t].view(2, 1, 1, 1) * noise
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(x0, snapshots[0])
    torch.testing.assert_close(noise, snapshots[1])


def test_q_sample_rejects_invalid_timestep_batches():
    schedule = minimal_ddpm.make_linear_schedule(8)
    x0 = torch.zeros(2, 1, 4, 4)
    with pytest.raises(ValueError, match="one timestep per batch item"):
        minimal_ddpm.q_sample(x0, torch.tensor([1]), schedule)
    with pytest.raises(ValueError, match="inside the schedule"):
        minimal_ddpm.q_sample(x0, torch.tensor([0, 8]), schedule)
```

- [ ] **Step 2: Run the tests and verify import failure**

```bash
python -m pytest files/assets/examples/diffusion/test_minimal_ddpm.py -q
```

Expected: FAIL because `minimal_ddpm.py` does not exist.

- [ ] **Step 3: Implement the schedule and forward process**

Create `minimal_ddpm.py` with this public API:

```python
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DiffusionSchedule:
    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bars: torch.Tensor
    alpha_bars_prev: torch.Tensor
    sqrt_alpha_bars: torch.Tensor
    sqrt_one_minus_alpha_bars: torch.Tensor
    posterior_variance: torch.Tensor


def make_linear_schedule(num_steps, beta_start=1e-4, beta_end=2e-2, device=None):
    if not isinstance(num_steps, int) or isinstance(num_steps, bool) or num_steps < 2:
        raise ValueError("num_steps must be an integer of at least 2")
    if not 0.0 < beta_start < beta_end < 1.0:
        raise ValueError("betas must satisfy 0 < beta_start < beta_end < 1")
    betas = torch.linspace(beta_start, beta_end, num_steps, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    alpha_bars_prev = torch.cat((torch.ones_like(alpha_bars[:1]), alpha_bars[:-1]))
    posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)
    return DiffusionSchedule(
        betas=betas,
        alphas=alphas,
        alpha_bars=alpha_bars,
        alpha_bars_prev=alpha_bars_prev,
        sqrt_alpha_bars=torch.sqrt(alpha_bars),
        sqrt_one_minus_alpha_bars=torch.sqrt(1.0 - alpha_bars),
        posterior_variance=posterior_variance,
    )


def _extract(values, t, x):
    if t.dtype != torch.long or t.ndim != 1 or t.shape[0] != x.shape[0]:
        raise ValueError("t must provide one timestep per batch item")
    if t.device != x.device:
        raise ValueError("t and x must be on the same device")
    if values.device != x.device:
        raise ValueError("schedule and x must be on the same device")
    if torch.any(t < 0) or torch.any(t >= values.shape[0]):
        raise ValueError("timesteps must be inside the schedule")
    return values.gather(0, t).reshape(t.shape[0], *((1,) * (x.ndim - 1)))


def q_sample(x0, t, schedule, noise=None):
    if not torch.is_floating_point(x0) or x0.ndim != 4:
        raise ValueError("x0 must be a floating BCHW tensor")
    noise = torch.randn_like(x0) if noise is None else noise
    if noise.shape != x0.shape or noise.device != x0.device:
        raise ValueError("noise must match x0 shape and device")
    return (
        _extract(schedule.sqrt_alpha_bars, t, x0) * x0
        + _extract(schedule.sqrt_one_minus_alpha_bars, t, x0) * noise
    )
```

- [ ] **Step 4: Add bounded example dependencies**

Create `requirements.txt`:

```text
torch>=2.2,<3
torchvision>=0.17,<1
pytest>=8,<10
```

- [ ] **Step 5: Run tests and commit**

```bash
python -m pytest files/assets/examples/diffusion/test_minimal_ddpm.py -q
git add files/assets/examples/diffusion
git commit -m "feat: add minimal DDPM forward process"
```

Expected: three tests pass.

## Task 6: Add the denoiser and training loss with TDD

**Files:**
- Modify: `files/assets/examples/diffusion/minimal_ddpm.py`
- Modify: `files/assets/examples/diffusion/test_minimal_ddpm.py`

- [ ] **Step 1: Add failing model and optimizer-update tests**

Append:

```python
def test_tiny_denoiser_preserves_image_shape():
    model = minimal_ddpm.TinyDenoiser(in_channels=1, base_channels=16, time_dim=32)
    x = torch.randn(3, 1, 28, 28)
    t = torch.tensor([0, 3, 7])
    assert model(x, t).shape == x.shape


def test_diffusion_loss_is_finite_and_updates_parameters():
    torch.manual_seed(7)
    schedule = minimal_ddpm.make_linear_schedule(8)
    model = minimal_ddpm.TinyDenoiser(in_channels=1, base_channels=16, time_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x0 = torch.rand(4, 1, 28, 28) * 2.0 - 1.0
    t = torch.tensor([0, 2, 5, 7])
    before = [parameter.detach().clone() for parameter in model.parameters()]
    loss = minimal_ddpm.diffusion_loss(model, x0, t, schedule)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
    assert any(
        not torch.equal(old, new.detach())
        for old, new in zip(before, model.parameters())
    )
```

- [ ] **Step 2: Run and verify missing API failures**

```bash
python -m pytest files/assets/examples/diffusion/test_minimal_ddpm.py -q
```

Expected: FAIL because `TinyDenoiser` and `diffusion_loss` do not exist.

- [ ] **Step 3: Implement the time embedding and small U-Net-shaped denoiser**

Add `SinusoidalTimeEmbedding`, a residual block with GroupNorm and time conditioning, and `TinyDenoiser`. The model must use one 28→14 downsample, one 14→28 upsample, a same-resolution skip concatenation, and a final convolution back to `in_channels`. Validate BCHW input, integer timestep batch shape, and spatial dimensions divisible by two.

The time embedding formula must be:

```python
half = self.dimension // 2
frequencies = torch.exp(
    -torch.log(torch.tensor(10_000.0, device=t.device))
    * torch.arange(half, device=t.device)
    / max(half - 1, 1)
)
angles = t.float()[:, None] * frequencies[None]
embedding = torch.cat((angles.sin(), angles.cos()), dim=1)
```

Pad one zero feature only when `dimension` is odd.

- [ ] **Step 4: Implement the noise-prediction loss**

Add:

```python
def diffusion_loss(model, x0, t, schedule, noise=None):
    noise = torch.randn_like(x0) if noise is None else noise
    xt = q_sample(x0, t, schedule, noise)
    prediction = model(xt, t)
    if prediction.shape != noise.shape:
        raise ValueError("model prediction must match the sampled noise")
    return torch.nn.functional.mse_loss(prediction, noise)
```

- [ ] **Step 5: Run tests and commit**

```bash
python -m pytest files/assets/examples/diffusion/test_minimal_ddpm.py -q
git add files/assets/examples/diffusion/minimal_ddpm.py files/assets/examples/diffusion/test_minimal_ddpm.py
git commit -m "feat: add minimal DDPM denoiser training"
```

Expected: all example tests pass and the optimizer-update assertion observes at least one changed parameter.

## Task 7: Add reverse sampling, the CLI, and the code walkthrough

**Files:**
- Modify: `files/assets/examples/diffusion/minimal_ddpm.py`
- Modify: `files/assets/examples/diffusion/test_minimal_ddpm.py`
- Create: `files/assets/examples/diffusion/README.md`
- Create: `files/diffusion/minimal-ddpm-from-scratch.md`
- Modify: `tests/test_diffusion_curriculum.py`

- [ ] **Step 1: Add failing reverse-process tests**

Append tests that use a zero-noise model:

```python
class ZeroNoise(torch.nn.Module):
    def forward(self, x, t):
        return torch.zeros_like(x)


def test_p_sample_is_deterministic_at_t_zero():
    schedule = minimal_ddpm.make_linear_schedule(8)
    xt = torch.randn(2, 1, 8, 8)
    t = torch.zeros(2, dtype=torch.long)
    first = minimal_ddpm.p_sample(ZeroNoise(), xt, t, schedule)
    second = minimal_ddpm.p_sample(ZeroNoise(), xt, t, schedule)
    torch.testing.assert_close(first, second)


def test_sample_loop_is_seeded_finite_and_shape_stable():
    schedule = minimal_ddpm.make_linear_schedule(8)
    first = minimal_ddpm.sample_loop(ZeroNoise(), (2, 1, 8, 8), schedule, seed=7)
    second = minimal_ddpm.sample_loop(ZeroNoise(), (2, 1, 8, 8), schedule, seed=7)
    assert first.shape == (2, 1, 8, 8)
    assert torch.all(torch.isfinite(first))
    torch.testing.assert_close(first, second)
```

- [ ] **Step 2: Run and verify missing reverse API failures**

```bash
python -m pytest files/assets/examples/diffusion/test_minimal_ddpm.py -q
```

Expected: FAIL because `p_sample` and `sample_loop` do not exist.

- [ ] **Step 3: Implement reverse sampling**

Use the DDPM mean from the training page and `schedule.posterior_variance`. `p_sample` must add sampled noise only where `t>0`; `sample_loop` must create a local `torch.Generator`, seed it, initialize Gaussian noise, and iterate all timesteps in reverse. Reject non-positive batch/spatial shapes and schedule/device mismatches.

The core update must be:

```python
predicted_noise = model(xt, t)
mean = torch.rsqrt(_extract(schedule.alphas, t, xt)) * (
    xt
    - _extract(schedule.betas, t, xt)
    / _extract(schedule.sqrt_one_minus_alpha_bars, t, xt)
    * predicted_noise
)
variance = _extract(schedule.posterior_variance, t, xt)
mask = (t > 0).reshape(t.shape[0], *((1,) * (xt.ndim - 1)))
return mean + mask * torch.sqrt(variance) * noise
```

- [ ] **Step 4: Add a bounded smoke CLI and MNIST training mode**

Use subcommands:

```text
python minimal_ddpm.py smoke --seed 7 --timesteps 8
python minimal_ddpm.py train --data-dir .cache/diffusion-mnist --download --epochs 1 --seed 7
```

`smoke` uses a synthetic BCHW batch, performs one optimizer update, samples two 8×8 tensors, and prints finite `loss=`, `sample_mean=`, and `sample_std=` fields. `train` imports torchvision lazily, normalizes MNIST to `[-1,1]`, and downloads only when `--download` is explicitly passed.

- [ ] **Step 5: Document environment and download impact**

In `README.md`, give an isolated `.venv-diffusion-course` setup, the two commands above, and this rollback boundary:

```bash
deactivate
rm -rf .venv-diffusion-course .cache/diffusion-mnist
```

Explain that the deletion is scoped to the example environment and MNIST cache, and must be run from the repository root after checking both names. State that the smoke run proves only code-path integrity, not image quality or convergence.

- [ ] **Step 6: Write the formula-to-code page**

Use this sequence:

```markdown
# 从零实现最小 DDPM：把每个公式映射到 PyTorch
## 本章回答什么
## 先修知识
## 核心机制
## 运行环境与数据范围
## 第一步：构造噪声日程
## 第二步：闭式采样 x_t
## 第三步：让 denoiser 读取图像和时间步
## 第四步：计算噪声预测损失
## 第五步：实现一个反向采样步
## 第六步：串起完整采样循环
## 怎样判断代码真的接通了
## 常见误解与失败
## 自检
## 下一步
```

Link directly to `../assets/examples/diffusion/minimal_ddpm.py`, its test file, and README. Every code variable must use the same meaning as `training.md`.

- [ ] **Step 7: Add a curriculum link test**

Append:

```python
    def test_minimal_implementation_page_links_runnable_assets(self):
        text = (DIFFUSION_ROOT / "minimal-ddpm-from-scratch.md").read_text(
            encoding="utf-8"
        )
        for required in (
            "../assets/examples/diffusion/minimal_ddpm.py",
            "../assets/examples/diffusion/test_minimal_ddpm.py",
            "../assets/examples/diffusion/README.md",
            "python files/assets/examples/diffusion/minimal_ddpm.py smoke --seed 7 --timesteps 8",
            "python files/assets/examples/diffusion/minimal_ddpm.py train --data-dir .cache/diffusion-mnist --download --epochs 1 --seed 7",
        ):
            self.assertIn(required, text)
```

Run:

```bash
python -m pytest files/assets/examples/diffusion/test_minimal_ddpm.py -q
python files/assets/examples/diffusion/minimal_ddpm.py smoke --seed 7 --timesteps 8
python -m pytest tests/test_diffusion_curriculum.py -q
```

Expected: example tests pass, smoke CLI exits 0 with finite labeled fields, and only not-yet-created architecture/flow curriculum checks may remain failing.

- [ ] **Step 8: Commit the runnable example batch**

```bash
git add \
  files/assets/examples/diffusion \
  files/diffusion/minimal-ddpm-from-scratch.md \
  tests/test_diffusion_curriculum.py
git commit -m "feat: add runnable minimal DDPM tutorial"
```

## Task 8: Add the architecture and Stable Diffusion systems chapter

**Files:**
- Create: `files/diffusion/architecture-and-stable-diffusion.md`
- Modify: `tests/test_diffusion_curriculum.py`

- [ ] **Step 1: Add a failing architecture coverage test**

Append:

```python
    def test_architecture_page_connects_network_and_system_components(self):
        text = (
            DIFFUSION_ROOT / "architecture-and-stable-diffusion.md"
        ).read_text(encoding="utf-8")
        for required in (
            "U-Net",
            "DiT",
            "timestep embedding",
            "VAE encoder",
            "VAE decoder",
            "tokenizer",
            "text encoder",
            "cross-attention",
            "denoiser",
            "scheduler",
            "训练数据流",
            "推理数据流",
        ):
            self.assertIn(required, text)
```

- [ ] **Step 2: Run the focused test and verify file-not-found failure**

```bash
python -m pytest tests/test_diffusion_curriculum.py -k architecture -q
```

Expected: FAIL because the page does not exist.

- [ ] **Step 3: Write the denoiser-interface half**

Use the shared contract and add sections for the interface `(x_t, t, c) -> prediction`, timestep embeddings, residual blocks, U-Net skip connections, attention, and DiT patch tokens. Include one Mermaid diagram that shows inputs entering the denoiser and the prediction returning to the sampler; do not imply the scheduler is part of the neural network.

- [ ] **Step 4: Write the Stable Diffusion systems half**

Add side-by-side Mermaid flows:

```text
training: image -> VAE encoder -> latent -> add noise -> denoiser + text condition -> loss
sampling: prompt -> tokenizer/text encoder; random latent -> repeated denoiser/scheduler -> VAE decoder -> image
```

Mark which components normally run once and which repeat for every sampling step. Explain that model families may freeze, fine-tune, or replace components; do not state that every Stable Diffusion training recipe freezes exactly the same modules.

- [ ] **Step 5: Link the chapter and verify**

Link backward to `training.md`, forward to `minimal-ddpm-from-scratch.md`, and outward to the Guidance and noise-parameterization pages. Run:

```bash
python -m pytest tests/test_diffusion_curriculum.py -k architecture -q
make build
```

Expected: architecture test and strict build pass.

- [ ] **Step 6: Commit**

```bash
git add files/diffusion/architecture-and-stable-diffusion.md tests/test_diffusion_curriculum.py
git commit -m "docs: explain diffusion architectures and Stable Diffusion"
```

## Task 9: Split Flow Matching from the Consistency chapter

**Files:**
- Create: `files/diffusion/flow-matching-and-rectified-flow.md`
- Refocus: `files/diffusion/consistency-models-and-rectified-flow.md`
- Modify: `tests/test_diffusion_curriculum.py`

- [ ] **Step 1: Add failing boundary tests**

Append:

```python
    def test_flow_page_owns_velocity_and_rectification(self):
        text = (
            DIFFUSION_ROOT / "flow-matching-and-rectified-flow.md"
        ).read_text(encoding="utf-8")
        for required in (
            "条件概率路径",
            "conditional velocity",
            "marginal velocity",
            "ODE 采样",
            "路径曲率",
            "Rectified Flow",
            "v-prediction",
            "Flow velocity",
            "画面运动速度",
        ):
            self.assertIn(required, text)

    def test_consistency_page_links_to_flow_without_repeating_its_full_section(self):
        text = (
            DIFFUSION_ROOT / "consistency-models-and-rectified-flow.md"
        ).read_text(encoding="utf-8")
        self.assertIn("Consistency Models", text)
        self.assertIn("LCM", text)
        self.assertIn("](flow-matching-and-rectified-flow.md)", text)
        self.assertNotIn("## Flow Matching：直接学习路径上的速度", text)
```

- [ ] **Step 2: Run and verify the boundary failures**

```bash
python -m pytest tests/test_diffusion_curriculum.py -k flow -q
```

Expected: FAIL because the new page is absent and the old page still contains the full Flow sections.

- [ ] **Step 3: Write the Flow page around one linear path convention**

Use a single time convention throughout. If `t=0` is data and `t=1` is noise, state it before:

```math
x_t=(1-t)x_0+t\epsilon,
\qquad
u_t=\frac{d x_t}{dt}=\epsilon-x_0.
```

Explain that other papers reverse time or choose different probability paths. Show how a learned velocity enters an ODE solver, then explain reflow/rectification and why straighter paths can reduce discretization error without guaranteeing one-step fidelity.

- [ ] **Step 4: Refocus the existing Consistency page**

Keep its filename for URL stability. Retain the endpoint-consistency objective, boundary condition, teacher-student training, LCM, and failure modes. Replace detailed Flow derivations with a short interface comparison table and a link to `flow-matching-and-rectified-flow.md`.

- [ ] **Step 5: Verify formulas, links, and build**

```bash
python -m pytest tests/test_diffusion_curriculum.py -k flow -q
make math-quality
make build
```

Expected: boundary tests, math-quality, and strict build pass.

- [ ] **Step 6: Commit**

```bash
git add \
  files/diffusion/flow-matching-and-rectified-flow.md \
  files/diffusion/consistency-models-and-rectified-flow.md \
  tests/test_diffusion_curriculum.py
git commit -m "docs: separate flow matching from consistency models"
```

## Task 10: Refocus advanced pages and finalize navigation

**Files:**
- Modify: `files/diffusion/evolution.md`
- Modify: `files/diffusion/noise-schedules-and-parameterization.md`
- Modify: `files/diffusion/score-matching-sde-and-probability-flow.md`
- Modify: `files/diffusion/guidance-and-conditioning.md`
- Modify: `files/diffusion/inference.md`
- Modify: `files/diffusion/comparison-table.md`
- Modify: `files/diffusion/distillation.md`
- Modify: `files/diffusion/practical-training-recipes-and-failure-analysis.md`
- Modify: `files/diffusion/video-and-multimodal-diffusion.md`
- Modify: `mkdocs.yml:183-194`
- Modify: `tests/test_diffusion_curriculum.py`

- [ ] **Step 1: Add the final ordered-navigation test**

Append this method, which parses Markdown paths from nested YAML lines without depending on group indentation:

```python
    def test_navigation_uses_the_final_grouped_course_order(self):
        nav = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
        start = nav.index("  - 扩散与生成模型:")
        end = nav.index("  - 思考探索:", start)
        section = nav[start:end]
        actual = re.findall(r"(diffusion/[^\s]+\.md)", section)
        self.assertEqual(
            actual,
            [
                "diffusion/index.md",
                "diffusion/beginner-primer.md",
                "diffusion/training.md",
                "diffusion/architecture-and-stable-diffusion.md",
                "diffusion/minimal-ddpm-from-scratch.md",
                "diffusion/evolution.md",
                "diffusion/noise-schedules-and-parameterization.md",
                "diffusion/score-matching-sde-and-probability-flow.md",
                "diffusion/flow-matching-and-rectified-flow.md",
                "diffusion/guidance-and-conditioning.md",
                "diffusion/inference.md",
                "diffusion/comparison-table.md",
                "diffusion/distillation.md",
                "diffusion/consistency-models-and-rectified-flow.md",
                "diffusion/practical-training-recipes-and-failure-analysis.md",
                "diffusion/video-and-multimodal-diffusion.md",
            ],
        )
```

Also append the shared-contract test now that all five pages exist:

```python
    def test_foundation_pages_follow_the_shared_contract(self):
        for filename in self.FOUNDATION_PAGES:
            with self.subTest(filename=filename):
                page = DIFFUSION_ROOT / filename
                self.assertTrue(page.is_file(), filename)
                headings = [
                    line for line in page.read_text(encoding="utf-8").splitlines()
                    if line.startswith("## ")
                ]
                self.assert_ordered_subsequence(self.REQUIRED_HEADINGS, headings)
```

- [ ] **Step 2: Add the final grouped navigation**

Use nested groups `基础主线`, `进阶理论`, `条件与采样`, `少步与工程`, and `应用扩展`, preserving the flattened order above. Modify only the diffusion block.

- [ ] **Step 3: Add prerequisite and next-step bridges to every advanced page**

At the top of each advanced page, state what the page assumes and link to the relevant foundation chapter. At the bottom, replace generic topic-index-only exits with a specific next page plus the index. Do not force every old page into the exact shared H2 contract.

- [ ] **Step 4: Remove duplicated foundation explanations**

For each page, compare against the design responsibility table:

- `evolution.md`: history only, after foundation.
- noise/parameterization: schedule, SNR, epsilon/x0/v coordinates.
- Score/SDE: score, reverse SDE, probability-flow ODE.
- Guidance: condition training, dropout, CFG, structural controls.
- inference: DDPM/DDIM/solver, stochasticity, NFE and latency.
- comparison: post-learning decision guide.
- distillation: teacher/student, trajectory and distribution matching.
- practical training: data, loss buckets, EMA, sampling replay, failure diagnosis.
- video: tokenizer, spatiotemporal backbone, condition, evaluation.

Keep a short definition plus a link when removing duplicated passages; never leave a dangling pronoun or undefined symbol.

- [ ] **Step 5: Add local-link resolution checks**

Add `import unicodedata` and `from urllib.parse import unquote`, then add these helpers and test:

```python
    def markdown_fragment_ids(self, page):
        ids = set()
        for line in page.read_text(encoding="utf-8").splitlines():
            match = re.match(r"^#{1,6}\s+(.+)$", line)
            if match is None:
                continue
            title = match.group(1).strip()
            custom = re.search(r"\s+\{#([^}]+)\}\s*$", title)
            if custom is not None:
                ids.add(custom.group(1))
                continue
            normalized = unicodedata.normalize("NFKC", title).casefold()
            words_only = re.sub(r"[^\w\s-]", "", normalized)
            slug = re.sub(r"[-\s]+", "-", words_only).strip("-")
            if slug:
                ids.add(slug)
        return ids

    def assert_local_links_resolve(self, page):
        text = page.read_text(encoding="utf-8")
        page_links = set()
        for image_marker, target in re.findall(r"(!?)\[[^]]*\]\(([^)]+)\)", text):
            if target.startswith(("http://", "https://")):
                continue
            path, _, fragment = target.partition("#")
            target_file = page if not path else (page.parent / path).resolve()
            self.assertTrue(target_file.is_file(), f"{page.name}: {target}")
            if path and not image_marker and target_file.suffix == ".md":
                page_links.add(path)
            if fragment:
                self.assertIn(
                    unquote(fragment),
                    self.markdown_fragment_ids(target_file),
                    f"{page.name}: {target}",
                )
        return page_links

    def test_all_diffusion_local_links_resolve(self):
        for page in DIFFUSION_ROOT.glob("*.md"):
            with self.subTest(page=page.name):
                links = self.assert_local_links_resolve(page)
                if page.name in self.FOUNDATION_PAGES:
                    self.assertGreaterEqual(len(links), 2, page.name)
```

- [ ] **Step 6: Run the full curriculum and strict build checks**

```bash
python -m pytest tests/test_diffusion_curriculum.py -q
make quality
make build
```

Expected: all curriculum tests pass; quality and strict build exit 0.

- [ ] **Step 7: Commit the advanced-page integration**

```bash
git add files/diffusion mkdocs.yml tests/test_diffusion_curriculum.py
git commit -m "docs: connect the complete diffusion learning path"
```

Before committing, inspect `git diff --cached -- mkdocs.yml` and confirm it contains only the diffusion block.

## Task 11: Complete the six-visual evidence map and source audit

**Files:**
- Modify: `files/diffusion/index.md`
- Modify: `files/diffusion/beginner-primer.md`
- Modify: `files/diffusion/training.md`
- Modify: `files/diffusion/architecture-and-stable-diffusion.md`
- Modify: `files/diffusion/flow-matching-and-rectified-flow.md`
- Reuse without modifying: `files/assets/images/paper-figures/diffusion/ddpm-orthogonal-coefficients.svg`
- Modify: `tests/test_diffusion_curriculum.py`

- [ ] **Step 1: Add a visual-coverage test**

Append:

```python
    def test_foundation_visual_map_has_all_six_groups(self):
        expected = {
            "index.md": ("flowchart LR",),
            "beginner-primer.md": ("diffusion-distribution-path.svg",),
            "training.md": ("ddpm-orthogonal-coefficients.svg",),
            "architecture-and-stable-diffusion.md": (
                "denoiser-interface",
                "stable-diffusion-training-inference",
            ),
            "flow-matching-and-rectified-flow.md": (
                "diffusion-method-coordinate-map",
            ),
        }
        self.assertEqual(sum(map(len, expected.values())), 6)
        for filename, markers in expected.items():
            text = (DIFFUSION_ROOT / filename).read_text(encoding="utf-8")
            for marker in markers:
                self.assertIn(marker, text, f"{filename}: {marker}")
```

Use Mermaid comments such as `%% denoiser-interface` as stable markers without changing rendered output.

- [ ] **Step 2: Add the remaining Mermaid visuals**

Create:

- A denoiser interface diagram separating neural network and scheduler.
- A training/inference Stable Diffusion diagram with repeated and one-time components.
- A method coordinate map whose columns are `learned target`, `generation dynamics`, `requires retraining/distillation`, and `main cost` rather than a misleading chronological arrow.

- [ ] **Step 3: Audit every visual explanation**

Immediately after each visual, write what it demonstrates and what it does not demonstrate. For the distribution SVG, state that the picture shows distribution corruption, not a unique reversible pairing. For the coefficient SVG, state the orthogonal-axis picture is an intuition under a chosen parameterization, not proof that real data and sampled noise are geometrically orthogonal in pixel space.

- [ ] **Step 4: Audit external sources by claim type**

Use DDPM for forward/reverse and loss claims, LDM for latent/cross-attention claims, Score SDE for reverse-SDE/probability-flow claims, and primary Flow/Rectified Flow sources for velocity and rectification claims. Keep tutorials in external-reading sections as teaching aids, not as the sole support for technical equivalence or performance claims.

- [ ] **Step 5: Run visual, math, and build validation**

```bash
python -m pytest tests/test_diffusion_curriculum.py -q
make image-quality
make math-quality
make build
```

Expected: all commands exit 0.

- [ ] **Step 6: Commit the visual and evidence pass**

```bash
git add \
  files/diffusion/index.md \
  files/diffusion/beginner-primer.md \
  files/diffusion/training.md \
  files/diffusion/architecture-and-stable-diffusion.md \
  files/diffusion/flow-matching-and-rectified-flow.md \
  tests/test_diffusion_curriculum.py
git commit -m "docs: add diffusion curriculum visuals and evidence"
```

Stage only the explicitly reviewed image files, not unrelated assets.

## Task 12: Final end-to-end verification and handoff

**Files:**
- Verify: all files above
- No new implementation files expected

- [ ] **Step 1: Run the complete automated verification suite**

In the isolated example environment:

```bash
python -m pytest files/assets/examples/diffusion/test_minimal_ddpm.py -q
python files/assets/examples/diffusion/minimal_ddpm.py smoke --seed 7 --timesteps 8
python -m pytest tests/test_diffusion_curriculum.py tests/test_content_quality_report.py -q
make quality
make image-quality
make math-quality
make build
```

Expected: all tests pass, smoke output contains finite labeled values, every quality command exits 0, and strict build exits 0.

- [ ] **Step 2: Perform the manual site review**

Run:

```bash
make serve DEV_ADDR=127.0.0.1:8010
```

Check desktop and narrow mobile widths for the topic index, `training.md`, the architecture page, the code page, and the Flow page. Verify nested navigation, MathJax wrapping, Mermaid rendering, SVG readability, table horizontal scrolling, and code-copy behavior. Stop the server with `Ctrl-C`.

- [ ] **Step 3: Audit scope and URL stability**

Run:

```bash
git diff --name-status $(git merge-base HEAD dev/v4)..HEAD
git status --short
```

Expected: only planned diffusion pages, assets, tests, `mkdocs.yml`, and plan/spec files appear; no existing diffusion file was renamed or deleted; the worktree is clean.

- [ ] **Step 4: Prepare the integration note**

Record:

- commits created by Tasks 1–11;
- exact automated commands and results;
- manual pages and viewports checked;
- the fact that `mkdocs.yml` must be reconciled with the original checkout's user-owned uncommitted changes before merge;
- any optional MNIST cache or `.venv-diffusion-course` left in the worktree.

- [ ] **Step 5: Use `finishing-a-development-branch` for integration choice**

Do not merge, rebase, delete the worktree, or overwrite the user's current checkout without the user's explicit integration choice.
