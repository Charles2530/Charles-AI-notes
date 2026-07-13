# Prerequisite Math and Foundations Curriculum Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (\`- [ ]\`) syntax for tracking.

**Goal:** Build an explicit learning chain from prerequisite mathematics through AI foundations to downstream topic guides and paper deep dives.

**Architecture:** Keep every existing URL stable, group the navigation by conceptual dependency, add one vector-calculus page and one training-signal bridge page, then deepen only the existing pages that carry unresolved downstream concepts. Every foundations article receives a short prerequisite card, while index pages provide both sequential and reverse lookup routes.

**Tech Stack:** MkDocs Material, Markdown, MathJax/Arithmatex, Mermaid, local SVG assets, pytest content checks.

---

## Working-Tree Rule

The current checkout contains user-owned changes in \`.gitignore\`, \`mkdocs.yml\`, Fisher content, DM0.5 content, and thinking-exploration pages. Do not reset, stash, overwrite, or commit those changes. Read each overlapping file immediately before editing and preserve its current content. Implementation changes remain uncommitted unless the user explicitly requests a commit.

### Task 1: Record the baseline and verify the current site

**Files:**
- Read: \`mkdocs.yml\`
- Read: \`Makefile\`
- Read: \`tests/test_content_quality_report.py\`
- Read: \`files/prerequisite-math/*.md\`
- Read: \`files/foundations/*.md\`

- [ ] **Step 1: Record the dirty working tree**

Run:

~~~bash
git status --short
~~~

Expected: user-owned modifications and untracked files are visible; no file is changed by this command.

- [ ] **Step 2: Run the content quality baseline**

Run:

~~~bash
python -m pytest tests/test_content_quality_report.py
~~~

Expected: the current content test passes before curriculum edits.

- [ ] **Step 3: Run the strict build baseline**

Run:

~~~bash
make build
~~~

Expected: MkDocs strict build exits 0; the Material for MkDocs future-version warning is allowed.

### Task 2: Add the vector-calculus bridge page and visual

**Files:**
- Create: \`files/prerequisite-math/differential-jacobian-and-gradient-flow.md\`
- Create: \`files/assets/images/prerequisite-math/jacobian-local-linear-map.svg\`
- Modify: \`files/prerequisite-math/calculus-and-newton-leibniz.md\`

- [ ] **Step 1: Create the Jacobian visual**

Create a 960 by 420 SVG with three regions:

1. Input perturbation \(\Delta x\) in a two-dimensional space.
2. A local linear map labelled \(J_f(x)\).
3. Output perturbation \(\Delta y\approx J_f(x)\Delta x\), showing rotation, stretching, and compression.

The bottom caption inside the SVG must state:

~~~text
Local rule: a nonlinear map behaves like its Jacobian for a sufficiently small perturbation.
~~~

- [ ] **Step 2: Write the page in story order**

Use these exact H2 sections:

~~~markdown
## 导数其实在做局部线性近似
## Jacobian：把每个输入方向对每个输出的影响排成矩阵
## 梯度和 Jacobian 到底差在哪里
## 向量链式法则：局部线性映射怎样接起来
## JVP 与 VJP：自动微分真正计算的对象
## Softmax Jacobian：改一个 logit，为什么所有概率都会动
## 误差传播：量化扰动为什么可能被后续网络放大
## Jacobian、Hessian 与 Fisher 的边界
## 常见误读
## 读完以后怎么判断
## 相关阅读与下一步
~~~

Include the formulas:

\[
\Delta y\approx J_f(x)\Delta x,
\qquad
[J_f(x)]_{ij}=\frac{\partial f_i}{\partial x_j}
\]

\[
J_{g\circ f}(x)=J_g(f(x))J_f(x)
\]

\[
\frac{\partial p_i}{\partial z_j}=p_i(\delta_{ij}-p_j)
\]

\[
\|\delta f\|_2\lesssim \|J\|_2\|\delta x\|_2
\]

Use one two-input/two-output numerical Jacobian example and one three-class softmax example. Link to autograd, Attn-QAT, quantization comparison, KVSlimmer, and the Fisher section.

- [ ] **Step 3: Keep calculus focused**

After the matrix-backprop example in \`calculus-and-newton-leibniz.md\`, add one paragraph explaining that vector-to-vector derivatives form a Jacobian and link to the new page. Do not duplicate the Jacobian derivation.

- [ ] **Step 4: Validate this batch**

Run:

~~~bash
xmllint --noout files/assets/images/prerequisite-math/jacobian-local-linear-map.svg
rg -n "^## |JVP|VJP|Softmax Jacobian|Hessian 与 Fisher" files/prerequisite-math/differential-jacobian-and-gradient-flow.md
~~~

Expected: XML validation exits 0 and all required sections are found.

### Task 3: Add the training-signal bridge page and visual

**Files:**
- Create: \`files/foundations/gradient-estimators-and-training-signals.md\`
- Create: \`files/assets/images/foundations/gradient-estimator-decision-map.svg\`

- [ ] **Step 1: Create the estimator decision map**

Create a 1000 by 560 SVG with one starting question and four branches:

~~~text
Can the random operation be rewritten as deterministic parameters + parameter-free noise?
Continuous sample -> reparameterization / pathwise gradient
Discrete sample -> score-function or continuous relaxation
Hard non-differentiable operator -> straight-through surrogate
Teacher or old distribution -> stop-gradient, EMA, or importance weighting
~~~

Use a restrained multi-color palette and stable box dimensions. The visual must remain legible at 760 px width.

- [ ] **Step 2: Write the page in story order**

Use these exact H2 sections:

~~~markdown
## 链式法则会在哪些地方断掉
## 重参数化：把随机性搬到参数之外
## Score-function estimator：不能穿过样本，就给样本概率加权
## Straight-Through Estimator：前向做硬选择，反向借一条近似梯度
## Gumbel-Softmax：用连续分布逼近离散选择
## Stop-gradient 与 EMA teacher：有些分支不该被反向更新
## Importance weighting：数据来自旧分布时怎样改权重
## 偏差、方差和适用边界
## 五个下游例子串起来
## 常见误读
## 读完以后怎么判断
## 相关阅读与下一步
~~~

Include the formulas:

\[
z=\mu_\theta(x)+\sigma_\theta(x)\odot\epsilon,
\qquad \epsilon\sim\mathcal N(0,I)
\]

\[
\nabla_\theta\mathbb E_{z\sim p_\theta}[f(z)]
=
\mathbb E[f(z)\nabla_\theta\log p_\theta(z)]
\]

\[
y_{\mathrm{st}}
=
y_{\mathrm{soft}}+
\operatorname{sg}(y_{\mathrm{hard}}-y_{\mathrm{soft}})
\]

\[
\bar\theta_t=\tau\bar\theta_{t-1}+(1-\tau)\theta_t
\]

\[
\mathbb E_p[f(x)]
=
\mathbb E_q\left[\frac{p(x)}{q(x)}f(x)\right]
\]

The five examples must be VAE, DreamerV2, Attn-QAT, JEPA/BYOL-style teacher training, and PPO/off-policy correction. Explicitly distinguish parameter score from diffusion data score and distinguish VAE reparameterization from scheduler parameterization.

- [ ] **Step 3: Validate this batch**

Run:

~~~bash
xmllint --noout files/assets/images/foundations/gradient-estimator-decision-map.svg
rg -n "^## |重参数化|Score-function|Straight-Through|Gumbel|EMA teacher|Importance" files/foundations/gradient-estimators-and-training-signals.md
~~~

Expected: XML validation exits 0 and all required sections are found.

### Task 4: Deepen the five prerequisite-math pages

**Files:**
- Modify: \`files/prerequisite-math/probability-distributions-and-expectation.md\`
- Modify: \`files/prerequisite-math/sampling-estimation-and-calibration.md\`
- Modify: \`files/prerequisite-math/latent-likelihood-and-generative-objectives.md\`
- Modify: \`files/prerequisite-math/dynamic-systems-and-control.md\`
- Modify: \`files/prerequisite-math/calculus-and-newton-leibniz.md\`

- [ ] **Step 1: Add covariance after variance**

Add:

\[
\operatorname{Cov}(X,Y)
=
\mathbb E[(X-\mathbb E[X])(Y-\mathbb E[Y])]
\]

\[
\rho_{XY}
=
\frac{\operatorname{Cov}(X,Y)}
{\sqrt{\operatorname{Var}(X)\operatorname{Var}(Y)}}
\]

\[
\Sigma
=
\mathbb E[(x-\mu)(x-\mu)^\top]
\]

Explain a two-axis robot pose error and how covariance rotates the uncertainty ellipse. State that zero covariance is not generally independence and correlation is not causation.

- [ ] **Step 2: Add importance sampling, Wilson, and bootstrap**

Add the importance-sampling identity and explain support coverage and effective sample collapse. Follow the current success-rate normal interval with the Wilson center and half-width:

\[
\tilde p
=
\frac{\hat p+z^2/(2n)}{1+z^2/n},
\qquad
h
=
\frac{z}{1+z^2/n}
\sqrt{\frac{\hat p(1-\hat p)}{n}+\frac{z^2}{4n^2}}
\]

Add a four-step bootstrap procedure: resample with replacement, recompute metric, repeat, take empirical quantiles. Do not present bootstrap as proof of unbiasedness.

- [ ] **Step 3: Clarify Jensen, reparameterization, and EBM**

In the ELBO section, show:

\[
\log p_\theta(x)
=
\log\mathbb E_{q_\phi(z\mid x)}
\left[
\frac{p_\theta(x,z)}{q_\phi(z\mid x)}
\right]
\ge
\mathbb E_{q_\phi}
\left[
\log\frac{p_\theta(x,z)}{q_\phi(z\mid x)}
\right]
\]

Then link reparameterized sampling to the new training-signal page. Add an EBM section with:

\[
p_\theta(y\mid x)
=
\frac{\exp(-E_\theta(x,y))}
{Z_\theta(x)},
\qquad
Z_\theta(x)=\int \exp(-E_\theta(x,y'))dy'
\]

Explain energy, partition function, optimization/sampling inference, and collapse prevention without reproducing H-JEPA-specific architecture.

- [ ] **Step 4: Add MPC and CEM**

Use:

\[
\min_{a_{t:t+H-1}}
\sum_{k=0}^{H-1}c(s_{t+k},a_{t+k})
+c_T(s_{t+H})
\]

Explain rollout, candidate scoring, execute-first-action, reobserve. Then show one CEM iteration with a Gaussian action-sequence distribution, elite selection, and mean/variance update. Link PlaNet, V-JEPA 2-AC, RSSM/Dreamer, and embodied planning.

- [ ] **Step 5: Check section placement and duplication**

Run:

~~~bash
rg -n "^## .*协方差|^## .*重要性采样|Wilson|bootstrap|^## .*Energy|^## .*MPC|^## .*CEM|Jacobian" files/prerequisite-math
~~~

Expected: each concept has one primary explanation and links elsewhere rather than duplicate derivations.

### Task 5: Deepen the four foundations pages

**Files:**
- Modify: \`files/foundations/pretraining-objectives-and-representation-learning.md\`
- Modify: \`files/foundations/probability-latent-variables-and-generative-models.md\`
- Modify: \`files/foundations/autograd-activation-checkpointing-and-memory.md\`
- Modify: \`files/foundations/model-compression-pruning-and-nas-basics.md\`

- [ ] **Step 1: Explain representation collapse**

Add one section after contrastive learning that defines constant-output collapse and compares:

| Mechanism | What prevents collapse | What it does not prove |
| --- | --- | --- |
| Negative samples | Raises incompatible pairs | Complete coverage of high-dimensional negatives |
| stop-gradient | Prevents both branches chasing each other | Informative target representations |
| EMA teacher | Provides a slower target | Correct semantic invariance |
| variance/covariance regularization | Keeps dimensions active and less redundant | Downstream control usefulness |
| bottleneck/capacity constraint | Limits answer-copying shortcuts | Sufficient state for every task |

Connect H-JEPA, V-JEPA, VICReg, and BYOL without turning the page into a paper survey.

- [ ] **Step 2: Compare generative model interfaces**

Add a table comparing normalized likelihood, latent-variable likelihood, energy-based compatibility, implicit generation, and score-based generation across training signal, tractable density, and inference method. Link mathematical details to the latent-objectives and training-signal pages.

- [ ] **Step 3: Clarify VJP in autograd**

Explain that backward propagates an upstream cotangent through a VJP:

\[
v^\top J_f(x)
\]

and normally does not materialize the full Jacobian. Connect activation storage to the local values required by these products.

- [ ] **Step 4: Bridge STE to QAT**

Add a concise paragraph in compression/pruning/NAS explaining that round/argmax masks are not differentiable, QAT often uses STE, and the estimator is biased. Link the training-signal page and Attn-QAT.

- [ ] **Step 5: Validate terminology**

Run:

~~~bash
rg -n "表示坍缩|EMA teacher|variance/covariance|能量模型|VJP|Straight-Through|有偏" files/foundations
~~~

Expected: all terms appear in their intended primary page and have cross-links.

### Task 6: Add prerequisite cards to every foundations article

**Files:**
- Modify: all \`files/foundations/*.md\` except \`files/foundations/index.md\`

- [ ] **Step 1: Add the standardized card**

Place the card after the opening introduction and before the first H2:

~~~markdown
!!! tip "数学先修"
    本文会用到 [具体概念](../prerequisite-math/file.md)。如果这些公式还不熟，先读对应数学页；已经熟悉可直接继续。
~~~

Use the exact prerequisite mapping in the approved design:

- symbols -> prerequisite index
- tensors -> linear algebra, numerics
- linear/GEMM -> linear algebra, numerics
- autograd -> calculus, Jacobian
- gradient estimators -> calculus, probability, sampling
- runtime numerics -> numerics, linear algebra
- Transformer -> linear algebra, sequence/masking, probability
- positional encoding -> sequence/masking, linear algebra
- normalization/residual -> linear algebra, optimization, Jacobian
- convolution -> convolution/frequency, linear algebra
- MoE -> discrete selection, probability, numerics
- Mamba -> dynamic systems, sequence/masking, calculus
- optimization/training -> optimization, calculus, sampling
- pretraining objectives -> information theory, latent objectives, linear algebra
- probability/latent foundations -> probability, information theory, latent objectives
- distillation -> information theory, probability, gradient estimators
- data -> probability, causality/distribution shift
- metrics -> sampling, scaling curves
- generalization -> causality/distribution shift, sampling, optimization
- generation -> probability, discrete selection, sequence/masking
- multimodal reasoning -> linear algebra, information theory, 3D geometry
- Prompt/CoT/RAG -> sequence/masking, probability, linear algebra
- compression/NAS -> spectral/Fisher, numerics, gradient estimators

- [ ] **Step 2: Audit coverage**

Run:

~~~bash
for f in files/foundations/*.md; do
  [ "$f" = "files/foundations/index.md" ] && continue
  rg -q '!!! tip "数学先修"' "$f" || echo "missing $f"
done
~~~

Expected: no output.

- [ ] **Step 3: Audit links**

Run:

~~~bash
rg -n '\.\./prerequisite-math/' files/foundations/*.md
~~~

Expected: every foundations article appears at least once.

### Task 7: Rebuild the two indexes, navigation, and source ledger

**Files:**
- Modify: \`files/prerequisite-math/index.md\`
- Modify: \`files/foundations/index.md\`
- Modify: \`mkdocs.yml\`
- Modify: \`files/references/external-reading-ledger.md\`

- [ ] **Step 1: Rebuild the prerequisite index**

Create five H2 learning groups matching navigation. Each group gets a short purpose paragraph and a table of questions/pages. Replace the dense dependency graph with a smaller Mermaid graph containing:

~~~text
linear algebra -> calculus -> Jacobian -> optimization/autograd
probability -> sampling/information/stochastic processes
probability + calculus -> latent objectives -> diffusion/world models
stochastic processes -> RL/control -> 3D embodied systems
spectral + Jacobian -> Fisher/Hessian -> compression/quantization
~~~

Add a reverse-lookup table for training/quantization, diffusion, RL/world models, VLM/VLA/3D, and inference/operators.

- [ ] **Step 2: Rebuild the foundations index**

List all 23 foundations articles after the new page is added. Include a four-column route table:

~~~text
Math prerequisite | Foundation mechanism | Topic entry | Paper example
~~~

Cover distributed training, low-bit/quantization, inference/KV, diffusion, world models, VLM, VLA/embodied, and RAG/Agent.

- [ ] **Step 3: Update navigation without losing current user edits**

Read the live \`mkdocs.yml\` immediately before patching. Replace only the \`前置数学知识\` and \`基础概念\` blocks. Preserve the current DM0.5 and thinking-exploration entries.

- [ ] **Step 4: Register external sources**

Add source-ledger entries for primary references actually used by the two new pages, choosing from:

- Baydin et al., Automatic Differentiation in Machine Learning
- Kingma and Welling, Auto-Encoding Variational Bayes
- Williams, REINFORCE
- Jang et al., Categorical Reparameterization with Gumbel-Softmax
- Grill et al., BYOL
- Bardes et al., VICReg
- Schulman et al., PPO

Do not add a source that is not cited or used.

- [ ] **Step 5: Validate navigation coverage**

Run:

~~~bash
rg -n "微分、Jacobian 与梯度传播|梯度估计、离散变量与训练信号|数学语言与变化|计算、梯度与运行时" mkdocs.yml files/prerequisite-math/index.md files/foundations/index.md
~~~

Expected: both new pages and both group names appear in navigation and indexes.

### Task 8: Full verification and rendered-site audit

**Files:**
- Verify: all changed Markdown and SVG files
- Verify: generated \`site/\` output

- [ ] **Step 1: Scan for placeholders and whitespace errors**

Run:

~~~bash
rg -n "TODO|TBD|待补|占位|FIXME|后续补充|未完成" files/prerequisite-math files/foundations
git diff --check
~~~

Expected: no placeholder matches in changed pages and no whitespace errors.

- [ ] **Step 2: Validate both SVG files**

Run:

~~~bash
xmllint --noout \
  files/assets/images/prerequisite-math/jacobian-local-linear-map.svg \
  files/assets/images/foundations/gradient-estimator-decision-map.svg
~~~

Expected: exit 0.

- [ ] **Step 3: Run content tests**

Run:

~~~bash
python -m pytest tests/test_content_quality_report.py
~~~

Expected: 1 test passes.

- [ ] **Step 4: Run strict build**

Run:

~~~bash
make build
~~~

Expected: MkDocs strict build exits 0.

- [ ] **Step 5: Verify generated pages and assets**

Run:

~~~bash
test -f site/prerequisite-math/differential-jacobian-and-gradient-flow/index.html
test -f site/foundations/gradient-estimators-and-training-signals/index.html
test -f site/assets/images/prerequisite-math/jacobian-local-linear-map.svg
test -f site/assets/images/foundations/gradient-estimator-decision-map.svg
rg -n "Softmax Jacobian|五个下游例子串起来|数学先修" \
  site/prerequisite-math/differential-jacobian-and-gradient-flow/index.html \
  site/foundations/gradient-estimators-and-training-signals/index.html \
  site/foundations/autograd-activation-checkpointing-and-memory/index.html
~~~

Expected: every file exists and rendered text is found.

- [ ] **Step 6: Review the final diff**

Run:

~~~bash
git status --short
git diff --stat
git diff --check
~~~

Expected: only approved curriculum files plus pre-existing user-owned changes are present; no unrelated file has been modified.
