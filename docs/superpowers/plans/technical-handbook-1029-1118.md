# Technical handbook implementation plan

Goal: Turn the existing AI reference site into a coherent technical introduction with worked mechanisms and free topic navigation. No course, checkpoints, exams or mandatory progression.

Architecture: Keep MkDocs/Material, existing page URLs, local reader storage and paper article ownership. Build a real root home, organize existing pages by technical responsibility, and rewrite six foundational pages with connected numerical explanations. Add a small dependency-free reference example only where it makes the mathematics verifiable. Paper regeneration is owned by the concurrent chat.

Technology: Markdown, MkDocs Material, Python, static CSS, existing browser scripts.

## Boundaries

- Work only in current main, no branches/worktrees, no broad staging/reset/checkout.
- Concurrent chat 01a185a0-ddc6-7182-b712-5f3c51f54837 owns paper article bodies, images and generation tools. Parent coordinates mkdocs.yml and paper index. Confirm shared audit ownership before mutation.
- Existing Markdown names are canonical site URL contracts and stay unchanged, as approved. New non-index Markdown uses Shanghai timestamps. Preserve prior delivered reports.
- Base strict build passes; pytest has 27 failed / 164 passed / 12 skipped. Do not hide unrelated baseline failures.

## Task A — Foundational explanations

Files: six existing files in files/foundations: tensors-shapes-and-computation-graphs.md, linear-layers-mlp-and-gemm.md, optimization-and-training-basics.md, transformer-attention-and-tokenization.md, generation-and-decoding-basics.md, autograd-activation-checkpointing-and-memory.md.

- [ ] Read all six sources and current related articles; preserve valid technical distinctions and citations.
- [ ] Rewrite around a complete mechanism with numerical input, intermediate values and output. Explain notation before use and separate training from inference.
- [ ] Keep core proof/derivation when useful; move unrelated system details to existing links. No forced chapter template, exercises or completion criteria.
- [ ] Check local paths, formulas and explanatory calculations; report each substantive factual correction.

## Task B — Homepage, taxonomy and index connections

Files: files/index.md (required home filename), mkdocs.yml, files/foundations/index.md, files/prerequisite-math/index.md, new files/model-architectures/index.md and files/technical-reference/index.md, topic indexes in training, inference, training-and-inference-systems, diffusion, reinforcement-learning, world-models-and-embodied-ai, vlm; paper index after coordination; README.md; a new handbook.css resource.

- [ ] Add root home with concise technical map, eight thematic destinations and question-based direct links. Reuse visual tokens, emphasize readable text, no progress UI.
- [ ] Move math into foundational reference, separate model structures, training, inference/efficiency, generation, RL, multimodal/embodied and papers; tools/QA/exploration get an auxiliary entry. Every existing nav page remains accessible.
- [ ] Rewrite indexes around objects and mechanism relationships, not lists of lecture tasks. Ensure VLM's currently omitted pages have a home.
- [ ] Put paper index before articles and move PDF reader to tools with a backlink. Preserve IndexedDB keys and paper URLs.
- [ ] Add navigation invariants: root home, paper landing, full former-page coverage, no duplicate nav paths or missing targets.
- [ ] Build strict and inspect desktop/narrow viewport and old links.

## Task C — Verifiable mechanism example

Files: files/assets/examples/foundations/mechanisms.py, tests/test_foundation_mechanisms.py.

- [ ] Write tests for numerical affine transform, finite difference vs analytic linear-regression gradient, causal attention no future leakage, stable softmax and greedy vs sequence probability counterexample.
- [ ] Run tests and observe expected missing implementation failure.
- [ ] Implement small Python standard-library reference functions and CLI emitting numeric values. Validate dimensions and finite softmax inputs; use exact conventions documented by the articles.
- [ ] Run tests, CLI and cross-check prose numeric values. No training or reproduction claims.

## Task D — Honest quality checks

Files: scripts/paper_deep_dive_audit.py and tests/test_paper_deep_dive_audit.py after peer coordination; scripts/content_quality_report.py and tests/test_content_quality_report.py.

- [ ] Add regression cases accepting meaningful alternative headings and long evidence-rich articles while still rejecting missing metadata, bad local links/images, missing source information and index omissions.
- [ ] Run new tests red, replace fixed-title/length semantic gates with non-blocking review hints; retain actual integrity checks. Output must say structural verification, not evidence correctness.
- [ ] Exclude vendor documentation from content pedagogy statistics while checking example pages and normal resources.
- [ ] Investigate baseline exact-string tests and missing reading links separately. Update stale tests to structural/behavior invariants; repair genuine missing links using their underlying meaning. Avoid editing peer-owned papers to satisfy wording assertions.

## Task E — Integration and review

- [ ] Compare Git status with ownership boundaries. Run relevant tests, make quality, strict build, full pytest and mathematics checks for changed pages.
- [ ] Read final source and inspect rendered formulas/tables; verify root, topic, paper and tools navigation at desktop and narrow width.
- [ ] Independent review of spec coverage and correctness, fix substantive findings and re-run affected checks.
- [ ] Send exact changed files and paper integration notes to peer chat. Deliver a timestamped change report with actual checks and remaining evidence limitations. No automatic deployment of unfinished peer work.
