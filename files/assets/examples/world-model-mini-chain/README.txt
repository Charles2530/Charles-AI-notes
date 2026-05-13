World Model Mini-Chain Fixture

This directory contains a tiny reproducible fixture used by the docs:

- episodes.jsonl: four aligned episode summaries with camera token counts.
- rollouts.jsonl: candidate action rollouts with predicted and observed outcomes.
- training-config.yaml: a minimal training/evaluation config.
- eval_mini_chain.py: a dependency-free script that computes token, rollout, calibration, and failure replay metrics.

Run:

python3 files/assets/examples/world-model-mini-chain/eval_mini_chain.py

This is a documentation fixture, not a claim about a trained production model.
