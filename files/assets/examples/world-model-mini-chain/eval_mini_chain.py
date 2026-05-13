#!/usr/bin/env python3
"""Evaluate the documentation mini-chain fixture.

This is a tiny, dependency-free check for the docs. It does not claim to train a
real model; it makes the evidence module reproducible by computing token counts,
rollout ranking metrics, risk calibration, and failure replay rows from fixtures.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load_jsonl(name: str) -> list[dict]:
    rows = []
    with (ROOT / name).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def token_stats(episodes: list[dict]) -> dict:
    raw = 0
    compressed = 0
    for ep in episodes:
        obs = ep["observations"]
        for cam in obs["cameras"]:
            raw += cam["frames"] * cam["patch_tokens_per_frame"]
            compressed += cam["frames"] * cam["resampler_tokens_per_frame"]
        compressed += (
            obs["memory_tokens"]
            + obs["language_tokens"]
            + obs["proprio_tokens"]
            + obs["action_tokens"]
        )
    return {
        "raw_patch_tokens": raw,
        "compressed_state_tokens": compressed,
        "compression_ratio": raw / compressed,
    }


def risk_ece(rows: list[dict]) -> float:
    bins = [(0.0, 0.33), (0.33, 0.66), (0.66, 1.01)]
    total = len(rows)
    ece = 0.0
    for lo, hi in bins:
        bucket = [r for r in rows if lo <= r["predicted_risk"] < hi]
        if not bucket:
            continue
        conf = sum(r["predicted_risk"] for r in bucket) / len(bucket)
        acc = sum(1.0 if r["actual_violation"] else 0.0 for r in bucket) / len(bucket)
        ece += (len(bucket) / total) * abs(conf - acc)
    return ece


def rollout_metrics(rows: list[dict], sensitivity_threshold: float = 8.0) -> dict:
    by_episode: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_episode[row["episode_id"]].append(row)

    sensitivity_pass = 0
    top1_safe_success = 0
    failure_replay = []

    for episode_id, candidates in sorted(by_episode.items()):
        max_delta = max(r["latent_delta_to_noop"] for r in candidates)
        if max_delta >= sensitivity_threshold:
            sensitivity_pass += 1

        top = max(candidates, key=lambda r: r["predicted_success"] - r["predicted_risk"])
        if top["actual_success"] and not top["actual_violation"]:
            top1_safe_success += 1
        else:
            failure_replay.append(top)

    return {
        "episodes": len(by_episode),
        "action_sensitivity_pass": sensitivity_pass,
        "top1_safe_success_agreement": top1_safe_success,
        "risk_ece": risk_ece(rows),
        "failure_replay": failure_replay,
    }


def main() -> None:
    episodes = load_jsonl("episodes.jsonl")
    rollouts = load_jsonl("rollouts.jsonl")
    tokens = token_stats(episodes)
    metrics = rollout_metrics(rollouts)

    print("# world-model-mini-chain")
    print()
    print("| Metric | Value |")
    print("| --- | ---: |")
    print(f"| episodes | {len(episodes)} |")
    print(f"| raw_patch_tokens | {tokens['raw_patch_tokens']} |")
    print(f"| compressed_state_tokens | {tokens['compressed_state_tokens']} |")
    print(f"| compression_ratio | {tokens['compression_ratio']:.2f}x |")
    print(
        "| action_sensitivity_pass | "
        f"{metrics['action_sensitivity_pass']}/{metrics['episodes']} |"
    )
    print(
        "| top1_safe_success_agreement | "
        f"{metrics['top1_safe_success_agreement']}/{metrics['episodes']} |"
    )
    print(f"| risk_ece_3bin | {metrics['risk_ece']:.3f} |")
    print()
    print("## failure replay")
    print()
    print("| episode | top_candidate | predicted_risk | predicted_success | actual_event |")
    print("| --- | --- | ---: | ---: | --- |")
    for row in metrics["failure_replay"]:
        print(
            f"| {row['episode_id']} | {row['candidate']} | "
            f"{row['predicted_risk']:.2f} | {row['predicted_success']:.2f} | "
            f"{row['actual_event']} |"
        )


if __name__ == "__main__":
    main()
