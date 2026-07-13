"""Small NumPy objectives used in the LLM reinforcement-learning chapter."""

from __future__ import annotations

import argparse
import math

import numpy as np


def _real_finite_array(value, name: str) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a regular numeric array") from error
    if array.size == 0 or array.dtype == np.bool_ or not np.issubdtype(
        array.dtype, np.number
    ):
        raise ValueError(f"{name} must be a non-empty real numeric array")
    if np.iscomplexobj(array):
        raise ValueError(f"{name} must be real")
    result = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _positive_finite_scalar(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or np.iscomplexobj(value):
        raise ValueError(f"{name} must be a positive finite real scalar")
    try:
        scalar_array = np.asarray(value)
        if scalar_array.ndim != 0:
            raise ValueError(f"{name} must be a positive finite real scalar")
        scalar = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a positive finite real scalar") from error
    if not math.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be a positive finite real scalar")
    return scalar


def group_advantages(rewards, epsilon: float = 1e-8) -> np.ndarray:
    """Normalize rewards within each last-axis group.

    A one-dimensional input is one candidate group. A two-dimensional input is
    a batch of groups. Groups whose population standard deviation is at most
    ``epsilon`` receive exact zero advantages.
    """

    reward_array = _real_finite_array(rewards, "rewards")
    epsilon_value = _positive_finite_scalar(epsilon, "epsilon")
    if reward_array.ndim not in (1, 2):
        raise ValueError("rewards must be one- or two-dimensional")
    if reward_array.shape[-1] < 2:
        raise ValueError("each group must contain at least two candidates")

    group_scale = np.max(np.abs(reward_array), axis=-1, keepdims=True)
    scaled_rewards = np.zeros_like(reward_array)
    np.divide(
        reward_array,
        group_scale,
        out=scaled_rewards,
        where=group_scale > 0.0,
    )
    centered = scaled_rewards - scaled_rewards.mean(axis=-1, keepdims=True)
    scaled_std = scaled_rewards.std(axis=-1, keepdims=True)
    scaled_epsilon = np.full_like(group_scale, np.inf)
    with np.errstate(over="ignore", under="ignore"):
        np.divide(
            epsilon_value,
            group_scale,
            out=scaled_epsilon,
            where=group_scale > 0.0,
        )

    advantages = np.zeros_like(centered)
    stable_groups = np.broadcast_to(scaled_std > scaled_epsilon, centered.shape)
    denominator = np.broadcast_to(scaled_std + scaled_epsilon, centered.shape)
    advantages[stable_groups] = centered[stable_groups] / denominator[stable_groups]
    if not np.all(np.isfinite(advantages)):
        raise ValueError("group advantage normalization produced non-finite values")
    return advantages


def sampled_forward_kl(policy_logp, reference_logp) -> np.ndarray:
    """Return sampled pointwise ``log pi(a|s) - log pi_ref(a|s)`` values.

    This log-ratio is a per-sample estimator used in KL penalties. Individual
    values can be negative; this function does not compute the full expected KL.
    """

    policy = _real_finite_array(policy_logp, "policy_logp")
    reference = _real_finite_array(reference_logp, "reference_logp")
    if policy.shape != reference.shape:
        raise ValueError("policy_logp and reference_logp must have identical shapes")
    return policy - reference


def clipped_policy_loss(
    new_logp, old_logp, advantages, clip_epsilon: float
) -> float:
    """Return the negative mean PPO clipped surrogate objective."""

    new = _real_finite_array(new_logp, "new_logp")
    old = _real_finite_array(old_logp, "old_logp")
    advantage_array = _real_finite_array(advantages, "advantages")
    epsilon_value = _positive_finite_scalar(clip_epsilon, "clip_epsilon")
    if epsilon_value >= 1.0:
        raise ValueError("clip_epsilon must be strictly between zero and one")
    if new.shape != old.shape or new.shape != advantage_array.shape:
        raise ValueError("new_logp, old_logp, and advantages must have identical shapes")

    with np.errstate(over="raise", invalid="raise"):
        try:
            ratio = np.exp(new - old)
            unclipped = ratio * advantage_array
            clipped = np.clip(ratio, 1.0 - epsilon_value, 1.0 + epsilon_value)
            loss = -np.mean(np.minimum(unclipped, clipped * advantage_array))
        except FloatingPointError as error:
            raise ValueError("log-probability ratio produced a non-finite objective") from error
    if not math.isfinite(float(loss)):
        raise ValueError("clipped policy loss must be finite")
    return float(loss)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    prompts = ("format", "arithmetic")
    rewards = np.array([[1.0, 1.0, 1.0], [0.0, 1.0, 2.0]])
    advantages = group_advantages(rewards)
    old_logp = np.log(np.array([[0.30, 0.40, 0.30], [0.20, 0.50, 0.30]]))
    new_logp = old_logp + rng.normal(0.0, 0.04, size=old_logp.shape)
    reference_logp = old_logp - 0.03
    pointwise_kl = sampled_forward_kl(new_logp, reference_logp)
    loss = clipped_policy_loss(new_logp, old_logp, advantages, 0.2)

    print("prompt      candidate  raw_reward  group_advantage  pointwise_kl_penalty")
    for prompt_index, prompt in enumerate(prompts):
        for candidate_index in range(rewards.shape[1]):
            print(
                f"{prompt:<11} {candidate_index:<9d} "
                f"{rewards[prompt_index, candidate_index]:>10.3f} "
                f"{advantages[prompt_index, candidate_index]:>16.6f} "
                f"{pointwise_kl[prompt_index, candidate_index]:>20.6f}"
            )
    print("pointwise_kl_penalty: sampled log-ratio estimator, not full KL")
    print("zero-variance group -> exact zero advantages")
    print(f"clipped_policy_loss={loss:.6f}")


if __name__ == "__main__":
    main()
