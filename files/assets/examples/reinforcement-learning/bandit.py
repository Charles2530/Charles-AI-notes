"""Seeded 10-armed bandit comparison for the exploration chapter."""

import argparse
import numbers

import numpy as np


def _make_rngs(seed):
    """Return independent, reproducible RNGs for the task and policy."""
    task_seed, policy_seed = np.random.SeedSequence(seed).spawn(2)
    return np.random.default_rng(task_seed), np.random.default_rng(policy_seed)


def _validate_values_and_counts(values, counts):
    values_array = np.asarray(values)
    counts_array = np.asarray(counts)
    if (
        values_array.ndim != 1
        or counts_array.ndim != 1
        or values_array.shape != counts_array.shape
    ):
        raise ValueError("values and counts must be matching one-dimensional arrays")
    if values_array.size == 0:
        raise ValueError("values and counts must not be empty")
    if not (
        np.issubdtype(counts_array.dtype, np.integer)
        or np.issubdtype(counts_array.dtype, np.floating)
    ):
        raise ValueError("counts must contain nonnegative integers")
    if (
        not np.all(np.isfinite(counts_array))
        or np.any(counts_array < 0)
        or np.any(counts_array != np.floor(counts_array))
    ):
        raise ValueError("counts must contain nonnegative integers")
    return values_array, counts_array


def _validate_update_inputs(values, counts, arm, reward):
    if not isinstance(values, np.ndarray) or not isinstance(counts, np.ndarray):
        raise ValueError("values and counts must be NumPy arrays")
    if (
        values.ndim != 1
        or counts.ndim != 1
        or values.shape != counts.shape
        or values.size == 0
    ):
        raise ValueError("values and counts must be matching non-empty 1D arrays")
    if not np.issubdtype(values.dtype, np.floating) or not values.flags.writeable:
        raise ValueError("values must be a writable floating-point array")
    if not np.issubdtype(counts.dtype, np.integer) or not counts.flags.writeable:
        raise ValueError("counts must be a writable integer array")
    if np.any(counts < 0):
        raise ValueError("counts must contain nonnegative integers")
    if (
        isinstance(arm, (bool, np.bool_))
        or not isinstance(arm, (int, np.integer))
        or not 0 <= arm < values.size
    ):
        raise ValueError("arm must index values and counts")
    if (
        isinstance(reward, (bool, np.bool_))
        or not isinstance(reward, numbers.Real)
        or not np.isfinite(reward)
    ):
        raise ValueError("reward must be a finite real number")


def update_sample_average(values, counts, arm, reward):
    _validate_update_inputs(values, counts, arm, reward)
    new_count = int(counts[arm]) + 1
    if new_count > np.iinfo(counts.dtype).max:
        raise ValueError("count cannot be incremented without overflow")
    new_value = float(values[arm] + (reward - values[arm]) / new_count)

    values[arm] = new_value
    counts[arm] = new_count


def epsilon_greedy(values, epsilon, rng):
    """Choose randomly with probability epsilon, otherwise the first argmax."""
    values = np.asarray(values)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("values must be a non-empty one-dimensional array")
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be in [0, 1]")

    if rng.random() < epsilon:
        return int(rng.integers(values.size))
    return int(np.argmax(values))


def ucb_action(values, counts, step):
    """Choose UCB1 action at a one-based interaction step.

    The first untried arm is returned before scores are computed, avoiding
    division by zero and log(0).
    """
    values, counts = _validate_values_and_counts(values, counts)
    if (
        isinstance(step, (bool, np.bool_))
        or not isinstance(step, (int, np.integer))
        or step <= 0
    ):
        raise ValueError("step must use the one-based convention and be positive")
    if step != sum(int(count) for count in counts) + 1:
        raise ValueError("step must equal counts.sum() + 1")

    untried = np.flatnonzero(counts == 0)
    if untried.size:
        return int(untried[0])

    scores = values + np.sqrt(2.0 * np.log(step) / counts)
    return int(np.argmax(scores))


def _run_policies(true_means, noise_rows, action_selectors):
    runs = {
        method: {
            "values": np.zeros_like(true_means),
            "counts": np.zeros(true_means.size, dtype=int),
            "cumulative_reward": 0.0,
            "regret": 0.0,
            "best_selections": 0,
        }
        for method in action_selectors
    }
    best_arm = int(np.argmax(true_means))
    steps = 0

    for steps, noise_row in enumerate(noise_rows, start=1):
        for method, choose_action in action_selectors.items():
            run = runs[method]
            arm = choose_action(run["values"], run["counts"], steps)
            reward = float(true_means[arm] + noise_row[arm])
            update_sample_average(run["values"], run["counts"], arm, reward)
            run["cumulative_reward"] += reward
            run["regret"] += float(true_means[best_arm] - true_means[arm])
            run["best_selections"] += arm == best_arm

    return {
        method: {
            "cumulative_reward": float(run["cumulative_reward"]),
            "average_regret": float(run["regret"] / steps),
            "best_arm_fraction": float(run["best_selections"] / steps),
        }
        for method, run in runs.items()
    }


def run_experiment(seed, steps, epsilon):
    """Compare epsilon-greedy and UCB on one fixed seeded 10-arm task."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be in [0, 1]")

    environment_rng, action_rng = _make_rngs(seed)
    true_means = environment_rng.normal(0.0, 1.0, size=10)
    noise_rows = (
        environment_rng.normal(0.0, 1.0, size=true_means.size)
        for _ in range(steps)
    )
    return _run_policies(
        true_means,
        noise_rows,
        {
            "epsilon_greedy": lambda values, _counts, _step: epsilon_greedy(
                values, epsilon, action_rng
            ),
            "ucb": lambda values, counts, step: ucb_action(values, counts, step),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=2000)
    args = parser.parse_args()

    summaries = run_experiment(args.seed, args.steps, epsilon=0.1)
    for method, summary in summaries.items():
        print(
            f"{method}: cumulative_reward={summary['cumulative_reward']:.3f}, "
            f"average_regret={summary['average_regret']:.3f}, "
            f"best_arm_fraction={summary['best_arm_fraction']:.3f}"
        )


if __name__ == "__main__":
    main()
