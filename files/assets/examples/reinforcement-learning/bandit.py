"""Seeded 10-armed bandit comparison for the exploration chapter."""

import argparse

import numpy as np


def update_sample_average(values, counts, arm, reward):
    counts[arm] += 1
    values[arm] += (reward - values[arm]) / counts[arm]


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
    values = np.asarray(values)
    counts = np.asarray(counts)
    if values.ndim != 1 or counts.ndim != 1 or values.shape != counts.shape:
        raise ValueError("values and counts must be matching one-dimensional arrays")
    if values.size == 0:
        raise ValueError("values and counts must not be empty")
    if step <= 0:
        raise ValueError("step must use the one-based convention and be positive")

    untried = np.flatnonzero(counts == 0)
    if untried.size:
        return int(untried[0])

    scores = values + np.sqrt(2.0 * np.log(step) / counts)
    return int(np.argmax(scores))


def _run_policy(true_means, noise, choose_action):
    values = np.zeros_like(true_means)
    counts = np.zeros(true_means.size, dtype=int)
    rewards = []
    regret = 0.0
    best_arm = int(np.argmax(true_means))
    best_selections = 0

    for index in range(noise.shape[0]):
        arm = choose_action(values, counts, index + 1)
        reward = float(true_means[arm] + noise[index, arm])
        update_sample_average(values, counts, arm, reward)
        rewards.append(reward)
        regret += float(true_means[best_arm] - true_means[arm])
        best_selections += arm == best_arm

    return {
        "cumulative_reward": float(np.sum(rewards)),
        "average_regret": float(regret / noise.shape[0]),
        "best_arm_fraction": float(best_selections / noise.shape[0]),
    }


def run_experiment(seed, steps, epsilon):
    """Compare epsilon-greedy and UCB on one fixed seeded 10-arm task."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be in [0, 1]")

    environment_rng = np.random.default_rng(seed)
    action_rng = np.random.default_rng(seed)
    true_means = environment_rng.normal(0.0, 1.0, size=10)
    noise = environment_rng.normal(0.0, 1.0, size=(steps, true_means.size))

    epsilon_summary = _run_policy(
        true_means,
        noise,
        lambda values, _counts, _step: epsilon_greedy(
            values, epsilon, action_rng
        ),
    )
    ucb_summary = _run_policy(
        true_means,
        noise,
        lambda values, counts, step: ucb_action(values, counts, step),
    )
    return {"epsilon_greedy": epsilon_summary, "ucb": ucb_summary}


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
