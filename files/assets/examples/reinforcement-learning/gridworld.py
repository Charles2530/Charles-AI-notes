"""Small deterministic GridWorld and tabular reinforcement-learning updates."""

import argparse
import numbers

import numpy as np


class GridWorld:
    """A 2x3 map with row-major states 0..5 and terminal goal state 5.

    Actions 0, 1, 2, and 3 move up, right, down, and left. Moving into a
    boundary leaves the agent in place. Non-terminal moves reward -1 and the
    transition into the goal rewards 0.
    """

    n_rows = 2
    n_columns = 3
    n_states = 6
    n_actions = 4
    start_state = 0
    terminal_state = 5
    _moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

    def __init__(self):
        self.state = self.start_state

    def reset(self):
        self.state = self.start_state
        return self.state

    def transition(self, state, action):
        self._validate_state(state)
        self._validate_action(action)
        if state == self.terminal_state:
            return state, 0.0, True

        row, column = divmod(state, self.n_columns)
        row_delta, column_delta = self._moves[action]
        next_row = min(max(row + row_delta, 0), self.n_rows - 1)
        next_column = min(max(column + column_delta, 0), self.n_columns - 1)
        next_state = next_row * self.n_columns + next_column
        terminated = next_state == self.terminal_state
        reward = 0.0 if terminated else -1.0
        return next_state, reward, terminated

    def step(self, action):
        next_state, reward, terminated = self.transition(self.state, action)
        self.state = next_state
        return next_state, reward, terminated

    def _validate_state(self, state):
        if not isinstance(state, (int, np.integer)) or not 0 <= state < self.n_states:
            raise ValueError(f"state must be an integer in [0, {self.n_states})")

    def _validate_action(self, action):
        if not isinstance(action, (int, np.integer)) or not 0 <= action < self.n_actions:
            raise ValueError(f"action must be an integer in [0, {self.n_actions})")


def _finite_real(value, name):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, numbers.Real)
        or not np.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite real number")
    return float(value)


def _validate_index(value, size, name):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or not 0 <= value < size
    ):
        raise ValueError(f"{name} must be an integer in [0, {size})")


def _validate_gamma(gamma):
    gamma = _finite_real(gamma, "gamma")
    if not 0.0 <= gamma < 1.0:
        raise ValueError("gamma must be in [0, 1)")
    return gamma


def _validate_update_arrays(values, counts):
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


def _validate_solver_settings(tolerance, max_iterations):
    tolerance = _finite_real(tolerance, "tolerance")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    if (
        isinstance(max_iterations, (bool, np.bool_))
        or not isinstance(max_iterations, (int, np.integer))
        or max_iterations <= 0
    ):
        raise ValueError("max_iterations must be a positive integer")
    return tolerance, int(max_iterations)


def policy_evaluation(env, policy, gamma=0.9, tolerance=1e-10, max_iterations=10_000):
    """Evaluate a tabular stochastic policy by iterative Bellman updates."""
    gamma = _validate_gamma(gamma)
    tolerance, max_iterations = _validate_solver_settings(
        tolerance, max_iterations
    )
    policy = np.asarray(policy, dtype=float)
    if policy.shape != (env.n_states, env.n_actions):
        raise ValueError("policy must have shape (n_states, n_actions)")
    if np.any(policy < 0.0) or not np.allclose(policy.sum(axis=1), 1.0):
        raise ValueError("each policy row must be a probability distribution")

    values = np.zeros(env.n_states)
    for _ in range(max_iterations):
        updated = np.zeros_like(values)
        for state in range(env.n_states):
            if state == env.terminal_state:
                continue
            for action in range(env.n_actions):
                next_state, reward, terminated = env.transition(state, action)
                updated[state] += policy[state, action] * td_target(
                    reward, gamma, values[next_state], terminated
                )
        if np.max(np.abs(updated - values)) < tolerance:
            return updated
        values = updated
    raise RuntimeError("policy evaluation did not converge within max_iterations")


def value_iteration(env, gamma=0.9, tolerance=1e-10, max_iterations=10_000):
    """Return optimal state values and a first-argmax deterministic policy."""
    gamma = _validate_gamma(gamma)
    tolerance, max_iterations = _validate_solver_settings(
        tolerance, max_iterations
    )
    values = np.zeros(env.n_states)
    for _ in range(max_iterations):
        updated = np.zeros_like(values)
        for state in range(env.n_states):
            if state == env.terminal_state:
                continue
            action_values = []
            for action in range(env.n_actions):
                next_state, reward, terminated = env.transition(state, action)
                action_values.append(
                    td_target(reward, gamma, values[next_state], terminated)
                )
            updated[state] = max(action_values)
        if np.max(np.abs(updated - values)) < tolerance:
            values = updated
            break
        values = updated
    else:
        raise RuntimeError("value iteration did not converge within max_iterations")

    greedy_policy = np.full(env.n_states, -1, dtype=int)
    for state in range(env.n_states):
        if state == env.terminal_state:
            continue
        action_values = []
        for action in range(env.n_actions):
            next_state, reward, terminated = env.transition(state, action)
            action_values.append(td_target(reward, gamma, values[next_state], terminated))
        greedy_policy[state] = int(np.argmax(action_values))
    return values, greedy_policy


def monte_carlo_update(values, counts, state, return_):
    _validate_update_arrays(values, counts)
    _validate_index(state, values.size, "state")
    return_ = _finite_real(return_, "return")
    new_count = int(counts[state]) + 1
    if new_count > np.iinfo(counts.dtype).max:
        raise ValueError("count cannot be incremented without overflow")
    new_value = float(values[state] + (return_ - values[state]) / new_count)

    values[state] = new_value
    counts[state] = new_count


def td_target(reward, gamma, next_value, terminated):
    return float(reward) if terminated else float(reward + gamma * next_value)


def q_learning_update(
    q, state, action, reward, next_state, alpha, gamma, terminated
):
    if (
        not isinstance(q, np.ndarray)
        or q.ndim != 2
        or not np.issubdtype(q.dtype, np.floating)
        or not q.flags.writeable
    ):
        raise ValueError("q must be a writable floating-point 2D array")
    _validate_index(state, q.shape[0], "state")
    _validate_index(action, q.shape[1], "action")
    _validate_index(next_state, q.shape[0], "next_state")
    reward = _finite_real(reward, "reward")
    alpha = _finite_real(alpha, "alpha")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    gamma = _validate_gamma(gamma)

    bootstrap = 0.0 if terminated else float(np.max(q[next_state]))
    target = reward + gamma * bootstrap
    updated = float(q[state, action] + alpha * (target - q[state, action]))
    q[state, action] = updated
    return float(target)


def _learn_q_values(env, seed, episodes=200):
    rng = np.random.default_rng(seed)
    q = np.zeros((env.n_states, env.n_actions))
    for _ in range(episodes):
        state = env.reset()
        for _ in range(30):
            if rng.random() < 0.2:
                action = int(rng.integers(env.n_actions))
            else:
                action = int(np.argmax(q[state]))
            next_state, reward, terminated = env.step(action)
            q_learning_update(
                q, state, action, reward, next_state, 0.2, 0.9, terminated
            )
            state = next_state
            if terminated:
                break
    return q


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    env = GridWorld()
    uniform_policy = np.full((env.n_states, env.n_actions), 1.0 / env.n_actions)
    policy_values = policy_evaluation(env, uniform_policy)
    optimal_values, greedy_policy = value_iteration(env)
    q = _learn_q_values(env, args.seed)

    print(f"policy_value_mean={np.mean(policy_values):.3f}")
    print(
        f"optimal_value_start={optimal_values[env.start_state]:.3f}, "
        f"greedy_policy={greedy_policy.tolist()}"
    )
    print(f"q_mean={np.mean(q):.3f}, q_max={np.max(q):.3f}")


if __name__ == "__main__":
    main()
