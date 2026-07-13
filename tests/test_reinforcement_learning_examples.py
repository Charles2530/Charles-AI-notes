import importlib.util
from pathlib import Path
from unittest import TestCase, skipUnless

try:
    import numpy as np
except ModuleNotFoundError:
    np = None


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "files" / "assets" / "examples" / "reinforcement-learning"


def load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@skipUnless(np is not None, "course examples require NumPy")
class ReinforcementLearningExampleTests(TestCase):
    def test_sample_average_updates_selected_arm(self):
        bandit = load_module("bandit")
        values = np.array([0.0, 0.0])
        counts = np.array([0, 0])

        bandit.update_sample_average(values, counts, arm=1, reward=2.0)
        bandit.update_sample_average(values, counts, arm=1, reward=4.0)

        np.testing.assert_allclose(values, [0.0, 3.0])
        np.testing.assert_array_equal(counts, [0, 2])

    def test_epsilon_greedy_uses_first_argmax_without_exploration(self):
        bandit = load_module("bandit")
        rng = np.random.default_rng(7)

        action = bandit.epsilon_greedy(
            np.array([1.0, 3.0, 3.0]), epsilon=0.0, rng=rng
        )

        self.assertEqual(action, 1)

    def test_ucb_tries_each_unselected_arm_before_scoring(self):
        bandit = load_module("bandit")
        values = np.array([0.5, 20.0, 30.0])

        with np.errstate(all="raise"):
            first = bandit.ucb_action(values, np.array([2, 0, 0]), step=3)
            second = bandit.ucb_action(values, np.array([2, 1, 0]), step=4)
            scored = bandit.ucb_action(values, np.array([2, 1, 1]), step=5)

        self.assertEqual(first, 1)
        self.assertEqual(second, 2)
        self.assertIn(scored, range(3))

    def test_same_seed_reproduces_bandit_summary(self):
        bandit = load_module("bandit")

        first = bandit.run_experiment(seed=11, steps=50, epsilon=0.1)
        second = bandit.run_experiment(seed=11, steps=50, epsilon=0.1)

        self.assertEqual(first, second)
        for method in first.values():
            self.assertTrue(all(np.isfinite(value) for value in method.values()))

    def test_terminal_transition_does_not_bootstrap(self):
        gridworld = load_module("gridworld")

        self.assertEqual(
            gridworld.td_target(1.0, 0.9, 10.0, terminated=True), 1.0
        )
        self.assertEqual(
            gridworld.td_target(1.0, 0.9, 10.0, terminated=False), 10.0
        )

    def test_q_learning_terminal_update_changes_only_selected_entry(self):
        gridworld = load_module("gridworld")
        q = np.arange(24, dtype=float).reshape(6, 4)
        before = q.copy()

        target = gridworld.q_learning_update(
            q,
            state=1,
            action=2,
            reward=2.0,
            next_state=5,
            alpha=0.5,
            gamma=0.9,
            terminated=True,
        )

        expected = before.copy()
        expected[1, 2] += 0.5 * (2.0 - expected[1, 2])
        self.assertEqual(target, 2.0)
        np.testing.assert_allclose(q, expected)

    def test_gridworld_reaches_terminal_state_on_documented_tiny_map(self):
        gridworld = load_module("gridworld")
        env = gridworld.GridWorld()

        self.assertEqual(env.reset(seed=7), 0)
        self.assertEqual(env.step(1), (1, -1.0, False))
        self.assertEqual(env.step(2), (4, -1.0, False))
        self.assertEqual(env.step(1), (5, 0.0, True))

    def test_dynamic_programming_outputs_are_finite_and_shape_stable(self):
        gridworld = load_module("gridworld")
        env = gridworld.GridWorld()
        policy = np.full((env.n_states, env.n_actions), 1.0 / env.n_actions)

        evaluated = gridworld.policy_evaluation(env, policy, gamma=0.9)
        optimal_values, greedy_policy = gridworld.value_iteration(env, gamma=0.9)

        self.assertEqual(evaluated.shape, (env.n_states,))
        self.assertEqual(optimal_values.shape, (env.n_states,))
        self.assertEqual(greedy_policy.shape, (env.n_states,))
        self.assertTrue(np.all(np.isfinite(evaluated)))
        self.assertTrue(np.all(np.isfinite(optimal_values)))

    def test_monte_carlo_update_uses_incremental_average(self):
        gridworld = load_module("gridworld")
        values = np.zeros(3)
        counts = np.zeros(3, dtype=int)

        gridworld.monte_carlo_update(values, counts, state=1, return_=2.0)
        gridworld.monte_carlo_update(values, counts, state=1, return_=4.0)

        np.testing.assert_allclose(values, [0.0, 3.0, 0.0])
        np.testing.assert_array_equal(counts, [0, 2, 0])
