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
    def test_dqn_targets_mask_only_true_terminals(self):
        deep_rl = load_module("deep_rl_smoke")
        rewards = np.array([1.0, 1.0])
        next_q = np.array([10.0, 10.0])
        terminated = np.array([True, False])
        snapshots = [array.copy() for array in (rewards, next_q, terminated)]

        targets = deep_rl.dqn_targets(rewards, next_q, terminated, gamma=0.9)

        np.testing.assert_allclose(targets, [1.0, 10.0])
        for array, snapshot in zip((rewards, next_q, terminated), snapshots):
            np.testing.assert_array_equal(array, snapshot)

    def test_dqn_targets_reject_invalid_arrays_and_gamma(self):
        deep_rl = load_module("deep_rl_smoke")
        valid = (np.ones(2), np.ones(2), np.array([False, True]))
        cases = [
            (np.ones((2, 1)), valid[1], valid[2], 0.9),
            (valid[0], np.ones(3), valid[2], 0.9),
            (valid[0], valid[1], np.array([0, 1]), 0.9),
            (np.array([1.0, np.nan]), valid[1], valid[2], 0.9),
            (valid[0], np.array([1.0, np.inf]), valid[2], 0.9),
            (*valid, -0.1),
            (*valid, 1.1),
            (*valid, np.nan),
            (*valid, True),
        ]
        for rewards, next_q, terminated, gamma in cases:
            with self.subTest(gamma=gamma):
                with self.assertRaises(ValueError):
                    deep_rl.dqn_targets(rewards, next_q, terminated, gamma)

    def test_discounted_returns_reset_at_episode_boundaries(self):
        deep_rl = load_module("deep_rl_smoke")
        rewards = np.array([1.0, 2.0, 3.0, 4.0])
        terminated = np.array([False, True, False, False])

        returns = deep_rl.discounted_returns(rewards, terminated, gamma=0.5)

        np.testing.assert_allclose(returns, [2.0, 2.0, 5.0, 4.0])
        np.testing.assert_array_equal(rewards, [1.0, 2.0, 3.0, 4.0])
        np.testing.assert_array_equal(terminated, [False, True, False, False])

    def test_discounted_returns_validate_inputs(self):
        deep_rl = load_module("deep_rl_smoke")
        valid_rewards = np.ones(2)
        valid_terminated = np.array([False, True])
        cases = [
            (np.ones((2, 1)), valid_terminated, 0.9),
            (valid_rewards, np.array([False]), 0.9),
            (valid_rewards, np.array([0, 1]), 0.9),
            (np.array([1.0, np.inf]), valid_terminated, 0.9),
            (valid_rewards, valid_terminated, -0.1),
            (valid_rewards, valid_terminated, 1.1),
        ]
        for rewards, terminated, gamma in cases:
            with self.subTest(gamma=gamma):
                with self.assertRaises(ValueError):
                    deep_rl.discounted_returns(rewards, terminated, gamma)

    def test_gae_matches_manual_recurrence_and_terminal_reset(self):
        deep_rl = load_module("deep_rl_smoke")
        rewards = np.array([1.0, 2.0, 3.0])
        values = np.array([0.5, 1.0, 2.0, 4.0])
        terminated = np.array([False, True, False])

        advantages = deep_rl.gae(
            rewards, values, terminated, gamma=0.9, gae_lambda=0.8
        )

        np.testing.assert_allclose(advantages, [2.12, 1.0, 4.6])
        np.testing.assert_array_equal(values, [0.5, 1.0, 2.0, 4.0])

    def test_gae_validates_bootstrap_shape_and_factors(self):
        deep_rl = load_module("deep_rl_smoke")
        rewards = np.ones(2)
        values = np.ones(3)
        terminated = np.array([False, True])
        cases = [
            (np.ones((2, 1)), values, terminated, 0.9, 0.8),
            (rewards, np.ones(2), terminated, 0.9, 0.8),
            (rewards, values, np.array([False]), 0.9, 0.8),
            (rewards, np.array([1.0, np.nan, 1.0]), terminated, 0.9, 0.8),
            (rewards, values, terminated, -0.1, 0.8),
            (rewards, values, terminated, 0.9, 1.1),
        ]
        for args in cases:
            with self.subTest(gamma=args[3], gae_lambda=args[4]):
                with self.assertRaises(ValueError):
                    deep_rl.gae(*args)

    def test_ppo_clipped_loss_handles_positive_and_negative_advantages(self):
        deep_rl = load_module("deep_rl_smoke")

        positive = deep_rl.ppo_clipped_loss(
            np.array([np.log(1.4)]), np.array([0.0]), np.array([2.0]), 0.2
        )
        negative = deep_rl.ppo_clipped_loss(
            np.array([np.log(0.6)]), np.array([0.0]), np.array([-2.0]), 0.2
        )

        self.assertIsInstance(positive, float)
        self.assertAlmostEqual(positive, -2.4)
        self.assertAlmostEqual(negative, 1.6)

    def test_ppo_clipped_loss_rejects_broadcasting_and_nonfinite_values(self):
        deep_rl = load_module("deep_rl_smoke")
        valid = np.zeros(2)
        cases = [
            (np.zeros((2, 1)), valid, valid, 0.2),
            (valid, np.zeros(3), valid, 0.2),
            (valid, valid, np.array([0.0, np.inf]), 0.2),
            (np.array([0.0, np.nan]), valid, valid, 0.2),
            (valid, valid, valid, 0.0),
            (valid, valid, valid, 1.0),
            (valid, valid, valid, True),
        ]
        for new_logp, old_logp, advantages, epsilon in cases:
            with self.subTest(epsilon=epsilon):
                with self.assertRaises(ValueError):
                    deep_rl.ppo_clipped_loss(
                        new_logp, old_logp, advantages, epsilon
                    )

    def test_sac_targets_include_entropy_and_mask_terminals(self):
        deep_rl = load_module("deep_rl_smoke")
        rewards = np.array([1.0, 1.0])
        next_min_q = np.array([10.0, 10.0])
        next_logp = np.array([2.0, 2.0])
        terminated = np.array([True, False])
        snapshots = [
            array.copy() for array in (rewards, next_min_q, next_logp, terminated)
        ]

        targets = deep_rl.sac_targets(
            rewards, next_min_q, next_logp, terminated, gamma=0.9, alpha=0.5
        )

        np.testing.assert_allclose(targets, [1.0, 9.1])
        for array, snapshot in zip(
            (rewards, next_min_q, next_logp, terminated), snapshots
        ):
            np.testing.assert_array_equal(array, snapshot)

    def test_sac_targets_validate_shapes_data_and_scalars(self):
        deep_rl = load_module("deep_rl_smoke")
        valid = (np.ones(2), np.ones(2), np.zeros(2), np.array([False, True]))
        cases = [
            (np.ones((2, 1)), *valid[1:], 0.9, 0.2),
            (valid[0], np.ones(3), *valid[2:], 0.9, 0.2),
            (*valid[:2], np.array([0.0, np.nan]), valid[3], 0.9, 0.2),
            (*valid[:3], np.array([0, 1]), 0.9, 0.2),
            (*valid, -0.1, 0.2),
            (*valid, 1.1, 0.2),
            (*valid, 0.9, -0.1),
            (*valid, 0.9, np.inf),
        ]
        for args in cases:
            with self.subTest(gamma=args[-2], alpha=args[-1]):
                with self.assertRaises(ValueError):
                    deep_rl.sac_targets(*args)

    def test_sample_average_updates_selected_arm(self):
        bandit = load_module("bandit")
        values = np.array([0.0, 0.0])
        counts = np.array([0, 0])

        bandit.update_sample_average(values, counts, arm=1, reward=2.0)
        bandit.update_sample_average(values, counts, arm=1, reward=4.0)

        np.testing.assert_allclose(values, [0.0, 3.0])
        np.testing.assert_array_equal(counts, [0, 2])

    def test_sample_average_rejects_invalid_inputs_without_mutation(self):
        bandit = load_module("bandit")
        read_only_values = np.array([1.0, 2.0])
        read_only_values.flags.writeable = False
        read_only_counts = np.array([0, 0])
        read_only_counts.flags.writeable = False
        cases = [
            (np.array([1, 2]), np.array([0, 0]), 0, 3.0),
            (np.array([1.0, 2.0]), np.array([0.0, 0.0]), 0, 3.0),
            (np.array([1.0, 2.0]), np.array([0, -1]), 0, 3.0),
            (read_only_values, np.array([0, 0]), 0, 3.0),
            (np.array([1.0, 2.0]), read_only_counts, 0, 3.0),
            (np.array([[1.0], [2.0]]), np.array([0, 0]), 0, 3.0),
            (np.array([1.0, 2.0]), np.array([0, 0, 0]), 0, 3.0),
            (np.array([1.0, 2.0]), np.array([0, 0]), -1, 3.0),
            (np.array([1.0, 2.0]), np.array([0, 0]), 2, 3.0),
            (np.array([1.0, 2.0]), np.array([0, 0]), True, 3.0),
            (np.array([1.0, 2.0]), np.array([0, 0]), 0, np.nan),
            (np.array([1.0, 2.0]), np.array([0, 0]), 0, np.inf),
            (np.array([1.0, 2.0]), np.array([0, 0]), 0, "bad"),
            (np.array([1.0, 2.0]), np.array([0, 0]), 0, True),
        ]

        for values, counts, arm, reward in cases:
            with self.subTest(
                values_dtype=values.dtype,
                counts_dtype=counts.dtype,
                arm=arm,
                reward=reward,
            ):
                values_before = values.copy()
                counts_before = counts.copy()
                with self.assertRaises(ValueError):
                    bandit.update_sample_average(values, counts, arm, reward)
                np.testing.assert_array_equal(values, values_before)
                np.testing.assert_array_equal(counts, counts_before)

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

    def test_ucb_rejects_negative_and_fractional_counts(self):
        bandit = load_module("bandit")
        values = np.array([0.5, 1.0])

        for counts in (np.array([1, -1]), np.array([1.0, 0.5])):
            with self.subTest(counts=counts):
                with self.assertRaises(ValueError):
                    bandit.ucb_action(values, counts, step=3)

    def test_ucb_requires_next_positive_integer_step(self):
        bandit = load_module("bandit")
        values = np.array([0.5, 1.0])
        counts = np.array([1, 1])

        for step in (0, 2.5, 3.0, np.nan, True, 4):
            with self.subTest(step=step):
                with self.assertRaises(ValueError):
                    bandit.ucb_action(values, counts, step)

    def test_same_seed_reproduces_bandit_summary(self):
        bandit = load_module("bandit")

        first = bandit.run_experiment(seed=11, steps=50, epsilon=0.1)
        second = bandit.run_experiment(seed=11, steps=50, epsilon=0.1)

        self.assertEqual(first, second)
        for method in first.values():
            self.assertTrue(all(np.isfinite(value) for value in method.values()))

    def test_bandit_summary_matches_known_actions_and_noise(self):
        bandit = load_module("bandit")
        actions = iter([0, 1, 1])

        summaries = bandit._run_policies(
            true_means=np.array([1.0, 3.0]),
            noise_rows=iter(
                [
                    np.array([0.5, -0.5]),
                    np.array([-1.0, 2.0]),
                    np.array([0.0, -1.0]),
                ]
            ),
            action_selectors={
                "fixed": lambda _values, _counts, _step: next(actions)
            },
        )

        self.assertEqual(
            summaries["fixed"],
            {
                "cumulative_reward": 8.5,
                "average_regret": 2.0 / 3.0,
                "best_arm_fraction": 2.0 / 3.0,
            },
        )

    def test_seeded_task_and_policy_rngs_are_independent_and_reproducible(self):
        bandit = load_module("bandit")

        first_task_rng, first_policy_rng = bandit._make_rngs(seed=11)
        second_task_rng, second_policy_rng = bandit._make_rngs(seed=11)
        first_task_draws = first_task_rng.random(4)
        first_policy_draws = first_policy_rng.random(4)

        np.testing.assert_array_equal(first_task_draws, second_task_rng.random(4))
        np.testing.assert_array_equal(first_policy_draws, second_policy_rng.random(4))
        self.assertFalse(np.array_equal(first_task_draws, first_policy_draws))

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

    def test_q_learning_rejects_invalid_inputs_without_mutation(self):
        gridworld = load_module("gridworld")
        read_only_q = np.zeros((6, 4))
        read_only_q.flags.writeable = False
        cases = [
            (np.zeros((6, 4), dtype=int), 1, 2, 1.0, 5, 0.2, 0.9),
            (read_only_q, 1, 2, 1.0, 5, 0.2, 0.9),
            (np.zeros(24), 1, 2, 1.0, 5, 0.2, 0.9),
            (np.zeros((6, 4)), -1, 2, 1.0, 5, 0.2, 0.9),
            (np.zeros((6, 4)), 6, 2, 1.0, 5, 0.2, 0.9),
            (np.zeros((6, 4)), True, 2, 1.0, 5, 0.2, 0.9),
            (np.zeros((6, 4)), 1, -1, 1.0, 5, 0.2, 0.9),
            (np.zeros((6, 4)), 1, 4, 1.0, 5, 0.2, 0.9),
            (np.zeros((6, 4)), 1, True, 1.0, 5, 0.2, 0.9),
            (np.zeros((6, 4)), 1, 2, 1.0, -1, 0.2, 0.9),
            (np.zeros((6, 4)), 1, 2, 1.0, 6, 0.2, 0.9),
            (np.zeros((6, 4)), 1, 2, 1.0, True, 0.2, 0.9),
            (np.zeros((6, 4)), 1, 2, np.nan, 5, 0.2, 0.9),
            (np.zeros((6, 4)), 1, 2, np.inf, 5, 0.2, 0.9),
            (np.zeros((6, 4)), 1, 2, "bad", 5, 0.2, 0.9),
            (np.zeros((6, 4)), 1, 2, True, 5, 0.2, 0.9),
            (np.zeros((6, 4)), 1, 2, 1.0, 5, np.nan, 0.9),
            (np.zeros((6, 4)), 1, 2, 1.0, 5, True, 0.9),
            (np.zeros((6, 4)), 1, 2, 1.0, 5, -0.1, 0.9),
            (np.zeros((6, 4)), 1, 2, 1.0, 5, 1.1, 0.9),
            (np.zeros((6, 4)), 1, 2, 1.0, 5, 0.2, np.nan),
            (np.zeros((6, 4)), 1, 2, 1.0, 5, 0.2, True),
            (np.zeros((6, 4)), 1, 2, 1.0, 5, 0.2, -0.1),
            (np.zeros((6, 4)), 1, 2, 1.0, 5, 0.2, 1.0),
        ]

        for q, state, action, reward, next_state, alpha, gamma in cases:
            with self.subTest(
                q_shape=q.shape,
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                alpha=alpha,
                gamma=gamma,
            ):
                before = q.copy()
                with self.assertRaises(ValueError):
                    gridworld.q_learning_update(
                        q,
                        state,
                        action,
                        reward,
                        next_state,
                        alpha,
                        gamma,
                        terminated=True,
                    )
                np.testing.assert_array_equal(q, before)

    def test_gridworld_reaches_terminal_state_on_documented_tiny_map(self):
        gridworld = load_module("gridworld")
        env = gridworld.GridWorld()

        self.assertEqual(env.reset(), 0)
        self.assertEqual(env.step(1), (1, -1.0, False))
        self.assertEqual(env.step(2), (4, -1.0, False))
        self.assertEqual(env.step(1), (5, 0.0, True))

    def test_gridworld_reset_does_not_accept_a_seed(self):
        gridworld = load_module("gridworld")
        env = gridworld.GridWorld()

        with self.assertRaises(TypeError):
            env.reset(seed=7)

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

    def test_dynamic_programming_validates_solver_settings(self):
        gridworld = load_module("gridworld")
        env = gridworld.GridWorld()
        policy = np.full((env.n_states, env.n_actions), 1.0 / env.n_actions)

        for function, args in (
            (gridworld.policy_evaluation, (env, policy)),
            (gridworld.value_iteration, (env,)),
        ):
            for tolerance in (0.0, -1.0, np.nan, np.inf, True, "bad"):
                with self.subTest(function=function.__name__, tolerance=tolerance):
                    with self.assertRaises(ValueError):
                        function(*args, tolerance=tolerance)
            for max_iterations in (0, -1, 1.5, True):
                with self.subTest(
                    function=function.__name__, max_iterations=max_iterations
                ):
                    with self.assertRaises(ValueError):
                        function(*args, max_iterations=max_iterations)

    def test_dynamic_programming_raises_when_iteration_budget_is_exhausted(self):
        gridworld = load_module("gridworld")
        env = gridworld.GridWorld()
        policy = np.full((env.n_states, env.n_actions), 1.0 / env.n_actions)

        with self.assertRaises(RuntimeError):
            gridworld.policy_evaluation(
                env, policy, tolerance=1e-12, max_iterations=1
            )
        with self.assertRaises(RuntimeError):
            gridworld.value_iteration(env, tolerance=1e-12, max_iterations=1)

    def test_value_iteration_matches_gridworld_bellman_solution(self):
        gridworld = load_module("gridworld")
        env = gridworld.GridWorld()
        tolerance = 1e-10

        values, policy = gridworld.value_iteration(env, tolerance=tolerance)

        np.testing.assert_allclose(values, [-1.9, -1.0, 0.0, -1.0, 0.0, 0.0])
        np.testing.assert_array_equal(policy, [1, 1, 2, 1, 1, -1])
        residuals = []
        for state in range(env.n_states):
            if state == env.terminal_state:
                self.assertEqual(values[state], 0.0)
                continue
            targets = []
            for action in range(env.n_actions):
                next_state, reward, terminated = env.transition(state, action)
                targets.append(
                    gridworld.td_target(
                        reward, 0.9, values[next_state], terminated
                    )
                )
            residuals.append(abs(values[state] - max(targets)))
        self.assertLess(max(residuals), tolerance)

    def test_monte_carlo_update_uses_incremental_average(self):
        gridworld = load_module("gridworld")
        values = np.zeros(3)
        counts = np.zeros(3, dtype=int)

        gridworld.monte_carlo_update(values, counts, state=1, return_=2.0)
        gridworld.monte_carlo_update(values, counts, state=1, return_=4.0)

        np.testing.assert_allclose(values, [0.0, 3.0, 0.0])
        np.testing.assert_array_equal(counts, [0, 2, 0])

    def test_monte_carlo_rejects_invalid_inputs_without_mutation(self):
        gridworld = load_module("gridworld")
        read_only_values = np.zeros(3)
        read_only_values.flags.writeable = False
        read_only_counts = np.zeros(3, dtype=int)
        read_only_counts.flags.writeable = False
        cases = [
            (np.zeros(3, dtype=int), np.zeros(3, dtype=int), 1, 2.0),
            (np.zeros(3), np.zeros(3), 1, 2.0),
            (np.zeros(3), np.array([0, -1, 0]), 1, 2.0),
            (read_only_values, np.zeros(3, dtype=int), 1, 2.0),
            (np.zeros(3), read_only_counts, 1, 2.0),
            (np.zeros((3, 1)), np.zeros(3, dtype=int), 1, 2.0),
            (np.zeros(3), np.zeros(2, dtype=int), 1, 2.0),
            (np.zeros(3), np.zeros(3, dtype=int), -1, 2.0),
            (np.zeros(3), np.zeros(3, dtype=int), 3, 2.0),
            (np.zeros(3), np.zeros(3, dtype=int), True, 2.0),
            (np.zeros(3), np.zeros(3, dtype=int), 1, np.nan),
            (np.zeros(3), np.zeros(3, dtype=int), 1, np.inf),
            (np.zeros(3), np.zeros(3, dtype=int), 1, "bad"),
            (np.zeros(3), np.zeros(3, dtype=int), 1, True),
        ]

        for values, counts, state, return_ in cases:
            with self.subTest(
                values_shape=values.shape,
                counts_dtype=counts.dtype,
                state=state,
                return_=return_,
            ):
                values_before = values.copy()
                counts_before = counts.copy()
                with self.assertRaises(ValueError):
                    gridworld.monte_carlo_update(values, counts, state, return_)
                np.testing.assert_array_equal(values, values_before)
                np.testing.assert_array_equal(counts, counts_before)
