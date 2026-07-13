import importlib.util
import math
from pathlib import Path
import subprocess
import sys
from unittest import TestCase, skipUnless
import warnings

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None
    nn = None


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "files" / "assets" / "examples" / "reinforcement-learning"
RL_COURSE = ROOT / "files" / "reinforcement-learning"


def load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@skipUnless(np is not None, "course examples require NumPy")
class ReinforcementLearningExampleTests(TestCase):
    def test_group_advantages_normalizes_each_group_without_mutation(self):
        objectives = load_module("policy_objectives")
        one_group = np.array([1.0, 2.0, 4.0])
        batch = np.array([[1.0, 2.0, 4.0], [10.0, 7.0, 3.0]])
        snapshots = (one_group.copy(), batch.copy())

        one_result = objectives.group_advantages(one_group)
        batch_result = objectives.group_advantages(batch)

        self.assertEqual(one_result.shape, one_group.shape)
        self.assertEqual(batch_result.shape, batch.shape)
        np.testing.assert_allclose(one_result.sum(), 0.0, atol=1e-12)
        np.testing.assert_allclose(batch_result.sum(axis=-1), 0.0, atol=1e-12)
        np.testing.assert_allclose(one_result.std(ddof=0), 1.0, atol=1e-7)
        np.testing.assert_allclose(batch_result.std(axis=-1, ddof=0), 1.0, atol=1e-7)
        np.testing.assert_array_equal(one_group, snapshots[0])
        np.testing.assert_array_equal(batch, snapshots[1])

    def test_group_advantages_returns_exact_zero_for_constant_groups_without_warning(self):
        objectives = load_module("policy_objectives")
        rewards = np.array([[3.0, 3.0], [4.0, 6.0]])

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = objectives.group_advantages(rewards)

        np.testing.assert_array_equal(result[0], np.zeros(2))
        np.testing.assert_allclose(result[1], [-0.99999999, 0.99999999])

    def test_group_advantages_handles_constant_near_float_limit_without_warning(self):
        objectives = load_module("policy_objectives")
        rewards = np.array([1e308, 1e308])
        snapshot = rewards.copy()

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = objectives.group_advantages(rewards)

        np.testing.assert_array_equal(result, np.zeros(2))
        self.assertTrue(np.all(np.isfinite(result)))
        np.testing.assert_array_equal(rewards, snapshot)

    def test_group_advantages_normalizes_nonconstant_near_float_limit_groups(self):
        objectives = load_module("policy_objectives")
        adjacent = np.nextafter(1e308, np.inf)
        cases = (
            np.array([-1e308, 1e308]),
            np.array([1e308, adjacent]),
        )

        self.assertGreater(adjacent - 1e308, 1e-8)
        for rewards in cases:
            with self.subTest(rewards=rewards), warnings.catch_warnings():
                warnings.simplefilter("error")
                result = objectives.group_advantages(rewards)

                self.assertTrue(np.all(np.isfinite(result)))
                np.testing.assert_allclose(result.sum(), 0.0, atol=1e-12)
                np.testing.assert_allclose(result.std(ddof=0), 1.0, atol=1e-7)

    def test_group_advantages_normalizes_mixed_near_limit_batch_per_group(self):
        objectives = load_module("policy_objectives")
        adjacent = np.nextafter(1e308, np.inf)
        rewards = np.array(
            [
                [1e308, 1e308],
                [-1e308, 1e308],
                [1e308, adjacent],
            ]
        )
        snapshot = rewards.copy()

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = objectives.group_advantages(rewards)

        self.assertTrue(np.all(np.isfinite(result)))
        np.testing.assert_array_equal(result[0], np.zeros(2))
        np.testing.assert_allclose(result[1:].sum(axis=-1), 0.0, atol=1e-12)
        np.testing.assert_allclose(result[1:].std(axis=-1, ddof=0), 1.0, atol=1e-7)
        np.testing.assert_array_equal(rewards, snapshot)

    def test_group_advantages_rejects_invalid_inputs(self):
        objectives = load_module("policy_objectives")
        cases = [
            ([1.0], 1e-8),
            ([[1.0], [2.0]], 1e-8),
            (np.ones((2, 2, 2)), 1e-8),
            ([[1.0, 2.0], [3.0]], 1e-8),
            ([1.0 + 0.0j, 2.0 + 0.0j], 1e-8),
            ([1.0, np.nan], 1e-8),
            ([1.0, np.inf], 1e-8),
            ([1.0, 2.0], 0.0),
            ([1.0, 2.0], -1.0),
            ([1.0, 2.0], np.inf),
            ([1.0, 2.0], 1e-8 + 0.0j),
            ([1.0, 2.0], np.array([1e-8])),
        ]
        for rewards, epsilon in cases:
            with self.subTest(rewards=rewards, epsilon=epsilon):
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    with self.assertRaises(ValueError):
                        objectives.group_advantages(rewards, epsilon)

    def test_sampled_forward_kl_is_pointwise_log_ratio(self):
        objectives = load_module("policy_objectives")
        policy_logp = np.array([[-0.2, -0.8], [-1.0, -0.5]])
        reference_logp = np.array([[-0.5, -0.7], [-1.2, -0.5]])

        result = objectives.sampled_forward_kl(policy_logp, reference_logp)

        np.testing.assert_allclose(result, [[0.3, -0.1], [0.2, 0.0]])
        self.assertEqual(result.shape, policy_logp.shape)
        np.testing.assert_array_equal(policy_logp, [[-0.2, -0.8], [-1.0, -0.5]])

    def test_sampled_forward_kl_rejects_broadcasting_and_invalid_values(self):
        objectives = load_module("policy_objectives")
        cases = [
            (np.zeros((2, 1)), np.zeros(2)),
            (np.zeros(2), np.zeros(3)),
            (np.array([0.0, np.nan]), np.zeros(2)),
            (np.zeros(2), np.array([0.0, np.inf])),
            (np.array([0.0 + 0.0j]), np.zeros(1)),
            (np.zeros(0), np.zeros(0)),
        ]
        for policy_logp, reference_logp in cases:
            with self.subTest(shapes=(policy_logp.shape, reference_logp.shape)):
                with self.assertRaises(ValueError):
                    objectives.sampled_forward_kl(policy_logp, reference_logp)

    def test_clipped_policy_loss_matches_positive_and_negative_exact_cases(self):
        objectives = load_module("policy_objectives")
        positive = objectives.clipped_policy_loss(
            np.array([math.log(1.5)]), np.array([0.0]), np.array([2.0]), 0.2
        )
        negative = objectives.clipped_policy_loss(
            np.array([math.log(0.5)]), np.array([0.0]), np.array([-2.0]), 0.2
        )

        self.assertAlmostEqual(positive, -2.4)
        self.assertAlmostEqual(negative, 1.6)

    def test_clipped_policy_loss_validates_exact_shapes_and_finite_real_inputs(self):
        objectives = load_module("policy_objectives")
        valid = np.zeros(2)
        cases = [
            (np.zeros((2, 1)), valid, valid, 0.2),
            (valid, np.zeros(1), valid, 0.2),
            (valid, valid, np.zeros(3), 0.2),
            (np.array([0.0, np.nan]), valid, valid, 0.2),
            (valid, np.array([0.0, np.inf]), valid, 0.2),
            (valid, valid, np.array([0.0 + 0.0j, 1.0]), 0.2),
            (valid, valid, valid, 0.0),
            (valid, valid, valid, 1.0),
            (valid, valid, valid, np.nan),
            (valid, valid, valid, 0.2 + 0.0j),
            (valid, valid, valid, np.array([0.2])),
        ]
        for args in cases:
            with self.subTest(shapes=[getattr(value, "shape", None) for value in args]):
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    with self.assertRaises(ValueError):
                        objectives.clipped_policy_loss(*args)

    def test_policy_objectives_cli_is_deterministic_and_names_estimator_boundaries(self):
        script = EXAMPLES / "policy_objectives.py"
        base_command = [sys.executable, str(script)]
        default = subprocess.run(base_command, check=True, capture_output=True, text=True)
        seeded = subprocess.run(
            [*base_command, "--seed", "7"], check=True, capture_output=True, text=True
        )
        repeated = subprocess.run(
            [*base_command, "--seed", "7"], check=True, capture_output=True, text=True
        )

        self.assertEqual(default.stdout, seeded.stdout)
        self.assertEqual(seeded.stdout, repeated.stdout)
        for field in (
            "prompt",
            "raw_reward",
            "group_advantage",
            "pointwise_kl_penalty",
            "sampled log-ratio estimator, not full KL",
            "clipped_policy_loss",
        ):
            self.assertIn(field, seeded.stdout)
        self.assertIn("zero-variance group -> exact zero advantages", seeded.stdout)
        loss_line = next(
            line for line in seeded.stdout.splitlines() if line.startswith("clipped_policy_loss=")
        )
        self.assertTrue(math.isfinite(float(loss_line.split("=", maxsplit=1)[1])))

    def test_example_readme_owns_tabular_apis_and_exact_runtime_contract(self):
        text = (EXAMPLES / "README.md").read_text(encoding="utf-8")
        for dependency in (
            "NumPy `>=1.26,<3`",
            "PyTorch `>=2.2,<3`",
            "Gymnasium `>=1.0,<2`",
        ):
            self.assertIn(dependency, text)
        for api in (
            "`update_sample_average(values, counts, arm, reward)`",
            "`epsilon_greedy(values, epsilon, rng)`",
            "`ucb_action(values, counts, step)`",
            "`run_experiment(seed, steps, epsilon)`",
            "`GridWorld.reset()` / `GridWorld.step(action)`",
            "`policy_evaluation(env, policy, ...)`",
            "`value_iteration(env, ...)`",
            "`monte_carlo_update(values, counts, state, return_)`",
            "`td_target(reward, gamma, next_value, terminated)`",
            "`q_learning_update(q, ...)`",
        ):
            self.assertIn(api, text)
        for output_fact in (
            "uniform smoke-quality floor",
            "interaction_count",
            "reward_per_interaction",
            "completed_episode_count",
            "completed_episode_return_mean=unavailable",
            "current_partial_episode_return",
            "sampled_approximate_kl",
            "actor_parameter_delta",
            "critic_parameter_delta",
            "scale-corrected environment-action logprob",
        ):
            self.assertIn(output_fact, text)

    def test_ppo_page_describes_final_sampled_kl_and_separate_deltas(self):
        text = (
            RL_COURSE / "policy-gradient-actor-critic-ppo-grpo.md"
        ).read_text(encoding="utf-8")
        for fact in (
            "完整 rollout",
            "old_logp - final_logp",
            "sampled approximate KL",
            "可能为负",
            "actor_parameter_delta",
            "critic_parameter_delta",
        ):
            self.assertIn(fact, text)

    def test_sac_page_names_target_entropy_and_logprob_coordinate(self):
        text = (RL_COURSE / "continuous-control-and-sac.md").read_text(
            encoding="utf-8"
        )
        for fact in (
            "target_entropy=-1.0",
            "scale-corrected environment-action logprob",
            "completed_episode_return_mean=unavailable",
            "current_partial_episode_return",
        ):
            self.assertIn(fact, text)

    def test_deep_objectives_reject_complex_without_warning(self):
        deep_rl = load_module("deep_rl_smoke")
        real = np.ones(2)
        terminal = np.array([False, True])
        complex_values = np.array([1.0 + 1.0j, 2.0 + 0.0j])
        cases = {
            "dqn_array": lambda: deep_rl.dqn_targets(
                complex_values, real, terminal, 0.9
            ),
            "dqn_gamma": lambda: deep_rl.dqn_targets(
                real, real, terminal, 0.9 + 0.0j
            ),
            "returns_array": lambda: deep_rl.discounted_returns(
                complex_values, terminal, 0.9
            ),
            "returns_gamma": lambda: deep_rl.discounted_returns(
                real, terminal, 0.9 + 0.0j
            ),
            "gae_array": lambda: deep_rl.gae(
                real, np.array([1.0 + 1.0j, 1.0, 1.0]), terminal, 0.9, 0.8
            ),
            "gae_lambda": lambda: deep_rl.gae(
                real, np.ones(3), terminal, 0.9, 0.8 + 0.0j
            ),
            "ppo_array": lambda: deep_rl.ppo_clipped_loss(
                complex_values, real, real, 0.2
            ),
            "ppo_epsilon": lambda: deep_rl.ppo_clipped_loss(
                real, real, real, 0.2 + 0.0j
            ),
            "sac_array": lambda: deep_rl.sac_targets(
                real, real, complex_values, terminal, 0.9, 0.2
            ),
            "sac_alpha": lambda: deep_rl.sac_targets(
                real, real, real, terminal, 0.9, 0.2 + 0.0j
            ),
        }

        for name, call in cases.items():
            with self.subTest(name=name), warnings.catch_warnings():
                warnings.simplefilter("error")
                with self.assertRaises(ValueError):
                    call()

    def test_interaction_summary_handles_no_completed_episode(self):
        deep_rl = load_module("deep_rl_smoke")

        summary = deep_rl.interaction_summary(
            interaction_count=3,
            total_reward=6.0,
            completed_episode_returns=[],
            current_partial_episode_return=6.0,
        )

        self.assertEqual(
            summary,
            {
                "interaction_count": 3,
                "reward_per_interaction": 2.0,
                "completed_episode_count": 0,
                "completed_episode_return_mean": None,
                "current_partial_episode_return": 6.0,
            },
        )

    def test_interaction_summary_separates_completed_and_partial_returns(self):
        deep_rl = load_module("deep_rl_smoke")

        summary = deep_rl.interaction_summary(
            interaction_count=10,
            total_reward=12.0,
            completed_episode_returns=[5.0, 3.0],
            current_partial_episode_return=4.0,
        )

        self.assertEqual(summary["reward_per_interaction"], 1.2)
        self.assertEqual(summary["completed_episode_count"], 2)
        self.assertEqual(summary["completed_episode_return_mean"], 4.0)
        self.assertEqual(summary["current_partial_episode_return"], 4.0)

    def test_managed_environment_closes_after_body_raises(self):
        deep_rl = load_module("deep_rl_smoke")

        class FakeEnvironment:
            closed = False

            def close(self):
                self.closed = True

        environment = FakeEnvironment()
        with self.assertRaisesRegex(RuntimeError, "body failed"):
            with deep_rl.managed_environment(environment):
                raise RuntimeError("body failed")

        self.assertTrue(environment.closed)

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


@skipUnless(torch is not None, "deep RL Torch primitives require PyTorch")
class DeepRLTorchPrimitiveTests(TestCase):
    def setUp(self):
        self.deep_rl = load_module("deep_rl_smoke")
        torch.manual_seed(7)

    def test_dqn_torch_target_and_loss_mask_terminal_without_target_gradient(self):
        predicted_q = torch.tensor([0.0, 0.0], requires_grad=True)
        next_q = torch.tensor([10.0, 10.0], requires_grad=True)

        targets, loss = self.deep_rl.torch_dqn_target_and_loss(
            predicted_q,
            torch.tensor([1.0, 1.0]),
            next_q,
            torch.tensor([True, False]),
            gamma=0.9,
        )

        torch.testing.assert_close(targets, torch.tensor([1.0, 10.0]))
        torch.testing.assert_close(loss, torch.tensor(5.0))
        loss.backward()
        self.assertIsNotNone(predicted_q.grad)
        self.assertIsNone(next_q.grad)

    def test_ppo_torch_losses_clip_exactly_and_own_separate_gradients(self):
        actor = nn.Linear(1, 2, bias=False)
        critic = nn.Linear(1, 1, bias=False)
        nn.init.zeros_(actor.weight)
        nn.init.zeros_(critic.weight)
        states = torch.tensor([[1.0]])
        actions = torch.tensor([0])
        current_logp = torch.tensor([-math.log(2.0)])
        old_logp = current_logp - math.log(1.4)

        losses = self.deep_rl.torch_ppo_minibatch_losses(
            actor,
            critic,
            states,
            actions,
            old_logp,
            torch.tensor([2.0]),
            torch.tensor([2.0]),
            clip_epsilon=0.2,
            value_coefficient=0.5,
            entropy_coefficient=0.01,
        )

        torch.testing.assert_close(losses["policy_loss"], torch.tensor(-2.4))
        torch.testing.assert_close(losses["value_loss"], torch.tensor(4.0))
        torch.testing.assert_close(losses["entropy"], torch.tensor(math.log(2.0)))
        expected_combined = -2.4 + 0.5 * 4.0 - 0.01 * math.log(2.0)
        torch.testing.assert_close(
            losses["combined_loss"], torch.tensor(expected_combined)
        )

        losses["policy_loss"].backward(retain_graph=True)
        self.assertTrue(any(parameter.grad is not None for parameter in actor.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in critic.parameters()))
        actor.zero_grad(set_to_none=True)
        critic.zero_grad(set_to_none=True)
        losses["value_loss"].backward()
        self.assertTrue(all(parameter.grad is None for parameter in actor.parameters()))
        self.assertTrue(any(parameter.grad is not None for parameter in critic.parameters()))

    def test_ppo_combined_loss_updates_actor_and_critic_separately(self):
        actor = nn.Linear(1, 2)
        critic = nn.Linear(1, 1)
        optimizer = torch.optim.SGD(
            list(actor.parameters()) + list(critic.parameters()), lr=0.1
        )
        actor_before = [parameter.detach().clone() for parameter in actor.parameters()]
        critic_before = [parameter.detach().clone() for parameter in critic.parameters()]
        states = torch.tensor([[1.0], [2.0]])
        actions = torch.tensor([0, 1])
        with torch.no_grad():
            old_logp = torch.distributions.Categorical(
                logits=actor(states)
            ).log_prob(actions)

        losses = self.deep_rl.torch_ppo_minibatch_losses(
            actor,
            critic,
            states,
            actions,
            old_logp,
            torch.tensor([1.0, -0.5]),
            torch.tensor([2.0, -1.0]),
        )
        optimizer.zero_grad()
        losses["combined_loss"].backward()
        optimizer.step()

        actor_delta = sum(
            torch.sum(torch.abs(before - after.detach())).item()
            for before, after in zip(actor_before, actor.parameters())
        )
        critic_delta = sum(
            torch.sum(torch.abs(before - after.detach())).item()
            for before, after in zip(critic_before, critic.parameters())
        )
        self.assertGreater(actor_delta, 0.0)
        self.assertGreater(critic_delta, 0.0)

    def test_sampled_approximate_kl_uses_complete_sample_mean_and_can_be_negative(self):
        positive = self.deep_rl.torch_sampled_approximate_kl(
            torch.tensor([0.0, 0.0]), torch.tensor([-0.2, -0.3])
        )
        negative = self.deep_rl.torch_sampled_approximate_kl(
            torch.tensor([-0.2]), torch.tensor([-0.1])
        )

        torch.testing.assert_close(positive, torch.tensor(0.25))
        torch.testing.assert_close(negative, torch.tensor(-0.1))

    def test_sac_torch_target_masks_terminal_and_keeps_entropy(self):
        targets = self.deep_rl.torch_sac_targets(
            torch.tensor([1.0, 1.0]),
            torch.tensor([10.0, 10.0]),
            torch.tensor([2.0, 2.0]),
            torch.tensor([True, False]),
            gamma=0.9,
            alpha=torch.tensor(0.5),
        )

        torch.testing.assert_close(targets, torch.tensor([1.0, 9.1]))

    def test_temperature_loss_uses_named_target_and_updates_log_alpha_direction(self):
        self.assertEqual(self.deep_rl.SAC_TARGET_ENTROPY, -1.0)
        log_alpha = torch.tensor(math.log(0.2), requires_grad=True)
        logp = torch.tensor([0.0], requires_grad=True)
        optimizer = torch.optim.SGD([log_alpha], lr=0.1)
        before = float(log_alpha.exp().detach())

        loss = self.deep_rl.torch_temperature_loss(
            log_alpha, logp, self.deep_rl.SAC_TARGET_ENTROPY
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        self.assertLess(float(log_alpha.exp().detach()), before)
        self.assertIsNone(logp.grad)

    def test_polyak_update_matches_weighted_average(self):
        target = nn.Linear(1, 1, bias=False)
        source = nn.Linear(1, 1, bias=False)
        target.weight.data.fill_(2.0)
        source.weight.data.fill_(10.0)

        self.deep_rl.polyak_update(
            list(target.parameters()), list(source.parameters()), tau=0.25
        )

        torch.testing.assert_close(target.weight, torch.tensor([[4.0]]))

    def test_frozen_parameters_route_actor_gradient_without_critic_gradient(self):
        actor = nn.Linear(1, 1, bias=False)
        critic = nn.Linear(2, 1, bias=False)
        state = torch.tensor([[1.0]])
        critic_parameters = list(critic.parameters())

        with self.deep_rl.frozen_parameters(critic_parameters):
            action = actor(state)
            loss = -critic(torch.cat((state, action), dim=-1)).mean()
            loss.backward()

        self.assertTrue(any(parameter.grad is not None for parameter in actor.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in critic_parameters))
        self.assertTrue(all(parameter.requires_grad for parameter in critic_parameters))
