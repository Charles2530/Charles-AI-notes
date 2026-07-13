# Reinforcement learning examples

These examples support the tabular and classic deep-reinforcement-learning
chapters. `bandit.py` and `gridworld.py` use NumPy only. `deep_rl_smoke.py` uses
Gymnasium Classic Control and PyTorch to verify that DQN, REINFORCE, PPO, and
SAC each collect real transitions and execute at least one finite optimizer
update. None of the short commands is a convergence benchmark.

## Isolated environment setup

Create a course-only virtual environment from the repository root. This keeps
NumPy, PyTorch, and Gymnasium out of the MkDocs site environment and does not
change the root `requirements.txt`:

```bash
python -m venv .venv-rl-course
source .venv-rl-course/bin/activate
python -m pip install -r files/assets/examples/reinforcement-learning/requirements.txt
```

The bounded dependencies are NumPy `>=1.26,<3`, PyTorch `>=2.2,<3`, and
Gymnasium `>=1.0,<2` with the Classic Control extra. A display and rendering
backend are not required because the smoke modes never call `render`.

To roll back the isolated setup, first leave the environment and then delete
only its directory:

```bash
deactivate
rm -rf .venv-rl-course
```

`rm -rf` is intentionally scoped to `.venv-rl-course`; run it from the
repository root and verify the name before deletion. Removing that directory
does not remove course source or modify the site environment.

## Commands

With `.venv-rl-course` active, run:

```bash
python files/assets/examples/reinforcement-learning/bandit.py --seed 7 --steps 2000
python files/assets/examples/reinforcement-learning/gridworld.py --seed 7
python files/assets/examples/reinforcement-learning/deep_rl_smoke.py dqn --seed 7 --steps 256
python files/assets/examples/reinforcement-learning/deep_rl_smoke.py reinforce --seed 7 --steps 256
python files/assets/examples/reinforcement-learning/deep_rl_smoke.py ppo --seed 7 --steps 256
python files/assets/examples/reinforcement-learning/deep_rl_smoke.py sac --seed 7 --steps 256
```

Every deep mode rejects `--steps` below 128. The value 128 is a
uniform smoke-quality floor chosen to ensure update coverage across all four
modes, not the mathematical minimum for every optimizer. REINFORCE finishes the current
trajectory, so `interaction_count` can exceed `requested_steps`. On a CPU
laptop each mode is a short smoke-class run, normally seconds rather than a
training job; package installation is usually slower than the commands.

## Deep objective API

`deep_rl_smoke.py` imports NumPy at module load and delays PyTorch/Gymnasium
until CLI dispatch. Numeric tests therefore work without optional deep-RL
dependencies. The public helpers reject shape mismatch, non-finite values, and
implicit broadcasting:

| API | Objective |
| --- | --- |
| `dqn_targets(...)` | Terminal-masked one-step Q target. |
| `discounted_returns(...)` | Reverse return-to-go recurrence with episode resets. |
| `gae(...)` | GAE from rewards, one value per step, and one bootstrap value. |
| `ppo_clipped_loss(...)` | Scalar negative clipped PPO surrogate. |
| `sac_targets(...)` | Entropy-regularized twin-Q target with terminal masking. |

Every summary reports `interaction_count`, `reward_per_interaction`,
`completed_episode_count`, and `current_partial_episode_return`. It reports a
mean only over completed episodes; if there are none, the output is exactly
`completed_episode_return_mean=unavailable`, never the partial return disguised
as a mean.

DQN adds update count, final loss, and network change. REINFORCE adds requested
steps, policy loss, and policy change. PPO reports `sampled_approximate_kl`
after recomputing the complete rollout under the final actor, plus separate
`actor_parameter_delta` and `critic_parameter_delta`. The sampled mean can be
negative and is not the exact nonnegative KL divergence. SAC reports critic and
actor losses, temperature, update count, and separate actor/critic changes.
Its entropy terms use a scale-corrected environment-action logprob: both the
tanh Jacobian and environment action-scale Jacobian are included, so the named
`target_entropy=-1.0` is interpreted in those density coordinates. The stable
contract is the labeled fields and a zero exit code, not exact decimals across
platforms.

## Terminated and truncated

Gymnasium returns `(observation, reward, terminated, truncated, info)`.
`terminated=True` means the task reached a true MDP terminal and the bootstrap
term is zero. `truncated=True` usually means a time limit or sampling boundary;
when its final observation is valid, DQN, PPO, and SAC retain bootstrap value.
Episode control resets on either flag.

PPO computes GAE separately for every episode or rollout segment. This keeps a
truncation bootstrap while preventing the recurrence from leaking through the
subsequent environment reset. REINFORCE has no critic to estimate an omitted
tail: it ends its sampled Monte Carlo trajectory at truncation and documents
that finite-horizon estimate instead of pretending to bootstrap it.

## Reproducibility and interpretation

The scripts seed Python, NumPy, PyTorch, the Gymnasium environment, and its
action space. A repeated run under the same package versions and hardware is
intended to be reproducible, but seeds do not guarantee bit-identical behavior
across PyTorch builds, processors, BLAS implementations, or Gymnasium versions.
A single seed also gives no uncertainty estimate.

These commands prove only that environment interaction, stored data, target or
advantage construction, loss, backward pass, optimizer step, and finite metrics
are connected. They do not prove monotonic learning, hyperparameter quality,
sample efficiency, robustness, task solution, or convergence. Those claims
require longer training, held-out evaluation episodes, multiple seeds, and
uncertainty reporting.

## Bandit API and output

`bandit.py` supports the [multi-armed bandit and exploration
chapter](../../../reinforcement-learning/multi-armed-bandits-and-exploration.md).

| API | Responsibility |
| --- | --- |
| `update_sample_average(values, counts, arm, reward)` | Atomically update one floating estimate and its nonnegative integer count. |
| `epsilon_greedy(values, epsilon, rng)` | Explore uniformly with probability epsilon; otherwise choose the first maximum. |
| `ucb_action(values, counts, step)` | Try unselected arms before computing the UCB1 score. |
| `run_experiment(seed, steps, epsilon)` | Compare epsilon-greedy and UCB on one seeded task. |

The CLI reports `cumulative_reward`, `average_regret`, and `best_arm_fraction`
for each policy. Their field names and finite values are the output contract;
one run does not establish statistical superiority. Separate task and policy
random streams make the true means and potential reward noise reproducible
without letting epsilon-greedy choices advance the task stream. A seed does not
replace repeated tasks, uncertainty estimates, or non-stationary evaluation.

## GridWorld API, terminal semantics, and output

`gridworld.py` supports [dynamic programming, Monte Carlo, and
TD](../../../reinforcement-learning/dynamic-programming-monte-carlo-td.md) plus
the tabular bridge in [Q-learning and
DQN](../../../reinforcement-learning/q-learning-and-dqn.md).

| API | Responsibility |
| --- | --- |
| `GridWorld.reset()` / `GridWorld.step(action)` | Reset or execute one deterministic map transition. |
| `policy_evaluation(env, policy, ...)` | Evaluate a tabular stochastic policy by synchronous backups. |
| `value_iteration(env, ...)` | Solve optimal values and a first-argmax policy. |
| `monte_carlo_update(values, counts, state, return_)` | Apply an incremental first-visit value sample average. |
| `td_target(reward, gamma, next_value, terminated)` | Form a one-step target with true-terminal masking. |
| `q_learning_update(q, ...)` | Atomically update one selected Q-table entry and return its target. |

The deterministic 2-by-3 map numbers states row-major from 0 through 5; state 5
is terminal. Ordinary transitions return `-1`, while entering or stepping from
the terminal returns `0` with `terminated=True`. A true terminal removes the
bootstrap term, so the TD/Q-learning target is just its reward. This semantic
mask is separate from index validation.

The CLI reports the uniform-policy value mean, optimal start value and greedy
policy, plus learned Q summary fields. The transition model is deterministic;
the seed controls only exploratory choices in the small Q-learning run. These
finite teaching summaries do not prove convergence on other maps, stochastic
dynamics, or function approximation.
