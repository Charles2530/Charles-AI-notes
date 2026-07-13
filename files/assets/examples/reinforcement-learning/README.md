# Reinforcement learning examples

These executable examples support the bandit, exploration, dynamic-programming,
Monte Carlo, and temporal-difference sections of the reinforcement-learning
course. They deliberately use small arrays and a deterministic map so each
update can be inspected without a training framework or an external simulator.

## Dependencies and commands

The scripts require Python 3 and NumPy. The focused tests additionally require
pytest. No Gym installation, GPU, network access, or package-specific working
directory is required. From the repository root, run:

```bash
python files/assets/examples/reinforcement-learning/bandit.py --seed 7 --steps 2000
python files/assets/examples/reinforcement-learning/gridworld.py --seed 7
python -m pytest tests/test_reinforcement_learning_examples.py -q
```

If the repository virtual environment is active, the same commands work with
its `python`. Otherwise, replace `python` with the absolute path to any Python
interpreter that has NumPy installed. All file arguments above are relative to
the repository root, so the commands do not depend on the worktree's absolute
location.

## Bandit API and output

`bandit.py` contains four public teaching functions:

| API | Purpose |
| --- | --- |
| `update_sample_average(values, counts, arm, reward)` | Increment the chosen arm's count and sample-mean value in place. |
| `epsilon_greedy(values, epsilon, rng)` | Explore uniformly with probability epsilon; otherwise return the first maximum-value arm. |
| `ucb_action(values, counts, step)` | Return the first untried arm, then use the UCB1 score once every arm has a sample. |
| `run_experiment(seed, steps, epsilon)` | Compare epsilon-greedy and UCB on one seeded 10-arm task and shared potential reward table. |

The command-line interface defaults to seed `7` and 2000 interaction steps.
For each method it prints `cumulative_reward`, `average_regret`, and
`best_arm_fraction`. Cumulative reward is the sum of observed noisy rewards.
Average regret uses the difference between the true best mean and the mean of
the selected arm, averaged over steps. Best-arm fraction is the proportion of
actions that select the arm with the largest true mean. The output contract is
the field names and finite numeric values, not exact decimal values that should
be copied into documentation.

Both methods receive the same true arm means and the same pre-generated
time-by-arm noise table. Their observed reward sequences still differ because
they choose different arms. This setup isolates action selection more carefully
than running each method on a separately sampled task, while remaining only one
seeded comparison.

## GridWorld API and terminal semantics

`gridworld.py` defines a deterministic 2-by-3 `GridWorld`. States are numbered
row-major from 0 to 5; state 0 is the start and state 5 is terminal. Actions 0,
1, 2, and 3 mean up, right, down, and left. A boundary action leaves the state
unchanged. Ordinary transitions return reward -1, while entering or stepping
from the terminal state returns reward 0 and `terminated=True`.

| API | Purpose |
| --- | --- |
| `GridWorld.reset()` / `GridWorld.step(action)` | Reset the state and execute deterministic transitions. |
| `policy_evaluation(env, policy, ...)` | Iteratively evaluate a tabular stochastic policy. |
| `value_iteration(env, ...)` | Compute optimal state values and a deterministic first-argmax policy. |
| `monte_carlo_update(values, counts, state, return_)` | Apply an in-place incremental sample-average return update. |
| `td_target(reward, gamma, next_value, terminated)` | Form a one-step TD target with terminal masking. |
| `q_learning_update(q, ...)` | Update exactly one state-action entry and return its target. |

Terminal masking is part of the algorithm, not a reporting detail. A normal TD
target is `reward + gamma * next_value`, and a normal Q-learning target uses
`reward + gamma * max(q[next_state])`. When `terminated=True`, the bootstrap
term is zero, so both targets reduce to the terminal reward. Bootstrapping from
an arbitrary value stored for terminal state 5 would leak value beyond the end
of an episode.

The GridWorld CLI prints a mean value under the uniform policy, the optimal
start-state value and greedy action array, plus summary statistics for learned
Q-values. These are compact smoke-test summaries; inspect the returned arrays
directly when studying individual states or updates.

## Reproducibility and scope

The bandit seed controls the sampled arm means, potential reward noise, and
epsilon-greedy choices. The GridWorld transition function itself is
deterministic; its seed controls the exploratory choices in the small Q-learning
demonstration. Repeating a command with the same arguments reproduces its output
under the same Python and NumPy behavior. Tests assert same-seed equality and
finite results rather than using a rounded CLI transcript as a golden file.

These examples demonstrate update equations, exploration decisions, fair use of
one sampled bandit task, deterministic transitions, and correct terminal
masking. They do not establish that UCB is universally better than
epsilon-greedy, measure statistical significance across seeds, cover
non-stationary or heavy-tailed bandits, or provide confidence intervals. The
bandit assumes fixed reward means and independent Gaussian noise.

The GridWorld is fully observable and tabular. It does not cover truncation,
stochastic dynamics, function approximation, replay buffers, target networks,
large-scale optimization, or production environment APIs. Its short capped
training loop is a deterministic teaching run, not evidence that the chosen
hyperparameters generalize to other tasks.
