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

The bounded dependencies are NumPy 1.x/2.x, PyTorch 2.x, and Gymnasium 1.x with
Classic Control. A display and rendering backend are not required because the
smoke modes never call `render`.

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

Every deep mode rejects `--steps` below 128 with an exact CLI error because a
replay or rollout batch must be large enough for an optimizer smoke update.
REINFORCE finishes the current trajectory, so `collected_steps` can exceed the
requested count. On a CPU laptop each mode is a short smoke-class run, normally
seconds rather than a training job; package installation is usually slower
than the commands.

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

The DQN summary reports update count, final loss, observed episode-return mean,
and online-network parameter change. REINFORCE reports requested/collected
steps, complete episodes, policy loss, return mean, and parameter change. PPO
adds minibatch count and approximate KL. SAC reports critic/actor loss,
temperature, return mean, and actor/critic changes. The stable contract is the
labeled finite fields and a zero exit code, not exact decimals across platforms.

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

## Tabular examples

`bandit.py` contains sample-average epsilon-greedy and UCB1 on one seeded task.
It reports cumulative reward, average regret, and best-arm fraction. Independent
task and policy RNG streams keep action selection from changing the task noise
stream.

`gridworld.py` defines a deterministic 2-by-3 map and implements policy
evaluation, value iteration, first-visit Monte Carlo updates, TD targets, and a
tabular Q-learning update. States are row-major from 0 to 5; state 5 is
terminal. Its CLI summary is also a teaching smoke run, not a generalization or
convergence result.
