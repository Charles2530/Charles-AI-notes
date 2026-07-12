# Reinforcement Learning Curriculum Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the reinforcement-learning topic into a complete, sequential course for deep-learning developers, with prerequisite math bridges, runnable teaching examples, modern LLM/VLA/world-model chapters, and verified MkDocs navigation.

**Architecture:** Keep every existing reinforcement-learning URL stable, make the section index the explicit entry point, and organize the navigation into four dependency-ordered stages. Put reusable teaching code under `files/assets/examples/reinforcement-learning/`, enforce the page contract and numeric objectives with focused tests, and treat existing MDP, PPO, world-model, and verl pages as stable anchors that are narrowed or extended in place.

**Tech Stack:** MkDocs Material, Markdown, MathJax/Arithmatex, Python 3.11+, NumPy, PyTorch, Gymnasium Classic Control, unittest/pytest, browser-based responsive checks.

---

## Working-Tree Rule

The checkout already contains user-owned changes in `mkdocs.yml`, four prerequisite-math pages, site JavaScript, foundations pages, and unrelated topic pages. Do not reset, stash, overwrite, or broadly stage those changes. Read every overlapping file immediately before editing, apply only local patches, and inspect its complete diff afterward. Commits may include only cleanly attributable reinforcement-learning files and newly created files; leave pre-existing changes in overlapping files unstaged unless the user explicitly asks to include them.

## File Map

### Course pages

- Rewrite: `files/reinforcement-learning/index.md` - definition, course map, entry routes.
- Create: `files/reinforcement-learning/rl-agent-environment-loop.md` - vocabulary and end-to-end data flow.
- Create: `files/reinforcement-learning/multi-armed-bandits-and-exploration.md` - exploration foundation.
- Refocus: `files/reinforcement-learning/mdp-value-bellman.md` - MDP, return, value, Q, advantage, Bellman only.
- Create: `files/reinforcement-learning/dynamic-programming-monte-carlo-td.md` - value-estimation families.
- Create: `files/reinforcement-learning/q-learning-and-dqn.md` - value-based deep RL.
- Create: `files/reinforcement-learning/policy-gradient-and-reinforce.md` - score-function path.
- Refocus: `files/reinforcement-learning/policy-gradient-actor-critic-ppo-grpo.md` - Actor-Critic, GAE, PPO.
- Create: `files/reinforcement-learning/continuous-control-and-sac.md` - maximum-entropy continuous control.
- Create: `files/reinforcement-learning/offline-reinforcement-learning.md` - fixed-dataset RL.
- Extend: `files/reinforcement-learning/rl-for-world-models.md` - model-based RL and world models.
- Create: `files/reinforcement-learning/llm-rlhf-rlvr-and-grpo.md` - LLM policy optimization.
- Create: `files/reinforcement-learning/vla-robotics-and-closed-loop-rl.md` - embodied constraints.
- Extend: `files/reinforcement-learning/verl-code-flow.md` - systems chapter.
- Create: `files/reinforcement-learning/practice-and-algorithm-selection.md` - capstone and decision guide.

### Math bridges

- Modify: `files/prerequisite-math/probability-distributions-and-expectation.md`.
- Modify: `files/prerequisite-math/sampling-estimation-and-calibration.md`.
- Modify: `files/prerequisite-math/stochastic-processes-and-markov.md`.
- Modify: `files/prerequisite-math/information-theory-and-divergence.md`.

### Examples and tests

- Create: `files/assets/examples/reinforcement-learning/README.md`.
- Create: `files/assets/examples/reinforcement-learning/requirements.txt`.
- Create: `files/assets/examples/reinforcement-learning/bandit.py`.
- Create: `files/assets/examples/reinforcement-learning/gridworld.py`.
- Create: `files/assets/examples/reinforcement-learning/policy_objectives.py`.
- Create: `files/assets/examples/reinforcement-learning/deep_rl_smoke.py`.
- Create: `tests/test_reinforcement_learning_curriculum.py`.
- Create: `tests/test_reinforcement_learning_examples.py`.

### Site integration

- Modify: `mkdocs.yml` - explicit staged reinforcement-learning navigation.

## Shared Page Contract

Every non-index course page must contain these exact H2 sections, with topic-specific sections allowed between them:

```markdown
## 本章回答什么
## 先修知识
## 最小任务
## 核心机制
## 训练数据流
## 常见失败
## 自检与练习
## 下一步
```

Each page must link to at least one earlier course or prerequisite page and one next course page. The index is exempt from the exact headings but must contain `## 什么是强化学习`, `## Agent 与环境怎样形成闭环`, `## 强化学习和监督学习有什么不同`, and `## 完整学习路线`.

### Task 1: Establish the baseline and curriculum contract test

**Files:**
- Create: `tests/test_reinforcement_learning_curriculum.py`
- Read: `mkdocs.yml`
- Read: `Makefile`
- Read: `files/reinforcement-learning/*.md`

- [ ] **Step 1: Record the current baseline**

Run:

```bash
git status --short
python -m pytest tests/test_content_quality_report.py -q
make build
```

Expected: status records the user-owned changes; the existing content test and strict build exit 0. If the baseline build fails, save the exact pre-existing error and do not attribute it to curriculum work.

- [ ] **Step 2: Write the failing curriculum structure test**

Create `tests/test_reinforcement_learning_curriculum.py` with:

```python
from pathlib import Path
from unittest import TestCase


ROOT = Path(__file__).resolve().parents[1]
RL_ROOT = ROOT / "files" / "reinforcement-learning"

class ReinforcementLearningCurriculumTests(TestCase):
    def test_index_starts_with_definition_and_learning_route(self):
        text = (RL_ROOT / "index.md").read_text(encoding="utf-8")
        for heading in (
            "## 什么是强化学习",
            "## Agent 与环境怎样形成闭环",
            "## 强化学习和监督学习有什么不同",
            "## 完整学习路线",
        ):
            self.assertIn(heading, text)

    def test_nav_starts_with_course_index(self):
        nav = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
        start = nav.index("  - 强化学习:")
        end = nav.index("  - 世界模型与具身智能:", start)
        section = nav[start:end]
        first_page = next(line for line in section.splitlines() if "reinforcement-learning/" in line)
        self.assertIn("reinforcement-learning/index.md", first_page)
```

- [ ] **Step 3: Run the contract test and verify it fails for missing pages**

Run:

```bash
python -m pytest tests/test_reinforcement_learning_curriculum.py -q
```

Expected: FAIL because the index headings are absent and the navigation starts with the MDP page.

### Task 2: Build the course entry and explicit navigation

**Files:**
- Modify: `files/reinforcement-learning/index.md`
- Create: `files/reinforcement-learning/rl-agent-environment-loop.md`
- Modify: `mkdocs.yml`

- [ ] **Step 1: Rewrite the index around the definition**

Use these H2 sections in this order:

```markdown
## 什么是强化学习
## Agent 与环境怎样形成闭环
## 强化学习和监督学习有什么不同
## 强化学习真正难在哪里
## 什么任务适合强化学习
## 完整学习路线
## 按方向进入课程
## 学完后如何判断自己真的理解了
```

The definition must state that an agent learns a policy through environment interaction to maximize expected cumulative return. Include one warehouse-robot time step and a comparison table for supervised learning, imitation learning, offline RL, and online RL. State that RL is a poor first choice when correct action labels are cheap, interaction is unsafe, or reward cannot represent the real goal.

- [ ] **Step 2: Write the agent-environment loop chapter**

Follow the shared page contract. Define observation, state, action, reward, termination, episode, trajectory, policy, rollout, and environment step. Include this data flow:

```text
observation_t -> policy -> action_t -> environment
environment -> reward_t + observation_{t+1} + terminated/truncated
trajectory batch -> return / advantage / target -> loss -> policy update
```

Use the same warehouse task to show sparse reward, delayed reward, partial observability, and the fact that a changed policy collects a changed data distribution.

- [ ] **Step 3: Add the entry pages without referencing unfinished chapters**

Preserve all other current `mkdocs.yml` edits. At this stage, add the index and loop before the four existing pages so strict builds never reference an absent file:

```yaml
  - 强化学习:
      - 强化学习导读: reinforcement-learning/index.md
      - Agent、环境与训练闭环: reinforcement-learning/rl-agent-environment-loop.md
      - MDP、价值函数与 Bellman: reinforcement-learning/mdp-value-bellman.md
      - Policy Gradient、Actor-Critic、PPO 与 GRPO: reinforcement-learning/policy-gradient-actor-critic-ppo-grpo.md
      - 世界模型中的强化学习: reinforcement-learning/rl-for-world-models.md
      - verl 训练流程: reinforcement-learning/verl-code-flow.md
```

- [ ] **Step 4: Validate the entry batch**

Run:

```bash
python -m pytest tests/test_reinforcement_learning_curriculum.py::ReinforcementLearningCurriculumTests::test_index_starts_with_definition_and_learning_route -q
make build
```

Expected: both entry contract tests pass and strict build exits 0.

- [ ] **Step 5: Commit cleanly attributable entry files**

Run:

```bash
git add files/reinforcement-learning/index.md files/reinforcement-learning/rl-agent-environment-loop.md tests/test_reinforcement_learning_curriculum.py
git commit -m "docs: add reinforcement learning course entry"
```

Do not stage `mkdocs.yml` because it contained user-owned changes before this task.

### Task 3: Complete the prerequisite probability bridges

**Files:**
- Modify: `files/prerequisite-math/probability-distributions-and-expectation.md`
- Modify: `files/prerequisite-math/sampling-estimation-and-calibration.md`
- Modify: `files/prerequisite-math/stochastic-processes-and-markov.md`
- Modify: `files/prerequisite-math/information-theory-and-divergence.md`

- [ ] **Step 1: Add conditional expectation and total expectation**

Add a section `## 条件期望：知道当前状态后，未来平均会怎样` containing:

```math
\mathbb E[X\mid Y=y],
\qquad
\mathbb E[X]=\mathbb E_Y[\mathbb E[X\mid Y]].
```

Use a two-state delivery example and link the result to `V^\pi(s)=\mathbb E[G_t\mid S_t=s]` without deriving Bellman on the math page.

- [ ] **Step 2: Add Monte Carlo variance, baseline, and importance sampling**

Add sections `## Monte Carlo 估计为什么会抖`, `## Baseline 为什么能降方差`, and `## 重要性采样：数据来自另一个分布时怎样改权重`. Include:

```math
\hat\mu_N=\frac{1}{N}\sum_{i=1}^N X_i,
\qquad
\operatorname{Var}(\hat\mu_N)=\frac{\sigma^2}{N}
```

```math
\mathbb E_p[f(X)]
=
\mathbb E_q\left[\frac{p(X)}{q(X)}f(X)\right].
```

State the support requirement `q(x)>0` wherever `p(x)f(x)` contributes, and explain that a state-only baseline does not change the expected policy gradient.

- [ ] **Step 3: Strengthen Markov and POMDP boundaries**

Add `## State 是对有用历史的压缩` and `## 从 MDP 到 POMDP`. Explain that the Markov property is conditional independence given a sufficient state, not absence of history. Link belief state to history windows, recurrent state, and latent world models.

- [ ] **Step 4: Add policy entropy and KL direction**

Add `## Policy entropy：探索分布还剩多少选择` and `## KL 用在哪个方向必须写清楚`. Include:

```math
H(\pi(\cdot\mid s))=-\sum_a \pi(a\mid s)\log\pi(a\mid s)
```

```math
D_{\mathrm{KL}}(\pi_\theta\|\pi_{\mathrm{ref}})
=
\mathbb E_{a\sim\pi_\theta}
[\log\pi_\theta(a\mid s)-\log\pi_{\mathrm{ref}}(a\mid s)].
```

State that reverse direction is a different quantity and that token-level approximations must name the sampling distribution.

- [ ] **Step 5: Validate without staging user-owned math changes**

Run:

```bash
rg -n "条件期望|Monte Carlo 估计为什么会抖|重要性采样|State 是对有用历史的压缩|Policy entropy|KL 用在哪个方向" files/prerequisite-math
python scripts/math_rendering_report.py --fail-on-issues --list
```

Expected: all six concepts are found and the math report exits 0. Inspect each file's diff to confirm pre-existing content remains intact; do not commit these overlapping files.

### Task 4: Add the Bandit and GridWorld teaching foundation

**Files:**
- Create: `files/reinforcement-learning/multi-armed-bandits-and-exploration.md`
- Create: `files/assets/examples/reinforcement-learning/bandit.py`
- Create: `files/assets/examples/reinforcement-learning/gridworld.py`
- Create: `files/assets/examples/reinforcement-learning/README.md`
- Create: `tests/test_reinforcement_learning_examples.py`

- [ ] **Step 1: Write failing deterministic example tests**

Create `tests/test_reinforcement_learning_examples.py` with import-by-path setup and these assertions:

```python
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

    def test_terminal_transition_does_not_bootstrap(self):
        gridworld = load_module("gridworld")
        self.assertEqual(gridworld.td_target(1.0, 0.9, 10.0, terminated=True), 1.0)
        self.assertEqual(gridworld.td_target(1.0, 0.9, 10.0, terminated=False), 10.0)
```

- [ ] **Step 2: Run tests and verify missing modules fail**

Run:

```bash
python -m pytest tests/test_reinforcement_learning_examples.py -q
```

Expected: FAIL because `bandit.py` and `gridworld.py` do not exist.

- [ ] **Step 3: Implement the minimal deterministic APIs**

`bandit.py` must expose `update_sample_average`, `epsilon_greedy`, `ucb_action`, and `run_experiment(seed, steps, epsilon)`. `gridworld.py` must expose `GridWorld`, `policy_evaluation`, `value_iteration`, `monte_carlo_update`, `td_target`, and `q_learning_update`. Use `numpy.random.default_rng(seed)` and keep environment dynamics deterministic apart from explicitly seeded action selection.

Use these exact update rules:

```python
def update_sample_average(values, counts, arm, reward):
    counts[arm] += 1
    values[arm] += (reward - values[arm]) / counts[arm]


def td_target(reward, gamma, next_value, terminated):
    return float(reward) if terminated else float(reward + gamma * next_value)


def q_learning_update(q, state, action, reward, next_state, alpha, gamma, terminated):
    bootstrap = 0.0 if terminated else float(np.max(q[next_state]))
    target = reward + gamma * bootstrap
    q[state, action] += alpha * (target - q[state, action])
    return float(target)
```

- [ ] **Step 4: Write the Bandit chapter and example README**

Follow the shared page contract. Compare epsilon-greedy and UCB on the same seeded ten-arm task; define regret as the reward gap to the best arm; explain why Bandit has no state transition or delayed return. `README.md` must document:

```bash
python files/assets/examples/reinforcement-learning/bandit.py --seed 7 --steps 2000
python files/assets/examples/reinforcement-learning/gridworld.py --seed 7
```

- [ ] **Step 5: Run tests and examples**

Run:

```bash
python -m pytest tests/test_reinforcement_learning_examples.py -q
python files/assets/examples/reinforcement-learning/bandit.py --seed 7 --steps 2000
python files/assets/examples/reinforcement-learning/gridworld.py --seed 7
```

Expected: tests pass; both scripts exit 0 and print finite value or reward summaries.

- [ ] **Step 6: Commit the foundation batch**

Run:

```bash
git add files/reinforcement-learning/multi-armed-bandits-and-exploration.md files/assets/examples/reinforcement-learning tests/test_reinforcement_learning_examples.py
git commit -m "docs: add reinforcement learning foundation examples"
```

### Task 5: Refocus MDP and add DP, Monte Carlo, and TD

**Files:**
- Modify: `files/reinforcement-learning/mdp-value-bellman.md`
- Create: `files/reinforcement-learning/dynamic-programming-monte-carlo-td.md`

- [ ] **Step 1: Refocus the MDP page**

Preserve its URL and useful examples. Use the shared page contract plus topic sections for MDP tuple, Markov property, policy, reward versus return, `V`, `Q`, advantage, Bellman expectation, and Bellman optimality. Replace the current full DQN, Offline RL, and world-model sections with short bridge paragraphs linking their dedicated chapters.

- [ ] **Step 2: Write the value-estimation chapter**

Explain policy evaluation, policy improvement, value iteration, first-visit Monte Carlo, TD(0), n-step return, and the bias-variance boundary. Use the shared GridWorld and include:

```math
G_t^{(n)}=\sum_{k=0}^{n-1}\gamma^k r_{t+k}+\gamma^n V(s_{t+n})
```

```math
\delta_t=r_t+\gamma(1-d_t)V(s_{t+1})-V(s_t).
```

State that `d_t` masks true terminal transitions; environment time-limit truncation must be handled according to whether a valid successor value exists.

- [ ] **Step 3: Validate formulas, links, and contract**

Run:

```bash
python -m pytest tests/test_reinforcement_learning_examples.py tests/test_reinforcement_learning_curriculum.py -q
python scripts/math_rendering_report.py --fail-on-issues --list
```

Expected: example and entry-contract tests pass; math report exits 0. Confirm both edited pages contain every shared-contract heading with `rg -n '^## ' files/reinforcement-learning/mdp-value-bellman.md files/reinforcement-learning/dynamic-programming-monte-carlo-td.md`.

### Task 6: Add Q-learning, DQN, and deep-RL smoke support

**Files:**
- Create: `files/reinforcement-learning/q-learning-and-dqn.md`
- Create: `files/assets/examples/reinforcement-learning/requirements.txt`
- Create: `files/assets/examples/reinforcement-learning/deep_rl_smoke.py`
- Modify: `tests/test_reinforcement_learning_examples.py`

- [ ] **Step 1: Declare isolated example dependencies**

Write:

```text
numpy>=1.26,<3
torch>=2.2,<3
gymnasium[classic-control]>=1.0,<2
```

Do not add these packages to the MkDocs `requirements.txt`.

- [ ] **Step 2: Add numeric DQN target tests**

Extend the example test with:

```python
def test_dqn_target_masks_terminal_transition(self):
    deep_rl = load_module("deep_rl_smoke")
    rewards = np.array([1.0, 1.0])
    next_q = np.array([10.0, 10.0])
    terminated = np.array([True, False])
    target = deep_rl.dqn_targets(rewards, next_q, terminated, gamma=0.9)
    np.testing.assert_allclose(target, [1.0, 10.0])
```

- [ ] **Step 3: Implement the smoke module interface**

`deep_rl_smoke.py` must provide NumPy functions `dqn_targets`, `discounted_returns`, `gae`, and `ppo_clipped_loss`, plus CLI modes `dqn`, `reinforce`, `ppo`, and `sac`. The CLI uses PyTorch and Gymnasium only after argument parsing so objective tests remain importable before optional packages are installed. Each mode runs a fixed-seed short training or update loop and exits after confirming all losses and parameters are finite.

Implement the target as:

```python
def dqn_targets(rewards, next_q, terminated, gamma):
    rewards = np.asarray(rewards, dtype=np.float64)
    next_q = np.asarray(next_q, dtype=np.float64)
    mask = 1.0 - np.asarray(terminated, dtype=np.float64)
    return rewards + gamma * mask * next_q
```

- [ ] **Step 4: Write the Q-learning and DQN chapter**

Cover SARSA versus Q-learning, tabular update, neural function approximation, replay buffer, target network, epsilon schedule, Huber loss, terminal masks, training/evaluation separation, and common reward-curve misreads. Link exact commands for `gridworld.py` and `deep_rl_smoke.py dqn`.

- [ ] **Step 5: Verify the objective and optional smoke run**

Run:

```bash
python -m pytest tests/test_reinforcement_learning_examples.py -q
python -m venv .venv-rl-course
source .venv-rl-course/bin/activate
python -m pip install -r files/assets/examples/reinforcement-learning/requirements.txt
python files/assets/examples/reinforcement-learning/deep_rl_smoke.py dqn --seed 7 --steps 256
```

Expected: the objective test passes in the existing environment; course dependencies install only inside `.venv-rl-course`; the DQN smoke command exits 0 with finite loss. The environment change is limited to that untracked directory. Roll back by deactivating it and deleting only `.venv-rl-course` after confirming no shell is using it.

### Task 7: Split Policy Gradient from Actor-Critic and PPO

**Files:**
- Create: `files/reinforcement-learning/policy-gradient-and-reinforce.md`
- Modify: `files/reinforcement-learning/policy-gradient-actor-critic-ppo-grpo.md`
- Modify: `tests/test_reinforcement_learning_examples.py`

- [ ] **Step 1: Add return, GAE, and PPO objective tests**

Add exact numerical cases for discounted returns, GAE with a terminal mask, and PPO clipping:

```python
def test_ppo_clipping_limits_positive_advantage(self):
    deep_rl = load_module("deep_rl_smoke")
    loss = deep_rl.ppo_clipped_loss(
        new_logp=np.log(np.array([1.4])),
        old_logp=np.log(np.array([1.0])),
        advantage=np.array([2.0]),
        clip_epsilon=0.2,
    )
    self.assertAlmostEqual(loss, -2.4)
```

- [ ] **Step 2: Implement objective helpers**

Use reverse-time recurrences for `discounted_returns` and `gae`; reset continuation at terminal transitions. Implement PPO loss as the negative mean of `min(ratio * advantage, clip(ratio) * advantage)`.

- [ ] **Step 3: Write the Policy Gradient chapter**

Move score-function estimator, REINFORCE, return-to-go, baseline, variance, and exploration material from the existing combined page. Include a trajectory-to-loss table and link the `reinforce` smoke command.

- [ ] **Step 4: Refocus the existing page on Actor-Critic, GAE, and PPO**

Keep the stable URL. Explain actor and critic targets, GAE, old-policy logprob, probability ratio, clipping, entropy, KL, minibatch reuse, and approximate on-policy constraints. Move full GRPO, visual-generation RL, LLM rollout-cost, and method-comparison material to the later LLM chapter; retain concise summaries and links.

- [ ] **Step 5: Verify objectives and smoke modes**

Run:

```bash
python -m pytest tests/test_reinforcement_learning_examples.py -q
python files/assets/examples/reinforcement-learning/deep_rl_smoke.py reinforce --seed 7 --steps 256
python files/assets/examples/reinforcement-learning/deep_rl_smoke.py ppo --seed 7 --steps 256
```

Expected: numeric objective tests pass; both smoke modes exit 0 with finite losses.

### Task 8: Add continuous control and SAC

**Files:**
- Create: `files/reinforcement-learning/continuous-control-and-sac.md`
- Modify: `files/assets/examples/reinforcement-learning/deep_rl_smoke.py`

- [ ] **Step 1: Implement SAC smoke mode**

Use Pendulum-v1, a squashed Gaussian actor, twin Q networks, replay samples, target Q values, and Polyak target updates. The smoke path needs only to collect seeded transitions and complete finite actor, critic, and temperature updates; it does not claim convergence.

- [ ] **Step 2: Write the SAC chapter**

Explain why DQN cannot enumerate continuous actions, reparameterized actions, `tanh` log-prob correction, maximum-entropy objective, twin-Q minimum, off-policy replay, target networks, and temperature. Explicitly separate environment action scale from the normalized actor output.

- [ ] **Step 3: Run the SAC smoke command**

Run:

```bash
python files/assets/examples/reinforcement-learning/deep_rl_smoke.py sac --seed 7 --steps 256
```

Expected: exit 0 and finite actor, critic, and temperature losses.

- [ ] **Step 4: Commit the classic deep-RL batch**

Run:

```bash
git add files/reinforcement-learning/mdp-value-bellman.md files/reinforcement-learning/dynamic-programming-monte-carlo-td.md files/reinforcement-learning/q-learning-and-dqn.md files/reinforcement-learning/policy-gradient-and-reinforce.md files/reinforcement-learning/policy-gradient-actor-critic-ppo-grpo.md files/reinforcement-learning/continuous-control-and-sac.md files/assets/examples/reinforcement-learning/deep_rl_smoke.py files/assets/examples/reinforcement-learning/requirements.txt tests/test_reinforcement_learning_examples.py
git commit -m "docs: build classic reinforcement learning course"
```

### Task 9: Add Offline RL and refocus model-based RL

**Files:**
- Create: `files/reinforcement-learning/offline-reinforcement-learning.md`
- Modify: `files/reinforcement-learning/rl-for-world-models.md`

- [ ] **Step 1: Write the Offline RL chapter**

Follow the shared contract. Compare behavioral cloning, off-policy online RL, and offline RL. Explain dataset support, extrapolation error, overestimated Q values, behavior constraints, conservative value estimation, and offline policy evaluation. Use one fixed robot dataset example and show why selecting an unseen high-Q action is unsafe.

- [ ] **Step 2: Extend the world-model page into model-based RL**

Prepend Dyna, learned transition/reward models, planning, MPC, and model predictive control before latent world models. Preserve Dreamer, video RL, reward-model, and model-exploitation material that belongs here. Separate open-loop prediction quality from closed-loop return.

- [ ] **Step 3: Validate page boundaries**

Run:

```bash
rg -n "distribution support|extrapolation|保守|Dyna|MPC|model exploitation|closed-loop" files/reinforcement-learning/offline-reinforcement-learning.md files/reinforcement-learning/rl-for-world-models.md
python scripts/content_quality_report.py --include-hints --list-pages
```

Expected: all boundary concepts are present; neither page is flagged for missing links, missing formula context, or excessive section count.

### Task 10: Add LLM RL, VLA, and policy-objective examples

**Files:**
- Create: `files/reinforcement-learning/llm-rlhf-rlvr-and-grpo.md`
- Create: `files/reinforcement-learning/vla-robotics-and-closed-loop-rl.md`
- Create: `files/assets/examples/reinforcement-learning/policy_objectives.py`
- Modify: `tests/test_reinforcement_learning_examples.py`

- [ ] **Step 1: Write group-advantage and KL tests**

Add tests asserting that normalized group advantages sum to approximately zero, identical group rewards produce zero advantages, and sampled forward KL uses `policy_logp - reference_logp`.

- [ ] **Step 2: Implement the policy-objective helpers**

Expose:

```python
def group_advantages(rewards, epsilon=1e-8):
    rewards = np.asarray(rewards, dtype=np.float64)
    centered = rewards - rewards.mean(axis=-1, keepdims=True)
    scale = rewards.std(axis=-1, keepdims=True)
    return np.where(scale > epsilon, centered / (scale + epsilon), 0.0)


def sampled_forward_kl(policy_logp, reference_logp):
    return np.asarray(policy_logp) - np.asarray(reference_logp)
```

The CLI must print a small prompt/group table, group advantages, KL penalties, and a finite clipped policy loss.

- [ ] **Step 3: Write the LLM RL chapter**

Map prompt/prefix, token, response, verifier/reward model, old policy, and reference policy onto the shared RL vocabulary. Explain RLHF, RLVR, PPO, GRPO, group normalization, zero-variance groups, KL, rollout cost, and reward hacking. Compare DPO as offline preference optimization and state precisely why it is not an online environment-interaction algorithm.

- [ ] **Step 4: Write the VLA chapter**

Explain POMDP observations, continuous action or action chunks, demonstrations, offline pretraining, simulation, world-model rollouts, real deployment data, safety constraints, and recovery. Include a table separating open-loop action prediction, simulator success, and real closed-loop success.

- [ ] **Step 5: Run policy-objective tests and example**

Run:

```bash
python -m pytest tests/test_reinforcement_learning_examples.py -q
python files/assets/examples/reinforcement-learning/policy_objectives.py
```

Expected: tests pass; output shows zero-centered group advantages and finite KL-adjusted objective values.

- [ ] **Step 6: Commit the modern application batch**

Run:

```bash
git add files/reinforcement-learning/offline-reinforcement-learning.md files/reinforcement-learning/rl-for-world-models.md files/reinforcement-learning/llm-rlhf-rlvr-and-grpo.md files/reinforcement-learning/vla-robotics-and-closed-loop-rl.md files/assets/examples/reinforcement-learning/policy_objectives.py tests/test_reinforcement_learning_examples.py
git commit -m "docs: add modern reinforcement learning applications"
```

### Task 11: Complete systems engineering and algorithm selection

**Files:**
- Modify: `files/reinforcement-learning/verl-code-flow.md`
- Create: `files/reinforcement-learning/practice-and-algorithm-selection.md`
- Modify: `files/assets/examples/reinforcement-learning/README.md`

- [ ] **Step 1: Bring verl into the shared course contract**

Preserve the existing code-flow analysis. Add learning goals, prerequisites, one full field-level flow from prompt batch to rollout to reward to old/reference logprob to advantage to training minibatch, common staleness and mask failures, self-check questions, and the capstone next link.

- [ ] **Step 2: Write the algorithm-selection capstone**

Organize decisions by observation/state quality, discrete versus continuous action, online interaction availability, dataset-only constraints, reward quality, environment model availability, and safety cost. Include a decision table with Bandit, DQN, PPO, SAC, Offline RL, Model-based RL, and GRPO. Provide three capstones: GridWorld diagnosis, Pendulum algorithm selection, and verifier-based LLM objective inspection.

- [ ] **Step 3: Finish the example README**

Document environment creation, dependency installation, every command, expected runtime class, seed behavior, and what each smoke test proves and does not prove. Include:

```bash
python -m venv .venv-rl-course
source .venv-rl-course/bin/activate
python -m pip install -r files/assets/examples/reinforcement-learning/requirements.txt
python -m pytest tests/test_reinforcement_learning_examples.py -q
```

- [ ] **Step 4: Extend the contract test to cover the complete course**

Add these constants after `RL_ROOT` in `tests/test_reinforcement_learning_curriculum.py`:

```python
COURSE_PAGES = [
    "rl-agent-environment-loop.md",
    "multi-armed-bandits-and-exploration.md",
    "mdp-value-bellman.md",
    "dynamic-programming-monte-carlo-td.md",
    "q-learning-and-dqn.md",
    "policy-gradient-and-reinforce.md",
    "policy-gradient-actor-critic-ppo-grpo.md",
    "continuous-control-and-sac.md",
    "offline-reinforcement-learning.md",
    "rl-for-world-models.md",
    "llm-rlhf-rlvr-and-grpo.md",
    "vla-robotics-and-closed-loop-rl.md",
    "verl-code-flow.md",
    "practice-and-algorithm-selection.md",
]

REQUIRED_SECTIONS = [
    "## 本章回答什么",
    "## 先修知识",
    "## 最小任务",
    "## 核心机制",
    "## 训练数据流",
    "## 常见失败",
    "## 自检与练习",
    "## 下一步",
]
```

Add these methods inside `ReinforcementLearningCurriculumTests`:

```python
def test_all_course_pages_exist_and_follow_contract(self):
    for name in COURSE_PAGES:
        with self.subTest(page=name):
            text = (RL_ROOT / name).read_text(encoding="utf-8")
            for heading in REQUIRED_SECTIONS:
                self.assertIn(heading, text)
            local_course_links = {
                target.split("#", 1)[0]
                for target in re.findall(r"\[[^\]]+\]\(([^)]+\.md(?:#[^)]+)?)\)", text)
            }
            self.assertGreaterEqual(len(local_course_links), 2)

def test_nav_exposes_all_course_pages_after_index(self):
    nav = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    start = nav.index("  - 强化学习:")
    end = nav.index("  - 世界模型与具身智能:", start)
    section = nav[start:end]
    index_position = section.index("reinforcement-learning/index.md")
    for name in COURSE_PAGES:
            self.assertGreater(section.index(f"reinforcement-learning/{name}"), index_position)
```

Also add `import re` at the top of the test file. The two-link assertion enforces one backward/prerequisite route and one forward or course-index route without coupling tests to link labels.

Run:

```bash
python -m pytest tests/test_reinforcement_learning_curriculum.py -q
```

Expected: the page-contract test passes and the complete-navigation test fails because `mkdocs.yml` still has the temporary short navigation.

- [ ] **Step 5: Replace the temporary navigation with the complete course navigation**

Preserve all non-RL changes in `mkdocs.yml`. Replace only the reinforcement-learning block with:

```yaml
  - 强化学习:
      - 强化学习导读: reinforcement-learning/index.md
      - 建立决策语言:
          - Agent、环境与训练闭环: reinforcement-learning/rl-agent-environment-loop.md
          - 多臂老虎机与探索: reinforcement-learning/multi-armed-bandits-and-exploration.md
          - MDP、价值函数与 Bellman: reinforcement-learning/mdp-value-bellman.md
          - 动态规划、Monte Carlo 与 TD: reinforcement-learning/dynamic-programming-monte-carlo-td.md
      - 经典深度强化学习:
          - Q-learning 与 DQN: reinforcement-learning/q-learning-and-dqn.md
          - Policy Gradient 与 REINFORCE: reinforcement-learning/policy-gradient-and-reinforce.md
          - Actor-Critic、GAE 与 PPO: reinforcement-learning/policy-gradient-actor-critic-ppo-grpo.md
          - 连续控制与 SAC: reinforcement-learning/continuous-control-and-sac.md
      - 低成本与模型驱动学习:
          - Offline RL: reinforcement-learning/offline-reinforcement-learning.md
          - Model-based RL 与世界模型: reinforcement-learning/rl-for-world-models.md
      - 大模型、机器人与系统:
          - LLM RLHF、RLVR 与 GRPO: reinforcement-learning/llm-rlhf-rlvr-and-grpo.md
          - VLA、机器人与闭环 RL: reinforcement-learning/vla-robotics-and-closed-loop-rl.md
          - verl 训练流程: reinforcement-learning/verl-code-flow.md
          - 综合实践与算法选择: reinforcement-learning/practice-and-algorithm-selection.md
```

- [ ] **Step 6: Run the full curriculum contract**

Run:

```bash
python -m pytest tests/test_reinforcement_learning_curriculum.py tests/test_reinforcement_learning_examples.py -q
```

Expected: all curriculum pages exist, every page satisfies the shared contract, index headings exist, navigation order is correct, and all numeric objective tests pass.

- [ ] **Step 7: Commit the systems and capstone pages**

Run:

```bash
git add files/reinforcement-learning/verl-code-flow.md files/reinforcement-learning/practice-and-algorithm-selection.md files/assets/examples/reinforcement-learning/README.md
git commit -m "docs: complete reinforcement learning course path"
```

Do not stage `mkdocs.yml`; its reinforcement-learning navigation remains part of the final working-tree handoff because the file already contained unrelated user changes.

### Task 12: Final content, build, and responsive verification

**Files:**
- Verify: `files/reinforcement-learning/*.md`
- Verify: `files/prerequisite-math/*.md`
- Verify: `files/assets/examples/reinforcement-learning/*`
- Verify: `mkdocs.yml`

- [ ] **Step 1: Run content and math checks**

Run:

```bash
python scripts/content_quality_report.py --include-hints --list-pages
python scripts/math_rendering_report.py --fail-on-issues --list
git diff --check
```

Expected: no RL page has broken local links, unexplained display formulas, missing image sources, template flags, or whitespace errors. Any unrelated pre-existing report findings are listed separately.

- [ ] **Step 2: Run the complete focused test suite and strict build**

Run:

```bash
python -m pytest tests/test_reinforcement_learning_curriculum.py tests/test_reinforcement_learning_examples.py tests/test_content_quality_report.py -q
make build
```

Expected: all focused tests pass and MkDocs strict build exits 0.

- [ ] **Step 3: Start a local server without taking over an occupied port**

Run:

```bash
make serve DEV_ADDR=127.0.0.1:8011
```

Expected: the site is available at `http://127.0.0.1:8011/reinforcement-learning/`. If `8011` is already occupied, increment to the next free port and record it.

- [ ] **Step 4: Verify desktop and mobile rendering**

Use browser automation at 1440x900 and 390x844. Check the index, `mdp-value-bellman/`, `policy-gradient-actor-critic-ppo-grpo/`, `continuous-control-and-sac/`, and `llm-rlhf-rlvr-and-grpo/`. Confirm:

```text
The index first viewport contains the definition and interaction loop.
The sidebar starts with 强化学习导读 and follows course order.
Math, tables, code blocks, and navigation do not overlap or clip.
Previous/next links work.
No console error prevents MathJax or diagram rendering.
```

- [ ] **Step 5: Review the final diff against the design**

Run:

```bash
git status --short
git diff -- files/reinforcement-learning files/assets/examples/reinforcement-learning tests/test_reinforcement_learning_curriculum.py tests/test_reinforcement_learning_examples.py mkdocs.yml files/prerequisite-math
```

Expected: every changed line maps to the curriculum design; unrelated user-owned changes remain present and are neither reverted nor silently staged.
