import re
from pathlib import Path
from unittest import TestCase

ROOT = Path(__file__).resolve().parents[1]
RL_ROOT = ROOT / "files" / "reinforcement-learning"

class ReinforcementLearningCurriculumTests(TestCase):
    REQUIRED_CHAPTER_HEADINGS = [
        "## 本章回答什么",
        "## 先修知识",
        "## 最小任务",
        "## 核心机制",
        "## 训练数据流",
        "## 常见失败",
        "## 自检与练习",
        "## 下一步",
    ]

    def assert_ordered_subsequence(self, required, actual):
        remaining = iter(actual)
        for item in required:
            self.assertIn(item, remaining)

    def assert_local_links_resolve(self, page):
        text = page.read_text(encoding="utf-8")
        local_targets = []
        for target in re.findall(r"!?\[[^]]*\]\(([^)]+)\)", text):
            if target.startswith(("http://", "https://", "#")):
                continue
            path = target.split("#", maxsplit=1)[0]
            if not path:
                continue
            local_targets.append(path)
            self.assertTrue((page.parent / path).resolve().is_file(), target)
        self.assertGreaterEqual(len(local_targets), 2)

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

    def test_index_has_complete_ordered_course_contract(self):
        text = (RL_ROOT / "index.md").read_text(encoding="utf-8")
        headings = [line for line in text.splitlines() if line.startswith("## ")]
        self.assertEqual(
            headings,
            [
                "## 什么是强化学习",
                "## Agent 与环境怎样形成闭环",
                "## 强化学习和监督学习有什么不同",
                "## 强化学习真正难在哪里",
                "## 什么任务适合强化学习",
                "## 完整学习路线 {#完整学习路线}",
                "## 按方向进入课程",
                "## 学完后如何判断自己真的理解了",
            ],
        )
        opening = text.split("## 什么是强化学习", maxsplit=1)[0]
        self.assertIn("agent 通过与环境交互学习 policy", opening)
        self.assertIn("期望累计回报最大化", opening)
        for row in ("监督学习", "模仿学习", "Offline RL", "Online RL"):
            self.assertIn(f"| {row} |", text)
        for warning in ("正确动作标签很便宜", "交互不安全", "reward 无法代表真实目标"):
            self.assertIn(warning, text)
        course_map = text.split("## 完整学习路线", maxsplit=1)[1].split(
            "## 按方向进入课程", maxsplit=1
        )[0]
        chapter_titles = []
        for row in course_map.splitlines():
            cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
            if len(cells) != 3:
                continue
            numbered_title = re.fullmatch(r"\d+\.\s+(.+)", cells[1])
            if numbered_title is None:
                continue
            title = re.sub(
                r"^\[([^]]+)\]\([^)]+\)$", r"\1", numbered_title.group(1)
            )
            chapter_titles.append(title)
        self.assertEqual(
            chapter_titles,
            [
                "强化学习专题首页",
                "Agent、环境与训练闭环",
                "多臂老虎机与探索",
                "MDP、回报、价值函数与 Bellman",
                "动态规划、Monte Carlo 与 TD",
                "Q-learning 与 DQN",
                "Policy Gradient 与 REINFORCE",
                "Actor-Critic、GAE 与 PPO",
                "连续控制与 SAC",
                "Offline Reinforcement Learning",
                "Model-based RL 与世界模型",
                "LLM 强化学习：RLHF、RLVR 与 GRPO",
                "VLA、机器人与闭环强化学习",
                "RL 系统工程与 verl",
                "综合实践与算法选择",
            ],
        )
        self.assertIn(
            "| 长期价值 | 3. [多臂老虎机与探索](multi-armed-bandits-and-exploration.md) | 可阅读 |",
            course_map,
        )
        self.assertIn(
            "| 长期价值 | 4. [MDP、回报、价值函数与 Bellman](mdp-value-bellman.md) | 可阅读 |",
            course_map,
        )
        self.assertIn(
            "| 长期价值 | 5. [动态规划、Monte Carlo 与 TD](dynamic-programming-monte-carlo-td.md) | 可阅读 |",
            course_map,
        )
        beginner_route = text.split("## 按方向进入课程", maxsplit=1)[1]
        self.assertIn(
            "[交互闭环](rl-agent-environment-loop.md) -> "
            "[多臂老虎机与探索](multi-armed-bandits-and-exploration.md) -> "
            "[MDP 与 Bellman](mdp-value-bellman.md) -> "
            "[动态规划、Monte Carlo 与 TD](dynamic-programming-monte-carlo-td.md)",
            beginner_route,
        )
        for target in re.findall(r"\]\(([^)]+\.md)\)", text):
            self.assertTrue((RL_ROOT / target).resolve().is_file(), target)

    def test_agent_environment_loop_has_page_contract(self):
        text = (RL_ROOT / "rl-agent-environment-loop.md").read_text(encoding="utf-8")
        headings = [line for line in text.splitlines() if line.startswith("## ")]
        required_headings = [
            "## 本章回答什么",
            "## 先修知识",
            "## 最小任务",
            "## 核心机制",
            "## 训练数据流",
            "## 常见失败",
            "## 自检与练习",
            "## 下一步",
        ]

        def assert_ordered_subsequence(actual):
            remaining = iter(actual)
            for required in required_headings:
                self.assertIn(required, remaining)

        assert_ordered_subsequence(headings)
        synthetic_headings = required_headings[:4] + ["## 主题扩展"] + required_headings[4:]
        assert_ordered_subsequence(synthetic_headings)
        for term in (
            "observation",
            "state",
            "action",
            "reward",
            "termination",
            "truncation",
            "episode",
            "trajectory",
            "policy",
            "rollout",
            "environment step",
        ):
            self.assertIn(f"| {term} |", text)
        flow = """observation_t -> policy -> action_t -> environment
environment -> reward_t + observation_{t+1} + terminated/truncated
trajectory batch -> return / advantage / target -> loss -> policy update"""
        self.assertIn(flow, text)
        observation_section = text.split(
            "### Observation 不等于 state", maxsplit=1
        )[1].split("### Reward 不等于完整目标", maxsplit=1)[0]
        self.assertIn("部分可观测马尔可夫决策过程（POMDP）", observation_section)
        self.assertIn("环境内部仍有潜在的 Markov state", observation_section)
        self.assertIn(
            "agent 每步只收到这个 state 的部分 observation", observation_section
        )
        self.assertIn("历史帧", observation_section)
        self.assertIn("记忆模块", observation_section)
        self.assertIn("belief state", observation_section)
        self.assertIn("推断", observation_section)
        reward_section = text.split("### Reward 不等于完整目标", maxsplit=1)[1].split(
            "### Policy 既是模型，也是数据采集规则", maxsplit=1
        )[0]
        self.assertIn("稀疏的是成功信号", reward_section)
        self.assertIn("每步都有 `-0.01` 的时间惩罚", reward_section)
        targets = re.findall(r"\]\(([^)]+\.md)\)", text)
        self.assertGreaterEqual(len(targets), 2)
        self.assertIn("mdp-value-bellman.md", targets)
        self.assertTrue(
            {
                "index.md",
                "../prerequisite-math/probability-distributions-and-expectation.md",
                "../prerequisite-math/stochastic-processes-and-markov.md",
                "../prerequisite-math/value-policy-and-rl-objectives.md",
            }
            & set(targets)
        )
        for target in targets:
            self.assertTrue((RL_ROOT / target).resolve().is_file(), target)

    def test_mdp_page_has_value_and_bellman_contract(self):
        page = RL_ROOT / "mdp-value-bellman.md"
        text = page.read_text(encoding="utf-8")
        headings = [line for line in text.splitlines() if line.startswith("## ")]
        self.assert_ordered_subsequence(self.REQUIRED_CHAPTER_HEADINGS, headings)
        for term in (
            "\\mathcal{M}=(\\mathcal{S},\\mathcal{A},P,R,\\gamma)",
            "Markov property",
            "\\pi(a\\mid s)",
            "reward",
            "return",
            "V^\\pi",
            "Q^\\pi",
            "A^\\pi",
            "Bellman expectation",
            "Bellman optimality",
            "state",
            "observation",
        ):
            self.assertIn(term, text)
        for target in (
            "../prerequisite-math/stochastic-processes-and-markov.md",
            "rl-agent-environment-loop.md",
            "dynamic-programming-monte-carlo-td.md",
        ):
            self.assertIn(f"]({target})", text)
        self.assert_local_links_resolve(page)

    def test_dynamic_programming_monte_carlo_td_page_contract(self):
        page = RL_ROOT / "dynamic-programming-monte-carlo-td.md"
        self.assertTrue(page.is_file())
        text = page.read_text(encoding="utf-8")
        headings = [line for line in text.splitlines() if line.startswith("## ")]
        self.assert_ordered_subsequence(self.REQUIRED_CHAPTER_HEADINGS, headings)
        for term in (
            "policy evaluation",
            "policy improvement",
            "policy iteration",
            "value iteration",
            "first-visit Monte Carlo",
            "TD(0)",
            "G_t^{(n)}",
            "bias-variance",
            "\\delta_t=r_t+\\gamma(1-d_t)V(s_{t+1})-V(s_t)",
            "terminated",
            "truncated",
        ):
            self.assertIn(term, text)
        for target in (
            "mdp-value-bellman.md",
            "../assets/examples/reinforcement-learning/gridworld.py",
            "../assets/examples/reinforcement-learning/README.md",
            "index.md#完整学习路线",
        ):
            self.assertIn(f"]({target})", text)
        self.assert_local_links_resolve(page)

    def test_nav_uses_temporary_build_safe_course_list(self):
        nav = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
        start = nav.index("  - 强化学习:")
        end = nav.index("  - 世界模型与具身智能:", start)
        section = nav[start:end]
        pages = [
            line.strip()
            for line in section.splitlines()
            if "reinforcement-learning/" in line
        ]
        self.assertEqual(
            pages,
            [
                "- 强化学习导读: reinforcement-learning/index.md",
                "- Agent、环境与训练闭环: reinforcement-learning/rl-agent-environment-loop.md",
                "- 多臂老虎机与探索: reinforcement-learning/multi-armed-bandits-and-exploration.md",
                "- MDP、价值函数与 Bellman: reinforcement-learning/mdp-value-bellman.md",
                "- 动态规划、Monte Carlo 与 TD: reinforcement-learning/dynamic-programming-monte-carlo-td.md",
                "- Policy Gradient、Actor-Critic、PPO 与 GRPO: reinforcement-learning/policy-gradient-actor-critic-ppo-grpo.md",
                "- 世界模型中的强化学习: reinforcement-learning/rl-for-world-models.md",
                "- verl 训练流程: reinforcement-learning/verl-code-flow.md",
            ],
        )
