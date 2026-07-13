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

    def test_index_has_complete_ordered_course_contract(self):
        import re

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
                "## 完整学习路线",
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
        for chapter in (
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
        ):
            self.assertIn(chapter, text)
        for target in re.findall(r"\]\(([^)]+\.md)\)", text):
            self.assertTrue((RL_ROOT / target).resolve().is_file(), target)

    def test_agent_environment_loop_has_page_contract(self):
        import re

        text = (RL_ROOT / "rl-agent-environment-loop.md").read_text(encoding="utf-8")
        headings = [line for line in text.splitlines() if line.startswith("## ")]
        self.assertEqual(
            headings,
            [
                "## 本章回答什么",
                "## 先修知识",
                "## 最小任务",
                "## 核心机制",
                "## 训练数据流",
                "## 常见失败",
                "## 自检与练习",
                "## 下一步",
            ],
        )
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
        self.assertIn("POMDP", text)
        self.assertIn("部分可观测马尔可夫决策过程", text)
        targets = re.findall(r"\]\(([^)]+\.md)\)", text)
        self.assertGreaterEqual(len(targets), 2)
        for target in targets:
            self.assertTrue((RL_ROOT / target).resolve().is_file(), target)

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
                "- MDP、价值函数与 Bellman: reinforcement-learning/mdp-value-bellman.md",
                "- Policy Gradient、Actor-Critic、PPO 与 GRPO: reinforcement-learning/policy-gradient-actor-critic-ppo-grpo.md",
                "- 世界模型中的强化学习: reinforcement-learning/rl-for-world-models.md",
                "- verl 训练流程: reinforcement-learning/verl-code-flow.md",
            ],
        )
