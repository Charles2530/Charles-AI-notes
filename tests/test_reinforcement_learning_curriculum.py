import re
from pathlib import Path
from unittest import TestCase

ROOT = Path(__file__).resolve().parents[1]
RL_ROOT = ROOT / "files" / "reinforcement-learning"

class ReinforcementLearningCurriculumTests(TestCase):
    COURSE_PAGES = (
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
    )
    CLASSIC_PAGES = (
        "q-learning-and-dqn.md",
        "policy-gradient-and-reinforce.md",
        "policy-gradient-actor-critic-ppo-grpo.md",
        "continuous-control-and-sac.md",
    )
    MODERN_PAGES = (
        "offline-reinforcement-learning.md",
        "rl-for-world-models.md",
        "llm-rlhf-rlvr-and-grpo.md",
        "vla-robotics-and-closed-loop-rl.md",
    )
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
        local_link_targets = []
        for image_marker, target in re.findall(r"(!?)\[[^]]*\]\(([^)]+)\)", text):
            if target.startswith(("http://", "https://", "#")):
                continue
            path = target.split("#", maxsplit=1)[0]
            if not path:
                continue
            if not image_marker:
                local_link_targets.append(path)
            self.assertTrue((page.parent / path).resolve().is_file(), target)
        self.assertGreaterEqual(len(set(local_link_targets)), 2)

    def h2_section(self, text, heading):
        start = text.index(f"## {heading}")
        following = re.search(r"^## ", text[start + 3 :], flags=re.MULTILINE)
        if following is None:
            return text[start:]
        return text[start : start + 3 + following.start()]

    def first_markdown_table(self, section):
        groups = []
        current = []
        for line in section.splitlines():
            if line.strip().startswith("|") and line.strip().endswith("|"):
                current.append(line)
            elif current:
                groups.append(current)
                current = []
        if current:
            groups.append(current)
        self.assertTrue(groups, "section must contain a Markdown table")
        table = groups[0]
        self.assertGreaterEqual(len(table), 3)

        def cells(line):
            return [cell.strip() for cell in line.strip().strip("|").split("|")]

        header = cells(table[0])
        rows = {row[0]: row[1:] for row in map(cells, table[2:])}
        return header, rows

    def normalize_formula_text(self, section):
        without_spacing_commands = re.sub(r"\\[,;:!]", "", section)
        without_delimiters = without_spacing_commands.replace("`", "").replace("$", "")
        return re.sub(r"\s+", "", without_delimiters)

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

    def test_all_course_pages_share_final_contract_and_resolve_links(self):
        for filename in self.COURSE_PAGES:
            with self.subTest(filename=filename):
                page = RL_ROOT / filename
                self.assertTrue(page.is_file(), filename)
                text = page.read_text(encoding="utf-8")
                headings = [line for line in text.splitlines() if line.startswith("## ")]
                self.assert_ordered_subsequence(self.REQUIRED_CHAPTER_HEADINGS, headings)
                self.assert_local_links_resolve(page)

                next_section = self.h2_section(text, "下一步")
                course_targets = {
                    target.split("#", maxsplit=1)[0]
                    for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", next_section)
                }
                self.assertTrue(
                    course_targets.intersection({"index.md", *self.COURSE_PAGES}),
                    f"{filename} 下一步 must link to the course or its index",
                )

    def test_published_course_exits_have_no_stale_planning_phrases(self):
        stale_phrases = (
            "规划中",
            "规划状态",
            "未来章节",
            "尚未实现",
            "没有独立课程页",
            "还没有独立课程页",
            "尚无独立课程页",
        )
        index = (RL_ROOT / "index.md").read_text(encoding="utf-8")
        for stale in stale_phrases:
            self.assertNotIn(stale, index)
        for filename in self.COURSE_PAGES:
            with self.subTest(filename=filename):
                text = (RL_ROOT / filename).read_text(encoding="utf-8")
                next_section = self.h2_section(text, "下一步")
                for stale in stale_phrases:
                    self.assertNotIn(stale, next_section)

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
        course_rows = []
        for row in course_map.splitlines():
            cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
            if len(cells) != 3 or not re.match(r"\d+\.\s+", cells[1]):
                continue
            match = re.fullmatch(r"(\d+)\.\s+\[([^]]+)\]\(([^)]+)\)", cells[1])
            self.assertIsNotNone(match, cells[1])
            course_rows.append((cells[0], int(match.group(1)), match.group(3), cells[2]))
        self.assertEqual(
            course_rows,
            [
                ("建立语言", 1, "index.md", "可阅读"),
                ("建立语言", 2, "rl-agent-environment-loop.md", "可阅读"),
                ("长期价值", 3, "multi-armed-bandits-and-exploration.md", "可阅读"),
                ("长期价值", 4, "mdp-value-bellman.md", "可阅读"),
                ("长期价值", 5, "dynamic-programming-monte-carlo-td.md", "可阅读"),
                ("经典深度 RL", 6, "q-learning-and-dqn.md", "可阅读"),
                ("经典深度 RL", 7, "policy-gradient-and-reinforce.md", "可阅读"),
                ("经典深度 RL", 8, "policy-gradient-actor-critic-ppo-grpo.md", "可阅读"),
                ("经典深度 RL", 9, "continuous-control-and-sac.md", "可阅读"),
                ("低成本与模型驱动", 10, "offline-reinforcement-learning.md", "可阅读"),
                ("低成本与模型驱动", 11, "rl-for-world-models.md", "可阅读"),
                ("大模型与具身系统", 12, "llm-rlhf-rlvr-and-grpo.md", "可阅读"),
                ("大模型与具身系统", 13, "vla-robotics-and-closed-loop-rl.md", "可阅读"),
                ("大模型与具身系统", 14, "verl-code-flow.md", "可阅读"),
                ("综合", 15, "practice-and-algorithm-selection.md", "可阅读"),
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
        for row in (
            "| 经典深度 RL | 6. [Q-learning 与 DQN](q-learning-and-dqn.md) | 可阅读 |",
            "| 经典深度 RL | 7. [Policy Gradient 与 REINFORCE](policy-gradient-and-reinforce.md) | 可阅读 |",
            "| 经典深度 RL | 8. [Actor-Critic、GAE 与 PPO](policy-gradient-actor-critic-ppo-grpo.md) | 可阅读 |",
            "| 经典深度 RL | 9. [连续控制与 SAC](continuous-control-and-sac.md) | 可阅读 |",
        ):
            self.assertIn(row, course_map)
        beginner_route = text.split("## 按方向进入课程", maxsplit=1)[1]
        self.assertIn(
            "[交互闭环](rl-agent-environment-loop.md) -> "
            "[多臂老虎机与探索](multi-armed-bandits-and-exploration.md) -> "
            "[MDP 与 Bellman](mdp-value-bellman.md) -> "
            "[动态规划、Monte Carlo 与 TD](dynamic-programming-monte-carlo-td.md) -> "
            "[Q-learning 与 DQN](q-learning-and-dqn.md) -> "
            "[Policy Gradient 与 REINFORCE](policy-gradient-and-reinforce.md) -> "
            "[Actor-Critic、GAE 与 PPO](policy-gradient-actor-critic-ppo-grpo.md) -> "
            "[连续控制与 SAC](continuous-control-and-sac.md)",
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

    def test_mdp_transition_reward_and_terminal_contract(self):
        text = (RL_ROOT / "mdp-value-bellman.md").read_text(encoding="utf-8")
        mdp_section = text.split(
            "### MDP 五元组与 Markov property", maxsplit=1
        )[1].split("### State 与 observation", maxsplit=1)[0]
        self.assertIn(
            "R(s,a,s')=\\mathbb{E}[r_t\\mid s_t=s,a_t=a,s_{t+1}=s']",
            mdp_section,
        )
        self.assertIn(
            "p(s_{t+1},r_t\\mid s_0,a_0,r_0,\\ldots,s_t,a_t)"
            "=p(s_{t+1},r_t\\mid s_t,a_t)",
            mdp_section,
        )

        expectation = text.split(
            "### Bellman expectation：评估固定策略", maxsplit=1
        )[1].split("### Bellman optimality：比较动作", maxsplit=1)[0]
        for term in (
            "非 terminal state",
            "V^\\pi(s_{\\mathrm{terminal}})=0",
            "transition 的随机性",
            "条件期望 reward",
            "若 successor \\(s'\\) 是 terminal",
            "\\sum_{a'}\\pi(a'\\mid s')Q^\\pi(s',a')=0",
        ):
            self.assertIn(term, expectation)

        optimality = text.split(
            "### Bellman optimality：比较动作", maxsplit=1
        )[1].split("### Discount 与 effective horizon", maxsplit=1)[0]
        self.assertIn("非 terminal state", optimality)
        self.assertIn("V^*(s_{\\mathrm{terminal}})=0", optimality)

    def test_mdp_keeps_algorithm_and_data_source_bridges(self):
        text = (RL_ROOT / "mdp-value-bellman.md").read_text(encoding="utf-8")
        bridge = text.split("### 算法桥接", maxsplit=1)[1].split(
            "## 训练数据流", maxsplit=1
        )[0]
        for term in (
            "y_t=r_t+\\gamma(1-d_t)\\max_{a'}Q(s_{t+1},a')",
            "真正 terminal",
            "当前 action-value estimate",
            "tabular Q-learning",
            "frozen target network",
            "函数逼近稳定机制",
            "replay buffer",
            "online RL",
            "off-policy RL",
            "offline RL",
            "固定数据集",
            "分布外（OOD）动作",
            "value 外推",
        ):
            self.assertIn(term, bridge)
        self.assertNotIn("Q_{\\mathrm{target}}", bridge)
        for target in (
            "https://arxiv.org/abs/1312.5602",
            "https://doi.org/10.1038/nature14236",
            "https://arxiv.org/abs/2005.01643",
            "q-learning-and-dqn.md",
            "offline-reinforcement-learning.md",
        ):
            self.assertIn(f"]({target})", bridge)

        early_dqn = next(
            paragraph
            for paragraph in bridge.split("\n\n")
            if "https://arxiv.org/abs/1312.5602" in paragraph
        )
        self.assertIn("早期 DQN", early_dqn)
        self.assertIn("experience replay", early_dqn)
        self.assertNotIn("frozen target network", early_dqn)

        nature_dqn = next(
            paragraph
            for paragraph in bridge.split("\n\n")
            if "https://doi.org/10.1038/nature14236" in paragraph
        )
        self.assertIn("2015 Nature DQN", nature_dqn)
        self.assertIn("frozen target network", nature_dqn)

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

    def test_td_updates_each_predecessor_and_n_step_stops_at_terminal(self):
        text = (RL_ROOT / "dynamic-programming-monte-carlo-td.md").read_text(
            encoding="utf-8"
        )
        td_section = text.split("### 5. TD(0)：一步样本加 bootstrap", maxsplit=1)[
            1
        ].split("### 6. n-step return 与 bias-variance", maxsplit=1)[0]
        for term in (
            "每次 transition 都更新 predecessor state",
            "非终止 target 为",
            "r_t+\\gamma V(s_{t+1})",
            "terminal target 为",
            "r_t",
        ):
            self.assertIn(term, td_section)

        n_step_section = text.split(
            "### 6. n-step return 与 bias-variance", maxsplit=1
        )[1].split("### 7. Terminal mask 与 truncation 边界", maxsplit=1)[0]
        for term in (
            "若 \\(n\\) 步内遇到真正 terminal",
            "求和在该转移结束",
            "不添加 bootstrap 项",
        ):
            self.assertIn(term, n_step_section)

    def test_dp_page_links_authoritative_chapters(self):
        text = (RL_ROOT / "dynamic-programming-monte-carlo-td.md").read_text(
            encoding="utf-8"
        )
        next_section = text.split("## 下一步", maxsplit=1)[1]
        self.assertIn(
            "](http://incompleteideas.net/book/the-book-2nd.html)", next_section
        )
        for term in (
            "Sutton & Barto",
            "Chapter 4",
            "Chapter 5",
            "Chapter 6",
            "Dynamic Programming",
            "Monte Carlo Methods",
            "Temporal-Difference Learning",
        ):
            self.assertIn(term, next_section)

    def test_classic_deep_rl_pages_share_contract_and_links(self):
        for filename in self.CLASSIC_PAGES:
            with self.subTest(filename=filename):
                page = RL_ROOT / filename
                self.assertTrue(page.is_file())
                text = page.read_text(encoding="utf-8")
                headings = [
                    line for line in text.splitlines() if line.startswith("## ")
                ]
                self.assert_ordered_subsequence(
                    self.REQUIRED_CHAPTER_HEADINGS, headings
                )
                self.assertIn(
                    "](../assets/examples/reinforcement-learning/deep_rl_smoke.py)",
                    text,
                )
                self.assertIn(
                    "](../assets/examples/reinforcement-learning/README.md)", text
                )
                self.assertIn("smoke", text.lower())
                self.assertIn("收敛", text)
                self.assert_local_links_resolve(page)

    def test_q_learning_and_dqn_core_contract(self):
        text = (RL_ROOT / "q-learning-and-dqn.md").read_text(encoding="utf-8")
        minimum_task = text.split("## 最小任务", maxsplit=1)[1].split(
            "## 核心机制", maxsplit=1
        )[0]
        for command in (
            "python files/assets/examples/reinforcement-learning/gridworld.py --seed 7",
            "python files/assets/examples/reinforcement-learning/deep_rl_smoke.py dqn --seed 7 --steps 256",
        ):
            self.assertIn(command, minimum_task)
        core = text.split("## 核心机制", maxsplit=1)[1].split(
            "## 训练数据流", maxsplit=1
        )[0]
        for term in (
            "SARSA",
            "on-policy",
            "Q-learning",
            "off-policy",
            "Q(s_t,a_t)\\leftarrow",
            "frozen target network",
            "replay buffer",
            "epsilon-greedy",
            "Huber loss",
            "terminated",
        ):
            self.assertIn(term, core)
        for schedule_fact in (
            "epsilon=max(0.1, 1-step/steps)",
            "`step=0`",
            "epsilon=1.0",
            "`--steps`",
            "floor 0.1",
        ):
            self.assertIn(schedule_fact, core)
        self.assertIn(
            "deep_rl_smoke.py dqn --seed 7 --steps 256", text
        )
        self.assertIn("](dynamic-programming-monte-carlo-td.md)", text)
        self.assertIn("](policy-gradient-and-reinforce.md)", text)
        self.assertIn("](https://arxiv.org/abs/1312.5602)", text)
        self.assertIn("](https://doi.org/10.1038/nature14236)", text)

    def test_reinforce_core_contract(self):
        text = (RL_ROOT / "policy-gradient-and-reinforce.md").read_text(
            encoding="utf-8"
        )
        core = text.split("## 核心机制", maxsplit=1)[1].split(
            "## 训练数据流", maxsplit=1
        )[0]
        for term in (
            "score-function estimator",
            "log-prob trick",
            "J(\\theta)",
            "return-to-go",
            "action-independent baseline",
            "方差",
            "entropy",
            "Categorical",
        ):
            self.assertIn(term, core)
        flow = text.split("## 训练数据流", maxsplit=1)[1].split(
            "## 常见失败", maxsplit=1
        )[0]
        for column in ("trajectory", "logp", "return", "loss"):
            self.assertIn(f"| {column} |", flow)
        self.assertIn(
            "deep_rl_smoke.py reinforce --seed 7 --steps 256", text
        )
        self.assertIn("](q-learning-and-dqn.md)", text)
        self.assertIn("](policy-gradient-actor-critic-ppo-grpo.md)", text)
        self.assertIn("](https://doi.org/10.1007/BF00992696)", text)

    def test_actor_critic_gae_ppo_core_contract(self):
        page = RL_ROOT / "policy-gradient-actor-critic-ppo-grpo.md"
        text = page.read_text(encoding="utf-8")
        self.assertTrue(text.startswith("# Actor-Critic、GAE 与 PPO"))
        core = text.split("## 核心机制", maxsplit=1)[1].split(
            "## 训练数据流", maxsplit=1
        )[0]
        for term in (
            "actor",
            "critic",
            "advantage",
            "GAE",
            "terminated",
            "old logprob",
            "probability ratio",
            "clipped surrogate",
            "entropy",
            "KL",
            "minibatch",
            "approximately on-policy",
            "reference policy",
        ):
            self.assertIn(term, core)
        self.assertIn("deep_rl_smoke.py ppo --seed 7 --steps 256", text)
        self.assertNotIn("## GRPO", text)
        self.assertNotIn("## 视觉生成", text)
        self.assertNotIn("## DeepSeek", text)
        self.assertIn("](policy-gradient-and-reinforce.md)", text)
        self.assertIn("](continuous-control-and-sac.md)", text)

    def test_sac_core_contract(self):
        text = (RL_ROOT / "continuous-control-and-sac.md").read_text(
            encoding="utf-8"
        )
        minimum_task = text.split("## 最小任务", maxsplit=1)[1].split(
            "## 核心机制", maxsplit=1
        )[0]
        for interaction_fact in (
            "总 interaction 数",
            "64 条 transition",
            "第 64 条 transition 写入后",
            "256-64+1=193",
            "193 次 optimizer update",
        ):
            self.assertIn(interaction_fact, minimum_task)
        core = text.split("## 核心机制", maxsplit=1)[1].split(
            "## 训练数据流", maxsplit=1
        )[0]
        for term in (
            "DQN",
            "连续动作",
            "reparameterization",
            "squashed Gaussian",
            "tanh Jacobian correction",
            "action scaling",
            "maximum entropy",
            "twin Q",
            "replay buffer",
            "target network",
            "temperature",
            "terminated",
        ):
            self.assertIn(term, core)
        self.assertIn("deep_rl_smoke.py sac --seed 7 --steps 256", text)
        self.assertIn("](policy-gradient-actor-critic-ppo-grpo.md)", text)
        self.assertIn("](rl-for-world-models.md)", text)
        self.assertIn("](https://arxiv.org/abs/1801.01290)", text)

    def test_modern_rl_pages_share_ordered_contract_and_course_routes(self):
        expected_routes = {
            "offline-reinforcement-learning.md": (
                "continuous-control-and-sac.md",
                "rl-for-world-models.md",
            ),
            "rl-for-world-models.md": (
                "offline-reinforcement-learning.md",
                "llm-rlhf-rlvr-and-grpo.md",
            ),
            "llm-rlhf-rlvr-and-grpo.md": (
                "rl-for-world-models.md",
                "vla-robotics-and-closed-loop-rl.md",
            ),
            "vla-robotics-and-closed-loop-rl.md": (
                "llm-rlhf-rlvr-and-grpo.md",
                "verl-code-flow.md",
            ),
        }
        for filename in self.MODERN_PAGES:
            with self.subTest(filename=filename):
                page = RL_ROOT / filename
                self.assertTrue(page.is_file())
                text = page.read_text(encoding="utf-8")
                headings = [line for line in text.splitlines() if line.startswith("## ")]
                self.assert_ordered_subsequence(self.REQUIRED_CHAPTER_HEADINGS, headings)
                previous, following = expected_routes[filename]
                next_section = text.split("## 下一步", maxsplit=1)[1]
                self.assertIn(f"]({previous})", next_section)
                self.assertIn(f"]({following})", next_section)
                self.assert_local_links_resolve(page)

    def test_offline_rl_core_has_support_shift_safety_and_evidence_boundaries(self):
        text = (RL_ROOT / "offline-reinforcement-learning.md").read_text(encoding="utf-8")
        core = text.split("## 核心机制", maxsplit=1)[1].split(
            "## 训练数据流", maxsplit=1
        )[0]
        for term in (
            "behavior cloning",
            "off-policy online RL",
            "fixed dataset",
            "dataset support",
            "distribution shift",
            "OOD action",
            "extrapolation error",
            "value overestimation",
            "behavior constraint",
            "conservative value estimation",
            "BCQ",
            "CQL",
            "IQL",
        ):
            self.assertIn(term, core)
        failures = text.split("## 常见失败", maxsplit=1)[1].split(
            "## 自检与练习", maxsplit=1
        )[0]
        for term in ("offline policy evaluation", "logged data", "real closed-loop", "安全"):
            self.assertIn(term, failures)
        self.assertIn("](https://arxiv.org/abs/2005.01643)", text)

    def test_offline_rl_comparison_and_ope_keep_data_boundaries_distinct(self):
        text = (RL_ROOT / "offline-reinforcement-learning.md").read_text(
            encoding="utf-8"
        )
        minimum_task = text.split("## 最小任务", maxsplit=1)[1].split(
            "## 核心机制", maxsplit=1
        )[0]
        header, rows = self.first_markdown_table(minimum_task)
        self.assertEqual(header, ["方法", "训练时的数据来源", "能否主动纠正数据盲区", "主要风险"])
        self.assertEqual(
            set(rows), {"behavior cloning", "off-policy online RL", "offline RL"}
        )
        behavior = " ".join(rows["behavior cloning"])
        off_policy = " ".join(rows["off-policy online RL"])
        offline = " ".join(rows["offline RL"])
        for term in ("fixed dataset", "不能", "示范"):
            self.assertIn(term, behavior)
        for term in ("replay", "持续环境交互", "能"):
            self.assertIn(term, off_policy)
        for term in ("fixed dataset", "不再交互", "不能", "OOD action"):
            self.assertIn(term, offline)

        ope = text.split("### Offline policy evaluation 的边界", maxsplit=1)[1].split(
            "## 训练数据流", maxsplit=1
        )[0]
        self.assertIn(
            "FQE 等直接价值估计方法会引入函数逼近和 Bellman 外推误差",
            ope,
        )
        self.assertIn(
            "基于 learned dynamics 的 OPE 还会额外引入 model bias", ope
        )
        self.assertNotRegex(ope, r"FQE[^。]*(模型化|model-based)")

    def test_model_based_page_stays_on_planning_and_learned_dynamics(self):
        text = (RL_ROOT / "rl-for-world-models.md").read_text(encoding="utf-8")
        core = text.split("## 核心机制", maxsplit=1)[1].split(
            "## 训练数据流", maxsplit=1
        )[0]
        for term in (
            "known model",
            "learned model",
            "transition",
            "reward",
            "done",
            "Dyna",
            "MPC",
            "PlaNet",
            "Dreamer",
            "latent dynamics",
            "model bias",
            "compounding error",
            "model exploitation",
            "action sensitivity",
            "open-loop",
            "closed-loop return",
        ):
            self.assertIn(term, core)
        self.assertNotIn("## 视频生成里的 RL", text)
        self.assertNotIn("## RLVR-World", text)
        self.assertNotIn("## Decision Transformer", text)

    def test_model_based_evidence_and_migration_ownership_tables(self):
        page = RL_ROOT / "rl-for-world-models.md"
        text = page.read_text(encoding="utf-8")
        evidence = text.split(
            "### Open-loop prediction 与 closed-loop return", maxsplit=1
        )[1].split("## 相邻路线的归属边界", maxsplit=1)[0]
        evidence_header, evidence_rows = self.first_markdown_table(evidence)
        self.assertEqual(evidence_header, ["证据", "测量什么", "不能单独证明什么"])
        self.assertEqual(
            set(evidence_rows),
            {
                "one-step prediction",
                "open-loop rollout",
                "counterfactual action test",
                "model-based closed-loop return",
            },
        )
        self.assertIn("策略会选择正确动作", evidence_rows["open-loop rollout"][1])
        self.assertIn(
            "reward/risk 校准正确", evidence_rows["counterfactual action test"][1]
        )
        self.assertIn(
            "跨环境与真实机器人泛化",
            evidence_rows["model-based closed-loop return"][1],
        )

        ownership = text.split("## 相邻路线的归属边界", maxsplit=1)[1].split(
            "## 训练数据流", maxsplit=1
        )[0]
        ownership_header, ownership_rows = self.first_markdown_table(ownership)
        self.assertEqual(
            ownership_header, ["路线", "课程归属", "优化对象", "证据边界"]
        )
        self.assertEqual(
            list(ownership_rows),
            [
                "RLVR-World",
                "Decision Transformer",
                "DDPO / Flow-GRPO / video generator RL",
                "interactive video worlds",
            ],
        )
        rlvr = " ".join(ownership_rows["RLVR-World"])
        for term in (
            "](rl-for-world-models.md",
            "learned transition/world predictor",
            "verifiable reward",
            "decoded prediction",
            "不能证明 real closed-loop return",
            "](https://arxiv.org/abs/2505.13934)",
        ):
            self.assertIn(term, rlvr)
        decision_transformer = " ".join(ownership_rows["Decision Transformer"])
        for term in (
            "](offline-reinforcement-learning.md)",
            "fixed trajectory",
            "不是 online environment interaction",
            "](https://arxiv.org/abs/2106.01345)",
        ):
            self.assertIn(term, decision_transformer)
        generator_rl = " ".join(
            ownership_rows["DDPO / Flow-GRPO / video generator RL"]
        )
        for term in (
            "](llm-rlhf-rlvr-and-grpo.md)",
            "](../diffusion/video-and-multimodal-diffusion.md)",
            "generator/sampler",
            "不是 learned environment transition",
            "不是 LLM 算法",
            "](https://arxiv.org/abs/2305.13301)",
            "](https://arxiv.org/abs/2505.05470)",
        ):
            self.assertIn(term, generator_rl)
        interactive = " ".join(ownership_rows["interactive video worlds"])
        for term in (
            "](../world-models/evaluation-and-failure-modes.md)",
            "](../vla/closed-loop-recovery-and-failure-analysis.md)",
            "visual playability/open-loop demo",
            "不能证明 real closed-loop",
            "](https://openai.com/index/video-generation-models-as-world-simulators/)",
            "](https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/)",
        ):
            self.assertIn(term, interactive)
        self.assert_local_links_resolve(page)

    def test_published_deep_rl_pages_do_not_keep_stale_exit_claims(self):
        routes = {
            "policy-gradient-actor-critic-ppo-grpo.md": (
                "llm-rlhf-rlvr-and-grpo.md",
            ),
            "continuous-control-and-sac.md": (
                "offline-reinforcement-learning.md",
                "vla-robotics-and-closed-loop-rl.md",
            ),
            "mdp-value-bellman.md": (
                "q-learning-and-dqn.md",
                "offline-reinforcement-learning.md",
            ),
        }
        stale_phrases = ("未来章节", "尚未实现", "没有独立课程页", "还没有独立课程页", "尚无独立课程页")
        for filename, targets in routes.items():
            with self.subTest(filename=filename):
                text = (RL_ROOT / filename).read_text(encoding="utf-8")
                next_section = text.split("## 下一步", maxsplit=1)[1]
                for target in targets:
                    self.assertIn(f"]({target})", next_section)
                for stale in stale_phrases:
                    self.assertNotIn(stale, next_section)

        mdp = (RL_ROOT / "mdp-value-bellman.md").read_text(encoding="utf-8")
        bridge = mdp.split("### 算法桥接：Q-learning 与数据来源", maxsplit=1)[
            1
        ].split("## 训练数据流", maxsplit=1)[0]
        self.assertIn("](q-learning-and-dqn.md)", bridge)
        self.assertIn("](offline-reinforcement-learning.md)", bridge)
        self.assertNotIn("独立课程页出现前", bridge)
        self.assertNotIn("未来专章", bridge)

    def test_llm_rl_page_maps_trajectory_and_objectives_without_blurring_dpo(self):
        page = RL_ROOT / "llm-rlhf-rlvr-and-grpo.md"
        text = page.read_text(encoding="utf-8")
        core = text.split("## 核心机制", maxsplit=1)[1].split(
            "## 训练数据流", maxsplit=1
        )[0]
        for term in (
            "prompt",
            "prefix state",
            "token action",
            "response trajectory",
            "reward model",
            "verifier",
            "old policy",
            "reference policy",
            "RLHF",
            "RLVR",
            "PPO",
            "GRPO",
            "RLOO",
            "zero-variance",
            "DPO",
            "offline preference optimization",
            "pointwise",
            "expected KL",
            "cross-sample",
        ):
            self.assertIn(term, core)
        self.assertIn("](../assets/examples/reinforcement-learning/policy_objectives.py)", text)
        self.assertIn("](https://arxiv.org/abs/1707.06347)", text)
        self.assertIn("](https://arxiv.org/abs/2305.18290)", text)
        self.assertIn("](https://arxiv.org/abs/2501.12948)", text)

    def test_llm_objective_sections_match_runnable_helper_semantics(self):
        text = (RL_ROOT / "llm-rlhf-rlvr-and-grpo.md").read_text(
            encoding="utf-8"
        )
        ppo = text.split("### PPO：旧策略上的 clipped surrogate", maxsplit=1)[
            1
        ].split("### GRPO：同一 prompt 内做相对归一化", maxsplit=1)[0]
        ppo_normalized = self.normalize_formula_text(ppo)
        ratio_formula = (
            r"r_t(\theta)=\exp\left(\log\pi_\theta(a_t\mids_t)-"
            r"\log\pi_{\text{old}}(a_t\mids_t)\right)"
        )
        reversed_ratio_formula = (
            r"r_t(\theta)=\exp\left(\log\pi_{\text{old}}(a_t\mids_t)-"
            r"\log\pi_\theta(a_t\mids_t)\right)"
        )
        clipped_formula = (
            r"L^{\text{clip}}(\theta)=\mathbbE\left[\min\left(r_tA_t,"
            r"\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t\right)\right]"
        )
        self.assertIn(ratio_formula, ppo_normalized)
        self.assertNotIn(reversed_ratio_formula, ppo_normalized)
        self.assertIn("new_logp-old_logp", ppo_normalized)
        self.assertNotIn("old_logp-new_logp", ppo_normalized)
        self.assertIn(
            "分子策略是当前policy，分母策略是采样这批response的oldpolicy",
            ppo_normalized,
        )
        self.assertIn(clipped_formula, ppo_normalized)
        self.assertIn("clipped_policy_loss返回这个surrogate的负均值", ppo_normalized)
        self.assertIn("作为loss交给optimizer最小化", ppo_normalized)
        self.assertIn("L^{\\text{clip}}写的是要最大化的正surrogate", ppo_normalized)
        self.assertLess(
            ppo_normalized.index(ratio_formula),
            ppo_normalized.index(clipped_formula),
        )

        group = text.split("### GRPO：同一 prompt 内做相对归一化", maxsplit=1)[1].split(
            "### RLOO 的边界", maxsplit=1
        )[0]
        group_normalized = self.normalize_formula_text(group)
        group_formula = (
            r"A_i=\begin{cases}\dfrac{R_i-\barR}{\sigma_R+\epsilon},"
            r"&\sigma_R>\epsilon,\\0,&\sigma_R\le\epsilon.\end{cases}"
        )
        self.assertIn(group_formula, group_normalized)
        self.assertNotIn(r"\barR-R_i", group_normalized)
        self.assertIn("同一prompt", group_normalized)
        self.assertIn("lastaxis当作同promptgroup", group_normalized)
        self.assertIn("先减去组内均值，再按总体标准差", group_normalized)
        self.assertIn("zero-variancegroup", group_normalized)
        self.assertIn("返回精确零", group_normalized)

        sampled_kl = text.split(
            "### Pointwise log-ratio 不等于 full KL", maxsplit=1
        )[1].split("### DPO 是离线偏好优化对照", maxsplit=1)[0]
        sampled_kl_normalized = self.normalize_formula_text(sampled_kl)
        sampled_formula = (
            r"k_t=\log\pi_\theta(a_t\mids_t)-"
            r"\log\pi_{\text{ref}}(a_t\mids_t)"
        )
        reversed_sampled_formula = (
            r"k_t=\log\pi_{\text{ref}}(a_t\mids_t)-"
            r"\log\pi_\theta(a_t\mids_t)"
        )
        self.assertIn(sampled_formula, sampled_kl_normalized)
        self.assertNotIn(reversed_sampled_formula, sampled_kl_normalized)
        self.assertIn("pointwisesampledlog-ratioestimator", sampled_kl_normalized)
        self.assertIn("不是fullKL", sampled_kl_normalized)
        self.assertIn("expectedKL", sampled_kl_normalized)

        self.assertIn(
            "](../assets/examples/reinforcement-learning/policy_objectives.py)", text
        )

        dpo = text.split("### DPO 是离线偏好优化对照", maxsplit=1)[1].split(
            "## 训练数据流", maxsplit=1
        )[0]
        for term in (
            "offline preference optimization",
            "固定的 chosen/rejected preference pair",
            "不向在线环境采新 response",
            "在线 PPO rollout",
            "固定偏好数据上的离线优化",
        ):
            self.assertIn(term, dpo)

    def test_vla_page_separates_prediction_from_sim_and_real_closed_loop_evidence(self):
        text = (RL_ROOT / "vla-robotics-and-closed-loop-rl.md").read_text(
            encoding="utf-8"
        )
        core = text.split("## 核心机制", maxsplit=1)[1].split(
            "## 训练数据流", maxsplit=1
        )[0]
        for term in (
            "POMDP",
            "belief",
            "continuous action",
            "action chunk",
            "demonstration",
            "offline pretraining",
            "simulation",
            "world-model rollout",
            "real deployment data",
            "safety constraint",
            "recovery",
            "distribution shift",
            "reward design",
            "imitation success",
            "human intervention",
        ):
            self.assertIn(term, core)
        for evidence in (
            "open-loop action prediction",
            "simulator closed-loop success",
            "real closed-loop success",
            "能证明什么",
        ):
            self.assertIn(evidence, text)

    def test_vla_evidence_table_keeps_cannot_prove_boundaries(self):
        text = (RL_ROOT / "vla-robotics-and-closed-loop-rl.md").read_text(
            encoding="utf-8"
        )
        minimum_task = text.split("## 最小任务", maxsplit=1)[1].split(
            "## 核心机制", maxsplit=1
        )[0]
        header, rows = self.first_markdown_table(minimum_task)
        self.assertEqual(header, ["评测层", "指标例子", "能证明什么", "不能证明什么"])
        self.assertEqual(
            list(rows),
            [
                "open-loop action prediction",
                "simulator closed-loop success",
                "real closed-loop success",
            ],
        )
        self.assertIn("policy 能纠错", rows["open-loop action prediction"][2])
        self.assertIn("任务会成功", rows["open-loop action prediction"][2])
        self.assertIn("真实摩擦", rows["simulator closed-loop success"][2])
        self.assertIn("长尾安全", rows["simulator closed-loop success"][2])
        self.assertIn("未测试场景", rows["real closed-loop success"][2])
        self.assertIn("普遍泛化", rows["real closed-loop success"][2])

    def test_verl_has_field_level_flow_worker_boundaries_and_system_checks(self):
        page = RL_ROOT / "verl-code-flow.md"
        text = page.read_text(encoding="utf-8")
        minimum_task = self.h2_section(text, "最小任务")
        for term in ("prompt batch", "distributed rollout", "distributed update"):
            self.assertIn(term, minimum_task)

        core = self.h2_section(text, "核心机制")
        workers = core.split("### Worker 角色与边界", maxsplit=1)[1]
        worker_header, worker_rows = self.first_markdown_table(workers)
        self.assertEqual(worker_header, ["角色", "职责", "产物或边界"])
        self.assertEqual(
            list(worker_rows),
            [
                "driver / controller",
                "actor / rollout",
                "critic（可选）",
                "reward",
                "reference",
                "trainer",
            ],
        )
        self.assertIn("placement", " ".join(worker_rows["driver / controller"]))
        self.assertIn("communication", " ".join(worker_rows["driver / controller"]))
        self.assertIn("不更新", " ".join(worker_rows["reference"]))
        self.assertIn("PPO", " ".join(worker_rows["critic（可选）"]))

        flow = self.h2_section(text, "训练数据流")
        flow_header, flow_rows = self.first_markdown_table(flow)
        self.assertEqual(flow_header, ["阶段", "输入字段", "输出字段", "首要边界"])
        self.assertEqual(
            list(flow_rows),
            [
                "prompt batch",
                "rollout",
                "reward",
                "PPO estimates",
                "advantage",
                "minibatch",
                "optimizer",
                "weight sync",
            ],
        )
        expected_fields = {
            "prompt batch": ("prompt", "input_ids", "attention_mask"),
            "rollout": ("response", "action_mask"),
            "reward": ("reward",),
            "PPO estimates": ("old_logp", "reference_logp", "value"),
            "advantage": ("advantage", "returns"),
            "minibatch": ("minibatch",),
            "optimizer": ("optimizer",),
            "weight sync": ("weight sync",),
        }
        for row, fields in expected_fields.items():
            joined = " ".join(flow_rows[row])
            for field in fields:
                self.assertIn(field, joined)
        for term in (
            "terminated",
            "truncated",
            "response_mask",
            "action_mask",
            "padding",
            "token-level",
            "sequence-level",
            "policy_version_id",
            "rollout_version_id",
            "staleness",
            "prefix/KV reuse",
            "throughput",
            "freshness",
            "reproducibility metadata",
        ):
            self.assertIn(term, text)
        old_reference = core.split("### Old policy 不等于 reference policy", maxsplit=1)[1]
        self.assertIn("PPO ratio", old_reference)
        self.assertIn("KL", old_reference)
        self.assertIn("不是同一个角色", old_reference)
        failures = self.h2_section(text, "常见失败")
        for term in (
            "stale logp",
            "wrong mask",
            "reward/logp misalignment",
            "reference/old confusion",
            "duplicate samples",
            "nondeterministic serving",
        ):
            self.assertIn(f"| {term} |", failures)
        for source in (
            "https://github.com/verl-project/verl",
            "https://verl.readthedocs.io/en/latest/workers/ray_trainer.html",
            "https://arxiv.org/abs/2409.19256",
            "https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo",
            "https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html",
            "https://arxiv.org/abs/1707.06347",
            "https://arxiv.org/abs/1801.01290",
        ):
            self.assertIn(f"]({source})", text)
        self.assertIn("](practice-and-algorithm-selection.md)", self.h2_section(text, "下一步"))
        self.assert_local_links_resolve(page)

    def test_capstone_has_exact_algorithm_decision_table(self):
        page = RL_ROOT / "practice-and-algorithm-selection.md"
        text = page.read_text(encoding="utf-8")
        core = self.h2_section(text, "核心机制")
        decision = core.split("### 算法选择表", maxsplit=1)[1]
        header, rows = self.first_markdown_table(decision)
        self.assertEqual(header, ["算法", "适用条件", "数据", "动作", "主要风险", "首个诊断"])
        self.assertEqual(
            list(rows),
            ["Bandit", "DQN", "PPO", "SAC", "Offline RL", "Model-based RL", "GRPO"],
        )
        for row in rows.values():
            self.assertTrue(all(cell for cell in row))
        for axis in (
            "observation/state/POMDP",
            "discrete / continuous action",
            "online interaction",
            "fixed dataset",
            "reward / verifier",
            "learned model",
            "calibration",
        ):
            self.assertIn(axis, core)

    def test_capstone_defines_three_executable_projects_and_acceptance(self):
        page = RL_ROOT / "practice-and-algorithm-selection.md"
        text = page.read_text(encoding="utf-8")
        projects = self.h2_section(text, "最小任务")
        expected = (
            (
                "### Capstone 1：GridWorld 中比较 DP、TD 与 Q-learning",
                "python files/assets/examples/reinforcement-learning/gridworld.py --seed 7",
                ("输入", "步骤", "预期产物", "验收标准", "失败诊断", "terminal mask"),
            ),
            (
                "### Capstone 2：Pendulum 中比较 SAC 与 PPO",
                "python files/assets/examples/reinforcement-learning/deep_rl_smoke.py sac --seed 7 --steps 256",
                ("输入", "步骤", "预期产物", "验收标准", "失败诊断", "actor", "critic", "alpha", "return", "smoke 不是收敛"),
            ),
            (
                "### Capstone 3：Verifier LLM 的组内策略目标",
                "python files/assets/examples/reinforcement-learning/policy_objectives.py --seed 7",
                ("输入", "步骤", "预期产物", "验收标准", "失败诊断", "group advantage", "zero group", "sampled KL", "clipped loss", "reward hacking"),
            ),
        )
        for index, (heading, command, terms) in enumerate(expected):
            self.assertIn(heading, projects)
            section = projects.split(heading, maxsplit=1)[1]
            if index + 1 < len(expected):
                section = section.split(expected[index + 1][0], maxsplit=1)[0]
            self.assertIn(command, section)
            for term in terms:
                self.assertIn(term, section)
        self.assert_local_links_resolve(page)

    def test_index_marks_modern_application_route_readable_and_linked(self):
        text = (RL_ROOT / "index.md").read_text(encoding="utf-8")
        course_map = text.split("## 完整学习路线", maxsplit=1)[1].split(
            "## 按方向进入课程", maxsplit=1
        )[0]
        expected_rows = (
            "| 低成本与模型驱动 | 10. [Offline Reinforcement Learning](offline-reinforcement-learning.md) | 可阅读 |",
            "| 低成本与模型驱动 | 11. [Model-based RL 与世界模型](rl-for-world-models.md) | 可阅读 |",
            "| 大模型与具身系统 | 12. [LLM 强化学习：RLHF、RLVR 与 GRPO](llm-rlhf-rlvr-and-grpo.md) | 可阅读 |",
            "| 大模型与具身系统 | 13. [VLA、机器人与闭环强化学习](vla-robotics-and-closed-loop-rl.md) | 可阅读 |",
        )
        for row in expected_rows:
            self.assertIn(row, course_map)
        self.assertIn(
            "[连续控制与 SAC](continuous-control-and-sac.md) -> "
            "[Offline RL](offline-reinforcement-learning.md) -> "
            "[Model-based RL 与世界模型](rl-for-world-models.md) -> "
            "[LLM 强化学习](llm-rlhf-rlvr-and-grpo.md) -> "
            "[VLA 与机器人闭环](vla-robotics-and-closed-loop-rl.md) -> "
            "[verl](verl-code-flow.md)",
            text,
        )

    def test_nav_uses_final_grouped_course_order(self):
        nav = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
        start = nav.index("  - 强化学习:")
        end = nav.index("  - 世界模型与具身智能:", start)
        section = nav[start:end]
        stage_labels = []
        pages = []
        for line in section.splitlines()[1:]:
            page_match = re.fullmatch(
                r"\s+-\s+[^:]+:\s+(reinforcement-learning/[^\s]+\.md)", line
            )
            if page_match:
                pages.append(page_match.group(1))
                continue
            stage_match = re.fullmatch(r"      - ([^:]+):", line)
            if stage_match:
                stage_labels.append(stage_match.group(1))
        self.assertEqual(
            stage_labels,
            [
                "建立决策语言",
                "经典深度强化学习",
                "低成本与模型驱动学习",
                "大模型、机器人与系统",
            ],
        )
        self.assertEqual(
            pages,
            [
                "reinforcement-learning/index.md",
                "reinforcement-learning/rl-agent-environment-loop.md",
                "reinforcement-learning/multi-armed-bandits-and-exploration.md",
                "reinforcement-learning/mdp-value-bellman.md",
                "reinforcement-learning/dynamic-programming-monte-carlo-td.md",
                "reinforcement-learning/q-learning-and-dqn.md",
                "reinforcement-learning/policy-gradient-and-reinforce.md",
                "reinforcement-learning/policy-gradient-actor-critic-ppo-grpo.md",
                "reinforcement-learning/continuous-control-and-sac.md",
                "reinforcement-learning/offline-reinforcement-learning.md",
                "reinforcement-learning/rl-for-world-models.md",
                "reinforcement-learning/llm-rlhf-rlvr-and-grpo.md",
                "reinforcement-learning/vla-robotics-and-closed-loop-rl.md",
                "reinforcement-learning/verl-code-flow.md",
                "reinforcement-learning/practice-and-algorithm-selection.md",
            ],
        )
