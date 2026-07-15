import re
from pathlib import Path
from unittest import TestCase


ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = ROOT / "files"


class WorldModelsEmbodiedCurriculumTests(TestCase):
    CORE_PAGES = (
        "world-models/beginner-world-models-primer.md",
        "embodied-ai/beginner-learning-path.md",
    )
    CORE_HEADINGS = (
        "## 本文解决什么问题",
        "## 前置知识",
        "## 核心机制",
        "## 常见误解与失效边界",
        "## 检查理解",
        "## 相关阅读与下一步",
    )

    def assert_ordered_subsequence(self, required, actual):
        remaining = iter(actual)
        for item in required:
            self.assertIn(item, remaining)

    def assert_local_links_resolve(self, page):
        text = page.read_text(encoding="utf-8")
        targets = re.findall(r"!?\[[^]]*\]\(([^)]+)\)", text)
        local_targets = set()
        for target in targets:
            target = target.split("#", 1)[0].split("?", 1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            path = (page.parent / target).resolve()
            local_targets.add(path)
            self.assertTrue(path.exists(), f"{page}: {target}")
        self.assertGreaterEqual(len(local_targets), 2, str(page))

    def assert_teaching_page(self, relative, required_terms):
        page = DOCS_ROOT / relative
        self.assertTrue(page.is_file(), str(page))
        text = page.read_text(encoding="utf-8")
        headings = [line for line in text.splitlines() if line.startswith("## ")]
        self.assert_ordered_subsequence(
            (
                "## 本章回答什么",
                "## 前置知识",
                "## 核心机制",
                "## 常见错误与失效边界",
                "## 检查理解",
                "## 相关阅读与下一步",
            ),
            headings,
        )
        for term in required_terms:
            self.assertIn(term, text)
        self.assertNotRegex(text, r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
        self.assert_local_links_resolve(page)

    def test_coordinate_transforms_page_covers_hand_eye_and_action_frames(self):
        self.assert_teaching_page(
            "embodied-ai/coordinate-transforms-and-hand-eye-calibration.md",
            (
                r"{}^{A}T_B",
                "eye-in-hand",
                "eye-to-hand",
                "AX=XB",
                "body-frame",
                "world-frame",
                "np.linalg.inv",
                "工程检查表",
            ),
        )

    def test_robot_kinematics_page_covers_dh_fk_and_failure_boundaries(self):
        self.assert_teaching_page(
            "embodied-ai/robot-kinematics-and-dh-parameters.md",
            (
                "关节空间",
                "任务空间",
                "Forward Kinematics",
                "Denavit--Hartenberg",
                r"a_i",
                r"\alpha_i",
                r"d_i",
                r"\theta_i",
                "revolute",
                "prismatic",
                "奇异",
                "工程检查表",
            ),
        )

    def test_unified_index_exposes_model_policy_and_system_layers(self):
        page = DOCS_ROOT / "world-models-and-embodied-ai" / "index.md"
        self.assertTrue(page.is_file())
        text = page.read_text(encoding="utf-8")
        headings = [line for line in text.splitlines() if line.startswith("## ")]
        self.assertEqual(
            headings,
            [
                "## 世界模型与具身智能是什么关系",
                "## 一条从观测到行动再到学习的闭环",
                "## 世界模型技术脉络",
                "## 具身智能系统脉络",
                "## VLA 作为策略接口",
                "## 按问题进入",
            ],
        )
        self.assertIn("%% embodied-world-model-loop", text)
        self.assert_local_links_resolve(page)

    def test_topic_indexes_define_complete_reading_paths(self):
        expected = {
            "world-models/index.md": (
                "## 世界模型的统一接口",
                "## 三类建模对象",
                "## 从预测到决策",
                "## 应用与工程",
                "## 按问题进入",
            ),
            "embodied-ai/index.md": (
                "## 具身系统的统一接口",
                "## 系统闭环",
                "## 感知与状态",
                "## 策略、规划与控制",
                "## 数据、评测与部署",
                "## 按问题进入",
            ),
            "vla/index.md": (
                "## VLA 的统一接口",
                "## 训练与执行是两条链",
                "## 动作表示与策略学习",
                "## 闭环、评测与部署",
                "## 与世界模型的关系",
            ),
        }
        for relative, headings in expected.items():
            page = DOCS_ROOT / relative
            actual = [
                line
                for line in page.read_text(encoding="utf-8").splitlines()
                if line.startswith("## ")
            ]
            with self.subTest(page=relative):
                self.assertEqual(actual, list(headings))
                self.assert_local_links_resolve(page)

    def test_foundation_pages_follow_the_shared_contract_without_reader_labels(self):
        forbidden = ("初学者", "新手", "零基础", "小白", "从零")
        for relative in self.CORE_PAGES:
            page = DOCS_ROOT / relative
            text = page.read_text(encoding="utf-8")
            headings = [line for line in text.splitlines() if line.startswith("## ")]
            with self.subTest(page=relative):
                self.assert_ordered_subsequence(self.CORE_HEADINGS, headings)
                for phrase in forbidden:
                    self.assertNotIn(phrase, text)
                self.assertNotRegex(text, r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
                self.assert_local_links_resolve(page)

    def test_core_pages_cover_state_prediction_policy_and_execution_interfaces(self):
        expected_terms = {
            "world-models/beginner-world-models-primer.md": (
                "belief",
                r"p_\theta(z_{t+1}\mid z_t,a_t)",
                "aleatoric uncertainty",
                "epistemic uncertainty",
                "posterior training",
                "prior rollout",
            ),
            "embodied-ai/beginner-learning-path.md": (
                "operational design domain",
                "policy action",
                "planner",
                "controller",
                "safety layer",
                "commanded action",
                "executed action",
            ),
            "vla/index.md": (
                "%% vla-training-execution",
                r"\pi_\theta",
                "action token",
                "动作块",
                "闭环状态分布",
                r"p(z_{t+1}\mid z_t,a_t)",
            ),
        }
        for relative, terms in expected_terms.items():
            text = (DOCS_ROOT / relative).read_text(encoding="utf-8")
            for term in terms:
                with self.subTest(page=relative, term=term):
                    self.assertIn(term, text)

    def test_navigation_exposes_every_topic_page_in_grouped_order(self):
        nav = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
        start = nav.index("  - 世界模型与具身智能:")
        end = nav.index("  - 扩散与生成模型:", start)
        section = nav[start:end]
        paths = re.findall(
            r"((?:world-models-and-embodied-ai|world-models|embodied-ai|vla)/[^\s]+\.md)",
            section,
        )
        embodied_sequence = [
            "embodied-ai/cameras-depth-and-robot-vision.md",
            "embodied-ai/coordinate-transforms-and-hand-eye-calibration.md",
            "embodied-ai/robot-kinematics-and-dh-parameters.md",
            "embodied-ai/stereo-matching-and-cost-volume.md",
        ]
        self.assert_ordered_subsequence(embodied_sequence, paths)
        self.assertEqual(
            paths[:6],
            [
                "world-models-and-embodied-ai/index.md",
                "world-models/index.md",
                "world-models/beginner-world-models-primer.md",
                "world-models/rssm-dreamer-and-planning.md",
                "world-models/masked-jepa-and-latent-prediction.md",
                "world-models/generative-simulation-and-video.md",
            ],
        )
        expected = {"world-models-and-embodied-ai/index.md"}
        for directory in ("world-models", "embodied-ai", "vla"):
            expected.update(
                f"{directory}/{page.name}"
                for page in (DOCS_ROOT / directory).glob("*.md")
            )
        self.assertEqual(set(paths), expected)
