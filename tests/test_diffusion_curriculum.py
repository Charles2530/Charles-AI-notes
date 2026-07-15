import re
from pathlib import Path
from unittest import TestCase


ROOT = Path(__file__).resolve().parents[1]
DIFFUSION_ROOT = ROOT / "files" / "diffusion"


class DiffusionCurriculumTests(TestCase):
    CORE_PAGES = (
        "foundations-and-core-intuition.md",
        "training.md",
        "architecture-and-stable-diffusion.md",
        "minimal-ddpm-implementation.md",
        "flow-matching-and-rectified-flow.md",
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
        local_targets = []
        for target in targets:
            target = target.split("#", 1)[0].split("?", 1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            path = (page.parent / target).resolve()
            local_targets.append(path)
            self.assertTrue(path.exists(), f"{page.name}: {target}")
        self.assertGreaterEqual(len(set(local_targets)), 2, page.name)

    def test_index_exposes_the_complete_diffusion_lineage(self):
        text = (DIFFUSION_ROOT / "index.md").read_text(encoding="utf-8")
        headings = [line for line in text.splitlines() if line.startswith("## ")]
        self.assertEqual(
            headings,
            [
                "## 扩散模型是什么",
                "## 训练与生成是两条不同的执行链",
                "## 核心技术脉络",
                "## 方法分支与工程问题",
                "## 按问题进入",
            ],
        )

    def test_core_pages_exist_and_follow_a_common_contract(self):
        for filename in self.CORE_PAGES:
            with self.subTest(filename=filename):
                page = DIFFUSION_ROOT / filename
                self.assertTrue(page.is_file(), filename)
                headings = [
                    line
                    for line in page.read_text(encoding="utf-8").splitlines()
                    if line.startswith("## ")
                ]
                self.assert_ordered_subsequence(self.CORE_HEADINGS, headings)
                self.assert_local_links_resolve(page)

    def test_new_core_pages_do_not_label_the_reader(self):
        forbidden = ("初学者", "新手", "零基础", "小白")
        for filename in self.CORE_PAGES:
            page = DIFFUSION_ROOT / filename
            if not page.exists():
                continue
            text = page.read_text(encoding="utf-8")
            for phrase in forbidden:
                self.assertNotIn(phrase, text, f"{filename}: {phrase}")
            self.assertNotRegex(text, r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

    def test_core_pages_cover_the_required_mathematical_and_system_interfaces(self):
        expected_terms = {
            "training.md": (
                r"q(x_t\mid x_0)",
                r"q(x_{t-1}\mid x_t,x_0)",
                r"\mathcal L_{\text{simple}}",
                "## DDPM 训练算法",
                "## DDPM 采样算法",
            ),
            "architecture-and-stable-diffusion.md": (
                "U-Net",
                "DiT",
                "VAE encoder",
                "text encoder",
                "scheduler / solver",
                "%% stable-diffusion-training-inference",
            ),
            "flow-matching-and-rectified-flow.md": (
                "conditional velocity",
                "marginal velocity",
                "continuity equation",
                "DDPM \\(v\\)-prediction",
                "reflow",
                "%% diffusion-method-coordinate-map",
            ),
            "minimal-ddpm-implementation.md": (
                "make_linear_schedule",
                "q_sample",
                "diffusion_loss",
                "p_sample",
                "sample_loop",
                "../assets/examples/diffusion/minimal_ddpm.py",
            ),
        }
        for filename, terms in expected_terms.items():
            text = (DIFFUSION_ROOT / filename).read_text(encoding="utf-8")
            for term in terms:
                with self.subTest(filename=filename, term=term):
                    self.assertIn(term, text)

    def test_navigation_starts_with_index_and_contains_the_full_core_path(self):
        nav = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
        start = nav.index("  - 扩散与生成模型:")
        end = nav.index("  - 思考探索:", start)
        section = nav[start:end]
        paths = re.findall(r"(diffusion/[^\s]+\.md)", section)
        self.assertEqual(
            paths[:5],
            [
                "diffusion/index.md",
                "diffusion/foundations-and-core-intuition.md",
                "diffusion/training.md",
                "diffusion/architecture-and-stable-diffusion.md",
                "diffusion/minimal-ddpm-implementation.md",
            ],
        )
        nav_paths = set(paths)
        diffusion_pages = {
            f"diffusion/{page.name}" for page in DIFFUSION_ROOT.glob("*.md")
        }
        self.assertEqual(nav_paths, diffusion_pages)
