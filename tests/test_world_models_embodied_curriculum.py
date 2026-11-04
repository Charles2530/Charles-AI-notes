"""Preserve cross-topic access without requiring a shared course template."""
from pathlib import Path
from unittest import TestCase

ROOT = Path(__file__).resolve().parents[1] / 'files'


class WorldModelsEmbodiedHandbookTests(TestCase):
    def test_overview_connects_perception_prediction_policy_and_execution(self):
        text = (ROOT / 'world-models-and-embodied-ai/index.md').read_text(encoding='utf-8')
        for target in (
            '../vlm/index.md', '../world-models/beginner-world-models-primer.md',
            '../world-models/rssm-dreamer-and-planning.md', '../vla/index.md',
            '../embodied-ai/index.md', '../vla/action-representation-and-interfaces.md',
            '../prerequisite-math/geometry-vision-and-3d.md',
        ):
            with self.subTest(target=target):
                self.assertIn(f']({target})', text)
        self.assertIn('```mermaid', text)

    def test_topic_indexes_link_their_core_mechanisms(self):
        for topic, targets in {
            'world-models': ('beginner-world-models-primer.md', 'rssm-dreamer-and-planning.md',
                             'evaluation-and-failure-modes.md'),
            'embodied-ai': ('beginner-learning-path.md', 'coordinate-transforms-and-hand-eye-calibration.md',
                            'robot-kinematics-and-dh-parameters.md'),
            'vla': ('action-representation-and-interfaces.md', 'data-and-policy-learning.md',
                    'closed-loop-recovery-and-failure-analysis.md'),
        }.items():
            text = (ROOT / topic / 'index.md').read_text(encoding='utf-8')
            for target in targets:
                with self.subTest(topic=topic, target=target):
                    self.assertIn(f']({target})', text)
