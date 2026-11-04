"""Integration links for the training/serving reference, independent of phrasing."""
from pathlib import Path
from unittest import TestCase

ROOT = Path(__file__).resolve().parents[1] / 'files'


class TrainingInferenceHandbookTests(TestCase):
    def test_system_overview_connects_both_execution_paths(self):
        text = (ROOT / 'training-and-inference-systems/index.md').read_text(encoding='utf-8')
        for target in (
            '../training/beginner-model-training-primer.md',
            '../training/objectives-optimizers-and-schedules.md',
            '../inference/serving-systems.md',
            '../inference/cost-modeling-and-slo-design.md',
            '../quantization/index.md', '../operators/index.md',
        ):
            with self.subTest(target=target):
                self.assertIn(f']({target})', text)
        self.assertIn('```mermaid', text)

    def test_topic_indexes_keep_their_articles_discoverable(self):
        for topic in ('training', 'inference'):
            text = (ROOT / topic / 'index.md').read_text(encoding='utf-8')
            for page in (ROOT / topic).glob('*.md'):
                if page.name in {'index.md', 'image-sources.md'}:
                    continue
                with self.subTest(page=str(page)):
                    self.assertIn(f']({page.name})', text)
