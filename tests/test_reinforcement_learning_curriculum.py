"""Keep algorithms and runnable examples discoverable without fixing prose headings.

Numerical, masking and gradient contracts live in
`test_reinforcement_learning_examples.py`.
"""
from pathlib import Path
from unittest import TestCase
import re

ROOT = Path(__file__).resolve().parents[1]
RL_ROOT = ROOT / 'files/reinforcement-learning'


class ReinforcementLearningHandbookTests(TestCase):
    def test_index_links_every_algorithm_article(self):
        text = (RL_ROOT / 'index.md').read_text(encoding='utf-8')
        for page in RL_ROOT.glob('*.md'):
            if page.name != 'index.md':
                with self.subTest(page=page.name):
                    self.assertIn(f']({page.name})', text)
        self.assertIn('{#完整学习路线}', text)  # Preserve existing inbound links.

    def test_executable_examples_are_linked_from_their_explanations(self):
        examples = {
            'multi-armed-bandits-and-exploration.md': 'bandit.py',
            'mdp-value-bellman.md': 'gridworld.py',
            'dynamic-programming-monte-carlo-td.md': 'gridworld.py',
            'q-learning-and-dqn.md': 'deep_rl_smoke.py',
            'policy-gradient-and-reinforce.md': 'deep_rl_smoke.py',
            'policy-gradient-actor-critic-ppo-grpo.md': 'deep_rl_smoke.py',
            'continuous-control-and-sac.md': 'deep_rl_smoke.py',
            'llm-rlhf-rlvr-and-grpo.md': 'policy_objectives.py',
        }
        for page, example in examples.items():
            with self.subTest(page=page):
                self.assertIn(
                    f'](../assets/examples/reinforcement-learning/{example})',
                    (RL_ROOT / page).read_text(encoding='utf-8'),
                )
                self.assertTrue((ROOT / 'files/assets/examples/reinforcement-learning' / example).is_file())

    def test_verl_source_references_pin_the_implementation_under_discussion(self):
        text = (RL_ROOT / 'verl-code-flow.md').read_text(encoding='utf-8')
        refs = re.findall(r'https://github.com/verl-project/verl/blob/([^/]+)/([^\s)]+)', text)
        self.assertTrue(refs)
        self.assertEqual({ref for ref, _ in refs}, {'e5687fce0516d31e1fdc4580499074a9bd94c751'})
        self.assertTrue({'ray_trainer.py', 'core_algos.py', 'rollout_corr_helper.py', 'agent_loop.py'}.issubset(
            {Path(path.split('#')[0]).name for _, path in refs}
        ))

    def test_runtime_instructions_remain_available(self):
        for page in ('index.md', 'practice-and-algorithm-selection.md'):
            self.assertIn('](../assets/examples/reinforcement-learning/README.md)',
                          (RL_ROOT / page).read_text(encoding='utf-8'))
