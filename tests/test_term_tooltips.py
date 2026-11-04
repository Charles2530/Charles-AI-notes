"""Exercise the production decorator on terms with conflicting paper meanings."""

import json
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "files/assets/javascripts/term-tooltips.js"

# Only the DOM output sink is replaced. Matching, tip/link selection and limits
# all run through the production decorateTextNode function.
HARNESS = r"""
const fs = require('node:fs');
const vm = require('node:vm');
const payload = JSON.parse(fs.readFileSync(0, 'utf8'));
const source = fs.readFileSync(process.argv[1], 'utf8').replace(
  '  if (window.document$) {',
  '  globalThis.decorate = decorateTextNode;\n  if (window.document$) {'
);
const document = {
  getElementById() { return null; },
  createDocumentFragment() { return {children: [], appendChild(x) {this.children.push(x);}}; },
  createTextNode(text) { return {textContent: text}; },
  createElement(tag) { return {tag, dataset: {}, setAttribute() {}}; }
};
const context = {
  document, URL,
  window: {location: {href: 'https://example.test/paper/'}, document$: {subscribe() {}}}
};
vm.runInNewContext(source, context);
const results = payload.map(text => {
  let output = [{textContent: text}];
  const node = {nodeValue: text, parentNode: {replaceChild(fragment) {output = fragment.children;}}};
  context.decorate(node, new Map(), new Set());
  return {
    text: output.map(x => x.textContent).join(''),
    tips: output.filter(x => x.dataset).map(x => ({term: x.textContent, tip: x.dataset.tip, href: x.href || ''}))
  };
});
process.stdout.write(JSON.stringify(results));
"""


def decorate(texts):
    result = subprocess.run(
        ["node", "-e", HARNESS, str(SCRIPT)],
        input=json.dumps(texts), text=True, capture_output=True, check=True,
    )
    return json.loads(result.stdout)


class TermTooltipTests(unittest.TestCase):
    def test_ambiguous_paper_terms_do_not_receive_unrelated_definitions(self):
        cases = [
            ("专家通信使用 EP 组 16。", "EP", "ONNX"),
            ("sparse mixture-of-experts Transformer", "mixture", "Data Mixture"),
            ("FACTS Grounding (Adjusted)", "Grounding", "图像区域"),
            ("Critic 按 rubric 评价回答。", "Critic", "未来回报"),
            ("EN text-to-text Policy Violations", "Policy", "强化学习"),
            ("混入精选样本的 PTX loss。", "PTX", "GPU"),
            ("Agentless-Single-Patch", "Patch", "图像"),
            ("single patch without test", "patch", "图像"),
            ("KernelBench L1", "L1", "回归损失"),
            ("SLA 的稀疏与线性 attention 分支", "SLA", "服务可靠性"),
            ("L2", "L2", "GPU"),
            ("Seed-OSS-36B-Instruct", "Seed", "随机种子"),
            ("Euler angle 约定", "Euler", "数值积分"),
            ("回放 5%–10% 的前阶段数据", "回放", "测试新模型"),
        ]
        for (text, term, wrong_meaning), result in zip(cases, decorate([x[0] for x in cases])):
            with self.subTest(text=text):
                self.assertEqual(result["text"], text)
                self.assertFalse(any(t["term"] == term and wrong_meaning in t["tip"] for t in result["tips"]), result)

    def test_specific_terms_keep_useful_definitions(self):
        cases = [
            ("Execution Provider", "ONNX"),
            ("Sparse-Linear Attention", "稀疏"),
            ("Service Level Agreement", "服务"),
            ("Expert Parallelism", "专家"),
            ("mixture-of-experts", "专家"),
            ("data mixture", "采样配比"),
            ("visual grounding", "图像区域"),
            ("Actor-Critic", "未来回报"),
            ("behavior policy", "策略"),
            ("PTX loss", "预训练"),
            ("PTX assembly", "GPU"),
            ("image patch", "图像"),
            ("L1 loss", "绝对误差"),
            ("L2 cache", "缓存"),
            ("random seed", "随机种子"),
            ("Euler method", "数值积分"),
            ("Euler angle", "旋转"),
            ("experience replay", "样本"),
            ("data replay", "训练"),
            ("replay set", "测试"),
        ]
        for (text, meaning), result in zip(cases, decorate([x[0] for x in cases])):
            with self.subTest(text=text):
                self.assertEqual(result["text"], text)
                self.assertTrue(any(meaning in t["tip"] for t in result["tips"]), result)

    def test_pretraining_loss_does_not_link_to_gpu_assembly(self):
        result = decorate(["联合 RL 使用 PTX loss 保持原有能力。"])[0]
        self.assertFalse(any("ptx-sass" in t["href"] for t in result["tips"]), result)


if __name__ == "__main__":
    unittest.main()
