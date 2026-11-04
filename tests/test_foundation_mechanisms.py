import importlib.util
import math
from pathlib import Path
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "files/assets/examples/foundations/mechanisms.py"
spec = importlib.util.spec_from_file_location("foundation_mechanisms", SOURCE)
mechanisms = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mechanisms)


class FoundationMechanismsTests(unittest.TestCase):
    def test_affine_uses_input_by_output_weight_orientation(self):
        self.assertEqual(mechanisms.affine([1, 2], [[1, 2], [3, 4]], [.5, -.5]), [7.5, 9.5])

    def test_affine_rejects_ragged_or_mismatched_dimensions(self):
        for weights, bias in [([[1, 2], [3]], [0, 0]), ([[1, 2]], [0, 0]), ([[1], [2]], [0, 0])]:
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                mechanisms.affine([1, 2], weights, bias)

    def test_regression_gradient_matches_finite_differences(self):
        x, y, w, b = [1, 2], [2, 4], 1.0, 0.0
        loss, dw, db = mechanisms.regression_loss_and_gradients(x, y, w, b)
        self.assertEqual((loss, dw, db), (1.25, -2.5, -1.5))
        eps = 1e-5
        loss_at = lambda weight, bias: mechanisms.regression_loss_and_gradients(x, y, weight, bias)[0]
        self.assertAlmostEqual(dw, (loss_at(w + eps, b) - loss_at(w - eps, b)) / (2 * eps))
        self.assertAlmostEqual(db, (loss_at(w, b + eps) - loss_at(w, b - eps)) / (2 * eps))
        self.assertAlmostEqual(loss_at(w - .1 * dw, b - .1 * db), .545625)

    def test_regression_rejects_empty_or_unpaired_samples(self):
        for x, y in [([], []), ([1, 2], [1])]:
            with self.subTest(x=x), self.assertRaises(ValueError):
                mechanisms.regression_loss_and_gradients(x, y, 1, 0)

    def test_softmax_is_stable_and_shift_invariant(self):
        result = mechanisms.softmax([1000, 1001])
        shifted = mechanisms.softmax([0, 1])
        self.assertAlmostEqual(sum(result), 1)
        for a, b in zip(result, shifted):
            self.assertAlmostEqual(a, b)
        self.assertGreater(result[1], result[0])

    def test_softmax_rejects_empty_and_nonfinite_logits(self):
        for logits in [[], [0, math.inf], [math.nan, 0]]:
            with self.subTest(logits=logits), self.assertRaises(ValueError):
                mechanisms.softmax(logits)

    def test_causal_attention_cannot_read_future_values(self):
        q = k = [[1, 0], [0, 1]]
        output, weights = mechanisms.attention(q, k, [[1, 0], [0, 1]], causal=True)
        altered, _ = mechanisms.attention(q, k, [[1, 0], [100, 200]], causal=True)
        self.assertEqual(output[0], [1.0, 0.0])
        self.assertEqual(altered[0], output[0])
        self.assertEqual(weights[0], [1.0, 0.0])
        self.assertAlmostEqual(output[1][1], 1 / (1 + math.exp(-1 / math.sqrt(2))))
        self.assertNotEqual(output[1], altered[1])

    def test_unmasked_attention_can_read_both_positions(self):
        q = k = v = [[1, 0], [0, 1]]
        output, weights = mechanisms.attention(q, k, v)
        self.assertGreater(output[0][1], 0)
        for row in weights:
            self.assertAlmostEqual(sum(row), 1)

    def test_attention_rejects_invalid_shapes(self):
        with self.assertRaises(ValueError):
            mechanisms.attention([[1, 0]], [[1]], [[2]])
        with self.assertRaises(ValueError):
            mechanisms.attention([[1]], [[1], [2]], [[3]])


if __name__ == "__main__":
    unittest.main()
