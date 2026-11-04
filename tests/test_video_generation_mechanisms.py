"""Numerical contracts for the video handbook's runnable examples."""
import importlib.util
import math
from pathlib import Path
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'files/assets/examples/diffusion/video_generation_mechanisms.py'
spec = importlib.util.spec_from_file_location('video_mechanisms', SOURCE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class VideoMechanismTests(unittest.TestCase):
    def test_video_budget_matches_explicit_grid_counts(self):
        result = m.video_budget(81, 480, 832)
        self.assertEqual(result['latent_shape'], (16, 21, 60, 104))
        self.assertEqual(result['tokens'], 32760)
        self.assertEqual(result['patch_features'], 64)
        self.assertEqual(result['dense_pairs'], 1073217600)
        self.assertEqual(result['factorized_pairs'], 51793560)
        self.assertEqual(result['latent_elements'], 2096640)

    def test_architecture_specific_frame_and_patch_constraints(self):
        for sizes in [(80, 480, 832), (81, 479, 832), (81, 488, 832)]:
            with self.subTest(sizes=sizes), self.assertRaises(ValueError):
                m.video_budget(*sizes)
        self.assertEqual(m.video_budget(1, 16, 16)['tokens'], 1)

    def test_flow_path_endpoints_and_derivative(self):
        clean, noise = [2., -1.], [-2., 3.]
        state, velocity = m.flow_pair(clean, noise, .75)
        self.assertEqual(state, [-1., 2.])
        self.assertEqual(velocity, [-4., 4.])
        self.assertEqual(m.flow_pair(clean, noise, 0)[0], clean)
        self.assertEqual(m.flow_pair(clean, noise, 1)[0], noise)
        before = m.flow_pair(clean, noise, .75 - 1e-5)[0]
        after = m.flow_pair(clean, noise, .75 + 1e-5)[0]
        for a, b, v in zip(before, after, velocity):
            self.assertAlmostEqual((b - a) / 2e-5, v)

    def test_masked_loss_ignores_padding_in_numerator_and_denominator(self):
        self.assertEqual(m.masked_mse([-3, 5, 1e6], [-4, 4, 0], [1, 1, 0]), 1.)
        self.assertEqual(m.masked_mse([-3, 5], [-4, 4], [1, 1]), 1.)
        with self.assertRaises(ValueError):
            m.masked_mse([1], [0], [0])

    def test_cfg_endpoints_and_extrapolation(self):
        self.assertEqual(m.cfg([1., -1.], [3., 2.], 0), [1., -1.])
        self.assertEqual(m.cfg([1., -1.], [3., 2.], 1), [3., 2.])
        self.assertEqual(m.cfg([1., -1.], [3., 2.], 2), [5., 5.])

    def test_reverse_euler_follows_negative_time_step(self):
        states = m.euler([-2., 3.], [1., .75, .5, .25, 0.], lambda z, tau: [-4., 4.])
        self.assertEqual(states, [[-2., 3.], [-1., 2.], [0., 1.], [1., 0.], [2., -1.]])

    def test_gaussian_velocity_matches_analytic_transport_derivative(self):
        tau, seed, mean, std, eps = .37, -.8, 2., .5, 1e-6
        def transport(t):
            return (1-t)*mean + math.sqrt((1-t)**2 * std**2 + t**2)*seed
        numerical = (transport(tau+eps) - transport(tau-eps)) / (2*eps)
        self.assertAlmostEqual(m.gaussian_velocity([transport(tau)], tau, mean, std)[0], numerical)

    def test_finer_euler_grid_converges_to_gaussian_endpoint(self):
        errors=[]
        for steps in (20, 200, 2000):
            output=m.euler([-1., 0., 1.], [1-i/steps for i in range(steps+1)], m.gaussian_velocity)[-1]
            errors.append(max(abs(a-b) for a,b in zip(output, [1.5,2.,2.5])))
            self.assertLess(output[0], output[1])
            self.assertLess(output[1], output[2])
        self.assertGreater(errors[0], errors[1])
        self.assertGreater(errors[1], errors[2])
        self.assertLess(errors[2], .001)

    def test_rejects_invalid_time_grids(self):
        for grid in ([1.], [1.,1.,0.], [0.,1.], [1.,-.1]):
            with self.subTest(grid=grid), self.assertRaises(ValueError):
                m.euler([0.], grid, m.gaussian_velocity)

    def test_shapes_cannot_silently_truncate(self):
        with self.assertRaises(ValueError):
            m.flow_pair([1,2], [3], .5)
        with self.assertRaises(ValueError):
            m.cfg([1], [2,3], 1)
        with self.assertRaises(ValueError):
            m.masked_mse([1,2], [1], [1,1])


if __name__ == '__main__':
    unittest.main()
