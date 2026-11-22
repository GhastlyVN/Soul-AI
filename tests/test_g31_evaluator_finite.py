import unittest

import numpy as np

from soul.g31.evaluator import inverse_squash, physical_metrics


class G31EvaluatorFiniteTests(unittest.TestCase):
    def test_inverse_squash_rejects_nonfinite_values(self):
        for value in (np.nan, np.inf):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, 'outside invertible squash range'
            ):
                inverse_squash([value, 0.0, 0.0])

    def test_physical_metrics_reject_nonfinite_inputs(self):
        for predicted, target in (
            ([np.nan, 0.0], [0.0, 0.0]),
            ([0.0, 0.0], [np.inf, 0.0]),
        ):
            with self.subTest(predicted=predicted, target=target), self.assertRaisesRegex(
                ValueError, 'finite'
            ):
                physical_metrics(predicted, target)

    def test_physical_metrics_still_works_for_finite_inputs(self):
        metrics = physical_metrics([np.pi + 0.1, 1.0], [np.pi - 0.1, 0.5])
        self.assertAlmostEqual(metrics['velocity_mae'], 0.5)
        self.assertAlmostEqual(metrics['theta_mae'], 0.2)
