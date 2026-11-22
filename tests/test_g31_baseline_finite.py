import unittest

import numpy as np

from soul.g31.baselines import fit_train_delta


def transition(before, after):
    return {'raw_observation_before': before, 'raw_observation_after': after}


class G31BaselineFiniteTests(unittest.TestCase):
    def test_empty_train_fails_closed(self):
        with self.assertRaisesRegex(ValueError, 'required'):
            fit_train_delta([])

    def test_nonfinite_train_delta_fails_closed(self):
        with self.assertRaisesRegex(ValueError, 'not finite'):
            fit_train_delta([transition([np.nan, 0.0, 0.0], [0.0, 0.0, 0.0])])

    def test_finite_baseline_is_unchanged(self):
        mean_dx, mean_dv = fit_train_delta([
            transition([0.0, 0.0, 0.0], [0.1, 0.2, 0.0]),
            transition([0.0, 0.0, 0.0], [0.3, 0.4, 0.0]),
        ])
        self.assertAlmostEqual(mean_dx, 0.2)
        self.assertAlmostEqual(mean_dv, 0.3)
