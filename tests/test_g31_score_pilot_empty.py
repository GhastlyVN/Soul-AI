import unittest

from soul.g31.score_pilot import aggregate_episode_metrics, paired_differences


class ScorePilotEmptyTests(unittest.TestCase):
    def test_empty_aggregate_fails_closed(self):
        with self.assertRaisesRegex(ValueError, 'no episode metrics'):
            aggregate_episode_metrics({})

    def test_empty_paired_fails_closed(self):
        with self.assertRaisesRegex(ValueError, 'no episode metrics'):
            paired_differences({}, {})

    def test_paired_requires_the_same_episodes(self):
        model = {'ep1': {'mae': 1.0}}
        baseline = {'ep2': {'mae': 2.0}}
        with self.assertRaisesRegex(ValueError, 'different episodes'):
            paired_differences(model, baseline)


if __name__ == '__main__':
    unittest.main()
