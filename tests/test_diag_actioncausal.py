import unittest
from unittest.mock import patch

import torch

from soul.diag_actioncausal import _shuffle_action_timesteps, measure_paired_actions


class ActionShuffleTests(unittest.TestCase):
    def test_shuffle_is_per_sequence_over_time_and_preserves_padding(self):
        actions = torch.tensor(
            [
                [[1.0], [2.0], [3.0]],
                [[10.0], [20.0], [30.0]],
            ]
        )
        valid = torch.tensor([[1.0, 1.0, 1.0], [1.0, 0.0, 1.0]])
        def reverse_indices(count, device=None, generator=None):
            return torch.arange(count - 1, -1, -1, device=device)

        with patch("soul.diag_actioncausal.torch.randperm", side_effect=reverse_indices):
            shuffled = _shuffle_action_timesteps(actions, valid)

        self.assertEqual(shuffled[0, :, 0].tolist(), [3.0, 2.0, 1.0])
        self.assertEqual(shuffled[1, valid[1] > 0, 0].tolist(), [30.0, 10.0])
        self.assertEqual(shuffled[1, 1, 0].item(), 20.0)
        self.assertTrue(
            all(value in {1.0, 2.0, 3.0} for value in shuffled[0, :, 0].tolist())
        )
        self.assertTrue(
            all(value in {10.0, 20.0, 30.0} for value in shuffled[1, :, 0].tolist())
        )

    def test_private_shuffle_generator_does_not_advance_latent_rng(self):
        actions = torch.tensor([[[1.0], [2.0], [3.0]]])
        valid = torch.ones(1, 3)
        torch.manual_seed(91)
        before = torch.get_rng_state().clone()
        generator = torch.Generator().manual_seed(7)

        _shuffle_action_timesteps(actions, valid, generator=generator)

        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_paired_measurements_restore_latent_rng_and_replay_seed(self):
        draws = []

        def fake_measure(_s, rng, shuffle, n_batches=10, shuffle_seed=0):
            if shuffle:
                generator = torch.Generator().manual_seed(shuffle_seed)
                torch.randperm(5, generator=generator)
            values = torch.rand(n_batches)
            draws.append((shuffle, values.clone(), rng.integers(0, 1000)))
            return float(values.sum())

        with patch("soul.diag_actioncausal.measure", side_effect=fake_measure):
            first = measure_paired_actions(None, seed=123, n_batches=4)
            first_draws = [(flag, values.clone(), replay) for flag, values, replay in draws]
            draws.clear()
            second = measure_paired_actions(None, seed=123, n_batches=4)
            second_draws = [(flag, values.clone(), replay) for flag, values, replay in draws]

        self.assertEqual(first, second)
        self.assertEqual([flag for flag, _, _ in first_draws], [False, True])
        self.assertEqual([flag for flag, _, _ in second_draws], [False, True])
        for (_, first_values, first_replay), (_, second_values, second_replay) in zip(
            first_draws, second_draws
        ):
            self.assertTrue(torch.equal(first_values, second_values))
            self.assertEqual(first_replay, second_replay)
        self.assertTrue(torch.equal(first_draws[0][1], first_draws[1][1]))
        self.assertEqual(first_draws[0][2], first_draws[1][2])

    def test_shuffle_rejects_incompatible_shapes(self):
        with self.assertRaisesRegex(ValueError, r"actions must be"):
            _shuffle_action_timesteps(torch.zeros(2, 3), torch.ones(2, 3))


if __name__ == "__main__":
    unittest.main()
