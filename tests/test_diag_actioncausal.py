import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from soul.diag_actioncausal import (
    _shuffle_action_timesteps,
    _write_json_report_atomic,
    intervene_actions,
    measure_ablation,
    measure_paired_actions,
    summarize_ablation,
)


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


    def test_paired_measurements_restore_the_callers_rng_state(self):
        torch.manual_seed(991)
        before = torch.get_rng_state().clone()

        with patch(
            "soul.diag_actioncausal.measure",
            side_effect=lambda *_args, **_kwargs: float(torch.rand(())),
        ):
            measure_paired_actions(None, seed=123, n_batches=1)

        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_paired_measurements_restore_rng_state_when_measurement_fails(self):
        torch.manual_seed(992)
        before = torch.get_rng_state().clone()

        with patch(
            "soul.diag_actioncausal.measure",
            side_effect=RuntimeError("synthetic measurement failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "synthetic measurement failure"):
                measure_paired_actions(None, seed=123, n_batches=1)

        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_shuffle_rejects_incompatible_shapes(self):
        with self.assertRaisesRegex(ValueError, r"actions must be"):
            _shuffle_action_timesteps(torch.zeros(2, 3), torch.ones(2, 3))



class ActionInterventionTests(unittest.TestCase):
    def _sequence(self):
        actions = torch.tensor([
            [[1.0], [2.0], [3.0], [4.0]],
            [[10.0], [20.0], [30.0], [40.0]],
        ])
        valid = torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 1.0, 1.0]])
        return actions, valid

    def test_real_keeps_every_action(self):
        actions, valid = self._sequence()
        self.assertTrue(torch.equal(intervene_actions(actions, valid, 'real'), actions))

    def test_zero_and_negate_only_change_valid_timesteps(self):
        actions, valid = self._sequence()
        zeroed = intervene_actions(actions, valid, 'zero')
        self.assertEqual(zeroed[0, :, 0].tolist(), [0.0, 0.0, 0.0, 0.0])
        self.assertEqual(zeroed[1, [0, 2, 3], 0].tolist(), [0.0, 0.0, 0.0])
        self.assertEqual(zeroed[1, 1, 0].item(), 20.0)
        negated = intervene_actions(actions, valid, 'negate')
        self.assertEqual(negated[0, :, 0].tolist(), [-1.0, -2.0, -3.0, -4.0])
        self.assertEqual(negated[1, [0, 2, 3], 0].tolist(), [-10.0, -30.0, -40.0])
        self.assertEqual(negated[1, 1, 0].item(), 20.0)

    def test_lag_and_lead_use_explicit_boundary_zero(self):
        actions, valid = self._sequence()
        lag = intervene_actions(actions, valid, 'lag')
        self.assertEqual(lag[0, :, 0].tolist(), [0.0, 1.0, 2.0, 3.0])
        self.assertEqual(lag[1, [0, 2, 3], 0].tolist(), [0.0, 10.0, 30.0])
        self.assertEqual(lag[1, 1, 0].item(), 20.0)
        lead = intervene_actions(actions, valid, 'lead')
        self.assertEqual(lead[0, :, 0].tolist(), [2.0, 3.0, 4.0, 0.0])
        self.assertEqual(lead[1, [0, 2, 3], 0].tolist(), [30.0, 40.0, 0.0])
        self.assertEqual(lead[1, 1, 0].item(), 20.0)

    def test_shuffle_preserves_multiset_and_padding(self):
        actions, valid = self._sequence()
        def reverse_indices(count, device=None, generator=None):
            return torch.arange(count - 1, -1, -1, device=device)

        with patch('soul.diag_actioncausal.torch.randperm', side_effect=reverse_indices):
            shuffled = intervene_actions(actions, valid, 'shuffle')

        self.assertEqual(shuffled[0, :, 0].tolist(), [4.0, 3.0, 2.0, 1.0])
        self.assertEqual(shuffled[1, [0, 2, 3], 0].tolist(), [40.0, 30.0, 10.0])
        self.assertEqual(shuffled[1, 1, 0].item(), 20.0)

    def test_unknown_intervention_is_rejected(self):
        actions, valid = self._sequence()
        with self.assertRaisesRegex(ValueError, 'unknown action intervention'):
            intervene_actions(actions, valid, 'reverse-time')


class AblationReportTests(unittest.TestCase):
    def test_ablation_is_deterministic_and_pairs_latent_and_replay_streams(self):
        calls = []

        def fake_series(_s, rng, kind, n_batches=10, intervention_seed=0):
            calls.append((kind, intervention_seed, int(rng.integers(0, 1000)),
                          float(torch.rand(()))))
            return [1.0 + index for index in range(n_batches)], n_batches * 8

        interventions = ('real', 'zero', 'negate')
        with patch('soul.diag_actioncausal.measure_kl_series', side_effect=fake_series):
            first = measure_ablation(None, seed=5, n_batches=2,
                                     interventions=interventions)
            first_calls = list(calls)
            calls.clear()
            second = measure_ablation(None, seed=5, n_batches=2,
                                      interventions=interventions)
            second_calls = list(calls)

        self.assertEqual(first, second)
        self.assertEqual([call[0] for call in first_calls], list(interventions))
        self.assertEqual([call[0] for call in second_calls], list(interventions))
        self.assertEqual(len({call[1] for call in first_calls}), len(interventions))
        self.assertEqual(len({call[2] for call in first_calls}), 1)
        self.assertEqual(len({call[3] for call in first_calls}), 1)
        self.assertEqual(first_calls, second_calls)

    def test_ablation_rejects_duplicate_or_unknown_interventions(self):
        with self.assertRaisesRegex(ValueError, 'unique'):
            measure_ablation(None, seed=1, interventions=('real', 'real'))
        with self.assertRaisesRegex(ValueError, 'unknown action intervention'):
            measure_ablation(None, seed=1, interventions=('real', 'bogus'))

    def test_report_refuses_checkpoint_and_dataset_aliases(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            protected = []
            for name in ('checkpoint.ckpt', 'train.npz', 'pilot.npz'):
                path = root / name
                path.write_bytes(name.encode('utf-8'))
                protected.append(path)

            for source in protected:
                before = source.read_bytes()
                with self.subTest(source=source.name):
                    with self.assertRaisesRegex(
                        ValueError, 'must not overwrite protected input'
                    ):
                        _write_json_report_atomic(source, {'unsafe': True}, protected)
                    self.assertEqual(source.read_bytes(), before)

            hardlink = root / 'train-hardlink.npz'
            try:
                os.link(protected[1], hardlink)
            except OSError:
                pass
            else:
                with self.assertRaisesRegex(
                    ValueError, 'must not overwrite protected input'
                ):
                    _write_json_report_atomic(hardlink, {'unsafe': True}, protected)
                self.assertEqual(protected[1].read_bytes(), b'train.npz')

            symlink = root / 'pilot-symlink.npz'
            try:
                symlink.symlink_to(protected[2])
            except (OSError, NotImplementedError):
                pass
            else:
                with self.assertRaisesRegex(ValueError, 'must not be a symlink'):
                    _write_json_report_atomic(symlink, {'unsafe': True}, protected)
                self.assertEqual(protected[2].read_bytes(), b'pilot.npz')

    def test_report_write_failure_preserves_existing_file_and_cleans_temp(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            report = root / 'report.json'
            report.write_text('previous report\n', encoding='utf-8')

            with patch(
                'soul.diag_actioncausal.json.dump',
                side_effect=OSError('synthetic serialization failure'),
            ):
                with self.assertRaisesRegex(OSError, 'synthetic serialization failure'):
                    _write_json_report_atomic(report, {'complete': True}, [])

            self.assertEqual(report.read_text(encoding='utf-8'), 'previous report\n')
            self.assertEqual(list(root.glob('.report.json.*.tmp')), [])


    def test_summary_keeps_per_episode_values_and_paired_effects(self):
        result = {
            'series': {'real': [1.0, 3.0], 'zero': [2.0, 5.0]},
            'valid_transitions': 12,
        }
        summary = summarize_ablation(result)

        self.assertEqual(summary['interventions']['real']['per_episode_kl'], [1.0, 3.0])
        self.assertEqual(summary['interventions']['real']['mean'], 2.0)
        self.assertEqual(summary['interventions']['zero']['dispersion'], 1.5)
        self.assertEqual(summary['paired_effects_vs_real']['zero']['mean'], 1.5)
        self.assertEqual(summary['paired_effects_vs_real']['zero']['median'], 1.5)
        self.assertEqual(summary['valid_transitions'], 12)


if __name__ == "__main__":
    unittest.main()
