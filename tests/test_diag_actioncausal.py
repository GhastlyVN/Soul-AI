import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from soul.diag_actioncausal import (
    _shuffle_action_timesteps,
    _write_json_report_atomic,
    checkpoint_exists_or_training,
    load_checkpoint_snapshot,
    sha256_file,
    intervene_actions,
    measure_ablation,
    measure_paired_actions,
    summarize_ablation,
)


class CheckpointBoundaryTests(unittest.TestCase):
    def test_missing_checkpoint_requires_explicit_training_opt_in(self):
        missing = Path(tempfile.gettempdir()) / f'soul-action-missing-{os.getpid()}.ckpt'
        missing.unlink(missing_ok=True)
        with self.assertRaisesRegex(FileNotFoundError, '--train-if-missing'):
            checkpoint_exists_or_training(missing, allow_training=False)
        self.assertFalse(checkpoint_exists_or_training(missing, allow_training=True))

    def test_checkpoint_snapshot_hashes_the_exact_bytes_loaded(self):
        class CapturingSoul:
            def load(self, path):
                self.loaded_path = Path(path)
                self.loaded_bytes = self.loaded_path.read_bytes()

        source = Path(tempfile.gettempdir()) / f'soul-action-snapshot-{os.getpid()}.ckpt'
        payload = b'checkpoint bytes are hashed and loaded from the same snapshot'
        source.write_bytes(payload)
        soul = CapturingSoul()
        try:
            expected = sha256_file(source)
            actual = load_checkpoint_snapshot(soul, source, expected_sha256=expected)
            self.assertEqual(actual, expected)
            self.assertEqual(soul.loaded_bytes, payload)
            self.assertFalse(soul.loaded_path.exists())
        finally:
            source.unlink(missing_ok=True)

    def test_checkpoint_snapshot_rejects_digest_mismatch_before_load(self):
        class MustNotLoad:
            def load(self, _path):
                raise AssertionError('checkpoint must not deserialize after a hash mismatch')

        source = Path(tempfile.gettempdir()) / f'soul-action-mismatch-{os.getpid()}.ckpt'
        source.write_bytes(b'known checkpoint bytes')
        try:
            with self.assertRaisesRegex(ValueError, 'does not match the expected digest'):
                load_checkpoint_snapshot(MustNotLoad(), source, expected_sha256='0' * 64)
        finally:
            source.unlink(missing_ok=True)

    def test_checkpoint_symlinks_are_rejected(self):
        source = Path(tempfile.gettempdir()) / f'soul-action-symlink-target-{os.getpid()}.ckpt'
        alias = Path(tempfile.gettempdir()) / f'soul-action-symlink-{os.getpid()}.ckpt'
        source.write_bytes(b'checkpoint')
        try:
            try:
                alias.symlink_to(source)
            except (OSError, NotImplementedError):
                self.skipTest('symlink creation is unavailable on this runner')
            with self.assertRaisesRegex(ValueError, 'must not be a symlink'):
                checkpoint_exists_or_training(alias, allow_training=False)

            class MustNotLoad:
                def load(self, _path):
                    raise AssertionError('a linked checkpoint must never be deserialized')

            with self.assertRaisesRegex(ValueError, 'regular non-symlink file'):
                load_checkpoint_snapshot(MustNotLoad(), alias)
        finally:
            alias.unlink(missing_ok=True)
            source.unlink(missing_ok=True)


    def test_checkpoint_snapshot_rejects_oversized_file_before_load(self):
        class MustNotLoad:
            def load(self, _path):
                raise AssertionError('oversized checkpoint must not be deserialized')

        source = Path(tempfile.gettempdir()) / f'soul-action-oversized-{os.getpid()}.ckpt'
        source.write_bytes(b'12345')
        try:
            with patch('soul.diag_actioncausal.MAX_CHECKPOINT_BYTES', 4):
                with self.assertRaisesRegex(ValueError, 'exceeds size limit'):
                    load_checkpoint_snapshot(MustNotLoad(), source)
        finally:
            source.unlink(missing_ok=True)

    def test_provenance_hash_is_bounded(self):
        source = Path(tempfile.gettempdir()) / f'soul-action-provenance-large-{os.getpid()}.bin'
        source.write_bytes(b'12345')
        try:
            with patch('soul.diag_actioncausal.MAX_PROVENANCE_FILE_BYTES', 4):
                with self.assertRaisesRegex(ValueError, 'provenance input exceeds size limit'):
                    sha256_file(source)
        finally:
            source.unlink(missing_ok=True)

    def test_provenance_hash_rejects_symlink_before_open(self):
        source = Path(tempfile.gettempdir()) / f'soul-action-provenance-target-{os.getpid()}.bin'
        alias = Path(tempfile.gettempdir()) / f'soul-action-provenance-link-{os.getpid()}.bin'
        source.write_bytes(b'provenance')
        try:
            try:
                alias.symlink_to(source)
            except (OSError, NotImplementedError):
                self.skipTest('symlink creation is unavailable on this runner')
            with patch(
                'soul.diag_actioncausal.os.open',
                side_effect=AssertionError('a symlink must be rejected before opening'),
            ):
                with self.assertRaisesRegex(ValueError, 'regular non-symlink file'):
                    sha256_file(alias)
        finally:
            alias.unlink(missing_ok=True)
            source.unlink(missing_ok=True)

    @unittest.skipIf(os.name == 'nt', 'POSIX FIFO inputs are unavailable on Windows')
    def test_provenance_hash_rejects_fifo_before_open(self):
        fifo = Path(tempfile.gettempdir()) / f'soul-action-provenance-fifo-{os.getpid()}'
        os.mkfifo(fifo)
        try:
            with patch(
                'soul.diag_actioncausal.os.open',
                side_effect=AssertionError('a FIFO must be rejected before opening'),
            ):
                with self.assertRaisesRegex(ValueError, 'regular non-symlink file'):
                    sha256_file(fifo)
        finally:
            fifo.unlink(missing_ok=True)

    @unittest.skipIf(os.name == 'nt', 'open-file replacement semantics differ on Windows')
    def test_provenance_hash_rejects_path_replacement_after_open(self):
        source = Path(tempfile.gettempdir()) / f'soul-action-provenance-race-{os.getpid()}.bin'
        replacement = Path(tempfile.gettempdir()) / f'soul-action-provenance-race-new-{os.getpid()}.bin'
        source.write_bytes(b'original provenance bytes')
        replacement.write_bytes(b'replacement provenance bytes')
        original_open = os.open
        replaced = False

        def replace_after_open(path, flags, *args, **kwargs):
            nonlocal replaced
            descriptor = original_open(path, flags, *args, **kwargs)
            if Path(path) == source and not replaced:
                os.replace(replacement, source)
                replaced = True
            return descriptor

        try:
            with patch('soul.diag_actioncausal.os.open', side_effect=replace_after_open):
                with self.assertRaisesRegex(ValueError, 'path changed while it was being opened'):
                    sha256_file(source)
            self.assertTrue(replaced)
        finally:
            source.unlink(missing_ok=True)
            replacement.unlink(missing_ok=True)

    @unittest.skipIf(os.name == 'nt', 'open-file replacement semantics differ on Windows')
    def test_checkpoint_snapshot_rejects_path_replacement_after_open(self):
        class MustNotLoad:
            def load(self, _path):
                raise AssertionError('replaced checkpoint must not be deserialized')

        source = Path(tempfile.gettempdir()) / f'soul-action-race-{os.getpid()}.ckpt'
        replacement = Path(tempfile.gettempdir()) / f'soul-action-race-new-{os.getpid()}.ckpt'
        source.write_bytes(b'original checkpoint')
        replacement.write_bytes(b'replacement checkpoint')
        original_open = os.open
        replaced = False

        def replace_after_open(path, flags, *args, **kwargs):
            nonlocal replaced
            descriptor = original_open(path, flags, *args, **kwargs)
            if Path(path) == source and not replaced:
                os.replace(replacement, source)
                replaced = True
            return descriptor

        try:
            with patch('soul.diag_actioncausal.os.open', side_effect=replace_after_open):
                with self.assertRaisesRegex(ValueError, 'path changed while it was being opened'):
                    load_checkpoint_snapshot(MustNotLoad(), source)
            self.assertTrue(replaced)
        finally:
            source.unlink(missing_ok=True)
            replacement.unlink(missing_ok=True)


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

    def test_ablation_requires_positive_integer_batch_count(self):
        for n_batches in (0, -1, True, 1.5):
            with self.subTest(n_batches=n_batches):
                with self.assertRaisesRegex(ValueError, 'positive integer'):
                    measure_ablation(None, seed=1, n_batches=n_batches)

    def test_ablation_rejects_missing_batches_and_mismatched_transition_counts(self):
        with patch(
            'soul.diag_actioncausal.measure_kl_series',
            return_value=([1.0], 8),
        ):
            with self.assertRaisesRegex(ValueError, 'expected 2'):
                measure_ablation(None, seed=1, n_batches=2, interventions=('real',))

        def mismatched_count(_s, _rng, kind, n_batches=10, intervention_seed=0):
            return [1.0] * n_batches, 8 if kind == 'real' else 7

        with patch(
            'soul.diag_actioncausal.measure_kl_series',
            side_effect=mismatched_count,
        ):
            with self.assertRaisesRegex(ValueError, 'valid-transition count differs'):
                measure_ablation(
                    None, seed=1, n_batches=2, interventions=('real', 'zero')
                )

    def test_summary_rejects_mismatched_paired_series_lengths(self):
        with self.assertRaisesRegex(ValueError, 'same number of paired batches'):
            summarize_ablation({
                'series': {'real': [1.0, 2.0], 'zero': [3.0]},
                'valid_transitions': 8,
            })

    def test_summary_rejects_empty_and_nonfinite_series(self):
        for series in (
            {'real': []},
            {'real': [1.0], 'zero': [float('nan')]},
            {'real': [1.0], 'zero': [True]},
        ):
            with self.subTest(series=series):
                with self.assertRaisesRegex(ValueError, 'at least one batch|finite numeric'):
                    summarize_ablation({
                        'series': series,
                        'valid_transitions': 8,
                    })

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


    def test_report_write_round_trips_sorted_json(self):
        with tempfile.TemporaryDirectory() as temp:
            report = Path(temp) / 'nested' / 'report.json'
            payload = {'z': 1, 'a': {'d': 4, 'b': 2}}

            _write_json_report_atomic(report, payload, [])

            self.assertEqual(
                report.read_text(encoding='utf-8'),
                '{\n  "a": {\n    "b": 2,\n    "d": 4\n  },\n  "z": 1\n}\n',
            )

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
