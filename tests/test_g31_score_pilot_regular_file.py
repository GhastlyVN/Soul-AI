"""Frozen-checkpoint loader refuses non-regular paths before deserialization."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest

from soul.g31 import score_pilot


class FrozenCheckpointRegularFileTests(unittest.TestCase):
    def test_symlink_checkpoint_is_refused_before_open(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            real = root / 'checkpoint.pt'
            real.write_bytes(b'not a checkpoint')
            alias = root / 'alias.pt'
            try:
                alias.symlink_to(real)
            except (OSError, NotImplementedError) as error:
                self.skipTest(f'symlink creation unavailable: {error}')
            with patch('soul.g31.score_pilot.os.open', side_effect=AssertionError('must not open a symlink')):
                with self.assertRaisesRegex(ValueError, 'regular non-symlink file'):
                    score_pilot.load_frozen_checkpoint(alias)

    @unittest.skipUnless(hasattr(os, 'mkfifo'), 'POSIX FIFO test')
    def test_fifo_checkpoint_is_refused_before_open(self):
        with tempfile.TemporaryDirectory() as temp:
            fifo = Path(temp) / 'checkpoint.pt'
            os.mkfifo(fifo)
            with patch('soul.g31.score_pilot.os.open', side_effect=AssertionError('must not open a FIFO')):
                with self.assertRaisesRegex(ValueError, 'regular non-symlink file'):
                    score_pilot.load_frozen_checkpoint(fifo)

    def test_checkpoint_path_replacement_between_lstat_and_open_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'checkpoint.pt'
            victim = root / 'outside.pt'
            source.write_bytes(b'original checkpoint bytes')
            victim.write_bytes(b'outside file bytes')
            real_open = os.open
            swapped = False

            def replace_before_open(path, flags, *args, **kwargs):
                nonlocal swapped
                if Path(path) == source and not swapped:
                    source.unlink()
                    try:
                        source.symlink_to(victim)
                    except (OSError, NotImplementedError) as error:
                        self.skipTest(f'symlink creation unavailable: {error}')
                    swapped = True
                return real_open(path, flags, *args, **kwargs)

            with patch('soul.g31.score_pilot.os.open', side_effect=replace_before_open):
                with self.assertRaisesRegex(ValueError, 'could not be opened safely|path changed while it was being opened'):
                    score_pilot.load_frozen_checkpoint(source)
            self.assertTrue(swapped)
            self.assertEqual(victim.read_bytes(), b'outside file bytes')

    @unittest.skipIf(os.name == 'nt', 'Windows sharing semantics differ for replacing an open file')
    def test_checkpoint_path_replacement_after_open_is_rejected_before_load(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'checkpoint.pt'
            replacement = root / 'replacement.pt'
            source.write_bytes(b'original checkpoint bytes')
            replacement.write_bytes(b'replacement checkpoint bytes')
            real_open = os.open
            replaced = False

            def replace_after_open(path, flags, *args, **kwargs):
                nonlocal replaced
                descriptor = real_open(path, flags, *args, **kwargs)
                if Path(path) == source and not replaced:
                    os.replace(replacement, source)
                    replaced = True
                return descriptor

            with patch('soul.g31.score_pilot.os.open', side_effect=replace_after_open):
                with self.assertRaisesRegex(ValueError, 'path changed while it was being opened'):
                    score_pilot.load_frozen_checkpoint(source)
            self.assertTrue(replaced)


if __name__ == '__main__':
    unittest.main()
