"""Frozen-checkpoint loader refuses non-regular paths before deserialization."""
import os
import stat
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
            alias.symlink_to(real)
            with patch.object(score_pilot.Path, 'open', side_effect=AssertionError('must not open a symlink')):
                with self.assertRaisesRegex(ValueError, 'regular non-symlink file'):
                    score_pilot.load_frozen_checkpoint(alias)

    @unittest.skipUnless(hasattr(os, 'mkfifo'), 'POSIX FIFO test')
    def test_fifo_checkpoint_is_refused_before_open(self):
        with tempfile.TemporaryDirectory() as temp:
            fifo = Path(temp) / 'checkpoint.pt'
            os.mkfifo(fifo)
            with patch.object(score_pilot.Path, 'open', side_effect=AssertionError('must not open a FIFO')):
                with self.assertRaisesRegex(ValueError, 'regular non-symlink file'):
                    score_pilot.load_frozen_checkpoint(fifo)


if __name__ == '__main__':
    unittest.main()
