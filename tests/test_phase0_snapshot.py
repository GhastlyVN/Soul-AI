import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.phase0_score import checkpoint_snapshot


class Phase0CheckpointSnapshotTests(unittest.TestCase):
    def test_snapshot_contains_the_allowlisted_bytes(self):
        payload = b'phase0 checkpoint bytes'
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'soul.pt'
            path.write_bytes(payload)
            expected = hashlib.sha256(payload).hexdigest()
            snapshot, digest = checkpoint_snapshot(path, expected)
            try:
                self.assertEqual(digest, expected)
                self.assertEqual(snapshot.read_bytes(), payload)
            finally:
                snapshot.unlink(missing_ok=True)

    def test_rejects_unlisted_digest_before_deserialization(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'soul.pt'
            path.write_bytes(b'unlisted checkpoint bytes')
            with self.assertRaisesRegex(ValueError, 'does not match the expected digest'):
                checkpoint_snapshot(path, '0' * 64)


if __name__ == '__main__':
    unittest.main()

