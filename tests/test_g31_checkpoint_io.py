import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from soul.g31.checkpoint_io import save_checkpoint_atomic, seed_training_rng


class G31CheckpointIOTests(unittest.TestCase):
    def test_seed_reproduces_model_initialization(self):
        seed_training_rng(1001)
        first = torch.nn.Linear(4, 3).state_dict()
        seed_training_rng(1001)
        second = torch.nn.Linear(4, 3).state_dict()
        seed_training_rng(1002)
        different = torch.nn.Linear(4, 3).state_dict()

        for name in first:
            self.assertTrue(torch.equal(first[name], second[name]))
        self.assertFalse(torch.equal(first["weight"], different["weight"]))

    def test_serialization_failure_preserves_existing_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "checkpoint.pt"
            target.write_bytes(b"previous checkpoint")

            def fail_after_partial_write(_payload, stream):
                stream.write(b"partial checkpoint")
                raise OSError("simulated serialization failure")

            with patch("soul.g31.checkpoint_io.torch.save", side_effect=fail_after_partial_write):
                with self.assertRaisesRegex(OSError, "simulated serialization failure"):
                    save_checkpoint_atomic({"weights": [1, 2, 3]}, target)

            self.assertEqual(target.read_bytes(), b"previous checkpoint")
            self.assertEqual(list(Path(directory).glob(".checkpoint.pt.*.tmp")), [])

    def test_replace_failure_preserves_existing_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "checkpoint.pt"
            target.write_bytes(b"previous checkpoint")

            with patch(
                "soul.g31.checkpoint_io.os.replace",
                side_effect=OSError("simulated replacement failure"),
            ):
                with self.assertRaisesRegex(OSError, "simulated replacement failure"):
                    save_checkpoint_atomic({"weights": [1, 2, 3]}, target)

            self.assertEqual(target.read_bytes(), b"previous checkpoint")
            self.assertEqual(list(Path(directory).glob(".checkpoint.pt.*.tmp")), [])

    def test_successful_save_replaces_existing_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "checkpoint.pt"
            target.write_bytes(b"previous checkpoint")

            save_checkpoint_atomic({"weights": torch.tensor([1.0, 2.0])}, target)

            self.assertNotEqual(target.read_bytes(), b"previous checkpoint")
            self.assertGreater(target.stat().st_size, 0)
            self.assertEqual(list(Path(directory).glob(".checkpoint.pt.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
