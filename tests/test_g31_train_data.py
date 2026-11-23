import hashlib
import math
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from soul.g31.train_data import TRAIN_SEEDS, load_train_jsonl_snapshot, read_file_snapshot, rows_to_episodes


def valid_row():
    return {
        "episode_id": "train_ep1001",
        "split": "train",
        "episode_seed": 1001,
        "timestep": 0,
        "raw_observation_before": [0.0, 0.0, 0.0],
        "applied_action": [0.1],
        "reward": 1.0,
        "terminal": False,
        "raw_observation_after": [0.1, 0.2, 0.3],
        "environment_config": {"env_id": "Pendulum-v1"},
    }


class G31TrainDataTests(unittest.TestCase):
    def _load_row(self, row):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "train.jsonl"
        encoded = (json.dumps(row, separators=(",", ":")) + "\n").encode("utf-8")
        path.write_bytes(encoded)
        return path, encoded

    def test_valid_row_hashes_the_exact_snapshot(self):
        path, encoded = self._load_row(valid_row())

        rows, digest = load_train_jsonl_snapshot(path)

        self.assertEqual(len(rows), 1)
        self.assertEqual(digest, hashlib.sha256(encoded).hexdigest())

    @unittest.skipUnless(hasattr(os, "symlink") and os.name != "nt", "POSIX symlinks required")
    def test_snapshot_rejects_symlink_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.jsonl"
            alias = root / "alias.jsonl"
            target.write_bytes(b"private data")
            alias.symlink_to(target)
            with self.assertRaisesRegex(ValueError, "regular non-symlink"):
                read_file_snapshot(alias, 1024)

    @unittest.skipUnless(hasattr(os, "mkfifo") and os.name != "nt", "POSIX FIFO required")
    def test_snapshot_rejects_fifo_swapped_in_after_path_check_without_blocking(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "train.jsonl"
            fifo = root / "replacement.fifo"
            source.write_bytes(b"{}\\n")
            os.mkfifo(fifo)
            original_open = os.open

            def swap_then_open(path, flags, *args, **kwargs):
                os.replace(fifo, source)
                return original_open(path, flags, *args, **kwargs)

            with patch("soul.g31.train_data.os.open", side_effect=swap_then_open):
                with self.assertRaisesRegex(RuntimeError, "changed before it could be read"):
                    read_file_snapshot(source, 1024)

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks are unavailable")
    def test_snapshot_rejects_symlink_swap_before_reading_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "train.jsonl"
            target = root / "target.jsonl"
            alias = root / "alias.jsonl"
            source.write_bytes(b"original")
            target.write_bytes(b"must not be read")
            alias.symlink_to(target)
            original_open = os.open

            def swap_then_open(path, flags, *args, **kwargs):
                os.replace(alias, source)
                return original_open(path, flags, *args, **kwargs)

            with patch("soul.g31.train_data.os.open", side_effect=swap_then_open):
                with self.assertRaisesRegex(ValueError, "opened safely|changed before"):
                    read_file_snapshot(source, 1024)

    @unittest.skipIf(os.name == "nt", "open-file replacement semantics differ on Windows")
    def test_snapshot_rejects_path_replacement_after_open(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "train.jsonl"
            replacement = root / "replacement.jsonl"
            source.write_bytes(b"original")
            replacement.write_bytes(b"replacement")
            original_open = os.open

            def open_then_replace(path, flags, *args, **kwargs):
                descriptor = original_open(path, flags, *args, **kwargs)
                os.replace(replacement, source)
                return descriptor

            with patch("soul.g31.train_data.os.open", side_effect=open_then_replace):
                with self.assertRaisesRegex(RuntimeError, "changed before it could be read|path changed while it was being opened"):
                    read_file_snapshot(source, 1024)

    def test_committed_train_dataset_matches_the_strict_schema(self):
        path = Path(__file__).resolve().parents[1] / "artifacts/g31/train.jsonl"

        rows, digest = load_train_jsonl_snapshot(path)

        self.assertEqual(len(rows), 100)
        self.assertEqual({row["episode_seed"] for row in rows}, set(TRAIN_SEEDS))
        self.assertEqual(len({row["episode_id"] for row in rows}), 5)
        self.assertEqual(len(digest), 64)

    def test_observations_reject_booleans_and_numeric_strings(self):
        cases = (
            ("raw_observation_before", ["0.1", 0.0, 0.0]),
            ("raw_observation_after", [0.1, True, 0.3]),
        )
        for field, value in cases:
            with self.subTest(field=field, value=value):
                row = valid_row()
                row[field] = value
                path, _ = self._load_row(row)

                with self.assertRaisesRegex(ValueError, "invalid observation value"):
                    load_train_jsonl_snapshot(path)

    def test_applied_action_must_stay_within_pendulum_bounds(self):
        for action in (-2.0001, 2.0001):
            with self.subTest(action=action):
                row = valid_row()
                row["applied_action"] = [action]
                path, _ = self._load_row(row)

                with self.assertRaisesRegex(ValueError, "applied action outside"):
                    load_train_jsonl_snapshot(path)

    def test_applied_action_accepts_both_inclusive_bounds(self):
        for action in (-2.0, 2.0):
            with self.subTest(action=action):
                row = valid_row()
                row["applied_action"] = [action]
                path, _ = self._load_row(row)

                rows, _ = load_train_jsonl_snapshot(path)
                self.assertEqual(rows[0]["applied_action"], [action])

    def test_applied_action_rejects_booleans_and_numeric_strings(self):
        for action in (True, '1.0'):
            with self.subTest(action=action):
                row = valid_row()
                row['applied_action'] = [action]
                path, _ = self._load_row(row)
                with self.assertRaisesRegex(ValueError, 'numeric value'):
                    load_train_jsonl_snapshot(path)

    def test_action_bounds_are_checked_before_float32_rounding(self):
        for action in (-2.00000001, 2.00000001):
            with self.subTest(action=action):
                row = valid_row()
                row['applied_action'] = [action]
                path, _ = self._load_row(row)
                with self.assertRaisesRegex(ValueError, 'applied action outside'):
                    load_train_jsonl_snapshot(path)
    def test_rows_to_episodes_rejects_episode_id_reused_across_seeds(self):
        first = valid_row()
        second = valid_row()
        second["episode_seed"] = 1002
        second["timestep"] = 1

        with self.assertRaisesRegex(ValueError, "episode_seed changes within episode"):
            rows_to_episodes([first, second], None, 0)

    def test_loader_rejects_multiple_episodes_for_one_train_seed(self):
        first = valid_row()
        second = valid_row()
        second["episode_id"] = "train_ep1001_duplicate"
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "train.jsonl"
        path.write_text(
            "\n".join(json.dumps(row) for row in (first, second)) + "\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "assigned to multiple episodes"):
            load_train_jsonl_snapshot(path)

    def test_rows_to_episodes_rejects_seed_reused_by_multiple_episode_ids(self):
        first = valid_row()
        second = valid_row()
        second["episode_id"] = "train_ep1001_duplicate"

        with self.assertRaisesRegex(ValueError, "assigned to multiple episodes"):
            rows_to_episodes([first, second], None, 0)

    def test_rows_to_episodes_rejects_transitions_after_terminal(self):
        first = valid_row()
        first["terminal"] = True
        second = valid_row()
        second["timestep"] = 1
        second["raw_observation_before"] = first["raw_observation_after"]
        second["raw_observation_after"] = [0.2, 0.3, 0.4]

        with self.assertRaisesRegex(ValueError, "terminal transition must be the final timestep"):
            rows_to_episodes([first, second], None, 0)

    def test_missing_reward_is_rejected_before_episode_conversion(self):
        row = valid_row()
        row.pop("reward")
        path, _ = self._load_row(row)

        with self.assertRaisesRegex(ValueError, "missing reward"):
            load_train_jsonl_snapshot(path)

    def test_terminal_must_be_a_json_boolean(self):
        row = valid_row()
        row["terminal"] = "false"
        path, _ = self._load_row(row)

        with self.assertRaisesRegex(ValueError, "terminal must be a boolean"):
            load_train_jsonl_snapshot(path)

    def test_timestep_must_be_a_nonnegative_integer_not_a_boolean(self):
        for timestep in (-1, 0.0, True):
            with self.subTest(timestep=timestep):
                row = valid_row()
                row["timestep"] = timestep
                path, _ = self._load_row(row)

                with self.assertRaisesRegex(ValueError, "invalid timestep"):
                    load_train_jsonl_snapshot(path)

    def test_episode_id_must_be_nonempty_text(self):
        for episode_id in ("", "   ", [], None):
            with self.subTest(episode_id=episode_id):
                row = valid_row()
                row["episode_id"] = episode_id
                path, _ = self._load_row(row)

                with self.assertRaisesRegex(ValueError, "invalid episode_id"):
                    load_train_jsonl_snapshot(path)

    def test_reward_must_be_a_finite_json_number(self):
        for reward in (True, "1.0", float("nan"), float("inf")):
            with self.subTest(reward=reward):
                row = valid_row()
                row["reward"] = reward
                path, _ = self._load_row(row)

                expected = "nonfinite reward" if isinstance(reward, float) and not math.isfinite(reward) else "invalid reward"
                with self.assertRaisesRegex(ValueError, expected):
                    load_train_jsonl_snapshot(path)


if __name__ == "__main__":
    unittest.main()
