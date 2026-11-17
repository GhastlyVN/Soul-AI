import hashlib
import math
import json
import tempfile
import unittest
from pathlib import Path

from soul.g31.train_data import TRAIN_SEEDS, load_train_jsonl_snapshot


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

    def test_committed_train_dataset_matches_the_strict_schema(self):
        path = Path(__file__).resolve().parents[1] / "artifacts/g31/train.jsonl"

        rows, digest = load_train_jsonl_snapshot(path)

        self.assertEqual(len(rows), 100)
        self.assertEqual({row["episode_seed"] for row in rows}, set(TRAIN_SEEDS))
        self.assertEqual(len({row["episode_id"] for row in rows}), 5)
        self.assertEqual(len(digest), 64)

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
