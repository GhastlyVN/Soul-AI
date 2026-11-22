import pathlib
import tempfile
import unittest

from soul.g31.real_data import persist_episodes


class PersistEpisodesTests(unittest.TestCase):
    def sample(self):
        return [[{"split": "pilot", "episode": 1}]]

    def test_existing_dataset_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as dirname:
            target = pathlib.Path(dirname) / "pilot.jsonl"
            target.write_text("original", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                persist_episodes(self.sample(), target)
            self.assertEqual(target.read_text(encoding="utf-8"), "original")

    def test_linked_dataset_is_not_written_through(self):
        with tempfile.TemporaryDirectory() as dirname:
            root = pathlib.Path(dirname)
            target = root / "pilot.jsonl"
            victim = root / "victim.jsonl"
            victim.write_text("original", encoding="utf-8")
            target.symlink_to(victim)
            with self.assertRaises(FileExistsError):
                persist_episodes(self.sample(), target)
            self.assertEqual(victim.read_text(encoding="utf-8"), "original")

    def test_dangling_link_is_not_created(self):
        with tempfile.TemporaryDirectory() as dirname:
            root = pathlib.Path(dirname)
            target = root / "pilot.jsonl"
            victim = root / "victim.jsonl"
            target.symlink_to(victim)
            with self.assertRaises(FileExistsError):
                persist_episodes(self.sample(), target)
            self.assertFalse(victim.exists())


if __name__ == "__main__":
    unittest.main()
