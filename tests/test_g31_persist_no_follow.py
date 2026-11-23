import os
import pathlib
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.modules.setdefault("gymnasium", types.ModuleType("gymnasium"))
from soul.g31 import real_data
from soul.g31.real_data import persist_episodes


class PersistEpisodesTests(unittest.TestCase):
    def sample(self):
        return [[{"split": "pilot", "episode": 1}]]

    def temporary_files(self, root):
        return list(root.glob(".pilot.jsonl.*.tmp"))

    def test_publishes_a_complete_dataset_and_removes_the_temporary_link(self):
        with tempfile.TemporaryDirectory() as dirname:
            root = pathlib.Path(dirname)
            target = root / "pilot.jsonl"

            persist_episodes(self.sample(), target)

            self.assertEqual(
                target.read_text(encoding="utf-8"),
                '{"episode": 1, "split": "pilot"}\n',
            )
            self.assertEqual(self.temporary_files(root), [])

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

    def test_serialization_failure_leaves_no_partial_dataset_or_temp_file(self):
        with tempfile.TemporaryDirectory() as dirname:
            root = pathlib.Path(dirname)
            target = root / "pilot.jsonl"
            with patch.object(real_data.json, "dumps", side_effect=TypeError("bad row")):
                with self.assertRaisesRegex(TypeError, "bad row"):
                    persist_episodes(self.sample(), target)
            self.assertFalse(target.exists())
            self.assertEqual(self.temporary_files(root), [])

    def test_concurrent_destination_creation_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as dirname:
            root = pathlib.Path(dirname)
            target = root / "pilot.jsonl"
            original_link = os.link

            def create_competing_file(source, destination):
                pathlib.Path(destination).write_text("concurrent", encoding="utf-8")
                return original_link(source, destination)

            with patch.object(real_data.os, "link", side_effect=create_competing_file):
                with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                    persist_episodes(self.sample(), target)

            self.assertEqual(target.read_text(encoding="utf-8"), "concurrent")
            self.assertEqual(self.temporary_files(root), [])


if __name__ == "__main__":
    unittest.main()
