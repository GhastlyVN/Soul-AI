import unittest

from soul.g31.collector import check_split_disjoint


def ep(eid, split, seed):
    return {"episode_id": eid, "split": split, "episode_seed": seed}


class SplitSeedIntegrityTests(unittest.TestCase):
    def test_rejects_missing_episode_seed(self):
        with self.assertRaisesRegex(ValueError, "missing episode_seed"):
            check_split_disjoint([ep("t1", "train", None), ep("p1", "pilot", 2001)])

    def test_missing_seeds_are_not_silently_deduplicated(self):
        with self.assertRaisesRegex(ValueError, "missing episode_seed"):
            check_split_disjoint([ep("t1", "train", None), ep("t2", "train", None)])

    def test_valid_disjoint_seeds_still_pass(self):
        check_split_disjoint([
            ep("t1", "train", 1001),
            ep("p1", "pilot", 2001),
            ep("c1", "confirm", 3001),
        ])
