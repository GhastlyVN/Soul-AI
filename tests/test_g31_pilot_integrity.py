import unittest
from unittest.mock import patch

from soul.g31 import pilot as pilot_module


class SyntheticPilotIntegrityTests(unittest.TestCase):
    def test_split_labels_are_checked_without_asserts(self):
        original = pilot_module.record_transition

        def wrong_split(*args, **kwargs):
            row = dict(original(*args, **kwargs))
            row["split"] = "pilot"
            return row

        with patch.object(pilot_module, "check_split_disjoint"), patch.object(
            pilot_module, "record_transition", side_effect=wrong_split
        ):
            with self.assertRaisesRegex(ValueError, "non-train row"):
                pilot_module.run_pilot()


if __name__ == "__main__":
    unittest.main()
