import contextlib
import io
import unittest
from unittest.mock import patch

from soul.g31 import __main__ as g31_cli


class G31CliGuardTests(unittest.TestCase):
    def run_cli(self, command):
        out = io.StringIO()
        with patch("sys.argv", ["g31", command]), contextlib.redirect_stdout(out):
            g31_cli.main()
        return out.getvalue()

    def test_evaluate_stays_blocked(self):
        self.assertIn("confirm blocked", self.run_cli("evaluate"))

    def test_lock_threshold_does_not_set_it(self):
        out = self.run_cli("lock-threshold")
        self.assertIn("TO_LOCK", out)
        self.assertIn("not set", out)


if __name__ == "__main__":
    unittest.main()
