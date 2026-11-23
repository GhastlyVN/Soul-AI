import unittest

from soul.g31.pilot import run_pilot


class SyntheticPilotStatusTests(unittest.TestCase):
    def test_report_never_claims_a_capability_or_confirm_access(self):
        report = run_pilot()
        self.assertEqual(report['threshold'], 'TO_LOCK')
        self.assertEqual(report['confirm'], 'NOT RUN')
        self.assertEqual(report['pilot_classification'], 'INCONCLUSIVE')
        self.assertFalse(report['integrity']['confirm_accessed'])
        self.assertTrue(report['integrity']['train_pilot_disjoint'])
        self.assertEqual(report['g31_capability_claim'], 'NONE — pilot only, not confirm')


if __name__ == '__main__':
    unittest.main()
