import unittest

import k3_primary_evidence as evidence


class K3PrimaryEvidenceTests(unittest.TestCase):
    def test_exact_counts_are_not_launch_page_rounding(self) -> None:
        self.assertEqual(evidence.K3_TOTAL_B, 2780.0)
        self.assertEqual(evidence.K3_TOTAL_T, 2.78)
        self.assertEqual(evidence.K3_ACTIVE_B, 104.2)
        self.assertEqual(evidence.K3_TOTAL_T_DISPLAY, 2.8)

    def test_source_is_primary_report(self) -> None:
        self.assertEqual(
            evidence.K3_PARAMETER_SOURCE,
            "Kimi K3 official technical report Table 1",
        )
        self.assertTrue(evidence.K3_EVIDENCE_PATH.is_file())


if __name__ == "__main__":
    unittest.main()
