from __future__ import annotations

import unittest

from frontier_target_signals import AA_TARGET_SIGNALS, TARGET_SLUGS


class FrontierTargetSignalsTest(unittest.TestCase):
    def test_exact_targets_are_complete_and_not_display_rounded(self) -> None:
        self.assertEqual(set(AA_TARGET_SIGNALS), set(TARGET_SLUGS))
        self.assertAlmostEqual(
            AA_TARGET_SIGNALS["Claude Fable 5"]["score"], 59.8606463217303
        )
        self.assertAlmostEqual(
            AA_TARGET_SIGNALS["GPT-5.6 Sol"]["score"], 58.889831189723
        )
        self.assertAlmostEqual(
            AA_TARGET_SIGNALS["Claude Opus 5"]["score"], 60.6918740157091
        )
        self.assertTrue(
            all(row["source_sha256"] == next(iter(AA_TARGET_SIGNALS.values()))["source_sha256"] for row in AA_TARGET_SIGNALS.values())
        )


if __name__ == "__main__":
    unittest.main()
