from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "regression_results.json"
AA = ROOT / "sources/aa_detailed_model_signals_2026-07-31.csv"
AA_SCORE_TIMING = ROOT / "sources/aa_score_availability_2026-07-31.json"
AA_CHANGELOG = ROOT / "sources/aa_changelog_2026-07-31.json.gz"
PARAMETER_TRUTH = ROOT / "sources/open_model_parameter_truth_reconciliation_2026-07-31.json"


class FrontierEquivalenceExactInputTest(unittest.TestCase):
    def test_main_aa_branch_uses_exact_audited_crosswalk(self) -> None:
        data = json.loads(RESULT.read_text(encoding="utf-8"))
        audit = data["aa_exact_input_audit"]
        self.assertEqual(audit["calibration_matches"], 50)
        self.assertEqual(audit["frontier_matches"], 12)
        self.assertEqual(
            audit["sha256"], hashlib.sha256(AA.read_bytes()).hexdigest()
        )
        timing = audit["score_availability_timing"]
        self.assertEqual(
            timing["ledger_sha256"],
            hashlib.sha256(AA_SCORE_TIMING.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            timing["raw_changelog_sha256"],
            hashlib.sha256(AA_CHANGELOG.read_bytes()).hexdigest(),
        )
        self.assertIn("explicitly unverified", timing["policy"])
        truth = data["open_model_parameter_truth_reconciliation"]
        self.assertEqual(
            truth["sha256"], hashlib.sha256(PARAMETER_TRUTH.read_bytes()).hexdigest()
        )
        self.assertGreaterEqual(truth["corrected_aa_rows"], 6)
        self.assertGreaterEqual(truth["corrected_eci_rows"], 6)
        self.assertFalse(truth["checkpoint_identity_changed"])
        kimi = [row for row in data["open_models"] if row["family"] == "kimi_k2"]
        self.assertTrue(kimi)
        self.assertTrue(all(row["total_b"] == 1040.0 for row in kimi))
        self.assertTrue(all(row["active_b"] == 32.6 for row in kimi))
        calibration = audit["calibration_match_audit"]
        self.assertEqual(len({row["aa_model_id"] for row in calibration}), 50)
        date_conflicts = [row for row in calibration if row["aa_minus_canonical_release_days"]]
        self.assertEqual(
            [(row["model"], row["aa_minus_canonical_release_days"]) for row in date_conflicts],
            [("Qwen3-235B-A22B-Instruct-2507", 4)],
        )

        targets = {row["model"]: row for row in data["frontier_predictions"]}
        self.assertAlmostEqual(targets["Claude Fable 5"]["aa_score"], 59.8606463217303)
        self.assertAlmostEqual(targets["GPT-5.6 Sol"]["aa_score"], 58.889831189723)
        self.assertAlmostEqual(targets["Claude Opus 5"]["aa_score"], 60.6918740157091)


if __name__ == "__main__":
    unittest.main()
