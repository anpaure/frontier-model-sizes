from __future__ import annotations

import csv
import hashlib
import json
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/019f6c42-2d53-7743-ab07-6293e2618dd7"
RESULT = OUT / "aa_score_availability_timing_audit_2026-07-31.json"
PREDICTIONS = OUT / "aa_score_availability_timing_changes_2026-07-31.csv"


class AaScoreAvailabilityTimingAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = json.loads(RESULT.read_text(encoding="utf-8"))
        with PREDICTIONS.open(newline="", encoding="utf-8") as handle:
            cls.predictions = list(csv.DictReader(handle))

    def test_coverage_and_source_shortfall_are_explicit(self) -> None:
        coverage = self.result["coverage"]
        self.assertEqual(coverage["live_aa_rows"], 50)
        self.assertEqual(coverage["live_aa_verified"], 28)
        self.assertEqual(coverage["live_aa_fallback"], 22)
        self.assertEqual(coverage["live_aa_verified_after_release"], 16)
        self.assertEqual(coverage["detailed_aa_rows"], 275)
        self.assertEqual(coverage["detailed_aa_verified"], 64)
        self.assertEqual(coverage["detailed_aa_verified_after_release"], 36)
        self.assertIs(coverage["api_total_reconciles"], False)

    def test_release_order_counterfactual_reproduces_the_prior_backtest(self) -> None:
        impact = self.result["validation_impact"]
        baseline = impact["available_component_ensemble"]["frontier_like"][
            "release_order_baseline"
        ]
        corrected = impact["available_component_ensemble"]["frontier_like"][
            "score_timing_corrected"
        ]
        self.assertEqual(baseline["n"], 26)
        self.assertTrue(
            math.isclose(
                baseline["median_multiplicative_error"],
                2.086202144829668,
                rel_tol=0,
                abs_tol=1e-12,
            )
        )
        self.assertEqual(corrected["n"], 27)
        self.assertTrue(
            math.isclose(
                corrected["median_multiplicative_error"],
                2.0237034416634527,
                rel_tol=0,
                abs_tol=1e-12,
            )
        )

    def test_impact_is_not_misreported_as_uniform_improvement(self) -> None:
        impact = self.result["validation_impact"]
        aa = impact["aa"]["all"]
        self.assertGreater(
            aa["score_timing_corrected"]["median_multiplicative_error"],
            aa["release_order_baseline"]["median_multiplicative_error"],
        )
        ensemble = impact["available_component_ensemble"]["all"]
        self.assertLess(
            ensemble["score_timing_corrected"]["median_multiplicative_error"],
            ensemble["release_order_baseline"]["median_multiplicative_error"],
        )
        changed = self.result["changed_rows"]
        self.assertEqual(changed["aa_common_predictions_changed"], 13)
        self.assertEqual(changed["aa_newly_eligible_predictions"], 1)
        self.assertEqual(changed["ensemble_common_predictions_changed"], 3)
        self.assertEqual(changed["ensemble_newly_eligible_predictions"], 1)

    def test_current_fit_and_centers_are_not_changed(self) -> None:
        decision = self.result["decision"]
        self.assertIs(decision["change_current_fit"], False)
        self.assertIs(decision["change_live_weights"], False)
        self.assertIs(decision["change_headline_centers"], False)
        self.assertIs(decision["change_validation_and_uncertainty"], True)

    def test_prediction_ledger_and_source_hashes_reconcile(self) -> None:
        self.assertTrue(self.predictions)
        self.assertEqual(
            len(
                {
                    (row["comparison"], row["release_date"], row["model"])
                    for row in self.predictions
                }
            ),
            len(self.predictions),
        )
        for relative, expected in self.result["source_files"].items():
            path = ROOT / relative
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)


if __name__ == "__main__":
    unittest.main()
