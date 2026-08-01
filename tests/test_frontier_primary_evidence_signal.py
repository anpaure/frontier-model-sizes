import csv
import json
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "019f6c42-2d53-7743-ab07-6293e2618dd7"
RESULT = OUT / "frontier_primary_evidence_audit_2026-07-18.json"
CONTROLS = OUT / "frontier_primary_evidence_controls_2026-07-18.csv"


class FrontierPrimaryEvidenceSignalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = json.loads(RESULT.read_text(encoding="utf-8"))
        with CONTROLS.open(newline="", encoding="utf-8") as handle:
            cls.controls = list(csv.DictReader(handle))

    def test_direct_measurement_and_current_projection_reconcile(self):
        official = self.result["official_measurements"]
        current = self.result["current_mapping"]
        self.assertEqual(official["gpt_5_6_sol_nocot_minutes"], 3.6)
        self.assertEqual(official["gpt_5_5_comparator_minutes"], 2.3)
        self.assertAlmostEqual(official["suite_rebased_sol_minutes"], 3.0 * 3.6 / 2.3)
        expected = math.sqrt(1.4 * 3.0) * 2 ** (
            current["effective_date_to_sol_days"]
            / current["exact_date_adjusted_doubling_days"]
        )
        self.assertAlmostEqual(current["projected_sol_horizon_minutes"], expected)
        self.assertAlmostEqual(current["projected_sol_horizon_prior_t"], 2.9850434724007573)

    def test_heldout_rule_is_chronological_and_developer_excluded(self):
        inventory = self.result["inventory"]
        self.assertEqual(inventory["open_weight_no_cot_ground_truth_models"], 35)
        self.assertEqual(inventory["open_weight_developers"], 6)
        self.assertEqual(inventory["chronological_developer_holdout_predictions"], 16)
        self.assertEqual(inventory["heldout_developers"], 5)
        backtest = self.result["heldout_backtest"]
        self.assertLess(
            backtest["horizon_equal_developer_mean_absolute_ln_error"],
            backtest["baseline_equal_developer_mean_absolute_ln_error"],
        )
        bootstrap = backtest["incremental_bootstrap"]
        self.assertLess(bootstrap["ci_90"][0], 0)
        self.assertGreater(bootstrap["ci_90"][1], 0)
        self.assertLess(bootstrap["bootstrap_probability_horizon_better"], 0.8)

    def test_mapping_uncertainty_blocks_promotion(self):
        self.assertEqual(self.result["k3_anchor"]["total_parameters_t"], 2.78)
        self.assertEqual(
            self.result["k3_anchor"]["source"],
            "Kimi K3 official technical report Table 1",
        )
        sensitivity = self.result["sol_mapping_sensitivity"]
        self.assertGreater(sensitivity["direct_model_level_horizon_regression_t"], 1.5)
        self.assertLess(sensitivity["direct_model_level_horizon_regression_t"], 2.0)
        self.assertGreater(sensitivity["direct_pooled_paper_elasticity_t"], 6.0)
        self.assertGreater(sensitivity["direct_moe_paper_elasticity_t"], 9.0)
        self.assertGreater(sensitivity["nonbaseline_method_max_over_min"], 5.0)
        decision = self.result["decision"]
        self.assertFalse(decision["promote_direct_sol_horizon_increment"])
        self.assertEqual(decision["incremental_live_weight"], 0.0)
        self.assertFalse(decision["change_headline_forecasts"])

    def test_same_size_controls_and_fable_identity_are_retained(self):
        exact = [
            row for row in self.controls if row["record_type"] == "exact_open_lineage_control"
        ]
        self.assertEqual(len(exact), 1)
        self.assertEqual(exact[0]["model"], "Kimi K2.6")
        self.assertAlmostEqual(float(exact[0]["total_parameters_b"]), 1040.0)
        self.assertLess(float(exact[0]["horizon_ratio_vs_previous"]), 0.5)
        decision = self.result["decision"]
        self.assertTrue(decision["apply_fable_mythos_shared_weight_identity"])
        self.assertFalse(decision["treat_opus_fallback_as_shared_base"])


if __name__ == "__main__":
    unittest.main()
