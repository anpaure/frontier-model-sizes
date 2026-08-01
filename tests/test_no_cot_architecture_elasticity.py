import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "019f6c42-2d53-7743-ab07-6293e2618dd7"
RESULT = OUT / "no_cot_architecture_elasticity_audit_2026-07-18.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class NoCotArchitectureElasticityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = json.loads(RESULT.read_text(encoding="utf-8"))

    def test_full_open_weight_panel_is_exact_and_day_level(self):
        inventory = self.result["inventory"]
        self.assertEqual(inventory["models"], 35)
        self.assertEqual(inventory["families"], 6)
        self.assertEqual(inventory["dense_models"], 19)
        self.assertEqual(inventory["moe_models"], 16)
        self.assertEqual(inventory["day_precision_dates"], 35)

    def test_paper_architecture_factors_are_reproduced(self):
        reproduction = self.result["paper_relationship_reproduction"]
        self.assertEqual(reproduction["dense"]["pareto_n"], 8)
        self.assertEqual(reproduction["moe"]["pareto_n"], 5)
        self.assertAlmostEqual(
            reproduction["dense"]["deterministic_bootstrap_median_reproduction"],
            2.176488788489135,
            places=10,
        )
        self.assertAlmostEqual(
            reproduction["moe"]["deterministic_bootstrap_median_reproduction"],
            8.132252384929462,
            places=10,
        )

    def test_architecture_specific_predictive_slope_does_not_promote(self):
        comparison = self.result["paired_comparisons"][
            "chronological_developer_holdout"
        ]["direct_architecture_minus_pooled"]
        self.assertEqual(comparison["n"], 16)
        self.assertGreater(comparison["observed_delta"], 0)
        self.assertGreater(comparison["ci_90"][0], 0)
        self.assertLess(comparison["probability_architecture_specific_better"], 0.1)
        pareto = self.result["paired_comparisons"][
            "chronological_developer_holdout"
        ]["training_pareto_architecture_minus_pooled"]
        self.assertLess(pareto["observed_delta"], 0)
        self.assertLess(pareto["ci_90"][0], 0)
        self.assertGreater(pareto["ci_90"][1], 0)

    def test_same_parameter_controls_block_elasticity_identity(self):
        controls = self.result["same_parameter_controls"]
        kimi = next(row for row in controls if row["family"] == "kimi")
        self.assertEqual(kimi["total_parameters_b"], 1040.0)
        self.assertGreater(kimi["horizon_max_over_min"], 2.0)
        self.assertFalse(
            self.result["decision"]["replace_pooled_live_elasticity_with_moe_specific"]
        )
        self.assertFalse(self.result["decision"]["change_headline_forecasts"])

    def test_raw_paper_labels_and_canonical_truth_are_not_conflated(self):
        overlays = self.result["parameter_truth_overlays_in_panel"]
        kimi = [
            row
            for row in overlays
            if row["parameter_truth_id"]
            == "moonshot-kimi-k2-family-report-table-1"
        ]
        self.assertGreaterEqual(len(kimi), 3)
        self.assertTrue(all(row["raw_total_parameters_b"] == 1000.0 for row in kimi))
        self.assertTrue(
            all(row["canonical_total_parameters_b"] == 1040.0 for row in kimi)
        )
        canonical = self.result["canonical_parameter_relationship_sensitivity"]
        self.assertAlmostEqual(
            canonical["moe"]["factor_per_horizon_doubling"],
            8.319917432714167,
            places=10,
        )
        self.assertNotAlmostEqual(
            canonical["moe"]["factor_per_horizon_doubling"],
            self.result["paper_relationship_reproduction"]["moe"][
                "deterministic_bootstrap_median_reproduction"
            ],
            places=6,
        )

    def test_source_hashes_reconcile(self):
        for relative, expected in self.result["source_hashes"].items():
            self.assertEqual(sha256(ROOT / relative), expected)


if __name__ == "__main__":
    unittest.main()
