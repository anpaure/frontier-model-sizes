import csv
import hashlib
import json
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/019f6c42-2d53-7743-ab07-6293e2618dd7"
DATE = "2026-07-18"
RESULT = OUT / f"openrouter_active_price_audit_{DATE}.json"
MATCHES = OUT / f"openrouter_active_parameter_match_audit_{DATE}.csv"
PREDICTIONS = OUT / f"openrouter_active_price_predictions_{DATE}.csv"
TARGETS = OUT / f"openrouter_active_price_targets_{DATE}.csv"


def rows(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class OpenRouterActivePriceSignalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = json.loads(RESULT.read_text(encoding="utf-8"))
        cls.matches = rows(MATCHES)
        cls.predictions = rows(PREDICTIONS)
        cls.targets = rows(TARGETS)

    def test_exact_identity_inventory_is_complete_and_unique(self):
        inventory = self.result["inventory"]
        self.assertEqual(len(self.matches), 93)
        self.assertEqual(inventory["calibration_rows_audited"], 93)
        self.assertEqual(len({row["canonical_checkpoint_id"] for row in self.matches}), 93)
        disclosed = [
            row for row in self.matches if row["status"] == "active_parameter_match"
        ]
        dense = [
            row
            for row in self.matches
            if row["status"] == "dense_config_active_equals_total"
        ]
        active = disclosed + dense
        self.assertEqual(len(disclosed), 45)
        self.assertEqual(len(dense), 18)
        self.assertEqual(inventory["active_parameter_matches"], 63)
        self.assertEqual(inventory["aa_disclosed_active_parameter_matches"], 45)
        self.assertEqual(inventory["dense_config_active_equals_total_controls"], 18)
        self.assertEqual(inventory["developers"], 15)
        self.assertEqual(len({row["aa_checkpoint_group_id"] for row in active}), 63)
        self.assertEqual(inventory["identity_conflicts"], 0)
        self.assertEqual(inventory["ambiguous_hf_repositories"], 2)
        self.assertEqual(inventory["unresolved_active_hf_ambiguities"], 0)
        self.assertLessEqual(inventory["max_total_parameter_ratio"], 1.07)
        self.assertLessEqual(inventory["max_dense_control_total_parameter_ratio"], 1.08)
        self.assertEqual(
            inventory["match_method_counts"],
            {
                "exact_hf_repo": 28,
                "exact_hf_repo_and_audited_crosswalk": 14,
                "audited_aa_epoch_crosswalk": 2,
                "audited_crosswalk_resolves_multiple_exact_hf_configs": 1,
                "primary_hf_config_dense_active_equals_total": 18,
            },
        )
        for row in disclosed:
            self.assertTrue(row["aa_checkpoint_group_id"])
            self.assertGreater(float(row["aa_active_parameters_b"]), 0)
            self.assertGreater(float(row["epoch_total_parameters_b"]), float(row["aa_active_parameters_b"]))
            self.assertIn("Artificial Analysis", row["active_parameter_source"])
        for row in dense:
            self.assertEqual(row["hf_config_classifications"], "dense_config")
            self.assertFalse(row["aa_active_parameters_b"])
            self.assertIn("active equals Epoch total", row["active_parameter_source"])

    def test_all_source_hashes_reconcile(self):
        for relative, expected in self.result["source_files"].items():
            self.assertEqual(sha256(ROOT / relative), expected, relative)

    def test_every_fold_is_strictly_release_ordered_and_family_held_out(self):
        inventory = self.result["inventory"]
        self.assertEqual(len(self.predictions), 45)
        self.assertEqual(inventory["release_ordered_predictions"], 45)
        self.assertEqual(len({row["developer"] for row in self.predictions}), 11)
        self.assertEqual(inventory["prediction_developers"], 11)
        for row in self.predictions:
            self.assertLess(row["train_max_date"], row["release_date"])
            self.assertEqual(row["test_developer_excluded"], "True")
            self.assertEqual(row["current_price_snapshot_not_historical"], "True")
            self.assertGreaterEqual(int(row["train_n"]), 12)
            self.assertGreaterEqual(int(row["train_developers"]), 5)
            for prefix, actual in (("active", "actual_active_b"), ("total", "actual_total_b")):
                for feature in ("date_price", "score_date", "score_date_price"):
                    predicted = float(row[f"predicted_{prefix}_{feature}_b"])
                    expected_error = math.log10(predicted / float(row[actual]))
                    self.assertAlmostEqual(float(row[f"{prefix}_{feature}_log10_error"]), expected_error, places=12)

    def test_active_price_is_easier_but_price_increment_is_not_decisive(self):
        comparisons = self.result["active_vs_total_predictability"]
        for feature in ("date_price", "score_date", "score_date_price"):
            self.assertLess(
                comparisons[feature]["active"]["median_multiplicative_error"],
                comparisons[feature]["total"]["median_multiplicative_error"],
            )
        incremental = comparisons["price_incremental_to_score_date"]
        self.assertLess(incremental["active"]["observed_delta"], 0)
        self.assertGreater(incremental["active"]["ci_90"][1], 0)
        self.assertLess(abs(incremental["total"]["observed_delta"]), 0.01)
        self.assertGreater(incremental["total"]["ci_90"][1], 0)

    def test_high_sparsity_transport_fails_predeclared_promotion_gate(self):
        inventory = self.result["inventory"]
        transport = self.result["high_sparsity_total_transport"]
        candidate = transport["candidate"]
        baseline = transport["direct_total_baseline"]
        bootstrap = transport["paired_cluster_bootstrap"]
        gates = self.result["promotion_gates"]
        self.assertEqual(candidate["n"], 16)
        self.assertEqual(inventory["high_sparsity_transport_predictions"], 16)
        self.assertEqual(bootstrap["developers"], 7)
        self.assertEqual(inventory["high_sparsity_transport_developers"], 7)
        for metric in ("median_multiplicative_error", "mean_absolute_log10_error", "p80_multiplicative_error"):
            self.assertLess(candidate[metric], baseline[metric])
        self.assertLess(bootstrap["observed_delta"], 0)
        self.assertGreater(bootstrap["ci_90"][1], 0)
        self.assertIs(gates["performance_gate_passed"], False)
        self.assertIs(gates["coverage_gate_passed"], False)
        self.assertEqual((gates["observed_tests"], gates["required_tests"]), (16, 20))
        self.assertEqual((gates["observed_developers"], gates["required_developers"]), (7, 8))

    def test_target_sensitivities_are_k3_anchored_and_extrapolative(self):
        by_model = {row["model"]: row for row in self.targets}
        self.assertEqual(set(by_model), {"Claude Fable 5", "GPT-5.6 Sol", "Kimi K3"})
        self.assertAlmostEqual(float(by_model["Kimi K3"]["k3_anchored_total_date_price_t"]), 2.78, places=12)
        self.assertAlmostEqual(float(by_model["Kimi K3"]["k3_anchored_total_score_date_price_t"]), 2.78, places=12)
        for row in by_model.values():
            self.assertAlmostEqual(
                float(row["k3_anchored_total_date_price_t"]),
                2.78
                * float(row["predicted_active_date_price_b"])
                / float(by_model["Kimi K3"]["predicted_active_date_price_b"]),
                places=12,
            )
            self.assertAlmostEqual(
                float(row["k3_anchored_total_score_date_price_t"]),
                2.78
                * float(row["predicted_active_score_date_price_b"])
                / float(by_model["Kimi K3"]["predicted_active_score_date_price_b"]),
                places=12,
            )
        self.assertGreater(float(by_model["Claude Fable 5"]["price_over_training_max"]), 6)
        self.assertGreater(float(by_model["GPT-5.6 Sol"]["price_over_training_max"]), 3)
        self.assertEqual(by_model["Claude Fable 5"]["status"], "0%-weight extrapolative sensitivity")
        self.assertEqual(self.result["metadata"]["generated_on"], "2026-07-31")
        self.assertEqual(
            self.result["metadata"]["compatibility_filename_date"], DATE
        )

    def test_decision_is_zero_weight_and_caveat_is_explicit(self):
        metadata = self.result["metadata"]
        decision = self.result["decision"]
        self.assertIn("not a genuinely historical prospective backtest", metadata["critical_nonprospective_caveat"])
        self.assertIs(decision["promote_active_price_transport"], False)
        self.assertEqual(decision["incremental_live_weight"], 0)
        self.assertIs(decision["replace_existing_price_branch"], False)
        self.assertIs(decision["change_headline_forecasts"], False)


if __name__ == "__main__":
    unittest.main()
