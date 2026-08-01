from __future__ import annotations

import csv
import hashlib
import json
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/019f6c42-2d53-7743-ab07-6293e2618dd7"
METADATA = ROOT / "sources/ikp_source_metadata_2026-07-18.json"
RESULT = OUT / "ikp_conditional_benchmark_signal_audit_2026-07-18.json"
PREDICTIONS = OUT / "ikp_conditional_benchmark_predictions_2026-07-18.csv"
SITE_COPY = ROOT / "site/public/data/ikp-conditional-benchmark-signal.json"
PARAMETER_SIGNAL = OUT / "ikp_parameter_signal_audit_2026-07-18.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class IkpConditionalBenchmarkSignalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metadata = json.loads(METADATA.read_text(encoding="utf-8"))
        cls.result = json.loads(RESULT.read_text(encoding="utf-8"))
        cls.parameter_signal = json.loads(PARAMETER_SIGNAL.read_text(encoding="utf-8"))
        with PREDICTIONS.open(newline="", encoding="utf-8") as handle:
            cls.predictions = list(csv.DictReader(handle))

    def test_all_21_pinned_ikp_files_reconcile(self):
        self.assertEqual(len(self.metadata["files"]), 21)
        labels = {row["label"] for row in self.metadata["files"]}
        self.assertIn("upstream_benchmark_scores", labels)
        self.assertIn("upstream_benchmark_joined_panel", labels)
        self.assertIn("upstream_benchmark_narrative_summary", labels)
        for record in self.metadata["files"]:
            path = ROOT / record["local_path"]
            self.assertEqual(sha256(path), record["sha256"])
            self.assertEqual(path.stat().st_size, record["bytes"])
        for relative, expected in self.result["source_files"].items():
            self.assertEqual(sha256(ROOT / relative), expected)

    def test_raw_vendor_tables_and_generated_upstream_outputs_reproduce(self):
        inventory = self.result["source_inventory"]
        self.assertEqual(inventory["raw_ikp_configurations"], 100)
        self.assertEqual(inventory["benchmark_rows"], 81)
        self.assertEqual(inventory["post_exclusion_configurations"], 93)
        self.assertEqual(inventory["post_collapse_weight_bases"], 87)
        self.assertEqual(inventory["serving_variants_collapsed"], 6)
        raw = inventory["raw_benchmark_provenance"]
        self.assertEqual(raw["benchmark_rows_matched"], 81)
        self.assertEqual(raw["populated_score_cells_matched"], 173)
        self.assertGreaterEqual(raw["distinct_primary_urls"], 50)
        self.assertEqual(raw["mismatches"], [])

        upstream = self.result["upstream_reproduction"]
        self.assertEqual(upstream["joined_panel"]["rows"], 93)
        self.assertTrue(upstream["joined_panel"]["order_identical"])
        self.assertEqual(upstream["joined_panel"]["mismatches"], [])
        self.assertTrue(
            upstream["regression_summary"]["reproduction"][
                "exact_after_upstream_rounding"
            ]
        )
        self.assertEqual(
            upstream["regression_summary"]["reproduction"]["mismatches"], []
        )
        self.assertTrue(
            upstream["time_coefficients"]["reproduction"][
                "exact_after_upstream_rounding"
            ]
        )
        self.assertEqual(
            upstream["time_coefficients"]["reproduction"]["mismatches"], []
        )

    def test_stale_upstream_narrative_is_explicit_not_silently_used(self):
        narrative = self.result["upstream_reproduction"]["narrative_summary_audit"]
        self.assertFalse(narrative["all_claims_match_generated_outputs"])
        self.assertEqual(narrative["stale_claim_count"], 6)
        claims = {row["claim"]: row for row in narrative["stale_claims"]}
        self.assertEqual(claims["headline_full_set_configurations"]["narrative"], 89)
        self.assertEqual(
            claims["headline_full_set_configurations"]["generated_output"], 93
        )
        self.assertEqual(claims["headline_full_set_r_squared"]["narrative"], 0.917)
        self.assertEqual(
            claims["headline_full_set_r_squared"]["generated_output"], 0.9105
        )
        self.assertLess(
            claims["headline_full_set_time_slope_pp_per_month"]["narrative"], 0
        )
        self.assertGreater(
            claims["headline_full_set_time_slope_pp_per_month"][
                "generated_output"
            ],
            0,
        )

    def test_every_prediction_is_strictly_chronological_and_vendor_held_out(self):
        self.assertEqual(len(self.predictions), 196)
        keys = [
            (
                row["benchmark"],
                row["specification"],
                row["training_weighting"],
                row["base_key"],
            )
            for row in self.predictions
        ]
        self.assertEqual(len(keys), len(set(keys)))
        for row in self.predictions:
            self.assertLess(row["train_max_date"], row["release_date"])
            self.assertEqual(row["test_vendor_excluded"], "True")
            self.assertGreaterEqual(int(row["train_rows"]), 10)
            self.assertGreaterEqual(int(row["train_vendors"]), 5)

    def test_gpqa_is_robust_and_mmlu_is_supportive_not_universal(self):
        results = self.result["heldout_results"]
        self.assertEqual(results["gpqa_diamond"]["strict_prediction_models"], 18)
        self.assertEqual(results["gpqa_diamond"]["strict_prediction_vendors"], 9)
        self.assertEqual(results["gpqa_diamond"]["passing_specifications"], 4)
        for specification in results["gpqa_diamond"]["specifications"].values():
            self.assertTrue(specification["passes_predeclared_gate"])
            self.assertLess(
                specification["candidate_with_ikp"]["median_multiplicative_error"],
                specification["baseline"]["median_multiplicative_error"],
            )
            self.assertLess(specification["paired_vendor_bootstrap"]["ci_90"][1], 0)

        self.assertEqual(results["mmlu"]["strict_prediction_models"], 16)
        self.assertEqual(results["mmlu"]["strict_prediction_vendors"], 11)
        self.assertEqual(results["mmlu"]["passing_specifications"], 3)
        weakest = results["mmlu"]["specifications"][
            "vendor_equal__score_date_arch"
        ]
        self.assertFalse(weakest["passes_predeclared_gate"])
        self.assertGreater(weakest["paired_vendor_bootstrap"]["ci_90"][1], 0)

    def test_sparse_benchmarks_are_not_overclaimed(self):
        results = self.result["heldout_results"]
        self.assertEqual(results["mmlu_pro"]["strict_prediction_vendors"], 7)
        self.assertEqual(results["mmlu_pro"]["passing_specifications"], 0)
        self.assertEqual(results["simpleqa"]["strict_prediction_models"], 0)
        self.assertEqual(results["simpleqa"]["passing_specifications"], 0)

    def test_decision_strengthens_validation_without_double_counting(self):
        decision = self.result["decision"]
        self.assertTrue(decision["conditional_incremental_signal_corroborated"])
        self.assertTrue(decision["gpqa_passes_all_four_sensitivity_specifications"])
        self.assertFalse(decision["change_live_ikp_weight"])
        primary = self.parameter_signal["decision"]
        self.assertEqual(
            decision["primary_parameter_signal_promoted"],
            primary["promote_incremental_ikp_weight"],
        )
        self.assertTrue(
            math.isclose(
                decision["retain_current_final_fable_ikp_weight"],
                primary["incremental_final_weight_when_crowd_is_50pct"],
                rel_tol=0,
                abs_tol=1e-15,
            )
        )
        self.assertEqual(RESULT.read_bytes(), SITE_COPY.read_bytes())


if __name__ == "__main__":
    unittest.main()
