from __future__ import annotations

import csv
import hashlib
import json
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/019f6c42-2d53-7743-ab07-6293e2618dd7"
RESULT = OUT / "posttraining_lineage_audit_2026-07-18.json"
EDGES = OUT / "posttraining_lineage_edges_2026-07-18.csv"
MEASUREMENTS = OUT / "posttraining_lineage_measurements_2026-07-18.csv"
PREDICTIONS = OUT / "posttraining_lineage_predictions_2026-07-18.csv"
FRONTIER = OUT / "frontier_shared_base_sensitivity_2026-07-18.csv"
EVIDENCE = OUT / "frontier_lineage_evidence_2026-07-18.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


class PosttrainingLineageSignalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = json.loads(RESULT.read_text(encoding="utf-8"))
        cls.edges = rows(EDGES)
        cls.measurements = rows(MEASUREMENTS)
        cls.predictions = rows(PREDICTIONS)
        cls.frontier = rows(FRONTIER)
        cls.evidence = rows(EVIDENCE)

    def test_epoch_lineage_inventory_and_admission_are_exact(self) -> None:
        inventory = self.result["inventory"]
        self.assertEqual(inventory["epoch_rows"], 3574)
        self.assertEqual(inventory["base_model_links"], 686)
        self.assertEqual(inventory["unique_exact_parent_matches"], 630)
        self.assertEqual(inventory["same_parameter_links_1pct"], 342)
        self.assertEqual(inventory["same_parameter_both_open_links"], 240)
        self.assertEqual(inventory["candidate_open_language_same_parameter_links"], 212)
        self.assertEqual(len(self.edges), inventory["admitted_measured_lineage_edges"])
        self.assertEqual(len({row["edge_id"] for row in self.edges}), len(self.edges))
        self.assertEqual(len({row["base_cluster_id"] for row in self.edges}), 6)
        self.assertEqual(len({row["child_organization"] for row in self.edges}), 5)
        self.assertEqual(
            sum(bool(row["finetune_compute_flop"]) for row in self.edges),
            inventory["edges_with_finetune_compute"],
        )
        self.assertEqual(
            {row["edge_id"] for row in self.edges},
            {
                "checkpoint:moonshot:kimi-k2-6->checkpoint:moonshot:kimi-k2-7-code",
                "checkpoint:moonshot:kimi-k2-5->checkpoint:moonshot:kimi-k2-6",
                "checkpoint:epoch:qwen3-next-80b-a3b->checkpoint:epoch:qwen3-coder-next",
                "checkpoint:deepseek:deepseek-v3->checkpoint:epoch:deepseek-v3-1",
                "checkpoint:mistral:mistral-small-3-1->checkpoint:mistral:magistral-small-1-1",
                "checkpoint:deepseek:deepseek-v3->checkpoint:deepseek:deepseek-r1-may-2025",
                "checkpoint:deepseek:deepseek-v3->checkpoint:deepseek:deepseek-r1",
                "checkpoint:meta:llama-2-70b->checkpoint:stability-ai:stable-beluga-2",
            },
        )
        self.assertIn("DeepSeek-V3.1", {row["child_model"] for row in self.edges})
        for row in self.edges:
            ratio = float(row["parameter_ratio_child_over_parent"])
            self.assertLessEqual(abs(ratio - 1), 0.01)
            self.assertLessEqual(row["parent_release_date"], row["child_release_date"])
            self.assertEqual(
                row["admission_status"],
                "admitted_exact_open_same_parameter_lineage",
            )

    def test_measurements_are_unique_and_reconcile_to_raw_differences(self) -> None:
        self.assertEqual(
            len(self.measurements), self.result["inventory"]["matched_measurements"]
        )
        keys = [
            (row["edge_id"], row["metric"], row["benchmark"])
            for row in self.measurements
        ]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(
            sum(row["metric"] == "ECI component" for row in self.measurements),
            37,
        )
        self.assertEqual(
            len(
                {
                    row["edge_id"]
                    for row in self.measurements
                    if row["metric"] == "No-CoT horizon"
                }
            ),
            1,
        )
        self.assertFalse(
            any(row["metric"] == "METR p50 horizon" for row in self.measurements)
        )
        for row in self.measurements:
            expected = float(row["child_value"]) - float(row["parent_value"])
            self.assertTrue(
                math.isclose(
                    float(row["difference_child_minus_parent"]),
                    expected,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            )
            if row["metric"] == "ECI component":
                expected_eci = float(row["child_component_implied_eci"]) - float(
                    row["parent_component_implied_eci"]
                )
                self.assertTrue(
                    math.isclose(
                        float(row["component_implied_eci_delta"]),
                        expected_eci,
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                )

    def test_prediction_folds_are_strict_and_arithmetic_reconciles(self) -> None:
        by_signal = {}
        for row in self.predictions:
            by_signal[row["signal"]] = by_signal.get(row["signal"], 0) + 1
            self.assertLess(row["train_max_date"], row["parent_release_date"])
            self.assertEqual(row["test_group_excluded"], "True")
            actual = float(row["actual_parameters_b"])
            for branch in ("baseline", "collapsed", "parent_only"):
                predicted = float(row[f"{branch}_predicted_b"])
                expected_error = math.log10(predicted) - math.log10(actual)
                self.assertTrue(
                    math.isclose(
                        float(row[f"{branch}_log10_error"]),
                        expected_error,
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                )
            self.assertGreater(
                float(row["implied_child_over_parent_parameter_ratio"]), 0
            )
        self.assertEqual(
            by_signal,
            {
                "ECI aggregate": self.result["inventory"]["eci_prediction_edges"],
                "AA Intelligence Index": self.result["inventory"]["aa_prediction_edges"],
            },
        )

    def test_backtest_support_is_real_but_not_promotable(self) -> None:
        eci = self.result["lineage_backtests"]["eci"]
        aa = self.result["lineage_backtests"]["aa"]
        self.assertEqual(eci["rows"], self.result["inventory"]["eci_prediction_edges"])
        self.assertEqual(eci["bases"], 5)
        self.assertGreater(eci["collapsed_vs_baseline"]["ci_90"][1], 0)
        self.assertGreater(eci["median_implied_child_over_parent_parameter_ratio"], 1)
        self.assertEqual(aa["rows"], 3)
        self.assertEqual(aa["bases"], 3)
        # The promotion statistic is equal-base mean absolute log10 error.  Do
        # not require every descriptive summary (especially a three-row
        # median) to move in the same direction after source-timing changes.
        self.assertLess(
            aa["collapsed"]["mean_absolute_log10_error"],
            aa["baseline"]["mean_absolute_log10_error"],
        )
        self.assertLess(
            aa["collapsed"]["p80_multiplicative_error"],
            aa["baseline"]["p80_multiplicative_error"],
        )
        self.assertLess(aa["collapsed_vs_baseline"]["ci_90"][1], 0)
        gates = self.result["promotion_gates"]
        self.assertIs(gates["verified_open_lineage_bases_at_least_8"], False)
        self.assertIs(gates["eci_signal_bases_at_least_6"], False)
        self.assertIs(gates["aa_signal_bases_at_least_6"], False)
        self.assertIs(
            gates["proprietary_shared_base_claims_publicly_verified"], False
        )
        self.assertIs(
            self.result["decision"]["promote_posttraining_correction"], False
        )
        self.assertEqual(self.result["decision"]["incremental_live_weight"], 0)
        self.assertIs(self.result["decision"]["change_headline_forecasts"], False)

    def test_same_checkpoint_reasoning_control_is_preserved(self) -> None:
        control = self.result["hard_same_checkpoint_control"][
            "open_weight_reasoning_pairs"
        ]
        self.assertEqual(control["pairs"], 53)
        self.assertEqual(control["creators"], 12)
        self.assertTrue(
            math.isclose(
                control["equal_creator_median_aa_uplift"],
                5.50904364575783,
                rel_tol=1e-12,
            )
        )
        low, high = control["equal_creator_median_bootstrap_90_ci"]
        self.assertGreater(low, 0)
        self.assertGreater(high, low)

    def test_component_category_result_is_a_diagnostic_not_a_branch(self) -> None:
        component = self.result["component_posttraining_sensitivity"]
        knowledge = component["categories"]["knowledge"]
        other = component["categories"]["other"]
        self.assertEqual(knowledge["bases"], 5)
        self.assertEqual(other["bases"], 4)
        self.assertLess(
            knowledge["median_component_implied_eci_delta"],
            other["median_component_implied_eci_delta"],
        )
        self.assertEqual(component["incremental_live_weight"], 0)

    def test_frontier_claims_are_labeled_and_extrapolation_is_visible(self) -> None:
        self.assertEqual(len(self.frontier), 18)
        self.assertEqual(len({row["chain"] for row in self.frontier}), 2)
        for row in self.frontier:
            self.assertEqual(
                row["evidence_grade"], "user_asserted_not_publicly_disclosed"
            )
            self.assertEqual(float(row["live_weight"]), 0)
            if row["score_extrapolates_above_calibration_max"] == "True":
                self.assertGreater(float(row["score_over_calibration_max"]), 1)
        latest_gpt = max(
            (
                row
                for row in self.frontier
                if row["chain"] == "GPT-5 through GPT-5.5"
                and row["mode"] == "max_reasoning"
            ),
            key=lambda row: int(row["sequence"]),
        )
        self.assertGreater(
            float(latest_gpt["date_adjusted_implied_parameter_ratio_vs_first"]),
            10,
        )

    def test_primary_source_evidence_does_not_overstate_identity(self) -> None:
        self.assertEqual(len(self.evidence), 4)
        by_lineage = {row["lineage"]: row for row in self.evidence}
        self.assertEqual(
            by_lineage["GPT-5.5 and GPT-5.5 Pro"]["evidence_grade"],
            "primary_source_explicit",
        )
        self.assertIn(
            "do not extend",
            by_lineage["GPT-5.5 to GPT-5.6 Sol/Terra"]["model_treatment"],
        )
        self.assertEqual(
            by_lineage["Claude Opus 4.5-4.8"]["evidence_grade"],
            "user_asserted_not_publicly_disclosed",
        )

    def test_source_hashes_reconcile(self) -> None:
        for relative, digest in self.result["source_files"].items():
            path = ROOT / relative
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)


if __name__ == "__main__":
    unittest.main()
