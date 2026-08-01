from __future__ import annotations

import csv
import hashlib
import json
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/019f6c42-2d53-7743-ab07-6293e2618dd7"
RESULT = OUT / "eci_multivariate_component_audit_2026-07-18.json"
PREDICTIONS = OUT / "eci_multivariate_component_predictions_2026-07-18.csv"
NARROW_CI_PREDICTIONS = (
    OUT / "eci_multivariate_component_narrow_eci_ci_predictions_2026-07-18.csv"
)
TARGETS = OUT / "eci_multivariate_component_targets_2026-07-18.csv"
COVERAGE = OUT / "eci_multivariate_component_coverage_2026-07-18.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


class ECIMultivariateComponentAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = json.loads(RESULT.read_text(encoding="utf-8"))
        cls.predictions = rows(PREDICTIONS)
        cls.narrow_ci_predictions = rows(NARROW_CI_PREDICTIONS)
        cls.targets = rows(TARGETS)
        cls.coverage = rows(COVERAGE)

    def test_inventory_and_component_coverage_are_exact(self) -> None:
        inventory = self.result["inventory"]
        self.assertEqual(inventory["parameter_map_checkpoints"], 89)
        self.assertEqual(inventory["parameter_map_families"], 40)
        self.assertEqual(inventory["narrow_eci_ci_checkpoints"], 41)
        self.assertEqual(inventory["broad_eci_ci_checkpoints"], 48)
        self.assertEqual(inventory["unique_component_measurements"], 723)
        self.assertEqual(inventory["component_benchmarks"], 50)
        self.assertEqual(len(self.coverage), 50)
        self.assertEqual(len({row["benchmark"] for row in self.coverage}), 50)
        self.assertEqual(sum(int(row["models"]) for row in self.coverage), 723)

    def test_primary_outer_predictions_are_strict_and_unique(self) -> None:
        inventory = self.result["inventory"]
        self.assertEqual(len(self.predictions), inventory["outer_predictions"])
        self.assertEqual(len(self.predictions), 73)
        self.assertEqual(len({row["family"] for row in self.predictions}), 34)
        keys = [
            (row["training_scope"], row["release_date"], row["model"])
            for row in self.predictions
        ]
        self.assertEqual(len(keys), len(set(keys)))
        for row in self.predictions:
            self.assertEqual(row["training_scope"], "all_half_weight")
            self.assertLess(row["train_max_date"], row["release_date"])
            self.assertEqual(row["test_family_excluded"], "True")
            self.assertGreaterEqual(int(row["train_n"]), 12)
            self.assertGreaterEqual(int(row["train_families"]), 5)

    def test_narrow_eci_ci_replication_never_uses_broad_ci_outcomes(self) -> None:
        inventory = self.result["inventory"]
        self.assertEqual(
            len(self.narrow_ci_predictions),
            inventory["narrow_eci_ci_only_outer_predictions"],
        )
        self.assertEqual(len(self.narrow_ci_predictions), 28)
        self.assertEqual(
            len({row["family"] for row in self.narrow_ci_predictions}), 11
        )
        for row in self.narrow_ci_predictions:
            self.assertEqual(row["training_scope"], "narrow_eci_ci_only")
            self.assertEqual(row["broad_eci_ci"], "0")
            self.assertLess(row["train_max_date"], row["release_date"])
            self.assertEqual(row["test_family_excluded"], "True")

    def test_uncertainty_flag_is_not_parameter_disclosure_metadata(self) -> None:
        regression = json.loads(
            (ROOT / "regression_results.json").read_text(
                encoding="utf-8"
            )
        )
        for row in regression["eci"]["open_models"]:
            expected = int(float(row["ci_width"]) > 10.0)
            self.assertEqual(int(row["estimated"]), expected)
            self.assertEqual(int(row["broad_eci_ci"]), expected)
            self.assertEqual(row["parameter_disclosure_status"], "not classified")
        limitations = " ".join(self.result["limitations"])
        self.assertIn("score uncertainty", limitations)
        self.assertIn("No parameter-disclosure classification", limitations)

    def test_nested_policy_fields_and_error_arithmetic_reconcile(self) -> None:
        allowed_sets = set(self.result["predeclared_model"]["feature_sets"])
        allowed_alphas = {
            float(value) for value in self.result["predeclared_model"]["ridge_alphas"]
        }
        for row in self.predictions + self.narrow_ci_predictions:
            for target in ("active", "total"):
                selected = [
                    value
                    for value in row[f"{target}_selected_benchmarks"].split("|")
                    if value
                ]
                observed = [
                    value
                    for value in row[
                        f"{target}_observed_selected_benchmarks"
                    ].split("|")
                    if value
                ]
                self.assertIn(row[f"{target}_feature_set"], allowed_sets)
                self.assertIn(float(row[f"{target}_alpha"]), allowed_alphas)
                self.assertEqual(
                    len(selected), int(row[f"{target}_selected_benchmark_count"])
                )
                self.assertEqual(
                    len(observed), int(row[f"{target}_observed_selected_count"])
                )
                self.assertTrue(set(observed).issubset(selected))
                actual = float(row[f"actual_{target}_b"])
                for branch in ("baseline", "candidate"):
                    predicted = float(row[f"{branch}_{target}_predicted_b"])
                    expected_error = math.log10(predicted) - math.log10(actual)
                    self.assertTrue(
                        math.isclose(
                            float(row[f"{branch}_{target}_log10_error"]),
                            expected_error,
                            rel_tol=1e-12,
                            abs_tol=1e-12,
                        )
                    )

    def test_point_gains_do_not_pass_interval_or_promotion_gates(self) -> None:
        primary = self.result["backtest"]["total"]["all"]
        narrow_ci = self.result["narrow_eci_ci_only_training_backtest"]["total"]
        self.assertLess(
            primary["candidate"]["median_multiplicative_error"],
            primary["baseline"]["median_multiplicative_error"],
        )
        for comparison in (primary, narrow_ci):
            self.assertLess(
                comparison["candidate"]["mean_absolute_log10_error"],
                comparison["baseline"]["mean_absolute_log10_error"],
            )
            self.assertLess(
                comparison["candidate"]["rmse_log10"],
                comparison["baseline"]["rmse_log10"],
            )
            self.assertGreater(
                comparison["paired_family_bootstrap"]["ci_90"][1], 0
            )
        self.assertGreaterEqual(
            narrow_ci["candidate"]["median_multiplicative_error"],
            narrow_ci["baseline"]["median_multiplicative_error"],
        )
        gates = self.result["promotion_gates"]
        self.assertIs(gates["all_point_metrics_improve"], True)
        self.assertIs(
            gates["narrow_eci_ci_only_training_point_metrics_improve"], False
        )
        self.assertIs(gates["all_equal_family_ci_wholly_favorable"], False)
        self.assertIs(gates["narrow_eci_ci_only_training_ci_wholly_favorable"], False)
        self.assertIs(gates["target_full_vs_narrow_eci_ci_adjustment_stable"], False)
        self.assertIs(gates["target_component_coverage_gate"], False)
        decision = self.result["decision"]
        self.assertIs(decision["promote_multivariate_component_branch"], False)
        self.assertEqual(decision["incremental_live_weight"], 0)
        self.assertIs(decision["change_headline_forecasts"], False)

    def test_target_predictions_are_honest_sparse_and_zero_weight(self) -> None:
        self.assertEqual({row["model"] for row in self.targets}, {"Claude Fable 5", "GPT-5.6 Sol"})
        by_model = {row["model"]: row for row in self.targets}
        for row in self.targets:
            self.assertLess(row["full_training_max_date"], row["release_date"])
            self.assertEqual(row["target_family_excluded"], "True")
            self.assertEqual(float(row["incremental_live_weight"]), 0)
            full_adjustment = float(row["component_adjustment_factor"])
            narrow_ci_adjustment = float(
                row["narrow_ci_training_adjustment_factor"]
            )
            self.assertTrue(
                math.isclose(
                    full_adjustment,
                    float(row["raw_multivariate_candidate_t"])
                    / float(row["raw_multivariate_baseline_t"]),
                    rel_tol=1e-12,
                )
            )
            self.assertTrue(
                math.isclose(
                    narrow_ci_adjustment,
                    float(row["narrow_ci_training_raw_candidate_t"])
                    / float(row["narrow_ci_training_raw_baseline_t"]),
                    rel_tol=1e-12,
                )
            )
            expected_ratio = max(full_adjustment, narrow_ci_adjustment) / min(
                full_adjustment, narrow_ci_adjustment
            )
            self.assertTrue(
                math.isclose(
                    float(row["full_vs_narrow_ci_adjustment_ratio"]),
                    expected_ratio,
                    rel_tol=1e-12,
                )
            )
        self.assertEqual(int(by_model["Claude Fable 5"]["observed_selected_count"]), 1)
        self.assertEqual(
            by_model["Claude Fable 5"]["full_vs_narrow_ci_direction_agrees"],
            "False",
        )
        self.assertEqual(int(by_model["GPT-5.6 Sol"]["observed_selected_count"]), 2)

    def test_source_hashes_reconcile(self) -> None:
        for relative, digest in self.result["source_files"].items():
            path = ROOT / relative
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)


if __name__ == "__main__":
    unittest.main()
