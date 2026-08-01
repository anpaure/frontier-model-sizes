from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import unittest

import analyze_active_parameter_shrinkage_challenger as audit


class ActiveParameterShrinkageChallengerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result, cls.rows = audit.build_audit()

    def test_committed_outputs_are_byte_exact(self) -> None:
        expected_json = json.dumps(self.result, indent=2) + "\n"
        self.assertEqual(expected_json, audit.RESULT.read_text(encoding="utf-8"))

        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(
            buffer,
            fieldnames=audit.PREDICTION_FIELDS,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(
            {field: row.get(field, "") for field in audit.PREDICTION_FIELDS}
            for row in self.rows
        )
        self.assertEqual(buffer.getvalue(), audit.PREDICTIONS.read_text(encoding="utf-8"))

    def test_every_source_and_nested_fold_is_strict(self) -> None:
        for test in self.rows:
            self.assertTrue(test["source_test_developer_excluded"])
            self.assertLess(
                test["source_train_max_date"],
                test["prediction_information_date"],
            )
            self.assertGreaterEqual(
                test["actual_total_to_active_ratio"],
                audit.HIGH_SPARSITY_THRESHOLD,
            )
            if not test["nested_eligible"]:
                continue
            prior = [
                row
                for row in self.rows
                if row["parameter_training_eligibility_date"]
                < test["prediction_information_date"]
                and row["developer"] != test["developer"]
            ]
            self.assertEqual(len(prior), test["nested_meta_train_n"])
            self.assertEqual(
                len({row["developer"] for row in prior}),
                test["nested_meta_train_developers"],
            )
            self.assertFalse(
                any(row["developer"] == test["developer"] for row in prior)
            )
            self.assertEqual(
                max(row["parameter_training_eligibility_date"] for row in prior),
                test["nested_meta_train_max_eligibility_date"],
            )
            self.assertLess(
                test["nested_meta_train_max_eligibility_date"],
                test["prediction_information_date"],
            )

    def test_empirical_gates_pass_but_live_applicability_fails(self) -> None:
        gates = self.result["promotion_gates"]
        self.assertTrue(all(gates["empirical"].values()))
        self.assertFalse(any(gates["applicability_and_prospective"].values()))
        self.assertFalse(gates["all_gates_pass"])
        decision = self.result["decision"]
        self.assertFalse(decision["promote_to_live_factor"])
        self.assertEqual(decision["incremental_live_weight"], 0.0)
        self.assertFalse(decision["change_headline_forecasts"])
        self.assertTrue(decision["preserve_as_challenger"])

    def test_fixed_and_nested_results_improve_declared_metrics(self) -> None:
        fixed = self.result["fixed_50_50_evaluation"]
        nested = self.result["nested_weight_evaluation"]
        for evaluation in (fixed, nested):
            for cohort in ("all_high_sparsity", "frontier_like_high_sparsity"):
                observed = evaluation[cohort]
                self.assertLess(
                    observed["candidate"]["mean_absolute_log10_error"],
                    observed["baseline"]["mean_absolute_log10_error"],
                )
                self.assertLess(
                    observed["paired_developer_bootstrap"]["ci_90"][1], 0
                )
        for cohort in ("all_high_sparsity", "frontier_like_high_sparsity"):
            observed = fixed[cohort]
            self.assertLess(
                observed["candidate"]["median_multiplicative_error"],
                observed["baseline"]["median_multiplicative_error"],
            )
            self.assertGreater(
                observed["candidate"]["within_2x"],
                observed["baseline"]["within_2x"],
            )

    def test_k3_uses_exact_primary_truth_and_is_not_called_prospective(self) -> None:
        k3 = self.result["kimi_k3_external_check"]
        self.assertEqual(k3["disclosed_total_b"], 2780.0)
        self.assertEqual(k3["disclosed_active_b"], 104.2)
        self.assertTrue(k3["source_component_developer_excluded"])
        self.assertLess(k3["source_train_max_date"], k3["release_date"])
        self.assertLess(k3["fixed_50_50_error_factor"], 1.03)
        self.assertIn("retrospective", k3["status"])
        k3_row = next(row for row in self.rows if row["model"] == "Kimi K3")
        self.assertEqual(k3_row["source_input_actual_total_b"], 2800.0)
        self.assertEqual(k3_row["source_input_actual_active_b"], 104.0)
        self.assertEqual(k3_row["actual_total_b"], 2780.0)
        self.assertEqual(k3_row["actual_active_b"], 104.2)
        self.assertTrue(k3_row["actual_value_override"])
        expected = math.sqrt(
            k3["direct_total_predicted_b"] * k3["active_transport_predicted_b"]
        )
        self.assertTrue(
            math.isclose(expected, k3["fixed_50_50_predicted_b"], abs_tol=1e-9)
        )

    def test_source_hashes_and_target_non_use(self) -> None:
        for relative, declared in self.result["source_files"].items():
            path = audit.ROOT / relative
            observed = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(observed, declared)
        forbidden = {"Claude Fable 5", "GPT-5.6 Sol", "Claude Opus 5"}
        self.assertTrue(forbidden.isdisjoint({row["model"] for row in self.rows}))


if __name__ == "__main__":
    unittest.main()
