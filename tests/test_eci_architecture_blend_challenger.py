from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import unittest
from collections import defaultdict

import numpy as np

import analyze_eci_architecture_blend_challenger as audit


class EciArchitectureBlendChallengerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result, cls.rows = audit.build_audit()

    def test_committed_outputs_are_byte_exact(self) -> None:
        self.assertEqual(
            json.dumps(self.result, indent=2) + "\n",
            audit.RESULT.read_text(encoding="utf-8"),
        )
        self.assertEqual(
            audit.render_csv(self.rows),
            audit.PREDICTIONS.read_text(encoding="utf-8"),
        )
        self.assertEqual(
            audit.render_report(self.result),
            audit.REPORT.read_text(encoding="utf-8"),
        )

    def test_outer_folds_are_strict_whole_developer_holdouts(self) -> None:
        self.assertEqual(len(self.rows), 69)
        self.assertEqual(len({row["developer"] for row in self.rows}), 13)
        keys = [(row["release_date"], row["model"]) for row in self.rows]
        self.assertEqual(len(keys), len(set(keys)))
        for row in self.rows:
            self.assertTrue(row["source_test_developer_excluded"])
            self.assertLess(
                row["source_train_max_eligibility_date"],
                row["prediction_information_date"],
            )
            self.assertGreaterEqual(
                row["source_train_n"], audit.MIN_OUTER_TRAIN_ROWS
            )
            self.assertGreaterEqual(
                row["source_train_developers"],
                audit.MIN_OUTER_TRAIN_DEVELOPERS,
            )

    def test_fixed_blends_recompute_exactly_in_log_space(self) -> None:
        for row in self.rows:
            actual = float(row["actual_b"])
            baseline_log = (
                audit.SCORE_ONLY_WEIGHT
                * math.log10(float(row["score_only_predicted_b"]))
                + audit.ARCHITECTURE_COMPONENT_WEIGHT
                * math.log10(float(row["score_date_predicted_b"]))
            )
            challenger_log = (
                audit.SCORE_ONLY_WEIGHT
                * math.log10(float(row["score_only_predicted_b"]))
                + audit.ARCHITECTURE_COMPONENT_WEIGHT
                * math.log10(float(row["architecture_component_predicted_b"]))
            )
            self.assertTrue(
                math.isclose(
                    float(row["baseline_predicted_b"]),
                    10**baseline_log,
                    rel_tol=1e-12,
                )
            )
            self.assertTrue(
                math.isclose(
                    float(row["challenger_predicted_b"]),
                    10**challenger_log,
                    rel_tol=1e-12,
                )
            )
            self.assertTrue(
                math.isclose(
                    float(row["baseline_log10_error"]),
                    baseline_log - math.log10(actual),
                    abs_tol=1e-12,
                )
            )
            self.assertTrue(
                math.isclose(
                    float(row["challenger_log10_error"]),
                    challenger_log - math.log10(actual),
                    abs_tol=1e-12,
                )
            )

    def test_nested_selection_uses_only_earlier_heldout_errors(self) -> None:
        eligible = [row for row in self.rows if row["nested_eligible"]]
        self.assertEqual(len(eligible), 57)
        self.assertEqual(len({row["developer"] for row in eligible}), 12)
        self.assertEqual(
            self.result["nested_evaluation"]["selection_counts"],
            {"baseline": 5, "challenger": 52},
        )
        for test in eligible:
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
            self.assertLess(
                test["nested_meta_train_max_eligibility_date"],
                test["prediction_information_date"],
            )
            grouped: dict[str, dict[str, list[float]]] = defaultdict(
                lambda: {"baseline": [], "challenger": []}
            )
            for row in prior:
                grouped[row["developer"]]["baseline"].append(
                    abs(float(row["baseline_log10_error"]))
                )
                grouped[row["developer"]]["challenger"].append(
                    abs(float(row["challenger_log10_error"]))
                )
            baseline_mae = float(
                np.mean([np.mean(value["baseline"]) for value in grouped.values()])
            )
            challenger_mae = float(
                np.mean(
                    [np.mean(value["challenger"]) for value in grouped.values()]
                )
            )
            expected = "challenger" if challenger_mae < baseline_mae else "baseline"
            self.assertEqual(test["nested_selected_specification"], expected)
            self.assertTrue(
                math.isclose(
                    test["nested_selected_predicted_b"],
                    test[f"{expected}_predicted_b"],
                    abs_tol=1e-12,
                )
            )

    def test_retrospective_gain_does_not_pass_promotion_gates(self) -> None:
        fixed = self.result["fixed_evaluation"]
        nested = self.result["nested_evaluation"]
        self.assertLess(
            fixed["all"]["challenger"]["mean_absolute_log10_error"],
            fixed["all"]["baseline"]["mean_absolute_log10_error"],
        )
        self.assertLess(
            fixed["all"]["paired_developer_bootstrap"]["ci_90"][1], 0
        )
        self.assertLess(
            fixed["frontier_like"]["paired_developer_bootstrap"]["ci_90"][1],
            0,
        )
        self.assertLess(
            nested["all"]["paired_developer_bootstrap"]["ci_90"][1], 0
        )
        self.assertGreaterEqual(
            nested["frontier_like"]["paired_developer_bootstrap"]["ci_90"][1],
            0,
        )
        gates = self.result["promotion_gates"]
        self.assertFalse(gates["all_pass"])
        self.assertFalse(gates["results"]["fixed_all_metrics_non_worse"])
        self.assertFalse(gates["results"]["live_target_architectures_observed"])
        decision = self.result["decision"]
        self.assertFalse(decision["promote_to_live_eci"])
        self.assertEqual(decision["incremental_live_weight"], 0.0)
        self.assertFalse(decision["change_live_weights"])
        self.assertFalse(decision["change_central_forecasts"])

    def test_k3_and_grok_anchor_truth_and_architecture_status(self) -> None:
        k3 = self.result["anchor_checks"]["kimi_k3"]
        self.assertEqual(k3["disclosed_total_b"], 2780.0)
        self.assertEqual(k3["disclosed_active_b"], 104.2)
        self.assertTrue(k3["target_developer_excluded"])
        self.assertLess(k3["train_max_eligibility_date"], k3["release_date"])
        self.assertGreater(
            k3["challenger_multiplicative_error"],
            k3["baseline_multiplicative_error"],
        )

        grok = self.result["anchor_checks"]["grok_4_5"]
        self.assertEqual(grok["disclosed_total_b"], 1500.0)
        self.assertTrue(grok["target_developer_excluded"])
        self.assertFalse(grok["target_architecture_observed"])
        self.assertEqual(len(grok["scenario_predictions"]), 4)
        self.assertEqual(
            {(row["moe"], row["reasoning"]) for row in grok["scenario_predictions"]},
            {(0, 0), (0, 1), (1, 0), (1, 1)},
        )
        self.assertTrue(grok["working_assumption"]["not_an_observation"])
        self.assertGreater(
            grok["working_assumption"]["multiplicative_error"],
            grok["baseline_multiplicative_error"],
        )

    def test_live_target_architecture_is_explicitly_unobserved(self) -> None:
        targets = self.result["live_target_applicability"]
        self.assertEqual(
            {row["model"] for row in targets}, set(audit.LIVE_TARGET_MODELS)
        )
        self.assertTrue(all(not row["architecture_observed"] for row in targets))
        self.assertTrue(
            all(row["status"] == "target architecture unobserved" for row in targets)
        )
        self.assertTrue(
            set(audit.LIVE_TARGET_MODELS).isdisjoint(
                {row["model"] for row in self.rows}
            )
        )

    def test_source_hashes_reconcile_and_offline_rebuild_is_exact(self) -> None:
        for relative, expected in self.result["source_files"].items():
            path = audit.ROOT / relative
            self.assertTrue(path.is_file(), relative)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)

        outputs = (audit.RESULT, audit.PREDICTIONS, audit.REPORT)
        before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in outputs}
        subprocess.run(
            [sys.executable, str(audit.ROOT / audit.__file__.split("/")[-1])],
            cwd=audit.ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in outputs}
        self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
