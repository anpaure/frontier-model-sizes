#!/usr/bin/env python3

from __future__ import annotations

import csv
import hashlib
import json
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/019f6c42-2d53-7743-ab07-6293e2618dd7"
RESULT = OUT / "active_parameter_transport_audit_2026-07-18.json"
PREDICTIONS = OUT / "active_parameter_transport_predictions_2026-07-18.csv"
TARGETS = OUT / "active_parameter_transport_targets_2026-07-18.csv"
PANEL = OUT / "aa_detailed_parameter_panel_2026-07-18.csv"
FACTS = ROOT / "sources/kimi_k3_release_evidence_2026-07-31.json"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ActiveParameterTransportAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = json.loads(RESULT.read_text(encoding="utf-8"))
        cls.facts = json.loads(FACTS.read_text(encoding="utf-8"))
        cls.predictions = read_csv(PREDICTIONS)
        cls.targets = read_csv(TARGETS)
        cls.panel = read_csv(PANEL)

    def test_official_k3_and_k2_facts_reconcile(self) -> None:
        k3 = self.facts["kimi_k3"]
        k2 = self.facts["kimi_k2_comparator"]
        derived = self.facts["derived_quantities"]
        self.assertEqual(k3["total_parameters_b_exact"], 2780.0)
        self.assertEqual(k3["activated_parameters_b_exact"], 104.2)
        self.assertEqual(k3["selected_routed_experts_per_token"], 16)
        self.assertEqual(k3["routed_experts"], 896)
        self.assertTrue(k3["activated_parameter_count_disclosed"])
        self.assertEqual(k2["total_parameters_b_exact"], 1040.0)
        self.assertEqual(k2["activated_parameters_b_exact"], 32.6)
        self.assertAlmostEqual(
            derived["selected_routed_expert_fraction"], 16 / 896, places=15
        )
        self.assertAlmostEqual(
            derived["total_to_activated_parameter_ratio"], 2780 / 104.2, places=12
        )
        self.assertAlmostEqual(
            derived["activated_parameter_fraction"], 104.2 / 2780, places=12
        )
        self.assertAlmostEqual(
            derived["k2_total_to_activated_parameter_ratio"],
            1040 / 32.6,
            places=12,
        )
        self.assertEqual(k3["initial_model_release_date"], "2026-07-16")
        self.assertEqual(k3["weights_release_date"], "2026-07-27")

    def test_inventory_and_source_hashes_are_exact(self) -> None:
        inventory = self.result["inventory"]
        active_panel = [row for row in self.panel if row["active_parameters_b"]]
        self.assertEqual(
            inventory["detailed_total_parameter_checkpoints"], len(self.panel)
        )
        self.assertEqual(inventory["active_parameter_checkpoints"], len(active_panel))
        self.assertEqual(
            inventory["active_parameter_developers"],
            len({row["creator_slug"] for row in active_panel}),
        )
        self.assertEqual(inventory["chronological_predictions"], len(self.predictions))
        self.assertEqual(
            inventory["frontier_like_predictions"],
            sum(float(row["frontier_score_rank"]) >= 0.90 for row in self.predictions),
        )
        self.assertEqual(
            inventory["high_sparsity_conversion_predictions"],
            sum(
                bool(row["active_converted_total_b"])
                and float(row["actual_total_to_active_ratio"]) >= 15
                for row in self.predictions
            ),
        )
        self.assertEqual(len(self.targets), 3)
        for relative, expected in self.result["source_files"].items():
            self.assertEqual(sha256(ROOT / relative), expected)

    def test_every_fold_is_strictly_chronological_and_developer_held_out(self) -> None:
        for row in self.predictions:
            self.assertLess(
                row["train_max_date"], row["prediction_information_date"]
            )
            self.assertEqual(row["test_developer_excluded"], "True")
            self.assertGreaterEqual(int(row["active_train_n"]), 30)
            self.assertGreaterEqual(int(row["active_train_developers"]), 8)
            actual_active = float(row["actual_active_b"])
            predicted_active = float(row["predicted_active_b"])
            self.assertAlmostEqual(
                float(row["active_log10_error"]),
                math.log10(predicted_active / actual_active),
                places=12,
            )
            actual_total = float(row["actual_total_b"])
            predicted_total = float(row["predicted_full_panel_total_b"])
            self.assertAlmostEqual(
                float(row["full_panel_total_log10_error"]),
                math.log10(predicted_total / actual_total),
                places=12,
            )
            if row["active_converted_total_b"]:
                self.assertAlmostEqual(
                    float(row["active_converted_total_b"]),
                    predicted_active * float(row["high_sparsity_reference_ratio"]),
                    places=9,
                )

    def test_active_signal_is_suggestive_but_not_promotion_grade(self) -> None:
        active = self.result["active_parameter_predictability"]
        self.assertLess(
            active["active_score_date"]["median_multiplicative_error"],
            active["total_score_date_same_active_checkpoint_panel"][
                "median_multiplicative_error"
            ],
        )
        self.assertLess(
            active["frontier_like"]["active_score_date"][
                "median_multiplicative_error"
            ],
            active["frontier_like"]["total_score_date_same_panel"][
                "median_multiplicative_error"
            ],
        )
        active_ci = active["paired_active_vs_same_panel_total"]["ci_90"]
        self.assertLess(active_ci[0], 0)
        self.assertLess(active_ci[1], 0)

        transport = self.result["high_sparsity_total_transport"]
        self.assertLess(
            transport["candidate"]["mean_absolute_log10_error"],
            transport["direct_total_baseline"]["mean_absolute_log10_error"],
        )
        self.assertGreater(
            transport["candidate"]["median_multiplicative_error"],
            transport["direct_total_baseline"]["median_multiplicative_error"],
        )
        transport_ci = transport["paired_cluster_bootstrap"]["ci_90"]
        self.assertLess(transport_ci[0], 0)
        self.assertLess(transport_ci[1], 0)
        self.assertFalse(
            self.result["decision"]["promote_active_transport_to_live_factor"]
        )
        self.assertFalse(
            self.result["decision"]["independent_target_architecture_observed"]
        )
        self.assertEqual(self.result["decision"]["incremental_live_weight"], 0.0)
        self.assertFalse(self.result["decision"]["change_headline_forecasts"])

    def test_k3_anchor_and_target_sensitivities_reconcile(self) -> None:
        audit = self.result["kimi_k3_external_architecture_check"]
        eligible_training = [
            row
            for row in self.panel
            if row["active_parameters_b"]
            and max(
                row["release_date"],
                row.get("parameter_label_available_date") or row["release_date"],
            )
            < "2026-07-16"
            and row["creator_slug"] not in {"kimi", "moonshot"}
        ]
        self.assertNotIn(
            "motif-0714", {row["selected_slug"] for row in eligible_training}
        )
        self.assertEqual(audit["training_rows"], len(eligible_training))
        self.assertEqual(
            audit["training_developers"],
            len({row["creator_slug"] for row in eligible_training}),
        )
        self.assertTrue(audit["kimi_developer_removed"])
        self.assertGreater(audit["predicted_k3_active_b"], 50.0)
        self.assertLess(audit["predicted_k3_active_b"], 80.0)
        self.assertEqual(audit["k3_disclosed_active_b"], 104.2)
        self.assertGreater(audit["k3_active_prediction_calibration_factor"], 1.0)
        self.assertLess(audit["k3_active_prediction_multiplicative_error"], 2.0)
        self.assertAlmostEqual(
            audit["k3_disclosed_total_to_active_ratio"], 2780 / 104.2, places=12
        )
        self.assertGreater(
            audit["k2_disclosed_total_to_active_ratio"],
            audit["k3_disclosed_total_to_active_ratio"],
        )
        self.assertGreater(audit["active_fraction_over_selected_expert_fraction"], 2.0)

        targets = {row["model"]: row for row in self.result["target_sensitivity"]}
        self.assertAlmostEqual(targets["Kimi K3"]["k3_anchored_total_t"], 2.78)
        self.assertAlmostEqual(targets["Kimi K3"]["k3_calibrated_active_b"], 104.2)
        for target in targets.values():
            self.assertAlmostEqual(
                target["k3_anchored_total_t"],
                target["k3_calibrated_active_b"]
                * audit["k3_disclosed_total_to_active_ratio"]
                / 1000,
            )
            self.assertAlmostEqual(
                target["k3_sparsity_total_t"], target["k3_anchored_total_t"]
            )
        self.assertGreater(
            targets["Claude Fable 5"]["k3_anchored_total_t"],
            targets["GPT-5.6 Sol"]["k3_anchored_total_t"],
        )
        self.assertGreater(
            targets["Claude Fable 5"]["k2_sparsity_total_t"],
            targets["Claude Fable 5"]["k3_sparsity_total_t"],
        )

    def test_compute_branch_distinguishes_estimate_from_disclosure(self) -> None:
        compute = self.result["compute_branch_independence"]
        self.assertEqual(compute["target_models"], 9)
        self.assertEqual(compute["target_models_with_epoch_training_compute_estimate"], 1)
        self.assertEqual(compute["epoch_compute_estimate_target_names"], ["Kimi K3"])
        self.assertEqual(compute["target_models_with_disclosed_training_compute"], 0)
        self.assertEqual(compute["disclosed_training_compute_target_names"], [])
        self.assertEqual(len(compute["epoch_compute_estimate_records"]), 1)
        self.assertEqual(
            compute["epoch_compute_estimate_records"][0]["epoch_confidence"],
            "Speculative",
        )
        self.assertFalse(compute["independent_target_evidence"])
        self.assertFalse(compute["change_numeric_weight"])
        self.assertFalse(
            self.result["decision"]["classify_compute_as_independent_evidence"]
        )
        self.assertTrue(
            self.result["decision"]["update_compute_description_to_show_dependency"]
        )


if __name__ == "__main__":
    unittest.main()
