from __future__ import annotations

import csv
import hashlib
import json
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/019f6c42-2d53-7743-ab07-6293e2618dd7"
RESULT = OUT / "aa_inference_budget_audit_2026-07-18.json"
PANEL = OUT / "aa_detailed_parameter_panel_2026-07-18.csv"
PAIRS = OUT / "aa_reasoning_pair_audit_2026-07-18.csv"
CROSSCHECK = OUT / "aa_detailed_epoch_crosscheck_2026-07-18.csv"
PREDICTIONS = OUT / "aa_inference_budget_predictions_2026-07-18.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class AAInferenceBudgetSignalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = json.loads(RESULT.read_text(encoding="utf-8"))
        cls.panel = rows(PANEL)
        cls.pairs = rows(PAIRS)
        cls.crosscheck = rows(CROSSCHECK)
        cls.predictions = rows(PREDICTIONS)

    def test_detailed_panel_deduplication_is_complete(self) -> None:
        audit = self.data["data_audit"]
        self.assertEqual(audit["raw_models"], 587)
        self.assertEqual(audit["open_weight_parameter_score_date_configurations"], 335)
        self.assertEqual(audit["unique_checkpoint_groups"], 275)
        self.assertEqual(audit["lower_score_configurations_removed"], 60)
        self.assertEqual(audit["creators"], 48)
        self.assertEqual(audit["token_covered_checkpoint_groups"], 79)
        self.assertEqual(len(self.panel), 275)
        self.assertEqual(len({row["checkpoint_group_id"] for row in self.panel}), 275)
        self.assertEqual(
            sum(int(row["configuration_count"]) for row in self.panel), 335
        )
        self.assertEqual(
            sum(int(row["lower_score_configurations_removed"]) for row in self.panel),
            60,
        )
        self.assertTrue(
            all(
                float(row["parameters_b"]) > 0
                and row["release_date"]
                and row["intelligence_index"]
                for row in self.panel
            )
        )

        overrides = audit["primary_metadata_overrides"]
        self.assertEqual(len(overrides), 2)
        overrides_by_slug = {row["slug"]: row for row in overrides}
        self.assertEqual(
            overrides_by_slug["motif-0714"]["lineage_class"],
            "independent_pretrain",
        )
        self.assertIsNone(overrides_by_slug["motif-0714"]["base_model_id"])
        motif = next(row for row in self.panel if row["selected_slug"] == "motif-0714")
        self.assertEqual(float(motif["parameters_b"]), 314.0)
        self.assertEqual(float(motif["active_parameters_b"]), 13.0)
        self.assertEqual(
            motif["model_weights_source_url"],
            "https://huggingface.co/Motif-Technologies/Motif-3-Beta",
        )
        self.assertEqual(
            motif["parameter_source"], motif["calibration_override_source_url"]
        )
        self.assertEqual(motif["release_date"], "2026-07-14")
        self.assertEqual(motif["parameter_label_available_date"], "2026-07-20")
        self.assertEqual(
            motif["parameter_training_eligibility_date"], "2026-07-20"
        )
        self.assertEqual(motif["lineage_class"], "independent_pretrain")

        motif2 = next(
            row for row in self.panel if row["selected_slug"] == "motif-2-12-7b"
        )
        self.assertEqual(float(motif2["parameters_b"]), 12.703860896)
        self.assertFalse(motif2["active_parameters_b"])
        self.assertEqual(motif2["architecture_class"], "dense")
        self.assertEqual(int(motif2["exact_tensor_parameters"]), 12_703_860_896)
        self.assertEqual(motif2["release_date"], "2025-12-04")
        self.assertEqual(motif2["parameter_label_available_date"], "2025-12-01")
        self.assertEqual(
            motif2["parameter_training_eligibility_date"], "2025-12-04"
        )
        self.assertEqual(motif2["lineage_class"], "same_base_posttrain")
        self.assertEqual(
            motif2["base_model_id"], "Motif-Technologies/Motif-2-12.7B-Base"
        )
        self.assertEqual(
            overrides_by_slug["motif-2-12-7b"]["lineage_family_id"],
            "Motif-Technologies/Motif-2-12.7B-Base",
        )

    def test_reasoning_pairs_cover_open_and_closed_models(self) -> None:
        self.assertEqual(len(self.pairs), 100)
        self.assertEqual(
            Counter(row["is_open_weights"] for row in self.pairs),
            Counter({"True": 61, "False": 39}),
        )
        self.assertEqual(len({row["checkpoint_group_id"] for row in self.pairs}), 100)
        self.assertEqual(len({row["creator_slug"] for row in self.pairs}), 17)
        self.assertTrue(
            all(
                abs(
                    float(row["aa_uplift"])
                    - (
                        float(row["reasoning_aa"])
                        - float(row["nonreasoning_aa"])
                    )
                )
                < 1e-10
                for row in self.pairs
            )
        )
        self.assertEqual(
            sum(float(row["aa_uplift"]) < 0 for row in self.pairs), 1
        )
        exact = self.data["same_weight_reasoning_pairs"]
        all_pairs = self.data["reasoning_configuration_pairs"]["all"]
        self.assertEqual(exact["pairs"], 53)
        self.assertEqual(all_pairs["pairs"], 100)
        self.assertLess(
            all_pairs["equal_creator_median_bootstrap_90_ci"][0], 6.0
        )
        self.assertGreater(
            all_pairs["equal_creator_median_bootstrap_90_ci"][1], 6.0
        )
        self.assertGreater(
            all_pairs["creator_medians"]["openai"],
            all_pairs["creator_medians"]["anthropic"],
        )

    def test_epoch_crosscheck_matches_every_exact_checkpoint_without_hiding_conflicts(self) -> None:
        audit = self.data["data_audit"]
        self.assertEqual(len(self.crosscheck), 62)
        self.assertEqual(audit["epoch_exact_crosschecks"], 62)
        self.assertEqual(audit["epoch_exact_unmatched"], ["Inkling"])
        self.assertEqual(len({row["epoch_checkpoint_id"] for row in self.crosscheck}), 62)
        self.assertEqual(len({row["aa_detailed_group_id"] for row in self.crosscheck}), 62)
        conflicts = [
            row
            for row in self.crosscheck
            if row["creator_agreement"] != "True"
            or row["date_within_45_days"] != "True"
            or row["parameters_within_20_percent"] != "True"
        ]
        self.assertEqual(len(conflicts), 3)
        self.assertEqual(
            {row["epoch_checkpoint_id"] for row in conflicts},
            {
                "checkpoint:epoch:phi-4-mini",
                "checkpoint:epoch:granite-4-0-h-tiny",
                "checkpoint:epoch:ring-flash-linear-2-0",
            },
        )
        self.assertEqual(audit["epoch_crosschecks_with_metadata_disagreement"], 3)
        self.assertAlmostEqual(audit["epoch_parameter_ratio_median"], 1.0)

    def test_every_backtest_is_strictly_chronological_and_family_held_out(self) -> None:
        counts = Counter(row["comparison"] for row in self.predictions)
        self.assertEqual(
            counts,
            Counter(
                {
                    "reasoning_standardization_portable": 229,
                    "reasoning_standardization_creator_aware": 229,
                    "token_budget_incremental": 54,
                    "current_50_vs_detailed": 34,
                }
            ),
        )
        self.assertEqual(
            len(
                {
                    (row["comparison"], row["release_date"], row["model"])
                    for row in self.predictions
                }
            ),
            len(self.predictions),
        )
        for row in self.predictions:
            self.assertLess(
                row["baseline_train_max_date"],
                row["prediction_information_date"],
            )
            self.assertLess(
                row["candidate_train_max_date"],
                row["prediction_information_date"],
            )
            self.assertEqual(row["test_developer_excluded"], "True")
            if "strictly earlier" in row.get("standardization_method", ""):
                self.assertLess(
                    row["standardization_history_max_date"], row["release_date"]
                )

    def test_results_support_diagnostics_but_not_live_weight(self) -> None:
        detailed = self.data["detailed_panel_backtest"]["scopes"]
        self.assertLess(
            detailed["all"]["candidate"]["median_multiplicative_error"],
            detailed["all"]["baseline"]["median_multiplicative_error"],
        )
        self.assertGreater(detailed["all"]["paired_cluster_bootstrap"]["ci_90"][1], 0)
        self.assertGreater(
            detailed["frontier_like"]["paired_cluster_bootstrap"]["ci_90"][1],
            0,
        )
        token = self.data["inference_budget_backtest"]["scopes"]
        # On the 54-row token-covered backtest the candidate improves the
        # unweighted median, mean, and tail. The developer-clustered interval
        # still crosses zero, so the signal remains diagnostic-only.
        self.assertLess(
            token["all"]["candidate"]["median_multiplicative_error"],
            token["all"]["baseline"]["median_multiplicative_error"],
        )
        self.assertLess(
            token["all"]["candidate"]["mean_absolute_log10_error"],
            token["all"]["baseline"]["mean_absolute_log10_error"],
        )
        self.assertLess(
            token["all"]["candidate"]["p80_multiplicative_error"],
            token["all"]["baseline"]["p80_multiplicative_error"],
        )
        self.assertGreater(token["all"]["paired_cluster_bootstrap"]["observed_delta"], 0)
        self.assertLess(
            token["frontier_like"]["paired_cluster_bootstrap"]["ci_90"][0], 0
        )
        self.assertGreater(
            token["frontier_like"]["paired_cluster_bootstrap"]["ci_90"][1], 0
        )
        standardized = self.data["reasoning_standardization_backtest"]["portable"][
            "scopes"
        ]
        self.assertLess(
            standardized["frontier_like"]["candidate"]["median_multiplicative_error"],
            standardized["frontier_like"]["baseline"]["median_multiplicative_error"],
        )
        # After exact score-publication timing, the portable correction has a
        # narrowly favorable frontier interval, but the all-row interval still
        # crosses zero and the full promotion rule remains unmet.
        self.assertLess(
            standardized["frontier_like"]["paired_cluster_bootstrap"]["ci_90"][1],
            0,
        )
        self.assertGreater(
            standardized["all"]["paired_cluster_bootstrap"]["ci_90"][1],
            0,
        )
        decision = self.data["decision"]
        self.assertIs(decision["change_live_aa_branch"], False)
        self.assertEqual(decision["incremental_detailed_panel_weight"], 0)
        self.assertEqual(decision["incremental_inference_budget_weight"], 0)
        self.assertEqual(decision["incremental_reasoning_standardization_weight"], 0)

    def test_source_hashes_reconcile(self) -> None:
        self.assertEqual(self.data["k3_anchor"]["total_parameters_b"], 2780)
        self.assertEqual(
            self.data["k3_anchor"]["source"],
            "Kimi K3 official technical report Table 1",
        )
        for relative, expected in self.data["source_manifest"].items():
            actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
