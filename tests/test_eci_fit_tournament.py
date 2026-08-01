from __future__ import annotations

import csv
import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "019f6c42-2d53-7743-ab07-6293e2618dd7"
RESULT = OUT / "eci_fit_tournament_2026-07-18.json"
PREDICTIONS = OUT / "eci_fit_tournament_predictions_2026-07-18.csv"
TARGETS = OUT / "eci_fit_tournament_frontier_sensitivity_2026-07-18.csv"
HISTORICAL = ROOT / "sources" / "epoch_eci_historical_model_scores_2026-07-18.csv"
FIT_METADATA = ROOT / "sources" / "epoch_eci_historical_fit_metadata_2026-07-18.json"
COLLECTION_METADATA = ROOT / "sources" / "epoch_eci_historical_collection_metadata_2026-07-18.json"
ARCHIVE = ROOT / "sources" / "epoch_eci_historical_snapshots_2026-07-18.tar.gz"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class EciFitTournamentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = json.loads(RESULT.read_text())
        cls.fit_metadata = json.loads(FIT_METADATA.read_text())
        cls.collection = json.loads(COLLECTION_METADATA.read_text())
        with PREDICTIONS.open(newline="", encoding="utf-8") as handle:
            cls.predictions = list(csv.DictReader(handle))
        with TARGETS.open(newline="", encoding="utf-8") as handle:
            cls.targets = list(csv.DictReader(handle))
        with HISTORICAL.open(newline="", encoding="utf-8") as handle:
            cls.historical = list(csv.DictReader(handle))

    def test_historical_source_inventory_and_hashes(self) -> None:
        self.assertEqual(self.collection["inventory"]["archive_members"], 20)
        self.assertEqual(self.collection["inventory"]["canonical_csv_captures"], 5)
        self.assertEqual(self.collection["inventory"]["benchmark_zip_captures"], 15)
        self.assertEqual(self.collection["inventory"]["benchmark_zip_eci_era"], 10)
        self.assertEqual(self.collection["inventory"]["benchmark_zip_pre_eci_preserved"], 5)
        self.assertEqual(self.collection["archive_sha256"], sha256(ARCHIVE))
        self.assertEqual(self.fit_metadata["source_archive_sha256"], sha256(ARCHIVE))
        self.assertEqual(self.fit_metadata["official_commit"], "542567e72a415b72624e5bbd12603cfd3f485179")
        self.assertEqual(len(self.fit_metadata["inventory"]), 15)
        self.assertEqual(len(self.fit_metadata["schema_compatibility_shims"]), 1)
        self.assertEqual(
            self.fit_metadata["schema_compatibility_shims"][0]["snapshot_timestamp"],
            "20251113094011",
        )
        terminal = self.fit_metadata["archival_terminal_fit_crosscheck"]
        self.assertTrue(terminal["exact_within_1e_8"])
        self.assertLess(
            terminal["maximum_absolute_eci_difference"],
            1e-12,
        )
        self.assertEqual(terminal["timestamp"], "20260716153134")
        self.assertEqual(
            self.collection["archival_terminal_exact_match"]["pinned_file"],
            "sources/epoch_eci_benchmarks_2026-07-17.csv",
        )
        self.assertEqual(
            self.collection["current_live_successor_reference"]["pinned_file"],
            "sources/epoch_eci_benchmarks_2026-07-31.csv",
        )
        self.assertFalse(
            self.collection["current_live_successor_reference"][
                "byte_equality_assertion_against_archival_terminal"
            ]
        )

    def test_historical_score_ledger_is_unique(self) -> None:
        keys = [(row["snapshot_timestamp"], row["Model"]) for row in self.historical]
        self.assertEqual(len(keys), 2308)
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(len({row["snapshot_timestamp"] for row in self.historical}), 15)

    def test_outer_prediction_ledger_is_complete(self) -> None:
        inventory = self.result["inventory"]
        self.assertEqual(inventory["parameter_map_checkpoints"], 89)
        self.assertEqual(inventory["parameter_families"], 40)
        self.assertEqual(inventory["archive_compatible_checkpoints"], 88)
        self.assertEqual(
            set(inventory["retired_historical_checkpoints"]),
            {"DeepSeek-V3.1", "Kimi K2 (Sep 2025)"},
        )
        self.assertEqual(
            set(inventory["current_checkpoints_absent_from_frozen_archive"]),
            {"Kimi K3", "Gemma 4 31B IT", "Qwen 3.6 35B-A3B"},
        )
        expected = 27 * 10 * 2
        self.assertEqual(len(self.predictions), expected)
        keys = [
            (row["model"], row["candidate"], row["weight_mode"])
            for row in self.predictions
        ]
        self.assertEqual(len(keys), len(set(keys)))
        prospective = {
            row["model"]
            for row in self.predictions
            if row["interval_prospective"] == "True"
        }
        self.assertEqual(prospective, {"Kimi K2.5", "Kimi K2.7 Code"})

    def test_selection_improves_but_promotion_gates_fail(self) -> None:
        primary = self.result["tournament"]["live_inverse_eci_ci"]
        self.assertEqual(primary["selected_challenger"], "ridge_flexible")
        baseline = primary["candidates"]["live_60_40_blend"][
            "selection_first_observed_backfills"
        ]
        challenger = primary["candidates"]["ridge_flexible"][
            "selection_first_observed_backfills"
        ]
        for field in (
            "median_multiplicative_error",
            "mean_absolute_log10_error",
            "rmse_log10",
            "p80_multiplicative_error",
        ):
            self.assertLess(challenger[field], baseline[field])
        decision = self.result["decision"]
        self.assertTrue(decision["same_challenger_selected_across_weight_modes"])
        self.assertTrue(decision["selection_point_metrics_all_improve"])
        self.assertFalse(decision["selection_equal_family_ci_wholly_favorable"])
        self.assertTrue(decision["interval_prospective_metrics_improve_in_both_weight_modes"])
        self.assertFalse(decision["frontier_sensitivity_below_1_25x"])
        self.assertFalse(decision["change_live_eci_functional_form"])

    def test_frontier_is_outside_training_support_and_unstable(self) -> None:
        self.assertEqual(len(self.targets), 2 * 10 * 2 * 2)
        self.assertGreater(self.result["frontier_extrapolation"]["Claude Fable 5"], 4)
        self.assertGreater(self.result["frontier_extrapolation"]["GPT-5.6 Sol"], 5)
        for model in ("Claude Fable 5", "GPT-5.6 Sol"):
            sensitivity = self.result["decision"]["frontier_sensitivity"][model]
            self.assertGreater(sensitivity["max_over_min"], 1.25)
            self.assertFalse(sensitivity["passes_1_25x_gate"])

    def test_output_hashes_are_self_consistent(self) -> None:
        self.assertEqual(self.result["outputs"]["predictions_sha256"], sha256(PREDICTIONS))
        self.assertEqual(self.result["outputs"]["frontier_sha256"], sha256(TARGETS))
        self.assertEqual(self.result["sources"]["historical_scores_sha256"], sha256(HISTORICAL))
        self.assertEqual(self.result["sources"]["historical_metadata_sha256"], sha256(FIT_METADATA))
        self.assertEqual(
            self.result["sources"]["collection_metadata_sha256"],
            sha256(COLLECTION_METADATA),
        )
        boundary = self.result["method"]["archive_boundary"]
        self.assertEqual(boundary["terminal_timestamp"], 20260716153134)
        self.assertTrue(boundary["current_live_excluded_from_archive_fit"])


if __name__ == "__main__":
    unittest.main()
