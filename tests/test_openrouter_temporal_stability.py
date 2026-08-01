from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import unittest
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATE = "2026-07-18"
OUT = ROOT / "outputs/019f6c42-2d53-7743-ab07-6293e2618dd7"
RESULT = OUT / f"openrouter_temporal_stability_audit_{DATE}.json"
DAILY = ROOT / f"sources/openrouter_throughput_daily_{DATE}.csv"
ENDPOINTS = OUT / f"openrouter_endpoint_temporal_stability_{DATE}.csv"
MODELS = OUT / f"openrouter_model_temporal_stability_{DATE}.csv"
REFRESH = OUT / f"openrouter_refresh_stability_{DATE}.csv"
PREDICTIONS = OUT / f"openrouter_tier_counterfactual_predictions_{DATE}.csv"
HISTORY_MANIFEST = ROOT / f"sources/openrouter_snapshot_history_manifest_{DATE}.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class OpenRouterTemporalStabilityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = json.loads(RESULT.read_text(encoding="utf-8"))
        cls.daily = rows(DAILY)
        cls.endpoints = rows(ENDPOINTS)
        cls.models = rows(MODELS)
        cls.refresh = rows(REFRESH)
        cls.predictions = rows(PREDICTIONS)
        cls.history_manifest = rows(HISTORY_MANIFEST)

    def test_inventory_is_lossless_and_service_tiers_are_separate(self) -> None:
        inventory = self.result["inventory"]
        self.assertEqual(inventory["immutable_snapshots"], len(self.history_manifest))
        self.assertGreaterEqual(inventory["immutable_snapshots"], 5)
        self.assertEqual(inventory["current_daily_rows"], len(self.daily))
        self.assertEqual(
            inventory["history_daily_rows"],
            sum(int(row["daily_throughput_row_count"]) for row in self.history_manifest),
        )
        tier_counts = Counter(row["service_tier"] for row in self.daily)
        self.assertEqual(
            sum(inventory["service_tier_row_counts"].values()), len(self.daily)
        )
        self.assertEqual(
            inventory["service_tier_row_counts"],
            dict(sorted(tier_counts.items())),
        )
        tiers_by_endpoint: dict[tuple[str, str], set[str]] = defaultdict(set)
        for row in self.daily:
            tiers_by_endpoint[(row["openrouter_model_id"], row["endpoint_id"])].add(
                row["service_tier"]
            )
        self.assertEqual(
            inventory["endpoint_models_with_multiple_service_tiers"],
            sum(len(tiers) > 1 for tiers in tiers_by_endpoint.values()),
        )
        self.assertEqual(
            inventory["current_daily_unmatched_endpoint_models"],
            len(
                {
                    (row["openrouter_model_id"], row["endpoint_id"])
                    for row in self.daily
                    if row["endpoint_metadata_match"] != "True"
                }
            ),
        )
        self.assertEqual(inventory["calibration_checkpoints"], 93)
        self.assertEqual(inventory["calibration_families"], 19)

    def test_every_endpoint_stat_recomputes_from_default_daily_rows(self) -> None:
        grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
        for row in self.daily:
            value = float(row["throughput_tps"])
            if row["service_tier"] == "default" and value > 0:
                grouped[(row["openrouter_model_id"], row["endpoint_id"])].append(value)
        self.assertEqual(len(grouped), len(self.endpoints))
        for row in self.endpoints:
            values = grouped[(row["openrouter_model_id"], row["endpoint_id"])]
            self.assertEqual(int(row["observations"]), len(values))
            self.assertAlmostEqual(float(row["median_tps"]), statistics.median(values))
            self.assertAlmostEqual(float(row["min_tps"]), min(values))
            self.assertAlmostEqual(float(row["max_tps"]), max(values))
            self.assertAlmostEqual(
                float(row["max_over_min"]), max(values) / min(values)
            )

    def test_temporal_volatility_is_material_and_focal_values_are_exact(self) -> None:
        temporal = self.result["temporal_stability"]
        endpoint = temporal[
            "endpoint_default_tier_with_at_least_four_days_max_over_min"
        ]
        model = temporal[
            "model_daily_provider_median_with_at_least_four_days_max_over_min"
        ]
        refresh = temporal[
            "model_default_tier_median_across_all_refreshes_max_over_min"
        ]
        self.assertGreater(endpoint["median"], 1.5)
        self.assertGreater(model["median"], 1.5)
        self.assertGreater(refresh["median"], 1.05)
        self.assertLess(refresh["median"], 1.25)
        focal = {row["openrouter_model_id"]: row for row in temporal["focal_models"]}
        model_rows = {row["openrouter_model_id"]: row for row in self.models}
        refresh_rows = {row["openrouter_model_id"]: row for row in self.refresh}
        for model_id in (
            "anthropic/claude-fable-5",
            "openai/gpt-5.6-sol",
            "moonshotai/kimi-k3",
            "anthropic/claude-opus-4.8",
        ):
            self.assertAlmostEqual(
                focal[model_id]["within_week_median_tps"],
                float(model_rows[model_id]["median_tps"]),
            )
            self.assertAlmostEqual(
                focal[model_id]["refresh_max_over_min"],
                float(refresh_rows[model_id]["max_over_min"]),
            )

    def test_corrected_tok_s_still_fails_heldout_gate(self) -> None:
        corrected = self.result["corrected_default_tier_backtest"]
        bootstraps = self.result["service_tier_counterfactual"]["paired_bootstraps"][
            "default_only"
        ]
        family = next(row for row in bootstraps if row["mode"] == "family")
        chronological = next(
            row for row in bootstraps if row["mode"] == "chronological_family"
        )
        expected_weight = corrected["promotion_gate_from_main_audit"][
            "recommended_incremental_tok_s_weight_in_live_ensemble"
        ]
        if expected_weight == 0:
            passes_all_gates = (
                family["bootstrap_probability_candidate_better"] >= 0.90
                and family["ci_90"][1] < 0
                and chronological["observed_delta"] < 0
            )
            self.assertIs(passes_all_gates, False)
            self.assertGreater(family["observed_delta"], 0)
            self.assertLess(chronological["observed_delta"], 0)
        else:
            self.assertGreaterEqual(
                family["bootstrap_probability_candidate_better"], 0.90
            )
            self.assertLess(family["ci_90"][1], 0)
            self.assertLess(chronological["observed_delta"], 0)
        self.assertEqual(
            self.result["decision"]["recommended_incremental_tok_s_weight"],
            expected_weight,
        )
        self.assertIs(self.result["decision"]["change_live_forecast"], False)
        self.assertEqual(
            self.result["metadata"]["snapshot_date"], "2026-07-31"
        )
        self.assertEqual(
            self.result["metadata"]["compatibility_filename_date"], DATE
        )

    def test_predictions_are_complete_and_source_hashes_reconcile(self) -> None:
        keys = [
            (
                row["throughput_tier_policy"],
                row["mode"],
                row["specification"],
                row["canonical_checkpoint_id"],
            )
            for row in self.predictions
        ]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(
            {row["throughput_tier_policy"] for row in self.predictions},
            {"default_only", "mixed_default_priority_flex_counterfactual"},
        )
        for relative, expected in self.result["source_manifest"].items():
            self.assertEqual(sha256(ROOT / relative), expected, relative)


if __name__ == "__main__":
    unittest.main()
