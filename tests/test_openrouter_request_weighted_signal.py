from __future__ import annotations

import csv
import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/019f6c42-2d53-7743-ab07-6293e2618dd7"
RESULT = OUT / "openrouter_request_weighted_operational_audit_2026-07-18.json"
PREDICTIONS = OUT / "openrouter_request_weighted_operational_predictions_2026-07-18.csv"


class OpenRouterRequestWeightedSignalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = json.loads(RESULT.read_text(encoding="utf-8"))
        with PREDICTIONS.open(newline="", encoding="utf-8") as handle:
            cls.predictions = list(csv.DictReader(handle))

    def test_inventory_and_request_gate_are_exact(self) -> None:
        inventory = self.data["inventory"]
        self.assertEqual(inventory["endpoint_default_rows"], 1036)
        self.assertEqual(inventory["complete_checkpoints"], 81)
        self.assertEqual(inventory["complete_families"], 16)
        self.assertEqual(len(inventory["incomplete_checkpoints"]), 12)
        feature_metadata = inventory["feature_metadata"]
        self.assertEqual(
            feature_metadata["throughput_p50_all"]["eligible_endpoint_rows"],
            965,
        )
        for name in (
            "throughput_p50_supported",
            "latency_p50_supported",
            "throughput_p90_over_p50_supported",
            "latency_p90_over_p50_supported",
        ):
            self.assertEqual(feature_metadata[name]["eligible_endpoint_rows"], 799)
            self.assertEqual(feature_metadata[name]["models"], 264)
            self.assertEqual(feature_metadata[name]["minimum_requests"], 100)
            self.assertIs(feature_metadata[name]["request_weighted"], True)

    def test_every_fold_is_family_held_out_and_chronological_when_required(
        self,
    ) -> None:
        specifications = set(self.data["candidate_specifications"])
        self.assertEqual(len(specifications), 7)
        self.assertEqual(len(self.predictions), 7 * (81 + 57))
        for row in self.predictions:
            self.assertIn(row["specification"], specifications)
            self.assertGreaterEqual(int(row["training_rows"]), 20)
            self.assertGreaterEqual(int(row["training_families"]), 5)
            if row["mode"] == "chronological_family":
                self.assertLess(row["training_max_date"], row["release_date"])

    def test_no_operational_candidate_passes_promotion_gates(self) -> None:
        self.assertEqual(self.data["supported_candidates"], [])
        self.assertIs(
            self.data["decision"]["promote_operational_feature"], False
        )
        self.assertEqual(self.data["decision"]["incremental_live_weight"], 0)
        family = {
            row["candidate"]: row
            for row in self.data["paired_family_bootstraps"]
            if row["mode"] == "family"
        }
        for candidate, comparison in family.items():
            passes_family_gate = (
                comparison["bootstrap_probability_candidate_better"] >= 0.90
                and comparison["ci_90"][1] < 0
            )
            self.assertIs(passes_family_gate, False, candidate)
        self.assertTrue(
            all(comparison["observed_delta"] > 0 for comparison in family.values())
        )
        self.assertEqual(self.data["snapshot_date"], "2026-07-31")
        self.assertEqual(
            self.data["compatibility_filename_date"], "2026-07-18"
        )

    def test_source_hashes_reconcile(self) -> None:
        for relative, expected in self.data["source_manifest"].items():
            actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            self.assertEqual(actual, expected)
