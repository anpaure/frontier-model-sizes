from __future__ import annotations

import csv
import hashlib
import json
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/019f6c42-2d53-7743-ab07-6293e2618dd7"
RESULT = OUT / "openrouter_incremental_price_backtest_2026-07-18.json"


class OpenRouterIncrementalBacktestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = json.loads(RESULT.read_text(encoding="utf-8"))
        prediction_path = ROOT / cls.result["prediction_csv"]
        with prediction_path.open(newline="", encoding="utf-8") as handle:
            cls.predictions = list(csv.DictReader(handle))

    def test_overlap_is_unique_and_explicit(self) -> None:
        coverage = self.result["coverage"]
        self.assertEqual(coverage["explicit_crosswalk_entries"], 17)
        self.assertEqual(coverage["strictly_chronological_paired_models"], 10)
        self.assertEqual(coverage["developer_families"], 6)
        keys = [row["canonical_checkpoint_id"] for row in self.predictions]
        self.assertEqual(len(keys), 10)
        self.assertEqual(len(keys), len(set(keys)))

    def test_current_small_price_weight_improves_the_clean_overlap(self) -> None:
        metrics = self.result["fixed_weight_metrics"]
        self.assertLess(
            metrics["0.0675"]["mean_absolute_log10_error"],
            metrics["0"]["mean_absolute_log10_error"],
        )
        self.assertLess(metrics["0.0675"]["rmse_log10"], metrics["0"]["rmse_log10"])
        bootstrap = self.result["paired_developer_family_bootstraps"]["0.0675"]
        promotion_probability = self.result["decision"]["bootstrap_promotion_probability"]
        self.assertEqual(promotion_probability, 0.90)
        self.assertGreaterEqual(
            bootstrap["bootstrap_probability_blend_better"], promotion_probability
        )
        self.assertLess(bootstrap["ci_90"][1], 0)
        self.assertEqual(bootstrap["random_seed"], 20260718)
        self.assertEqual(
            self.result["decision"]["current_weight_has_favorable_paired_bootstrap"],
            bootstrap["bootstrap_probability_blend_better"] >= promotion_probability
            and bootstrap["ci_90"][1] < 0,
        )
        self.assertIs(self.result["decision"]["current_weight_has_favorable_paired_bootstrap"], True)
        self.assertIs(self.result["decision"]["change_live_weight"], False)

    def test_prediction_arithmetic_recomputes(self) -> None:
        for row in self.predictions:
            evidence = float(row["evidence_predicted_b"])
            price = float(row["price_predicted_b"])
            for weight in (0.03375, 0.0675, 0.1, 0.5, 1.0):
                expected = math.exp((1 - weight) * math.log(evidence) + weight * math.log(price))
                self.assertAlmostEqual(float(row[f"blend_w{weight:g}_b"]), expected, places=8)

    def test_nested_meta_predictions_are_strictly_chronological_and_family_held_out(self) -> None:
        nested = self.result["nested_chronological_weight_learning"]["predictions"]
        self.assertGreaterEqual(len(nested), 5)
        for row in nested:
            self.assertLess(row["training_max_date"], row["release_date"])
            self.assertIs(row["test_family_excluded"], True)
            self.assertGreaterEqual(row["training_families"], 4)

    def test_disclosed_anchors_are_external_only(self) -> None:
        by_id = {
            row["model_id"]: row
            for row in self.result["external_disclosed_checks"]
        }
        ids = set(by_id)
        self.assertEqual(ids, {"moonshotai/kimi-k3", "x-ai/grok-4.5"})
        self.assertEqual(by_id["moonshotai/kimi-k3"]["actual_b"], 2780)
        self.assertEqual(
            by_id["moonshotai/kimi-k3"]["actual_parameter_source"],
            "Kimi K3 official technical report Table 1",
        )
        overlap = "|".join(row["canonical_checkpoint_id"] for row in self.predictions)
        self.assertNotIn("kimi-k3", overlap)
        self.assertNotIn("grok-4.5", overlap)

    def test_source_hashes_reconcile(self) -> None:
        for relative, digest in self.result["source_manifest"].items():
            path = ROOT / relative
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)


if __name__ == "__main__":
    unittest.main()
