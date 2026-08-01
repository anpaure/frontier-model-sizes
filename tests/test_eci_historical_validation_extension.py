from __future__ import annotations

import csv
import hashlib
import json
import unittest
from datetime import datetime
from pathlib import Path

import collect_eci_validation_extension as collector
import fit_eci_validation_extension as extension_fit


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "019f6c42-2d53-7743-ab07-6293e2618dd7"
RESULT = OUT / "eci_historical_validation_extension_2026-07-31.json"
PREDICTIONS = OUT / "eci_historical_validation_extension_predictions_2026-07-31.csv"
TARGETS = ROOT / "sources" / "eci_historical_validation_targets_2026-07-31.csv"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class EciHistoricalValidationExtensionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = json.loads(RESULT.read_text(encoding="utf-8"))
        with PREDICTIONS.open(newline="", encoding="utf-8") as handle:
            cls.predictions = list(csv.DictReader(handle))
        with TARGETS.open(newline="", encoding="utf-8") as handle:
            cls.targets = list(csv.DictReader(handle))

    def test_july_22_capture_and_official_fit_are_pinned(self) -> None:
        collected = collector.verify_existing()
        fitted = extension_fit.verify_existing()
        self.assertEqual(
            collected["sha256"],
            "85eb265713ebd29068b5cc5430c33596d5896c7b98a03b5f18f8d0e1e66615f8",
        )
        self.assertEqual(collected["rows"], 2034)
        self.assertEqual(collected["models"], 212)
        self.assertEqual(collected["benchmarks"], 53)
        self.assertEqual(collected["k3_component_rows"], 8)
        self.assertEqual(fitted["models"], 212)
        self.assertAlmostEqual(fitted["k3_eci"], 155.61602792683064, places=10)
        self.assertEqual(
            fitted["official_commit"],
            "542567e72a415b72624e5bbd12603cfd3f485179",
        )

    def test_target_identity_and_classification_are_explicit(self) -> None:
        by_model = {row["model"]: row for row in self.targets}
        self.assertEqual(
            set(by_model),
            {"Kimi K2.5", "Kimi K2.7 Code", "Grok 4.5", "GLM-5.2", "Kimi K3"},
        )
        self.assertEqual(by_model["Grok 4.5"]["total_b"], "1500")
        self.assertEqual(by_model["GLM-5.2"]["total_b"], "744")
        self.assertEqual(
            by_model["Kimi K3"]["validation_class"],
            "score_vintage_only_not_project_prospective",
        )
        self.assertEqual(by_model["Kimi K3"]["project_prospective"], "false")
        glm = by_model["GLM-5.2"]
        parse = lambda value: datetime.fromisoformat(value.replace("Z", "+00:00"))
        self.assertLess(
            parse(glm["prior_capture_timestamp_utc"]),
            parse(glm["release_timestamp_utc"]),
        )
        self.assertLess(
            parse(glm["release_timestamp_utc"]),
            parse(glm["first_score_capture_timestamp_utc"]),
        )

    def test_prediction_ledger_is_complete_and_leakage_labeled(self) -> None:
        self.assertEqual(len(self.predictions), 5 * 10 * 2)
        keys = [
            (row["model"], row["candidate"], row["weight_mode"])
            for row in self.predictions
        ]
        self.assertEqual(len(keys), len(set(keys)))
        interval = {
            row["model"]
            for row in self.predictions
            if row["historical_interval_prospective"] == "True"
        }
        self.assertEqual(
            interval, {"Kimi K2.5", "Kimi K2.7 Code", "Grok 4.5", "GLM-5.2"}
        )
        k3 = [row for row in self.predictions if row["model"] == "Kimi K3"]
        self.assertTrue(k3)
        self.assertTrue(all(row["score_vintage_holdout"] == "True" for row in k3))
        self.assertTrue(all(row["project_prospective"] == "False" for row in k3))
        glm = [row for row in self.predictions if row["model"] == "GLM-5.2"]
        self.assertTrue(all(row["timestamp_rescued"] == "True" for row in glm))

    def test_frozen_baseline_values_and_zero_weight_decision(self) -> None:
        baseline = {
            row["model"]: row
            for row in self.result["baseline_predictions"]
        }
        expected = {
            "Kimi K2.5": 629.7169782132821,
            "Kimi K2.7 Code": 799.4074593711471,
            "Grok 4.5": 1361.8344292349534,
            "GLM-5.2": 1226.9008331135165,
            "Kimi K3": 1352.7074414444476,
        }
        for model, value in expected.items():
            self.assertAlmostEqual(baseline[model]["predicted_b"], value, places=8)
        self.assertFalse(self.result["decision"]["change_live_weights"])
        self.assertFalse(self.result["decision"]["change_live_functional_form"])
        self.assertEqual(self.result["role"], "external validation only; zero live-model weight")
        self.assertEqual(self.result["linked_component_panel"]["live_weight"], 0)

    def test_hashes_and_pipeline_wiring(self) -> None:
        sources = self.result["sources"]
        outputs = self.result["outputs"]
        self.assertEqual(sources["target_ledger_sha256"], sha256(TARGETS))
        self.assertEqual(
            sources["july_22_capture_sha256"], sha256(collector.SOURCE)
        )
        self.assertEqual(
            sources["july_22_scores_sha256"], sha256(extension_fit.OUTPUT)
        )
        self.assertEqual(outputs["predictions_sha256"], sha256(PREDICTIONS))
        pipeline = (ROOT / "run_forecast_pipeline.py").read_text(encoding="utf-8")
        collector_pos = pipeline.index("collect_eci_validation_extension.py")
        fit_pos = pipeline.index("fit_eci_validation_extension.py")
        analysis_pos = pipeline.index("analyze_eci_historical_validation_extension.py")
        self.assertLess(collector_pos, fit_pos)
        self.assertLess(fit_pos, analysis_pos)
        self.assertIn("tests.test_eci_historical_validation_extension", pipeline)


if __name__ == "__main__":
    unittest.main()
