from __future__ import annotations

import csv
import hashlib
import json
import unittest
from pathlib import Path

from analyze_openrouter_parameter_signal import DISCLOSED_ANCHORS, MANUAL_EPOCH_MAP


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/019f6c42-2d53-7743-ab07-6293e2618dd7"
DATE = "2026-07-18"
TRUTH = ROOT / "sources/open_model_parameter_truth_reconciliation_2026-07-31.json"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class OpenRouterParameterSignalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit_path = OUT / f"openrouter_epoch_match_audit_{DATE}.csv"
        cls.calibration_path = OUT / f"openrouter_parameter_calibration_{DATE}.csv"
        cls.predictions_path = OUT / f"openrouter_parameter_backtest_predictions_{DATE}.csv"
        cls.frontier_path = OUT / f"openrouter_frontier_operational_estimates_{DATE}.csv"
        cls.result_path = OUT / f"openrouter_parameter_signal_backtest_{DATE}.json"
        cls.audit = rows(cls.audit_path)
        cls.calibration = rows(cls.calibration_path)
        cls.predictions = rows(cls.predictions_path)
        cls.frontier = rows(cls.frontier_path)
        cls.result = json.loads(cls.result_path.read_text(encoding="utf-8"))
        cls.truth_ids = {
            row["truth_id"]
            for row in json.loads(TRUTH.read_text(encoding="utf-8"))["records"]
        }

    def assert_epoch_parameter_source(self, value: str) -> None:
        prefix = "Epoch Parameters"
        self.assertTrue(value == prefix or value.startswith(prefix + "; canonicalized by "))
        if value != prefix:
            self.assertIn(value.removeprefix(prefix + "; canonicalized by "), self.truth_ids)

    def test_manual_map_is_exactly_reconciled(self) -> None:
        by_id = {row["openrouter_model_id"]: row for row in self.audit}
        self.assertEqual(len(by_id), len(self.audit))
        self.assertEqual(len(self.audit), self.result["data_audit"]["openrouter_catalog_models"])
        self.assertEqual(
            sum(row["match_status"] == "matched_epoch_manual" for row in self.audit),
            len(MANUAL_EPOCH_MAP),
        )
        for model_id, checkpoint_id in MANUAL_EPOCH_MAP.items():
            self.assertIn(model_id, by_id)
            self.assertEqual(by_id[model_id]["canonical_checkpoint_id"], checkpoint_id)
            self.assert_epoch_parameter_source(by_id[model_id]["parameter_value_source"])
            self.assertGreater(float(by_id[model_id]["total_parameters_b"]), 0)
            self.assertIn(
                by_id[model_id]["catalog_date_comparison_flag"],
                {"within_60_days", "reviewed_large_delta"},
            )

    def test_calibration_has_no_duplicate_models_or_non_epoch_targets(self) -> None:
        checkpoint_ids = [row["canonical_checkpoint_id"] for row in self.calibration]
        self.assertEqual(len(checkpoint_ids), len(set(checkpoint_ids)))
        self.assertEqual(len(self.calibration), self.result["data_audit"]["unique_epoch_calibration_checkpoints"])
        self.assertGreaterEqual(len(self.calibration), 90)
        self.assertGreaterEqual(len({row["family"] for row in self.calibration}), 15)
        for row in self.calibration:
            self.assert_epoch_parameter_source(row["parameter_value_source"])
            self.assertGreater(float(row["blended_price_usd_per_mtoken"]), 0)
            self.assertGreater(float(row["raw_throughput_tps_1w"]), 0)
            self.assertGreater(float(row["provider_normalized_throughput_ratio"]), 0)
        self.assertGreaterEqual(
            sum("canonicalized by" in row["parameter_value_source"] for row in self.calibration),
            6,
        )

    def test_disclosures_are_external_checks_not_training_rows(self) -> None:
        calibration_ids = "|".join(row["openrouter_model_ids"] for row in self.calibration)
        frontier_by_id = {row["openrouter_model_id"]: row for row in self.frontier}
        for model_id, record in DISCLOSED_ANCHORS.items():
            self.assertNotIn(model_id, calibration_ids)
            self.assertEqual(float(frontier_by_id[model_id]["disclosed_total_parameters_b"]), record["total_parameters_b"])
            self.assertEqual(frontier_by_id[model_id]["status"], "external disclosed check")
        self.assertEqual(
            frontier_by_id["moonshotai/kimi-k3"]["disclosed_parameter_source"],
            "Kimi K3 official technical report Table 1",
        )

    def test_held_out_evidence_supports_price_but_not_throughput(self) -> None:
        conclusion = self.result["conclusion"]
        self.assertIs(conclusion["openrouter_price_is_family_heldout_predictive"], True)
        self.assertIs(conclusion["tok_s_adds_robust_incremental_information_beyond_price"], False)
        self.assertEqual(conclusion["recommended_incremental_tok_s_weight_in_live_ensemble"], 0.0)
        family = self.result["heldout_metrics"]["family"]
        self.assertLess(
            family["date_price"]["mean_absolute_log10_error"],
            family["date_only"]["mean_absolute_log10_error"],
        )
        self.assertGreaterEqual(
            family["date_price_provider_normalized_throughput"]["mean_absolute_log10_error"],
            family["date_price"]["mean_absolute_log10_error"],
        )

    def test_same_base_controls_measure_operational_noise(self) -> None:
        controls = self.result["same_base_operational_noise_controls"]
        opus = controls["Claude Opus 4.5–4.8 shared-base control"]
        self.assertGreater(opus["raw_throughput_max_over_min"], 1.5)
        self.assertEqual(opus["price_max_over_min"], 1.0)
        self.assertGreater(controls["GPT-5–5.5 same-base control (standard endpoints only)"]["raw_throughput_max_over_min"], 1.05)

    def test_manifest_hashes_and_prediction_keys(self) -> None:
        self.assertEqual(self.result["snapshot_date"], "2026-07-31")
        self.assertEqual(
            self.result["compatibility_filename_date"], DATE
        )
        self.assertTrue(
            all(row["snapshot_date"] == "2026-07-31" for row in self.audit)
        )
        for relative_path, digest in self.result["source_manifest"].items():
            path = ROOT / relative_path
            self.assertTrue(path.exists())
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
        prediction_keys = [
            (row["mode"], row["specification"], row["canonical_checkpoint_id"])
            for row in self.predictions
        ]
        self.assertEqual(len(prediction_keys), len(set(prediction_keys)))


if __name__ == "__main__":
    unittest.main()
