from __future__ import annotations

import csv
import hashlib
import io
import json
import tarfile
import unittest
import zipfile
from collections import defaultdict
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "019f6c42-2d53-7743-ab07-6293e2618dd7"
ARCHIVE = ROOT / "sources" / "epoch_eci_historical_snapshots_2026-07-18.tar.gz"
FOLDS = ROOT / "sources" / "eci_historical_common_component_folds_2026-07-31.csv"
RESULT = OUT / "eci_historical_common_component_audit_2026-07-31.json"
PREDICTIONS = OUT / "eci_historical_common_component_predictions_2026-07-31.csv"
TRAINING = OUT / "eci_historical_common_component_training_panel_2026-07-31.csv"
LOCKED = "GPQA diamond|MATH level 5|OTIS Mock AIME 2024-2025"
SNAPSHOT_HASHES = {
    "20250305190317": "b1b89260bd87fc0f84561046d88b035526f2d8f66b849c4613ec75ebe9a565cf",
    "20250403051524": "d74d26b56e0201ebbbd990b67c45f0eafe234d9a9d32a556e36ead1e14c416be",
    "20250510183121": "ad0e0a70b60b6e1ec7b8df60ec52903a029fe58ed21c2c31e385fef53f720f45",
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class EciHistoricalCommonComponentsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = json.loads(RESULT.read_text(encoding="utf-8"))
        cls.folds = read_csv(FOLDS)
        cls.predictions = read_csv(PREDICTIONS)
        cls.training = read_csv(TRAINING)

    def test_fold_and_benchmark_selection_is_locked(self) -> None:
        self.assertEqual(len(self.folds), 4)
        self.assertEqual(
            {row["target_model"] for row in self.folds},
            {
                "Gemma 3 27B",
                "Mistral Small 3.1",
                "Llama 4 Scout",
                "Llama 4 Maverick",
            },
        )
        self.assertEqual({row["locked_benchmarks"] for row in self.folds}, {LOCKED})
        expected_versions = {
            "Gemma 3 27B": "gemma-3-27b-it",
            "Mistral Small 3.1": "mistral-small-2503",
            "Llama 4 Scout": "Llama-4-Scout-17B-16E-Instruct",
            "Llama 4 Maverick": "Llama-4-Maverick-17B-128E-Instruct-FP8",
        }
        self.assertEqual(
            {row["target_model"]: row["target_model_version"] for row in self.folds},
            expected_versions,
        )
        collection = json.loads(
            (
                ROOT
                / "sources"
                / "epoch_eci_historical_collection_metadata_2026-07-18.json"
            ).read_text(encoding="utf-8")
        )
        ordered = sorted(
            row["timestamp"]
            for row in collection["captures"]
            if row["kind"] == "benchmark_zip"
        )
        for row in self.folds:
            index = ordered.index(row["training_snapshot_timestamp"])
            self.assertEqual(
                ordered[index + 1], row["target_score_snapshot_timestamp"]
            )

    def test_snapshot_and_inner_table_hashes_are_exact(self) -> None:
        inventory = self.result["sources"]["snapshot_inventory"]
        self.assertEqual(set(inventory), set(SNAPSHOT_HASHES))
        with tarfile.open(ARCHIVE, mode="r:gz") as archive:
            for timestamp, expected in SNAPSHOT_HASHES.items():
                handle = archive.extractfile(
                    f"benchmark_zip/benchmark_data_{timestamp}.zip"
                )
                self.assertIsNotNone(handle)
                payload = handle.read()
                self.assertEqual(sha256_bytes(payload), expected)
                self.assertEqual(inventory[timestamp]["sha256"], expected)
                with zipfile.ZipFile(io.BytesIO(payload)) as zipped:
                    for name, digest in inventory[timestamp][
                        "required_member_sha256"
                    ].items():
                        self.assertEqual(sha256_bytes(zipped.read(name)), digest)

    def test_prediction_and_parameter_truth_inventory(self) -> None:
        self.assertEqual(len(self.predictions), 8)
        keys = [
            (row["target_model"], row["weight_mode"]) for row in self.predictions
        ]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(
            {row["weight_mode"] for row in self.predictions},
            {"equal_developer", "equal_checkpoint"},
        )
        truth = {
            "Gemma 3 27B": 27.0,
            "Mistral Small 3.1": 24.0,
            "Llama 4 Scout": 109.0,
            "Llama 4 Maverick": 400.0,
        }
        for row in self.predictions:
            self.assertEqual(row["locked_benchmarks"], LOCKED)
            self.assertEqual(float(row["actual_b"]), truth[row["target_model"]])
            self.assertEqual(row["project_preregistered"], "False")
            self.assertTrue(row["parameter_source"].startswith("https://"))
            for field in (
                "target_gpqa_diamond",
                "target_math_level_5",
                "target_otis_mock_aime",
            ):
                self.assertGreaterEqual(float(row[field]), 0)
                self.assertLessEqual(float(row[field]), 1)

    def test_outer_training_is_strict_and_has_no_identity_leakage(self) -> None:
        fold_by_target = {row["target_model"]: row for row in self.folds}
        keys = [(row["target_model"], row["training_model"]) for row in self.training]
        self.assertEqual(len(keys), len(set(keys)))
        expected_counts = {
            "Gemma 3 27B": 14,
            "Mistral Small 3.1": 13,
            "Llama 4 Scout": 11,
            "Llama 4 Maverick": 11,
        }
        observed_counts = defaultdict(int)
        for row in self.training:
            observed_counts[row["target_model"]] += 1
            fold = fold_by_target[row["target_model"]]
            self.assertLess(
                date.fromisoformat(row["training_release_date"]),
                date.fromisoformat(fold["target_release_date"]),
            )
            self.assertNotIn(
                fold["target_developer_token"].lower(),
                row["training_organization"].lower(),
            )
            family = fold["target_family_token"].lower()
            self.assertNotIn(family, row["training_model"].lower())
            self.assertNotIn(family, row["training_model_version"].lower())
        self.assertEqual(dict(observed_counts), expected_counts)

    def test_equal_developer_weights_really_equalize_developers(self) -> None:
        by_target: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in self.training:
            by_target[row["target_model"]].append(row)
        for target, rows in by_target.items():
            totals: dict[str, float] = defaultdict(float)
            for row in rows:
                totals[row["training_organization"]] += float(
                    row["equal_developer_weight"]
                )
            self.assertAlmostEqual(sum(totals.values()), len(rows), places=12)
            values = list(totals.values())
            self.assertLess(max(values) - min(values), 1e-12, target)

    def test_primary_predictions_and_zero_weight_gates_are_frozen(self) -> None:
        primary = {
            row["target_model"]: float(row["predicted_b"])
            for row in self.predictions
            if row["weight_mode"] == "equal_developer"
        }
        expected = {
            "Gemma 3 27B": 89.02469923705371,
            "Mistral Small 3.1": 28.44036701073199,
            "Llama 4 Scout": 37.83984897374788,
            "Llama 4 Maverick": 113.71757123172002,
        }
        for model, value in expected.items():
            self.assertAlmostEqual(primary[model], value, places=10)
        summary = self.result["summary"]["equal_developer"]
        self.assertAlmostEqual(
            summary["median_multiplicative_error"], 3.0818530540228397, places=12
        )
        self.assertEqual(summary["within_2x"], 0.25)
        decision = self.result["decision"]
        self.assertFalse(decision["promote_to_live_model"])
        self.assertEqual(decision["live_weight"], 0)
        failed = {
            name for name, gate in decision["gates"].items() if not gate["passes"]
        }
        self.assertEqual(
            failed,
            {
                "target_count",
                "target_developer_count",
                "within_2x",
                "median_error",
                "project_preregistration",
            },
        )

    def test_output_hashes_and_pipeline_wiring(self) -> None:
        self.assertEqual(self.result["sources"]["fold_ledger_sha256"], sha256(FOLDS))
        self.assertEqual(
            self.result["sources"]["historical_archive_sha256"], sha256(ARCHIVE)
        )
        self.assertEqual(
            self.result["outputs"]["predictions_sha256"], sha256(PREDICTIONS)
        )
        self.assertEqual(
            self.result["outputs"]["training_panel_sha256"], sha256(TRAINING)
        )
        pipeline = (ROOT / "run_forecast_pipeline.py").read_text(encoding="utf-8")
        aggregate = pipeline.index("analyze_eci_historical_validation_extension.py")
        component = pipeline.index("analyze_eci_historical_common_components.py")
        self.assertLess(aggregate, component)
        self.assertIn("tests.test_eci_historical_common_components", pipeline)


if __name__ == "__main__":
    unittest.main()
