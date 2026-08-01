from __future__ import annotations

import csv
import gzip
import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATE = "2026-07-18"
OUT = ROOT / "outputs/019f6c42-2d53-7743-ab07-6293e2618dd7"
RAW = ROOT / f"sources/huggingface_architecture_config_snapshot_{DATE}.json.gz"
SIGNALS = ROOT / f"sources/huggingface_architecture_config_signals_{DATE}.csv"
AUDIT = OUT / f"huggingface_architecture_config_collection_audit_{DATE}.json"


class HuggingFaceArchitectureConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with gzip.open(RAW, "rt", encoding="utf-8") as handle:
            cls.raw = json.load(handle)
        with SIGNALS.open(newline="", encoding="utf-8") as handle:
            cls.rows = list(csv.DictReader(handle))
        cls.audit = json.loads(AUDIT.read_text(encoding="utf-8"))

    def test_inventory_is_unique_and_complete(self) -> None:
        self.assertEqual(len(self.rows), 87)
        self.assertEqual(
            len({row["hugging_face_repo"] for row in self.rows}), 87
        )
        self.assertEqual(len(self.raw["entries"]), 87)
        self.assertEqual(self.audit["calibration_checkpoints"], 93)
        self.assertEqual(self.audit["successful_json_configs"], 73)
        self.assertEqual(
            self.audit["architecture_classification_counts"],
            {"dense_config": 25, "moe_config": 48, "unavailable": 14},
        )
        self.assertEqual(self.audit["http_status_counts"], {"200": 73, "401": 14})

    def test_primary_configs_distinguish_dense_and_moe(self) -> None:
        by_repo = {row["hugging_face_repo"]: row for row in self.rows}
        self.assertEqual(
            by_repo["qwen/qwen2.5-coder-32b-instruct"][
                "architecture_classification"
            ],
            "dense_config",
        )
        deepseek = by_repo["deepseek-ai/deepseek-v3"]
        self.assertEqual(deepseek["architecture_classification"], "moe_config")
        self.assertEqual(float(deepseek["routed_expert_count"]), 256)
        self.assertEqual(float(deepseek["experts_per_token"]), 8)
        ernie = by_repo["baidu/ernie-4.5-vl-424b-a47b-pt"]
        self.assertEqual(ernie["architecture_classification"], "moe_config")
        self.assertEqual(
            by_repo["meta-llama/llama-3.3-70b-instruct"][
                "architecture_classification"
            ],
            "unavailable",
        )

    def test_no_http_or_parse_failure_is_silently_classified_dense(self) -> None:
        for row in self.rows:
            if row["architecture_classification"] == "unavailable":
                self.assertNotEqual(row["status_code"], "200")
                self.assertTrue(row["error"])
            else:
                self.assertEqual(row["status_code"], "200")
                self.assertFalse(row["error"])
                self.assertEqual(len(row["content_sha256"]), 64)

    def test_hashes_reconcile(self) -> None:
        for relative, expected in self.audit["source_hashes"].items():
            actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
