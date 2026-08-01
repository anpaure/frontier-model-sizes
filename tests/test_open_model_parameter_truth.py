from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path

from open_model_parameter_truth import apply_parameter_truth, load_parameter_truth


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "sources/open_model_parameter_truth_reconciliation_2026-07-31.json"


class OpenModelParameterTruthTest(unittest.TestCase):
    def test_sources_and_primary_values_reconcile(self) -> None:
        payload = load_parameter_truth()
        records = {row["truth_id"]: row for row in payload["records"]}
        kimi = records["moonshot-kimi-k2-family-report-table-1"]
        self.assertEqual(kimi["canonical_total_parameters_b"], 1040.0)
        self.assertEqual(kimi["canonical_active_parameters_b"], 32.6)
        self.assertTrue(kimi["parameter_shape"]["same_parameter_shape"])
        self.assertEqual(kimi["parameter_shape"]["same_weight_identity"], "unproven")
        for key in (
            "minimax-m2-5-official-safetensors",
            "minimax-m2-7-official-safetensors",
        ):
            self.assertEqual(records[key]["exact_tensor_parameters"], 228_703_644_928)
            self.assertAlmostEqual(records[key]["canonical_total_parameters_b"], 228.703644928)
        for source in payload["source_files"]:
            path = ROOT / source["path"]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), source["sha256"])

    def test_overlay_preserves_raw_values_and_checkpoint_identity(self) -> None:
        kimi = apply_parameter_truth(
            {"model": "Kimi K2.6 Reasoning", "total_b": 1000, "active_b": 32, "checkpoint": "k2.6"}
        )
        self.assertEqual(kimi["checkpoint"], "k2.6")
        self.assertEqual(kimi["total_b"], 1040.0)
        self.assertEqual(kimi["active_b"], 32.6)
        self.assertEqual(kimi["raw_total_b"], 1000)
        self.assertEqual(kimi["raw_active_b"], 32)
        minimax = apply_parameter_truth(
            {"name": "MiniMax-M2.7", "parameters_b": 230, "active_parameters_b": 10}
        )
        self.assertAlmostEqual(minimax["parameters_b"], 228.703644928)
        self.assertEqual(minimax["raw_parameters_b"], 230)

    def test_unrecognized_models_are_unchanged(self) -> None:
        row = {"model": "Other", "total_b": 7.0, "active_b": 7.0}
        self.assertEqual(apply_parameter_truth(row), row)

    def test_offline_collector_is_byte_stable(self) -> None:
        before = LEDGER.read_bytes()
        subprocess.run(
            [sys.executable, str(ROOT / "collect_open_model_parameter_truth.py")],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(before, LEDGER.read_bytes())


if __name__ == "__main__":
    unittest.main(verbosity=2)
