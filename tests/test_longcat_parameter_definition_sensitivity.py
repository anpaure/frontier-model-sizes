from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "outputs/019f6c42-2d53-7743-ab07-6293e2618dd7/longcat_parameter_definition_sensitivity_2026-07-31.json"
REPORT = ROOT / "LONGCAT_PARAMETER_DEFINITION_SENSITIVITY.md"
TARGET_CSV = ROOT / "outputs/019f6c42-2d53-7743-ab07-6293e2618dd7/longcat_parameter_definition_target_sensitivity_2026-07-31.csv"
METRICS_CSV = ROOT / "outputs/019f6c42-2d53-7743-ab07-6293e2618dd7/longcat_parameter_definition_backtest_sensitivity_2026-07-31.csv"
SCRIPT = ROOT / "analyze_longcat_parameter_definition_sensitivity.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class LongCatParameterDefinitionSensitivityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = json.loads(RESULT.read_text(encoding="utf-8"))

    def test_definition_inventory_reconciles(self) -> None:
        evidence = self.data["definition_evidence"]
        hf = evidence["hugging_face_serialized_inventory"]
        tensor = evidence["tensor_name_inventory"]
        derived = evidence["derived_reconciliation"]
        self.assertEqual(evidence["publisher_definition"]["total_parameters"], 1_600_000_000_000)
        self.assertEqual(hf["safetensors_total_elements"], 1_775_560_491_136)
        self.assertEqual(sum(row["elements"] for row in tensor.values()), hf["safetensors_total_elements"])
        self.assertEqual(sum(row["tensor_entries"] for row in tensor.values()), hf["tensor_entries"])
        self.assertEqual(tensor["mtp"]["elements"], 136_943_683_712)
        self.assertEqual(derived["serialized_elements_excluding_mtp"], 1_638_616_807_424)
        self.assertTrue(derived["excluding_mtp_rounds_to_publisher_label"])

    def test_counterfactual_is_narrow_and_does_not_cherry_pick(self) -> None:
        scenarios = self.data["scenarios"]
        self.assertEqual(set(scenarios), {"publisher_model_total", "hf_serialized_tensor_elements"})
        self.assertEqual(
            scenarios["publisher_model_total"]["frontier_regression"]["selected_spec"],
            scenarios["hf_serialized_tensor_elements"]["frontier_regression"]["selected_spec"],
        )
        changes = {row["model"]: row for row in self.data["target_changes"]}
        self.assertEqual(len(changes), 12)
        self.assertLess(max(abs(row["change_percent"]) for row in changes.values()), 1.0)
        self.assertEqual(
            {row["model"] for row in self.data["changed_aa_predictions"]},
            {"Inkling (xhigh)", "LongCat 2.0"},
        )

    def test_matched_ensemble_and_live_forecasts_are_invariant(self) -> None:
        scenarios = self.data["scenarios"]
        publisher = scenarios["publisher_model_total"]["backtest"]["metrics"]
        hf = scenarios["hf_serialized_tensor_elements"]["backtest"]["metrics"]
        self.assertEqual(publisher["ensemble_all"], hf["ensemble_all"])
        self.assertEqual(publisher["ensemble_frontier"], hf["ensemble_frontier"])
        live = self.data["live_dependency_audit"]
        self.assertFalse(live["longcat_parameter_target_consumed_by_live_center"])
        self.assertEqual(len(live["downstream_targets"]), 3)
        for row in live["downstream_targets"]:
            self.assertEqual(row["evidence_delta_percent"], 0.0)
            self.assertEqual(row["final_delta_percent"], 0.0)
        self.assertEqual(self.data["decision"]["incremental_live_weight"], 0.0)
        self.assertFalse(self.data["decision"]["change_live_forecast"])

    def test_regeneration_is_deterministic_and_offline(self) -> None:
        paths = (RESULT, REPORT, TARGET_CSV, METRICS_CSV)
        before = {path: sha256(path) for path in paths}
        subprocess.run([sys.executable, str(SCRIPT)], cwd=ROOT, check=True, capture_output=True, text=True)
        after = {path: sha256(path) for path in paths}
        self.assertEqual(before, after)
        regenerated = json.loads(RESULT.read_text(encoding="utf-8"))
        self.assertEqual(regenerated["metadata"]["network_reads"], 0)
        self.assertFalse(regenerated["metadata"]["raw_sources_modified"])


if __name__ == "__main__":
    unittest.main()
