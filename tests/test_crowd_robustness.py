from __future__ import annotations

import hashlib
import csv
import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = (
    ROOT
    / "outputs"
    / "019f6c42-2d53-7743-ab07-6293e2618dd7"
    / "crowd_robustness_audit_2026-07-31.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CrowdRobustnessAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = json.loads(RESULT.read_text(encoding="utf-8"))
        cls.forecast = json.loads(
            (ROOT / "site/public/data/forecast-model.json").read_text(encoding="utf-8")
        )
        with (ROOT / "sources/human_parameter_forecasts_2026-07-17.csv").open(
            newline="", encoding="utf-8-sig"
        ) as handle:
            ledger = list(csv.DictReader(handle))
        superseded = {row["supersedes"] for row in ledger if row["supersedes"]}
        cls.active = [row for row in ledger if row["forecast_id"] not in superseded]

    def test_pools_and_current_centers_reconcile(self) -> None:
        rows = {row["model"]: row for row in self.result["targets"]}
        forecast = {row["name"]: row for row in self.forecast["models"]}
        for model, row in rows.items():
            expected_n = sum(item["model"] == model for item in self.active)
            self.assertEqual(row["n"], expected_n)
            self.assertAlmostEqual(
                row["displayed_final_t"], forecast[model]["currentFinalT"]
            )

    def test_single_contributor_robustness_does_not_claim_independence(self) -> None:
        for row in self.result["targets"]:
            self.assertLess(
                row["leave_one_contributor_out"][
                    "maximum_absolute_final_shift_fraction"
                ],
                0.10,
            )
            self.assertGreater(row["individual_point_span_x"], 9)
        dependence = self.result["cross_target_dependence"]
        self.assertEqual(dependence["paired_contributors"], 18)
        self.assertGreater(dependence["pearson_correlation_of_log_points"], 0.7)
        decision = self.result["decision"]
        self.assertTrue(decision["center_is_single_forecaster_robust"])
        self.assertFalse(decision["crowd_is_independent_calibration_evidence"])
        self.assertFalse(decision["narrow_predictive_intervals"])
        self.assertFalse(decision["change_user_selected_crowd_weight"])

    def test_sources_reconcile_and_rebuild_is_byte_exact(self) -> None:
        for relative, expected in self.result["source_files"].items():
            self.assertEqual(sha256(ROOT / relative), expected)
        before = sha256(RESULT)
        subprocess.run(
            [sys.executable, str(ROOT / "analyze_crowd_robustness.py")],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(sha256(RESULT), before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
