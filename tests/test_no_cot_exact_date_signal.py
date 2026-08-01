import csv
import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/019f6c42-2d53-7743-ab07-6293e2618dd7"
RESULT = OUTPUT / "no_cot_exact_date_audit_2026-07-18.json"
MODEL_AUDIT = OUTPUT / "no_cot_exact_date_model_audit_2026-07-18.csv"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class NoCotExactDateSignalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = json.loads(RESULT.read_text(encoding="utf-8"))
        with MODEL_AUDIT.open(newline="", encoding="utf-8") as handle:
            cls.rows = list(csv.DictReader(handle))
        cls.by_model = {row["model"]: row for row in cls.rows}

    def test_every_no_cot_model_has_one_day_level_date(self):
        inventory = self.result["inventory"]
        self.assertEqual(inventory["no_cot_models"], 49)
        self.assertEqual(inventory["models_with_day_level_dates"], 49)
        self.assertEqual(inventory["models_remaining_month_only"], 0)
        self.assertEqual(len(self.rows), 49)
        self.assertEqual(len(self.by_model), 49)

    def test_overrides_add_dates_without_parameter_identities(self):
        self.assertEqual(self.result["inventory"]["explicit_date_only_overrides"], 4)
        self.assertEqual(self.result["inventory"]["parameter_identities_added_by_overrides"], 0)
        for model in ["GPT-2", "GPT-3", "GPT-3.5", "Qwen 3 30B-A3B (2507)"]:
            self.assertEqual(
                self.by_model[model]["parameter_join_policy"],
                "date_only_no_epoch_parameter_join",
            )

    def test_exact_date_adjustment_is_small_and_recomputable(self):
        for axis in ["time_horizon", "token_horizon"]:
            branch = self.result[axis]
            month = branch["month_date_approximation"]["doubling_time_days"]
            exact = branch["exact_date_approximation"]["doubling_time_days"]
            law = branch["adjusted_reported_law"]
            ratio = exact / month
            self.assertAlmostEqual(law["exact_date_adjustment_ratio"], ratio, places=12)
            self.assertAlmostEqual(
                law["adjusted_point_days"],
                law["paper_reported_point_days"] * ratio,
                places=12,
            )
            self.assertLess(abs(ratio - 1.0), 0.03)
        self.assertAlmostEqual(
            self.result["time_horizon"]["adjusted_reported_law"]["adjusted_point_days"],
            365.8606542885574,
            places=9,
        )

    def test_exact_dates_change_time_frontier_membership_transparently(self):
        month = set(self.result["time_horizon"]["month_date_approximation"]["frontier_models"])
        exact = set(self.result["time_horizon"]["exact_date_approximation"]["frontier_models"])
        self.assertNotIn("Opus 4.7", month)
        self.assertIn("Opus 4.7", exact)
        self.assertIn("GPT-5.5", month)
        self.assertIn("GPT-5.5", exact)

    def test_all_source_hashes_reconcile(self):
        for relative, expected in self.result["source_hashes"].items():
            self.assertEqual(sha256(ROOT / relative), expected)
        self.assertTrue(self.result["decision"]["use_exact_date_adjusted_time_law"])
        self.assertFalse(self.result["decision"]["change_live_no_cot_weight"])


if __name__ == "__main__":
    unittest.main()
