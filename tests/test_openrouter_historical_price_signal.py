import csv
import gzip
import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATE = "2026-07-18"
OUT = ROOT / "outputs/019f6c42-2d53-7743-ab07-6293e2618dd7"
RAW = ROOT / f"sources/openrouter_historical_price_ledger_{DATE}.json.gz"
SIGNALS = ROOT / f"sources/openrouter_historical_price_change_points_{DATE}.csv"
METADATA = ROOT / f"sources/openrouter_historical_price_collection_metadata_{DATE}.json"
MATCHES = OUT / f"openrouter_historical_price_match_audit_{DATE}.csv"
PREDICTIONS = OUT / f"openrouter_historical_price_backtest_predictions_{DATE}.csv"
TARGETS = OUT / f"openrouter_historical_price_frontier_targets_{DATE}.csv"
RESULT = OUT / f"openrouter_historical_price_audit_{DATE}.json"


def rows(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


class OpenRouterHistoricalPriceAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.signals = rows(SIGNALS)
        cls.matches = rows(MATCHES)
        cls.predictions = rows(PREDICTIONS)
        cls.targets = rows(TARGETS)
        cls.result = json.loads(RESULT.read_text())
        cls.metadata = json.loads(METADATA.read_text())

    def test_lossless_pinned_source_inventory(self):
        with gzip.open(RAW, "rb") as handle:
            payload = handle.read()
        self.assertEqual(
            hashlib.sha256(payload).hexdigest(),
            "d572cdb4b64b0b4ed878e570490b937666a09b49f5cef1199dc8869825473acc",
        )
        parsed = json.loads(payload)
        self.assertEqual(parsed["as_of"], "2026-07-31")
        self.assertEqual(parsed["model_count"], 914)
        self.assertEqual(len(self.signals), 2566)
        self.assertEqual(
            len({row["openrouter_model_id"] for row in self.signals}), 914
        )
        self.assertEqual(
            len(
                {
                    (row["openrouter_model_id"], row["change_index"])
                    for row in self.signals
                }
            ),
            2566,
        )
        self.assertEqual(
            self.metadata["source"]["pinned_commit"],
            "1cd0e2ec5fccb271df9d1140abc91aaf20b3e878",
        )
        self.assertEqual(self.metadata["generated_on"], "2026-07-31")
        self.assertEqual(
            self.metadata["compatibility_filename_date"], "2026-07-18"
        )
        self.assertTrue(
            self.metadata["integrity_policy"]["all_change_points_preserved"]
        )

    def test_all_calibration_aliases_are_exactly_matched_once(self):
        self.assertEqual(len(self.matches), 93)
        self.assertEqual(
            len({row["canonical_checkpoint_id"] for row in self.matches}), 93
        )
        self.assertTrue(
            all(row["all_aliases_exactly_matched"] == "True" for row in self.matches)
        )
        self.assertEqual(
            self.result["inventory"]["calibration_aliases_missing_from_history"], 0
        )
        self.assertEqual(
            self.result["inventory"]["eligible_total_rows_by_window"]["1"], 73
        )
        self.assertEqual(
            self.result["inventory"]["eligible_active_rows_by_window"]["1"], 55
        )

    def test_every_historical_fold_is_strictly_prospective_and_developer_held_out(self):
        self.assertEqual(len(self.predictions), 3098)
        for row in self.predictions:
            self.assertLess(
                row["train_max_price_availability_date"],
                row["price_availability_date"],
            )
            self.assertEqual(row["test_developer_excluded"], "True")
            self.assertEqual(
                row["historical_price_is_prospective_at_fold_date"], "True"
            )
            if "current_price" in row["specification"]:
                self.assertEqual(
                    row["current_price_is_nonprospective_comparison"], "True"
                )
            else:
                self.assertEqual(
                    row["current_price_is_nonprospective_comparison"], "False"
                )

    def test_price_beats_date_but_not_score_date_robustly(self):
        self.assertEqual(self.result["metadata"]["generated_on"], "2026-07-31")
        self.assertEqual(
            self.result["metadata"]["compatibility_filename_date"], DATE
        )
        decision = self.result["decision"]
        self.assertTrue(
            decision[
                "launch_vintage_price_predicts_total_beyond_date_robustly_across_all_windows"
            ]
        )
        self.assertFalse(
            decision[
                "launch_vintage_price_adds_robust_information_beyond_score_date_for_active_parameters"
            ]
        )
        self.assertFalse(decision["change_existing_live_price_weight"])
        self.assertEqual(decision["incremental_live_weight_from_this_audit"], 0.0)
        comparisons = self.result["paired_developer_bootstraps"]
        total = [row for row in comparisons if row["panel"] == "total_calibration"]
        active = [
            row
            for row in comparisons
            if row["panel"] == "active_label_common_panel"
            and row["target"] == "active_b"
        ]
        self.assertEqual(len(total), 7)
        self.assertEqual(len(active), 7)
        self.assertTrue(all(row["ci_90"][1] < 0 for row in total))
        self.assertTrue(all(row["ci_90"][1] >= 0 for row in active))

    def test_frontier_first_day_prices_and_anchor_are_exact(self):
        by_model = {row["model"]: row for row in self.targets}
        expected = {
            "Claude Fable 5": (10.0, 50.0),
            "GPT-5.6 Sol": (5.0, 30.0),
            "Kimi K3": (3.0, 15.0),
        }
        for model, (prompt, completion) in expected.items():
            self.assertEqual(
                float(by_model[model]["first_day_prompt_price_usd_per_mtoken"]),
                prompt,
            )
            self.assertEqual(
                float(
                    by_model[model]["first_day_completion_price_usd_per_mtoken"]
                ),
                completion,
            )
        headline = self.result["headline_crosscheck"]
        self.assertAlmostEqual(headline["k3_disclosed_t"], 2.78)
        self.assertGreater(
            headline["fable_k3_anchored_score_date_first_day_price_t"], 5.0
        )
        self.assertGreater(
            headline["sol_k3_anchored_score_date_first_day_price_t"], 3.5
        )
        self.assertEqual(headline["status"], "zero-weight crosscheck")


if __name__ == "__main__":
    unittest.main()
