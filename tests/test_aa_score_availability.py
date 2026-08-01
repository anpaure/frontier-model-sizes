from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from aa_calibration_overrides import parameter_training_eligibility_date
from aa_score_availability import (
    LEDGER_PATH,
    RAW_PATH,
    aa_prediction_information_date,
    aa_score_available_date,
    aa_score_availability_verified,
    load_aa_score_availability,
)


ROOT = Path(__file__).resolve().parents[1]


class AaScoreAvailabilityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ledger = load_aa_score_availability()
        cls.by_slug = {
            row["aa_slug"]: row for row in cls.ledger["records"]
        }

    def test_raw_snapshot_is_pinned_and_api_shortfall_is_explicit(self) -> None:
        summary = self.ledger["summary"]
        self.assertEqual(summary["pages"], 4)
        self.assertEqual(summary["total_changelog_events"], 1250)
        self.assertEqual(summary["api_reported_total_events"], 1561)
        self.assertIs(summary["api_total_reconciles"], False)
        self.assertEqual(summary["verified_score_slugs"], 136)
        evidence = self.ledger["raw_evidence"]
        self.assertEqual(
            hashlib.sha256(RAW_PATH.read_bytes()).hexdigest(),
            evidence["sha256"],
        )

    def test_known_score_publication_dates_are_exact_slug_matches(self) -> None:
        expected = {
            "kimi-k3": "2026-07-16",
            "motif-0714": "2026-07-20",
            "nex-n2-pro": "2026-06-24",
            "minimax-m3": "2026-06-03",
            "longcat-2-0": "2026-07-14",
        }
        for slug, available in expected.items():
            with self.subTest(slug=slug):
                self.assertEqual(
                    self.by_slug[slug]["score_available_date"], available
                )
                row = {"aa_slug": slug, "release_date": "2025-01-01"}
                self.assertEqual(aa_score_available_date(row), available)
                self.assertTrue(aa_score_availability_verified(row))

    def test_release_feature_and_information_date_remain_separate(self) -> None:
        row = {
            "model": "LongCat 2.0",
            "aa_slug": "longcat-2-0",
            "release_date": "2026-06-29",
            "parameters_b": 1600.0,
        }
        self.assertEqual(row["release_date"], "2026-06-29")
        self.assertEqual(aa_prediction_information_date(row), "2026-07-14")
        self.assertEqual(parameter_training_eligibility_date(row), "2026-07-14")

    def test_parameter_and_score_dates_both_gate_training(self) -> None:
        minimax = {
            "model": "MiniMax-M3",
            "aa_slug": "minimax-m3",
            "release_date": "2026-06-01",
            "parameters_b": 428.0,
            "model_weights_source_url": "https://huggingface.co/MiniMaxAI/MiniMax-M3",
        }
        self.assertEqual(aa_score_available_date(minimax), "2026-06-03")
        self.assertEqual(parameter_training_eligibility_date(minimax), "2026-06-12")

    def test_pipeline_verifies_pinned_timing_before_model_fit(self) -> None:
        source = (ROOT / "run_forecast_pipeline.py").read_text(encoding="utf-8")
        collector = 'ROOT / "collect_aa_score_availability.py"'
        test = '"tests.test_aa_score_availability"'
        fit = 'ROOT / "analyze_frontier_equivalence.py"'
        self.assertIn(collector, source)
        self.assertIn(test, source)
        self.assertLess(source.index(collector), source.index(fit))
        self.assertLess(source.index(test), source.index(fit))

    def test_regression_rows_preserve_publication_provenance(self) -> None:
        regression = json.loads((ROOT / "regression_results.json").read_text())
        # This assertion becomes active after the generator is rerun and guards
        # against future loss of the exact slug/date crosswalk.
        rows = regression["open_models"]
        self.assertTrue(all("aa_slug" in row for row in rows))
        by_model = {row["model"]: row for row in rows}
        self.assertEqual(
            by_model["LongCat 2.0"]["aa_score_available_date"], "2026-07-14"
        )
        self.assertTrue(
            by_model["LongCat 2.0"]["aa_score_availability_verified"]
        )


if __name__ == "__main__":
    unittest.main()
