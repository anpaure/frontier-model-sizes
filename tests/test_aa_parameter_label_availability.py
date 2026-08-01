from __future__ import annotations

import copy
import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from aa_calibration_overrides import parameter_training_eligibility_date
from aa_parameter_label_availability import (
    LEDGER_PATH,
    load_parameter_label_availability,
    resolve_parameter_label_available_date,
)


ROOT = Path(__file__).resolve().parents[1]


class AAParameterLabelAvailabilityTest(unittest.TestCase):
    def test_six_primary_source_records_are_hash_pinned(self) -> None:
        payload = load_parameter_label_availability()
        self.assertEqual(len(payload["records"]), 6)
        self.assertEqual(
            {record["identity"]["canonical_model_name"] for record in payload["records"]},
            {
                "LongCat Flash Lite",
                "MiniMax-M2.7",
                "MiMo-V2.5",
                "MiniMax-M3",
                "Nex-N2-Pro",
                "LongCat 2.0",
            },
        )
        self.assertGreaterEqual(
            sum(len(record["local_evidence"]) for record in payload["records"]),
            24,
        )

    def test_launch_labels_are_not_replaced_by_later_weight_dates(self) -> None:
        records = {
            record["identity"]["canonical_model_name"]: record
            for record in load_parameter_label_availability()["records"]
        }
        mimo = records["MiMo-V2.5"]
        self.assertEqual(
            mimo["timing"]["parameter_label_available_date"], "2026-04-22"
        )
        self.assertEqual(mimo["timing"]["weights_available_date"], "2026-04-27")
        longcat = records["LongCat 2.0"]
        self.assertEqual(
            longcat["timing"]["parameter_label_available_date"], "2026-06-30"
        )
        self.assertEqual(
            longcat["timing"]["weights_available_date"], "2026-07-05"
        )

    def test_current_panel_resolves_five_post_release_label_lags(self) -> None:
        regression = json.loads((ROOT / "regression_results.json").read_text())
        rows = regression["open_models"]
        delayed = {
            row["model"]: resolve_parameter_label_available_date(row)
            for row in rows
            if resolve_parameter_label_available_date(row) > row["release_date"]
        }
        self.assertEqual(
            delayed,
            {
                "LongCat Flash Lite": "2026-01-29",
                "MiniMax-M2.7": "2026-04-09",
                "MiniMax-M3": "2026-06-12",
                "Nex-N2-Pro": "2026-06-03",
                "LongCat 2.0": "2026-06-30",
            },
        )
        self.assertTrue(
            all(
                resolve_parameter_label_available_date(row) <= "2026-07-31"
                for row in rows
            )
        )

    def test_exactly_five_live_aa_backtest_folds_lose_a_premature_label(self) -> None:
        regression = json.loads((ROOT / "regression_results.json").read_text())
        rows = regression["open_models"]
        affected: dict[str, list[str]] = {}
        for test in rows:
            premature = [
                row["model"]
                for row in rows
                if row["release_date"] < test["release_date"]
                and resolve_parameter_label_available_date(row) >= test["release_date"]
                and row["family"] != test["family"]
            ]
            if premature:
                affected[test["model"]] = premature
        self.assertEqual(
            affected,
            {
                "GLM-5.1 Reasoning": ["MiniMax-M2.7"],
                "Nex-N2-Pro": ["MiniMax-M3"],
                "Nemotron 3 Ultra 550B A55B": ["MiniMax-M3"],
                "North Mini Code": ["MiniMax-M3"],
                "Kimi K2.7 Code": ["MiniMax-M3"],
            },
        )

    def test_rounded_minimax_value_is_explicit_but_other_mismatches_fail(self) -> None:
        base = {
            "model": "MiniMax-M2.7",
            "release_date": "2026-03-18",
            "total_b": 229.0,
            "parameter_source": "https://huggingface.co/MiniMaxAI/MiniMax-M2.7",
        }
        self.assertEqual(resolve_parameter_label_available_date(base), "2026-04-09")
        rounded = {**base, "total_b": 230.0}
        self.assertEqual(
            resolve_parameter_label_available_date(rounded), "2026-04-09"
        )
        canonicalized = {
            **base,
            "total_b": 228.703644928,
            "raw_total_b": 230.0,
            "parameter_truth_id": "minimax-m2-7-official-safetensors",
        }
        self.assertEqual(
            resolve_parameter_label_available_date(canonicalized), "2026-04-09"
        )
        with self.assertRaisesRegex(ValueError, "parameter mismatch"):
            resolve_parameter_label_available_date({**base, "total_b": 231.0})
        with self.assertRaisesRegex(ValueError, "release mismatch"):
            resolve_parameter_label_available_date(
                {**base, "release_date": "2026-03-19"}
            )

    def test_tampered_evidence_hash_fails_closed(self) -> None:
        payload = copy.deepcopy(
            json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
        )
        payload["records"][0]["local_evidence"][0]["sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "timing.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "evidence hash mismatch"):
                load_parameter_label_availability(path)


if __name__ == "__main__":
    unittest.main()
