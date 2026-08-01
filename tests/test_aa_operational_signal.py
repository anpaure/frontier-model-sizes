import csv
import hashlib
import json
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/019f6c42-2d53-7743-ab07-6293e2618dd7"
RESULT = OUT / "aa_operational_signal_audit_2026-07-18.json"
PANEL = OUT / "aa_operational_parameter_panel_2026-07-18.csv"
PREDICTIONS = OUT / "aa_operational_backtest_predictions_2026-07-18.csv"
CROSSCHECK = OUT / "aa_openrouter_operational_crosscheck_2026-07-18.csv"
DETAIL = ROOT / "sources/aa_detailed_model_signals_2026-07-31.csv"
OPENROUTER = OUT / "openrouter_parameter_calibration_2026-07-18.csv"


def rows(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def numeric(value):
    return None if value in (None, "") else float(value)


class AAOperationalSignalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = json.loads(RESULT.read_text(encoding="utf-8"))
        cls.panel = rows(PANEL)
        cls.predictions = rows(PREDICTIONS)
        cls.crosscheck = rows(CROSSCHECK)

    def test_complete_panel_reconciles_to_every_selected_raw_record(self):
        raw = {row["slug"]: row for row in rows(DETAIL)}
        self.assertEqual(len(raw), 587)
        self.assertEqual(len(self.panel), 275)
        self.assertEqual(
            len({row["checkpoint_group_id"] for row in self.panel}), 275
        )
        mapping = {
            "price_input_usd_per_mtoken": "price_input_usd_per_mtoken",
            "price_output_usd_per_mtoken": "price_output_usd_per_mtoken",
            "price_cache_hit_usd_per_mtoken": "price_cache_hit_usd_per_mtoken",
            "price_blended_7_2_1_usd_per_mtoken": "price_blended_7_2_1_usd_per_mtoken",
            "median_output_speed_tps": "median_output_speed_tps",
            "median_time_to_first_chunk_seconds": "median_time_to_first_chunk_seconds",
            "intelligence_cost_per_task_usd": "intelligence_cost_per_task_usd",
            "intelligence_time_per_task_seconds": "intelligence_time_per_task_seconds",
        }
        for row in self.panel:
            source = raw[row["selected_slug"]]
            self.assertEqual(row["release_date"], source["release_date"])
            self.assertEqual(
                row["performance_data_source_type"],
                source["performance_data_source_type"],
            )
            self.assertEqual(
                row["performance_provider_name"],
                source["performance_provider_name"],
            )
            for output_field, source_field in mapping.items():
                self.assertEqual(numeric(row[output_field]), numeric(source[source_field]))
            if row["parameter_truth_id"]:
                self.assertEqual(
                    numeric(row["raw_parameter_total_b"]),
                    numeric(source["parameters_b"]),
                )
            elif not row["calibration_override_id"]:
                self.assertEqual(
                    numeric(row["parameters_b"]),
                    numeric(source["parameters_b"]),
                )

        truth_rows = [row for row in self.panel if row["parameter_truth_id"]]
        self.assertGreaterEqual(len(truth_rows), 7)
        self.assertEqual(
            {row["parameter_truth_id"] for row in truth_rows},
            {
                "moonshot-kimi-k2-family-report-table-1",
                "minimax-m2-5-official-safetensors",
                "minimax-m2-7-official-safetensors",
            },
        )

        audit = self.result["data_audit"]
        coverage_fields = {
            "blended_price": "price_blended_7_2_1_usd_per_mtoken",
            "output_speed": "median_output_speed_tps",
            "latency_ttfc": "median_time_to_first_chunk_seconds",
            "cost_per_task": "intelligence_cost_per_task_usd",
            "time_per_task": "intelligence_time_per_task_seconds",
        }
        for signal, field in coverage_fields.items():
            covered = [row for row in self.panel if row[field]]
            self.assertEqual(audit["coverage"][signal]["checkpoints"], len(covered))
            self.assertEqual(
                audit["coverage"][signal]["developers"],
                len({row["creator_slug"] for row in covered}),
            )

    def test_all_predictions_are_strictly_chronological_and_developer_held_out(self):
        expected_specs = {
            specification: audit["scopes"]["all"]["n"]
            for specification, audit in self.result["backtests"].items()
        }
        self.assertEqual(len(self.predictions), sum(expected_specs.values()))
        counts = {}
        for row in self.predictions:
            counts[row["specification"]] = counts.get(row["specification"], 0) + 1
            self.assertEqual(row["test_developer_excluded"], "True")
            self.assertGreaterEqual(
                row["prediction_information_date"], row["release_date"]
            )
            self.assertLess(
                row["train_max_date"], row["prediction_information_date"]
            )
            self.assertGreaterEqual(int(row["train_n"]), int(row["minimum_train_rows"]))
            self.assertGreaterEqual(
                int(row["train_developers"]),
                int(row["minimum_train_developers"]),
            )
            actual = float(row["actual_parameters_b"])
            for prefix in ("baseline", "candidate"):
                prediction = float(row[f"{prefix}_predicted_b"])
                error = float(row[f"{prefix}_log10_error"])
                self.assertTrue(
                    math.isclose(
                        math.log10(prediction) - math.log10(actual),
                        error,
                        rel_tol=0,
                        abs_tol=2e-12,
                    )
                )
        self.assertEqual(counts, expected_specs)

    def test_exact_openrouter_crosscheck_is_lossless(self):
        self.assertEqual(len(self.crosscheck), 28)
        self.assertEqual(
            len({row["canonical_checkpoint_id"] for row in self.crosscheck}), 28
        )
        panel_by_slug = {row["selected_slug"]: row for row in self.panel}
        router = {
            row["canonical_checkpoint_id"]: row for row in rows(OPENROUTER)
        }
        for row in self.crosscheck:
            aa = panel_by_slug[row["aa_slug"]]
            source = router[row["canonical_checkpoint_id"]]
            self.assertEqual(
                numeric(row["aa_blended_price_usd_per_mtoken"]),
                numeric(aa["price_blended_7_2_1_usd_per_mtoken"]),
            )
            self.assertEqual(
                numeric(row["aa_output_speed_tps"]),
                numeric(aa["median_output_speed_tps"]),
            )
            self.assertEqual(
                numeric(row["openrouter_blended_price_usd_per_mtoken"]),
                numeric(source["blended_price_usd_per_mtoken"]),
            )
            self.assertEqual(
                numeric(row["openrouter_raw_throughput_tps"]),
                numeric(source["raw_throughput_tps_1w"]),
            )

        summary = self.result["aa_openrouter_exact_crosscheck"]
        self.assertEqual(summary["price"]["n_positive_pairs"], 26)
        self.assertGreater(summary["price"]["spearman"], 0.80)
        self.assertLess(
            summary["raw_speed"]["spearman"],
            summary["price"]["spearman"] - 0.30,
        )

    def test_decision_matches_heldout_evidence(self):
        tests = self.result["backtests"]
        median_price = tests["price_provider_median"]["scopes"]
        self.assertLess(
            median_price["all"]["candidate"]["mean_absolute_log10_error"],
            median_price["all"]["baseline"]["mean_absolute_log10_error"],
        )
        self.assertLess(
            median_price["frontier_like"]["candidate"]["mean_absolute_log10_error"],
            median_price["frontier_like"]["baseline"]["mean_absolute_log10_error"],
        )
        self.assertLess(
            median_price["frontier_like"]["paired_developer_bootstrap"]["ci_90"][1],
            0,
        )
        first_party = tests["price_first_party"]["scopes"]["frontier_like"]
        self.assertGreater(
            first_party["candidate"]["mean_absolute_log10_error"],
            first_party["baseline"]["mean_absolute_log10_error"],
        )
        speed = tests["output_speed"]["scopes"]["frontier_like"]
        self.assertGreater(
            speed["paired_developer_bootstrap"]["ci_90"][1], 0
        )
        decision = self.result["decision"]
        self.assertIs(decision["validates_existing_price_direction"], True)
        self.assertIs(decision["change_live_price_weight"], False)
        self.assertEqual(decision["incremental_aa_operational_price_weight"], 0)
        self.assertEqual(decision["incremental_aa_speed_weight"], 0)
        self.assertEqual(decision["incremental_aa_latency_weight"], 0)

    def test_source_hashes_reconcile(self):
        for relative, expected in self.result["source_manifest"].items():
            actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
