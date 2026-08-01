from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import sys
import unittest
from pathlib import Path

import analyze_eci_vintage_knowledge_residual as audit


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/019f6c42-2d53-7743-ab07-6293e2618dd7"
RESULT = OUT / "eci_vintage_knowledge_residual_audit_2026-07-31.json"
PREDICTIONS = OUT / "eci_vintage_knowledge_residual_predictions_2026-07-31.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


class EciVintageKnowledgeResidualTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = json.loads(RESULT.read_text(encoding="utf-8"))
        cls.predictions = rows(PREDICTIONS)

    def test_archive_inventory_is_exact_and_complete(self) -> None:
        inventory = self.result["inventory"]
        self.assertEqual(inventory["historical_snapshots"], 15)
        self.assertEqual(inventory["historical_score_rows"], 2308)
        self.assertEqual(inventory["historical_component_rows"], 20350)
        self.assertEqual(inventory["historical_knowledge_rows"], 4066)
        self.assertEqual(inventory["parameter_checkpoints"], 89)
        self.assertEqual(inventory["archive_benchmark_zip_eci_captures"], 10)
        self.assertEqual(inventory["archive_canonical_csv_captures"], 5)
        self.assertEqual(
            self.result["predeclared_model"]["knowledge_benchmarks"],
            list(audit.KNOWLEDGE_BENCHMARKS),
        )

    def test_outer_and_inner_splits_are_strict_developer_holdouts(self) -> None:
        self.assertEqual(len(self.predictions), self.result["inventory"]["outer_predictions"])
        keys = [(row["snapshot_timestamp"], row["model"]) for row in self.predictions]
        self.assertEqual(len(keys), len(set(keys)))
        for row in self.predictions:
            self.assertLess(row["train_max_date"], row["release_date"])
            self.assertEqual(row["test_developer_excluded"], "True")
            self.assertEqual(row["inner_strict_chronology"], "True")
            self.assertGreaterEqual(int(row["train_n"]), audit.MIN_OUTER_TRAIN_ROWS)
            self.assertGreaterEqual(
                int(row["train_developers"]), audit.MIN_OUTER_TRAIN_DEVELOPERS
            )
            self.assertGreaterEqual(int(row["inner_predictions"]), audit.MIN_INNER_PREDICTIONS)
            self.assertGreaterEqual(
                int(row["inner_developers"]), audit.MIN_INNER_VALIDATION_DEVELOPERS
            )
            self.assertIn(float(row["selected_alpha"]), audit.RIDGE_ALPHAS)

    def test_sparse_coverage_shrinkage_and_prediction_arithmetic_recompute(self) -> None:
        for row in self.predictions:
            observed = int(row["observed_benchmark_count"])
            expected_shrinkage = observed / (
                observed + audit.COVERAGE_PRIOR_BENCHMARKS
            )
            self.assertTrue(
                math.isclose(
                    float(row["coverage_shrinkage"]),
                    expected_shrinkage,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            )
            actual = float(row["actual_b"])
            for branch in ("baseline", "candidate"):
                predicted = float(row[f"{branch}_predicted_b"])
                expected_error = math.log10(predicted) - math.log10(actual)
                self.assertTrue(
                    math.isclose(
                        float(row[f"{branch}_log10_error"]),
                        expected_error,
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                )
                self.assertTrue(
                    math.isclose(
                        float(row[f"{branch}_multiplicative_error"]),
                        10 ** abs(expected_error),
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                )

    def test_component_residuals_use_same_snapshot_parameters(self) -> None:
        source = audit.load_vintage_sources()
        for row in self.predictions:
            timestamp = row["snapshot_timestamp"]
            model = row["model"]
            aggregate = source["scores"][timestamp][model]
            residuals = json.loads(row["target_component_residuals_json"])
            for benchmark, residual in residuals.items():
                performance = source["raw_components"][timestamp][model][benchmark]
                edi, slope = source["parameters"][(timestamp, benchmark)]
                expected = edi + audit.logit(performance) / slope - aggregate
                self.assertTrue(
                    math.isclose(residual, expected, rel_tol=1e-12, abs_tol=1e-12)
                )

    def test_archive_signal_is_positive_but_promotion_stays_zero_weight(self) -> None:
        all_rows = self.result["cohorts"]["all_first_observed"]
        self.assertLess(
            all_rows["candidate"]["median_multiplicative_error"],
            all_rows["baseline"]["median_multiplicative_error"],
        )
        self.assertLess(
            all_rows["paired_developer_bootstrap"]["ci_90"][1], 0
        )
        gates = self.result["promotion_gates"]
        self.assertFalse(gates["results"]["outer_coverage"])
        self.assertFalse(gates["results"]["prospective_coverage"])
        self.assertFalse(gates["results"]["current_target_component_coverage"])
        self.assertFalse(gates["all_pass"])
        decision = self.result["decision"]
        self.assertFalse(decision["promote_archive_vintage_knowledge_branch"])
        self.assertEqual(decision["incremental_live_weight"], 0.0)
        self.assertFalse(decision["change_live_forecasts"])

    def test_current_target_coverage_is_exactly_and_honestly_sparse(self) -> None:
        coverage = {row["model"]: row for row in self.result["current_target_coverage"]}
        self.assertEqual(coverage["Claude Fable 5"]["observed_count"], 1)
        self.assertEqual(coverage["GPT-5.6 Sol"]["observed_count"], 2)
        self.assertAlmostEqual(coverage["Claude Fable 5"]["coverage_shrinkage"], 1 / 3)
        self.assertAlmostEqual(coverage["GPT-5.6 Sol"]["coverage_shrinkage"], 1 / 2)
        self.assertTrue(all(not row["passes_minimum"] for row in coverage.values()))

    def test_source_hashes_reconcile(self) -> None:
        for relative, digest in self.result["source_files"].items():
            path = ROOT / relative
            self.assertTrue(path.is_file(), relative)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)

    def test_offline_rebuild_is_byte_exact(self) -> None:
        before = {
            RESULT: hashlib.sha256(RESULT.read_bytes()).hexdigest(),
            PREDICTIONS: hashlib.sha256(PREDICTIONS.read_bytes()).hexdigest(),
        }
        subprocess.run(
            [sys.executable, str(ROOT / "analyze_eci_vintage_knowledge_residual.py")],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        after = {
            path: hashlib.sha256(path.read_bytes()).hexdigest() for path in before
        }
        self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
