from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import unittest
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/019f6c42-2d53-7743-ab07-6293e2618dd7"
METADATA = ROOT / "sources/ikp_source_metadata_2026-07-18.json"
CALIBRATION = ROOT / "sources/ikp_upstream_calibration_2026-07-18.json"
RESULT = OUT / "ikp_parameter_signal_audit_2026-07-18.json"
PREDICTIONS = OUT / "ikp_parameter_chronological_predictions_2026-07-18.csv"
OVERLAP = OUT / "ikp_parameter_incremental_overlap_2026-07-18.csv"
SITE_COPY = ROOT / "site/public/data/ikp-parameter-signal.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def metrics_from_rows(rows: list[dict[str, str]], prediction: str) -> dict[str, float]:
    errors = [
        math.log10(float(row[prediction]) / float(row["actual_b"])) for row in rows
    ]
    absolute = [abs(value) for value in errors]
    ordered = sorted(absolute)
    middle = len(ordered) // 2
    median = (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2
    )
    return {
        "n": len(rows),
        "families": len({row["family"] for row in rows}),
        "vendors": len({row.get("vendor", row["family"]) for row in rows}),
        "mean_absolute_log10_error": sum(absolute) / len(absolute),
        "median_multiplicative_error": 10**median,
        "rmse_log10": math.sqrt(sum(value * value for value in errors) / len(errors)),
        "p80_multiplicative_error": 10 ** quantile(absolute, 0.8),
        "within_2x": sum(value <= math.log10(2) for value in absolute) / len(absolute),
        "signed_bias_factor": 10 ** (sum(errors) / len(errors)),
    }


class IkpParameterSignalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metadata = json.loads(METADATA.read_text(encoding="utf-8"))
        cls.calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
        cls.result = json.loads(RESULT.read_text(encoding="utf-8"))
        with PREDICTIONS.open(newline="", encoding="utf-8") as handle:
            cls.predictions = list(csv.DictReader(handle))
        cls.primary_predictions = {
            row["base_key"]: row
            for row in cls.predictions
            if row["collapse_policy"] == "mean" and row["form"] == "forward_inverse"
        }
        with OVERLAP.open(newline="", encoding="utf-8") as handle:
            cls.overlap = list(csv.DictReader(handle))

    def assert_metrics_match_rows(self, observed, rows, prediction):
        expected = metrics_from_rows(rows, prediction)
        self.assertEqual(observed["n"], expected["n"])
        self.assertEqual(observed["families"], expected["families"])
        self.assertEqual(observed["vendors"], expected["vendors"])
        for field in (
            "mean_absolute_log10_error",
            "median_multiplicative_error",
            "rmse_log10",
            "p80_multiplicative_error",
            "within_2x",
            "signed_bias_factor",
        ):
            self.assertTrue(
                math.isclose(observed[field], expected[field], rel_tol=1e-12, abs_tol=1e-12),
                f"{field}: {observed[field]} != {expected[field]}",
            )

    def test_immutable_sources_and_hashes_reconcile(self):
        self.assertEqual(
            self.metadata["upstream"]["commit"],
            "e5c4231985048bb2db5dc2611b6eb659b891791d",
        )
        self.assertEqual(
            self.metadata["replication"]["commit"],
            "c44e4dc82132e268dc9a2c86350863f59282fddb",
        )
        self.assertEqual(len(self.metadata["files"]), 21)
        for record in self.metadata["files"]:
            path = ROOT / record["local_path"]
            self.assertEqual(sha256(path), record["sha256"])
            self.assertEqual(path.stat().st_size, record["bytes"])
        for relative, expected in self.result["source_files"].items():
            self.assertEqual(sha256(ROOT / relative), expected)

    def test_source_fit_is_exactly_reproduced_without_duplicate_configs(self):
        points = self.calibration["calibration_points"]
        self.assertEqual(len(points), 93)
        self.assertEqual(len({row["model"] for row in points}), 93)
        self.assertEqual(self.result["source_inventory"]["calibration_configurations"], 93)
        self.assertEqual(self.result["source_inventory"]["calibration_weight_bases"], 87)
        self.assertEqual(self.result["source_inventory"]["serving_variants_collapsed"], 6)
        self.assertEqual(self.result["published_reproduction"]["summary_mismatches"], [])
        self.assertTrue(
            math.isclose(
                self.result["published_reproduction"]["r_squared"],
                self.calibration["fit"]["r_squared"],
                rel_tol=0,
                abs_tol=1e-14,
            )
        )

    def test_every_prediction_is_strictly_earlier_and_vendor_held_out(self):
        self.assertEqual(len(self.predictions), 960)
        keys = [
            (row["collapse_policy"], row["form"], row["base_key"])
            for row in self.predictions
        ]
        self.assertEqual(len(keys), len(set(keys)))
        for row in self.predictions:
            self.assertLess(row["train_max_date"], row["release_date"])
            self.assertEqual(row["ikp_test_vendor_excluded"] if "ikp_test_vendor_excluded" in row else row["test_vendor_excluded"], "True")
            self.assertGreaterEqual(int(row["train_rows"]), 20)
            self.assertGreaterEqual(int(row["train_vendors"]), 6)

    def test_incremental_overlap_has_exact_identity_and_supported_gain(self):
        incremental = self.result["incremental_overlap"]
        self.assertEqual(len(self.overlap), incremental["models"])
        families = {row["family"] for row in self.overlap}
        self.assertEqual(len(families), incremental["families"])
        self.assertEqual(
            len({row["normalized_model"] for row in self.overlap}), len(self.overlap)
        )
        self.assertEqual(
            len({row["ikp_base_key"] for row in self.overlap}), len(self.overlap)
        )

        self.assert_metrics_match_rows(
            incremental["existing"], self.overlap, "existing_predicted_b"
        )
        self.assert_metrics_match_rows(
            incremental["ikp"], self.overlap, "ikp_predicted_b"
        )
        self.assert_metrics_match_rows(
            incremental["blend_10pct"], self.overlap, "blended_predicted_b"
        )

        family_deltas = defaultdict(list)
        for row in self.overlap:
            weight = float(row["blend_weight"])
            expected_blend = 10 ** (
                (1 - weight) * math.log10(float(row["existing_predicted_b"]))
                + weight * math.log10(float(row["ikp_predicted_b"]))
            )
            self.assertTrue(
                math.isclose(
                    float(row["blended_predicted_b"]),
                    expected_blend,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            )
            expected_delta = (
                abs(math.log10(expected_blend / float(row["actual_b"])))
                - abs(
                    math.log10(
                        float(row["existing_predicted_b"]) / float(row["actual_b"])
                    )
                )
            )
            self.assertTrue(
                math.isclose(
                    float(row["blend_minus_existing_abs_log10_error"]),
                    expected_delta,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            )
            family_deltas[row["family"]].append(expected_delta)
            ikp_actual = float(self.primary_predictions[row["ikp_base_key"]]["actual_b"])
            self.assertLessEqual(
                abs(math.log10(float(row["actual_b"]) / ikp_actual)),
                math.log10(1.06),
            )
        expected_family_means = {
            family: sum(values) / len(values)
            for family, values in sorted(family_deltas.items())
        }
        self.assertEqual(set(incremental["family_mean_deltas"]), families)
        for family, expected in expected_family_means.items():
            self.assertTrue(
                math.isclose(
                    incremental["family_mean_deltas"][family],
                    expected,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            )
        self.assertEqual(
            incremental["families_improved"],
            sum(value < 0 for value in expected_family_means.values()),
        )
        bootstrap = incremental["family_bootstrap"]
        self.assertEqual(bootstrap["families"], len(families))
        self.assertTrue(
            math.isclose(
                bootstrap["observed_delta"],
                sum(expected_family_means.values()) / len(expected_family_means),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        )
        self.assertLessEqual(bootstrap["ci_90"][0], bootstrap["ci_90"][1])
        self.assertGreaterEqual(bootstrap["probability_blend_better"], 0)
        self.assertLessEqual(bootstrap["probability_blend_better"], 1)
        correlation = incremental["signed_error_correlation_existing_vs_ikp"]
        self.assertTrue(math.isfinite(correlation))
        self.assertGreaterEqual(correlation, -1)
        self.assertLessEqual(correlation, 1)

    def test_later_chronological_subset_and_nested_family_holdout_are_clean(self):
        incremental = self.result["incremental_overlap"]
        chronological = incremental["chronological_fixed_weight_subset"]
        protocol_counts = re.search(
            r"at least (\d+) earlier overlap rows from (\d+) other families",
            chronological["protocol"],
        )
        self.assertIsNotNone(protocol_counts)
        minimum_rows, minimum_families = map(int, protocol_counts.groups())
        selected = []
        for row in self.overlap:
            train = [
                candidate
                for candidate in self.overlap
                if candidate["release_date"] < row["release_date"]
                and candidate["family"] != row["family"]
            ]
            if len(train) >= minimum_rows and len({item["family"] for item in train}) >= minimum_families:
                selected.append(row)
        self.assertEqual(chronological["models"], len(selected))
        self.assertEqual(
            chronological["families"], len({row["family"] for row in selected})
        )
        self.assert_metrics_match_rows(
            chronological["existing"], selected, "existing_predicted_b"
        )
        self.assert_metrics_match_rows(
            chronological["blend"], selected, "blended_predicted_b"
        )
        chronological_bootstrap = chronological["family_bootstrap"]
        self.assertEqual(chronological_bootstrap["families"], chronological["families"])
        self.assertLessEqual(
            chronological_bootstrap["ci_90"][0], chronological_bootstrap["ci_90"][1]
        )
        self.assertGreaterEqual(chronological_bootstrap["probability_blend_better"], 0)
        self.assertLessEqual(chronological_bootstrap["probability_blend_better"], 1)
        nested = incremental["nested_leave_one_family_out_weight_learning"]
        self.assertTrue(nested["all_test_families_excluded"])
        self.assertEqual(nested["metrics"]["n"], incremental["models"])
        self.assertEqual(nested["metrics"]["families"], incremental["families"])
        self.assertEqual(
            sum(nested["selected_weight_counts"].values()), incremental["models"]
        )

    def test_fable_target_is_release_and_vendor_held_out(self):
        target = self.result["target_signal"]["fable"]
        strict = target["strict_open_only_release_and_vendor_holdout"]
        self.assertEqual(set(strict), {"mean", "nonthinking", "max"})
        for policy in strict.values():
            self.assertTrue(policy["test_vendor_excluded"])
            self.assertLess(policy["train_max_date"], "2026-06-09")
            self.assertEqual(policy["train_rows"], 86)
        primary = strict["mean"]["estimates"]["forward_inverse"]["estimated_b"]
        published = target["published_lambda0_estimate_b"]
        self.assertLess(max(primary / published, published / primary), 1.05)
        self.assertLess(target["strict_open_only_model_form_min_b"], primary)
        self.assertGreater(target["strict_open_only_model_form_max_b"], primary)
        self.assertFalse(self.result["target_signal"]["sol"]["observed"])

    def test_promotion_matches_declared_gates_and_site_copy_is_identical(self):
        incremental = self.result["incremental_overlap"]
        decision = self.result["decision"]
        gates = decision["evidence_gates"]
        full_bootstrap = incremental["family_bootstrap"]
        chronological = incremental["chronological_fixed_weight_subset"]
        chronological_bootstrap = chronological["family_bootstrap"]
        nested = incremental["nested_leave_one_family_out_weight_learning"]
        expected_promotion = all(
            (
                incremental["models"] >= gates["minimum_overlap_models"],
                incremental["families"] >= gates["minimum_overlap_families"],
                full_bootstrap["ci_90"][1]
                < gates["maximum_bootstrap_ci90_upper"],
                full_bootstrap["probability_blend_better"]
                >= gates["minimum_bootstrap_probability_better"],
                chronological["models"]
                >= gates["minimum_chronological_subset_models"],
                chronological["families"]
                >= gates["minimum_chronological_subset_families"],
                chronological_bootstrap["ci_90"][1]
                < gates["maximum_bootstrap_ci90_upper"],
                chronological_bootstrap["probability_blend_better"]
                >= gates["minimum_bootstrap_probability_better"],
                nested["all_test_families_excluded"],
            )
        )
        self.assertEqual(decision["promote_incremental_ikp_weight"], expected_promotion)
        tested_weights = {float(row["blend_weight"]) for row in self.overlap}
        self.assertEqual(len(tested_weights), 1)
        tested_weight = next(iter(tested_weights))
        expected_weight = tested_weight if expected_promotion else 0.0
        self.assertEqual(decision["incremental_evidence_weight"], expected_weight)
        self.assertEqual(
            decision["incremental_final_weight_when_crowd_is_50pct"],
            expected_weight * 0.5,
        )
        self.assertEqual(decision["change_fable_center"], expected_promotion)
        self.assertFalse(decision["change_sol_center"])
        self.assertEqual(RESULT.read_bytes(), SITE_COPY.read_bytes())

    def test_same_base_controls_are_not_mislabeled_as_disclosures(self):
        controls = {
            row["label"]: row for row in self.result["serving_and_same_base_controls"]
        }
        self.assertEqual(
            controls["GPT-5.5 vs GPT-5.5 Pro"]["identity_status"],
            "first-party same underlying model",
        )
        self.assertEqual(
            controls["GPT-5 vs GPT-5.5"]["identity_status"],
            "user-supplied same-base assumption",
        )
        self.assertEqual(
            controls["Opus 4.7 vs Opus 4.8"]["identity_status"],
            "user-supplied same-base assumption",
        )


if __name__ == "__main__":
    unittest.main()
