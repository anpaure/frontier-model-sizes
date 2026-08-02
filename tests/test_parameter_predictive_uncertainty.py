from __future__ import annotations

import csv
import hashlib
import json
import math
import unittest
from pathlib import Path

import analyze_parameter_predictive_uncertainty as audit


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/019f6c42-2d53-7743-ab07-6293e2618dd7"
RESULT = OUT / "frontier_parameter_predictive_uncertainty_2026-07-18.json"
LEDGER = OUT / "frontier_parameter_predictive_uncertainty_calibration_2026-07-18.csv"
SITE_OUTPUT = ROOT / "site/public/data/predictive-uncertainty.json"
SITE_MODEL = ROOT / "site/public/data/forecast-model.json"


class ParameterPredictiveUncertaintyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = json.loads(RESULT.read_text(encoding="utf-8"))
        with LEDGER.open(newline="", encoding="utf-8-sig") as handle:
            cls.ledger = list(csv.DictReader(handle))
        cls.site = json.loads(SITE_MODEL.read_text(encoding="utf-8"))

    def test_site_copy_is_byte_identical(self) -> None:
        self.assertEqual(RESULT.read_bytes(), SITE_OUTPUT.read_bytes())

    def test_calibration_source_is_strict_and_developer_balanced(self) -> None:
        cohorts = self.result["cohorts"]
        self.assertEqual(cohorts["all_ensemble"]["rows"], 44)
        self.assertEqual(cohorts["all_ensemble"]["families"], 25)
        self.assertEqual(cohorts["frontier_like"]["rows"], 27)
        self.assertEqual(cohorts["frontier_like"]["families"], 16)
        self.assertEqual(cohorts["frontier_like"]["developers"], 11)
        self.assertEqual(
            cohorts["frontier_like"]["latest_per_family_rows"], 16
        )
        self.assertEqual(
            cohorts["frontier_like"]["latest_per_developer_rows"], 11
        )
        primary = cohorts["frontier_like"]
        self.assertEqual(
            set(primary["holdout_specs"]),
            {audit.LINEAGE_HOLDOUT, audit.DEVELOPER_HOLDOUT},
        )
        for level in ("50", "80", "90"):
            candidates = primary["intervals"][level][
                "candidate_multiplicative_factors"
            ]
            self.assertAlmostEqual(
                primary["intervals"][level]["multiplicative_factor"],
                max(value for value in candidates.values() if value is not None),
                places=12,
            )
        self.assertAlmostEqual(primary["intervals"]["50"]["multiplicative_factor"], 3.0156279830601624)
        self.assertAlmostEqual(primary["intervals"]["80"]["multiplicative_factor"], 5.276399681855711)
        self.assertAlmostEqual(primary["intervals"]["90"]["multiplicative_factor"], 6.1864280178083515)
        frontier_ledger = [
            row for row in self.ledger if row["cohort"] == "frontier_like"
        ]
        self.assertEqual(len(frontier_ledger), 54)
        for spec in (audit.LINEAGE_HOLDOUT, audit.DEVELOPER_HOLDOUT):
            self.assertEqual(
                len([row for row in frontier_ledger if row["holdout_spec"] == spec]),
                27,
            )

    def test_latest_date_ties_are_resolved_by_worst_error_not_model_name(self) -> None:
        rows = [
            {
                "developer": "Lab",
                "release_date": "2026-01-01",
                "model": "A-low-error",
                "log10_error": math.log10(1.2),
            },
            {
                "developer": "Lab",
                "release_date": "2026-01-01",
                "model": "Z-high-error",
                "log10_error": math.log10(4.0),
            },
            {
                "developer": "Lab",
                "release_date": "2025-12-01",
                "model": "Older-even-higher",
                "log10_error": math.log10(9.0),
            },
        ]
        selected = audit.latest_per_group(rows, "developer")
        self.assertEqual([row["model"] for row in selected], ["Z-high-error"])

    def test_architecture_matched_subset_is_not_used_to_narrow_intervals(self) -> None:
        diagnostic = self.result["cohorts"]["frontier_moe_reasoning"]
        self.assertEqual(diagnostic["rows"], 11)
        self.assertEqual(diagnostic["families"], 5)
        self.assertEqual(diagnostic["developers"], 5)
        self.assertIn("insufficient", diagnostic["target_specificity_status"])
        self.assertIs(
            self.result["decision"]["use_frontier_moe_reasoning_cohort"], False
        )
        self.assertEqual(self.result["method"]["primary_cohort"], "frontier_like")
        self.assertFalse(diagnostic["intervals"]["90"]["supported"])
        self.assertIsNone(diagnostic["intervals"]["90"]["multiplicative_factor"])
        self.assertIsNone(diagnostic["intervals"]["90"]["order_statistic_rank"])

    def test_unsupported_order_statistic_never_clips_to_finite_maximum(self) -> None:
        rows = [
            {
                "developer": f"lab-{index}",
                "release_date": "2026-01-01",
                "model": f"model-{index}",
                "log10_error": math.log10(1.1 + index),
            }
            for index in range(5)
        ]
        result = audit.order_statistic_factor(rows, 0.90)
        self.assertEqual(result["raw_order_statistic_rank"], 6)
        self.assertFalse(result["supported"])
        self.assertIsNone(result["order_statistic_rank"])
        self.assertIsNone(result["multiplicative_factor"])

    def test_chronological_coverage_is_honestly_recorded(self) -> None:
        blocks = self.result["cohorts"]["frontier_like"]["holdout_specs"]
        for spec, block in blocks.items():
            coverage = block["chronological_coverage"]
            for level in ("50", "80", "90"):
                summary = coverage[level]
                self.assertEqual(summary["candidate_tests"], 20)
                self.assertEqual(
                    summary["candidate_tests"],
                    summary["eligible_tests"] + summary["unsupported_tests"],
                )
                supported = [row for row in summary["tests"] if row["supported"]]
                self.assertEqual(len(supported), summary["eligible_tests"])
                covered = sum(bool(row["covered"]) for row in supported)
                self.assertEqual(
                    summary["raw_checkpoint_coverage"]["covered"], covered
                )
                self.assertTrue(
                    math.isclose(
                        summary["raw_checkpoint_coverage"]["rate"],
                        covered / len(supported),
                    )
                )
                self.assertLessEqual(
                    summary["latest_developer_coverage"]["developers"],
                    summary["test_developers"],
                )
            self.assertEqual(coverage["90"]["eligible_tests"], 7)
            self.assertEqual(coverage["90"]["unsupported_tests"], 13)
        strict = blocks[audit.DEVELOPER_HOLDOUT]["chronological_coverage"]
        self.assertAlmostEqual(
            strict["50"]["raw_checkpoint_coverage"]["rate"], 9 / 20
        )
        self.assertAlmostEqual(
            strict["50"]["developer_balanced_coverage"]["rate"],
            0.4425925925925926,
        )
        self.assertAlmostEqual(
            strict["50"]["latest_developer_coverage"]["rate"], 3 / 9
        )
        self.assertIn(
            "raw/developer-balanced/latest-developer",
            self.result["decision"]["sequential_coverage_warning"],
        )

    def test_target_intervals_recompute_and_do_not_move_centers(self) -> None:
        targets = {row["model_id"]: row for row in self.result["targets"]}
        self.assertEqual(
            set(targets), {"claude-fable-5", "gpt-56-sol", "claude-opus-5"}
        )
        site_centers = {
            row["id"]: row["currentEvidenceT"] for row in self.site["models"]
        }
        for model_id, target in targets.items():
            self.assertTrue(math.isclose(target["center_t"], site_centers[model_id]))
        expected_developers = {
            "claude-fable-5": 11,
            "gpt-56-sol": 10,
            "claude-opus-5": 11,
        }
        for model_id, target in targets.items():
            self.assertEqual(
                target["calibration_families"], expected_developers[model_id]
            )
            self.assertEqual(
                target["calibration_developers"], expected_developers[model_id]
            )
            previous_factor = 1.0
            for level in ("50", "80", "90"):
                interval = target["intervals"][level]
                factor = interval["multiplicative_factor"]
                self.assertGreaterEqual(factor, previous_factor)
                self.assertTrue(
                    math.isclose(interval["low_t"], target["center_t"] / factor)
                )
                self.assertTrue(
                    math.isclose(interval["high_t"], target["center_t"] * factor)
                )
                previous_factor = factor
                candidates = interval["candidate_multiplicative_factors"]
                self.assertAlmostEqual(
                    factor,
                    max(value for value in candidates.values() if value is not None),
                    places=12,
                )
            self.assertTrue(
                math.isclose(
                    target["intervals"]["80"]["multiplicative_factor"],
                    5.276399681855711,
                )
            )
            self.assertGreaterEqual(
                target["intervals"]["80"]["multiplicative_factor"],
                target["lineage_intervals"]["80"]["multiplicative_factor"],
            )
        self.assertIs(self.result["decision"]["change_central_forecasts"], False)
        self.assertFalse(self.result["decision"]["formal_coverage_guarantee"])
        self.assertFalse(self.result["method"]["formal_conformal_coverage_claim"])
        self.assertFalse(
            self.result["post_freeze_diagnostic_correction"]["freeze_rewritten"]
        )

    def test_crowd_never_narrows_or_recenters_calibrated_error_bands(self) -> None:
        self.assertIn("do not narrow", self.result["method"]["crowd_policy"])
        site_by_id = {row["id"]: row for row in self.site["models"]}
        for target in self.result["targets"]:
            model = site_by_id[target["model_id"]]
            self.assertAlmostEqual(target["center_t"], model["currentEvidenceT"])
            self.assertAlmostEqual(
                target["displayed_final_center_t"], model["currentFinalT"]
            )
            factor = target["intervals"]["80"]["multiplicative_factor"]
            self.assertAlmostEqual(
                target["intervals"]["80"]["low_t"],
                model["currentEvidenceT"] / factor,
            )
            self.assertAlmostEqual(
                target["intervals"]["80"]["high_t"],
                model["currentEvidenceT"] * factor,
            )

    def test_k3_efficiency_projection_changes_only_upper_tail(self) -> None:
        self.assertTrue(
            self.result["decision"]["k3_efficiency_projection_live_for_upper_tail"]
        )
        self.assertFalse(
            self.result["decision"]["k3_efficiency_projection_changes_centers"]
        )
        self.assertEqual(
            self.result["decision"][
                "k3_efficiency_default_projection_strength"
            ],
            0.8,
        )
        for target in self.result["targets"]:
            projection = target["k3_efficiency_projection"]
            self.assertEqual(projection["default_projection_percent"], 80)
            self.assertFalse(projection["lower_tail_changed"])
            self.assertFalse(projection["point_center_changed"])
            self.assertFalse(projection["formal_coverage_guarantee"])
            self.assertFalse(projection["literal_conditioning"])
            self.assertFalse(
                projection["literal_ceiling_enforced_when_reference_below_center"]
            )
            self.assertGreaterEqual(
                projection["binding_probability_against_raw_upper_draws"], 0
            )
            self.assertLessEqual(
                projection["binding_probability_against_raw_upper_draws"], 1
            )
            self.assertGreaterEqual(projection["center_override_probability"], 0)
            self.assertLessEqual(projection["center_override_probability"], 1)
            self.assertEqual(
                projection["center_override_probability"],
                projection["literal_reference_violation_probability_at_100pct"],
            )
            previous_by_level = {
                level: float(target["intervals"][level]["high_t"])
                for level in ("50", "80", "90")
            }
            for percent in range(0, 101, 5):
                grid = projection["strength_grid"][str(percent)]
                for level in ("50", "80", "90"):
                    raw = target["intervals"][level]
                    projected = grid[level]
                    self.assertEqual(projected["low_t"], raw["low_t"])
                    self.assertEqual(projected["raw_low_t"], raw["low_t"])
                    self.assertEqual(projected["raw_high_t"], raw["high_t"])
                    self.assertGreaterEqual(
                        projected["high_t"], target["center_t"]
                    )
                    self.assertLessEqual(projected["high_t"], raw["high_t"])
                    self.assertLessEqual(
                        projected["high_t"], previous_by_level[level] + 1e-12
                    )
                    previous_by_level[level] = projected["high_t"]
            self.assertEqual(
                projection["projected_intervals"], projection["strength_grid"]["80"]
            )
            self.assertLess(
                projection["projected_intervals"]["80"]["high_t"],
                target["intervals"]["80"]["high_t"],
            )

        by_id = {row["model_id"]: row for row in self.result["targets"]}
        self.assertTrue(
            by_id["gpt-56-sol"]["k3_efficiency_projection"][
                "order_projection_applied"
            ]
        )
        self.assertFalse(
            by_id["claude-fable-5"]["k3_efficiency_projection"][
                "order_projection_applied"
            ]
        )
        self.assertGreater(
            by_id["claude-fable-5"]["k3_efficiency_projection"][
                "center_override_probability"
            ],
            0.8,
        )

    def test_opus_5_is_an_unlocked_distinct_target_with_exact_inputs(self) -> None:
        opus = next(row for row in self.site["models"] if row["id"] == "claude-opus-5")
        self.assertEqual(opus["name"], "Claude Opus 5")
        self.assertEqual(opus["releaseDate"], "2026-07-24")
        self.assertFalse(opus["lockedAnchor"])
        self.assertIsNone(opus["disclosedT"])
        self.assertIsNone(opus["factors"]["crowd"])
        self.assertIsNone(opus["factors"]["ikp"])
        self.assertTrue(math.isclose(opus["aaScore"], 60.6918740157091))
        self.assertTrue(math.isclose(opus["eciScore"], 159.3778667882398))
        self.assertEqual(
            opus["eciCi90"], [157.24933114170264, 162.20640578425878]
        )
        self.assertIsNotNone(opus["aaConfiguration"])
        self.assertEqual(opus["aaFallbackModel"], "Claude Opus 4.8")

    def test_source_hashes_reconcile(self) -> None:
        for relative, digest in self.result["source_files"].items():
            path = ROOT / relative
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)


if __name__ == "__main__":
    unittest.main()
