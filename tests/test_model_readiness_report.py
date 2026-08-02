from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "019f6c42-2d53-7743-ab07-6293e2618dd7"
RESULT = OUT / "model_readiness_report_2026-07-31.json"
MARKDOWN = ROOT / "MODEL_READINESS.md"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ModelReadinessReportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = json.loads(RESULT.read_text(encoding="utf-8"))
        cls.forecast = json.loads(
            (ROOT / "site/public/data/forecast-model.json").read_text(encoding="utf-8")
        )
        cls.backtest = json.loads(
            (
                OUT / "frontier_parameter_chronological_backtest_2026-07-17.json"
            ).read_text(encoding="utf-8")
        )
        cls.uncertainty = json.loads(
            (
                OUT / "frontier_parameter_predictive_uncertainty_2026-07-18.json"
            ).read_text(encoding="utf-8")
        )
        cls.k3_efficiency = json.loads(
            (OUT / "k3_efficiency_prior_2026-08-01.json").read_text(
                encoding="utf-8"
            )
        )

    def test_headline_forecasts_and_intervals_are_exact(self) -> None:
        rows = {row["model"]: row for row in self.result["headline_forecasts"]}
        self.assertEqual(set(rows), {"Claude Fable 5", "GPT-5.6 Sol", "Claude Opus 5"})
        forecast_rows = {row["name"]: row for row in self.forecast["models"]}
        uncertainty_rows = {
            row["model"]: row for row in self.uncertainty["targets"]
        }
        for row in rows.values():
            source = forecast_rows[row["model"]]
            projection = uncertainty_rows[row["model"]]["k3_efficiency_projection"]
            self.assertAlmostEqual(row["final_center_t"], source["currentFinalT"])
            self.assertAlmostEqual(row["evidence_center_t"], source["currentEvidenceT"])
            self.assertFalse(row["parameter_count_disclosed"])
            self.assertGreater(row["empirical_50_high_t"], row["empirical_50_low_t"])
            self.assertGreater(row["empirical_80_factor"], row["empirical_50_factor"])
            self.assertAlmostEqual(
                row["k3_efficiency_reference_median_t"],
                projection["pooled_reference_quantiles_t"]["median"],
            )
            self.assertAlmostEqual(
                row["k3_efficiency_80_low_t"], row["empirical_80_low_t"]
            )
            self.assertLess(
                row["k3_efficiency_80_high_t"], row["empirical_80_high_t"]
            )
            self.assertFalse(row["k3_efficiency_point_center_changed"])
            self.assertFalse(row["k3_efficiency_lower_tail_changed"])

    def test_k3_efficiency_projection_is_center_preserving_and_does_not_reweight_centers(self) -> None:
        policy = self.result["k3_efficiency_projection"]
        self.assertEqual(policy["default_projection_strength"], 0.8)
        self.assertEqual(policy["point_center_weight"], 0)
        self.assertEqual(policy["crowd_weight_for_fable_and_sol"], 0.5)
        self.assertEqual(policy["rejected_nonlinear_eci_weight"], 0)
        self.assertIn("no greater than", policy["logical_direction"])
        self.assertIn(
            "log10(parameters)", policy["diminishing_returns_interpretation"]
        )
        self.assertFalse(policy["formal_coverage_guarantee"])
        self.assertFalse(policy["literal_conditioning"])

    def test_precision_and_zero_weight_decisions_are_honest(self) -> None:
        precision = self.result["heldout_precision"]
        source_frontier = self.backtest["frontier_like_metrics"][
            "Available-components ensemble"
        ]
        self.assertEqual(precision["frontier_lineage_holdout"], source_frontier)
        self.assertGreaterEqual(
            precision["frontier_lineage_holdout"]["n"],
            20,
        )
        self.assertEqual(precision["latest_per_developer"]["developers"], 11)
        self.assertGreater(
            precision["published_prequential_factors"]["80"]["multiplicative_factor"],
            5,
        )
        self.assertGreater(
            precision["published_prequential_factors"]["50"]["multiplicative_factor"],
            precision["frontier_lineage_holdout"]["median_multiplicative_error"],
        )
        self.assertFalse(precision["formal_coverage_guarantee"])
        self.assertFalse(
            precision["post_freeze_diagnostic_correction"]["freeze_rewritten"]
        )
        k3 = precision["k3_external_aa_reconciliation"]
        self.assertAlmostEqual(k3["actual_b"], 2780.0)
        self.assertGreater(k3["aa_all_kimi_held_out_predicted_b"], 2000.0)
        self.assertLess(k3["aa_all_kimi_held_out_error_x"], 1.3)
        self.assertAlmostEqual(
            k3["prior_incomplete_eci_compute_predicted_b"], 384.5334531305
        )
        self.assertGreater(k3["prior_incomplete_eci_compute_error_x"], 7.0)
        self.assertLess(k3["corrected_available_component_error_x"], 4.0)
        diagnostics = self.result["diagnostic_decisions"]
        self.assertEqual(diagnostics["vintage_knowledge"]["live_weight"], 0)
        self.assertFalse(diagnostics["vintage_knowledge"]["promotion_gates"]["all_pass"])
        self.assertEqual(diagnostics["active_parameters"]["live_weight"], 0)
        self.assertGreater(
            diagnostics["active_parameters"]["transport_candidate_median_error_x"],
            diagnostics["active_parameters"]["transport_baseline_median_error_x"],
        )
        self.assertFalse(diagnostics["direct_weight_optimization"]["update_live_weights"])
        architecture = diagnostics["eci_architecture_blend"]
        self.assertEqual(architecture["live_weight"], 0)
        self.assertLess(
            architecture["fixed_all"]["challenger"]["median_multiplicative_error"],
            architecture["fixed_all"]["baseline"]["median_multiplicative_error"],
        )
        self.assertFalse(architecture["promotion_gates"]["all_pass"])
        longcat = diagnostics["longcat_parameter_definition"]
        self.assertEqual(longcat["canonical_total_b"], 1600.0)
        self.assertAlmostEqual(longcat["hf_serialized_total_b"], 1775.560491136)
        self.assertLess(longcat["maximum_legacy_target_change_percent"], 1.0)
        self.assertTrue(longcat["ensemble_all_invariant"])
        self.assertFalse(longcat["change_live_forecast"])
        self.assertEqual(longcat["live_weight"], 0)
        shrinkage = diagnostics["active_parameter_shrinkage"]
        self.assertEqual(shrinkage["inventory"]["high_sparsity_rows"], 47)
        self.assertLess(
            shrinkage["fixed_all"]["candidate"]["median_multiplicative_error"],
            shrinkage["fixed_all"]["baseline"]["median_multiplicative_error"],
        )
        self.assertTrue(shrinkage["promotion_gates"]["empirical"]["fixed_all_ci90_wholly_favorable"])
        self.assertFalse(shrinkage["promotion_gates"]["all_gates_pass"])
        self.assertEqual(shrinkage["live_weight"], 0)
        common = self.result["external_validation"]["common_component_panel"]
        self.assertEqual(common["summary"]["n"], 4)
        self.assertGreater(common["summary"]["median_multiplicative_error"], 3)
        self.assertEqual(common["live_weight"], 0)
        commitment = self.result["prospective_commitment"]
        self.assertEqual(commitment["status"], "LOCKED_PRE_DISCLOSURE")
        self.assertEqual(commitment["post_outcome_refitting"], "FORBIDDEN")
        self.assertEqual(len(commitment["artifact_sha256"]), 64)
        self.assertTrue(commitment["frozen_point_centers_match_current"])
        self.assertTrue(commitment["current_bands_are_post_freeze_diagnostics"])
        self.assertFalse(commitment["freeze_rewritten"])
        self.assertTrue(commitment["privacy_redacted"])
        self.assertFalse(commitment["respondent_name_mapping_retained"])
        self.assertEqual(len(commitment["privacy_prior_artifact_sha256"]), 64)
        self.assertEqual(
            set(commitment["frozen_empirical_intervals"]),
            {"Claude Fable 5", "GPT-5.6 Sol", "Claude Opus 5"},
        )
        self.assertEqual(
            len(self.result["data_inventory"]["aa_primary_metadata_overrides"]), 2
        )
        self.assertEqual(
            self.result["data_inventory"]["aa_parameter_label_timing_records"], 6
        )
        self.assertEqual(
            self.result["data_inventory"]["aa_post_release_parameter_label_records"],
            5,
        )
        score_timing = self.result["data_inventory"]["aa_score_timing"]
        self.assertEqual(score_timing["live_rows"], 50)
        self.assertEqual(score_timing["live_verified"], 28)
        self.assertEqual(score_timing["live_fallback"], 22)
        self.assertEqual(score_timing["live_verified_after_release"], 16)
        self.assertEqual(score_timing["ensemble_rows_before"], 43)
        self.assertEqual(score_timing["ensemble_rows_after"], 44)
        self.assertFalse(score_timing["api_total_reconciles"])
        truth = self.result["data_inventory"]["open_model_parameter_truth"]
        self.assertEqual(truth["records"], 3)
        self.assertTrue(truth["raw_values_preserved"])
        self.assertFalse(truth["checkpoints_deduplicated"])
        crowd = self.result["crowd_robustness"]
        self.assertTrue(crowd["decision"]["center_is_single_forecaster_robust"])
        self.assertFalse(crowd["decision"]["crowd_is_independent_calibration_evidence"])
        self.assertGreater(
            crowd["cross_target_dependence"]["pearson_correlation_of_log_points"],
            0.7,
        )

    def test_sources_and_markdown_reconcile(self) -> None:
        for relative, expected in self.result["source_files"].items():
            path = ROOT / relative
            self.assertTrue(path.is_file(), relative)
            self.assertEqual(sha256(path), expected)
        text = MARKDOWN.read_text(encoding="utf-8")
        for row in self.result["headline_forecasts"]:
            self.assertIn(f"**{row['final_center_t']:.1f}T**", text)
        median = self.result["heldout_precision"]["frontier_lineage_holdout"][
            "median_multiplicative_error"
        ]
        self.assertIn(f"median error {median:.2f}×", text)
        self.assertIn("not project-prospective", text)
        self.assertIn("post-outcome refitting is **FORBIDDEN**", text)
        self.assertIn("prospective interval scoring must use the immutable bands", text)
        self.assertIn("privacy-redacted to stable anonymous respondent IDs", text)
        self.assertIn("after 2 explicit primary-source overrides", text)
        self.assertIn("AA parameter-label timing: 6 pinned records", text)
        self.assertIn("AA score-publication timing: 28 of 50", text)
        self.assertIn("ECI architecture-blend challenger", text)
        self.assertIn("LongCat parameter-definition audit", text)
        self.assertIn("Parameter-truth reconciliation: 3", text)
        self.assertIn("High-sparsity shrinkage challenger", text)
        self.assertIn("Crowd center robustness", text)
        self.assertIn("Kimi K3 audit correction", text)
        self.assertIn("K3 efficiency upper-tail stress test", text)
        self.assertIn("rejected nonlinear ECI extrapolation 0% weight", text)
        self.assertIn("center-preserving winsorized structural stress test", text)

    def test_offline_rebuild_is_byte_exact(self) -> None:
        before = {path: sha256(path) for path in (RESULT, MARKDOWN)}
        subprocess.run(
            [sys.executable, str(ROOT / "generate_model_readiness_report.py")],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual({path: sha256(path) for path in before}, before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
