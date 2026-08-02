from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/019f6c42-2d53-7743-ab07-6293e2618dd7"
RESULT = OUT / "k3_efficiency_prior_2026-08-01.json"
DRAWS = OUT / "k3_efficiency_prior_cap_draws_2026-08-01.csv"
SCRIPT = ROOT / "analyze_k3_efficiency_prior.py"


class K3EfficiencyPriorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = json.loads(RESULT.read_text(encoding="utf-8"))
        with DRAWS.open(newline="", encoding="utf-8") as handle:
            cls.draws = list(csv.DictReader(handle))

    def test_exact_anchor_and_one_sided_decision(self) -> None:
        anchor = self.result["anchor"]
        self.assertEqual(anchor["model"], "Kimi K3")
        self.assertEqual(anchor["release_date"], "2026-07-16")
        self.assertEqual(anchor["total_parameters_t"], 2.780)
        self.assertEqual(anchor["activated_parameters_b"], 104.2)
        decision = self.result["decision"]
        self.assertTrue(decision["apply_center_preserving_upper_tail_projection"])
        self.assertFalse(decision["change_point_centers"])
        self.assertEqual(decision["incremental_point_center_weight"], 0)
        self.assertFalse(decision["change_crowd_weight"])
        self.assertEqual(decision["crowd_weight_for_fable_and_sol"], 0.5)
        self.assertEqual(decision["rejected_nonlinear_eci_weight"], 0)
        self.assertEqual(decision["default_projection_strength"], 0.8)
        self.assertFalse(
            decision["literal_constraint_enforced_when_reference_below_center"]
        )

    def test_only_log_linear_diminishing_return_laws_are_used(self) -> None:
        method = self.result["method"]
        self.assertEqual(method["nonlinear_forms_used"], [])
        self.assertIn("log10(parameters)", method["diminishing_returns_interpretation"])
        variants = self.result["linear_eci_slope_variants"]
        self.assertEqual(len(variants), 5)
        self.assertEqual(
            {row["id"] for row in variants},
            {
                "live_inverse_eci_ci__all_rows",
                "live_inverse_eci_ci__collapsed",
                "equal_family__all_rows",
                "equal_family__collapsed",
                "canonical_live_coefficients",
            },
        )
        self.assertTrue(all(row["blended_log10_slope"] > 0 for row in variants))
        aa_audit = json.loads(
            (
                OUT / "aa_expanded_parameter_audit_2026-07-18.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(
            method["aa_log10_parameter_slope"],
            aa_audit["full_fit"]["current_live_k3_anchored_coefficients"][
                "score_slope"
            ],
        )

    def test_canonical_equivalents_reproduce_equations(self) -> None:
        targets = {row["model"]: row for row in self.result["targets"]}
        expected = {
            "Claude Fable 5": (3.7007065133226313, 5.068457682392811),
            "GPT-5.6 Sol": (3.3450145821002732, 5.69288105582019),
            "Claude Opus 5": (4.035163064229386, 4.327984378272586),
        }
        for model, (aa, eci) in expected.items():
            self.assertTrue(
                math.isclose(
                    targets[model]["aa_same_efficiency_equivalent_t"],
                    aa,
                    rel_tol=0,
                    abs_tol=1e-12,
                )
            )
            self.assertTrue(
                math.isclose(
                    targets[model]["eci_same_efficiency_equivalent_t"]["canonical"],
                    eci,
                    rel_tol=0,
                    abs_tol=1e-12,
                )
            )

    def test_support_and_rejected_extrapolation_are_explicit(self) -> None:
        validation = self.result["validation"]
        self.assertEqual(validation["k3_parameter_labels_used_in_slope_fit"], 0)
        self.assertGreater(validation["k3_eci_points_beyond_calibration"], 4)
        self.assertTrue(
            all(
                value > validation["k3_eci_points_beyond_calibration"]
                for value in validation[
                    "target_eci_points_beyond_calibration"
                ].values()
            )
        )
        self.assertFalse(validation["existing_eci_tournament_changed_live_form"])
        self.assertFalse(
            validation["existing_eci_tournament_frontier_stability_gate_passed"]
        )
        rejected = validation["rejected_convex_extrapolation"]
        self.assertEqual(rejected["live_weight"], 0)
        self.assertEqual(rejected["claimed_fable_t"], [8.5, 10.6])
        self.assertEqual(rejected["claimed_sol_t"], [9.7, 12.2])

    def test_draw_ledger_and_joint_order_prior(self) -> None:
        self.assertEqual(len(self.draws), self.result["method"]["draws"])
        self.assertEqual(
            [int(row["draw"]) for row in self.draws], list(range(len(self.draws)))
        )
        for row in self.draws:
            self.assertLessEqual(
                float(row["sol_ordered_reference_t"]),
                float(row["fable_ordered_reference_t"]),
            )
            self.assertEqual(
                float(row["fable_raw_reference_t"]),
                float(row["fable_ordered_reference_t"]),
            )
            self.assertEqual(
                float(row["opus5_raw_reference_t"]),
                float(row["opus5_ordered_reference_t"]),
            )
        digest = hashlib.sha256(DRAWS.read_bytes()).hexdigest()
        self.assertEqual(digest, self.result["outputs"]["draw_ledger_sha256"])

    def test_declared_source_hashes_reconcile(self) -> None:
        sources = self.result["sources"]
        for key, relative in sources.items():
            if not key.endswith("_sha256"):
                continue
            path_key = key.removesuffix("_sha256")
            path = ROOT / sources[path_key]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), relative)

    def test_rebuild_is_byte_deterministic(self) -> None:
        before_result = RESULT.read_bytes()
        before_draws = DRAWS.read_bytes()
        subprocess.run([sys.executable, SCRIPT], cwd=ROOT, check=True, capture_output=True)
        self.assertEqual(RESULT.read_bytes(), before_result)
        self.assertEqual(DRAWS.read_bytes(), before_draws)


if __name__ == "__main__":
    unittest.main()
