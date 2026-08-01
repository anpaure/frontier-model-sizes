#!/usr/bin/env python3
"""Integrity tests for developer-holdout and Epoch-vintage sensitivity."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import analyze_parameter_vintage_sensitivity as audit
from run_parameter_backtest import (
    CURRENT_WEIGHTS,
    _current_ensemble,
    _load_panels,
    _load_parameter_truth_registry,
)


ROOT = Path(__file__).resolve().parents[1]
RESULT = (
    ROOT
    / "outputs/019f6c42-2d53-7743-ab07-6293e2618dd7"
    / "parameter_developer_vintage_sensitivity_2026-07-31.json"
)


class ParameterVintageSensitivityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = json.loads(RESULT.read_text(encoding="utf-8"))

    def test_diagnostic_cannot_change_live_weights_or_centers(self) -> None:
        self.assertFalse(self.data["decision"]["change_forecast_weights"])
        self.assertFalse(self.data["decision"]["change_central_forecasts"])
        self.assertEqual(self.data["method"]["current_weights"], CURRENT_WEIGHTS)
        self.assertEqual(self.data["method"]["vintage_scope"], "ECI only; no full-ensemble vintage claim")

    def test_developer_holdout_recomputes_strict_predictions(self) -> None:
        panels, _ = _load_panels()
        lookup = audit.developer_lookup()
        converted = audit.with_developers(panels, lookup)
        selected = audit.selected_developer_predictions(converted)
        ensemble = _current_ensemble(
            selected,
            equal_weight=False,
            parameter_registry=_load_parameter_truth_registry(),
        )
        self.assertEqual(len(ensemble), 44)
        self.assertTrue(
            all(
                row["train_max_date"]
                < (row.get("prediction_information_date") or row["release_date"])
                for row in ensemble
            )
        )
        self.assertTrue(all(row["test_family_excluded"] for row in ensemble))
        frontier = [row for row in ensemble if row["frontier_signal_rank"] >= 0.90]
        self.assertEqual(len(frontier), 27)
        paired = self.data["developer_holdout_current_snapshot"]["paired_frontier"]
        self.assertEqual(paired["paired_models"], 27)
        self.assertAlmostEqual(
            paired["challenger"]["median_multiplicative_error"],
            self.data["developer_holdout_current_snapshot"]["developer_frontier"][
                "median_multiplicative_error"
            ],
            places=12,
        )

    def test_effective_independent_inventory_is_developer_balanced(self) -> None:
        current = self.data["developer_holdout_current_snapshot"]
        self.assertEqual(current["latest_per_developer"]["developers"], 11)
        self.assertEqual(current["lineage_latest_per_developer"]["developers"], 11)
        self.assertEqual(
            current["latest_per_developer"]["conformal_factors"]["80"]["rank"],
            10,
        )
        self.assertAlmostEqual(
            current["latest_per_developer"]["conformal_factors"]["80"][
                "multiplicative_factor"
            ],
            4.655997411471875,
            places=12,
        )
        self.assertAlmostEqual(
            current["lineage_latest_per_developer"]["conformal_factors"]["80"][
                "multiplicative_factor"
            ],
            5.276399681855711,
            places=12,
        )
        # K3's exact-score AA holdout must be present.  The old ECI+compute-only
        # row produced a spurious 7.23x tail and is no longer admissible.
        self.assertLess(
            current["latest_per_developer"]["metrics"]["p90_multiplicative_error"],
            5.0,
        )
        # Llama 4 Maverick and Scout share a release date. The conservative
        # cluster keeps the larger-error Maverick rather than selecting Scout
        # lexicographically.
        self.assertIn("Llama 4 Maverick", current["latest_per_developer"]["latest_models"])
        self.assertNotIn("Llama 4 Scout", current["latest_per_developer"]["latest_models"])
        self.assertIn("largest absolute residual", current["latest_per_developer"]["cluster_policy"])

    def test_vintage_predictions_are_first_observed_and_leakage_controlled(self) -> None:
        block = self.data["eci_vintage"]
        self.assertEqual(block["inventory"]["historical_snapshots"], 15)
        self.assertEqual(block["inventory"]["historical_score_rows"], 2308)
        self.assertEqual(block["inventory"]["current_parameter_checkpoints"], 89)
        self.assertEqual(block["inventory"]["first_observed_targets"], 25)
        self.assertEqual(len(block["predictions"]), 25)
        for row in block["predictions"]:
            self.assertGreater(row["first_snapshot_timestamp"], "20251113094011")
            self.assertGreaterEqual(row["availability_lag_days"], 0)
            self.assertGreaterEqual(row["lineage_train_n"], 20)
            self.assertGreaterEqual(row["developer_train_n"], 20)
            self.assertGreaterEqual(row["lineage_train_groups"], 6)
            self.assertGreaterEqual(row["developer_train_groups"], 6)

    def test_only_two_same_developer_targets_are_interval_prospective(self) -> None:
        prospective = self.data["eci_vintage"]["cohorts"]["interval_prospective"]
        self.assertEqual(prospective["rows"], 2)
        self.assertEqual(prospective["developers"], 1)
        self.assertEqual(
            set(prospective["models"]), {"Kimi K2.5", "Kimi K2.7 Code"}
        )

    def test_paired_vintage_penalty_and_small_ensemble_overlap_are_explicit(self) -> None:
        paired = self.data["eci_vintage"]["paired_current_vs_vintage"]
        self.assertEqual(paired["all_first_observed"]["n"], 25)
        self.assertEqual(paired["frontier_like"]["n"], 22)
        self.assertGreater(
            paired["frontier_like"]["first_observed_vintage"][
                "median_multiplicative_error"
            ],
            paired["frontier_like"]["current_snapshot"][
                "median_multiplicative_error"
            ],
        )
        swap = self.data["eci_vintage"]["frontier_ensemble_eci_swap"]
        self.assertEqual(swap["n"], 9)
        self.assertEqual(len(swap["models"]), 9)

    def test_source_hashes_reconcile(self) -> None:
        for relative, expected in self.data["sources"].items():
            path = ROOT / relative
            self.assertTrue(path.exists(), relative)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
