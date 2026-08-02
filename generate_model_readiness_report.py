#!/usr/bin/env python3
"""Generate the current model/readiness summary from audited artifacts.

This report replaces hand-copied headline metrics with one deterministic view
of the current forecasts, held-out accuracy, empirical uncertainty, validation
extensions, and promotion decisions.  It is descriptive only and never feeds
the forecasting model.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "019f6c42-2d53-7743-ab07-6293e2618dd7"
DATE = "2026-07-31"  # durable compatibility suffix
GENERATED_ON = "2026-08-01"

INPUTS = {
    "forecast": ROOT / "site/public/data/forecast-model.json",
    "backtest": OUT / "frontier_parameter_chronological_backtest_2026-07-17.json",
    "vintage": OUT / "parameter_developer_vintage_sensitivity_2026-07-31.json",
    "uncertainty": OUT / "frontier_parameter_predictive_uncertainty_2026-07-18.json",
    "k3_efficiency": OUT / "k3_efficiency_prior_2026-08-01.json",
    "eci_extension": OUT / "eci_historical_validation_extension_2026-07-31.json",
    "common_components": OUT / "eci_historical_common_component_audit_2026-07-31.json",
    "knowledge": OUT / "eci_vintage_knowledge_residual_audit_2026-07-31.json",
    "active": OUT / "active_parameter_transport_audit_2026-07-18.json",
    "active_shrinkage": OUT / "active_parameter_shrinkage_challenger_2026-07-31.json",
    "eci_architecture_blend": OUT / "eci_architecture_blend_challenger_2026-07-31.json",
    "longcat_definition": OUT / "longcat_parameter_definition_sensitivity_2026-07-31.json",
    "optimizer": OUT / "factor_weight_optimization_2026-07-18.json",
    "aa_inference": OUT / "aa_inference_budget_audit_2026-07-18.json",
    "aa_label_availability": ROOT / "sources/aa_parameter_label_availability_2026-07-31.json",
    "aa_score_availability": ROOT / "sources/aa_score_availability_2026-07-31.json",
    "aa_score_timing_audit": OUT / "aa_score_availability_timing_audit_2026-07-31.json",
    "open_model_parameter_truth": ROOT / "sources/open_model_parameter_truth_reconciliation_2026-07-31.json",
    "crowd": ROOT / "sources/human_parameter_forecasts_2026-07-17.csv",
    "crowd_robustness": OUT / "crowd_robustness_audit_2026-07-31.json",
    "regression": ROOT / "regression_results.json",
    "prospective_freeze": ROOT / "forecast_freezes/2026-07-31-frontier-parameters-v1/forecast_freeze.json",
    "prospective_freeze_digest": ROOT / "forecast_freezes/2026-07-31-frontier-parameters-v1/forecast_freeze.sha256",
}

RESULT = OUT / f"model_readiness_report_{DATE}.json"
MARKDOWN = ROOT / "MODEL_READINESS.md"
TARGETS = ("Claude Fable 5", "GPT-5.6 Sol", "Claude Opus 5")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def load_inputs() -> dict[str, Any]:
    missing = [relative(path) for path in INPUTS.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing readiness inputs: {missing}")
    return {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in INPUTS.items()
        if path.suffix == ".json"
    }


def model_by_name(forecast: dict[str, Any], name: str) -> dict[str, Any]:
    rows = [row for row in forecast["models"] if row["name"] == name]
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one forecast row for {name}; found {len(rows)}")
    return rows[0]


def target_interval(uncertainty: dict[str, Any], name: str) -> dict[str, Any]:
    rows = [row for row in uncertainty["targets"] if row["model"] == name]
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one uncertainty row for {name}; found {len(rows)}")
    return rows[0]


def build_result(data: dict[str, Any]) -> dict[str, Any]:
    forecast = data["forecast"]
    backtest = data["backtest"]
    vintage = data["vintage"]
    uncertainty = data["uncertainty"]
    k3_efficiency = data["k3_efficiency"]
    extension = data["eci_extension"]
    common_components = data["common_components"]
    knowledge = data["knowledge"]
    active = data["active"]
    active_shrinkage = data["active_shrinkage"]
    eci_architecture_blend = data["eci_architecture_blend"]
    longcat_definition = data["longcat_definition"]
    optimizer = data["optimizer"]
    aa = data["aa_inference"]
    aa_label_availability = data["aa_label_availability"]
    aa_score_availability = data["aa_score_availability"]
    aa_score_timing = data["aa_score_timing_audit"]
    open_model_parameter_truth = data["open_model_parameter_truth"]
    crowd_robustness = data["crowd_robustness"]
    prospective_freeze = data["prospective_freeze"]

    frozen_targets = {
        row["identity"]["canonical_name"]: row
        for row in prospective_freeze["targets"]
    }
    if set(frozen_targets) != set(TARGETS):
        raise ValueError(
            "Prospective freeze target identities do not match readiness targets"
        )

    target_rows = []
    efficiency_by_model = {
        row["model"]: row for row in k3_efficiency["targets"]
    }
    if set(efficiency_by_model) != set(TARGETS):
        raise ValueError("K3 efficiency-reference targets do not match readiness targets")
    for name in TARGETS:
        model = model_by_name(forecast, name)
        interval = target_interval(uncertainty, name)
        frozen = frozen_targets[name]
        efficiency = efficiency_by_model[name]
        projection = interval["k3_efficiency_projection"]
        if abs(model["currentFinalT"] - interval["displayed_final_center_t"]) > 1e-12:
            raise ValueError(f"Forecast/uncertainty center mismatch for {name}")
        if abs(model["currentFinalT"] - frozen["forecast"]["final_center_t"]) > 1e-12:
            raise ValueError(f"Current/frozen point forecast mismatch for {name}")
        target_rows.append(
            {
                "model": name,
                "release_date": model["releaseDate"],
                "evidence_center_t": model["currentEvidenceT"],
                "crowd_center_t": model["factors"].get("crowd"),
                "crowd_forecasters": model["crowd"]["n"],
                "final_center_t": model["currentFinalT"],
                "empirical_50_factor": interval["intervals"]["50"]["multiplicative_factor"],
                "empirical_50_low_t": interval["intervals"]["50"]["low_t"],
                "empirical_50_high_t": interval["intervals"]["50"]["high_t"],
                "empirical_80_factor": interval["intervals"]["80"]["multiplicative_factor"],
                "empirical_80_low_t": interval["intervals"]["80"]["low_t"],
                "empirical_80_high_t": interval["intervals"]["80"]["high_t"],
                "k3_efficiency_reference_median_t": projection["pooled_reference_quantiles_t"]["median"],
                "k3_efficiency_reference_p10_t": projection["pooled_reference_quantiles_t"]["p10"],
                "k3_efficiency_reference_p90_t": projection["pooled_reference_quantiles_t"]["p90"],
                "k3_efficiency_default_strength": projection["default_projection_strength"],
                "k3_efficiency_80_low_t": projection["projected_intervals"]["80"]["low_t"],
                "k3_efficiency_80_high_t": projection["projected_intervals"]["80"]["high_t"],
                "k3_efficiency_center_override_probability": projection["center_override_probability"],
                "k3_efficiency_literal_conditioning": projection["literal_conditioning"],
                "k3_efficiency_point_center_changed": projection["point_center_changed"],
                "k3_efficiency_lower_tail_changed": projection["lower_tail_changed"],
                "parameter_count_disclosed": bool(model["lockedAnchor"]),
            }
        )
        pooled_reference = efficiency["pooled_parameter_equivalent_reference_t"]
        if any(
            abs(projection["pooled_reference_quantiles_t"][key] - pooled_reference[key]) > 1e-12
            for key in ("p10", "median", "p90")
        ):
            raise ValueError(f"K3 efficiency reference mismatch for {name}")

    frontier = backtest["frontier_like_metrics"]["Available-components ensemble"]
    developer = vintage["developer_holdout_current_snapshot"]["developer_frontier"]
    latest = vintage["developer_holdout_current_snapshot"]["latest_per_developer"]
    eci_validation = extension["summary"]["live_inverse_eci_ci"]["live_60_40_blend"]
    knowledge_all = knowledge["cohorts"]["all_first_observed"]
    active_predictability = active["active_parameter_predictability"]
    active_transport = active["high_sparsity_total_transport"]
    k3_ensemble = next(
        row for row in backtest["ensemble_predictions"] if row["model"] == "Kimi K3"
    )
    k3_aa = next(
        component for component in k3_ensemble["components"] if component["panel"] == "AA"
    )
    k3_non_aa = [
        component for component in k3_ensemble["components"] if component["panel"] != "AA"
    ]
    k3_old_log = sum(
        component["weight"] * math.log(float(component["predicted_b"]))
        for component in k3_non_aa
    ) / sum(component["weight"] for component in k3_non_aa)
    k3_old_prediction_b = math.exp(k3_old_log)

    result = {
        "generated_on": GENERATED_ON,
        "role": "descriptive current-state report; never a model input",
        "headline_forecasts": target_rows,
        "live_weights_percent": forecast["defaultWeights"],
        "k3_efficiency_projection": {
            "status": "live center-preserving upper-tail projection; no point-center weight",
            "default_projection_strength": k3_efficiency["decision"]["default_projection_strength"],
            "point_center_weight": k3_efficiency["decision"]["incremental_point_center_weight"],
            "crowd_weight_for_fable_and_sol": k3_efficiency["decision"]["crowd_weight_for_fable_and_sol"],
            "rejected_nonlinear_eci_weight": k3_efficiency["decision"]["rejected_nonlinear_eci_weight"],
            "logical_direction": k3_efficiency["method"]["logical_direction"],
            "diminishing_returns_interpretation": k3_efficiency["method"]["diminishing_returns_interpretation"],
            "literal_conditioning": False,
            "formal_coverage_guarantee": False,
        },
        "heldout_precision": {
            "frontier_lineage_holdout": frontier,
            "frontier_whole_developer_holdout": developer,
            "latest_per_developer": latest,
            "published_prequential_factors": uncertainty["cohorts"]["frontier_like"]["intervals"],
            "holdout_specification_factors": uncertainty["cohorts"]["frontier_like"]["holdout_specs"],
            "formal_coverage_guarantee": uncertainty["decision"]["formal_coverage_guarantee"],
            "post_freeze_diagnostic_correction": uncertainty["post_freeze_diagnostic_correction"],
            "k3_external_aa_reconciliation": {
                "actual_b": k3_ensemble["actual_b"],
                "aa_all_kimi_held_out_predicted_b": k3_aa["predicted_b"],
                "aa_all_kimi_held_out_error_x": max(
                    k3_aa["predicted_b"] / k3_ensemble["actual_b"],
                    k3_ensemble["actual_b"] / k3_aa["predicted_b"],
                ),
                "prior_incomplete_eci_compute_predicted_b": k3_old_prediction_b,
                "prior_incomplete_eci_compute_error_x": max(
                    k3_old_prediction_b / k3_ensemble["actual_b"],
                    k3_ensemble["actual_b"] / k3_old_prediction_b,
                ),
                "corrected_available_component_predicted_b": k3_ensemble["predicted_b"],
                "corrected_available_component_error_x": k3_ensemble["multiplicative_error"],
                "policy": (
                    "K3 remains outside the AA parameter target panel. Its exact-score, "
                    "all-Kimi-held-out AA prediction is included in the available-component "
                    "ensemble so an absent target row cannot masquerade as absent evidence."
                ),
            },
            "interpretation": "Row-level median error is about 2x, but the conservative latest-developer 50% envelope is about 3x; tails are much wider and neither the empirical bands nor the crowd layer have a formal outcome-calibrated coverage claim.",
        },
        "external_validation": {
            "eci_four_target_retrospective": eci_validation["four_target_historical_interval"],
            "k3_score_vintage_only": eci_validation["k3_score_vintage_only"],
            "project_prospective_targets": extension["inventory"]["project_prospective_targets"],
            "change_live_weights": extension["decision"]["change_live_weights"],
            "common_component_panel": {
                "summary": common_components["summary"]["equal_developer"],
                "gates": common_components["decision"]["gates"],
                "live_weight": common_components["decision"]["live_weight"],
            },
        },
        "prospective_commitment": {
            "freeze_id": prospective_freeze["freeze_id"],
            "status": prospective_freeze["status"],
            "locked_at_utc": prospective_freeze["locked_at_utc"],
            "targets": [row["identity"]["canonical_name"] for row in prospective_freeze["targets"]],
            "canonical_payload_sha256": prospective_freeze["artifact_integrity"]["canonical_payload_sha256"],
            "artifact_sha256": (ROOT / "forecast_freezes/2026-07-31-frontier-parameters-v1/forecast_freeze.sha256").read_text(encoding="utf-8").split()[0],
            "post_outcome_refitting": prospective_freeze["evaluation_policy"]["post_outcome_refitting"],
            "frozen_point_centers_match_current": True,
            "current_bands_are_post_freeze_diagnostics": uncertainty["post_freeze_diagnostic_correction"]["applied_after_forecast_freeze"],
            "freeze_rewritten": uncertainty["post_freeze_diagnostic_correction"]["freeze_rewritten"],
            "privacy_redacted": bool(prospective_freeze.get("privacy_redaction")),
            "privacy_prior_artifact_sha256": prospective_freeze.get("privacy_redaction", {}).get("prior_artifact_sha256"),
            "respondent_name_mapping_retained": prospective_freeze.get("privacy_redaction", {}).get("name_to_id_mapping_retained"),
            "frozen_empirical_intervals": {
                name: frozen_targets[name]["forecast"]["empirical_intervals"]
                for name in TARGETS
            },
        },
        "diagnostic_decisions": {
            "vintage_knowledge": {
                "rows": knowledge_all["rows"],
                "developers": knowledge_all["developers"],
                "baseline_median_error_x": knowledge_all["baseline"]["median_multiplicative_error"],
                "candidate_median_error_x": knowledge_all["candidate"]["median_multiplicative_error"],
                "ci90": knowledge_all["paired_developer_bootstrap"]["ci_90"],
                "promotion_gates": knowledge["promotion_gates"],
                "live_weight": knowledge["decision"]["incremental_live_weight"],
            },
            "active_parameters": {
                "checkpoints": active["inventory"]["active_parameter_checkpoints"],
                "developers": active["inventory"]["active_parameter_developers"],
                "active_vs_total_ci90": active_predictability["paired_active_vs_same_panel_total"]["ci_90"],
                "transport_baseline_median_error_x": active_transport["direct_total_baseline"]["median_multiplicative_error"],
                "transport_candidate_median_error_x": active_transport["candidate"]["median_multiplicative_error"],
                "target_sparsity_observed": active["decision"]["independent_target_architecture_observed"],
                "live_weight": active["decision"]["incremental_live_weight"],
            },
            "active_parameter_shrinkage": {
                "inventory": active_shrinkage["inventory"],
                "fixed_all": active_shrinkage["fixed_50_50_evaluation"]["all_high_sparsity"],
                "fixed_frontier": active_shrinkage["fixed_50_50_evaluation"]["frontier_like_high_sparsity"],
                "nested_all": active_shrinkage["nested_weight_evaluation"]["all_high_sparsity"],
                "nested_frontier": active_shrinkage["nested_weight_evaluation"]["frontier_like_high_sparsity"],
                "k3_external_check": active_shrinkage["kimi_k3_external_check"],
                "promotion_gates": active_shrinkage["promotion_gates"],
                "live_weight": active_shrinkage["decision"]["incremental_live_weight"],
            },
            "direct_weight_optimization": {
                "outer_predictions": optimizer["nested_outer_evaluation"]["eligible_predictions"],
                "current_median_error_x": optimizer["nested_outer_evaluation"]["metrics"]["current_weights"]["median_multiplicative_error"],
                "optimized_median_error_x": optimizer["nested_outer_evaluation"]["metrics"]["optimized_mse"]["median_multiplicative_error"],
                "ci90": optimizer["nested_outer_evaluation"]["paired_family_bootstrap"]["ci_90"],
                "update_live_weights": optimizer["decision"]["update_live_weights"],
            },
            "eci_architecture_blend": {
                "inventory": eci_architecture_blend["inventory"],
                "fixed_all": eci_architecture_blend["fixed_evaluation"]["all"],
                "fixed_frontier": eci_architecture_blend["fixed_evaluation"]["frontier_like"],
                "nested_all": eci_architecture_blend["nested_evaluation"]["all"],
                "nested_frontier": eci_architecture_blend["nested_evaluation"]["frontier_like"],
                "anchor_checks": eci_architecture_blend["anchor_checks"],
                "promotion_gates": eci_architecture_blend["promotion_gates"],
                "live_target_applicability": eci_architecture_blend["live_target_applicability"],
                "live_weight": eci_architecture_blend["decision"]["incremental_live_weight"],
            },
            "longcat_parameter_definition": {
                "canonical_total_b": longcat_definition["decision"]["canonical_total_b"],
                "hf_serialized_total_b": longcat_definition["definition_evidence"]["hugging_face_serialized_inventory"]["safetensors_total_elements"] / 1e9,
                "serialized_excluding_mtp_b": longcat_definition["definition_evidence"]["derived_reconciliation"]["serialized_elements_excluding_mtp"] / 1e9,
                "maximum_legacy_target_change_percent": max(
                    abs(row["change_percent"])
                    for row in longcat_definition["target_changes"]
                ),
                "ensemble_all_invariant": (
                    longcat_definition["scenarios"]["publisher_model_total"]["backtest"]["metrics"]["ensemble_all"]
                    == longcat_definition["scenarios"]["hf_serialized_tensor_elements"]["backtest"]["metrics"]["ensemble_all"]
                ),
                "live_weight": longcat_definition["decision"]["incremental_live_weight"],
                "change_live_forecast": longcat_definition["decision"]["change_live_forecast"],
            },
        },
        "data_inventory": {
            "aa_raw_configurations": aa["data_audit"]["raw_models"],
            "aa_calibration_configurations": aa["data_audit"]["open_weight_parameter_score_date_configurations"],
            "aa_calibration_checkpoints": aa["data_audit"]["unique_checkpoint_groups"],
            "aa_calibration_creators": aa["data_audit"]["creators"],
            "aa_primary_metadata_overrides": aa["data_audit"]["primary_metadata_overrides"],
            "aa_parameter_label_timing_records": len(aa_label_availability["records"]),
            "aa_post_release_parameter_label_records": sum(
                record["timing"]["parameter_label_available_date"]
                > record["identity"]["aa_release_date"]
                for record in aa_label_availability["records"]
            ),
            "aa_score_timing": {
                "verified_slugs": aa_score_availability["summary"]["verified_score_slugs"],
                "live_rows": aa_score_timing["coverage"]["live_aa_rows"],
                "live_verified": aa_score_timing["coverage"]["live_aa_verified"],
                "live_fallback": aa_score_timing["coverage"]["live_aa_fallback"],
                "live_verified_after_release": aa_score_timing["coverage"]["live_aa_verified_after_release"],
                "ensemble_rows_before": aa_score_timing["validation_impact"]["available_component_ensemble"]["all"]["release_order_baseline"]["n"],
                "ensemble_rows_after": aa_score_timing["validation_impact"]["available_component_ensemble"]["all"]["score_timing_corrected"]["n"],
                "ensemble_median_before": aa_score_timing["validation_impact"]["available_component_ensemble"]["all"]["release_order_baseline"]["median_multiplicative_error"],
                "ensemble_median_after": aa_score_timing["validation_impact"]["available_component_ensemble"]["all"]["score_timing_corrected"]["median_multiplicative_error"],
                "frontier_median_before": aa_score_timing["validation_impact"]["available_component_ensemble"]["frontier_like"]["release_order_baseline"]["median_multiplicative_error"],
                "frontier_median_after": aa_score_timing["validation_impact"]["available_component_ensemble"]["frontier_like"]["score_timing_corrected"]["median_multiplicative_error"],
                "api_total_reconciles": aa_score_timing["coverage"]["api_total_reconciles"],
            },
            "open_model_parameter_truth": {
                "records": len(open_model_parameter_truth["records"]),
                "aliases": sum(
                    len(record["aliases"])
                    for record in open_model_parameter_truth["records"]
                ),
                "truth_ids": [
                    record["truth_id"]
                    for record in open_model_parameter_truth["records"]
                ],
                "raw_values_preserved": open_model_parameter_truth["policy"]["preserve_raw_values"],
                "checkpoints_deduplicated": open_model_parameter_truth["policy"]["deduplicate_checkpoints"],
            },
        },
        "crowd_robustness": {
            "targets": crowd_robustness["targets"],
            "cross_target_dependence": crowd_robustness["cross_target_dependence"],
            "decision": crowd_robustness["decision"],
        },
        "limitations": [
            "No hidden target has a disclosed parameter count, so the final crowd-plus-model centers are not prospectively calibrated.",
            "Only 11 independent frontier calibration developers make empirical tail factors coarse.",
            "The targets extrapolate beyond the observed open-weight capability frontier.",
            "The central estimates are total-parameter equivalents at observed inference budgets, not physical weight disclosures.",
            "The K3-projected bands are center-preserving winsorized stress tests, not literal conditioning, empirical coverage intervals, or formal Bayesian credible intervals.",
        ],
        "source_files": {relative(path): sha256(path) for path in INPUTS.values()},
        "outputs": {"markdown": relative(MARKDOWN)},
    }
    if any(row["parameter_count_disclosed"] for row in target_rows):
        raise ValueError("A hidden readiness target unexpectedly became disclosed")
    if result["external_validation"]["project_prospective_targets"]:
        raise ValueError("Retrospective validation must not be labeled project-prospective")
    return result


def render_markdown(result: dict[str, Any]) -> str:
    def t(value: float | None) -> str:
        return "—" if value is None else f"{value:.1f}T"

    def x(value: float) -> str:
        return f"{value:.2f}×"

    rows = result["headline_forecasts"]
    efficiency = result["k3_efficiency_projection"]
    precision = result["heldout_precision"]
    lineage = precision["frontier_lineage_holdout"]
    developer = precision["frontier_whole_developer_holdout"]
    latest = precision["latest_per_developer"]
    published = precision["published_prequential_factors"]
    k3_reconciliation = precision["k3_external_aa_reconciliation"]
    extension = result["external_validation"]
    common = extension["common_component_panel"]
    commitment = result["prospective_commitment"]
    knowledge = result["diagnostic_decisions"]["vintage_knowledge"]
    active = result["diagnostic_decisions"]["active_parameters"]
    shrinkage = result["diagnostic_decisions"]["active_parameter_shrinkage"]
    optimizer = result["diagnostic_decisions"]["direct_weight_optimization"]
    architecture = result["diagnostic_decisions"]["eci_architecture_blend"]
    longcat = result["diagnostic_decisions"]["longcat_parameter_definition"]
    aa = result["data_inventory"]
    crowd = result["crowd_robustness"]
    aa_override_count = len(aa["aa_primary_metadata_overrides"])
    aa_override_label = (
        "override" if aa_override_count == 1 else "overrides"
    )
    active_ci = active["active_vs_total_ci90"]
    active_ci_label = (
        "wholly favorable"
        if active_ci[1] < 0
        else "favorably centered but crosses zero"
    )

    lines = [
        "# Current frontier-parameter model readiness",
        "",
        f"Generated automatically from the audited pipeline on {result['generated_on']}.",
        "",
        "## Current forecasts",
        "",
        "| Model | Evidence | Crowd | Final | Empirical 50% band | Empirical 80% band |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {t(row['evidence_center_t'])} | {t(row['crowd_center_t'])} | "
            f"**{t(row['final_center_t'])}** | {t(row['empirical_50_low_t'])}–{t(row['empirical_50_high_t'])} | "
            f"{t(row['empirical_80_low_t'])}–{t(row['empirical_80_high_t'])} |"
        )
    lines.extend(
        [
            "",
            "The intervals are calibrated around the evidence centers. Crowd forecasts shift Fable/Sol's displayed centers but do not narrow coverage.",
            "",
            "## K3 efficiency upper-tail stress test",
            "",
            f"The user-supplied assumption that the targets are at least as parameter-efficient as disclosed 2.780T Kimi K3 is represented by center-preserving winsorization on {efficiency['default_projection_strength']:.0%} of upper-tail draws. It has {efficiency['point_center_weight']:.0%} point-center weight, leaves every lower endpoint unchanged, preserves the exact {efficiency['crowd_weight_for_fable_and_sol']:.0%} Fable/Sol crowd blend, and assigns the rejected nonlinear ECI extrapolation {efficiency['rejected_nonlinear_eci_weight']:.0%} weight.",
            "",
            "| Model | Pooled K3-relative reference median | Reference 10–90% | Raw 80% band | Projected 80% stress test | Center override |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['model']} | {t(row['k3_efficiency_reference_median_t'])} | "
            f"{t(row['k3_efficiency_reference_p10_t'])}–{t(row['k3_efficiency_reference_p90_t'])} | "
            f"{t(row['empirical_80_low_t'])}–{t(row['empirical_80_high_t'])} | "
            f"{t(row['k3_efficiency_80_low_t'])}–{t(row['k3_efficiency_80_high_t'])} | "
            f"{row['k3_efficiency_center_override_probability']:.0%} |"
        )
    lines.extend(
        [
            "",
            "This is a center-preserving winsorized structural stress test—not literal conditioning, an empirical coverage interval, or a formal Bayesian credible interval. The pooled AA/ECI reference is not a strict ceiling under either mapping. The high Fable override rate exposes tension with the existing evidence center: when the reference is lower, the center wins rather than silently recentering Fable.",
            "",
            "## Precision",
            "",
            f"- Frontier lineage holdout: {lineage['n']} predictions, median error {x(lineage['median_multiplicative_error'])}, {lineage['within_2x']:.0%} within 2× and {lineage['within_3x']:.0%} within 3×.",
            f"- Whole-developer frontier holdout: {developer['n']} predictions, median error {x(developer['median_multiplicative_error'])}.",
            f"- Latest residual per developer under whole-developer refits: {latest['developers']} developers; median factor {x(latest['order_statistic_factors']['50']['multiplicative_factor'])}.",
            f"- Conservative envelope across lineage- and whole-developer-holdout specifications: {x(published['50']['multiplicative_factor'])} at 50%, {x(published['80']['multiplicative_factor'])} at 80%, and {x(published['90']['multiplicative_factor'])} at 90%.",
            f"- Kimi K3 audit correction: the exact-score AA fit with all Kimi lineages held out predicts {k3_reconciliation['aa_all_kimi_held_out_predicted_b'] / 1000:.2f}T ({x(k3_reconciliation['aa_all_kimi_held_out_error_x'])}). The prior {k3_reconciliation['prior_incomplete_eci_compute_predicted_b'] / 1000:.3f}T / {x(k3_reconciliation['prior_incomplete_eci_compute_error_x'])} row used only ECI and speculative compute because K3 was external to the AA target table; after incorporating the leakage-safe AA component, the available-component result is {k3_reconciliation['corrected_available_component_predicted_b'] / 1000:.3f}T / {x(k3_reconciliation['corrected_available_component_error_x'])}.",
            "- Practical reading: row-level point errors are roughly factor-of-two, but a new developer's empirical central band is closer to factor-of-three and the tails are much wider. These are prequential error bands, not formal conformal coverage guarantees.",
            "",
            "## External and vintage validation",
            "",
            f"- Frozen ECI form on four retrospective interval targets: median error {x(extension['eci_four_target_retrospective']['median_multiplicative_error'])}; {extension['eci_four_target_retrospective']['within_2x']:.0%} within 2×.",
            f"- Kimi K3 later score-vintage check: {x(extension['k3_score_vintage_only']['median_multiplicative_error'])} error; not project-prospective because K3's size was already known.",
            f"- Four-model common GPQA/MATH/AIME panel: median error {x(common['summary']['median_multiplicative_error'])}, {common['summary']['within_2x']:.0%} within 2×; live weight {common['live_weight']:.0%} after fixed gates fail.",
            f"- Vintage knowledge challenger: {knowledge['rows']} rows/{knowledge['developers']} developers, {x(knowledge['baseline_median_error_x'])} → {x(knowledge['candidate_median_error_x'])}; live weight {knowledge['live_weight']:.0%} because coverage gates fail.",
            f"- High-sparsity shrinkage challenger: {shrinkage['inventory']['high_sparsity_rows']} rows/{shrinkage['inventory']['high_sparsity_developers']} developers, {x(shrinkage['fixed_all']['baseline']['median_multiplicative_error'])} → {x(shrinkage['fixed_all']['candidate']['median_multiplicative_error'])}; live weight {shrinkage['live_weight']:.0%} because target sparsity and preregistration gates fail.",
            "",
            "## Prospective commitment",
            "",
            f"- Freeze `{commitment['freeze_id']}` is `{commitment['status']}` for {', '.join(commitment['targets'])}.",
            f"- Artifact SHA-256: `{commitment['artifact_sha256']}`; post-outcome refitting is **{commitment['post_outcome_refitting']}**.",
            "- The frozen point centers exactly equal the current centers. The table's uncertainty bands are later, corrected diagnostics; prospective interval scoring must use the immutable bands inside the freeze artifact, which was not rewritten.",
            "- Poll identities are privacy-redacted to stable anonymous respondent IDs in the public freeze. The redacted artifact preserves the prior digest and all numerical fields, and the project retains no name-to-ID mapping.",
            "",
            "## Retained decisions",
            "",
            f"- Active-parameter recovery has a 90% developer interval [{active_ci[0]:+.3f}, {active_ci[1]:+.3f}] ({active_ci_label}), while total transport worsens median error ({x(active['transport_baseline_median_error_x'])} → {x(active['transport_candidate_median_error_x'])}) and target sparsity is unobserved; live weight remains {active['live_weight']:.0%}.",
            f"- Direct factor-weight optimization worsens median error on {optimizer['outer_predictions']} outer tests ({x(optimizer['current_median_error_x'])} → {x(optimizer['optimized_median_error_x'])}); weights remain unchanged.",
            f"- ECI architecture-blend challenger: fixed whole-developer median improves on {architecture['fixed_all']['baseline']['n']} rows from {x(architecture['fixed_all']['baseline']['median_multiplicative_error'])} to {x(architecture['fixed_all']['challenger']['median_multiplicative_error'])}, but within-2× accuracy worsens, both frontier developer intervals cross zero, K3 error worsens, and target architecture is unobserved; live weight remains {architecture['live_weight']:.0%}.",
            f"- LongCat parameter-definition audit: retain the publisher's {longcat['canonical_total_b'] / 1000:.1f}T semantic model total; the exact {longcat['hf_serialized_total_b'] / 1000:.3f}T serialized inventory falls to {longcat['serialized_excluding_mtp_b'] / 1000:.3f}T after excluding MTP tensors. The alternative moves legacy target fits by at most {longcat['maximum_legacy_target_change_percent']:.2f}%, leaves the matched ensemble invariant, and receives {longcat['live_weight']:.0%} live weight.",
            f"- AA calibration view: {aa['aa_raw_configurations']} raw configurations, {aa['aa_calibration_configurations']} eligible configurations, {aa['aa_calibration_checkpoints']} checkpoints, and {aa['aa_calibration_creators']} creators after {aa_override_count} explicit primary-source {aa_override_label}.",
            f"- AA parameter-label timing: {aa['aa_parameter_label_timing_records']} pinned records, of which {aa['aa_post_release_parameter_label_records']} become eligible after nominal model release; chronological folds use the later date while the current fit is unchanged.",
            f"- AA score-publication timing: {aa['aa_score_timing']['live_verified']} of {aa['aa_score_timing']['live_rows']} live AA checkpoints have verified non-null changelog dates and {aa['aa_score_timing']['live_verified_after_release']} were published after nominal release. Correcting information dates changes the all-ensemble audit from {aa['aa_score_timing']['ensemble_rows_before']} rows/{x(aa['aa_score_timing']['ensemble_median_before'])} to {aa['aa_score_timing']['ensemble_rows_after']} rows/{x(aa['aa_score_timing']['ensemble_median_after'])}, while the frontier median changes from {x(aa['aa_score_timing']['frontier_median_before'])} to {x(aa['aa_score_timing']['frontier_median_after'])}; centers and live weights remain unchanged.",
            f"- Parameter-truth reconciliation: {aa['open_model_parameter_truth']['records']} narrow primary-source overlays canonicalize Moonshot K2 and MiniMax M2.5/M2.7 coarse labels while preserving raw values and all distinct checkpoints; no global match tolerance is widened.",
            f"- Crowd center robustness: leave-one-contributor-out final ranges are {t(crowd['targets'][0]['leave_one_contributor_out']['final_min_t'])}–{t(crowd['targets'][0]['leave_one_contributor_out']['final_max_t'])} for Fable and {t(crowd['targets'][1]['leave_one_contributor_out']['final_min_t'])}–{t(crowd['targets'][1]['leave_one_contributor_out']['final_max_t'])} for Sol. The {crowd['cross_target_dependence']['paired_contributors']} paired contributors have log-point correlation {crowd['cross_target_dependence']['pearson_correlation_of_log_points']:.2f}; crowd agreement remains correlated and does not narrow intervals.",
            "",
            "## Bottom line",
            "",
            "Ready for comparative forecasting and sensitivity analysis; not precise enough to claim literal hidden counts to a decimal place. The strongest remaining validation is a future unadjusted comparison against a parameter disclosure for Fable, Sol, or Opus 5.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    data = load_inputs()
    result = build_result(data)
    markdown = render_markdown(result)
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    MARKDOWN.write_text(markdown, encoding="utf-8")
    print(
        json.dumps(
            {
                "result": relative(RESULT),
                "markdown": relative(MARKDOWN),
                "targets": len(result["headline_forecasts"]),
                "frontier_median_error_x": result["heldout_precision"]["frontier_lineage_holdout"]["median_multiplicative_error"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
