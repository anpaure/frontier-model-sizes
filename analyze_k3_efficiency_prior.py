#!/usr/bin/env python3
"""Build the Kimi K3 relative-efficiency reference used by uncertainty reporting.

The assumption being represented is one-sided: a proprietary frontier model is
at least as parameter-efficient as Kimi K3. Conditional on any one
benchmark-to-size transport law, that makes the K3-relative parameter
equivalent an upper reference. The AA/ECI geometric pool is a judgmental
structural sensitivity, not a literal physical ceiling or an independent
point-estimate factor.

The retained AA/ECI laws are linear in log10(parameters), so diminishing
returns in raw parameters are already present.  The rejected quadratic ECI
challenger is deliberately absent from this module.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import analyze_eci_fit_tournament as eci_fit
from analyze_epoch_feedback_signal import (
    ECI_DATED_SCORE as ECI_CANONICAL_DATED_SLOPE,
)
from analyze_epoch_feedback_signal import (
    ECI_NODATE_SCORE as ECI_CANONICAL_NO_DATE_SLOPE,
)
from frontier_target_signals import AA_DETAILED_PATH, AA_TARGET_SIGNALS


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs/019f6c42-2d53-7743-ab07-6293e2618dd7"
REGRESSION = ROOT / "regression_results.json"
K3_EVIDENCE = ROOT / "sources/kimi_k3_release_evidence_2026-07-31.json"
ECI_TOURNAMENT = OUT / "eci_fit_tournament_2026-07-18.json"
AA_EXPANDED_AUDIT = OUT / "aa_expanded_parameter_audit_2026-07-18.json"
RESULT = OUT / "k3_efficiency_prior_2026-08-01.json"
DRAWS = OUT / "k3_efficiency_prior_cap_draws_2026-08-01.csv"
AUDIT = ROOT / "K3_EFFICIENCY_PRIOR_AUDIT.md"

GENERATED_ON = "2026-08-01"
SEED = 20260801
DRAW_COUNT = 20_000
K3_MODEL = "Kimi K3"
K3_RELEASE = "2026-07-16"
K3_TOTAL_T = 2.780
ECI_NO_DATE_WEIGHT = 0.60
ECI_DATED_WEIGHT = 0.40
DEFAULT_PROJECTION_STRENGTH = 0.80

TARGETS = (
    ("claude-fable-5", "Claude Fable 5"),
    ("gpt-56-sol", "GPT-5.6 Sol"),
    ("claude-opus-5", "Claude Opus 5"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def quantiles(values: np.ndarray) -> dict[str, float]:
    probabilities = (0.025, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975)
    names = ("p025", "p10", "p25", "median", "p75", "p90", "p95", "p975")
    return {
        name: float(value)
        for name, value in zip(names, np.quantile(values, probabilities))
    }


def split_half_normal(
    rng: np.random.Generator,
    center: float,
    low_90: float,
    high_90: float,
    size: int,
) -> np.ndarray:
    """Approximate an asymmetric 90% interval without inventing covariance.

    Each side receives half the probability mass.  Scaling a half-normal by
    the corresponding central-to-endpoint distance makes the supplied bounds
    the marginal 5th/95th percentiles.
    """

    if not low_90 <= center <= high_90:
        raise ValueError("ECI confidence interval does not contain its center")
    z_95 = 1.6448536269514722
    magnitude = np.abs(rng.normal(size=size))
    lower_side = rng.random(size) < 0.5
    return center + np.where(
        lower_side,
        -magnitude * (center - low_90) / z_95,
        magnitude * (high_90 - center) / z_95,
    )


def linear_eci_slope_variants(panel: pd.DataFrame, k3: dict[str, Any]) -> list[dict[str, Any]]:
    """Return only retained log-linear ECI sensitivity slopes.

    K3's parameter label is excluded from slope fitting.  It enters later only
    as the exact intercept anchor.  Target and K3 dates are held equal because
    this reference comparison explicitly holds algorithmic efficiency equal.
    """

    tests = pd.DataFrame(
        [
            {
                "release_date": K3_RELEASE,
                "score": float(k3["score"]),
                "reasoning": 1,
                "moe": 1,
                "coder": 0,
                "multimodal": 0,
            },
            {
                "release_date": K3_RELEASE,
                "score": float(k3["score"]) + 1.0,
                "reasoning": 1,
                "moe": 1,
                "coder": 0,
                "multimodal": 0,
            },
        ]
    )
    variants: list[dict[str, Any]] = []
    for weight_mode in eci_fit.WEIGHT_MODES:
        for collapsed in (False, True):
            train = eci_fit.collapse_same_base(panel) if collapsed else panel.copy()
            no_date = eci_fit.fit_predict_single(
                train, tests, "score", weight_mode
            )
            dated = eci_fit.fit_predict_single(
                train, tests, "score_date", weight_mode
            )
            variants.append(
                {
                    "id": f"{weight_mode}__{'collapsed' if collapsed else 'all_rows'}",
                    "weight_mode": weight_mode,
                    "same_base_collapsed": collapsed,
                    "training_rows": len(train),
                    "training_families": int(train["family"].nunique()),
                    "no_date_log10_slope": float(no_date[1] - no_date[0]),
                    "dated_same_efficiency_log10_slope": float(dated[1] - dated[0]),
                }
            )
    variants.append(
        {
            "id": "canonical_live_coefficients",
            "weight_mode": "canonical",
            "same_base_collapsed": False,
            "training_rows": None,
            "training_families": None,
            "no_date_log10_slope": ECI_CANONICAL_NO_DATE_SLOPE,
            "dated_same_efficiency_log10_slope": ECI_CANONICAL_DATED_SLOPE,
        }
    )
    for row in variants:
        row["blended_log10_slope"] = (
            ECI_NO_DATE_WEIGHT * row["no_date_log10_slope"]
            + ECI_DATED_WEIGHT * row["dated_same_efficiency_log10_slope"]
        )
        if row["blended_log10_slope"] <= 0:
            raise ValueError("K3 efficiency reference requires a positive ECI slope")
    return variants


def write_draws(rows: list[dict[str, Any]]) -> None:
    DRAWS.parent.mkdir(parents=True, exist_ok=True)
    with DRAWS.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def markdown(result: dict[str, Any]) -> str:
    target_lines = []
    for target in result["targets"]:
        q = target["pooled_parameter_equivalent_reference_t"]
        target_lines.append(
            f"| {target['model']} | {target['aa_same_efficiency_equivalent_t']:.2f}T | "
            f"{target['eci_same_efficiency_equivalent_t']['canonical']:.2f}T | "
            f"{q['median']:.2f}T | {q['p10']:.2f}–{q['p90']:.2f}T |"
        )
    return "\n".join(
        [
            "# K3 relative-efficiency reference audit",
            "",
            f"Generated {GENERATED_ON}. The center-preserving projection is **live for upper-tail sensitivity at "
            f"{100 * DEFAULT_PROJECTION_STRENGTH:.0f}% strength** and has **0% point-center weight**.",
            "",
            "## Contract",
            "",
            "The assumption ‘at least as parameter-efficient as Kimi K3’ is one-sided. "
            "For a fixed capability mapping it implies `target parameters <= K3-relative parameter equivalent`. "
            "It does not imply that a target is at least 2.78T; that is a separate size-floor assumption.",
            "",
            "The retained models are linear in log10(parameters), so they already encode diminishing "
            "returns in raw parameters. No quadratic, ridge-flexible, spline, or score-asymptote curve enters this reference.",
            "",
            "| Target | AA equivalent | Canonical ECI equivalent | Pooled reference median | Reference 10–90% |",
            "|---|---:|---:|---:|---:|",
            *target_lines,
            "",
            "The pooled reference distribution propagates current ECI confidence intervals and five retained "
            "linear slope sensitivities. AA and ECI receive equal log weight; they are explicitly correlated alternative "
            "mappings, not independent likelihoods. Their geometric mean is not a strict ceiling under either mapping.",
            "",
            "## Support audit",
            "",
            f"K3 is excluded from the ECI slope fit. The largest remaining calibration ECI is "
            f"{result['validation']['eci_calibration_max_excluding_k3']:.2f}; K3 is "
            f"{result['validation']['k3_eci_points_beyond_calibration']:.2f} points beyond it. "
            f"Fable, Sol, and Opus 5 are respectively "
            f"{result['validation']['target_eci_points_beyond_calibration']['Claude Fable 5']:.2f}, "
            f"{result['validation']['target_eci_points_beyond_calibration']['GPT-5.6 Sol']:.2f}, and "
            f"{result['validation']['target_eci_points_beyond_calibration']['Claude Opus 5']:.2f} points beyond it. "
            "The linear transport is therefore an explicit extrapolative sensitivity, not a newly validated frontier law.",
            "",
            "## Integration decision",
            "",
            "- Preserve every evidence center and the exact 50% crowd blend.",
            "- Winsorize only draws above the evidence center; lower tails are unchanged.",
            "- When the pooled reference is below the center, preserve the center and record that override explicitly.",
            "- Publish both raw empirical intervals and center-preserving K3-efficiency projection intervals.",
            "- The projected intervals are not conditioning, conformal intervals, or Bayesian credible intervals.",
            "- The user-supplied Sol < Fable ordering is applied to the Sol reference draws only; it does not alter centers or enforce actual-size ordering.",
            "",
            "## Why the rejected nonlinear result is absent",
            "",
            "The ECI functional-form tournament did not promote its flexible ridge challenger. The earlier "
            "8.5–12.2T K3-rebased result transported an out-of-support quadratic derivative and K3's entire residual. "
            "It remains a rejected stress test and contributes exactly zero weight here.",
            "",
        ]
    )


def main() -> None:
    regression = json.loads(REGRESSION.read_text(encoding="utf-8"))
    evidence = json.loads(K3_EVIDENCE.read_text(encoding="utf-8"))
    tournament = json.loads(ECI_TOURNAMENT.read_text(encoding="utf-8"))
    aa_audit = json.loads(AA_EXPANDED_AUDIT.read_text(encoding="utf-8"))
    aa_slope = float(
        aa_audit["full_fit"]["current_live_k3_anchored_coefficients"][
            "score_slope"
        ]
    )
    k3_aa = float(AA_TARGET_SIGNALS[K3_MODEL]["score"])
    k3_facts = evidence["kimi_k3"]
    if (
        float(k3_facts["total_parameters_b_exact"]) != 2780.0
        or float(k3_facts["activated_parameters_b_exact"]) != 104.2
        or not k3_facts["parameter_count_disclosed"]
    ):
        raise ValueError("K3 primary evidence must retain exact 2.780T / 104.2B")
    if tournament["decision"]["change_live_eci_functional_form"]:
        raise ValueError("K3 reference policy assumes the nonlinear ECI challenger remains rejected")
    if aa_slope <= 0:
        raise ValueError("K3 efficiency reference requires a positive AA score slope")

    # Reuse the tournament's canonical panel constructor so weighting and
    # same-base identifiers are exactly the ones already audited there.
    eci_panel, _ = eci_fit.load_panel()
    eci_panel = eci_panel.loc[eci_panel["current_canonical"]].copy()
    k3_rows = eci_panel.loc[eci_panel["model"] == K3_MODEL]
    if len(k3_rows) != 1:
        raise ValueError("Expected exactly one K3 ECI row")
    k3 = k3_rows.iloc[0].to_dict()
    if k3["release_date"] != K3_RELEASE or float(k3["total_b"]) != 2780.0:
        raise ValueError("K3 ECI identity does not match the primary anchor")
    slope_panel = eci_panel.loc[eci_panel["model"] != K3_MODEL].copy()
    variants = linear_eci_slope_variants(slope_panel, k3)

    frontier = {row["model"]: row for row in regression["frontier_predictions"]}
    if set(name for _, name in TARGETS) - frontier.keys():
        raise ValueError("Frontier prediction table is missing a K3-reference target")

    rng = np.random.default_rng(SEED)
    k3_score_draw = split_half_normal(
        rng,
        float(k3["score"]),
        float(k3["ci_low"]),
        float(k3["ci_high"]),
        DRAW_COUNT,
    )
    variant_index = rng.integers(0, len(variants), size=DRAW_COUNT)
    variant_slopes = np.asarray(
        [row["blended_log10_slope"] for row in variants], dtype=float
    )[variant_index]

    target_arrays: dict[str, np.ndarray] = {}
    raw_arrays: dict[str, np.ndarray] = {}
    target_results: list[dict[str, Any]] = []
    for model_id, model in TARGETS:
        row = frontier[model]
        aa_signal = AA_TARGET_SIGNALS[model]
        if (
            not math.isclose(
                float(row["aa_score"]), float(aa_signal["score"]), rel_tol=0, abs_tol=1e-12
            )
            or row["release_date"] != aa_signal["release_date"]
        ):
            raise ValueError(f"Frontier AA input does not match its frozen exact signal: {model}")
        aa_equivalent = K3_TOTAL_T * 10 ** (
            aa_slope * (float(aa_signal["score"]) - k3_aa)
        )
        target_eci_draw = split_half_normal(
            rng,
            float(row["eci"]),
            float(row["eci_low"]),
            float(row["eci_high"]),
            DRAW_COUNT,
        )
        eci_equivalent_draw = K3_TOTAL_T * 10 ** (
            variant_slopes * (target_eci_draw - k3_score_draw)
        )
        pooled_reference = np.sqrt(aa_equivalent * eci_equivalent_draw)
        raw_arrays[model] = pooled_reference
        target_arrays[model] = pooled_reference.copy()
        canonical_slope = (
            ECI_NO_DATE_WEIGHT * ECI_CANONICAL_NO_DATE_SLOPE
            + ECI_DATED_WEIGHT * ECI_CANONICAL_DATED_SLOPE
        )
        canonical_eci = K3_TOTAL_T * 10 ** (
            canonical_slope * (float(row["eci"]) - float(k3["score"]))
        )
        target_results.append(
            {
                "model_id": model_id,
                "model": model,
                "release_date": row["release_date"],
                "aa_score": float(row["aa_score"]),
                "eci_score": float(row["eci"]),
                "eci_ci_90": [float(row["eci_low"]), float(row["eci_high"])],
                "aa_same_efficiency_equivalent_t": float(aa_equivalent),
                "eci_same_efficiency_equivalent_t": {
                    "canonical": float(canonical_eci),
                    "variant_centers": [
                        {
                            "variant": variant["id"],
                            "equivalent_t": float(
                                K3_TOTAL_T
                                * 10
                                ** (
                                    variant["blended_log10_slope"]
                                    * (float(row["eci"]) - float(k3["score"]))
                                )
                            ),
                        }
                        for variant in variants
                    ],
                },
            }
        )

    # User-supplied ordering sensitivity: Sol is smaller than Fable. Project
    # only the pooled Sol reference not to exceed Fable's; this does not enforce
    # an ordering on actual parameter draws or point centers.
    target_arrays["GPT-5.6 Sol"] = np.minimum(
        target_arrays["GPT-5.6 Sol"], target_arrays["Claude Fable 5"]
    )

    draw_rows: list[dict[str, Any]] = []
    for index in range(DRAW_COUNT):
        draw_rows.append(
            {
                "draw": index,
                "eci_variant": variants[int(variant_index[index])]["id"],
                "k3_eci_draw": float(k3_score_draw[index]),
                "fable_raw_reference_t": float(raw_arrays["Claude Fable 5"][index]),
                "fable_ordered_reference_t": float(target_arrays["Claude Fable 5"][index]),
                "sol_raw_reference_t": float(raw_arrays["GPT-5.6 Sol"][index]),
                "sol_ordered_reference_t": float(target_arrays["GPT-5.6 Sol"][index]),
                "opus5_raw_reference_t": float(raw_arrays["Claude Opus 5"][index]),
                "opus5_ordered_reference_t": float(target_arrays["Claude Opus 5"][index]),
            }
        )
    write_draws(draw_rows)

    by_model = {row["model"]: row for row in target_results}
    for model_id, model in TARGETS:
        by_model[model]["pooled_parameter_equivalent_reference_t"] = quantiles(
            target_arrays[model]
        )
        by_model[model]["raw_unordered_pooled_reference_t"] = quantiles(
            raw_arrays[model]
        )
        by_model[model]["order_projection_applied"] = model == "GPT-5.6 Sol"

    result = {
        "schema_version": "1.0",
        "generated_on": GENERATED_ON,
        "question": "How should the assumption that frontier labs are at least as parameter-efficient as Kimi K3 affect upper-tail sensitivity without silently overriding the published centers?",
        "anchor": {
            "model": K3_MODEL,
            "release_date": K3_RELEASE,
            "total_parameters_t": K3_TOTAL_T,
            "activated_parameters_b": 104.2,
            "aa_score": k3_aa,
            "eci_score": float(k3["score"]),
            "eci_ci_90": [float(k3["ci_low"]), float(k3["ci_high"])],
        },
        "method": {
            "logical_direction": "under any specified capability mapping, at-least-K3-efficient implies target parameters are no greater than that mapping's K3-relative parameter equivalent",
            "aa_mapping": "exact K3-relative score difference under the retained log10-parameter slope; target and K3 dates held equal",
            "aa_log10_parameter_slope": aa_slope,
            "eci_mapping": "60/40 no-date/dated log-linear slope blend; target and K3 dates held equal; K3 parameter label excluded from slope fits",
            "aa_eci_pool": "equal log weight; judgmental pool of correlated alternative mappings, not an independent likelihood or a strict logical ceiling",
            "eci_score_uncertainty": "independent piecewise half-normal approximation to published 90% marginal intervals; covariance unavailable",
            "eci_model_form_sensitivity": "uniform over the canonical live coefficients and four retained linear weighting/base-collapse variants",
            "diminishing_returns_interpretation": "linear capability versus log10(parameters) already gives diminishing marginal capability per raw parameter",
            "nonlinear_forms_used": [],
            "draws": DRAW_COUNT,
            "seed": SEED,
            "reference_order_projection": "Sol pooled reference is projected not to exceed Fable's; actual-size draws and centers are untouched",
        },
        "linear_eci_slope_variants": variants,
        "targets": target_results,
        "validation": {
            "eci_slope_fit_rows_excluding_k3": int(len(slope_panel)),
            "eci_slope_fit_families_excluding_k3": int(slope_panel["family"].nunique()),
            "k3_parameter_labels_used_in_slope_fit": 0,
            "eci_calibration_max_excluding_k3": float(slope_panel["score"].max()),
            "k3_eci_points_beyond_calibration": float(
                float(k3["score"]) - float(slope_panel["score"].max())
            ),
            "target_eci_points_beyond_calibration": {
                model: float(
                    float(frontier[model]["eci"])
                    - float(slope_panel["score"].max())
                )
                for _, model in TARGETS
            },
            "existing_eci_tournament_changed_live_form": tournament["decision"][
                "change_live_eci_functional_form"
            ],
            "existing_eci_tournament_frontier_stability_gate_passed": tournament[
                "decision"
            ]["frontier_sensitivity_below_1_25x"],
            "chronological_validation_claim": "none beyond the existing ECI tournament; this artifact transports only retained linear slopes and remains a structural sensitivity",
            "rejected_convex_extrapolation": {
                "claimed_fable_t": [8.5, 10.6],
                "claimed_sol_t": [9.7, 12.2],
                "live_weight": 0.0,
                "reason": "out-of-support nonlinear derivative plus transport of K3's full model-specific residual",
            },
        },
        "decision": {
            "apply_center_preserving_upper_tail_projection": True,
            "default_projection_strength": DEFAULT_PROJECTION_STRENGTH,
            "change_point_centers": False,
            "incremental_point_center_weight": 0.0,
            "change_crowd_weight": False,
            "crowd_weight_for_fable_and_sol": 0.50,
            "rejected_nonlinear_eci_weight": 0.0,
            "literal_constraint_enforced_when_reference_below_center": False,
            "reason": "The efficiency statement supplies a one-sided structural reference. The live layer winsorizes upper-tail draws while preserving the evidence center, and therefore is not literal conditioning. The nonlinear challenger failed its promotion gates.",
        },
        "sources": {
            "regression_results": str(REGRESSION.relative_to(ROOT)),
            "regression_results_sha256": sha256(REGRESSION),
            "k3_primary_evidence": str(K3_EVIDENCE.relative_to(ROOT)),
            "k3_primary_evidence_sha256": sha256(K3_EVIDENCE),
            "eci_fit_tournament": str(ECI_TOURNAMENT.relative_to(ROOT)),
            "eci_fit_tournament_sha256": sha256(ECI_TOURNAMENT),
            "aa_live_fit_audit": str(AA_EXPANDED_AUDIT.relative_to(ROOT)),
            "aa_live_fit_audit_sha256": sha256(AA_EXPANDED_AUDIT),
            "aa_exact_target_snapshot": str(AA_DETAILED_PATH.relative_to(ROOT)),
            "aa_exact_target_snapshot_sha256": sha256(AA_DETAILED_PATH),
            "canonical_eci_coefficient_source": "analyze_epoch_feedback_signal.py",
            "canonical_eci_coefficient_source_sha256": sha256(
                ROOT / "analyze_epoch_feedback_signal.py"
            ),
        },
        "outputs": {
            "draw_ledger": str(DRAWS.relative_to(ROOT)),
            "draw_ledger_sha256": sha256(DRAWS),
        },
        "limitations": [
            "AA and ECI are post-training/deployment capability aggregates, not clean pretraining-loss measurements.",
            "The target architecture and sparsity ratios are undisclosed; the reference is conditional on the user's comparable-MoE assumption.",
            "Published ECI bootstrap covariance between target and K3 scores is unavailable, so marginal score draws are treated independently.",
            "The five uniformly sampled ECI slope variants are uncalibrated sensitivity specifications, not five independent datasets or posterior model probabilities.",
            "AA score/slope uncertainty and within-fit ECI coefficient uncertainty are not propagated; reference quantiles are specification sensitivities, not posterior intervals.",
            "The center-preserving upper-tail projections produced downstream are winsorized structural sensitivities, not conditioning, conformal intervals, or Bayesian credible intervals.",
        ],
    }
    RESULT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    AUDIT.write_text(markdown(result), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(RESULT),
                "draws": DRAW_COUNT,
                "targets": {
                    row["model"]: row["pooled_parameter_equivalent_reference_t"]
                    for row in target_results
                },
                "center_weight": 0.0,
                "upper_tail_strength": DEFAULT_PROJECTION_STRENGTH,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
