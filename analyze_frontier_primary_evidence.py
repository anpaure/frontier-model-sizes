#!/usr/bin/env python3
"""Audit how first-party frontier evidence should affect parameter forecasts.

The direct GPT-5.6 Sol no-CoT point is genuine new evidence, but a capability
measurement is not a parameter disclosure. This audit compares two mappings:

1. the paper's aggregate Pareto elasticity anchored to Kimi K3; and
2. a model-level regression tested chronologically while excluding the held-out
   developer.

It also records same-size family controls. The output deliberately keeps the
measurement even when the promotion gate rejects an incremental live weight.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from k3_primary_evidence import K3_EVIDENCE_PATH, K3_PARAMETER_SOURCE, K3_TOTAL_T


ROOT = Path(__file__).resolve().parent
DATE = "2026-07-18"
THREAD = "019f6c42-2d53-7743-ab07-6293e2618dd7"
OUT = ROOT / "outputs" / THREAD
UNIFIED = OUT / "unified_model_observations_compute_enriched_2026-07-17.csv"
EVIDENCE = ROOT / f"sources/frontier_primary_evidence_{DATE}.csv"
EVIDENCE_METADATA = ROOT / f"sources/frontier_primary_evidence_collection_metadata_{DATE}.json"
EXACT_DATE_AUDIT = OUT / f"no_cot_exact_date_audit_{DATE}.json"
LINEAGE_EDGES = OUT / f"posttraining_lineage_edges_{DATE}.csv"
RESULT = OUT / f"frontier_primary_evidence_audit_{DATE}.json"
CONTROLS = OUT / f"frontier_primary_evidence_controls_{DATE}.csv"

ORIGIN = date(2024, 1, 1)
K3_PRETRAIN_PROXY_MINUTES = 2.4
POOLED_TOTAL_FACTOR_PER_DOUBLING = 4.2
MOE_TOTAL_FACTOR_PER_DOUBLING = 8.1
SOL_RELEASE_DATE = date(2026, 7, 9)
GPT_5_4_RELEASE_DATE = date(2026, 3, 5)
GPT_5_5_RELEASE_DATE = date(2026, 4, 23)
GPT_SHARED_HORIZON = math.sqrt(1.4 * 3.0)
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20_260_718


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def design(row: dict[str, Any], include_horizon: bool) -> list[float]:
    release = date.fromisoformat(row["canonical_release_date"])
    values = [
        1.0,
        (release - ORIGIN).days / 365.25,
        float(row["architecture"] == "MoE"),
        float(row["reasoning"] != "Non-reasoning"),
    ]
    if include_horizon:
        values.append(math.log(float(row["nocot_time_horizon_minutes"])))
    return values


def fit(rows: list[dict[str, Any]], include_horizon: bool) -> np.ndarray:
    matrix = np.asarray([design(row, include_horizon) for row in rows], dtype=float)
    outcome = np.log(
        np.asarray([float(row["total_parameters_b"]) for row in rows], dtype=float)
    )
    coefficients, _, rank, _ = np.linalg.lstsq(matrix, outcome, rcond=None)
    if rank != matrix.shape[1]:
        raise ValueError(
            f"Rank-deficient parameter mapping: rank {rank}/{matrix.shape[1]}"
        )
    return coefficients


def predict(row: dict[str, Any], coefficients: np.ndarray, include_horizon: bool) -> float:
    return math.exp(float(np.dot(design(row, include_horizon), coefficients)))


def metrics(errors: list[float]) -> dict[str, Any]:
    values = np.asarray(errors, dtype=float)
    return {
        "n": len(errors),
        "mean_absolute_ln_error": float(values.mean()),
        "geomean_multiplicative_error": math.exp(float(values.mean())),
        "median_multiplicative_error": math.exp(float(np.median(values))),
        "p80_multiplicative_error": math.exp(float(np.quantile(values, 0.8))),
        "within_2x": float(np.mean(values <= math.log(2))),
    }


def equal_developer_error(
    predictions: list[dict[str, Any]], field: str
) -> tuple[float, dict[str, float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in predictions:
        grouped[row["developer"]].append(float(row[field]))
    per_developer = {key: float(np.mean(value)) for key, value in grouped.items()}
    return float(np.mean(list(per_developer.values()))), per_developer


def bootstrap_delta(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[row["developer"]].append(row)
    developers = sorted(grouped)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty(BOOTSTRAP_SAMPLES, dtype=float)
    for index in range(BOOTSTRAP_SAMPLES):
        sampled_developers = rng.choice(developers, len(developers), replace=True)
        developer_deltas = []
        for developer in sampled_developers:
            rows = grouped[str(developer)]
            sampled_rows = rng.integers(0, len(rows), len(rows))
            developer_deltas.append(
                float(
                    np.mean(
                        [
                            rows[row_index]["horizon_absolute_ln_error"]
                            - rows[row_index]["baseline_absolute_ln_error"]
                            for row_index in sampled_rows
                        ]
                    )
                )
            )
        draws[index] = float(np.mean(developer_deltas))
    return {
        "metric": "equal-developer mean absolute ln error; horizon model minus date/architecture/reasoning baseline",
        "ci_90": [float(np.quantile(draws, 0.05)), float(np.quantile(draws, 0.95))],
        "bootstrap_probability_horizon_better": float(np.mean(draws < 0)),
        "samples": BOOTSTRAP_SAMPLES,
        "seed": BOOTSTRAP_SEED,
        "developers": len(developers),
    }


def elasticity_prediction(horizon: float, factor_per_doubling: float) -> float:
    alpha = math.log2(factor_per_doubling)
    return K3_TOTAL_T * (horizon / K3_PRETRAIN_PROXY_MINUTES) ** alpha


def control_row(
    record_type: str,
    series: str,
    model: str,
    release_date: str,
    parameters_b: float | str,
    horizon: float | str,
    ratio: float | str,
    evidence_grade: str,
    interpretation: str,
) -> dict[str, Any]:
    return {
        "record_type": record_type,
        "series": series,
        "model": model,
        "release_date": release_date,
        "total_parameters_b": parameters_b,
        "nocot_time_horizon_minutes": horizon,
        "horizon_ratio_vs_previous": ratio,
        "evidence_grade": evidence_grade,
        "interpretation": interpretation,
    }


def main() -> None:
    unified = read_csv(UNIFIED)
    open_rows = [
        row
        for row in unified
        if row["source"] == "No-CoT"
        and row["record_type"] == "model"
        and row["source_locator"].startswith("tab:open-source-models")
    ]
    if len(open_rows) != 35:
        raise ValueError(f"Expected 35 open-weight no-CoT rows, found {len(open_rows)}")
    if len({row["source_model_name"] for row in open_rows}) != len(open_rows):
        raise ValueError("Duplicate open-weight no-CoT model")

    evidence_rows = read_csv(EVIDENCE)
    evidence_by_id = {row["evidence_id"]: row for row in evidence_rows}
    if len(evidence_by_id) != 5:
        raise ValueError("Primary evidence inventory is incomplete or duplicated")
    sol_direct = float(evidence_by_id["openai_gpt_5_6_sol_nocot_horizon"]["value"])
    gpt_5_5_direct = float(evidence_by_id["openai_gpt_5_5_nocot_comparator"]["value"])
    if (sol_direct, gpt_5_5_direct) != (3.6, 2.3):
        raise ValueError("Official no-CoT measurements changed")

    predictions: list[dict[str, Any]] = []
    for test in sorted(open_rows, key=lambda row: (row["canonical_release_date"], row["source_model_name"])):
        train = [
            row
            for row in open_rows
            if row["canonical_release_date"] < test["canonical_release_date"]
            and row["source_organization"] != test["source_organization"]
        ]
        if len(train) < 10 or len({row["source_organization"] for row in train}) < 3:
            continue
        try:
            baseline_coefficients = fit(train, include_horizon=False)
            horizon_coefficients = fit(train, include_horizon=True)
        except ValueError:
            # A chronologically valid window may still contain no variation in
            # architecture or reasoning. It cannot identify the predeclared
            # specification and is therefore omitted rather than pseudo-fit.
            continue
        actual = float(test["total_parameters_b"])
        baseline = predict(test, baseline_coefficients, include_horizon=False)
        horizon = predict(test, horizon_coefficients, include_horizon=True)
        predictions.append(
            {
                "model": test["source_model_name"],
                "developer": test["source_organization"],
                "release_date": test["canonical_release_date"],
                "architecture": test["architecture"],
                "reasoning": test["reasoning"],
                "nocot_time_horizon_minutes": float(test["nocot_time_horizon_minutes"]),
                "actual_parameters_b": actual,
                "baseline_prediction_b": baseline,
                "horizon_prediction_b": horizon,
                "baseline_absolute_ln_error": abs(math.log(baseline / actual)),
                "horizon_absolute_ln_error": abs(math.log(horizon / actual)),
                "train_n": len(train),
                "train_developers": len({row["source_organization"] for row in train}),
                "same_developer_excluded": True,
                "strictly_earlier_training_rows": True,
            }
        )
    if len(predictions) != 16 or len({row["developer"] for row in predictions}) != 5:
        raise ValueError(
            "Held-out prediction inventory changed: "
            f"{len(predictions)} rows / "
            f"{len({row['developer'] for row in predictions})} developers"
        )

    baseline_equal, baseline_by_developer = equal_developer_error(
        predictions, "baseline_absolute_ln_error"
    )
    horizon_equal, horizon_by_developer = equal_developer_error(
        predictions, "horizon_absolute_ln_error"
    )
    delta_bootstrap = bootstrap_delta(predictions)
    delta_bootstrap["observed_delta"] = horizon_equal - baseline_equal

    target = {
        "canonical_release_date": SOL_RELEASE_DATE.isoformat(),
        "architecture": "MoE",
        "reasoning": "Hybrid-reasoning",
        "nocot_time_horizon_minutes": sol_direct,
    }
    baseline_coefficients = fit(open_rows, include_horizon=False)
    horizon_coefficients = fit(open_rows, include_horizon=True)
    target_baseline_b = predict(target, baseline_coefficients, include_horizon=False)
    target_horizon_b = predict(target, horizon_coefficients, include_horizon=True)

    exact_date = json.loads(EXACT_DATE_AUDIT.read_text(encoding="utf-8"))
    time_doubling_days = float(
        exact_date["time_horizon"]["adjusted_reported_law"]["adjusted_point_days"]
    )
    gpt_effective_date = (
        GPT_5_4_RELEASE_DATE.toordinal() + GPT_5_5_RELEASE_DATE.toordinal()
    ) / 2
    days_to_sol = SOL_RELEASE_DATE.toordinal() - gpt_effective_date
    current_projected_horizon = GPT_SHARED_HORIZON * 2 ** (days_to_sol / time_doubling_days)
    current_projected_prior = elasticity_prediction(
        current_projected_horizon, POOLED_TOTAL_FACTOR_PER_DOUBLING
    )
    direct_pooled_prior = elasticity_prediction(
        sol_direct, POOLED_TOTAL_FACTOR_PER_DOUBLING
    )
    direct_moe_prior = elasticity_prediction(sol_direct, MOE_TOTAL_FACTOR_PER_DOUBLING)
    rebased_horizon = 3.0 * sol_direct / gpt_5_5_direct
    rebased_pooled_prior = elasticity_prediction(
        rebased_horizon, POOLED_TOTAL_FACTOR_PER_DOUBLING
    )

    controls: list[dict[str, Any]] = []
    for prediction in predictions:
        controls.append(
            control_row(
                "chronological_developer_holdout",
                prediction["developer"],
                prediction["model"],
                prediction["release_date"],
                prediction["actual_parameters_b"],
                prediction["nocot_time_horizon_minutes"],
                prediction["horizon_prediction_b"] / prediction["actual_parameters_b"],
                "open_weight_ground_truth",
                "Ratio field is horizon-model predicted parameters divided by actual parameters.",
            )
        )

    open_by_name = {row["source_model_name"]: row for row in open_rows}
    same_size_series = {
        "Kimi K2 stable-size family": ["Kimi K2-0905", "Kimi K2.5", "Kimi K2.6"],
        "DeepSeek V3 stable-size family": [
            "DeepSeek V3 (0324)",
            "DeepSeek V3.1-terminus",
            "DeepSeek V3.2",
        ],
    }
    for series, names in same_size_series.items():
        previous = None
        parameters = {float(open_by_name[name]["total_parameters_b"]) for name in names}
        if len(parameters) != 1:
            raise ValueError(f"Same-size control changed for {series}")
        for name in names:
            row = open_by_name[name]
            horizon = float(row["nocot_time_horizon_minutes"])
            ratio = "" if previous is None else horizon / previous
            controls.append(
                control_row(
                    "same_size_family_control",
                    series,
                    name,
                    row["canonical_release_date"],
                    float(row["total_parameters_b"]),
                    horizon,
                    ratio,
                    "public_same_parameter_family_not_always_exact_same_weights",
                    "Shows within-family horizon movement while reported total size is unchanged.",
                )
            )
            previous = horizon

    lineage_edges = read_csv(LINEAGE_EDGES)
    kimi_edge = next(
        row
        for row in lineage_edges
        if row["parent_model"] == "Kimi K2.5" and row["child_model"] == "Kimi K2.6"
    )
    if kimi_edge["admission_status"] != "admitted_exact_open_same_parameter_lineage":
        raise ValueError("Kimi exact lineage control lost its admitted status")
    controls.append(
        control_row(
            "exact_open_lineage_control",
            "Kimi K2.5 → Kimi K2.6",
            "Kimi K2.6",
            open_by_name["Kimi K2.6"]["canonical_release_date"],
            float(kimi_edge["child_parameters_b"]),
            float(open_by_name["Kimi K2.6"]["nocot_time_horizon_minutes"]),
            float(open_by_name["Kimi K2.6"]["nocot_time_horizon_minutes"])
            / float(open_by_name["Kimi K2.5"]["nocot_time_horizon_minutes"]),
            "epoch_structured_base_link_exact_open_same_parameters",
            "A direct same-parameter lineage control: no-CoT horizon falls despite unchanged total size.",
        )
    )

    proprietary_controls = [
        ("GPT-5 shared-base assertion", "GPT-5.4", "2026-03-05", 1.4, ""),
        ("GPT-5 shared-base assertion", "GPT-5.5", "2026-04-23", 3.0, 3.0 / 1.4),
        ("Opus shared-base assertion", "Opus 4.5", "2025-11-24", 2.4, ""),
        ("Opus shared-base assertion", "Opus 4.6", "2026-02-05", 2.5, 2.5 / 2.4),
        ("Opus shared-base assertion", "Opus 4.7", "2026-04-16", 2.8, 2.8 / 2.5),
    ]
    for series, model, release, horizon, ratio in proprietary_controls:
        controls.append(
            control_row(
                "user_asserted_proprietary_shared_base_control",
                series,
                model,
                release,
                "",
                horizon,
                ratio,
                "user_asserted_not_publicly_disclosed",
                "Sensitivity control only; it is not treated as a verified parameter identity.",
            )
        )

    sensitivity_rows = [
        ("current_date_projection", current_projected_horizon, current_projected_prior),
        ("direct_model_level_regression", sol_direct, target_horizon_b / 1000),
        ("direct_pooled_paper_elasticity", sol_direct, direct_pooled_prior),
        ("direct_moe_paper_elasticity", sol_direct, direct_moe_prior),
        ("gpt_5_5_suite_rebased_pooled_elasticity", rebased_horizon, rebased_pooled_prior),
    ]
    for method, horizon, prior in sensitivity_rows:
        controls.append(
            control_row(
                "sol_parameter_mapping_sensitivity",
                method,
                "GPT-5.6 Sol",
                SOL_RELEASE_DATE.isoformat(),
                prior * 1000,
                horizon,
                "",
                "derived_sensitivity_not_parameter_disclosure",
                "Total-parameter estimate varies by mapping; method dispersion is the decision-relevant result.",
            )
        )

    CONTROLS.parent.mkdir(parents=True, exist_ok=True)
    with CONTROLS.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(controls[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(controls)

    method_estimates = [target_horizon_b / 1000, direct_pooled_prior, direct_moe_prior, rebased_pooled_prior]
    method_spread = max(method_estimates) / min(method_estimates)
    promotion_gates = {
        "official_direct_measurement_present": True,
        "chronological_developer_holdout_predictions_at_least_30": len(predictions) >= 30,
        "heldout_developers_at_least_8": len({row["developer"] for row in predictions}) >= 8,
        "horizon_increment_bootstrap_ci_wholly_favorable": delta_bootstrap["ci_90"][1] < 0,
        "same_suite_uncertainty_interval_reported": False,
        "parameter_mapping_method_spread_below_2x": method_spread < 2,
    }
    promote = all(promotion_gates.values())
    if promote:
        raise ValueError("Unexpected promotion: review gates before changing the live model")

    result = {
        "metadata": {
            "generated_on": DATE,
            "question": "Does the official GPT-5.6 Sol no-CoT result justify changing the live parameter forecast?",
            "target_parameter_identity": "Claude Fable 5 and Claude Mythos 5 are one shared underlying-weight target; Opus fallback is serving behavior, not another base.",
            "heldout_rule": "For each open model, train only on strictly earlier rows and exclude the entire test developer; require at least 10 rows and 3 other developers.",
        },
        "inventory": {
            "primary_evidence_rows": len(evidence_rows),
            "open_weight_no_cot_ground_truth_models": len(open_rows),
            "open_weight_developers": len({row["source_organization"] for row in open_rows}),
            "chronological_developer_holdout_predictions": len(predictions),
            "heldout_developers": len({row["developer"] for row in predictions}),
            "control_rows": len(controls),
        },
        "official_measurements": {
            "gpt_5_6_sol_nocot_minutes": sol_direct,
            "gpt_5_5_comparator_minutes": gpt_5_5_direct,
            "official_ratio_sol_over_gpt_5_5": sol_direct / gpt_5_5_direct,
            "paper_gpt_5_5_minutes": 3.0,
            "suite_rebased_sol_minutes": rebased_horizon,
        },
        "current_mapping": {
            "gpt_5_4_5_5_geometric_horizon_minutes": GPT_SHARED_HORIZON,
            "effective_date_to_sol_days": days_to_sol,
            "exact_date_adjusted_doubling_days": time_doubling_days,
            "projected_sol_horizon_minutes": current_projected_horizon,
            "projected_sol_horizon_prior_t": current_projected_prior,
        },
        "heldout_backtest": {
            "baseline_specification": "ln(total parameters) ~ exact date + MoE + reasoning",
            "horizon_specification": "ln(total parameters) ~ exact date + MoE + reasoning + ln(no-CoT horizon)",
            "baseline_all_rows": metrics(
                [row["baseline_absolute_ln_error"] for row in predictions]
            ),
            "horizon_all_rows": metrics(
                [row["horizon_absolute_ln_error"] for row in predictions]
            ),
            "baseline_equal_developer_mean_absolute_ln_error": baseline_equal,
            "horizon_equal_developer_mean_absolute_ln_error": horizon_equal,
            "baseline_per_developer_mean_absolute_ln_error": baseline_by_developer,
            "horizon_per_developer_mean_absolute_ln_error": horizon_by_developer,
            "incremental_bootstrap": delta_bootstrap,
        },
        "sol_mapping_sensitivity": {
            "date_architecture_reasoning_baseline_t": target_baseline_b / 1000,
            "direct_model_level_horizon_regression_t": target_horizon_b / 1000,
            "direct_pooled_paper_elasticity_t": direct_pooled_prior,
            "direct_moe_paper_elasticity_t": direct_moe_prior,
            "gpt_5_5_suite_rebased_pooled_elasticity_t": rebased_pooled_prior,
            "nonbaseline_method_max_over_min": method_spread,
            "interpretation": "The same official 3.6-minute point maps to radically different parameter counts under individually defensible model forms; mapping uncertainty dominates measurement precision.",
        },
        "k3_anchor": {
            "total_parameters_t": K3_TOTAL_T,
            "source": K3_PARAMETER_SOURCE,
        },
        "same_size_controls": {
            "exact_open_lineage": "Kimi K2.5 → Kimi K2.6 keeps 1.0T in the no-CoT table and 1.04T in Epoch while horizon falls from 1.102 to 0.508 minutes.",
            "stable_size_families": "Kimi K2 and DeepSeek V3 sequences have unchanged reported total parameters but materially moving horizons.",
            "proprietary_shared_base_status": "GPT-5 and Opus controls remain user-asserted, not publicly disclosed; they are sensitivity rows only.",
        },
        "promotion_gates": promotion_gates,
        "decision": {
            "preserve_primary_measurement_in_pipeline": True,
            "apply_fable_mythos_shared_weight_identity": True,
            "treat_opus_fallback_as_shared_base": False,
            "promote_direct_sol_horizon_increment": promote,
            "incremental_live_weight": 0.0,
            "change_headline_forecasts": False,
            "reason": "The direct measurement is real, but the fully identified horizon specification has only 16 chronological developer-held-out predictions from five held-out developers, its 90% developer-balanced bootstrap interval crosses zero, the source reports no interval, and plausible parameter mappings span more than 2x. It is retained as a zero-weight diagnostic until the mapping is validated.",
        },
        "limitations": [
            "The official system-card statement reports point estimates but no confidence interval or task-level observations.",
            "The official GPT-5.5 comparator and paper GPT-5.5 estimate are not numerically identical, so cross-suite rebasing is a sensitivity rather than a merge.",
            "The model-level ground-truth panel has six developers and only 35 models; developer-held-out coverage is too small for a stable frontier extrapolation.",
            "MoE-specific paper elasticity is based on a small, unstable Pareto frontier and is shown only as a sensitivity.",
            "Release date, reasoning label, and MoE status do not measure RL compute, data quality, or inference budget directly.",
        ],
        "outputs": {"controls": str(CONTROLS.relative_to(ROOT))},
        "source_hashes": {
            str(UNIFIED.relative_to(ROOT)): sha256(UNIFIED),
            str(EVIDENCE.relative_to(ROOT)): sha256(EVIDENCE),
            str(EVIDENCE_METADATA.relative_to(ROOT)): sha256(EVIDENCE_METADATA),
            str(EXACT_DATE_AUDIT.relative_to(ROOT)): sha256(EXACT_DATE_AUDIT),
            str(LINEAGE_EDGES.relative_to(ROOT)): sha256(LINEAGE_EDGES),
            str(CONTROLS.relative_to(ROOT)): sha256(CONTROLS),
            str(K3_EVIDENCE_PATH.relative_to(ROOT)): sha256(K3_EVIDENCE_PATH),
        },
    }
    RESULT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(RESULT),
                "heldout_predictions": len(predictions),
                "heldout_developers": len({row["developer"] for row in predictions}),
                "sol_model_level_t": target_horizon_b / 1000,
                "sol_pooled_elasticity_t": direct_pooled_prior,
                "method_spread": method_spread,
                "incremental_live_weight": 0.0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
