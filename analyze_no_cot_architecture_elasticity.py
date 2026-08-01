#!/usr/bin/env python3
"""Audit whether the no-CoT parameter mapping needs a MoE-specific slope.

The no-CoT paper reports descriptive Pareto-frontier relationships: a 4.2x
increase in total parameters per horizon doubling in the pooled panel, 2.2x
for dense models, and 8.1x for MoEs.  Those coefficients are not automatically
valid as inverse parameter estimators.  This audit therefore does two things:

1. reproduce the paper's architecture result from the source-table bootstrap
   medians; and
2. test pooled and architecture-specific parameter mappings without allowing a
   held-out checkpoint or its developer into model fitting.

The primary split is strictly chronological and developer-held-out.  A
leave-one-developer-out analysis is retained as a higher-power secondary check.
All dates come from the audited day-precision canonical registry.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parent
THREAD = "019f6c42-2d53-7743-ab07-6293e2618dd7"
OUTPUT = ROOT / "outputs" / THREAD
UNIFIED = OUTPUT / "unified_model_observations_compute_enriched_2026-07-17.csv"
RESULT = OUTPUT / "no_cot_architecture_elasticity_audit_2026-07-18.json"
PREDICTIONS = OUTPUT / "no_cot_architecture_elasticity_predictions_2026-07-18.csv"

ORIGIN = date(2024, 1, 1)
PAPER_FACTORS = {"pooled": 4.2, "dense": 2.2, "moe": 8.1}
EXPECTED_MODELS = 35
EXPECTED_FAMILIES = 6
EXPECTED_DENSE = 19
EXPECTED_MOE = 16
MIN_TRAIN = 10
MIN_FAMILIES = 3
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20_260_718


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def family_for(model: str) -> str:
    lowered = model.lower()
    if "ministral" in lowered:
        return "mistral"
    for family in ("llama", "mistral", "qwen", "gemma", "deepseek", "kimi"):
        if family in lowered:
            return family
    raise ValueError(f"Unmapped no-CoT model family: {model}")


def load_panel() -> list[dict[str, Any]]:
    with UNIFIED.open(newline="", encoding="utf-8") as handle:
        source = list(csv.DictReader(handle))
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in source:
        checkpoint = row["canonical_checkpoint_id"]
        if (
            not row["nocot_time_horizon_minutes"]
            or not row["nocot_time_horizon_median_minutes"]
            or not row["total_parameters_b"]
            or checkpoint in seen
            or row["canonical_display_name"].startswith("GPT-4")
        ):
            continue
        seen.add(checkpoint)
        architecture = row["architecture"]
        if architecture not in {"Dense", "MoE"}:
            raise ValueError(
                f"Missing architecture for {row['canonical_display_name']}: {architecture!r}"
            )
        release = row["canonical_release_date"]
        if len(release) != 10:
            raise ValueError(f"Non-day-level date for {row['canonical_display_name']}: {release}")
        canonical_total_b = float(row["total_parameters_b"])
        raw_total_b = float(
            row.get("raw_total_parameters_b") or row["total_parameters_b"]
        )
        rows.append(
            {
                "checkpoint_id": checkpoint,
                "model": row["canonical_display_name"],
                "family": family_for(row["canonical_display_name"]),
                "release_date": release,
                "date_years": (date.fromisoformat(release) - ORIGIN).days / 365.25,
                # Predictive fits use the reconciled parameter truth.  The raw
                # source label is retained separately so the published-paper
                # relationship can still be reproduced byte-for-byte after a
                # more precise first-party parameter disclosure arrives.
                "total_b": canonical_total_b,
                "log_total": math.log(canonical_total_b),
                "raw_total_b": raw_total_b,
                "raw_log_total": math.log(raw_total_b),
                "parameter_truth_id": row.get("parameter_truth_id") or None,
                "point_horizon": float(row["nocot_time_horizon_minutes"]),
                "median_horizon": float(row["nocot_time_horizon_median_minutes"]),
                "log_point_horizon": math.log(
                    float(row["nocot_time_horizon_minutes"])
                ),
                "log_median_horizon": math.log(
                    float(row["nocot_time_horizon_median_minutes"])
                ),
                "architecture": architecture,
                "moe": int(architecture == "MoE"),
                "reasoning": int(row["reasoning"] != "Non-reasoning"),
            }
        )
    rows.sort(key=lambda row: (row["release_date"], row["model"]))
    architecture_counts = Counter(row["architecture"] for row in rows)
    if len(rows) != EXPECTED_MODELS:
        raise ValueError(f"Expected {EXPECTED_MODELS} open models, found {len(rows)}")
    if len({row["family"] for row in rows}) != EXPECTED_FAMILIES:
        raise ValueError("No-CoT developer-family count changed")
    if architecture_counts != {"Dense": EXPECTED_DENSE, "MoE": EXPECTED_MOE}:
        raise ValueError(f"Architecture counts changed: {architecture_counts}")
    return rows


def pareto_frontier(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return points not dominated on low parameters / high median horizon."""
    frontier = []
    for row in rows:
        dominated = any(
            candidate["log_total"] <= row["log_total"]
            and candidate["log_median_horizon"] >= row["log_median_horizon"]
            and (
                candidate["log_total"] < row["log_total"]
                or candidate["log_median_horizon"] > row["log_median_horizon"]
            )
            for candidate in rows
        )
        if not dominated:
            frontier.append(row)
    return sorted(frontier, key=lambda row: (row["total_b"], row["model"]))


def pareto_parameter_slope(rows: list[dict[str, Any]]) -> tuple[float, list[str]]:
    """Fit log(horizon) ~ log(parameters), then invert for parameter prediction."""
    frontier = pareto_frontier(rows)
    if len(frontier) < 3:
        raise ValueError(f"Only {len(frontier)} Pareto points")
    matrix = np.asarray(
        [[1.0, row["log_total"]] for row in frontier], dtype=float
    )
    outcome = np.asarray(
        [row["log_median_horizon"] for row in frontier], dtype=float
    )
    coefficients, _, rank, _ = np.linalg.lstsq(matrix, outcome, rcond=None)
    if rank != 2 or coefficients[1] <= 0:
        raise ValueError("Invalid Pareto slope")
    return float(1 / coefficients[1]), [row["model"] for row in frontier]


def family_weights(rows: list[dict[str, Any]]) -> np.ndarray:
    counts = Counter(row["family"] for row in rows)
    weights = np.asarray([1 / counts[row["family"]] for row in rows], dtype=float)
    return weights / weights.mean()


def design(rows: list[dict[str, Any]], specification: str) -> np.ndarray:
    matrix = []
    for row in rows:
        values = [1.0, row["date_years"], row["moe"], row["reasoning"]]
        if specification in {"direct_pooled", "direct_architecture"}:
            values.append(row["log_point_horizon"])
        if specification == "direct_architecture":
            values.append(row["log_point_horizon"] * row["moe"])
        matrix.append(values)
    return np.asarray(matrix, dtype=float)


def weighted_fit(
    train: list[dict[str, Any]], outcome: np.ndarray, specification: str
) -> np.ndarray:
    matrix = design(train, specification)
    sqrt_weights = np.sqrt(family_weights(train))
    coefficients, _, rank, _ = np.linalg.lstsq(
        matrix * sqrt_weights[:, None], outcome * sqrt_weights, rcond=None
    )
    if rank != matrix.shape[1]:
        raise ValueError(
            f"Rank-deficient {specification}: {rank}/{matrix.shape[1]}"
        )
    return coefficients


def direct_prediction(
    train: list[dict[str, Any]], test: dict[str, Any], architecture_slope: bool
) -> tuple[float, dict[str, Any]]:
    specification = "direct_architecture" if architecture_slope else "direct_pooled"
    outcome = np.asarray([row["log_total"] for row in train], dtype=float)
    coefficients = weighted_fit(train, outcome, specification)
    prediction = float((design([test], specification) @ coefficients).item())
    pooled_slope = float(coefficients[4])
    moe_slope = pooled_slope + (float(coefficients[5]) if architecture_slope else 0.0)
    return prediction, {
        "dense_log_parameter_per_log_horizon": pooled_slope,
        "moe_log_parameter_per_log_horizon": moe_slope,
    }


def training_pareto_prediction(
    train: list[dict[str, Any]], test: dict[str, Any], architecture_slope: bool
) -> tuple[float, dict[str, Any]]:
    if architecture_slope:
        dense_slope, dense_frontier = pareto_parameter_slope(
            [row for row in train if not row["moe"]]
        )
        moe_slope, moe_frontier = pareto_parameter_slope(
            [row for row in train if row["moe"]]
        )
    else:
        dense_slope, pooled_frontier = pareto_parameter_slope(train)
        moe_slope = dense_slope
        dense_frontier = pooled_frontier
        moe_frontier = pooled_frontier
    slopes = {0: dense_slope, 1: moe_slope}
    residual = np.asarray(
        [
            row["log_total"]
            - slopes[row["moe"]] * row["log_median_horizon"]
            for row in train
        ],
        dtype=float,
    )
    coefficients = weighted_fit(train, residual, "residual_base")
    base = float((design([test], "residual_base") @ coefficients).item())
    prediction = base + slopes[test["moe"]] * test["log_median_horizon"]
    return prediction, {
        "dense_log_parameter_per_log_horizon": dense_slope,
        "moe_log_parameter_per_log_horizon": moe_slope,
        "dense_factor_per_horizon_doubling": float(2**dense_slope),
        "moe_factor_per_horizon_doubling": float(2**moe_slope),
        "dense_frontier_n": len(dense_frontier),
        "moe_frontier_n": len(moe_frontier),
    }


PREDICTORS: dict[
    str,
    Callable[[list[dict[str, Any]], dict[str, Any]], tuple[float, dict[str, Any]]],
] = {
    "direct_pooled": lambda train, test: direct_prediction(train, test, False),
    "direct_architecture": lambda train, test: direct_prediction(train, test, True),
    "training_pareto_pooled": lambda train, test: training_pareto_prediction(
        train, test, False
    ),
    "training_pareto_architecture": lambda train, test: training_pareto_prediction(
        train, test, True
    ),
}


def backtest(
    rows: list[dict[str, Any]], split: str, specification: str
) -> list[dict[str, Any]]:
    predictions = []
    for test in rows:
        train = [row for row in rows if row["family"] != test["family"]]
        if split == "chronological_developer_holdout":
            train = [row for row in train if row["release_date"] < test["release_date"]]
        if len(train) < MIN_TRAIN or len({row["family"] for row in train}) < MIN_FAMILIES:
            continue
        try:
            prediction_log, diagnostics = PREDICTORS[specification](train, test)
        except ValueError:
            continue
        error = prediction_log - test["log_total"]
        comparison_pool = [
            row
            for row in rows
            if split != "chronological_developer_holdout"
            or row["release_date"] < test["release_date"]
        ]
        horizon_rank = sum(
            row["log_point_horizon"] <= test["log_point_horizon"]
            for row in comparison_pool
        ) / len(comparison_pool)
        predictions.append(
            {
                "split": split,
                "specification": specification,
                "release_date": test["release_date"],
                "model": test["model"],
                "family": test["family"],
                "architecture": test["architecture"],
                "actual_parameters_b": test["total_b"],
                "predicted_parameters_b": math.exp(prediction_log),
                "log_error": error,
                "absolute_log_error": abs(error),
                "multiplicative_error": math.exp(abs(error)),
                "train_n": len(train),
                "train_families": len({row["family"] for row in train}),
                "train_max_date": max(row["release_date"] for row in train),
                "test_family_excluded": not any(
                    row["family"] == test["family"] for row in train
                ),
                "strictly_earlier": all(
                    row["release_date"] < test["release_date"] for row in train
                ),
                "frontier_like": horizon_rank >= 0.75,
                **diagnostics,
            }
        )
    return predictions


def metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    errors = np.asarray([row["log_error"] for row in predictions], dtype=float)
    absolute = np.abs(errors)
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in predictions:
        grouped[row["family"]].append(row["absolute_log_error"])
    equal_family = float(np.mean([np.mean(values) for values in grouped.values()]))
    return {
        "n": len(predictions),
        "families": len(grouped),
        "median_multiplicative_error": float(math.exp(np.median(absolute))),
        "geomean_multiplicative_error": float(math.exp(np.mean(absolute))),
        "equal_family_geomean_multiplicative_error": float(math.exp(equal_family)),
        "rmse_log": float(np.sqrt(np.mean(errors**2))),
        "within_2x": float(np.mean(absolute <= math.log(2))),
    }


def matched(
    left: list[dict[str, Any]], right: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    left_index = {row["model"]: row for row in left}
    right_index = {row["model"]: row for row in right}
    common = sorted(left_index.keys() & right_index.keys())
    return [left_index[key] for key in common], [right_index[key] for key in common]


def paired_family_bootstrap(
    baseline: list[dict[str, Any]], candidate: list[dict[str, Any]], seed: int
) -> dict[str, Any]:
    baseline, candidate = matched(baseline, candidate)
    grouped: dict[str, list[float]] = defaultdict(list)
    for base, test in zip(baseline, candidate, strict=True):
        if base["model"] != test["model"] or base["family"] != test["family"]:
            raise ValueError("Paired prediction alignment failed")
        grouped[base["family"]].append(
            test["absolute_log_error"] - base["absolute_log_error"]
        )
    families = sorted(grouped)
    observed = float(np.mean([np.mean(grouped[family]) for family in families]))
    rng = np.random.default_rng(seed)
    draws = np.empty(BOOTSTRAP_SAMPLES, dtype=float)
    for draw in range(BOOTSTRAP_SAMPLES):
        sampled_families = rng.choice(families, len(families), replace=True)
        family_means = []
        for family in sampled_families:
            values = grouped[str(family)]
            sampled_rows = rng.choice(values, len(values), replace=True)
            family_means.append(float(np.mean(sampled_rows)))
        draws[draw] = float(np.mean(family_means))
    return {
        "metric": "equal-family mean absolute natural-log error; architecture-specific minus pooled",
        "n": len(baseline),
        "families": len(families),
        "observed_delta": observed,
        "ci_90": [float(np.quantile(draws, 0.05)), float(np.quantile(draws, 0.95))],
        "probability_architecture_specific_better": float(np.mean(draws < 0)),
        "samples": BOOTSTRAP_SAMPLES,
        "seed": seed,
        "per_family_delta": {
            family: float(np.mean(values)) for family, values in sorted(grouped.items())
        },
    }


def same_parameter_controls(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["family"], row["total_b"], row["architecture"])].append(row)
    controls = []
    for (family, total_b, architecture), members in sorted(grouped.items()):
        if len(members) < 2:
            continue
        controls.append(
            {
                "family": family,
                "architecture": architecture,
                "total_parameters_b": total_b,
                "models": [row["model"] for row in members],
                "horizon_min": min(row["point_horizon"] for row in members),
                "horizon_max": max(row["point_horizon"] for row in members),
                "horizon_max_over_min": max(
                    row["point_horizon"] for row in members
                )
                / min(row["point_horizon"] for row in members),
            }
        )
    return controls


def main() -> None:
    rows = load_panel()
    raw_paper_rows = [
        {
            **row,
            "total_b": row["raw_total_b"],
            "log_total": row["raw_log_total"],
        }
        for row in rows
    ]
    slope_groups = {
        "pooled": raw_paper_rows,
        "dense": [row for row in raw_paper_rows if not row["moe"]],
        "moe": [row for row in raw_paper_rows if row["moe"]],
    }
    reproduction = {}
    for label, group in slope_groups.items():
        slope, frontier = pareto_parameter_slope(group)
        reproduction[label] = {
            "paper_reported_factor_per_horizon_doubling": PAPER_FACTORS[label],
            "deterministic_bootstrap_median_reproduction": float(2**slope),
            "log_parameter_per_log_horizon": slope,
            "pareto_models": frontier,
            "pareto_n": len(frontier),
            "parameter_value_basis": "raw_no_cot_source_label",
        }

    canonical_relationship = {}
    for label, group in {
        "pooled": rows,
        "dense": [row for row in rows if not row["moe"]],
        "moe": [row for row in rows if row["moe"]],
    }.items():
        slope, frontier = pareto_parameter_slope(group)
        canonical_relationship[label] = {
            "factor_per_horizon_doubling": float(2**slope),
            "log_parameter_per_log_horizon": slope,
            "pareto_models": frontier,
            "pareto_n": len(frontier),
            "parameter_value_basis": "canonical_parameter_truth_overlay",
        }

    all_predictions: dict[tuple[str, str], list[dict[str, Any]]] = {}
    splits = ["chronological_developer_holdout", "leave_one_developer_out"]
    for split in splits:
        for specification in PREDICTORS:
            all_predictions[(split, specification)] = backtest(
                rows, split, specification
            )

    comparisons = {}
    for index, split in enumerate(splits):
        comparisons[split] = {
            "direct_architecture_minus_pooled": paired_family_bootstrap(
                all_predictions[(split, "direct_pooled")],
                all_predictions[(split, "direct_architecture")],
                BOOTSTRAP_SEED + index,
            ),
            "training_pareto_architecture_minus_pooled": paired_family_bootstrap(
                all_predictions[(split, "training_pareto_pooled")],
                all_predictions[(split, "training_pareto_architecture")],
                BOOTSTRAP_SEED + 10 + index,
            ),
        }

    model_metrics = {
        split: {
            specification: metrics(all_predictions[(split, specification)])
            for specification in PREDICTORS
        }
        for split in splits
    }
    frontier_metrics = {
        split: {
            specification: metrics(
                [
                    row
                    for row in all_predictions[(split, specification)]
                    if row["frontier_like"]
                ]
            )
            for specification in PREDICTORS
        }
        for split in splits
    }

    controls = same_parameter_controls(rows)
    primary_direct = comparisons["chronological_developer_holdout"][
        "direct_architecture_minus_pooled"
    ]
    primary_pareto = comparisons["chronological_developer_holdout"][
        "training_pareto_architecture_minus_pooled"
    ]
    promote = (
        primary_direct["ci_90"][1] < 0
        and primary_pareto["ci_90"][1] < 0
        and comparisons["leave_one_developer_out"][
            "direct_architecture_minus_pooled"
        ]["observed_delta"]
        < 0
    )
    if promote:
        raise ValueError("Unexpected promotion; review before changing the live elasticity")

    prediction_rows = [
        row
        for split in splits
        for specification in PREDICTORS
        for row in all_predictions[(split, specification)]
    ]
    PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "split",
        "specification",
        "release_date",
        "model",
        "family",
        "architecture",
        "actual_parameters_b",
        "predicted_parameters_b",
        "log_error",
        "absolute_log_error",
        "multiplicative_error",
        "train_n",
        "train_families",
        "train_max_date",
        "test_family_excluded",
        "strictly_earlier",
        "frontier_like",
        "dense_log_parameter_per_log_horizon",
        "moe_log_parameter_per_log_horizon",
        "dense_factor_per_horizon_doubling",
        "moe_factor_per_horizon_doubling",
        "dense_frontier_n",
        "moe_frontier_n",
    ]
    with PREDICTIONS.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(prediction_rows)

    result = {
        "metadata": {
            "generated_on": "2026-07-18",
            "primary_split": "strictly earlier release date and entire test developer excluded",
            "training_weighting": "inverse checkpoint count per developer family",
            "paper_method": "OLS of log horizon on log parameters along the per-axis Pareto frontier; deterministic reproduction uses reported per-model bootstrap medians and the raw parameter labels present in the no-CoT source",
            "predictive_parameter_basis": "canonical parameter-truth overlay; raw source labels remain preserved for paper reproduction",
            "prediction_question": "Does a separate MoE horizon slope improve held-out recovery of total parameter count?",
        },
        "inventory": {
            "models": len(rows),
            "families": len({row["family"] for row in rows}),
            "dense_models": sum(not row["moe"] for row in rows),
            "moe_models": sum(row["moe"] for row in rows),
            "day_precision_dates": sum(len(row["release_date"]) == 10 for row in rows),
            "same_parameter_control_groups": len(controls),
        },
        "paper_relationship_reproduction": reproduction,
        "canonical_parameter_relationship_sensitivity": canonical_relationship,
        "parameter_truth_overlays_in_panel": [
            {
                "model": row["model"],
                "raw_total_parameters_b": row["raw_total_b"],
                "canonical_total_parameters_b": row["total_b"],
                "parameter_truth_id": row["parameter_truth_id"],
            }
            for row in rows
            if not math.isclose(row["raw_total_b"], row["total_b"], rel_tol=0, abs_tol=1e-12)
        ],
        "heldout_metrics": model_metrics,
        "frontier_like_metrics": frontier_metrics,
        "paired_comparisons": comparisons,
        "same_parameter_controls": controls,
        "promotion_gates": {
            "chronological_direct_architecture_ci_wholly_favorable": primary_direct[
                "ci_90"
            ][1]
            < 0,
            "chronological_training_pareto_architecture_ci_wholly_favorable": primary_pareto[
                "ci_90"
            ][1]
            < 0,
            "secondary_direct_architecture_direction_favorable": comparisons[
                "leave_one_developer_out"
            ]["direct_architecture_minus_pooled"]["observed_delta"]
            < 0,
        },
        "decision": {
            "replace_pooled_live_elasticity_with_moe_specific": promote,
            "retain_paper_moe_factor_as_sensitivity": True,
            "change_live_horizon_weight": False,
            "change_headline_forecasts": False,
            "reason": "The 8.1x MoE Pareto relationship is exactly reproducible as a descriptive scaling result, but a separately learned architecture slope worsens held-out parameter recovery. Training-fold-only Pareto slopes have a favorable chronological point estimate but their family-bootstrap interval crosses zero, so the evidence does not support replacing the pooled live mapping.",
        },
        "limitations": [
            "The panel has only six developers and the full MoE Pareto frontier has five models.",
            "Architecture labels do not directly measure sparsity, routing, data quality, post-training compute, or inference budget.",
            "The secondary leave-one-developer-out split uses future models and is not a real-time forecast simulation.",
            "Same-parameter controls show that horizon can move materially with unchanged reported total size, so any elasticity is a noisy capability-to-parameter transport rather than an identity.",
            "The paper-reproduction block intentionally retains the paper's raw rounded labels, while every predictive fit and same-parameter control uses the reconciled canonical total; their small difference is reported as a sensitivity rather than silently changing the published result.",
        ],
        "outputs": {"predictions": str(PREDICTIONS.relative_to(ROOT))},
        "source_hashes": {
            str(UNIFIED.relative_to(ROOT)): sha256(UNIFIED),
            str(PREDICTIONS.relative_to(ROOT)): sha256(PREDICTIONS),
        },
    }
    RESULT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(RESULT),
                "models": len(rows),
                "moe_pareto_reproduction": reproduction["moe"][
                    "deterministic_bootstrap_median_reproduction"
                ],
                "primary_direct_comparison": primary_direct,
                "primary_pareto_comparison": primary_pareto,
                "change_headline_forecasts": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
