#!/usr/bin/env python3
"""Nested optimization audit for the live factor weights.

The component predictions are themselves strictly chronological, family-held-out
predictions produced by run_parameter_backtest.py.  This script adds a second,
meta-level split: each test checkpoint's factor weights are learned only from
earlier component-prediction rows, again excluding the test family.

Price is validated separately on the subset with audited OpenRouter coverage.
It is not jointly optimized here because that overlap is much smaller than the
37-row factor panel. IKP is evaluated separately on its exact-overlap panel
because it is a newly independent signal with different coverage. Crowd cannot
be optimized before Fable/Sol disclosure.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from artifact_paths import portable_path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs/019f6c42-2d53-7743-ab07-6293e2618dd7"
BACKTEST = OUT / "frontier_parameter_chronological_backtest_2026-07-17.json"
SITE_DATA = ROOT / "site/public/data/forecast-model.json"
OPENROUTER_INCREMENTAL = OUT / "openrouter_incremental_price_backtest_2026-07-18.json"
IKP_INCREMENTAL = OUT / "ikp_parameter_signal_audit_2026-07-18.json"
RESULT = OUT / "factor_weight_optimization_2026-07-18.json"
PREDICTIONS = OUT / "factor_weight_optimization_predictions_2026-07-18.csv"

FACTORS = ("AA", "ECI", "No-CoT", "Compute")
CURRENT_RAW = np.asarray([19.125, 19.125, 50.0, 5.0], dtype=float)
CURRENT = CURRENT_RAW / CURRENT_RAW.sum()
EQUAL = np.ones(len(FACTORS), dtype=float) / len(FACTORS)
GRID_STEP = 0.02
MIN_META_TRAIN = 12
MIN_META_FAMILIES = 5


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def simplex_grid() -> np.ndarray:
    units = round(1.0 / GRID_STEP)
    return np.asarray(
        [
            (a, b, c, units - a - b - c)
            for a in range(units + 1)
            for b in range(units + 1 - a)
            for c in range(units + 1 - a - b)
        ],
        dtype=float,
    ) / units


def matrices(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predictions = np.zeros((len(rows), len(FACTORS)), dtype=float)
    available = np.zeros_like(predictions)
    actual = np.zeros(len(rows), dtype=float)
    for row_index, row in enumerate(rows):
        actual[row_index] = math.log10(float(row["actual_b"]))
        for component in row["components"]:
            factor_index = FACTORS.index(component["panel"])
            predictions[row_index, factor_index] = math.log10(float(component["predicted_b"]))
            available[row_index, factor_index] = 1.0
    return predictions, available, actual


def candidate_errors(
    grid: np.ndarray,
    prediction: np.ndarray,
    available: np.ndarray,
    actual: np.ndarray,
    row_indices: list[int] | np.ndarray,
) -> np.ndarray:
    indices = np.asarray(row_indices, dtype=int)
    denominator = grid @ available[indices].T
    numerator = grid @ (prediction[indices] * available[indices]).T
    predicted = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > 0,
    )
    return predicted - actual[indices]


def optimize(
    grid: np.ndarray,
    prediction: np.ndarray,
    available: np.ndarray,
    actual: np.ndarray,
    row_indices: list[int] | np.ndarray,
    objective: str,
    required_availability: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    errors = candidate_errors(grid, prediction, available, actual, row_indices)
    valid = np.all(np.isfinite(errors), axis=1)
    if required_availability is not None:
        valid &= (grid @ required_availability) > 0
    if objective == "mse":
        loss = np.mean(errors**2, axis=1)
    elif objective == "mae":
        loss = np.mean(np.abs(errors), axis=1)
    elif objective == "median_absolute":
        loss = np.median(np.abs(errors), axis=1)
    elif objective == "p80_absolute":
        loss = np.quantile(np.abs(errors), 0.80, axis=1)
    else:
        raise ValueError(f"Unknown objective: {objective}")
    loss[~valid] = np.inf
    best = int(np.argmin(loss))
    if not np.isfinite(loss[best]):
        raise ValueError("No valid weight vector for the requested rows")
    return grid[best].copy(), float(loss[best])


def predict_one(
    weights: np.ndarray,
    prediction: np.ndarray,
    available: np.ndarray,
    row_index: int,
) -> float:
    denominator = float(weights @ available[row_index])
    if denominator <= 0:
        raise ValueError("Selected weights assign zero mass to every available test factor")
    return float(weights @ (prediction[row_index] * available[row_index]) / denominator)


def metrics(errors: list[float] | np.ndarray) -> dict[str, Any]:
    values = np.asarray(errors, dtype=float)
    absolute = np.abs(values)
    return {
        "n": int(len(values)),
        "median_multiplicative_error": float(10 ** np.median(absolute)),
        "geomean_multiplicative_error": float(10 ** np.mean(absolute)),
        "rmse_log10": float(np.sqrt(np.mean(values**2))),
        "p80_multiplicative_error": float(10 ** np.quantile(absolute, 0.80)),
        "within_2x": float(np.mean(absolute <= math.log10(2.0))),
        "signed_bias_factor": float(10 ** np.mean(values)),
    }


def weight_record(weights: np.ndarray) -> dict[str, float]:
    return {factor: float(weights[index]) for index, factor in enumerate(FACTORS)}


def family_bootstrap_weights(
    rows: list[dict[str, Any]],
    grid: np.ndarray,
    prediction: np.ndarray,
    available: np.ndarray,
    actual: np.ndarray,
    samples: int = 1000,
) -> dict[str, Any]:
    by_family: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_family[row["family"]].append(index)
    families = sorted(by_family)
    rng = np.random.default_rng(20260718)
    draws = []
    for _ in range(samples):
        selected = rng.choice(families, size=len(families), replace=True)
        indices = [index for family in selected for index in by_family[family]]
        weights, _ = optimize(grid, prediction, available, actual, indices, "mse")
        draws.append(weights)
    matrix = np.asarray(draws)
    return {
        "samples": samples,
        "family_clusters": len(families),
        "weight_quantiles_5_50_95": {
            factor: [float(value) for value in np.quantile(matrix[:, index], [0.05, 0.50, 0.95])]
            for index, factor in enumerate(FACTORS)
        },
        "probability_weight_is_zero": {
            factor: float(np.mean(matrix[:, index] == 0))
            for index, factor in enumerate(FACTORS)
        },
    }


def paired_family_bootstrap(
    prediction_rows: list[dict[str, Any]], samples: int = 10000
) -> dict[str, Any]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in prediction_rows:
        by_family[row["family"]].append(row)
    families = sorted(by_family)

    def delta(selected: list[str]) -> float:
        chosen = [row for family in selected for row in by_family[family]]
        return float(
            np.mean([abs(row["optimized_mse_log10_error"]) for row in chosen])
            - np.mean([abs(row["current_log10_error"]) for row in chosen])
        )

    observed = delta(families)
    rng = np.random.default_rng(20260718)
    draws = np.asarray(
        [delta(list(rng.choice(families, size=len(families), replace=True))) for _ in range(samples)]
    )
    return {
        "metric": "mean absolute log10 error, optimized MSE minus current weights",
        "observed_delta": observed,
        "ci_90": [float(value) for value in np.quantile(draws, [0.05, 0.95])],
        "bootstrap_probability_optimized_better": float(np.mean(draws < 0)),
        "samples": samples,
        "family_clusters": len(families),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    backtest = json.loads(BACKTEST.read_text(encoding="utf-8"))
    site = json.loads(SITE_DATA.read_text(encoding="utf-8"))
    price_validation = json.loads(OPENROUTER_INCREMENTAL.read_text(encoding="utf-8"))
    ikp_validation = json.loads(IKP_INCREMENTAL.read_text(encoding="utf-8"))
    ikp_overlap = ikp_validation["incremental_overlap"]
    ikp_decision = ikp_validation["decision"]
    ikp_promoted = bool(ikp_decision["promote_incremental_ikp_weight"])
    ikp_evidence_weight = float(ikp_decision["incremental_evidence_weight"])
    ikp_final_weight = float(
        ikp_decision["incremental_final_weight_when_crowd_is_50pct"]
    )
    if not ikp_promoted and (ikp_evidence_weight != 0 or ikp_final_weight != 0):
        raise ValueError("A non-promoted IKP signal must have zero live weight")
    if not math.isclose(
        float(site["defaultWeights"]["ikp"]),
        100 * ikp_final_weight,
        rel_tol=0,
        abs_tol=1e-12,
    ):
        raise ValueError("Site IKP weight disagrees with the current IKP decision")
    rows = backtest["ensemble_predictions"]
    prediction, available, actual = matrices(rows)
    grid = simplex_grid()
    all_indices = list(range(len(rows)))

    global_fits = {}
    for objective in ("mse", "mae", "median_absolute", "p80_absolute"):
        weights, loss = optimize(grid, prediction, available, actual, all_indices, objective)
        errors = [predict_one(weights, prediction, available, index) - actual[index] for index in all_indices]
        global_fits[objective] = {
            "weights": weight_record(weights),
            "objective_value": loss,
            "metrics": metrics(errors),
            "status": "diagnostic in-sample fit; not eligible for live weights",
        }

    nested_rows = []
    for test_index in sorted(all_indices, key=lambda index: (rows[index]["release_date"], rows[index]["model"])):
        test = rows[test_index]
        train_indices = [
            index
            for index, row in enumerate(rows)
            if row["release_date"] < test["release_date"] and row["family"] != test["family"]
        ]
        if len(train_indices) < MIN_META_TRAIN or len({rows[index]["family"] for index in train_indices}) < MIN_META_FAMILIES:
            continue
        mse_weights, _ = optimize(
            grid,
            prediction,
            available,
            actual,
            train_indices,
            "mse",
            required_availability=available[test_index],
        )
        mae_weights, _ = optimize(
            grid,
            prediction,
            available,
            actual,
            train_indices,
            "mae",
            required_availability=available[test_index],
        )
        optimized_mse = predict_one(mse_weights, prediction, available, test_index)
        optimized_mae = predict_one(mae_weights, prediction, available, test_index)
        current = predict_one(CURRENT, prediction, available, test_index)
        equal = predict_one(EQUAL, prediction, available, test_index)
        nested_rows.append(
            {
                "release_date": test["release_date"],
                "model": test["model"],
                "family": test["family"],
                "actual_b": float(test["actual_b"]),
                "available_factors": "|".join(component["panel"] for component in test["components"]),
                "meta_train_n": len(train_indices),
                "meta_train_family_n": len({rows[index]["family"] for index in train_indices}),
                "meta_train_max_date": max(rows[index]["release_date"] for index in train_indices),
                "optimized_mse_predicted_b": float(10**optimized_mse),
                "optimized_mse_log10_error": float(optimized_mse - actual[test_index]),
                "optimized_mae_predicted_b": float(10**optimized_mae),
                "optimized_mae_log10_error": float(optimized_mae - actual[test_index]),
                "current_predicted_b": float(10**current),
                "current_log10_error": float(current - actual[test_index]),
                "equal_predicted_b": float(10**equal),
                "equal_log10_error": float(equal - actual[test_index]),
                **{f"optimized_mse_weight_{factor}": float(mse_weights[index]) for index, factor in enumerate(FACTORS)},
                **{f"optimized_mae_weight_{factor}": float(mae_weights[index]) for index, factor in enumerate(FACTORS)},
            }
        )

    coverage = {factor: int(np.sum(available[:, index])) for index, factor in enumerate(FACTORS)}
    patterns = Counter(
        "|".join(sorted(component["panel"] for component in row["components"])) for row in rows
    )
    nested_metrics = {
        "optimized_mse": metrics([row["optimized_mse_log10_error"] for row in nested_rows]),
        "optimized_mae": metrics([row["optimized_mae_log10_error"] for row in nested_rows]),
        "current_weights": metrics([row["current_log10_error"] for row in nested_rows]),
        "equal_weights": metrics([row["equal_log10_error"] for row in nested_rows]),
    }
    bootstrap_weights = family_bootstrap_weights(rows, grid, prediction, available, actual)
    paired_bootstrap = paired_family_bootstrap(nested_rows)

    result = {
        "metadata": {
            "generated_on": "2026-07-18",
            "target": "log10 disclosed total parameters in billions",
            "base_predictions": "strictly-earlier chronological family-held-out factor predictions",
            "meta_split": "weights learned only from earlier base-prediction rows with the test family removed",
            "grid": f"non-negative simplex in {GRID_STEP:.2f} increments",
            "missing_factor_policy": "renormalize weights over factors available for that checkpoint",
            "nested_minimums": {"rows": MIN_META_TRAIN, "families": MIN_META_FAMILIES},
        },
        "live_final_weights_percent": site["defaultWeights"],
        "jointly_optimizable_core_weights_normalized": weight_record(CURRENT),
        "not_optimizable": {
            "API price": f"Validated separately on {price_validation['coverage']['strictly_chronological_paired_models']} strictly chronological checkpoints from {price_validation['coverage']['developer_families']} developers. The current 6.75% within-evidence price weight improves held-out error, but the overlap is too small and frontier prices are outside calibration support, so price is not jointly optimized on this factor panel.",
            "IKP factual capacity": f"Evaluated separately on {ikp_overlap['models']} exact model matches from {ikp_overlap['families']} families, including a {ikp_overlap['chronological_fixed_weight_subset']['models']}-model/{ikp_overlap['chronological_fixed_weight_subset']['families']}-family later chronological subset. The fixed 10% diagnostic blend has favorable bootstrap results, but promotion requires at least {ikp_decision['evidence_gates']['minimum_chronological_subset_models']} chronological models. Its live within-evidence weight is therefore {100 * ikp_evidence_weight:.0f}%; its different coverage also prevents joint optimization on the {len(rows)}-row core panel.",
            "Human crowd": "Fable and Sol parameter outcomes remain undisclosed, and the forecasts were collected after benchmark outcomes were visible.",
        },
        "separate_api_price_validation": {
            "coverage": price_validation["coverage"],
            "evidence_only_metrics": price_validation["fixed_weight_metrics"]["0"],
            "current_price_weight_metrics": price_validation["fixed_weight_metrics"]["0.0675"],
            "paired_developer_family_bootstrap": price_validation[
                "paired_developer_family_bootstraps"
            ]["0.0675"],
            "decision": price_validation["decision"],
        },
        "separate_ikp_validation": {
            "coverage": {
                "models": ikp_validation["incremental_overlap"]["models"],
                "families": ikp_validation["incremental_overlap"]["families"],
                "chronological_models": ikp_validation["incremental_overlap"]["chronological_fixed_weight_subset"]["models"],
                "chronological_families": ikp_validation["incremental_overlap"]["chronological_fixed_weight_subset"]["families"],
            },
            "existing_metrics": ikp_validation["incremental_overlap"]["existing"],
            "ikp_metrics": ikp_validation["incremental_overlap"]["ikp"],
            "fixed_10pct_blend_metrics": ikp_validation["incremental_overlap"]["blend_10pct"],
            "full_overlap_family_bootstrap": ikp_validation["incremental_overlap"]["family_bootstrap"],
            "chronological_family_bootstrap": ikp_validation["incremental_overlap"]["chronological_fixed_weight_subset"]["family_bootstrap"],
            "decision": ikp_validation["decision"],
        },
        "coverage": {
            "matched_checkpoints": len(rows),
            "families": len({row["family"] for row in rows}),
            "rows_by_factor": coverage,
            "availability_patterns": dict(sorted(patterns.items())),
        },
        "global_in_sample_optimization": global_fits,
        "global_mse_weight_family_bootstrap": bootstrap_weights,
        "nested_outer_evaluation": {
            "eligible_predictions": len(nested_rows),
            "metrics": nested_metrics,
            "paired_family_bootstrap": paired_bootstrap,
        },
        "decision": {
            "update_live_weights": False,
            "reason": f"Direct optimization of the four common core factors fails to improve genuinely held-out checkpoints and is extremely unstable across loss functions and family bootstrap samples. Price remains separately fixed; IKP's current audit decision assigns {100 * ikp_evidence_weight:.0f}% live evidence weight because it does not pass every promotion gate.",
            "next_data_needed": "Disclosed outcomes with overlapping AA, ECI, no-CoT, compute, and comparable provider-price coverage; eventual Fable/Sol disclosures for crowd calibration.",
        },
        "source_files": {
            portable_path(BACKTEST): sha256(BACKTEST),
            portable_path(SITE_DATA): sha256(SITE_DATA),
            portable_path(OPENROUTER_INCREMENTAL): sha256(OPENROUTER_INCREMENTAL),
            portable_path(IKP_INCREMENTAL): sha256(IKP_INCREMENTAL),
        },
    }
    RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_csv(PREDICTIONS, nested_rows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
