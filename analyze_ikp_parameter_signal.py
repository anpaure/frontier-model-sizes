#!/usr/bin/env python3
"""Audit Incompressible Knowledge Probes as an incremental size signal.

The published IKP leave-one-model-out result leaks vendor/family information
and counts thinking variants of the same weights as separate observations.  We
therefore reproduce the source first, collapse obvious thinking duplicates,
then run strictly earlier, vendor-held-out predictions.  Promotion is decided
only on the exact overlap with the existing chronological parameter ensemble.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs/019f6c42-2d53-7743-ab07-6293e2618dd7"
CALIBRATION = ROOT / "sources/ikp_upstream_calibration_2026-07-18.json"
SUMMARY = ROOT / "sources/ikp_upstream_evaluation_summary_2026-07-18.json"
CONFIG = ROOT / "sources/ikp_upstream_models_2026-07-18.json"
SENSITIVITY = ROOT / "sources/ikp_upstream_sensitivity_2026-07-18.json"
V2_VALIDATION = ROOT / "sources/ikp_upstream_v2_validation_2026-07-18.json"
REPLICATION = ROOT / "sources/ikp_replication_calibration_errors_2026-07-18.csv"
REPLICATION_WIKIDATA = ROOT / "sources/ikp_replication_wikidata_dominance_2026-07-18.json"
SOURCE_METADATA = ROOT / "sources/ikp_source_metadata_2026-07-18.json"
BACKTEST = OUT / "frontier_parameter_chronological_backtest_2026-07-17.json"

RESULT = OUT / "ikp_parameter_signal_audit_2026-07-18.json"
PREDICTIONS = OUT / "ikp_parameter_chronological_predictions_2026-07-18.csv"
OVERLAP = OUT / "ikp_parameter_incremental_overlap_2026-07-18.csv"
SITE_OUTPUT = ROOT / "site/public/data/ikp-parameter-signal.json"

MIN_TRAIN_ROWS = 20
MIN_TRAIN_VENDORS = 6
BOOTSTRAP_SAMPLES = 20_000
RANDOM_SEED = 20260718
PRIMARY_COLLAPSE = "mean"
PRIMARY_FORM = "forward_inverse"
TEST_WEIGHT = 0.10
META_WEIGHT_GRID = tuple(index / 20 for index in range(21))
META_MIN_PRIOR_ROWS = 5
META_MIN_PRIOR_FAMILIES = 5

# Exact checkpoint adjudications.  The IKP side is a base key after removing
# only the explicit ``-think`` serving variant; no fuzzy name matching is used.
BACKTEST_TO_IKP = {
    "deepseekv32": "deepseek-v3.2",
    "deepseekv4flash": "deepseek-v4-flash",
    "deepseekv4pro": "deepseek-v4-pro",
    "gemma227b": "gemma-2-27b",
    "glm46": "glm-4.6",
    "glm5": "glm-5",
    "gptoss120b": "gpt-oss-120b",
    "kimik2": "kimi-k2",
    "llama3170b": "llama-3.1-70b",
    "llama318b": "llama-3.1-8b",
    "llama3370b": "llama-3.3-70b",
    "llama370b": "llama-3-70b",
    "llama38b": "llama-3-8b",
    "llama4maverick": "llama-4-maverick",
    "llama4scout": "llama-4-scout",
    "mimov2flash": "mimo-v2-flash",
    "minimaxm27": "minimax-m2.7",
    "mixtral8x22b": "mixtral-8x22b",
    "mixtral8x7b": "mixtral-8x7b",
    "nemotron3nano30ba3b": "nemotron-3-nano-30b",
    "phi3mini38b": "phi-3-mini",
    "phi4": "phi-4",
    "qwen2572b": "qwen-2.5-72b",
    "qwen3next80ba3b": "qwen3-next-80b-a3b",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def years_since_2024(value: str) -> float:
    return (date.fromisoformat(value) - date(2024, 1, 1)).days / 365.25


def base_key(model: str) -> str:
    return model.removesuffix("-think")


def fit_ols(features: np.ndarray, target: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(features)), features])
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    return coefficients


def feature_vector(row: dict[str, Any], form: str) -> np.ndarray:
    values = [float(row["accuracy"])]
    if "date" in form:
        values.append(years_since_2024(row["release_date"]))
    if "moe" in form:
        values.append(float(row["moe"]))
    return np.asarray(values, dtype=float)


def predict_log10(
    train: list[dict[str, Any]], test: dict[str, Any], form: str
) -> tuple[float, dict[str, float]]:
    log_params = np.asarray([math.log10(row["params_b"]) for row in train])
    if form == "forward_inverse":
        accuracy = np.asarray([row["accuracy"] for row in train])
        slope, intercept = np.polyfit(log_params, accuracy, 1)
        return (float((test["accuracy"] - intercept) / slope), {
            "slope": float(slope),
            "intercept": float(intercept),
        })
    features = np.vstack([feature_vector(row, form) for row in train])
    coefficients = fit_ols(features, log_params)
    test_design = np.concatenate([[1.0], feature_vector(test, form)])
    return float(test_design @ coefficients), {
        f"coefficient_{index}": float(value)
        for index, value in enumerate(coefficients)
    }


def metrics(rows: list[dict[str, Any]], prediction_field: str) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    errors = np.asarray(
        [math.log10(float(row[prediction_field]) / float(row["actual_b"])) for row in rows]
    )
    absolute = np.abs(errors)
    return {
        "n": len(rows),
        "families": len({row["family"] for row in rows}),
        "vendors": len({row.get("vendor", row["family"]) for row in rows}),
        "mean_absolute_log10_error": float(np.mean(absolute)),
        "median_multiplicative_error": float(10 ** np.median(absolute)),
        "rmse_log10": float(np.sqrt(np.mean(errors**2))),
        "p80_multiplicative_error": float(10 ** np.quantile(absolute, 0.8)),
        "within_2x": float(np.mean(absolute <= math.log10(2))),
        "signed_bias_factor": float(10 ** np.mean(errors)),
    }


def collapse_rows(
    rows: list[dict[str, Any]], policy: str
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[base_key(row["model"])].append(row)
    output = []
    for key, variants in sorted(groups.items()):
        params = {float(row["params_b"]) for row in variants}
        families = {row["family"] for row in variants}
        vendors = {row["vendor"] for row in variants}
        architectures = {row["moe"] for row in variants}
        if len(params) != 1 or len(families) != 1 or len(vendors) != 1 or len(architectures) != 1:
            raise ValueError(f"Inconsistent IKP serving variants for {key}")
        if policy == "mean":
            accuracy = float(np.mean([row["accuracy"] for row in variants]))
        elif policy == "max":
            accuracy = max(row["accuracy"] for row in variants)
        elif policy == "nonthinking":
            nonthinking = [row for row in variants if not row["thinking"]]
            selected = nonthinking if nonthinking else variants
            accuracy = float(np.mean([row["accuracy"] for row in selected]))
        else:
            raise ValueError(policy)
        output.append(
            {
                "base_key": key,
                "model": key,
                "variants": [row["model"] for row in variants],
                "variant_count": len(variants),
                "params_b": next(iter(params)),
                "accuracy": accuracy,
                "family": next(iter(families)),
                "vendor": next(iter(vendors)),
                "moe": next(iter(architectures)),
                "release_date": min(row["release_date"] for row in variants),
            }
        )
    return output


def strict_predictions(
    rows: list[dict[str, Any]], policy: str, forms: list[str]
) -> list[dict[str, Any]]:
    output = []
    for test in sorted(rows, key=lambda row: (row["release_date"], row["base_key"])):
        train = [
            row
            for row in rows
            if row["release_date"] < test["release_date"]
            and row["vendor"] != test["vendor"]
        ]
        if len(train) < MIN_TRAIN_ROWS or len({row["vendor"] for row in train}) < MIN_TRAIN_VENDORS:
            continue
        for form in forms:
            predicted_log10, coefficients = predict_log10(train, test, form)
            error = predicted_log10 - math.log10(test["params_b"])
            output.append(
                {
                    "collapse_policy": policy,
                    "form": form,
                    "model": test["model"],
                    "base_key": test["base_key"],
                    "variants": "|".join(test["variants"]),
                    "release_date": test["release_date"],
                    "family": test["family"],
                    "vendor": test["vendor"],
                    "moe": test["moe"],
                    "accuracy": test["accuracy"],
                    "actual_b": test["params_b"],
                    "predicted_b": float(10**predicted_log10),
                    "log10_error": float(error),
                    "multiplicative_error": float(10 ** abs(error)),
                    "train_rows": len(train),
                    "train_families": len({row["family"] for row in train}),
                    "train_vendors": len({row["vendor"] for row in train}),
                    "train_max_date": max(row["release_date"] for row in train),
                    "test_vendor_excluded": all(row["vendor"] != test["vendor"] for row in train),
                    "coefficients_json": json.dumps(coefficients, sort_keys=True),
                }
            )
    return output


def family_bootstrap(values: dict[str, list[float]]) -> dict[str, Any]:
    family_means = np.asarray([np.mean(group) for group in values.values()], dtype=float)
    rng = np.random.default_rng(RANDOM_SEED)
    draws = rng.choice(
        family_means,
        size=(BOOTSTRAP_SAMPLES, len(family_means)),
        replace=True,
    ).mean(axis=1)
    return {
        "metric": "family-balanced mean absolute log10 error delta; blend minus existing ensemble",
        "observed_delta": float(family_means.mean()),
        "ci_90": [float(np.quantile(draws, 0.05)), float(np.quantile(draws, 0.95))],
        "probability_blend_better": float(np.mean(draws < 0)),
        "families": len(family_means),
        "samples": BOOTSTRAP_SAMPLES,
        "random_seed": RANDOM_SEED,
    }


def blended_prediction(row: dict[str, Any], weight: float) -> float:
    return float(
        10
        ** (
            (1 - weight) * math.log10(float(row["existing_predicted_b"]))
            + weight * math.log10(float(row["ikp_predicted_b"]))
        )
    )


def family_balanced_mae(rows: list[dict[str, Any]], weight: float) -> float:
    errors: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        prediction = blended_prediction(row, weight)
        errors[row["family"]].append(abs(math.log10(prediction / float(row["actual_b"]))))
    return float(np.mean([np.mean(values) for values in errors.values()]))


def leave_one_family_out_meta(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Learn the blend weight without the family whose errors are scored."""

    predictions = []
    for family in sorted({row["family"] for row in rows}):
        train = [row for row in rows if row["family"] != family]
        test = [row for row in rows if row["family"] == family]
        selected_weight = min(
            META_WEIGHT_GRID,
            key=lambda weight: (family_balanced_mae(train, weight), weight),
        )
        for row in test:
            prediction = blended_prediction(row, selected_weight)
            output = dict(row)
            output.update(
                {
                    "meta_predicted_b": prediction,
                    "meta_selected_ikp_weight": selected_weight,
                    "meta_train_rows": len(train),
                    "meta_train_families": len({item["family"] for item in train}),
                    "meta_test_family_excluded": all(
                        item["family"] != family for item in train
                    ),
                }
            )
            predictions.append(output)
    family_deltas: dict[str, list[float]] = defaultdict(list)
    for row in predictions:
        family_deltas[row["family"]].append(
            abs(math.log10(row["meta_predicted_b"] / row["actual_b"]))
            - row["existing_abs_log10_error"]
        )
    return {
        "protocol": (
            "outer leave-one-family-out; blend weight selected on all other families "
            "from a 0.00-1.00 grid in 0.05 increments"
        ),
        "metrics": metrics(predictions, "meta_predicted_b"),
        "family_bootstrap": family_bootstrap(family_deltas),
        "selected_weight_counts": {
            str(weight): sum(
                row["meta_selected_ikp_weight"] == weight for row in predictions
            )
            for weight in sorted({row["meta_selected_ikp_weight"] for row in predictions})
        },
        "all_test_families_excluded": all(
            row["meta_test_family_excluded"] for row in predictions
        ),
    }


def chronological_meta_subset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Score the fixed conservative blend where prior overlap can select a weight."""

    selected = []
    for row in sorted(rows, key=lambda item: (item["release_date"], item["family"])):
        train = [
            item
            for item in rows
            if item["release_date"] < row["release_date"]
            and item["family"] != row["family"]
        ]
        if (
            len(train) < META_MIN_PRIOR_ROWS
            or len({item["family"] for item in train}) < META_MIN_PRIOR_FAMILIES
        ):
            continue
        output = dict(row)
        output["meta_prior_rows"] = len(train)
        output["meta_prior_families"] = len({item["family"] for item in train})
        selected.append(output)
    family_deltas: dict[str, list[float]] = defaultdict(list)
    for row in selected:
        family_deltas[row["family"]].append(
            row["blend_minus_existing_abs_log10_error"]
        )
    return {
        "protocol": (
            f"fixed {TEST_WEIGHT:.0%} IKP blend, evaluated only after at least "
            f"{META_MIN_PRIOR_ROWS} earlier overlap rows from "
            f"{META_MIN_PRIOR_FAMILIES} other families exist"
        ),
        "models": len(selected),
        "families": len(family_deltas),
        "existing": metrics(selected, "existing_predicted_b"),
        "blend": metrics(selected, "blended_predicted_b"),
        "family_bootstrap": family_bootstrap(family_deltas),
    }


def refusal_adjusted_accuracy(model: dict[str, Any]) -> tuple[float, float]:
    tier_scores = []
    refusals = 0
    total = 0
    for stats in model["tier_stats"].values():
        attempted = stats["total"] - stats["refusal"]
        tier_scores.append(stats["correct"] / attempted if attempted else 0.0)
        refusals += stats["refusal"]
        total += stats["total"]
    return float(np.mean(tier_scores)), refusals / total


def main() -> None:
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    configs = json.loads(CONFIG.read_text(encoding="utf-8"))["models"]
    sensitivity = json.loads(SENSITIVITY.read_text(encoding="utf-8"))
    validation = json.loads(V2_VALIDATION.read_text(encoding="utf-8"))
    backtest = json.loads(BACKTEST.read_text(encoding="utf-8"))
    replication_wikidata = json.loads(REPLICATION_WIKIDATA.read_text(encoding="utf-8"))

    points = calibration["calibration_points"]
    if len(points) != 93 or len({row["model"] for row in points}) != 93:
        raise ValueError("Expected 93 unique IKP calibration configurations")
    summary_by_model = {row["model"]: row for row in summary}
    raw_rows = []
    summary_mismatches = []
    for point in points:
        model = point["model"]
        source = summary_by_model[model]
        config = configs[model]
        for field in ("accuracy", "params_B", "family", "vendor"):
            left = point.get(field)
            right = source.get(field)
            if left != right:
                summary_mismatches.append({"model": model, "field": field, "calibration": left, "summary": right})
        if not config.get("release_date"):
            raise ValueError(f"Missing IKP release date: {model}")
        raw_rows.append(
            {
                "model": model,
                "params_b": float(point["params_B"]),
                "accuracy": float(point["accuracy"]),
                "family": point["family"],
                "vendor": point["vendor"],
                "moe": int(point["arch"] == "moe"),
                "thinking": bool(point["thinking"]),
                "release_date": config["release_date"],
            }
        )
    if summary_mismatches:
        raise ValueError(f"IKP calibration/summary mismatch: {summary_mismatches[:3]}")

    log_params = np.log10([row["params_b"] for row in raw_rows])
    accuracies = np.asarray([row["accuracy"] for row in raw_rows])
    reproduced_slope, reproduced_intercept = np.polyfit(log_params, accuracies, 1)
    reproduced_fitted = reproduced_slope * log_params + reproduced_intercept
    reproduced_r2 = 1 - np.sum((accuracies - reproduced_fitted) ** 2) / np.sum(
        (accuracies - accuracies.mean()) ** 2
    )
    published_fit = calibration["fit"]
    if not (
        math.isclose(reproduced_slope, published_fit["slope"], abs_tol=1e-14)
        and math.isclose(reproduced_intercept, published_fit["intercept"], abs_tol=1e-14)
        and math.isclose(reproduced_r2, published_fit["r_squared"], abs_tol=1e-14)
    ):
        raise ValueError("Failed to reproduce published IKP fit")

    forms = [
        "forward_inverse",
        "inverse_accuracy",
        "inverse_accuracy_date",
        "inverse_accuracy_moe",
        "inverse_accuracy_date_moe",
    ]
    collapsed = {
        policy: collapse_rows(raw_rows, policy)
        for policy in ("mean", "nonthinking", "max")
    }
    prediction_rows = []
    for policy, rows in collapsed.items():
        prediction_rows.extend(strict_predictions(rows, policy, forms))
    if not all(row["train_max_date"] < row["release_date"] for row in prediction_rows):
        raise ValueError("IKP chronology violation")
    if not all(row["test_vendor_excluded"] for row in prediction_rows):
        raise ValueError("IKP vendor-holdout violation")

    heldout_metrics: dict[str, Any] = {}
    for policy in collapsed:
        heldout_metrics[policy] = {}
        for form in forms:
            selected = [
                row
                for row in prediction_rows
                if row["collapse_policy"] == policy and row["form"] == form
            ]
            heldout_metrics[policy][form] = {
                "all": metrics(selected, "predicted_b"),
                "frontier_70b_plus": metrics(
                    [row for row in selected if row["actual_b"] >= 70],
                    "predicted_b",
                ),
                "modern_2025_plus": metrics(
                    [row for row in selected if row["release_date"] >= "2025-01-01"],
                    "predicted_b",
                ),
            }

    primary_predictions = {
        row["base_key"]: row
        for row in prediction_rows
        if row["collapse_policy"] == PRIMARY_COLLAPSE and row["form"] == PRIMARY_FORM
    }
    existing = {row["normalized_model"]: row for row in backtest["ensemble_predictions"]}
    overlap_rows = []
    for backtest_key, ikp_key in BACKTEST_TO_IKP.items():
        if backtest_key not in existing or ikp_key not in primary_predictions:
            continue
        base = existing[backtest_key]
        ikp = primary_predictions[ikp_key]
        if abs(math.log10(base["actual_b"] / ikp["actual_b"])) > math.log10(1.06):
            raise ValueError(
                f"Parameter identity disagreement for {backtest_key}: "
                f"{base['actual_b']} vs {ikp['actual_b']}"
            )
        actual = float(base["actual_b"])
        existing_error = math.log10(float(base["predicted_b"]) / actual)
        ikp_error = math.log10(float(ikp["predicted_b"]) / actual)
        blended_log = (1 - TEST_WEIGHT) * math.log10(float(base["predicted_b"])) + TEST_WEIGHT * math.log10(float(ikp["predicted_b"]))
        blended = 10**blended_log
        overlap_rows.append(
            {
                "normalized_model": backtest_key,
                "ikp_base_key": ikp_key,
                "model": base["model"],
                "release_date": base["release_date"],
                "family": base["family"],
                "actual_b": actual,
                "existing_predicted_b": base["predicted_b"],
                "ikp_predicted_b": ikp["predicted_b"],
                "blend_weight": TEST_WEIGHT,
                "blended_predicted_b": blended,
                "existing_abs_log10_error": abs(existing_error),
                "ikp_abs_log10_error": abs(ikp_error),
                "blended_abs_log10_error": abs(math.log10(blended / actual)),
                "blend_minus_existing_abs_log10_error": abs(math.log10(blended / actual)) - abs(existing_error),
                "ikp_train_rows": ikp["train_rows"],
                "ikp_train_vendors": ikp["train_vendors"],
                "ikp_train_max_date": ikp["train_max_date"],
                "ikp_test_vendor_excluded": ikp["test_vendor_excluded"],
            }
        )

    family_deltas: dict[str, list[float]] = defaultdict(list)
    for row in overlap_rows:
        family_deltas[row["family"]].append(row["blend_minus_existing_abs_log10_error"])
    bootstrap = family_bootstrap(family_deltas)
    fixed_family_improvements = {
        family: float(np.mean(values))
        for family, values in sorted(family_deltas.items())
    }
    nested_family_holdout = leave_one_family_out_meta(overlap_rows)
    chronological_fixed = chronological_meta_subset(overlap_rows)
    existing_signed_errors = np.asarray(
        [
            math.log10(row["existing_predicted_b"] / row["actual_b"])
            for row in overlap_rows
        ]
    )
    ikp_signed_errors = np.asarray(
        [
            math.log10(row["ikp_predicted_b"] / row["actual_b"])
            for row in overlap_rows
        ]
    )
    signed_error_correlation = float(
        np.corrcoef(existing_signed_errors, ikp_signed_errors)[0, 1]
    )
    weight_grid = []
    for weight in np.linspace(0, 0.5, 11):
        family_errors: dict[str, list[float]] = defaultdict(list)
        for row in overlap_rows:
            prediction = 10 ** (
                (1 - weight) * math.log10(row["existing_predicted_b"])
                + weight * math.log10(row["ikp_predicted_b"])
            )
            family_errors[row["family"]].append(abs(math.log10(prediction / row["actual_b"])))
        weight_grid.append(
            {
                "ikp_weight": float(weight),
                "family_balanced_mean_absolute_log10_error": float(
                    np.mean([np.mean(values) for values in family_errors.values()])
                ),
            }
        )
    best_grid = min(weight_grid, key=lambda row: row["family_balanced_mean_absolute_log10_error"])

    proprietary = {row["model"]: row for row in calibration["proprietary_estimates"]}
    source_fable = proprietary["claude-fable-5"]
    summary_fable = summary_by_model["claude-fable-5"]
    fable_config = configs["claude-fable-5"]
    refusal_accuracy, refusal_rate = refusal_adjusted_accuracy(summary_fable)
    refusal_adjusted_b = 10 ** ((refusal_accuracy - published_fit["intercept"]) / published_fit["slope"])
    fable_open_only_estimates: dict[str, dict[str, Any]] = {}
    for policy, rows in collapsed.items():
        train = [
            row
            for row in rows
            if row["release_date"] < fable_config["release_date"]
            and row["vendor"] != fable_config["vendor"]
        ]
        estimates = {}
        for form in ("forward_inverse", "inverse_accuracy", "inverse_accuracy_date"):
            predicted_log10, coefficients = predict_log10(
                train,
                {
                    "accuracy": source_fable["accuracy"],
                    "release_date": fable_config["release_date"],
                    "moe": 0,
                },
                form,
            )
            estimates[form] = {
                "estimated_b": float(10**predicted_log10),
                "coefficients": coefficients,
            }
        fable_open_only_estimates[policy] = {
            "train_rows": len(train),
            "train_families": len({row["family"] for row in train}),
            "train_vendors": len({row["vendor"] for row in train}),
            "train_max_date": max(row["release_date"] for row in train),
            "test_vendor_excluded": all(
                row["vendor"] != fable_config["vendor"] for row in train
            ),
            "estimates": estimates,
        }
    open_only_values = [
        item["estimated_b"]
        for policy in fable_open_only_estimates.values()
        for item in policy["estimates"].values()
    ]
    fable_target = {
        "model": "Claude Fable 5",
        "observed_accuracy": source_fable["accuracy"],
        "published_lambda0_estimate_b": source_fable["estimated_B"],
        "published_pi90_b": [source_fable["pi_lo"], source_fable["pi_hi"]],
        "refusal_rate": refusal_rate,
        "refusal_confidence_tier": (
            "Reliable" if refusal_rate < 0.10 else "Caution" if refusal_rate <= 0.30 else "Low confidence"
        ),
        "refusal_adjusted_accuracy": refusal_accuracy,
        "refusal_adjusted_reference_b": refusal_adjusted_b,
        "lambda_sensitivity_b": {
            str(row["lambda"]): row["estimates"]["Claude Fable 5"]
            for row in sensitivity["rows"]
        },
        "lambda_sensitivity_min_b": min(
            row["estimates"]["Claude Fable 5"] for row in sensitivity["rows"]
        ),
        "lambda_sensitivity_max_b": max(
            row["estimates"]["Claude Fable 5"] for row in sensitivity["rows"]
        ),
        "strict_open_only_release_and_vendor_holdout": fable_open_only_estimates,
        "strict_open_only_model_form_min_b": min(open_only_values),
        "strict_open_only_model_form_max_b": max(open_only_values),
    }

    serving_controls = []
    for label, first, second, status in (
        ("GPT-5.5 vs GPT-5.5 Pro", "gpt-5.5", "gpt-5.5-pro", "first-party same underlying model"),
        ("GPT-5 vs GPT-5.5", "gpt-5-think", "gpt-5.5", "user-supplied same-base assumption"),
        ("Opus 4.7 vs Opus 4.8", "claude-opus-4.7-think", "claude-opus-4.8", "user-supplied same-base assumption"),
    ):
        left = proprietary[first]
        right = proprietary[second]
        ratio = max(left["estimated_B"], right["estimated_B"]) / min(left["estimated_B"], right["estimated_B"])
        serving_controls.append(
            {
                "label": label,
                "first_model": first,
                "first_estimated_b": left["estimated_B"],
                "second_model": second,
                "second_estimated_b": right["estimated_B"],
                "max_over_min": ratio,
                "identity_status": status,
            }
        )

    ambiguous_records = [value for value in replication_wikidata.values() if value.get("real_ambiguity")]
    replication_rows = list(csv.DictReader(REPLICATION.open(newline="", encoding="utf-8-sig")))
    support_gates = {
        "minimum_overlap_models": 12,
        "minimum_overlap_families": 10,
        "minimum_chronological_subset_models": 8,
        "minimum_chronological_subset_families": 7,
        "maximum_bootstrap_ci90_upper": 0.0,
        "minimum_bootstrap_probability_better": 0.90,
    }
    support_passes = (
        len(overlap_rows) >= support_gates["minimum_overlap_models"]
        and len(family_deltas) >= support_gates["minimum_overlap_families"]
        and bootstrap["ci_90"][1] < 0
        and bootstrap["probability_blend_better"] >= 0.90
        and chronological_fixed["models"]
        >= support_gates["minimum_chronological_subset_models"]
        and chronological_fixed["families"]
        >= support_gates["minimum_chronological_subset_families"]
        and chronological_fixed["family_bootstrap"]["ci_90"][1] < 0
        and chronological_fixed["family_bootstrap"]["probability_blend_better"]
        >= support_gates["minimum_bootstrap_probability_better"]
        and nested_family_holdout["all_test_families_excluded"]
    )
    result = {
        "generated_on": "2026-07-18",
        "source_inventory": {
            "upstream_commit": json.loads(SOURCE_METADATA.read_text(encoding="utf-8"))["upstream"]["commit"],
            "replication_commit": json.loads(SOURCE_METADATA.read_text(encoding="utf-8"))["replication"]["commit"],
            "calibration_configurations": len(raw_rows),
            "calibration_weight_bases": len(collapsed[PRIMARY_COLLAPSE]),
            "serving_variants_collapsed": len(raw_rows) - len(collapsed[PRIMARY_COLLAPSE]),
            "calibration_vendors": len({row["vendor"] for row in raw_rows}),
            "calibration_families": len({row["family"] for row in raw_rows}),
            "all_release_dates_present": all(row["release_date"] for row in raw_rows),
            "replication_calibration_rows": len(replication_rows),
        },
        "published_reproduction": {
            "summary_mismatches": summary_mismatches,
            "slope": float(reproduced_slope),
            "intercept": float(reproduced_intercept),
            "r_squared": float(reproduced_r2),
            "published_loo_median_error": calibration["loo_cv"]["median_fold_err"],
            "published_loo_within_2x": calibration["loo_cv"]["within_2x"],
            "upstream_raw_accuracy_reproduction_max_diff": validation["check1_max_accuracy_diff"],
            "raw_response_scope": "upstream validation claim; raw per-probe responses are not copied into this project",
        },
        "strict_protocol": {
            "primary_collapse": PRIMARY_COLLAPSE,
            "primary_form": PRIMARY_FORM,
            "training_rule": "strictly earlier release dates and test vendor excluded",
            "minimum_train_rows": MIN_TRAIN_ROWS,
            "minimum_train_vendors": MIN_TRAIN_VENDORS,
            "forms": forms,
        },
        "heldout_metrics": heldout_metrics,
        "incremental_overlap": {
            "models": len(overlap_rows),
            "families": len(family_deltas),
            "existing": metrics(overlap_rows, "existing_predicted_b"),
            "ikp": metrics(overlap_rows, "ikp_predicted_b"),
            "blend_10pct": metrics(overlap_rows, "blended_predicted_b"),
            "family_bootstrap": bootstrap,
            "family_mean_deltas": fixed_family_improvements,
            "families_improved": sum(
                value < 0 for value in fixed_family_improvements.values()
            ),
            "signed_error_correlation_existing_vs_ikp": signed_error_correlation,
            "weight_grid": weight_grid,
            "best_in_sample_grid_weight": best_grid,
            "nested_leave_one_family_out_weight_learning": nested_family_holdout,
            "chronological_fixed_weight_subset": chronological_fixed,
        },
        "target_signal": {
            "fable": fable_target,
            "sol": {
                "observed": False,
                "reason": "GPT-5.6 Sol is absent from the pinned IKP evaluation; no family transport is substituted.",
            },
        },
        "serving_and_same_base_controls": serving_controls,
        "replication_quality_flags": {
            "wikidata_flagged_records_rechecked": len(replication_wikidata),
            "wikidata_flagged_records_still_real_ambiguity": len(ambiguous_records),
            "scope": "flagged subset, not a denominator over all probes",
            "historical_replication_calibration_rows": len(replication_rows),
            "current_upstream_response": "v2 uses lambda=0, a cleaned probe set, and refusal-aware reporting",
        },
        "decision": {
            "promote_incremental_ikp_weight": support_passes,
            "incremental_evidence_weight": TEST_WEIGHT if support_passes else 0.0,
            "incremental_final_weight_when_crowd_is_50pct": (
                TEST_WEIGHT * 0.5 if support_passes else 0.0
            ),
            "change_fable_center": support_passes,
            "change_sol_center": False,
            "evidence_gates": support_gates,
            "reason": (
                "A conservative fixed 10% evidence-level IKP blend passes the encoded "
                "model-count, family-count, full-overlap bootstrap, and later chronological-subset gates."
                if support_passes
                else "IKP is retained as a direct Fable sensitivity because it does not pass every encoded incremental-validation gate."
            ),
        },
        "limitations": [
            "IKP measures effective factual-recall capacity and can be shifted by training-data coverage, refusal policy, and serving configuration.",
            "The strict backtest uses current benchmark responses for historical checkpoints, not release-vintage response snapshots.",
            "Fable is above the observed open-weight accuracy frontier and its estimate is extrapolative.",
            "The source's 90% interval is a model prediction interval, not a disclosure probability.",
            "The independent replication audited an earlier benchmark version; v2 fixes some but not all diversity and ambiguity concerns.",
        ],
        "source_files": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (
                CALIBRATION,
                SUMMARY,
                CONFIG,
                SENSITIVITY,
                V2_VALIDATION,
                REPLICATION,
                REPLICATION_WIKIDATA,
                SOURCE_METADATA,
                BACKTEST,
            )
        },
        "outputs": {
            "prediction_ledger": str(PREDICTIONS.relative_to(ROOT)),
            "overlap_ledger": str(OVERLAP.relative_to(ROOT)),
            "site_data": str(SITE_OUTPUT.relative_to(ROOT)),
        },
    }

    PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)
    with PREDICTIONS.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)
    with OVERLAP.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(overlap_rows[0]))
        writer.writeheader()
        writer.writerows(overlap_rows)
    RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    SITE_OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "result": str(RESULT),
                "weight_bases": len(collapsed[PRIMARY_COLLAPSE]),
                "strict_predictions": len(prediction_rows),
                "overlap": len(overlap_rows),
                "overlap_families": len(family_deltas),
                "bootstrap": bootstrap,
                "fable": fable_target,
                "decision": result["decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
