#!/usr/bin/env python3
"""Test request-supported OpenRouter throughput and latency after API price.

This is a deliberately conservative follow-up to the one-week throughput audit.
It uses the frozen 30-minute endpoint percentiles, requires at least 100 requests
for the request-supported features, and caps log request weights so one popular
endpoint cannot dominate a model.  All candidate specifications are evaluated
on the same complete checkpoint panel under developer-family and chronological
developer-family holdout.  Nothing is promoted from a favorable point estimate
unless the family-cluster interval also excludes zero.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

import analyze_openrouter_parameter_signal as base


ROOT = Path(__file__).resolve().parent
COMPATIBILITY_FILE_DATE = "2026-07-18"
DATE = COMPATIBILITY_FILE_DATE
OUT = ROOT / "outputs/019f6c42-2d53-7743-ab07-6293e2618dd7"
ENDPOINT_TIERS = ROOT / f"sources/openrouter_endpoint_tier_signals_{DATE}.csv"
RESULT = OUT / f"openrouter_request_weighted_operational_audit_{DATE}.json"
PREDICTIONS = OUT / f"openrouter_request_weighted_operational_predictions_{DATE}.csv"

MIN_REQUESTS = 100
REQUEST_WEIGHT_CAP = 10_000


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def positive(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed > 0 else None


def weighted_median(values: list[tuple[float, float]]) -> float:
    ordered = sorted(values)
    threshold = sum(weight for _, weight in ordered) / 2
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= threshold:
            return value
    return ordered[-1][0]


def normalized_endpoint_index(
    rows: list[dict[str, str]],
    numerator: str,
    *,
    denominator: str | None = None,
    minimum_requests: int = 1,
    request_weighted: bool = False,
) -> tuple[dict[str, float], dict[str, Any]]:
    usable: list[tuple[dict[str, str], float, float]] = []
    for row in rows:
        if row["service_tier"] != "default":
            continue
        requests = positive(row["request_count_30m"])
        value = positive(row[numerator])
        if denominator:
            divisor = positive(row[denominator])
            value = value / divisor if value is not None and divisor is not None else None
        if value is None or requests is None or requests < minimum_requests:
            continue
        usable.append((row, math.log(value), requests))

    by_provider_quantization: dict[tuple[str, str], list[float]] = defaultdict(list)
    by_provider: dict[str, list[float]] = defaultdict(list)
    by_quantization: dict[str, list[float]] = defaultdict(list)
    for row, log_value, _ in usable:
        provider = row["provider_slug"] or row["provider_name"]
        quantization = row["quantization"] or "unspecified"
        by_provider_quantization[(provider, quantization)].append(log_value)
        by_provider[provider].append(log_value)
        by_quantization[quantization].append(log_value)

    global_center = statistics.median(log_value for _, log_value, _ in usable)
    residuals: dict[str, list[tuple[float, float]]] = defaultdict(list)
    control_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row, log_value, requests in usable:
        provider = row["provider_slug"] or row["provider_name"]
        quantization = row["quantization"] or "unspecified"
        provider_quantization_values = by_provider_quantization[(provider, quantization)]
        provider_values = by_provider[provider]
        quantization_values = by_quantization[quantization]
        if len(provider_quantization_values) >= 4:
            center = statistics.median(provider_quantization_values)
            control = "provider+quantization"
        elif len(provider_values) >= 4:
            center = statistics.median(provider_values)
            control = "provider"
        elif len(quantization_values) >= 4:
            center = statistics.median(quantization_values)
            control = "quantization"
        else:
            center = global_center
            control = "global"
        weight = (
            min(math.log1p(requests), math.log1p(REQUEST_WEIGHT_CAP))
            if request_weighted
            else 1.0
        )
        model_id = row["openrouter_model_id"]
        residuals[model_id].append((log_value - center, weight))
        control_counts[model_id][control] += 1

    index = {
        model_id: math.exp(weighted_median(model_residuals))
        for model_id, model_residuals in residuals.items()
    }
    return index, {
        "eligible_endpoint_rows": len(usable),
        "models": len(index),
        "minimum_requests": minimum_requests,
        "request_weighted": request_weighted,
        "request_weight": (
            f"min(log1p(request_count), log1p({REQUEST_WEIGHT_CAP}))"
            if request_weighted
            else "equal endpoint weight"
        ),
        "normalization_hierarchy": "provider+quantization when n>=4, then provider, quantization, global",
        "control_counts": {
            model_id: dict(sorted(counts.items()))
            for model_id, counts in sorted(control_counts.items())
        },
    }


def build_complete_panel() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    endpoint_rows = read_csv(ENDPOINT_TIERS)
    match_audit, calibration, _, one_week_index = base.build_match_and_calibration()
    model_ids_by_checkpoint: dict[str, list[str]] = defaultdict(list)
    for row in match_audit:
        if row["canonical_checkpoint_id"]:
            model_ids_by_checkpoint[row["canonical_checkpoint_id"]].append(
                row["openrouter_model_id"]
            )

    definitions = {
        "throughput_p50_all": {
            "numerator": "p50_throughput_tps_30m",
            "minimum_requests": 1,
            "request_weighted": False,
        },
        "throughput_p50_supported": {
            "numerator": "p50_throughput_tps_30m",
            "minimum_requests": MIN_REQUESTS,
            "request_weighted": True,
        },
        "latency_p50_supported": {
            "numerator": "p50_latency_seconds_30m",
            "minimum_requests": MIN_REQUESTS,
            "request_weighted": True,
        },
        "throughput_p90_over_p50_supported": {
            "numerator": "p90_throughput_tps_30m",
            "denominator": "p50_throughput_tps_30m",
            "minimum_requests": MIN_REQUESTS,
            "request_weighted": True,
        },
        "latency_p90_over_p50_supported": {
            "numerator": "p90_latency_seconds_30m",
            "denominator": "p50_latency_seconds_30m",
            "minimum_requests": MIN_REQUESTS,
            "request_weighted": True,
        },
    }
    indices: dict[str, dict[str, float]] = {}
    metadata: dict[str, Any] = {}
    for name, definition in definitions.items():
        index, details = normalized_endpoint_index(endpoint_rows, **definition)
        indices[name] = index
        metadata[name] = details

    panel: list[dict[str, Any]] = []
    incomplete: list[dict[str, Any]] = []
    for row in calibration:
        checkpoint = row["canonical_checkpoint_id"]
        aliases = model_ids_by_checkpoint[checkpoint]
        copy = dict(row)
        copy["log10_one_week_throughput"] = math.log10(
            statistics.median(one_week_index[model_id] for model_id in aliases)
        )
        missing: list[str] = []
        for name, index in indices.items():
            values = [index[model_id] for model_id in aliases if model_id in index]
            if values:
                copy[f"log10_{name}"] = math.log10(statistics.median(values))
            else:
                copy[f"log10_{name}"] = None
                missing.append(name)
        if missing:
            incomplete.append(
                {
                    "canonical_checkpoint_id": checkpoint,
                    "model": row["canonical_model_name"],
                    "missing_features": missing,
                }
            )
        else:
            panel.append(copy)

    return panel, {
        "endpoint_default_rows": sum(
            row["service_tier"] == "default" for row in endpoint_rows
        ),
        "complete_checkpoints": len(panel),
        "complete_families": len({row["family"] for row in panel}),
        "incomplete_checkpoints": incomplete,
        "feature_metadata": metadata,
    }


SPECS = {
    "date_price": ("release_decimal_year", "log10_blended_price"),
    "date_price_one_week_throughput": (
        "release_decimal_year",
        "log10_blended_price",
        "log10_one_week_throughput",
    ),
    "date_price_p50_throughput_all": (
        "release_decimal_year",
        "log10_blended_price",
        "log10_throughput_p50_all",
    ),
    "date_price_p50_throughput_supported": (
        "release_decimal_year",
        "log10_blended_price",
        "log10_throughput_p50_supported",
    ),
    "date_price_p50_latency_supported": (
        "release_decimal_year",
        "log10_blended_price",
        "log10_latency_p50_supported",
    ),
    "date_price_p50_throughput_latency_supported": (
        "release_decimal_year",
        "log10_blended_price",
        "log10_throughput_p50_supported",
        "log10_latency_p50_supported",
    ),
    "date_price_p50_throughput_tail_spreads_supported": (
        "release_decimal_year",
        "log10_blended_price",
        "log10_throughput_p50_supported",
        "log10_throughput_p90_over_p50_supported",
        "log10_latency_p90_over_p50_supported",
    ),
}


def design(
    rows: list[dict[str, Any]], features: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(
        [[float(row[feature]) for feature in features] for row in rows], dtype=float
    )
    y = np.log10(
        np.asarray([float(row["total_parameters_b"]) for row in rows], dtype=float)
    )
    return x, y


def held_out_predictions(
    rows: list[dict[str, Any]], mode: str
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for specification, features in SPECS.items():
        for test in rows:
            train = [
                row
                for row in rows
                if row["family"] != test["family"]
                and (mode == "family" or row["release_date"] < test["release_date"])
            ]
            if len(train) < 20 or len({row["family"] for row in train}) < 5:
                continue
            train_x, train_y = design(train, features)
            fit = base.robust_ridge_fit(
                train_x, train_y, [row["family"] for row in train]
            )
            test_x, test_y = design([test], features)
            predicted_log = float(base.predict_fit(fit, test_x)[0])
            output.append(
                {
                    "mode": mode,
                    "specification": specification,
                    "canonical_checkpoint_id": test["canonical_checkpoint_id"],
                    "model": test["canonical_model_name"],
                    "family": test["family"],
                    "release_date": test["release_date"],
                    "actual_parameters_b": test["total_parameters_b"],
                    "predicted_parameters_b": 10**predicted_log,
                    "log10_error": predicted_log - float(test_y[0]),
                    "training_rows": len(train),
                    "training_families": len({row["family"] for row in train}),
                    "training_max_date": max(row["release_date"] for row in train),
                }
            )
    return output


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    source_dates = {row["snapshot_date"] for row in read_csv(ENDPOINT_TIERS)}
    if len(source_dates) != 1:
        raise ValueError(
            f"Expected one OpenRouter observation date, found {source_dates}"
        )
    observation_date = next(iter(source_dates))
    panel, inventory = build_complete_panel()
    predictions = held_out_predictions(panel, "family") + held_out_predictions(
        panel, "chronological_family"
    )
    write_csv(PREDICTIONS, predictions)
    metrics = base.metrics_by_mode_spec(predictions)
    comparisons = [
        base.paired_family_bootstrap(
            predictions,
            mode,
            candidate,
            "date_price",
            samples=20_000,
        )
        for mode in ("family", "chronological_family")
        for candidate in SPECS
        if candidate != "date_price"
    ]
    keyed = {(row["mode"], row["candidate"]): row for row in comparisons}
    supported_candidates = []
    for candidate in SPECS:
        if candidate == "date_price":
            continue
        family = keyed[("family", candidate)]
        chronological = keyed[("chronological_family", candidate)]
        if (
            family["bootstrap_probability_candidate_better"] >= 0.90
            and family["ci_90"][1] < 0
            and chronological["observed_delta"] < 0
        ):
            supported_candidates.append(candidate)

    result = {
        "schema_version": "1.0",
        "snapshot_date": observation_date,
        "compatibility_filename_date": COMPATIBILITY_FILE_DATE,
        "question": "Do request-supported 30-minute throughput, latency, or percentile spreads add parameter-count information after date and API price?",
        "inventory": inventory,
        "candidate_specifications": SPECS,
        "fit": "family-balanced Huber robust ridge with standardized features; every candidate uses the same complete checkpoint panel",
        "heldout_metrics": metrics,
        "paired_family_bootstraps": comparisons,
        "promotion_policy": "family bootstrap >=90% favorable, family 90% CI wholly below zero, and chronological-family observed direction favorable",
        "supported_candidates": supported_candidates,
        "decision": {
            "promote_operational_feature": bool(supported_candidates),
            "incremental_live_weight": 0.05 if supported_candidates else 0.0,
            "change_headline_forecasts": False,
            "reason": (
                "No request-supported throughput, latency, joint, or tail-spread candidate passes the predeclared family-interval and chronological-direction gates."
                if not supported_candidates
                else "At least one candidate passes every gate; prospective replication is still required before changing headline forecasts."
            ),
        },
        "limitations": [
            "The 30-minute window is a frozen cross-section and can reflect transient traffic or routing.",
            "Request count measures traffic, not statistical independence; capped log weights reduce but do not remove popularity bias.",
            "Latency and throughput remain serving-stack outcomes affected by hardware, batching, speculative decoding, quantization, and provider policy.",
            "Candidate specifications were tested together, so isolated favorable point estimates are not treated as confirmatory evidence.",
        ],
        "source_manifest": {
            str(ENDPOINT_TIERS.relative_to(ROOT)): sha256(ENDPOINT_TIERS),
            str(base.MODEL_SIGNALS.relative_to(ROOT)): sha256(base.MODEL_SIGNALS),
            str(base.PROVIDER_SIGNALS.relative_to(ROOT)): sha256(base.PROVIDER_SIGNALS),
            str(base.UNIFIED.relative_to(ROOT)): sha256(base.UNIFIED),
        },
        "outputs": {"predictions": str(PREDICTIONS.relative_to(ROOT))},
    }
    RESULT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "result": str(RESULT),
                "complete_checkpoints": len(panel),
                "complete_families": inventory["complete_families"],
                "supported_candidates": supported_candidates,
                "incremental_live_weight": result["decision"]["incremental_live_weight"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
