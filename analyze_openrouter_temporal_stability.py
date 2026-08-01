#!/usr/bin/env python3
"""Audit temporal and service-tier stability of OpenRouter throughput.

This is a guard against interpreting serving-system measurements as model
architecture.  It uses the lossless dated endpoint panel, every immutable
refresh, and an explicit mixed-tier counterfactual.  The live forecast can
only gain tok/s weight if the corrected default-tier feature passes the same
family-held-out and chronological gates as the main OpenRouter audit.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from analyze_openrouter_parameter_signal import (
    held_out_predictions,
    metrics_by_mode_spec,
    paired_family_bootstrap,
    provider_normalized_indices,
)


ROOT = Path(__file__).resolve().parent
COMPATIBILITY_FILE_DATE = "2026-07-18"
DATE = COMPATIBILITY_FILE_DATE
OUT = ROOT / "outputs/019f6c42-2d53-7743-ab07-6293e2618dd7"
DAILY = ROOT / f"sources/openrouter_throughput_daily_{DATE}.csv"
DAILY_HISTORY = ROOT / f"sources/openrouter_throughput_daily_history_{DATE}.csv"
HISTORY_MANIFEST = ROOT / f"sources/openrouter_snapshot_history_manifest_{DATE}.csv"
MODELS = ROOT / f"sources/openrouter_model_signals_{DATE}.csv"
PROVIDERS = ROOT / f"sources/openrouter_provider_signals_{DATE}.csv"
CALIBRATION = OUT / f"openrouter_parameter_calibration_{DATE}.csv"
PARAMETER_AUDIT = OUT / f"openrouter_parameter_signal_backtest_{DATE}.json"

ENDPOINT_OUTPUT = OUT / f"openrouter_endpoint_temporal_stability_{DATE}.csv"
MODEL_OUTPUT = OUT / f"openrouter_model_temporal_stability_{DATE}.csv"
REFRESH_OUTPUT = OUT / f"openrouter_refresh_stability_{DATE}.csv"
TIER_PREDICTIONS = OUT / f"openrouter_tier_counterfactual_predictions_{DATE}.csv"
RESULT = OUT / f"openrouter_temporal_stability_audit_{DATE}.json"

FOCAL_MODELS = {
    "anthropic/claude-fable-5": "Claude Fable 5",
    "openai/gpt-5.6-sol": "GPT-5.6 Sol",
    "moonshotai/kimi-k3": "Kimi K3",
    "anthropic/claude-opus-4.8": "Claude Opus 4.8",
}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, data: list[dict[str, Any]]) -> None:
    fields = list(data[0]) if data else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="raise", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(data)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def positive(value: str | float | int | None) -> float | None:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed > 0 else None


def quantiles(values: Iterable[float], probs: Iterable[float]) -> list[float]:
    clean = list(values)
    return [float(value) for value in np.quantile(clean, list(probs))] if clean else []


def distribution_summary(values: Iterable[float]) -> dict[str, Any]:
    clean = list(values)
    if not clean:
        return {"n": 0}
    q10, q50, q90 = quantiles(clean, [0.10, 0.50, 0.90])
    return {
        "n": len(clean),
        "p10": q10,
        "median": q50,
        "p90": q90,
        "mean": float(statistics.fmean(clean)),
    }


def temporal_row(values: list[float]) -> dict[str, Any]:
    logs = [math.log(value) for value in values]
    median_log = statistics.median(logs)
    return {
        "observations": len(values),
        "median_tps": statistics.median(values),
        "mean_tps": statistics.fmean(values),
        "min_tps": min(values),
        "max_tps": max(values),
        "max_over_min": max(values) / min(values),
        "log_standard_deviation": statistics.pstdev(logs),
        "median_absolute_log_deviation": statistics.median(
            abs(value - median_log) for value in logs
        ),
    }


def endpoint_and_model_stability(
    daily: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    endpoint_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    endpoint_meta: dict[tuple[str, str], dict[str, str]] = {}
    model_date_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    model_names: dict[str, str] = {}
    for row in daily:
        if row["service_tier"] != "default":
            continue
        value = positive(row["throughput_tps"])
        if value is None:
            continue
        model_id = row["openrouter_model_id"]
        endpoint_id = row["endpoint_id"]
        key = (model_id, endpoint_id)
        endpoint_values[key].append(value)
        endpoint_meta.setdefault(key, row)
        model_date_values[(model_id, row["observation_date"])].append(value)
        model_names[model_id] = row["openrouter_model_name"]

    endpoints: list[dict[str, Any]] = []
    for key, values in sorted(endpoint_values.items()):
        meta = endpoint_meta[key]
        endpoints.append(
            {
                "openrouter_model_id": key[0],
                "openrouter_model_name": meta["openrouter_model_name"],
                "endpoint_id": key[1],
                "provider_name": meta["provider_name"],
                "provider_slug": meta["provider_slug"],
                "quantization": meta["quantization"],
                "endpoint_metadata_match": meta["endpoint_metadata_match"],
                **temporal_row(values),
            }
        )

    by_model: dict[str, list[tuple[str, float, int]]] = defaultdict(list)
    for (model_id, observation_date), values in model_date_values.items():
        by_model[model_id].append(
            (observation_date, statistics.median(values), len(values))
        )
    models: list[dict[str, Any]] = []
    for model_id, observations in sorted(by_model.items()):
        observations.sort()
        values = [value for _, value, _ in observations]
        stats = temporal_row(values)
        models.append(
            {
                "openrouter_model_id": model_id,
                "openrouter_model_name": model_names[model_id],
                "dates": len(observations),
                "date_min": observations[0][0],
                "date_max": observations[-1][0],
                "median_endpoints_per_date": statistics.median(
                    count for _, _, count in observations
                ),
                **stats,
                "daily_provider_medians_json": json.dumps(
                    {day: value for day, value, _ in observations}, sort_keys=True
                ),
            }
        )
    return endpoints, models


def refresh_stability(
    history: list[dict[str, str]],
) -> list[dict[str, Any]]:
    endpoint_values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    names: dict[str, str] = {}
    for row in history:
        if row["service_tier"] != "default":
            continue
        value = positive(row["throughput_tps"])
        if value is None:
            continue
        endpoint_values[
            (
                row["history_snapshot_id"],
                row["openrouter_model_id"],
                row["endpoint_id"],
            )
        ].append(value)
        names[row["openrouter_model_id"]] = row["openrouter_model_name"]

    snapshot_model: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (snapshot, model_id, _), values in endpoint_values.items():
        snapshot_model[(snapshot, model_id)].append(statistics.median(values))
    by_model: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for (snapshot, model_id), values in snapshot_model.items():
        by_model[model_id].append((snapshot, statistics.median(values)))

    output: list[dict[str, Any]] = []
    for model_id, observations in sorted(by_model.items()):
        observations.sort()
        values = [value for _, value in observations]
        output.append(
            {
                "openrouter_model_id": model_id,
                "openrouter_model_name": names[model_id],
                "snapshots": len(observations),
                **temporal_row(values),
                "snapshot_medians_json": json.dumps(
                    {snapshot: value for snapshot, value in observations}, sort_keys=True
                ),
            }
        )
    return output


def mixed_tier_counterfactual(
    daily: list[dict[str, str]],
    providers: list[dict[str, str]],
    calibration: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    all_tier_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    default_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    tiers: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in daily:
        value = positive(row["throughput_tps"])
        if value is None:
            continue
        key = (row["openrouter_model_id"], row["endpoint_id"])
        all_tier_values[key].append(value)
        tiers[key].add(row["service_tier"])
        if row["service_tier"] == "default":
            default_values[key].append(value)

    mixed_provider_rows: list[dict[str, str]] = []
    model_mixed_values: dict[str, list[float]] = defaultdict(list)
    model_default_values: dict[str, list[float]] = defaultdict(list)
    for row in providers:
        key = (row["openrouter_model_id"], row["endpoint_id"])
        mixed = (
            statistics.median(all_tier_values[key]) if all_tier_values.get(key) else None
        )
        default = (
            statistics.median(default_values[key]) if default_values.get(key) else None
        )
        copy = dict(row)
        copy["throughput_median_tps_1w"] = "" if mixed is None else str(mixed)
        mixed_provider_rows.append(copy)
        if mixed is not None:
            model_mixed_values[row["openrouter_model_id"]].append(mixed)
        if default is not None:
            model_default_values[row["openrouter_model_id"]].append(default)

    mixed_normalized, _ = provider_normalized_indices(mixed_provider_rows)
    current_normalized, _ = provider_normalized_indices(providers)
    mixed_model = {
        model_id: statistics.median(values)
        for model_id, values in model_mixed_values.items()
    }
    default_model = {
        model_id: statistics.median(values)
        for model_id, values in model_default_values.items()
    }

    mixed_calibration: list[dict[str, Any]] = []
    changed_checkpoints = 0
    ratios: list[float] = []
    for row in calibration:
        aliases = row["openrouter_model_ids"].split("|")
        raw_values = [mixed_model[alias] for alias in aliases if alias in mixed_model]
        normalized_values = [
            mixed_normalized[alias] for alias in aliases if alias in mixed_normalized
        ]
        default_alias_values = [
            default_model[alias] for alias in aliases if alias in default_model
        ]
        if not raw_values or not normalized_values or not default_alias_values:
            raise ValueError(f"Missing tier counterfactual for {row['canonical_checkpoint_id']}")
        mixed_raw = statistics.median(raw_values)
        mixed_norm = statistics.median(normalized_values)
        default_raw = statistics.median(default_alias_values)
        ratio = mixed_raw / default_raw
        ratios.append(ratio)
        if abs(math.log(ratio)) > 1e-12:
            changed_checkpoints += 1
        copy: dict[str, Any] = dict(row)
        copy["raw_throughput_tps_1w"] = mixed_raw
        copy["provider_normalized_throughput_ratio"] = mixed_norm
        copy["log10_raw_throughput"] = math.log10(mixed_raw)
        copy["log10_provider_normalized_throughput"] = math.log10(mixed_norm)
        mixed_calibration.append(copy)

    default_rows: list[dict[str, Any]] = [dict(row) for row in calibration]
    predictions: list[dict[str, Any]] = []
    for policy, panel in (
        ("default_only", default_rows),
        ("mixed_default_priority_flex_counterfactual", mixed_calibration),
    ):
        panel_predictions = held_out_predictions(panel, "family") + held_out_predictions(
            panel, "chronological_family"
        )
        for prediction in panel_predictions:
            prediction["throughput_tier_policy"] = policy
        predictions.extend(panel_predictions)

    policy_metrics: dict[str, Any] = {}
    policy_bootstraps: dict[str, Any] = {}
    for policy in (
        "default_only",
        "mixed_default_priority_flex_counterfactual",
    ):
        selected = [
            row for row in predictions if row["throughput_tier_policy"] == policy
        ]
        policy_metrics[policy] = metrics_by_mode_spec(selected)
        policy_bootstraps[policy] = [
            paired_family_bootstrap(
                selected,
                mode,
                "date_price_provider_normalized_throughput",
                "date_price",
            )
            for mode in ("family", "chronological_family")
        ]

    multi_tier = [key for key, values in tiers.items() if len(values) > 1]
    return predictions, {
        "endpoint_models_with_multiple_service_tiers": len(multi_tier),
        "calibration_checkpoints_changed_by_tier_separation": changed_checkpoints,
        "mixed_over_default_model_throughput_ratio": distribution_summary(ratios),
        "heldout_metrics": policy_metrics,
        "paired_bootstraps": policy_bootstraps,
        "current_default_normalization_models": len(current_normalized),
        "mixed_tier_normalization_models": len(mixed_normalized),
    }


def focal_rows(
    model_stability: list[dict[str, Any]],
    refresh: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    within = {row["openrouter_model_id"]: row for row in model_stability}
    across = {row["openrouter_model_id"]: row for row in refresh}
    output = []
    for model_id, name in FOCAL_MODELS.items():
        output.append(
            {
                "model": name,
                "openrouter_model_id": model_id,
                "within_week_dates": within[model_id]["dates"],
                "within_week_median_tps": within[model_id]["median_tps"],
                "within_week_max_over_min": within[model_id]["max_over_min"],
                "refreshes": across[model_id]["snapshots"],
                "refresh_median_tps": across[model_id]["median_tps"],
                "refresh_max_over_min": across[model_id]["max_over_min"],
            }
        )
    return output


def main() -> None:
    daily = rows(DAILY)
    source_dates = {row["snapshot_date"] for row in daily}
    if len(source_dates) != 1:
        raise ValueError(
            f"Expected one OpenRouter observation date, found {source_dates}"
        )
    observation_date = next(iter(source_dates))
    history = rows(DAILY_HISTORY)
    manifest = rows(HISTORY_MANIFEST)
    models = rows(MODELS)
    providers = rows(PROVIDERS)
    calibration = rows(CALIBRATION)
    parameter_audit = json.loads(PARAMETER_AUDIT.read_text(encoding="utf-8"))

    endpoints, model_stability = endpoint_and_model_stability(daily)
    refresh = refresh_stability(history)
    predictions, tier_audit = mixed_tier_counterfactual(
        daily, providers, calibration
    )
    write_csv(ENDPOINT_OUTPUT, endpoints)
    write_csv(MODEL_OUTPUT, model_stability)
    write_csv(REFRESH_OUTPUT, refresh)
    write_csv(TIER_PREDICTIONS, predictions)

    tier_counts = Counter(row["service_tier"] for row in daily)
    endpoint_ratios = [
        row["max_over_min"] for row in endpoints if row["observations"] >= 4
    ]
    model_ratios = [
        row["max_over_min"] for row in model_stability if row["dates"] >= 4
    ]
    refresh_ratios = [
        row["max_over_min"]
        for row in refresh
        if row["snapshots"] == len(manifest)
    ]
    unmatched_rows = [
        row for row in daily if row["endpoint_metadata_match"] != "True"
    ]
    refresh_times = sorted(
        datetime.fromisoformat(row["fetched_at_utc"]) for row in manifest
    )
    refresh_span_hours = (
        (refresh_times[-1] - refresh_times[0]).total_seconds() / 3600
        if len(refresh_times) > 1
        else 0.0
    )
    unmatched_endpoint_models = {
        (row["openrouter_model_id"], row["endpoint_id"])
        for row in unmatched_rows
    }

    default_family = tier_audit["heldout_metrics"]["default_only"]["family"]
    default_chronological = tier_audit["heldout_metrics"]["default_only"][
        "chronological_family"
    ]
    result = {
        "metadata": {
            "snapshot_date": observation_date,
            "compatibility_filename_date": COMPATIBILITY_FILE_DATE,
            "question": "Does lossless daily and service-tier-separated OpenRouter throughput add a stable parameter-count signal beyond current price?",
            "tier_policy": "All tiers preserved; only default service feeds model aggregates and regression.",
            "history_policy": "Every committed refresh plus the current worktree is archived by fetched_at timestamp. Repeated rolling windows are correlated, not independent samples.",
        },
        "inventory": {
            "immutable_snapshots": len(manifest),
            "current_models": len(models),
            "current_provider_endpoints": len(providers),
            "current_daily_rows": len(daily),
            "history_daily_rows": len(history),
            "service_tier_row_counts": dict(sorted(tier_counts.items())),
            "endpoint_models_with_multiple_service_tiers": tier_audit[
                "endpoint_models_with_multiple_service_tiers"
            ],
            "current_daily_rows_without_endpoint_metadata": len(unmatched_rows),
            "current_daily_unmatched_endpoint_models": len(unmatched_endpoint_models),
            "calibration_checkpoints": len(calibration),
            "calibration_families": len({row["family"] for row in calibration}),
        },
        "temporal_stability": {
            "endpoint_default_tier_with_at_least_four_days_max_over_min": distribution_summary(
                endpoint_ratios
            ),
            "model_daily_provider_median_with_at_least_four_days_max_over_min": distribution_summary(
                model_ratios
            ),
            "model_default_tier_median_across_all_refreshes_max_over_min": distribution_summary(
                refresh_ratios
            ),
            "focal_models": focal_rows(model_stability, refresh),
        },
        "service_tier_counterfactual": tier_audit,
        "corrected_default_tier_backtest": {
            "family": {
                "date_only": default_family["date_only"],
                "date_price": default_family["date_price"],
                "date_price_plus_normalized_tok_s": default_family[
                    "date_price_provider_normalized_throughput"
                ],
            },
            "chronological_family": {
                "date_only": default_chronological["date_only"],
                "date_price": default_chronological["date_price"],
                "date_price_plus_normalized_tok_s": default_chronological[
                    "date_price_provider_normalized_throughput"
                ],
            },
            "promotion_gate_from_main_audit": parameter_audit["conclusion"],
        },
        "decision": {
            "recommended_incremental_tok_s_weight": parameter_audit["conclusion"][
                "recommended_incremental_tok_s_weight_in_live_ensemble"
            ],
            "change_live_forecast": False,
            "reason": "Separating service tiers fixes a real data defect and preserves the complete time series, but corrected default-tier tok/s still fails the predeclared family-bootstrap and chronological-direction promotion gate after price. Temporal and refresh volatility provide an additional operational-noise floor.",
        },
        "limitations": [
            f"The {len(manifest)} refreshes span {refresh_span_hours:.1f} hours and contain heavily overlapping rolling one-week windows.",
            "OpenRouter throughput remains a serving-stack measurement affected by batching, hardware, quantization, speculative decoding, traffic, and provider policy.",
            f"{len(unmatched_endpoint_models)} endpoint/model pairs appear in throughput history without public endpoint metadata; their measurements are retained with an explicit unmatched flag.",
            "Current prices are not launch-vintage historical prices and remain correlated with product strategy and active inference cost.",
        ],
        "source_manifest": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (
                DAILY,
                DAILY_HISTORY,
                HISTORY_MANIFEST,
                MODELS,
                PROVIDERS,
                CALIBRATION,
                PARAMETER_AUDIT,
            )
        },
        "outputs": {
            "endpoint_stability": str(ENDPOINT_OUTPUT.relative_to(ROOT)),
            "model_stability": str(MODEL_OUTPUT.relative_to(ROOT)),
            "refresh_stability": str(REFRESH_OUTPUT.relative_to(ROOT)),
            "tier_counterfactual_predictions": str(TIER_PREDICTIONS.relative_to(ROOT)),
        },
    }
    RESULT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(RESULT),
                "snapshots": len(manifest),
                "current_daily_rows": len(daily),
                "history_daily_rows": len(history),
                "multi_tier_endpoint_models": tier_audit[
                    "endpoint_models_with_multiple_service_tiers"
                ],
                "tok_s_weight": result["decision"][
                    "recommended_incremental_tok_s_weight"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
