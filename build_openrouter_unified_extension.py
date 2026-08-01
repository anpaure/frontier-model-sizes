#!/usr/bin/env python3
"""Append complete OpenRouter operational data to the canonical megafiles.

The original unified observations/measurements are never rewritten in place.
This produces an operationally enriched successor containing every original
row plus all OpenRouter model aggregates, provider endpoint observations, and
lossless dated provider-throughput observations.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
DATE = "2026-07-18"
OUT = ROOT / "outputs/019f6c42-2d53-7743-ab07-6293e2618dd7"
BASE_OBSERVATIONS = OUT / "unified_model_observations_compute_enriched_2026-07-17.csv"
BASE_MEASUREMENTS = OUT / "unified_model_measurements_long_compute_enriched_2026-07-17.csv"
BASE_MANIFEST = OUT / "unified_model_source_manifest_compute_enriched_2026-07-17.csv"
MODEL_SIGNALS = ROOT / f"sources/openrouter_model_signals_{DATE}.csv"
PROVIDER_SIGNALS = ROOT / f"sources/openrouter_provider_signals_{DATE}.csv"
TIER_SIGNALS = ROOT / f"sources/openrouter_endpoint_tier_signals_{DATE}.csv"
DAILY_SIGNALS = ROOT / f"sources/openrouter_throughput_daily_{DATE}.csv"
MODEL_HISTORY = ROOT / f"sources/openrouter_model_snapshot_history_{DATE}.csv"
PROVIDER_HISTORY = ROOT / f"sources/openrouter_provider_snapshot_history_{DATE}.csv"
TIER_HISTORY = ROOT / f"sources/openrouter_endpoint_tier_snapshot_history_{DATE}.csv"
DAILY_HISTORY = ROOT / f"sources/openrouter_throughput_daily_history_{DATE}.csv"
HISTORY_MANIFEST = ROOT / f"sources/openrouter_snapshot_history_manifest_{DATE}.csv"
RAW_SNAPSHOT = ROOT / f"sources/openrouter_operational_snapshot_{DATE}.json.gz"
COLLECTION_AUDIT = OUT / f"openrouter_collection_audit_{DATE}.json"
MATCH_AUDIT = OUT / f"openrouter_epoch_match_audit_{DATE}.csv"
SIGNAL_RESULT = OUT / f"openrouter_parameter_signal_backtest_{DATE}.json"
TEMPORAL_RESULT = OUT / f"openrouter_temporal_stability_audit_{DATE}.json"
REQUEST_WEIGHTED_RESULT = OUT / f"openrouter_request_weighted_operational_audit_{DATE}.json"
REQUEST_WEIGHTED_PREDICTIONS = OUT / f"openrouter_request_weighted_operational_predictions_{DATE}.csv"
ACTIVE_PRICE_RESULT = OUT / f"openrouter_active_price_audit_{DATE}.json"
ACTIVE_PRICE_MATCHES = OUT / f"openrouter_active_parameter_match_audit_{DATE}.csv"
ACTIVE_PRICE_PREDICTIONS = OUT / f"openrouter_active_price_predictions_{DATE}.csv"
ACTIVE_PRICE_TARGETS = OUT / f"openrouter_active_price_targets_{DATE}.csv"
HISTORICAL_PRICE_RAW = ROOT / f"sources/openrouter_historical_price_ledger_{DATE}.json.gz"
HISTORICAL_PRICE_POINTS = ROOT / f"sources/openrouter_historical_price_change_points_{DATE}.csv"
HISTORICAL_PRICE_METADATA = ROOT / f"sources/openrouter_historical_price_collection_metadata_{DATE}.json"
HISTORICAL_PRICE_MATCHES = OUT / f"openrouter_historical_price_match_audit_{DATE}.csv"
HISTORICAL_PRICE_PREDICTIONS = OUT / f"openrouter_historical_price_backtest_predictions_{DATE}.csv"
HISTORICAL_PRICE_TARGETS = OUT / f"openrouter_historical_price_frontier_targets_{DATE}.csv"
HISTORICAL_PRICE_RESULT = OUT / f"openrouter_historical_price_audit_{DATE}.json"
HF_ARCHITECTURE_SNAPSHOT = ROOT / f"sources/huggingface_architecture_config_snapshot_{DATE}.json.gz"
HF_ARCHITECTURE_SIGNALS = ROOT / f"sources/huggingface_architecture_config_signals_{DATE}.csv"
HF_ARCHITECTURE_AUDIT = OUT / f"huggingface_architecture_config_collection_audit_{DATE}.json"
OFFICIAL_SNAPSHOT = ROOT / f"sources/openrouter_official_endpoint_snapshot_{DATE}.json.gz"
OFFICIAL_PRICES = ROOT / f"sources/openrouter_official_endpoint_prices_{DATE}.csv"
OFFICIAL_COMPARISON = OUT / f"openrouter_official_endpoint_crosscheck_{DATE}.csv"
OFFICIAL_AUDIT = OUT / f"openrouter_official_endpoint_audit_{DATE}.json"
NO_COT_EXACT_DATE_AUDIT = OUT / f"no_cot_exact_date_audit_{DATE}.json"
NO_COT_EXACT_DATE_MODELS = OUT / f"no_cot_exact_date_model_audit_{DATE}.csv"
FRONTIER_PRIMARY_EVIDENCE = ROOT / f"sources/frontier_primary_evidence_{DATE}.csv"
FRONTIER_PRIMARY_METADATA = ROOT / f"sources/frontier_primary_evidence_collection_metadata_{DATE}.json"
FRONTIER_PRIMARY_AUDIT = OUT / f"frontier_primary_evidence_audit_{DATE}.json"
FRONTIER_PRIMARY_CONTROLS = OUT / f"frontier_primary_evidence_controls_{DATE}.csv"
OPUS5_EVIDENCE = ROOT / "sources/claude_opus_5_evidence_2026-07-31.json"

OBSERVATIONS = OUT / f"unified_model_observations_operational_enriched_{DATE}.csv"
MEASUREMENTS = OUT / f"unified_model_measurements_long_operational_enriched_{DATE}.csv"
MANIFEST = OUT / f"unified_model_source_manifest_operational_enriched_{DATE}.csv"
SUMMARY = OUT / f"unified_model_data_summary_operational_enriched_{DATE}.json"


def portable_path(path: Path) -> str:
    """Serialize local provenance without exposing the workstation account."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        try:
            return f"~/{resolved.relative_to(Path.home().resolve()).as_posix()}"
        except ValueError:
            return resolved.as_posix()


MODEL_NUMERIC_METRICS = {
    "context_length_tokens": "tokens",
    "endpoint_count": "endpoints",
    "endpoint_count_with_price": "endpoints",
    "endpoint_count_with_throughput_30m": "endpoints",
    "endpoint_count_with_throughput_1w": "endpoints",
    "provider_count_with_throughput_1w": "providers",
    "prompt_price_min_usd_per_mtoken": "USD/million tokens",
    "prompt_price_median_usd_per_mtoken": "USD/million tokens",
    "prompt_price_max_usd_per_mtoken": "USD/million tokens",
    "completion_price_min_usd_per_mtoken": "USD/million tokens",
    "completion_price_median_usd_per_mtoken": "USD/million tokens",
    "completion_price_max_usd_per_mtoken": "USD/million tokens",
    "blended_price_geomean_median_usd_per_mtoken": "USD/million tokens",
    "throughput_best_provider_median_tps_1w": "tokens/second",
    "throughput_median_provider_median_tps_1w": "tokens/second",
    "throughput_worst_provider_median_tps_1w": "tokens/second",
    "throughput_best_p50_tps_30m": "tokens/second",
    "throughput_median_p50_tps_30m": "tokens/second",
    "provider_normalized_throughput_ratio": "ratio",
}

PROVIDER_NUMERIC_METRICS = {
    "model_context_length_tokens": "tokens",
    "endpoint_context_length_tokens": "tokens",
    "max_completion_tokens": "tokens",
    "prompt_usd_per_mtoken": "USD/million tokens",
    "completion_usd_per_mtoken": "USD/million tokens",
    "cache_read_usd_per_mtoken": "USD/million tokens",
    "cache_write_usd_per_mtoken": "USD/million tokens",
    "p50_throughput_tps_30m": "tokens/second",
    "p75_throughput_tps_30m": "tokens/second",
    "p90_throughput_tps_30m": "tokens/second",
    "p50_latency_seconds_30m": "seconds",
    "request_count_30m": "requests",
    "window_minutes": "minutes",
    "throughput_daily_observations_1w": "daily observations",
    "throughput_median_tps_1w": "tokens/second",
    "throughput_mean_tps_1w": "tokens/second",
    "throughput_min_tps_1w": "tokens/second",
    "throughput_max_tps_1w": "tokens/second",
}

TIER_NUMERIC_METRICS = {
    "endpoint_context_length_tokens": "tokens",
    "max_prompt_tokens": "tokens",
    "max_completion_tokens": "tokens",
    "prompt_usd_per_mtoken": "USD/million tokens",
    "completion_usd_per_mtoken": "USD/million tokens",
    "cache_read_usd_per_mtoken": "USD/million tokens",
    "cache_write_usd_per_mtoken": "USD/million tokens",
    "high_context_min_prompt_tokens": "tokens",
    "high_context_prompt_usd_per_mtoken": "USD/million tokens",
    "high_context_completion_usd_per_mtoken": "USD/million tokens",
    "high_context_cache_read_usd_per_mtoken": "USD/million tokens",
    "high_context_cache_write_usd_per_mtoken": "USD/million tokens",
    "p50_throughput_tps_30m": "tokens/second",
    "p75_throughput_tps_30m": "tokens/second",
    "p90_throughput_tps_30m": "tokens/second",
    "p95_throughput_tps_30m": "tokens/second",
    "p99_throughput_tps_30m": "tokens/second",
    "p50_latency_seconds_30m": "seconds",
    "p75_latency_seconds_30m": "seconds",
    "p90_latency_seconds_30m": "seconds",
    "p95_latency_seconds_30m": "seconds",
    "p99_latency_seconds_30m": "seconds",
    "request_count_30m": "requests",
    "window_minutes": "minutes",
}

DAILY_NUMERIC_METRICS = {
    "throughput_tps": "tokens/second",
}


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def csv_record_count(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="raise", lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_token(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_")


def source_record(row: dict[str, str]) -> str:
    return json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def identity(
    model_id: str,
    audit_by_id: dict[str, dict[str, str]],
    base_by_checkpoint: dict[str, str],
) -> tuple[str, str, str, str, str]:
    audit = audit_by_id[model_id]
    checkpoint = audit["canonical_checkpoint_id"] or f"openrouter:{model_id}"
    if model_id.startswith("anthropic/claude-opus-4.") and not model_id.endswith("-fast"):
        base = "base:anthropic:claude-opus-4.5-4.8"
    elif model_id in {
        "openai/gpt-5", "openai/gpt-5.1", "openai/gpt-5.2", "openai/gpt-5.3-chat", "openai/gpt-5.4", "openai/gpt-5.5"
    }:
        base = "base:openai:gpt-5-through-5.5"
    else:
        base = base_by_checkpoint.get(checkpoint, f"base:{checkpoint}")
    display = audit["epoch_model_name"] or audit["openrouter_model_name"]
    matched_epoch = audit["epoch_model_name"]
    link_level = "checkpoint" if matched_epoch else ""
    return checkpoint, base, display, matched_epoch, link_level


def make_observation(
    base_fields: list[str],
    extra_fields: list[str],
    raw: dict[str, str],
    audit: dict[str, str],
    checkpoint: str,
    base: str,
    display: str,
    matched_epoch: str,
    link_level: str,
    *,
    kind: str,
    observation_id: str,
) -> dict[str, Any]:
    row = {field: "" for field in base_fields + extra_fields}
    is_model = kind == "model"
    is_daily = kind == "daily"
    is_tier = kind == "tier"
    source_url = raw.get("catalog_source_url") or raw.get("endpoint_source_url") or raw.get("throughput_source_url")
    source = (
        "OpenRouter"
        if is_model
        else "OpenRouter Provider Daily Stats"
        if is_daily
        else "OpenRouter Provider Tier Stats"
        if is_tier
        else "OpenRouter Provider Stats"
    )
    record_type = (
        "model_configuration"
        if is_model
        else "provider_operational_time_series_point"
        if is_daily
        else "provider_service_tier_measurement"
        if is_tier
        else "provider_operational_measurement"
    )
    benchmark = (
        "OpenRouter model aggregate"
        if is_model
        else "OpenRouter daily provider throughput"
        if is_daily
        else "OpenRouter endpoint service tier"
        if is_tier
        else "OpenRouter provider endpoint"
    )
    row.update(
        {
            "observation_id": observation_id,
            "source": source,
            "record_type": record_type,
            "dataset": (
                "OpenRouter public provider throughput time series"
                if is_daily
                else "OpenRouter public endpoint service-tier prices and statistics"
                if is_tier
                else "OpenRouter public catalog and frontend provider statistics"
            ),
            "snapshot_date": DATE,
            "source_locator": source_url,
            "source_model_name": raw["openrouter_model_name"],
            "source_configuration": "aggregate across provider endpoints" if is_model else "; ".join(
                value
                for value in (
                    f"provider={raw.get('provider_name', '')}" if raw.get("provider_name") else "",
                    f"quantization={raw.get('quantization', '')}" if raw.get("quantization") else "",
                    f"variant={raw.get('variant', '')}" if raw.get("variant") else "",
                    f"service_tier={raw.get('service_tier', '')}" if raw.get("service_tier") else "",
                    f"observation_date={raw.get('observation_date', '')}" if raw.get("observation_date") else "",
                )
                if value
            ),
            "source_organization": raw.get("author", ""),
            "source_provider": "OpenRouter" if is_model else raw.get("provider_name", ""),
            "model_level_include": "yes" if is_model else "no",
            "model_level_selection_reason": (
                "one aggregate row per OpenRouter model ID"
                if is_model
                else "dated provider/tier point retained as correlated operational detail"
                if is_daily
                else "endpoint service tier retained separately; no prompt-length or tier pooling"
                if is_tier
                else "provider endpoint retained as operational detail"
            ),
            "canonical_checkpoint_id": checkpoint,
            "canonical_display_name": display,
            "canonical_base_id": base,
            "canonical_release_date": audit["epoch_release_date"] or raw.get("created_date", ""),
            "canonical_release_date_source": "Epoch" if audit["epoch_release_date"] else "OpenRouter catalog",
            "source_release_date": raw.get("created_date", ""),
            "source_release_date_precision": "day",
            "matched_epoch_model": matched_epoch,
            "epoch_match_status": audit["match_status"],
            "epoch_match_method": "manual explicit OpenRouter ID map" if matched_epoch else "unmatched",
            "epoch_match_confidence": "high" if matched_epoch else "",
            "epoch_link_level": link_level,
            "benchmark_name": benchmark,
            "total_parameters_b": audit["total_parameters_b"],
            "parameter_value_source": audit["parameter_value_source"],
            "source_url": source_url,
            "notes": (
                audit["match_reason"]
                + (
                    "; endpoint metadata absent from public endpoint payload"
                    if is_daily and raw.get("endpoint_metadata_match") != "True"
                    else ""
                )
            ),
            "source_record_json": source_record(raw),
        }
    )
    for field, value in raw.items():
        row[f"or_{field}"] = value
    if is_model:
        row["or_provider_normalized_throughput_ratio"] = audit["provider_normalized_throughput_ratio"]
        row["or_normalization_controls_used"] = audit["normalization_controls_used"]
    return row


def measurement_rows(
    raw: dict[str, str],
    checkpoint: str,
    base: str,
    matched_epoch: str,
    link_level: str,
    observation_id: str,
    *,
    kind: str,
    normalized_ratio: str = "",
) -> list[dict[str, str]]:
    metrics = dict(
        MODEL_NUMERIC_METRICS
        if kind == "model"
        else DAILY_NUMERIC_METRICS
        if kind == "daily"
        else TIER_NUMERIC_METRICS
        if kind == "tier"
        else PROVIDER_NUMERIC_METRICS
    )
    values = dict(raw)
    if kind == "model":
        values["provider_normalized_throughput_ratio"] = normalized_ratio
    output = []
    for metric_name, unit in metrics.items():
        value = values.get(metric_name, "")
        if value == "":
            continue
        output.append(
            {
                "measurement_id": f"or:m:{stable_token(observation_id)}:{metric_name}",
                "observation_id": observation_id,
                "canonical_checkpoint_id": checkpoint,
                "canonical_base_id": base,
                "source": (
                    "OpenRouter"
                    if kind == "model"
                    else "OpenRouter Provider Daily Stats"
                    if kind == "daily"
                    else "OpenRouter Provider Tier Stats"
                    if kind == "tier"
                    else "OpenRouter Provider Stats"
                ),
                "source_model_name": raw["openrouter_model_name"],
                "source_configuration": "aggregate" if kind == "model" else f"{raw.get('provider_name', '')}; {raw.get('quantization', '')}; {raw.get('variant', '')}",
                "matched_epoch_model": matched_epoch,
                "epoch_link_level": link_level,
                "benchmark_name": (
                    "OpenRouter model aggregate"
                    if kind == "model"
                    else "OpenRouter daily provider throughput"
                    if kind == "daily"
                    else "OpenRouter endpoint service tier"
                    if kind == "tier"
                    else "OpenRouter provider endpoint"
                ),
                "metric_name": f"openrouter.{metric_name}",
                "value": value,
                "value_raw": value,
                "unit": unit,
                "ci_low": "",
                "ci_high": "",
                "measurement_notes": raw.get("throughput_source_url") or raw.get("endpoint_source_url") or raw.get("catalog_source_url", ""),
            }
        )
    return output


def make_historical_price_observation(
    base_fields: list[str],
    extra_fields: list[str],
    raw: dict[str, str],
    audit: dict[str, str] | None,
    base_by_checkpoint: dict[str, str],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    model_id = raw["openrouter_model_id"]
    if audit is not None:
        checkpoint = audit["canonical_checkpoint_id"] or f"openrouter:{model_id}"
        base = base_by_checkpoint.get(checkpoint, f"base:{checkpoint}")
        display = audit["epoch_model_name"] or raw["openrouter_model_name"]
        matched_epoch = audit["epoch_model_name"]
        link_level = "checkpoint" if matched_epoch else ""
        release_date = audit["epoch_release_date"] or raw["first_seen"]
        release_source = "Epoch" if audit["epoch_release_date"] else "OpenRouter first seen"
        parameters = audit["total_parameters_b"]
        parameter_source = audit["parameter_value_source"]
        match_status = audit["match_status"]
    else:
        checkpoint = f"openrouter:{model_id}"
        base = f"base:openrouter:{model_id}"
        display = raw["openrouter_model_name"]
        matched_epoch = ""
        link_level = ""
        release_date = raw["first_seen"]
        release_source = "OpenRouter first seen"
        parameters = ""
        parameter_source = ""
        match_status = "historical_unmatched"
    source_url = (
        f"{raw['source_repository']}/blob/{raw['source_commit']}/"
        "data/history/prices.json"
    )
    observation_id = (
        f"openrouter:historical-price:{model_id}:{raw['change_index']}"
    )
    row = {field: "" for field in base_fields + extra_fields}
    row.update(
        {
            "observation_id": observation_id,
            "source": "OpenRouter Historical Prices",
            "record_type": "model_price_change_point",
            "dataset": "Hash-pinned OpenRouter /api/v1/models price history",
            "snapshot_date": raw["effective_date"],
            "source_locator": source_url,
            "source_model_name": raw["openrouter_model_name"],
            "source_configuration": (
                f"model_id={model_id}; change_index={raw['change_index']}; "
                f"effective_date={raw['effective_date']}"
            ),
            "source_organization": model_id.split("/", 1)[0],
            "source_provider": "OpenRouter",
            "model_level_include": "no",
            "model_level_selection_reason": (
                "lossless dated price change point; correlated time-series detail"
            ),
            "canonical_checkpoint_id": checkpoint,
            "canonical_display_name": display,
            "canonical_base_id": base,
            "canonical_release_date": release_date,
            "canonical_release_date_source": release_source,
            "source_release_date": raw["first_seen"],
            "source_release_date_precision": "day",
            "matched_epoch_model": matched_epoch,
            "epoch_match_status": match_status,
            "epoch_match_method": (
                "manual explicit OpenRouter ID map" if matched_epoch else "unmatched"
            ),
            "epoch_match_confidence": "high" if matched_epoch else "",
            "epoch_link_level": link_level,
            "benchmark_name": "OpenRouter historical model price",
            "total_parameters_b": parameters,
            "parameter_value_source": parameter_source,
            "source_url": source_url,
            "notes": (
                "Exact price change point reconstructed from committed official "
                "/api/v1/models responses; free and unavailable states retained"
            ),
            "source_record_json": source_record(raw),
        }
    )
    for field, value in raw.items():
        row[f"orh_{field}"] = value

    measurements: list[dict[str, str]] = []
    metrics = {
        "prompt_price_usd_per_mtoken": "openrouter.history.prompt_price_usd_per_mtoken",
        "completion_price_usd_per_mtoken": "openrouter.history.completion_price_usd_per_mtoken",
        "blended_geomean_price_usd_per_mtoken": "openrouter.history.blended_geomean_price_usd_per_mtoken",
    }
    for field, metric_name in metrics.items():
        value = raw[field]
        if value == "":
            continue
        measurements.append(
            {
                "measurement_id": (
                    f"orh:m:{stable_token(observation_id)}:{field}"
                ),
                "observation_id": observation_id,
                "canonical_checkpoint_id": checkpoint,
                "canonical_base_id": base,
                "source": "OpenRouter Historical Prices",
                "source_model_name": raw["openrouter_model_name"],
                "source_configuration": (
                    f"change_index={raw['change_index']}; "
                    f"effective_date={raw['effective_date']}"
                ),
                "matched_epoch_model": matched_epoch,
                "epoch_link_level": link_level,
                "benchmark_name": "OpenRouter historical model price",
                "metric_name": metric_name,
                "value": value,
                "value_raw": value,
                "unit": "USD/million tokens",
                "ci_low": "",
                "ci_high": "",
                "measurement_notes": source_url,
            }
        )
    return row, measurements


def make_no_cot_date_audit_observation(
    base_fields: list[str],
    extra_fields: list[str],
    raw: dict[str, str],
    base_no_cot: dict[str, str],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    model = raw["model"]
    observation_id = f"nocot-date-audit:{stable_token(model)}"
    row = {field: "" for field in base_fields + extra_fields}
    row.update(
        {
            "observation_id": observation_id,
            "source": "No-CoT Exact-Date Audit",
            "record_type": "release_date_audit",
            "dataset": "No-CoT exact day-level release-date reconciliation",
            "snapshot_date": DATE,
            "source_locator": portable_path(NO_COT_EXACT_DATE_MODELS),
            "source_model_name": model,
            "source_configuration": (
                f"paper_month={raw['paper_month_date']}; "
                f"exact_release={raw['exact_release_date']}"
            ),
            "source_organization": base_no_cot["source_organization"],
            "source_provider": base_no_cot["source_provider"],
            "model_level_include": "no",
            "model_level_selection_reason": (
                "derived date-fidelity audit; linked to, but not counted as, "
                "an independent capability or parameter observation"
            ),
            "canonical_checkpoint_id": base_no_cot["canonical_checkpoint_id"],
            "canonical_display_name": base_no_cot["canonical_display_name"],
            "canonical_base_id": base_no_cot["canonical_base_id"],
            "canonical_release_date": raw["exact_release_date"],
            "canonical_release_date_source": raw["exact_date_source"],
            "source_release_date": raw["paper_month_date"],
            "source_release_date_precision": "month",
            "release_date_delta_days": raw["day_offset_from_month_start"],
            "matched_epoch_model": base_no_cot["matched_epoch_model"],
            "epoch_match_status": (
                "date_only_override"
                if raw["explicit_override"] == "True"
                else "existing_exact_checkpoint_crosswalk"
            ),
            "epoch_match_method": raw["parameter_join_policy"],
            "epoch_match_confidence": "high",
            "epoch_link_level": base_no_cot["epoch_link_level"],
            "benchmark_name": "No-CoT exact-date sensitivity",
            "source_url": raw["exact_date_source"],
            "notes": (
                "Date-only evidence. Parameter joins remain governed by the existing "
                f"checkpoint registry; policy={raw['parameter_join_policy']}"
            ),
            "source_record_json": source_record(raw),
        }
    )
    metric_units = {
        "day_offset_from_month_start": "days",
        "time_pareto_with_month_dates": "boolean",
        "time_pareto_with_exact_dates": "boolean",
        "token_pareto_with_month_dates": "boolean",
        "token_pareto_with_exact_dates": "boolean",
    }
    measurements = []
    for field, unit in metric_units.items():
        value = raw[field]
        numeric_value = (
            "1" if value == "True" else "0" if value == "False" else value
        )
        measurements.append(
            {
                "measurement_id": f"nocot-date:m:{stable_token(model)}:{field}",
                "observation_id": observation_id,
                "canonical_checkpoint_id": base_no_cot["canonical_checkpoint_id"],
                "canonical_base_id": base_no_cot["canonical_base_id"],
                "source": "No-CoT Exact-Date Audit",
                "source_model_name": model,
                "source_configuration": "exact-date sensitivity",
                "matched_epoch_model": base_no_cot["matched_epoch_model"],
                "epoch_link_level": base_no_cot["epoch_link_level"],
                "benchmark_name": "No-CoT exact-date sensitivity",
                "metric_name": f"nocot.date_audit.{field}",
                "value": numeric_value,
                "value_raw": value,
                "unit": unit,
                "ci_low": "",
                "ci_high": "",
                "measurement_notes": raw["exact_date_source"],
            }
        )
    return row, measurements


def make_no_cot_date_law_observation(
    base_fields: list[str],
    extra_fields: list[str],
    audit: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    observation_id = "nocot-date-audit:law"
    row = {field: "" for field in base_fields + extra_fields}
    row.update(
        {
            "observation_id": observation_id,
            "source": "No-CoT Exact-Date Audit",
            "record_type": "scaling_law_date_sensitivity",
            "dataset": "No-CoT exact day-level release-date reconciliation",
            "snapshot_date": DATE,
            "source_locator": portable_path(NO_COT_EXACT_DATE_AUDIT),
            "source_model_name": "__exact_date_law__",
            "source_configuration": "paper law × exact-date/month-date deterministic Pareto-OLS slope ratio",
            "model_level_include": "no",
            "model_level_selection_reason": "derived source-fidelity sensitivity; no independent evidence weight",
            "benchmark_name": "No-CoT exact-date scaling-law adjustment",
            "source_url": portable_path(NO_COT_EXACT_DATE_AUDIT),
            "notes": audit["decision"]["reason"],
            "source_record_json": json.dumps(audit, sort_keys=True, separators=(",", ":")),
        }
    )
    inventory = audit["inventory"]
    time_law = audit["time_horizon"]["adjusted_reported_law"]
    token_law = audit["token_horizon"]["adjusted_reported_law"]
    metric_values = {
        "no_cot_models": (inventory["no_cot_models"], "models"),
        "models_with_day_level_dates": (inventory["models_with_day_level_dates"], "models"),
        "models_remaining_month_only": (inventory["models_remaining_month_only"], "models"),
        "explicit_date_only_overrides": (inventory["explicit_date_only_overrides"], "models"),
        "parameter_identities_added_by_overrides": (inventory["parameter_identities_added_by_overrides"], "models"),
        "time_paper_reported_point_days": (time_law["paper_reported_point_days"], "days"),
        "time_adjusted_point_days": (time_law["adjusted_point_days"], "days"),
        "time_exact_date_adjustment_ratio": (time_law["exact_date_adjustment_ratio"], "ratio"),
        "token_paper_reported_point_days": (token_law["paper_reported_point_days"], "days"),
        "token_adjusted_point_days": (token_law["adjusted_point_days"], "days"),
        "token_exact_date_adjustment_ratio": (token_law["exact_date_adjustment_ratio"], "ratio"),
    }
    measurements = []
    for metric, (value, unit) in metric_values.items():
        measurements.append(
            {
                "measurement_id": f"nocot-date:m:law:{metric}",
                "observation_id": observation_id,
                "canonical_checkpoint_id": "",
                "canonical_base_id": "",
                "source": "No-CoT Exact-Date Audit",
                "source_model_name": "__exact_date_law__",
                "source_configuration": "exact-date sensitivity",
                "matched_epoch_model": "",
                "epoch_link_level": "",
                "benchmark_name": "No-CoT exact-date scaling-law adjustment",
                "metric_name": f"nocot.date_audit.{metric}",
                "value": str(value),
                "value_raw": str(value),
                "unit": unit,
                "ci_low": "",
                "ci_high": "",
                "measurement_notes": portable_path(NO_COT_EXACT_DATE_AUDIT),
            }
        )
    return row, measurements


def model_identity(
    model: str, identity_by_name: dict[str, dict[str, str]]
) -> tuple[str, str, str]:
    source = identity_by_name.get(model)
    if source is None:
        return "", "", model
    return (
        source["canonical_checkpoint_id"],
        source["canonical_base_id"],
        source["canonical_display_name"] or model,
    )


def make_primary_evidence_observation(
    base_fields: list[str],
    extra_fields: list[str],
    raw: dict[str, str],
    identity_by_name: dict[str, dict[str, str]],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    checkpoint, base, display = model_identity(raw["model"], identity_by_name)
    observation_id = f"frontier-primary:evidence:{raw['evidence_id']}"
    row = {field: "" for field in base_fields + extra_fields}
    row.update(
        {
            "observation_id": observation_id,
            "source": "Frontier Primary Evidence",
            "record_type": raw["evidence_type"],
            "dataset": "First-party frontier target evidence",
            "snapshot_date": DATE,
            "source_locator": portable_path(FRONTIER_PRIMARY_EVIDENCE),
            "source_model_name": raw["model"],
            "source_configuration": (
                f"comparator={raw['comparator_model']}; "
                f"identity_policy={raw['parameter_identity_policy']}"
            ),
            "source_organization": raw["developer"],
            "source_provider": raw["developer"],
            "model_level_include": "no",
            "model_level_selection_reason": raw["live_weight_policy"],
            "canonical_checkpoint_id": checkpoint,
            "canonical_display_name": display,
            "canonical_base_id": base,
            "canonical_release_date": raw["source_date"],
            "canonical_release_date_source": "first-party evidence publication date",
            "source_release_date": raw["source_date"],
            "source_release_date_precision": "day",
            "benchmark_name": raw["metric_name"] or raw["evidence_type"],
            "source_url": raw["source_url"],
            "notes": raw["claim_summary"],
            "source_record_json": source_record(raw),
        }
    )
    if raw["value"] == "":
        return row, []
    measurement = {
        "measurement_id": f"frontier-primary:m:evidence:{raw['evidence_id']}",
        "observation_id": observation_id,
        "canonical_checkpoint_id": checkpoint,
        "canonical_base_id": base,
        "source": "Frontier Primary Evidence",
        "source_model_name": raw["model"],
        "source_configuration": raw["page_or_section"],
        "matched_epoch_model": "",
        "epoch_link_level": "",
        "benchmark_name": raw["metric_name"],
        "metric_name": f"frontier.primary.{raw['metric_name']}",
        "value": raw["value"],
        "value_raw": raw["value"],
        "unit": raw["unit"],
        "ci_low": "",
        "ci_high": "",
        "measurement_notes": raw["claim_summary"],
    }
    return row, [measurement]


def numeric_leaves(value: Any, prefix: str = "") -> list[tuple[str, float, str]]:
    leaves: list[tuple[str, float, str]] = []
    if isinstance(value, bool):
        leaves.append((prefix, float(value), "boolean"))
    elif isinstance(value, (int, float)) and math.isfinite(float(value)):
        leaves.append((prefix, float(value), "derived audit value"))
    elif isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            leaves.extend(numeric_leaves(child, child_prefix))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_prefix = f"{prefix}.{index}" if prefix else str(index)
            leaves.extend(numeric_leaves(child, child_prefix))
    return leaves


def make_primary_audit_observation(
    base_fields: list[str],
    extra_fields: list[str],
    audit: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    observation_id = "frontier-primary:audit"
    row = {field: "" for field in base_fields + extra_fields}
    row.update(
        {
            "observation_id": observation_id,
            "source": "Frontier Primary Evidence Audit",
            "record_type": "statistical_signal_audit",
            "dataset": "Frontier primary-evidence parameter-mapping audit",
            "snapshot_date": DATE,
            "source_locator": portable_path(FRONTIER_PRIMARY_AUDIT),
            "source_model_name": "__frontier_primary_evidence_audit__",
            "source_configuration": audit["metadata"]["heldout_rule"],
            "model_level_include": "no",
            "model_level_selection_reason": "derived statistical audit; never an independent model observation",
            "benchmark_name": "Frontier primary-evidence audit",
            "source_url": portable_path(FRONTIER_PRIMARY_AUDIT),
            "notes": audit["decision"]["reason"],
            "source_record_json": json.dumps(
                audit, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ),
        }
    )
    sections = {
        key: audit[key]
        for key in (
            "inventory",
            "official_measurements",
            "current_mapping",
            "heldout_backtest",
            "sol_mapping_sensitivity",
            "promotion_gates",
            "decision",
        )
    }
    measurements = []
    for path, value, unit in numeric_leaves(sections):
        measurements.append(
            {
                "measurement_id": f"frontier-primary:m:audit:{stable_token(path)}",
                "observation_id": observation_id,
                "canonical_checkpoint_id": "",
                "canonical_base_id": "",
                "source": "Frontier Primary Evidence Audit",
                "source_model_name": "__frontier_primary_evidence_audit__",
                "source_configuration": "chronological developer-held-out mapping audit",
                "matched_epoch_model": "",
                "epoch_link_level": "",
                "benchmark_name": "Frontier primary-evidence audit",
                "metric_name": f"frontier.primary.audit.{path}",
                "value": str(value),
                "value_raw": str(value),
                "unit": unit,
                "ci_low": "",
                "ci_high": "",
                "measurement_notes": portable_path(FRONTIER_PRIMARY_AUDIT),
            }
        )
    return row, measurements


def make_primary_control_observation(
    base_fields: list[str],
    extra_fields: list[str],
    raw: dict[str, str],
    index: int,
    identity_by_name: dict[str, dict[str, str]],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    checkpoint, base, display = model_identity(raw["model"], identity_by_name)
    observation_id = (
        f"frontier-primary:control:{index:03d}:"
        f"{stable_token(raw['record_type'])}:{stable_token(raw['model'])}"
    )
    row = {field: "" for field in base_fields + extra_fields}
    row.update(
        {
            "observation_id": observation_id,
            "source": "Frontier Primary Evidence Audit",
            "record_type": raw["record_type"],
            "dataset": "Frontier primary-evidence control and sensitivity ledger",
            "snapshot_date": DATE,
            "source_locator": portable_path(FRONTIER_PRIMARY_CONTROLS),
            "source_model_name": raw["model"],
            "source_configuration": raw["series"],
            "model_level_include": "no",
            "model_level_selection_reason": raw["evidence_grade"],
            "canonical_checkpoint_id": checkpoint,
            "canonical_display_name": display,
            "canonical_base_id": base,
            "canonical_release_date": raw["release_date"],
            "canonical_release_date_source": "control-ledger release date",
            "source_release_date": raw["release_date"],
            "source_release_date_precision": "day" if raw["release_date"] else "",
            "benchmark_name": "No-CoT parameter-mapping control",
            "source_url": portable_path(FRONTIER_PRIMARY_CONTROLS),
            "notes": raw["interpretation"],
            "source_record_json": source_record(raw),
        }
    )
    metric_units = {
        "total_parameters_b": "billions of parameters",
        "nocot_time_horizon_minutes": "minutes",
        "horizon_ratio_vs_previous": "ratio",
    }
    measurements = []
    for field, unit in metric_units.items():
        value = raw[field]
        if value == "":
            continue
        measurements.append(
            {
                "measurement_id": f"frontier-primary:m:control:{index:03d}:{field}",
                "observation_id": observation_id,
                "canonical_checkpoint_id": checkpoint,
                "canonical_base_id": base,
                "source": "Frontier Primary Evidence Audit",
                "source_model_name": raw["model"],
                "source_configuration": raw["series"],
                "matched_epoch_model": "",
                "epoch_link_level": "",
                "benchmark_name": "No-CoT parameter-mapping control",
                "metric_name": f"frontier.primary.control.{field}",
                "value": value,
                "value_raw": value,
                "unit": unit,
                "ci_low": "",
                "ci_high": "",
                "measurement_notes": raw["evidence_grade"],
            }
        )
    return row, measurements


def opus5_unit(path: str) -> str:
    if path.endswith("_usd_per_mtok"):
        return "USD/million tokens"
    if "tokens" in path:
        return "tokens"
    if path.endswith("_page"):
        return "PDF page"
    if path.endswith("_score") or path.endswith(".score") or "eci" in path:
        return "index points"
    if path.endswith("_b"):
        return "billions of parameters"
    if path.endswith("_benchmarks") or path.endswith("_rows"):
        return "count"
    if path.endswith("disclosed") or path.endswith("_available"):
        return "boolean"
    return "source value"


def make_opus5_evidence_observation(
    base_fields: list[str],
    extra_fields: list[str],
    section: str,
    value: Any,
    identity_by_name: dict[str, dict[str, str]],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    checkpoint, base, display = model_identity("Claude Opus 5", identity_by_name)
    observation_id = f"opus5:evidence:{stable_token(section)}"
    row = {field: "" for field in base_fields + extra_fields}
    row.update(
        {
            "observation_id": observation_id,
            "source": "Claude Opus 5 Evidence",
            "record_type": "source_view",
            "dataset": "Claude Opus 5 normalized evidence bundle",
            "snapshot_date": "2026-07-31",
            "source_locator": f"{portable_path(OPUS5_EVIDENCE)}#{section}",
            "source_model_name": "Claude Opus 5",
            "source_configuration": section,
            "source_organization": "Anthropic",
            "source_provider": "Anthropic",
            "model_level_include": "no",
            "model_level_selection_reason": (
                "Correlated normalized source view; Opus 5 AA and ECI model-level "
                "signals already enter through their canonical source datasets."
            ),
            "canonical_checkpoint_id": checkpoint,
            "canonical_display_name": display or "Claude Opus 5",
            "canonical_base_id": base,
            "canonical_release_date": "2026-07-24",
            "canonical_release_date_source": "Anthropic first-party release",
            "source_release_date": "2026-07-24",
            "source_release_date_precision": "day",
            "benchmark_name": f"Claude Opus 5 evidence: {section}",
            "source_url": portable_path(OPUS5_EVIDENCE),
            "notes": (
                "Full normalized section retained; source bundle contains hashes and paths "
                "for every first-party and benchmark snapshot."
            ),
            "source_record_json": json.dumps(
                value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ),
        }
    )
    measurements: list[dict[str, str]] = []
    for path, numeric, _ in numeric_leaves(value):
        measurements.append(
            {
                "measurement_id": f"opus5:m:{stable_token(section)}:{stable_token(path)}",
                "observation_id": observation_id,
                "canonical_checkpoint_id": checkpoint,
                "canonical_base_id": base,
                "source": "Claude Opus 5 Evidence",
                "source_model_name": "Claude Opus 5",
                "source_configuration": section,
                "matched_epoch_model": "Claude Opus 5" if section == "epoch" else "",
                "epoch_link_level": "checkpoint" if section == "epoch" else "",
                "benchmark_name": f"Claude Opus 5 evidence: {section}",
                "metric_name": f"opus5.evidence.{section}.{path}",
                "value": str(numeric),
                "value_raw": str(numeric),
                "unit": opus5_unit(path),
                "ci_low": "",
                "ci_high": "",
                "measurement_notes": (
                    "Correlated source-view measurement; excluded from independent model-level weighting."
                ),
            }
        )
    return row, measurements


def main() -> None:
    base_fields, base_observations = read_csv(BASE_OBSERVATIONS)
    measurement_fields, base_measurements = read_csv(BASE_MEASUREMENTS)
    manifest_fields, base_manifest = read_csv(BASE_MANIFEST)
    model_fields, models = read_csv(MODEL_SIGNALS)
    provider_fields, providers = read_csv(PROVIDER_SIGNALS)
    tier_fields, tiers = read_csv(TIER_SIGNALS)
    daily_fields, daily = read_csv(DAILY_SIGNALS)
    historical_price_fields, historical_prices = read_csv(HISTORICAL_PRICE_POINTS)
    historical_price_metadata = json.loads(
        HISTORICAL_PRICE_METADATA.read_text(encoding="utf-8")
    )
    collection_audit = json.loads(COLLECTION_AUDIT.read_text(encoding="utf-8"))
    official_audit = json.loads(OFFICIAL_AUDIT.read_text(encoding="utf-8"))
    _, no_cot_date_models = read_csv(NO_COT_EXACT_DATE_MODELS)
    no_cot_date_audit = json.loads(NO_COT_EXACT_DATE_AUDIT.read_text(encoding="utf-8"))
    _, primary_evidence_rows = read_csv(FRONTIER_PRIMARY_EVIDENCE)
    _, primary_control_rows = read_csv(FRONTIER_PRIMARY_CONTROLS)
    primary_audit = json.loads(FRONTIER_PRIMARY_AUDIT.read_text(encoding="utf-8"))
    opus5_evidence = json.loads(OPUS5_EVIDENCE.read_text(encoding="utf-8"))
    _, audits = read_csv(MATCH_AUDIT)
    audit_by_id = {row["openrouter_model_id"]: row for row in audits}
    model_by_id = {row["openrouter_model_id"]: row for row in models}
    if set(audit_by_id) != set(model_by_id):
        raise ValueError("OpenRouter match audit and model aggregate IDs do not reconcile")

    base_by_checkpoint = {
        row["canonical_checkpoint_id"]: row["canonical_base_id"]
        for row in base_observations
        if row["source"] == "Epoch" and row["record_type"] == "model" and row["canonical_checkpoint_id"]
    }
    base_no_cot_by_model = {
        row["source_model_name"]: row
        for row in base_observations
        if row["source"] == "No-CoT" and row["source_model_name"] != "__scaling_law__"
    }
    if len(no_cot_date_models) != 49 or set(base_no_cot_by_model) != {
        row["model"] for row in no_cot_date_models
    }:
        raise ValueError("No-CoT exact-date audit does not reconcile to the 49 base model rows")
    if len(primary_evidence_rows) != 5 or len(primary_control_rows) != 33:
        raise ValueError(
            "Frontier primary evidence inventory mismatch: "
            f"{len(primary_evidence_rows)} evidence / {len(primary_control_rows)} controls"
        )
    identity_by_name: dict[str, dict[str, str]] = {}
    for source in ("Epoch", "ECI", "No-CoT"):
        for row in base_observations:
            if row["source"] != source or row["record_type"] != "model":
                continue
            identity_by_name.setdefault(row["source_model_name"], row)
            if row["canonical_display_name"]:
                identity_by_name.setdefault(row["canonical_display_name"], row)
    extra_fields = sorted(
        {f"or_{field}" for field in model_fields + provider_fields + tier_fields + daily_fields}
        | {f"orh_{field}" for field in historical_price_fields}
        | {"or_provider_normalized_throughput_ratio", "or_normalization_controls_used"}
    )
    all_observations: list[dict[str, Any]] = [dict(row) for row in base_observations]
    all_measurements: list[dict[str, str]] = [dict(row) for row in base_measurements]

    for model in models:
        model_id = model["openrouter_model_id"]
        audit = audit_by_id[model_id]
        checkpoint, base, display, matched_epoch, link_level = identity(model_id, audit_by_id, base_by_checkpoint)
        observation_id = f"openrouter:model:{model_id}"
        all_observations.append(
            make_observation(
                base_fields, extra_fields, model, audit, checkpoint, base, display, matched_epoch, link_level,
                kind="model", observation_id=observation_id,
            )
        )
        all_measurements.extend(
            measurement_rows(
                model, checkpoint, base, matched_epoch, link_level, observation_id,
                kind="model", normalized_ratio=audit["provider_normalized_throughput_ratio"],
            )
        )

    provider_key_counter: Counter[str] = Counter()
    for provider in providers:
        model_id = provider["openrouter_model_id"]
        audit = audit_by_id[model_id]
        checkpoint, base, display, matched_epoch, link_level = identity(model_id, audit_by_id, base_by_checkpoint)
        raw_key = f"{model_id}:{provider['endpoint_id']}:{provider['variant']}"
        provider_key_counter[raw_key] += 1
        observation_id = f"openrouter:provider:{raw_key}"
        all_observations.append(
            make_observation(
                base_fields, extra_fields, provider, audit, checkpoint, base, display, matched_epoch, link_level,
                kind="provider", observation_id=observation_id,
            )
        )
        all_measurements.extend(
            measurement_rows(
                provider, checkpoint, base, matched_epoch, link_level, observation_id, kind="provider"
            )
        )
    if any(count != 1 for count in provider_key_counter.values()):
        raise ValueError("Duplicate OpenRouter provider observation key")

    tier_key_counter: Counter[str] = Counter()
    for tier in tiers:
        model_id = tier["openrouter_model_id"]
        audit = audit_by_id[model_id]
        checkpoint, base, display, matched_epoch, link_level = identity(
            model_id, audit_by_id, base_by_checkpoint
        )
        raw_key = f"{model_id}:{tier['endpoint_id']}:{tier['service_tier']}"
        tier_key_counter[raw_key] += 1
        observation_id = f"openrouter:tier:{raw_key}"
        all_observations.append(
            make_observation(
                base_fields,
                extra_fields,
                tier,
                audit,
                checkpoint,
                base,
                display,
                matched_epoch,
                link_level,
                kind="tier",
                observation_id=observation_id,
            )
        )
        all_measurements.extend(
            measurement_rows(
                tier,
                checkpoint,
                base,
                matched_epoch,
                link_level,
                observation_id,
                kind="tier",
            )
        )
    if any(count != 1 for count in tier_key_counter.values()):
        raise ValueError("Duplicate OpenRouter endpoint service-tier observation key")

    daily_key_counter: Counter[str] = Counter()
    for point in daily:
        model_id = point["openrouter_model_id"]
        audit = audit_by_id[model_id]
        checkpoint, base, display, matched_epoch, link_level = identity(
            model_id, audit_by_id, base_by_checkpoint
        )
        raw_key = ":".join(
            (
                model_id,
                point["observation_time_raw"],
                point["endpoint_tier_key"],
            )
        )
        daily_key_counter[raw_key] += 1
        observation_id = f"openrouter:daily:{raw_key}"
        all_observations.append(
            make_observation(
                base_fields,
                extra_fields,
                point,
                audit,
                checkpoint,
                base,
                display,
                matched_epoch,
                link_level,
                kind="daily",
                observation_id=observation_id,
            )
        )
        all_measurements.extend(
            measurement_rows(
                point,
                checkpoint,
                base,
                matched_epoch,
                link_level,
                observation_id,
                kind="daily",
            )
        )
    if any(count != 1 for count in daily_key_counter.values()):
        raise ValueError("Duplicate OpenRouter daily throughput observation key")

    historical_price_key_counter: Counter[str] = Counter()
    historical_measurement_count = 0
    for point in historical_prices:
        raw_key = f"{point['openrouter_model_id']}:{point['change_index']}"
        historical_price_key_counter[raw_key] += 1
        observation, measurements = make_historical_price_observation(
            base_fields,
            extra_fields,
            point,
            audit_by_id.get(point["openrouter_model_id"]),
            base_by_checkpoint,
        )
        all_observations.append(observation)
        all_measurements.extend(measurements)
        historical_measurement_count += len(measurements)
    if any(count != 1 for count in historical_price_key_counter.values()):
        raise ValueError("Duplicate OpenRouter historical price change-point key")
    expected_historical_points = int(
        historical_price_metadata["inventory"]["price_change_points"]
    )
    if len(historical_prices) != expected_historical_points:
        raise ValueError(
            "Historical-price megafile reconciliation failed: "
            f"observations={len(historical_prices)} != {expected_historical_points}"
        )
    if not (2 * len(historical_prices) <= historical_measurement_count <= 3 * len(historical_prices)):
        raise ValueError(
            "Historical-price measurement coverage is outside the 2–3 metric contract: "
            f"{historical_measurement_count} for {len(historical_prices)} observations"
        )

    no_cot_date_measurement_count = 0
    for date_row in no_cot_date_models:
        observation, measurements = make_no_cot_date_audit_observation(
            base_fields,
            extra_fields,
            date_row,
            base_no_cot_by_model[date_row["model"]],
        )
        all_observations.append(observation)
        all_measurements.extend(measurements)
        no_cot_date_measurement_count += len(measurements)
    law_observation, law_measurements = make_no_cot_date_law_observation(
        base_fields, extra_fields, no_cot_date_audit
    )
    all_observations.append(law_observation)
    all_measurements.extend(law_measurements)
    no_cot_date_measurement_count += len(law_measurements)
    if no_cot_date_measurement_count != 256:
        raise ValueError(
            f"No-CoT exact-date measurement reconciliation failed: {no_cot_date_measurement_count}"
        )

    primary_measurement_count = 0
    for evidence_row in primary_evidence_rows:
        observation, measurements = make_primary_evidence_observation(
            base_fields, extra_fields, evidence_row, identity_by_name
        )
        all_observations.append(observation)
        all_measurements.extend(measurements)
        primary_measurement_count += len(measurements)
    audit_observation, audit_measurements = make_primary_audit_observation(
        base_fields, extra_fields, primary_audit
    )
    all_observations.append(audit_observation)
    all_measurements.extend(audit_measurements)
    primary_measurement_count += len(audit_measurements)
    for index, control in enumerate(primary_control_rows, start=1):
        observation, measurements = make_primary_control_observation(
            base_fields, extra_fields, control, index, identity_by_name
        )
        all_observations.append(observation)
        all_measurements.extend(measurements)
        primary_measurement_count += len(measurements)
    primary_observation_count = len(primary_evidence_rows) + 1 + len(primary_control_rows)
    if primary_measurement_count < 90:
        raise ValueError(
            f"Frontier primary evidence lost measurements: {primary_measurement_count}"
        )

    opus5_sections = (
        "identity",
        "anthropic_system_card",
        "api",
        "artificial_analysis",
        "epoch",
        "openrouter",
        "availability",
    )
    opus5_measurement_count = 0
    for section in opus5_sections:
        observation, measurements = make_opus5_evidence_observation(
            base_fields, extra_fields, section, opus5_evidence[section], identity_by_name
        )
        all_observations.append(observation)
        all_measurements.extend(measurements)
        opus5_measurement_count += len(measurements)
    if opus5_measurement_count < 40:
        raise ValueError(f"Claude Opus 5 evidence lost measurements: {opus5_measurement_count}")

    observation_ids = [row["observation_id"] for row in all_observations]
    measurement_ids = [row["measurement_id"] for row in all_measurements]
    if len(observation_ids) != len(set(observation_ids)):
        raise ValueError("Duplicate unified observation ID")
    if len(measurement_ids) != len(set(measurement_ids)):
        raise ValueError("Duplicate unified measurement ID")

    write_csv(OBSERVATIONS, base_fields + extra_fields, all_observations)
    write_csv(MEASUREMENTS, measurement_fields, all_measurements)

    manifest_additions = [
        ("Claude Opus 5 normalized evidence bundle", OPUS5_EVIDENCE, 7, "seven normalized evidence sections plus path/hash provenance maps", "Distinct undisclosed base; all numeric evidence retained as correlated source views and excluded from duplicate model-level weighting"),
        ("No-CoT exact-date audit", NO_COT_EXACT_DATE_AUDIT, 1, "JSON date-fidelity and scaling-law sensitivity audit", "49/49 exact checkpoint dates; exact-date/month-date Pareto slope ratios; no evidence-weight change"),
        ("No-CoT exact-date model ledger", NO_COT_EXACT_DATE_MODELS, len(no_cot_date_models), "one row per no-CoT checkpoint", "Paper month and exact date retained separately; four date-only overrides add zero parameter identities"),
        ("Frontier primary-evidence audit", FRONTIER_PRIMARY_AUDIT, 1, "JSON chronological developer-held-out statistical audit", "Official Sol no-CoT measurement, parameter-mapping sensitivity, Fable/Mythos identity, and zero-weight promotion gates"),
        ("Frontier primary-evidence controls", FRONTIER_PRIMARY_CONTROLS, len(primary_control_rows), "held-out predictions, same-size controls, and target sensitivities", "Every backtest and method-sensitivity row retained; derived rows excluded from model-level likelihoods"),
        ("OpenRouter raw snapshot", RAW_SNAPSHOT, collection_audit["catalog_model_count"], "gzip JSON", "Full catalog and per-model public endpoint responses"),
        ("OpenRouter model aggregates", MODEL_SIGNALS, len(models), f"{len(model_fields)} columns", "One row per eligible text model ID"),
        ("OpenRouter provider observations", PROVIDER_SIGNALS, len(providers), f"{len(provider_fields)} columns", "One row per model ID + endpoint UUID + variant"),
        ("OpenRouter endpoint service-tier observations", TIER_SIGNALS, len(tiers), f"{len(tier_fields)} columns", "Base and high-context prices plus p50-p99 throughput/latency retained separately by endpoint and service tier"),
        ("OpenRouter daily provider throughput", DAILY_SIGNALS, len(daily), f"{len(daily_fields)} columns", "One lossless row per model + date + endpoint/service-tier key"),
        ("OpenRouter model snapshot history", MODEL_HISTORY, csv_record_count(MODEL_HISTORY), "timestamped model aggregates", "Repeated refreshes retained as correlated operational views"),
        ("OpenRouter provider snapshot history", PROVIDER_HISTORY, csv_record_count(PROVIDER_HISTORY), "timestamped provider aggregates", "Repeated refreshes retained as correlated operational views"),
        ("OpenRouter endpoint-tier snapshot history", TIER_HISTORY, csv_record_count(TIER_HISTORY), "timestamped endpoint/service-tier prices and percentiles", "Every tier row reconstructed from each immutable raw response"),
        ("OpenRouter daily snapshot history", DAILY_HISTORY, csv_record_count(DAILY_HISTORY), "timestamped daily endpoint/tier rows", "All service tiers and refreshes preserved; rolling windows overlap"),
        ("OpenRouter snapshot history manifest", HISTORY_MANIFEST, csv_record_count(HISTORY_MANIFEST), "immutable refresh manifest", "Exact archived file hashes and fetched-at timestamps"),
        ("OpenRouter manual Epoch match audit", MATCH_AUDIT, len(audits), "explicit identity audit", "Unmatched models retained; fuzzy matches prohibited"),
        ("OpenRouter parameter-signal backtest", SIGNAL_RESULT, 1, "JSON statistical audit", "Family-held-out and chronological-family tests"),
        ("OpenRouter temporal-stability audit", TEMPORAL_RESULT, 1, "JSON statistical audit", "Service-tier correction, daily volatility, and repeated-refresh checks"),
        ("OpenRouter request-weighted operational audit", REQUEST_WEIGHTED_RESULT, 1, "JSON statistical audit", "Request-count-gated throughput, latency, percentile-spread, and promotion checks"),
        ("OpenRouter request-weighted predictions", REQUEST_WEIGHTED_PREDICTIONS, csv_record_count(REQUEST_WEIGHTED_PREDICTIONS), "held-out prediction ledger", "Common-panel developer-family and chronological-family predictions for every operational candidate"),
        ("OpenRouter active-price audit", ACTIVE_PRICE_RESULT, 1, "JSON statistical audit", "Active-capacity price tests, sparse-MoE transport, promotion gates, and frontier sensitivities"),
        ("OpenRouter active-parameter identity ledger", ACTIVE_PRICE_MATCHES, csv_record_count(ACTIVE_PRICE_MATCHES), "one row per exact Epoch/OpenRouter calibration checkpoint", "Disclosed active labels and primary-config dense controls are preserved separately"),
        ("OpenRouter active-price predictions", ACTIVE_PRICE_PREDICTIONS, csv_record_count(ACTIVE_PRICE_PREDICTIONS), "release-ordered developer-held-out predictions", "Every eligible active-capacity price prediction and transport error"),
        ("OpenRouter active-price frontier targets", ACTIVE_PRICE_TARGETS, csv_record_count(ACTIVE_PRICE_TARGETS), "zero-weight frontier sensitivity ledger", "K3-anchored active-to-total sensitivities; not part of the live ensemble"),
        ("OpenRouter historical price raw ledger", HISTORICAL_PRICE_RAW, historical_price_metadata["inventory"]["models"], f"gzip JSON with {len(historical_prices):,} ordered price change points", f"Hash-pinned compact ledger rebuilt from {historical_price_metadata['source']['full_git_history_rebuild_snapshot_count']:,} committed official /api/v1/models snapshots"),
        ("OpenRouter historical price change points", HISTORICAL_PRICE_POINTS, len(historical_prices), f"{len(historical_price_fields)} columns", "Every model, availability interval, change index, price state, and source commit retained"),
        ("OpenRouter historical price collection metadata", HISTORICAL_PRICE_METADATA, 1, "JSON provenance and integrity audit", "Pinned commit/blob/hash, collection schedule, inventory, and no-loss policies"),
        ("OpenRouter historical price identity audit", HISTORICAL_PRICE_MATCHES, csv_record_count(HISTORICAL_PRICE_MATCHES), "one row per exact calibration checkpoint", "All aliases matched; every tested price window and eligibility decision retained"),
        ("OpenRouter historical price backtest", HISTORICAL_PRICE_RESULT, 1, "JSON prospective statistical audit", "Price-availability-ordered developer holdouts across seven predeclared windows"),
        ("OpenRouter historical price predictions", HISTORICAL_PRICE_PREDICTIONS, csv_record_count(HISTORICAL_PRICE_PREDICTIONS), "prospective held-out prediction ledger", "Every total/active/common-panel specification and current-price nonprospective comparator retained"),
        ("OpenRouter historical price frontier targets", HISTORICAL_PRICE_TARGETS, csv_record_count(HISTORICAL_PRICE_TARGETS), "first-day price sensitivity ledger", "Exact first-day prompt/completion prices and K3-anchored zero-weight sensitivities"),
        ("Primary Hugging Face architecture config snapshot", HF_ARCHITECTURE_SNAPSHOT, 87, "gzip JSON with verbatim config objects", "Primary raw config.json responses, including explicit HTTP/gating failures"),
        ("Hugging Face architecture config signals", HF_ARCHITECTURE_SIGNALS, csv_record_count(HF_ARCHITECTURE_SIGNALS), "one row per exact Hugging Face repository", "Nested expert fields and conservative dense/MoE/unavailable classification"),
        ("Hugging Face architecture config audit", HF_ARCHITECTURE_AUDIT, 1, "JSON collection and integrity audit", "Inventory reconciliation, status counts, classifications, and hashes"),
        ("OpenRouter official endpoint snapshot", OFFICIAL_SNAPSHOT, official_audit["model_count_requested"], "gzip JSON", "Documented model-endpoints API responses for all retained model IDs"),
        ("OpenRouter official endpoint prices", OFFICIAL_PRICES, csv_record_count(OFFICIAL_PRICES), "official endpoint price schedules", "Independent core-API crosscheck of frontend price parsing"),
        ("OpenRouter official/frontend comparison", OFFICIAL_COMPARISON, csv_record_count(OFFICIAL_COMPARISON), "price-signature multiset comparison", "Exact matches and every live discrepancy retained"),
        ("OpenRouter official endpoint audit", OFFICIAL_AUDIT, 1, "JSON integrity audit", "Coverage, exact-match share, focal-model checks, and hashes"),
    ]
    manifest = list(base_manifest)
    for source, path, records, structure, notes in manifest_additions:
        manifest.append(
            {
                "source": source,
                "path": portable_path(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
                "records_parsed": records,
                "structure": structure,
                "notes": notes,
            }
        )
    write_csv(MANIFEST, manifest_fields, manifest)

    summary = {
        "schema_version": "1.0",
        "snapshot_date": DATE,
        "base_observations": len(base_observations),
        "openrouter_model_observations": len(models),
        "openrouter_provider_observations": len(providers),
        "openrouter_tier_observations": len(tiers),
        "openrouter_daily_observations": len(daily),
        "openrouter_historical_price_observations": len(historical_prices),
        "no_cot_exact_date_observations": len(no_cot_date_models) + 1,
        "frontier_primary_evidence_observations": primary_observation_count,
        "opus5_evidence_observations": len(opus5_sections),
        "total_observations": len(all_observations),
        "base_measurements": len(base_measurements),
        "openrouter_measurements": len(all_measurements) - len(base_measurements) - no_cot_date_measurement_count - primary_measurement_count - opus5_measurement_count,
        "openrouter_historical_price_measurements": historical_measurement_count,
        "no_cot_exact_date_measurements": no_cot_date_measurement_count,
        "frontier_primary_evidence_measurements": primary_measurement_count,
        "opus5_evidence_measurements": opus5_measurement_count,
        "total_measurements": len(all_measurements),
        "unique_observation_ids": len(observation_ids) == len(set(observation_ids)),
        "unique_measurement_ids": len(measurement_ids) == len(set(measurement_ids)),
        "source_manifest_rows": len(manifest),
        "match_status_counts": dict(sorted(Counter(row["match_status"] for row in audits).items())),
        "files": {
            "observations": {"path": str(OBSERVATIONS.relative_to(ROOT)), "sha256": sha256(OBSERVATIONS)},
            "measurements": {"path": str(MEASUREMENTS.relative_to(ROOT)), "sha256": sha256(MEASUREMENTS)},
            "manifest": {"path": str(MANIFEST.relative_to(ROOT)), "sha256": sha256(MANIFEST)},
        },
    }
    SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
