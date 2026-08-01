#!/usr/bin/env python3
"""Audit OpenRouter price/throughput as parameter-count signals.

This analysis deliberately keeps three layers separate:

1. the complete OpenRouter catalog and provider observations (collected by
   ``collect_openrouter_signals.py``);
2. a manually reviewed OpenRouter -> Epoch checkpoint map; and
3. held-out statistical tests of whether current API price or throughput helps
   predict Epoch's disclosed total parameter counts.

Throughput is normalized within provider and quantization before modeling.
Every regression is evaluated by held-out developer family; a second audit is
strictly chronological as well.  The script does not silently add either signal
to the live frontier ensemble: it emits a recommended incremental weight only
when a paired family bootstrap supports improvement.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from k3_primary_evidence import K3_EVIDENCE_PATH, K3_PARAMETER_SOURCE, K3_TOTAL_B


ROOT = Path(__file__).resolve().parent
COMPATIBILITY_FILE_DATE = "2026-07-18"
SNAPSHOT_DATE = COMPATIBILITY_FILE_DATE
OUT = ROOT / "outputs/019f6c42-2d53-7743-ab07-6293e2618dd7"
MODEL_SIGNALS = ROOT / f"sources/openrouter_model_signals_{SNAPSHOT_DATE}.csv"
PROVIDER_SIGNALS = ROOT / f"sources/openrouter_provider_signals_{SNAPSHOT_DATE}.csv"
DAILY_THROUGHPUT = ROOT / f"sources/openrouter_throughput_daily_{SNAPSHOT_DATE}.csv"
RAW_SNAPSHOT = ROOT / f"sources/openrouter_operational_snapshot_{SNAPSHOT_DATE}.json.gz"
UNIFIED = OUT / "unified_model_observations_compute_enriched_2026-07-17.csv"

MATCH_AUDIT = OUT / f"openrouter_epoch_match_audit_{SNAPSHOT_DATE}.csv"
CALIBRATION = OUT / f"openrouter_parameter_calibration_{SNAPSHOT_DATE}.csv"
PREDICTIONS = OUT / f"openrouter_parameter_backtest_predictions_{SNAPSHOT_DATE}.csv"
FRONTIER = OUT / f"openrouter_frontier_operational_estimates_{SNAPSHOT_DATE}.csv"
RESULT = OUT / f"openrouter_parameter_signal_backtest_{SNAPSHOT_DATE}.json"


# Every accepted match was checked against both the OpenRouter ID/name and the
# exact Epoch checkpoint row.  Multiple OpenRouter aliases may point to one
# checkpoint; they are collapsed before fitting so no checkpoint is duplicated.
MANUAL_EPOCH_MAP: dict[str, str] = {
    "arcee-ai/trinity-large-thinking": "checkpoint:epoch:arcee-trinity-large",
    "baidu/ernie-4.5-vl-424b-a47b": "checkpoint:epoch:ernie-4-5-vl-424b-a47b-4-5",
    "bytedance-seed/seed-1.6": "checkpoint:epoch:seed-1-6",
    "bytedance-seed/seed-1.6-flash": "checkpoint:epoch:seed-1-6",
    "cohere/command-a": "checkpoint:epoch:cohere-command-a",
    "deepseek/deepseek-chat": "checkpoint:deepseek:deepseek-v3",
    "deepseek/deepseek-chat-v3-0324": "checkpoint:deepseek:deepseek-v3-mar-2025",
    "deepseek/deepseek-chat-v3.1": "checkpoint:epoch:deepseek-v3-1",
    "deepseek/deepseek-r1": "checkpoint:deepseek:deepseek-r1",
    "deepseek/deepseek-r1-0528": "checkpoint:deepseek:deepseek-r1-may-2025",
    "deepseek/deepseek-r1-distill-llama-70b": "checkpoint:epoch:deepseek-r1-distill-llama-70b",
    "deepseek/deepseek-v3.1-terminus": "checkpoint:epoch:deepseek-v3-1-terminus",
    "deepseek/deepseek-v3.2-exp": "checkpoint:deepseek:deepseek-v3-2-exp",
    "deepseek/deepseek-v4-flash": "checkpoint:epoch:deepseek-v4-flash",
    "deepseek/deepseek-v4-pro": "checkpoint:deepseek:deepseek-v4-pro",
    "google/gemma-2-27b-it": "checkpoint:google:gemma-2-27b",
    "google/gemma-3-12b-it": "checkpoint:epoch:gemma-3-12b",
    "google/gemma-3-27b-it": "checkpoint:google:gemma-3-27b",
    "google/gemma-3-4b-it": "checkpoint:epoch:gemma-3-4b",
    "google/gemma-3n-e4b-it": "checkpoint:epoch:gemma-3n",
    "google/gemma-4-26b-a4b-it": "checkpoint:epoch:gemma-4-26b-a4b",
    "google/gemma-4-31b-it": "checkpoint:google:gemma-4-31b-it",
    "ibm-granite/granite-4.0-h-micro": "checkpoint:epoch:granite-4-0-h-micro",
    "meta-llama/llama-3.1-70b-instruct": "checkpoint:meta:llama-3-1-70b",
    "meta-llama/llama-3.1-8b-instruct": "checkpoint:meta:llama-3-1-8b",
    "meta-llama/llama-3.2-1b-instruct": "checkpoint:epoch:llama-3-2-1b",
    "meta-llama/llama-3.2-3b-instruct": "checkpoint:epoch:llama-3-2-3b",
    "meta-llama/llama-3.3-70b-instruct": "checkpoint:meta:llama-3-3-70b",
    "meta-llama/llama-4-maverick": "checkpoint:meta:llama-4-maverick",
    "meta-llama/llama-4-scout": "checkpoint:meta:llama-4-scout",
    "microsoft/phi-4": "checkpoint:microsoft-research:phi-4",
    "microsoft/wizardlm-2-8x22b": "checkpoint:epoch:wizardlm-2-8x22b",
    "minimax/minimax-01": "checkpoint:epoch:minimax-text-01",
    "minimax/minimax-m1": "checkpoint:epoch:minimax-m1-80k",
    "minimax/minimax-m2": "checkpoint:epoch:minimax-m2",
    "minimax/minimax-m2.1": "checkpoint:epoch:minimax-m2-1",
    "minimax/minimax-m2.5": "checkpoint:minimax:minimax-m2-5",
    "mistralai/devstral-2512": "checkpoint:epoch:devstral-2-123b",
    "mistralai/ministral-14b-2512": "checkpoint:epoch:ministral-3-14b",
    "mistralai/ministral-3b-2512": "checkpoint:epoch:ministral-3-3b",
    "mistralai/ministral-8b-2512": "checkpoint:epoch:ministral-3-8b",
    "mistralai/mistral-large-2407": "checkpoint:epoch:mistral-large-2",
    "mistralai/mistral-large-2512": "checkpoint:epoch:mistral-3-large",
    "mistralai/mistral-medium-3-5": "checkpoint:epoch:mistral-medium-3-5",
    "mistralai/mistral-nemo": "checkpoint:mistral:mistral-nemo",
    "mistralai/mistral-saba": "checkpoint:epoch:mistral-saba",
    "mistralai/mistral-small-24b-instruct-2501": "checkpoint:epoch:mistral-small-3",
    "mistralai/mistral-small-3.1-24b-instruct": "checkpoint:mistral:mistral-small-3-1",
    "mistralai/mistral-small-3.2-24b-instruct": "checkpoint:epoch:mistral-small-3-2",
    "mistralai/mixtral-8x22b-instruct": "checkpoint:mistral:mixtral-8x22b",
    "mistralai/voxtral-small-24b-2507": "checkpoint:epoch:voxtral-small",
    "moonshotai/kimi-k2": "checkpoint:epoch:kimi-k2",
    "moonshotai/kimi-k2-thinking": "checkpoint:moonshot:kimi-k2-thinking",
    "moonshotai/kimi-k2.5": "checkpoint:moonshot:kimi-k2-5",
    "moonshotai/kimi-k2.6": "checkpoint:moonshot:kimi-k2-6",
    "moonshotai/kimi-k2.7-code": "checkpoint:moonshot:kimi-k2-7-code",
    "nousresearch/hermes-3-llama-3.1-405b": "checkpoint:epoch:hermes-3-405b",
    "nousresearch/hermes-3-llama-3.1-70b": "checkpoint:epoch:hermes-3-70b",
    "nvidia/nemotron-3-nano-30b-a3b": "checkpoint:epoch:nemotron-3-nano-30b-a3b",
    "nvidia/nemotron-3-super-120b-a12b": "checkpoint:epoch:nemotron-3-super",
    "nvidia/nemotron-3-ultra-550b-a55b": "checkpoint:epoch:nemotron-3-ultra",
    "openai/gpt-oss-120b": "checkpoint:openai:gpt-oss-120b",
    "openai/gpt-oss-20b": "checkpoint:epoch:gpt-oss-20b",
    "qwen/qwen-2.5-72b-instruct": "checkpoint:epoch:qwen2-5-instruct-72b",
    "qwen/qwen-2.5-7b-instruct": "checkpoint:epoch:qwen2-5-instruct-7b",
    "qwen/qwen-2.5-coder-32b-instruct": "checkpoint:alibaba:qwen2-5-coder-32b",
    "qwen/qwen2.5-vl-72b-instruct": "checkpoint:epoch:qwen2-5-vl-72b",
    "qwen/qwen3-14b": "checkpoint:epoch:qwen3-14b",
    "qwen/qwen3-235b-a22b": "checkpoint:alibaba:qwen3-235b-a22b",
    "qwen/qwen3-235b-a22b-2507": "checkpoint:epoch:qwen3-235b-a22b-jul-2025",
    "qwen/qwen3-235b-a22b-thinking-2507": "checkpoint:alibaba:qwen3-235b-a22b-thinking-jul-2025",
    "qwen/qwen3-30b-a3b": "checkpoint:epoch:qwen3-30b-a3b",
    "qwen/qwen3-32b": "checkpoint:epoch:qwen3-32b",
    "qwen/qwen3-8b": "checkpoint:epoch:qwen3-8b",
    "qwen/qwen3-coder": "checkpoint:epoch:qwen3-coder-480b-a35b",
    "qwen/qwen3-coder-next": "checkpoint:epoch:qwen3-coder-next",
    "qwen/qwen3-max": "checkpoint:alibaba:qwen3-max",
    "qwen/qwen3-next-80b-a3b-instruct": "checkpoint:epoch:qwen3-next-80b-a3b",
    "qwen/qwen3-next-80b-a3b-thinking": "checkpoint:epoch:qwen3-next-80b-a3b",
    "qwen/qwen3.5-122b-a10b": "checkpoint:epoch:qwen3-5-122b-a10b",
    "qwen/qwen3.5-27b": "checkpoint:epoch:qwen3-5-27b",
    "qwen/qwen3.5-397b-a17b": "checkpoint:epoch:qwen3-5-397b-a17b",
    "qwen/qwen3.5-9b": "checkpoint:epoch:qwen3-5-9b",
    "rekaai/reka-edge": "checkpoint:epoch:reka-edge",
    "rekaai/reka-flash-3": "checkpoint:epoch:reka-flash-3",
    "tencent/hy3-preview": "checkpoint:epoch:tencent-hy3-preview",
    "z-ai/glm-4.5": "checkpoint:epoch:glm-4-5",
    "z-ai/glm-4.5-air": "checkpoint:epoch:glm-4-5-air",
    "z-ai/glm-4.5v": "checkpoint:epoch:glm-4-5v",
    "z-ai/glm-4.6": "checkpoint:z-ai-zhipu-ai-tsinghua-university:glm-4-6",
    "z-ai/glm-4.7": "checkpoint:z-ai-zhipu-ai:glm-4-7",
    "z-ai/glm-4.7-flash": "checkpoint:epoch:glm-4-7-flash",
    "z-ai/glm-5": "checkpoint:z-ai-zhipu-ai:glm-5",
    "z-ai/glm-5.1": "checkpoint:z-ai-zhipu-ai:glm-5-1",
    "z-ai/glm-5.2": "checkpoint:z-ai-zhipu-ai:glm-5-2",
}

DISCLOSED_ANCHORS = {
    "moonshotai/kimi-k3": {
        "total_parameters_b": K3_TOTAL_B,
        "source": K3_PARAMETER_SOURCE,
    },
    "x-ai/grok-4.5": {"total_parameters_b": 1500.0, "source": "user-provided disclosure"},
}

FRONTIER_TARGETS = {
    "anthropic/claude-fable-5": "Claude Fable 5",
    "openai/gpt-5.6-sol": "GPT-5.6 Sol",
    "anthropic/claude-opus-4.8": "Claude Opus 4.8",
    "anthropic/claude-opus-4.7": "Claude Opus 4.7",
    "anthropic/claude-opus-4.5": "Claude Opus 4.5",
    "openai/gpt-5.5": "GPT-5.5",
    "openai/gpt-5.6-terra": "GPT-5.6 Terra",
    "anthropic/claude-sonnet-5": "Claude Sonnet 5",
    "openai/gpt-5.6-luna": "GPT-5.6 Luna",
    **{key: key for key in DISCLOSED_ANCHORS},
}

SAME_BASE_CONTROLS = {
    "Claude Opus 4.5–4.8 shared-base control": [
        "anthropic/claude-opus-4.5",
        "anthropic/claude-opus-4.6",
        "anthropic/claude-opus-4.7",
        "anthropic/claude-opus-4.8",
    ],
    "GPT-5–5.5 same-base control (standard endpoints only)": [
        "openai/gpt-5",
        "openai/gpt-5.1",
        "openai/gpt-5.2",
        "openai/gpt-5.3-chat",
        "openai/gpt-5.4",
        "openai/gpt-5.5",
    ],
}

FEATURE_SPECS = {
    "date_only": ("release_decimal_year",),
    "date_price": ("release_decimal_year", "log10_blended_price"),
    "date_raw_throughput": ("release_decimal_year", "log10_raw_throughput"),
    "date_provider_normalized_throughput": (
        "release_decimal_year",
        "log10_provider_normalized_throughput",
    ),
    "date_price_provider_normalized_throughput": (
        "release_decimal_year",
        "log10_blended_price",
        "log10_provider_normalized_throughput",
    ),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str] | None = None) -> None:
    if fields is None:
        fields = list(rows[0]) if rows else []
    field_list = list(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=field_list, extrasaction="raise", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in field_list} for row in rows])


def number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def median(values: Iterable[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    return None if not clean else float(statistics.median(clean))


def decimal_year(value: str) -> float:
    parsed = date.fromisoformat(value)
    start = date(parsed.year, 1, 1)
    end = date(parsed.year + 1, 1, 1)
    return parsed.year + (parsed - start).days / (end - start).days


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def epoch_registry() -> dict[str, dict[str, str]]:
    rows = [
        row
        for row in read_csv(UNIFIED)
        if row["source"] == "Epoch"
        and row["record_type"] == "model"
        and row["total_parameters_b"]
        and row["canonical_release_date"]
        and "Epoch" in row["parameter_value_source"]
    ]
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["canonical_checkpoint_id"]].append(row)
    duplicates = {
        key: value
        for key, value in grouped.items()
        if key in set(MANUAL_EPOCH_MAP.values()) and len(value) != 1
    }
    if duplicates:
        raise ValueError(f"Epoch canonical model rows are not unique: {sorted(duplicates)[:5]}")
    return {key: value[0] for key, value in grouped.items() if len(value) == 1}


def provider_normalized_indices(
    providers: list[dict[str, str]],
) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    usable = [
        row
        for row in providers
        if number(row["throughput_median_tps_1w"]) is not None
        and number(row["throughput_median_tps_1w"]) > 0
    ]
    by_provider_quant: dict[tuple[str, str], list[float]] = defaultdict(list)
    by_provider: dict[str, list[float]] = defaultdict(list)
    by_quant: dict[str, list[float]] = defaultdict(list)
    for row in usable:
        provider = row["provider_slug"] or row["provider_name"]
        quantization = row["quantization"] or "unspecified"
        log_speed = math.log(float(row["throughput_median_tps_1w"]))
        by_provider_quant[(provider, quantization)].append(log_speed)
        by_provider[provider].append(log_speed)
        by_quant[quantization].append(log_speed)

    global_center = statistics.median(
        math.log(float(row["throughput_median_tps_1w"])) for row in usable
    )
    residuals: dict[str, list[float]] = defaultdict(list)
    controls_used: dict[str, Counter[str]] = defaultdict(Counter)
    for row in usable:
        provider = row["provider_slug"] or row["provider_name"]
        quantization = row["quantization"] or "unspecified"
        pq_values = by_provider_quant[(provider, quantization)]
        provider_values = by_provider[provider]
        quant_values = by_quant[quantization]
        if len(pq_values) >= 4:
            center = statistics.median(pq_values)
            control = "provider+quantization"
        elif len(provider_values) >= 4:
            center = statistics.median(provider_values)
            control = "provider"
        elif len(quant_values) >= 4:
            center = statistics.median(quant_values)
            control = "quantization"
        else:
            center = global_center
            control = "global"
        residuals[row["openrouter_model_id"]].append(
            math.log(float(row["throughput_median_tps_1w"])) - center
        )
        controls_used[row["openrouter_model_id"]][control] += 1

    indices = {model_id: math.exp(statistics.median(values)) for model_id, values in residuals.items()}
    metadata = {
        model_id: {
            "endpoint_observations": len(residuals[model_id]),
            "controls_used": dict(controls_used[model_id]),
        }
        for model_id in residuals
    }
    return indices, metadata


def build_match_and_calibration() -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, float]
]:
    models = read_csv(MODEL_SIGNALS)
    providers = read_csv(PROVIDER_SIGNALS)
    epoch = epoch_registry()
    model_by_id = {row["openrouter_model_id"]: row for row in models}
    normalized_speed, normalization_meta = provider_normalized_indices(providers)

    missing_model_ids = sorted(set(MANUAL_EPOCH_MAP) - set(model_by_id))
    missing_checkpoints = sorted(set(MANUAL_EPOCH_MAP.values()) - set(epoch))
    if missing_model_ids or missing_checkpoints:
        raise ValueError(
            f"Stale manual map: missing OpenRouter={missing_model_ids}, missing Epoch={missing_checkpoints}"
        )

    audit_rows: list[dict[str, Any]] = []
    matched_by_checkpoint: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for model in models:
        model_id = model["openrouter_model_id"]
        checkpoint_id = MANUAL_EPOCH_MAP.get(model_id, "")
        epoch_row = epoch.get(checkpoint_id)
        if epoch_row:
            status = "matched_epoch_manual"
            reason = "Manually verified exact model/checkpoint identity; Epoch parameter value retained"
        elif model_id in DISCLOSED_ANCHORS:
            status = "disclosed_anchor"
            reason = "User-provided disclosed total; excluded from Epoch calibration and used as external check"
        elif model_id in FRONTIER_TARGETS:
            status = "frontier_target"
            reason = "Closed frontier target with unknown parameter count"
        elif any(model_id in members for members in SAME_BASE_CONTROLS.values()):
            status = "same_base_control"
            reason = "Operational noise control; no parameter target assigned"
        else:
            status = "unmatched"
            reason = "No manually verified exact Epoch checkpoint; retained in raw OpenRouter files"

        catalog_date_delta = ""
        catalog_date_flag = ""
        if epoch_row and model["created_date"]:
            catalog_date_delta = (
                date.fromisoformat(model["created_date"])
                - date.fromisoformat(epoch_row["canonical_release_date"])
            ).days
            catalog_date_flag = "reviewed_large_delta" if abs(catalog_date_delta) > 60 else "within_60_days"

        audit = {
            "snapshot_date": model["snapshot_date"],
            "openrouter_model_id": model_id,
            "openrouter_model_name": model["openrouter_model_name"],
            "openrouter_created_date": model["created_date"],
            "match_status": status,
            "match_reason": reason,
            "canonical_checkpoint_id": checkpoint_id,
            "epoch_model_name": epoch_row["canonical_display_name"] if epoch_row else "",
            "epoch_release_date": epoch_row["canonical_release_date"] if epoch_row else "",
            "openrouter_catalog_date_minus_epoch_release_days": catalog_date_delta,
            "catalog_date_comparison_flag": catalog_date_flag,
            "total_parameters_b": epoch_row["total_parameters_b"] if epoch_row else DISCLOSED_ANCHORS.get(model_id, {}).get("total_parameters_b", ""),
            "parameter_value_source": epoch_row["parameter_value_source"] if epoch_row else DISCLOSED_ANCHORS.get(model_id, {}).get("source", ""),
            "prompt_price_usd_per_mtoken": model["prompt_price_median_usd_per_mtoken"],
            "completion_price_usd_per_mtoken": model["completion_price_median_usd_per_mtoken"],
            "blended_price_usd_per_mtoken": model["blended_price_geomean_median_usd_per_mtoken"],
            "raw_throughput_tps_1w": model["throughput_median_provider_median_tps_1w"],
            "provider_normalized_throughput_ratio": normalized_speed.get(model_id, ""),
            "normalization_endpoint_count": normalization_meta.get(model_id, {}).get("endpoint_observations", ""),
            "normalization_controls_used": json.dumps(normalization_meta.get(model_id, {}).get("controls_used", {}), sort_keys=True),
            "catalog_source_url": model["catalog_source_url"],
            "endpoint_source_url": model["endpoint_source_url"],
            "throughput_source_url": model["throughput_source_url"],
        }
        audit_rows.append(audit)
        if epoch_row:
            matched_by_checkpoint[checkpoint_id].append({"model": model, "audit": audit, "epoch": epoch_row})

    calibration_rows: list[dict[str, Any]] = []
    for checkpoint_id, observations in sorted(matched_by_checkpoint.items()):
        epoch_row = observations[0]["epoch"]
        prices = [number(item["model"]["blended_price_geomean_median_usd_per_mtoken"]) for item in observations]
        raw_speeds = [number(item["model"]["throughput_median_provider_median_tps_1w"]) for item in observations]
        normalized_speeds = [normalized_speed.get(item["model"]["openrouter_model_id"]) for item in observations]
        price = median(prices)
        raw_speed = median(raw_speeds)
        normalized = median(normalized_speeds)
        if price is None or price <= 0 or raw_speed is None or raw_speed <= 0 or normalized is None or normalized <= 0:
            continue
        model_ids = sorted(item["model"]["openrouter_model_id"] for item in observations)
        family = model_ids[0].split("/", 1)[0]
        calibration_rows.append(
            {
                "canonical_checkpoint_id": checkpoint_id,
                "canonical_model_name": epoch_row["canonical_display_name"],
                "family": family,
                "release_date": epoch_row["canonical_release_date"],
                "release_decimal_year": decimal_year(epoch_row["canonical_release_date"]),
                "total_parameters_b": float(epoch_row["total_parameters_b"]),
                "parameter_value_source": epoch_row["parameter_value_source"],
                "openrouter_alias_count": len(observations),
                "openrouter_model_ids": "|".join(model_ids),
                "blended_price_usd_per_mtoken": price,
                "raw_throughput_tps_1w": raw_speed,
                "provider_normalized_throughput_ratio": normalized,
                "log10_blended_price": math.log10(price),
                "log10_raw_throughput": math.log10(raw_speed),
                "log10_provider_normalized_throughput": math.log10(normalized),
            }
        )

    if len({row["canonical_checkpoint_id"] for row in calibration_rows}) != len(calibration_rows):
        raise ValueError("Duplicate checkpoint survived OpenRouter calibration collapse")
    return audit_rows, calibration_rows, model_by_id, normalized_speed


def design(rows: list[dict[str, Any]], features: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([[float(row[feature]) for feature in features] for row in rows], dtype=float)
    y = np.log10(np.asarray([float(row["total_parameters_b"]) for row in rows], dtype=float))
    return x, y


def robust_ridge_fit(
    x: np.ndarray,
    y: np.ndarray,
    families: list[str],
    alpha: float = 1.0,
) -> dict[str, np.ndarray]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-9] = 1.0
    z = (x - mean) / scale
    matrix = np.column_stack([np.ones(len(z)), z])
    counts = Counter(families)
    base_weights = np.asarray([1.0 / counts[family] for family in families], dtype=float)
    base_weights *= len(base_weights) / base_weights.sum()
    weights = base_weights.copy()
    beta = np.zeros(matrix.shape[1])
    penalty = np.diag([0.0] + [alpha] * z.shape[1])
    for _ in range(30):
        weighted = matrix * np.sqrt(weights)[:, None]
        target = y * np.sqrt(weights)
        beta_next = np.linalg.solve(weighted.T @ weighted + penalty, weighted.T @ target)
        residual = y - matrix @ beta_next
        center = statistics.median(float(value) for value in residual)
        mad = statistics.median(abs(float(value) - center) for value in residual)
        robust_scale = max(1.4826 * mad, 0.03)
        huber = np.minimum(1.0, (1.5 * robust_scale) / np.maximum(np.abs(residual - center), 1e-12))
        weights = base_weights * huber
        if np.max(np.abs(beta_next - beta)) < 1e-9:
            beta = beta_next
            break
        beta = beta_next
    return {"mean": mean, "scale": scale, "beta": beta}


def predict_fit(fit: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    z = (x - fit["mean"]) / fit["scale"]
    return np.column_stack([np.ones(len(z)), z]) @ fit["beta"]


def prediction_metrics(errors: Iterable[float]) -> dict[str, Any]:
    values = np.asarray(list(errors), dtype=float)
    absolute = np.abs(values)
    return {
        "n": int(len(values)),
        "median_multiplicative_error": float(10 ** np.median(absolute)),
        "mean_absolute_log10_error": float(np.mean(absolute)),
        "rmse_log10": float(np.sqrt(np.mean(values**2))),
        "p80_multiplicative_error": float(10 ** np.quantile(absolute, 0.80)),
        "within_2x": float(np.mean(absolute <= math.log10(2))),
        "signed_bias_factor": float(10 ** np.mean(values)),
    }


def held_out_predictions(
    rows: list[dict[str, Any]],
    mode: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for spec, features in FEATURE_SPECS.items():
        for test_index, test in enumerate(rows):
            if mode == "family":
                train_indices = [index for index, row in enumerate(rows) if row["family"] != test["family"]]
            elif mode == "chronological_family":
                train_indices = [
                    index
                    for index, row in enumerate(rows)
                    if row["family"] != test["family"] and row["release_date"] < test["release_date"]
                ]
            else:
                raise ValueError(mode)
            if len(train_indices) < 20 or len({rows[index]["family"] for index in train_indices}) < 5:
                continue
            train = [rows[index] for index in train_indices]
            train_x, train_y = design(train, features)
            fit = robust_ridge_fit(train_x, train_y, [row["family"] for row in train])
            test_x, test_y = design([test], features)
            predicted_log = float(predict_fit(fit, test_x)[0])
            output.append(
                {
                    "mode": mode,
                    "specification": spec,
                    "canonical_checkpoint_id": test["canonical_checkpoint_id"],
                    "model": test["canonical_model_name"],
                    "family": test["family"],
                    "release_date": test["release_date"],
                    "actual_parameters_b": test["total_parameters_b"],
                    "predicted_parameters_b": 10**predicted_log,
                    "log10_error": predicted_log - float(test_y[0]),
                    "training_rows": len(train),
                    "training_families": len({row["family"] for row in train}),
                }
            )
    return output


def metrics_by_mode_spec(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["mode"], row["specification"])].append(row)
    return {
        mode: {
            spec: prediction_metrics(row["log10_error"] for row in subset)
            for (candidate_mode, spec), subset in grouped.items()
            if candidate_mode == mode
        }
        for mode in sorted({row["mode"] for row in rows})
    }


def paired_family_bootstrap(
    predictions: list[dict[str, Any]],
    mode: str,
    candidate: str,
    baseline: str,
    samples: int = 10_000,
) -> dict[str, Any]:
    selected = [row for row in predictions if row["mode"] == mode and row["specification"] in {candidate, baseline}]
    keyed: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in selected:
        keyed[(row["canonical_checkpoint_id"], row["family"])][row["specification"]] = row
    pairs = [value for value in keyed.values() if candidate in value and baseline in value]
    by_family: dict[str, list[dict[str, dict[str, Any]]]] = defaultdict(list)
    for pair in pairs:
        by_family[pair[candidate]["family"]].append(pair)
    families = sorted(by_family)

    def delta(chosen: list[str]) -> float:
        observations = [pair for family in chosen for pair in by_family[family]]
        return float(
            np.mean([abs(pair[candidate]["log10_error"]) for pair in observations])
            - np.mean([abs(pair[baseline]["log10_error"]) for pair in observations])
        )

    observed = delta(families)
    rng = np.random.default_rng(20260718)
    draws = np.asarray(
        [delta(list(rng.choice(families, size=len(families), replace=True))) for _ in range(samples)]
    )
    return {
        "mode": mode,
        "candidate": candidate,
        "baseline": baseline,
        "metric": "mean absolute log10 error; candidate minus baseline",
        "paired_checkpoints": len(pairs),
        "family_clusters": len(families),
        "observed_delta": observed,
        "ci_90": [float(value) for value in np.quantile(draws, [0.05, 0.95])],
        "bootstrap_probability_candidate_better": float(np.mean(draws < 0)),
        "samples": samples,
    }


def target_features(
    model: dict[str, str],
    normalized_speed: dict[str, float],
    feature_names: tuple[str, ...],
) -> np.ndarray:
    model_id = model["openrouter_model_id"]
    price = float(model["blended_price_geomean_median_usd_per_mtoken"])
    raw_speed = float(model["throughput_median_provider_median_tps_1w"])
    values = {
        "release_decimal_year": decimal_year(model["created_date"]),
        "log10_blended_price": math.log10(price),
        "log10_raw_throughput": math.log10(raw_speed),
        "log10_provider_normalized_throughput": math.log10(normalized_speed[model_id]),
    }
    return np.asarray([[values[name] for name in feature_names]], dtype=float)


def frontier_estimates(
    calibration: list[dict[str, Any]],
    model_by_id: dict[str, dict[str, Any]],
    normalized_speed: dict[str, float],
    prediction_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # The held-out winner is price without throughput.  It remains a diagnostic
    # because frontier API prices extend beyond the calibration support.
    spec = "date_price"
    features = FEATURE_SPECS[spec]
    family_mode = [row for row in prediction_rows if row["mode"] == "family" and row["specification"] == spec]
    error_factor_80 = 10 ** float(np.quantile([abs(row["log10_error"]) for row in family_mode], 0.80))
    result_rows: list[dict[str, Any]] = []
    bootstrap_draws: dict[str, list[float]] = defaultdict(list)
    families = sorted({row["family"] for row in calibration})
    by_family = {family: [row for row in calibration if row["family"] == family] for family in families}
    rng = np.random.default_rng(20260718)

    calibration_prices = [float(row["blended_price_usd_per_mtoken"]) for row in calibration]
    calibration_price_min = min(calibration_prices)
    calibration_price_max = max(calibration_prices)
    for model_id, display_name in FRONTIER_TARGETS.items():
        model = model_by_id[model_id]
        excluded_family = model_id.split("/", 1)[0]
        train = [row for row in calibration if row["family"] != excluded_family]
        train_x, train_y = design(train, features)
        fit = robust_ridge_fit(train_x, train_y, [row["family"] for row in train])
        x = target_features(model, normalized_speed, features)
        central = 10 ** float(predict_fit(fit, x)[0])
        actual = DISCLOSED_ANCHORS.get(model_id, {}).get("total_parameters_b")
        target_price = float(model["blended_price_geomean_median_usd_per_mtoken"])
        result_rows.append(
            {
                "openrouter_model_id": model_id,
                "model": display_name if model_id not in DISCLOSED_ANCHORS else model["openrouter_model_name"],
                "release_date": model["created_date"],
                "prompt_price_usd_per_mtoken": model["prompt_price_median_usd_per_mtoken"],
                "completion_price_usd_per_mtoken": model["completion_price_median_usd_per_mtoken"],
                "raw_throughput_tps_1w": model["throughput_median_provider_median_tps_1w"],
                "provider_normalized_throughput_ratio": normalized_speed[model_id],
                "calibration_price_min_usd_per_mtoken": calibration_price_min,
                "calibration_price_max_usd_per_mtoken": calibration_price_max,
                "price_over_calibration_max": target_price / calibration_price_max,
                "operational_model_central_b": central,
                "heldout_error_80_low_b": central / error_factor_80,
                "heldout_error_80_high_b": central * error_factor_80,
                "disclosed_total_parameters_b": actual or "",
                "disclosed_parameter_source": DISCLOSED_ANCHORS.get(model_id, {}).get(
                    "source", ""
                ),
                "actual_over_prediction": float(actual) / central if actual else "",
                "status": (
                    "external disclosed check"
                    if actual
                    else "unknown target; out-of-domain price extrapolation"
                    if target_price > calibration_price_max
                    else "unknown target; diagnostic only"
                ),
            }
        )

    # Cluster bootstrap shows coefficient uncertainty; the broader held-out
    # interval above remains the relevant predictive uncertainty.
    for _ in range(1000):
        chosen = list(rng.choice(families, size=len(families), replace=True))
        sample = [row for family in chosen for row in by_family[family]]
        sample_x, sample_y = design(sample, features)
        fit = robust_ridge_fit(sample_x, sample_y, [row["family"] for row in sample])
        for model_id in FRONTIER_TARGETS:
            x = target_features(model_by_id[model_id], normalized_speed, features)
            bootstrap_draws[model_id].append(10 ** float(predict_fit(fit, x)[0]))
    for row in result_rows:
        q = np.quantile(bootstrap_draws[row["openrouter_model_id"]], [0.05, 0.5, 0.95])
        row["coefficient_bootstrap_p05_b"] = float(q[0])
        row["coefficient_bootstrap_p50_b"] = float(q[1])
        row["coefficient_bootstrap_p95_b"] = float(q[2])

    anchors = [row for row in result_rows if row["disclosed_total_parameters_b"] != ""]
    anchor_calibration_factor = math.exp(
        statistics.mean(
            math.log(float(row["disclosed_total_parameters_b"]) / float(row["operational_model_central_b"]))
            for row in anchors
        )
    )
    for row in result_rows:
        row["two_anchor_calibration_factor"] = anchor_calibration_factor
        row["anchor_calibrated_central_b"] = float(row["operational_model_central_b"]) * anchor_calibration_factor
        row["anchor_calibrated_heldout_error_80_low_b"] = float(row["heldout_error_80_low_b"]) * anchor_calibration_factor
        row["anchor_calibrated_heldout_error_80_high_b"] = float(row["heldout_error_80_high_b"]) * anchor_calibration_factor

    return result_rows, {
        "specification": spec,
        "family_exclusion": "target developer excluded from calibration when present",
        "coefficient_bootstrap_samples": 1000,
        "heldout_predictive_error_factor_p80": error_factor_80,
        "calibration_price_range_usd_per_mtoken": [calibration_price_min, calibration_price_max],
        "two_anchor_frontier_calibration_factor": anchor_calibration_factor,
        "two_anchor_calibration_note": "Geometric residual correction from disclosed Kimi K3 and Grok 4.5; reported separately from uncalibrated external checks.",
    }


def same_base_audit(
    model_by_id: dict[str, dict[str, Any]], normalized_speed: dict[str, float]
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for group_name, model_ids in SAME_BASE_CONTROLS.items():
        rows = []
        for model_id in model_ids:
            model = model_by_id[model_id]
            rows.append(
                {
                    "model_id": model_id,
                    "created_date": model["created_date"],
                    "blended_price_usd_per_mtoken": float(model["blended_price_geomean_median_usd_per_mtoken"]),
                    "raw_throughput_tps_1w": float(model["throughput_median_provider_median_tps_1w"]),
                    "provider_normalized_throughput_ratio": normalized_speed[model_id],
                }
            )
        raw = [row["raw_throughput_tps_1w"] for row in rows]
        normalized = [row["provider_normalized_throughput_ratio"] for row in rows]
        prices = [row["blended_price_usd_per_mtoken"] for row in rows]
        output[group_name] = {
            "models": rows,
            "raw_throughput_max_over_min": max(raw) / min(raw),
            "provider_normalized_throughput_max_over_min": max(normalized) / min(normalized),
            "price_max_over_min": max(prices) / min(prices),
            "interpretation": "Variation within an asserted shared base is an empirical operational-noise floor, not a size change.",
        }
    return output


def main() -> None:
    audit, calibration, model_by_id, normalized_speed = build_match_and_calibration()
    observation_dates = {row["snapshot_date"] for row in audit}
    if len(observation_dates) != 1:
        raise ValueError(
            f"Expected one OpenRouter observation date, found {observation_dates}"
        )
    observation_date = next(iter(observation_dates))
    write_csv(MATCH_AUDIT, audit)
    write_csv(CALIBRATION, calibration)

    family_predictions = held_out_predictions(calibration, "family")
    chronological_predictions = held_out_predictions(calibration, "chronological_family")
    prediction_rows = family_predictions + chronological_predictions
    write_csv(PREDICTIONS, prediction_rows)
    metrics = metrics_by_mode_spec(prediction_rows)

    comparisons = [
        paired_family_bootstrap(prediction_rows, mode, candidate, baseline)
        for mode in ("family", "chronological_family")
        for candidate, baseline in (
            ("date_price", "date_only"),
            ("date_provider_normalized_throughput", "date_only"),
            ("date_price_provider_normalized_throughput", "date_price"),
        )
    ]
    comparison_index = {
        (row["mode"], row["candidate"], row["baseline"]): row for row in comparisons
    }
    throughput_incremental = comparison_index[
        ("family", "date_price_provider_normalized_throughput", "date_price")
    ]
    chronological_throughput = comparison_index[
        ("chronological_family", "date_price_provider_normalized_throughput", "date_price")
    ]
    tok_s_supported = (
        throughput_incremental["bootstrap_probability_candidate_better"] >= 0.90
        and throughput_incremental["ci_90"][1] < 0
        and chronological_throughput["observed_delta"] < 0
    )
    recommended_incremental_weight = 0.05 if tok_s_supported else 0.0

    frontier_rows, frontier_method = frontier_estimates(
        calibration, model_by_id, normalized_speed, prediction_rows
    )
    write_csv(FRONTIER, frontier_rows)

    match_counts = Counter(row["match_status"] for row in audit)
    large_date_deltas = sum(row["catalog_date_comparison_flag"] == "reviewed_large_delta" for row in audit)
    result = {
        "snapshot_date": observation_date,
        "compatibility_filename_date": COMPATIBILITY_FILE_DATE,
        "conclusion": {
            "openrouter_price_is_family_heldout_predictive": comparison_index[("family", "date_price", "date_only")]["bootstrap_probability_candidate_better"] >= 0.90,
            "current_live_final_price_weight": 0.034,
            "price_weight_recommendation": "retain the existing low weight; do not increase until price is tested incrementally against AA/ECI/no-CoT on the same held-out checkpoints",
            "tok_s_adds_robust_incremental_information_beyond_price": tok_s_supported,
            "recommended_incremental_tok_s_weight_in_live_ensemble": recommended_incremental_weight,
            "policy": "No positive live weight unless family-held-out paired bootstrap is >=90% favorable, its 90% CI excludes zero, and chronological-family direction agrees.",
            "operational_prediction_status": "diagnostic; not automatically blended into the live frontier forecast",
        },
        "data_audit": {
            "openrouter_catalog_models": len(audit),
            "manual_epoch_model_matches": match_counts["matched_epoch_manual"],
            "unique_epoch_calibration_checkpoints": len(calibration),
            "calibration_families": len({row["family"] for row in calibration}),
            "duplicate_calibration_checkpoints": len(calibration) - len({row["canonical_checkpoint_id"] for row in calibration}),
            "match_status_counts": dict(sorted(match_counts.items())),
            "matched_rows_with_catalog_date_over_60_days_from_epoch": large_date_deltas,
            "matching_policy": "Explicit whitelist only; unmatched records remain in raw source files and are never fuzzy-matched.",
            "parameter_policy": "Epoch Parameters only for regression; Kimi K3 and Grok 4.5 disclosed totals are external validation anchors.",
            "date_policy": "Epoch release date is authoritative for matched calibration rows. OpenRouter created_date is a catalog/onboarding timestamp and large deltas are retained as explicit audit flags.",
        },
        "feature_definitions": {
            "price": "median provider geometric mean of prompt and completion USD per million tokens",
            "raw_throughput": "median across provider one-week median output tokens/second; default service tier only",
            "provider_normalized_throughput": "default-tier median endpoint log-speed residual after provider+quantization, provider, quantization, or global control hierarchy; exponentiated",
            "target": "log10 Epoch total parameters in billions",
            "fit": "family-balanced Huber robust ridge (alpha=1) with standardized features",
        },
        "heldout_metrics": metrics,
        "paired_family_bootstraps": comparisons,
        "same_base_operational_noise_controls": same_base_audit(model_by_id, normalized_speed),
        "frontier_operational_estimate_method": frontier_method,
        "external_anchor_checks": [
            row for row in frontier_rows if row["disclosed_total_parameters_b"] != ""
        ],
        "limitations": [
            "OpenRouter throughput measures serving stacks, hardware, batching, quantization, traffic, and routing as well as model compute.",
            "Current snapshot prices are not historical launch prices; older models may have been repriced.",
            "Epoch total parameters do not reveal active MoE parameters, which are closer to per-token inference cost.",
            "No disclosed Anthropic frontier parameter checkpoints exist in this calibration, so developer-family transfer is a real extrapolation.",
            "Provider-normalization reduces observed infrastructure confounding but cannot identify undisclosed batching or speculative decoding.",
        ],
        "source_manifest": {
            str(RAW_SNAPSHOT.relative_to(ROOT)): sha256(RAW_SNAPSHOT),
            str(MODEL_SIGNALS.relative_to(ROOT)): sha256(MODEL_SIGNALS),
            str(PROVIDER_SIGNALS.relative_to(ROOT)): sha256(PROVIDER_SIGNALS),
            str(DAILY_THROUGHPUT.relative_to(ROOT)): sha256(DAILY_THROUGHPUT),
            str(UNIFIED.relative_to(ROOT)): sha256(UNIFIED),
            str(K3_EVIDENCE_PATH.relative_to(ROOT)): sha256(K3_EVIDENCE_PATH),
        },
        "outputs": {
            "match_audit": str(MATCH_AUDIT.relative_to(ROOT)),
            "calibration": str(CALIBRATION.relative_to(ROOT)),
            "heldout_predictions": str(PREDICTIONS.relative_to(ROOT)),
            "frontier_operational_estimates": str(FRONTIER.relative_to(ROOT)),
        },
    }
    RESULT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(RESULT),
                "catalog_models": len(audit),
                "manual_matches": match_counts["matched_epoch_manual"],
                "calibration_checkpoints": len(calibration),
                "calibration_families": len({row["family"] for row in calibration}),
                "tok_s_supported": tok_s_supported,
                "recommended_incremental_tok_s_weight": recommended_incremental_weight,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
