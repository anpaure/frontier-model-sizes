#!/usr/bin/env python3
"""Collect reproducible OpenRouter model pricing and operational-speed signals.

The catalog and provider endpoints are public OpenRouter data.  The collector
preserves the complete responses in a compressed snapshot and writes two flat
tables:

* provider-level observations (prices, quantization, recent p50 throughput);
* model-level aggregates (robust one-week throughput and price summaries).

Throughput is an operational measurement, not a direct architecture label.  It
depends on provider hardware, batching, quantization, service tier, and traffic.
Consequently, downstream models must retain provider/quantization controls and
must demonstrate held-out improvement before receiving forecast weight.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
# Downstream artifacts deliberately retain their original compatibility names;
# the observation date lives inside every row, raw response, and audit record.
# Keeping these concepts separate lets a refresh update the evidence without
# either breaking consumers or falsely dating a newly fetched observation.
COMPATIBILITY_FILE_DATE = "2026-07-18"
DEFAULT_SNAPSHOT_DATE = datetime.now(tz=UTC).date().isoformat()
DEFAULT_WORKERS = 8
CATALOG_URL = "https://openrouter.ai/api/v1/models"
FRONTEND_BASE = "https://openrouter.ai/api/frontend/v1"
USER_AGENT = "frontier-parameter-research/1.0 (OpenRouter public-data audit)"


PROVIDER_FIELDS = [
    "snapshot_date",
    "fetched_at_utc",
    "openrouter_model_id",
    "openrouter_model_name",
    "canonical_slug",
    "author",
    "created_date",
     "permaslug",
    "hugging_face_id",
    "model_context_length_tokens",
    "endpoint_id",
    "provider_name",
    "provider_display_name",
    "provider_slug",
    "provider_model_id",
    "provider_region",
    "variant",
    "quantization",
    "endpoint_context_length_tokens",
    "max_completion_tokens",
    "prompt_usd_per_mtoken",
    "completion_usd_per_mtoken",
    "cache_read_usd_per_mtoken",
    "cache_write_usd_per_mtoken",
    "p50_throughput_tps_30m",
    "p75_throughput_tps_30m",
    "p90_throughput_tps_30m",
    "p50_latency_seconds_30m",
    "request_count_30m",
    "window_minutes",
    "throughput_daily_observations_1w",
    "throughput_median_tps_1w",
    "throughput_mean_tps_1w",
    "throughput_min_tps_1w",
    "throughput_max_tps_1w",
    "throughput_service_tier_1w",
    "endpoint_source_url",
    "throughput_source_url",
]

TIER_FIELDS = [
    "snapshot_date",
    "fetched_at_utc",
    "openrouter_model_id",
    "openrouter_model_name",
    "canonical_slug",
    "permaslug",
    "author",
    "created_date",
    "endpoint_id",
    "endpoint_tier_key",
    "service_tier",
    "provider_name",
    "provider_display_name",
    "provider_slug",
    "provider_region",
    "variant",
    "quantization",
    "endpoint_context_length_tokens",
    "max_prompt_tokens",
    "max_completion_tokens",
    "prompt_usd_per_mtoken",
    "completion_usd_per_mtoken",
    "cache_read_usd_per_mtoken",
    "cache_write_usd_per_mtoken",
    "high_context_min_prompt_tokens",
    "high_context_prompt_usd_per_mtoken",
    "high_context_completion_usd_per_mtoken",
    "high_context_cache_read_usd_per_mtoken",
    "high_context_cache_write_usd_per_mtoken",
    "p50_throughput_tps_30m",
    "p75_throughput_tps_30m",
    "p90_throughput_tps_30m",
    "p95_throughput_tps_30m",
    "p99_throughput_tps_30m",
    "p50_latency_seconds_30m",
    "p75_latency_seconds_30m",
    "p90_latency_seconds_30m",
    "p95_latency_seconds_30m",
    "p99_latency_seconds_30m",
    "request_count_30m",
    "window_minutes",
    "pricing_source",
    "stats_source",
    "endpoint_source_url",
]

DAILY_FIELDS = [
    "snapshot_date",
    "fetched_at_utc",
    "observation_time_raw",
    "observation_date",
    "openrouter_model_id",
    "openrouter_model_name",
    "canonical_slug",
    "endpoint_tier_key",
    "endpoint_id",
    "service_tier",
    "provider_name",
    "provider_display_name",
    "provider_slug",
    "provider_region",
    "variant",
    "quantization",
    "throughput_tps",
    "endpoint_metadata_match",
    "throughput_source_url",
]

MODEL_FIELDS = [
    "snapshot_date",
    "fetched_at_utc",
    "openrouter_model_id",
    "openrouter_model_name",
    "canonical_slug",
    "permaslug",
    "author",
    "hugging_face_id",
    "created_date",
    "context_length_tokens",
    "endpoint_count",
    "endpoint_count_with_price",
    "endpoint_count_with_throughput_30m",
    "endpoint_count_with_throughput_1w",
    "provider_count_with_throughput_1w",
    "quantizations",
    "prompt_price_min_usd_per_mtoken",
    "prompt_price_median_usd_per_mtoken",
    "prompt_price_max_usd_per_mtoken",
    "completion_price_min_usd_per_mtoken",
    "completion_price_median_usd_per_mtoken",
    "completion_price_max_usd_per_mtoken",
    "blended_price_geomean_median_usd_per_mtoken",
    "throughput_best_provider_median_tps_1w",
    "throughput_median_provider_median_tps_1w",
    "throughput_worst_provider_median_tps_1w",
    "throughput_best_p50_tps_30m",
    "throughput_median_p50_tps_30m",
    "catalog_source_url",
    "endpoint_source_url",
    "throughput_source_url",
]


@dataclass(frozen=True)
class ModelTask:
    model_id: str
    model_name: str
    canonical_slug: str
    permaslug: str
    author: str
    hugging_face_id: str
    created: int | None
    context_length: int | None


def _request_json(url: str, *, attempts: int = 4) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(attempts):
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "application/json",
                "Referer": "https://openrouter.ai/models",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as error:
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(0.5 * (2**attempt))
    raise RuntimeError(f"Failed after {attempts} attempts: {url}: {last_error}")


def _date_from_timestamp(value: int | None) -> str:
    if value is None:
        return ""
    return datetime.fromtimestamp(value, tz=UTC).date().isoformat()


def _number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _per_million(value: Any) -> float | None:
    parsed = _number(value)
    return None if parsed is None else parsed * 1_000_000


def _median(values: Iterable[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    return None if not clean else float(statistics.median(clean))


def _mean(values: Iterable[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    return None if not clean else float(statistics.fmean(clean))


def _minimum(values: Iterable[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    return None if not clean else float(min(clean))


def _maximum(values: Iterable[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    return None if not clean else float(max(clean))


def _seconds(value: Any) -> float | None:
    parsed = _number(value)
    return None if parsed is None else parsed / 1000


def _pricing_override(pricing: dict[str, Any]) -> dict[str, Any]:
    """Return the first token-count pricing override without inventing tiers."""
    overrides = [
        row
        for row in (pricing.get("overrides") or [])
        if isinstance(row, dict) and _number(row.get("min_prompt_tokens")) is not None
    ]
    if not overrides:
        return {}
    return min(overrides, key=lambda row: float(row["min_prompt_tokens"]))


def _display_pricing(tier: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Extract base and first high-context prices from a service-tier display block."""
    labels = {
        "Input Price": "prompt",
        "Output Price": "completion",
        "Cache Read": "input_cache_read",
        "Cache Write": "input_cache_write",
    }
    base: dict[str, Any] = {}
    high: dict[str, Any] = {}
    for entry in tier.get("display_pricing") or []:
        key = labels.get(str(entry.get("sku_label") or ""))
        if key is None:
            continue
        base[key] = entry.get("price")
        price_tiers = entry.get("tiers") or []
        if len(price_tiers) >= 2:
            high[key] = price_tiers[-1].get("price")
    return base, high


def _endpoint_tier_rows(
    task: ModelTask,
    fetched_at: str,
    snapshot_date: str,
    endpoint_url: str,
    endpoint_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    """Flatten current endpoint price and 30-minute stats by service tier.

    The frontend endpoint response contains separate default/flex/priority
    measurements and, for some models, higher prices beyond a prompt-token
    threshold.  Keeping these rows separate prevents service policy and
    long-context pricing from being silently pooled into one model signal.
    """
    rows: list[dict[str, Any]] = []
    for endpoint in endpoint_payload.get("data") or []:
        endpoint_id = str(endpoint.get("id") or "")
        base_pricing = endpoint.get("pricing") or {}
        base_override = _pricing_override(base_pricing)
        stats_by_tier = endpoint.get("statsByTier") or {}
        pricing_by_tier = endpoint.get("tiers") or {}
        service_tiers = sorted({"default", *stats_by_tier, *pricing_by_tier})
        for service_tier in service_tiers:
            if service_tier == "default":
                pricing = base_pricing
                high_pricing = base_override
                stats = stats_by_tier.get("default") or endpoint.get("stats") or {}
                pricing_source = "endpoint.pricing"
                stats_source = (
                    "endpoint.statsByTier.default"
                    if stats_by_tier.get("default")
                    else "endpoint.stats"
                )
            else:
                pricing, high_pricing = _display_pricing(
                    pricing_by_tier.get(service_tier) or {}
                )
                stats = stats_by_tier.get(service_tier) or {}
                pricing_source = f"endpoint.tiers.{service_tier}.display_pricing"
                stats_source = f"endpoint.statsByTier.{service_tier}"
            threshold = base_override.get("min_prompt_tokens") if high_pricing else None
            rows.append(
                {
                    "snapshot_date": snapshot_date,
                    "fetched_at_utc": fetched_at,
                    "openrouter_model_id": task.model_id,
                    "openrouter_model_name": task.model_name,
                    "canonical_slug": task.canonical_slug,
                    "permaslug": task.permaslug,
                    "author": task.author,
                    "created_date": _date_from_timestamp(task.created),
                    "endpoint_id": endpoint_id,
                    "endpoint_tier_key": f"{endpoint_id}::{service_tier}",
                    "service_tier": service_tier,
                    "provider_name": endpoint.get("provider_name") or "",
                    "provider_display_name": endpoint.get("provider_display_name")
                    or endpoint.get("provider_name")
                    or "",
                    "provider_slug": endpoint.get("provider_slug") or "",
                    "provider_region": endpoint.get("provider_region") or "",
                    "variant": endpoint.get("variant") or "standard",
                    "quantization": endpoint.get("quantization") or "unknown",
                    "endpoint_context_length_tokens": endpoint.get("context_length"),
                    "max_prompt_tokens": endpoint.get("max_prompt_tokens"),
                    "max_completion_tokens": endpoint.get("max_completion_tokens"),
                    "prompt_usd_per_mtoken": _per_million(pricing.get("prompt")),
                    "completion_usd_per_mtoken": _per_million(
                        pricing.get("completion")
                    ),
                    "cache_read_usd_per_mtoken": _per_million(
                        pricing.get("input_cache_read")
                    ),
                    "cache_write_usd_per_mtoken": _per_million(
                        pricing.get("input_cache_write")
                    ),
                    "high_context_min_prompt_tokens": threshold,
                    "high_context_prompt_usd_per_mtoken": _per_million(
                        high_pricing.get("prompt")
                    ),
                    "high_context_completion_usd_per_mtoken": _per_million(
                        high_pricing.get("completion")
                    ),
                    "high_context_cache_read_usd_per_mtoken": _per_million(
                        high_pricing.get("input_cache_read")
                    ),
                    "high_context_cache_write_usd_per_mtoken": _per_million(
                        high_pricing.get("input_cache_write")
                    ),
                    "p50_throughput_tps_30m": _number(
                        stats.get("p50_throughput")
                    ),
                    "p75_throughput_tps_30m": _number(
                        stats.get("p75_throughput")
                    ),
                    "p90_throughput_tps_30m": _number(
                        stats.get("p90_throughput")
                    ),
                    "p95_throughput_tps_30m": _number(
                        stats.get("p95_throughput")
                    ),
                    "p99_throughput_tps_30m": _number(
                        stats.get("p99_throughput")
                    ),
                    "p50_latency_seconds_30m": _seconds(stats.get("p50_latency")),
                    "p75_latency_seconds_30m": _seconds(stats.get("p75_latency")),
                    "p90_latency_seconds_30m": _seconds(stats.get("p90_latency")),
                    "p95_latency_seconds_30m": _seconds(stats.get("p95_latency")),
                    "p99_latency_seconds_30m": _seconds(stats.get("p99_latency")),
                    "request_count_30m": stats.get("request_count"),
                    "window_minutes": stats.get("window_minutes"),
                    "pricing_source": pricing_source,
                    "stats_source": stats_source,
                    "endpoint_source_url": endpoint_url,
                }
            )
    return rows


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.12g}"
    return value


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="raise", lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _task_from_catalog(model: dict[str, Any]) -> ModelTask | None:
    model_id = str(model.get("id") or "")
    canonical_slug = str(model.get("canonical_slug") or "")
    architecture = model.get("architecture") or {}
    output_modalities = architecture.get("output_modalities") or []
    if (
        not model_id
        or not canonical_slug
        or model_id.startswith("~")
        or model_id.startswith("openrouter/")
        or ":free" in model_id
        or (output_modalities and "text" not in output_modalities)
    ):
        return None
    return ModelTask(
        model_id=model_id,
        model_name=str(model.get("name") or model_id),
        canonical_slug=canonical_slug,
        permaslug=canonical_slug,
        author=model_id.split("/", 1)[0],
        hugging_face_id=str(model.get("hugging_face_id") or ""),
        created=int(model["created"]) if model.get("created") is not None else None,
        context_length=int(model["context_length"]) if model.get("context_length") is not None else None,
    )


def _urls(task: ModelTask) -> tuple[str, str]:
    encoded = urllib.parse.quote(task.permaslug, safe="")
    endpoint_url = f"{FRONTEND_BASE}/stats/endpoint?permaslug={encoded}&variant=standard"
    throughput_url = f"{FRONTEND_BASE}/stats/throughput-comparison?permaslug={encoded}&timeRange=1w"
    return endpoint_url, throughput_url


def _collect_one(task: ModelTask) -> dict[str, Any]:
    endpoint_url, throughput_url = _urls(task)
    errors: list[dict[str, str]] = []
    try:
        endpoint_payload = _request_json(endpoint_url)
    except Exception as error:  # retain failure and continue to other models
        endpoint_payload = {"data": []}
        errors.append({"kind": "endpoint", "url": endpoint_url, "error": str(error)})
    try:
        throughput_payload = _request_json(throughput_url)
    except Exception as error:  # models without traffic may legitimately have no series
        throughput_payload = {"data": []}
        errors.append({"kind": "throughput", "url": throughput_url, "error": str(error)})
    return {
        "task": task,
        "endpoint_url": endpoint_url,
        "throughput_url": throughput_url,
        "endpoint_payload": endpoint_payload,
        "throughput_payload": throughput_payload,
        "errors": errors,
    }


def _throughput_by_endpoint(
    payload: dict[str, Any], *, service_tier: str = "default"
) -> dict[str, list[float]]:
    """Return one-week observations for one serving tier per endpoint.

    OpenRouter's public series can contain ``default``, ``priority``, and
    ``flex`` observations under the same endpoint UUID.  Pooling those tiers
    creates an avoidable serving-policy confound.  The regression therefore
    uses only the default tier; the lossless daily table retains every tier.
    """
    values: dict[str, list[float]] = {}
    for point in payload.get("data") or []:
        for endpoint_tier, raw_value in (point.get("y") or {}).items():
            parts = str(endpoint_tier).split("::", 1)
            endpoint_id = parts[0]
            tier = parts[1] if len(parts) == 2 else "default"
            if tier != service_tier:
                continue
            value = _number(raw_value)
            if value is not None:
                values.setdefault(endpoint_id, []).append(value)
    return values


def _daily_rows(
    task: ModelTask,
    fetched_at: str,
    snapshot_date: str,
    throughput_url: str,
    throughput_payload: dict[str, Any],
    provider_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Flatten every dated provider/tier throughput observation losslessly."""
    metadata = {str(row["endpoint_id"]): row for row in provider_rows if row["endpoint_id"]}
    rows: list[dict[str, Any]] = []
    for point in throughput_payload.get("data") or []:
        observed_raw = str(point.get("x") or "")
        observed_date = observed_raw[:10] if len(observed_raw) >= 10 else ""
        for endpoint_tier, raw_value in (point.get("y") or {}).items():
            value = _number(raw_value)
            if value is None:
                continue
            parts = str(endpoint_tier).split("::", 1)
            endpoint_id = parts[0]
            service_tier = parts[1] if len(parts) == 2 else "default"
            provider = metadata.get(endpoint_id, {})
            rows.append(
                {
                    "snapshot_date": snapshot_date,
                    "fetched_at_utc": fetched_at,
                    "observation_time_raw": observed_raw,
                    "observation_date": observed_date,
                    "openrouter_model_id": task.model_id,
                    "openrouter_model_name": task.model_name,
                    "canonical_slug": task.canonical_slug,
                    "author": task.author,
                    "created_date": _date_from_timestamp(task.created),
                    "endpoint_tier_key": str(endpoint_tier),
                    "endpoint_id": endpoint_id,
                    "service_tier": service_tier,
                    "provider_name": provider.get("provider_name", ""),
                    "provider_display_name": provider.get("provider_display_name", ""),
                    "provider_slug": provider.get("provider_slug", ""),
                    "provider_region": provider.get("provider_region", ""),
                    "variant": provider.get("variant", ""),
                    "quantization": provider.get("quantization", ""),
                    "throughput_tps": value,
                    "endpoint_metadata_match": bool(provider),
                    "throughput_source_url": throughput_url,
                }
            )
    return rows


def _provider_rows(
    task: ModelTask,
    fetched_at: str,
    snapshot_date: str,
    endpoint_url: str,
    throughput_url: str,
    endpoint_payload: dict[str, Any],
    throughput_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    history = _throughput_by_endpoint(throughput_payload)
    rows: list[dict[str, Any]] = []
    for endpoint in endpoint_payload.get("data") or []:
        endpoint_id = str(endpoint.get("id") or "")
        pricing = endpoint.get("pricing") or {}
        stats = (endpoint.get("statsByTier") or {}).get("default") or endpoint.get("stats") or {}
        series = history.get(endpoint_id, [])
        rows.append(
            {
                "snapshot_date": snapshot_date,
                "fetched_at_utc": fetched_at,
                "openrouter_model_id": task.model_id,
                "openrouter_model_name": task.model_name,
                "canonical_slug": task.canonical_slug,
                "permaslug": task.permaslug,
                "author": task.author,
                "hugging_face_id": task.hugging_face_id,
                "created_date": _date_from_timestamp(task.created),
                "model_context_length_tokens": task.context_length,
                "endpoint_id": endpoint_id,
                "provider_name": endpoint.get("provider_name") or "",
                "provider_display_name": endpoint.get("provider_display_name") or endpoint.get("provider_name") or "",
                "provider_slug": endpoint.get("provider_slug") or "",
                "provider_model_id": endpoint.get("provider_model_id") or "",
                "provider_region": endpoint.get("provider_region") or "",
                "variant": endpoint.get("variant") or "standard",
                "quantization": endpoint.get("quantization") or "unknown",
                "endpoint_context_length_tokens": endpoint.get("context_length"),
                "max_completion_tokens": endpoint.get("max_completion_tokens"),
                "prompt_usd_per_mtoken": _per_million(pricing.get("prompt")),
                "completion_usd_per_mtoken": _per_million(pricing.get("completion")),
                "cache_read_usd_per_mtoken": _per_million(pricing.get("input_cache_read")),
                "cache_write_usd_per_mtoken": _per_million(pricing.get("input_cache_write")),
                "p50_throughput_tps_30m": _number(stats.get("p50_throughput")),
                "p75_throughput_tps_30m": _number(stats.get("p75_throughput")),
                "p90_throughput_tps_30m": _number(stats.get("p90_throughput")),
                "p50_latency_seconds_30m": _seconds(stats.get("p50_latency")),
                "request_count_30m": stats.get("request_count"),
                "window_minutes": stats.get("window_minutes"),
                "throughput_daily_observations_1w": len(series),
                "throughput_median_tps_1w": _median(series),
                "throughput_mean_tps_1w": _mean(series),
                "throughput_min_tps_1w": _minimum(series),
                "throughput_max_tps_1w": _maximum(series),
                "throughput_service_tier_1w": "default",
                "endpoint_source_url": endpoint_url,
                "throughput_source_url": throughput_url,
            }
        )
    return rows


def _model_row(task: ModelTask, fetched_at: str, snapshot_date: str, provider_rows: list[dict[str, Any]]) -> dict[str, Any]:
    endpoint_url, throughput_url = _urls(task)
    prompts = [_number(row["prompt_usd_per_mtoken"]) for row in provider_rows]
    completions = [_number(row["completion_usd_per_mtoken"]) for row in provider_rows]
    geomeans = [
        math.sqrt(prompt * completion)
        for prompt, completion in zip(prompts, completions, strict=True)
        if prompt is not None and completion is not None and prompt > 0 and completion > 0
    ]
    week = [_number(row["throughput_median_tps_1w"]) for row in provider_rows]
    recent = [_number(row["p50_throughput_tps_30m"]) for row in provider_rows]
    quantizations = sorted({str(row["quantization"]) for row in provider_rows if row["quantization"]})
    week_provider_names = {
        str(row["provider_slug"] or row["provider_display_name"])
        for row in provider_rows
        if _number(row["throughput_median_tps_1w"]) is not None
    }
    return {
        "snapshot_date": snapshot_date,
        "fetched_at_utc": fetched_at,
        "openrouter_model_id": task.model_id,
        "openrouter_model_name": task.model_name,
        "canonical_slug": task.canonical_slug,
        "permaslug": task.permaslug,
        "author": task.author,
        "hugging_face_id": task.hugging_face_id,
        "created_date": _date_from_timestamp(task.created),
        "context_length_tokens": task.context_length,
        "endpoint_count": len(provider_rows),
        "endpoint_count_with_price": sum(
            prompt is not None and completion is not None for prompt, completion in zip(prompts, completions, strict=True)
        ),
        "endpoint_count_with_throughput_30m": sum(value is not None for value in recent),
        "endpoint_count_with_throughput_1w": sum(value is not None for value in week),
        "provider_count_with_throughput_1w": len(week_provider_names),
        "quantizations": "|".join(quantizations),
        "prompt_price_min_usd_per_mtoken": _minimum(prompts),
        "prompt_price_median_usd_per_mtoken": _median(prompts),
        "prompt_price_max_usd_per_mtoken": _maximum(prompts),
        "completion_price_min_usd_per_mtoken": _minimum(completions),
        "completion_price_median_usd_per_mtoken": _median(completions),
        "completion_price_max_usd_per_mtoken": _maximum(completions),
        "blended_price_geomean_median_usd_per_mtoken": _median(geomeans),
        "throughput_best_provider_median_tps_1w": _maximum(week),
        "throughput_median_provider_median_tps_1w": _median(week),
        "throughput_worst_provider_median_tps_1w": _minimum(week),
        "throughput_best_p50_tps_30m": _maximum(recent),
        "throughput_median_p50_tps_30m": _median(recent),
        "catalog_source_url": CATALOG_URL,
        "endpoint_source_url": endpoint_url,
        "throughput_source_url": throughput_url,
    }


def collect(snapshot_date: str, workers: int) -> dict[str, Any]:
    fetched_at = datetime.now(tz=UTC).replace(microsecond=0).isoformat()
    catalog = _request_json(CATALOG_URL)
    tasks = [task for item in catalog.get("data") or [] if (task := _task_from_catalog(item)) is not None]
    tasks.sort(key=lambda task: task.model_id)

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {pool.submit(_collect_one, task): task for task in tasks}
        for future in as_completed(future_map):
            results.append(future.result())
    results.sort(key=lambda result: result["task"].model_id)

    provider_rows: list[dict[str, Any]] = []
    tier_rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    raw_models: list[dict[str, Any]] = []
    for result in results:
        task: ModelTask = result["task"]
        rows = _provider_rows(
            task,
            fetched_at,
            snapshot_date,
            result["endpoint_url"],
            result["throughput_url"],
            result["endpoint_payload"],
            result["throughput_payload"],
        )
        provider_rows.extend(rows)
        tier_rows.extend(
            _endpoint_tier_rows(
                task,
                fetched_at,
                snapshot_date,
                result["endpoint_url"],
                result["endpoint_payload"],
            )
        )
        daily_rows.extend(
            _daily_rows(
                task,
                fetched_at,
                snapshot_date,
                result["throughput_url"],
                result["throughput_payload"],
                rows,
            )
        )
        model_rows.append(_model_row(task, fetched_at, snapshot_date, rows))
        failures.extend({"openrouter_model_id": task.model_id, **failure} for failure in result["errors"])
        raw_models.append(
            {
                "openrouter_model_id": task.model_id,
                "canonical_slug": task.canonical_slug,
                "endpoint_url": result["endpoint_url"],
                "throughput_url": result["throughput_url"],
                "endpoint_payload": result["endpoint_payload"],
                "throughput_payload": result["throughput_payload"],
                "errors": result["errors"],
            }
        )

    provider_rows.sort(key=lambda row: (row["openrouter_model_id"], row["provider_slug"], row["endpoint_id"]))
    tier_rows.sort(
        key=lambda row: (
            row["openrouter_model_id"],
            row["provider_slug"],
            row["endpoint_id"],
            row["service_tier"],
        )
    )
    daily_rows.sort(
        key=lambda row: (
            row["openrouter_model_id"],
            row["observation_time_raw"],
            row["endpoint_tier_key"],
        )
    )
    model_rows.sort(key=lambda row: row["openrouter_model_id"])
    return {
        "snapshot_date": snapshot_date,
        "fetched_at_utc": fetched_at,
        "catalog_url": CATALOG_URL,
        "catalog_payload": catalog,
        "raw_models": raw_models,
        "provider_rows": provider_rows,
        "tier_rows": tier_rows,
        "daily_rows": daily_rows,
        "model_rows": model_rows,
        "failures": failures,
    }


def write_outputs(
    collected: dict[str, Any],
    *,
    file_date: str = COMPATIBILITY_FILE_DATE,
) -> dict[str, Any]:
    snapshot_date = collected["snapshot_date"]
    source_dir = ROOT / "sources"
    output_dir = ROOT / "outputs" / "019f6c42-2d53-7743-ab07-6293e2618dd7"
    raw_path = source_dir / f"openrouter_operational_snapshot_{file_date}.json.gz"
    provider_path = source_dir / f"openrouter_provider_signals_{file_date}.csv"
    tier_path = source_dir / f"openrouter_endpoint_tier_signals_{file_date}.csv"
    daily_path = source_dir / f"openrouter_throughput_daily_{file_date}.csv"
    model_path = source_dir / f"openrouter_model_signals_{file_date}.csv"
    audit_path = output_dir / f"openrouter_collection_audit_{file_date}.json"
    source_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_payload = {
        "schema_version": "1.0",
        "snapshot_date": snapshot_date,
        "compatibility_filename_date": file_date,
        "fetched_at_utc": collected["fetched_at_utc"],
        "catalog_url": collected["catalog_url"],
        "catalog_payload": collected["catalog_payload"],
        "models": collected["raw_models"],
    }
    with gzip.open(raw_path, "wt", encoding="utf-8") as handle:
        json.dump(raw_payload, handle, ensure_ascii=False, separators=(",", ":"))

    _write_csv(provider_path, PROVIDER_FIELDS, collected["provider_rows"])
    _write_csv(tier_path, TIER_FIELDS, collected["tier_rows"])
    _write_csv(daily_path, DAILY_FIELDS, collected["daily_rows"])
    _write_csv(model_path, MODEL_FIELDS, collected["model_rows"])

    endpoint_ids = [row["endpoint_id"] for row in collected["provider_rows"] if row["endpoint_id"]]
    provider_observation_keys = [
        (row["openrouter_model_id"], row["endpoint_id"], row["variant"])
        for row in collected["provider_rows"]
        if row["endpoint_id"]
    ]
    endpoint_id_counts: dict[str, int] = {}
    for endpoint_id in endpoint_ids:
        endpoint_id_counts[endpoint_id] = endpoint_id_counts.get(endpoint_id, 0) + 1
    model_ids = [row["openrouter_model_id"] for row in collected["model_rows"]]
    daily_keys = [
        (row["openrouter_model_id"], row["observation_time_raw"], row["endpoint_tier_key"])
        for row in collected["daily_rows"]
    ]
    tier_keys = [
        (row["openrouter_model_id"], row["endpoint_id"], row["service_tier"])
        for row in collected["tier_rows"]
    ]
    daily_dates = [row["observation_date"] for row in collected["daily_rows"] if row["observation_date"]]
    audit = {
        "schema_version": "1.0",
        "snapshot_date": snapshot_date,
        "compatibility_filename_date": file_date,
        "fetched_at_utc": collected["fetched_at_utc"],
        "catalog_model_count": len(collected["catalog_payload"].get("data") or []),
        "eligible_text_model_count": len(collected["model_rows"]),
        "provider_endpoint_row_count": len(collected["provider_rows"]),
        "endpoint_tier_row_count": len(collected["tier_rows"]),
        "endpoint_tier_service_tier_counts": {
            tier: sum(row["service_tier"] == tier for row in collected["tier_rows"])
            for tier in sorted({row["service_tier"] for row in collected["tier_rows"]})
        },
        "endpoint_tier_rows_with_high_context_price": sum(
            row["high_context_min_prompt_tokens"] is not None
            for row in collected["tier_rows"]
        ),
        "daily_throughput_row_count": len(collected["daily_rows"]),
        "daily_throughput_model_count": len(
            {row["openrouter_model_id"] for row in collected["daily_rows"]}
        ),
        "daily_throughput_endpoint_tier_count": len(
            {
                (row["openrouter_model_id"], row["endpoint_tier_key"])
                for row in collected["daily_rows"]
            }
        ),
        "daily_throughput_date_min": min(daily_dates) if daily_dates else None,
        "daily_throughput_date_max": max(daily_dates) if daily_dates else None,
        "models_with_price": sum(int(row["endpoint_count_with_price"]) > 0 for row in collected["model_rows"]),
        "models_with_throughput_30m": sum(
            int(row["endpoint_count_with_throughput_30m"]) > 0 for row in collected["model_rows"]
        ),
        "models_with_throughput_1w": sum(
            int(row["endpoint_count_with_throughput_1w"]) > 0 for row in collected["model_rows"]
        ),
        "unique_model_ids": len(model_ids) == len(set(model_ids)),
        "unique_provider_observation_keys": len(provider_observation_keys) == len(set(provider_observation_keys)),
        "unique_endpoint_tier_keys": len(tier_keys) == len(set(tier_keys)),
        "unique_daily_throughput_keys": len(daily_keys) == len(set(daily_keys)),
        "daily_rows_without_endpoint_metadata": sum(
            not row["endpoint_metadata_match"] for row in collected["daily_rows"]
        ),
        "reused_endpoint_id_count": sum(count > 1 for count in endpoint_id_counts.values()),
        "reused_endpoint_ids": sorted(endpoint_id for endpoint_id, count in endpoint_id_counts.items() if count > 1),
        "failure_count": len(collected["failures"]),
        "failures": collected["failures"],
        "files": {
            "raw_snapshot": str(raw_path.relative_to(ROOT)),
            "provider_signals": str(provider_path.relative_to(ROOT)),
            "endpoint_tier_signals": str(tier_path.relative_to(ROOT)),
            "daily_throughput": str(daily_path.relative_to(ROOT)),
            "model_signals": str(model_path.relative_to(ROOT)),
        },
        "method_notes": [
            "List prices are recorded per provider endpoint and converted from USD/token to USD/million tokens.",
            "The 30-minute p50 throughput includes request counts and is retained as a volatile audit signal.",
            "All available 30-minute throughput and latency percentiles are recorded separately for default, flex, and priority service tiers.",
            "Base and first high-context price schedules are retained per endpoint and service tier; no prompt-length mixture is assumed.",
            "The preferred speed summary is each provider endpoint's median across the one-week daily series.",
            "Default, priority, and flex service tiers are retained separately in the daily table; only default-tier observations feed the regression aggregate.",
            "Endpoint UUIDs may be reused by OpenRouter model variants; uniqueness is enforced on model ID + endpoint ID + variant.",
            "Throughput is not a parameter-count label; provider, quantization, batching, and service-tier effects remain confounders.",
        ],
    }
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "raw": raw_path,
        "providers": provider_path,
        "tiers": tier_path,
        "daily": daily_path,
        "models": model_path,
        "audit": audit_path,
        "summary": audit,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-date", default=DEFAULT_SNAPSHOT_DATE)
    parser.add_argument(
        "--file-date",
        default=COMPATIBILITY_FILE_DATE,
        help=(
            "Compatibility date embedded in canonical filenames. The actual "
            "observation date is --snapshot-date and is preserved in every row."
        ),
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--skip-history",
        action="store_true",
        help="Do not archive the prior/current canonical snapshot in the immutable history ledger.",
    )
    args = parser.parse_args()
    history_script = ROOT / "build_openrouter_snapshot_history.py"
    if not args.skip_history and history_script.exists():
        subprocess.run([sys.executable, str(history_script)], cwd=ROOT, check=True)
    collected = collect(args.snapshot_date, max(1, args.workers))
    outputs = write_outputs(collected, file_date=args.file_date)
    if not args.skip_history and history_script.exists():
        subprocess.run([sys.executable, str(history_script)], cwd=ROOT, check=True)
    print(json.dumps({"files": {key: str(value) for key, value in outputs.items() if key != "summary"}, **outputs["summary"]}, indent=2))


if __name__ == "__main__":
    main()
