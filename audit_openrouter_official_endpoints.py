#!/usr/bin/env python3
"""Crosscheck the frozen OpenRouter operational panel against the official API.

The performance panel comes from OpenRouter's public model-page statistics
payload because it exposes provider-level throughput percentiles and daily
series.  Prices are independently verified here against the documented
``/api/v1/models/{model}/endpoints`` API.  Every response is preserved so a
future audit can distinguish source changes from parser changes.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DATE = "2026-07-18"
OUT = ROOT / "outputs/019f6c42-2d53-7743-ab07-6293e2618dd7"
MODELS = ROOT / f"sources/openrouter_model_signals_{DATE}.csv"
TIERS = ROOT / f"sources/openrouter_endpoint_tier_signals_{DATE}.csv"
RAW = ROOT / f"sources/openrouter_official_endpoint_snapshot_{DATE}.json.gz"
OFFICIAL = ROOT / f"sources/openrouter_official_endpoint_prices_{DATE}.csv"
COMPARISON = OUT / f"openrouter_official_endpoint_crosscheck_{DATE}.csv"
AUDIT = OUT / f"openrouter_official_endpoint_audit_{DATE}.json"
BASE_URL = "https://openrouter.ai/api/v1/models"
USER_AGENT = "frontier-parameter-research/1.0 (OpenRouter official API crosscheck)"

OFFICIAL_FIELDS = [
    "fetched_at_utc",
    "openrouter_model_id",
    "openrouter_model_name",
    "provider_name",
    "provider_tag",
    "service_tier",
    "quantization",
    "context_length_tokens",
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
    "status",
    "uptime_last_30m",
    "uptime_last_5m",
    "uptime_last_1d",
    "source_url",
]


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        if len(fields) != len(set(fields)):
            raise ValueError(f"Duplicate CSV fields in {path}")
        return list(reader)


def number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def per_million(value: Any) -> float | None:
    parsed = number(value)
    return None if parsed is None else parsed * 1_000_000


def first_override(pricing: dict[str, Any]) -> dict[str, Any]:
    candidates = [
        row
        for row in (pricing.get("overrides") or [])
        if isinstance(row, dict) and number(row.get("min_prompt_tokens")) is not None
    ]
    return {} if not candidates else min(candidates, key=lambda row: float(row["min_prompt_tokens"]))


def service_tier(tag: str) -> str:
    for tier in ("flex", "priority"):
        if tag.endswith(f"/{tier}"):
            return tier
    return "default"


def request_json(url: str, attempts: int = 4) -> dict[str, Any]:
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(
                url, headers={"Accept": "application/json", "User-Agent": USER_AGENT}
            )
            with urllib.request.urlopen(request, timeout=45) as response:
                return json.loads(response.read().decode("utf-8"))
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            TimeoutError,
            json.JSONDecodeError,
        ) as error:
            last = error
            if attempt + 1 < attempts:
                time.sleep(0.5 * 2**attempt)
    raise RuntimeError(f"Failed after {attempts} attempts: {url}: {last}")


def endpoint_url(model_id: str) -> str:
    return f"{BASE_URL}/{urllib.parse.quote(model_id, safe='/:')}/endpoints"


def collect_one(model_id: str) -> dict[str, Any]:
    url = endpoint_url(model_id)
    try:
        return {"model_id": model_id, "url": url, "payload": request_json(url), "error": ""}
    except Exception as error:
        return {"model_id": model_id, "url": url, "payload": {"data": {}}, "error": str(error)}


def flatten(result: dict[str, Any], fetched_at: str) -> list[dict[str, Any]]:
    data = result["payload"].get("data") or {}
    output: list[dict[str, Any]] = []
    for endpoint in data.get("endpoints") or []:
        pricing = endpoint.get("pricing") or {}
        high = first_override(pricing)
        tag = str(endpoint.get("tag") or "")
        output.append(
            {
                "fetched_at_utc": fetched_at,
                "openrouter_model_id": result["model_id"],
                "openrouter_model_name": data.get("name") or result["model_id"],
                "provider_name": endpoint.get("provider_name") or "",
                "provider_tag": tag,
                "service_tier": service_tier(tag),
                "quantization": endpoint.get("quantization") or "unknown",
                "context_length_tokens": endpoint.get("context_length"),
                "max_prompt_tokens": endpoint.get("max_prompt_tokens"),
                "max_completion_tokens": endpoint.get("max_completion_tokens"),
                "prompt_usd_per_mtoken": per_million(pricing.get("prompt")),
                "completion_usd_per_mtoken": per_million(pricing.get("completion")),
                "cache_read_usd_per_mtoken": per_million(pricing.get("input_cache_read")),
                "cache_write_usd_per_mtoken": per_million(pricing.get("input_cache_write")),
                "high_context_min_prompt_tokens": high.get("min_prompt_tokens"),
                "high_context_prompt_usd_per_mtoken": per_million(high.get("prompt")),
                "high_context_completion_usd_per_mtoken": per_million(high.get("completion")),
                "high_context_cache_read_usd_per_mtoken": per_million(high.get("input_cache_read")),
                "high_context_cache_write_usd_per_mtoken": per_million(high.get("input_cache_write")),
                "status": endpoint.get("status"),
                "uptime_last_30m": endpoint.get("uptime_last_30m"),
                "uptime_last_5m": endpoint.get("uptime_last_5m"),
                "uptime_last_1d": endpoint.get("uptime_last_1d"),
                "source_url": result["url"],
            }
        )
    return output


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.12g}"
    return value


def write_csv(path: Path, fields: list[str], output: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in output:
            writer.writerow({field: csv_value(row.get(field)) for field in fields})


def signature(row: dict[str, Any], *, official: bool) -> tuple[Any, ...]:
    def rounded(key: str) -> float | None:
        value = number(row.get(key))
        return None if value is None else round(value, 9)

    context_key = "context_length_tokens" if official else "endpoint_context_length_tokens"
    return (
        str(row.get("quantization") or "unknown"),
        rounded(context_key),
        rounded("max_prompt_tokens"),
        rounded("max_completion_tokens"),
        rounded("prompt_usd_per_mtoken"),
        rounded("completion_usd_per_mtoken"),
        rounded("cache_read_usd_per_mtoken"),
        rounded("cache_write_usd_per_mtoken"),
        rounded("high_context_min_prompt_tokens"),
        rounded("high_context_prompt_usd_per_mtoken"),
        rounded("high_context_completion_usd_per_mtoken"),
        rounded("high_context_cache_read_usd_per_mtoken"),
        rounded("high_context_cache_write_usd_per_mtoken"),
    )


def frontend_tag(row: dict[str, str]) -> str:
    tag = row["provider_slug"]
    tier = row["service_tier"]
    return tag if tier == "default" else f"{tag}/{tier}"


def compare(
    official_rows: list[dict[str, Any]], frontend_rows: list[dict[str, str]]
) -> list[dict[str, Any]]:
    official_groups: dict[tuple[str, str], Counter[tuple[Any, ...]]] = defaultdict(Counter)
    frontend_groups: dict[tuple[str, str], Counter[tuple[Any, ...]]] = defaultdict(Counter)
    for row in official_rows:
        official_groups[(row["openrouter_model_id"], row["provider_tag"])][
            signature(row, official=True)
        ] += 1
    for row in frontend_rows:
        frontend_groups[(row["openrouter_model_id"], frontend_tag(row))][
            signature(row, official=False)
        ] += 1
    output: list[dict[str, Any]] = []
    for model_id, tag in sorted(set(official_groups) | set(frontend_groups)):
        left = official_groups.get((model_id, tag), Counter())
        right = frontend_groups.get((model_id, tag), Counter())
        if left == right:
            status = "exact"
        elif not left:
            status = "frontend_only"
        elif not right:
            status = "official_only"
        else:
            status = "signature_mismatch"
        output.append(
            {
                "openrouter_model_id": model_id,
                "provider_tag": tag,
                "status": status,
                "official_endpoint_rows": sum(left.values()),
                "frontend_endpoint_tier_rows": sum(right.values()),
                "official_signatures_json": json.dumps(sorted((str(k), v) for k, v in left.items())),
                "frontend_signatures_json": json.dumps(sorted((str(k), v) for k, v in right.items())),
                "official_source_url": endpoint_url(model_id),
                "frontend_source_url": next(
                    (
                        row["endpoint_source_url"]
                        for row in frontend_rows
                        if row["openrouter_model_id"] == model_id and frontend_tag(row) == tag
                    ),
                    "",
                ),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Fetch a new official endpoint snapshot; otherwise rebuild only from the frozen raw response.",
    )
    args = parser.parse_args()
    model_rows = rows(MODELS)
    frontend_rows = rows(TIERS)
    model_ids = sorted(row["openrouter_model_id"] for row in model_rows)
    if args.refresh:
        fetched_at = datetime.now(tz=UTC).replace(microsecond=0).isoformat()
        results: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=12) as pool:
            futures = {
                pool.submit(collect_one, model_id): model_id for model_id in model_ids
            }
            for future in as_completed(futures):
                results.append(future.result())
        results.sort(key=lambda row: row["model_id"])
        raw_payload = {
            "schema_version": "1.0",
            "fetched_at_utc": fetched_at,
            "base_url": BASE_URL,
            "models": results,
        }
        with gzip.open(RAW, "wt", encoding="utf-8") as handle:
            json.dump(
                raw_payload, handle, ensure_ascii=False, separators=(",", ":")
            )
    else:
        if not RAW.exists():
            raise FileNotFoundError(f"Missing frozen official snapshot; run {__file__} --refresh")
        with gzip.open(RAW, "rt", encoding="utf-8") as handle:
            raw_payload = json.load(handle)
        fetched_at = str(raw_payload["fetched_at_utc"])
        results = list(raw_payload["models"])
        frozen_ids = sorted(str(row["model_id"]) for row in results)
        if frozen_ids != model_ids:
            missing = sorted(set(model_ids) - set(frozen_ids))
            extra = sorted(set(frozen_ids) - set(model_ids))
            raise ValueError(
                f"Frozen official snapshot/model panel mismatch; missing={missing}, extra={extra}; refresh required"
            )
    failures = [
        {"openrouter_model_id": row["model_id"], "source_url": row["url"], "error": row["error"]}
        for row in results
        if row["error"]
    ]
    official_rows = [row for result in results for row in flatten(result, fetched_at)]
    official_rows.sort(
        key=lambda row: (
            row["openrouter_model_id"], row["provider_tag"], row["service_tier"]
        )
    )
    comparison = compare(official_rows, frontend_rows)

    write_csv(OFFICIAL, OFFICIAL_FIELDS, official_rows)
    comparison_fields = list(comparison[0]) if comparison else []
    write_csv(COMPARISON, comparison_fields, comparison)

    counts = Counter(row["status"] for row in comparison)
    exact_rows = sum(
        int(row["official_endpoint_rows"])
        for row in comparison
        if row["status"] == "exact"
    )
    official_with_uptime = [
        number(row["uptime_last_30m"])
        for row in official_rows
        if number(row["uptime_last_30m"]) is not None
    ]
    focal = {}
    for model_id in (
        "anthropic/claude-fable-5",
        "openai/gpt-5.6-sol",
        "moonshotai/kimi-k3",
    ):
        focal[model_id] = {
            "official_rows": sum(row["openrouter_model_id"] == model_id for row in official_rows),
            "frontend_rows": sum(row["openrouter_model_id"] == model_id for row in frontend_rows),
            "non_exact_groups": [
                row["provider_tag"]
                for row in comparison
                if row["openrouter_model_id"] == model_id and row["status"] != "exact"
            ],
        }
    audit = {
        "schema_version": "1.0",
        "fetched_at_utc": fetched_at,
        "network_refresh": bool(args.refresh),
        "documented_endpoint_pattern": f"{BASE_URL}/{{model_id}}/endpoints",
        "model_count_requested": len(model_ids),
        "model_count_succeeded": len(model_ids) - len(failures),
        "failure_count": len(failures),
        "failures": failures,
        "official_endpoint_rows": len(official_rows),
        "frontend_endpoint_tier_rows": len(frontend_rows),
        "comparison_group_counts": dict(sorted(counts.items())),
        "official_rows_in_exact_groups": exact_rows,
        "official_price_row_exact_share": (
            exact_rows / len(official_rows) if official_rows else None
        ),
        "official_rows_with_30m_uptime": len(official_with_uptime),
        "official_30m_uptime_median": (
            statistics.median(official_with_uptime) if official_with_uptime else None
        ),
        "focal_models": focal,
        "method_notes": [
            "Official provider tags are matched to frontend provider slugs; flex and priority are kept as separate service-tier suffixes.",
            "A group is exact only when the full multiset of quantization, context limits, base prices, and first high-context prices matches.",
            "Throughput values remain sourced from the public model-page statistics payload because the documented endpoint API currently returns null throughput fields.",
            "This is a near-contemporaneous crosscheck, not a guarantee that two independently fetched live endpoints cannot change between requests.",
        ],
        "files": {
            "raw_official_snapshot": str(RAW.relative_to(ROOT)),
            "official_endpoint_prices": str(OFFICIAL.relative_to(ROOT)),
            "comparison": str(COMPARISON.relative_to(ROOT)),
        },
        "source_hashes": {
            str(MODELS.relative_to(ROOT)): hashlib.sha256(MODELS.read_bytes()).hexdigest(),
            str(TIERS.relative_to(ROOT)): hashlib.sha256(TIERS.read_bytes()).hexdigest(),
            str(RAW.relative_to(ROOT)): hashlib.sha256(RAW.read_bytes()).hexdigest(),
            str(OFFICIAL.relative_to(ROOT)): hashlib.sha256(OFFICIAL.read_bytes()).hexdigest(),
            str(COMPARISON.relative_to(ROOT)): hashlib.sha256(COMPARISON.read_bytes()).hexdigest(),
        },
    }
    AUDIT.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
