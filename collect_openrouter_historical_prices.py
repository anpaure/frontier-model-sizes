#!/usr/bin/env python3
"""Freeze and normalize OpenRouter model-price history.

The upstream repository commits the raw public ``/api/v1/models`` response
twice daily.  Its compact ledger is rebuilt from those immutable git versions
and records every prompt/completion price change while retaining first/last
availability dates.  We pin one upstream commit and SHA-256 so a refresh can
never silently move the historical evidence.

Frozen mode (the pipeline default) never uses the network.  ``--refresh``
downloads only the pinned ledger, verifies its exact uncompressed hash, and
writes a deterministic gzip copy plus a lossless change-point CSV.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import urllib.request
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
# The stable filename is an API contract for downstream scripts.  The pinned
# source and all provenance fields below are current through OBSERVATION_DATE.
COMPATIBILITY_FILE_DATE = "2026-07-18"
OBSERVATION_DATE = "2026-07-31"
DATE = COMPATIBILITY_FILE_DATE
PINNED_COMMIT = "1cd0e2ec5fccb271df9d1140abc91aaf20b3e878"
UPSTREAM_GIT_BLOB = "2e00573a582c7c669f6d7e8d98dc8555397cf4ea"
EXPECTED_RAW_SHA256 = "d572cdb4b64b0b4ed878e570490b937666a09b49f5cef1199dc8869825473acc"
RAW_URL = (
    "https://raw.githubusercontent.com/jvrck/openrouterlist/"
    f"{PINNED_COMMIT}/data/history/prices.json"
)
REPOSITORY_URL = "https://github.com/jvrck/openrouterlist"
OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/models"

RAW = ROOT / f"sources/openrouter_historical_price_ledger_{DATE}.json.gz"
SIGNALS = ROOT / f"sources/openrouter_historical_price_change_points_{DATE}.csv"
METADATA = ROOT / f"sources/openrouter_historical_price_collection_metadata_{DATE}.json"

EXPECTED_MODELS = 914
EXPECTED_POINTS = 2566
EXPECTED_AS_OF = OBSERVATION_DATE
UPSTREAM_OUTPUT_SNAPSHOTS = 1062
UPSTREAM_FIRST_SNAPSHOT = "2024-09-21"
FULL_HISTORY_REBUILD_VERIFIED_ON = OBSERVATION_DATE


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def deterministic_gzip(payload: bytes) -> bytes:
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0) as handle:
        handle.write(payload)
    return buffer.getvalue()


def fetch_pinned() -> bytes:
    request = urllib.request.Request(
        RAW_URL,
        headers={"User-Agent": "frontier-parameter-model historical-price audit"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()
    actual = sha256_bytes(payload)
    if actual != EXPECTED_RAW_SHA256:
        raise ValueError(
            f"Pinned upstream payload hash changed: expected {EXPECTED_RAW_SHA256}, got {actual}"
        )
    return payload


def load_frozen() -> bytes:
    if not RAW.exists():
        raise FileNotFoundError(f"Missing frozen source {RAW}; run with --refresh once")
    with gzip.open(RAW, "rb") as handle:
        payload = handle.read()
    actual = sha256_bytes(payload)
    if actual != EXPECTED_RAW_SHA256:
        raise ValueError(
            f"Frozen upstream payload hash mismatch: expected {EXPECTED_RAW_SHA256}, got {actual}"
        )
    return payload


def finite_nonnegative(value: Any) -> bool:
    if value is None:
        return True
    return isinstance(value, (int, float)) and math.isfinite(float(value)) and value >= 0


def validate(payload: dict[str, Any]) -> dict[str, Any]:
    models = payload.get("models")
    if not isinstance(models, dict):
        raise ValueError("Historical ledger has no model dictionary")
    if payload.get("model_count") != len(models) or len(models) != EXPECTED_MODELS:
        raise ValueError(
            f"Historical model inventory mismatch: header={payload.get('model_count')}, "
            f"parsed={len(models)}, expected={EXPECTED_MODELS}"
        )
    if payload.get("as_of") != EXPECTED_AS_OF:
        raise ValueError(f"Unexpected historical ledger as_of={payload.get('as_of')}")

    point_count = 0
    same_day_points = 0
    statuses: Counter[str] = Counter()
    earliest = "9999-99-99"
    latest = "0000-00-00"
    for model_id, record in models.items():
        if not isinstance(model_id, str) or "/" not in model_id:
            raise ValueError(f"Invalid OpenRouter model ID: {model_id!r}")
        points = record.get("points")
        if not isinstance(points, list) or not points:
            raise ValueError(f"Missing price points for {model_id}")
        if record.get("first_seen") != points[0][0]:
            raise ValueError(f"first_seen does not equal first change point for {model_id}")
        if date.fromisoformat(record["first_seen"]) > date.fromisoformat(record["last_seen"]):
            raise ValueError(f"Invalid availability interval for {model_id}")
        previous = ""
        for index, point in enumerate(points):
            if not isinstance(point, list) or len(point) != 3:
                raise ValueError(f"Malformed point {index} for {model_id}")
            effective, prompt, completion = point
            date.fromisoformat(effective)
            if effective < previous:
                raise ValueError(f"Non-monotonic points for {model_id}")
            same_day_points += int(effective == previous)
            previous = effective
            if not finite_nonnegative(prompt) or not finite_nonnegative(completion):
                raise ValueError(f"Invalid price at {model_id} point {index}")
            if prompt is None or completion is None:
                statuses["unavailable"] += 1
            elif prompt == 0 or completion == 0:
                statuses["free_or_partly_free"] += 1
            else:
                statuses["priced"] += 1
            point_count += 1
        earliest = min(earliest, record["first_seen"])
        latest = max(latest, record["last_seen"])
    if point_count != EXPECTED_POINTS:
        raise ValueError(f"Expected {EXPECTED_POINTS} change points, found {point_count}")
    if earliest != UPSTREAM_FIRST_SNAPSHOT or latest != EXPECTED_AS_OF:
        raise ValueError(f"Unexpected ledger range {earliest} through {latest}")
    return {
        "models": len(models),
        "price_change_points": point_count,
        "same_day_additional_change_points": same_day_points,
        "price_status_counts": dict(sorted(statuses.items())),
        "first_snapshot_date": earliest,
        "as_of": latest,
        "present_now_models": sum(bool(record.get("present_now")) for record in models.values()),
    }


def normalize(payload: dict[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for model_id, record in payload["models"].items():
        for index, (effective, prompt, completion) in enumerate(record["points"]):
            if prompt is None or completion is None:
                blended = ""
                status = "unavailable"
            elif prompt == 0 or completion == 0:
                blended = 0.0
                status = "free_or_partly_free"
            else:
                blended = math.sqrt(float(prompt) * float(completion))
                status = "priced"
            output.append(
                {
                    "openrouter_model_id": model_id,
                    "openrouter_model_name": record["name"],
                    "first_seen": record["first_seen"],
                    "last_seen": record["last_seen"],
                    "present_now": str(bool(record["present_now"])).lower(),
                    "change_index": index,
                    "effective_date": effective,
                    "prompt_price_usd_per_mtoken": "" if prompt is None else prompt,
                    "completion_price_usd_per_mtoken": "" if completion is None else completion,
                    "blended_geomean_price_usd_per_mtoken": blended,
                    "price_status": status,
                    "source_repository": REPOSITORY_URL,
                    "source_commit": PINNED_COMMIT,
                    "upstream_endpoint": OPENROUTER_ENDPOINT,
                }
            )
    return output


def write_csv(rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    with SIGNALS.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Fetch the hash-pinned upstream ledger; default is frozen/offline mode.",
    )
    args = parser.parse_args()
    raw_payload = fetch_pinned() if args.refresh else load_frozen()
    parsed = json.loads(raw_payload)
    inventory = validate(parsed)

    RAW.parent.mkdir(parents=True, exist_ok=True)
    if args.refresh:
        RAW.write_bytes(deterministic_gzip(raw_payload))
    rows = normalize(parsed)
    write_csv(rows)
    metadata = {
        "generated_on": OBSERVATION_DATE,
        "compatibility_filename_date": COMPATIBILITY_FILE_DATE,
        "collection_mode": "hash-pinned refresh" if args.refresh else "frozen offline rebuild",
        "source": {
            "repository": REPOSITORY_URL,
            "license": "MIT",
            "pinned_commit": PINNED_COMMIT,
            "upstream_git_blob": UPSTREAM_GIT_BLOB,
            "raw_ledger_url": RAW_URL,
            "raw_ledger_sha256": EXPECTED_RAW_SHA256,
            "raw_endpoint_used_by_upstream": OPENROUTER_ENDPOINT,
            "upstream_output_snapshot_commits": UPSTREAM_OUTPUT_SNAPSHOTS,
            "upstream_collection_schedule": "official public endpoint fetched at approximately 00:00 and 12:00 UTC",
            "full_git_history_rebuild_verified_on": FULL_HISTORY_REBUILD_VERIFIED_ON,
            "full_git_history_rebuild_snapshot_count": UPSTREAM_OUTPUT_SNAPSHOTS,
            "full_git_history_rebuild_sha256": EXPECTED_RAW_SHA256,
        },
        "inventory": inventory,
        "integrity_policy": {
            "network_in_frozen_pipeline": False,
            "raw_payload_preserved_losslessly": True,
            "all_models_preserved": True,
            "all_change_points_preserved": True,
            "same_day_changes_preserved_in_order": True,
            "full_upstream_git_history_rebuild_matches_frozen_ledger": True,
            "negative_prices_allowed": False,
            "free_and_unavailable_prices_retained_but_not_log_modeled": True,
        },
        "files": {
            str(RAW.relative_to(ROOT)): {
                "sha256": sha256(RAW),
                "uncompressed_sha256": EXPECTED_RAW_SHA256,
            },
            str(SIGNALS.relative_to(ROOT)): {"sha256": sha256(SIGNALS)},
        },
    }
    METADATA.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"raw": str(RAW), "signals": str(SIGNALS), **inventory}, indent=2))


if __name__ == "__main__":
    main()
