#!/usr/bin/env python3
"""Materialize an immutable, lossless OpenRouter refresh history.

The canonical OpenRouter files intentionally keep stable names so downstream
code can consume a frozen snapshot.  This script prevents a refresh from
silently erasing the prior observation: it recovers every committed snapshot,
adds the current worktree snapshot, archives the exact raw/model/provider/audit
bytes under a timestamped directory, and rebuilds long history tables.

The daily history is derived from each preserved raw response and retains
default, priority, and flex service tiers separately.  Snapshot rows are
correlated repeated measurements and are never treated as independent model
labels by the forecasting regression.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import subprocess
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from collect_openrouter_signals import ModelTask, TIER_FIELDS, _endpoint_tier_rows


ROOT = Path(__file__).resolve().parent
COMPATIBILITY_FILE_DATE = "2026-07-18"
# Backwards-compatible alias used by the frozen output filenames below.
DATE = COMPATIBILITY_FILE_DATE
OUT = ROOT / "outputs/019f6c42-2d53-7743-ab07-6293e2618dd7"
RAW_REL = f"sources/openrouter_operational_snapshot_{DATE}.json.gz"
MODELS_REL = f"sources/openrouter_model_signals_{DATE}.csv"
PROVIDERS_REL = f"sources/openrouter_provider_signals_{DATE}.csv"
AUDIT_REL = (
    "outputs/019f6c42-2d53-7743-ab07-6293e2618dd7/"
    f"openrouter_collection_audit_{DATE}.json"
)

HISTORY_DIR = ROOT / "sources/openrouter_history"
MODEL_HISTORY = ROOT / f"sources/openrouter_model_snapshot_history_{DATE}.csv"
PROVIDER_HISTORY = ROOT / f"sources/openrouter_provider_snapshot_history_{DATE}.csv"
TIER_HISTORY = ROOT / f"sources/openrouter_endpoint_tier_snapshot_history_{DATE}.csv"
DAILY_HISTORY = ROOT / f"sources/openrouter_throughput_daily_history_{DATE}.csv"
MANIFEST = ROOT / f"sources/openrouter_snapshot_history_manifest_{DATE}.csv"

HISTORY_PREFIX = ["history_snapshot_id", "history_source_revision"]
DAILY_FIELDS = [
    "history_snapshot_id",
    "history_source_revision",
    "snapshot_date",
    "fetched_at_utc",
    "observation_time_raw",
    "observation_date",
    "openrouter_model_id",
    "openrouter_model_name",
    "canonical_slug",
    "author",
    "created_date",
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


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def git_bytes(revision: str, relative_path: str) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{revision}:{relative_path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return result.stdout


def git_revisions() -> list[str]:
    result = subprocess.run(
        ["git", "log", "--reverse", "--format=%H", "--", RAW_REL],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def parse_csv(payload: bytes) -> tuple[list[str], list[dict[str, str]]]:
    text = payload.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    return list(reader.fieldnames or []), list(reader)


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.12g}"
    return value


def write_csv(path: Path, fields: Iterable[str], rows: list[dict[str, Any]]) -> None:
    field_list = list(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=field_list, extrasaction="raise", lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field)) for field in field_list})


def number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def snapshot_id(fetched_at: str) -> str:
    return (
        fetched_at.replace("-", "")
        .replace(":", "")
        .replace("+00:00", "Z")
        .replace(".", "")
    )


def derive_daily_rows(
    raw_payload: bytes,
    provider_rows: list[dict[str, str]],
    history_snapshot_id: str,
    source_revision: str,
) -> list[dict[str, Any]]:
    with gzip.GzipFile(fileobj=io.BytesIO(raw_payload), mode="rb") as handle:
        raw = json.loads(handle.read().decode("utf-8"))
    providers: dict[tuple[str, str], dict[str, str]] = {}
    first_provider: dict[str, dict[str, str]] = {}
    for row in provider_rows:
        model_id = row["openrouter_model_id"]
        first_provider.setdefault(model_id, row)
        if row.get("endpoint_id"):
            providers[(model_id, row["endpoint_id"])] = row
    catalog = {
        str(row.get("id") or ""): row
        for row in raw.get("catalog_payload", {}).get("data", [])
    }
    rows: list[dict[str, Any]] = []
    for model in raw.get("models", []):
        model_id = str(model.get("openrouter_model_id") or "")
        fallback = first_provider.get(model_id, {})
        catalog_row = catalog.get(model_id, {})
        model_name = str(
            catalog_row.get("name") or fallback.get("openrouter_model_name") or model_id
        )
        canonical_slug = str(
            catalog_row.get("canonical_slug")
            or fallback.get("canonical_slug")
            or model.get("canonical_slug")
            or ""
        )
        author = model_id.split("/", 1)[0] if "/" in model_id else model_id
        created = catalog_row.get("created")
        created_date = ""
        if isinstance(created, (int, float)):
            created_date = datetime.fromtimestamp(created, tz=UTC).date().isoformat()
        throughput_url = str(model.get("throughput_url") or "")
        for point in model.get("throughput_payload", {}).get("data", []) or []:
            observed_raw = str(point.get("x") or "")
            observed_date = observed_raw[:10] if len(observed_raw) >= 10 else ""
            for endpoint_tier, raw_value in (point.get("y") or {}).items():
                value = number(raw_value)
                if value is None:
                    continue
                parts = str(endpoint_tier).split("::", 1)
                endpoint_id = parts[0]
                tier = parts[1] if len(parts) == 2 else "default"
                provider = providers.get((model_id, endpoint_id), {})
                rows.append(
                    {
                        "history_snapshot_id": history_snapshot_id,
                        "history_source_revision": source_revision,
                        "snapshot_date": raw.get("snapshot_date", DATE),
                        "fetched_at_utc": raw.get("fetched_at_utc", ""),
                        "observation_time_raw": observed_raw,
                        "observation_date": observed_date,
                        "openrouter_model_id": model_id,
                        "openrouter_model_name": model_name,
                        "canonical_slug": canonical_slug,
                        "author": author,
                        "created_date": created_date,
                        "endpoint_tier_key": str(endpoint_tier),
                        "endpoint_id": endpoint_id,
                        "service_tier": tier,
                        "provider_name": provider.get("provider_name", ""),
                        "provider_display_name": provider.get(
                            "provider_display_name", ""
                        ),
                        "provider_slug": provider.get("provider_slug", ""),
                        "provider_region": provider.get("provider_region", ""),
                        "variant": provider.get("variant", ""),
                        "quantization": provider.get("quantization", ""),
                        "throughput_tps": value,
                        "endpoint_metadata_match": bool(provider),
                        "throughput_source_url": throughput_url,
                    }
                )
    rows.sort(
        key=lambda row: (
            row["history_snapshot_id"],
            row["openrouter_model_id"],
            row["observation_time_raw"],
            row["endpoint_tier_key"],
        )
    )
    return rows


def derive_tier_rows(
    raw_payload: bytes,
    history_snapshot_id: str,
    source_revision: str,
) -> list[dict[str, Any]]:
    """Reconstruct every endpoint/service-tier price and 30-minute stats row."""
    with gzip.GzipFile(fileobj=io.BytesIO(raw_payload), mode="rb") as handle:
        raw = json.loads(handle.read().decode("utf-8"))
    catalog = {
        str(row.get("id") or ""): row
        for row in raw.get("catalog_payload", {}).get("data", [])
    }
    rows: list[dict[str, Any]] = []
    for model in raw.get("models", []):
        model_id = str(model.get("openrouter_model_id") or "")
        item = catalog.get(model_id, {})
        task = ModelTask(
            model_id=model_id,
            model_name=str(item.get("name") or model_id),
            canonical_slug=str(
                item.get("canonical_slug") or model.get("canonical_slug") or ""
            ),
            permaslug=str(
                item.get("canonical_slug") or model.get("canonical_slug") or ""
            ),
            author=model_id.split("/", 1)[0] if "/" in model_id else model_id,
            hugging_face_id=str(item.get("hugging_face_id") or ""),
            created=(
                int(item["created"]) if item.get("created") is not None else None
            ),
            context_length=(
                int(item["context_length"])
                if item.get("context_length") is not None
                else None
            ),
        )
        tier_rows = _endpoint_tier_rows(
            task,
            str(raw.get("fetched_at_utc") or ""),
            str(raw.get("snapshot_date") or DATE),
            str(model.get("endpoint_url") or ""),
            model.get("endpoint_payload") or {"data": []},
        )
        rows.extend(
            {
                "history_snapshot_id": history_snapshot_id,
                "history_source_revision": source_revision,
                **row,
            }
            for row in tier_rows
        )
    rows.sort(
        key=lambda row: (
            row["history_snapshot_id"],
            row["openrouter_model_id"],
            row["endpoint_id"],
            row["service_tier"],
        )
    )
    return rows


def union_fields(field_lists: list[list[str]]) -> list[str]:
    output: list[str] = []
    for fields in field_lists:
        for field in fields:
            if field not in output:
                output.append(field)
    return output


def load_snapshot(
    revision: str,
    raw_bytes: bytes,
    model_bytes: bytes,
    provider_bytes: bytes,
    audit_bytes: bytes,
) -> dict[str, Any]:
    audit = json.loads(audit_bytes.decode("utf-8"))
    fetched_at = str(audit["fetched_at_utc"])
    observation_date = str(audit["snapshot_date"])
    compatibility_date = str(
        audit.get("compatibility_filename_date") or COMPATIBILITY_FILE_DATE
    )
    if compatibility_date != COMPATIBILITY_FILE_DATE:
        raise ValueError(
            f"Unexpected OpenRouter compatibility filename date {compatibility_date}"
        )
    with gzip.GzipFile(fileobj=io.BytesIO(raw_bytes), mode="rb") as handle:
        raw = json.loads(handle.read().decode("utf-8"))
    if str(raw.get("snapshot_date") or "") != observation_date:
        raise ValueError("OpenRouter raw/audit snapshot-date mismatch")
    if str(raw.get("fetched_at_utc") or "") != fetched_at:
        raise ValueError("OpenRouter raw/audit fetched-at mismatch")
    raw_compatibility_date = str(
        raw.get("compatibility_filename_date") or COMPATIBILITY_FILE_DATE
    )
    if raw_compatibility_date != compatibility_date:
        raise ValueError("OpenRouter raw/audit compatibility-date mismatch")
    identifier = snapshot_id(fetched_at)
    model_fields, model_rows = parse_csv(model_bytes)
    provider_fields, provider_rows = parse_csv(provider_bytes)
    for table_name, rows in (("model", model_rows), ("provider", provider_rows)):
        if any(row.get("snapshot_date") != observation_date for row in rows):
            raise ValueError(f"OpenRouter {table_name}/audit snapshot-date mismatch")
        if any(row.get("fetched_at_utc") != fetched_at for row in rows):
            raise ValueError(f"OpenRouter {table_name}/audit fetched-at mismatch")
    daily_rows = derive_daily_rows(raw_bytes, provider_rows, identifier, revision)
    tier_rows = derive_tier_rows(raw_bytes, identifier, revision)
    return {
        "snapshot_id": identifier,
        "revision": revision,
        "fetched_at": fetched_at,
        "observation_date": observation_date,
        "compatibility_filename_date": compatibility_date,
        "audit": audit,
        "raw_bytes": raw_bytes,
        "model_bytes": model_bytes,
        "provider_bytes": provider_bytes,
        "audit_bytes": audit_bytes,
        "model_fields": model_fields,
        "model_rows": model_rows,
        "provider_fields": provider_fields,
        "provider_rows": provider_rows,
        "tier_rows": tier_rows,
        "daily_rows": daily_rows,
    }


def all_snapshots() -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    prior_revisions: dict[str, str] = {}
    if MANIFEST.exists():
        _, prior_manifest = parse_csv(MANIFEST.read_bytes())
        prior_revisions = {
            row["history_snapshot_id"]: row.get("source_revision", "ARCHIVE")
            for row in prior_manifest
        }

    if HISTORY_DIR.exists():
        for archive in sorted(path for path in HISTORY_DIR.iterdir() if path.is_dir()):
            paths = (
                archive / Path(RAW_REL).name,
                archive / Path(MODELS_REL).name,
                archive / Path(PROVIDERS_REL).name,
                archive / Path(AUDIT_REL).name,
            )
            if not all(path.exists() for path in paths):
                continue
            snapshot = load_snapshot(
                prior_revisions.get(archive.name, "ARCHIVE"),
                paths[0].read_bytes(),
                paths[1].read_bytes(),
                paths[2].read_bytes(),
                paths[3].read_bytes(),
            )
            by_id[snapshot["snapshot_id"]] = snapshot

    for revision in git_revisions():
        try:
            snapshot = load_snapshot(
                revision[:12],
                git_bytes(revision, RAW_REL),
                git_bytes(revision, MODELS_REL),
                git_bytes(revision, PROVIDERS_REL),
                git_bytes(revision, AUDIT_REL),
            )
        except subprocess.CalledProcessError:
            continue
        prior = by_id.get(snapshot["snapshot_id"])
        if prior is not None:
            for key in ("raw_bytes", "model_bytes", "provider_bytes", "audit_bytes"):
                if sha256_bytes(prior[key]) != sha256_bytes(snapshot[key]):
                    raise ValueError(
                        f"Snapshot {snapshot['snapshot_id']} changed without a fetched-at change"
                    )
            # A fetched snapshot is a content-addressed observation. Keep the
            # provenance label recorded when it first entered the immutable
            # archive instead of making every later git commit rewrite all
            # history rows.
            continue
        by_id[snapshot["snapshot_id"]] = snapshot

    current_paths = [ROOT / RAW_REL, ROOT / MODELS_REL, ROOT / PROVIDERS_REL, ROOT / AUDIT_REL]
    if all(path.exists() for path in current_paths):
        current = load_snapshot(
            "WORKTREE",
            current_paths[0].read_bytes(),
            current_paths[1].read_bytes(),
            current_paths[2].read_bytes(),
            current_paths[3].read_bytes(),
        )
        prior = by_id.get(current["snapshot_id"])
        if prior is not None:
            for key in ("raw_bytes", "model_bytes", "provider_bytes", "audit_bytes"):
                if sha256_bytes(prior[key]) != sha256_bytes(current[key]):
                    raise ValueError(
                        f"Snapshot {current['snapshot_id']} changed without a fetched-at change"
                    )
            current["revision"] = prior["revision"]
            for row in current["daily_rows"] + current["tier_rows"]:
                row["history_source_revision"] = current["revision"]
        by_id[current["snapshot_id"]] = current
    return sorted(by_id.values(), key=lambda row: row["fetched_at"])


def main() -> None:
    snapshots = all_snapshots()
    if not snapshots:
        raise ValueError("No OpenRouter snapshots found")

    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    model_fields = HISTORY_PREFIX + union_fields(
        [snapshot["model_fields"] for snapshot in snapshots]
    )
    provider_fields = HISTORY_PREFIX + union_fields(
        [snapshot["provider_fields"] for snapshot in snapshots]
    )
    model_history: list[dict[str, Any]] = []
    provider_history: list[dict[str, Any]] = []
    tier_history: list[dict[str, Any]] = []
    daily_history: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []

    for snapshot in snapshots:
        archive = HISTORY_DIR / snapshot["snapshot_id"]
        archive.mkdir(parents=True, exist_ok=True)
        archived_raw = archive / Path(RAW_REL).name
        archived_models = archive / Path(MODELS_REL).name
        archived_providers = archive / Path(PROVIDERS_REL).name
        archived_audit = archive / Path(AUDIT_REL).name
        archived_raw.write_bytes(snapshot["raw_bytes"])
        archived_models.write_bytes(snapshot["model_bytes"])
        archived_providers.write_bytes(snapshot["provider_bytes"])
        archived_audit.write_bytes(snapshot["audit_bytes"])

        model_history.extend(
            {
                "history_snapshot_id": snapshot["snapshot_id"],
                "history_source_revision": snapshot["revision"],
                **row,
            }
            for row in snapshot["model_rows"]
        )
        provider_history.extend(
            {
                "history_snapshot_id": snapshot["snapshot_id"],
                "history_source_revision": snapshot["revision"],
                **row,
            }
            for row in snapshot["provider_rows"]
        )
        tier_history.extend(snapshot["tier_rows"])
        daily_history.extend(snapshot["daily_rows"])
        audit = snapshot["audit"]
        manifest.append(
            {
                "history_snapshot_id": snapshot["snapshot_id"],
                "fetched_at_utc": snapshot["fetched_at"],
                "snapshot_date": audit.get("snapshot_date", DATE),
                "compatibility_filename_date": snapshot[
                    "compatibility_filename_date"
                ],
                "source_revision": snapshot["revision"],
                "catalog_model_count": audit.get("catalog_model_count", ""),
                "eligible_text_model_count": audit.get("eligible_text_model_count", ""),
                "provider_endpoint_row_count": len(snapshot["provider_rows"]),
                "endpoint_tier_row_count": len(snapshot["tier_rows"]),
                "daily_throughput_row_count": len(snapshot["daily_rows"]),
                "daily_throughput_model_count": len(
                    {row["openrouter_model_id"] for row in snapshot["daily_rows"]}
                ),
                "daily_throughput_endpoint_tier_count": len(
                    {
                        (row["openrouter_model_id"], row["endpoint_tier_key"])
                        for row in snapshot["daily_rows"]
                    }
                ),
                "failure_count": audit.get("failure_count", ""),
                "raw_sha256": sha256_bytes(snapshot["raw_bytes"]),
                "model_sha256": sha256_bytes(snapshot["model_bytes"]),
                "provider_sha256": sha256_bytes(snapshot["provider_bytes"]),
                "audit_sha256": sha256_bytes(snapshot["audit_bytes"]),
                "archive_directory": str(archive.relative_to(ROOT)),
            }
        )

    model_history.sort(
        key=lambda row: (row["history_snapshot_id"], row["openrouter_model_id"])
    )
    provider_history.sort(
        key=lambda row: (
            row["history_snapshot_id"],
            row["openrouter_model_id"],
            row.get("endpoint_id", ""),
            row.get("variant", ""),
        )
    )
    tier_history.sort(
        key=lambda row: (
            row["history_snapshot_id"],
            row["openrouter_model_id"],
            row["endpoint_id"],
            row["service_tier"],
        )
    )
    daily_history.sort(
        key=lambda row: (
            row["history_snapshot_id"],
            row["openrouter_model_id"],
            row["observation_time_raw"],
            row["endpoint_tier_key"],
        )
    )
    write_csv(MODEL_HISTORY, model_fields, model_history)
    write_csv(PROVIDER_HISTORY, provider_fields, provider_history)
    write_csv(TIER_HISTORY, HISTORY_PREFIX + TIER_FIELDS, tier_history)
    write_csv(DAILY_HISTORY, DAILY_FIELDS, daily_history)
    write_csv(MANIFEST, list(manifest[0]), manifest)

    unique_daily_keys = {
        (
            row["history_snapshot_id"],
            row["openrouter_model_id"],
            row["observation_time_raw"],
            row["endpoint_tier_key"],
        )
        for row in daily_history
    }
    if len(unique_daily_keys) != len(daily_history):
        raise ValueError("Duplicate OpenRouter daily history key")
    print(
        json.dumps(
            {
                "snapshots": len(snapshots),
                "model_rows": len(model_history),
                "provider_rows": len(provider_history),
                "tier_rows": len(tier_history),
                "daily_rows": len(daily_history),
                "unique_daily_keys": True,
                "files": {
                    "model_history": str(MODEL_HISTORY.relative_to(ROOT)),
                    "provider_history": str(PROVIDER_HISTORY.relative_to(ROOT)),
                    "tier_history": str(TIER_HISTORY.relative_to(ROOT)),
                    "daily_history": str(DAILY_HISTORY.relative_to(ROOT)),
                    "manifest": str(MANIFEST.relative_to(ROOT)),
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
