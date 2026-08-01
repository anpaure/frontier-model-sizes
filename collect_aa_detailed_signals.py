#!/usr/bin/env python3
"""Collect the machine-readable Artificial Analysis model snapshot.

Artificial Analysis embeds the model table used by its public model pages in
Next.js React Flight payloads.  A single page currently contains the complete
model inventory, including benchmark components, open-weight parameter counts,
prices, and Intelligence Index token-use fields.  This collector preserves the
raw HTML, extracts every complete model object, and writes a flat audit table
without discarding the original record JSON.

The generated snapshot is intentionally not refreshed during ordinary forecast
builds.  Network refreshes are explicit; downstream analysis consumes the
frozen, hashed artifacts.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import re
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
SNAPSHOT_DATE = "2026-07-31"
PRIOR_SNAPSHOT_DATE = "2026-07-18"
SOURCE_URL = "https://artificialanalysis.ai/models/claude-opus-5"
RAW = ROOT / f"sources/aa_detailed_snapshot_{SNAPSHOT_DATE}.html.gz"
MODELS = ROOT / f"sources/aa_detailed_model_signals_{SNAPSHOT_DATE}.csv"
METADATA = ROOT / f"sources/aa_detailed_collection_metadata_{SNAPSHOT_DATE}.json"
MANIFEST = ROOT / f"sources/aa_detailed_snapshot_manifest_{SNAPSHOT_DATE}.json"
DELTA = ROOT / (
    f"sources/aa_detailed_snapshot_delta_{PRIOR_SNAPSHOT_DATE}_to_{SNAPSHOT_DATE}.json"
)
PRIOR_RAW = ROOT / f"sources/aa_detailed_snapshot_{PRIOR_SNAPSHOT_DATE}.html.gz"


# These are the normalized fields that can affect identity matching, benchmark
# regressors, or architecture interpretation downstream.  The delta audit
# records every change in these fields rather than treating a newer snapshot as
# an opaque replacement.
IDENTITY_FIELDS = (
    "model_id",
    "slug",
    "name",
    "short_name",
    "creator_id",
    "creator_slug",
    "creator_name",
    "release_date",
)
SCORE_FIELDS = (
    "intelligence_index",
    "intelligence_index_estimated",
    "coding_index",
    "agentic_index",
    "gdpval",
    "gdpval_normalized",
    "tau_banking",
    "terminal_bench_v2_1",
    "scicode",
    "hle",
    "gpqa",
    "critpt",
    "omniscience",
    "omniscience_accuracy",
    "omniscience_hallucination_rate",
    "lcr",
    "ifbench",
    "apex_agents",
    "automation_bench_partial_score",
    "enterprise_ops_gym",
    "mmmu_pro",
)
ARCHITECTURE_FIELDS = (
    "is_reasoning",
    "reasoning_tokens_setting",
    "is_open_weights",
    "open_source_categorization",
    "parameters_b",
    "active_parameters_b",
    "size_class",
    "context_window_tokens",
    "model_weights_source_url",
    "license_name",
    "license_url",
    "commercial_allowed",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch() -> tuple[bytes, str]:
    request = urllib.request.Request(
        SOURCE_URL,
        headers={
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Encoding": "gzip",
            "User-Agent": "Mozilla/5.0 (compatible; FrontierParameterAudit/1.0)",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        raw = response.read()
        if response.headers.get("Content-Encoding", "").lower() == "gzip":
            raw = gzip.decompress(raw)
        content_type = response.headers.get("Content-Type", "")
    if b"self.__next_f.push" not in raw:
        raise ValueError("Artificial Analysis response lacks React Flight data")
    return raw, content_type


def flight_strings(html: str) -> list[str]:
    strings: list[str] = []
    for payload in re.findall(
        r"<script>self\.__next_f\.push\((.*?)\)</script>", html, re.DOTALL
    ):
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if (
            isinstance(decoded, list)
            and len(decoded) > 1
            and isinstance(decoded[1], str)
        ):
            strings.append(decoded[1])
    if not strings:
        raise ValueError("No decodable React Flight strings found")
    return strings


def extract_models(html: str) -> dict[str, dict[str, Any]]:
    decoder = json.JSONDecoder()
    models: dict[str, dict[str, Any]] = {}
    for text in flight_strings(html):
        position = 0
        while True:
            start = text.find('{"id":', position)
            if start < 0:
                break
            try:
                candidate, consumed = decoder.raw_decode(text[start:])
            except json.JSONDecodeError:
                position = start + 1
                continue
            position = start + max(consumed, 1)
            if not (
                isinstance(candidate, dict)
                and candidate.get("id")
                and candidate.get("slug")
                and candidate.get("name")
                and "intelligenceIndex" in candidate
            ):
                continue
            slug = str(candidate["slug"])
            # The same model can appear in several page components.  Retain the
            # most complete object and require exact agreement on core fields.
            existing = models.get(slug)
            if existing:
                for field in ("id", "name", "releaseDate", "intelligenceIndex"):
                    if (
                        existing.get(field) is not None
                        and candidate.get(field) is not None
                        and existing.get(field) != candidate.get(field)
                    ):
                        raise ValueError(
                            f"Conflicting duplicate {slug!r} field {field!r}"
                        )
            if existing is None or len(candidate) > len(existing):
                models[slug] = candidate
    if len(models) < 500:
        raise ValueError(f"Unexpectedly small AA inventory: {len(models)}")
    if len({row["id"] for row in models.values()}) != len(models):
        raise ValueError("AA inventory contains duplicate model IDs")
    return models


def nested(record: dict[str, Any], *path: str) -> Any:
    value: Any = record
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def flatten(record: dict[str, Any], snapshot_sha: str) -> dict[str, Any]:
    creator = record.get("creator") or {}
    performance = record.get("performanceDataSource") or {}
    return {
        "model_id": record.get("id"),
        "slug": record.get("slug"),
        "name": record.get("name"),
        "short_name": record.get("shortName"),
        "creator_id": creator.get("id"),
        "creator_slug": creator.get("slug"),
        "creator_name": creator.get("name"),
        "creator_url": creator.get("url"),
        "release_date": record.get("releaseDate"),
        "knowledge_cutoff_date": record.get("knowledgeCutoffDate"),
        "is_reasoning": record.get("isReasoning"),
        "reasoning_tokens_setting": record.get("reasoningTokens"),
        "is_open_weights": record.get("isOpenWeights"),
        "open_source_categorization": record.get("openSourceCategorization"),
        "parameters_b": record.get("parameters"),
        "active_parameters_b": record.get("inferenceParametersActiveBillions"),
        "size_class": record.get("sizeClass"),
        "context_window_tokens": record.get("contextWindowTokens"),
        "intelligence_index": record.get("intelligenceIndex"),
        "intelligence_index_estimated": record.get("intelligenceIndexIsEstimated"),
        "coding_index": record.get("codingIndex"),
        "agentic_index": record.get("agenticIndex"),
        "gdpval": record.get("gdpval"),
        "gdpval_normalized": record.get("gdpvalNormalized"),
        "tau_banking": record.get("tauBanking"),
        "terminal_bench_v2_1": record.get("terminalbenchV21"),
        "scicode": record.get("scicode"),
        "hle": record.get("hle"),
        "gpqa": record.get("gpqa"),
        "critpt": record.get("critpt"),
        "omniscience": record.get("omniscience"),
        "omniscience_accuracy": nested(record, "omniscienceBreakdown", "accuracy"),
        "omniscience_hallucination_rate": nested(
            record, "omniscienceBreakdown", "hallucinationRate"
        ),
        "lcr": record.get("lcr"),
        "ifbench": record.get("ifbench"),
        "apex_agents": record.get("apexAgents"),
        "automation_bench_partial_score": record.get("automationBenchPartialScore"),
        "enterprise_ops_gym": record.get("enterpriseOpsGym"),
        "mmmu_pro": record.get("mmmuPro"),
        "price_input_usd_per_mtoken": record.get("price1mInputTokens"),
        "price_output_usd_per_mtoken": record.get("price1mOutputTokens"),
        "price_cache_hit_usd_per_mtoken": record.get("cacheHitPrice"),
        "price_blended_7_2_1_usd_per_mtoken": record.get("price1mBlended7To2To1"),
        "intelligence_time_per_task_seconds": record.get(
            "intelligenceIndexTimePerTask"
        ),
        "intelligence_cost_total_usd": nested(
            record, "intelligenceIndexCost", "total"
        ),
        "intelligence_cost_per_task_usd": nested(
            record, "intelligenceIndexCostPerTask", "cost", "total"
        ),
        "intelligence_input_tokens_total": nested(
            record, "canonicalIntelligenceIndexTokenCount", "input"
        ),
        "intelligence_output_tokens_total": nested(
            record, "canonicalIntelligenceIndexTokenCount", "output"
        ),
        "intelligence_answer_tokens_total": nested(
            record, "canonicalIntelligenceIndexTokenCount", "answer"
        ),
        "intelligence_reasoning_tokens_total": nested(
            record, "canonicalIntelligenceIndexTokenCount", "reasoning"
        ),
        "intelligence_output_tokens_per_task": nested(
            record, "intelligenceIndexOutputTokensPerTask", "output"
        ),
        "intelligence_answer_tokens_per_task": nested(
            record, "intelligenceIndexOutputTokensPerTask", "answer"
        ),
        "intelligence_reasoning_tokens_per_task": nested(
            record, "intelligenceIndexOutputTokensPerTask", "reasoning"
        ),
        "median_output_speed_tps": nested(record, "timescaleData", "medianOutputSpeed"),
        "median_time_to_first_chunk_seconds": nested(
            record, "timescaleData", "medianTimeToFirstChunk"
        ),
        "performance_data_source_type": performance.get("type"),
        "performance_provider_name": performance.get("providerName"),
        "model_weights_source_url": record.get("modelWeightsSourceUrl"),
        "license_name": record.get("licenseName"),
        "license_url": record.get("licenseUrl"),
        "commercial_allowed": record.get("commercialAllowed"),
        "deprecated": record.get("deprecated"),
        "source_page_url": f"https://artificialanalysis.ai/models/{record.get('slug')}",
        "snapshot_html_sha256": snapshot_sha,
        "source_record_json": json.dumps(
            record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ),
    }


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")


def csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    if not rows:
        raise ValueError("Refusing to write an empty AA detailed table")
    buffer = io.StringIO(newline="")
    fields = list(rows[0])
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def byte_record(path: Path, payload: bytes) -> dict[str, Any]:
    return {
        "path": relative(path),
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
    }


def selected_changes(
    prior_by_slug: dict[str, dict[str, Any]],
    current_by_slug: dict[str, dict[str, Any]],
    fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    for slug in sorted(prior_by_slug.keys() & current_by_slug.keys()):
        prior = prior_by_slug[slug]
        current = current_by_slug[slug]
        field_changes = {
            field: {"before": prior.get(field), "after": current.get(field)}
            for field in fields
            if prior.get(field) != current.get(field)
        }
        if field_changes:
            changes.append(
                {
                    "slug": slug,
                    "name": current["name"],
                    "changes": field_changes,
                }
            )
    return changes


def build_delta(
    current_rows: list[dict[str, Any]], current_snapshot_sha: str
) -> dict[str, Any]:
    if not PRIOR_RAW.is_file():
        raise FileNotFoundError(
            f"Prior AA snapshot required for the migration audit: {PRIOR_RAW}"
        )
    prior_gzip = PRIOR_RAW.read_bytes()
    prior_html = gzip.decompress(prior_gzip)
    prior_sha = sha256_bytes(prior_html)
    prior_models = extract_models(prior_html.decode("utf-8"))
    prior_rows = [flatten(prior_models[slug], prior_sha) for slug in sorted(prior_models)]
    prior_by_slug = {row["slug"]: row for row in prior_rows}
    current_by_slug = {row["slug"]: row for row in current_rows}
    added_slugs = sorted(current_by_slug.keys() - prior_by_slug.keys())
    removed_slugs = sorted(prior_by_slug.keys() - current_by_slug.keys())

    added_fields = (
        "slug",
        "name",
        "short_name",
        "creator_name",
        "release_date",
        "intelligence_index",
        "intelligence_index_estimated",
        "is_reasoning",
        "is_open_weights",
        "open_source_categorization",
        "parameters_b",
        "active_parameters_b",
        "size_class",
        "context_window_tokens",
        "model_weights_source_url",
    )
    identity_changes = selected_changes(
        prior_by_slug, current_by_slug, IDENTITY_FIELDS
    )
    score_changes = selected_changes(prior_by_slug, current_by_slug, SCORE_FIELDS)
    architecture_changes = selected_changes(
        prior_by_slug, current_by_slug, ARCHITECTURE_FIELDS
    )
    return {
        "schema_version": "1.0",
        "from_snapshot_date": PRIOR_SNAPSHOT_DATE,
        "to_snapshot_date": SNAPSHOT_DATE,
        "prior_snapshot": {
            "path": relative(PRIOR_RAW),
            "gzip_bytes": len(prior_gzip),
            "gzip_sha256": sha256_bytes(prior_gzip),
            "uncompressed_bytes": len(prior_html),
            "uncompressed_sha256": prior_sha,
        },
        "current_snapshot": {
            "path": relative(RAW),
            "uncompressed_sha256": current_snapshot_sha,
        },
        "fields_audited": {
            "identity": list(IDENTITY_FIELDS),
            "scores": list(SCORE_FIELDS),
            "architecture": list(ARCHITECTURE_FIELDS),
        },
        "inventory": {
            "prior_models": len(prior_rows),
            "current_models": len(current_rows),
            "shared_slugs": len(prior_by_slug.keys() & current_by_slug.keys()),
            "added_slugs": added_slugs,
            "removed_slugs": removed_slugs,
            "added_models": [
                {field: current_by_slug[slug].get(field) for field in added_fields}
                for slug in added_slugs
            ],
        },
        "identity_changes": identity_changes,
        "score_changes": score_changes,
        "architecture_changes": architecture_changes,
        "counts": {
            "added_models": len(added_slugs),
            "removed_models": len(removed_slugs),
            "shared_models_with_identity_changes": len(identity_changes),
            "shared_models_with_score_changes": len(score_changes),
            "shared_models_with_architecture_changes": len(architecture_changes),
            "shared_models_with_intelligence_index_changes": sum(
                "intelligence_index" in row["changes"] for row in score_changes
            ),
        },
    }


def build_artifacts(
    raw_html: bytes,
    raw_gzip: bytes,
    *,
    fetched_at: str,
    content_type: str,
) -> tuple[bytes, bytes, bytes, dict[str, Any]]:
    snapshot_sha = sha256_bytes(raw_html)
    models = extract_models(raw_html.decode("utf-8"))
    rows = [flatten(models[slug], snapshot_sha) for slug in sorted(models)]
    models_payload = csv_bytes(rows)

    open_parameter_rows = [
        row
        for row in rows
        if row["is_open_weights"] is True
        and row["parameters_b"] is not None
        and row["intelligence_index"] is not None
        and row["release_date"]
    ]
    token_rows = [
        row
        for row in open_parameter_rows
        if row["intelligence_output_tokens_per_task"] is not None
    ]
    delta = build_delta(rows, snapshot_sha)
    delta_payload = canonical_json(delta)
    counts = {
        "models": len(rows),
        "unique_model_ids": len({row["model_id"] for row in rows}),
        "unique_slugs": len({row["slug"] for row in rows}),
        "open_weight_models": sum(row["is_open_weights"] is True for row in rows),
        "models_with_total_parameters": sum(
            row["parameters_b"] is not None for row in rows
        ),
        "models_with_active_parameters": sum(
            row["active_parameters_b"] is not None for row in rows
        ),
        "open_weight_parameter_score_date_rows": len(open_parameter_rows),
        "open_weight_rows_with_inference_token_measurement": len(token_rows),
        "creators_in_parameter_panel": len(
            {row["creator_slug"] for row in open_parameter_rows}
        ),
    }
    metadata = {
        "schema_version": "2.0",
        "snapshot_date": SNAPSHOT_DATE,
        "source_url": SOURCE_URL,
        "fetched_at_utc": fetched_at,
        "content_type": content_type,
        "parse_method": "Next.js React Flight model objects; most-complete duplicate retained after core-field agreement",
        "raw_html_uncompressed_bytes": len(raw_html),
        "raw_html_uncompressed_sha256": snapshot_sha,
        "raw_gzip_sha256": sha256_bytes(raw_gzip),
        "model_csv_sha256": sha256_bytes(models_payload),
        "delta_json_sha256": sha256_bytes(delta_payload),
        **counts,
        "outputs": {
            "raw": relative(RAW),
            "models": relative(MODELS),
            "delta": relative(DELTA),
            "manifest": relative(MANIFEST),
        },
    }
    return models_payload, canonical_json(metadata), delta_payload, counts


def require_record(payload: bytes, record: dict[str, Any], label: str) -> None:
    if len(payload) != record.get("bytes"):
        raise ValueError(
            f"Frozen {label} byte count changed: {len(payload)} != {record.get('bytes')}"
        )
    digest = sha256_bytes(payload)
    if digest != record.get("sha256"):
        raise ValueError(
            f"Frozen {label} SHA-256 changed: {digest} != {record.get('sha256')}"
        )


def build_manifest(
    *,
    raw_gzip: bytes,
    raw_html: bytes,
    models_payload: bytes,
    metadata_payload: bytes,
    delta_payload: bytes,
    counts: dict[str, Any],
    fetched_at: str,
    content_type: str,
) -> dict[str, Any]:
    prior_gzip = PRIOR_RAW.read_bytes()
    prior_html = gzip.decompress(prior_gzip)
    delta = json.loads(delta_payload)
    raw_record = byte_record(RAW, raw_gzip)
    raw_record.update(
        {
            "wire_format": "gzip",
            "uncompressed_bytes": len(raw_html),
            "uncompressed_sha256": sha256_bytes(raw_html),
        }
    )
    prior_record = byte_record(PRIOR_RAW, prior_gzip)
    prior_record.update(
        {
            "wire_format": "gzip",
            "uncompressed_bytes": len(prior_html),
            "uncompressed_sha256": sha256_bytes(prior_html),
        }
    )
    return {
        "schema_version": "1.0",
        "snapshot_date": SNAPSHOT_DATE,
        "source_url": SOURCE_URL,
        "fetched_at_utc": fetched_at,
        "content_type": content_type,
        "parse_method": "Next.js React Flight model objects; most-complete duplicate retained after core-field agreement",
        "files": {
            "raw": raw_record,
            "models": byte_record(MODELS, models_payload),
            "metadata": byte_record(METADATA, metadata_payload),
            "delta_from_prior": byte_record(DELTA, delta_payload),
            "prior_raw": prior_record,
        },
        "counts": counts,
        "delta_summary": delta["counts"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Fetch and replace the frozen public AA snapshot.",
    )
    args = parser.parse_args()

    if args.refresh:
        raw_html, content_type = fetch()
        raw_gzip = gzip.compress(raw_html, compresslevel=9, mtime=0)
        fetched_at = datetime.now(timezone.utc).isoformat()
        models_payload, metadata_payload, delta_payload, counts = build_artifacts(
            raw_html,
            raw_gzip,
            fetched_at=fetched_at,
            content_type=content_type,
        )
        manifest = build_manifest(
            raw_gzip=raw_gzip,
            raw_html=raw_html,
            models_payload=models_payload,
            metadata_payload=metadata_payload,
            delta_payload=delta_payload,
            counts=counts,
            fetched_at=fetched_at,
            content_type=content_type,
        )
        atomic_write(RAW, raw_gzip)
        atomic_write(MODELS, models_payload)
        atomic_write(METADATA, metadata_payload)
        atomic_write(DELTA, delta_payload)
        atomic_write(MANIFEST, canonical_json(manifest))
    else:
        if not RAW.is_file() or not MANIFEST.is_file():
            raise FileNotFoundError(
                "No frozen AA July 31 snapshot and manifest; run with --refresh"
            )
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != "1.0":
            raise ValueError("Unsupported AA manifest schema version")
        if manifest.get("snapshot_date") != SNAPSHOT_DATE:
            raise ValueError("AA manifest snapshot date does not match collector")
        if manifest.get("source_url") != SOURCE_URL:
            raise ValueError("AA manifest source URL does not match collector")
        expected_paths = {
            "raw": RAW,
            "models": MODELS,
            "metadata": METADATA,
            "delta_from_prior": DELTA,
            "prior_raw": PRIOR_RAW,
        }
        for key, path in expected_paths.items():
            if manifest["files"][key].get("path") != relative(path):
                raise ValueError(f"AA manifest {key} path does not match collector")
        raw_gzip = RAW.read_bytes()
        require_record(raw_gzip, manifest["files"]["raw"], "raw gzip")
        raw_html = gzip.decompress(raw_gzip)
        raw_spec = manifest["files"]["raw"]
        if len(raw_html) != raw_spec["uncompressed_bytes"]:
            raise ValueError("Frozen AA uncompressed byte count changed")
        if sha256_bytes(raw_html) != raw_spec["uncompressed_sha256"]:
            raise ValueError("Frozen AA uncompressed SHA-256 changed")
        prior_gzip = PRIOR_RAW.read_bytes()
        require_record(
            prior_gzip, manifest["files"]["prior_raw"], "prior raw gzip"
        )
        prior_html = gzip.decompress(prior_gzip)
        prior_spec = manifest["files"]["prior_raw"]
        if (
            len(prior_html) != prior_spec["uncompressed_bytes"]
            or sha256_bytes(prior_html) != prior_spec["uncompressed_sha256"]
        ):
            raise ValueError("Frozen prior AA snapshot changed")

        fetched_at = manifest["fetched_at_utc"]
        content_type = manifest["content_type"]
        models_payload, metadata_payload, delta_payload, counts = build_artifacts(
            raw_html,
            raw_gzip,
            fetched_at=fetched_at,
            content_type=content_type,
        )
        if counts != manifest["counts"]:
            raise ValueError("Frozen AA extracted counts changed")
        require_record(
            models_payload, manifest["files"]["models"], "normalized model CSV"
        )
        require_record(
            metadata_payload, manifest["files"]["metadata"], "collection metadata"
        )
        require_record(
            delta_payload, manifest["files"]["delta_from_prior"], "snapshot delta"
        )
        atomic_write(MODELS, models_payload)
        atomic_write(METADATA, metadata_payload)
        atomic_write(DELTA, delta_payload)

    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
