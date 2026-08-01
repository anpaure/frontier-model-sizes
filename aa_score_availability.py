#!/usr/bin/env python3
"""Pinned Artificial Analysis score-publication timing.

Artificial Analysis model release dates and Intelligence Index publication
dates are different objects.  The former remains the algorithmic-progress
feature.  The latter determines when a benchmark-based parameter estimate
could first have been made in a chronological audit.

The changelog only begins in 2025 and does not report every historical score
revision.  Exact non-null ``modelAdded`` events are therefore used when they
exist; unmatched checkpoints retain their nominal release date and are marked
unverified rather than being presented as historically reconstructed scores.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
LEDGER_PATH = ROOT / "sources/aa_score_availability_2026-07-31.json"
RAW_PATH = ROOT / "sources/aa_changelog_2026-07-31.json.gz"
SNAPSHOT_DATE = "2026-07-31"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _normalize_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def _release_date(row: dict[str, Any]) -> str:
    for field in ("release_date", "canonical_release_date"):
        value = str(row.get(field) or "")
        if value:
            date.fromisoformat(value)
            return value
    raise ValueError("AA row lacks a release date")


def _record_from_event(event: dict[str, Any]) -> dict[str, Any] | None:
    if event.get("type") != "modelAdded":
        return None
    model = event.get("model") or {}
    score = model.get("intelligenceIndex")
    slug = str(model.get("slug") or "")
    if not slug or not isinstance(score, (int, float)) or not math.isfinite(score):
        return None
    creator = model.get("creator") or {}
    return {
        "aa_slug": slug,
        "aa_model_id": str(model.get("id") or ""),
        "aa_name_at_publication": str(model.get("name") or ""),
        "creator_name_at_publication": str(creator.get("name") or ""),
        "score_available_date": str(event.get("dateLa") or ""),
        "intelligence_index_at_publication": float(score),
        "changelog_event_id": str(event.get("id") or ""),
        "changelog_event_type": "modelAdded",
    }


def _expected_records(raw: dict[str, Any]) -> list[dict[str, Any]]:
    earliest: dict[str, dict[str, Any]] = {}
    for page in raw.get("pages", []):
        for event in page.get("data", []):
            record = _record_from_event(event)
            if record is None:
                continue
            key = record["aa_slug"]
            current = earliest.get(key)
            candidate_order = (
                record["score_available_date"],
                record["changelog_event_id"],
            )
            current_order = (
                current["score_available_date"],
                current["changelog_event_id"],
            ) if current else None
            if current is None or candidate_order < current_order:
                earliest[key] = record
    return [earliest[key] for key in sorted(earliest)]


@lru_cache(maxsize=2)
def load_aa_score_availability(path: Path = LEDGER_PATH) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "1.0":
        raise ValueError("Unsupported AA score-availability schema")
    if payload.get("snapshot_date") != SNAPSHOT_DATE:
        raise ValueError("AA score-availability snapshot is not pinned")

    evidence = payload.get("raw_evidence") or {}
    relative = str(evidence.get("path") or "")
    raw_path = ROOT / relative
    if raw_path != RAW_PATH or not raw_path.is_file():
        raise ValueError("AA changelog evidence path is missing or unexpected")
    expected_hash = str(evidence.get("sha256") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        raise ValueError("AA changelog evidence hash is malformed")
    if _sha256(raw_path) != expected_hash:
        raise ValueError("AA changelog evidence hash mismatch")

    compressed = raw_path.read_bytes()
    raw_bytes = gzip.decompress(compressed)
    if _sha256_bytes(raw_bytes) != evidence.get("uncompressed_sha256"):
        raise ValueError("AA changelog uncompressed hash mismatch")
    raw = json.loads(raw_bytes)
    if raw.get("snapshot_date") != SNAPSHOT_DATE:
        raise ValueError("AA changelog raw snapshot date mismatch")
    pages = raw.get("pages")
    if not isinstance(pages, list) or not pages:
        raise ValueError("AA changelog raw snapshot has no pages")
    if [page.get("page") for page in pages] != list(range(1, len(pages) + 1)):
        raise ValueError("AA changelog pages are incomplete or unordered")
    if any(page.get("total") != raw.get("api_reported_total_events") for page in pages):
        raise ValueError("AA changelog page totals disagree")
    events = [event for page in pages for event in page.get("data", [])]
    if len(events) != raw.get("returned_events"):
        raise ValueError("AA changelog returned-event count disagrees with page contents")
    event_ids = [str(event.get("id") or "") for event in events]
    if any(not event_id for event_id in event_ids) or len(event_ids) != len(set(event_ids)):
        raise ValueError("AA changelog contains missing or duplicate event IDs")

    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("AA score-availability ledger is empty")
    expected_records = _expected_records(raw)
    if records != expected_records:
        raise ValueError("AA score-availability ledger does not reproduce raw evidence")
    slugs = [record["aa_slug"] for record in records]
    if len(slugs) != len(set(slugs)):
        raise ValueError("AA score-availability ledger has duplicate slugs")
    for record in records:
        date.fromisoformat(record["score_available_date"])
        if record["score_available_date"] > SNAPSHOT_DATE:
            raise ValueError("AA score-availability record postdates the snapshot")
        if not record["aa_model_id"] or not record["changelog_event_id"]:
            raise ValueError("AA score-availability record lacks stable identity")

    summary = payload.get("summary") or {}
    if summary.get("total_changelog_events") != len(events):
        raise ValueError("AA score-availability summary event count mismatch")
    if summary.get("api_reported_total_events") != raw.get("api_reported_total_events"):
        raise ValueError("AA score-availability API total mismatch")
    if summary.get("api_total_reconciles") is not (
        len(events) == raw.get("api_reported_total_events")
    ):
        raise ValueError("AA score-availability reconciliation flag mismatch")
    if summary.get("verified_score_slugs") != len(records):
        raise ValueError("AA score-availability summary record count mismatch")
    return payload


@lru_cache(maxsize=2)
def _records_by_slug() -> dict[str, dict[str, Any]]:
    return {
        record["aa_slug"]: record
        for record in load_aa_score_availability()["records"]
    }


@lru_cache(maxsize=2)
def _records_by_name() -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for record in load_aa_score_availability()["records"]:
        output.setdefault(_normalize_name(record["aa_name_at_publication"]), []).append(record)
    return output


def resolve_aa_score_availability(row: dict[str, Any]) -> dict[str, Any]:
    """Return the best pinned score-availability record for an AA row.

    Slug matches are exact.  A normalized-name fallback is accepted only when
    it resolves to one changelog record.  No fuzzy or family-level match is
    allowed.
    """

    for field in ("aa_slug", "selected_slug", "slug"):
        slug = str(row.get(field) or "")
        if slug and slug in _records_by_slug():
            return dict(_records_by_slug()[slug])
    for field in ("aa_name", "selected_name", "model", "name"):
        name = _normalize_name(row.get(field))
        candidates = _records_by_name().get(name, [])
        if len(candidates) == 1:
            return dict(candidates[0])
    return {}


def aa_score_available_date(row: dict[str, Any]) -> str:
    explicit = str(row.get("aa_score_available_date") or "")
    if explicit:
        date.fromisoformat(explicit)
        return explicit
    record = resolve_aa_score_availability(row)
    return str(record.get("score_available_date") or _release_date(row))


def aa_score_availability_verified(row: dict[str, Any]) -> bool:
    explicit = row.get("aa_score_availability_verified")
    if explicit not in (None, ""):
        if isinstance(explicit, str):
            return explicit.strip().lower() in {"1", "true", "yes"}
        return bool(explicit)
    return bool(resolve_aa_score_availability(row))


def aa_prediction_information_date(row: dict[str, Any]) -> str:
    """Date when this row's AA score could first support a prediction."""

    return max(_release_date(row), aa_score_available_date(row))
