#!/usr/bin/env python3
"""Collect or offline-verify Artificial Analysis score-publication timing."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from aa_score_availability import (
    LEDGER_PATH,
    RAW_PATH,
    ROOT,
    SNAPSHOT_DATE,
    _expected_records,
    load_aa_score_availability,
)


API = "https://artificialanalysis.ai/api/changelogs"


def _fetch_json(url: str) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(6):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "frontier-parameter-aa-timing-audit/1.0"},
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            last_error = error
            time.sleep(min(10.0, 1.5 * (attempt + 1)))
    raise RuntimeError(f"Could not fetch {url}: {last_error}")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def refresh() -> None:
    pages = []
    expected_total: int | None = None
    for page_number in range(1, 21):
        response = _fetch_json(f"{API}?page={page_number}")
        if response.get("page") != page_number:
            raise RuntimeError("Artificial Analysis changelog returned the wrong page")
        total = response.get("total")
        if not isinstance(total, int) or total <= 0:
            raise RuntimeError("Artificial Analysis changelog lacks a valid total")
        if expected_total is None:
            expected_total = total
        elif total != expected_total:
            raise RuntimeError("Artificial Analysis changelog total changed during collection")
        data = response.get("data")
        if not isinstance(data, list):
            raise RuntimeError("Artificial Analysis changelog page lacks data")
        pages.append(
            {
                "page": page_number,
                "has_more": bool(response.get("has_more")),
                "total": total,
                "data": data,
            }
        )
        if not response.get("has_more"):
            break
    else:
        raise RuntimeError("Artificial Analysis changelog exceeded the page limit")

    events = [event for page in pages for event in page["data"]]
    event_ids = [str(event.get("id") or "") for event in events]
    if any(not event_id for event_id in event_ids) or len(event_ids) != len(set(event_ids)):
        raise RuntimeError("Artificial Analysis changelog event IDs are not unique")

    raw = {
        "schema_version": "1.0",
        "snapshot_date": SNAPSHOT_DATE,
        "source": API,
        "api_reported_total_events": expected_total,
        "returned_events": len(events),
        "pages": pages,
    }
    raw_bytes = (
        json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    compressed = gzip.compress(raw_bytes, compresslevel=9, mtime=0)
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAW_PATH.write_bytes(compressed)

    records = _expected_records(raw)
    model_added = sum(event.get("type") == "modelAdded" for event in events)
    payload = {
        "schema_version": "1.0",
        "snapshot_date": SNAPSHOT_DATE,
        "purpose": (
            "Separate nominal model release from the first dated Artificial Analysis "
            "Intelligence Index publication event in chronological parameter audits."
        ),
        "policy": (
            "Use the earliest changelog modelAdded event with a non-null Intelligence "
            "Index for an exact AA slug. Keep release date as the regression feature. "
            "Unmatched rows fall back to nominal release and remain explicitly unverified."
        ),
        "raw_evidence": {
            "path": str(RAW_PATH.relative_to(ROOT)),
            "sha256": _sha256(compressed),
            "uncompressed_sha256": _sha256(raw_bytes),
            "source_url": API,
        },
        "summary": {
            "pages": len(pages),
            "total_changelog_events": len(events),
            "api_reported_total_events": expected_total,
            "api_total_reconciles": len(events) == expected_total,
            "model_added_events": model_added,
            "verified_score_slugs": len(records),
            "earliest_event_date": min(event["dateLa"] for event in events),
            "latest_event_date": max(event["dateLa"] for event in events),
        },
        "records": records,
    }
    LEDGER_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    load_aa_score_availability.cache_clear()
    load_aa_score_availability()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Fetch and replace the pinned changelog snapshot; default is offline verification.",
    )
    args = parser.parse_args()
    if args.refresh:
        refresh()
    payload = load_aa_score_availability()
    print(
        json.dumps(
            {
                "status": "PASS",
                "snapshot_date": payload["snapshot_date"],
                **payload["summary"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
