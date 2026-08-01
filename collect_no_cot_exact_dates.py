#!/usr/bin/env python3
"""Build the day-level release-date overrides for the four unmatched no-CoT rows.

The broader no-CoT crosswalk already resolves 45/49 models to exact ECI or
Epoch checkpoints.  GPT-2/3/3.5 remain intentionally excluded from parameter
joining by prior user instruction, but Epoch still supplies day-level release
dates.  Qwen3-30B-A3B-Instruct-2507 is absent from the frozen Epoch snapshot;
its first-party Hugging Face repository's immutable initial commit supplies the
remaining day-level date.

The default mode is offline and rebuilds the override CSV from frozen inputs.
``--refresh`` fetches only the first-party Qwen commit history and freezes the
verbatim response as deterministic gzip.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DATE = "2026-07-18"
EPOCH = ROOT / "sources/epoch_all_ai_models_2026-07-31.csv"
QWEN_RAW = ROOT / f"sources/qwen3_30b_a3b_instruct_2507_hf_commits_{DATE}.json.gz"
OVERRIDES = ROOT / f"sources/no_cot_exact_date_overrides_{DATE}.csv"
METADATA = ROOT / f"sources/no_cot_exact_date_collection_metadata_{DATE}.json"

QWEN_REPO = "Qwen/Qwen3-30B-A3B-Instruct-2507"
QWEN_COMMITS_URL = f"https://huggingface.co/api/models/{QWEN_REPO}/commits/main?limit=100"
QWEN_INITIAL_COMMIT = "c9051e5f23e735fd6549f86b616377617848a621"
QWEN_INITIAL_TIMESTAMP = "2025-07-28T07:31:28.000Z"

LEGACY_EPOCH_ROWS = {
    "GPT-2": ("GPT-2 (1.5B)", "2019-02-14", "2019-02-01"),
    "GPT-3": ("GPT-3 175B (davinci)", "2020-05-28", "2020-05-01"),
    "GPT-3.5": ("GPT-3.5 (davinci-002)", "2022-03-15", "2022-03-01"),
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def deterministic_gzip(payload: bytes) -> bytes:
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0) as handle:
        handle.write(payload)
    return buffer.getvalue()


def fetch_qwen() -> bytes:
    request = urllib.request.Request(
        QWEN_COMMITS_URL,
        headers={"User-Agent": "frontier-parameter-model no-CoT date audit"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def load_qwen() -> bytes:
    if not QWEN_RAW.exists():
        raise FileNotFoundError(f"Missing frozen source {QWEN_RAW}; run with --refresh once")
    with gzip.open(QWEN_RAW, "rb") as handle:
        return handle.read()


def validate_qwen(payload: bytes) -> dict[str, Any]:
    commits = json.loads(payload)
    if not isinstance(commits, list) or not commits:
        raise ValueError("Hugging Face commit response is empty or malformed")
    ids = [row.get("id") for row in commits]
    if len(ids) != len(set(ids)):
        raise ValueError("Hugging Face commit response contains duplicate commit IDs")
    initial = next((row for row in commits if row.get("id") == QWEN_INITIAL_COMMIT), None)
    if initial is None:
        raise ValueError("Pinned Qwen initial commit is missing")
    if initial.get("date") != QWEN_INITIAL_TIMESTAMP:
        raise ValueError(
            f"Pinned Qwen initial timestamp changed: {initial.get('date')} != {QWEN_INITIAL_TIMESTAMP}"
        )
    if initial.get("title") != "initial commit":
        raise ValueError(f"Unexpected Qwen initial-commit title: {initial.get('title')!r}")
    return {"commit_count": len(commits), "initial_commit": initial}


def epoch_rows() -> dict[str, dict[str, str]]:
    with EPOCH.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    by_name: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_name.setdefault(row["Model"].strip(), []).append(row)
    selected: dict[str, dict[str, str]] = {}
    for paper_model, (epoch_name, expected_date, _) in LEGACY_EPOCH_ROWS.items():
        matches = by_name.get(epoch_name, [])
        if len(matches) != 1:
            raise ValueError(f"Expected one Epoch row for {epoch_name}, found {len(matches)}")
        row = matches[0]
        if row["Publication date"] != expected_date:
            raise ValueError(
                f"Epoch date changed for {epoch_name}: {row['Publication date']} != {expected_date}"
            )
        selected[paper_model] = row
    return selected


def build_rows(epoch_selected: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for paper_model, (epoch_name, exact_date, paper_month) in LEGACY_EPOCH_ROWS.items():
        source = epoch_selected[paper_model]
        rows.append(
            {
                "paper_model": paper_model,
                "paper_month_date": paper_month,
                "exact_release_date": exact_date,
                "day_offset_from_month_start": (date.fromisoformat(exact_date) - date.fromisoformat(paper_month)).days,
                "date_source": "Epoch exact checkpoint date",
                "source_checkpoint": epoch_name,
                "source_url": source["Link"],
                "source_field": "Publication date",
                "parameter_join_policy": "date_only_no_epoch_parameter_join",
                "identity_note": "Prior instruction to exclude GPT-2/3/3.5 from parameter joining is preserved.",
            }
        )
    qwen_date = QWEN_INITIAL_TIMESTAMP[:10]
    qwen_month = "2025-07-01"
    rows.append(
        {
            "paper_model": "Qwen 3 30B-A3B (2507)",
            "paper_month_date": qwen_month,
            "exact_release_date": qwen_date,
            "day_offset_from_month_start": (date.fromisoformat(qwen_date) - date.fromisoformat(qwen_month)).days,
            "date_source": "First-party Hugging Face initial commit",
            "source_checkpoint": QWEN_REPO,
            "source_url": QWEN_COMMITS_URL,
            "source_field": f"commit {QWEN_INITIAL_COMMIT} date",
            "parameter_join_policy": "date_only_no_epoch_parameter_join",
            "identity_note": "Exact paper checkpoint repository; no Epoch parameter identity is inferred.",
        }
    )
    return rows


def write_csv(rows: list[dict[str, Any]]) -> None:
    with OVERRIDES.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Fetch and freeze first-party Qwen commit history; default is offline.",
    )
    args = parser.parse_args()

    raw = fetch_qwen() if args.refresh else load_qwen()
    qwen = validate_qwen(raw)
    QWEN_RAW.parent.mkdir(parents=True, exist_ok=True)
    if args.refresh:
        QWEN_RAW.write_bytes(deterministic_gzip(raw))
    selected = epoch_rows()
    rows = build_rows(selected)
    write_csv(rows)

    metadata = {
        "generated_on": DATE,
        "collection_mode": "first-party refresh" if args.refresh else "frozen offline rebuild",
        "decision_scope": "release dates only; no parameter identities added",
        "inventory": {
            "paper_rows_overridden": len(rows),
            "epoch_date_rows": len(selected),
            "first_party_huggingface_date_rows": 1,
            "qwen_commit_rows_preserved": qwen["commit_count"],
        },
        "qwen_source": {
            "repository": QWEN_REPO,
            "commits_url": QWEN_COMMITS_URL,
            "initial_commit": QWEN_INITIAL_COMMIT,
            "initial_timestamp": QWEN_INITIAL_TIMESTAMP,
            "raw_uncompressed_sha256": sha256_bytes(raw),
        },
        "integrity_policy": {
            "network_in_frozen_pipeline": False,
            "verbatim_qwen_commit_response_preserved": True,
            "legacy_epoch_parameter_join_exclusion_preserved": True,
            "qwen_epoch_parameter_join_not_inferred": True,
            "all_four_month_only_rows_resolved": True,
        },
        "files": {
            str(EPOCH.relative_to(ROOT)): {"sha256": sha256(EPOCH)},
            str(QWEN_RAW.relative_to(ROOT)): {"sha256": sha256(QWEN_RAW)},
            str(OVERRIDES.relative_to(ROOT)): {"sha256": sha256(OVERRIDES)},
        },
    }
    METADATA.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"overrides": str(OVERRIDES), **metadata["inventory"]}, indent=2))


if __name__ == "__main__":
    main()
