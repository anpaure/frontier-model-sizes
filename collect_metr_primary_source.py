#!/usr/bin/env python3
"""Freeze, normalize, and audit METR-Horizon-v1.1's official result asset.

The committed raw YAML is the source of truth.  Default runs are offline and
deterministic; ``--refresh`` re-fetches the official asset before rebuilding
the normalized table.  The parser is intentionally narrow and fail-closed for
the published schema, which avoids adding a runtime YAML dependency while
ensuring that an upstream structural change cannot pass silently.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DATE = "2026-07-18"
SOURCE_URL = "https://metr.org/assets/benchmark_results_1_1.yaml"
RAW = ROOT / f"sources/metr_benchmark_results_1_1_{DATE}.yaml"
SIGNALS = ROOT / f"sources/metr_horizon_official_signals_{DATE}.csv"
METADATA = ROOT / f"sources/metr_horizon_official_metadata_{DATE}.json"
AUDIT = ROOT / f"outputs/019f6c42-2d53-7743-ab07-6293e2618dd7/metr_primary_source_audit_{DATE}.json"
LEGACY = ROOT / "sources/metr_horizon_user_snapshot_2026-07-17.csv"

EXPECTED = {
    "benchmark_name": "METR-Horizon-v1.1",
    "all_time_stitched_point_estimate_days": "187.778",
    "from_2023_on_point_estimate_days": "128.744",
    "from_2023_on_ci_low_days": "104.428",
    "from_2023_on_ci_high_days": "158.012",
    "long_tasks_version": "799cc9c4b4483a93fc3445623a49ea1bd74fdeb2",
    "swaa_version": "f6cc84052e2a79dd540766c8a1b15ff399696371",
    "result_rows": 26,
}

SIGNAL_FIELDS = [
    "source_id",
    "benchmark_name",
    "release_date",
    "average_score",
    "is_sota",
    "p50_estimate_minutes",
    "p50_ci_low_minutes",
    "p50_ci_high_minutes",
    "p80_estimate_minutes",
    "p80_ci_low_minutes",
    "p80_ci_high_minutes",
    "scaffold_family",
    "scaffolds_json",
    "long_tasks_version",
    "swaa_version",
]

LEGACY_FIELDS = [field for field in SIGNAL_FIELDS if field != "scaffolds_json"]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def fetch() -> bytes:
    request = urllib.request.Request(
        SOURCE_URL,
        headers={
            "Accept": "application/yaml,text/yaml,text/plain;q=0.9,*/*;q=0.1",
            "User-Agent": "frontier-parameter-model METR primary-source audit",
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    if not payload.startswith(b"benchmark_name: METR-Horizon-v1.1\n"):
        raise ValueError("Official METR response does not have the expected document header")
    return payload


def scalar(lines: list[str], pattern: str, label: str) -> str:
    matches = [match.group(1) for line in lines if (match := re.fullmatch(pattern, line))]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {label}; found {len(matches)}")
    return matches[0]


def scaffold_family(scaffolds: list[str]) -> str:
    if not scaffolds:
        raise ValueError("Result row has no scaffold entries")
    if "," in scaffolds[0]:
        # METR's list also includes the separate SWAA ``generate`` scaffold.
        # The final list item is the primary agent and matches the terminal
        # component of the task-specific scaffold entries.
        family = scaffolds[-1]
        if "," in family:
            family = family.rsplit(",", 1)[-1]
        if scaffolds[0].rsplit(",", 1)[-1] != family:
            raise ValueError(f"First and final scaffold families disagree: {scaffolds}")
        return family
    return "/".join(scaffolds)


def parse_result(source_id: str, lines: list[str], versions: dict[str, str]) -> dict[str, str]:
    benchmark = scalar(lines, r"    benchmark_name: (\S+)", f"{source_id} benchmark")
    release_date = scalar(lines, r"    release_date: (\d{4}-\d{2}-\d{2})", f"{source_id} date")
    is_sota = scalar(lines, r"      is_sota: (true|false)", f"{source_id} is_sota")

    metric: str | None = None
    metrics: dict[str, dict[str, str]] = {}
    scaffolds: list[str] = []
    in_scaffolds = False
    allowed_structural = {"    metrics:", "    scaffolds:"}
    for line in lines:
        metric_match = re.fullmatch(
            r"      (average_score|p50_horizon_length|p80_horizon_length):", line
        )
        if metric_match:
            metric = metric_match.group(1)
            metrics[metric] = {}
            in_scaffolds = False
            continue
        value_match = re.fullmatch(r"        (estimate|ci_low|ci_high): ([0-9.]+)", line)
        if value_match:
            if metric is None:
                raise ValueError(f"Metric value before metric name in {source_id}: {line}")
            metrics[metric][value_match.group(1)] = value_match.group(2)
            continue
        if line == "    scaffolds:":
            in_scaffolds = True
            metric = None
            continue
        scaffold_match = re.fullmatch(r"    - (.+)", line)
        if scaffold_match:
            if not in_scaffolds:
                raise ValueError(f"Scaffold entry outside list in {source_id}: {line}")
            scaffolds.append(scaffold_match.group(1))
            continue
        if line in allowed_structural:
            continue
        if re.fullmatch(r"    benchmark_name: \S+", line):
            continue
        if re.fullmatch(r"    release_date: \d{4}-\d{2}-\d{2}", line):
            continue
        if re.fullmatch(r"      is_sota: (true|false)", line):
            continue
        raise ValueError(f"Unrecognized YAML line in {source_id}: {line!r}")

    expected_metric_keys = {
        "average_score": {"estimate"},
        "p50_horizon_length": {"estimate", "ci_low", "ci_high"},
        "p80_horizon_length": {"estimate", "ci_low", "ci_high"},
    }
    actual_metric_keys = {name: set(values) for name, values in metrics.items()}
    if actual_metric_keys != expected_metric_keys:
        raise ValueError(
            f"Unexpected metric schema for {source_id}: {actual_metric_keys!r}"
        )
    return {
        "source_id": source_id,
        "benchmark_name": benchmark,
        "release_date": release_date,
        "average_score": metrics["average_score"]["estimate"],
        "is_sota": is_sota,
        "p50_estimate_minutes": metrics["p50_horizon_length"]["estimate"],
        "p50_ci_low_minutes": metrics["p50_horizon_length"]["ci_low"],
        "p50_ci_high_minutes": metrics["p50_horizon_length"]["ci_high"],
        "p80_estimate_minutes": metrics["p80_horizon_length"]["estimate"],
        "p80_ci_low_minutes": metrics["p80_horizon_length"]["ci_low"],
        "p80_ci_high_minutes": metrics["p80_horizon_length"]["ci_high"],
        "scaffold_family": scaffold_family(scaffolds),
        "scaffolds_json": json.dumps(scaffolds, ensure_ascii=False, separators=(",", ":")),
        "long_tasks_version": versions["long_tasks_version"],
        "swaa_version": versions["swaa_version"],
    }


def parse_document(payload: bytes) -> tuple[dict[str, str], list[dict[str, str]]]:
    text = payload.decode("utf-8")
    if "\r" in text:
        raise ValueError("Unexpected CR bytes in official METR YAML")
    lines = text.splitlines()
    try:
        results_index = lines.index("results:")
    except ValueError as exc:
        raise ValueError("Official METR YAML has no results section") from exc
    if not lines[-1].startswith("swaa_version: "):
        raise ValueError("Official METR YAML no longer ends with swaa_version")

    metadata_lines = lines[:results_index] + [lines[-1]]
    metadata = {
        "benchmark_name": scalar(metadata_lines, r"benchmark_name: (\S+)", "benchmark name"),
        "all_time_stitched_point_estimate_days": "",
        "from_2023_on_ci_high_days": scalar(
            metadata_lines, r"    ci_high: ([0-9.]+)", "from-2023 CI high"
        ),
        "from_2023_on_ci_low_days": scalar(
            metadata_lines, r"    ci_low: ([0-9.]+)", "from-2023 CI low"
        ),
        "from_2023_on_point_estimate_days": "",
        "long_tasks_version": scalar(
            metadata_lines, r"long_tasks_version: ([0-9a-f]{40})", "long-tasks version"
        ),
        "swaa_version": scalar(
            metadata_lines, r"swaa_version: ([0-9a-f]{40})", "SWAA version"
        ),
        "exclusion_rule": "excludes points with central estimate p50 > 16 hrs",
    }

    # Two point-estimate lines share the same indentation.  Resolve them by
    # their enclosing blocks while still checking the complete fixed schema.
    point_positions = [
        (index, re.fullmatch(r"    point_estimate: ([0-9.]+)", line))
        for index, line in enumerate(metadata_lines)
    ]
    point_values = [match.group(1) for _, match in point_positions if match]
    if point_values != ["187.778", "128.744"]:
        raise ValueError(f"Unexpected trend point estimates/order: {point_values}")
    metadata["all_time_stitched_point_estimate_days"] = point_values[0]
    metadata["from_2023_on_point_estimate_days"] = point_values[1]

    blocks: list[tuple[str, list[str]]] = []
    current_id: str | None = None
    current_lines: list[str] = []
    for line in lines[results_index + 1 : -1]:
        start = re.fullmatch(r"  ([a-zA-Z0-9_]+):", line)
        if start:
            if current_id is not None:
                blocks.append((current_id, current_lines))
            current_id = start.group(1)
            current_lines = []
        else:
            if current_id is None:
                raise ValueError(f"Unexpected line before first result: {line!r}")
            current_lines.append(line)
    if current_id is not None:
        blocks.append((current_id, current_lines))

    rows = [parse_result(source_id, block_lines, metadata) for source_id, block_lines in blocks]
    if len(rows) != EXPECTED["result_rows"]:
        raise ValueError(f"Expected 26 METR result rows; found {len(rows)}")
    if len({row['source_id'] for row in rows}) != len(rows):
        raise ValueError("Official METR results contain duplicate source IDs")
    for key, expected in EXPECTED.items():
        if key == "result_rows":
            continue
        if metadata[key] != expected:
            raise ValueError(f"Official METR {key} changed: {metadata[key]!r} != {expected!r}")
    return metadata, rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SIGNAL_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def compare_legacy(rows: list[dict[str, str]]) -> dict[str, Any]:
    legacy = read_csv(LEGACY)
    by_id = {row["source_id"]: row for row in legacy}
    if len(by_id) != len(legacy):
        raise ValueError("Legacy METR snapshot has duplicate source IDs")
    mismatches: list[dict[str, str]] = []
    exact_rows = 0
    for row in rows:
        old = by_id.get(row["source_id"])
        if old is None:
            mismatches.append({"source_id": row["source_id"], "field": "source_id", "official": row["source_id"], "legacy": "MISSING"})
            continue
        row_mismatches = 0
        for field in LEGACY_FIELDS:
            if row[field] != old[field]:
                row_mismatches += 1
                mismatches.append({"source_id": row["source_id"], "field": field, "official": row[field], "legacy": old[field]})
        if row_mismatches == 0:
            exact_rows += 1
    extra = sorted(set(by_id) - {row["source_id"] for row in rows})
    for source_id in extra:
        mismatches.append({"source_id": source_id, "field": "source_id", "official": "MISSING", "legacy": source_id})
    return {
        "official_rows": len(rows),
        "legacy_rows": len(legacy),
        "common_fields_compared": LEGACY_FIELDS,
        "exact_rows": exact_rows,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-fetch the official METR YAML before rebuilding normalized artifacts.",
    )
    args = parser.parse_args()

    if args.refresh:
        RAW.parent.mkdir(parents=True, exist_ok=True)
        RAW.write_bytes(fetch())
    if not RAW.exists():
        raise FileNotFoundError(f"Missing frozen source {RAW}; run with --refresh once")

    payload = RAW.read_bytes()
    trend, rows = parse_document(payload)
    write_csv(SIGNALS, rows)
    comparison = compare_legacy(rows)
    if comparison["mismatch_count"]:
        raise ValueError(f"Official/legacy METR comparison failed: {comparison['mismatches'][:5]}")

    metadata = {
        "schema_version": "1.0",
        "snapshot_date": DATE,
        "source": {
            "name": "METR-Horizon-v1.1 official benchmark results",
            "url": SOURCE_URL,
            "authority": "first_party",
            "retrieval_mode": "live_refresh" if args.refresh else "frozen_offline",
        },
        "trend": trend,
        "inventory": {
            "result_rows": len(rows),
            "unique_source_ids": len({row["source_id"] for row in rows}),
            "full_scaffold_entries": sum(len(json.loads(row["scaffolds_json"])) for row in rows),
            "scaffold_families": sorted({row["scaffold_family"] for row in rows}),
        },
        "integrity_policy": {
            "parser": "fail_closed_published_schema",
            "official_source_is_authoritative": True,
            "all_result_fields_preserved": True,
            "full_scaffold_arrays_preserved": True,
            "legacy_snapshot_used_only_as_exact_crosscheck": True,
        },
        "files": {
            relative(RAW): {"sha256": sha256(RAW), "bytes": RAW.stat().st_size},
            relative(SIGNALS): {"sha256": sha256(SIGNALS), "rows": len(rows)},
            relative(LEGACY): {"sha256": sha256(LEGACY), "rows": comparison["legacy_rows"]},
        },
    }
    METADATA.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    audit = {
        "schema_version": "1.0",
        "snapshot_date": DATE,
        "status": "PASS",
        "official_asset": {
            "url": SOURCE_URL,
            "raw_sha256": sha256(RAW),
            "raw_bytes": RAW.stat().st_size,
            "result_rows": len(rows),
            "unique_source_ids": len({row["source_id"] for row in rows}),
        },
        "trend": trend,
        "legacy_exact_crosscheck": comparison,
        "losslessness": {
            "normalized_columns": SIGNAL_FIELDS,
            "full_scaffold_arrays_preserved": True,
            "full_scaffold_entries": metadata["inventory"]["full_scaffold_entries"],
            "legacy_collapsed_scaffold_family_preserved": True,
        },
        "files": {
            relative(RAW): {"sha256": sha256(RAW)},
            relative(SIGNALS): {"sha256": sha256(SIGNALS)},
            relative(METADATA): {"sha256": sha256(METADATA)},
            relative(LEGACY): {"sha256": sha256(LEGACY)},
        },
    }
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "PASS",
                "official_rows": len(rows),
                "legacy_exact_rows": comparison["exact_rows"],
                "raw_sha256": sha256(RAW),
                "signals_sha256": sha256(SIGNALS),
                "full_scaffold_entries": metadata["inventory"]["full_scaffold_entries"],
                "audit": relative(AUDIT),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
